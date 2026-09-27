"""Small exact mechanisms and a declared MAG-topology teaching specialization."""
import json
from pathlib import Path
import numpy as np
import torch
from neural_memory import initialize_memory, read_memory, write_memory


def linear_write(weight, momentum, key, value, rate=0.5, retention=0.0, decay=0.0):
    residual = weight @ key - value
    gradient = np.outer(residual, key)
    next_momentum = retention * momentum - rate * gradient
    return (1-decay) * weight + next_momentum, next_momentum, gradient


def associative_trace(keys, values, rate=0.5, retention=0.0, decay=0.0):
    weight = np.zeros((len(values[0]), len(keys[0])))
    momentum = np.zeros_like(weight)
    records = []
    for key, value in zip(np.array(keys), np.array(values)):
        residual = weight @ key - value
        new_weight, new_momentum, gradient = linear_write(
            weight, momentum, key, value, rate, retention, decay)
        records.append({'key': key.tolist(), 'value': value.tolist(),
            'before': weight.tolist(), 'loss': float(0.5 * residual @ residual),
            'gradient': gradient.tolist(), 'momentum': new_momentum.tolist(),
            'after': new_weight.tolist(), 'read_first': (new_weight @ keys[0]).tolist()})
        weight, momentum = new_weight, new_momentum
    return records


def gated_sequence(tokens, prefix=True, rate=0.5, state=None):
    # Identity Q/K/V; two-token causal window, including current observed token.
    weight, momentum, recent = ((np.zeros((2, 2)), np.zeros((2, 2)), [])
                               if state is None else (state[0].copy(), state[1].copy(),
                                                      [value.copy() for value in state[2]]))
    records = []
    for token in np.array(tokens, dtype=float):
        recent = (recent + [token])[-2:]
        context = np.array(([np.array([1.0, 0.0])] if prefix else []) + recent)
        logits = context @ token / np.sqrt(2)
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        short = probabilities @ context
        weight, momentum, _ = linear_write(weight, momentum, token, token, rate, 0.5, 0.0)
        long = weight @ token
        output = short * np.tanh(long)
        records.append({'input': token.tolist(), 'context': context.tolist(),
            'attention': probabilities.tolist(), 'short': short.tolist(),
            'weight': weight.tolist(), 'momentum': momentum.tolist(),
            'long': long.tolist(), 'output': output.tolist()})
    return records, (weight, momentum, recent)


def chunk_trace(anchor=False, chunk_size=2, rate=0.5):
    weight = 0.0
    anchor_weight = 0.0
    records = []
    for index, target in enumerate([1.0, 2.0]):
        if index % chunk_size == 0:
            anchor_weight = weight
        gradient = (anchor_weight if anchor else weight) - target
        weight -= rate * gradient
        records.append({'gradient': gradient, 'weight': weight})
    return records


def scalar_outer(rate_value=0.25):
    rate = torch.tensor(rate_value, dtype=torch.float64, requires_grad=True)
    weight = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    inner_loss = 0.5 * (weight * 2 - 1).square()
    gradient, = torch.autograd.grad(inner_loss, weight, create_graph=True)
    next_weight = weight - rate * gradient
    outer_loss = 0.5 * (next_weight * 3 - 2).square()
    derivative, = torch.autograd.grad(outer_loss, rate)
    return {'weight': float(next_weight.detach()), 'prediction': float((next_weight*3).detach()),
            'loss': float(outer_loss.detach()), 'derivative': float(derivative)}


def nonlinear_outer(rate_value, derivative=False):
    weights = initialize_memory(3, input_size=2, hidden_size=3)
    state = tuple(torch.zeros_like(weight) for weight in weights)
    rate = torch.tensor(rate_value, dtype=torch.float64, requires_grad=True)
    key = torch.tensor([1.0, -0.5], dtype=torch.float64)
    target = torch.tensor(0.8, dtype=torch.float64)
    new_weights, _, _, _ = write_memory(weights, state, key, target, rate, 0.0, 0.0, True)
    query = torch.tensor([-0.25, 0.75], dtype=torch.float64)
    loss = 0.5 * (read_memory(new_weights, query) + 0.3).square()
    if derivative:
        gradient, = torch.autograd.grad(loss, rate)
        return float(loss.detach()), float(gradient)
    return float(loss.detach())


def calculations():
    orthogonal = associative_trace([[1, 0], [0, 1]], [[2], [4]])
    correlated = associative_trace([[1, 0], [1, 1]], [[2], [4]])
    loss, gradient = nonlinear_outer(0.1, True)
    epsilon = 1e-6
    finite_difference = (nonlinear_outer(0.1+epsilon) - nonlinear_outer(0.1-epsilon)) / (2*epsilon)
    tokens = [[1, 0], [0, 1], [1, 1], [1, -1]]
    full, final = gated_sequence(tokens)
    prefixless, _ = gated_sequence(tokens, prefix=False)
    disabled, _ = gated_sequence(tokens, rate=0.0)
    first, checkpoint = gated_sequence(tokens[:2])
    rest, _ = gated_sequence(tokens[2:], state=checkpoint)
    fresh, _ = gated_sequence(tokens[2:])
    perturbed, _ = gated_sequence(tokens[:3] + [[-2, 1]])
    state_parameter_count = 2*2048*512  # Bias-free two-layer memory, d -> H -> d.
    result = {'orthogonal': orthogonal, 'correlated': correlated,
        'sequential': chunk_trace(), 'anchor': chunk_trace(True),
        'anchor_size_one': chunk_trace(True, 1), 'scalar_outer': scalar_outer(),
        'nonlinear_outer': {'loss': loss, 'autograd': gradient, 'finite_difference': finite_difference},
        'gated': full, 'prefixless': prefixless, 'writes_disabled': disabled,
        'continuation': first + rest, 'fresh_tail': fresh, 'future_changed': perturbed,
        'memory_bytes': {'parameters_per_layer': state_parameter_count,
            'fast_weights_plus_momentum_fp32_per_layer': 2*state_parameter_count*4,
            'layers_24_batch_4_GiB': 24*4*2*state_parameter_count*4/2**30,
            'window_2048_layers_24_batch_4_kv_heads_8_head_dim_128_fp16_GiB':
                2*4*24*2048*8*128*2/2**30,
            'full_131072_layers_24_batch_4_kv_heads_8_head_dim_128_fp16_GiB':
                2*4*24*131072*8*128*2/2**30},
        'decay_half_life_steps_alpha_001': float(np.log(0.5)/np.log(0.99)),
        'momentum_reversal_steps_eta_09': int(np.floor(np.log(0.5)/np.log(0.9)) + 1)}
    (Path(__file__).resolve().parent / 'mechanism-results.json').write_text(
        json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ['scalar_outer', 'nonlinear_outer', 'memory_bytes']}, indent=2))
    return result


if __name__ == '__main__':
    calculations()
