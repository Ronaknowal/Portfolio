"""Complete operator bridges, not replicas of whole named language models.

PyTorch is required; --griffin additionally needs recurrentgemma with Torch extras.
The optional specialist package run is prepared, not claimed executed.
"""
import argparse
import math
from pathlib import Path
import runpy
import torch
from torch.nn import functional as F


def attention_read(queries, keys, values, allowed):
    if not allowed.any(dim=-1).all():
        raise ValueError("each query must have a legal key")
    scores = queries @ keys.transpose(-1, -2) / math.sqrt(queries.shape[-1])
    weights = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
    return weights @ values


def check_segmented_attention():
    torch.manual_seed(4)
    queries, keys = [torch.randn(1, 1, 7, 4, dtype=torch.float64) for _ in range(2)]
    values = torch.randn(1, 1, 7, 3, dtype=torch.float64)
    cached_keys, cached_values = keys[..., :0, :], values[..., :0, :]
    retained_positions = torch.empty(0, dtype=torch.long)
    outputs = []
    for start in range(0, 7, 3):
        stop = min(start + 3, 7)
        positions = torch.cat([retained_positions, torch.arange(start, stop)])
        visible_keys = torch.cat([cached_keys, keys[..., start:stop, :]], dim=-2)
        visible_values = torch.cat([cached_values, values[..., start:stop, :]], dim=-2)
        legal = positions[None, :] <= torch.arange(start, stop)[:, None]
        query = queries[..., start:stop, :]
        manual = attention_read(query, visible_keys, visible_values, legal)
        native = F.scaled_dot_product_attention(query, visible_keys, visible_values,
                                                attn_mask=legal, dropout_p=0.0)
        torch.testing.assert_close(manual, native, atol=1e-12, rtol=1e-12)
        outputs.append(native)
        # The cache belongs to this stream; detaching limits gradient history.
        cached_keys, cached_values = visible_keys[..., -4:, :].detach(), visible_values[..., -4:, :].detach()
        retained_positions = positions[-4:]
    print({"segmented_output_shape": tuple(torch.cat(outputs, dim=-2).shape)})


def check_latent_read():
    namespace = runpy.run_path(str(Path(__file__).with_name("latent_trajectory_classifier.py")))
    torch.manual_seed(9)
    layer = namespace["Attention"](8).double()
    queries = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    inputs = torch.randn(2, 11, 8, dtype=torch.float64, requires_grad=True)
    valid = torch.ones(2, 11, dtype=torch.bool)
    valid[1, -2:] = False
    manual, _ = layer(queries, inputs, valid)
    read = F.scaled_dot_product_attention(
        layer.query(queries)[:, None], layer.key(inputs)[:, None],
        layer.value(inputs)[:, None], attn_mask=valid[:, None, None, :], dropout_p=0.0,
    )[:, 0]
    native = layer.output(read)
    torch.testing.assert_close(manual, native, atol=1e-12, rtol=1e-12)
    variables = (queries, inputs, *layer.parameters())
    manual_gradients = torch.autograd.grad(manual.square().sum(), variables, retain_graph=True)
    native_gradients = torch.autograd.grad(native.square().sum(), variables)
    for manual, library in zip(manual_gradients, native_gradients):
        torch.testing.assert_close(manual, library, atol=1e-11, rtol=1e-11)
    print("latent attention values and gradients agree")


def rglru_reference(inputs, positions, input_weight, input_bias,
                    recurrence_weight, recurrence_bias, raw_decay, initial=None):
    """Block-diagonal learned gates and sequential state; float32 scope."""
    batch, length, width = inputs.shape
    heads, block_width, _ = input_weight.shape
    blocks = inputs.reshape(batch, length, heads, block_width)
    input_gate = torch.sigmoid(torch.einsum("bthi,hij->bthj", blocks, input_weight) + input_bias)
    recurrence_gate = torch.sigmoid(torch.einsum("bthi,hij->bthj", blocks, recurrence_weight) + recurrence_bias)
    input_gate, recurrence_gate = input_gate.flatten(-2), recurrence_gate.flatten(-2)
    log_decay = -8 * recurrence_gate * F.softplus(raw_decay)
    decay = torch.exp(log_decay)
    reset = positions == 0
    # expm1 avoids cancellation as log_decay approaches zero. In ordinary
    # precision the derivative differs from the package only in its clipped tail.
    injection_scale = torch.sqrt(-torch.expm1(2 * log_decay))
    injection_scale = torch.where(reset[..., None], 1.0, injection_scale)
    state = torch.zeros(batch, width, device=inputs.device) if initial is None else initial
    states = []
    for time in range(length):
        previous = torch.where(reset[:, time, None], 0.0, state)
        state = decay[:, time] * previous + injection_scale[:, time] * input_gate[:, time] * inputs[:, time]
        states.append(state)
    return torch.stack(states, dim=1), state


def check_griffin_component():
    from recurrentgemma.torch.layers import RGLRU

    torch.manual_seed(12)
    layer = RGLRU(width=8, num_heads=2)
    # Moderate decay keeps the demonstration outside the package's special
    # clipped-sqrt derivative regime; that is a stated comparison boundary.
    with torch.no_grad():
        layer.a_param.zero_()
    inputs = torch.randn(2, 7, 8, requires_grad=True)
    positions = torch.tensor([[0, 1, 2, 0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6]])
    arguments = (layer.input_gate.w, layer.input_gate.b, layer.a_gate.w,
                 layer.a_gate.b, layer.a_param)
    manual, manual_state = rglru_reference(inputs, positions, *arguments)
    native, native_state = layer(inputs, positions)
    torch.testing.assert_close(manual, native, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(manual_state, native_state, atol=2e-6, rtol=2e-5)
    first, cache = layer(inputs[:, :4], positions[:, :4])
    second, _ = layer(inputs[:, 4:], positions[:, 4:], cache=cache)
    torch.testing.assert_close(native, torch.cat([first, second], dim=1))
    variables = (inputs, *arguments)
    left = torch.autograd.grad(manual.square().sum(), variables, retain_graph=True)
    right = torch.autograd.grad(native.square().sum(), variables)
    for manual_gradient, native_gradient in zip(left, right):
        torch.testing.assert_close(manual_gradient, native_gradient, atol=2e-5, rtol=2e-4)
    print("RG-LRU matched-state forward, reset, chunk and gradient comparisons passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--griffin", action="store_true")
    arguments = parser.parse_args()
    check_segmented_attention()
    check_latent_read()
    if arguments.griffin:
        check_griffin_component()
