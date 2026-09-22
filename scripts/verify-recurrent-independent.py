"""Complementary review: new nonzero states, gate limits and state ownership; no fits."""
from pathlib import Path
import hashlib
import importlib.util
import json
import platform
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'public/learn-assets/rnns-lstms-grus/recurrent-mechanics.py'
spec = importlib.util.spec_from_file_location('reviewed_mechanics', SOURCE)
mechanics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mechanics)
torch.set_num_threads(1)
torch.manual_seed(607)
report = {'passed': False, 'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__}, 'groups': [], 'sequences': [], 'limits': [], 'boundaries': []}
def array(value): return value.detach().tolist()
def error(a, b): return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
def close(a, b):
    difference = error(a, b)
    assert difference < 2e-13, difference
    return difference

# Different shapes/parameters from both the learned handwriting model and author fixtures.
inputs = torch.randn(3, 7, 3, dtype=torch.float64) * .45
hidden = torch.randn(1, 3, 4, dtype=torch.float64) * .6
cell = torch.randn(1, 3, 4, dtype=torch.float64) * .8
largest = 0.
for kind, constructor in [('rnn', nn.RNN), ('lstm', nn.LSTM), ('gru', nn.GRU)]:
    module = constructor(3, 4, batch_first=True).double().eval()
    head = nn.Linear(4, 5).double().eval()
    weights = {'recurrent.' + key: array(value) for key, value in module.state_dict().items()}
    weights.update({'readout.' + key: array(value) for key, value in head.state_dict().items()})
    initial = (hidden, cell) if kind == 'lstm' else hidden
    output, final = module(inputs, initial)
    for sample in range(3):
        manual = mechanics.manual_sequence(kind, array(inputs[sample]), weights, array(hidden[0, sample]), array(cell[0, sample]))
        largest = max(largest, close([row['hidden'] for row in manual], array(output[sample])))
        largest = max(largest, close([row['probabilities'] for row in manual], array(head(output[sample]).softmax(-1))))
        if kind == 'lstm': close(manual[-1]['gates']['cell'], array(final[1][0, sample]))
        report['sequences'].append({'kind': kind, 'points': array(inputs[sample]), 'weights': weights, 'initial': {'hidden': array(hidden[0, sample]), 'cell': array(cell[0, sample])}, 'output': array(output[sample]), 'probabilities': array(head(output[sample]).softmax(-1)), 'finalCell': array(final[1][0, sample]) if kind == 'lstm' else None})
    prefix, state = module(inputs[:, :3], initial)
    suffix, _ = module(inputs[:, 3:], state)
    close(array(torch.cat([prefix, suffix], 1)), array(output))
    order = torch.tensor([2, 0, 1])
    permuted_state = tuple(value[:, order] for value in state) if kind == 'lstm' else state[:, order]
    owned, _ = module(inputs[order, 3:], permuted_state)
    close(array(owned), array(suffix[order]))
    wrong, _ = module(inputs[order, 3:], state)
    assert error(array(wrong), array(suffix[order])) > 1e-4
    if kind == 'lstm':
        missing_cell, _ = module(inputs[:, 3:], (state[0], torch.zeros_like(state[1])))
        assert error(array(missing_cell), array(suffix)) > 1e-3
report['groups'].append({'name': 'Nine nonzero-state sequences across all cells, native/scratch outputs and probabilities', 'maximumDifference': largest})
report['groups'].append({'name': 'All-cell chunk equivalence and correctly permuted stream state; missing LSTM cell changes output', 'passed': True})

# Reset is applied to the hidden affine INCLUDING its bias; z retains the old state.
for reset, retain in [(0, 0), (1, 0), (0, 1), (1, 1)]:
    module = nn.GRU(3, 4, batch_first=True).double().eval()
    with torch.no_grad():
        for parameter in module.parameters(): parameter.zero_()
        module.bias_ih_l0[:4] = 80 if reset else -80
        module.bias_ih_l0[4:8] = 80 if retain else -80
        module.bias_ih_l0[8:] = torch.tensor([.3, -.2, .6, -.4])
        module.bias_hh_l0[8:] = torch.tensor([-.5, .7, .2, -.9])
        module.weight_hh_l0[8:] = torch.tensor([[.2, -.1, .4, .3], [.5, .2, -.6, -.1], [.1, -.3, .7, .2], [-.2, .8, .1, .4]])
    output, _ = module(inputs[:, :1], hidden)
    expected = hidden[0] if retain else torch.tanh(module.bias_ih_l0[8:] + reset * (hidden[0] @ module.weight_hh_l0[8:].T + module.bias_hh_l0[8:]))
    close(array(output[:, 0]), array(expected))
    weights = {'recurrent.' + key: array(value) for key, value in module.state_dict().items()}
    report['limits'].append({'reset': reset, 'retain': retain, 'points': array(inputs[0, :1]), 'initial': {'hidden': array(hidden[0, 0])}, 'weights': weights, 'expected': array(expected[0])})
report['groups'].append({'name': 'Four GRU gate limits with nonzero hidden bias and mixed hidden coordinates', 'passed': True})

# Unsorted packed lengths must preserve both directions and BOTH LSTM state tensors.
module = nn.LSTM(3, 4, batch_first=True, bidirectional=True).double().eval()
lengths = [7, 2, 5]
h0 = torch.randn(2, 3, 4, dtype=torch.float64)
c0 = torch.randn(2, 3, 4, dtype=torch.float64)
padded = inputs.clone()
for i, length in enumerate(lengths): padded[i, length:] = 37.
_, packed_state = module(pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False), (h0, c0))
for i, length in enumerate(lengths):
    _, separate = module(inputs[i:i+1, :length], (h0[:, i:i+1], c0[:, i:i+1]))
    for tensor, expected in zip(packed_state, separate): close(array(tensor[:, i:i+1]), array(expected))
report['groups'].append({'name': 'Unsorted [7,2,5] bidirectional packing matches separate sequences with nonzero h/c', 'passed': True})

module = nn.GRU(3, 4, batch_first=True).double().eval()
weights = {key: array(value) for key, value in module.state_dict().items()}
for mode in ['carry', 'detach', 'reset']:
    x = inputs[:1].clone().requires_grad_()
    prefix, state = module(x[:, :4])
    if mode == 'detach': state = state.detach()
    if mode == 'reset': state = torch.zeros_like(state)
    suffix, _ = module(x[:, 4:], state)
    loss = suffix.square().sum()
    gradients = torch.autograd.grad(loss, x)[0]
    report['boundaries'].append({'mode': mode, 'boundary': 4, 'points': array(x[0]), 'weights': weights, 'output': array(torch.cat([prefix, suffix], 1)[0]), 'gradients': array(gradients[0]), 'loss': float(loss.detach())})
report['groups'].append({'name': 'New three-coordinate seven-position carry/detach/reset derivative fixture', 'passed': True})
report['sourceHashes'] = {str(SOURCE.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(SOURCE.read_bytes()).hexdigest()}
report['passed'] = True
destination = ROOT / 'docs/teaching/evidence/recurrent-independent-native.json'
destination.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'passed': True, 'groups': len(report['groups']), 'maximumScratchNativeDifference': largest}))
