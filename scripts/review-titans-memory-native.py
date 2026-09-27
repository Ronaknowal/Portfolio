"""Independent chained-write/autograd and changed-arrival oracles; no fitting."""
import json
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / 'docs/teaching/drafts/titans-multi-memory-architecture'
sys.path.insert(0, str(PACKET))
from neural_memory import initialize_memory, read_memory, write_memory
from rental_memory_study import load_observations, make_keys, replay

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
array = lambda tensors: [value.detach().tolist() for value in tensors]
result = {'environment': {'torch': torch.__version__, 'threads': 1, 'dtype': 'float64'}, 'chains': [], 'arrivals': []}
for dimension, hidden, seed in [(2, 5, 101), (2, 5, 202), (9, 4, 101), (9, 4, 202)]:
    generator = torch.Generator().manual_seed(seed + 800)
    initial = initialize_memory(seed, dimension, hidden)
    momentum = tuple(torch.randn(p.shape, generator=generator, dtype=torch.float64) * .03 for p in initial)
    keys = torch.randn((3, dimension), generator=generator, dtype=torch.float64) / dimension**.5
    targets = torch.randn(3, generator=generator, dtype=torch.float64)
    query = torch.randn(dimension, generator=generator, dtype=torch.float64)
    rate = torch.tensor(.07, dtype=torch.float64, requires_grad=True)
    record = {'initial': array(initial), 'initialMomentum': array(momentum), 'keys': keys.tolist(), 'targets': targets.tolist(), 'query': query.tolist(), 'rate': .07, 'retention': .3, 'decay': .02, 'steps': []}
    parameters = initial
    for key, target in zip(keys, targets):
        parameters, momentum, loss, gradient = write_memory(parameters, momentum, key, target, rate, .3, .02, differentiable=True)
        record['steps'].append({'parameters': array(parameters), 'momentum': array(momentum), 'loss': loss.detach().item(), 'gradient': array(gradient)})
    output = read_memory(parameters, query)
    loss = .5 * (output + .4).square()
    derivatives = torch.autograd.grad(loss, (rate, *initial))
    record.update(output=output.detach().item(), loss=loss.detach().item(), rateGradient=derivatives[0].item(), initialGradient=array(derivatives[1:]))
    result['chains'].append(record)

days, original = load_observations()
mean, scale = original[:365].mean(), original[:365].std()
saved = json.loads((PACKET / 'rental-results.json').read_text())
for seed in ['3', '7', '19']:
    snapshot = saved['seeds'][seed]
    for day, value in [(365, 20000), (547, 0), (730, 20000)]:
        counts = original.copy()
        counts[day] = value
        parameters = tuple(torch.tensor(p, dtype=torch.float64, requires_grad=True) for p in snapshot['initial_parameters'])
        trace, parameters, momentum = replay(parameters, make_keys(days, counts, mean, scale), torch.tensor((counts-mean)/scale, dtype=torch.float64))
        result['arrivals'].append({'seed': seed, 'day': day, 'value': value, 'predictions': [r['prediction_z'] for r in trace], 'parameters': array(parameters), 'momentum': array(momentum)})
path = ROOT / 'docs/teaching/deep-learning-completion/titans-multi-memory-architecture/independent-native-fixtures.json'
path.write_text(json.dumps(result, separators=(',', ':'))+'\n', encoding='utf-8')
print(json.dumps({'passed': True, 'chainedNativeCases': len(result['chains']), 'changedRealArrivals': len(result['arrivals']), 'fits': 0, 'output': str(path)}))
