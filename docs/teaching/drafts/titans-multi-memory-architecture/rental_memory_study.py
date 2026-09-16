"""Predeclared, offline, predict-observe-write comparison on real daily counts."""
import csv
import hashlib
import json
import platform
from datetime import date
from pathlib import Path
import numpy as np
import torch
from neural_memory import (copy_parameters, initialize_memory,
                           read_memory, write_memory)

ROOT = Path(__file__).resolve().parent
SOURCE_HASH = 'a6bcf826782d3c0fbfdcbeead17cd0884185a0dafe8ff10cd48a874ee7ba18be'


def load_observations():
    source = ROOT / 'bike-sharing-daily.csv'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == SOURCE_HASH
    with source.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    days = [date.fromisoformat(row['dteday']) for row in rows]
    counts = np.array([float(row['cnt']) for row in rows])
    assert len(rows) == 731 and all((b - a).days == 1 for a, b in zip(days, days[1:]))
    assert all(int(row['casual']) + int(row['registered']) == int(row['cnt']) for row in rows)
    return days, counts


def make_keys(days, counts, mean, scale):
    keys = []
    for target in range(7, len(counts)):
        phase = 2 * np.pi * days[target].weekday() / 7
        key = np.r_[(counts[target-7:target] - mean) / scale,
                    np.sin(phase), np.cos(phase)]
        keys.append(key / np.linalg.norm(key))
    return torch.tensor(np.array(keys), dtype=torch.float64)


def fit_initial(keys, targets, seed):
    parameters = initialize_memory(seed)
    optimizer = torch.optim.Adam(parameters, lr=0.01)
    for _ in range(1000):
        optimizer.zero_grad()
        loss = 0.5 * (read_memory(parameters, keys) - targets).square().mean()
        loss.backward()
        optimizer.step()
    return copy_parameters(parameters), float(loss.detach())


def replay(parameters, keys, targets, start=365, stop=731, momentum=None,
           rate=0.005, retention=0.5, decay=0.0001):
    weights = copy_parameters(parameters)
    state = (tuple(torch.zeros_like(weight) for weight in weights)
             if momentum is None else tuple(value.clone() for value in momentum))
    records = []
    for target_index in range(start, stop):
        key = keys[target_index - 7]
        prediction = float(read_memory(weights, key).detach())
        next_weights, next_state, loss, gradients = write_memory(
            weights, state, key, targets[target_index], rate, retention, decay)
        gradient_norm = torch.sqrt(sum(gradient.square().sum() for gradient in gradients))
        update_norm = torch.sqrt(sum((new-old).square().sum()
                                     for new, old in zip(next_weights, weights)))
        records.append({'index': target_index, 'prediction_z': prediction,
                        'loss_before_write': float(loss.detach()),
                        'gradient_norm': float(gradient_norm.detach()),
                        'update_norm': float(update_norm.detach())})
        weights, state = next_weights, next_state
    return records, weights, state


def score(predictions, observed):
    error = np.asarray(predictions) - observed
    return {'mae': float(np.abs(error).mean()),
            'rmse': float(np.sqrt(np.mean(error**2)))}


def run_study():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    days, counts = load_observations()
    mean, scale = float(counts[:365].mean()), float(counts[:365].std())
    keys = make_keys(days, counts, mean, scale)
    targets = torch.tensor((counts - mean) / scale, dtype=torch.float64)
    result = {'environment': {'python': platform.python_version(),
                             'numpy': np.__version__, 'torch': torch.__version__,
                             'device': 'cpu', 'dtype': 'float64', 'threads': 1},
              'source_sha256': SOURCE_HASH, 'mean': mean, 'scale': scale,
              'fit_examples': 358, 'windows': {}, 'seeds': {}}
    for name, lower, upper in [('development', 365, 548), ('assessment', 548, 731)]:
        result['windows'][name] = {'first': str(days[lower]), 'last': str(days[upper-1]),
            'n': upper-lower,
            'previous_day': score(counts[lower-1:upper-1], counts[lower:upper]),
            'previous_week': score(counts[lower-7:upper-7], counts[lower:upper])}
    for seed in [3, 7, 19]:
        fitted, fit_loss = fit_initial(keys[:358], targets[7:365], seed)
        trace, final, state = replay(fitted, keys, targets)
        frozen = read_memory(fitted, keys[358:]).detach().numpy() * scale + mean
        adaptive = np.array([row['prediction_z'] for row in trace]) * scale + mean
        windows = {}
        for name, lower, upper in [('development', 0, 183), ('assessment', 183, 366)]:
            observed = counts[365+lower:365+upper]
            windows[name] = {'frozen': score(frozen[lower:upper], observed),
                             'adaptive': score(adaptive[lower:upper], observed)}
        for offset, row in enumerate(trace):
            row.update(date=str(days[row['index']]), observed=float(counts[row['index']]),
                       frozen=float(frozen[offset]), adaptive=float(adaptive[offset]))
        result['seeds'][str(seed)] = {'fit_loss': fit_loss, 'windows': windows,
            'initial_parameters': [value.detach().tolist() for value in fitted],
            'final_parameters': [value.detach().tolist() for value in final],
            'final_momentum': [value.tolist() for value in state], 'trace': trace}
        print(seed, 'development', windows['development'], 'assessment', windows['assessment'])
    (ROOT / 'rental-results.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    run_study()
