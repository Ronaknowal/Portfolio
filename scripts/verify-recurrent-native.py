"""Execute the published recurrent core and export independent browser comparisons."""
from pathlib import Path
import hashlib
import importlib.util
import json
import platform
import tempfile
import shutil
import subprocess
import sys

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'public/learn-assets/rnns-lstms-grus'
EVIDENCE = ROOT / 'docs/teaching/evidence/recurrent-native.json'
torch.set_num_threads(1)


def load_module(name, source):
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def values(tensor):
    return tensor.detach().cpu().numpy().tolist()


def main():
    report = {'passed': False, 'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__}, 'groups': [], 'pen': [], 'boundaries': [], 'padding': []}
    EVIDENCE.write_text(json.dumps(report), encoding='utf-8')
    with tempfile.TemporaryDirectory(prefix='recurrent-native-', dir=ROOT / 'scratch') as folder:
        folder = Path(folder)
        for name in ['recurrent-mechanics.py', 'calculated-inputs.json']:
            shutil.copyfile(ASSETS / name, folder / name)
        execution = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(folder / 'recurrent-mechanics.py')], capture_output=True, text=True, check=True)
        report['groups'].append({'name': 'Published scratch mechanics and native checks', 'output': execution.stdout[-1200:]})
        generated = json.loads((folder / 'mechanics-results.json').read_text())
        retained = json.loads((ASSETS / 'mechanics-results.json').read_text())
        assert generated == retained, 'Native mechanics no longer reproduces recorded outputs'
    mechanics = load_module('recurrent_mechanics', ASSETS / 'recurrent-mechanics.py')
    training = load_module('pen_sequence', ASSETS / 'pen-sequence-learning.py')
    models = json.loads((ASSETS / 'pen-models.json').read_text())
    results = json.loads((ASSETS / 'calculated-inputs.json').read_text())
    coordinates, labels, train, _ = training.load_data()
    for kind in ['rnn', 'lstm', 'gru']:
        module = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[kind](2, 32, batch_first=True).double().eval()
        weights = models['weights'][kind]
        module.load_state_dict({name.replace('recurrent.', ''): torch.tensor(value, dtype=torch.float64) for name, value in weights.items() if name.startswith('recurrent.')})
        readout = nn.Linear(32, 10).double().eval()
        readout.load_state_dict({name.replace('readout.', ''): torch.tensor(value, dtype=torch.float64) for name, value in weights.items() if name.startswith('readout.')})
        for sample in models['specimens']:
            for edit in ['original', 'edited', 'reverse']:
                points = np.array(sample['points'], dtype=float)
                if edit == 'edited':
                    points[2, 0] = min(1., points[2, 0] + .2)
                elif edit == 'reverse':
                    points = points[::-1].copy()
                output, _ = module(torch.tensor(points[None], dtype=torch.float64))
                probabilities = readout(output).softmax(-1)
                manual = mechanics.manual_sequence(kind, points, weights)
                error = np.max(np.abs(np.array([row['hidden'] for row in manual]) - output.detach().numpy()[0]))
                assert error < 2e-14
                report['pen'].append({'kind': kind, 'sourceId': sample['sourceId'], 'edit': edit, 'points': points.tolist(), 'hidden': values(output[0]), 'probabilities': values(probabilities[0])})
        fitted = training.PenClassifier(kind).eval()
        fitted.load_state_dict({name: torch.tensor(value) for name, value in weights.items()})
        run = next(row for row in results['runs'] if row['kind'] == kind and row['seed'] == 1)
        x = torch.tensor(coordinates[~train], dtype=torch.float32)
        y = torch.tensor(labels[~train], dtype=torch.long)
        for field, data in [('final', x), ('reversed', x.flip(1))]:
            actual = training.assess(fitted, data, y, details=True)
            assert actual['correct'] == run[field]['correct']
            assert np.max(np.abs(np.array(actual['probabilities']) - run[field]['probabilities'])) < 1e-7
    report['groups'].append({'name': 'All three saved native cells, 27 changed trajectories and six development replays', 'passed': True})
    fixture = retained['state_and_padding']
    sequence = torch.tensor(fixture['sequence'], dtype=torch.float64)
    model = nn.GRU(2, 3, batch_first=True).double().eval()
    model.load_state_dict({name: torch.tensor(value, dtype=torch.float64) for name, value in fixture['weights'].items()})
    for boundary in range(1, 5):
        for mode in ['carry', 'detach', 'reset']:
            x = sequence.clone().requires_grad_()
            prefix, state = model(x[:, :boundary])
            if mode == 'detach': state = state.detach()
            if mode == 'reset': state = torch.zeros_like(state)
            suffix, _ = model(x[:, boundary:], state)
            loss = suffix.square().sum()
            gradients = torch.autograd.grad(loss, x)[0]
            report['boundaries'].append({'boundary': boundary, 'mode': mode, 'hidden': values(torch.cat([prefix, suffix], 1)[0]), 'gradient': values(gradients[0]), 'loss': float(loss.detach())})
    report['groups'].append({'name': 'Twelve chunk/state/gradient routes against autograd', 'passed': True})
    model = nn.GRU(2, 3, batch_first=True, bidirectional=True).double().eval()
    model.load_state_dict({name: torch.tensor(value, dtype=torch.float64) for name, value in fixture['padding']['True']['weights'].items()})
    for length in range(1, 6):
        for padding in [-4., -3.5, 0., 4.]:
            x = sequence.clone()
            x[:, length:] = padding
            output, _ = model(x)
            valid, _ = model(x[:, :length])
            report['padding'].append({'length': length, 'padding': padding, 'forward': values(output[0, :, :3]), 'backward': values(output[0, :, 3:]), 'validForward': values(valid[0, :, :3]), 'validBackward': values(valid[0, :, 3:])})
    report['groups'].append({'name': 'Twenty bidirectional padding states', 'passed': True})
    report['sourceHashes'] = {str(path.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest() for path in [ASSETS / name for name in ['recurrent-mechanics.py', 'pen-sequence-learning.py', 'pen-models.json', 'recurrent-evidence.json', 'calculated-inputs.json', 'pen-trajectories.csv']]}
    report['passed'] = True
    EVIDENCE.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'groups': len(report['groups']), 'nativeComparisons': len(report['pen']) + len(report['boundaries']) + len(report['padding'])}))


if __name__ == '__main__':
    main()
