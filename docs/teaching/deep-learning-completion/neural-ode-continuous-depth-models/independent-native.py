"""Independent SciPy and functional PyTorch references; imports no lesson code."""
from pathlib import Path
import json
import numpy as np
import scipy
from scipy.linalg import expm
import torch
from torch.nn.functional import linear

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PACKET = ROOT / 'public/learn-assets/neural-ode-continuous-depth-models'
torch.set_num_threads(2)
torch.set_default_dtype(torch.float64)


def reference(snapshot, raw, metadata, steps, method, endpoint):
    weights = {key: torch.tensor(value) for key, value in snapshot['state'].items()}
    initial = torch.tensor(raw, requires_grad=True)
    z = (initial - torch.tensor(metadata['mean'])) / torch.tensor(metadata['scale'])
    if snapshot['kind'] == 'augmented_ode':
        z = torch.cat((z, torch.zeros(2)))
    history = [z]
    # Generic explicit Butcher tableau implementation, separate from the
    # lesson's four hand-written state equations and nn.Module hierarchy.
    a, b, c = ([[]], [1.], [0.]) if method == 'euler' else (
        [[], [.5], [0., .5], [0., 0., 1.]], [1/6, 1/3, 1/3, 1/6], [0., .5, .5, 1.])
    h = endpoint / steps
    for interval in range(steps):
        stages = []
        for row, clock in zip(a, c):
            probe = z + h * sum((coefficient * derivative for coefficient, derivative in zip(row, stages)), torch.zeros_like(z))
            probe = torch.cat((probe, torch.tensor([(interval + clock) * h])))
            hidden = torch.tanh(linear(probe, weights['field.input.weight'], weights['field.input.bias']))
            stages.append(linear(hidden, weights['field.output.weight'], weights['field.output.bias']))
        z = z + h * sum((coefficient * derivative for coefficient, derivative in zip(b, stages)), torch.zeros_like(z))
        history.append(z)
    logits = linear(z, weights['readout.weight'], weights['readout.bias'])
    probabilities = torch.softmax(logits, dim=0)
    gradient = torch.autograd.grad(probabilities[2], initial)[0]
    return {'trace': torch.stack(history).detach().tolist(), 'logits': logits.detach().tolist(),
            'probabilities': probabilities.detach().tolist(), 'class2_gradient_per_cm': gradient.tolist()}


def main():
    matrices = [[[0., 3.], [0., 0.]], [[-2., 3.], [0., -2.]], [[.7, -2.9], [2.9, .7]],
                [[2.9, -2.7], [-.3, -1.8]], [[1., 1.], [1e-10, 1.]],
                [[1., 1.], [-1e-10, 1.]], [[0., 0.], [0., 0.]],
                [[2.5, -3.], [3., -2.5]], [[-3., -3.], [-3., -3.]]]
    matrix_cases = [{'matrix': a, 'time': t, 'exponential': expm(np.array(a) * t).tolist()}
                    for a in matrices for t in [-1.3, 0., .00001, .13, .83, 2.]]
    models = json.loads((PACKET / 'fitted-models.json').read_text(encoding='utf-8'))
    metadata = json.loads((PACKET / 'study-results.json').read_text(encoding='utf-8'))['data']
    raw_cases = [[4.87, 3.22, 1.66, .34], [6.17, 2.81, 4.73, 1.53], [7.21, 3.08, 6.42, 2.21]]
    cases = []
    for model in models:
        if 'ode' not in model['kind']:
            continue
        for raw in raw_cases:
            for method, steps, endpoint in [('euler', 3, .73), ('rk4', 7, 1.31), ('rk4', 4, 1.), ('euler', 16, 1.)]:
                cases.append({'kind': model['kind'], 'seed': model['seed'], 'raw': raw,
                              'method': method, 'steps': steps, 'endpoint': endpoint,
                              **reference(model, raw, metadata, steps, method, endpoint)})
    result = {'passed': True, 'torch': torch.__version__, 'scipy': scipy.__version__, 'threads': 2,
              'matrix_cases': matrix_cases, 'classifier_cases': cases,
              'reference': 'SciPy expm and independently implemented functional Torch Butcher-tableau solve; no lesson imports or fitting.'}
    (HERE / 'independent-native-results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'matrixCases': len(matrix_cases), 'fullNetworkCases': len(cases), 'inputGradients': 4 * len(cases)}))


if __name__ == '__main__':
    main()
