"""Native MLA verification without training; writes only scoped evidence."""
from pathlib import Path
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
sys.dont_write_bytecode = True
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ID = 'multi-head-latent-attention-mla'
DRAFT = ROOT / 'docs/teaching/drafts' / ID
EVIDENCE = ROOT / 'docs/teaching/deep-learning-completion' / ID
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, DRAFT / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    checks = []
    with tempfile.TemporaryDirectory(prefix='mla-native-') as temporary:
        temporary = Path(temporary)
        for name in ['mechanism-calculations.py', 'practice-calculations.py', 'mla_sdpa_bridge.py', 'author-calculations.py', 'forecast-model.json', 'movement_libras.data']:
            shutil.copyfile(DRAFT / name, temporary / name)
        for program in ['mechanism-calculations.py', 'practice-calculations.py', 'mla_sdpa_bridge.py']:
            process = subprocess.run([sys.executable, '-B', str(temporary / program)], capture_output=True, text=True, check=True)
            checks.append({'name': 'Executed complete ' + program, 'passed': True, 'stdout': process.stdout})
        mechanism = json.loads((temporary / 'mechanism-fixtures.json').read_text())
        practice = json.loads((temporary / 'practice-fixtures.json').read_text())
        reference = next(part.split('```')[0] for part in (DRAFT / 'lesson.md').read_text(encoding='utf-8').split('```python')[1:] if 'def mla_read' in part)
        (temporary / 'displayed.py').write_text(reference, encoding='utf-8')
        result = subprocess.run([sys.executable, str(temporary / 'displayed.py')], capture_output=True, text=True, check=True)
        assert result.stdout.count('True') == 2
        checks.append({'name': 'Executed exact displayed NumPy reference', 'passed': True, 'stdout': result.stdout})
    study = load_module('author-calculations')
    saved = json.loads((DRAFT / 'forecast-model.json').read_text())
    model = study.LatentForecaster()
    model.load_state_dict({key: torch.tensor(value) for key, value in saved['weights'].items()})
    model.eval()
    points, pairs, _, _ = study.load_records()
    gradients = study.gradient_equivalence(model, points[76:78, :7])
    assert gradients['all_parameter_gradient_max_error'] < 1e-9
    checks.append({'name': 'Complete same-state expanded/absorbed output and every parameter gradient', 'passed': True, 'result': gradients})
    cases = []
    with torch.no_grad():
        for rank in [None, 8, 4, 2]:
            basis = None if rank is None else torch.tensor(saved['complete_basis'])[:, :rank]
            for length, edit, shift, wrong in [(27, False, 0, False), (27, True, 0, False), (2, False, 100, False), (40, True, 100, True)]:
                prefix = points[76:77, :length].clone()
                if edit:
                    prefix[:, min(19, length - 1), 1] *= -1
                output, _, trace = model(prefix, torch.arange(shift, shift + length), basis=basis, wrong_scale=wrong, return_trace=True)
                expanded, _ = model(prefix, torch.arange(shift, shift + length), basis=basis, wrong_scale=wrong, mode='expanded')
                assert torch.allclose(output, expanded, atol=2e-6, rtol=2e-6)
                cases.append({'rank': rank, 'length': length, 'edited': edit, 'shift': shift, 'wrong': wrong, 'points': ((prefix[0] + 1) / 2).tolist(), 'predictions': ((output[0] + 1) / 2).tolist(), 'weights': trace['attention'][0, :, -1].tolist(), 'heads': trace['head_outputs'][0, :, -1].tolist(), 'content': trace['content_scores'][0, :, -1].tolist(), 'rotary': trace['rotary_scores'][0, :, -1].tolist()})
        recorded = json.loads((DRAFT / 'author-results.json').read_text())
        for label, basis in [('full_metrics', None), ('rank4_intervention_metrics', torch.tensor(saved['rank4_basis']))]:
            actual = study.metrics(model, pairs, basis=basis)
            for split in actual:
                assert abs(actual[split]['rmse_original_coordinate'] - recorded[label][split]['rmse_original_coordinate']) < 2e-6
    checks.append({'name': 'Sixteen selected-model cases and full/rank4 held-out metric replay', 'passed': True})
    matrices = [[[3., 1.], [0., 2.]], [[0., 0.], [0., 0.]], [[1., 0.], [0., 1.]], [[0., 0.], [0., 3.]], [[-2., 4.], [3., -1.]]]
    rank_cases = []
    for matrix in matrices:
        matrix = np.array(matrix)
        for vector in [[2., -3.], [0., 0.], [5., 1.]]:
            _, singular, right = np.linalg.svd(matrix)
            direction = np.array([1., 0.]) if abs(singular[0] - singular[1]) < 1e-12 else right[0]
            reduced = matrix @ np.outer(direction, direction)
            rank_cases.append({'matrix': matrix.tolist(), 'input': vector, 'full': (matrix @ vector).tolist(), 'output': (reduced @ vector).tolist(), 'matrixError': float(np.sum((matrix - reduced)**2))})
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / 'native-fixtures.json').write_text(json.dumps({'cases': cases, 'mechanism': mechanism, 'practice': {key: practice[key] for key in ['I1', 'I2', 'I3', 'I4']}, 'rankCases': rank_cases}, separators=(',', ':')))
    receipt = {'topicId': ID, 'passed': True, 'checks': checks, 'environment': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__, 'threads': 2}, 'limitations': ['Original selected model fitting reused; no new training.', 'No specialized MLA GPU execution or performance measurement.', 'Browser acceptance remains separate.']}
    (EVIDENCE / 'native-checks.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('MLA native verification passed: four complete programs, all model gradients, 16 model cases, 15 rank/input cases and metric replay.')


if __name__ == '__main__':
    main()
