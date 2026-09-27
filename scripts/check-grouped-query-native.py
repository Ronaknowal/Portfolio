"""Replay supplied native operators and selected models without refitting them.

Run with scratch/lesson-tools/Scripts/python.exe. Writes topic evidence only;
temporary mechanism output lives in a TemporaryDirectory and is removed.
"""
from pathlib import Path
import hashlib
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
ID = 'grouped-query-attention-gqa-multi-query-attention-mqa'
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
    with tempfile.TemporaryDirectory(prefix='gqa-native-') as temporary:
        target = Path(temporary) / 'mechanism-calculations.py'
        shutil.copyfile(DRAFT / target.name, target)
        process = subprocess.run([sys.executable, str(target)], capture_output=True, text=True, check=True)
        fresh = json.loads((target.parent / 'mechanism-fixtures.json').read_text())
        assert fresh['native_checks']['native_vs_grouped_max_error'] < 1e-12
        checks.append({'name': 'Executed complete supplied NumPy/PyTorch operator, masking, allocation and shared-gradient program', 'passed': True, 'stdout': process.stdout})
        code = (DRAFT / 'lesson.md').read_text(encoding='utf-8').split('```python')[2].split('```')[0]
        # Locate the complete displayed reference rather than the preceding einsum fragment.
        programs = (DRAFT / 'lesson.md').read_text(encoding='utf-8').split('```python')[1:]
        reference = next(part.split('```')[0] for part in programs if 'def grouped_attention' in part)
        reference_file = Path(temporary) / 'displayed.py'
        reference_file.write_text(reference, encoding='utf-8')
        displayed = subprocess.run([sys.executable, str(reference_file)], capture_output=True, text=True, check=True)
        assert displayed.stdout.count('True') == 2
        checks.append({'name': 'Executed exact complete displayed NumPy program', 'passed': True, 'stdout': displayed.stdout})
    study = load_module('author-calculations')
    raw, points, pairs, split, duplicates = study.load_records()
    models = json.loads((DRAFT / 'forecast-models.json').read_text())
    cases = []
    with torch.no_grad():
        for heads in [4, 2, 1]:
            saved = models[str(heads)]
            model = study.CausalForecaster(heads)
            model.load_state_dict({key: torch.tensor(value) for key, value in saved['weights'].items()})
            model.eval()
            for length, edit, shift in [(32, False, 0), (32, True, 0), (2, False, 0), (40, True, 100)]:
                source = points[76:77, :length].clone()
                if edit:
                    source[:, min(23, length - 1), 0] *= -1
                prediction, cache, trace = model(source, torch.arange(shift, shift + length), return_trace=True)
                incremental = []
                current_cache = None
                for index in range(length):
                    step, current_cache = model(source[:, index:index + 1], torch.tensor([shift + index]), current_cache)
                    incremental.append(step)
                assert float((torch.cat(incremental, 1) - prediction).abs().max()) < 2e-6
                rollout = [prediction[:, -1:]]
                current_cache = cache
                for step in range(4):
                    predicted, current_cache = model(rollout[-1], torch.tensor([shift + length + step]), current_cache)
                    rollout.append(predicted)
                cases.append({'heads': heads, 'length': length, 'edited': edit, 'shift': shift,
                              'points': ((source[0] + 1) / 2).tolist(),
                              'prediction': ((prediction[0] + 1) / 2).tolist(),
                              'rollout': ((torch.cat(rollout, 1)[0] + 1) / 2).tolist(),
                              'weights': trace['attention'][0, :, -1].tolist(),
                              'headOutputs': trace['head_outputs'][0, :, -1].tolist()})
            baseline = model(points[76:77, :32])[0]
            assert np.max(np.abs(((baseline[0, -1] + 1) / 2).numpy() - saved['investigation']['next_prediction_original_coordinates'])) < 2e-6
            edited = points[76:77, :32].clone()
            edited[:, 23, 0] *= -1
            assert torch.equal(baseline[:, :23], model(edited)[0][:, :23])
            recorded = json.loads((DRAFT / 'author-results.json').read_text())['branches'][str(heads)]['metrics']
            actual = study.metrics(model, pairs)
            for split_name in actual:
                assert abs(actual[split_name]['rmse_original_coordinate'] - recorded[split_name]['rmse_original_coordinate']) < 2e-6
    checks.append({'name': 'Reloaded all three saved models, replayed held-out metrics, exact causal-prefix null and full/incremental equivalence on twelve cases', 'passed': True})
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / 'native-fixtures.json').write_text(json.dumps({'cases': cases, 'mechanisms': fresh}, separators=(',', ':')))
    receipt = {'topicId': ID, 'passed': True, 'checks': checks, 'environment': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__, 'threads': 2}, 'limitations': ['Reused original training and checkpoint-selection evidence; no new fits or GPU performance measurements.', 'Browser checks are separate.']}
    (EVIDENCE / 'native-checks.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('GQA native verification passed: complete programs, 3 selected models, 12 inference cases, all held-out metric replays.')


if __name__ == '__main__':
    main()
