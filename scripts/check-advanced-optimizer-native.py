"""Native source execution and saved-state verification, without repeating fits."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import tempfile

import numpy as np
import torch
from torch.nn import functional as F

sys.dont_write_bytecode = True
torch.set_num_threads(2)
torch.set_num_interop_threads(2)
topic = 'advanced-optimizers-lion-sophia-prodigy-schedule-free'
source = Path('docs/teaching/drafts') / topic
output = Path('docs/teaching/deep-learning-completion') / topic
checks, bridges, cases = [], {}, []

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result

with tempfile.TemporaryDirectory(prefix='optimizer-native-') as temporary:
    folder = Path(temporary)
    for filename in ['optimizer_rules.py', 'optimizer_study.py', 'optimizer_calculations.py', 'optimizer_library_bridge.py', 'lion_pytorch.py', 'sophia.py', 'digits-400.csv', 'fitted-optimizer-states.json', 'study-results.json']:
        shutil.copyfile(source / filename, folder / filename)
    sys.path.insert(0, temporary)
    rules = module('optimizer_rules', folder / 'optimizer_rules.py')
    study = module('optimizer_study', folder / 'optimizer_study.py')
    calculations = module('optimizer_calculations', folder / 'optimizer_calculations.py')
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        calculations.main()
    checks.append({'name': 'Complete arithmetic program executes native AdamW parity, finite estimator enumeration, Prodigy expanded histories, stateful real edits and nulls', 'passed': True})
    bridge = module('optimizer_library_bridge', folder / 'optimizer_library_bridge.py')
    for method in ['lion', 'sophia', 'prodigy', 'schedule_free']:
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            bridge.main(method)
        bridges[method] = capture.getvalue()
        checks.append({'name': f'Actual {method} API: 8 updates, checkpoint/RNG/mode resume, matching final logits', 'passed': True})
    # Executed displayed program is exactly the linked bridge (apart from whitespace).
    manuscript = (source / 'lesson.md').read_text(encoding='utf-8')
    displayed = manuscript.split('```python\n', 1)[1].split('\n```', 1)[0]
    assert displayed.strip() == (source / 'optimizer_library_bridge.py').read_text(encoding='utf-8').strip()
    raw, features, labels, roles = study.load_data()
    reports = json.loads((source / 'study-results.json').read_text(encoding='utf-8'))
    snapshots = json.loads((source / 'fitted-optimizer-states.json').read_text(encoding='utf-8'))
    assert len(set(raw[:, 0])) == len(raw) == len(np.unique(raw[:, 1:65], axis=0)) == 400
    assert [len(v) for v in roles.values()] == [240, 80, 80]
    for method, rates in reports['candidates'].items():
        selected = min(rates, key=lambda rate: np.mean([r['validation']['cross_entropy'] for r in reports['runs'] if r['method'] == method and r['rate'] == rate]))
        assert selected == reports['selected'][method]
    for saved in snapshots:
        parameters = np.array(saved['parameters'])
        state = {k: np.array(v) if isinstance(v, list) else v for k, v in saved['state'].items()}
        evaluation = state.get('average', parameters)
        report = next(r for r in reports['runs'] if r['method'] == saved['method'] and r['seed'] == saved['seed'] and r['rate'] == saved['rate'])
        for role in ['validation', 'assessment']:
            actual = rules.metrics(features[roles[role]], labels[roles[role]], evaluation)
            assert actual['correct'] == report[role]['correct']
            assert abs(actual['cross_entropy'] - report[role]['cross_entropy']) < 1e-12
        # New current entities, not the manuscript's saved single-pixel case.
        index = int(np.flatnonzero(raw[:, 0] == 312)[0])
        pixels = raw[index, 1:65].copy()
        pixels[17], pixels[28] = 9, 5
        x = np.r_[pixels / 16, 1][None]
        target = np.array([7])
        optimizer = rules.Optimizer(parameters, saved['method'], saved['rate'])
        optimizer.state = state
        before = rules.probabilities(x, optimizer.evaluation_parameters())[0]
        training = rules.probabilities(x, optimizer.parameters)[0]
        gradient = rules.gradient(x, target, optimizer.parameters)
        tensor = torch.tensor(parameters, dtype=torch.float64, requires_grad=True)
        torch_gradient, = torch.autograd.grad(F.cross_entropy(torch.tensor(x) @ tensor, torch.tensor(target)), tensor)
        np.testing.assert_allclose(gradient, torch_gradient.numpy(), atol=1e-13, rtol=1e-13)
        curvature = x[0, :, None] ** 2 * (training * (1 - training))[None, :] if saved['method'] == 'sophia_g' else None
        diagnostics = optimizer.step(gradient, curvature)
        cases.append({'method': saved['method'], 'seed': saved['seed'], 'pixels': pixels.tolist(), 'target': 7, 'before': before.tolist(), 'training': training.tolist(), 'gradient': gradient.tolist(), 'after': rules.probabilities(x, optimizer.evaluation_parameters())[0].tolist(), 'parameters': optimizer.parameters.tolist(), 'evaluationParameters': optimizer.evaluation_parameters().tolist(), 'state': optimizer.state, 'diagnostics': diagnostics})
    checks.append({'name': 'All 12 saved final models reproduce validation/assessment loss and correctness; all24 candidates retain validation-only selection', 'passed': True})
    checks.append({'name': 'All12 new pixel/label gradients match independent torch autograd; full stateful diagnostic updates exported', 'passed': True})
    # No-gradient is distinct from a zero current gradient after momentum exists.
    p = torch.tensor([.5], dtype=torch.float64, requires_grad=True)
    lion = bridge.make_optimizer([p], 'lion')
    p.grad = torch.tensor([1.], dtype=torch.float64); lion.step()
    saved = p.detach().clone(); p.grad = None; lion.step(); torch.testing.assert_close(p, saved, rtol=0, atol=0)
    p.grad = torch.zeros_like(p); lion.step(); assert not torch.equal(p, saved)
    checks.append({'name': 'Official Lion skips absent gradient but uses retained momentum for zero gradient', 'passed': True})
    calculated = json.loads((folder / 'calculated-inputs.json').read_text(encoding='utf-8'))

def serializable(value):
    return value.tolist() if isinstance(value, np.ndarray) else value.item() if isinstance(value, np.generic) else str(value)

(output / 'native-fixtures.json').write_text(json.dumps({'calculated': calculated, 'cases': cases}, default=serializable) + '\n', encoding='utf-8')
receipt = {'topicId': topic, 'passed': True, 'environment': {'torch': torch.__version__, 'numpy': np.__version__, 'maximumThreads': 2, 'python': sys.version}, 'checks': checks, 'bridgeOutputs': bridges, 'limitations': ['Historical24 fits reused; no new model fitting or hardware timing.', 'Paper Prodigy/Schedule-Free rules and maintained package variants are not falsely equated.']}
(output / 'native-checks.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps(receipt, indent=2))
