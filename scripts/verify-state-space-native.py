"""Native source-bound inference and mechanism checks; no training replay."""
from pathlib import Path
import hashlib
import json
import runpy
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'public/learn-code/state-space-models-s4-mamba-mamba-2'
destination = ROOT / 'docs/teaching/evidence/state-space-native.json'
destination.write_text(json.dumps({'passed': False, 'reason': 'Verification running'})+'\n')
torch.set_num_threads(2)
models = runpy.run_path(str(FOLDER / 'trajectory_state_models.py'))
mechanisms = runpy.run_path(str(FOLDER / 'state_space_mechanisms.py'))
x, labels, roles, groups = models['prepare'](FOLDER)
parameters = np.load(FOLDER / 'trajectory-state-fits.npz')
cases = []
maximum = 0.
for kind in ('diagonal', 'selective'):
    prefix = f'{kind}_seed17::'
    model = models['TrajectoryClassifier'](kind)
    model.load_state_dict({key[len(prefix):]: torch.from_numpy(parameters[key])
                           for key in parameters.files if key.startswith(prefix) and not key.endswith('logits')})
    model.eval()
    with torch.inference_mode():
        actual = model(x[roles[1]], recurrent=True)
        saved = torch.from_numpy(parameters[prefix + 'logits'][roles[1]])
        difference = float((actual - saved).abs().max())
        assert difference < 1e-4
        maximum = max(maximum, difference)
        for row, output in zip(roles[1], actual):
            cases.append({'kind': kind, 'row': int(row + 1), 'edit': 'original', 'logits': output.tolist()})
        for edit in ('reflect', 'reverse', 'zero', 'one'):
            changed = x[6:7].clone()
            if edit == 'reflect': changed[0, 22, 0] *= -1
            if edit == 'reverse': changed = changed.flip(1)
            if edit == 'zero': changed[:] = -1  # unit coordinates zero
            if edit == 'one': changed[:] = 1
            cases.append({'kind': kind, 'row': 7, 'edit': edit, 'logits': model(changed, recurrent=True)[0].tolist()})
Ad, Bd = mechanisms['held_discretize'](np.zeros((2, 2)), np.eye(2), .5)
assert np.allclose(Ad, np.eye(2)) and np.allclose(Bd, .5*np.eye(2))
rng = np.random.default_rng(316)
for length in (1, 7, 16):
    A = np.diag(rng.uniform(-3, 0, 2)); B = rng.normal(size=(2, 1)); C = rng.normal(size=2)
    u = rng.normal(size=length); initial = rng.normal(size=2)
    Ad, Bd = mechanisms['held_discretize'](A, B, .7)
    recurrent, _ = mechanisms['recurrent'](Ad, Bd, C, .3, u, initial)
    taps = mechanisms['kernel'](Ad, Bd, C, length)
    response = np.array([C @ np.linalg.matrix_power(Ad, t+1) @ initial for t in range(length)])
    assert np.max(np.abs(recurrent - mechanisms['fft_convolve'](u, taps) - .3*u - response)) < 1e-10
report = {'passed': True, 'cases': cases, 'validationRows': len(roles[1]),
          'saved_vs_fresh_max_logit_error': maximum,
          'nativeChecks': ['singular block exponential', 'initial-state FFT comparison at lengths 1/7/16',
                           'both frozen seed-17 models on all validation rows plus reflection/reversal/zero/one edits'],
          'cudaExecuted': False, 'torch': torch.__version__,
          'sourceHashes': {name: hashlib.sha256((FOLDER / name).read_bytes()).hexdigest()
                           for name in ['trajectory_state_models.py', 'state_space_mechanisms.py', 'trajectory-state-fits.npz', 'movement_libras.data']}}
report['verifierSha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
destination.write_text(json.dumps(report, indent=2)+'\n')
print(f'Passed native mechanisms and {len(cases)} fitted inference cases; CUDA bridge not executed.')
