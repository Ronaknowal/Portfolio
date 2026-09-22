"""Execute the prepared scratch/library contracts against the retained model states."""
from pathlib import Path
import hashlib
import json
import platform
import runpy
import numpy as np
import sklearn
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
ID = 'depthwise-separable-dilated-convolutions'
ASSETS = ROOT / 'public/learn-code' / ID
PACKET = ROOT / 'docs/teaching/drafts' / ID
REPORT = ROOT / 'docs/teaching/evidence/depthwise-convolution-native.json'
REPORT.write_text(json.dumps({'passed': False, 'status': 'running'}), encoding='utf-8')
torch.set_num_threads(1)
checks = []
for filename in ['convolution-factorization.py', 'context-blocks.py', 'author-checks.py', 'digits-400.csv', 'calculated-inputs.json']:
    assert (ASSETS / filename).read_bytes() == (PACKET / filename).read_bytes()
checks.append('Canonical downloadable programs, CSV and measured six-run record equal frozen packet bytes')
reference = runpy.run_path(str(ASSETS / 'author-checks.py'))
reference['main']()
actual = json.loads((ASSETS / 'author-check-results.json').read_text())
expected = json.loads((PACKET / 'author-check-results.json').read_text())
assert actual == expected
checks.append('Reexecuted every direct-loop, gradient, geometry, rank, boundary, saved-inference and practice calculation; full JSON identical')
blocks = runpy.run_path(str(ASSETS / 'context-blocks.py'))
blocks['main']()
assert json.loads((ASSETS / 'block-check-results.json').read_text()) == json.loads((PACKET / 'block-check-results.json').read_text())
checks.append('Mobile/context building blocks execute real input, forward shapes and finite nonzero stem gradients')
source = runpy.run_path(str(ASSETS / 'convolution-factorization.py'))
records = json.loads((ASSETS / 'calculated-inputs.json').read_text())
maximum_reconstruction_error = 0.
for run in records['runs']:
    assert run['parameters'] == 2886 and run['convolution_linear_macs_per_image'] == 61824
    assert run['trace'][-1]['training']['correct'] == 280
    if run['seed'] != 1:
        continue
    model = source['DigitClassifier'](run['dilation'])
    model.load_state_dict({name: torch.tensor(value) for name, value in run['dense_state'].items()})
    for rank in [1, 2, 4, 9]:
        pair, singular = source['factor_spatial'](model.spatial, rank)
        reconstructed = source['reconstructed_weight'](pair)
        discarded = sum(sum(value * value for value in row[rank:]) for row in singular)
        error_squared = float((reconstructed - model.spatial.weight).square().sum())
        assert np.isclose(discarded, error_squared, atol=1e-9, rtol=1e-5)
        for example in run['examples']:
            image = torch.tensor(example['input'])[None, None]
            with torch.no_grad():
                stem = F.relu(model.stem(image))
                grouped = pair(stem)
                dense = F.conv2d(stem, reconstructed, pair[1].bias, padding=run['dilation'], dilation=run['dilation'])
            difference = float((grouped - dense).abs().max())
            maximum_reconstruction_error = max(maximum_reconstruction_error, difference)
            torch.testing.assert_close(grouped, dense, atol=1e-5, rtol=1e-5)
            if rank == 9:
                torch.testing.assert_close(reconstructed, model.spatial.weight, atol=1e-6, rtol=1e-5)
checks.append('Native per-input SVD conversion, spectral-error equality, grouped/effective-kernel output parity and full-rank reconstruction across both retained models and all ranks')
# Execute the complete displayed library equality program without rewriting it.
manuscript = (PACKET / 'lesson.md').read_text(encoding='utf-8')
program = manuscript.split('```python\n', 1)[1].split('\n```', 1)[0]
namespace = {}
exec(compile(program, 'displayed-depthwise-equality.py', 'exec'), namespace)
torch.testing.assert_close(namespace['separated'], namespace['standard'], atol=1e-12, rtol=1e-12)
assert tuple(namespace['filtered'].shape) == (1, 6, 7, 7)
assert tuple(namespace['separated'].shape) == (1, 5, 7, 7)
checks.append('Complete displayed float64 effective-kernel program executes and agrees at 1e-12')
report = {'passed': True, 'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__, 'sklearn': sklearn.__version__}, 'checks': checks, 'maximumFloat32GroupedDifference': maximum_reconstruction_error, 'sourceHashes': {str(file.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(file.read_bytes()).hexdigest() for file in ASSETS.iterdir() if file.is_file()}, 'limits': 'Historical six training runs are preserved source-bound evidence, not rerun for this presentation-only integration. Saved inference, conversion, operators and blocks executed now.'}
REPORT.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'passed': True, 'groups': len(checks), 'maximumFloat32GroupedDifference': maximum_reconstruction_error}))
