"""Native integration checks. Full sweep is opt-in; the current run already exists.

Use --mup-path for an isolated mup 1.0.0 target. No shared package mutation.
"""
import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import platform
import re
import runpy
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--mup-path')
parser.add_argument('--replay-full', action='store_true')
args = parser.parse_args()
if args.mup_path:
    sys.path.insert(0, str(Path(args.mup_path).resolve()))
import numpy as np
import sklearn
import torch
torch.set_num_threads(1)
root = Path(__file__).resolve().parents[1]
packet = root / 'docs/teaching/drafts/weight-initialization-xavier-kaiming-p'
assets = root / 'public/learn-assets/weight-initialization-xavier-kaiming-p'
evidence = root / 'docs/teaching/evidence/weight-initialization-native.json'
evidence.write_text(json.dumps({'passed': False, 'status': 'running'}), encoding='utf-8')
checks = []
hash_file = lambda file: hashlib.sha256(file.read_bytes()).hexdigest()
if args.replay_full:
    runpy.run_path(str(assets / 'initialization-experiments.py'), run_name='__main__')
actual = json.loads((assets / 'calculated-inputs.json').read_text(encoding='utf-8'))
expected = json.loads((packet / 'calculated-inputs.json').read_text(encoding='utf-8'))
assert actual == expected
assert hash_file(assets / 'initialization-experiments.py') == hash_file(packet / 'initialization-experiments.py')
assert hash_file(assets / 'digits-400.csv') == hash_file(packet / 'digits-400.csv')
checks.append('Complete saved 18-digit/54-width/18-probe output equals the conserved packet exactly; producer and data bytes unchanged')
output = io.StringIO()
with contextlib.redirect_stdout(output):
    bridge = runpy.run_path(str(assets / 'initialization_library_bridge.py'))
    bridge['check_orthogonal']()
    bridge['check_mup']()
checks.append('Actual mup1.0.0 outputs, gradients and two Adam updates match the explicit model at widths32/96')
matrix = bridge['orthogonal_matrix'](3, 7, gain=.5, generator=torch.Generator().manual_seed(21))
torch.testing.assert_close(matrix @ matrix.T, .25 * torch.eye(3, dtype=torch.float64), atol=1e-12, rtol=1e-12)
assert torch.linalg.matrix_rank(matrix).item() == 3
checks.append('Changed rectangular exercise:3x7 gain0.5 has row Gram0.25I and rank3')
experiments = runpy.run_path(str(assets / 'initialization-experiments.py'))
model = experiments['WidthMLP'](160, 'mu', 7)
assert model.multiplier == 5
assert all(math.isclose(group['lr'], expected, rel_tol=1e-14, abs_tol=1e-16)
           for group, expected in zip(model.optimizer(.003).param_groups, [.003, .0006, .003]))
checks.append('Changed width160 exercise checks multiplier5 and separate Adam group rates')
namespace = {}
with contextlib.redirect_stdout(output):
    for block in re.findall(r'```python\n(.*?)\n```', (packet / 'lesson.md').read_text(encoding='utf-8'), re.S):
        exec(compile(block, 'prepared-lesson-python', 'exec'), namespace)
assert tuple(namespace['logits'].shape) == (2, 3)
checks.append('Both inline Python blocks execute; ordinary nn.init example produces the declared2x3 logits')
assert len(actual['digit_fits']) == 18 and len(actual['width_fits']) == 54 and len(actual['propagation']['records']) == 18
for seed in (1, 2, 3):
    for rate in (.001, .003, .01):
        runs = [row['trace'] for row in actual['width_fits'] if row['width'] == 32 and row['seed'] == seed and row['rate'] == rate]
        assert len(runs) == 2 and runs[0] == runs[1]
checks.append('All9 base-width standard/mu traces are exactly identical')
record = {'passed': True, 'scope': 'Native phase-two checks; source-bound. --replay-full reruns the full sweep. Initial22September implementation ran that full producer before this verification.',
          'fullReplayInThisCommand': args.replay_full,
          'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__, 'sklearn': sklearn.__version__, 'mup': importlib.metadata.version('mup')},
          'checks': checks, 'stdout': output.getvalue(),
          'sources': {str(file.relative_to(root)).replace('\\', '/'): hash_file(file) for file in [assets/'initialization-experiments.py', assets/'initialization_library_bridge.py', assets/'digits-400.csv', assets/'calculated-inputs.json']},
          'limits': ['CPU-only teaching experiments, no benchmark speed or generalization claim', 'LSUV/Fixup are optional explanations, not implemented experimental procedures', 'No arbitrary-architecture muP or SGD equivalence claim']}
serialized = json.dumps(record, indent=2) + '\n'
evidence.write_text(serialized, encoding='utf-8')
(assets / 'native-verification.json').write_text(serialized, encoding='utf-8')
print(json.dumps({'passed': True, 'checks': checks, 'environment': record['environment']}, indent=2))
