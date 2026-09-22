"""Independent review: changed native cases, no training replay.

Needs NumPy, scikit-learn, PyTorch and mup==1.0.0. An isolated installation
may be supplied with --mup-path; no retained scratch folder is required.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import runpy
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--mup-path')
arguments = parser.parse_args()
if arguments.mup_path:
    sys.path.insert(0, str(Path(arguments.mup_path).resolve()))
from mup import MuAdam, MuReadout, set_base_shapes

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
assets = ROOT / 'public/learn-assets/weight-initialization-xavier-kaiming-p'
output = ROOT / 'docs/teaching/evidence/initialization-independent-native.json'
output.write_text(json.dumps({'passed': False}), encoding='utf-8')
bridge = runpy.run_path(str(assets / 'initialization_library_bridge.py'))
program = runpy.run_path(str(assets / 'initialization-experiments.py'))
groups, cases = [], []

# Singular values, rather than repeating the author's rectangular Gram check.
for rows, columns in [(1, 9), (9, 1), (2, 5), (5, 2)]:
    for gain in [0., .2, 1.7]:
        matrix = bridge['orthogonal_matrix'](rows, columns, gain,
                    torch.Generator().manual_seed(43))
        torch.testing.assert_close(torch.linalg.svdvals(matrix),
            torch.full((min(rows, columns),), gain), atol=1e-12, rtol=1e-12)
groups.append('12 rectangular/singleton/zero-gain cases checked by singular values')

# Derivatives produced by autograd, used later to check the JS mechanism.
for x in [-2.3, 0., .7]:
    for target in [-1.2, .4]:
        w = torch.tensor([-.8, .45], requires_grad=True)
        v = torch.tensor([.2, -.9], requires_grad=True)
        prediction = (torch.tanh(w * x) * v).sum()
        loss = .5 * (prediction - target).square()
        dw, dv = torch.autograd.grad(loss, (w, v))
        cases.append(dict(input=x, target=target, weights=w.tolist(), outgoing=v.tolist(),
                          output=prediction.item(), loss=loss.item(),
                          hiddenGradient=dw.tolist(), headGradient=dv.tolist()))
groups.append('Six changed signed/zero-input symmetry gradients via torch autograd')

# An independent ordinary-package bridge at a noninteger width multiplier.
class Library(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.lower = torch.nn.Linear(64, width, bias=False)
        self.upper = torch.nn.Linear(width, width, bias=False)
        self.readout = MuReadout(width, 10, bias=False)
    def forward(self, inputs):
        return self.readout(torch.relu(self.upper(torch.relu(self.lower(inputs)))))

manual = program['WidthMLP'](48, 'mu', seed=19).double()
library = Library(48)
set_base_shapes(library, Library(32), delta=Library(64), rescale_params=False)
library.load_state_dict(manual.state_dict())
left_opt = manual.optimizer(.0017)
right_opt = MuAdam(library.parameters(), lr=.0017, eps=1e-8, weight_decay=0.)
inputs = torch.randn(7, 64, generator=torch.Generator().manual_seed(63))
targets = torch.tensor([9, 8, 3, 0, 0, 6, 1])
for _ in range(3):
    torch.testing.assert_close(manual(inputs), library(inputs), atol=1e-12, rtol=1e-12)
    for model, optimizer in [(manual, left_opt), (library, right_opt)]:
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model(inputs), targets).backward()
    for left, right in zip(manual.parameters(), library.parameters()):
        torch.testing.assert_close(left.grad, right.grad, atol=1e-12, rtol=1e-12)
    left_opt.step()
    right_opt.step()
    for left, right in zip(manual.parameters(), library.parameters()):
        torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-12)
groups.append('Width48 multiplier1.5: changed seed/rate/batch; all outputs, gradients and3Adam updates match mup1.0.0')

record = dict(passed=True, groups=groups, symmetryCases=cases, torch=torch.__version__,
    sources={str(path.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in [assets/'initialization-experiments.py', assets/'initialization_library_bridge.py']},
    limits='No training replay, browser check or arbitrary architecture/SGD equivalence claim.')
output.write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
print(json.dumps(dict(passed=True, groups=groups)))
