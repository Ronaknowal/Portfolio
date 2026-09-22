"""Independent, bounded residual review. No training campaign or source edits."""
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'public/learn-assets/residual-connections/residual-experiments.py'
REPORT = ROOT / 'docs/teaching/evidence/residual-connections-independent-native.json'
spec = importlib.util.spec_from_file_location('residual_lesson', SOURCE)
lesson = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lesson)
checks = []


def close(actual, expected):
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


# The parameter-shape extension must sum over examples, and sum over features
# only when one scalar is shared. Use a nonzero two-row branch, unlike the prose
# fixture whose first coordinate vanishes.
inputs = torch.tensor([[1., 2.], [-.5, 3.]], dtype=torch.float64)
weights = torch.tensor([[.4, -.1], [.3, .2]], dtype=torch.float64)
targets = torch.tensor([[.2, -.3], [1., .4]], dtype=torch.float64)
branch = inputs @ weights.T
for initial in (-.3, 0., .27):
    scalar = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
    feature = torch.full((2,), initial, dtype=torch.float64, requires_grad=True)
    scalar_output = inputs + scalar * branch
    feature_output = inputs + feature * branch
    scalar_gradient, = torch.autograd.grad(.5 * (scalar_output-targets).square().sum(), scalar)
    feature_gradient, = torch.autograd.grad(.5 * (feature_output-targets).square().sum(), feature)
    expected_feature = ((feature_output-targets)*branch).sum(0)
    close(scalar_output, feature_output)
    close(feature_gradient, expected_feature)
    close(scalar_gradient, feature_gradient.sum())
    epsilon = 1e-6
    objective = lambda scale: .5 * (inputs + scale * branch-targets).square().sum()
    torch.testing.assert_close((objective(initial+epsilon)-objective(initial-epsilon))/(2*epsilon), scalar_gradient, atol=1e-7, rtol=1e-7)
checks.append('Scalar/channel gate gradients: changed two-row inputs, nonzero feature corrections, batch reduction and finite differences')

# An exactly zero ReZero gate preserves both values and derivative, but the
# branch learns only after the gate has actually moved. Execute both updates.
block = lesson.Refinement(4, 'rezero', 6, 917).double()
value = torch.tensor([[.3, -1.2, 2., .7], [-.2, .6, 1.4, -1.]], dtype=torch.float64)
target = torch.tensor([[1., .2, -.5, 0.], [.7, -.4, 0., 1.]], dtype=torch.float64)
close(block(value), value)
close(torch.func.jacrev(lambda v: block(v[None])[0])(value[0]), torch.eye(4, dtype=torch.float64))
optimizer = torch.optim.SGD(block.parameters(), lr=.03)
loss = .5 * (block(value)-target).square().sum()
loss.backward()
initial_gate_gradient = float(block.scale.grad)
assert abs(initial_gate_gradient) > 1e-8
assert torch.count_nonzero(block.lower.weight.grad) == 0
assert torch.count_nonzero(block.upper.weight.grad) == 0
optimizer.step()
optimizer.zero_grad(set_to_none=True)
(.5 * (block(value)-target).square().sum()).backward()
assert block.lower.weight.grad.norm() > 1e-8
assert block.upper.weight.grad.norm() > 1e-8
checks.append('Actual zero gate: identity Jacobian, zero first branch gradient, nonzero gate step, nonzero next branch gradients')

# Registration is a real API distinction, not a label in the page. Fixed scales
# persist in state_dict while remaining absent from optimizer parameters.
for mode in ('plain', 'residual', 'scaled', 'rezero'):
    block = lesson.Refinement(4, mode, 6, 59)
    assert 'scale' in block.state_dict()
    parameters = dict(block.named_parameters())
    buffers = dict(block.named_buffers())
    assert ('scale' in parameters) == (mode == 'rezero')
    assert ('scale' in buffers) == (mode != 'rezero')
    if mode != 'rezero':
        before = block.scale.clone()
        stepper = torch.optim.Adam(block.parameters(), lr=.003)
        stepper.zero_grad(set_to_none=True)
        block(value.float()).square().mean().backward()
        stepper.step()
        close(block.scale.double(), before.double())
checks.append('Fixed/trainable scale: saved-state, buffer/parameter and optimizer-update contracts')

# Mutation demonstrates that the shape guard really fires. Projection semantics
# need design review separately; a runtime shape check cannot prove them.
bad = lesson.Refinement(4, 'residual', 6, 2)
bad.upper = nn.Linear(4, 1)
try:
    bad(value.float())
except ValueError as error:
    assert 'same shape' in str(error)
else:
    raise AssertionError('Broadcastable correction bypassed the residual shape guard')
checks.append('Falsification: broadcastable width-one correction is rejected by canonical forward')

# Reuse is truthful: construction seeds explicitly match branch/stem/head state,
# and measurement/diagnostics must not accidentally accumulate parameter .grad.
left = lesson.DigitNetwork(2, 'plain', 7)
right = lesson.DigitNetwork(6, 'residual', 7)
for name in ('stem', 'head'):
    for a, b in zip(getattr(left, name).parameters(), getattr(right, name).parameters()):
        close(a.double(), b.double())
for i in range(2):
    for a, b in zip(left.blocks[i].parameters(), right.blocks[i].parameters()):
        close(a.double(), b.double())
generator = torch.Generator().manual_seed(29)
pixels = torch.randn(5, 64, generator=generator)
targets = torch.tensor([0, 2, 4, 6, 8])
before = {name: tensor.detach().clone() for name, tensor in right.state_dict().items()}
lesson.score(right, pixels, targets)
lesson.diagnostics(right, pixels, targets)
assert all(parameter.grad is None for parameter in right.parameters())
for name, tensor in right.state_dict().items():
    close(tensor.double(), before[name].double())
checks.append('Matched construction and diagnostics: same-seed prefix/stem/head, no parameter gradient accumulation or state mutation')

result = {'passed': True, 'reviewer': '/root/decision_depth_prose', 'torch': torch.__version__,
          'canonicalSourceHash': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          'checks': checks, 'initialGateGradient': initial_gate_gradient,
          'scope': 'Complementary exact/native checks only; no 39-fit rerun, browser or pretrained training.'}
REPORT.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
print(json.dumps(result, indent=2))
