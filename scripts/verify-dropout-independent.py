"""Independent finite scope: mask adjoints, locked axes and BN state.

Imports canonical teaching programs without running any training campaign.
Produces native BN cases for a separate JavaScript model comparison.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "public/learn-assets/dropout-droppath-stochastic-depth"
REPORT = ROOT / "docs/teaching/evidence/dropout-implementation/independent-native.json"
REPORT.write_text(json.dumps({"passed": False, "state": "running"}), encoding="utf-8")
sys.path.insert(0, str(ROOT / "scratch/dropout-runtime-deps"))


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ASSETS / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scratch = load("review_dropout_scratch", "dropout-experiments.py")
library = load("review_dropout_library", "dropout-library-checks.py")
torch.set_num_threads(1)
torch.manual_seed(816)
cases = []

# A ones probe recovers the exact sampled operator without losing the mask at
# zero-valued activations. Replaying the seed is deliberate within one scratch
# operation; there is no assumed cross-library RNG equivalence.
for mode, shape in [("element", (3, 2, 3, 2)), ("channel", (3, 2, 3, 2)),
                    ("row", (3, 5, 4)), ("batch", (3, 5, 4))]:
    for probability in (0., .13, .61, 1.):
        for training in (True, False):
            value = torch.randn(shape, dtype=torch.float64)
            value.flatten()[::5] = 0
            value.requires_grad_()
            upstream = torch.randn(shape, dtype=torch.float64)
            torch.manual_seed(417)
            scale = scratch.mask_values(torch.ones_like(value), probability, training, mode)
            torch.manual_seed(417)
            output = scratch.mask_values(value, probability, training, mode)
            torch.testing.assert_close(output, scale * value)
            derivative, = torch.autograd.grad((output * upstream).sum(), value)
            torch.testing.assert_close(derivative, scale * upstream)
            if mode == "channel":
                torch.testing.assert_close(scale, scale[:, :, :1, :1].expand_as(scale))
            elif mode == "row":
                torch.testing.assert_close(scale, scale[:, :1, :1].expand_as(scale))
            elif mode == "batch":
                torch.testing.assert_close(scale, scale.flatten()[0].expand_as(scale))
            cases.append({"kind": "mask_adjoint_zero_inputs", "mode": mode,
                          "p": probability, "training": training})

for shape in ((1, 1, 1), (2, 4, 3), (3, 2, 5)):
    for probability in (.2, .7):
        value = torch.randn(shape, dtype=torch.float64, requires_grad=True)
        fixed = (torch.arange(shape[0] * shape[2]).reshape(shape[0], 1, shape[2]) % 3 != 0).double()
        upstream = torch.randn_like(value)
        output = library.locked_features(value, probability, True, fixed)
        derivative, = torch.autograd.grad((output * upstream).sum(), value)
        torch.testing.assert_close(output, value * fixed / (1 - probability))
        torch.testing.assert_close(derivative, upstream * fixed / (1 - probability))
        cases.append({"kind": "locked_axis_weighted_adjoint", "shape": shape, "p": probability})

for mask in (torch.ones(2, 4, 3), torch.full((2, 1, 3), .3)):
    try:
        library.locked_features(torch.ones(2, 4, 3), .3, True, mask)
    except ValueError:
        cases.append({"kind": "locked_invalid_mask", "shape": list(mask.shape)})
    else:
        raise AssertionError("Invalid locked mask was accepted")

# Both modes with both autograd settings, changed batch sizes and momentum.
# Save actual PyTorch values; JS must match these, not its own copied formula.
bn_cases = []
for count in (2, 5, 11):
    for training in (False, True):
        for recording in (False, True):
            for momentum in (.17, 1.):
                bn = torch.nn.BatchNorm1d(1, momentum=momentum, eps=1e-5, affine=False).double()
                bn.running_mean.fill_(1.75)
                bn.running_var.fill_(2.25)
                bn.num_batches_tracked.fill_(3)
                bn.train(training)
                value = torch.linspace(-2.2, 3.7, count, dtype=torch.float64).square().reshape(-1, 1)
                value.requires_grad_()
                with torch.set_grad_enabled(recording):
                    output = bn(value)
                assert output.requires_grad == recording
                bn_cases.append({"inputs": {"values": value.detach().flatten().tolist(),
                    "runningMean": 1.75, "runningVariance": 2.25, "batches": 3,
                    "batchTraining": training, "recordGradients": recording, "momentum": momentum},
                    "expected": {"output": output.detach().flatten().tolist(),
                        "runningMean": bn.running_mean.item(), "runningVariance": bn.running_var.item(),
                        "batches": bn.num_batches_tracked.item(), "recordGradients": output.requires_grad}})

sources = ["dropout-experiments.py", "dropout-library-checks.py"]
record = {"passed": True, "method": "Weighted adjoints including zero inputs; changed locked axes and invalid mask shapes; independent PyTorch BatchNorm state/gradient-mode oracle.",
          "maskAndLockedCases": len(cases), "batchNormCases": len(bn_cases), "cases": cases,
          "batchNorm": bn_cases, "versions": {"torch": torch.__version__, "torchvision": library.torchvision.__version__},
          "sourceHashes": {str((ASSETS / path).relative_to(ROOT)).replace('\\', '/'): hashlib.sha256((ASSETS / path).read_bytes()).hexdigest() for path in sources}}
REPORT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"passed": True, "maskAndLockedCases": len(cases), "batchNormCases": len(bn_cases)}))
