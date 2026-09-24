"""Complementary independent review: bilinear adjoints and weighted pooling.

Imports the public teaching implementation without running its experiment.
No fitting and no source mutation. Run from the repository root.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "public/learn-code/convolution-pooling-receptive-fields"
REPORT = ROOT / "docs/teaching/evidence/convolution-independent-native.json"
REPORT.write_text(json.dumps({"passed": False, "state": "running"}), encoding="utf-8")
spec = importlib.util.spec_from_file_location("convolution_pullbacks", ASSETS / "convolution_pullbacks.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
torch.set_num_threads(1)
rng = np.random.default_rng(901)
records = []

for shape, kernel_shape in [((1, 1, 2, 4), (1, 1, 1, 3)), ((2, 2, 5, 7), (3, 2, 3, 2)), ((1, 3, 3, 4), (2, 3, 3, 4)), ((3, 1, 4, 3), (2, 1, 2, 1))]:
    image, weight = rng.normal(size=shape), rng.normal(size=kernel_shape)
    bias = rng.normal(size=kernel_shape[0])
    result = module.convolution_forward(image, weight, bias)
    upstream = rng.normal(size=result.shape)
    dx, dw, db = module.convolution_backward(image, weight, upstream)
    # Bilinearity gives independent directional identities without calling torch backward.
    direction_x = rng.normal(size=image.shape)
    direction_w = rng.normal(size=weight.shape)
    zero_bias = np.zeros_like(bias)
    lhs_x = np.vdot(module.convolution_forward(direction_x, weight, zero_bias), upstream)
    lhs_w = np.vdot(module.convolution_forward(image, direction_w, zero_bias), upstream)
    np.testing.assert_allclose(lhs_x, np.vdot(direction_x, dx), atol=1e-10)
    np.testing.assert_allclose(lhs_w, np.vdot(direction_w, dw), atol=1e-10)
    np.testing.assert_allclose(db, upstream.sum(axis=(0, 2, 3)), atol=1e-12)
    # Forward oracle is a different lowering: sliding windows + one full contraction.
    windows = np.lib.stride_tricks.sliding_window_view(image, weight.shape[-2:], axis=(-2, -1))
    oracle = np.einsum("bihwrc,oirc->bohw", windows, weight) + bias[None, :, None, None]
    np.testing.assert_allclose(result, oracle, atol=1e-12)
    records.append({"kind": "convolution_bilinear_adjoints", "image": shape, "kernel": kernel_shape})

for length in [1, 2, 3, 5, 9]:
    for output_size in [1, 2, 4, 7, 11]:
        values, upstream = rng.normal(size=length), rng.normal(size=output_size)
        result, saved = module.adaptive_average1d(values, output_size)
        derivative = module.adaptive_average1d_backward(upstream, saved)
        # Dense incidence matrix independently handles repeated/overlapping bins.
        incidence = np.zeros((output_size, length))
        for i in range(output_size):
            start = int(np.floor(i * length / output_size))
            end = int(np.ceil((i + 1) * length / output_size))
            incidence[i, start:end] = 1 / (end - start)
        np.testing.assert_allclose(result, incidence @ values, atol=2e-14)
        np.testing.assert_allclose(derivative, incidence.T @ upstream, atol=2e-14)
        records.append({"kind": "adaptive_incidence_pullback", "input": length, "output": output_size})

for values in [np.array([1., 4., 3., 2., -1.]), np.array([2., 2., 1., 2., 2.]), np.array([-1., -4., -2., -3., -5.])]:
    for size in [1, 2, 3, 5]:
        for stride in [1, 2]:
            for mode in ["max", "mean"]:
                result, saved = module.pool1d(values, size, stride, mode)
                upstream = rng.normal(size=len(result))
                derivative = module.pool1d_backward(upstream, saved)
                native = torch.tensor(values, requires_grad=True)
                operation = F.max_pool1d if mode == "max" else F.avg_pool1d
                expected = operation(native[None, None], size, stride).flatten()
                (expected * torch.tensor(upstream)).sum().backward()
                np.testing.assert_allclose(result, expected.detach().numpy(), atol=1e-12)
                np.testing.assert_allclose(derivative, native.grad.numpy(), atol=1e-12)
                records.append({"kind": "weighted_pooling", "values": values.tolist(), "size": size, "stride": stride, "mode": mode})

report = {"passed": True, "reviewer": "residual_implementation reviewing convolution", "scope": "4 bilinear-adjoint cases, 25 adaptive incidence-matrix cases and 48 weighted pooling cases; no fit replay", "cases": records, "sourceHashes": {str((ASSETS / name).relative_to(ROOT)).replace('\\', '/'): hashlib.sha256((ASSETS / name).read_bytes()).hexdigest() for name in ["convolution_pullbacks.py", "convolution-experiments.py"]}}
REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(f"Independent convolution native review passed: {len(records)} cases")
