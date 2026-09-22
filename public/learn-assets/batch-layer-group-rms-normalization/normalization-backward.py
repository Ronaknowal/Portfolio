"""Explicit normalization VJPs in NumPy, matched to torch modules.

Run python normalization-backward.py; numpy2.3.5, torch2.14.0+cpu.
PyTorch performs only the library comparison. All fixtures are float64.
"""
import json
import numpy as np
import torch


def normalize(values, axes, epsilon=1e-5, centered=True):
    mean = values.mean(axis=axes, keepdims=True) if centered else 0.
    deviation = values - mean
    inverse_scale = 1 / np.sqrt((deviation ** 2).mean(axis=axes, keepdims=True) + epsilon)
    normalized = deviation * inverse_scale
    return normalized, (normalized, inverse_scale, axes, centered)


def backward(upstream, cache):
    """VJP for normalization WITHOUT affine scale/shift."""
    normalized, inverse_scale, axes, centered = cache
    correlated = (upstream * normalized).mean(axis=axes, keepdims=True)
    gradient = upstream - normalized * correlated
    if centered:
        gradient = gradient - upstream.mean(axis=axes, keepdims=True)
    return inverse_scale * gradient


def main():
    torch.set_num_threads(1)
    rng = np.random.default_rng(7)
    values = rng.normal(size=(2, 4, 2, 3))
    upstream = rng.normal(size=values.shape)
    results = {}
    variants = {
        "BatchNorm": ((0, 2, 3), True, torch.nn.BatchNorm2d(4, affine=False, dtype=torch.float64)),
        "LayerNorm": ((1, 2, 3), True, torch.nn.LayerNorm((4, 2, 3), elementwise_affine=False, dtype=torch.float64)),
        "InstanceNorm": ((2, 3), True, torch.nn.InstanceNorm2d(4, affine=False, dtype=torch.float64)),
        "RMSNorm": ((3,), False, torch.nn.RMSNorm(3, eps=1e-5, elementwise_affine=False, dtype=torch.float64)),
    }
    for name, (axes, centered, layer) in variants.items():
        own, cache = normalize(values, axes, centered=centered)
        gradient = backward(upstream, cache)
        tensor = torch.tensor(values, requires_grad=True)
        output = layer(tensor)
        (output * torch.tensor(upstream)).sum().backward()
        np.testing.assert_allclose(own, output.detach().numpy(), atol=1e-12)
        np.testing.assert_allclose(gradient, tensor.grad.numpy(), atol=1e-12)
        results[name] = dict(forward_error=float(np.max(abs(own-output.detach().numpy()))),
                             gradient_error=float(np.max(abs(gradient-tensor.grad.numpy()))))
    # GroupNorm changes the grouping/reshape, not the centered derivative rule.
    grouped = values.reshape(2, 2, 2, 2, 3)
    own_group, cache = normalize(grouped, (2, 3, 4))
    own_gradient = backward(upstream.reshape(grouped.shape), cache).reshape(values.shape)
    tensor = torch.tensor(values, requires_grad=True)
    layer = torch.nn.GroupNorm(2, 4, affine=False, dtype=torch.float64)
    output = layer(tensor)
    (output * torch.tensor(upstream)).sum().backward()
    np.testing.assert_allclose(own_group.reshape(values.shape), output.detach().numpy(), atol=1e-12)
    np.testing.assert_allclose(own_gradient, tensor.grad.numpy(), atol=1e-12)
    results["GroupNorm"] = dict(gradient_error=float(np.max(abs(own_gradient-tensor.grad.numpy()))))
    # The affine part is distinct: multiply before the VJP, reduce for parameters.
    token = np.array([[1., 3., -2.], [.5, -.4, 1.2]])
    scale, shift = np.array([.5, 1.2, -.7]), np.array([.1, -.2, .4])
    incoming = np.array([[.3, -.2, .8], [-.1, .7, .4]])
    normalized, cache = normalize(token, (1,))
    input_gradient = backward(incoming * scale, cache)
    scale_gradient = (incoming * normalized).sum(0)
    shift_gradient = incoming.sum(0)
    tensor = torch.tensor(token, requires_grad=True)
    layer = torch.nn.LayerNorm(3, dtype=torch.float64)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(scale)); layer.bias.copy_(torch.tensor(shift))
    (layer(tensor) * torch.tensor(incoming)).sum().backward()
    np.testing.assert_allclose(input_gradient, tensor.grad.numpy(), atol=1e-12)
    np.testing.assert_allclose(scale_gradient, layer.weight.grad.numpy(), atol=1e-12)
    np.testing.assert_allclose(shift_gradient, layer.bias.grad.numpy(), atol=1e-12)
    results["affine_LayerNorm"] = dict(input_gradient=input_gradient.tolist(),
                                     scale_gradient=scale_gradient.tolist(),
                                     shift_gradient=shift_gradient.tolist())
    # Finite differences supply an independent check of the explicit derivative.
    numerical = np.zeros_like(token)
    for index in np.ndindex(token.shape):
        plus, minus = token.copy(), token.copy()
        plus[index] += 1e-6; minus[index] -= 1e-6
        yp, _ = normalize(plus, (1,)); ym, _ = normalize(minus, (1,))
        numerical[index] = ((yp-ym) * scale * incoming).sum() / 2e-6
    np.testing.assert_allclose(input_gradient, numerical, atol=1e-8, rtol=1e-8)
    results["finite_difference_max_error"] = float(np.max(abs(input_gradient-numerical)))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
