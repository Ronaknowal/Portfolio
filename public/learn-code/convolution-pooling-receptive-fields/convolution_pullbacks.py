"""Vectorized valid cross-correlation and pooling with explicit pullbacks.

NumPy + PyTorch, float64, NCHW, stride 1 and no padding. The existing direct_conv2d
owns general grouped/strided/dilated indexing; this file opens batching/backward.
"""
import numpy as np
import torch
from torch.nn import functional as F


def convolution_forward(images, weights, bias):
    batch, channels, height, width = images.shape
    outputs, filter_channels, kh, kw = weights.shape
    if channels != filter_channels or kh > height or kw > width:
        raise ValueError("the dense valid filter must fit the image and channels")
    oh, ow = height - kh + 1, width - kw + 1
    result = np.broadcast_to(bias[None, :, None, None], (batch, outputs, oh, ow)).copy()
    for row in range(kh):
        for column in range(kw):
            patch = images[:, :, row:row+oh, column:column+ow]
            result += np.einsum("bihw,oi->bohw", patch, weights[:, :, row, column], optimize=True)
    return result


def convolution_backward(images, weights, upstream):
    _, _, kh, kw = weights.shape
    oh, ow = upstream.shape[-2:]
    input_gradient = np.zeros_like(images)
    weight_gradient = np.zeros_like(weights)
    for row in range(kh):
        for column in range(kw):
            patch = images[:, :, row:row+oh, column:column+ow]
            weight_gradient[:, :, row, column] = np.einsum(
                "bohw,bihw->oi", upstream, patch, optimize=True)
            input_gradient[:, :, row:row+oh, column:column+ow] += np.einsum(
                "bohw,oi->bihw", upstream, weights[:, :, row, column], optimize=True)
    return input_gradient, weight_gradient, upstream.sum(axis=(0, 2, 3))


def pool1d(values, size, stride, mode="max"):
    """Finite 1-D valid windows; ties route to the first maximum."""
    if size < 1 or stride < 1 or size > len(values) or mode not in {"max", "mean"}:
        raise ValueError("invalid pooling window or mode")
    starts = np.arange(0, len(values) - size + 1, stride)
    windows = np.lib.stride_tricks.sliding_window_view(values, size)[::stride]
    if mode == "max":
        chosen = windows.argmax(axis=-1)
        return windows[np.arange(len(windows)), chosen], (starts, chosen, size, len(values), mode)
    return windows.mean(axis=-1), (starts, None, size, len(values), mode)


def pool1d_backward(upstream, saved):
    starts, chosen, size, length, mode = saved
    result = np.zeros(length, dtype=upstream.dtype)
    if mode == "max":
        np.add.at(result, starts + chosen, upstream)
    else:
        for offset in range(size):
            np.add.at(result, starts + offset, upstream / size)
    return result


def adaptive_average1d(values, output_size):
    """Floor/ceiling bins; prefix sums avoid repeatedly reducing overlapping bins."""
    if values.ndim != 1 or len(values) == 0 or output_size < 1:
        raise ValueError("use a nonempty vector and positive output size")
    length = len(values)
    indices = np.arange(output_size)
    starts = indices * length // output_size
    ends = ((indices + 1) * length + output_size - 1) // output_size
    prefix = np.concatenate((np.zeros(1, dtype=values.dtype), np.cumsum(values)))
    result = (prefix[ends] - prefix[starts]) / (ends - starts)
    return result, (starts, ends, length)


def adaptive_average1d_backward(upstream, saved):
    """A range-add difference array accumulates every bin's uniform pullback."""
    starts, ends, length = saved
    differences = np.zeros(length + 1, dtype=upstream.dtype)
    shares = upstream / (ends - starts)
    np.add.at(differences, starts, shares)
    np.add.at(differences, ends, -shares)
    return np.cumsum(differences)[:-1]


def main():
    rng = np.random.default_rng(13)
    images, weights, bias = rng.normal(size=(2, 3, 5, 6)), rng.normal(size=(4, 3, 2, 3)), rng.normal(size=4)
    output = convolution_forward(images, weights, bias)
    upstream = rng.normal(size=output.shape)
    manual = convolution_backward(images, weights, upstream)
    native_inputs = [torch.tensor(array, requires_grad=True) for array in (images, weights, bias)]
    native = F.conv2d(*native_inputs)
    (native * torch.tensor(upstream)).sum().backward()
    np.testing.assert_allclose(output, native.detach().numpy(), atol=1e-12, rtol=1e-12)
    for calculated, tensor in zip(manual, native_inputs):
        np.testing.assert_allclose(calculated, tensor.grad.numpy(), atol=1e-12, rtol=1e-12)
    for mode in ("max", "mean"):
        values = np.array([1., 4., 3., 2., -1.])
        output, saved = pool1d(values, 3, 1, mode)
        derivative = pool1d_backward(np.ones_like(output), saved)
        inputs = torch.tensor(values, requires_grad=True)
        operation = F.max_pool1d if mode == "max" else F.avg_pool1d
        native = operation(inputs[None, None], 3, 1).flatten()
        native.sum().backward()
        np.testing.assert_allclose(output, native.detach().numpy())
        np.testing.assert_allclose(derivative, inputs.grad.numpy())
    for length, output_size in ((5, 3), (3, 5), (5, 1)):
        values = rng.normal(size=length)
        output, saved = adaptive_average1d(values, output_size)
        upstream = rng.normal(size=output_size)
        derivative = adaptive_average1d_backward(upstream, saved)
        inputs = torch.tensor(values, requires_grad=True)
        native = F.adaptive_avg_pool1d(inputs[None, None], output_size).flatten()
        (native * torch.tensor(upstream)).sum().backward()
        np.testing.assert_allclose(output, native.detach().numpy(), atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(derivative, inputs.grad.numpy(), atol=1e-12, rtol=1e-12)
    print("dense valid convolution, overlapping pooling and adaptive pooling values/pullbacks agree")


if __name__ == "__main__":
    main()
