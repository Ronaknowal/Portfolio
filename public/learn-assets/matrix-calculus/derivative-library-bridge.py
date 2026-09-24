"""Manual derivative maps and PyTorch transforms on identical float64 fixtures.

Run: python derivative-library-bridge.py
Dependencies: numpy==2.3.5, torch==2.14.0 (tested CPU wheel).
This file owns derivative-product/API mapping; the Backpropagation lesson owns
the general reverse-mode engine. No dense Jacobian is built for the affine loss.
"""
import numpy as np
import torch
from torch.func import grad, jacrev, jvp, vjp


def affine_loss_pullback(x, weight, bias, target):
    """Row batches: W[in,out], loss=sum(residual**2)/(2*batch_size)."""
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[0]:
        raise ValueError("Expected X[batch,in] and W[in,out]")
    shape = (x.shape[0], weight.shape[1])
    if x.shape[0] == 0 or bias.shape != (shape[1],) or target.shape != shape:
        raise ValueError("Use a nonempty batch, bias[out], and exact target shape")
    residual = x @ weight + bias - target
    upstream = residual / len(x)
    return (np.sum(residual * residual) / (2 * len(x)),
            upstream @ weight.T, x.T @ upstream, upstream.sum(axis=0))


def vector_function(x):
    return torch.stack((x[0] ** 2 + x[1], x[0] * x[1]))


def scalar_function(x):
    return x[0] ** 2 * x[1] + torch.sin(x[0])


def main():
    torch.set_num_threads(1)
    x = torch.tensor([2., 3.], dtype=torch.float64)
    direction = torch.tensor([.01, -.02], dtype=x.dtype)
    cotangent = torch.tensor([1., -1.], dtype=x.dtype)
    jacobian = np.array([[4., 1.], [3., 2.]])
    tangent = jvp(vector_function, (x,), (direction,))[1]
    _, pullback = vjp(vector_function, x)
    adjoint = pullback(cotangent)[0]
    np.testing.assert_allclose(jacrev(vector_function)(x), jacobian)
    np.testing.assert_allclose(tangent, jacobian @ direction.numpy())
    np.testing.assert_allclose(adjoint, jacobian.T @ cotangent.numpy())
    print("JVP", np.round(tangent.numpy(), 6).tolist())
    print("VJP", adjoint.numpy().tolist())
    print("adjoint pairing", round(float(cotangent @ tangent), 8))

    # A rectangular batch makes a transpose or an extra output mean observable.
    batch = np.array([[1., 2., -1.], [0., 1., 2.], [2., -1., 1.], [-1., 0., 1.]])
    weight = np.array([[.2, -.3], [.4, .1], [-.2, .5]])
    bias = np.array([.1, -.2])
    target = np.array([[1., 0.], [0., 1.], [.5, -.5], [-1., 1.]])
    loss, gx, gw, gb = affine_loss_pullback(batch, weight, bias, target)
    tx, tw, tb = [torch.tensor(a, requires_grad=True) for a in (batch, weight, bias)]
    # F.linear expects [out,in]; transpose our parameter, retaining its AD link.
    residual = torch.nn.functional.linear(tx, tw.T, tb) - torch.from_numpy(target)
    objective = residual.square().sum() / (2 * len(batch))
    objective.backward()
    error = max(np.max(np.abs(a - b.grad.numpy()))
                for a, b in ((gx, tx), (gw, tw), (gb, tb)))
    np.testing.assert_allclose(loss, objective.item())
    assert error < 1e-12
    print("affine loss", round(float(loss), 8), "pullback max_error", f"{error:.2e}")
    point = torch.tensor([1.2, -.7], dtype=torch.float64)
    gradient = grad(scalar_function)(point)
    manual = [2 * 1.2 * -.7 + np.cos(1.2), 1.2 ** 2]
    np.testing.assert_allclose(gradient, manual)
    velocity = torch.tensor([2., -1.], dtype=point.dtype)
    slope = jvp(scalar_function, (point,), (velocity,))[1]
    np.testing.assert_allclose(slope, gradient @ velocity)
    print("scalar gradient", np.round(gradient.numpy(), 6).tolist())
    print("unnormalized direction [2,-1] slope", round(float(slope), 6))


if __name__ == "__main__":
    main()
