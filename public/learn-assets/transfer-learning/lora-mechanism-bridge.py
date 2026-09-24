"""A manual NumPy LoRA backward/update, checked against ordinary PyTorch ops."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def forward_and_gradients(x, target, weight, a, b, scale):
    """Dense real matrices; mean MSE over every row and output coordinate."""
    if any(values.ndim != 2 for values in (x, target, weight, a, b)):
        raise ValueError("X, target, W, A and B must be matrices.")
    if (any(size == 0 for values in (x, target, weight, a, b) for size in values.shape)
            or target.shape != (x.shape[0], weight.shape[0])
            or weight.shape[1] != x.shape[1] or a.shape[1] != x.shape[1]
            or b.shape != (weight.shape[0], a.shape[0])):
        raise ValueError("Use X:(N,k), target:(N,d), W:(d,k), A:(r,k), B:(d,r).")
    hidden = x @ a.T
    output = x @ weight.T + scale * hidden @ b.T
    residual = output - target
    loss = np.mean(residual**2)
    incoming = 2 * residual / residual.size
    grad_b = scale * incoming.T @ hidden
    grad_a = scale * (incoming @ b).T @ x
    grad_x = incoming @ weight + scale * (incoming @ b) @ a
    return output, loss, grad_a, grad_b, grad_x


def compare_case(name, x, target, a, b, rate=.1):
    weight = np.eye(2)
    scale = 1.
    output, loss, grad_a, grad_b, grad_x = forward_and_gradients(x, target, weight, a, b, scale)
    base = nn.Linear(2, 2, bias=False, dtype=torch.float64).requires_grad_(False)
    with torch.no_grad():
        base.weight.copy_(torch.from_numpy(weight))
    library_a = nn.Parameter(torch.from_numpy(a.copy()))
    library_b = nn.Parameter(torch.from_numpy(b.copy()))
    library_x = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    prediction = base(library_x) + scale * F.linear(F.linear(library_x, library_a), library_b)
    library_loss = F.mse_loss(prediction, torch.from_numpy(target))
    library_loss.backward()
    error = max(np.max(np.abs(grad_a - library_a.grad.numpy())),
                np.max(np.abs(grad_b - library_b.grad.numpy())),
                np.max(np.abs(grad_x - library_x.grad.numpy())))
    optimizer = torch.optim.SGD([library_a, library_b], lr=rate)
    optimizer.step()
    next_a, next_b = a - rate * grad_a, b - rate * grad_b
    after = forward_and_gradients(x, target, weight, next_a, next_b, scale)
    with torch.no_grad():
        library_after = base(library_x) + scale * F.linear(F.linear(library_x, library_a), library_b)
    merged = x @ (weight + scale * next_b @ next_a).T
    print(name, "loss", round(float(loss), 9), "after", round(float(after[1]), 9))
    print("grad_A", grad_a.tolist(), "grad_B", grad_b.tolist(), "grad_input", grad_x.tolist())
    print("gradient_max_error", f"{error:.3e}",
          "step_max_error", f"{np.max(np.abs(after[0] - library_after.numpy())):.3e}",
          "merge_max_error", f"{np.max(np.abs(after[0] - merged)):.3e}")
    print("base_has_gradient", base.weight.grad is not None,
          "base_max_change", float(np.max(np.abs(base.weight.detach().numpy() - weight))))


if __name__ == "__main__":
    torch.set_num_threads(1)
    target = np.zeros((1, 2))
    a = np.array([[1., -1.]])
    b = np.zeros((2, 1))
    compare_case("zero_B", np.array([[2., 1.]]), target, a, b)
    compare_case("both_zero", np.array([[2., 1.]]), target, np.zeros_like(a), b)
    compare_case("null_measurement", np.array([[1., 1.]]), target, a, b)
    compare_case("zero_rate", np.array([[2., 1.]]), target, a, b, rate=0.)
    compare_case("two_rows_nonzero_B", np.array([[2., 1.], [-1., 2.]]),
                 np.zeros((2, 2)), a, np.array([[-.2], [-.1]]))
