import torch
from torch.utils.checkpoint import checkpoint

def block(x):
    return torch.tanh(x @ x.T / x.shape[0])

x = torch.tensor([[0.2, -0.4], [0.7, 0.1]],
                 requires_grad=True, dtype=torch.float64)
ordinary = block(x).sum()
gradient_a = torch.autograd.grad(ordinary, x)[0]
recomputed = checkpoint(block, x, use_reentrant=False).sum()
gradient_b = torch.autograd.grad(recomputed, x)[0]
print(torch.allclose(gradient_a, gradient_b, atol=1e-12, rtol=1e-12))
