import torch

class HardSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.clamp(0.2 * x + 0.5, 0.0, 1.0)

    @staticmethod
    def backward(ctx, incoming):
        (x,) = ctx.saved_tensors
        active = (x > -2.5) & (x < 2.5)
        return incoming * 0.2 * active

x = torch.tensor([-3., -2., 0., 2., 3.],
                 requires_grad=True, dtype=torch.float64)
print(torch.autograd.gradcheck(HardSigmoid.apply, (x,)))
HardSigmoid.apply(x).sum().backward()
print(x.grad.tolist())
