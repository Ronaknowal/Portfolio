import torch

x = torch.tensor([0.3, 0.7], requires_grad=True, dtype=torch.float64)
v = torch.tensor([1.0, 2.0], dtype=torch.float64)
loss = x[0] ** 2 + x[0] * x[1] + 3 * x[1] ** 2
gradient = torch.autograd.grad(loss, x, create_graph=True)[0]
hessian_vector = torch.autograd.grad(gradient @ v, x)[0]
print(gradient.tolist())
print(hessian_vector.tolist())
