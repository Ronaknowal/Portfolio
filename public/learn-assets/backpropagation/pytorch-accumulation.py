import torch

x = torch.tensor(3.0, requires_grad=True)
loss = x * x
loss.backward(retain_graph=True)
print(x.grad.item())
loss.backward()
print(x.grad.item())
x.grad = None
new_loss = x * x
new_loss.backward()
print(x.grad.item())
