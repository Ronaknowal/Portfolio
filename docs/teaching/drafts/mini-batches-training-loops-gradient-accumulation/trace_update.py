import torch

torch.set_default_dtype(torch.float64)
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([2.0, 0.0, 1.0])
w = torch.nn.Parameter(torch.tensor(0.0))
optimizer = torch.optim.SGD([w], lr=0.1, momentum=0.9)
optimizer.zero_grad(set_to_none=True)
print("start", f"w={w.item():.6f}", "grad=None")
for rows in ([0, 1], [2]):
    loss_sum = 0.5 * ((w * x[rows] - y[rows]) ** 2).sum()
    (loss_sum / len(x)).backward()
    print("backward", rows, f"grad={w.grad.item():.6f}", f"w={w.item():.6f}")
optimizer.step()
print("step", f"w={w.item():.6f}", f"momentum={optimizer.state[w]['momentum_buffer'].item():.6f}")
optimizer.zero_grad(set_to_none=True)
print("clear", "grad=None", f"w={w.item():.6f}")
