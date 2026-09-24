import torch

def function(x):
    return torch.stack([x[0] * x[1], torch.sin(x[0]), x[1] ** 2])

x = torch.tensor([0.3, 0.7], dtype=torch.float64)
direction = torch.tensor([1.0, 2.0], dtype=torch.float64)
output_weight = torch.tensor([1.0, -1.0, 2.0], dtype=torch.float64)
value, tangent = torch.func.jvp(function, (x,), (direction,))
value, pullback = torch.func.vjp(function, x)
sensitivity = pullback(output_weight)[0]
print(value.tolist())
print(tangent.tolist())
print(sensitivity.tolist())
print(float(output_weight @ tangent), float(sensitivity @ direction))
