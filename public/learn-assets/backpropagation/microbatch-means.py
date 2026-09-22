import torch

x = torch.arange(1., 6., dtype=torch.float64)
targets = 2 * x
for mode in ("full", "weighted", "unweighted"):
    weight = torch.tensor(0., requires_grad=True, dtype=torch.float64)
    slices = (slice(None),) if mode == "full" else (slice(0, 2), slice(2, 5))
    for part in slices:
        errors = weight * x[part] - targets[part]
        loss = (errors ** 2).mean()
        if mode == "weighted":
            loss = loss * len(errors) / len(x)
        loss.backward()
    print(mode, weight.grad.item())
