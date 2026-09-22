"""Complete mask-contract checks and a locked-feature customization.

Run with compatible torch/torchvision installations. The website's retained
verification uses PyTorch 2.14 and Torchvision 0.29 on CPU.
"""
import json
import torch
import torchvision
from torch import nn
from torchvision.ops import stochastic_depth


def locked_features(values, probability, training, mask=None):
    """One [B,1,D] mask shared across time in a [B,T,D] tensor."""
    if values.ndim != 3 or not 0 <= probability <= 1:
        raise ValueError("use [B,T,D] and a drop probability from zero to one")
    if not training or probability == 0:
        return values
    if probability == 1:
        return values * 0
    shape = (values.shape[0], 1, values.shape[2])
    if mask is None:
        mask = values.new_empty(shape).bernoulli_(1 - probability)
    if tuple(mask.shape) != shape or not ((mask == 0) | (mask == 1)).all():
        raise ValueError("fixed mask must have shape [B,1,D] and binary entries")
    return values * mask / (1 - probability)


def run():
    torch.manual_seed(9)
    values = torch.arange(1., 17., dtype=torch.float64).reshape(2, 2, 2, 2)
    checked = []
    for name, operation in [
        ("element", nn.Dropout(.25)),
        ("channel", nn.Dropout2d(.25)),
        ("row", lambda x: stochastic_depth(x, .25, "row", True)),
        ("batch", lambda x: stochastic_depth(x, .25, "batch", True)),
    ]:
        x = values.clone().requires_grad_()
        output = operation(x)
        # Recover this exact forward mask, never assume identical RNG streams.
        bits = (output.detach() != 0).to(x.dtype)
        expected = x * bits / .75
        torch.testing.assert_close(output, expected)
        output.sum().backward()
        torch.testing.assert_close(x.grad, bits / .75)
        if name == "channel":
            assert (bits == bits[:, :, :1, :1]).all()
        if name == "row":
            assert (bits == bits[:, :1, :1, :1]).all()
        if name == "batch":
            assert (bits == bits[:1, :1, :1, :1]).all()
        checked.append(name + " forward/backward geometry")
    for probability in (0., 1.):
        for layer in (nn.Dropout(probability), nn.Dropout2d(probability)):
            layer.train()
            torch.testing.assert_close(layer(values), values if probability == 0 else values * 0)
            layer.eval()
            torch.testing.assert_close(layer(values), values)
        for mode in ("row", "batch"):
            torch.testing.assert_close(stochastic_depth(values, probability, mode, True),
                                       values if probability == 0 else values * 0)
            torch.testing.assert_close(stochastic_depth(values, probability, mode, False), values)
    sequence = torch.tensor([[[1., 2.], [3., 4.]]], requires_grad=True)
    fixed = torch.tensor([[[1., 0.]]])
    output = locked_features(sequence, .5, True, fixed)
    torch.testing.assert_close(output, torch.tensor([[[2., 0.], [6., 0.]]]))
    output.sum().backward()
    torch.testing.assert_close(sequence.grad, torch.tensor([[[2., 0.], [2., 0.]]]))
    for probability in (0., .5, 1.):
        torch.testing.assert_close(locked_features(sequence, probability, False), sequence)
    torch.testing.assert_close(locked_features(sequence, 0., True), sequence)
    torch.testing.assert_close(locked_features(sequence, 1., True), sequence * 0)
    return {"torch": torch.__version__, "torchvision": torchvision.__version__,
            "passed": True, "checks": checked + ["p0/p1 train/eval boundaries",
                      "locked-feature fixed-mask outputs and gradients",
                      "locked-feature evaluation and probability boundaries"],
            "locked_output": output.detach().tolist(), "locked_gradient": sequence.grad.tolist()}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
