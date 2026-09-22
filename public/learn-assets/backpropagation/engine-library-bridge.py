"""Run beside teaching-autodiff.py; compare one identical training step."""
from pathlib import Path
import importlib.util
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

spec = importlib.util.spec_from_file_location(
    "teaching_autodiff", Path(__file__).with_name("teaching-autodiff.py"))
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)


def run(rate):
    torch.set_num_threads(1)
    features = np.array([[.2, -.4], [1.1, .3], [-.7, .8]])
    labels = np.array([0, 1, 0])
    parameters = engine.make_parameters(8, (2, 3, 2))
    library = nn.Sequential(nn.Linear(2, 3), nn.Tanh(), nn.Linear(3, 2)).double()
    with torch.no_grad():
        library[0].weight.copy_(torch.from_numpy(parameters[0].data.T.copy()))
        library[0].bias.copy_(torch.from_numpy(parameters[1].data.copy()))
        library[2].weight.copy_(torch.from_numpy(parameters[2].data.T.copy()))
        library[2].bias.copy_(torch.from_numpy(parameters[3].data.copy()))

    logits = engine.predict(features, parameters)
    loss = logits.cross_entropy(labels)
    loss.backward()
    optimizer = torch.optim.SGD(library.parameters(), lr=rate)
    optimizer.zero_grad(set_to_none=True)
    library_logits = library(torch.from_numpy(features))
    library_loss = F.cross_entropy(library_logits, torch.from_numpy(labels))
    library_loss.backward()
    print("rate", rate, "before_loss", round(float(loss.data), 12), round(library_loss.item(), 12))
    print("logit_max_error", f"{np.max(np.abs(logits.data - library_logits.detach().numpy())):.3e}")
    library_gradients = [library[0].weight.grad.T, library[0].bias.grad,
                         library[2].weight.grad.T, library[2].bias.grad]
    for name, parameter, gradient in zip(("W1", "b1", "W2", "b2"), parameters, library_gradients):
        error = np.max(np.abs(parameter.grad - gradient.numpy()))
        print(name, "gradient_shape", parameter.grad.shape, "max_error", f"{error:.3e}")

    # Both sides use the old model's gradients for one simultaneous SGD step.
    updated = [engine.Tensor(p.data - rate * p.grad, True) for p in parameters]
    optimizer.step()
    after = engine.predict(features, updated).cross_entropy(labels)
    library_after = F.cross_entropy(library(torch.from_numpy(features)), torch.from_numpy(labels))
    print("after_loss", round(float(after.data), 12), round(library_after.item(), 12))


if __name__ == "__main__":
    run(.2)
    run(0.)
