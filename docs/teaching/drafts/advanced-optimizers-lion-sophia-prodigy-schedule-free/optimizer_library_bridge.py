"""Four actual optimizer APIs on one small fixed classification problem.

Targets: torch 2.14.0, prodigyopt 1.1.2, schedulefree 1.4.1; the lesson
pins the original Lion/Sophia source files. Choose --method explicitly.
This is an API/state demonstration, not an optimizer quality benchmark.
"""
import argparse
from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from optimizer_rules import Optimizer


def source_class(filename, class_name):
    path = Path(__file__).with_name(filename)
    spec = spec_from_file_location(path.stem, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def make_optimizer(parameters, method):
    if method == "lion":
        return source_class("lion_pytorch.py", "Lion")(
            parameters, lr=.01, betas=(.9, .99), weight_decay=0.)
    if method == "sophia":
        return source_class("sophia.py", "SophiaG")(
            parameters, lr=.01, betas=(.965, .99), rho=.04, weight_decay=0.)
    if method == "prodigy":
        from prodigyopt import Prodigy
        return Prodigy(parameters, lr=1., betas=(.9, .999), d0=1e-6,
                       weight_decay=0., use_bias_correction=False,
                       safeguard_warmup=False, slice_p=1)
    if method == "schedule_free":
        from schedulefree import AdamWScheduleFree
        return AdamWScheduleFree(parameters, lr=.01, betas=(.9, .999),
                                  weight_decay=0., warmup_steps=2, foreach=False)
    raise ValueError(method)


def switch(model, optimizer, method, training):
    model.train(training)
    if method == "schedule_free":
        optimizer.train() if training else optimizer.eval()


def update(model, optimizer, method, inputs, labels, step):
    switch(model, optimizer, method, True)
    optimizer.zero_grad(set_to_none=True)
    if method == "sophia" and step % 2 == 0:
        # Curvature and true-label gradients are evaluated at the same parameters.
        logits = model(inputs)
        fake_labels = torch.distributions.Categorical(logits=logits.detach()).sample()
        F.cross_entropy(logits, fake_labels, reduction="mean").backward()
        optimizer.update_hessian()   # EMA of squared mean gradient, not B*g^2
        optimizer.zero_grad(set_to_none=True)
    F.cross_entropy(model(inputs), labels, reduction="mean").backward()
    if method == "sophia":
        optimizer.step(bs=len(inputs))   # B is applied once, in Sophia's denominator
    else:
        optimizer.step()


def build(method):
    model = torch.nn.Linear(2, 3, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[.1, -.2], [-.1, .2], [.05, -.05]]))
    return model, make_optimizer(model.parameters(), method)


def main(method):
    torch.manual_seed(23)
    inputs = torch.tensor([[1., 0.], [0., 1.], [1., 1.], [-1., 1.]], dtype=torch.float64)
    labels = torch.tensor([0, 1, 2, 1])
    model, optimizer = build(method)
    reference = Optimizer(model.weight.detach().numpy(), "lion", .01) if method == "lion" else None
    checkpoint = None
    for step in range(8):
        if reference is not None:
            loss = F.cross_entropy(model(inputs), labels)
            gradient = torch.autograd.grad(loss, model.weight)[0].detach().numpy()
            reference.step(gradient)
        update(model, optimizer, method, inputs, labels, step)
        if reference is not None:
            np.testing.assert_allclose(model.weight.detach().numpy(), reference.parameters,
                                       atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(optimizer.state[model.weight]["exp_avg"].numpy(),
                                       reference.state["momentum"], atol=1e-12)
        if step == 3:
            switch(model, optimizer, method, False)
            checkpoint = deepcopy({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                   "rng": torch.get_rng_state(), "next_step": step+1})
    switch(model, optimizer, method, False)
    with torch.no_grad():
        expected_logits = model(inputs).clone()
    resumed, resumed_optimizer = build(method)
    resumed.load_state_dict(checkpoint["model"])
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    torch.set_rng_state(checkpoint["rng"])
    for step in range(checkpoint["next_step"], 8):
        update(resumed, resumed_optimizer, method, inputs, labels, step)
    switch(resumed, resumed_optimizer, method, False)
    with torch.no_grad():
        torch.testing.assert_close(resumed(inputs), expected_logits, rtol=1e-12, atol=1e-12)
        print(method, "evaluation loss:", F.cross_entropy(expected_logits, labels).item())
    print("State keys:", sorted(optimizer.state[model.weight]))
    print("Uninterrupted/resumed evaluation maximum error:",
          (resumed(inputs).detach()-expected_logits).abs().max().item())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["lion", "sophia", "prodigy", "schedule_free"], required=True)
    main(parser.parse_args().method)
