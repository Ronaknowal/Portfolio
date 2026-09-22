"""A matched Euler program, adaptive solve, and continuous-adjoint comparison.

Authoring targets: torch 2.14.0, torchdiffeq 0.2.5. CPU float64.
Run beside neural_ode_study.py; no fitting or dataset download occurs.
"""
import torch
from torch import nn
from torchdiffeq import odeint, odeint_adjoint
from neural_ode_study import integrate


class Decay(nn.Module):
    def __init__(self, rate=-.7):
        super().__init__()
        self.rate = nn.Parameter(torch.tensor(rate, dtype=torch.float64))

    def forward(self, time, state):
        return self.rate * state


def value_and_derivatives(route):
    field = Decay()
    initial = torch.tensor([[1.5], [-.5]], dtype=torch.float64, requires_grad=True)
    times = torch.tensor([0., 1.], dtype=torch.float64)
    if route == "scratch_euler":
        result = integrate(field, initial, steps=8, method="euler")[-1]
    elif route == "library_euler":
        result = odeint(field, initial, times, method="euler",
                        options={"step_size": 1/8})[-1]
    elif route == "adaptive":
        result = odeint(field, initial, times, method="dopri5", rtol=1e-9, atol=1e-11)[-1]
    elif route == "adjoint":
        result = odeint_adjoint(field, initial, times, method="dopri5",
                                rtol=1e-9, atol=1e-11, adjoint_method="dopri5",
                                adjoint_rtol=1e-9, adjoint_atol=1e-11)[-1]
    else:
        raise ValueError(route)
    loss = result.square().sum()/2
    initial_gradient, rate_gradient = torch.autograd.grad(loss, (initial, field.rate))
    return result.detach(), initial_gradient.detach(), rate_gradient.detach()


def main():
    scratch, library = [value_and_derivatives(route)
                        for route in ("scratch_euler", "library_euler")]
    for left, right in zip(scratch, library):
        torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)
    initial = torch.tensor([[1.5], [-.5]], dtype=torch.float64)
    factor = torch.exp(torch.tensor(-.7, dtype=torch.float64))
    exact = (initial*factor, initial*factor.square(), initial.square().sum()*factor.square())
    for route in ("adaptive", "adjoint"):
        actual = value_and_derivatives(route)
        for value, reference in zip(actual, exact):
            torch.testing.assert_close(value, reference, rtol=2e-7, atol=2e-9)
        print(route, "endpoint/initial-gradient/rate-gradient:", *actual)
    print("Euler endpoint, initial gradient, rate gradient:", *library)


if __name__ == "__main__":
    main()
