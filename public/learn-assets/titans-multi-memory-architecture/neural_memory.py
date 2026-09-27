"""Inspectable fast weights; no global mutable model or request state."""
import torch
from torch.nn import functional as F


def read_memory(parameters, key):
    input_weight, input_bias, output_weight, output_bias = parameters
    hidden = F.silu(key @ input_weight.T + input_bias)
    return (hidden @ output_weight.T + output_bias).squeeze(-1)


def write_memory(parameters, momentum, key, value, rate, retention,
                 decay, differentiable=False):
    loss = 0.5 * (read_memory(parameters, key) - value).square().mean()
    gradients = torch.autograd.grad(loss, parameters, create_graph=differentiable)
    next_momentum = tuple(retention * old - rate * gradient
                          for old, gradient in zip(momentum, gradients))
    next_parameters = tuple((1 - decay) * weight + change
                            for weight, change in zip(parameters, next_momentum))
    if not differentiable:
        next_parameters = tuple(weight.detach().requires_grad_()
                                for weight in next_parameters)
        next_momentum = tuple(change.detach() for change in next_momentum)
    return next_parameters, next_momentum, loss, gradients


def initialize_memory(seed, input_size=9, hidden_size=8):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(input_size, hidden_size),
                                torch.nn.SiLU(), torch.nn.Linear(hidden_size, 1))
    return tuple(model.double().parameters())


def copy_parameters(parameters):
    return tuple(weight.detach().clone().requires_grad_() for weight in parameters)
