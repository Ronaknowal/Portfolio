"""Explicit SGD/AdaGrad/RMSProp/Adam/AdamW versus PyTorch, on supplied gradients.
Install: python -m pip install numpy==2.3.5 torch==2.14.0
Run: python optimizer_library_bridge.py
This checks optimizer arithmetic; it does not benchmark training or autograd.
"""
import copy
import numpy as np
import torch


METHODS = ("sgd", "adagrad", "rmsprop", "adam", "adamw")


def make_optimizer(parameter, name, decay=.1):
    common = dict(lr=.05, weight_decay=decay, foreach=False)
    if name == "sgd":
        return torch.optim.SGD([parameter], momentum=.9, dampening=0, nesterov=False, **common)
    if name == "adagrad":
        return torch.optim.Adagrad([parameter], lr_decay=0, initial_accumulator_value=0,
                                   eps=1e-8, fused=False, **common)
    if name == "rmsprop":
        return torch.optim.RMSprop([parameter], alpha=.95, momentum=0,
                                   centered=False, eps=1e-8, **common)
    if name == "adam":
        return torch.optim.Adam([parameter], betas=(.9, .95), eps=1e-8,
                                amsgrad=False, fused=False,
                                decoupled_weight_decay=False, **common)
    if name == "adamw":
        return torch.optim.AdamW([parameter], betas=(.9, .95), eps=1e-8,
                                 amsgrad=False, fused=False, **common)
    raise ValueError("Unknown optimizer")


def manual_step(theta, first, second, step, gradient, name, decay=.1):
    """Same recurrences as SmallOptimizer above; add an explicit missing event.

    Fixed lr=.05, beta1=.9, beta2=.95 and eps=1e-8. Return a new finite state;
    inputs are never modified, including when arithmetic fails.
    """
    if name not in METHODS or not np.isfinite(decay) or decay < 0:
        raise ValueError("Choose a supported optimizer and finite nonnegative decay")
    if (type(step) is not int or step < 0 or theta.ndim != 1 or theta.size == 0
            or first.shape != theta.shape or second.shape != theta.shape
            or not all(np.isfinite(value).all() for value in (theta, first, second))
            or np.any(second < 0)):
        raise ValueError("Use finite vector state, nonnegative square history and integer clock")
    # None means no update, no decay and no increment of this parameter's clock.
    if gradient is None:
        return theta.copy(), first.copy(), second.copy(), step
    gradient = np.asarray(gradient, dtype=np.float64)
    if gradient.shape != theta.shape or not np.isfinite(gradient).all():
        raise ValueError("Gradient must be finite and match the parameter")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        effective = gradient if name == "adamw" else gradient + decay * theta
        if name == "sgd":
            first = .9 * first + effective
            direction = first
        elif name == "adagrad":
            second = second + effective**2
            direction = effective / (np.sqrt(second) + 1e-8)
        elif name == "rmsprop":
            second = .95 * second + .05 * effective**2
            direction = effective / (np.sqrt(second) + 1e-8)
        else:
            first = .9 * first + .1 * effective
            second = .95 * second + .05 * effective**2
            corrected_first = first / (1 - .9**(step + 1))
            corrected_second = second / (1 - .95**(step + 1))
            direction = corrected_first / (np.sqrt(corrected_second) + 1e-8)
        theta = theta * (1 - .05 * decay) if name == "adamw" else theta.copy()
        theta = theta - .05 * direction
        if not all(np.isfinite(value).all() for value in (theta, first, second)):
            raise FloatingPointError("Unrepresentable update")
    return theta, first, second, step + 1


def check_state(optimizer, parameter, first, second, step, name):
    state = optimizer.state[parameter]
    pairs = ([(state["momentum_buffer"], first)] if name == "sgd" else
             [(state["sum"], second)] if name == "adagrad" else
             [(state["square_avg"], second)] if name == "rmsprop" else
             [(state["exp_avg"], first), (state["exp_avg_sq"], second)])
    if name != "sgd":
        assert int(state["step"]) == step
    for actual, expected in pairs:
        assert np.allclose(actual.numpy(), expected, rtol=1e-12, atol=1e-13)


def main():
    gradients = [np.array([2., 0.]), np.array([2., 4.]), None,
                 np.zeros(2), np.array([-1., .5]), np.array([1e-12, -1e-12])]
    for name in METHODS:
        parameter = torch.tensor([1., -2.], dtype=torch.float64, requires_grad=True)
        optimizer = make_optimizer(parameter, name)
        theta = parameter.detach().numpy().copy()
        first, second, step = np.zeros(2), np.zeros(2), 0
        for index, gradient in enumerate(gradients):
            parameter.grad = None if gradient is None else torch.tensor(gradient, dtype=torch.float64)
            before = parameter.detach().clone()
            optimizer.step()
            theta, first, second, step = manual_step(theta, first, second, step, gradient, name)
            assert np.allclose(parameter.detach().numpy(), theta, rtol=1e-12, atol=1e-12)
            check_state(optimizer, parameter, first, second, step, name)
            if gradient is None:
                assert torch.equal(before, parameter)
            if index == 3:
                assert not torch.equal(before, parameter)  # zero is an active step
                print(f"{name}: None skipped; zero gradient moved parameters")
            if index == 1:
                resumed = parameter.detach().clone().requires_grad_(True)
                resumed_optimizer = make_optimizer(resumed, name)
                resumed_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            elif index > 1:
                resumed.grad = None if gradient is None else torch.tensor(gradient, dtype=torch.float64)
                resumed_optimizer.step()
                assert torch.equal(resumed, parameter)
                check_state(resumed_optimizer, resumed, first, second, step, name)
        print(f"{name}: final {np.round(theta, 6).tolist()}; all parameter/state steps and resumed suffix passed")

        # An epsilon-sensitive first update is not hidden by old large moments.
        tiny = np.array([1e-12, -1e-12])
        fresh = torch.zeros(2, dtype=torch.float64, requires_grad=True)
        fresh_optimizer = make_optimizer(fresh, name, decay=0)
        fresh.grad = torch.tensor(tiny)
        fresh_optimizer.step()
        expected = manual_step(np.zeros(2), np.zeros(2), np.zeros(2), 0, tiny, name, decay=0)
        assert np.allclose(fresh.detach().numpy(), expected[0], rtol=1e-12, atol=1e-16)
        check_state(fresh_optimizer, fresh, *expected[1:], name)

    # Without decay, a zero gradient does not move AdaGrad/basic RMSProp.
    # RMSProp's square history still decays; AdaGrad's cumulative sum does not.
    for name in ("adagrad", "rmsprop"):
        parameter = torch.ones(2, dtype=torch.float64, requires_grad=True)
        optimizer = make_optimizer(parameter, name, decay=0)
        parameter.grad = torch.tensor([2., 4.], dtype=torch.float64)
        optimizer.step()
        theta, first, second, step = manual_step(np.ones(2), np.zeros(2), np.zeros(2), 0,
                                                np.array([2., 4.]), name, decay=0)
        before = second.copy()
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        after = manual_step(theta, first, second, step, np.zeros(2), name, decay=0)
        assert np.array_equal(after[0], theta) and after[3] == step + 1
        assert np.allclose(after[2], before if name == "adagrad" else .95 * before)
        assert np.allclose(parameter.detach().numpy(), after[0], rtol=1e-12, atol=1e-13)
        check_state(optimizer, parameter, *after[1:], name)
    print("Tiny-gradient epsilon checks passed; zero without decay keeps AdaGrad/RMSProp parameters fixed")
    try:
        manual_step(np.ones(2), np.zeros(2), np.zeros(2), 0, np.array([np.nan, 0]), "adamw")
    except ValueError:
        print("Nonfinite gradient rejected before update")
    else:
        raise AssertionError("Expected validation failure")
    initial = np.ones(2)
    try:
        manual_step(initial, np.zeros(2), np.zeros(2), 0, np.full(2, 1e308), "adam")
    except FloatingPointError:
        assert np.array_equal(initial, np.ones(2))
        print("Overflow rejected without modifying supplied state")
    else:
        raise AssertionError("Expected overflow rejection")


if __name__ == "__main__":
    main()
