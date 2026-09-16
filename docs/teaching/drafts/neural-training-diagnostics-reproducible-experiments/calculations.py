"""Author arithmetic, mode, and interactive-fixture checks; no browser code."""

from fractions import Fraction as F
import json
from pathlib import Path

import torch
from torch import nn

PACKET = Path(__file__).resolve().parent
torch.set_default_dtype(torch.float64)


def scalar_step(xs, ys, weight, rate, momentum=F(0), velocity=F(0), omitted=False):
    gradient = sum((weight * x - y) * x for x, y in zip(xs, ys)) / len(xs)
    next_velocity = momentum * velocity + gradient
    return gradient, next_velocity, weight if omitted else weight - rate * next_velocity


def finite_difference():
    x = torch.tensor([1.0, 2.0])
    y = torch.tensor([1.0, 0.0])
    weight = torch.tensor(0.5, requires_grad=True)
    loss = ((weight * x - y).square() / 2).mean()
    loss.backward()
    rows = []
    for step in [0.01, 0.0001, 0.000001]:
        upper = (((weight.detach() + step) * x - y).square() / 2).mean()
        lower = (((weight.detach() - step) * x - y).square() / 2).mean()
        rows.append({"step": step, "finite_difference": ((upper - lower) / (2 * step)).item()})
    return {"loss": loss.item(), "autograd": weight.grad.item(), "estimates": rows}


def mode_probe(values=(1.0, 3.0), stored_mean=0.0, stored_variance=1.0, module="batchnorm"):
    rows = []
    for training in [True, False]:
        for grad_enabled in [True, False]:
            if module == "batchnorm":
                layer = nn.BatchNorm1d(1)
                layer.running_mean.fill_(stored_mean)
                layer.running_var.fill_(stored_variance)
            else:
                layer = nn.Linear(1, 1)
                with torch.no_grad():
                    layer.weight.fill_(2.0)
                    layer.bias.fill_(1.0)
            layer.train(training)
            with torch.set_grad_enabled(grad_enabled):
                output = layer(torch.tensor(values).reshape(2, 1))
            rows.append({"training": training, "grad_enabled": grad_enabled,
                         "output_requires_grad": output.requires_grad,
                         "running_mean": layer.running_mean.item() if module == "batchnorm" else None,
                         "running_variance": layer.running_var.item() if module == "batchnorm" else None,
                         "output": output.detach().flatten().tolist()})
    return rows


def main():
    # Editable scalar investigation: two contrasting inputs plus a zero-gradient null.
    fixtures = []
    for name, xs, ys, weight, rate in [
        ("mechanics_bridge", [F(1), F(2), F(3)], [F(2), F(0), F(1)], F(0), F(1, 10)),
        ("fresh_input", [F(1), F(2)], [F(1), F(0)], F(1, 2), F(1, 10)),
        ("independent_case_a", [F(2), F(3)], [F(1), F(2)], F(1, 4), F(1, 5)),
        ("stationary_null", [F(1), F(2)], [F(1), F(2)], F(1), F(1, 10)),
    ]:
        for omitted in [False, True]:
            g, v, w = scalar_step(xs, ys, weight, rate, omitted=omitted)
            fixtures.append({"case": name, "omitted_step": omitted,
                             "gradient": str(g), "weight_after": str(w)})
    assert scalar_step([F(1), F(2)], [F(1), F(0)], F(1, 2), F(1, 10))[2] == F(17, 40)
    assert scalar_step([F(2), F(3)], [F(1), F(2)], F(1, 4), F(1, 5))[2] == F(29, 40)
    # Restart recurrence: L(w)=(w-3)^2/2, v'=mu*v+g, w'=w-rate*v'.
    g1, v1, w1 = scalar_step([F(1)], [F(3)], F(1), F(1, 10), F(1, 2))
    g2, v2, w2 = scalar_step([F(1)], [F(3)], w1, F(1, 10), F(1, 2), v1)
    _, _, reset_weight = scalar_step([F(1)], [F(3)], w1, F(1, 10), F(1, 2), F(0))
    assert (w1, w2, reset_weight) == (F(6, 5), F(37, 25), F(69, 50))
    # Independently changed practice and the momentum=0 null.
    _, first_velocity, first_weight = scalar_step([F(1)], [F(4)], F(2), F(1, 4), F(3, 4))
    _, _, continued = scalar_step([F(1)], [F(4)], first_weight, F(1, 4), F(3, 4), first_velocity)
    _, _, reset = scalar_step([F(1)], [F(4)], first_weight, F(1, 4), F(3, 4), F(0))
    assert (continued, reset) == (F(13, 4), F(23, 8))
    assert scalar_step([F(1)], [F(3)], w1, F(1, 10), F(0), v1)[2] == scalar_step([F(1)], [F(3)], w1, F(1, 10), F(0), F(0))[2]
    fresh_modes = mode_probe((-2.0, 6.0), 1.0, 4.0)
    linear_modes = mode_probe(module="linear")
    matching_modes = mode_probe((0.0, 1.0), 0.5, 0.5)
    assert all(row["output"] == [3.0, 7.0] for row in linear_modes)
    assert all(row["running_mean"] == 0.5 and row["running_variance"] == 0.5 for row in matching_modes)
    assert abs(fresh_modes[0]["running_mean"] - 1.1) < 1e-12
    assert abs(fresh_modes[0]["running_variance"] - 6.8) < 1e-12
    paired_differences = [F(1, 100), F(-3, 100), F(5, 100)]
    paired_mean = sum(paired_differences) / 3
    paired_variance = sum((v - paired_mean) ** 2 for v in paired_differences) / 2
    assert (paired_mean, paired_variance) == (F(1, 100), F(1, 625))
    fresh_differences = [F(-1, 10), F(1, 20), F(-1, 20)]
    fresh_mean = sum(fresh_differences) / 3
    fresh_variance = sum((v - fresh_mean) ** 2 for v in fresh_differences) / 2
    assert (fresh_mean, fresh_variance) == (F(-1, 30), F(7, 1200))
    assert scalar_step([F(2), F(3)], [F(1), F(2)], F(1, 4), F(0))[2] == F(1, 4)
    result = {"finite_difference": finite_difference(), "mode_probe": mode_probe(),
              "mode_fresh_input": fresh_modes, "mode_linear_null": linear_modes,
              "mode_matching_buffers_null": matching_modes,
              "paired_practice": {"mean": str(paired_mean), "sample_variance": str(paired_variance)},
              "paired_fresh_loss_fixture": {"differences": [str(v) for v in fresh_differences], "mean": str(fresh_mean), "sample_variance": str(fresh_variance)},
              "scalar_fault_fixtures": fixtures,
              "restart_example": {"after_first": str(w1), "continued": str(w2), "reset": str(reset_weight)},
              "restart_practice": {"continued": str(continued), "reset": str(reset)},
              "nulls": {"zero_gradient_hides_omitted_step": True, "zero_momentum_hides_reset_velocity": True}}
    (PACKET / "calculation-results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
