"""Executed arithmetic and meaningful changed/null fixtures for the manuscript."""
from copy import deepcopy
from itertools import product
from pathlib import Path
import json
import numpy as np
import torch
from optimizer_rules import Optimizer, probabilities, gradient, metrics
from optimizer_study import load_data, serializable

DIRECTORY = Path(__file__).resolve().parent
torch.set_num_threads(1)


def schedule_free_sgd(initial, target, noise, beta=.9, rate=.2):
    average = fast = float(initial)
    trace = []
    for step, perturbation in enumerate(noise, 1):
        training = beta * average + (1 - beta) * fast
        derivative = training - target + perturbation
        fast -= rate * derivative
        average += (fast - average) / step
        trace.append(dict(step=step, training=training, gradient=derivative,
                          fast=fast, average=average, loss=.5 * (average-target)**2))
    return trace


def main():
    results = {}
    initial = np.array([.5, -.7])
    gradients = np.array([[2., -4.], [.3, -.1], [-.2, .8], [0., 0.]])
    own = Optimizer(initial, "adamw", .03)
    native = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
    reference = torch.optim.AdamW([native], lr=.03, weight_decay=.1, foreach=False)
    parity = []
    for derivative in gradients:
        own.step(derivative, weight_decay=.1)
        native.grad = torch.tensor(derivative, dtype=torch.float64)
        reference.step()
        parity.append(float(np.max(np.abs(own.parameters-native.detach().numpy()))))
    assert max(parity) < 1e-14
    results["adamw"] = {"native_max_errors": parity, "final_parameters": own.parameters,
        "impulse_at_step_1000_ratio": (.1 / (1-.9**1000)) / np.sqrt(.001/(1-.999**1000))}
    lion = Optimizer(np.array([.5]), "lion", .01)
    lion.state["momentum"][:] = .2
    lion.step(np.array([-1.]), weight_decay=.2)
    results["lion"] = {"worked": {"parameter": lion.parameters, "momentum": lion.state["momentum"]},
        "fresh": {"initial": -.4, "momentum": -.3, "gradient": 2., "rate": .02,
                  "decay": .1, "blend": -.07, "next_parameter": -.3792, "next_momentum": -.277},
        "zero_blend": {"momentum": 0., "gradient": 0., "next_parameter_with_no_decay": .5}}
    fresh_lion = Optimizer(np.array([-.4]), "lion", .02)
    fresh_lion.state["momentum"][:] = -.3
    fresh_lion.step(np.array([2.]), weight_decay=.1)
    assert np.allclose(fresh_lion.parameters, [-.3792], rtol=0, atol=1e-15)
    assert np.allclose(fresh_lion.state["momentum"], [-.277], rtol=0, atol=1e-15)
    null_lion = Optimizer(np.array([.5]), "lion", .02)
    null_lion.step(np.array([0.]))
    assert null_lion.parameters[0] == .5
    # Exact enumeration proves the batch factor without a noisy Monte Carlo claim.
    inputs = np.array([2., 1.])
    positive_probabilities = np.array([.8, .3])
    mean_estimate = 0.0
    outcomes = []
    for labels in product([0, 1], repeat=2):
        labels = np.array(labels)
        probability = np.prod(np.where(labels, positive_probabilities, 1-positive_probabilities))
        derivative = np.mean(inputs * (positive_probabilities - labels))
        estimate = 2 * derivative**2
        mean_estimate += probability * estimate
        outcomes.append({"labels": labels, "probability": probability, "estimate": estimate})
    exact_diagonal = np.mean(inputs**2 * positive_probabilities * (1-positive_probabilities))
    assert abs(mean_estimate-exact_diagonal) < 1e-15
    matrix = np.array([[2., 3.], [3., 1.]])
    probes = [np.array(probe) for probe in product([-1., 1.], repeat=2)]
    estimates = [probe * (matrix @ probe) for probe in probes]
    assert np.allclose(np.mean(estimates, axis=0), np.diag(matrix))
    results["curvature"] = {"binary_single_true_label_gradient_squared": .16,
        "binary_single_ggn": .64, "batch_outcomes": outcomes,
        "batch_expectation": mean_estimate, "batch_exact": exact_diagonal,
        "hutchinson_matrix": matrix, "hutchinson_estimates": estimates,
        "worked_ratio": [.08, -6, 0], "worked_after_decay_and_update": [.972, 2.06, 2.94]}
    fresh_inputs, fresh_probabilities = np.array([3., 1.]), np.array([.25, .6])
    fresh_outcomes = []
    for sampled_labels in product([0, 1], repeat=2):
        sampled_labels = np.array(sampled_labels)
        mass = np.prod(np.where(sampled_labels, fresh_probabilities, 1-fresh_probabilities))
        average_gradient = np.mean(fresh_inputs*(fresh_probabilities-sampled_labels))
        fresh_outcomes.append(dict(labels=sampled_labels, probability=mass,
                                   gradient=average_gradient, estimate=2*average_gradient**2))
    fresh_exact = float(np.mean(fresh_inputs**2*fresh_probabilities*(1-fresh_probabilities)))
    assert abs(sum(row["probability"]*row["estimate"] for row in fresh_outcomes)-fresh_exact) < 1e-15
    results["curvature"]["fresh"] = dict(inputs=fresh_inputs, probabilities=fresh_probabilities,
        outcomes=fresh_outcomes, exact_diagonal=fresh_exact,
        actual_labels=[1, 0], actual_mean_gradient_squared=float(np.mean(
            fresh_inputs*(fresh_probabilities-np.array([1, 0])))**2))
    prodigy = Optimizer(np.array([0.]), "prodigy", 1.)
    prodigy.state["distance"] = .01
    trace, gradient_history, distance_history = [], [], []
    for step in range(12):
        derivative = prodigy.parameters - 3
        gradient_history.append(derivative.copy())
        distance_history.append(prodigy.state["distance"])
        diagnostics = prodigy.step(derivative)
        # Separately expand the EMA sum rather than repeat its recurrence.
        exact_first = sum(.1 * .9**(step-index) * distance_history[index] * value
                          for index, value in enumerate(gradient_history))
        exact_second = sum(.001 * .999**(step-index) * distance_history[index]**2 * value**2
                           for index, value in enumerate(gradient_history))
        assert np.allclose(exact_first, prodigy.state["momentum"], atol=1e-15)
        assert np.allclose(exact_second, prodigy.state["second"], atol=1e-15)
        trace.append({"step": step+1, "parameter": float(prodigy.parameters[0]), **diagnostics})
    zero = Optimizer(np.array([0.]), "prodigy", 1.)
    zero.step(np.array([0.]))
    assert zero.parameters[0] == 0 and zero.state["distance"] == 1e-6
    results["prodigy"] = {"target": 3, "initial_distance": .01, "trace": trace, "zero_gradient_initial_null": True}
    for name, scale in [("fresh", .3), ("fresh_scale_contrast", 1.)]:
        fresh = Optimizer(np.array([1.]), "prodigy", scale)
        fresh.state["distance"] = .01
        fresh_trace = []
        for step in range(12):
            diagnostics = fresh.step(fresh.parameters+2)
            fresh_trace.append(dict(step=step+1, parameter=float(fresh.parameters[0]), **diagnostics))
        results["prodigy"][name] = dict(initial=1., target=-2., scale=scale, trace=fresh_trace)
    results["schedule_free"] = {
        "worked": schedule_free_sgd(2, 0, [1, -1, .5, -.5]),
        "fresh": schedule_free_sgd(-1, 1, [-.5, .5, 1, -1]),
        "fresh_beta_zero": schedule_free_sgd(-1, 1, [-.5, .5, 1, -1], beta=0),
        "fresh_beta_one": schedule_free_sgd(-1, 1, [-.5, .5, 1, -1], beta=1),
        "stationary_null": schedule_free_sgd(1, 1, [0, 0, 0, 0]),
        "warmup_weights_at_update_three": [1/14, 4/14, 9/14]}
    rotation = np.array([[np.cos(np.pi/6), -np.sin(np.pi/6)],
                         [np.sin(np.pi/6), np.cos(np.pi/6)]])
    hessian = rotation @ np.diag([1., 20.]) @ rotation.T
    start = np.array([2., -1.])
    derivative = hessian @ start
    results["geometry"] = {"hessian": hessian, "initial": start, "gradient": derivative,
        "gd_rate_008": start-.08*derivative,
        "diagonal_newton": start-derivative/np.diag(hessian),
        "full_newton": start-np.linalg.solve(hessian, derivative)}
    results["memory"] = {"parameters": 70_000_000_000, "weight_bytes_bfloat16": 140_000_000_000,
        "adamw_two_fp32_buffers": 560_000_000_000, "lion_one_fp32_buffer": 280_000_000_000,
        "adafactor_matrix_shape": [4096, 4096], "factored_float32_bytes": 32768,
        "dense_second_float32_bytes": 67108864}
    polar_input = np.array([[2., 1.], [1., 2.]])
    left, singular_values, right = np.linalg.svd(polar_input)
    assert np.allclose(left @ right, np.eye(2), rtol=0, atol=1e-15)
    results["polar"] = dict(matrix=polar_input, singular_values=singular_values,
                             ideal_polar=left@right, entrywise_sign=np.sign(polar_input))
    results["practice_loss"] = {
        "A": float(-np.log([.51, .99, .99, .99]).mean()),
        "B": float(-np.log([.49, .999, .999, .999]).mean()),
        "changed_A": float(-np.log([.51, .9, .9, .9]).mean())}
    raw, features, labels, roles = load_data()
    models = json.loads((DIRECTORY / "fitted-optimizer-states.json").read_text())
    edited_fixtures = []
    for role_index in [0, 1]:
        row_index = roles["validation"][role_index]
        image = features[row_index:row_index+1].copy()
        changed = image.copy()
        changed[0, 28] = 1 - changed[0, 28]
        for model in [model for model in models if model["seed"] == 11]:
            optimizer = Optimizer(np.array(model["parameters"]), model["method"], model["rate"])
            optimizer.state = {key: np.array(value, dtype=float) if isinstance(value, list) else value
                               for key, value in deepcopy(model["state"]).items()}
            before = probabilities(changed, optimizer.evaluation_parameters())[0]
            correct_gradient = gradient(changed, labels[row_index:row_index+1], optimizer.parameters)
            wrong_gradient = gradient(changed, (labels[row_index:row_index+1]+1)%10, optimizer.parameters)
            assert np.array_equal(before, probabilities(changed, optimizer.evaluation_parameters())[0])
            assert not np.array_equal(correct_gradient, wrong_gradient)
            original_parameters = optimizer.parameters.copy()
            # A deterministic expected-label diagonal refresh for the diagnostic only.
            # This is explicitly distinguished from the fit's sampled Sophia-G refresh.
            curvature = None
            if optimizer.method == "sophia_g":
                distribution = probabilities(changed, optimizer.parameters)[0]
                curvature = changed[0, :, None]**2 * (distribution*(1-distribution))[None, :]
            optimizer.step(correct_gradient, curvature)
            after = probabilities(changed, optimizer.evaluation_parameters())[0]
            if optimizer.method == "adamw_cosine":
                assert np.array_equal(original_parameters, optimizer.parameters)
                assert np.array_equal(before, after)
            edited_fixtures.append({"source_id": int(raw[row_index, 0]), "method": optimizer.method,
                "label": int(labels[row_index]), "edited_pixel_index": 28,
                "original_intensity": float(image[0, 28]), "edited_intensity": float(changed[0, 28]),
                "before": before, "after_one_diagnostic_update": after})
    results["real_pixel_edit"] = edited_fixtures
    results["environment"] = {"numpy": np.__version__, "torch": torch.__version__, "dtype": "float64"}
    (DIRECTORY / "calculated-inputs.json").write_text(json.dumps(results, indent=2, default=serializable, allow_nan=False)+"\n")
    print(json.dumps({"adamw": results["adamw"], "batch_ggn": mean_estimate,
                      "prodigy_trace": trace, "schedule_free": results["schedule_free"],
                      "fresh_row": edited_fixtures[6]}, default=serializable, indent=2))


if __name__ == "__main__":
    main()
