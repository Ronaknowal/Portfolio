"""Bounded independent mathematical review; not the author's exhaustive test suite."""

from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "src/learn/data/second-order-methods-models.js"
EXPORT = r"""
import * as models from './src/learn/data/second-order-methods-models.js';
const secant = [];
for (const memory of [0, 1, 2, 3]) {
  for (const scaled of [false, true]) {
    for (const reversed of [false, true]) {
      secant.push(models.lbfgsHistoryTrace(memory, scaled, reversed));
    }
  }
}
const fisher = [0, 0.37, 2].flatMap(strength =>
  [0.01, 0.63].map(damping => models.kfacFactorState(strength, damping)));
const shampoo = [1, 2, 3].flatMap(updates =>
  [[0.0001, 17], [0.37, 0], [1, 90]].map(([epsilon, angle]) =>
    models.shampooMatrixState(updates, epsilon, angle)));
const natural = [[0.05, 0.9, 1], [0.95, 0.1, 1], [0.8, 0.8, 1],
  [0.2, 0.8, 0.25], [0.5, 0.25, 0.5], [0.2, 0.8, 0.01]]
  .map(args => models.bernoulliGeometry(...args));
const newton = [[0.2, 0], [0.6, 0.01], [0, 1], [-1.4, 3]]
  .map(args => models.safeguardedNewtonState(...args));
const curvature = [[200, 0, 1.6], [118, 35, 1.9], [2, 90, 1], [6, 17, 0.1]]
  .map(args => models.curvatureTrace(...args, 24));
console.log(JSON.stringify({secant, fisher, shampoo, natural, newton, curvature}));
"""
states = json.loads(subprocess.check_output(
    ["node", "--input-type=module", "-e", EXPORT], cwd=ROOT, text=True,
    encoding="utf-8"))
errors = {}


def close(actual, expected, name, tolerance=2e-11):
    actual, expected = np.asarray(actual), np.asarray(expected)
    error = float(np.max(np.abs(actual - expected)))
    errors[name] = max(errors.get(name, 0), error)
    assert np.allclose(actual, expected, atol=tolerance, rtol=tolerance), (
        name, actual.tolist(), expected.tolist(), error)


# Complement the author's inverse-BFGS formula with the Hessian-form update,
# then invert. This catches switching the two secant directions or V's order.
for state in states["secant"]:
    hessian = np.eye(2) / state["gamma"]
    for pair in state["pairs"]:
        step, change = np.array(pair["step"]), np.array(pair["change"])
        product = hessian @ step
        hessian += (np.outer(change, change) / (step @ change)
                    - np.outer(product, product) / (step @ product))
        assert np.linalg.eigvalsh(hessian).min() > 0
    inverse = np.linalg.inv(hessian)
    close(state["inverseApproximation"], inverse, "inverse from Hessian BFGS")
    close(state["transformed"], np.linalg.solve(hessian, state["gradient"]),
          "two-loop versus Hessian solve")
    if state["pairs"]:
        newest = state["pairs"][-1]
        close(inverse @ newest["change"], newest["step"], "newest inverse secant")


# Differentiate the complete joint log-probability in the declared flattened
# coordinates, then average its score outer products over eight input/outcomes.
# This oracle does not construct the score with the model's outer-product code.
for state in states["fisher"]:
    weights = np.array(state["weights"])
    vector = weights.ravel(order="F")
    fisher = np.zeros((4, 4))
    supervised_gradient = np.zeros(4)
    labels = [(1, 0), (0, 1)]
    inputs = [(1, -1), (1, 2)]

    def log_mass(parameters, input_vector, outcome):
        logits = parameters.reshape((2, 2), order="F") @ input_vector
        return np.sum(np.array(outcome) * logits - np.logaddexp(0, logits))

    def score_for(input_vector, outcome):
        scores = []
        for coordinate in range(4):
            delta = np.eye(4)[coordinate] * 2e-6
            scores.append((log_mass(vector + delta, input_vector, outcome)
                           - log_mass(vector - delta, input_vector, outcome)) / 4e-6)
        return np.array(scores)

    for input_vector, label in zip(inputs, labels):
        for outcome in itertools.product([0, 1], repeat=2):
            mass = np.exp(log_mass(vector, input_vector, outcome))
            score = score_for(input_vector, outcome)
            fisher += 0.5 * mass * np.outer(score, score)
        supervised_gradient -= 0.5 * score_for(input_vector, label)
    close(state["exactFisher"], fisher, "finite-difference joint score Fisher", 1e-9)
    close(state["flatGradient"], supervised_gradient,
          "finite-difference supervised column order", 1e-9)
    close(state["exactDirection"], np.linalg.solve(
        fisher + state["damping"] * np.eye(4), -supervised_gradient),
        "finite-difference Fisher direction", 3e-9)
    input_factor, output_factor = np.array(state["inputFactor"]), np.array(state["outputFactor"])
    approximation = np.kron(input_factor, output_factor)
    damping = state["damping"]
    cross_terms = np.sqrt(damping) * (
        np.kron(input_factor, np.eye(2)) + np.kron(np.eye(2), output_factor))
    close(state["factorDampedFisher"], approximation + damping * np.eye(4) + cross_terms,
          "factor damping cross terms")
    gradient = np.array(state["gradient"])
    two_solves = np.linalg.solve(input_factor, np.linalg.solve(output_factor, gradient).T).T
    close(two_solves.ravel(order="F"), np.linalg.solve(approximation, gradient.ravel(order="F")),
          "column-major two-factor inverse")
    if state["strength"] == 0:
        close(state["approximationError"], 0, "constant covariance exactness")
    else:
        assert state["approximationError"] > 0


def inverse_quarter(matrix):
    values, vectors = np.linalg.eigh(matrix)
    assert values.min() > 0
    return (vectors * values ** (-0.25)) @ vectors.T


for state in states["shampoo"]:
    left, right = [state["epsilon"] * np.eye(2) for _ in range(2)]
    rotation = np.array(state["rotation"])
    original_left, original_right = left.copy(), right.copy()
    for original, frame in zip(state["originalGradients"], state["frames"]):
        original = np.array(original)
        gradient = rotation @ original
        left += gradient @ gradient.T
        right += gradient.T @ gradient
        original_left += original @ original.T
        original_right += original.T @ original
        left_root, right_root = inverse_quarter(left), inverse_quarter(right)
        direction = left_root @ gradient @ right_root
        close(frame["leftRoot"]["matrix"], left_root, "Shampoo NumPy spectral left root")
        close(frame["rightRoot"]["matrix"], right_root, "Shampoo NumPy spectral right root")
        close(np.linalg.matrix_power(np.array(frame["leftRoot"]["matrix"]), 4) @ left,
              np.eye(2), "fourth-root inverse residual")
        close(frame["direction"], direction, "Shampoo direction")
        original_direction = inverse_quarter(original_left) @ original @ inverse_quarter(original_right)
        close(direction, rotation @ original_direction, "orthogonal row equivariance")


for state in states["natural"]:
    p, q, fraction = [state[key] for key in ("probability", "target", "fraction")]
    masses = np.array([1 - p, p])
    scores = np.array([-1 / (1 - p), 1 / p])
    fisher = masses @ scores ** 2
    close(state["probabilityFisher"], fisher, "Bernoulli model expectation")
    close(state["mappedLogitDirection"], q - p, "natural tangent mapping")
    direct = p + fraction * (q - p)
    transformed = 1 / (1 + np.exp(-(np.log(p / (1 - p)) + fraction * (q - p) / (p * (1 - p)))))
    close(state["directNext"], direct, "natural direct endpoint")
    close(state["logitNext"], transformed, "natural logit endpoint")
    for endpoint, key in [(direct, "directKl"), (transformed, "logitKl")]:
        divergence = np.sum(masses * np.log(masses / np.array([1 - endpoint, endpoint])))
        close(state[key], divergence, "Bernoulli actual KL")
    if q != p:
        assert abs(direct - transformed) > 1e-7

for state in states["curvature"]:
    hessian = np.array(state["hessian"])
    transform = np.eye(2) - state["rate"] * hessian
    expected = np.linalg.matrix_power(transform, 24) @ np.ones(2)
    close(state["frames"][-1]["point"], expected, "GD matrix power")
    close(state["newtonDirection"], -np.ones(2), "coupled Newton solve")

for state in states["newton"]:
    if state["singular"]:
        assert state["direction"] is None and state["samples"] == []
        continue
    point, direction = np.array(state["point"]), np.array(state["direction"])
    close(np.diag(state["shifted"]) @ direction, -np.array(state["gradient"]),
          "damped Newton solve")
    if state["accepted"]:
        assert state["slope"] < 0
        assert state["accepted"]["value"] <= state["accepted"]["threshold"]
        assert all(not trial["passes"] for trial in state["attempts"][:-1])
    for sample in state["samples"]:
        multiplier = sample["multiplier"]
        x, y = point + multiplier * direction
        actual = (x * x - 1) ** 2 / 4 + y * y / 2
        close(sample["actual"], actual, "nonlinear loss along fixed direction")

# Six changed hand calculations, using exact rationals where applicable.
hessian = np.array([[6, 2], [2, 2]])
gradient = hessian @ [2, 3]
assert list(gradient) == [18, 10]
close(np.linalg.solve(hessian, -gradient), [-2, -3], "practice 1 coupled solve")
raw_gradient = Fraction(-24, 125)
raw_curvature = Fraction(-22, 25)
raw_direction = -raw_gradient / raw_curvature
assert raw_direction == Fraction(-12, 55)
assert Fraction(1, 5) + raw_direction == Fraction(-1, 55)
assert raw_direction * raw_gradient > 0
inverse = np.array([[0.75, -0.5], [-0.5, 1]])
close(inverse @ [2, 1], [1, 0], "practice 3 secant")
close(-inverse @ [1, 2], [0.25, -1.5], "practice 3 descent")
assert Fraction(1, 2) + Fraction(1, 2) * (Fraction(1, 4) - Fraction(1, 2)) == Fraction(3, 8)
exact = (Fraction(1, 4) + 4 * Fraction(4, 25)) / 2
factored = Fraction(5, 2) * (Fraction(1, 4) + Fraction(4, 25)) / 2
assert (exact, factored, factored - exact) == (Fraction(89, 200), Fraction(41, 80), Fraction(27, 400))
diagonal_gradient = np.diag([3., 1.])
diagonal_root = inverse_quarter(np.diag([10., 2.]))
close(diagonal_root @ diagonal_gradient @ diagonal_root,
      np.diag([3 / np.sqrt(10), 1 / np.sqrt(2)]), "practice 6 spectral transform")

result = {
    "status": "passed",
    "reviewed_at": datetime.now(timezone.utc).isoformat(),
    "review_type": "Independent mathematical draft review; not final source freeze or browser evidence",
    "model_sha256_at_review": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
    "case_counts": {key: len(value) for key, value in states.items()},
    "exact_or_changed_hand_exercises": 6,
    "seventh_exercise": "Reviewed experiment requirements; author preparing worked native comparison separately",
    "maximum_absolute_errors": errors,
    "limitations": ["Finite fixtures complement, not replace, author native and browser verification.",
                    "No final hash assertion: author source remains under active review."]
}
destination = ROOT / "docs/teaching/evidence/second-order-independent-mathematics.json"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
