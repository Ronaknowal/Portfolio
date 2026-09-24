"""Independent LP, scalar minimization, exact arithmetic and native-output checks."""
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linprog, minimize_scalar
from scipy.special import xlogy
from scipy.stats import wasserstein_distance

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "scratch/optimal-transport-review"
fixtures = json.loads((DIRECTORY / "model-fixtures.json").read_text())
comparisons = 0


def near(left, right, tolerance=1e-8):
    global comparisons
    comparisons += 1
    assert np.allclose(left, right, atol=tolerance, rtol=tolerance), (left, right, tolerance)


def lp(a, b, costs):
    costs = np.asarray(costs)
    rows, columns = costs.shape
    equations = np.vstack([np.kron(np.eye(rows), np.ones((1, columns))),
                           np.kron(np.ones((1, rows)), np.eye(columns))])
    result = linprog(costs.ravel(), A_eq=equations, b_eq=np.r_[a, b], bounds=(0, None), method="highs")
    assert result.success, result.message
    return result


def entropic_objective(plan, costs, epsilon):
    matrix = np.array(plan)
    return float(np.sum(matrix * costs) + epsilon * np.sum(xlogy(matrix, matrix) - matrix))


def scalar_reference(a, b, costs, epsilon):
    lower, upper = max(0., a + b - 1), min(a, b)
    def matrix(t):
        return np.maximum([[t, a - t], [b - t, 1 - a - b + t]], 0)
    if upper - lower < 1e-14:
        return matrix(lower), entropic_objective(matrix(lower), costs, epsilon)
    optimum = minimize_scalar(lambda t: entropic_objective(matrix(t), costs, epsilon),
                              bounds=(lower, upper), method="bounded",
                              options={"xatol": 1e-15, "maxiter": 1000})
    candidates = [lower, upper, optimum.x]
    best = min(candidates, key=lambda t: entropic_objective(matrix(t), costs, epsilon))
    return matrix(best), entropic_objective(matrix(best), costs, epsilon)


lp_cache = {}
for state in fixtures["twoLocation"]:
    key = json.dumps([state["source"], state["target"], state["costs"]])
    if key not in lp_cache:
        lp_cache[key] = lp(state["source"], state["target"], state["costs"])
    reference = lp_cache[key]
    near(state["optimumCost"], reference.fun)
    near(state["optimalCertificate"]["value"], reference.fun)
    p = np.array(state["plan"])
    f, g = np.array(state["dual"]["sourcePotential"]), np.array(state["dual"]["targetPotential"])
    slack = np.array(state["costs"]) - f[:, None] - g[None, :]
    assert slack.min() >= -1e-12
    near(state["dualGap"], np.sum(p * slack))
    near(p.sum(axis=1), state["source"])
    near(p.sum(axis=0), state["target"])

for state in fixtures["ordered"]:
    reference = lp(state["a"], state["b"], state["costs"])
    near(state["cost"], reference.fun)
    near(state["distance"] ** state["order"], reference.fun)
    if state["order"] == 1:
        near(state["distance"], wasserstein_distance(state["source"], state["target"], state["a"], state["b"]))
        near(state["cdfArea"], state["distance"])

for state in fixtures["scaling"]:
    reference, objective = scalar_reference(state["source"][0], state["target"][0], np.array(state["costs"]), state["epsilon"])
    assert state["converged"], state["residual"]
    near(state["plan"], reference, 3e-7)
    near(state["objective"], objective)
    near(state["cost"], np.sum(np.array(state["plan"]) * state["costs"]))
    if state["trace"]:
        kernel = np.exp(-np.array(state["costs"]) / state["epsilon"])
        near(state["trace"][0]["plan"], kernel)
        # Independent direct matrix normalization, not the log-factor recurrence.
        for index, snapshot in enumerate(state["trace"][1:], 1):
            if index % 2:
                kernel *= (np.array(state["source"]) / kernel.sum(axis=1))[:, None]
            else:
                kernel *= (np.array(state["target"]) / kernel.sum(axis=0))[None, :]
            near(snapshot["plan"], kernel, 1e-11)
    unregularized = lp(state["source"], state["target"], state["costs"]).fun
    marginal_entropy = min(-np.sum(xlogy(state["source"], state["source"])), -np.sum(xlogy(state["target"], state["target"])))
    assert -1e-8 <= state["cost"] - unregularized <= state["epsilon"] * marginal_entropy + 1e-8

for state in fixtures["equalWeight"]:
    _, objective = scalar_reference(.5, .5, np.array(state["costs"]), state["epsilon"])
    near(state["objective"], objective, 1e-8)
    near(np.array(state["plan"]).sum(axis=0), [.5, .5], 1e-12)
    near(np.array(state["plan"]).sum(axis=1), [.5, .5], 1e-12)

for state in fixtures["comparisons"]:
    independent = []
    for key in ["cross", "sourceSelf", "targetSelf"]:
        part = state[key]
        _, reference = scalar_reference(.5, .5, np.array(part["costs"]), state["epsilon"])
        independent.append(reference)
        near(part["analytic"]["objective"], reference, 1e-8)
        assert part["converged"] == (part["residual"] <= part["tolerance"])
        if part["converged"]:
            near(part["plan"], part["analytic"]["plan"], 1e-8)
    divergence = independent[0] - (independent[1] + independent[2]) / 2
    near(state["analyticDivergence"], divergence, 1e-8)
    assert divergence >= -1e-8
    target = state["targetLocations"]
    exact = lp([.5, .5], [.5, .5], np.array([[abs(x - y) ** 2 for y in target] for x in [0, 2]])).fun
    near(state["exactSquaredDistance"], exact)
    if target == [0, 2]:
        near(state["analyticDivergence"], 0, 1e-12)
    if target == [1, 3]:
        near(state["analyticDivergence"], 1, 1e-12)

for state in fixtures["stability"]:
    near(state["stable"]["plan"], state["reference"]["plan"], 1e-10)
    near(state["stable"]["cost"] - state["reference"]["cost"], state["offset"], 1e-9)
for state in fixtures["cumulative"]:
    near(state["distance"], wasserstein_distance(state["locations"], state["locations"], state["source"], state["target"]))
    near(sum(gap["area"] for gap in state["gaps"]), state["distance"])

# Changed practice and extension calculations use exact values or independent minimization.
practice = {}
practice["changed_plan"] = F(1, 2) * 2 + F(3, 10)
assert practice["changed_plan"] == F(13, 10)
practice["root_units"] = [F(3, 4) + F(1, 4) * 3, F(3, 4) + F(1, 4) * 9]
assert practice["root_units"] == [F(3, 2), F(3)]
practice["cdf_changed"] = F(1, 2) + F(1, 4) * 2
assert practice["cdf_changed"] == 1
_, practice["entropy_self"] = scalar_reference(.5, .5, np.array([[0, 1], [1, 0]]), 1)
near(practice["entropy_self"], -2.0064088681, 1e-10)
practice["projection_variance"] = F(1, 4) * (-2 - 1) ** 2 + F(3, 4) * (2 - 1) ** 2
assert practice["projection_variance"] == 3
def relaxed(mass):
    return mass + 2 * (mass * math.log(mass / 2) - mass + 2 + mass * math.log(mass / 8) - mass + 8)
numerical = minimize_scalar(relaxed, bounds=(.001, 10), method="bounded")
near(numerical.x, 4 * math.exp(-.25), 1e-6)
near(numerical.fun, 7.539187470857522, 1e-8)
# Independent exact covariance identity for a split source used by the native program.
assert F(1, 4) * 4 + F(1, 4) * 4 == 2
assert F(1, 2) * 1 + F(1, 2) * 1 == 1

examples = json.loads((DIRECTORY / "examples.json").read_text())
for key, example in examples.items():
    result = subprocess.run([sys.executable, "-c", example["code"]], capture_output=True, text=True, check=True, timeout=30)
    assert result.stdout.strip() == example["expected"], (key, result.stdout)
    assert not result.stderr
result = {
    "at": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
    "scipy": scipy.__version__, "numpy": np.__version__, "numericComparisons": comparisons,
    "independentTwoLocationLPs": len(lp_cache), "orderedLPs": len(fixtures["ordered"]),
    "programs": list(examples), "practiceReferences": {key: str(value) for key, value in practice.items()},
    "scope": "Independent finite LP/scalar minimization/direct matrix normalization, SciPy W1, exact practice arithmetic and exact native stdout. No theorem proof is inferred from numeric coverage.",
}
(DIRECTORY / "native-results.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
