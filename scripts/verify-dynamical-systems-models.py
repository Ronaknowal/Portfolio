"""Independent analytic, rational, linear-algebra and adaptive-ODE checks."""
import itertools
import json
import math
import platform
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import quad, solve_ivp
from scipy.linalg import expm

ROOT = Path("scratch/dynamical-systems-review")
data = json.loads((ROOT / "model-fixtures.json").read_text())
comparisons = 0
max_scalar_error = 0.0
max_lorenz_error = 0.0


def close(actual, expected, tolerance=1e-10):
    global comparisons
    assert math.isfinite(float(actual)) and math.isfinite(float(expected))
    assert abs(actual - expected) <= tolerance * max(1, abs(expected)), (actual, expected, tolerance)
    comparisons += 1


for case in data["cooling"]:
    spec = case["input"]
    for index, row in enumerate(case["output"]["rows"]):
        time = spec["step"] * index
        close(row["exact"], spec["initial"] * math.exp(-spec["decay"] * time))
        exact_fraction = Fraction(spec["initial"]) * (1 - Fraction(spec["decay"]) * Fraction(spec["step"])) ** index
        close(row["euler"], float(exact_fraction))

for case in data["scalar"]:
    spec, result = case["input"], case["output"]
    parameter = spec["parameter"]
    if spec["kind"] == "pitchfork":
        field = lambda t, state: parameter * state - state**3
        potential = lambda value: value**4 / 4 - parameter * value**2 / 2
    else:
        field = lambda t, state: parameter + state - state**3
        potential = lambda value: value**4 / 4 - value**2 / 2 - parameter * value
    times = np.array([row["time"] for row in result["rows"]])
    reference = solve_ivp(field, [0, spec["duration"]], [spec["initial"]], t_eval=times,
                          method="DOP853", rtol=2e-13, atol=2e-14)
    assert reference.success
    values = np.array([row["value"] for row in result["rows"]])
    error = float(np.max(np.abs(values - reference.y[0])))
    max_scalar_error = max(max_scalar_error, error)
    assert error < 3e-7, (spec, error)
    for row in result["rows"]:
        close(row["potential"], potential(row["value"]))
    assert np.max(np.diff([row["potential"] for row in result["rows"]])) < 1e-10
    for root in result["equilibria"]:
        close(float(field(0, root["value"])), 0, 1e-12)

for case in data["planar"]:
    spec = case["input"]
    mode, parameter = spec["mode"], spec["parameter"]
    initial = np.array(spec["initial"])
    matrices = {
        "center": [[0, -1], [1, 0]],
        "spiral": [[-0.4, -1], [1, -0.4]],
        "saddle": [[0.4, 0], [0, -0.5]],
        "transient": [[-1, 6], [0, -2]],
    }
    rows = case["output"]["rows"]
    if mode == "hopf":
        def field(time, state):
            radius_squared = np.dot(state, state)
            return np.array([(parameter - radius_squared) * state[0] - state[1],
                             state[0] + (parameter - radius_squared) * state[1]])
        reference = solve_ivp(field, [0, spec["duration"]], initial,
                              t_eval=[row["time"] for row in rows],
                              method="DOP853", rtol=2e-13, atol=2e-14)
        assert reference.success
        expected_states = reference.y.T
    else:
        expected_states = [expm(np.array(matrices[mode]) * row["time"]) @ initial for row in rows]
    for row, expected in zip(rows, expected_states):
        for actual, value in zip(row["position"], expected):
            close(actual, float(value), 2e-10)
        close(row["radius"], float(np.linalg.norm(expected)), 2e-10)

for case in data["logistic"]:
    spec, result = case["input"], case["output"]
    exact = Fraction(spec["initial"])
    growth = Fraction(spec["growth"])
    for actual in result["values"]:
        close(actual, float(exact), 2e-11)
        exact = growth * exact * (1 - exact)
    for point in result["fixedPoints"]:
        close(spec["growth"] * point["value"] * (1 - point["value"]), point["value"])
    if result["twoCycle"] is not None:
        left, right = result["twoCycle"]["values"]
        close(spec["growth"] * left * (1 - left), right)
        close(spec["growth"] * right * (1 - right), left)
        product = spec["growth"]**2 * (1 - 2 * left) * (1 - 2 * right)
        close(product, result["twoCycle"]["multiplier"])

for case in data["sensitivity"]:
    spec, result = case["input"], case["output"]
    reference = spec["initial"]
    perturbed = reference + spec["difference"]
    gain = Fraction(1)
    for index, row in enumerate(result["rows"]):
        close(row["reference"], reference)
        close(row["other"], perturbed)
        close(row["separation"], abs(perturbed - reference))
        if gain == 0:
            assert row["tangentVanished"] and row["logGain"] is None
        else:
            close(row["logGain"], math.log(abs(float(gain))), 2e-11)
        gain *= Fraction(spec["growth"] * (1 - 2 * reference))
        reference = spec["growth"] * reference * (1 - reference)
        perturbed = spec["growth"] * perturbed * (1 - perturbed)
assert data["criticalEstimate"]["zeroDerivatives"] == 1
assert data["criticalEstimate"]["estimate"] is None
close(data["fixedEstimate"]["estimate"], math.log(2))
for case in data["estimates"]:
    growth = case["input"]["growth"]
    value = case["output"]["estimate"]
    if growth in [2.5, 3.2, 3.5, 3.83]:
        assert value < 0
    if growth in [3.9, 4]:
        assert value > 0
    if growth == 2.5:
        close(value, math.log(0.5))
    if growth == 3.2:
        close(value, math.log(abs(4 + 2 * growth - growth**2)) / 2)

for depth, cylinder_group in itertools.groupby(data["cylinders"], key=lambda row: len(row["word"])):
    cylinders = list(cylinder_group)
    intervals = sorted(row["interval"] for row in cylinders)
    close(intervals[0][0], 0)
    close(intervals[-1][1], 1)
    for previous, following in zip(intervals, intervals[1:]):
        close(previous[1], following[0])
    for row in cylinders:
        close(row["interval"][1] - row["interval"][0], 2**-depth)
        point = Fraction(row["intercept"]) / (1 - Fraction(row["slope"]))
        close(row["periodicPoint"], float(point))
        iterate = point
        for unused in range(depth):
            iterate = 2 * iterate if iterate <= Fraction(1, 2) else 2 * (1 - iterate)
        assert iterate == point
        midpoint = sum(row["interval"]) / 2
        itinerary = ""
        for unused in range(depth):
            itinerary += "L" if midpoint <= 0.5 else "R"
            midpoint = 2 * midpoint if midpoint <= 0.5 else 2 * (1 - midpoint)
        assert itinerary == row["word"]

for row in data["conjugacy"]:
    value = row["value"]
    close(row["logistic"], (1 - math.cos(math.pi * value)) / 2)
    close(4 * row["logistic"] * (1 - row["logistic"]), math.sin(math.pi * row["tent"] / 2)**2)
    expected, error = quad(lambda angle: 2 / math.pi, 0, math.asin(math.sqrt(value)))
    close(row["cdf"], expected)
for low, high in [(0, 0.25), (0.25, 0.75), (0, 1), (0.9, 1)]:
    probability, error = quad(lambda x: 1 / (math.pi * math.sqrt(x * (1 - x))), low, high,
                              epsabs=2e-11, epsrel=2e-11)
    close(probability, 2 / math.pi * (math.asin(math.sqrt(high)) - math.asin(math.sqrt(low))), 2e-10)
for case in data["quantized"]:
    spec = case["input"]
    exact = Fraction(spec["numerator"], 2**spec["bits"])
    for row in case["output"]["rows"]:
        assert Fraction(row["numerator"], row["denominator"]) == exact
        exact = 2 * exact if exact <= Fraction(1, 2) else 2 * (1 - exact)
    assert case["output"]["rows"][spec["bits"] + 1]["numerator"] == 0

for case in data["oscillator"]:
    spec = case["input"]
    step = spec["step"]
    start = np.array(spec["initial"])
    forward = np.array([[1, step], [-step, 1]])
    symplectic = np.array([[1 - step**2, step], [-step, 1]])
    metric = np.array([[1, -step / 2], [-step / 2, 1]])
    assert np.max(np.abs(symplectic.T @ metric @ symplectic - metric)) < 1e-12
    close(float(np.linalg.det(symplectic)), 1)
    initial_energy = float(start @ start) / 2
    for index, row in enumerate(case["output"]["rows"]):
        for key, matrix in [("euler", forward), ("symplectic", symplectic)]:
            expected = np.linalg.matrix_power(matrix, index) @ start
            for actual, value in zip(row[key], expected):
                close(actual, float(value), 2e-9)
        close(row["eulerEnergy"], initial_energy * (1 + step**2)**index, 2e-9)
        close(row["exactEnergy"], initial_energy)
        if step < 2:
            close(row["modifiedEnergy"], float(start @ metric @ start) / 2)

for case in data["lorenz"]:
    spec, result = case["input"], case["output"]
    def field(time, state):
        first, second, third = state
        return [10 * (second - first), first * (spec["rho"] - third) - second,
                first * second - (8 / 3) * third]
    for equilibrium in result["equilibria"]:
        assert max(abs(value) for value in field(0, equilibrium)) < 2e-12
    close(result["divergence"], -41 / 3)
    times = [row["time"] for row in result["rows"]]
    reference = solve_ivp(field, [0, spec["duration"]], spec["initial"], t_eval=times,
                          method="DOP853", rtol=2e-13, atol=2e-14, max_step=0.01)
    assert reference.success
    actual = np.array([row["state"] for row in result["rows"]])
    error = float(np.max(np.abs(actual - reference.y.T)))
    max_lorenz_error = max(max_lorenz_error, error)
    assert error < 3e-5, (spec, error)

result = {
    "verifiedAt": datetime.now(timezone.utc).isoformat(),
    "sourceSha256": data["sourceSha256"],
    "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
    "comparisons": comparisons,
    "cases": {key: len(data[key]) for key in ["cooling", "scalar", "planar", "logistic", "sensitivity", "estimates", "cylinders", "conjugacy", "quantized", "oscillator", "lorenz"]},
    "maxScalarAbsoluteErrorAgainstDop853": max_scalar_error,
    "maxLorenzShortTimeAbsoluteErrorAgainstDop853": max_lorenz_error,
    "invalidCases": data["invalidCases"],
    "limits": "Selected bounded mathematical fixtures; short-time ODE comparison does not certify long chaotic trajectories or arbitrary observed systems. Native displayed programs and browser review remain separate.",
}
(ROOT / "model-results.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
