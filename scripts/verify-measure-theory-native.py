"""Independent finite enumeration, weighted least-squares and quadrature oracles."""
import contextlib
from datetime import datetime, timezone
from fractions import Fraction as F
import io
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.integrate import quad, dblquad

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scratch/measure-theory-verification"
data = json.loads((OUT / "model-states.json").read_text(encoding="utf-8"))
counts = json.loads((OUT / "model-counts.json").read_text(encoding="utf-8"))


def near(actual, expected, tolerance=2e-10):
    assert math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance), (actual, expected)


programs = {}
for key, example in data["examples"].items():
    file = OUT / (key + ".py")
    file.write_text(example["code"] + "\n", encoding="utf-8")
    result = subprocess.run([sys.executable, str(file)], capture_output=True, text=True, check=True)
    assert result.stdout.rstrip() == example["expected"], (key, result.stdout, example["expected"])
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(example["code"], str(file), "exec"), namespace)
    programs[key] = namespace

# Construct distinguishability labels, then test whether event indicator is a function of label.
labels = {
    "none": [0] * 6, "parity": [1, 0, 1, 0, 1, 0],
    "pairs": [0, 0, 1, 1, 2, 2], "full": list(range(6)),
}
for state in data["events"]:
    label = labels[state["partitionId"]]
    indicator = [bool(state["eventMask"] & (1 << i)) for i in range(6)]
    observable = all(indicator[i] == indicator[j] for i in range(6) for j in range(6) if label[i] == label[j])
    assert state["observable"] == observable
    expected_events = {
        tuple(i + 1 for i in range(6) if selection[label[i]])
        for selection in itertools.product([False, True], repeat=max(label) + 1)
    }
    assert set(map(tuple, state["algebra"])) == expected_events
    near(state["ambientProbability"], F(sum(indicator), 6))
    assert state["observableEventCount"] == len(expected_events)
    full = frozenset(range(1, 7))
    algebra = set(map(frozenset, expected_events))
    assert all(full - event in algebra for event in algebra)
    assert all(a | b in algebra for a in algebra for b in algebra)

for state in data["preimages"]:
    values = [0, 0, 0, 1, 2, 3]
    expected = [i + 1 for i, value in enumerate(values) if value <= state["threshold"]]
    assert state["preimage"] == expected
    near(state["probability"], F(len(expected), 6))
    near(state["mean"], 1)

for state in data["mixtures"]:
    w, low, high = map(lambda key: F(str(state[key])), ["atomWeight", "lower", "upper"])
    continuous = (1 - w) * max(F(0), min(F(1), high) - max(F(0), low))
    atom = w if low <= 0 <= high else 0
    near(state["probability"], continuous + atom)
    near(state["continuousMass"], continuous)
    near(state["probability"], state["cdfAtUpper"] - state["cdfBeforeLower"])
    near(state["mean"], (1 - w) / 2)
    near(state["secondMoment"], (1 - w) / 3)
    assert state["hasLebesgueDensity"] == (w == 0)

for state in data["cantor"]:
    n = state["level"]
    starts = [sum(F(2 * bit, 3 ** (j + 1)) for j, bit in enumerate(bits)) for bits in itertools.product([0, 1], repeat=n)]
    for interval, low in zip(state["intervals"], starts):
        near(interval["low"], low)
        near(interval["high"], low + F(1, 3**n))
        near(interval["mass"], F(1, 2**n))
    near(sum(item["high"] - item["low"] for item in state["intervals"]), F(2, 3)**n)
    near(sum(item["mass"] for item in state["intervals"]), 1)
    assert all(a["high"] < b["low"] for a, b in zip(state["intervals"], state["intervals"][1:]))

last = 0
for state in data["simple"]:
    count = 2 ** state["level"]
    # Integrate the actual floor function, independently of the contribution list.
    cutpoints = [math.sqrt(k / count) for k in range(1, count)]
    oracle = quad(lambda x: math.floor(count * x*x) / count, 0, 1, points=cutpoints, limit=200)[0]
    near(state["lowerIntegral"], oracle)
    assert last <= oracle <= 1/3 <= state["upperIntegral"] + 1e-12
    last = oracle
    near(sum(step["mass"] for step in state["steps"]), 1)
    near(programs["simple"]["lower_integral"](state["level"]), oracle)

quadrature_cache = {}
for state in data["limits"]:
    mode, n, x = state["mode"], state["n"], state["x"]
    if mode == "spike":
        value = n if 0 < x < 1/n else 0
        limit = 0
    elif mode == "bounded":
        value, limit = x**n, int(x == 1)
    else:
        value = n if x == 0 else min(n, x**-.5)
        limit = "+Infinity" if x == 0 else x**-.5
    near(state["value"], value)
    if isinstance(limit, str):
        assert state["pointwiseLimit"] == limit
    else:
        near(state["pointwiseLimit"], limit)
    cache_key = (mode, n)
    if cache_key not in quadrature_cache:
        if mode == "spike":
            oracle = quad(lambda u: n, 0, 1/n)[0]
        elif mode == "bounded":
            oracle = quad(lambda u: u**n, 0, 1)[0]
        else:
            oracle = quad(lambda u: min(n, u**-.5), 0, 1, points=[1/n**2], epsabs=1e-11)[0]
        quadrature_cache[cache_key] = oracle
    near(state["integral"], quadrature_cache[cache_key])
    near(state["integralOfLimit"], 2 if mode == "increasing" else 0)
    for point in state["points"]:
        assert 0 <= point["x"] <= 1 and 0 <= point["y"] <= state["yMaximum"] + 1e-10

for state in data["joint"]:
    a, b = state["scaleX"], state["scaleY"]
    original = dblquad(lambda y, x: 4*x*y, state["x0"], state["x1"], state["y0"], state["y1"])[0]
    transformed = dblquad(lambda v, u: 4*u*v/(a*a*b*b), state["u0"], state["u1"], state["v0"], state["v1"])[0]
    near(state["probability"], original)
    near(transformed, original)
    near(state["transformedDensity"] * state["transformedArea"], original)
    near(sum(cell["probability"] for cell in state["cells"]), 1)
    near(state["marginalCellProbability"], quad(lambda x: 2*x, state["x0"], state["x1"])[0])

for state in data["arrays"]:
    r, c = state["rows"], state["columns"]
    # Count nonzero positions independently; include only indices contained in this window.
    plus = min(r, c)
    minus = min(r, max(0, c-1))
    assert state["finiteSum"] == plus - minus
    assert state["finiteAbsoluteSum"] == plus + minus
    assert sum(state["columnSums"]) == sum(state["rowSums"]) == plus - minus

for state in data["conditional"]:
    y, p = np.array(state["values"], float), np.array(state["weights"], float)
    label = labels[state["partitionId"]]
    matrix = np.eye(max(label)+1)[label]
    coefficients = np.linalg.lstsq(np.sqrt(p)[:, None] * matrix, np.sqrt(p) * y, rcond=None)[0]
    prediction = matrix @ coefficients
    actual = np.array([row["prediction"] for row in state["rows"]])
    # Null columns have no uniquely determined least-squares coefficient.
    np.testing.assert_allclose(actual[p > 0], prediction[p > 0], atol=2e-11)
    near(state["risk"], np.dot(p, (y - prediction)**2))
    near(state["mean"], np.dot(p, y))
    near(state["variance"], state["predictionVariance"] + state["risk"])
    np.testing.assert_allclose(matrix.T @ (p * (y-actual)), 0, atol=3e-12)
    competitor = matrix @ (coefficients + np.arange(len(coefficients)) - 2)
    near(np.dot(p, (y-competitor)**2), state["risk"] + np.dot(p, (actual-competitor)**2))

for state in data["ratios"]:
    supported = all(row["p"] > 0 or row["q"] == 0 for row in state["rows"])
    assert state["supported"] == supported
    if supported:
        for row in state["rows"]:
            near(row["p"] * row["ratio"], row["q"])
        near(state["weightedMean"], state["targetMean"])
        near(state["ratioMean"], 1)
    else:
        assert state["weightedMean"] is None and state["ratioMean"] is None

# Changed practice, not copied default fixtures.
p = list(map(F, ["1/10", "4/10", "2/10", "3/10"]))
y = [F(v) for v in [0, 5, 2, 8]]
z = [F(4), F(4), F(28, 5), F(28, 5)]
mean = sum(a*b for a, b in zip(p, y))
risk = sum(a*(b-c)**2 for a, b, c in zip(p, y, z))
variance = sum(a*(b-mean)**2 for a, b in zip(p, y))
assert (mean, risk, variance) == (F(24,5), F(158,25), F(174,25))
assert F(2,5)*(-1) + F(3,5) == F(1,5)
assert F(2,5) + F(3,10) == F(7,10)
near(dblquad(lambda v, u: u*v/9, 0, 1.5, 1, 2)[0], F(3,16))
assert F(2,3)*3 + F(1,3)*9 == 5

# Actual runnable density-ratio helper, independent changed distributions and support.
for p0 in [F(0), F(1,4), F(1,2), F(1)]:
    for q0 in [F(0), F(1,3), F(2,3), F(1)]:
        source, target = [p0, 1-p0], [q0, 1-q0]
        supported = all(p or not q for p, q in zip(source, target))
        if supported:
            ratio = programs["reweighting"]["density_ratio"](source, target)
            assert [p*w for p, w in zip(source, ratio)] == target
        else:
            try:
                programs["reweighting"]["density_ratio"](source, target)
                raise AssertionError("Missing support was not rejected")
            except ValueError:
                pass

result = {
    "status": "passed",
    "completedAt": datetime.now(timezone.utc).isoformat(),
    "counts": counts,
    "independentQuadratures": len(quadrature_cache) + 2*len(data["joint"]) + len(data["simple"]),
    "oracles": ["Exact finite event enumeration and Fraction arithmetic", "Direct ternary-prefix Cantor intervals", "SciPy quadrature of actual functions and transformed cells", "NumPy weighted least-squares for conditional projections", "Changed practice and actual runnable helper support cases"],
    "limits": "Finite tests complement the written infinite-limit proofs; they do not establish those theorems or all measurable-space cases.",
}
(OUT / "native-results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
