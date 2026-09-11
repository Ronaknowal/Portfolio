"""Independent Fraction enumeration of selection, assignment and shared-error laws."""

import contextlib
import io
import itertools
import json
import math
from fractions import Fraction as F
from pathlib import Path

payload = json.loads(Path("scratch/sampling-measurement-verification/cases.json").read_text(encoding="utf-8"))
cases = payload["cases"]


def close(actual, expected):
    assert math.isclose(actual, float(expected), rel_tol=2e-11, abs_tol=2e-11), (actual, expected)


def mean(values):
    return sum(values, F(0)) / len(values)


def moments(values, weights=None):
    weights = weights or [F(1, len(values))] * len(values)
    mu = sum(w * x for w, x in zip(weights, values))
    return mu, sum(w * (x - mu) ** 2 for w, x in zip(weights, values))


for state in cases["samples"]:
    population, frame, n = list(map(F, state["population"])), state["eligible"], state["size"]
    # Bit masks provide a separate enumeration from the production recursive collector.
    samples = [[frame[j] for j in range(len(frame)) if mask & (1 << j)] for mask in range(1 << len(frame)) if mask.bit_count() == n]
    estimates = [mean([population[i] for i in subset]) for subset in samples]
    mu, variance = moments(estimates)
    target = mean(population)
    close(state["expectation"], mu)
    close(state["variance"], variance)
    close(state["formulaVariance"], variance)
    close(state["mse"], mean([(estimate - target) ** 2 for estimate in estimates]))
    assert {tuple(sorted(s)) for s in samples} == {tuple(s["indices"]) for s in state["samples"]}

for state in cases["inclusion"]:
    values = list(map(F, state["values"]))
    weights = {"equal": [1] * 6, "unequal": [4, 2, 1, 1, 1, 1], "uncovered": [1, 0, 0, 0, 0, 0]}[state["mode"]]
    p = [F(w, sum(weights)) for w in weights]
    subsets = list(itertools.combinations(range(4), 2))
    pi = [sum(probability for subset, probability in zip(subsets, p) if unit in subset) for unit in range(4)]
    for actual, exact in zip(state["inclusion"], pi):
        close(actual, exact)
    if 0 in pi:
        assert state["ht"] is None and state["ratio"] is None and not state["covered"]
        continue
    ht = [sum(values[i] / pi[i] for i in subset) / 4 for subset in subsets]
    ratio = [sum(values[i] / pi[i] for i in subset) / sum(1 / pi[i] for i in subset) for subset in subsets]
    raw = [mean([values[i] for i in subset]) for subset in subsets]
    for name, estimates in [("ht", ht), ("ratio", ratio), ("raw", raw)]:
        mu, variance = moments(estimates, p)
        close(state[name]["mean"], mu)
        close(state[name]["variance"], variance)

for state in cases["readings"]:
    G, m = state["units"], state["repeats"]
    u, e, b = map(lambda value: F(str(value)), [state["unitVariance"], state["readingVariance"], state["bias"]])
    # Sum all covariance entries for the grand mean's weights, not its simplified formula.
    covariance_total = F(0)
    for a in range(G * m):
        for c in range(G * m):
            covariance_total += u * (a // m == c // m) + e * (a == c)
    variance = covariance_total / (G * m) ** 2
    close(state["variance"], variance)
    close(state["mse"], variance + b * b)
    if u + e == 0:
        assert state["correlation"] is None and state["designEffect"] is None
    else:
        close(state["designEffect"], variance / ((u + e) / (G * m)))

for state in cases["assignments"]:
    baseline, treatment = list(map(F, state["baseline"])), list(map(F, state["treated"]))
    allocations = []
    for bits in itertools.product([0, 1], repeat=6):
        if sum(bits) != 3:
            continue
        if state["blocks"] and not all(bits[a] + bits[b] == 1 for a, b in state["blocks"]):
            continue
        allocations.append(bits)
    estimates = [(sum(treatment[i] for i, z in enumerate(bits) if z) - sum(baseline[i] for i, z in enumerate(bits) if not z)) / 3 for bits in allocations]
    mu, variance = moments(estimates)
    close(state["target"], mean([b - a for a, b in zip(baseline, treatment)]))
    close(state["expectation"], mu)
    close(state["variance"], variance)
    if state["design"] == "complete":
        close(state["neymanVariance"], variance)
    by_units = {tuple(i for i, z in enumerate(bits) if z): estimate for bits, estimate in zip(allocations, estimates)}
    assert len(by_units) == len(state["states"])
    for result in state["states"]:
        close(result["difference"], by_units[tuple(result["indices"])])
        assert result["observed"] == [float(treatment[i] if i in result["indices"] else baseline[i]) for i in range(6)]

for state in cases["factorial"]:
    interaction, q = F(state["interaction"]), F(str(state["highBShare"]))
    function = lambda a, b: 10 + 2 * a - b + interaction * a * b
    low, high = function(1, 0) - function(0, 0), function(1, 1) - function(0, 1)
    close(state["averageAEffect"], (1 - q) * low + q * high)
    assert state["cells"] == [[float(function(a, b)) for a in [0, 1]] for b in [0, 1]]

for item in cases["missing"]:
    values, state = item["values"], item["state"]
    possible = []
    for replacements in itertools.product([0, 10], repeat=values.count(None)):
        replacements = iter(replacements)
        possible.append(mean([F(next(replacements)) if value is None else F(value) for value in values]))
    close(state["lower"], min(possible))
    close(state["upper"], max(possible))

native_scopes = {}
for example in payload["examples"]:
    stream, scope = io.StringIO(), {"__name__": "__main__"}
    with contextlib.redirect_stdout(stream):
        exec(compile(example["code"], example["id"] + ".py", "exec"), scope)
    assert stream.getvalue().strip() == example["expected"], example["id"]
    native_scopes[example["id"]] = scope

native_changed = 0
for y0 in itertools.product([-1, 1, 4], repeat=3):
    for effects in [(0, 1, 3), (2, -1, 0)]:
        y1 = [y + e for y, e in zip(y0, effects)]
        for n1 in [1, 2]:
            result = native_scopes["assignments"]["assignment_distribution"](y0, y1, n1)
            estimates = []
            for permutation in itertools.permutations(range(3)):
                group = permutation[:n1]
                estimates.append(mean([F(y1[i]) for i in group]) - mean([F(y0[i]) for i in permutation[n1:]]))
            mu, variance = moments(estimates)
            assert result[1:] == (mean(list(map(F, effects))), mu, variance)
            native_changed += 1
for values in itertools.product([-2, 0, 5], repeat=4):
    pi, expected = native_scopes["inclusion-weights"]["compare_estimators"](values, [4, 2, 1, 1, 1, 1])
    assert expected[1] == mean(list(map(F, values)))
    native_changed += 1
for N in [2, 3, 5, 7]:
    population = [F(i * i - 3 * i) for i in range(N)]
    for n in range(1, N + 1):
        count, mu, variance, bias, mse = native_scopes["finite-samples"]["describe_sampling"](population, list(range(N)), n)
        expected_variance = (1 - F(n, N)) * sum((y - mean(population)) ** 2 for y in population) / ((N - 1) * n)
        assert mu == mean(population) and variance == expected_variance and bias == 0 and mse == variance
        native_changed += 1

print(json.dumps({"passed": True, "modelCases": {key: len(value) for key, value in cases.items()}, "actualPrograms": len(payload["examples"]), "changedNativeHelperCases": native_changed, "oracles": ["Fraction bit-mask subset laws", "Exact weighted design laws", "Full covariance-matrix sum for grouped readings", "Boolean assignment spaces and complete native permutations", "Corner enumeration of missing outcomes"], "limits": ["Finite bounded fixtures and model contracts; no empirical collection or beginner study", "Browser/visual evidence is separate"]}))
