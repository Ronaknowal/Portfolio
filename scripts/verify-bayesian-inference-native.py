"""Independent library integrals/distributions and executed lesson-program checks."""
import contextlib
import ast
from datetime import datetime, timezone
from fractions import Fraction
import io
import itertools
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.stats import beta, betabinom, binom, gamma, norm, t

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "scratch/bayesian-inference-review"
fixture = json.loads((DESTINATION / "model-fixtures.json").read_text(encoding="utf-8"))
checks = 0


def close(actual, expected, *, atol=3e-11, rtol=3e-10):
    global checks
    assert math.isclose(actual, float(expected), abs_tol=atol, rel_tol=rtol), (actual, expected)
    checks += 1


for row in fixture["beta"]:
    distribution = beta(row["a"], row["b"])
    if "p" in row:
        close(row["quantile"], distribution.ppf(row["p"]))
    else:
        x = row["x"]
        close(row["density"], distribution.pdf(x))
        close(row["cdf"], distribution.cdf(x))
        close(row["sf"], distribution.sf(x))
        close(row["cdf"] + row["sf"], 1)

for row in fixture["updates"]:
    distribution = beta(row["a"], row["b"])
    tail = (1 - row["level"]) / 2
    close(row["mean"], distribution.mean())
    close(row["variance"], distribution.var())
    close(row["low"], distribution.ppf(tail))
    close(row["high"], distribution.isf(tail))
    close(row["above"], distribution.sf(row["threshold"]))
    close(quad(distribution.pdf, row["low"], row["high"], epsabs=1e-11)[0], row["level"])
    for point in row["points"]:
        close(point["posterior"], distribution.pdf(point["x"]))
        close(point["prior"], beta.pdf(point["x"], row["prior"]["a"], row["prior"]["b"]))
    for point in row["shade"]:
        close(point["density"], distribution.pdf(point["x"]))

for row in fixture["batches"]:
    distribution = betabinom(row["m"], row["a"], row["b"])
    plugin = binom(row["m"], row["meanRate"])
    close(row["mean"], distribution.mean())
    close(row["variance"], distribution.var())
    close(row["pluginVariance"], plugin.var())
    close(row["integratedTail"], distribution.sf(row["threshold"] - 1))
    close(row["pluginTail"], plugin.sf(row["threshold"] - 1))
    close(sum(point["integrated"] for point in row["points"]), 1)
    for point in row["points"]:
        close(point["integrated"], distribution.pmf(point["k"]))
        close(point["plugin"], plugin.pmf(point["k"]))
    if row["m"] == 2:
        # Integrate the likelihood and normalized continuous prior independently.
        close(row["points"][2]["integrated"], quad(lambda theta: theta**2 * beta.pdf(theta, row["a"], row["b"]), 0, 1)[0])

for row in fixture["gamma"]:
    distribution = gamma(row["shape"], scale=1 / row["rate"])
    if "p" in row:
        close(row["quantile"], distribution.ppf(row["p"]))
    else:
        close(row["density"], distribution.pdf(row["x"]))
        close(row["cdf"], distribution.cdf(row["x"]))
        close(row["sf"], distribution.sf(row["x"]))

for row in fixture["exposures"]:
    distribution = gamma(row["shape"], scale=1 / row["rate"])
    close(row["mean"], distribution.mean())
    close(row["variance"], distribution.var())
    close(row["low"], distribution.ppf(.025))
    close(row["high"], distribution.ppf(.975))
    assert distribution.cdf(row["max"]) >= .995 - 1e-12
    for point in row["points"]:
        close(point["posterior"], distribution.pdf(point["x"]))
    # The same lambda/exposure product after hours-to-minutes conversion.
    close(math.exp(-row["mean"] * 2), math.exp(-(row["mean"] / 60) * 120))

for row in fixture["normals"]:
    # Independent density quadrature: prior times a normal sample-mean likelihood.
    likelihood_sd = row["noiseSd"] / math.sqrt(row["n"])
    center = row["mean"]
    scale = math.sqrt(row["variance"])
    log_peak = norm.logpdf(center, row["priorMean"], row["priorSd"]) + norm.logpdf(row["observedMean"], center, likelihood_sd)
    def kernel(z):
        mu = center + scale * z
        return math.exp(norm.logpdf(mu, row["priorMean"], row["priorSd"]) + norm.logpdf(row["observedMean"], mu, likelihood_sd) - log_peak)
    mass = quad(kernel, -12, 12)[0]
    integrated_mean = quad(lambda z: (center + scale * z) * kernel(z), -12, 12)[0] / mass
    integrated_variance = quad(lambda z: (scale * z)**2 * kernel(z), -12, 12)[0] / mass
    close(row["mean"], integrated_mean)
    close(row["variance"], integrated_variance)
    for interval in row["intervals"]:
        close(norm.cdf(interval["high"], interval["mean"], interval["sd"]) - norm.cdf(interval["low"], interval["mean"], interval["sd"]), .95)

sequences = list(itertools.product([0, 1], repeat=10))
for row in fixture["patterns"]:
    weights = [Fraction(0) for _ in range(11)]
    a, b = (2, 2) if row["conditioning"] == "prior" else (10, 4)
    prior_integral = Fraction(math.factorial(a - 1) * math.factorial(b - 1), math.factorial(a + b - 1))
    for sequence in sequences:
        successes = sum(sequence)
        if row["conditioning"] == "same-count" and successes != 8:
            continue
        runs = sum(1 for _ in itertools.groupby(sequence))
        if row["conditioning"] == "same-count":
            weight = Fraction(1, 45)
        else:
            integral = Fraction(math.factorial(a + successes - 1) * math.factorial(b + 9 - successes), math.factorial(a + b + 9))
            weight = integral / prior_integral
        weights[runs] += weight
    assert sum(weights) == 1
    for point in row["masses"]:
        close(point["mass"], weights[point["runs"]])
    close(row["lowerTail"], sum(weights[:row["observedRuns"] + 1]))

native = []
namespaces = {}
for key, example in fixture["examples"].items():
    filename = DESTINATION / "native" / f"{key}.py"
    filename.write_text(example["code"] + "\n", encoding="utf-8")
    run = subprocess.run([sys.executable, str(filename)], capture_output=True, text=True, check=True)
    assert run.stdout.rstrip() == example["expected"], key
    native.append(key)
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(example["code"], key, "exec"), namespace)
    namespaces[key] = namespace

# Independently integrate the nonconjugate likelihood: no grid or log-rescaling.
def exact_detector(successes, failures, sensitivity=.9, false_positive=.1):
    def kernel(theta):
        reported = false_positive + (sensitivity - false_positive) * theta
        return 6 * theta * (1 - theta) * reported**successes * (1 - reported)**failures
    evidence = quad(kernel, 0, 1, epsabs=1e-13)[0]
    return (
        quad(lambda theta: theta * kernel(theta), 0, 1, epsabs=1e-13)[0] / evidence,
        quad(kernel, .7, 1, epsabs=1e-13)[0] / evidence,
        evidence,
    )

for cells in [100, 1000, 10000]:
    approximate = namespaces["nonconjugateGrid"]["grid_inference"](cells)
    exact = exact_detector(8, 2)
    for actual, expected in zip(approximate, exact):
        close(actual, expected, atol=1 / cells**2)

# Independent changed practice cases, including a modified runnable grid program.
close(beta.mean(8, 7), Fraction(8, 15))
close((8 - 1) / (8 + 7 - 2), Fraction(7, 13))
for k, mass in enumerate([Fraction(3, 10), Fraction(2, 5), Fraction(3, 10)]):
    close(betabinom.pmf(k, 2, 2, 2), mass)
close((4 / 6)**7, .05852766346593506)
close(1 / (1/4 + 2), Fraction(4, 9))
close((2 * 2) / (1/4 + 2), Fraction(16, 9))
changed_ast = ast.parse(fixture["examples"]["nonconjugateGrid"]["code"])
changed_counts = 0
for node in ast.walk(changed_ast):
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult) and isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Call):
        if node.left.value == 8 and ast.unparse(node.right) == "log(reported_probability)":
            node.left.value = 4
            changed_counts += 1
        elif node.left.value == 2 and ast.unparse(node.right) == "log1p(-reported_probability)":
            node.left.value = 6
            changed_counts += 1
assert changed_counts == 2
changed_code = ast.unparse(changed_ast)
assert changed_code != fixture["examples"]["nonconjugateGrid"]["code"]
changed = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(changed_code, "changed-practice-grid", "exec"), changed)
changed_exact = exact_detector(4, 6)
for actual, expected in zip(changed["grid_inference"](10000), changed_exact):
    close(actual, expected, atol=1e-8)
close(exact_detector(0, 0)[0], .5)
close(exact_detector(4, 6, 1, 0)[0], Fraction(3, 7))
close(exact_detector(4, 6, .5, .5)[0], .5)
close(beta.var(100, 1), .00009611, atol=5e-9)
assert beta.var(100, 2) > beta.var(100, 1)
close(8 / 2, 4)
close(t.var(8, scale=math.sqrt(15.4 * 6 / (4 * 5))), 6.16)

result = {
    "at": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
    "scipy": scipy.__version__, "numpy": np.__version__,
    "independentNumericComparisons": checks, "exactSequenceReferenceStates": 6,
    "executedExactOutputPrograms": native, "changedPracticeGrid": changed_exact,
    "invalidModelInputs": fixture["invalidCases"],
    "limits": "Bounded analytic teaching models; numerical comparisons are not arbitrary-parameter error proofs. No clinical data or benchmark measurements."
}
(DESTINATION / "native-results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
