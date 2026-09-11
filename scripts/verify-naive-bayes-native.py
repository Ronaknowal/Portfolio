"""Independent Fraction, density/quadrature and library oracles for actual lesson code/models."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
from datetime import datetime, timezone
import contextlib
import hashlib
import io
import json
import math
import warnings

import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.stats import norm, multivariate_normal, dirichlet_multinomial
from sklearn.naive_bayes import MultinomialNB

root = Path("scratch/naive-bayes-verification")
examples = json.loads((root / "examples.json").read_text(encoding="utf-8"))
models = json.loads((root / "model-cases.json").read_text(encoding="utf-8"))
namespaces = {}
program_results = []
for item in examples:
    output = io.StringIO()
    namespace = {"__name__": "verified_example"}
    with contextlib.redirect_stdout(output), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exec(compile(item["code"], item["id"] + ".py", "exec"), namespace)
    assert not caught, (item["id"], [str(w.message) for w in caught])
    assert output.getvalue().rstrip() == item["expected"], item["id"]
    namespaces[item["id"]] = namespace
    program_results.append({"id": item["id"], "stdoutMatch": True,
                            "codeSha256": hashlib.sha256(item["code"].encode()).hexdigest()})


def close(actual, expected, tolerance=2e-11):
    assert math.isfinite(actual) and math.isfinite(float(expected)), (actual, expected)
    assert math.isclose(actual, float(expected), rel_tol=tolerance, abs_tol=tolerance), (actual, expected)


def posterior(weights):
    return None if not sum(weights) else [weight / sum(weights) for weight in weights]


words = ["free", "money", "win", "meeting", "agenda"]
raw_counts = [[0, 0, 0, 1, 1], [3, 2, 1, 0, 0]]
class_counts = [1, 2]
for case in models["token"]:
    alpha = F(str(case["alpha"]))
    theta = [[(F(value) + alpha) / (sum(row) + 5 * alpha) for value in row] for row in raw_counts]
    tokens = case["text"].split()
    weights = [F(1, 3), F(2, 3)]
    for step, frame in enumerate(case["frames"]):
        if step:
            j = words.index(tokens[step - 1])
            weights = [weights[c] * theta[c][j] for c in [0, 1]]
        expected = posterior(weights)
        if expected is None:
            assert frame["probabilities"] is None
        else:
            for c in [0, 1]:
                close(frame["probabilities"][c], expected[c])
        for c in [0, 1]:
            if weights[c] == 0:
                assert frame["scores"][c] == "-Infinity"
            else:
                close(frame["scores"][c], math.log(float(weights[c])))

for case in models["presence"]:
    counts = case["counts"]
    binary = [int(value > 0) for value in counts]
    occurrence = [[F(v, 3) for v in [1, 1, 1, 2, 2]], [F(v, 4) for v in [3, 3, 2, 1, 1]]]
    weights = []
    count_weights = []
    for c in [0, 1]:
        term = F(class_counts[c], 3)
        for j in range(5):
            term *= occurrence[c][j] if binary[j] else 1 - occurrence[c][j]
        weights.append(term)
        term = F(class_counts[c], 3)
        for j in range(5):
            term *= F(raw_counts[c][j] + 1, sum(raw_counts[c]) + 5) ** counts[j]
        count_weights.append(term)
    for c in [0, 1]:
        close(case["bernoulli"]["probabilities"][c], posterior(weights)[c])
        close(case["multinomial"]["probabilities"][c], posterior(count_weights)[c])

max_integral_error = 0.0
for case in models["gaussian"]:
    x, width = case["reading"], case["width"]
    weights = []
    for c, scale in enumerate([1, 2]):
        density = norm.pdf(x, scale=scale)
        close(case["densities"][c], density)
        integral = quad(lambda t: norm.pdf(t, scale=scale), x - width/2, x + width/2,
                        epsabs=1e-13, epsrel=1e-13)[0]
        error = abs(case["masses"][c] - integral)
        max_integral_error = max(max_integral_error, error)
        assert error < 1e-9, error
        weights.append(density / 2)
    for c in [0, 1]:
        close(case["probabilities"][c], posterior(weights)[c])
    for c, curve in enumerate(case["curves"]):
        for x, density in curve:
            close(density, norm.pdf(x, scale=[1, 2][c]))

for case in models["geometry"]:
    v0, v1 = case["variances"]
    scores = [multivariate_normal.logpdf(case["point"], mean, variance * np.eye(2)) + math.log(.5)
              for mean, variance in [([-2, -2], v0), ([2, 2], v1)]]
    for c in [0, 1]:
        close(case["scores"][c], scores[c])
    for point in case["boundary"]:
        delta = multivariate_normal.logpdf(point, [2, 2], v1*np.eye(2)) - multivariate_normal.logpdf(point, [-2, -2], v0*np.eye(2))
        assert abs(delta) < 8e-13, (case["mode"], point, delta)
    for cell in case["grid"]:
        x, y = cell["x"], cell["y"]
        delta = 8*(x+y) if case["mode"] == "equal" else .75*(x*x+y*y)+5*(x+y)+6-math.log(4)
        assert cell["label"] == int(delta > 0), (cell, delta)

for case in models["copied"]:
    prior = F(str(case["prior"]))
    accuracy, brier = F(0), F(0)
    for observed, row in zip([True, False], case["rows"]):
        likelihood = [F(2, 5), F(4, 5)] if observed else [F(3, 5), F(1, 5)]
        actual = [(1-prior)*likelihood[0], prior*likelihood[1]]
        fake = [(1-prior)*likelihood[0]**case["copies"], prior*likelihood[1]**case["copies"]]
        p = posterior(fake)[1]
        chosen = int(p > F(1, 2))
        assert row["prediction"] == chosen
        close(row["reported"], p)
        close(row["actual"], posterior(actual)[1])
        accuracy += actual[chosen]
        brier += actual[0]*p*p + actual[1]*(1-p)**2
    close(case["accuracy"], accuracy)
    close(case["brier"], brier)

base_cases = [(F(str(p)), y) for p, y in [
    (.05, 0), (.1, 0), (.1, 1), (.2, 0), (.25, 0), (.4, 1),
    (.55, 0), (.6, 1), (.75, 1), (.8, 0), (.9, 1), (.95, 1)]]
for case in models["reliability"]:
    points = [(F(3, 20) + F(7, 10)*p if case["compression"] else p, y) for p, y in base_cases]
    for bin_index, displayed in enumerate(case["bins"]):
        selected = [(p, y) for p, y in points if min(case["binCount"]-1, int(p*case["binCount"])) == bin_index]
        assert displayed["count"] == len(selected)
        if selected:
            close(displayed["meanPrediction"], sum(p for p, _ in selected) / len(selected))
            close(displayed["observedFraction"], F(sum(y for _, y in selected), len(selected)))
        else:
            assert displayed["meanPrediction"] is None and displayed["observedFraction"] is None
    close(case["brier"], sum((p-y)**2 for p, y in points)/len(points))
    close(case["logLoss"], sum(-math.log(float(p if y else 1-p)) for p, y in points)/len(points))

# Actual CountNaiveBayes implementation versus independently enumerated count laws.
Count = namespaces["count-classifier"]["CountNaiveBayes"]
native_count_cases = 0
vectors = list(product(range(3), repeat=2))
for first, second in product(vectors, repeat=2):
    X = np.array([first, second])
    y = np.array([0, 1])
    for alpha in [F(0), F(1, 2), F(1), F(2)]:
        if alpha == 0 and (sum(first) == 0 or sum(second) == 0):
            try:
                Count(float(alpha)).fit(X, y)
            except ValueError:
                native_count_cases += 1
                continue
            raise AssertionError("Undefined token table accepted")
        fitted = Count(float(alpha)).fit(X, y)
        for query in vectors:
            theta = [[(F(value)+alpha)/(sum(row)+2*alpha) for value in row] for row in X]
            weights = [F(1, 2)*theta[c][0]**query[0]*theta[c][1]**query[1] for c in [0, 1]]
            truth = posterior(weights)
            try:
                actual = fitted.predict_proba([query])[0]
            except ValueError:
                assert truth is None
            else:
                assert truth is not None
                for c in [0, 1]:
                    close(actual[c], truth[c])
            native_count_cases += 1

Gaussian = namespaces["gaussian-fit"]["DiagonalGaussianNB"]
rng = np.random.default_rng(811)
native_gaussian_cases = 0
for dimensions in [1, 2, 4]:
    for _ in range(12):
        X = rng.integers(-7, 8, size=(10, dimensions)).astype(float)
        y = np.array([0]*4+[1]*6)
        fitted = Gaussian(variance_floor=.25).fit(X, y)
        exact_means, exact_variances = [], []
        for c in [0, 1]:
            rows = X[y == c].astype(int).tolist()
            means = [sum(F(row[j]) for row in rows) / len(rows) for j in range(dimensions)]
            variances = [sum((F(row[j])-means[j])**2 for row in rows) / len(rows) + F(1, 4)
                         for j in range(dimensions)]
            exact_means.append(means)
            exact_variances.append(variances)
        for query in rng.integers(-8, 9, size=(5, dimensions)):
            expected_scores = [multivariate_normal.logpdf(query, np.array(exact_means[c], dtype=float),
                               np.diag(np.array(exact_variances[c], dtype=float))) + math.log([.4, .6][c]) for c in [0, 1]]
            actual = fitted.predict_log_proba([query])[0]
            for c in [0, 1]:
                close(actual[c], expected_scores[c]-logsumexp(expected_scores))
            native_gaussian_cases += 1

mass = namespaces["count-law"]["multinomial_mass"]
integrated = namespaces["count-law"]["integrated_mass"]
native_integrated_cases = 0
for a, b in product(range(1, 6), repeat=2):
    for total in range(8):
        assert sum(mass([k, total-k], [F(a, a+b), F(b, a+b)]) for k in range(total+1)) == 1
        assert sum(integrated([k, total-k], [a, b]) for k in range(total+1)) == 1
        for k in range(total+1):
            close(integrated([k, total-k], [a, b]), dirichlet_multinomial.pmf([k, total-k], [a, b], total))
            native_integrated_cases += 1

changed = next(item for item in examples if item["id"] == "calibrated-report")["code"]
for before, after in [
    ("np.random.default_rng(2026)", "np.random.default_rng(2027)"),
    ("reading = 1.5 * y", "reading = 1.0 * y"),
    ("X_cal, y_cal = draw_cases(rng, 800)", "X_cal, y_cal = draw_cases(rng, 400)"),
]:
    assert before in changed, before
    changed = changed.replace(before, after)
output = io.StringIO()
with contextlib.redirect_stdout(output):
    exec(compile(changed, "changed_calibration.py", "exec"), {})
(root / "changed-capstone.py").write_text(changed, encoding="utf-8")
(root / "changed-capstone-output.txt").write_text(output.getvalue(), encoding="utf-8")
result = {
    "completedAt": datetime.now(timezone.utc).isoformat(),
    "programs": program_results,
    "modelCases": {key: len(value) for key, value in models.items()},
    "nativeCountCases": native_count_cases, "nativeGaussianCases": native_gaussian_cases,
    "nativeIntegratedCases": native_integrated_cases,
    "maximumGaussianIntervalAbsoluteError": max_integral_error,
    "changedCapstoneOutput": output.getvalue(),
    "status": "passed; production browser and independent cross-review remain separate",
}
(root / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
