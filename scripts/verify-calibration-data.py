"""Rebuild the calibration lesson's measured data module from the files it serves.

This is the data layer's trust root. It does three things.

  1. **Re-derives both frozen result files.** Every number the page quotes from
     `checked-results.json` and `experiment-results.json` is recomputed here from
     the two CSV files this lesson serves, by code that does not import the
     packet's module. The conformal ranks are re-derived by scanning integers in
     exact rational arithmetic; the isotonic map by the max-min formula over
     lower and upper sets; the Platt map by a damped Newton fit of the same
     objective; the ridge coefficients by solving the penalised normal equations
     directly; and every count, loss, width and coverage figure by explicit
     loops rather than by a metrics call.

  2. **Counts its own coverage honestly.** Coverage is the number of LEAF paths
     in those two JSON files that an assertion actually reached, collected by
     the comparison helpers themselves rather than by a hand-maintained tally.
     Anything left uncovered is named in the evidence file with its reason.

  3. **Generates `src/learn/data/calibration-data.js`.** Read-only by default:
     without `--write` the module on disk must equal a fresh generation byte for
     byte, so an edit to the generated file fails here.

Two boundaries are enforced rather than trusted. The banknote CSV's four data
roles are taken by POSITION, and this checks that the positions still line up
with the legacy split labels the file carries — a reordered CSV would silently
move eighty rows from probability calibration into conformal calibration. And
the eighty assessment rows are checked to be disjoint from every other role, so
no investigation on the page can reach a number the learner has not earned.

Run:
  scratch/lesson-tools/Scripts/python.exe scripts/verify-calibration-data.py --write
  scratch/lesson-tools/Scripts/python.exe scripts/verify-calibration-data.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.frozen import FrozenEstimator
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "docs/teaching/drafts/calibration-conformal-prediction"
ASSETS = ROOT / "public/learn-assets/calibration"
MODULE = ROOT / "src/learn/data/calibration-data.js"
EVIDENCE = ROOT / "docs/teaching/evidence/calibration-data.json"

write = "--write" in sys.argv
failures: list[str] = []
checks = 0
covered: set[str] = set()


def note(condition, label):
    global checks
    checks += 1
    if not condition:
        failures.append(label)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def leaf_paths(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaf_paths(item, f"{prefix}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaf_paths(item, f"{prefix}/{index}")
    else:
        yield prefix


def navigate(root, path):
    node = root
    for step in [part for part in path.split("/") if part]:
        node = node[int(step)] if isinstance(node, list) else node[step]
    return node


def agree(file_key, path, actual, tolerance=1e-9, label=None):
    """Compare a recomputed value with the frozen file, marking every leaf it reaches."""
    global checks
    checks += 1
    root = FROZEN[file_key]
    try:
        expected = navigate(root, path)
    except (KeyError, IndexError, TypeError):
        failures.append(f"{file_key}{path}: the frozen file has no such path")
        return
    problems = _compare(expected, actual, f"{file_key}{path}", tolerance)
    for leaf in leaf_paths(expected, path):
        covered.add(f"{file_key}{leaf}")
    if problems:
        failures.append(label or problems[0])


def _compare(expected, actual, where, tolerance):
    problems = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{where}: expected an object, recomputed {type(actual).__name__}"]
        if set(expected) != set(actual):
            return [f"{where}: keys differ, frozen {sorted(expected)} recomputed {sorted(actual)}"]
        for key in expected:
            problems += _compare(expected[key], actual[key], f"{where}/{key}", tolerance)
        return problems
    if isinstance(expected, list):
        if not isinstance(actual, (list, tuple)) or len(expected) != len(actual):
            return [f"{where}: length {len(expected)} versus "
                    f"{len(actual) if hasattr(actual, '__len__') else 'not a list'}"]
        for index, item in enumerate(expected):
            problems += _compare(item, actual[index], f"{where}/{index}", tolerance)
            if len(problems) > 3:
                return problems
        return problems
    if expected is None or actual is None:
        return [] if expected is actual else [f"{where}: {expected!r} versus {actual!r}"]
    if isinstance(expected, bool) or isinstance(actual, bool):
        return [] if bool(expected) == bool(actual) else [f"{where}: {expected} versus {actual}"]
    if isinstance(expected, (int, float)):
        gap = abs(float(expected) - float(actual))
        if gap <= tolerance * max(1.0, abs(float(expected))):
            return []
        return [f"{where}: {expected} versus {actual} (gap {gap})"]
    return [] if expected == actual else [f"{where}: {expected!r} versus {actual!r}"]


# ------------------------------------------------ independent building blocks

def independent_rank(n: int, alpha) -> int:
    """Smallest rank whose share of the n+1 places reaches the requested coverage.

    Exact rational arithmetic, no ceiling and no quantile call.
    """
    target = 1 - Fraction(str(alpha))
    for j in range(1, n + 2):
        if Fraction(j, n + 1) >= target:
            return j
    raise AssertionError("no rank reaches the requested coverage")


def independent_threshold(scores, alpha):
    k = independent_rank(len(scores), alpha)
    if k > len(scores):
        return math.inf, k
    return sorted(float(value) for value in scores)[k - 1], k


def independent_isotonic_blocks(scores, labels):
    """Isotonic fit by the max-min formula; returns the distinct knots and values."""
    order = sorted(range(len(scores)), key=lambda index: (scores[index], index))
    knots: list[float] = []
    totals: list[float] = []
    weights: list[int] = []
    for index in order:
        if knots and knots[-1] == scores[index]:
            totals[-1] += float(labels[index])
            weights[-1] += 1
        else:
            knots.append(float(scores[index]))
            totals.append(float(labels[index]))
            weights.append(1)
    size = len(knots)
    prefix_total = np.concatenate([[0.0], np.cumsum(totals)])
    prefix_weight = np.concatenate([[0.0], np.cumsum(weights)])

    def block_mean(u, v):
        return (prefix_total[v + 1] - prefix_total[u]) / (prefix_weight[v + 1] - prefix_weight[u])

    fitted = []
    for i in range(size):
        best = max(min(block_mean(u, v) for v in range(i, size)) for u in range(0, i + 1))
        fitted.append(best)
    return knots, fitted


def independent_stack_pav(scores, labels):
    """The stack algorithm again, written independently of the packet's.

    Blocks are compared by CROSS-MULTIPLICATION rather than by dividing, which
    is exact on these integer totals and weights, and the trace it produces is
    then checked twice over: against the packet's recorded trace, and against
    the max-min solution, which must give every merged block the same mean.
    """
    order = sorted(range(len(scores)), key=lambda index: (scores[index], index))
    knots: list[float] = []
    totals: list[float] = []
    weights: list[int] = []
    for index in order:
        if knots and knots[-1] == scores[index]:
            totals[-1] += float(labels[index])
            weights[-1] += 1
        else:
            knots.append(float(scores[index]))
            totals.append(float(labels[index]))
            weights.append(1)
    stack: list[list] = []
    trace: list[dict] = []
    for position in range(len(knots)):
        stack.append([position, position, totals[position], weights[position]])
        while len(stack) >= 2 and stack[-2][2] * stack[-1][3] > stack[-1][2] * stack[-2][3]:
            right = stack.pop()
            left = stack.pop()
            merged = [left[0], right[1], left[2] + right[2], left[3] + right[3]]
            trace.append({"left": left, "right": right, "merged": merged})
            stack.append(merged)
    return knots, stack, trace


def independent_linear_quantile(values, level):
    """NumPy's default linear convention, written out."""
    ordered = sorted(float(value) for value in values)
    position = level * (len(ordered) - 1)
    lower = math.floor(position)
    if lower >= len(ordered) - 1:
        return ordered[-1]
    return ordered[lower] + (position - lower) * (ordered[lower + 1] - ordered[lower])


def independent_higher_quantile(values, level):
    ordered = sorted(float(value) for value in values)
    return ordered[math.ceil(level * (len(ordered) - 1))]


def independent_temperature(logits, labels, bounds=(.05, 20)):
    """Minimise the same multiclass log loss over inverse temperature by bisecting
    its derivative.

    The objective is convex in the inverse temperature, so the derivative — the
    probability-weighted mean logit minus the true-class logit — crosses zero
    exactly once. Bisecting that root shares nothing with a bounded scalar
    minimiser.
    """
    z = np.asarray(logits, float)
    y = np.asarray(labels, int)

    def loss(beta):
        scaled = z * beta
        largest = scaled.max(axis=1, keepdims=True)
        total = largest[:, 0] + np.log(np.exp(scaled - largest).sum(axis=1))
        return float(np.mean(total - scaled[np.arange(len(y)), y]))

    def derivative(beta):
        scaled = z * beta
        largest = scaled.max(axis=1, keepdims=True)
        weights = np.exp(scaled - largest)
        weights = weights / weights.sum(axis=1, keepdims=True)
        return float(np.mean((weights * z).sum(axis=1) - z[np.arange(len(y)), y]))

    low, high = bounds
    if derivative(low) > 0:
        best = low
    elif derivative(high) < 0:
        best = high
    else:
        for _ in range(400):
            middle = (low + high) / 2
            if derivative(middle) < 0:
                low = middle
            else:
                high = middle
        best = (low + high) / 2
    return best, loss, bounds


def interpolate_clipped(knots, fitted, values):
    out = []
    for x in values:
        if x <= knots[0]:
            out.append(fitted[0])
        elif x >= knots[-1]:
            out.append(fitted[-1])
        else:
            upper = next(index for index, knot in enumerate(knots) if knot >= x)
            lower = upper - 1
            span = knots[upper] - knots[lower]
            out.append(fitted[upper] if span == 0
                       else fitted[lower] + (x - knots[lower]) / span * (fitted[upper] - fitted[lower]))
    return np.array(out)


def independent_platt(scores, labels):
    """Damped Newton on Platt's smoothed objective. Second route to sklearn's fit."""
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, float)
    positives, negatives = int(labels.sum()), int((1 - labels).sum())
    targets = np.where(labels == 1, (positives + 1) / (positives + 2), 1 / (negatives + 2))

    def objective(a, b):
        z = a * scores + b
        return float(np.mean(np.logaddexp(0, z) - targets * z))

    a, b = 0.0, 0.0
    for _ in range(200):
        z = a * scores + b
        q = expit(z)
        residual = q - targets
        w = q * (1 - q)
        g0, g1 = float(np.mean(residual * scores)), float(np.mean(residual))
        if max(abs(g0), abs(g1)) < 1e-14:
            break
        h00, h01, h11 = float(np.mean(w * scores * scores)), float(np.mean(w * scores)), float(np.mean(w))
        determinant = h00 * h11 - h01 * h01
        da, db = ((h11 * g0 - h01 * g1) / determinant, (h00 * g1 - h01 * g0) / determinant) \
            if determinant > 1e-18 else (g0, g1)
        base = objective(a, b)
        step = 1.0
        for _attempt in range(60):
            if objective(a - step * da, b - step * db) < base:
                a, b = a - step * da, b - step * db
                break
            step /= 2
        else:
            break
    return a, b


def independent_auc(scores, labels):
    order = sorted(range(len(scores)), key=lambda index: scores[index])
    ranks = [0.0] * len(scores)
    position = 0
    while position < len(order):
        end = position
        while end + 1 < len(order) and scores[order[end + 1]] == scores[order[position]]:
            end += 1
        average = (position + end) / 2 + 1
        for index in range(position, end + 1):
            ranks[order[index]] = average
        position = end + 1
    positives = sum(1 for value in labels if value == 1)
    negatives = len(labels) - positives
    total = sum(ranks[index] for index, value in enumerate(labels) if value == 1)
    return (total - positives * (positives + 1) / 2) / (positives * negatives)


def independent_bins(probabilities, outcomes, edges):
    rows = []
    for index in range(len(edges) - 1):
        members = []
        for position, value in enumerate(probabilities):
            above = next((slot for slot, edge in enumerate(edges) if edge > value), len(edges))
            if min(above - 1, len(edges) - 2) == index:
                members.append(position)
        if not members:
            rows.append({"bin": index, "count": 0, "mean_p": None, "fraction_positive": None})
            continue
        positive = int(sum(outcomes[position] for position in members))
        rows.append({
            "bin": index, "count": len(members), "positive": positive,
            "mean_p": float(sum(probabilities[position] for position in members) / len(members)),
            "fraction_positive": positive / len(members),
            "source_indices": members,
        })
    ece = sum(row["count"] * abs(row["mean_p"] - row["fraction_positive"])
              for row in rows if row["count"]) / len(probabilities)
    return {"bins": rows, "ece": ece, "count": len(probabilities)}


def load_csv(name, feature_names, target):
    with (ASSETS / name).open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    ids = [int(row["source_row"]) for row in rows]
    splits = [row["split"] for row in rows]
    x = np.array([[float(row[field]) for field in feature_names] for row in rows])
    y = np.array([float(row[target]) for row in rows])
    return rows, ids, splits, x, y


FROZEN = {
    "checked": json.loads((PACKET / "checked-results.json").read_text(encoding="utf-8")),
    "experiment": json.loads((PACKET / "experiment-results.json").read_text(encoding="utf-8")),
}

METHOD_LABELS = {
    "raw_sigmoid_score": "Naive sigmoid of the SVC score",
    "sigmoid": "Held-out sigmoid calibration",
    "isotonic": "Held-out isotonic calibration",
    "temperature": "Held-out temperature scaling",
    "prior_constant": "Training-prior constant",
}
# A second, shorter name for table rows. The full names are what a reader needs
# beside a figure; in a nine-column table they pushed it 254 pixels past its own
# container at desktop width, where the last columns could only be reached by
# scrolling sideways.
METHOD_SHORT = {
    "raw_sigmoid_score": "naive sigmoid",
    "sigmoid": "sigmoid",
    "isotonic": "isotonic",
    "temperature": "temperature",
    "prior_constant": "prior constant",
}
INTERVAL_LABELS = {
    "constant": "Training-mean constant + conformal residuals",
    "ridge_absolute": "Ridge + conformal absolute residuals",
    "raw_quantiles": "Raw fitted .05/.95 quantiles",
    "cqr": "Conformalized quantile regression",
}
INTERVAL_SHORT = {
    "constant": "constant + residuals",
    "ridge_absolute": "ridge + residuals",
    "raw_quantiles": "raw .05/.95 quantiles",
    "cqr": "CQR",
}


def classification():
    names = ["variance", "skewness", "curtosis", "entropy"]
    rows, ids, splits, x, y = load_csv("banknote-subset.csv", names, "class")
    y = y.astype(int)
    note(len(rows) == 480, "the served banknote subset no longer has 480 rows")
    note(len(set(ids)) == 480, "the served banknote subset has repeated source rows")
    train, probability_calibration = np.arange(240), np.arange(240, 320)
    conformal, test = np.arange(320, 400), np.arange(400, 480)
    # The four roles are taken by POSITION. A reordered file would move rows
    # between roles in silence, so the legacy labels are checked against them.
    note(set(splits[:320]) == {"pool"}, "the first 320 rows are no longer the legacy pool rows")
    note(set(splits[320:400]) == {"development"}, "rows 321-400 are no longer the legacy development rows")
    note(set(splits[400:]) == {"test"}, "the last 80 rows are no longer the legacy test rows")
    role_ids = {name: [ids[index] for index in block] for name, block in
                [("train", train), ("probability_calibration", probability_calibration),
                 ("conformal", conformal), ("test", test)]}
    everything = [value for block in role_ids.values() for value in block]
    note(len(set(everything)) == 480,
         f"the four data roles do not partition the 480 rows: {len(set(everything))} distinct rows are assigned")
    note(not (set(role_ids["test"]) & set(role_ids["train"] + role_ids["probability_calibration"]
                                          + role_ids["conformal"])),
         "an assessment row also appears in a fitting or calibration role")
    agree("experiment", "/classification/data_roles", role_ids)
    agree("experiment", "/classification/test_labels", [float(value) for value in y[test]])
    agree("experiment", "/classification/alpha", 0.1)
    agree("experiment", "/classification/primary_predeclared_method", "sigmoid")
    agree("experiment", "/classification/base_parameters",
          {"model": "StandardScaler + SVC", "C": 1, "kernel": "rbf"})

    base = make_pipeline(StandardScaler(), SVC(C=1.0, kernel="rbf")).fit(x[train], y[train])
    models = {}
    for method in ("sigmoid", "isotonic", "temperature"):
        model = CalibratedClassifierCV(FrozenEstimator(base), method=method)
        model.fit(x[probability_calibration], y[probability_calibration])
        note(list(model.classes_) == [0, 1], f"the {method} calibrator reports an unexpected class order")
        models[method] = model

    alpha = 0.1
    edges = [0, 0.2, 0.4, 0.6, 0.8, 1]
    decision_calibration = base.decision_function(x[probability_calibration])
    decision_conformal = base.decision_function(x[conformal])
    decision_test = base.decision_function(x[test])

    # Second route to the two fitted maps that matter, on the same scores.
    platt_a, platt_b = independent_platt(decision_calibration, y[probability_calibration])
    platt_test = expit(platt_a * decision_test + platt_b)
    knots, fitted = independent_isotonic_blocks(list(decision_calibration), list(y[probability_calibration]))
    isotonic_test = interpolate_clipped(knots, fitted, decision_test)

    out = {}
    for name in ("raw_sigmoid_score", "sigmoid", "isotonic", "temperature", "prior_constant"):
        if name == "raw_sigmoid_score":
            conformal_probability, test_probability = expit(decision_conformal), expit(decision_test)
        elif name == "prior_constant":
            constant = float(y[train].mean())
            conformal_probability = np.full(len(conformal), constant)
            test_probability = np.full(len(test), constant)
        else:
            conformal_probability = models[name].predict_proba(x[conformal])[:, 1]
            test_probability = models[name].predict_proba(x[test])[:, 1]
        true_class_probability = np.where(y[conformal] == 1, conformal_probability, 1 - conformal_probability)
        scores = 1 - true_class_probability
        q, k = independent_threshold(scores, alpha)
        agree("experiment", f"/classification/methods/{name}/calibration_scores", [float(v) for v in scores])
        agree("experiment", f"/classification/methods/{name}/rank", k)
        agree("experiment", f"/classification/methods/{name}/q", q)
        agree("experiment", f"/classification/methods/{name}/test_probabilities_class1",
              [float(v) for v in test_probability])
        note(k == 73, f"{name}: the rank for 80 conformal scores at alpha .1 should be 73, not {k}")

        # Sets, coverage, sizes and every loss recomputed with explicit loops.
        sets = [[bool(1 - (1 - p) <= q), bool(1 - p <= q)] for p in test_probability]
        agree("experiment", f"/classification/methods/{name}/test_sets", sets)
        coverage = [row[int(label)] for row, label in zip(sets, y[test])]
        sizes = [sum(1 for value in row if value) for row in sets]
        correct = sum(1 for p, label in zip(test_probability, y[test]) if (p >= 0.5) == (label == 1))
        brier = sum((p - label) ** 2 for p, label in zip(test_probability, y[test])) / len(test)
        log_loss = -sum(math.log(max(min(p if label == 1 else 1 - p, 1 - 1e-15), 1e-15))
                        for p, label in zip(test_probability, y[test])) / len(test)
        agree("experiment", f"/classification/methods/{name}/correct", correct)
        agree("experiment", f"/classification/methods/{name}/brier", brier)
        agree("experiment", f"/classification/methods/{name}/log_loss", log_loss, tolerance=1e-7)
        agree("experiment", f"/classification/methods/{name}/auc",
              independent_auc(list(test_probability), list(y[test])))
        agree("experiment", f"/classification/methods/{name}/covered", sum(1 for value in coverage if value))
        agree("experiment", f"/classification/methods/{name}/test_n", 80)
        agree("experiment", f"/classification/methods/{name}/mean_set_size", sum(sizes) / len(sizes))
        agree("experiment", f"/classification/methods/{name}/size_counts",
              {str(size): sum(1 for value in sizes if value == size) for size in range(3)})
        agree("experiment", f"/classification/methods/{name}/class_coverage", {
            str(label): {
                "covered": sum(1 for value, actual in zip(coverage, y[test]) if actual == label and value),
                "n": int(sum(1 for actual in y[test] if actual == label)),
            } for label in (0, 1)})
        agree("experiment", f"/classification/methods/{name}/reliability",
              independent_bins(list(test_probability), list(y[test]), edges))

        out[name] = {
            "key": name, "label": METHOD_LABELS[name], "shortLabel": METHOD_SHORT[name],
            "rank": k, "q": q,
            "calibrationScores": [float(v) for v in scores],
            "testProbabilities": [float(v) for v in test_probability],
            "correct": correct, "brier": brier, "logLoss": log_loss,
            "auc": independent_auc(list(test_probability), list(y[test])),
            "covered": sum(1 for value in coverage if value),
            "meanSetSize": sum(sizes) / len(sizes),
            "sizeCounts": {str(size): sum(1 for value in sizes if value == size) for size in range(3)},
            "classCoverage": {str(label): {
                "covered": sum(1 for value, actual in zip(coverage, y[test]) if actual == label and value),
                "n": int(sum(1 for actual in y[test] if actual == label)),
            } for label in (0, 1)},
            "reliability": independent_bins(list(test_probability), list(y[test]), edges),
        }

    # The two independently fitted maps must agree with the library's.
    note(max(abs(a - b) for a, b in zip(platt_test, out["sigmoid"]["testProbabilities"])) < 1e-6,
         "the independently fitted Platt map disagrees with scikit-learn's sigmoid calibration")
    note(max(abs(a - b) for a, b in zip(isotonic_test, out["isotonic"]["testProbabilities"])) < 1e-9,
         "the max-min isotonic map disagrees with scikit-learn's isotonic calibration")
    # The constant baseline's tied boundary is the whole point of the fixture.
    constant = float(y[train].mean())
    note(abs(constant - 103 / 240) < 1e-15, "the training prior is no longer 103/240")
    note((1 - constant <= 1 - constant) and not (constant >= 1 - (1 - constant)),
         "the tied-boundary fixture no longer distinguishes the two algebraically equal comparisons")
    note(out["prior_constant"]["covered"] == 80 and out["prior_constant"]["meanSetSize"] == 2,
         "the constant baseline must return both labels everywhere under the direct score comparison")
    note(out["sigmoid"]["sizeCounts"]["0"] == 9 and out["sigmoid"]["sizeCounts"]["1"] == 71,
         "the sigmoid procedure's nine empty sets and 71 singletons have moved")
    note(out["isotonic"]["q"] == 0.0,
         "isotonic's threshold is zero in this run because many conformal scores tie at zero")
    note(all(abs(out[name]["auc"] - 1) < 1e-12
             for name in ("raw_sigmoid_score", "sigmoid", "isotonic", "temperature")),
         "the four SVC-derived scores no longer all have AUC 1 on the assessment rows")

    return {
        "alpha": alpha, "rank": 73, "binEdges": edges,
        "primaryPredeclaredMethod": "sigmoid",
        "baseParameters": {"model": "StandardScaler + SVC", "C": 1, "kernel": "rbf"},
        # Only the assessment rows' identifiers travel to the browser. The page
        # names the other roles by size and never needs to list their rows, and
        # shipping fewer of them keeps the boundary visible in the bundle too.
        "roleSizes": {name: len(block) for name, block in role_ids.items()},
        "testIds": role_ids["test"],
        "testLabels": [int(value) for value in y[test]],
        "trainingPrior": constant,
        "methods": out,
        "methodOrder": ["raw_sigmoid_score", "sigmoid", "isotonic", "temperature", "prior_constant"],
    }


def regression():
    names = ["frequency_hz", "attack_angle_deg", "chord_length_m",
             "free_stream_velocity_m_s", "displacement_thickness_m"]
    rows, ids, splits, x, y = load_csv("airfoil-subset.csv", names, "scaled_sound_pressure_db")
    train = np.array([index for index, row in enumerate(rows) if row["split"] == "train"])
    cal = np.array([index for index, row in enumerate(rows) if row["split"] == "conformal"])
    test = np.array([index for index, row in enumerate(rows) if row["split"] == "test"])
    note([len(train), len(cal), len(test)] == [240, 120, 120], "the airfoil splits are no longer 240/120/120")
    note(len(set(ids)) == 480, "the served airfoil subset has repeated source rows")
    note(not (set(ids[index] for index in test) & set(ids[index] for index in list(train) + list(cal))),
         "an airfoil assessment row also appears in a fitting or calibration role")
    role_ids = {"train": [ids[index] for index in train], "conformal": [ids[index] for index in cal],
                "test": [ids[index] for index in test]}
    agree("experiment", "/regression/data_roles", role_ids)
    agree("experiment", "/regression/alpha", 0.1)
    agree("experiment", "/regression/target_unit", "dB")
    agree("experiment", "/regression/features", names)
    agree("experiment", "/regression/parameters",
          {"ridge_alpha": 1, "quantile_levels": [0.05, 0.95], "trees": 80, "max_depth": 2, "seed": 67})
    agree("experiment", "/regression/test_y", [float(value) for value in y[test]])
    agree("experiment", "/regression/test_frequency_hz", [float(value) for value in x[test, 0]])

    point = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(x[train], y[train])
    prediction = point.predict(x[test])
    # Second route to the same ridge fit: standardise with the training mean and
    # population standard deviation, centre, and solve the penalised normal
    # equations directly. No estimator object is involved.
    mean, deviation = x[train].mean(axis=0), x[train].std(axis=0)
    standard_train = (x[train] - mean) / deviation
    centred = standard_train - standard_train.mean(axis=0)
    target = y[train] - y[train].mean()
    weights = np.linalg.solve(centred.T @ centred + np.eye(x.shape[1]), centred.T @ target)
    own_prediction = ((x[test] - mean) / deviation - standard_train.mean(axis=0)) @ weights + y[train].mean()
    note(float(np.max(np.abs(own_prediction - prediction))) < 1e-8,
         "the directly solved ridge coefficients disagree with the fitted pipeline")

    low = GradientBoostingRegressor(loss="quantile", alpha=.05, n_estimators=80, max_depth=2,
                                    random_state=67).fit(x[train], y[train])
    high = GradientBoostingRegressor(loss="quantile", alpha=.95, n_estimators=80, max_depth=2,
                                     random_state=67).fit(x[train], y[train])
    lower_cal, upper_cal = low.predict(x[cal]), high.predict(x[cal])
    lower_test, upper_test = low.predict(x[test]), high.predict(x[test])
    crossed = {"cal": int((lower_cal > upper_cal).sum()), "test": int((lower_test > upper_test).sum())}
    agree("experiment", "/regression/quantile_crossings", crossed)
    lower_cal, upper_cal = np.minimum(lower_cal, upper_cal), np.maximum(lower_cal, upper_cal)
    lower_test, upper_test = np.minimum(lower_test, upper_test), np.maximum(lower_test, upper_test)

    alpha = 0.1
    absolute_scores = np.abs(y[cal] - point.predict(x[cal]))
    cqr_scores = np.maximum(lower_cal - y[cal], y[cal] - upper_cal)
    constant = float(y[train].mean())
    q_absolute, rank = independent_threshold(absolute_scores, alpha)
    q_cqr, _ = independent_threshold(cqr_scores, alpha)
    q_constant, _ = independent_threshold(np.abs(y[cal] - constant), alpha)
    agree("experiment", "/regression/rank", rank)
    agree("experiment", "/regression/q_absolute", q_absolute)
    agree("experiment", "/regression/q_cqr", q_cqr)
    agree("experiment", "/regression/q_constant", q_constant)
    agree("experiment", "/regression/absolute_scores", [float(v) for v in absolute_scores])
    agree("experiment", "/regression/cqr_scores", [float(v) for v in cqr_scores])
    agree("experiment", "/regression/point_predictions", [float(v) for v in prediction])
    agree("experiment", "/regression/point_mae",
          float(sum(abs(a - b) for a, b in zip(prediction, y[test])) / len(test)))
    agree("experiment", "/regression/constant_mae",
          float(sum(abs(constant - b) for b in y[test]) / len(test)))
    note(rank == 109, f"the rank for 120 calibration scores at alpha .1 should be 109, not {rank}")

    intervals = {
        "constant": (np.full(len(test), constant - q_constant), np.full(len(test), constant + q_constant)),
        "ridge_absolute": (prediction - q_absolute, prediction + q_absolute),
        "raw_quantiles": (lower_test, upper_test),
        "cqr": (lower_test - q_cqr, upper_test + q_cqr),
    }
    out = {}
    frequency = x[test, 0]
    for name, (lower, upper) in intervals.items():
        covered_rows = [bool(lo <= value <= hi) for lo, hi, value in zip(lower, upper, y[test])]
        widths = [max(hi - lo, 0) for lo, hi in zip(lower, upper)]
        groups = {}
        for label, mask in [("below_2000_hz", frequency < 2000), ("at_least_2000_hz", frequency >= 2000)]:
            chosen = [index for index, keep in enumerate(mask) if keep]
            groups[label] = {
                "covered": sum(1 for index in chosen if covered_rows[index]),
                "n": len(chosen),
                "mean_width": sum(widths[index] for index in chosen) / len(chosen),
            }
        agree("experiment", f"/regression/methods/{name}/lower", [float(v) for v in lower])
        agree("experiment", f"/regression/methods/{name}/upper", [float(v) for v in upper])
        agree("experiment", f"/regression/methods/{name}/covered", sum(1 for v in covered_rows if v))
        agree("experiment", f"/regression/methods/{name}/n", 120)
        agree("experiment", f"/regression/methods/{name}/mean_width", sum(widths) / len(widths))
        agree("experiment", f"/regression/methods/{name}/empty_count",
              sum(1 for lo, hi in zip(lower, upper) if lo > hi))
        agree("experiment", f"/regression/methods/{name}/width_quantiles",
              [float(v) for v in np.quantile(widths, [0, .25, .5, .75, 1])])
        agree("experiment", f"/regression/methods/{name}/frequency_groups", groups)
        out[name] = {
            "key": name, "label": INTERVAL_LABELS[name], "shortLabel": INTERVAL_SHORT[name],
            "covered": sum(1 for v in covered_rows if v), "n": 120,
            "meanWidth": sum(widths) / len(widths),
            "emptyCount": sum(1 for lo, hi in zip(lower, upper) if lo > hi),
            "widthQuantiles": [float(v) for v in np.quantile(widths, [0, .25, .5, .75, 1])],
            "frequencyGroups": groups,
        }
    note(out["ridge_absolute"]["frequencyGroups"]["below_2000_hz"] == {"covered": 54, "n": 66,
                                                                      "mean_width": out["ridge_absolute"]["meanWidth"]}
         or out["ridge_absolute"]["frequencyGroups"]["below_2000_hz"]["covered"] == 54,
         "the ridge slice below 2000 Hz is no longer 54 of 66")
    note(out["ridge_absolute"]["frequencyGroups"]["at_least_2000_hz"]["covered"] == 54
         and out["ridge_absolute"]["frequencyGroups"]["at_least_2000_hz"]["n"] == 54,
         "the ridge slice at or above 2000 Hz is no longer 54 of 54")
    note(out["cqr"]["covered"] > out["raw_quantiles"]["covered"],
         "CQR no longer improves on the unadjusted quantiles in this sample")
    note(out["cqr"]["meanWidth"] < out["ridge_absolute"]["meanWidth"]
         and out["cqr"]["covered"] < out["ridge_absolute"]["covered"],
         "the tradeoff the table is placed beside has disappeared: CQR should be narrower and cover less")

    return {
        "alpha": alpha, "rank": rank, "targetUnit": "dB", "features": names,
        "parameters": {"ridgeAlpha": 1, "quantileLevels": [0.05, 0.95], "trees": 80, "maxDepth": 2, "seed": 67},
        "roleSizes": {name: len(block) for name, block in role_ids.items()},
        "qAbsolute": q_absolute, "qCqr": q_cqr, "qConstant": q_constant,
        "trainingMean": constant,
        "quantileCrossings": crossed,
        "testIds": role_ids["test"],
        "testY": [float(value) for value in y[test]],
        "testFrequencyHz": [float(value) for value in x[test, 0]],
        "pointPredictions": [float(v) for v in prediction],
        "rawLower": [float(v) for v in lower_test],
        "rawUpper": [float(v) for v in upper_test],
        "absoluteScores": [float(v) for v in absolute_scores],
        "cqrScores": [float(v) for v in cqr_scores],
        "pointMae": float(sum(abs(a - b) for a, b in zip(prediction, y[test])) / len(test)),
        "constantMae": float(sum(abs(constant - b) for b in y[test]) / len(test)),
        "frequencySplitHz": 2000,
        "methods": out,
        "methodOrder": ["constant", "ridge_absolute", "raw_quantiles", "cqr"],
    }


def constructed():
    """The constructed entries of checked-results.json, re-derived here."""
    checked = FROZEN["checked"]
    probabilities = [.2] * 5 + [.8] * 5
    outcomes = [1, 1, 0, 0, 0, 1, 1, 1, 0, 0]
    agree("checked", "/reliability/predictions", probabilities)
    agree("checked", "/reliability/outcomes", [float(v) for v in outcomes])
    agree("checked", "/reliability/two_bins", independent_bins(probabilities, outcomes, [0, .5, 1]))
    agree("checked", "/reliability/one_bin", independent_bins(probabilities, outcomes, [0, 1]))
    agree("checked", "/reliability/repaired",
          independent_bins([.4] * 5 + [.6] * 5, outcomes, [0, .5, 1]))
    agree("checked", "/reliability/boundary_fixture", independent_bins([0, .5, 1], [0, 1, 1], [0, .5, 1]))
    agree("checked", "/reliability/brier_before",
          sum((p - v) ** 2 for p, v in zip(probabilities, outcomes)) / 10)
    agree("checked", "/reliability/brier_repaired",
          sum((p - v) ** 2 for p, v in zip([.4] * 5 + [.6] * 5, outcomes)) / 10)
    agree("checked", "/reliability/auc_before", independent_auc(probabilities, outcomes))
    agree("checked", "/reliability/auc_repaired", independent_auc([.4] * 5 + [.6] * 5, outcomes))

    scores, labels = [-3, -2, -1, 0, 1, 2, 3, 4], [0, 1, 0, 0, 1, 0, 1, 1]
    knots, fitted = independent_isotonic_blocks(scores, labels)
    agree("checked", "/pav/knots", knots)
    agree("checked", "/pav/fitted", fitted)
    for name, (s, l) in {"pav": (scores, labels),
                         "pav_ties": ([-2, -2, 0, 1], [1, 0, 0, 1]),
                         "pav_ties_changed": ([-2, -2, 0, 1], [1, 0, 1, 1]),
                         "practice/pav": ([1, 2, 3, 4, 5, 6], [0, 1, 0, 1, 0, 1])}.items():
        own_knots, own_fitted = independent_isotonic_blocks(s, l)
        agree("checked", f"/{name}/knots", own_knots)
        agree("checked", f"/{name}/fitted", own_fitted)
        # The stack trace the page steps through, and its blocks.
        stack_knots, stack_blocks, stack_trace = independent_stack_pav(s, l)
        note(stack_knots == own_knots, f"{name}: the two groupings of equal scores disagree")
        agree("checked", f"/{name}/blocks", stack_blocks)
        agree("checked", f"/{name}/merges", stack_trace)
        # Each block's mean must equal the max-min solution over its own span,
        # so the trace is not merely a second copy of the same stack.
        for block in stack_blocks:
            start, end, total, weight = block
            for position in range(start, end + 1):
                note(abs(total / weight - own_fitted[position]) < 1e-12,
                     f"{name}: block {start}-{end} disagrees with the max-min value at knot {position}")
    # sklearn is a third route to the same fits.
    library = IsotonicRegression(out_of_bounds="clip").fit(np.array(scores, float), np.array(labels, float))
    note(float(np.max(np.abs(library.predict(np.array(knots, float)) - np.array(fitted)))) < 1e-12,
         "the max-min isotonic solution disagrees with scikit-learn on the eight-score fixture")

    a, b = independent_platt(scores, labels)
    agree("checked", "/sigmoid/a", a, tolerance=1e-8)
    agree("checked", "/sigmoid/b", b, tolerance=1e-8)
    agree("checked", "/sigmoid/probabilities", [float(expit(a * s + b)) for s in scores], tolerance=1e-8)
    positives = sum(labels)
    negatives = len(labels) - positives
    smoothed = [(positives + 1) / (positives + 2) if v == 1 else 1 / (negatives + 2) for v in labels]
    agree("checked", "/sigmoid/smoothed_targets", smoothed)
    agree("checked", "/sigmoid/objective",
          float(np.mean([np.logaddexp(0, a * s + b) - t * (a * s + b) for s, t in zip(scores, smoothed)])),
          tolerance=1e-9)

    for temperature in ("0.5", "1", "2", "4"):
        scaled = np.array([3, 1, 0], float) / float(temperature)
        shifted = np.exp(scaled - scaled.max())
        agree("checked", f"/temperature/{temperature}", [float(v) for v in shifted / shifted.sum()])

    fit_logits = [[3, 1, 0], [3, 1, 0], [3, 1, 0], [0, 2, 1]]
    fit_labels = [0, 0, 1, 2]
    beta, temperature_loss, temperature_bounds = independent_temperature(fit_logits, fit_labels)
    agree("checked", "/temperature_fit/temperature", 1 / beta, tolerance=1e-6)
    agree("checked", "/temperature_fit/nll_before", temperature_loss(1.0))
    agree("checked", "/temperature_fit/nll_after", temperature_loss(beta), tolerance=1e-9)
    agree("checked", "/temperature_fit/inverse_temperature_bounds", list(temperature_bounds))
    agree("checked", "/temperature_fit/near_search_boundary",
          min(beta - temperature_bounds[0], temperature_bounds[1] - beta) < 1e-5)
    scaled_fit = np.asarray(fit_logits, float) * beta
    shifted_fit = np.exp(scaled_fit - scaled_fit.max(axis=1, keepdims=True))
    agree("checked", "/temperature_fit/probabilities",
          [[float(value) for value in row / row.sum()] for row in shifted_fit], tolerance=1e-6)
    note(temperature_loss(beta) < temperature_loss(1.0),
         "the fitted temperature must reduce the log loss it was fitted on")

    agree("checked", "/rank/linear_quantile_at_k_over_n",
          independent_linear_quantile([.05, .1, .15, .2, .25, .3, .4, .6, .9], 8 / 9))
    agree("checked", "/rank/higher_quantile_at_k_over_n",
          independent_higher_quantile([.05, .1, .15, .2, .25, .3, .4, .6, .9], 8 / 9))

    calibration = [.05, .1, .15, .2, .25, .3, .4, .6, .9]
    q, k = independent_threshold(calibration, .2)
    agree("checked", "/rank/calibration_scores", calibration)
    agree("checked", "/rank/k", k)
    agree("checked", "/rank/q", q)
    agree("checked", "/rank/tiny_alpha_rank", independent_rank(9, .05))
    agree("checked", "/rank/probabilities", [[.8, .15, .05], [.45, .4, .15], [.34, .33, .33]])
    agree("checked", "/rank/prediction_sets",
          [[bool(1 - value <= q) for value in row] for row in [[.8, .15, .05], [.45, .4, .15], [.34, .33, .33]]])
    for key, source in (("rotation", calibration + [.95]), ("tie_rotation", [.2] * 10)):
        rows = []
        for index, held in enumerate(source):
            others = [value for slot, value in enumerate(source) if slot != index]
            own_q, _ = independent_threshold(others, .2)
            rows.append({"test": held, "q": own_q if math.isfinite(own_q) else "infinity",
                         "covered": bool(held <= own_q)})
        agree("checked", f"/rank/{key}/rows", rows)
        agree("checked", f"/rank/{key}/covered", sum(1 for row in rows if row["covered"]))
        agree("checked", f"/rank/{key}/total", len(rows))

    agree("checked", "/aps/probabilities", [.5, .3, .2])
    order = sorted(range(3), key=lambda index: (-[.5, .3, .2][index], index))
    running, aps = 0.0, [0.0] * 3
    for index in order:
        running += [.5, .3, .2][index]
        aps[index] = running
    agree("checked", "/aps/scores", aps)
    agree("checked", "/group_example/weights", [.8, .2])
    agree("checked", "/group_example/coverage", [1, .5])
    agree("checked", "/group_example/marginal", .8 * 1 + .2 * .5)
    agree("checked", "/resolution/rates", [.1, .3])
    agree("checked", "/resolution/coarse", .2)
    agree("checked", "/resolution/coarse_brier",
          sum(.5 * (r * (1 - .2) ** 2 + (1 - r) * .2 ** 2) for r in (.1, .3)))
    agree("checked", "/resolution/full_brier",
          sum(.5 * (r * (1 - r) ** 2 + (1 - r) * r ** 2) for r in (.1, .3)))
    agree("checked", "/resolution/coarse_cost", sum(.5 * 80 * r for r in (.1, .3)))
    agree("checked", "/resolution/full_cost", .5 * 80 * .1 + .5 * 20)
    agree("checked", "/label_shift/sensitivity", .8)
    agree("checked", "/label_shift/false_positive_rate", .2)
    agree("checked", "/label_shift/ppv_at_prior_half", (.8 * .5) / (.8 * .5 + .2 * .5))
    agree("checked", "/label_shift/ppv_at_prior_tenth", (.8 * .1) / (.8 * .1 + .2 * .9))
    agree("checked", "/practice/rank_n14_alpha_point2", independent_rank(14, .2))
    agree("checked", "/practice/cqr_scores", [-2, -1, 0, 1, 3])
    agree("checked", "/practice/cqr_q", independent_threshold([-2, -1, 0, 1, 3], .4)[0])
    agree("checked", "/practice/cqr_negative_q", independent_threshold([-5, -4, -3, -2, -1], .4)[0])

    residuals = [.5, 1, 1.5, 2, 3, 4, 5, 6, 8]
    scales = [1, 1, 1, 1, 2, 2, 2, 3, 4]
    changed = [.5, 1, 1.5, 2, 3, 4, 5, 12, 16]
    agree("checked", "/adaptive/residuals", residuals)
    agree("checked", "/adaptive/scales", scales)
    agree("checked", "/adaptive/changed_residuals", changed)
    agree("checked", "/adaptive/query_predictions", [10, 20])
    agree("checked", "/adaptive/query_scales", [1, 3])
    agree("checked", "/adaptive/absolute_q", independent_threshold(residuals, .2)[0])
    agree("checked", "/adaptive/normalized_q",
          independent_threshold([r / s for r, s in zip(residuals, scales)], .2)[0])
    agree("checked", "/adaptive/changed_q",
          independent_threshold([r / s for r, s in zip(changed, scales)], .2)[0])
    agree("checked", "/adaptive/all_scales_doubled_q",
          independent_threshold([r / (2 * s) for r, s in zip(residuals, scales)], .2)[0])
    boundary = 103 / 240
    agree("checked", "/floating_boundary/p", boundary)
    agree("checked", "/floating_boundary/q", 1 - boundary)
    agree("checked", "/floating_boundary/score_comparison", 1 - boundary <= 1 - boundary)
    agree("checked", "/floating_boundary/rearranged_comparison", boundary >= 1 - (1 - boundary))

    return {
        "sigmoid": {
            "a": checked["sigmoid"]["a"], "b": checked["sigmoid"]["b"],
            "objective": checked["sigmoid"]["objective"],
            "probabilities": checked["sigmoid"]["probabilities"],
            "smoothedTargets": checked["sigmoid"]["smoothed_targets"],
            "solver": "SciPy BFGS on the smoothed log-loss objective, recorded by the content packet",
        },
        "temperatureFit": {
            "temperature": checked["temperature_fit"]["temperature"],
            "nllBefore": checked["temperature_fit"]["nll_before"],
            "nllAfter": checked["temperature_fit"]["nll_after"],
            "inverseTemperatureBounds": checked["temperature_fit"]["inverse_temperature_bounds"],
            "nearSearchBoundary": checked["temperature_fit"]["near_search_boundary"],
            "probabilities": checked["temperature_fit"]["probabilities"],
            "logits": [[3, 1, 0], [3, 1, 0], [3, 1, 0], [0, 2, 1]],
            "labels": [0, 0, 1, 2],
        },
    }


def provenance():
    entries = {}
    for name, description in [
        ("banknote-subset.csv", "Banknote Authentication subset, 480 of 1372 rows"),
        ("airfoil-subset.csv", "Airfoil Self-Noise subset, 480 of 1503 rows"),
        ("calibration_calculations.py", "Every constructed mechanism, complete and runnable"),
        ("uncertainty_experiments.py", "Both offline experiments, complete and runnable"),
        ("reliability_bins.py", "Displayed: binning and ECE"),
        ("monotone_map.py", "Displayed: pool-adjacent-violators"),
        ("conformal_rank.py", "Displayed: the exact finite rank"),
        ("ATTRIBUTION.txt", "Sources, licences, selection rules and role allocation"),
    ]:
        data = (ASSETS / name).read_bytes()
        entries[name] = {"file": f"/learn-assets/calibration/{name}", "bytes": len(data),
                         "sha256": sha(data), "describes": description}
    for name in ("banknote-subset.csv", "airfoil-subset.csv",
                 "calibration_calculations.py", "uncertainty_experiments.py"):
        note((ASSETS / name).read_bytes() == (PACKET / name).read_bytes(),
             f"the served {name} is not byte-identical to the frozen packet's")
    return {
        "attribution": "/learn-assets/calibration/ATTRIBUTION.txt",
        "files": entries,
        "banknote": {
            "name": "Banknote Authentication", "creator": "Volker Lohweg (2012)",
            "doi": "https://doi.org/10.24432/C55P57",
            "record": "https://archive.ics.uci.edu/dataset/267/banknote+authentication",
            "license": "CC BY 4.0", "licenseUrl": "https://creativecommons.org/licenses/by/4.0/",
            "sourceRows": 1372, "retainedRows": 480, "retrieved": "2026-09-12",
            "selection": "NumPy default_rng(23).permutation(1372)[:480]",
            "features": ["variance", "skewness", "curtosis", "entropy"],
            "target": "class", "targetNote": "the source record supplies the codes 0 and 1 and no semantics for them",
        },
        "airfoil": {
            "name": "Airfoil Self-Noise", "creator": "Brooks, Pope and Marcolini (1989)",
            "doi": "https://doi.org/10.24432/C5VW2C",
            "record": "https://archive.ics.uci.edu/dataset/291/airfoil+self+noise",
            "license": "CC BY 4.0", "licenseUrl": "https://creativecommons.org/licenses/by/4.0/",
            "sourceRows": 1503, "retainedRows": 480, "retrieved": "2026-09-12",
            "selection": "NumPy default_rng(67).permutation(1503)[:480]",
            "features": ["frequency in Hz", "angle of attack in degrees", "chord length in m",
                         "free-stream velocity in m/s", "displacement thickness in m"],
            "target": "scaled sound pressure level in dB",
            "targetNote": "a width in dB is a width on the supplied decibel scale, not a linear pressure difference",
        },
        "environment": {"python": "3.12.14", "numpy": "2.3.5", "scipy": "1.18.1", "scikitLearn": "1.9.1"},
    }


def main():
    import scipy
    import sklearn
    agree("experiment", "/versions",
          {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__})
    note((np.__version__, scipy.__version__, sklearn.__version__) == ("2.3.5", "1.18.1", "1.9.1"),
         "the shared runtime's library versions have moved away from the recorded ones")
    module = {
        "provenance": provenance(),
        "classification": classification(),
        "regression": regression(),
        "constructedRecord": constructed(),
    }

    checked_leaves = set(f"checked{path}" for path in leaf_paths(FROZEN["checked"]))
    experiment_leaves = set(f"experiment{path}" for path in leaf_paths(FROZEN["experiment"]))
    all_leaves = checked_leaves | experiment_leaves
    uncovered = sorted(all_leaves - covered)
    note(len(covered & all_leaves) / len(all_leaves) > 0.95,
         f"trust-root coverage fell to {len(covered & all_leaves)} of {len(all_leaves)} asserted leaf paths")

    text = (
        "// Measured data for the Calibration & Conformal Prediction lesson.\n"
        "//\n"
        "// Generated by scripts/verify-calibration-data.py --write. Do not edit by hand.\n"
        "//\n"
        "// Every number here is recomputed from the two CSV files this lesson serves, by code that\n"
        "// does not import the content packet's program, and checked against the packet's frozen\n"
        "// checked-results.json and experiment-results.json leaf by leaf.\n"
        "//\n"
        "// Three kinds of quantity live in this file and are never mixed. The conformal thresholds\n"
        "// and fitted maps are CALIBRATION quantities, learned from their own held-out rows. The\n"
        "// coverage counts, Brier losses and widths are ASSESSMENT quantities: finite counts on 80\n"
        "// and 120 reserved rows, with their denominators kept attached. Nothing here is a\n"
        "// population guarantee, and no investigation on the page reads any of it.\n"
        "export const calibrationData = "
    )
    fresh = text + json.dumps(module, indent=2) + ";\n"
    if write:
        MODULE.write_text(fresh, encoding="utf-8", newline="\n")
    else:
        note(MODULE.exists(), "src/learn/data/calibration-data.js has not been generated")
        if MODULE.exists():
            note(MODULE.read_text(encoding="utf-8") == fresh,
                 "src/learn/data/calibration-data.js differs from a fresh generation")

    if failures:
        for problem in failures[:40]:
            print(f"FAIL {problem}")
        raise SystemExit(f"{len(failures)} of {checks} data checks failed")

    # A counter that is reported but never floored is decoration: a block that
    # stops running still prints PASS, with a smaller number nobody reads. The
    # models verifier has had these floors since phase A; the guard audit found
    # that the other four verifiers report their headline counts unfloored.
    if checks < 285:
        raise SystemExit(f"only {checks} data checks ran; a block did not execute, so this PASS covers "
                         "less than it claims")

    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps({
        "checkedAt": datetime.now(timezone.utc).isoformat(),
        "verifier": "scripts/verify-calibration-data.py",
        "verifierSha256": sha(Path(__file__).read_bytes()),
        "generatedModuleSha256": sha(fresh.encode("utf-8")),
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__},
        "checks": checks,
        "trustRootLeafPaths": {
            "checked-results.json": len(checked_leaves),
            "experiment-results.json": len(experiment_leaves),
            "total": len(all_leaves),
            "asserted": len(covered & all_leaves),
            "uncovered": len(uncovered),
            "uncoveredPaths": uncovered,
        },
        "servedSha256": {path.name: sha(path.read_bytes())
                         for path in sorted(ASSETS.iterdir()) if path.is_file()},
        "scope": "Both frozen result files re-derived from the two served CSV files by code that does not "
                 "import the packet's program. Conformal ranks are scanned in exact rational arithmetic; the "
                 "isotonic map is solved by the max-min formula over lower and upper sets and cross-checked "
                 "against scikit-learn; the Platt map is refitted by damped Newton and compared with "
                 "scikit-learn's sigmoid calibration on the assessment rows; the ridge coefficients are "
                 "obtained by solving the penalised normal equations directly; and every count, loss, set "
                 "size, width and frequency-slice figure is recomputed with explicit loops rather than a "
                 "metrics call. The banknote roles are taken by position and checked against the file's own "
                 "legacy split labels, and the assessment rows are shown disjoint from every fitting and "
                 "calibration role. Coverage below is the number of leaf paths an assertion actually "
                 "reached, collected by the comparison helper itself.",
        "limitations": [
            "The two gradient-boosted quantile models are refitted with the same library, settings and seed "
            "rather than reimplemented; their predictions are reproduced, not independently derived. "
            "Everything computed FROM them — scores, thresholds, intervals, coverage, widths and the "
            "frequency slices — is recomputed here by explicit loops.",
            "scikit-learn's temperature calibration is likewise reproduced by re-running it. The sigmoid and "
            "isotonic maps, which the lesson actually teaches, are both refitted independently.",
            "Byte-identical regeneration of the packet's JSON files is checked by "
            "scripts/verify-calibration-examples.py, which runs the packet's own programs.",
        ],
        "passed": True,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")

    print(f"PASS: {checks} data checks. {len(covered & all_leaves)} of {len(all_leaves)} trust-root leaf paths "
          f"asserted ({len(uncovered)} uncovered), both experiments re-derived from the served CSV files.")


def _record_failure(evidence_path, verifier, error):
    """Write a FAILING evidence record.

    Every check in this file raises or exits, and a raise skips the evidence
    write at the end of main() — which leaves the PREVIOUS run's
    `"passed": true` on disk describing a tree that fails. A reviewer reading
    the evidence directory afterwards sees green for red. A sibling lesson
    shipped exactly that, so the failure path writes its own record.
    """
    import traceback
    try:
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(json.dumps({
            "checkedAt": datetime.now(timezone.utc).isoformat(),
            "verifier": verifier,
            "passed": False,
            "failure": {
                "type": type(error).__name__,
                "message": str(error)[:2000],
                "traceback": traceback.format_exc()[-4000:],
            },
            "note": "This run failed. Written from the failure path so a red tree cannot be read as green "
                    "from an earlier run's evidence file.",
        }, indent=2) + "\n", encoding="utf-8", newline="\n")
    except Exception:  # noqa: BLE001 - the original failure must still surface
        pass


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:  # noqa: BLE001 - includes SystemExit from a FAIL path
        if isinstance(error, SystemExit) and not error.code:
            raise
        _record_failure(EVIDENCE, __file__.replace("\\", "/").split("/scripts/")[-1], error)
        raise
