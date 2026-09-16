"""Execute the anomaly-detection lesson's displayed Python programs.

Run with the isolated lesson Python. `--write` regenerates
src/learn/data/anomaly-detection-examples.js from the executed programs;
without it, the recorded output must match a fresh execution.

The temperature program reads the two data files the lesson offers for
download. It runs here against the copies in public/learn-assets, which are the
same bytes a learner receives.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import contextlib
import hashlib
import importlib.metadata
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ASSETS = Path("public/learn-assets/anomaly-detection")

PROGRAMS = {
    "isolation": {
        "file": "isolation_example.py",
        "title": "A compact isolation forest you can inspect",
        "question": "The forest is asked for 256-row subsamples but the data have five rows. Which sample size ends up in the normalizer, and what would change if you used 256 anyway?",
        "code": r"""
import math
import numpy as np


def correction(n):
    if n <= 1:
        return 0.0
    return 2 * sum(1 / j for j in range(1, n)) - 2 * (n - 1) / n


def grow(values, depth, limit, rng):
    lo, hi = float(values.min()), float(values.max())
    if len(values) <= 1 or depth == limit or lo == hi:
        return {"size": len(values)}
    cut = rng.uniform(lo, hi)
    left = values < cut
    if not left.any() or left.all():
        return {"size": len(values)}
    return {
        "cut": cut,
        "left": grow(values[left], depth + 1, limit, rng),
        "right": grow(values[~left], depth + 1, limit, rng),
    }


def path(tree, query, depth=0):
    if "size" in tree:
        return depth + correction(tree["size"])
    child = "left" if query < tree["cut"] else "right"
    return path(tree[child], query, depth + 1)


def fit_forest(values, n_trees=200, max_samples=256, seed=17):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Use at least two finite one-dimensional observations.")
    if type(max_samples) is not int or max_samples < 2:
        raise ValueError("Use an integer max_samples >= 2.")
    if type(n_trees) is not int or n_trees < 1:
        raise ValueError("Use a positive integer number of trees.")
    size = min(max_samples, len(values))
    rng = np.random.default_rng(seed)
    trees = [grow(rng.choice(values, size, replace=False), 0,
                  math.ceil(math.log2(size)), rng) for _ in range(n_trees)]
    return trees, size


def anomaly_scores(fitted, queries):
    trees, size = fitted
    mean_paths = [np.mean([path(tree, x) for tree in trees]) for x in queries]
    return 2 ** (-np.asarray(mean_paths) / correction(size))


x = np.array([0., 1., 2., 3., 12.])
fitted = fit_forest(x)
scores = anomaly_scores(fitted, x)
print("Fitted sample size:", fitted[1])
print("Most easily isolated position:", x[np.argmax(scores)])
print("Scores:", np.round(scores, 4))
print("Duplicate-only scores:", anomaly_scores(fit_forest([2., 2., 2.]), [2., 2.]))
""",
    },
    "lofModes": {
        "file": "lof_modes.py",
        "title": "Training factors and new-query scores are different objects",
        "question": "Both calls use the same six coordinates. Before running: should scoring the training array as queries return the training factors?",
        "code": r"""
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

x = np.array([0., 1., 2., 20., 24., 28.])[:, None]
model = LocalOutlierFactor(n_neighbors=2, novelty=True).fit(x)
print("Training LOF:", np.round(-model.negative_outlier_factor_, 6))
print("New-query LOF:", np.round(-model.score_samples([[6.], [17.]]), 6))
print("Wrong training comparison:", np.round(-model.score_samples(x), 6))
""",
    },
    "lofArithmetic": {
        "file": "lof_arithmetic.py",
        "title": "The core LOF arithmetic in a small program",
        "question": "Which point owns the radius inside each maximum: the query or the neighbour?",
        "code": r"""
import numpy as np

x = np.array([0., 1., 2., 20., 24., 28.])
k = 2
if len(np.unique(x)) != len(x) or not 1 <= k < len(x):
    raise ValueError("Use distinct coordinates and 1 <= k < n.")

distance = abs(x[:, None] - x[None, :])
np.fill_diagonal(distance, np.inf)  # Exclude this row's own identity.
neighbors = np.argsort(distance, axis=1, kind="stable")[:, :k]
radius = distance[np.arange(len(x)), neighbors[:, -1]]
reach = np.maximum(distance[np.arange(len(x))[:, None], neighbors],
                   radius[neighbors])
lrd = 1 / reach.mean(axis=1)
lof = (lrd[neighbors] / lrd[:, None]).mean(axis=1)

queries = np.array([4., 6., 17.])
query_distance = abs(queries[:, None] - x[None, :])
query_neighbors = np.argsort(query_distance, axis=1, kind="stable")[:, :k]
query_reach = np.maximum(
    query_distance[np.arange(len(queries))[:, None], query_neighbors],
    radius[query_neighbors],
)
query_lrd = 1 / query_reach.mean(axis=1)
query_lof = (lrd[query_neighbors] / query_lrd[:, None]).mean(axis=1)
print("Reference radii:", radius)
print("Reference lrd:", np.round(lrd, 6))
print("Reference LOF:", np.round(lof, 6))
print("Query LOF:", np.round(query_lof, 6))
""",
    },
    "kernel": {
        "file": "kernel_boundary.py",
        "title": "Two reference observations expose the kernel mechanism",
        "question": "Both reference points sit on the boundary at every gamma. Will the midpoint between them be inside the region at gamma 1?",
        "code": r"""
import numpy as np


def decision(x, gamma):
    rho = (1 + np.exp(-4 * gamma)) / 2
    return (np.exp(-gamma * (x + 1)**2)
            + np.exp(-gamma * (x - 1)**2)) / 2 - rho


for gamma in [0.1, 1.0]:
    print(f"gamma={gamma:.1f}: midpoint={decision(0, gamma):+.6f}, "
          f"reference={decision(1, gamma):+.6f}")
""",
    },
    "temperature": {
        "file": "temperature_monitor.py",
        "title": "The complete offline temperature analysis",
        "question": "The calibration period supplies the threshold and the later period supplies the scores. Which of the two quantiles do you expect to lose a window hit?",
        "code": r"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

base = Path(__file__).resolve().parent
raw = pd.read_csv(base / "machine_temperature_system_failure.csv",
                  parse_dates=["timestamp"])
level = raw.groupby("timestamp", sort=True)["value"].mean()
previous = level.reindex(level.index - pd.Timedelta(hours=1))
features = pd.DataFrame({
    "level": level.to_numpy(),
    "one_hour_change": level.to_numpy() - previous.to_numpy(),
}, index=level.index).dropna()

fit = features.index < "2013-12-06"
cal = (features.index >= "2013-12-06") & (features.index < "2013-12-10")
test = features.index >= "2013-12-10"
scaler = StandardScaler().fit(features.loc[fit])
x_fit, x_cal, x_test = [scaler.transform(features.loc[m]) for m in [fit, cal, test]]

median = features.loc[fit, "level"].median()
mad = (features.loc[fit, "level"] - median).abs().median()
if mad <= 0:
    raise ValueError("This baseline requires a positive reference MAD.")
scores = {"Baseline": (
    abs(features.loc[cal, "level"].to_numpy() - median) / mad,
    abs(features.loc[test, "level"].to_numpy() - median) / mad,
)}
models = {
    "Isolation Forest": IsolationForest(n_estimators=100, max_samples=256,
        contamination="auto", random_state=17, n_jobs=1),
    "One-Class SVM": OneClassSVM(kernel="rbf", gamma=0.5, nu=0.05),
    "LOF novelty": LocalOutlierFactor(n_neighbors=20, novelty=True,
                                     contamination="auto"),
}
for name, model in models.items():
    model.fit(x_fit)
    scores[name] = (-model.score_samples(x_cal), -model.score_samples(x_test))

windows = json.loads((base / "nab-event-windows.json").read_text())[
    "realKnownCause/machine_temperature_system_failure.csv"]
times = features.index[test]
window_masks = [(times >= pd.Timestamp(a)) & (times <= pd.Timestamp(b))
                for a, b in windows]
inside = np.logical_or.reduce(window_masks)
print("fit/cal/test:", int(fit.sum()), int(cal.sum()), int(test.sum()))
for name, (cal_score, test_score) in scores.items():
    for q in [0.95, 0.99]:
        threshold = np.quantile(cal_score, q, method="higher")
        alert = test_score > threshold
        hits = sum(bool((alert & mask).any()) for mask in window_masks)
        print(f"{name:16s} q={q:.2f} alerts={alert.sum():5d} "
              f"inside={np.sum(alert & inside):4d} "
              f"outside={np.sum(alert & ~inside):4d} windows={hits}/4")
""",
    },
}

module_path = Path("src/learn/data/anomaly-detection-examples.js")
evidence_path = Path("docs/teaching/evidence/anomaly-native.json")
write = "--write" in sys.argv


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


records, namespaces = {}, {}
for key, item in PROGRAMS.items():
    code = item["code"].strip() + "\n"
    namespace = {"__file__": str((ASSETS / item["file"]).resolve())}
    with threadpool_limits(limits=1), contextlib.redirect_stdout(io.StringIO()) as output:
        exec(compile(code, f"anomaly-example:{key}", "exec"), namespace)
    namespaces[key] = namespace
    records[key] = {
        "title": item["title"],
        "question": item["question"],
        "file": item["file"],
        "code": code.rstrip("\n"),
        "expected": output.getvalue().strip(),
        "language": "python",
    }
    print(f"Executed {key}: {len(code.splitlines())} lines, {len(records[key]['expected'].splitlines())} output lines")

if not write:
    existing = module_path.read_text(encoding="utf-8")
    prefix = "export const anomalyExamples = "
    recorded = json.loads(existing[existing.index(prefix) + len(prefix):].rstrip().rstrip(";"))
    for key, record in records.items():
        assert recorded[key]["code"] == record["code"], f"{key}: displayed code changed"
        assert recorded[key]["expected"] == record["expected"], (
            f"{key}: output changed:\n{recorded[key]['expected']}\n---\n{record['expected']}")

# Oracles: the manuscript's stated values must come out of these runs.
isolation = namespaces["isolation"]
assert isolation["fitted"][1] == 5, "the fitted subsample is five rows, not 256"
assert float(isolation["x"][int(np.argmax(isolation["scores"]))]) == 12.0
assert np.allclose(isolation["anomaly_scores"](isolation["fit_forest"]([2., 2., 2.]), [2., 2.]), [0.5, 0.5])
assert abs(isolation["correction"](5) - 77 / 30) < 1e-12
lof_arithmetic = namespaces["lofArithmetic"]
assert np.allclose(lof_arithmetic["lrd"], [2 / 3, 1 / 2, 2 / 3, 1 / 6, 1 / 8, 1 / 6])
assert np.allclose(lof_arithmetic["lof"], [7 / 8, 4 / 3, 7 / 8, 7 / 8, 4 / 3, 7 / 8])
assert np.allclose(lof_arithmetic["query_lof"], [35 / 24, 21 / 8, 35 / 32])
modes = namespaces["lofModes"]
assert np.allclose(-modes["model"].negative_outlier_factor_, [7 / 8, 4 / 3, 7 / 8, 7 / 8, 4 / 3, 7 / 8])
assert np.allclose(-modes["model"].score_samples([[17.]]), [35 / 32])
assert np.allclose(-modes["model"].score_samples(modes["x"]), np.full(6, 7 / 8)), "training-array queries collapse"
kernel = namespaces["kernel"]
assert abs(kernel["decision"](0, 0.1) - 0.06967739501813985) < 1e-15
assert abs(kernel["decision"](0, 1.0) + 0.1412783783) < 1e-9
assert abs(kernel["decision"](1, 1.0)) < 1e-15
assert abs(kernel["decision"](0, 0.25) - kernel["decision"](0, 0.25)) == 0
temperature = namespaces["temperature"]
assert (int(temperature["fit"].sum()), int(temperature["cal"].sum()), int(temperature["test"].sum())) == (885, 1152, 20634)
published = {
    "Baseline": {0.95: (4416, 1379, 3037), 0.99: (1461, 1016, 445)},
    "Isolation Forest": {0.95: (10530, 1722, 8808), 0.99: (1548, 971, 577)},
    "One-Class SVM": {0.95: (9719, 1347, 8372), 0.99: (9232, 1319, 7913)},
    "LOF novelty": {0.95: (8771, 1275, 7496), 0.99: (7837, 1138, 6699)},
}
for name, rows in published.items():
    cal_score, test_score = temperature["scores"][name]
    for quantile, (total, hit, miss) in rows.items():
        threshold = np.quantile(cal_score, quantile, method="higher")
        alert = test_score > threshold
        assert (int(alert.sum()), int((alert & temperature["inside"]).sum()),
                int((alert & ~temperature["inside"]).sum())) == (total, hit, miss), (name, quantile)
        assert all(bool((alert & mask).any()) for mask in temperature["window_masks"]), (name, quantile)
oracle_count = 16

if write:
    module_path.write_text(
        "// Complete displayed programs for the anomaly-detection lesson, executed by\n"
        "// scripts/verify-anomaly-detection-examples.py. `file` is the filename the\n"
        "// lesson asks the learner to save the block as.\n"
        "export const anomalyExamples = " + json.dumps(records, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8",
    )

evidence = {
    "verifiedAt": datetime.now(timezone.utc).isoformat(),
    "stage": "native author verification of displayed programs; browser, independent and integration review are separate",
    "source": str(module_path).replace("\\", "/"),
    "sourceHash": hashlib.sha256(module_path.read_bytes()).hexdigest(),
    "verifier": "scripts/verify-anomaly-detection-examples.py",
    "verifierHash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "versions": {name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scikit-learn"]},
    "programs": {key: {"file": record["file"], "codeHash": digest(record["code"]),
                       "stdoutHash": digest(record["expected"]), "stdout": record["expected"]}
                 for key, record in records.items()},
    "oracles": oracle_count,
    "limits": [
        "The temperature program ran against the served copies of the pinned NAB files; no network access was used.",
        "Window hits count published annotation windows, not verified faults or onset times.",
        "Outside-window alerts are unmatched workload, not measured false positives.",
        "The compact forest is a teaching construction; its scores need not match the library's.",
    ],
}
evidence_path.parent.mkdir(parents=True, exist_ok=True)
evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
print(f"PASS: {len(records)} displayed programs executed, {oracle_count} oracle assertions.")
