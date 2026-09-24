"""Execute the bias-variance lesson's displayed Python programs.

Run with the isolated lesson Python. `--write` regenerates
src/learn/data/bias-variance-examples.js from the executed programs; without it,
the recorded output must match a fresh execution.

The two airfoil programs read the tab-separated `.dat` this lesson serves, from
public/learn-assets/bias-variance, which is the same bytes a learner receives.
The second is the manuscript's "append the following to the same program", so it
runs in the first one's namespace rather than as a separate script. Nothing is
downloaded.

Run:
  scratch/lesson-tools/Scripts/python.exe scripts/verify-bias-variance-examples.py --write
  scratch/lesson-tools/Scripts/python.exe scripts/verify-bias-variance-examples.py
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
from fractions import Fraction
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
ASSETS = (ROOT / "public/learn-assets/bias-variance").resolve()

PROGRAMS = {
    "finiteWorlds": {
        "file": "bias_variance_worlds.py",
        "title": "Every possible tiny training set, enumerated",
        "question": "eight equally likely training sets, three fitted degrees and two probe inputs. Which fit has zero squared bias at the probe, and does the lowest expected error belong to it?",
        "code": r"""
from itertools import product
import numpy as np

train_x = np.array([-1., 0., 1.])
probe = np.array([0., .5])
curvature, sigma = 1., .5
signs = np.array(list(product([-1., 1.], repeat=len(train_x))))
targets = 1 + train_x + curvature * train_x**2 + sigma * signs
truth = 1 + probe + curvature * probe**2

for degree in (0, 1, 2):
    design = np.vander(train_x, degree + 1, increasing=True)
    coefficients = np.linalg.lstsq(design, targets.T, rcond=None)[0]
    predicted = (
        np.vander(probe, degree + 1, increasing=True) @ coefficients
    ).T
    mean = predicted.mean(axis=0)
    bias_squared = (mean - truth)**2
    variance = predicted.var(axis=0)
    expected_error = ((predicted - truth)**2).mean(axis=0) + sigma**2
    print(degree, np.round(mean, 6), np.round(bias_squared, 6),
          np.round(variance, 6), np.round(expected_error, 6))
""",
    },
    "airfoilLearningCurve": {
        "file": "airfoil_learning_curve.py",
        "title": "Learning curves for four prespecified procedures",
        "question": "four procedures, five fitted sizes and five folds. Which of them improves most as training size grows, and does requiring twenty items per leaf help at every size?",
        "code": r"""
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split, learning_curve

data = np.loadtxt("airfoil-self-noise.dat")
x, y = data[:, :5], data[:, 5]
development, untouched = train_test_split(
    np.arange(len(y)), train_size=1200, random_state=41
)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
models = {
    "mean": DummyRegressor(strategy="mean"),
    "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.)),
    "tree_leaf1": DecisionTreeRegressor(
        min_samples_leaf=1, random_state=43),
    "tree_leaf20": DecisionTreeRegressor(
        min_samples_leaf=20, random_state=43),
}
for name, model in models.items():
    sizes, train_scores, valid_scores = learning_curve(
        model, x[development], y[development],
        train_sizes=[60, 120, 240, 480, 900], cv=cv,
        scoring="neg_mean_squared_error",
        shuffle=True, random_state=44, n_jobs=1,
    )
    print(name)
    for n, tr, va in zip(sizes, train_scores, valid_scores):
        print(n, round(-tr.mean(), 4), round(-va.mean(), 4))
""",
    },
    "settingAndTrajectory": {
        "dependsOn": "airfoilLearningCurve",
        "title": "One changed setting, then one training trajectory",
        "question": "the same development pool, a restriction grid and a boosting run. Does restricting the leaf size improve the score here, and where does the monitoring error reach its lowest inspected value?",
        "code": r"""
from sklearn.model_selection import validation_curve
from sklearn.ensemble import GradientBoostingRegressor

leaves = [1, 2, 5, 10, 20, 40]
tr, va = validation_curve(
    DecisionTreeRegressor(random_state=43), x[development], y[development],
    param_name="min_samples_leaf", param_range=leaves, cv=cv,
    scoring="neg_mean_squared_error", n_jobs=1,
)
for size, train_score, valid_score in zip(leaves, tr, va):
    print("leaf", size, round(-train_score.mean(), 4),
          round(-valid_score.mean(), 4))

fit_ids, monitor_ids = train_test_split(
    development, test_size=240, random_state=45
)
boosted = GradientBoostingRegressor(
    n_estimators=120, max_depth=2, learning_rate=.1, random_state=46
).fit(x[fit_ids], y[fit_ids])
train_mse = [np.mean((y[fit_ids] - p)**2)
             for p in boosted.staged_predict(x[fit_ids])]
monitor_mse = [np.mean((y[monitor_ids] - p)**2)
               for p in boosted.staged_predict(x[monitor_ids])]
for i in [0, 9, 29, 59, 119]:
    print("round", i + 1, round(train_mse[i], 4),
          round(monitor_mse[i], 4))
print("best inspected round", int(np.argmin(monitor_mse)) + 1)
""",
    },
}

module_path = ROOT / "src/learn/data/bias-variance-examples.js"
evidence_path = ROOT / "docs/teaching/evidence/bias-variance-native.json"
write = "--write" in sys.argv


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


records, namespaces = {}, {}
for key, item in PROGRAMS.items():
    code = item["code"].strip() + "\n"
    parent = item.get("dependsOn")
    namespace = namespaces[parent] if parent else {"__file__": str(ASSETS / item.get("file", "appended.py"))}
    with contextlib.chdir(ASSETS), threadpool_limits(limits=1), contextlib.redirect_stdout(io.StringIO()) as output:
        exec(compile(code, f"bias-variance-example:{key}", "exec"), namespace)
    namespaces[key] = namespace
    record = {
        "title": item["title"],
        "question": item["question"],
        "code": code.rstrip("\n"),
        "expected": output.getvalue().strip(),
        "language": "python",
    }
    if item.get("file"):
        record["file"] = item["file"]
    if parent:
        record["appendedTo"] = PROGRAMS[parent]["file"]
    records[key] = record
    print(f"Executed {key}: {len(code.splitlines())} lines, {len(record['expected'].splitlines())} output lines")

if not write:
    existing = module_path.read_text(encoding="utf-8")
    prefix = "export const biasVarianceExamples = "
    recorded = json.loads(existing[existing.index(prefix) + len(prefix):].rstrip().rstrip(";"))
    for key, record in records.items():
        assert recorded[key]["code"] == record["code"], f"{key}: displayed code changed"
        assert recorded[key]["expected"] == record["expected"], (
            f"{key}: output changed:\n{recorded[key]['expected']}\n---\n{record['expected']}")

# Oracles: the manuscript's stated values must come out of these runs.
oracle_count = 0


def oracle(condition, label):
    global oracle_count
    assert condition, label
    oracle_count += 1


def parse_row(line):
    """`0 [1.666667 1.666667] [0.444444 0.006944] ...` into degree plus four pairs."""
    degree = int(line.split(" ", 1)[0])
    values = [float(token) for token in line.replace("[", " ").replace("]", " ").split()[1:]]
    assert len(values) == 8, line
    pairs = [(values[index], values[index + 1]) for index in range(0, 8, 2)]
    return degree, pairs


worlds = namespaces["finiteWorlds"]
printed = records["finiteWorlds"]["expected"].splitlines()
oracle(len(printed) == 3, "one line per fitted degree")
# The manuscript's section 3 table at the probe x = 0.5, as exact fractions.
stated = {
    0: (Fraction(5, 3), Fraction(1, 144), Fraction(1, 12), Fraction(49, 144)),
    1: (Fraction(13, 6), Fraction(25, 144), Fraction(11, 96), Fraction(155, 288)),
    2: (Fraction(7, 4), Fraction(0), Fraction(23, 128), Fraction(55, 128)),
}
rows = dict(parse_row(line) for line in printed)
for degree, expected in stated.items():
    mean_at, bias_at, variance_at, error_at = (pair[1] for pair in rows[degree])
    for actual, target, name in zip(
        (mean_at, bias_at, variance_at, error_at), expected,
        ("mean prediction", "squared bias", "prediction variance", "expected error"),
    ):
        oracle(abs(actual - float(target)) < 1e-6, f"degree {degree} {name} at probe 0.5 is {target}")
# At the zero probe the constant and the line coincide, and the quadratic costs .5.
zero_constant = [pair[0] for pair in rows[0]]
zero_line = [pair[0] for pair in rows[1]]
oracle(zero_constant == zero_line, "constant and line agree at the zero probe, term by term")
oracle(abs(zero_constant[1] - 4 / 9) < 1e-6, "the constant's squared bias at zero is 4/9")
oracle(abs([pair[0] for pair in rows[2]][3] - 0.5) < 1e-6, "the quadratic's expected error at zero is 0.5")
oracle(abs([pair[1] for pair in rows[2]][1]) < 5e-7, "the quadratic is unbiased at the 0.5 probe")
oracle(worlds["targets"].shape == (8, 3), "eight equally likely training sets of three outcomes")
oracle(sorted(set(np.round(worlds["targets"][:, 0], 9))) == [0.5, 1.5],
       "the first training outcome is the truth plus or minus sigma")
oracle(np.allclose(np.sort(worlds["targets"][0]), np.sort([0.5, 0.5, 2.5])),
       "the first enumerated world is the all-minus one")
oracle(any(np.allclose(np.sort(row), np.sort([0.5, 1.5, 2.5])) for row in worlds["targets"]),
       "the manuscript's example noisy set [.5, 1.5, 2.5] is one of the eight")
# The quadratic's interpolation weights, recomputed here rather than read back.
design = np.vander(worlds["train_x"], 3, increasing=True)
probe_row = np.vander(np.array([0.5]), 3, increasing=True)[0]
weights = probe_row @ np.linalg.inv(design.T @ design) @ design.T
oracle(np.allclose(weights, [-0.125, 0.75, 0.375], atol=1e-12), "the stated interpolation weights")
oracle(abs(weights.sum() - 1) < 1e-12, "the interpolation weights sum to one")
oracle(abs(0.25 * float(weights @ weights) - 23 / 128) < 1e-12, "sigma squared times the squared weights is 23/128")
oracle((weights < 0).sum() == 1, "exactly one weight is negative")

curve = namespaces["airfoilLearningCurve"]
lines = records["airfoilLearningCurve"]["expected"].splitlines()
stated_validation = {
    "mean": [45.4545, 45.081, 45.2469, 45.2065, 45.222],
    "ridge": [24.9831, 23.8044, 23.6478, 23.4994, 23.372],
    "tree_leaf1": [41.8405, 24.4041, 20.1754, 13.7464, 8.1412],
    "tree_leaf20": [39.9286, 32.3389, 28.1945, 21.6109, 16.6298],
}
blocks = {}
current = None
for line in lines:
    if line in stated_validation:
        current = line
        blocks[current] = []
    else:
        blocks[current].append([float(token) for token in line.split()])
for name, expected in stated_validation.items():
    oracle([row[0] for row in blocks[name]] == [60.0, 120.0, 240.0, 480.0, 900.0],
           f"{name} prints the five requested fitted sizes")
    oracle([row[2] for row in blocks[name]] == expected, f"{name} prints the recorded validation table")
oracle([row[1] for row in blocks["tree_leaf1"]] == [0.0] * 5,
       "the leaf-1 tree's training MSE is zero at every inspected size")
oracle(blocks["ridge"][-1][1] == 22.7528 and blocks["ridge"][-1][2] == 23.372,
       "Ridge prints 22.7528 and 23.372 at 900 fitted rows")
oracle(blocks["tree_leaf20"][-1][1] == 12.5438 and blocks["tree_leaf20"][-1][2] == 16.6298,
       "the leaf-20 tree prints 12.5438 and 16.6298 at 900 fitted rows")
crossing = [row[0] for row, ridge in zip(blocks["tree_leaf1"], blocks["ridge"]) if row[2] < ridge[2]]
oracle(crossing and crossing[0] == 240.0, "the unrestricted tree first passes Ridge at 240 fitted rows")
helps = [row[0] for row, free in zip(blocks["tree_leaf20"], blocks["tree_leaf1"]) if row[2] < free[2]]
oracle(helps == [60.0], "the leaf-20 restriction helps only at the smallest inspected size")
oracle(len(curve["development"]) == 1200 and len(curve["untouched"]) == 303,
       "1,200 development rows and 303 reserved rows")
oracle(not set(curve["development"].tolist()) & set(curve["untouched"].tolist()),
       "the reserved rows are disjoint from development")
oracle(curve["data"].shape == (1503, 6), "the tab-separated file loads as 1503 rows by six columns")
folds = list(curve["cv"].split(curve["x"][curve["development"]]))
oracle(all(len(train) == 960 and len(valid) == 240 for train, valid in folds),
       "each of the five folds holds 960 available training rows and 240 validation rows")
oracle(len(curve["train_scores"]) == 5 and len(curve["train_scores"][0]) == 5,
       "the returned arrays have shape five sizes by five folds")

trajectory = namespaces["settingAndTrajectory"]
staged = records["settingAndTrajectory"]["expected"].splitlines()
leaf_lines = [line for line in staged if line.startswith("leaf ")]
oracle([float(line.split()[3]) for line in leaf_lines]
       == [8.2079, 8.5475, 10.1516, 12.3799, 15.7762, 21.8071],
       "the six printed validation-curve scores")
oracle([int(line.split()[1]) for line in leaf_lines] == [1, 2, 5, 10, 20, 40], "the six inspected leaf sizes")
oracle(min(float(line.split()[3]) for line in leaf_lines) == 8.2079,
       "this inspected restriction never improves the score")
oracle(float(leaf_lines[0].split()[3]) != 8.1412,
       "the 960-row leaf-1 score differs from the 900-row learning-curve value")
round_lines = [line for line in staged if line.startswith("round ")]
oracle([[int(line.split()[1]), float(line.split()[2]), float(line.split()[3])] for line in round_lines] == [
    [1, 41.5026, 42.9204], [10, 26.9413, 28.0979], [30, 17.5512, 20.0168],
    [60, 13.0405, 15.3584], [120, 9.1042, 12.3338],
], "the five printed boosting rounds")
oracle(staged[-1] == "best inspected round 120", "the best inspected round is the last one")
oracle(len(trajectory["monitor_mse"]) == 120 and int(np.argmin(trajectory["monitor_mse"])) == 119,
       "the argmin is taken over all 120 recorded rounds, not the five printed ones")
oracle(len(trajectory["fit_ids"]) == 960 and len(trajectory["monitor_ids"]) == 240,
       "the trajectory's own split is 960 fitting and 240 monitoring rows")
oracle(not set(trajectory["fit_ids"].tolist()) & set(trajectory["monitor_ids"].tolist()),
       "its two parts are disjoint")
oracle(set(trajectory["fit_ids"].tolist()) | set(trajectory["monitor_ids"].tolist())
       == set(curve["development"].tolist()),
       "and together they are exactly the development pool, so no reserved row is used")
oracle(any(later > earlier for earlier, later in zip(trajectory["monitor_mse"], trajectory["monitor_mse"][1:])),
       "the monitoring trace does rise at some rounds, so the drawn curve is not smoothed")
oracle(oracle_count > 0, "oracles ran")

if write:
    module_path.write_text(
        "// Complete displayed programs for the bias-variance and learning-curve\n"
        "// lesson, executed by scripts/verify-bias-variance-examples.py. `file` is the\n"
        "// filename the lesson asks the learner to save the block as; `appendedTo`\n"
        "// marks a block that runs inside an earlier program's namespace.\n"
        "export const biasVarianceExamples = " + json.dumps(records, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8", newline="\n",
    )

evidence = {
    "verifiedAt": datetime.now(timezone.utc).isoformat(),
    "stage": "native author verification of displayed programs; browser, independent and integration review are separate",
    "source": str(module_path.relative_to(ROOT)).replace("\\", "/"),
    "sourceHash": hashlib.sha256(module_path.read_bytes()).hexdigest(),
    "verifier": "scripts/verify-bias-variance-examples.py",
    "verifierHash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "versions": {name: importlib.metadata.version(name) for name in ["numpy", "scikit-learn"]},
    "programs": {key: {"file": record.get("file", record.get("appendedTo")),
                       "appended": "appendedTo" in record,
                       "codeHash": digest(record["code"]),
                       "stdoutHash": digest(record["expected"]),
                       "stdout": record["expected"]}
                 for key, record in records.items()},
    "oracles": oracle_count,
    "limits": [
        "The airfoil programs ran against this lesson's served copy of the supplied .dat; no network access was used.",
        "One row-level split of one collection; the scores are model-development results.",
        "No reserved row was predicted or scored by any program here.",
        "Numerical fitting can differ on other library versions.",
    ],
}
evidence_path.parent.mkdir(parents=True, exist_ok=True)
evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8", newline="\n")
print(f"PASS: {len(records)} displayed programs executed, {oracle_count} oracle assertions.")
