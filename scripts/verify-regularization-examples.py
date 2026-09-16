"""Execute the regularization lesson's displayed Python programs.

Run with the isolated lesson Python. `--write` regenerates
src/learn/data/regularization-examples.js from the executed programs; without
it, the recorded output must match a fresh execution.

The airfoil program reads the tab-separated `.dat` the lesson serves, from
public/learn-assets/regularization, which is the same bytes a learner receives.
Nothing is downloaded.

Run:
  scratch/lesson-tools/Scripts/python.exe scripts/verify-regularization-examples.py --write
  scratch/lesson-tools/Scripts/python.exe scripts/verify-regularization-examples.py
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

ASSETS = Path("public/learn-assets/regularization").resolve()

PROGRAMS = {
    "coordinateFit": {
        "file": "coordinate_regularization.py",
        "title": "Coordinate descent for the common objective",
        "question": "the four constructed rows have orthogonal columns and data preference (3, 0.4). Will one sweep be enough for all three families, and which coefficient reaches exactly zero?",
        "code": r"""
import numpy as np

def soft_threshold(value, threshold):
    return np.sign(value) * np.maximum(np.abs(value) - threshold, 0.0)

def fit_coordinates(X, y, strength, ratio, tolerance=1e-10, max_sweeps=10000):
    X, y = np.asarray(X, float), np.asarray(y, float)
    if X.ndim != 2 or y.shape != (len(X),) or len(X) == 0:
        raise ValueError("X must be nonempty rows by features, y one target per row")
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise ValueError("Use finite data; handle missing observations before fitting")
    if not np.isfinite([strength, ratio]).all() or strength < 0 or not 0 <= ratio <= 1:
        raise ValueError("Use nonnegative strength and a mixing ratio in [0, 1]")
    n, p = X.shape
    mean_x, mean_y = X.mean(axis=0), y.mean()
    Z, target = X - mean_x, y - mean_y
    curvature = np.mean(Z * Z, axis=0)
    weight = np.zeros(p)
    residual = target.copy()
    for sweep in range(1, max_sweeps + 1):
        for j in range(p):
            partial = residual + Z[:, j] * weight[j]
            association = Z[:, j] @ partial / n
            denominator = curvature[j] + strength * (1 - ratio)
            weight[j] = (soft_threshold(association, strength * ratio)
                         / denominator) if denominator > 0 else 0.0
            residual = partial - Z[:, j] * weight[j]
        gradient = -(Z.T @ residual) / n + strength * (1 - ratio) * weight
        violation = np.where(
            weight != 0,
            np.abs(gradient + strength * ratio * np.sign(weight)),
            np.maximum(np.abs(gradient) - strength * ratio, 0),
        )
        optimality_residual = float(violation.max(initial=0))
        if optimality_residual <= tolerance:
            intercept = float(mean_y - mean_x @ weight)
            return weight, intercept, sweep, optimality_residual
    raise RuntimeError("Coordinate fit did not meet the requested optimality tolerance")

X = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], float)
y = np.array([3.4, 2.6, -2.6, -3.4])
for name, ratio in [("ridge", 0), ("lasso", 1), ("elastic_net", .5)]:
    weight, intercept, sweeps, residual = fit_coordinates(X, y, 1, ratio)
    print(name, np.round(weight, 6), round(intercept, 6), sweeps)
""",
    },
    "airfoilComparison": {
        "file": "airfoil_regularization.py",
        "title": "The complete offline airfoil comparison",
        "question": "three families each search the same six strengths on the same three folds. Will any of them beat unpenalized least squares, and will the selected lasso fit be sparse?",
        "code": r"""
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

def make_model(family, strength, n_fit):
    if family == "ridge":
        fit = Ridge(alpha=n_fit * strength, solver="svd")
    elif family == "lasso":
        fit = Lasso(alpha=strength, max_iter=50000, tol=1e-8)
    elif family == "elastic_net":
        fit = ElasticNet(alpha=strength, l1_ratio=.5, max_iter=50000, tol=1e-8)
    elif family == "ols":
        fit = LinearRegression()
    else:
        raise ValueError(family)
    return make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False), StandardScaler(), fit,
    )

# The supplied .dat is tab separated, 1503 rows by 6 columns, which is the
# whitespace layout np.loadtxt reads by default.
data = np.loadtxt(Path(__file__).with_name("airfoil-self-noise.dat"))
development, reserved = train_test_split(
    np.arange(len(data)), train_size=1200, random_state=41,
)
X, y = data[development, :5], data[development, 5]
splits = list(KFold(n_splits=3, shuffle=True, random_state=202).split(X))
strengths = [.001, .01, .1, 1, 10, 100]
mean_errors, ols_errors = [], []
for train, valid in splits:
    mean_errors.append(np.mean((y[valid] - y[train].mean()) ** 2))
    ols = make_model("ols", 0, len(train)).fit(X[train], y[train])
    ols_errors.append(np.mean((ols.predict(X[valid]) - y[valid]) ** 2))
print("baseline mean MSE", round(float(np.mean(mean_errors)), 6))
print("OLS mean MSE", round(float(np.mean(ols_errors)), 6))

for family in ["ridge", "lasso", "elastic_net"]:
    candidates = []
    for strength in strengths:
        errors, nonzero = [], []
        for train, valid in splits:
            model = make_model(family, strength, len(train)).fit(X[train], y[train])
            errors.append(np.mean((model.predict(X[valid]) - y[valid]) ** 2))
            nonzero.append(int(np.count_nonzero(model[-1].coef_)))
        score = float(np.mean(errors))
        candidates.append((score, strength))
        print(family, strength, round(score, 6), nonzero)
    score, selected_strength = min(candidates)  # lower strength resolves an exact tie
    fitted = make_model(family, selected_strength, len(X)).fit(X, y)
    print("selected", family, selected_strength,
          "development CV MSE", round(score, 6),
          "refit nonzero", int(np.count_nonzero(fitted[-1].coef_)))
""",
    },
    "dropoutMasks": {
        "file": "dropout_masks.py",
        "title": "Enumerate every dropout mask exactly",
        "question": "the mean noisy prediction equals the clean prediction. Will the mean noisy loss equal the clean loss?",
        "code": r"""
from itertools import product

x, weight, target, keep = [2.0, 1.0], [1.0, -1.0], 1.0, 0.5
expected_prediction = expected_loss = 0.0
for mask in product([0, 1], repeat=2):
    probability = 1.0
    for kept in mask:
        probability *= keep if kept else 1 - keep
    prediction = sum(a * w * m / keep for a, w, m in zip(x, weight, mask))
    loss = (target - prediction) ** 2 / 2
    expected_prediction += probability * prediction
    expected_loss += probability * loss
    print(mask, probability, prediction, loss)
print(expected_prediction, expected_loss)
""",
    },
}

module_path = Path("src/learn/data/regularization-examples.js")
evidence_path = Path("docs/teaching/evidence/regularization-native.json")
write = "--write" in sys.argv


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


records, namespaces = {}, {}
for key, item in PROGRAMS.items():
    code = item["code"].strip() + "\n"
    namespace = {"__file__": str(ASSETS / item["file"])}
    with contextlib.chdir(ASSETS), threadpool_limits(limits=1), contextlib.redirect_stdout(io.StringIO()) as output:
        exec(compile(code, f"regularization-example:{key}", "exec"), namespace)
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
    prefix = "export const regularizationExamples = "
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


fit = namespaces["coordinateFit"]
lines = records["coordinateFit"]["expected"].splitlines()
# The manuscript recorded these lines before the standalone program was executed
# and dropped NumPy's column padding on the third. The executed output is what is
# published; the values are identical.
oracle(lines == ["ridge [1.5 0.2] 0.0 1", "lasso [2. 0.] 0.0 1", "elastic_net [1.666667 0.      ] 0.0 1"],
       "the three printed lines, with NumPy's actual column padding")
oracle([" ".join(line.split()) for line in lines]
       == ["ridge [1.5 0.2] 0.0 1", "lasso [2. 0.] 0.0 1", "elastic_net [1.666667 0. ] 0.0 1"],
       "the manuscript's values, ignoring NumPy's alignment spaces")
X, y = fit["X"], fit["y"]
for name, ratio, expected in [("ridge", 0, [1.5, 0.2]), ("lasso", 1, [2.0, 0.0]), ("elastic_net", .5, [5 / 3, 0.0])]:
    weight, intercept, sweeps, residual = fit["fit_coordinates"](X, y, 1, ratio)
    oracle(np.allclose(weight, expected, atol=1e-12), f"{name} slopes")
    oracle(abs(intercept) < 1e-12 and sweeps == 1 and residual <= 1e-10, f"{name} intercept, sweep count and optimality")
oracle(fit["fit_coordinates"](X, y, 1, 1)[0][1] == 0.0, "the lasso second slope is exactly zero, not a small number")
# The two contrasts investigation 2 records.
changed = y.copy()
changed[0] = 7.4
weight, intercept, _, _ = fit["fit_coordinates"](X, changed, 1, 1)
oracle(np.allclose(weight, [3.0, 0.4], atol=1e-12) and abs(intercept - 1) < 1e-12,
       "row 0's target 3.4 to 7.4 gives slopes (3, 0.4) and intercept 1")
weight, intercept, _, _ = fit["fit_coordinates"](X, y + 7, 1, 1)
oracle(np.allclose(weight, [2.0, 0.0], atol=1e-12) and abs(intercept - 7) < 1e-12,
       "adding seven to every target changes only the intercept")
weight, _, _, _ = fit["fit_coordinates"](X, y, 0.5, 1)
oracle(np.allclose(weight, [2.5, 0.0], atol=1e-12), "practice 2's lasso slopes at strength 0.5")
duplicate_X = np.array([[-1., -1.], [1., 1.]])
duplicate_y = np.array([-2., 2.])
weight, _, _, _ = fit["fit_coordinates"](duplicate_X, duplicate_y, 1, 1)
oracle(np.allclose(weight, [1.0, 0.0], atol=1e-12), "the duplicate lasso fit returns the first endpoint under this order")
reversed_columns, _, _, _ = fit["fit_coordinates"](duplicate_X[:, ::-1], duplicate_y, 1, 1)
oracle(np.allclose(reversed_columns, [1.0, 0.0], atol=1e-12), "reversing the coordinate order returns the other endpoint")
weight, _, _, _ = fit["fit_coordinates"](duplicate_X, duplicate_y, 1, 0)
oracle(np.allclose(weight, [2 / 3, 2 / 3], atol=1e-8), "the duplicate ridge fit balances the two coefficients")
weight, _, _, _ = fit["fit_coordinates"](duplicate_X, duplicate_y, 1, .5)
oracle(np.allclose(weight, [0.6, 0.6], atol=1e-8), "the duplicate elastic-net fit is (0.6, 0.6)")
try:
    fit["fit_coordinates"](X, [3.4, 2.6, float("nan"), -3.4], 1, 1)
    raise AssertionError("a non-finite target must be refused")
except ValueError:
    oracle(True, "a non-finite target is refused rather than silently replaced")
try:
    fit["fit_coordinates"](X, y, -1, 1)
    raise AssertionError("a negative strength must be refused")
except ValueError:
    oracle(True, "a negative strength is refused")

airfoil = namespaces["airfoilComparison"]
printed = records["airfoilComparison"]["expected"].splitlines()
oracle(printed[0] == "baseline mean MSE 45.072768", "the stated mean baseline")
oracle(printed[1] == "OLS mean MSE 17.350268", "the stated unpenalized OLS score")
recorded_table = {
    0.001: ["17.315634", "17.335409", "17.325484"],
    0.01: ["17.335789", "17.347152", "17.332697"],
    0.1: ["17.6948", "17.629206", "17.650934"],
    1: ["20.962663", "22.179353", "21.658337"],
    10: ["34.763447", "45.072768", "45.072768"],
    100: ["43.547835", "45.072768", "45.072768"],
}
for column, family in enumerate(["ridge", "lasso", "elastic_net"]):
    for strength, expected in recorded_table.items():
        line = next(row for row in printed if row.startswith(f"{family} {strength} "))
        oracle(line.split()[2] == expected[column], f"{family} at {strength} prints {expected[column]}")
for family in ["ridge", "lasso", "elastic_net"]:
    line = next(row for row in printed if row.startswith(f"selected {family} "))
    oracle(line.split()[2] == "0.001", f"{family} selects the smallest strength in the grid")
    oracle(line.endswith("refit nonzero 20"), f"the selected {family} refit keeps all twenty terms")
oracle(next(row for row in printed if row.startswith("lasso 0.1 ")).endswith("[9, 9, 9]"),
       "lasso at 0.1 leaves nine nonzero coefficients in each fold")
for strength in (10, 100):
    oracle(next(row for row in printed if row.startswith(f"lasso {strength} ")).endswith("[0, 0, 0]"),
           f"lasso at {strength} leaves no nonzero slope")
oracle(next(row for row in printed if row.startswith("lasso 0.001 ")).endswith("[19, 20, 19]"),
       "two folds keep nineteen terms even though the final refit keeps twenty")
oracle(len(airfoil["development"]) == 1200 and len(airfoil["reserved"]) == 303, "1200 development and 303 reserved rows")
oracle(all(len(train) == 800 and len(valid) == 400 for train, valid in airfoil["splits"]), "800 fit and 400 validation rows per fold")
oracle(airfoil["data"].shape == (1503, 6), "the tab-separated file loads as 1503 rows by six columns")
# The reserved rows are split off and never predicted or scored.
oracle(len(set(airfoil["development"].tolist()) & set(airfoil["reserved"].tolist())) == 0,
       "the reserved rows are disjoint from development")
mean_baseline = float(np.mean(airfoil["mean_errors"]))
for family in ["lasso", "elastic_net"]:
    line = next(row for row in printed if row.startswith(f"{family} 10 "))
    oracle(abs(float(line.split()[2]) - round(mean_baseline, 6)) < 1e-9,
           f"{family} at strength 10 reproduces the mean baseline exactly")

dropout = namespaces["dropoutMasks"]
mask_lines = records["dropoutMasks"]["expected"].splitlines()
oracle(mask_lines == [
    "(0, 0) 0.25 0.0 0.5",
    "(0, 1) 0.25 -2.0 4.5",
    "(1, 0) 0.25 4.0 4.5",
    "(1, 1) 0.25 2.0 0.5",
    "1.0 2.5",
], "the mask table and the final line the manuscript records")
oracle(abs(dropout["expected_prediction"] - 1.0) < 1e-12, "the mean prediction is the clean prediction")
oracle(abs(dropout["expected_loss"] - 2.5) < 1e-12, "the expected noisy half-loss is 2.5")
oracle(abs(dropout["expected_loss"] - (1 - 0.5) / (2 * 0.5) * (4 + 1)) < 1e-12,
       "the analytic penalty formula matches the enumeration exactly")
oracle(oracle_count > 0, "oracles ran")

if write:
    module_path.write_text(
        "// Complete displayed programs for the regularization lesson, executed by\n"
        "// scripts/verify-regularization-examples.py. `file` is the filename the lesson\n"
        "// asks the learner to save the block as.\n"
        "export const regularizationExamples = " + json.dumps(records, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8", newline="\n",
    )

evidence = {
    "verifiedAt": datetime.now(timezone.utc).isoformat(),
    "stage": "native author verification of displayed programs; browser, independent and integration review are separate",
    "source": str(module_path).replace("\\", "/"),
    "sourceHash": hashlib.sha256(module_path.read_bytes()).hexdigest(),
    "verifier": "scripts/verify-regularization-examples.py",
    "verifierHash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "versions": {name: importlib.metadata.version(name) for name in ["numpy", "scikit-learn"]},
    "programs": {key: {"file": record["file"], "codeHash": digest(record["code"]),
                       "stdoutHash": digest(record["expected"]), "stdout": record["expected"]}
                 for key, record in records.items()},
    "oracles": oracle_count,
    "limits": [
        "The airfoil program ran against the served copy of the supplied .dat; no network access was used.",
        "One row-level split of one collection; the scores are development comparison and selection records.",
        "No reserved row was predicted or scored by any program here.",
        "Numerical fitting can differ on other library versions.",
    ],
}
evidence_path.parent.mkdir(parents=True, exist_ok=True)
evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8", newline="\n")
print(f"PASS: {len(records)} displayed programs executed, {oracle_count} oracle assertions.")
