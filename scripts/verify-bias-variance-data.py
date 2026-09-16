"""Regenerate and verify the airfoil data module for the bias-variance lesson.

The declared campaign is recomputed from the dataset this lesson serves, twice:
once from an independent implementation of the protocol (KFold, one shared
permutation stream, prefixes of each fold's training rows) and once through
scikit-learn's own `learning_curve` and `validation_curve`. The two must agree
before either is compared with the content packet's recorded
`calculated-inputs.json`.

This script is READ-ONLY unless given `--write`. Without it, the module text is
rebuilt in memory and must be byte-identical to the file already on disk, and
the served asset must already match the packet bytes.

Run:
  scratch/lesson-tools/Scripts/python.exe scripts/verify-bias-variance-data.py --write
  scratch/lesson-tools/Scripts/python.exe scripts/verify-bias-variance-data.py
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import hashlib
import json
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
    KFold, learning_curve, train_test_split, validation_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_random_state

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "docs/teaching/drafts/bias-variance-tradeoff-learning-curves"
PACKET_DATASET = PACKET / "airfoil-self-noise.dat"
ASSET_DIR = ROOT / "public/learn-assets/bias-variance"
ASSET_DATASET = ASSET_DIR / "airfoil-self-noise.dat"
MODULE = ROOT / "src/learn/data/bias-variance-data.js"
EVIDENCE = ROOT / "docs/teaching/evidence/bias-variance-data.json"

EXPECTED_SHA = "74c75fd71783f1e6b71f8a622b993dc592897a97cd689c5090a07147a1b097b3"
EXPECTED_BYTES = 59984
RAW_NAMES = ["frequency_hz", "attack_degrees", "chord_m", "speed_mps", "displacement_m"]
RAW_LABELS = ["frequency", "angle of attack", "chord length", "free-stream velocity",
              "suction-side displacement thickness"]
RAW_UNITS = ["Hz", "degrees", "m", "m/s", "m"]
TRAIN_SIZES = [60, 120, 240, 480, 900]
LEAF_GRID = [1, 2, 5, 10, 20, 40]
PRINTED_ROUNDS = [1, 10, 30, 60, 120]
SPLIT_SEED = 41
FOLD_SEED = 42
SUBSET_SEED = 44
TREE_SEED = 43
STAGE_SPLIT_SEED = 45
STAGE_SEED = 46
LINE = "," + chr(10)

PROCEDURES = [
    ("mean", "training-mean baseline",
     "Predicts the training mean of every fitted subset; no input is used.",
     lambda: DummyRegressor(strategy="mean")),
    ("ridge", "standardized Ridge, penalty 1",
     "Standardization is fitted inside each training subset, then a linear Ridge fit.",
     lambda: make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
    ("tree_leaf1", "regression tree, 1 item per leaf",
     "An unrestricted regression tree: leaves may hold a single training row.",
     lambda: DecisionTreeRegressor(min_samples_leaf=1, random_state=TREE_SEED)),
    ("tree_leaf20", "regression tree, 20 items per leaf",
     "The same tree with one capacity restriction: at least twenty rows per leaf.",
     lambda: DecisionTreeRegressor(min_samples_leaf=20, random_state=TREE_SEED)),
]


def close(actual, expected, label, tolerance=1e-9):
    difference = abs(float(actual) - float(expected))
    assert difference <= tolerance, f"{label}: {actual} versus {expected} (differ by {difference})"


def independent_learning_curve(factory, X, y, folds, sizes, seed):
    """The protocol, implemented here rather than delegated.

    One random stream permutes each fold's training rows in fold order; a
    requested size takes that many rows from the front of the permutation. Both
    scores are measured on the rows actually used: the fitted subset and the
    fold's own held-out rows, which never change with size.
    """
    rng = check_random_state(seed)
    permuted = [(rng.permutation(train), valid) for train, valid in folds]
    train_rows, valid_rows = [], []
    for size in sizes:
        per_size_train, per_size_valid = [], []
        for train, valid in permuted:
            subset = train[:size]
            model = factory().fit(X[subset], y[subset])
            per_size_train.append(float(np.mean((model.predict(X[subset]) - y[subset]) ** 2)))
            per_size_valid.append(float(np.mean((model.predict(X[valid]) - y[valid]) ** 2)))
        train_rows.append(per_size_train)
        valid_rows.append(per_size_valid)
    return train_rows, valid_rows


def independent_validation_curve(X, y, folds, settings, seed):
    train_rows, valid_rows = [], []
    for setting in settings:
        per_setting_train, per_setting_valid = [], []
        for train, valid in folds:
            model = DecisionTreeRegressor(min_samples_leaf=setting, random_state=seed)
            model.fit(X[train], y[train])
            per_setting_train.append(float(np.mean((model.predict(X[train]) - y[train]) ** 2)))
            per_setting_valid.append(float(np.mean((model.predict(X[valid]) - y[valid]) ** 2)))
        train_rows.append(per_setting_train)
        valid_rows.append(per_setting_valid)
    return train_rows, valid_rows


def numbers(values, digits=9):
    return "[" + ", ".join(repr(round(float(value), digits)) for value in values) + "]"


def fold_block(rows, indent, digits=9):
    pad = " " * indent
    return ("[" + chr(10)
            + LINE.join(pad + "  " + numbers(row, digits) for row in rows)
            + chr(10) + pad + "]")


def main() -> None:
    write = "--write" in sys.argv

    packet_bytes = PACKET_DATASET.read_bytes()
    digest = hashlib.sha256(packet_bytes).hexdigest()
    assert len(packet_bytes) == EXPECTED_BYTES, f"packet dataset is {len(packet_bytes)} bytes, not {EXPECTED_BYTES}"
    assert digest == EXPECTED_SHA, "the packet dataset is not the checkpointed file"
    assert b"\t" in packet_bytes and b"\r\n" in packet_bytes, "the packet file is tab separated with CRLF endings"

    if ASSET_DATASET.exists():
        served = ASSET_DATASET.read_bytes()
        assert served == packet_bytes, "the served copy differs from the checkpointed packet bytes"
    else:
        assert write, "the served dataset is missing; rerun with --write to publish this lesson's own copy"
        served = packet_bytes

    packet_path = PACKET / "calculated-inputs.json"
    packet_bytes_raw = packet_path.read_bytes()
    packet_sha = hashlib.sha256(packet_bytes_raw).hexdigest()
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    recorded = packet["real"]

    # The trust root, re-derived here rather than assumed.
    #
    # scripts/verify-bias-variance-models.mjs compares the browser model layer
    # against this file's `finiteExperiments` and `doubleDescentApproximation`
    # blocks — 19,764 grid values — but until now nothing in the repository
    # re-derived them, so a regenerated-from-buggy-code or hand-edited block
    # would have produced agreement and a green run. They are recomputed below
    # from the recorded settings alone, through normal equations rather than the
    # author script's `lstsq`, and through the manuscript's piecewise expression.
    finite_checks = 0
    for record in packet["finiteExperiments"]:
        train_x = np.asarray(record["trainX"], dtype=float)
        grid = np.asarray(record["grid"], dtype=float)
        degree, curvature, sigma = record["degree"], record["curvature"], record["sigma"]
        signs = np.array([
            [1.0 if (mask >> (len(train_x) - 1 - position)) & 1 else -1.0 for position in range(len(train_x))]
            for mask in range(2 ** len(train_x))
        ])
        targets = 1 + train_x + curvature * train_x ** 2 + sigma * signs
        design = np.vander(train_x, degree + 1, increasing=True)
        gram = design.T @ design
        coefficients = np.linalg.solve(gram, design.T @ targets.T)
        predictions = (np.vander(grid, degree + 1, increasing=True) @ coefficients).T
        truth = 1 + grid + curvature * grid ** 2
        mean = predictions.mean(axis=0)
        variance = predictions.var(axis=0)
        bias_squared = (mean - truth) ** 2
        expected = ((predictions - truth) ** 2).mean(axis=0) + sigma ** 2
        label = f"finite experiment degree={degree} c={curvature} sigma={sigma} n={len(train_x)}"
        np.testing.assert_allclose(targets, record["targets"], rtol=0, atol=1e-12, err_msg=f"{label}: targets")
        np.testing.assert_allclose(predictions, record["predictions"], rtol=0, atol=1e-9, err_msg=f"{label}: predictions")
        np.testing.assert_allclose(truth, record["truth"], rtol=0, atol=1e-12, err_msg=f"{label}: truth")
        np.testing.assert_allclose(mean, record["mean"], rtol=0, atol=1e-9, err_msg=f"{label}: mean")
        np.testing.assert_allclose(bias_squared, record["biasSquared"], rtol=0, atol=1e-9, err_msg=f"{label}: squared bias")
        np.testing.assert_allclose(variance, record["variance"], rtol=0, atol=1e-9, err_msg=f"{label}: variance")
        np.testing.assert_allclose(expected, record["expectedError"], rtol=0, atol=1e-9, err_msg=f"{label}: expected error")
        # The identity the whole section rests on, checked pointwise.
        np.testing.assert_allclose(expected - sigma ** 2, bias_squared + variance, rtol=0, atol=1e-9,
                                   err_msg=f"{label}: bias squared plus variance is the excess over noise")
        assert record["identityMaxError"] < 1e-12, f"{label}: the packet recorded a large identity residual"
        finite_checks += targets.size + predictions.size + 5 * grid.size
    assert len(packet["finiteExperiments"]) == 21, "the packet should hold twenty-one finite experiments"

    descent_checks = 0
    for record in packet["doubleDescentApproximation"]:
        ratio = record["ratio"]
        assert ratio != 1, "the packet must not record a value at the singular boundary"
        bias = (1 - ratio) ** 2 if ratio < 1 else 0.0
        variance = (ratio * (1 - ratio) + 0.04 * ratio / (1 - ratio)) if ratio < 1 else 0.04 / (ratio - 1)
        close(bias, record["biasSquared"], f"double descent squared bias at {ratio}", 1e-12)
        close(variance, record["varianceApprox"], f"double descent variance at {ratio}", 1e-12)
        close(bias + variance + 0.04, record["riskApprox"], f"double descent risk at {ratio}", 1e-12)
        descent_checks += 3
    trust_root_checks = finite_checks + descent_checks

    data = np.loadtxt(PACKET_DATASET)
    assert data.shape == (1503, 6), f"expected 1503 rows and six columns, found {data.shape}"
    x, y = data[:, :5], data[:, 5]

    development, untouched = train_test_split(np.arange(len(y)), train_size=1200, random_state=SPLIT_SEED)
    assert development.tolist() == recorded["developmentIds"], "development row IDs differ from the packet"
    assert untouched.tolist() == recorded["untouchedIds"], "reserved row IDs differ from the packet"
    assert len(development) == 1200 and len(untouched) == 303
    assert not set(development.tolist()) & set(untouched.tolist()), "the reserved rows must stay disjoint"

    X, Y = x[development], y[development]
    folds = list(KFold(n_splits=5, shuffle=True, random_state=FOLD_SEED).split(X))
    assert all(len(train) == 960 and len(valid) == 240 for train, valid in folds), "each fold holds 960/240 rows"

    caught: list[str] = []
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")

        procedures = []
        for name, label, description, factory in PROCEDURES:
            mine_train, mine_valid = independent_learning_curve(factory, X, Y, folds, TRAIN_SIZES, SUBSET_SEED)
            sizes, library_train, library_valid = learning_curve(
                factory(), X, Y, train_sizes=TRAIN_SIZES, cv=KFold(n_splits=5, shuffle=True, random_state=FOLD_SEED),
                scoring="neg_mean_squared_error", shuffle=True, random_state=SUBSET_SEED, n_jobs=1,
            )
            assert sizes.tolist() == TRAIN_SIZES, "the library returned different fitted sizes"
            np.testing.assert_allclose(mine_train, -library_train, rtol=0, atol=1e-9,
                                       err_msg=f"{name}: independent training scores differ from the library")
            np.testing.assert_allclose(mine_valid, -library_valid, rtol=0, atol=1e-9,
                                       err_msg=f"{name}: independent validation scores differ from the library")
            saved = recorded["learning"][name]
            assert saved["sizes"] == TRAIN_SIZES
            np.testing.assert_allclose(mine_train, saved["trainMse"], rtol=0, atol=1e-9,
                                       err_msg=f"{name}: training scores differ from the packet")
            np.testing.assert_allclose(mine_valid, saved["validationMse"], rtol=0, atol=1e-9,
                                       err_msg=f"{name}: validation scores differ from the packet")
            procedures.append({
                "model": name, "label": label, "description": description,
                "trainMse": mine_train, "validationMse": mine_valid,
            })

        mine_leaf_train, mine_leaf_valid = independent_validation_curve(X, Y, folds, LEAF_GRID, TREE_SEED)
        library_leaf_train, library_leaf_valid = validation_curve(
            DecisionTreeRegressor(random_state=TREE_SEED), X, Y,
            param_name="min_samples_leaf", param_range=LEAF_GRID,
            cv=KFold(n_splits=5, shuffle=True, random_state=FOLD_SEED),
            scoring="neg_mean_squared_error", n_jobs=1,
        )
        np.testing.assert_allclose(mine_leaf_train, -library_leaf_train, rtol=0, atol=1e-9)
        np.testing.assert_allclose(mine_leaf_valid, -library_leaf_valid, rtol=0, atol=1e-9)
        np.testing.assert_allclose(mine_leaf_train, recorded["validation"]["trainMse"], rtol=0, atol=1e-9)
        np.testing.assert_allclose(mine_leaf_valid, recorded["validation"]["validationMse"], rtol=0, atol=1e-9)
        assert recorded["validation"]["minSamplesLeaf"] == LEAF_GRID

        fit_ids, monitor_ids = train_test_split(development, test_size=240, random_state=STAGE_SPLIT_SEED)
        assert fit_ids.tolist() == recorded["stage"]["trainIds"], "the boosting fit rows differ from the packet"
        assert monitor_ids.tolist() == recorded["stage"]["validationIds"], "the boosting monitor rows differ"
        assert len(fit_ids) == 960 and len(monitor_ids) == 240
        boosted = GradientBoostingRegressor(
            n_estimators=120, max_depth=2, learning_rate=0.1, random_state=STAGE_SEED,
        ).fit(x[fit_ids], y[fit_ids])
        stage_train = [float(np.mean((y[fit_ids] - p) ** 2)) for p in boosted.staged_predict(x[fit_ids])]
        stage_monitor = [float(np.mean((y[monitor_ids] - p) ** 2)) for p in boosted.staged_predict(x[monitor_ids])]
        np.testing.assert_allclose(stage_train, recorded["stage"]["trainMse"], rtol=0, atol=1e-9)
        np.testing.assert_allclose(stage_monitor, recorded["stage"]["validationMse"], rtol=0, atol=1e-9)
        caught = [str(item.message) for item in observed]

    assert caught == [], f"the campaign raised warnings: {caught}"

    # The manuscript's printed values, recomputed rather than copied across.
    stated_validation = {
        60: [45.4545, 24.9831, 41.8405, 39.9286],
        120: [45.0810, 23.8044, 24.4041, 32.3389],
        240: [45.2469, 23.6478, 20.1754, 28.1945],
        480: [45.2065, 23.4994, 13.7464, 21.6109],
        900: [45.2220, 23.3720, 8.1412, 16.6298],
    }
    for position, size in enumerate(TRAIN_SIZES):
        for column, procedure in enumerate(procedures):
            close(round(float(np.mean(procedure["validationMse"][position])), 4),
                  stated_validation[size][column],
                  f"{procedure['model']} at {size} fitted rows", 1e-9)
    leaf1 = next(row for row in procedures if row["model"] == "tree_leaf1")
    assert all(value == 0.0 for row in leaf1["trainMse"] for value in row), \
        "the leaf-1 tree's training MSE must be exactly zero at every inspected size"
    ridge_row = next(row for row in procedures if row["model"] == "ridge")
    leaf20 = next(row for row in procedures if row["model"] == "tree_leaf20")
    close(float(np.mean(ridge_row["trainMse"][-1])), 22.7528, "Ridge training MSE at 900 rows", 5e-5)
    close(float(np.mean(leaf20["trainMse"][-1])), 12.5438, "leaf-20 training MSE at 900 rows", 5e-5)
    # The stated crossover: the unrestricted tree first beats Ridge at 240 rows.
    tree_means = [float(np.mean(row)) for row in leaf1["validationMse"]]
    ridge_means = [float(np.mean(row)) for row in ridge_row["validationMse"]]
    leaf20_means = [float(np.mean(row)) for row in leaf20["validationMse"]]
    crossings = [size for size, tree, ridge in zip(TRAIN_SIZES, tree_means, ridge_means) if tree < ridge]
    assert crossings and crossings[0] == 240, f"expected the tree to pass Ridge at 240 rows, found {crossings}"
    helps = [size for size, restricted, free in zip(TRAIN_SIZES, leaf20_means, tree_means) if restricted < free]
    assert helps == [60], f"the leaf-20 restriction should help only at 60 rows, found {helps}"
    stated_leaf = [8.2079, 8.5475, 10.1516, 12.3799, 15.7762, 21.8071]
    for setting, row, expected in zip(LEAF_GRID, mine_leaf_valid, stated_leaf):
        close(round(float(np.mean(row)), 4), expected, f"validation curve at leaf {setting}", 1e-9)
    assert min(range(6), key=lambda index: float(np.mean(mine_leaf_valid[index]))) == 0, \
        "this inspected restriction never improves the score"
    stated_stage = {1: (41.5026, 42.9204), 10: (26.9413, 28.0979), 30: (17.5512, 20.0168),
                    60: (13.0405, 15.3584), 120: (9.1042, 12.3338)}
    for round_number, (expected_train, expected_monitor) in stated_stage.items():
        close(round(stage_train[round_number - 1], 4), expected_train, f"boosting round {round_number} training", 1e-9)
        close(round(stage_monitor[round_number - 1], 4), expected_monitor, f"boosting round {round_number} monitoring", 1e-9)
    best_round = int(np.argmin(stage_monitor)) + 1
    assert best_round == 120, f"the best inspected round is {best_round}, not 120"

    ranges = [[float(X[:, column].min()), float(X[:, column].max())] for column in range(5)]
    target_range = [float(Y.min()), float(Y.max())]

    module = f'''/** Airfoil Self-Noise learning-curve results for the bias-variance lesson.
 *
 * Generated by scripts/verify-bias-variance-data.py, which recomputes the whole
 * declared campaign from the served dataset — once from an independent
 * implementation of the protocol and once through scikit-learn's own helpers —
 * and matches every value against the content packet before writing this file.
 * Do not edit by hand.
 *
 * Source: Brooks, Pope and Marcolini (1989), Airfoil Self-Noise, UCI dataset
 * 291, https://doi.org/10.24432/C5VW2C, licensed CC BY 4.0. This lesson serves
 * its own copy of the unchanged tab-separated file at
 * /learn-assets/bias-variance/airfoil-self-noise.dat
 * ({EXPECTED_BYTES} bytes, SHA-256 {EXPECTED_SHA}).
 *
 * Protocol: 1,200 development rows and 303 reserved rows split with seed {SPLIT_SEED};
 * five shuffled folds with seed {FOLD_SEED} inside development, each holding 960
 * available training rows and 240 validation rows. Learning-curve subsets are
 * prefixes of one permutation stream with seed {SUBSET_SEED}, so a requested size is a
 * count of fitted rows per fold. Trees use random state {TREE_SEED}. The boosting
 * trajectory uses a separate 960/240 split of development with seed {STAGE_SPLIT_SEED} and a
 * gradient-boosting fit with seed {STAGE_SEED}. No reserved row receives a prediction or a
 * score here or in the packet. Errors are mean squared error in squared decibels.
 *
 * Fitted with scikit-learn {sklearn.__version__} on Python {platform.python_version()}, NumPy {np.__version__}.
 */

export const provenance = {{
  name: "Airfoil Self-Noise",
  authors: "Brooks, T., Pope, D. and Marcolini, M. (1989)",
  doi: "https://doi.org/10.24432/C5VW2C",
  page: "https://archive.ics.uci.edu/dataset/291/airfoil+self+noise",
  license: "CC BY 4.0",
  licenseUrl: "https://creativecommons.org/licenses/by/4.0/",
  file: "/learn-assets/bias-variance/airfoil-self-noise.dat",
  attribution: "/learn-assets/bias-variance/ATTRIBUTION.txt",
  separator: "tab",
  lineEnding: "CRLF",
  bytes: {EXPECTED_BYTES},
  sha256: "{EXPECTED_SHA}",
  rows: 1503,
  columns: 6,
  developmentRows: 1200,
  reservedRows: 303,
  reservedPredictionsComputed: false,
  splitSeed: {SPLIT_SEED},
  foldSeed: {FOLD_SEED},
  folds: 5,
  foldTrainRows: 960,
  foldValidationRows: 240,
  subsetSeed: {SUBSET_SEED},
  treeSeed: {TREE_SEED},
  stageSplitSeed: {STAGE_SPLIT_SEED},
  stageSeed: {STAGE_SEED},
  stageFitRows: 960,
  stageMonitorRows: 240,
  limits: [
    "The collection holds related experimental settings and supplies no deployment-ready independent-run identifier, so a random row split is a row-level diagnostic.",
    "It cannot establish performance on a new airfoil family or a new experiment.",
    "These are model-development results; the 303 reserved rows were used for no reported selection or score."
  ]
}};

export const rawFeatureNames = {json.dumps(RAW_NAMES)};
export const rawFeatureLabels = {json.dumps(RAW_LABELS)};
export const rawFeatureUnits = {json.dumps(RAW_UNITS)};
export const targetLabel = "scaled sound-pressure level";
export const targetUnit = "dB";
/** Observed development minimum and maximum for each raw column, and the target. */
export const rawFeatureRanges = {json.dumps(ranges)};
export const targetRange = {json.dumps(target_range)};

/** Fitted rows per fold, not rows shared between folds. */
export const trainSizes = {json.dumps(TRAIN_SIZES)};

/** Four prespecified procedures. Each carries one row of five folds per fitted
 * size, in the fold order the protocol produced; a mean line is computed from
 * these values, never stored beside them. */
export const procedures = [
{LINE.join(f"""  {{
    model: {json.dumps(row["model"])},
    label: {json.dumps(row["label"])},
    description: {json.dumps(row["description"])},
    sizes: {json.dumps(TRAIN_SIZES)},
    trainMse: {fold_block(row["trainMse"], 4)},
    validationMse: {fold_block(row["validationMse"], 4)}
  }}""" for row in procedures)}
];

/** One tree setting changed at the full 960 fitted rows per fold. */
export const minSamplesLeafGrid = {json.dumps(LEAF_GRID)};
export const validationCurveRecord = {{
  minSamplesLeaf: {json.dumps(LEAF_GRID)},
  fitRowsPerFold: 960,
  trainMse: {fold_block(mine_leaf_train, 2)},
  validationMse: {fold_block(mine_leaf_valid, 2)}
}};

/** One prespecified boosting fit, traced through all 120 rounds on a separate
 * 960/240 split of the same development pool. */
export const boostingTrajectory = {{
  rounds: 120,
  maxDepth: 2,
  learningRate: 0.1,
  fitRows: 960,
  monitorRows: 240,
  trainMse: {numbers(stage_train)},
  monitorMse: {numbers(stage_monitor)}
}};
/** The rounds the manuscript prints; the best round is an argmin over all 120. */
export const printedRounds = {json.dumps(PRINTED_ROUNDS)};
'''

    if write:
        ASSET_DIR.mkdir(parents=True, exist_ok=True)
        ASSET_DATASET.write_bytes(packet_bytes)
        (ASSET_DIR / "ATTRIBUTION.txt").write_text(
            "Airfoil Self-Noise\n"
            "Brooks, T., Pope, D. and Marcolini, M. (1989). UCI Machine Learning Repository.\n"
            "https://doi.org/10.24432/C5VW2C  ·  https://archive.ics.uci.edu/dataset/291/airfoil+self+noise\n"
            "Licensed CC BY 4.0: https://creativecommons.org/licenses/by/4.0/\n"
            f"airfoil-self-noise.dat is the unchanged member, {EXPECTED_BYTES} bytes, SHA-256 {EXPECTED_SHA}.\n"
            "Tab separated, CRLF line endings, 1,503 rows and six columns: frequency Hz, attack angle degrees,\n"
            "chord m, free-stream velocity m/s, suction-side displacement thickness m, scaled sound pressure dB.\n"
            "This lesson serves its own copy so that no other lesson's asset edit can change it.\n",
            encoding="utf-8", newline="\n",
        )
        MODULE.write_text(module, encoding="utf-8", newline="\n")
    else:
        assert MODULE.exists(), "the data module is missing; rerun with --write"
        existing = MODULE.read_text(encoding="utf-8")
        assert existing == module, (
            "a fresh regeneration is not byte-identical to src/learn/data/bias-variance-data.js; "
            "rerun with --write and inspect the difference"
        )
        assert (ASSET_DIR / "ATTRIBUTION.txt").exists(), "the served attribution file is missing"

    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps({
        "checkedAt": datetime.now(timezone.utc).isoformat(),
        "mode": "write" if write else "read-only",
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
        "datasetSha256": digest,
        "packetCalculatedInputsSha256": packet_sha,
        "trustRootChecks": trust_root_checks,
        "datasetBytes": len(packet_bytes),
        "servedCopy": str(ASSET_DATASET.relative_to(ROOT)).replace("\\", "/"),
        "servedMatchesPacket": served == packet_bytes,
        "rows": int(data.shape[0]),
        "split": {"development": len(development), "reserved": len(untouched)},
        "fits": {
            "learningCurve": len(PROCEDURES) * len(TRAIN_SIZES) * 5,
            "learningCurveLibraryRepeat": len(PROCEDURES) * len(TRAIN_SIZES) * 5,
            "validationCurve": len(LEAF_GRID) * 5,
            "validationCurveLibraryRepeat": len(LEAF_GRID) * 5,
            "boosting": 1,
        },
        "warnings": caught,
        "statedValidationTable": {str(size): stated_validation[size] for size in TRAIN_SIZES},
        "crossoverSize": crossings[0],
        "restrictionHelpsAt": helps,
        "bestInspectedBoostingRound": best_round,
        "leafOneTrainingExactlyZero": True,
        "reservedPredictionsComputed": False,
        "moduleSha256": hashlib.sha256(MODULE.read_bytes()).hexdigest(),
        "verifierSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": (
            "Recomputed the 100-fit learning curve, the 30-fit validation curve and the 120-round boosting "
            "trajectory from the served tab-separated .dat using an independent implementation of the protocol, "
            "repeated them through scikit-learn's learning_curve and validation_curve, matched both against the "
            "content packet's calculated-inputs.json and against every value the manuscript prints, then checked "
            "that a fresh regeneration of src/learn/data/bias-variance-data.js is byte-identical. Also re-derived "
            "the packet's own trust root — every value of its finiteExperiments and doubleDescentApproximation "
            "blocks, which verify-bias-variance-models.mjs consumes as ground truth — from the recorded settings "
            "alone, through normal equations rather than the author script's lstsq, and recorded that file's hash."
        ),
        "limitations": [
            "One row-level split of one collection; related experimental settings are not independent runs.",
            "Development results only; no reserved row was predicted or scored.",
            "Numerical fitting can differ on other library versions.",
        ],
        "passed": True,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        f"PASS: {data.shape[0]} rows; 100 learning fits, 30 validation fits and one 120-round trajectory "
        f"recomputed twice and matched; {trust_root_checks:,} packet trust-root values re-derived; "
        f"module {'written' if write else 'byte-identical'} ({MODULE.stat().st_size / 1024:.0f} KB)."
    )


if __name__ == "__main__":
    main()
