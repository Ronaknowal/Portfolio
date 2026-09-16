"""Regenerate and verify the airfoil data module for the regularization lesson.

Refits the whole declared campaign from the supplied dataset rather than copying
the content packet, checks every refitted value against the packet's recorded
`calculated-inputs.json`, then writes src/learn/data/regularization-data.js and
serves the unchanged data file from public/learn-assets/regularization/.

The supplied file is a tab-separated `.dat` with 1,503 rows and six columns and
CRLF line endings; its exact bytes are copied, never rewritten.

Run: scratch/lesson-tools/Scripts/python.exe scripts/verify-regularization-data.py
"""
from __future__ import annotations

import hashlib
import json
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "docs/teaching/drafts/regularization-l1-l2-elastic-net-dropout"
DATASET = PACKET / "airfoil-self-noise.dat"
MODULE = ROOT / "src/learn/data/regularization-data.js"
ASSET_DIR = ROOT / "public/learn-assets/regularization"
EVIDENCE = ROOT / "docs/teaching/evidence/regularization-data.json"

EXPECTED_SHA = "74c75fd71783f1e6b71f8a622b993dc592897a97cd689c5090a07147a1b097b3"
EXPECTED_BYTES = 59984
RAW_NAMES = ["frequency_hz", "attack_degrees", "chord_m", "speed_mps", "displacement_m"]
FAMILIES = ["ridge", "lasso", "elastic_net"]
STRENGTHS = [0.001, 0.01, 0.1, 1, 10, 100]
RATIO = 0.5
LINE = "," + chr(10)


def make_model(family: str, strength: float, n_fit: int):
    if family == "ridge":
        estimator = Ridge(alpha=n_fit * strength, solver="svd")
    elif family == "lasso":
        estimator = Lasso(alpha=strength, max_iter=50000, tol=1e-8)
    elif family == "elastic_net":
        estimator = ElasticNet(alpha=strength, l1_ratio=RATIO, max_iter=50000, tol=1e-8)
    elif family == "ols":
        estimator = LinearRegression()
    else:
        raise ValueError(family)
    return make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False), StandardScaler(), estimator,
    )


def close(actual, expected, label, tolerance=1e-9):
    difference = abs(float(actual) - float(expected))
    assert difference <= tolerance, f"{label}: {actual} versus {expected} (differ by {difference})"


def main() -> None:
    raw = DATASET.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert len(raw) == EXPECTED_BYTES, f"dataset size {len(raw)} is not the checkpointed {EXPECTED_BYTES}"
    assert digest == EXPECTED_SHA, "the supplied dataset is not the checkpointed file"
    assert b"\t" in raw and b"\r\n" in raw, "the packet file is tab separated with CRLF endings"

    packet = json.loads((PACKET / "calculated-inputs.json").read_text(encoding="utf-8"))
    recorded = packet["airfoil"]
    assert recorded["reserved_predictions_computed"] is False, "the packet must contain no reserved score"

    data = np.loadtxt(DATASET)
    assert data.shape == (1503, 6), f"expected 1503 rows and six columns, found {data.shape}"

    development, reserved = train_test_split(np.arange(len(data)), train_size=1200, random_state=41)
    assert development.tolist() == recorded["development_indices"], "development row IDs differ from the packet"
    assert reserved.tolist() == recorded["reserved_indices"], "reserved row IDs differ from the packet"
    assert len(development) == 1200 and len(reserved) == 303

    X, y = data[development, :5], data[development, 5]
    splits = list(KFold(n_splits=3, shuffle=True, random_state=202).split(X))
    for fold, (train, valid) in enumerate(splits):
        assert development[train].tolist() == recorded["folds"][fold]["train_indices"]
        assert development[valid].tolist() == recorded["folds"][fold]["validation_indices"]
        assert (len(train), len(valid)) == (800, 400)

    caught: list[str] = []
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")

        baselines = []
        for train, valid in splits:
            ols = make_model("ols", 0, len(train)).fit(X[train], y[train])
            baselines.append({
                "mean_prediction": float(y[train].mean()),
                "mean_mse": float(np.mean((y[valid] - y[train].mean()) ** 2)),
                "ols_mse": float(np.mean((ols.predict(X[valid]) - y[valid]) ** 2)),
            })
        for index, (refit, saved) in enumerate(zip(baselines, recorded["baselines"])):
            for field in ("mean_prediction", "mean_mse", "ols_mse"):
                close(refit[field], saved[field], f"baseline fold {index} {field}", 1e-9)

        rows = []
        for family in FAMILIES:
            for strength in STRENGTHS:
                fits = []
                for train, valid in splits:
                    model = make_model(family, strength, len(train)).fit(X[train], y[train])
                    prediction = model.predict(X[valid])
                    fits.append({
                        "validation_mse": float(np.mean((prediction - y[valid]) ** 2)),
                        "training_mse": float(np.mean((model.predict(X[train]) - y[train]) ** 2)),
                        "coefficients": model[-1].coef_.tolist(),
                        "intercept": float(model[-1].intercept_),
                        "nonzero": int(np.count_nonzero(model[-1].coef_)),
                    })
                rows.append({
                    "family": family, "strength": strength,
                    "mean_validation_mse": float(np.mean([fit["validation_mse"] for fit in fits])),
                    "fits": fits,
                })

        feature_names = None
        selected = []
        inference = None
        for family in FAMILIES:
            candidates = [(row["mean_validation_mse"], row["strength"]) for row in rows if row["family"] == family]
            score, strength = min(candidates)  # lower strength resolves an exact tie
            fitted = make_model(family, strength, len(X)).fit(X, y)
            feature_names = fitted[0].get_feature_names_out(RAW_NAMES).tolist()
            entry = {
                "family": family, "strength": strength, "selection_mse": score,
                "coefficients": fitted[-1].coef_.tolist(), "intercept": float(fitted[-1].intercept_),
                "nonzero": int(np.count_nonzero(fitted[-1].coef_)),
                "scale_mean": fitted[1].mean_.tolist(), "scale_scale": fitted[1].scale_.tolist(),
                "development_row0_prediction": float(fitted.predict(X[:1])[0]),
            }
            selected.append(entry)
            if family == "ridge":
                changed = X[0].copy()
                changed[0] = 1750.0
                inference = {
                    "row_id": int(development[0]), "raw_X": X[0].tolist(),
                    "observed_target": float(y[0]),
                    "base_prediction": float(fitted.predict(X[:1])[0]),
                    "changed_X": changed.tolist(),
                    "changed_prediction": float(fitted.predict(changed[None, :])[0]),
                }
        caught = [str(item.message) for item in observed]

    assert caught == [], f"the campaign raised warnings: {caught}"
    assert feature_names == recorded["feature_names"], "the polynomial term order differs from the packet"

    for refit, saved in zip(rows, recorded["candidate_results"]):
        assert (refit["family"], refit["strength"]) == (saved["family"], saved["strength"])
        close(refit["mean_validation_mse"], saved["mean_validation_mse"],
              f"{refit['family']} lambda={refit['strength']} mean validation MSE", 1e-9)
        for fold, (fit, packet_fit) in enumerate(zip(refit["fits"], saved["fits"])):
            close(fit["validation_mse"], packet_fit["validation_mse"],
                  f"{refit['family']} lambda={refit['strength']} fold {fold} MSE", 1e-9)
            close(fit["training_mse"], packet_fit["training_mse"],
                  f"{refit['family']} lambda={refit['strength']} fold {fold} training MSE", 1e-9)
            assert fit["nonzero"] == packet_fit["nonzero"], (refit["family"], refit["strength"], fold)
            np.testing.assert_allclose(fit["coefficients"], packet_fit["coefficients"], rtol=0, atol=1e-8)
            close(fit["intercept"], packet_fit["intercept"], "fold intercept", 1e-9)

    for refit, saved in zip(selected, recorded["selected"]):
        assert refit["family"] == saved["family"] and refit["strength"] == saved["strength"]
        close(refit["selection_mse"], saved["selection_mse"], f"{refit['family']} selected score", 1e-9)
        assert refit["nonzero"] == saved["nonzero"] == 20, "every selected refit keeps all twenty terms"
        np.testing.assert_allclose(refit["coefficients"], saved["coefficients"], rtol=0, atol=1e-8)
        np.testing.assert_allclose(refit["scale_mean"], saved["scale_mean"], rtol=0, atol=1e-9)
        np.testing.assert_allclose(refit["scale_scale"], saved["scale_scale"], rtol=0, atol=1e-9)
        close(refit["intercept"], saved["intercept"], "selected intercept", 1e-9)
        close(refit["development_row0_prediction"], saved["development_row0_prediction"], "row 1203 prediction", 1e-9)

    for field in ("row_id", "observed_target", "base_prediction", "changed_prediction"):
        assert abs(float(inference[field]) - float(recorded["inference_fixture"][field])) < 1e-9, field
    np.testing.assert_allclose(inference["raw_X"], recorded["inference_fixture"]["raw_X"], rtol=0, atol=0)
    np.testing.assert_allclose(inference["changed_X"], recorded["inference_fixture"]["changed_X"], rtol=0, atol=0)

    # The manuscript's stated aggregates, recomputed rather than copied.
    baseline_mean = float(np.mean([row["mean_mse"] for row in baselines]))
    ols_mean = float(np.mean([row["ols_mse"] for row in baselines]))
    close(baseline_mean, 45.072768, "the stated mean baseline", 1e-6)
    close(ols_mean, 17.350268, "the stated OLS mean", 1e-6)
    for family in FAMILIES:
        best = min((row["mean_validation_mse"], row["strength"]) for row in rows if row["family"] == family)
        assert best[1] == 0.001, f"{family} selects the smallest grid value"
    lasso_tenth = next(row for row in rows if row["family"] == "lasso" and row["strength"] == 0.1)
    assert [fit["nonzero"] for fit in lasso_tenth["fits"]] == [9, 9, 9], "lasso at 0.1 keeps nine terms in each fold"
    for strength in (10, 100):
        for family in ("lasso", "elastic_net"):
            row = next(r for r in rows if r["family"] == family and r["strength"] == strength)
            assert [fit["nonzero"] for fit in row["fits"]] == [0, 0, 0]
            close(row["mean_validation_mse"], baseline_mean, f"{family} at {strength} reaches the baseline", 1e-12)
    lasso_thousandth = next(row for row in rows if row["family"] == "lasso" and row["strength"] == 0.001)
    assert [fit["nonzero"] for fit in lasso_thousandth["fits"]] == [19, 20, 19], "two folds dropped one term"

    # The exact arithmetic the browser repeats for the inference trace, in the
    # same order the module uses, so the published prediction is reproducible.
    ridge = next(entry for entry in selected if entry["family"] == "ridge")
    def expand(values):
        terms = list(values)
        for i in range(5):
            for j in range(i, 5):
                terms.append(values[i] * values[j])
        return terms
    assert len(expand(inference["raw_X"])) == 20
    manual = ridge["intercept"] + sum(
        coefficient * (term - mean) / scale
        for coefficient, term, mean, scale
        in zip(ridge["coefficients"], expand(inference["raw_X"]), ridge["scale_mean"], ridge["scale_scale"])
    )
    close(manual, inference["base_prediction"], "the browser's term-by-term prediction", 1e-9)
    manual_changed = ridge["intercept"] + sum(
        coefficient * (term - mean) / scale
        for coefficient, term, mean, scale
        in zip(ridge["coefficients"], expand(inference["changed_X"]), ridge["scale_mean"], ridge["scale_scale"])
    )
    close(manual_changed, inference["changed_prediction"], "the browser's changed prediction", 1e-9)

    ranges = [[float(X[:, column].min()), float(X[:, column].max())] for column in range(5)]

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    (ASSET_DIR / "airfoil-self-noise.dat").write_bytes(raw)
    (ASSET_DIR / "ATTRIBUTION.txt").write_text(
        "Airfoil Self-Noise\n"
        "Brooks, T., Pope, D. and Marcolini, M. (1989). UCI Machine Learning Repository.\n"
        "https://doi.org/10.24432/C5VW2C  ·  https://archive.ics.uci.edu/dataset/291/airfoil+self+noise\n"
        "Licensed CC BY 4.0: https://creativecommons.org/licenses/by/4.0/\n"
        f"airfoil-self-noise.dat is the unchanged member, {EXPECTED_BYTES} bytes, SHA-256 {EXPECTED_SHA}.\n"
        "Tab separated, CRLF line endings, 1,503 rows and six columns: frequency Hz, attack angle degrees,\n"
        "chord m, free-stream speed m/s, suction-side displacement thickness m, scaled sound pressure dB.\n",
        encoding="utf-8", newline="\n",
    )

    def numbers(values, digits=None):
        return json.dumps([float(value) if digits is None else round(float(value), digits) for value in values])

    candidate_rows = [
        [row["family"], row["strength"], round(row["mean_validation_mse"], 6),
         [round(fit["validation_mse"], 6) for fit in row["fits"]],
         [fit["nonzero"] for fit in row["fits"]]]
        for row in rows
    ]
    path_rows = [
        {"family": row["family"], "strength": row["strength"],
         "folds": [[round(value, 9) for value in fit["coefficients"]] for fit in row["fits"]]}
        for row in rows
    ]

    module = f'''/** Airfoil Self-Noise results for the regularization lesson.
 *
 * Generated by scripts/verify-regularization-data.py, which refits the whole
 * declared campaign from the supplied dataset and matches every value against
 * the content packet before writing this file. Do not edit by hand.
 *
 * Source: Brooks, Pope and Marcolini (1989), Airfoil Self-Noise, UCI dataset
 * 291, https://doi.org/10.24432/C5VW2C, licensed CC BY 4.0. The unchanged
 * tab-separated file is served at /learn-assets/regularization/airfoil-self-noise.dat
 * ({EXPECTED_BYTES} bytes, SHA-256 {EXPECTED_SHA}).
 *
 * Protocol: 1,200 development rows and 303 reserved rows split with seed 41;
 * three shuffled folds with seed 202 inside development. No reserved row
 * receives a prediction or a score here or in the packet. Errors are mean
 * squared error in squared decibels. Coefficients are in the fold's own
 * standardized coordinates and are not comparable across folds.
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
  file: "/learn-assets/regularization/airfoil-self-noise.dat",
  separator: "tab",
  lineEnding: "CRLF",
  bytes: {EXPECTED_BYTES},
  sha256: "{EXPECTED_SHA}",
  rows: 1503,
  columns: 6,
  developmentRows: 1200,
  reservedRows: 303,
  reservedPredictionsComputed: false,
  splitSeed: 41,
  foldSeed: 202,
  folds: 3,
  foldTrainRows: 800,
  foldValidationRows: 400,
  limits: [
    "Related experimental settings occur in the collection and independent run IDs are not available, so this is a row-level diagnostic.",
    "It cannot substantiate performance on a new airfoil or a new experimental run.",
    "These are development comparison and selection scores, not independent performance claims."
  ]
}};

export const rawFeatureNames = {json.dumps(RAW_NAMES)};
export const rawFeatureLabels = ["frequency", "angle of attack", "chord length", "free-stream speed", "displacement thickness"];
export const rawFeatureUnits = ["Hz", "degrees", "m", "m/s", "m"];
/** Observed development minimum and maximum for each raw column. */
export const rawFeatureRanges = {json.dumps(ranges)};

/** The twenty degree-two terms, in the exact PolynomialFeatures order. */
export const featureNames = [
{LINE.join("  " + json.dumps(name) for name in feature_names)}
];

export const strengths = {json.dumps(STRENGTHS)};
export const elasticNetRatio = {RATIO};

/** family, lambda, mean validation MSE, the three fold MSEs, the three nonzero counts. */
export const candidates = [
{LINE.join("  " + json.dumps(row) for row in candidate_rows)}
];

/** Per-fold training means, their validation MSE, and unpenalized OLS on the same folds. */
export const baselines = {json.dumps([{ "meanPrediction": round(row["mean_prediction"], 9), "meanMse": round(row["mean_mse"], 9), "olsMse": round(row["ols_mse"], 9)} for row in baselines], indent=2)};
export const baselineMeanMse = {round(baseline_mean, 9)!r};
export const olsMeanMse = {round(ols_mean, 9)!r};

/** Signed coefficient paths: one row per fitted candidate, three folds of twenty
 * standardized coefficients each. Exact zeros are stored as 0. */
export const coefficientPaths = [
{LINE.join("  " + json.dumps(row) for row in path_rows)}
];

/** The selected candidate per family, refitted on all 1,200 development rows. */
export const selected = {json.dumps([{
        "family": entry["family"], "strength": entry["strength"],
        "selectionMse": round(entry["selection_mse"], 9), "nonzero": entry["nonzero"],
    } for entry in selected], indent=2)};

/** The saved final ridge model at lambda = 0.001, in raw-feature units in and decibels out. */
export const ridgeModel = {{
  family: "ridge",
  strength: 0.001,
  intercept: {ridge["intercept"]!r},
  coefficients: {numbers(ridge["coefficients"])},
  scaleMean: {numbers(ridge["scale_mean"])},
  scaleScale: {numbers(ridge["scale_scale"])}
}};

/** One development observation, traced through the saved ridge model. */
export const inferenceFixture = {{
  rowId: {inference["row_id"]},
  rawFeatures: {numbers(inference["raw_X"])},
  observedTarget: {inference["observed_target"]!r},
  basePrediction: {inference["base_prediction"]!r},
  changedFeatures: {numbers(inference["changed_X"])},
  changedPrediction: {inference["changed_prediction"]!r},
  note: "A development-row inference trace under one fitted model, not a held-out accuracy demonstration and not a causal effect."
}};
'''
    MODULE.write_text(module, encoding="utf-8", newline="\n")

    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps({
        "checkedAt": datetime.now(timezone.utc).isoformat(),
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
        "datasetSha256": digest,
        "datasetBytes": len(raw),
        "rows": int(data.shape[0]),
        "split": {"development": len(development), "reserved": len(reserved)},
        "fits": {"candidates": 54, "ols": 3, "selectedRefits": 3, "total": 60},
        "warnings": caught,
        "baselineMeanMse": baseline_mean,
        "olsMeanMse": ols_mean,
        "selected": [[entry["family"], entry["strength"], entry["selection_mse"], entry["nonzero"]] for entry in selected],
        "reservedPredictionsComputed": False,
        "scope": (
            "Refitted the full 60-fit campaign from the supplied tab-separated .dat, matched the split, the fold "
            "memberships, every fold MSE, training MSE, coefficient vector, nonzero count and intercept, the three "
            "selected refits and the inference fixture against the content packet's calculated-inputs.json, then "
            "generated src/learn/data/regularization-data.js and served the unchanged data file."
        ),
        "limitations": [
            "One row-level split of one collection; related experimental settings are not independent runs.",
            "Development selection scores only; no reserved row was predicted or scored.",
            "Numerical fitting can differ on other library versions.",
        ],
        "passed": True,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        f"PASS: {data.shape[0]} rows, 60 fits refitted and matched against the packet, "
        f"module {MODULE.stat().st_size / 1024:.0f} KB."
    )


if __name__ == "__main__":
    main()
