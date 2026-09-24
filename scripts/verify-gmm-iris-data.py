"""Regenerate and verify the Iris data module for the Gaussian mixture lesson.

Refits every candidate from the supplied CSV rather than copying the content
packet, then checks the refit against the packet's recorded numbers before
writing src/learn/data/gmm-iris-data.js.

Run: scratch/lesson-tools/Scripts/python.exe scripts/verify-gmm-iris-data.py
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "docs/teaching/drafts/gaussian-mixture-models-gmm-em-algorithm"
MODULE = ROOT / "src/learn/data/gmm-iris-data.js"
ASSET = ROOT / "public/learn-assets/gmm/iris.csv"
EVIDENCE = ROOT / "docs/teaching/evidence/gmm-iris-data.json"
FAMILIES = ["full", "tied", "diag", "spherical"]
TOLERANCE = 1e-9
SEPARATOR = "," + chr(10)


def number(value, digits):
    """Round for display without letting %g truncate an embedded value.

    A rounded negative zero is published as zero: a covariance whose off-diagonal
    prints as -0.0 reads as asymmetric when it is not.
    """
    rounded = float(np.round(float(value), digits))
    return 0.0 if rounded == 0 else rounded


def main() -> None:
    checked = json.loads((PACKET / "checked-results.json").read_text(encoding="utf-8"))
    raw = (PACKET / "iris.csv").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == checked["iris"]["csv_sha256"], "the supplied CSV is not the checkpointed snapshot"

    rows = list(csv.DictReader((PACKET / "iris.csv").open(newline="", encoding="utf-8")))
    assert len(rows) == 150
    ids = np.array([int(row["observation_id"]) for row in rows])
    assert ids.tolist() == list(range(1, 151))
    X = np.array([[float(row["sepal_length_cm"]), float(row["sepal_width_cm"])] for row in rows])
    species = np.array([row["species"] for row in rows])
    assert sorted(set(species)) == ["setosa", "versicolor", "virginica"]
    assert all((species == name).sum() == 50 for name in set(species))

    # One fixed label-blind permutation, exactly as the manuscript states.
    order = np.random.default_rng(16).permutation(len(X))
    train, validation, test = order[:90], order[90:120], order[120:]
    assert (ids[train]).tolist() == checked["iris"]["train_ids"], "training IDs differ from the packet"
    assert (ids[validation]).tolist() == checked["iris"]["validation_ids"]
    assert (ids[test]).tolist() == checked["iris"]["test_ids"]

    scaler = StandardScaler().fit(X[train])
    np.testing.assert_allclose(scaler.mean_, checked["iris"]["scale_mean"], rtol=0, atol=TOLERANCE)
    np.testing.assert_allclose(scaler.scale_, checked["iris"]["scale"], rtol=0, atol=TOLERANCE)
    Z = scaler.transform(X)

    fits = []
    for kind in FAMILIES:
        for k in range(1, 5):
            model = GaussianMixture(
                n_components=k, covariance_type=kind, n_init=5, random_state=16,
                reg_covar=1e-4, tol=1e-6, max_iter=500,
            ).fit(Z[train])
            assert model.converged_, f"unfinished fit: {kind}, K={k}"
            fits.append({
                "covariance_type": kind,
                "components": k,
                "train_mean_log_density": float(model.score(Z[train])),
                "validation_mean_log_density": float(model.score(Z[validation])),
                "train_bic": float(model.bic(Z[train])),
                "train_aic": float(model.aic(Z[train])),
                "model": model,
            })

    for recorded, refit in zip(checked["iris"]["candidates"], fits):
        assert recorded["covariance_type"] == refit["covariance_type"]
        assert recorded["components"] == refit["components"]
        for recorded_field, refit_field in (
            ("train_mean_log_density", "train_mean_log_density"),
            ("validation_mean_log_density", "validation_mean_log_density"),
            ("bic", "train_bic"),
            ("aic", "train_aic"),
        ):
            assert abs(recorded[recorded_field] - refit[refit_field]) < 1e-6, (recorded_field, recorded, refit)
        assert recorded["converged"] is True

    best = max(fits, key=lambda fit: fit["validation_mean_log_density"])
    assert [best["covariance_type"], best["components"]] == checked["iris"]["selected"]
    baseline = fits[0]
    assert (baseline["covariance_type"], baseline["components"]) == ("full", 1)
    lowest_bic = min(fits, key=lambda fit: fit["train_bic"])
    assert (lowest_bic["covariance_type"], lowest_bic["components"]) == ("full", 4)

    selected = best["model"]
    test_score = float(selected.score(Z[test]))
    baseline_score = float(baseline["model"].score(Z[test]))
    assignment = selected.predict(Z[test])
    ari = float(adjusted_rand_score(species[test], assignment))
    assert abs(test_score - checked["iris"]["test_mean_log_density"]) < 1e-9
    assert abs(baseline_score - checked["iris"]["baseline_test_mean_log_density"]) < 1e-9
    assert abs(ari - checked["iris"]["test_ari"]) < 1e-9
    row_log_density = selected.score_samples(Z[test])
    row_responsibility = selected.predict_proba(Z[test])
    np.testing.assert_allclose(row_log_density, checked["iris"]["test_log_density"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(row_responsibility, checked["iris"]["test_responsibilities"], rtol=0, atol=1e-9)
    assert ids[test].tolist() == checked["iris"]["test_ids_in_order"]

    # The training-BIC winner's narrow component, which is the diagnostic F5 shows.
    narrow_model = lowest_bic["model"]
    widths = narrow_model.covariances_[:, 1, 1]
    narrow = int(np.argmin(widths))
    narrow_component = {
        "component": narrow,
        "weight": float(narrow_model.weights_[narrow]),
        "mean": [float(value) for value in narrow_model.means_[narrow]],
        "covariance": [[float(value) for value in row] for row in narrow_model.covariances_[narrow]],
    }
    assert abs(narrow_component["covariance"][1][1] - 1e-4) < 1e-12, "the narrow width is the regularization level"
    raw_mean = scaler.inverse_transform(np.array([narrow_component["mean"]]))[0]
    raw_width_sd = float(np.sqrt(narrow_component["covariance"][1][1]) * scaler.scale_[1])

    bayesian = []
    for concentration in [0.01, 1.0, 10.0]:
        model = BayesianGaussianMixture(
            n_components=6, covariance_type="full",
            weight_concentration_prior_type="dirichlet_process",
            weight_concentration_prior=concentration,
            n_init=3, random_state=16, reg_covar=1e-4, tol=1e-6, max_iter=1000,
        ).fit(Z[train])
        assert model.converged_
        bayesian.append({
            "concentration": concentration,
            "weights": [number(value, 4) for value in model.weights_],
            "above_one_percent": int((model.weights_ > 0.01).sum()),
            "above_five_percent": int((model.weights_ > 0.05).sum()),
        })
    for recorded, refit in zip(checked["bayesian_sensitivity"], bayesian):
        assert recorded["concentration"] == refit["concentration"]
        np.testing.assert_allclose([number(v, 4) for v in recorded["weights"]], refit["weights"], rtol=0, atol=1e-4)
        assert recorded["active_above_point01"] == refit["above_one_percent"], (recorded, refit)
        assert recorded["converged"] is True

    split_of = np.empty(len(X), dtype=int)
    split_of[train] = 0
    split_of[validation] = 1
    split_of[test] = 2

    observations = [
        [int(ids[index]), number(X[index][0], 1), number(X[index][1], 1), int(split_of[index]),
         ["setosa", "versicolor", "virginica"].index(species[index])]
        for index in range(len(X))
    ]
    candidates = [
        [fit["covariance_type"], fit["components"], number(fit["validation_mean_log_density"], 6),
         number(fit["train_bic"], 3), number(fit["train_mean_log_density"], 6)]
        for fit in fits
    ]
    test_rows = [
        [int(ids[index]), number(row_log_density[position], 6), number(row_responsibility[position][0], 9),
         int(assignment[position])]
        for position, index in enumerate(test)
    ]
    selected_parameters = {
        "weights": [number(value, 9) for value in selected.weights_],
        "means": [[number(value, 9) for value in mean] for mean in selected.means_],
        "covariances": [[[number(value, 9) for value in row] for row in matrix] for matrix in selected.covariances_],
    }

    module = f'''/** Iris measurements and the fitted candidates for the Gaussian mixture lesson.
 *
 * Generated by scripts/verify-gmm-iris-data.py, which refits every candidate
 * from the supplied CSV and checks the result against the content packet before
 * writing this file. Do not edit by hand.
 *
 * Source: Fisher (1936), Iris, UCI record 53, CC BY 4.0, exported from the
 * corrected scikit-learn {sklearn.__version__} snapshot. The same 150 rows are served at
 * /learn-assets/gmm/iris.csv. Measurements are centimetres; the fitted values
 * below are in training-standardized coordinates, which are dimensionless.
 *
 * Fitted with scikit-learn {sklearn.__version__} on Python {platform.python_version()}, NumPy {np.__version__}.
 */

/** id, sepal length cm, sepal width cm, split (0 train, 1 validation, 2 test), species (0 setosa, 1 versicolor, 2 virginica). */
export const observations = [
{SEPARATOR.join("  " + json.dumps(row) for row in observations)}
];
export const speciesNames = ["setosa", "versicolor", "virginica"];
export const splitNames = ["train", "validation", "test"];
export const splitCounts = {{ train: 90, validation: 30, test: 30 }};

/** Training-only standardization, frozen and applied everywhere. */
export const scaler = {{
  mean: {json.dumps([number(v, 9) for v in scaler.mean_])},
  scale: {json.dumps([number(v, 9) for v in scaler.scale_])}
}};

/** covariance family, K, validation mean log-density, training BIC, training mean log-density. */
export const candidates = [
{SEPARATOR.join("  " + json.dumps(row) for row in candidates)}
];
export const selection = {{
  rule: "highest validation mean log-density",
  selected: {json.dumps(checked["iris"]["selected"])},
  baseline: ["full", 1],
  trainingBicWinner: ["full", 4],
  testMeanLogDensity: {number(test_score, 9)!r},
  baselineTestMeanLogDensity: {number(baseline_score, 9)!r},
  testAri: {number(ari, 9)!r},
  testRows: 30
}};
export const selectedParameters = {json.dumps(selected_parameters)};

/** The training-BIC winner's narrowest component, in standardized coordinates. */
export const narrowComponent = {{
  family: "full",
  components: 4,
  component: {narrow_component["component"]},
  weight: {number(narrow_component["weight"], 9)!r},
  mean: {json.dumps([number(v, 9) for v in narrow_component["mean"]])},
  covariance: {json.dumps([[number(v, 12) for v in row] for row in narrow_component["covariance"]])},
  regularization: 1e-4,
  rawMean: {json.dumps([number(v, 6) for v in raw_mean])},
  rawWidthStandardDeviation: {number(raw_width_sd, 9)!r},
  recordedResolution: 0.1
}};

/** id, log-density, first-component responsibility, argmax component. */
export const testRows = [
{SEPARATOR.join("  " + json.dumps(row) for row in test_rows)}
];

/** Dirichlet-process weight sensitivity on the same 90 training rows. */
export const bayesianSensitivity = {json.dumps(bayesian, indent=2)};
'''
    MODULE.write_text(module, encoding="utf-8", newline="\n")
    ASSET.parent.mkdir(parents=True, exist_ok=True)
    ASSET.write_bytes(raw)

    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps({
        "checkedAt": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__,
        },
        "csvSha256": digest,
        "rows": len(rows),
        "split": {"train": len(train), "validation": len(validation), "test": len(test)},
        "candidates": len(fits),
        "selected": checked["iris"]["selected"],
        "testMeanLogDensity": test_score,
        "baselineTestMeanLogDensity": baseline_score,
        "testAri": ari,
        "scope": (
            "Refitted all 16 candidates, the selected model's per-row test densities and responsibilities, and the three "
            "Dirichlet-process fits from the supplied CSV, then matched every value against the content packet's recorded "
            "results before generating src/learn/data/gmm-iris-data.js."
        ),
        "limitations": [
            "One fixed split of one curated balanced collection; not a sampling study.",
            "Numerical fitting can differ on other library versions.",
            "Species labels enter only the final ARI diagnostic.",
        ],
        "passed": True,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"PASS: 150 rows, {len(fits)} candidates refitted and matched, module {MODULE.stat().st_size / 1024:.0f} KB.")


if __name__ == "__main__":
    main()
