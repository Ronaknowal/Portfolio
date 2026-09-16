"""Regenerate and verify the penguin data module for the feature-preparation lesson.

Recomputes every fitted statistic, split, transformed row and comparison count
from the packet's own `penguins.csv` rather than copying the author's JSON, then
asserts each recorded value in `calculated-inputs.json` before writing
`src/learn/data/scaling-data.js` and serving the CSV from
`public/learn-assets/feature-scaling/`.

Read-only by default, so an independent reviewer can re-run it inside a no-write
boundary: it recomputes everything and asserts that the module already on disk is
byte-identical to what it would generate. `--write` regenerates the module, the
served asset and the evidence file.

Run: scratch/lesson-tools/Scripts/python.exe scripts/verify-scaling-data.py [--write]
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler,
)

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "docs/teaching/drafts/feature-scaling-encoding-imputation"
MODULE = ROOT / "src/learn/data/scaling-data.js"
ASSET = ROOT / "public/learn-assets/feature-scaling/penguins.csv"
EVIDENCE = ROOT / "docs/teaching/evidence/scaling-data.json"

NUMERIC = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
CATEGORICAL = ["sex"]
SPECIES = ["Adelie", "Chinstrap", "Gentoo"]
CATEGORIES = ["female", "male", "not_recorded"]
METHODS = ["raw", "standard", "minmax", "robust"]
SEPARATOR = "," + chr(10)


def cell(value):
    """A JSON-safe cell: missing measurements stay null, never a silent zero."""
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return float(value)


def cross_fit(categories, target, folds, smoothing=2.0):
    encoded = np.empty(len(target), dtype=float)
    priors = {}
    for held in np.unique(folds):
        donors = folds != held
        prior = float(target[donors].mean())
        priors[int(held)] = prior
        for row in np.flatnonzero(~donors):
            matching = donors & (categories == categories[row])
            encoded[row] = (target[matching].sum() + smoothing * prior) / (matching.sum() + smoothing)
    return encoded, priors


def main() -> None:
    recorded = json.loads((PACKET / "calculated-inputs.json").read_text(encoding="utf-8"))
    raw_bytes = (PACKET / "penguins.csv").read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    assert digest == recorded["data_sha256"], "the supplied CSV is not the checkpointed file"
    assert len(raw_bytes) == 15241, "the provenance records a 15,241-byte public CSV"

    data = pd.read_csv(PACKET / "penguins.csv")
    assert len(data) == 344, "344 observations"
    assert list(data.columns) == [
        "species", "island", "bill_length_mm", "bill_depth_mm",
        "flipper_length_mm", "body_mass_g", "sex", "year",
    ]
    missing = {name: int(count) for name, count in data.isna().sum().items()}
    assert missing == recorded["missing_counts"], (missing, recorded["missing_counts"])
    assert all(missing[name] == 2 for name in NUMERIC), "two missing cells per measurement"
    assert missing["sex"] == 11, "eleven unrecorded sexes"
    assert sorted(data["species"].unique()) == SPECIES

    X, y = data[NUMERIC + CATEGORICAL], data["species"]
    train, test = train_test_split(
        np.arange(len(data)), test_size=0.25, random_state=20, stratify=y,
    )
    assert train.tolist() == recorded["splits"]["train"], "training indices differ from the packet"
    assert test.tolist() == recorded["splits"]["test"], "held-out indices differ from the packet"
    assert len(train) == 258 and len(test) == 86
    assert set(train).isdisjoint(set(test)) and len(set(train) | set(test)) == 344

    def make_model(scaler):
        numeric_steps = Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", scaler),
        ])
        category_steps = Pipeline([
            ("impute", SimpleImputer(
                strategy="constant", fill_value="not_recorded", keep_empty_features=True,
            )),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        prepare = ColumnTransformer([
            ("numeric", numeric_steps, NUMERIC),
            ("category", category_steps, CATEGORICAL),
        ])
        return Pipeline([("prepare", prepare), ("classify", KNeighborsClassifier(n_neighbors=5))])

    scalers = {
        "raw": "passthrough",
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler(),
    }
    models, comparison, predictions = {}, [], {}
    for name in METHODS:
        model = make_model(scalers[name])
        model.fit(X.iloc[train], y.iloc[train])
        predicted = model.predict(X.iloc[test])
        models[name] = model
        predictions[name] = predicted
        matrix = confusion_matrix(y.iloc[test], predicted, labels=SPECIES).tolist()
        comparison.append({
            "method": name,
            "correct": int(np.sum(predicted == y.iloc[test].to_numpy())),
            "accuracy": float(accuracy_score(y.iloc[test], predicted)),
            "confusion": matrix,
        })

    for row, author in zip(comparison, recorded["results"]):
        assert row["method"] == author["method"]
        assert row["correct"] == author["correct"], (row, author)
        assert abs(row["accuracy"] - author["accuracy"]) < 1e-12
        assert row["confusion"] == author["confusion"], (row, author)
    assert [row["correct"] for row in comparison] == [67, 84, 85, 85], "the manuscript's counts"

    baseline = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(train), 1)), y.iloc[train])
    baseline_predicted = baseline.predict(np.zeros((len(test), 1)))
    baseline_correct = int(np.sum(baseline_predicted == y.iloc[test].to_numpy()))
    assert baseline_correct == 38, "the majority baseline gets 38 of 86"
    assert abs(accuracy_score(y.iloc[test], baseline_predicted) - recorded["baseline_accuracy"]) < 1e-12
    assert set(baseline_predicted) == {"Adelie"}, "the majority class in training is Adelie"

    author_scaled = recorded["scaled_inspection"]
    prepare = models["standard"].named_steps["prepare"]
    feature_names = prepare.get_feature_names_out().tolist()
    assert feature_names == author_scaled["feature_names"]
    assert feature_names[4:] == [
        "category__sex_female", "category__sex_male", "category__sex_not_recorded",
    ]
    numeric_branch = prepare.named_transformers_["numeric"]
    medians = numeric_branch.named_steps["impute"].statistics_.tolist()
    assert medians == author_scaled["train_medians"] == [45.0, 17.3, 197.0, 4000.0]
    standard = numeric_branch.named_steps["scale"]
    np.testing.assert_allclose(standard.mean_, author_scaled["mean"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(standard.scale_, author_scaled["scale"], rtol=0, atol=1e-12)

    # The fitted statistics belong to the training rows alone. Recompute them by
    # hand from the imputed training block to prove no held-out row took part.
    training_block = X.iloc[train][NUMERIC].to_numpy(dtype=float)
    imputed_training = np.where(np.isnan(training_block), np.array(medians), training_block)
    np.testing.assert_allclose(np.median(training_block, axis=0), medians, rtol=0, atol=1e-12)
    np.testing.assert_allclose(imputed_training.mean(axis=0), standard.mean_, rtol=0, atol=1e-9)
    np.testing.assert_allclose(imputed_training.std(axis=0, ddof=0), standard.scale_, rtol=0, atol=1e-9)

    fitted = {"medians": medians, "categories": CATEGORIES}
    for name in ["standard", "minmax", "robust"]:
        scale_step = models[name].named_steps["prepare"].named_transformers_["numeric"].named_steps["scale"]
        if name == "standard":
            fitted["standard"] = {
                "center": [float(v) for v in scale_step.mean_],
                "scale": [float(v) for v in scale_step.scale_],
            }
        elif name == "minmax":
            np.testing.assert_allclose(scale_step.data_min_, imputed_training.min(axis=0), rtol=0, atol=1e-12)
            fitted["minmax"] = {
                "center": [float(v) for v in scale_step.data_min_],
                "scale": [float(v) for v in (scale_step.data_max_ - scale_step.data_min_)],
                "dataMax": [float(v) for v in scale_step.data_max_],
            }
        else:
            quartiles = np.percentile(imputed_training, [25, 75], axis=0)
            np.testing.assert_allclose(scale_step.center_, np.median(imputed_training, axis=0), rtol=0, atol=1e-12)
            np.testing.assert_allclose(scale_step.scale_, quartiles[1] - quartiles[0], rtol=0, atol=1e-12)
            fitted["robust"] = {
                "center": [float(v) for v in scale_step.center_],
                "scale": [float(v) for v in scale_step.scale_],
                "quartiles": [[float(v) for v in row] for row in quartiles],
            }

    transformed = prepare.transform(X.iloc[test])
    np.testing.assert_allclose(transformed, author_scaled["transformed_test"], rtol=0, atol=1e-12)
    assert author_scaled["test_source_rows"] == test.tolist()
    first = author_scaled["raw_test"][0]
    assert (first["bill_length_mm"], first["bill_depth_mm"], first["flipper_length_mm"],
            first["body_mass_g"], first["sex"]) == (51.0, 18.8, 203.0, 4100.0, "male")
    assert int(test[0]) == 309, "the first held-out row is zero-based source row 309"
    np.testing.assert_allclose(
        transformed[0],
        [1.3091367504848355, 0.877270699293386, 0.16451000467236732, -0.11279257796742845, 0.0, 1.0, 0.0],
        rtol=0, atol=1e-12)
    assert predictions["standard"].tolist() == author_scaled["predictions"]
    assert y.iloc[test].tolist() == author_scaled["truth"]

    # Practice 6: the same record with a missing body mass takes the fitted median.
    missing_mass = X.iloc[test[:1]].copy()
    missing_mass.loc[missing_mass.index[0], "body_mass_g"] = np.nan
    practice_row = prepare.transform(missing_mass)[0]
    expected_mass = (4000.0 - standard.mean_[3]) / standard.scale_[3]
    assert abs(practice_row[3] - expected_mass) < 1e-12
    assert abs(practice_row[3] + 0.2368) < 5e-5, "the practice answer is about -0.2368"
    np.testing.assert_allclose(
        np.delete(practice_row, 3), np.delete(transformed[0], 3), rtol=0, atol=1e-12,
        err_msg="only the edited coordinate moves")

    # An unknown category encodes as zeros across the block; a missing one takes
    # the fitted not_recorded coordinate.
    unknown = X.iloc[test[:1]].copy()
    unknown.loc[unknown.index[0], "sex"] = "unrecorded_by_a_new_device"
    np.testing.assert_allclose(prepare.transform(unknown)[0][4:], [0.0, 0.0, 0.0], rtol=0, atol=0)
    absent = X.iloc[test[:1]].copy()
    absent.loc[absent.index[0], "sex"] = np.nan
    np.testing.assert_allclose(prepare.transform(absent)[0][4:], [0.0, 0.0, 1.0], rtol=0, atol=0)

    values = np.array([1.0, 2.0, 3.0, 4.0, 100.0])[:, None]
    scale_fixture = {}
    for name, scaler in [("standard", StandardScaler()), ("minmax", MinMaxScaler()), ("robust", RobustScaler())]:
        scaler.fit(values)
        scale_fixture[name] = {
            "values": [float(v) for v in scaler.transform(values).ravel()],
            "new150": float(scaler.transform([[150.0]])[0, 0]),
        }
        author = recorded["scale_fixture"][name]
        np.testing.assert_allclose(scale_fixture[name]["values"], author["values"], rtol=0, atol=1e-12)
        assert abs(scale_fixture[name]["new150"] - author["new_150"]) < 1e-12
    assert abs(scale_fixture["standard"]["values"][0] + 0.5383) < 1e-4
    assert scale_fixture["robust"]["values"] == [-1.0, -0.5, 0.0, 0.5, 48.5]
    assert scale_fixture["robust"]["new150"] == 73.5
    scale_fixture["statistics"] = {
        "standard": {"center": 22.0, "scale": float(values.std(ddof=0))},
        "minmax": {"center": 1.0, "scale": 99.0, "dataMax": 100.0},
        "robust": {"center": 3.0, "scale": 2.0, "quartiles": [2.0, 4.0]},
    }
    assert abs(scale_fixture["statistics"]["standard"]["scale"] - 39.0128) < 1e-4

    categories = np.array(["A", "A", "B", "B", "C", "C"])
    target = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    folds = np.array([0, 1, 0, 1, 0, 1])
    encoded, priors = cross_fit(categories, target, folds)
    np.testing.assert_allclose(encoded, recorded["target_encoding"]["encoded"], rtol=0, atol=1e-12)
    changed = target.copy()
    changed[0] = 0.0
    changed_encoded, _ = cross_fit(categories, changed, folds)
    np.testing.assert_allclose(changed_encoded, recorded["target_encoding"]["changed_encoded"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(encoded, [5 / 9, 7 / 9, 2 / 9, 4 / 9, 2 / 9, 7 / 9], rtol=0, atol=1e-12)
    np.testing.assert_allclose(changed_encoded, [5 / 9, 2 / 9, 2 / 9, 2 / 9, 2 / 9, 5 / 9], rtol=0, atol=1e-12)
    assert abs(priors[0] - 1 / 3) < 1e-12 and abs(priors[1] - 2 / 3) < 1e-12
    # Practice 7: changing row 3's target moves row 0 through the prior alone.
    practice = target.copy()
    practice[3] = 1.0
    practice_encoded, _ = cross_fit(categories, practice, folds)
    assert abs(practice_encoded[0] - 7 / 9) < 1e-12 and abs(practice_encoded[3] - 4 / 9) < 1e-12

    donors = np.array([[1.0, 10.0, 100.0], [3.0, np.nan, 300.0], [np.nan, 14.0, 500.0]])
    query = np.array([[2.0, 12.0, np.nan]])
    knn = KNNImputer(n_neighbors=2).fit(donors).transform(query)
    np.testing.assert_allclose(knn, recorded["knn_imputed"], rtol=0, atol=1e-12)
    assert knn.tolist() == [[2.0, 12.0, 200.0]]
    # The overlap-adjusted distances the manuscript states.
    assert abs(1.5 * (1 ** 2 + 2 ** 2) - 7.5) < 1e-12
    removed = donors.copy()
    removed[1, 2] = np.nan
    np.testing.assert_allclose(
        KNNImputer(n_neighbors=2).fit(removed).transform(query), [[2.0, 12.0, 300.0]], rtol=0, atol=1e-12)
    moved = donors.copy()
    moved[0, 2] = 140.0
    np.testing.assert_allclose(
        KNNImputer(n_neighbors=2).fit(moved).transform(query), [[2.0, 12.0, 220.0]], rtol=0, atol=1e-12)

    split_of = np.zeros(len(data), dtype=int)
    split_of[test] = 1
    sex_index = {"female": 0, "male": 1}
    rows = [
        [
            int(index), SPECIES.index(data["species"].iloc[index]),
            cell(data["bill_length_mm"].iloc[index]), cell(data["bill_depth_mm"].iloc[index]),
            cell(data["flipper_length_mm"].iloc[index]), cell(data["body_mass_g"].iloc[index]),
            sex_index.get(data["sex"].iloc[index]) if isinstance(data["sex"].iloc[index], str) else None,
            int(split_of[index]),
        ]
        for index in range(len(data))
    ]
    assert sum(row[7] for row in rows) == 86

    truth_index = [SPECIES.index(name) for name in y.iloc[test]]
    prediction_index = {name: [SPECIES.index(value) for value in predictions[name]] for name in METHODS}
    for name in METHODS:
        wrong = sum(1 for a, b in zip(truth_index, prediction_index[name]) if a != b)
        assert wrong == 86 - dict((row["method"], row["correct"]) for row in comparison)[name]
    assert sum(1 for a, b in zip(truth_index, prediction_index["standard"]) if a != b) == 2
    assert sum(1 for a, b in zip(truth_index, prediction_index["raw"]) if a != b) == 19

    module = f'''/** Palmer Penguins observations and the fitted preparation for the
 * feature-scaling lesson.
 *
 * Generated by scripts/verify-scaling-data.py, which recomputes every value
 * below from the packet's own penguins.csv and asserts it against
 * calculated-inputs.json before writing this file. Do not edit by hand.
 *
 * Source: Palmer Penguins (Horst, Hill and Gorman), CC0, from the project's
 * public CSV. The same 344 rows are served at
 * /learn-assets/feature-scaling/penguins.csv. Bill length, bill depth and
 * flipper length are millimetres; body mass is grams. Missing cells are null,
 * never zero.
 *
 * Every fitted statistic here was fitted on the 258 training rows alone and is
 * applied to the 86 held-out rows frozen.
 *
 * Computed with scikit-learn {sklearn.__version__} on Python {platform.python_version()}, NumPy {np.__version__}, pandas {pd.__version__}.
 */

export const provenance = {json.dumps({
    "name": "Palmer Penguins",
    "rows": 344,
    "bytes": len(raw_bytes),
    "sha256": digest,
    "licence": "CC0",
    "credit": "Allison Horst, Alison Hill and Kristen Gorman; observations collected by Gorman and colleagues at Palmer Station",
    "url": "https://allisonhorst.github.io/palmerpenguins/",
    "asset": "/learn-assets/feature-scaling/penguins.csv",
    "missing": missing,
    "limits": [
        "The original research context is ecological measurement; this small classification exercise is our own use.",
        "One fixed stratified split of one curated collection, not a demonstrated generalization to another population or protocol.",
        "Island, year and row position are excluded from the features; the row index is a file position, not an animal identifier.",
    ],
}, indent=2)};

export const columns = {{
  numeric: {json.dumps(NUMERIC)},
  units: {json.dumps(["mm", "mm", "mm", "g"])},
  categorical: {json.dumps(CATEGORICAL)},
  target: "species"
}};
export const speciesNames = {json.dumps(SPECIES)};
export const recordedSexes = {json.dumps(["female", "male"])};
export const fittedCategories = {json.dumps(CATEGORIES)};
export const featureNames = {json.dumps(feature_names)};

/** source row (zero-based), species, bill length mm, bill depth mm, flipper length mm, body mass g, sex (0 female, 1 male, null unrecorded), split (0 training, 1 held out). */
export const rows = [
{SEPARATOR.join("  " + json.dumps(row) for row in rows)}
];

export const split = {{
  rule: "train_test_split(test_size=0.25, random_state=20, stratify=species)",
  training: {len(train)},
  heldOut: {len(test)},
  /** Held-out source rows in the order the split produced them. */
  heldOutOrder: {json.dumps([int(v) for v in test])}
}};

/** Fitted on the 258 training rows only, then frozen. */
export const fitted = {json.dumps(fitted, indent=2)};

/** The seven output coordinates of each held-out row under the standard model. */
export const transformedHeldOut = [
{SEPARATOR.join("  " + json.dumps([float(v) for v in row]) for row in transformed)}
];

export const heldOutTruth = {json.dumps(truth_index)};
export const heldOutPredictions = {json.dumps(prediction_index, indent=2)};

/** One fixed split, one neighbour count, one categorical preparation. */
export const comparison = {json.dumps([
    {
        "method": "majority",
        "label": "Majority-species baseline",
        "correct": baseline_correct,
        "accuracy": float(accuracy_score(y.iloc[test], baseline_predicted)),
        "confusion": None,
    },
] + [
    {
        "method": row["method"],
        "label": label,
        "correct": row["correct"],
        "accuracy": row["accuracy"],
        "confusion": row["confusion"],
    }
    for row, label in zip(comparison, [
        "Median imputation, no scaling",
        "Median imputation, standard scaling",
        "Median imputation, min–max scaling",
        "Median imputation, robust scaling",
    ])
], indent=2)};
export const heldOutRows = {len(test)};
export const neighbours = 5;

/** The constructed five-value training column of section 2. */
export const scaleFixture = {json.dumps(scale_fixture, indent=2)};

/** The constructed six-row cross-fitting table of section 8. */
export const targetEncodingFixture = {json.dumps({
    "categories": categories.tolist(),
    "target": [int(v) for v in target],
    "folds": [int(v) for v in folds],
    "smoothing": 2.0,
    "priors": [priors[0], priors[1]],
    "encoded": [float(v) for v in encoded],
    "changedTargetRowZero": [int(v) for v in changed],
    "changedEncoded": [float(v) for v in changed_encoded],
}, indent=2)};

/** The constructed donor table of section 4, checked against KNNImputer. */
export const donorFixture = {json.dumps({
    "features": ["a", "b", "c"],
    "donors": [[1.0, 10.0, 100.0], [3.0, None, 300.0], [None, 14.0, 500.0]],
    "donorNames": ["D1", "D2", "D3"],
    "query": [2.0, 12.0, None],
    "neighbours": 2,
    "estimate": 200.0,
    "removedDonorTarget": 300.0,
    "movedDonorTarget": 220.0,
}, indent=2)};
'''

    write = "--write" in sys.argv
    if write:
        MODULE.write_text(module, encoding="utf-8", newline="\n")
        ASSET.parent.mkdir(parents=True, exist_ok=True)
        ASSET.write_bytes(raw_bytes)
    else:
        assert MODULE.exists(), "the data module is missing; run once with --write"
        assert MODULE.read_text(encoding="utf-8") == module, (
            "src/learn/data/scaling-data.js differs from a fresh regeneration; re-run with --write")
        assert ASSET.exists() and ASSET.read_bytes() == raw_bytes, (
            "the served CSV differs from the packet copy; re-run with --write")

    evidence = json.dumps({
        "checkedAt": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "sklearn": sklearn.__version__,
        },
        "csvSha256": digest,
        "csvBytes": len(raw_bytes),
        "rows": len(data),
        "missing": missing,
        "split": {"training": len(train), "heldOut": len(test)},
        "correctCounts": {
            "majority": baseline_correct,
            **{row["method"]: row["correct"] for row in comparison},
        },
        "fittedMedians": medians,
        "standardCenter": [float(v) for v in standard.mean_],
        "standardScale": [float(v) for v in standard.scale_],
        "scope": (
            "Refitted the four preparation pipelines, the majority baseline, every fitted statistic, all 86 transformed "
            "held-out rows, the practice missing-mass coordinate, the unknown/missing category policies, the five-value "
            "scaler fixture, the six-row cross-fitting table and the three-donor KNN imputation from the packet's CSV, "
            "and matched each against calculated-inputs.json before generating src/learn/data/scaling-data.js."
        ),
        "limitations": [
            "One fixed stratified split of one curated collection; not a sampling study.",
            "Numerical results can differ on other library versions.",
            "Two rows lack every measurement and eleven lack recorded sex; both are carried through, not dropped.",
        ],
        "passed": True,
    }, indent=2) + "\n"
    if write:
        EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
        EVIDENCE.write_text(evidence, encoding="utf-8", newline="\n")
    print(
        f"PASS: 344 rows, {len(train)}/{len(test)} split, four preparations refitted and matched, "
        f"module {MODULE.stat().st_size / 1024:.0f} KB"
        f"{', module and evidence rewritten' if write else ' (read-only, module already matches)'}."
    )


if __name__ == "__main__":
    main()
