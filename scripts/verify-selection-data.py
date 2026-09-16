"""Regenerate and verify the Wine data module for the feature-selection lesson.

Re-runs the whole declared study from the served dataset rather than copying the
content packet, checks every refitted value against the packet's recorded
`calculated-inputs.json`, then emits src/learn/data/selection-data.js.

The script is read-only unless given `--write`. Without the flag it regenerates
the module text in memory and asserts it is byte-identical to the file on disk,
so a stale module is a failure rather than a silent overwrite.

Declared protocol, from the packet:
  178 source rows -> 138 development / 40 reserved, stratified, seed 51
  138 development -> 100 fitting / 38 inspection, stratified, seed 52
  100 fitting -> three stratified folds, seed 53
  SelectKBest(mutual_info_classif, discrete_features=False, n_neighbors=3,
              random_state=54) with k in {3, 6, 13}
  DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=55)
  twenty permutation donor orderings, seeds 56 + r
  Exactly eleven fits: nine inner CV candidates, one selected refit, one
  separately predeclared four-field refit. No reserved row is ever scored.

Run:
  scratch/lesson-tools/Scripts/python.exe scripts/verify-selection-data.py --write
  scratch/lesson-tools/Scripts/python.exe scripts/verify-selection-data.py
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import warnings
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np
import sklearn
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "docs/teaching/drafts/feature-selection-importance-shap-permutation-mutual-info"
DATASET = PACKET / "wine.data"
MODULE = ROOT / "src/learn/data/selection-data.js"
ASSET_DIR = ROOT / "public/learn-assets/feature-selection"
EVIDENCE = ROOT / "docs/teaching/evidence/selection-data.json"

EXPECTED_SHA = "6be6b1203f3d51df0b553a70e57b8a723cd405683958204f96d23d7cd6aea659"
EXPECTED_BYTES = 10782
NAMES = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
    "color_intensity", "hue", "od280_od315", "proline",
]
LABELS = [
    "alcohol", "malic acid", "ash", "alkalinity of ash", "magnesium",
    "total phenols", "flavanoids", "nonflavanoid phenols", "proanthocyanins",
    "color intensity", "hue", "OD280/OD315", "proline",
]
SIZES = [3, 6, 13]
SMALL_COLUMNS = [0, 1, 6, 12]
REPEATS = 20
NL = chr(10)

# Control bounds for the raw-field investigation. They are wider than every
# observed source value so the manuscript's contrast edits (alcohol 13.5, malic
# acid 4.1) and an explicit extrapolation are all reachable, and narrow enough
# that a typo cannot produce an absurd specimen.
FIELD_LIMITS = [
    {"minimum": 9.0, "maximum": 17.0, "step": 0.01, "decimals": 2},
    {"minimum": 0.1, "maximum": 8.0, "step": 0.01, "decimals": 2},
    {"minimum": 0.0, "maximum": 7.0, "step": 0.01, "decimals": 2},
    {"minimum": 100.0, "maximum": 2000.0, "step": 1.0, "decimals": 1},
]

ATTRIBUTION = NL.join([
    "Wine",
    "Aeberhard, S. and Forina, M. (1992). UCI Machine Learning Repository.",
    "https://doi.org/10.24432/C5PC7J  .  https://archive.ics.uci.edu/static/public/109/wine.zip",
    "Licensed CC BY 4.0: https://creativecommons.org/licenses/by/4.0/",
    f"wine.data is the unchanged archive member, {EXPECTED_BYTES} bytes, SHA-256 {EXPECTED_SHA}.",
    "Comma separated, 178 rows and 14 columns: the cultivar class 1/2/3 followed by thirteen",
    "numeric measurements (alcohol, malic acid, ash, alkalinity of ash, magnesium, total phenols,",
    "flavanoids, nonflavanoid phenols, proanthocyanins, color intensity, hue, OD280/OD315, proline).",
    "The provider's names file does not establish physical units for all fields, so the lesson",
    "keeps raw source scales. The split, models and attributions in the lesson are its own work.",
    "",
])


def close(actual, expected, label, tolerance=1e-12):
    difference = abs(float(actual) - float(expected))
    assert difference <= tolerance, f"{label}: {actual} versus {expected} (differ by {difference})"


def vector(actual, expected, label, tolerance=1e-12):
    assert len(actual) == len(expected), f"{label}: length {len(actual)} versus {len(expected)}"
    for index, (left, right) in enumerate(zip(actual, expected)):
        close(left, right, f"{label}[{index}]", tolerance)


def shapley(values, dimension):
    """Exact Shapley values of a game given as 2**dimension coalition values."""
    result = np.zeros(dimension)
    for j in range(dimension):
        for mask in range(1 << dimension):
            if mask & (1 << j):
                continue
            size = mask.bit_count()
            weight = (math.factorial(size) * math.factorial(dimension - size - 1)
                      / math.factorial(dimension))
            result[j] += weight * (values[mask | (1 << j)] - values[mask])
    return result


def background_game(predict, instance, background):
    values = []
    for mask in range(1 << len(instance)):
        hybrid = background.copy()
        kept = [j for j in range(len(instance)) if mask & (1 << j)]
        hybrid[:, kept] = instance[kept]
        values.append(float(predict(hybrid).mean()))
    return np.array(values)


def impurity_importances(tree, feature_count):
    """Weighted training-impurity decrease per feature, normalised as sklearn does."""
    total = tree.n_node_samples[0]
    raw = np.zeros(feature_count)
    for node in range(tree.node_count):
        left, right = tree.children_left[node], tree.children_right[node]
        if left == -1:
            continue
        decrease = (tree.n_node_samples[node] * tree.impurity[node]
                    - tree.n_node_samples[left] * tree.impurity[left]
                    - tree.n_node_samples[right] * tree.impurity[right]) / total
        raw[tree.feature[node]] += decrease
    return raw / raw.sum()


def build():
    raw_bytes = DATASET.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    assert len(raw_bytes) == EXPECTED_BYTES, f"dataset size {len(raw_bytes)} is not the checkpointed {EXPECTED_BYTES}"
    assert digest == EXPECTED_SHA, "the supplied dataset is not the checkpointed file"
    assert b"," in raw_bytes, "the packet file is comma separated"

    packet = json.loads((PACKET / "calculated-inputs.json").read_text(encoding="utf-8"))
    recorded = packet["wine"]
    assert recorded["reserved_scored"] is False, "the packet must contain no reserved score"
    assert recorded["feature_names"] == NAMES

    data = np.loadtxt(DATASET, delimiter=",")
    assert data.shape == (178, 14), f"expected 178 rows and fourteen columns, found {data.shape}"
    features, target = data[:, 1:], data[:, 0].astype(int)
    assert np.isfinite(data).all(), "the source file has no missing entries"
    assert sorted(set(target.tolist())) == [1, 2, 3]

    development, reserved = train_test_split(
        np.arange(len(target)), train_size=138, stratify=target, random_state=51)
    fitting, inspection = train_test_split(
        development, train_size=100, stratify=target[development], random_state=52)
    assert development.tolist() == recorded["development_ids"], "development row IDs differ from the packet"
    assert reserved.tolist() == recorded["reserved_ids"], "reserved row IDs differ from the packet"
    assert fitting.tolist() == recorded["fitting_ids"], "fitting row IDs differ from the packet"
    assert inspection.tolist() == recorded["inspection_ids"], "inspection row IDs differ from the packet"
    assert len(set(development.tolist()) & set(reserved.tolist())) == 0
    assert len(set(fitting.tolist()) & set(inspection.tolist())) == 0

    folds = list(StratifiedKFold(3, shuffle=True, random_state=53).split(
        features[fitting], target[fitting]))
    score_mi = partial(mutual_info_classif, discrete_features=False, n_neighbors=3, random_state=54)

    def pipeline(k):
        return make_pipeline(
            SelectKBest(score_mi, k=k),
            DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=55))

    caught = []
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")

        candidates = []
        for index, k in enumerate(SIZES):
            saved = recorded["candidates"][index]
            assert saved["k"] == k
            runs = []
            for fold, (local_fit, local_validation) in enumerate(folds):
                fit_ids, validation_ids = fitting[local_fit], fitting[local_validation]
                saved_fold = saved["folds"][fold]
                assert fit_ids.tolist() == saved_fold["fit_ids"]
                assert validation_ids.tolist() == saved_fold["validation_ids"]
                model = pipeline(k).fit(features[fit_ids], target[fit_ids])
                prediction = model.predict(features[validation_ids])
                selected = np.flatnonzero(model[0].get_support()).tolist()
                mi_nats = model[0].scores_.tolist()
                assert selected == saved_fold["selected"], f"k={k} fold {fold} mask differs"
                vector(mi_nats, saved_fold["mi_nats"], f"k={k} fold {fold} MI nats", 1e-12)
                assert prediction.tolist() == saved_fold["predictions"]
                correct = int(np.sum(prediction == target[validation_ids]))
                assert correct == saved_fold["correct"]
                runs.append({
                    "fold": fold, "correct": correct, "total": int(len(validation_ids)),
                    "accuracy": float(np.mean(prediction == target[validation_ids])),
                    "selected": selected,
                    "miNats": [round(value, 9) for value in mi_nats],
                })
            mean_accuracy = float(np.mean([run["accuracy"] for run in runs]))
            close(mean_accuracy, saved["mean_accuracy"], f"k={k} mean accuracy")
            candidates.append({"k": k, "meanAccuracy": mean_accuracy, "folds": runs})

        # The declared size rule: the best mean, ties resolved toward the first,
        # smaller listed size. max() keeps the first maximum, which is that rule.
        chosen_k = max(candidates, key=lambda row: row["meanAccuracy"])["k"]
        assert chosen_k == recorded["selected"]["k"] == 6
        selected_model = pipeline(chosen_k).fit(features[fitting], target[fitting])
        selected_columns = np.flatnonzero(selected_model[0].get_support()).tolist()
        assert selected_columns == recorded["selected"]["columns"]
        selected_prediction = selected_model.predict(features[inspection])
        assert selected_prediction.tolist() == recorded["selected"]["inspection_predictions"]
        selected_correct = int(np.sum(selected_prediction == target[inspection]))
        assert selected_correct == recorded["selected"]["correct"] == 36

        small_features = features[:, SMALL_COLUMNS]
        small_model = DecisionTreeClassifier(
            max_depth=3, min_samples_leaf=5, random_state=55).fit(
            small_features[fitting], target[fitting])
        small_prediction = small_model.predict(small_features[inspection])
        assert small_prediction.tolist() == recorded["four_feature_model"]["inspection_predictions"]
        small_correct = int(np.sum(small_prediction == target[inspection]))
        assert small_correct == recorded["four_feature_model"]["correct"] == 36
        tree = small_model.tree_
        assert tree.node_count == 9, f"the predeclared tree has nine nodes, found {tree.node_count}"
        saved_tree = recorded["four_feature_model"]["tree"]
        assert tree.children_left.tolist() == saved_tree["children_left"]
        assert tree.children_right.tolist() == saved_tree["children_right"]
        assert tree.feature.tolist() == saved_tree["feature"]
        vector(tree.threshold.tolist(), saved_tree["threshold"], "tree thresholds", 0.0)
        assert tree.n_node_samples.tolist() == saved_tree["n_node_samples"]
        vector(tree.impurity.tolist(), saved_tree["impurity"], "tree impurity", 1e-15)
        for node, row in enumerate(tree.value[:, 0, :].tolist()):
            vector(row, saved_tree["value"][node], f"tree value {node}", 1e-15)

        mdi = small_model.feature_importances_.tolist()
        vector(mdi, recorded["four_feature_model"]["mdi"], "impurity importances", 1e-12)
        # Independent recomputation of that normalisation from the saved tree.
        vector(impurity_importances(tree, 4).tolist(), mdi, "recomputed impurity importances", 1e-12)

        majority_class = int(np.argmax(np.bincount(target[fitting])))
        majority_correct = int(np.sum(target[inspection] == majority_class))
        assert majority_class == recorded["majority_class_baseline"]["class"] == 2
        assert majority_correct == recorded["majority_class_baseline"]["correct"] == 15

        donor_orders = [np.random.default_rng(56 + r).permutation(len(inspection)) for r in range(REPEATS)]
        for order, saved_order in zip(donor_orders, recorded["donor_orders"]):
            assert order.tolist() == saved_order
        baseline_accuracy = float(np.mean(small_prediction == target[inspection]))
        permutation = []
        for column in range(4):
            drops = []
            for donor in donor_orders:
                changed = small_features[inspection].copy()
                changed[:, column] = changed[donor, column]
                drops.append(baseline_accuracy - float(np.mean(
                    small_model.predict(changed) == target[inspection])))
            saved_perm = recorded["permutation"][column]
            assert saved_perm["feature"] == NAMES[SMALL_COLUMNS[column]]
            vector(drops, saved_perm["drops"], f"permutation drops for {saved_perm['feature']}", 1e-12)
            close(float(np.mean(drops)), saved_perm["mean"], "permutation mean")
            close(float(np.std(drops)), saved_perm["std_population"], "permutation SD")
            permutation.append({
                "column": column, "name": NAMES[SMALL_COLUMNS[column]],
                "label": LABELS[SMALL_COLUMNS[column]],
                "drops": [round(value, 12) for value in drops],
                "mean": float(np.mean(drops)), "sd": float(np.std(drops)),
                "mdi": mdi[column],
            })

        background_ids = np.sort(fitting)
        assert background_ids.tolist() == recorded["background_ids"]
        background = small_features[background_ids]
        class_index = int(np.flatnonzero(small_model.classes_ == 1)[0])
        assert small_model.classes_.tolist() == [1, 2, 3] and class_index == 0

        def probability_one(rows):
            return small_model.predict_proba(rows)[:, class_index]

        explained_ids = inspection[:12]
        explained = []
        for position, row_id in enumerate(explained_ids):
            values = background_game(probability_one, small_features[row_id], background)
            phi = shapley(values, 4)
            saved_case = recorded["explained"][position]
            assert saved_case["source_id"] == int(row_id)
            vector(values.tolist(), saved_case["coalitions"], f"coalitions of row {row_id}", 1e-12)
            vector(phi.tolist(), saved_case["phi"], f"attributions of row {row_id}", 1e-12)
            assert abs(values[0] + phi.sum() - values[-1]) <= 1e-10
            explained.append({
                "sourceId": int(row_id), "input": small_features[row_id].tolist(),
                "actualClass": int(target[row_id]),
                "coalitions": [round(value, 12) for value in values.tolist()],
                "baseline": float(values[0]), "prediction": float(values[-1]),
                "phi": [round(value, 12) for value in phi.tolist()],
                "efficiencyError": float(values[0] + phi.sum() - values[-1]),
            })

        first = small_features[explained_ids[0]]
        changed_inference = []
        for saved_case, (name, index, value) in zip(
                recorded["changed_inference"],
                [("alcohol_contrast", 0, 13.5), ("malic_acid_null", 1, 4.1)]):
            changed = first.copy()
            changed[index] = value
            game = background_game(probability_one, changed, background)
            phi = shapley(game, 4)
            assert saved_case["name"] == name
            vector(game.tolist(), saved_case["coalitions"], f"{name} coalitions", 1e-12)
            vector(phi.tolist(), saved_case["phi"], f"{name} attributions", 1e-12)
            changed_inference.append({
                "name": name, "field": index, "value": value, "input": changed.tolist(),
                "coalitions": [round(entry, 12) for entry in game.tolist()],
                "phi": [round(entry, 12) for entry in phi.tolist()],
                "baseline": float(game[0]), "prediction": float(game[-1]),
            })

        alternative_ids = np.sort(fitting)[-12:]
        assert alternative_ids.tolist() == recorded["alternative_reference"]["background_ids"]
        assert set(target[alternative_ids].tolist()) == {3}, "the contrast cohort is entirely class 3"
        alternative_game = background_game(probability_one, first, small_features[alternative_ids])
        alternative_phi = shapley(alternative_game, 4)
        vector(alternative_game.tolist(), recorded["alternative_reference"]["coalitions"],
               "alternative coalitions", 1e-12)
        vector(alternative_phi.tolist(), recorded["alternative_reference"]["phi"],
               "alternative attributions", 1e-12)

    caught = [str(entry.message) for entry in observed]

    fit_ranges = [[float(small_features[fitting, j].min()), float(small_features[fitting, j].max())]
                  for j in range(4)]
    for index, bounds in enumerate(FIELD_LIMITS):
        assert bounds["minimum"] < fit_ranges[index][0] and bounds["maximum"] > fit_ranges[index][1], (
            f"the control range for field {index} must contain the whole observed fitting column")
    assert FIELD_LIMITS[0]["maximum"] >= 13.5 and FIELD_LIMITS[1]["maximum"] >= 4.1, (
        "the manuscript's contrast edits must be reachable inside the control bounds")

    return {
        "digest": digest,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "sklearn": sklearn.__version__},
        "warnings": caught,
        "developmentIds": development.tolist(), "reservedIds": reserved.tolist(),
        "fittingIds": fitting.tolist(), "inspectionIds": inspection.tolist(),
        "folds": [{"fold": index, "fitIds": fitting[local_fit].tolist(),
                   "validationIds": fitting[local_validation].tolist()}
                  for index, (local_fit, local_validation) in enumerate(folds)],
        "candidates": candidates,
        "selected": {
            "k": chosen_k, "columns": selected_columns,
            "names": [NAMES[column] for column in selected_columns],
            "labels": [LABELS[column] for column in selected_columns],
            "miNats": [round(value, 9) for value in selected_model[0].scores_.tolist()],
            "correct": selected_correct, "total": int(len(inspection)),
            "accuracy": float(np.mean(selected_prediction == target[inspection])),
        },
        "small": {
            "columns": SMALL_COLUMNS,
            "names": [NAMES[column] for column in SMALL_COLUMNS],
            "labels": [LABELS[column] for column in SMALL_COLUMNS],
            "correct": small_correct, "total": int(len(inspection)),
            "accuracy": baseline_accuracy,
            "classes": small_model.classes_.tolist(),
            "classIndex": class_index,
            "confusion": confusion_matrix(target[inspection], small_prediction, labels=[1, 2, 3]).tolist(),
            "mdi": mdi,
            "tree": {
                "childrenLeft": tree.children_left.tolist(),
                "childrenRight": tree.children_right.tolist(),
                "feature": tree.feature.tolist(),
                "threshold": tree.threshold.tolist(),
                "value": tree.value[:, 0, :].tolist(),
                "samples": tree.n_node_samples.tolist(),
                "impurity": tree.impurity.tolist(),
            },
            "fitRanges": fit_ranges,
            "limits": FIELD_LIMITS,
        },
        "majority": {"class": majority_class, "correct": majority_correct,
                     "total": int(len(inspection)),
                     "accuracy": float(np.mean(target[inspection] == majority_class))},
        "permutation": permutation,
        "backgroundIds": background_ids.tolist(),
        "backgroundRows": small_features[background_ids].tolist(),
        "inspectionRows": small_features[inspection].tolist(),
        "inspectionClasses": target[inspection].tolist(),
        "inspectionPredictions": small_prediction.tolist(),
        "explained": explained,
        "changedInference": changed_inference,
        "alternative": {
            "ids": alternative_ids.tolist(),
            "rows": small_features[alternative_ids].tolist(),
            "classes": target[alternative_ids].tolist(),
            "coalitions": [round(value, 12) for value in alternative_game.tolist()],
            "phi": [round(value, 12) for value in alternative_phi.tolist()],
            "baseline": float(alternative_game[0]),
            "prediction": float(alternative_game[-1]),
        },
    }


def dumps(value, indent=0):
    return json.dumps(value, ensure_ascii=False, indent=indent if indent else None)


def rows_text(rows, lead="  "):
    return ("," + NL).join(lead + json.dumps(row, ensure_ascii=False) for row in rows)


def render(study):
    versions = study["versions"]
    small = study["small"]
    return f'''/** Wine study records for the feature-selection lesson.
 *
 * Generated by scripts/verify-selection-data.py, which re-runs the whole
 * declared study from the supplied dataset and matches every value against the
 * content packet before writing this file. Do not edit by hand.
 *
 * Source: Aeberhard, S. and Forina, M. (1992), Wine, UCI Machine Learning
 * Repository, https://doi.org/10.24432/C5PC7J, licensed CC BY 4.0. The
 * unchanged comma-separated member is served at
 * /learn-assets/feature-selection/wine.data ({EXPECTED_BYTES} bytes, SHA-256
 * {EXPECTED_SHA}).
 *
 * Protocol: 138 development and 40 reserved rows, stratified, seed 51; 100
 * fitting and 38 inspection rows inside development, stratified, seed 52; three
 * stratified folds inside the fitting rows, seed 53. Eleven fits in total: nine
 * inner selection candidates, one selected refit and one separately predeclared
 * four-field refit. No reserved row receives a prediction or a score, here or in
 * the packet.
 *
 * Fitted with scikit-learn {versions["sklearn"]} on Python {versions["python"]}, NumPy {versions["numpy"]}.
 */

export const provenance = {{
  name: "Wine",
  creator: "Aeberhard, S. and Forina, M. (1992)",
  doi: "https://doi.org/10.24432/C5PC7J",
  archive: "https://archive.ics.uci.edu/static/public/109/wine.zip",
  license: "CC BY 4.0",
  licenseUrl: "https://creativecommons.org/licenses/by/4.0/",
  file: "/learn-assets/feature-selection/wine.data",
  attribution: "/learn-assets/feature-selection/ATTRIBUTION.txt",
  separator: "comma",
  bytes: {EXPECTED_BYTES},
  sha256: "{EXPECTED_SHA}",
  rows: 178,
  columns: 14,
  missingEntries: 0,
  units: "The provider's names file does not establish physical units for all fields; raw source scales are kept.",
  task: "Cultivar classification of three cultivars grown in one Italian region. Not wine quality.",
  limits: [
    "One small historical collection under one declared row-level split.",
    "No vineyard, vintage, run or laboratory grouping metadata is supplied, so no transfer claim is available.",
    "Inner fold scores participated in choosing the retained size; they are selection records, not independent evidence.",
    "The forty reserved rows receive no prediction and no score."
  ]
}};

export const featureNames = {dumps(NAMES)};
export const featureLabels = {dumps(LABELS)};
export const classLabels = [1, 2, 3];

export const split = {{
  total: 178,
  development: 138,
  reserved: 40,
  fitting: 100,
  inspection: 38,
  splitSeed: 51,
  innerSeed: 52,
  foldSeed: 53,
  selectorSeed: 54,
  treeSeed: 55,
  donorSeedBase: 56,
  donorRepeats: {REPEATS},
  treeDepth: 3,
  minSamplesLeaf: 5,
  neighbors: 3,
  sizes: {dumps(SIZES)},
  fits: {{ selectionCv: 9, selectedRefit: 1, fourFieldRefit: 1, total: 11 }},
  reservedScored: false
}};

export const developmentIds = {dumps(study["developmentIds"])};
export const reservedIds = {dumps(study["reservedIds"])};
export const fittingIds = {dumps(study["fittingIds"])};
export const inspectionIds = {dumps(study["inspectionIds"])};

/** The three inner folds, by source row ID. */
export const foldMemberships = [
{rows_text(study["folds"])}
];

/** One entry per evaluated retained size. `miNats` is that fold's own selector
 * estimate for all thirteen columns, in nats; `selected` is the mask it kept. */
export const candidates = [
{("," + NL).join("  " + dumps(row) for row in study["candidates"])}
];

/** The size the declared rule chose, refitted on all 100 fitting rows. */
export const selectedModel = {dumps(study["selected"], 2)};

/** The separately predeclared four-field tree. Not chosen by the MI search. */
export const fourFieldModel = {dumps(small, 2)};

export const majorityBaseline = {dumps(study["majority"], 2)};

/** Twenty seeded donor orderings per field, as accuracy decreases. The spread
 * describes donor randomization at a fixed model and fixed assessment rows. */
export const permutationRecords = [
{("," + NL).join("  " + dumps(row) for row in study["permutation"])}
];

/** The explanation reference: all 100 fitting rows, equally weighted, in four
 * raw source-scale fields. */
export const backgroundIds = {dumps(study["backgroundIds"])};
export const backgroundRows = [
{rows_text(study["backgroundRows"])}
];

/** The 38 inspection rows in the same four fields, with their actual cultivar
 * and the four-field tree's hard prediction. */
export const inspectionRows = [
{rows_text(study["inspectionRows"])}
];
export const inspectionClasses = {dumps(study["inspectionClasses"])};
export const inspectionPredictions = {dumps(study["inspectionPredictions"])};

/** The twelve predeclared explained cases, with their recorded coalition values
 * and attributions. The browser recomputes these from the saved tree; these are
 * the recorded oracle they must reproduce. */
export const explainedCases = [
{("," + NL).join("  " + dumps(case_) for case_ in study["explained"])}
];

/** Two recorded input contrasts on the first explained case. */
export const changedInference = [
{("," + NL).join("  " + dumps(case_) for case_ in study["changedInference"])}
];

/** An explicitly labelled class-3 cohort used as a contrasting reference. It is
 * the last twelve fitting rows in source order, not a representative sample. */
export const alternativeReference = {dumps(study["alternative"], 2)};
'''


def main():
    write = "--write" in sys.argv
    study = build()
    module = render(study)

    if write:
        MODULE.write_text(module, encoding="utf-8", newline="\n")
        ASSET_DIR.mkdir(parents=True, exist_ok=True)
        (ASSET_DIR / "wine.data").write_bytes(DATASET.read_bytes())
        (ASSET_DIR / "ATTRIBUTION.txt").write_text(ATTRIBUTION, encoding="utf-8", newline="\n")
    else:
        assert MODULE.exists(), "the data module has not been generated; run with --write"
        current = MODULE.read_text(encoding="utf-8")
        assert current == module, "the emitted data module differs from the file on disk"
        served = ASSET_DIR / "wine.data"
        assert served.exists(), "the served dataset copy is missing; run with --write"
        assert served.read_bytes() == DATASET.read_bytes(), "the served copy is not the packet file"
        assert (ASSET_DIR / "ATTRIBUTION.txt").read_text(encoding="utf-8") == ATTRIBUTION

    evidence = {
        "verifiedAt": datetime.now(timezone.utc).isoformat(),
        "stage": "native data derivation for the browser lesson; model, example and browser review are separate",
        "mode": "write" if write else "read-only comparison",
        "generator": "scripts/verify-selection-data.py",
        "generatorHash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "module": str(MODULE.relative_to(ROOT)).replace("\\", "/"),
        "moduleHash": hashlib.sha256(MODULE.read_bytes()).hexdigest(),
        "moduleBytes": MODULE.stat().st_size,
        "dataset": {"file": "public/learn-assets/feature-selection/wine.data",
                    "sha256": study["digest"], "bytes": EXPECTED_BYTES},
        "versions": {name: importlib.metadata.version(name) for name in ["numpy", "scikit-learn"]},
        "warnings": study["warnings"],
        "fits": {"selectionCv": 9, "selectedRefit": 1, "fourFieldRefit": 1, "total": 11},
        "reservedScored": False,
        "checks": [
            "Dataset bytes and SHA-256 match the checkpointed archive member; 178 rows, 14 columns, no missing entry.",
            "All four row-ID sets (138/40/100/38) and the three inner fold memberships reproduce the packet exactly.",
            "Nine inner candidate fits reproduce every selected mask, all thirteen MI estimates in nats, every fold prediction and correct count.",
            "The size rule selects six; the refit mask, its 36/38 inspection result and the majority 15/38 baseline reproduce.",
            "The predeclared four-field tree reproduces node for node, including thresholds at full double precision and normalised leaf values.",
            "Impurity importances match sklearn's feature_importances_ and an independent weighted-decrease recomputation.",
            "All twenty donor orderings and all eighty accuracy decreases reproduce; means and population SDs match the packet.",
            "All twelve explained cases reproduce their sixteen coalition values and four attributions, each reconstructing within 1e-10.",
            "Both recorded input contrasts and the class-3 contrast reference reproduce exactly.",
            "Every control bound strictly contains the observed fitting column and admits the manuscript's contrast edits.",
        ],
        "limits": [
            "One row-level stratified split of one small historical collection.",
            "Inner fold scores participated in selection; they are not independent performance estimates.",
            "No reserved row was predicted or scored.",
            "Numerical estimates can differ on other library versions.",
        ],
        "passed": True,
    }
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps(evidence, indent=2) + NL, encoding="utf-8", newline="\n")
    print(f"PASS: 11 fits re-run and matched against the packet, module "
          f"{MODULE.stat().st_size / 1024:.0f} KB ({'written' if write else 'byte-identical'}).")


main()
