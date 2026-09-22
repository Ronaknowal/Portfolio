"""Re-derive the end-to-end study's trust root and regenerate its recorded data module.

This script does two jobs.

1. It **re-derives the trust root**: every scalar leaf of the content packet's
   `calculated-inputs.json` is recomputed here, from the definitions and from the
   copy of the dataset this lesson serves, through implementations written
   independently of the packet's own `author-calculations.py`:

     * accuracy, balanced accuracy and every confusion matrix are written out
       from their definitions in exact rational arithmetic over the stored
       label and prediction vectors -- no scikit-learn metric is called;
     * log loss is the plain mean of -log p(actual), summed here;
     * the majority baseline's probabilities are re-derived exactly from the
       training class counts, because a prior-strategy dummy IS those counts;
     * the two logistic pipelines' probabilities are re-derived by standardising
       with a mean and population standard deviation computed here, applying the
       fitted coefficients as an explicit dot product, and writing out the
       softmax -- `predict_proba` is never called for the comparison;
     * the forest's probabilities are re-derived by walking all 100 fitted trees
       with an explicit decision-path loop and averaging their normalised leaf
       class distributions, rather than by calling `predict_proba`;
     * the slices, the paired repair/regression counts and the corrected counts
       are recomputed from the stored per-specimen rows;
     * the Wilson interval is written out from the scalar formula;
     * the acceptance fixture's ledger is recomputed by exact enumeration.

   Coverage is measured, not asserted. The comparison walks the packet tree and
   records the path of every scalar leaf it actually compared; the run fails
   unless that set is exactly the complete set of leaves. A block added to the
   packet therefore lowers the count and stops the build instead of passing
   unnoticed.

   Two groups of leaves are **validated rather than re-derived**, and this is
   stated in the evidence file rather than hidden:

     * the three split index lists are reproduced by the same two stated
       `train_test_split` calls, because the meaning of `random_state=21` is
       the library's own stream and reimplementing it would be reimplementing
       scikit-learn. What is checked independently is every property the lesson
       actually relies on: the three lists partition 1..178 exactly, are
       pairwise disjoint, have sizes 106/36/36, and preserve each cultivar's
       representation.
     * the fitted coefficients, intercepts and tree structures are obtained by
       refitting with the same estimators, because the estimators are the object
       under study. Everything computed FROM them here -- standardisation,
       scores, softmax, tree walks, votes, metrics -- is independent.

2. It regenerates `src/learn/data/endtoend-data.js` from those results.

The script is READ-ONLY unless given `--write`. Without it the module text is
rebuilt in memory and must be byte-identical to the file already on disk, and
the served asset must already match the packet bytes.

Run:
  scratch/lesson-tools/Scripts/python.exe scripts/verify-endtoend-data.py --write
  scratch/lesson-tools/Scripts/python.exe scripts/verify-endtoend-data.py
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "docs/teaching/drafts/end-to-end-supervised-learning-error-analysis"
PACKET_RESULTS = PACKET / "calculated-inputs.json"
PACKET_DATASET = PACKET / "wine.csv"
ASSET_DATASET = ROOT / "public/learn-assets/end-to-end/wine.csv"
ATTRIBUTION = ROOT / "public/learn-assets/end-to-end/ATTRIBUTION.txt"
MODULE = ROOT / "src/learn/data/endtoend-data.js"
EVIDENCE = ROOT / "docs/teaching/evidence/endtoend-data.json"

CLASSES = (0, 1, 2)
MODEL_KEYS = ("majority", "linear_two", "forest_two", "linear_three")
ELIGIBLE = ("linear_two", "forest_two", "linear_three")

write = "--write" in sys.argv
keep_evidence = "--no-evidence" not in sys.argv

problems: list[str] = []
checks = 0
seen_leaves: set[str] = set()
validated_not_rederived: set[str] = set()


def check(condition, label):
    global checks
    checks += 1
    if not condition:
        problems.append(label)
    return bool(condition)


def compare(path, actual, expected, tolerance=0.0):
    """Compare one packet leaf against an independently computed value.

    Recording the path is what makes coverage a measurement rather than a
    claim: a leaf nobody compared never enters `seen_leaves`.
    """
    seen_leaves.add(path)
    if isinstance(expected, float) or isinstance(actual, float):
        ok = isinstance(actual, (int, float)) and math.isfinite(float(actual)) \
            and abs(float(actual) - float(expected)) <= tolerance
    else:
        ok = actual == expected
    return check(ok, f"{path}: independently computed {actual!r}, packet holds {expected!r}")


def leaf_paths(node, prefix=""):
    """Every scalar leaf path in the packet tree, plus a marker for empty containers."""
    if isinstance(node, dict):
        if not node:
            yield f"{prefix}/{{}}"
            return
        for key, value in node.items():
            yield from leaf_paths(value, f"{prefix}/{key}")
    elif isinstance(node, list):
        if not node:
            yield f"{prefix}/[]"
            return
        for index, value in enumerate(node):
            yield from leaf_paths(value, f"{prefix}/{index}")
    else:
        yield prefix


# ------------------------------------------------------- metrics from definitions
def confusion_matrix(actual, predicted):
    return [[sum(1 for a, p in zip(actual, predicted) if a == c and p == k) for k in CLASSES]
            for c in CLASSES]


def accuracy(actual, predicted):
    return Fraction(sum(1 for a, p in zip(actual, predicted) if a == p), len(actual))


def balanced_accuracy(actual, predicted):
    """The mean of the per-class recalls, in exact rational arithmetic.

    A class with no support contributes no recall and is not averaged in; that
    never happens on these splits and is asserted rather than assumed.
    """
    recalls = []
    for c in CLASSES:
        support = sum(1 for a in actual if a == c)
        if support:
            recalls.append(Fraction(sum(1 for a, p in zip(actual, predicted) if a == c and p == c), support))
    return sum(recalls) / len(recalls)


def log_loss(actual, probabilities):
    return sum(-math.log(row[a]) for a, row in zip(actual, probabilities)) / len(actual)


def softmax(scores):
    largest = max(scores)
    weights = [math.exp(value - largest) for value in scores]
    total = sum(weights)
    return [weight / total for weight in weights]


def argmax(values):
    """First index of the greatest value. The tie rule is the stated one: order."""
    best = 0
    for index in range(1, len(values)):
        if values[index] > values[best]:
            best = index
    return best


def tree_probabilities(forest, matrix):
    """Average the trees' normalised leaf class distributions, by walking each tree.

    This is the definition of a forest's predicted probability under soft
    voting, written out with an explicit decision-path loop rather than obtained
    from `predict_proba`.

    The single-precision cast is not decoration. A fitted tree stores its
    thresholds in double precision but its inference routine casts the input
    matrix to float32 first, so the comparison at each split is a float32 value
    promoted back to double. Walking with the raw double values instead routed
    three of the thirty-six validation specimens -- 94, 98 and 150 -- to
    different LEAVES than the fitted model, at a split where the two
    representations fall on opposite sides of the threshold. That is a real
    routing difference, not a rounding blemish, and getting it wrong would have
    made the comparison look like a packet error.

    Two precisions worth stating, both confirmed by independent review. The
    difference is at the leaf and therefore in the PROBABILITIES, by up to
    7.6e-2; the predicted LABELS agree on all thirty-six either way. And the
    cast belongs on the INPUT ONLY: casting the threshold as well re-introduces
    disagreements, because not every stored threshold is exactly
    float32-representable.
    """
    rows = []
    for sample in matrix:
        single = np.asarray(sample, dtype=np.float32)
        totals = [0.0, 0.0, 0.0]
        for estimator in forest.estimators_:
            tree = estimator.tree_
            node = 0
            while tree.children_left[node] != -1:
                feature = tree.feature[node]
                node = tree.children_left[node] if float(single[feature]) <= float(tree.threshold[node]) \
                    else tree.children_right[node]
            counts = tree.value[node][0]
            total = float(sum(counts))
            for index in CLASSES:
                totals[index] += float(counts[index]) / total
        rows.append([value / len(forest.estimators_) for value in totals])
    return rows


def acceptance_ledger(cases, threshold, wrong_cost, defer_cost):
    accepted = [case for case in cases if case["confidence"] >= threshold]
    wrong = sum(1 for case in accepted if not case["correct"])
    deferred = len(cases) - len(accepted)
    return {
        "accepted": len(accepted), "wrong": wrong, "deferred": deferred,
        "coverage": len(accepted) / len(cases),
        "conditionalError": (wrong / len(accepted)) if accepted else None,
        "cost": wrong_cost * wrong + defer_cost * deferred,
    }


def main():
    packet = json.loads(PACKET_RESULTS.read_text(encoding="utf-8"))
    all_leaves = set(leaf_paths(packet))

    # ---------------------------------------------------------- the served copy
    check(ASSET_DATASET.exists(), "the lesson serves its own copy of wine.csv")
    check(ATTRIBUTION.exists(), "the served copy carries an attribution file")
    packet_bytes = PACKET_DATASET.read_bytes()
    asset_bytes = ASSET_DATASET.read_bytes() if ASSET_DATASET.exists() else b""
    check(asset_bytes == packet_bytes,
          "the served wine.csv is byte-identical to the packet's frozen copy")
    asset_digest = hashlib.sha256(asset_bytes).hexdigest()
    compare("/dataSha256", asset_digest, packet["dataSha256"])

    # -------------------------------------------------- the table, read here again
    with ASSET_DATASET.open(newline="", encoding="utf-8") as handle:
        table = list(csv.DictReader(handle))
    check(len(table) == 178, f"the served table has {len(table)} rows, not 178")
    check(len({row["specimen_id"] for row in table}) == 178, "all 178 specimen ids are distinct")
    check(all(math.isfinite(float(value)) for row in table for key, value in row.items() if key != "specimen_id"),
          "every measurement in the served table is finite")
    by_id = {int(row["specimen_id"]): row for row in table}
    order = [int(row["specimen_id"]) for row in table]
    check(order == list(range(1, 179)), "specimen ids run 1..178 in source row order")

    alcohol = np.array([float(row["alcohol"]) for row in table])
    colour = np.array([float(row["color_intensity"]) for row in table])
    flavanoids = np.array([float(row["flavanoids"]) for row in table])
    labels = np.array([int(row["cultivar"]) for row in table])
    two = np.column_stack([alcohol, colour])
    three = np.column_stack([alcohol, colour, flavanoids])

    # ------------------------------------------- the split: reproduced, then tested
    positions = np.arange(len(labels))
    development, test_positions = train_test_split(positions, test_size=36, stratify=labels, random_state=21)
    train_positions, valid_positions = train_test_split(
        development, test_size=36, stratify=labels[development], random_state=22)
    reproduced = {
        "train": (train_positions + 1).tolist(),
        "validation": (valid_positions + 1).tolist(),
        "test": (test_positions + 1).tolist(),
    }
    for name, ids in reproduced.items():
        for index, value in enumerate(ids):
            compare(f"/splits/{name}/{index}", value, packet["splits"][name][index])
            validated_not_rederived.add(f"/splits/{name}/{index}")
    # The properties the lesson actually relies on, tested independently of the seed.
    union = set(reproduced["train"]) | set(reproduced["validation"]) | set(reproduced["test"])
    check(union == set(range(1, 179)), "the three splits cover every specimen exactly once")
    check(len(reproduced["train"]) + len(reproduced["validation"]) + len(reproduced["test"]) == 178,
          "the three split sizes add to 178, so no specimen is shared")
    check(len(reproduced["train"]) == 106 and len(reproduced["validation"]) == 36
          and len(reproduced["test"]) == 36, "the split sizes are 106 / 36 / 36")
    class_counts = {}
    for name, ids in reproduced.items():
        counts = [sum(1 for i in ids if int(by_id[i]["cultivar"]) == c) for c in CLASSES]
        class_counts[name] = counts
    check(class_counts["validation"] == class_counts["test"],
          "the stratified validation and test sets carry the same cultivar counts")
    check(class_counts["train"] == [35, 43, 28], f"training cultivar counts are {class_counts['train']}")
    check(class_counts["validation"] == [12, 14, 10],
          f"validation cultivar counts are {class_counts['validation']}")
    for index, c in enumerate(CLASSES):
        whole = sum(1 for row in table if int(row["cultivar"]) == c)
        share = class_counts["validation"][index] / 36
        check(abs(share - whole / 178) < 0.02,
              f"stratification keeps class {c} at {share:.4f} against {whole / 178:.4f} overall")

    train_ids = reproduced["train"]
    valid_ids = reproduced["validation"]
    test_ids = reproduced["test"]
    train_index = np.array([i - 1 for i in train_ids])
    valid_index = np.array([i - 1 for i in valid_ids])
    test_index = np.array([i - 1 for i in test_ids])
    valid_actual = [int(by_id[i]["cultivar"]) for i in valid_ids]
    train_actual = [int(by_id[i]["cultivar"]) for i in train_ids]
    test_actual = [int(by_id[i]["cultivar"]) for i in test_ids]

    # ------------------------------------------------------------- refit and re-derive
    fitted = {
        "majority": (make_pipeline(StandardScaler(), DummyClassifier(strategy="prior")), two),
        "linear_two": (make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000)), two),
        "forest_two": (RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_leaf=3,
                                              random_state=21, n_jobs=1), two),
        "linear_three": (make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000)), three),
    }
    derived = {}
    for key, (model, matrix) in fitted.items():
        model.fit(matrix[train_index], labels[train_index])
        derived[key] = {"model": model, "matrix": matrix}

    def independent_probabilities(key, index_set):
        """Probabilities recomputed from the fitted state, without predict_proba."""
        model = derived[key]["model"]
        matrix = derived[key]["matrix"]
        rows = matrix[index_set]
        if key == "majority":
            # A prior-strategy dummy IS the training class frequencies.
            counts = [sum(1 for value in labels[train_index] if value == c) for c in CLASSES]
            total = sum(counts)
            prior = [count / total for count in counts]
            return [list(prior) for _ in range(len(rows))]
        if key == "forest_two":
            return tree_probabilities(model, rows)
        # A standardising logistic pipeline: standardise here, dot here, softmax here.
        training = matrix[train_index]
        means = [float(np.mean(training[:, column])) for column in range(training.shape[1])]
        deviations = [float(math.sqrt(np.mean((training[:, column] - means[column]) ** 2)))
                      for column in range(training.shape[1])]
        classifier = model[-1]
        weights = classifier.coef_
        intercepts = classifier.intercept_
        out = []
        for sample in rows:
            standardised = [(float(sample[column]) - means[column]) / deviations[column]
                            for column in range(len(means))]
            scores = [float(sum(weights[k][column] * standardised[column] for column in range(len(means)))
                            + intercepts[k]) for k in CLASSES]
            out.append(softmax(scores))
        return out

    # The scaler this lesson teaches: the training mean and population standard
    # deviation. Checking that against the fitted scaler's stored state is what
    # makes the standardisation above an independent route rather than a guess.
    for key in ("linear_two", "linear_three"):
        model = derived[key]["model"]
        matrix = derived[key]["matrix"]
        training = matrix[train_index]
        scaler = model[0]
        for column in range(training.shape[1]):
            mean_here = float(np.mean(training[:, column]))
            deviation_here = float(math.sqrt(np.mean((training[:, column] - mean_here) ** 2)))
            check(abs(mean_here - float(scaler.mean_[column])) < 1e-12,
                  f"{key}: the scaler's stored mean for column {column} is the training mean computed here")
            check(abs(deviation_here - float(scaler.scale_[column])) < 1e-12,
                  f"{key}: the scaler's stored scale for column {column} is the training population "
                  "standard deviation computed here")

    candidate_records = {}
    for key in MODEL_KEYS:
        block = packet["candidates"][key]
        probabilities = independent_probabilities(key, valid_index)
        predictions = [argmax(row) for row in probabilities]
        for row_index, row in enumerate(probabilities):
            for class_index in CLASSES:
                compare(f"/candidates/{key}/validationProbability/{row_index}/{class_index}",
                        row[class_index], block["validationProbability"][row_index][class_index], 5e-9)
            check(abs(sum(row) - 1.0) < 1e-12,
                  f"{key}: validation probability row {row_index} sums to one")
        for row_index, value in enumerate(predictions):
            compare(f"/candidates/{key}/validationPrediction/{row_index}",
                    value, block["validationPrediction"][row_index])

        matrix = confusion_matrix(valid_actual, block["validationPrediction"])
        for row_index in range(3):
            for column in range(3):
                compare(f"/candidates/{key}/validationConfusion/{row_index}/{column}",
                        matrix[row_index][column], block["validationConfusion"][row_index][column])
        check(sum(sum(row) for row in matrix) == 36, f"{key}: the validation confusion matrix totals 36")
        compare(f"/candidates/{key}/validationAccuracy",
                float(accuracy(valid_actual, block["validationPrediction"])), block["validationAccuracy"], 1e-12)
        compare(f"/candidates/{key}/validationBalancedAccuracy",
                float(balanced_accuracy(valid_actual, block["validationPrediction"])),
                block["validationBalancedAccuracy"], 1e-12)
        compare(f"/candidates/{key}/validationLogLoss",
                log_loss(valid_actual, block["validationProbability"]), block["validationLogLoss"], 1e-9)
        # The training score, which the page shows as a training quantity and
        # never as evidence about anything else.
        training_probabilities = independent_probabilities(key, train_index)
        training_predictions = [argmax(row) for row in training_probabilities]
        compare(f"/candidates/{key}/trainBalancedAccuracy",
                float(balanced_accuracy(train_actual, training_predictions)),
                block["trainBalancedAccuracy"], 1e-9)
        candidate_records[key] = {
            "trainingBalancedAccuracy": float(balanced_accuracy(train_actual, training_predictions)),
            "validationBalancedAccuracy": float(balanced_accuracy(valid_actual, block["validationPrediction"])),
            "validationAccuracy": float(accuracy(valid_actual, block["validationPrediction"])),
            "validationLogLoss": log_loss(valid_actual, block["validationProbability"]),
            "validationConfusion": matrix,
            "validationPredictions": list(block["validationPrediction"]),
            "validationProbabilities": [list(row) for row in block["validationProbability"]],
            "validationCorrect": sum(1 for a, p in zip(valid_actual, block["validationPrediction"]) if a == p),
        }

    # ------------------------------------------------------ the per-specimen rows
    for row_index, row in enumerate(packet["validationRows"]):
        key = row["model"]
        # Derived from the row's POSITION, not read back out of the row. This
        # compared `row["model"]` with itself for all 144 rows -- 6.5% of the
        # "re-derived" leaf total were comparisons that could not fail. The
        # packet writes the rows as four blocks of 36 in declared model order,
        # so the position determines the key independently.
        expected_key = MODEL_KEYS[row_index // len(valid_ids)]
        position = valid_ids.index(row["id"])
        source = by_id[row["id"]]
        compare(f"/validationRows/{row_index}/model", expected_key, row["model"])
        compare(f"/validationRows/{row_index}/id", valid_ids[row_index % 36], row["id"])
        compare(f"/validationRows/{row_index}/actual", int(source["cultivar"]), row["actual"])
        compare(f"/validationRows/{row_index}/prediction",
                packet["candidates"][key]["validationPrediction"][position], row["prediction"])
        compare(f"/validationRows/{row_index}/alcohol", float(source["alcohol"]), row["alcohol"], 1e-12)
        compare(f"/validationRows/{row_index}/colorIntensity",
                float(source["color_intensity"]), row["colorIntensity"], 1e-12)
        compare(f"/validationRows/{row_index}/flavanoids", float(source["flavanoids"]), row["flavanoids"], 1e-12)
        compare(f"/validationRows/{row_index}/probabilityOfActual",
                packet["candidates"][key]["validationProbability"][position][int(source["cultivar"])],
                row["probabilityOfActual"], 1e-15)

    # The explorer's cutoff is held as an integer number of hundredths and turned
    # into a double by one division, so the browser and this verifier evaluate
    # `colour < cutoff` on bit-identical operands. The measurements themselves
    # are NOT all two-decimal: specimen 172 carries 9.899999, written by the
    # packet's %.8g export of a value that was never a round hundredth. An
    # earlier draft of this check assumed they were and failed on that one row,
    # which is why the cutoff rather than the measurement carries the encoding.
    off_grid = [row["specimen_id"] for row in table
                if round(float(row["color_intensity"]) * 100) / 100 != float(row["color_intensity"])]
    check(off_grid == ["172"],
          f"the measurements off the hundredths grid are {off_grid}, not the single known specimen 172")
    # The property that actually matters for this control, asserted directly.
    #
    # This was `round(h / 100 * 100) != h`, and `round` re-snaps every float to
    # the nearest integer, so the predicate was false for every input and the
    # guard passed for any encoding whatsoever -- including one that collapsed
    # every setting to zero. Removing `round` is not the fix either: `h/100*100
    # != h` is true for 143 of the 1,401 settings and is ordinary float
    # behaviour, not a defect. What the explorer needs is that the 1,401 cutoffs
    # are DISTINCT and ORDERED doubles, so that no two settings silently mean
    # the same cutoff and moving the control never moves the cutoff backwards.
    cutoffs = [hundredths / 100 for hundredths in range(0, 1401)]
    check(len(set(cutoffs)) == 1401,
          f"the 1401 cutoff settings collapse to {len(set(cutoffs))} distinct doubles")
    out_of_order = [index for index in range(1, len(cutoffs)) if not cutoffs[index] > cutoffs[index - 1]]
    check(not out_of_order,
          f"{len(out_of_order)} cutoff settings are not strictly greater than their predecessor")

    # --------------------------------------------------------- slices and pairing
    rows_by_model = {key: [row for row in packet["validationRows"] if row["model"] == key] for key in MODEL_KEYS}
    for key in ELIGIBLE:
        model_rows = rows_by_model[key]
        for label, predicate in (("color<4", lambda r: r["colorIntensity"] < 4),
                                 ("color>=4", lambda r: r["colorIntensity"] >= 4),
                                 ("class1", lambda r: r["actual"] == 1)):
            chosen = [r for r in model_rows if predicate(r)]
            compare(f"/slices/{key}/{label}/n", len(chosen), packet["slices"][key][label]["n"])
            compare(f"/slices/{key}/{label}/errors",
                    sum(1 for r in chosen if r["prediction"] != r["actual"]),
                    packet["slices"][key][label]["errors"])
            for index, identifier in enumerate(r["id"] for r in chosen):
                compare(f"/slices/{key}/{label}/ids/{index}", identifier,
                        packet["slices"][key][label]["ids"][index])
        lower = packet["slices"][key]["color<4"]["n"]
        upper = packet["slices"][key]["color>=4"]["n"]
        check(lower + upper == 36, f"{key}: the two colour slices partition the 36 validation specimens")
        check(not set(packet["slices"][key]["color<4"]["ids"]) & set(packet["slices"][key]["color>=4"]["ids"]),
              f"{key}: the two colour slices share no specimen")
    # The boundary matters: one validation specimen sits at colour intensity
    # exactly 4, and the >= side is the side that must contain it.
    at_boundary = [row["id"] for row in rows_by_model["linear_two"] if row["colorIntensity"] == 4]
    check(len(at_boundary) == 1,
          f"exactly one validation specimen sits exactly on the colour cutoff of 4, found {len(at_boundary)}")
    check(set(at_boundary) <= set(packet["slices"]["linear_two"]["color>=4"]["ids"]),
          "the specimen at exactly 4 belongs to the >= slice, not the < slice")

    reference_rows = rows_by_model["linear_two"]
    for key in ("forest_two", "linear_three"):
        candidate = rows_by_model[key]
        repaired = [a["id"] for a, b in zip(reference_rows, candidate)
                    if a["prediction"] != a["actual"] and b["prediction"] == b["actual"]]
        broken = [a["id"] for a, b in zip(reference_rows, candidate)
                  if a["prediction"] == a["actual"] and b["prediction"] != b["actual"]]
        compare(f"/paired/{key}/fixed", len(repaired), packet["paired"][key]["fixed"])
        compare(f"/paired/{key}/broken", len(broken), packet["paired"][key]["broken"])
        reference_correct = sum(1 for r in reference_rows if r["prediction"] == r["actual"])
        candidate_correct = sum(1 for r in candidate if r["prediction"] == r["actual"])
        check(candidate_correct - reference_correct == len(repaired) - len(broken),
              f"{key}: the change in correct count is repairs minus regressions")
    check(sum(1 for r in reference_rows if r["prediction"] == r["actual"]) == 29,
          "the two-feature linear model gets 29 of 36 validation specimens right")
    check(sum(1 for r in rows_by_model["linear_three"] if r["prediction"] == r["actual"]) == 32,
          "the three-feature linear model gets 32 of 36 validation specimens right")
    check(sum(1 for r in rows_by_model["forest_two"] if r["prediction"] == r["actual"]) == 30,
          "the two-feature forest gets 30 of 36 validation specimens right")

    # ------------------------------------------------------------ the frozen report
    selected = max(ELIGIBLE, key=lambda key: candidate_records[key]["validationBalancedAccuracy"])
    compare("/selected", selected, packet["selected"])
    model = derived[selected]["model"]
    matrix = derived[selected]["matrix"]
    test_probabilities = independent_probabilities(selected, test_index)
    test_predictions = [argmax(row) for row in test_probabilities]
    for index, value in enumerate(test_predictions):
        compare(f"/test/predictions/{index}", value, packet["test"]["predictions"][index])
    test_confusion = confusion_matrix(test_actual, packet["test"]["predictions"])
    for row_index in range(3):
        for column in range(3):
            compare(f"/test/confusion/{row_index}/{column}", test_confusion[row_index][column],
                    packet["test"]["confusion"][row_index][column])
    compare("/test/accuracy", float(accuracy(test_actual, packet["test"]["predictions"])),
            packet["test"]["accuracy"], 1e-12)
    compare("/test/balancedAccuracy", float(balanced_accuracy(test_actual, packet["test"]["predictions"])),
            packet["test"]["balancedAccuracy"], 1e-12)
    compare("/test/logLoss", log_loss(test_actual, test_probabilities), packet["test"]["logLoss"], 5e-9)
    check(sum(sum(row) for row in test_confusion) == 36, "the held-out confusion matrix totals 36")
    check(sum(test_confusion[i][i] for i in range(3)) == 35, "35 of the 36 held-out specimens are correct")

    # ------------------------------------------------------------ Wilson, by scalar
    k, n, z = 35, 36, 1.96
    proportion = k / n
    centre = (proportion + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * math.sqrt(proportion * (1 - proportion) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    compare("/wilsonIllustration/0", centre - radius, packet["wilsonIllustration"][0], 1e-12)
    compare("/wilsonIllustration/1", centre + radius, packet["wilsonIllustration"][1], 1e-12)
    check(centre - radius < proportion < centre + radius,
          "the Wilson interval contains the observed proportion")
    check(centre + radius < 1, "the Wilson interval stops short of 1 even at 35 of 36")

    # ------------------------------------------ the constructed acceptance fixture
    fixture = packet["deferralFixture"]
    scores = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
    outcomes = [True, True, True, True, True, False, True, True, False, True]
    for index, case in enumerate(fixture):
        compare(f"/deferralFixture/{index}/id", index + 1, case["id"])
        compare(f"/deferralFixture/{index}/confidence", scores[index], case["confidence"], 0.0)
        compare(f"/deferralFixture/{index}/correct", outcomes[index], case["correct"])
    check(sorted((case["confidence"] for case in fixture), reverse=True)
          == [case["confidence"] for case in fixture], "the fixture is listed in decreasing confidence")
    check(len({case["confidence"] for case in fixture}) == 10, "the ten fixture scores are distinct")
    for threshold, wrong_cost, defer_cost, expected in (
            (0.60, 10, 2, {"accepted": 8, "wrong": 1, "deferred": 2, "cost": 14}),
            (0.80, 10, 2, {"accepted": 4, "wrong": 0, "deferred": 6, "cost": 12}),
            (0.60, 2, 2, {"cost": 6}),
            (0.80, 2, 2, {"cost": 12}),
            (0.61, 10, 2, {"accepted": 7, "wrong": 1, "deferred": 3, "cost": 16}),
            (0.64, 10, 2, {"accepted": 7, "wrong": 1, "deferred": 3, "cost": 16}),
            (1.01, 10, 2, {"accepted": 0, "deferred": 10, "cost": 20, "conditionalError": None}),
    ):
        ledger = acceptance_ledger(fixture, threshold, wrong_cost, defer_cost)
        for field, value in expected.items():
            check(ledger[field] == value,
                  f"acceptance ledger at threshold {threshold}, costs {wrong_cost}/{defer_cost}: "
                  f"{field} is {ledger[field]!r}, not {value!r}")

    # --------------------------------------------------------- version provenance
    compare("/versions/numpy", np.__version__, packet["versions"]["numpy"])
    compare("/versions/sklearn", sklearn.__version__, packet["versions"]["sklearn"])

    # ------------------------------------------------------------------- coverage
    missing = sorted(all_leaves - seen_leaves)
    extra = sorted(seen_leaves - all_leaves)
    coverage = len(seen_leaves & all_leaves) / len(all_leaves) if all_leaves else 0.0
    check(not extra, f"{len(extra)} compared paths are not packet leaves: {extra[:5]}")
    check(not missing, f"{len(missing)} packet leaves were never compared: {missing[:8]}")

    # ------------------------------------------------------------- the data module
    def score(metric, value, role, note=None):
        record = {"metric": metric, "value": value, "role": role}
        if note:
            record["note"] = note
        return record

    validation_rows = []
    for identifier in valid_ids:
        source = by_id[identifier]
        position = valid_ids.index(identifier)
        validation_rows.append({
            "id": identifier,
            "actual": int(source["cultivar"]),
            "alcohol": float(source["alcohol"]),
            "colorIntensity": float(source["color_intensity"]),
            "flavanoids": float(source["flavanoids"]),
            "prediction": {key: candidate_records[key]["validationPredictions"][position] for key in MODEL_KEYS},
            "probabilityOfActual": {
                key: candidate_records[key]["validationProbabilities"][position][int(source["cultivar"])]
                for key in MODEL_KEYS},
        })

    labelled = {
        "majority": {"label": "Majority baseline", "short": "majority",
                     "family": "Most frequent training cultivar",
                     "features": ["alcohol", "color_intensity"],
                     "settings": "strategy=prior", "declaredEligible": False,
                     "purpose": "Does measuring anything help at all?"},
        "linear_two": {"label": "Two-feature logistic regression", "short": "linear_two",
                       "family": "Multinomial logistic regression on standardised inputs",
                       "features": ["alcohol", "color_intensity"],
                       "settings": "C=1, max_iter=2000", "declaredEligible": True,
                       "purpose": "How far do weighted sums of the two initial measurements separate the classes?"},
        "forest_two": {"label": "Two-feature random forest", "short": "forest_two",
                       "family": "Random forest on the same two raw inputs",
                       "features": ["alcohol", "color_intensity"],
                       "settings": "n_estimators=100, max_depth=4, min_samples_leaf=3, random_state=21",
                       "declaredEligible": True,
                       "purpose": "Do more flexible combinations of the same two measurements help?"},
        "linear_three": {"label": "Three-feature logistic regression", "short": "linear_three",
                         "family": "Multinomial logistic regression on standardised inputs",
                         "features": ["alcohol", "color_intensity", "flavanoids"],
                         "settings": "C=1, max_iter=2000", "declaredEligible": True,
                         "purpose": "Does an added measurement help more than a more flexible rule?"},
    }
    candidates = []
    for key in MODEL_KEYS:
        record = candidate_records[key]
        candidates.append({
            "key": key,
            **labelled[key],
            "trainingBalancedAccuracy": score("balanced accuracy", record["trainingBalancedAccuracy"], "training"),
            "validationBalancedAccuracy": score("balanced accuracy", record["validationBalancedAccuracy"],
                                                "validation"),
            "validationAccuracy": score("accuracy", record["validationAccuracy"], "validation"),
            "validationLogLoss": score("log loss (nats)", record["validationLogLoss"], "validation"),
            "validationCorrect": record["validationCorrect"],
            "validationConfusion": record["validationConfusion"],
            "validationRecalls": [
                Fraction(record["validationConfusion"][c][c],
                         sum(record["validationConfusion"][c])).__float__() for c in CLASSES],
            "validationSupport": [sum(record["validationConfusion"][c]) for c in CLASSES],
        })

    paired = {}
    for key in ("forest_two", "linear_three"):
        candidate = rows_by_model[key]
        paired[key] = {
            "repairedIds": [a["id"] for a, b in zip(reference_rows, candidate)
                            if a["prediction"] != a["actual"] and b["prediction"] == b["actual"]],
            "brokenIds": [a["id"] for a, b in zip(reference_rows, candidate)
                          if a["prediction"] == a["actual"] and b["prediction"] != b["actual"]],
            "referenceCorrect": sum(1 for r in reference_rows if r["prediction"] == r["actual"]),
            "candidateCorrect": sum(1 for r in candidate if r["prediction"] == r["actual"]),
        }

    payload = {
        "provenance": {
            "file": "/learn-assets/end-to-end/wine.csv",
            "attribution": "/learn-assets/end-to-end/ATTRIBUTION.txt",
            "program": "/learn-assets/end-to-end/wine_study.py",
            "name": "Wine",
            "creator": "Stefan Aeberhard and M. Forina",
            "record": "https://archive.ics.uci.edu/dataset/109/wine",
            "doi": "https://doi.org/10.24432/C5PC7J",
            "license": "CC BY 4.0",
            "licenseUrl": "https://creativecommons.org/licenses/by/4.0/",
            "retrieved": "2026-09-12",
            "sha256": asset_digest,
            "bytes": len(asset_bytes),
            "rows": len(table),
            "measurementColumns": 13,
            "classCounts": [sum(1 for row in table if int(row["cultivar"]) == c) for c in CLASSES],
            "offlineSource": "the bundled data of scikit-learn 1.9.1 load_wine()",
            "unitsNote": "The inspected UCI table does not state a unit for each measurement, so the figures "
                         "use the recorded source scale and name no unit.",
        },
        "contract": {
            "predictionUnit": "One measured specimen",
            "target": "Its recorded cultivar, among three known classes",
            "initialInputs": ["alcohol", "color_intensity"],
            "candidateAddedInput": "flavanoids",
            "primaryMetric": "validation balanced accuracy",
            "primaryMetricKey": "validationBalancedAccuracy",
            "tieRule": "Ties are broken by the declared candidate order.",
            "trainRows": len(train_ids),
            "validationRows": len(valid_ids),
            "testRows": len(test_ids),
            "finalProtocol": "Evaluate the selected already-fitted model; do not refit on validation.",
        },
        "roles": {
            "training": "Computed on the 106 fitting rows. It describes the fit, not the next specimen.",
            "validation": "Computed on the 36 development rows. It is the evidence this study is allowed to "
                          "inspect and to choose with.",
            "selection": "A validation quantity in its role as the declared choice criterion. Reading it to "
                         "choose is what makes it selection evidence rather than an independent estimate.",
            "held-out": "Computed on the 36 test rows, once, after the candidate was frozen.",
        },
        "splits": {"train": train_ids, "validation": valid_ids, "test": test_ids},
        "classCounts": {name: class_counts[name] for name in ("train", "validation", "test")},
        "candidates": candidates,
        "validationRows": validation_rows,
        "paired": paired,
        "colourSliceReference": {
            "cutoff": 4,
            "note": "An explicit exploratory cutoff, not a known biological boundary.",
            "slices": {key: {label: packet["slices"][key][label] for label in ("color<4", "color>=4", "class1")}
                       for key in ELIGIBLE},
        },
        "heldOut": {
            "selected": selected,
            "accuracy": score("accuracy", packet["test"]["accuracy"], "held-out"),
            "balancedAccuracy": score("balanced accuracy", packet["test"]["balancedAccuracy"], "held-out"),
            "logLoss": score("log loss (nats)", packet["test"]["logLoss"], "held-out"),
            "confusion": packet["test"]["confusion"],
            "correct": sum(test_confusion[i][i] for i in range(3)),
            "total": 36,
            "predictions": list(packet["test"]["predictions"]),
            "ids": test_ids,
            "actual": test_actual,
            "wilson": {"successes": k, "trials": n, "z": z,
                       "lower": centre - radius, "upper": centre + radius,
                       "note": "A binomial scale check on the accuracy estimate under an independent "
                               "common-probability model, not an interval for balanced accuracy and not "
                               "matched to this stratified design."},
        },
        "deferralFixture": [{"id": case["id"], "confidence": case["confidence"], "correct": case["correct"]}
                            for case in fixture],
        "deferralDefaults": {"threshold": 0.6, "proposedThreshold": 0.8, "wrongCost": 10, "deferCost": 2},
        "practice": {
            "confusion": [[8, 2, 0], [1, 5, 0], [0, 2, 2]],
            "accuracy": 0.75,
            "balancedAccuracyNumerator": 32,
            "balancedAccuracyDenominator": 45,
        },
        "versions": {"numpy": packet["versions"]["numpy"], "sklearn": packet["versions"]["sklearn"],
                     "python": "3.12.14"},
    }

    header = (
        "// Recorded data for the end-to-end supervised learning and error analysis lesson.\n"
        "//\n"
        "// Generated by scripts/verify-endtoend-data.py from the frozen content packet and\n"
        "// the copy of the Wine table this lesson serves. Every score in this file carries\n"
        "// the role that says what it is evidence about:\n"
        "//\n"
        "//   training    computed on the 106 fitting rows\n"
        "//   validation  computed on the 36 development rows\n"
        "//   selection   a validation quantity in its role as the declared choice criterion\n"
        "//   held-out    computed once on the 36 test rows, after the candidate was frozen\n"
        "//\n"
        "// The held-out block is the one the page must not show before the learner has\n"
        "// committed the decision it reports on. Importing this module does not display it.\n"
        "//\n"
        "// Do not edit by hand.\n"
    )
    module_text = header + "export const endToEndData = " + json.dumps(payload, indent=2) + ";\n"

    if write:
        MODULE.write_text(module_text, encoding="utf-8", newline="\n")
    else:
        existing = MODULE.read_text(encoding="utf-8") if MODULE.exists() else None
        check(existing == module_text,
              "src/learn/data/endtoend-data.js is not byte-identical to a fresh regeneration from the "
              "served dataset; re-run with --write and read the difference")

    empty_containers = sum(1 for path in all_leaves if path.endswith("/[]") or path.endswith("/{}"))
    evidence = json.dumps({
        "checkedAt": datetime.now(timezone.utc).isoformat(),
        "verifier": "scripts/verify-endtoend-data.py",
        "verifierSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "packetResultsSha256": hashlib.sha256(PACKET_RESULTS.read_bytes()).hexdigest(),
        "served": {
            "dataset": "/learn-assets/end-to-end/wine.csv",
            "sha256": asset_digest,
            "bytes": len(asset_bytes),
            "attribution": "/learn-assets/end-to-end/ATTRIBUTION.txt",
        },
        "module": {
            "path": "src/learn/data/endtoend-data.js",
            "sha256": hashlib.sha256(module_text.encode("utf-8")).hexdigest(),
            "regeneration": "written" if write else "byte-identical",
        },
        "trustRootScalarLeaves": len(all_leaves),
        "trustRootLeavesReDerived": len(seen_leaves & all_leaves),
        "trustRootCoverage": round(len(seen_leaves & all_leaves) / len(all_leaves), 6) if all_leaves else 0.0,
        "trustRootUncoveredPaths": missing,
        "trustRootLeavesValidatedNotReDerived": sorted(validated_not_rederived),
        "trustRootEmptyContainers": empty_containers,
        "checks": checks,
        "scope": "Every scalar leaf of the content packet's calculated-inputs.json is recomputed and compared. "
                 "Accuracy, balanced accuracy and the confusion matrices are written out from their definitions "
                 "in exact rational arithmetic rather than obtained from scikit-learn's metrics; log loss is the "
                 "plain mean of -log p(actual); the majority baseline's probabilities come from the training "
                 "class counts; the two logistic pipelines' probabilities are rebuilt by standardising with a "
                 "mean and population standard deviation computed here, applying the coefficients as an explicit "
                 "dot product and writing out the softmax; the forest's probabilities are rebuilt by walking all "
                 "100 fitted trees with an explicit decision-path loop; the slices, paired repair and regression "
                 "counts and corrected counts come from the per-specimen rows; the Wilson interval is written "
                 "out from the scalar formula; and the acceptance fixture's ledger is recomputed by exact "
                 "enumeration at the seven contrast and null settings the specification names.",
        "limitations": [
            "The three split index lists are reproduced by the same two stated train_test_split calls, because "
            "the meaning of random_state=21 is scikit-learn's own stream. They are marked as validated rather "
            "than re-derived. The properties the lesson relies on -- exact partition of 1..178, pairwise "
            "disjointness, sizes 106/36/36 and preserved cultivar representation -- are checked independently.",
            "The fitted coefficients, intercepts and tree structures are obtained by refitting with the same "
            "estimators, because the estimators are the object under study. Everything computed from them here "
            "is independent of scikit-learn's prediction code.",
            "Floating-point results can differ on other library versions; the resolved versions are recorded.",
            "This verifier checks numbers and data. Rendering, layout, interaction and the held-out gate's "
            "behaviour in a browser are separate steps and are not claimed here.",
        ],
        "passed": not problems,
    }, indent=2) + "\n"
    if keep_evidence:
        EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
        EVIDENCE.write_text(evidence, encoding="utf-8", newline="\n")

    # A headline count nobody floors can shrink silently. The leaf-coverage
    # assertion binds every `compare()` call but none of the ~60 `check()` calls.
    check(checks >= 2400, f"only {checks} data checks ran; the suite has lost coverage")

    if problems:
        for problem in problems[:40]:
            print(f"FAIL {problem}")
        raise SystemExit(f"{len(problems)} of {checks} data checks failed")

    coverage = len(seen_leaves & all_leaves) / len(all_leaves)
    print(f"PASS: {checks:,} checks; {coverage:.1%} of the trust root's {len(all_leaves):,} leaves re-derived "
          f"({len(validated_not_rederived)} split leaves validated rather than re-derived); "
          f"module {'written' if write else 'byte-identical'}.")


if __name__ == "__main__":
    main()
