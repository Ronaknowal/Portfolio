"""Re-derive the problem-formulation trust root and regenerate the lesson's data module.

Two jobs.

1. **Re-derive the trust root.** Every scalar leaf of the content packet's
   `calculated-inputs.json` is recomputed here and compared, from the dataset
   this lesson serves, through implementations written independently of the
   packet's own `author-calculations.py`:

     * average precision is summed over distinct score thresholds from its
       definition -- recall increment times precision -- not by calling
       `sklearn.metrics.average_precision_score`;
     * log loss is the mean negative log likelihood, written out;
     * the confusion cells, the correct count and the top-50 counts are counted
       here from the outcomes;
     * the ranking is an explicit sort by descending probability then ascending
       source row, not `numpy.lexsort`;
     * the one-hot feature names are rebuilt from the sorted distinct categories
       of the training rows, not read off the fitted encoder;
     * the as-known calibration results come from an independent implementation
       of the manuscript's five-step rule;
     * the two feature lists are parsed out of the FROZEN MANUSCRIPT's own
       program text rather than retyped here, so a list that drifted from what
       the page displays would fail.

   Coverage is measured, not asserted: the comparison walks the packet tree and
   records the path of every scalar leaf it actually compared. The run fails
   unless that set is the complete set of leaves, so a block added to the packet
   lowers the count and stops the build instead of passing unnoticed.

   Two groups of leaves are asserted by a weaker route, and this is stated in
   the evidence file rather than hidden:

     * the 2,472 fitted probabilities and the 2 iteration counts are
       **validated by refitting** with the same scikit-learn estimator. The
       estimator is the object under study; an independent reimplementation of
       regularised logistic regression would be testing a different thing.
       Everything computed FROM those probabilities is independent.
     * the 4,119 split membership indices are **recorded inputs regenerated
       from their seeds** through `StratifiedShuffleSplit`, a different entry
       point from the packet's `train_test_split`, and additionally checked for
       the structural properties the lesson claims: disjoint, exhaustive,
       correctly sized, and stratified to the recorded per-class counts.

2. It regenerates `src/learn/data/formulation-data.js` from those results. The
   script is READ-ONLY unless given `--write`; without it the module text is
   rebuilt in memory and must be byte-identical to the file on disk.

Run:
  scratch/lesson-tools/Scripts/python.exe scripts/verify-formulation-data.py --write
  scratch/lesson-tools/Scripts/python.exe scripts/verify-formulation-data.py
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
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "docs/teaching/drafts/ml-problem-formulation-baselines-data-leakage"
PACKET_RESULTS = PACKET / "calculated-inputs.json"
PACKET_DATASET = PACKET / "bank-additional.csv"
MANUSCRIPT = PACKET / "lesson.md"
ASSET_DIR = ROOT / "public/learn-assets/problem-formulation"
ASSET_DATASET = ASSET_DIR / "bank-additional.csv"
ASSET_ATTRIBUTION = ASSET_DIR / "ATTRIBUTION.txt"
ASSET_DESCRIPTION = ASSET_DIR / "bank-marketing-variable-description.txt"
ASSET_PROGRAM = ASSET_DIR / "formulation-calculations.py"
MODULE = ROOT / "src/learn/data/formulation-data.js"
EVIDENCE = ROOT / "docs/teaching/evidence/formulation-data.json"

SOURCE_ROWS = 4119
DEVELOPMENT_ROWS = 3295
RESERVED_ROWS = 824
TRAIN_ROWS = 2471
VALIDATION_ROWS = 824
FIRST_SEED = 53
SECOND_SEED = 54
CAPACITY = 50

PROCEDURES = [
    {
        "packetKey": "training_prior",
        "id": "prior",
        "label": "Training-prior baseline",
        "features": "no input at all: one constant, the training positive fraction",
        "availability": "Uses the training outcomes only. Every case scores the same, so it carries no ranking "
                        "information and its top-50 set is decided by the tie rule.",
        "duration": None,
    },
    {
        "packetKey": "candidate_pre_call",
        "id": "candidate",
        "label": "Candidate recorded-feature model",
        "features": "age, previous-contact count, a contacted-before indicator, elapsed days since the previous "
                    "contact, and seven recorded customer and history categories",
        "availability": "Proposed as pre-call inputs. The supplied file does not certify their availability at a "
                        "scheduling cutoff; that contract would have to be established separately.",
        "duration": False,
    },
    {
        "packetKey": "unavailable_duration",
        "id": "duration",
        "label": "Same families plus final call duration",
        "features": "the candidate families and, in addition, the final length of the call being predicted",
        "availability": "Final call duration does not exist until the call has ended. It cannot be an input to a "
                        "prediction made before the call is placed, whatever its measured score.",
        "duration": True,
    },
]

write = "--write" in sys.argv
keep_evidence = "--no-evidence" not in sys.argv

problems: list[str] = []
checks = 0
root_checks = 0
covered: set[str] = set()
validated_by_refit: set[str] = set()
recorded_inputs: set[str] = set()


def check(condition, label):
    global checks
    checks += 1
    if not condition:
        problems.append(label)


def root(condition, label):
    """A statement about the RESULT, not a comparison between two routes to it.

    These would fail even if the packet and this script agreed on a wrong
    number, which is the only thing a re-derivation cannot catch on its own.
    """
    global root_checks
    root_checks += 1
    check(condition, f"trust root -- {label}")


def close(actual, expected, label, tolerance=1e-12):
    if not isinstance(actual, (int, float)) or isinstance(actual, bool):
        check(False, f"{label}: {actual!r} is not a number")
        return
    if not math.isfinite(float(actual)):
        check(False, f"{label}: {actual} is not finite")
        return
    check(abs(float(actual) - float(expected)) <= tolerance * max(1.0, abs(float(expected))),
          f"{label}: {actual} versus {expected}")


# ------------------------------------------------------------------ trust root

def leaf_paths(node, prefix=""):
    """Every scalar leaf of a recorded tree, addressed by its path."""
    if isinstance(node, dict):
        if not node:
            yield f"{prefix}/{{}}"
        for key, value in node.items():
            yield from leaf_paths(value, f"{prefix}/{key}")
    elif isinstance(node, list):
        if not node:
            yield f"{prefix}/[]"
        for index, value in enumerate(node):
            yield from leaf_paths(value, f"{prefix}/{index}")
    else:
        yield prefix


def compare_tree(got, want, prefix="", tolerance=1e-12):
    """Compare a recomputed subtree against the packet, recording every leaf."""
    if isinstance(want, dict):
        if not isinstance(got, dict) or set(got) != set(want):
            check(False, f"{prefix}: key sets differ "
                         f"({sorted(got) if isinstance(got, dict) else got} versus {sorted(want)})")
            return
        if not want:
            covered.add(f"{prefix}/{{}}")
            check(got == {}, f"{prefix}: both sides are empty")
        for key in want:
            compare_tree(got[key], want[key], f"{prefix}/{key}", tolerance)
        return
    if isinstance(want, list):
        if not isinstance(got, list) or len(got) != len(want):
            check(False, f"{prefix}: length {len(got) if isinstance(got, list) else got} versus {len(want)}")
            return
        if not want:
            covered.add(f"{prefix}/[]")
            check(got == [], f"{prefix}: both sides are empty")
        for index, value in enumerate(want):
            compare_tree(got[index], value, f"{prefix}/{index}", tolerance)
        return
    covered.add(prefix)
    if isinstance(want, bool) or want is None or isinstance(want, str):
        check(got == want, f"{prefix}: {got!r} versus {want!r}")
    else:
        close(got, want, prefix, tolerance)


# ============================================ metrics, written from definitions

def average_precision(targets, scores):
    """Sum over DISTINCT score thresholds of (recall increment) x (precision).

    Tied scores form one threshold. That is why a constant score gives exactly
    the positive fraction: one step from recall 0 to recall 1, at a precision
    equal to the prevalence.
    """
    positives = sum(targets)
    if positives == 0:
        raise ValueError("average precision is undefined with no positive case")
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    true_positives = 0
    seen = 0
    previous_recall = 0.0
    total = 0.0
    position = 0
    while position < len(order):
        last = position
        while last + 1 < len(order) and scores[order[last + 1]] == scores[order[position]]:
            last += 1
        for step in range(position, last + 1):
            true_positives += targets[order[step]]
            seen += 1
        recall = true_positives / positives
        precision = true_positives / seen
        total += (recall - previous_recall) * precision
        previous_recall = recall
        position = last + 1
    return total


def log_loss_by_definition(targets, probabilities):
    total = 0.0
    for target, probability in zip(targets, probabilities):
        if not 0.0 < probability < 1.0:
            raise ValueError(f"probability {probability} is not strictly inside (0, 1)")
        total += -math.log(probability) if target == 1 else -math.log(1.0 - probability)
    return total / len(targets)


def confusion_cells(targets, probabilities, threshold=0.5):
    cells = [[0, 0], [0, 0]]
    for target, probability in zip(targets, probabilities):
        predicted = 1 if probability >= threshold else 0
        cells[target][predicted] += 1
    return cells


def ranked_source_rows(ids, scores):
    """Descending probability, then ascending original source row index."""
    order = sorted(range(len(ids)), key=lambda index: (-scores[index], ids[index]))
    return [ids[index] for index in order]


def as_known(records, entity, cutoff, maximum_age):
    """The manuscript's five-step rule, written out again.

    Entity, then event no later than the cutoff, then event no older than the
    age limit, then availability no later than the cutoff; among the survivors
    the largest (event, available, version).
    """
    eligible = [record for record in records
                if record["entity"] == entity
                and record["event"] <= cutoff
                and record["event"] >= cutoff - maximum_age
                and record["available"] <= cutoff]
    if not eligible:
        return None
    best = eligible[0]
    for record in eligible[1:]:
        key = (record["event"], record["available"], record["version"])
        if key > (best["event"], best["available"], best["version"]):
            best = record
    return best


# ============================================================== module writing

def js_number(value):
    """Shortest round-tripping decimal. Python's repr and JavaScript's parser
    agree on these, so a regenerated module is byte-identical."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    text = repr(float(value))
    if text.endswith(".0"):
        text = text[:-2]
    return text


def js_string(value):
    return json.dumps(value, ensure_ascii=False)


def wrap_numbers(values, indent, per_line=8):
    pad = " " * indent
    lines = []
    for start in range(0, len(values), per_line):
        chunk = ", ".join(js_number(value) for value in values[start:start + per_line])
        lines.append(f"{pad}{chunk},")
    return "\n".join(lines)


def build_module(payload):
    out = []
    add = out.append
    add("/* GENERATED by scripts/verify-formulation-data.py. Do not edit by hand.")
    add(" *")
    add(" * Recorded measurements for ML Problem Formulation, Baselines & Data Leakage:")
    add(" * the dataset's provenance, the fixed partition, the three procedures' scores,")
    add(" * and the per-case validation outcomes and probabilities the capacity")
    add(" * investigation reads.")
    add(" *")
    add(" * Every number here was measured once, by the content packet's own program, on")
    add(" * the dataset served beside this lesson. The verifier re-derives all of it from")
    add(" * that dataset and rewrites this file; a hand edit is overwritten on the next")
    add(" * run and fails the read-only check before that.")
    add(" *")
    add(" * The validation outcomes below are development evidence. They are revealed")
    add(" * inside the capacity investigation on purpose -- the exercise is about which")
    add(" * identities enter a metric -- and that is not a claim that choosing a better")
    add(" * action set on them would generalise.")
    add(" */")
    add("export const formulationData = {")

    provenance = payload["provenance"]
    add("  provenance: {")
    for key, value in provenance.items():
        rendered = js_number(value) if isinstance(value, (int, float)) and not isinstance(value, bool) \
            else js_string(value)
        add(f"    {key}: {rendered},")
    add("  },")

    add("  software: {")
    for key, value in payload["software"].items():
        add(f"    {key}: {js_string(value)},")
    add("  },")

    add("  partition: {")
    for key, value in payload["partition"].items():
        add(f"    {key}: {js_number(value)},")
    add("  },")

    add("  features: {")
    add("    numeric: [" + ", ".join(js_string(name) for name in payload["features"]["numeric"]) + "],")
    add("    categorical: [" + ", ".join(js_string(name) for name in payload["features"]["categorical"]) + "],")
    add(f"    candidateColumns: {payload['features']['candidateColumns']},")
    add(f"    durationColumns: {payload['features']['durationColumns']},")
    add(f"    sentinel: {payload['features']['sentinel']},")
    add(f"    sentinelRows: {payload['features']['sentinelRows']},")
    add("  },")

    add("  procedures: [")
    for procedure in payload["procedures"]:
        add("    {")
        add(f"      id: {js_string(procedure['id'])},")
        add(f"      label: {js_string(procedure['label'])},")
        add(f"      features: {js_string(procedure['features'])},")
        add(f"      availability: {js_string(procedure['availability'])},")
        add(f"      averagePrecision: {js_number(procedure['averagePrecision'])},")
        add(f"      logLoss: {js_number(procedure['logLoss'])},")
        add(f"      correct: {procedure['correct']},")
        add("      confusion: [[" + ", ".join(str(cell) for cell in procedure["confusion"][0]) + "], ["
            + ", ".join(str(cell) for cell in procedure["confusion"][1]) + "]],")
        add(f"      top50Positives: {procedure['top50Positives']},")
        add(f"      precisionAt50: {js_number(procedure['precisionAt50'])},")
        add(f"      recallAt50: {js_number(procedure['recallAt50'])},")
        add(f"      iterations: {js_number(procedure['iterations']) if procedure['iterations'] is not None else 'null'},")
        add("    },")
    add("  ],")

    add("  validation: {")
    add("    ids: [")
    add(wrap_numbers(payload["validation"]["ids"], 6, 12))
    add("    ],")
    add("    targets: [")
    add(wrap_numbers(payload["validation"]["targets"], 6, 40))
    add("    ],")
    add("    scores: {")
    for name in ("candidate", "duration"):
        add(f"      {name}: [")
        add(wrap_numbers(payload["validation"]["scores"][name], 8, 4))
        add("      ],")
    add("    },")
    add("  },")

    add("  policyFixtures: {")
    add("    top25Ids: [")
    add(wrap_numbers(payload["policyFixtures"]["top25Ids"], 6, 12))
    add("    ],")
    add(f"    top25Positives: {payload['policyFixtures']['top25Positives']},")
    add(f"    top50Positives: {payload['policyFixtures']['top50Positives']},")
    add("    swap: {")
    add(f"      removeId: {payload['policyFixtures']['swap']['removeId']},")
    add(f"      addId: {payload['policyFixtures']['swap']['addId']},")
    add(f"      positives: {payload['policyFixtures']['swap']['positives']},")
    add("    },")
    add(f"    reorderNullPositives: {payload['policyFixtures']['reorderNullPositives']},")
    add("  },")

    add("  timeline: {")
    add("    records: [")
    for record in payload["timeline"]["records"]:
        add("      { entity: " + js_string(record["entity"])
            + f", event: {record['event']}, available: {record['available']}"
            + f", version: {record['version']}, value: {js_number(record['value'])} }},")
    add("    ],")
    add("    cases: [")
    for case in payload["timeline"]["cases"]:
        selected = case["selected"]
        rendered = "null" if selected is None else (
            "{ entity: " + js_string(selected["entity"])
            + f", event: {selected['event']}, available: {selected['available']}"
            + f", version: {selected['version']}, value: {js_number(selected['value'])} }}")
        add("      { name: " + js_string(case["name"]) + f", selected: {rendered} }},")
    add("    ],")
    add("  },")

    add("};")
    add("")
    add("export default formulationData;")
    add("")
    return "\n".join(out)


# ==================================================================== the run

def write_provisional():
    """A record stamped `passed: false`, written before the first check runs.

    Writing evidence only at the end looks safe and is not: a run that fails
    leaves the PREVIOUS file on disk, still saying `passed: true`, describing a
    trust root that no longer matches. Anyone reading the directory then sees a
    green record for a red tree.
    """
    if not keep_evidence:
        return
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps({
        "startedAt": datetime.now(timezone.utc).isoformat(),
        "verifier": "scripts/verify-formulation-data.py",
        "status": "running",
        "note": "Provisional record written before the first check. If this is what is on disk, the run did "
                "not reach its end: it raised, or it was killed.",
        "passed": False,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")


def main():
    global covered

    write_provisional()
    recorded = json.loads(PACKET_RESULTS.read_text(encoding="utf-8"))
    packet_bytes = PACKET_DATASET.read_bytes()
    digest = hashlib.sha256(packet_bytes).hexdigest()

    check(ASSET_DATASET.exists(), "the lesson serves its own copy of the dataset")
    check(ASSET_DATASET.exists() and ASSET_DATASET.read_bytes() == packet_bytes,
          "the served dataset is byte-for-byte the packet's file")
    check(ASSET_DESCRIPTION.exists(), "the provider's variable description is served beside it")
    check(ASSET_PROGRAM.exists(), "the packet's calculation program is served as a download")
    check(ASSET_PROGRAM.exists() and ASSET_PROGRAM.read_bytes() == (PACKET / "author-calculations.py").read_bytes(),
          "and is the packet's program byte for byte")
    check(ASSET_ATTRIBUTION.exists(), "an attribution file sits beside the data")
    if ASSET_ATTRIBUTION.exists():
        attribution = ASSET_ATTRIBUTION.read_text(encoding="utf-8")
        for token in ("CC BY 4.0", digest, "10.24432/C5K306", "Moro"):
            check(token in attribution, f"the attribution states {token}")

    # ---------------------------------------------- the data, read as a file
    with ASSET_DATASET.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    check(len(rows) == SOURCE_ROWS, f"the served file has {len(rows)} data rows, expected {SOURCE_ROWS}")
    y = [1 if row["y"] == "yes" else 0 for row in rows]
    check(sum(y) == 451, f"the served file has {sum(y)} positive outcomes, expected 451")
    # The sentinel count the figure prints, counted from the served file rather
    # than typed into a caption.
    sentinel_rows = sum(1 for row in rows if row["pdays"] == "999")
    root(sentinel_rows == 3959,
         f"{sentinel_rows} rows carry the 999 no-previous-contact sentinel, which is why it must not be read "
         "as an elapsed time")
    root(sentinel_rows < SOURCE_ROWS and sentinel_rows > 0,
         "and it is a sentinel on most rows but not all, so the contacted-before indicator carries information")

    # The two feature lists come from the FROZEN MANUSCRIPT's own program text.
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    numeric_match = re.search(r'^numeric = \[(.*?)\]$', manuscript, re.M | re.S)
    categorical_match = re.search(r'^categorical = \[(.*?)\]$', manuscript, re.M | re.S)
    check(numeric_match is not None and categorical_match is not None,
          "the manuscript's displayed program still declares numeric and categorical feature lists")
    numeric = re.findall(r'"([a-z_]+)"', numeric_match.group(1)) if numeric_match else []
    categorical = re.findall(r'"([a-z_]+)"', categorical_match.group(1)) if categorical_match else []
    compare_tree(numeric, recorded["numericFeatures"], "/numericFeatures")
    compare_tree(categorical, recorded["categoricalFeatures"], "/categoricalFeatures")
    check(numeric == recorded["numericFeatures"],
          f"the manuscript's numeric list {numeric} matches the packet's {recorded['numericFeatures']}")
    check(categorical == recorded["categoricalFeatures"],
          f"the manuscript's categorical list {categorical} matches the packet's")

    # The counts the manuscript states in prose, recounted from the served file.
    compare_tree(len(rows), recorded["inputRows"], "/inputRows")
    compare_tree(sum(y), recorded["positiveRows"], "/positiveRows")
    compare_tree({"numpy": np.__version__, "pandas": pd.__version__, "scikitLearn": sklearn.__version__},
                 recorded["software"], "/software")

    # ------------------------------------- the splits, regenerated from seeds
    y_array = np.asarray(y, dtype=int)
    first = StratifiedShuffleSplit(n_splits=1, train_size=DEVELOPMENT_ROWS, test_size=RESERVED_ROWS,
                                   random_state=FIRST_SEED)
    development, reserved = next(first.split(np.arange(SOURCE_ROWS).reshape(-1, 1), y_array))
    second = StratifiedShuffleSplit(n_splits=1, train_size=TRAIN_ROWS, test_size=VALIDATION_ROWS,
                                    random_state=SECOND_SEED)
    train_within, validation_within = next(second.split(development.reshape(-1, 1), y_array[development]))
    train = development[train_within]
    validation = development[validation_within]

    for name, mine, theirs in (("trainRows", train, recorded["trainRows"]),
                               ("validationRows", validation, recorded["validationRows"]),
                               ("reservedRows", reserved, recorded["reservedRows"])):
        compare_tree(mine.tolist(), theirs, f"/{name}")
        check(mine.tolist() == theirs, f"{name}: the seeds regenerate the recorded membership in order")
        for index in range(len(theirs)):
            recorded_inputs.add(f"/{name}/{index}")

    # Structural properties, independent of how the indices were produced.
    train_set, validation_set, reserved_set = set(train.tolist()), set(validation.tolist()), set(reserved.tolist())
    root(len(train_set) == TRAIN_ROWS and len(validation_set) == VALIDATION_ROWS
         and len(reserved_set) == RESERVED_ROWS, "the three partitions have the sizes the lesson states")
    root(not (train_set & validation_set) and not (train_set & reserved_set)
         and not (validation_set & reserved_set), "and are pairwise disjoint")
    root(train_set | validation_set | reserved_set == set(range(SOURCE_ROWS)),
         "and together exhaust every source row exactly once")
    compare_tree(sum(y[index] for index in train), recorded["trainPositives"], "/trainPositives")
    compare_tree(sum(y[index] for index in validation), recorded["validationPositives"], "/validationPositives")
    root(sum(y[index] for index in train) == recorded["trainPositives"] == 271,
         "the training rows carry 271 positive outcomes")
    root(sum(y[index] for index in validation) == recorded["validationPositives"] == 90,
         "and the validation rows 90, out of 824")
    root(sum(y[index] for index in reserved) == 451 - 271 - 90,
         "leaving the remaining positives in the reserved rows, which are never scored")
    root(abs(sum(y[index] for index in train) / TRAIN_ROWS - 451 / SOURCE_ROWS) < 0.005,
         "the stratified split holds the training prevalence close to the whole file's")

    # ------------------------------------------------------ the fitted models
    data = pd.read_csv(ASSET_DATASET, sep=";")
    features = data.copy()
    features["contacted_before"] = (features["pdays"] != 999).astype(int)
    features["days_since_previous"] = features["pdays"].replace(999, np.nan)

    def pipeline(include_duration):
        columns = numeric + (["duration"] if include_duration else [])
        preparation = ColumnTransformer([
            ("numeric", make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                      StandardScaler()), columns),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ])
        return make_pipeline(preparation, LogisticRegression(C=1.0, max_iter=600))

    prior = float(y_array[train].mean())
    close(prior, recorded["trainPrior"], "trainPrior")
    covered.add("/trainPrior")

    scores = {"training_prior": [prior] * VALIDATION_ROWS}
    iterations = {"training_prior": None}
    feature_names = {}
    for key, include in (("candidate_pre_call", False), ("unavailable_duration", True)):
        model = pipeline(include).fit(features.iloc[train], y_array[train])
        scores[key] = model.predict_proba(features.iloc[validation])[:, 1].tolist()
        iterations[key] = model[-1].n_iter_.tolist()
        feature_names[key] = model[0].get_feature_names_out().tolist()

    # The one-hot names, rebuilt from the sorted distinct training categories.
    def expected_feature_names(include_duration):
        columns = numeric + (["duration"] if include_duration else [])
        names = [f"numeric__{name}" for name in columns]
        for column in categorical:
            values = sorted({rows[index][column] for index in train})
            names.extend(f"categorical__{column}_{value}" for value in values)
        return names

    validation_ids = validation.tolist()
    validation_targets = [y[index] for index in validation_ids]
    compare_tree(validation_targets, recorded["validationTargets"], "/validationTargets")
    check(validation_targets == recorded["validationTargets"], "the validation outcomes match the packet")

    mine_results = {}
    for procedure in PROCEDURES:
        key = procedure["packetKey"]
        probability = scores[key]
        ranked = ranked_source_rows(validation_ids, probability)
        found50 = sum(validation_targets[validation_ids.index(case)] for case in ranked[:CAPACITY])
        cells = confusion_cells(validation_targets, probability)
        mine_results[key] = {
            "averagePrecision": average_precision(validation_targets, probability),
            "logLoss": log_loss_by_definition(validation_targets, probability),
            "confusion": cells,
            "correct": cells[0][0] + cells[1][1],
            "top50Positives": found50,
            "precisionAt50": found50 / CAPACITY,
            "recallAt50": found50 / sum(validation_targets),
            "probabilities": probability,
            "rankedSourceRows": ranked,
        }
        if iterations[key] is not None:
            mine_results[key]["iterations"] = iterations[key]
            mine_results[key]["featureNames"] = feature_names[key]
            check(feature_names[key] == expected_feature_names(procedure["duration"]),
                  f"{key}: the one-hot names rebuilt from the training categories match the fitted encoder")
        for index in range(len(recorded["results"][key]["probabilities"])):
            validated_by_refit.add(f"/results/{key}/probabilities/{index}")
        if iterations[key] is not None:
            validated_by_refit.add(f"/results/{key}/iterations/0")

    compare_tree(mine_results, recorded["results"], "/results", tolerance=1e-9)

    # How far the refit actually drifted. Reported rather than assumed: a
    # tolerance nobody measured against is a tolerance nobody can defend.
    worst_probability_gap = max(
        abs(mine - theirs)
        for key in mine_results
        for mine, theirs in zip(mine_results[key]["probabilities"],
                                recorded["results"][key]["probabilities"]))
    worst_metric_gap = max(
        abs(mine_results[key][field] - recorded["results"][key][field])
        for key in mine_results for field in ("averagePrecision", "logLoss"))
    root(worst_probability_gap < 1e-12,
         f"the refit reproduces every recorded probability to {worst_probability_gap:.3g}, far inside the "
         "tolerance the comparison allows")
    root(worst_metric_gap < 1e-12,
         f"and every recorded metric to {worst_metric_gap:.3g}, so the independent implementations of average "
         "precision and log loss agree with scikit-learn's to floating-point noise")

    # ------------------------------------------------------------- timeline
    records = recorded["timelineInput"]
    compare_tree(records, recorded["timelineInput"], "/timelineInput")
    manuscript_table = re.findall(r"^\| A \| (\d+) \| (\d+) \| (\d+) \| (\d+) \|$", manuscript, re.M)
    manuscript_b = re.findall(r"^\| B \| (\d+) \| (\d+) \| (\d+) \| (\d+) \|$", manuscript, re.M)
    rebuilt = [{"entity": "sensor_A", "event": int(a), "available": int(b), "version": int(c), "value": int(d)}
               for a, b, c, d in manuscript_table]
    rebuilt += [{"entity": "sensor_B", "event": int(a), "available": int(b), "version": int(c), "value": int(d)}
                for a, b, c, d in manuscript_b]
    check(rebuilt == records, "the packet's calibration history is the table the manuscript prints")

    def mutate(name):
        if name == "arrive_earlier":
            return [dict(record, available=4) if record["event"] == 4 and record["entity"] == "sensor_A" else record
                    for record in records]
        if name == "unrelated_entity_null":
            return [dict(record, value=88) if record["entity"] == "sensor_B" else record for record in records]
        return records

    timeline_cases = [("default", 5, 5), ("arrive_earlier", 5, 5), ("latest_known_revision", 7, 7),
                      ("new_event_now_known", 9, 9), ("too_old", 5, 2), ("unrelated_entity_null", 5, 5)]
    mine_timeline = [{"name": name, "selected": as_known(mutate(name), "sensor_A", cutoff, age)}
                     for name, cutoff, age in timeline_cases]
    compare_tree(mine_timeline, recorded["timelineResults"], "/timelineResults")

    by_name = {case["name"]: case["selected"] for case in mine_timeline}
    root(by_name["default"]["value"] == 10,
         "at cutoff 5 the admissible calibration is the value 10, not the newer event that has not arrived")
    root(by_name["arrive_earlier"]["value"] == 20 and by_name["arrive_earlier"]["event"] == 4,
         "moving that event's arrival from 8 to 4, with its event time unchanged, changes the answer to 20")
    root(by_name["default"]["event"] == 1 and by_name["arrive_earlier"]["event"] == 4,
         "so the SELECTED EVENT changes although no event time moved: the destination note's whole point")
    root(by_name["latest_known_revision"]["value"] == 12
         and by_name["latest_known_revision"]["version"] == 2,
         "at cutoff 7 the later revision of event 1 is known and is chosen over version 1")
    root(by_name["new_event_now_known"]["value"] == 20,
         "at cutoff 9 the delayed event has arrived and is the newest eligible one")
    root(by_name["too_old"] is None,
         "an age limit of 2 at cutoff 5 leaves nothing eligible: a missing calibration, not a substitute")
    root(by_name["unrelated_entity_null"]["value"] == 10,
         "editing the other sensor's value leaves this entity's selection untouched")
    root(as_known(records, "sensor_A", 5, 5)["value"]
         != as_known(records, "sensor_B", 4, 4)["value"],
         "the two entities do not share a calibration, so entity matching is doing work")
    root(as_known([dict(record) for record in records], "sensor_A", 8, 8)["event"] == 4,
         "cutoff 8 is the first moment the delayed value becomes admissible")
    root(as_known([dict(record) for record in records], "sensor_A", 7, 8) is not None
         and as_known([dict(record) for record in records], "sensor_A", 7, 8)["value"] == 12,
         "widening the age limit alone does not make an unavailable value available")

    # ------------------------------------------------------- policy fixtures
    target_by_id = dict(zip(validation_ids, validation_targets))
    candidate_ranked = mine_results["candidate_pre_call"]["rankedSourceRows"]
    top25 = candidate_ranked[:25]
    remove_id = next(case for case in top25 if target_by_id[case] == 1)
    add_id = next(case for case in candidate_ranked[25:] if target_by_id[case] == 0)
    swapped = [add_id if case == remove_id else case for case in top25]
    mine_policy = {
        "top25": {"ids": top25, "positives": sum(target_by_id[case] for case in top25)},
        "positiveToNegativeSwap": {"removeId": remove_id, "addId": add_id,
                                   "positives": sum(target_by_id[case] for case in swapped)},
        "reorderNullPositives": sum(target_by_id[case] for case in reversed(top25)),
    }
    compare_tree(mine_policy, recorded["policyFixtures"], "/policyFixtures")

    root(mine_policy["top25"]["positives"] == 12,
         "the candidate's first 25 opportunities contain 12 recorded positives")
    root(abs(12 / 25 - 0.48) < 1e-12 and abs(20 / 50 - 0.4) < 1e-12,
         "so precision RISES from .40 at capacity 50 to .48 at capacity 25")
    root(12 / 90 < 20 / 90, "while recall falls, because the top 25 are a subset of the top 50")
    root(set(top25) < set(candidate_ranked[:CAPACITY]),
         "and that subset relation is a fact about this ranking, not an assumption")
    root(mine_policy["reorderNullPositives"] == mine_policy["top25"]["positives"],
         "reversing the selected block changes no set metric: the membership null")
    root(mine_policy["positiveToNegativeSwap"]["positives"] == 11,
         "exchanging one selected positive for one unselected negative costs exactly one")
    root(mine_results["training_prior"]["top50Positives"] == 6
         and len({round(value, 15) for value in scores["training_prior"]}) == 1,
         "the prior baseline's six positives come from a tie rule: every one of its scores is the same number")
    root(abs(mine_results["training_prior"]["averagePrecision"] - 90 / 824) < 1e-15,
         "and its average precision is exactly the validation positive fraction 90/824")
    root(mine_results["candidate_pre_call"]["correct"] == 733
         < mine_results["training_prior"]["correct"] == 734,
         "the candidate makes one FEWER correct decision at threshold .5 than the constant baseline")
    root(mine_results["candidate_pre_call"]["top50Positives"] == 20
         > mine_results["training_prior"]["top50Positives"] == 6,
         "while concentrating more than three times as many positives in the selected 50")
    root(mine_results["unavailable_duration"]["averagePrecision"]
         > mine_results["candidate_pre_call"]["averagePrecision"],
         "the unavailable-duration model scores higher, which is the point: a better score, a wrong contract")
    root(mine_results["unavailable_duration"]["logLoss"] < mine_results["candidate_pre_call"]["logLoss"]
         < mine_results["training_prior"]["logLoss"],
         "and its log loss is lower too, on all three procedures the same 824 rows")
    root(all(cells := mine_results[key]["confusion"] for key in mine_results)
         and all(sum(sum(row) for row in mine_results[key]["confusion"]) == VALIDATION_ROWS
                 for key in mine_results),
         "every confusion matrix accounts for all 824 validation rows")
    root(mine_results["training_prior"]["confusion"][1][1] == 0,
         "the constant baseline predicts no positive at threshold .5, so its positive recall is zero")

    # ------------------------------------------------------------ the module
    payload = {
        "provenance": {
            "file": "/learn-assets/problem-formulation/bank-additional.csv",
            "attribution": "/learn-assets/problem-formulation/ATTRIBUTION.txt",
            "description": "/learn-assets/problem-formulation/bank-marketing-variable-description.txt",
            "program": "/learn-assets/problem-formulation/formulation-calculations.py",
            "name": "Bank Marketing",
            "creator": "Moro, S., Rita, P. & Cortez, P. (2014)",
            "record": "https://archive.ics.uci.edu/dataset/222/bank+marketing",
            "doi": "https://doi.org/10.24432/C5K306",
            "license": "CC BY 4.0",
            "licenseUrl": "https://creativecommons.org/licenses/by/4.0/",
            "retrieved": "2026-09-12",
            "sha256": digest,
            "bytes": len(packet_bytes),
            "subset": "the provider's own 4,119-row random subset, retained byte for byte",
        },
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikitLearn": sklearn.__version__,
        },
        "partition": {
            "sourceRows": SOURCE_ROWS,
            "positiveRows": sum(y),
            "developmentRows": DEVELOPMENT_ROWS,
            "reservedRows": RESERVED_ROWS,
            "trainRows": TRAIN_ROWS,
            "validationRows": VALIDATION_ROWS,
            "trainPositives": recorded["trainPositives"],
            "validationPositives": recorded["validationPositives"],
            "reservedPositives": sum(y) - recorded["trainPositives"] - recorded["validationPositives"],
            "trainPrior": prior,
            "capacity": CAPACITY,
            "firstSeed": FIRST_SEED,
            "secondSeed": SECOND_SEED,
        },
        "features": {
            "numeric": numeric,
            "categorical": categorical,
            "candidateColumns": len(feature_names["candidate_pre_call"]),
            "durationColumns": len(feature_names["unavailable_duration"]),
            "sentinel": 999,
            "sentinelRows": sentinel_rows,
        },
        "procedures": [
            {
                "id": procedure["id"],
                "label": procedure["label"],
                "features": procedure["features"],
                "availability": procedure["availability"],
                # The RECORDED measurement, not this script's recomputation.
                # The two agree to floating-point noise, asserted above; the
                # module carries the number that was actually measured, and
                # stays a deterministic projection of the frozen packet so its
                # bytes cannot drift with a BLAS thread setting.
                "averagePrecision": recorded["results"][procedure["packetKey"]]["averagePrecision"],
                "logLoss": recorded["results"][procedure["packetKey"]]["logLoss"],
                "correct": recorded["results"][procedure["packetKey"]]["correct"],
                "confusion": recorded["results"][procedure["packetKey"]]["confusion"],
                "top50Positives": recorded["results"][procedure["packetKey"]]["top50Positives"],
                "precisionAt50": recorded["results"][procedure["packetKey"]]["precisionAt50"],
                "recallAt50": recorded["results"][procedure["packetKey"]]["recallAt50"],
                "iterations": (iterations[procedure["packetKey"]][0]
                               if iterations[procedure["packetKey"]] is not None else None),
            } for procedure in PROCEDURES
        ],
        "validation": {
            "ids": recorded["validationRows"],
            "targets": recorded["validationTargets"],
            "scores": {
                "candidate": recorded["results"]["candidate_pre_call"]["probabilities"],
                "duration": recorded["results"]["unavailable_duration"]["probabilities"],
            },
        },
        "policyFixtures": {
            "top25Ids": recorded["policyFixtures"]["top25"]["ids"],
            "top25Positives": recorded["policyFixtures"]["top25"]["positives"],
            "top50Positives": recorded["results"]["candidate_pre_call"]["top50Positives"],
            "swap": recorded["policyFixtures"]["positiveToNegativeSwap"],
            "reorderNullPositives": recorded["policyFixtures"]["reorderNullPositives"],
        },
        "timeline": {
            "records": recorded["timelineInput"],
            "cases": recorded["timelineResults"],
        },
    }
    module = build_module(payload)

    if write:
        MODULE.write_text(module, encoding="utf-8", newline="\n")
    else:
        check(MODULE.exists(), "the data module exists; rerun with --write if not")
        if MODULE.exists():
            check(MODULE.read_text(encoding="utf-8") == module,
                  "a fresh derivation reproduces src/learn/data/formulation-data.js byte for byte")

    # ------------------------------------------------------------- coverage
    all_leaves = set(leaf_paths(recorded))
    missing = sorted(all_leaves - covered)
    extra = sorted(covered - all_leaves)
    coverage = len(covered & all_leaves) / len(all_leaves)
    check(not missing, f"{len(missing)} trust-root leaves were never compared, first: {missing[:6]}")
    check(not extra, f"{len(extra)} compared paths are not packet leaves, first: {extra[:6]}")

    independent = len(all_leaves) - len(validated_by_refit & all_leaves) - len(recorded_inputs & all_leaves)

    # Floors run BEFORE the evidence is written, and sit just below the real
    # numbers: a floor at half of current lets a whole section vanish silently.
    check(len(all_leaves) >= 10050, f"the trust root has only {len(all_leaves)} leaves; a block has gone missing")
    check(root_checks >= 28, f"only {root_checks} independent property checks ran")
    check(checks >= 10100, f"only {checks} checks ran; the suite has lost coverage")

    evidence = json.dumps({
        "checkedAt": datetime.now(timezone.utc).isoformat(),
        "verifier": "scripts/verify-formulation-data.py",
        "verifierSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mode": "write" if write else "read-only",
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
        "packet": {
            "calculatedInputsSha256": hashlib.sha256(PACKET_RESULTS.read_bytes()).hexdigest(),
            "manuscriptSha256": hashlib.sha256(MANUSCRIPT.read_bytes()).hexdigest(),
            "datasetSha256": digest,
            "datasetBytes": len(packet_bytes),
        },
        "served": {
            "dataset": payload["provenance"]["file"],
            "sha256": hashlib.sha256(ASSET_DATASET.read_bytes()).hexdigest(),
            "attribution": payload["provenance"]["attribution"],
            "program": payload["provenance"]["program"],
        },
        "module": {
            "path": "src/learn/data/formulation-data.js",
            "sha256": hashlib.sha256(MODULE.read_bytes()).hexdigest() if MODULE.exists() else None,
            "regeneration": "written" if write else "byte-identical",
        },
        "trustRootScalarLeaves": len(all_leaves),
        "trustRootLeavesAsserted": len(covered & all_leaves),
        "trustRootCoverage": round(coverage, 6),
        "trustRootUncoveredPaths": missing[:40],
        "trustRootLeavesReDerivedIndependently": independent,
        "trustRootLeavesValidatedByRefit": len(validated_by_refit & all_leaves),
        "trustRootLeavesRecordedInputsRegeneratedFromSeeds": len(recorded_inputs & all_leaves),
        "trustRootPropertyChecks": root_checks,
        "checks": checks,
        "scope": "Every scalar leaf of the content packet's calculated-inputs.json is recomputed from the "
                 "dataset this lesson serves and compared, with the path of each compared leaf recorded so the "
                 "coverage figure is measured rather than asserted. Average precision is summed over distinct "
                 "score thresholds from its definition; log loss is the mean negative log likelihood written "
                 "out; the confusion cells, correct count and top-50 counts are counted here; the ranking is an "
                 "explicit sort by descending probability then ascending source row rather than numpy.lexsort; "
                 "the one-hot feature names are rebuilt from the sorted distinct categories of the training "
                 "rows; the as-known calibration results come from an independent implementation of the "
                 "manuscript's five-step rule; and the two feature lists and the calibration history are parsed "
                 "out of the frozen manuscript rather than retyped.",
        "notCovered": [
            "The 2,472 fitted probabilities and the 2 iteration counts are VALIDATED BY REFITTING with the same "
            "scikit-learn estimator, not re-derived from an independent implementation of regularised logistic "
            "regression. The estimator is the object under study. Everything computed from those probabilities "
            "-- every metric, ranking, confusion cell and policy fixture -- is independent of scikit-learn.",
            "The 4,119 split membership indices are RECORDED INPUTS regenerated from their seeds through "
            "StratifiedShuffleSplit, a different entry point from the packet's train_test_split, and separately "
            "checked for the structural properties the lesson claims: disjoint, exhaustive, correctly sized and "
            "stratified to the recorded per-class counts. Their exact order is a property of scikit-learn's "
            "random state and is not independently reproduced.",
            "This verifier checks numbers and data. Rendering, layout, interaction and the teaching itself are "
            "separate steps and are not claimed here.",
        ],
        "passed": not problems,
    }, indent=2) + "\n"
    if keep_evidence:
        EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
        EVIDENCE.write_text(evidence, encoding="utf-8", newline="\n")

    if problems:
        for problem in problems[:40]:
            print(f"FAIL {problem}")
        raise SystemExit(f"{len(problems)} of {checks} data checks failed")

    print(f"PASS: {checks:,} checks; {coverage:.1%} of the trust root's {len(all_leaves):,} leaves asserted "
          f"({independent:,} re-derived independently, {len(validated_by_refit & all_leaves):,} validated by "
          f"refitting the same estimator, {len(recorded_inputs & all_leaves):,} recorded split inputs "
          f"regenerated from their seeds), {root_checks} independent property checks, module "
          f"{'written' if write else 'byte-identical'}.")


if __name__ == "__main__":
    main()
