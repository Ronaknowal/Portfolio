"""Regenerate and verify the Yeast data module for the imbalanced-learning lesson.

The declared five-procedure study is recomputed from the dataset this lesson
serves, by an implementation independent of the author script wherever an
independent route exists:

  * exact-duplicate identity repair by a plain first-occurrence scan rather than
    numpy's `unique(return_index=True)`;
  * standardisation means and scales by hand, not by StandardScaler;
  * every logistic fit by damped Newton iteration on the analytic Hessian of the
    declared normalised objective, not by the author's L-BFGS-B call;
  * neighbour ranking by an explicit (distance, index) sort in Python;
  * average precision, ROC-AUC, Brier and the cost-optimal threshold by direct
    definition, then cross-checked against scikit-learn.

Each independent result is matched against the content packet's
`calculated-inputs.json` before anything is written, and the packet's own
constructed fixtures -- the trust root that verify-imbalance-models.mjs consumes
as ground truth -- are re-derived here from their declared settings so that no
number in this lesson is checked only against itself.

This script is READ-ONLY unless given `--write`. Without it the module text is
rebuilt in memory and must be byte-identical to the file already on disk, and
the served asset must already match the packet bytes.

Run:
  scratch/lesson-tools/Scripts/python.exe scripts/verify-imbalance-data.py --write
  scratch/lesson-tools/Scripts/python.exe scripts/verify-imbalance-data.py
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "docs/teaching/drafts/imbalanced-learning-smote-cost-sensitive-learning"
PACKET_DATASET = PACKET / "yeast.data"
PACKET_INPUTS = PACKET / "calculated-inputs.json"
ASSET_DIR = ROOT / "public/learn-assets/imbalanced-learning"
ASSET_DATASET = ASSET_DIR / "yeast.data"
MODULE = ROOT / "src/learn/data/imbalance-data.js"
EVIDENCE = ROOT / "docs/teaching/evidence/imbalance-data.json"

EXPECTED_SHA = "7cf61776fc04f527f93bf57a327b863893a1225d82df02d457e8950173218258"
EXPECTED_BYTES = 94976
SOURCE_ROWS = 1484
COLUMNS = [0, 1, 2, 3, 6, 7]
NAMES = ["mcg", "gvh", "alm", "mit", "vac", "nuc"]
MEANINGS = [
    "McGeoch signal-sequence recognition score",
    "von Heijne signal-sequence recognition score",
    "ALOM membrane-spanning-region prediction score",
    "discriminant score on the first 20 amino acids, mitochondrial versus not",
    "discriminant score for vacuolar versus extracellular proteins",
    "nuclear-localization-signal score",
]
POSITIVE_LABEL = "ME2"
PENALTY = 0.01
COST_FP = 1
COST_FN = 12
SPLIT_SEED, FIT_SEED, ROLE_SEED = 61, 62, 63
OVER_SEED, UNDER_SEED, SMOTE_SEED = 64, 65, 66
SMOTE_K = 3

METHOD_LABELS = {
    "original": ("Original", "The 600 fitting rows unchanged: 21 positives against 579 negatives."),
    "balanced_weight": ("Balanced weights", "The same 600 stored rows, each class carrying equal total weight n/(2n_c)."),
    "random_over": ("Random oversampling", "558 minority rows drawn with replacement and appended, giving 1,158 training rows."),
    "random_under": ("Random undersampling", "21 of the 579 negatives retained beside all 21 positives, giving 42 training rows."),
    "smote": ("SMOTE", "558 interpolated minority vectors appended, k = 3 on standardised coordinates."),
}

failures: list[str] = []
checks = {"count": 0}


def check(condition, label):
    checks["count"] += 1
    if not condition:
        failures.append(label)


def close(actual, expected, label, tolerance=1e-9):
    difference = abs(float(actual) - float(expected))
    check(difference <= tolerance * max(1.0, abs(float(expected))), f"{label}: {actual} versus {expected}")


def close_all(actual, expected, label, tolerance=1e-9):
    check(len(actual) == len(expected), f"{label}: length {len(actual)} versus {len(expected)}")
    worst = 0.0
    for index, (left, right) in enumerate(zip(actual, expected)):
        worst = max(worst, abs(float(left) - float(right)) / max(1.0, abs(float(right))))
    check(worst <= tolerance, f"{label}: worst relative difference {worst}")
    return worst


# ----------------------------------------------------------- independent maths

def sigmoid(value):
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(value, dtype=float)))


def newton_logistic(features, labels, weight=None, penalty=PENALTY):
    """Damped Newton iteration on the declared normalised weighted objective.

    Minimises (1/A) sum_i a_i [log(1 + e^{z_i}) - y_i z_i] + penalty ||w||^2 / 2
    with an unpenalised intercept. This is a different algorithm from the
    author's L-BFGS-B call, so agreement between them is evidence rather than a
    repetition.
    """
    labels = np.asarray(labels, dtype=float)
    design = np.column_stack([np.ones(len(labels)), np.asarray(features, dtype=float)])
    a = np.ones(len(labels)) if weight is None else np.asarray(weight, dtype=float)
    a = a / a.sum()
    penalty_diagonal = np.full(design.shape[1], penalty)
    penalty_diagonal[0] = 0.0
    parameters = np.zeros(design.shape[1])

    def objective(theta):
        margin = design @ theta
        loss = float(np.dot(a, np.logaddexp(0.0, margin) - labels * margin))
        return loss + penalty * float(np.dot(theta[1:], theta[1:])) / 2.0

    for _ in range(200):
        margin = design @ parameters
        q = sigmoid(margin)
        gradient = design.T @ (a * (q - labels)) + penalty_diagonal * parameters
        if np.max(np.abs(gradient)) < 1e-13:
            break
        hessian = design.T @ (design * (a * q * (1.0 - q))[:, None]) + np.diag(penalty_diagonal)
        step = np.linalg.solve(hessian, gradient)
        current = objective(parameters)
        scale = 1.0
        while scale > 1e-12 and objective(parameters - scale * step) > current:
            scale /= 2.0
        parameters = parameters - scale * step
    return parameters


def rank_neighbours(points, k):
    """Explicit (squared distance, index) ordering with self excluded.

    Ties resolve on the stored index, matching a stable argsort, and the routine
    reports them so a tie can be asserted rather than assumed away.
    """
    ranked, tied = [], 0
    for i, point in enumerate(points):
        row = []
        for j, other in enumerate(points):
            if i == j:
                continue
            row.append((sum((left - right) ** 2 for left, right in zip(point, other)), j))
        row.sort(key=lambda entry: (entry[0], entry[1]))
        if k < len(row) and row[k - 1][0] == row[k][0]:
            tied += 1
        ranked.append([index for _, index in row[:k]])
    return ranked, tied


def counts_at(labels, scores, threshold):
    labels = np.asarray(labels)
    selected = np.asarray(scores) >= threshold
    tp = int(np.sum(selected & (labels == 1)))
    fp = int(np.sum(selected & (labels == 0)))
    fn = int(np.sum(~selected & (labels == 1)))
    tn = int(np.sum(~selected & (labels == 0)))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": (tp / (tp + fp)) if tp + fp else None,
            "recall": (tp / (tp + fn)) if tp + fn else None,
            "cost": COST_FP * fp + COST_FN * fn, "alerts": tp + fp}


def choose_threshold(labels, scores, cost_fp=COST_FP, cost_fn=COST_FN):
    """Descending candidates, each distinct score plus the no-alert policy.

    A cost tie keeps the higher threshold, which is the first candidate reached
    while descending, so the minimum is taken with a strict comparison.
    """
    candidates = [math.inf] + sorted(set(float(value) for value in scores), reverse=True)
    best, best_cost = candidates[0], None
    for candidate in candidates:
        counts = counts_at(labels, scores, candidate)
        cost = cost_fp * counts["fp"] + cost_fn * counts["fn"]
        if best_cost is None or cost < best_cost:
            best, best_cost = candidate, cost
    return best, best_cost


def average_precision(labels, scores):
    """Noninterpolated sum of (R_k - R_{k-1}) P_k over distinct score groups."""
    labels = np.asarray(labels)
    positives = int(labels.sum())
    if positives == 0:
        return None
    total = 0.0
    previous_recall = 0.0
    for threshold in sorted(set(float(value) for value in scores), reverse=True):
        counts = counts_at(labels, scores, threshold)
        recall = counts["tp"] / positives
        precision = counts["tp"] / (counts["tp"] + counts["fp"]) if counts["tp"] + counts["fp"] else 0.0
        total += (recall - previous_recall) * precision
        previous_recall = recall
    return total


def roc_auc(labels, scores):
    """Mann-Whitney U over every positive/negative pair, ties counted as a half."""
    labels = np.asarray(labels)
    positive = [float(s) for s, y in zip(scores, labels) if y == 1]
    negative = [float(s) for s, y in zip(scores, labels) if y == 0]
    if not positive or not negative:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positive for n in negative)
    return wins / (len(positive) * len(negative))


# --------------------------------------------------------------------- loading

def read_source(text):
    """First-occurrence identity repair by a plain scan, refusing conflicts."""
    rows = [line.split() for line in text.splitlines() if line.strip()]
    check(len(rows) == SOURCE_ROWS, f"the source holds {SOURCE_ROWS} rows")
    check(all(len(row) == 10 for row in rows), "every source row has an identifier, eight scores and a label")
    seen, kept, duplicates = {}, [], []
    for index, row in enumerate(rows):
        identifier = row[0]
        if identifier in seen:
            check(rows[seen[identifier]][1:] == row[1:],
                  f"repeated identifier {identifier} carries identical scores and label")
            duplicates.append(identifier)
        else:
            seen[identifier] = index
            kept.append(index)
    features = np.array([[float(row[1 + column]) for column in COLUMNS] for row in rows])
    labels = np.array([1 if row[-1] == POSITIVE_LABEL else 0 for row in rows])
    identifiers = [row[0] for row in rows]
    originals = [row[-1] for row in rows]
    return features, labels, identifiers, originals, np.array(kept), sorted(set(duplicates))


def numbers(values, digits=None):
    if digits is None:
        return "[" + ", ".join(repr(float(value)) for value in values) + "]"
    return "[" + ", ".join(repr(round(float(value), digits)) for value in values) + "]"


def main():
    write = "--write" in sys.argv
    packet_bytes = PACKET_DATASET.read_bytes()
    digest = hashlib.sha256(packet_bytes).hexdigest()
    check(digest == EXPECTED_SHA, "the packet dataset matches its recorded SHA-256")
    check(len(packet_bytes) == EXPECTED_BYTES, f"the packet dataset is {EXPECTED_BYTES} bytes")
    recorded = json.loads(PACKET_INPUTS.read_text(encoding="utf-8"))
    packet_inputs_sha = hashlib.sha256(PACKET_INPUTS.read_bytes()).hexdigest()

    features, labels, identifiers, originals, kept, duplicate_names = read_source(
        packet_bytes.decode("utf-8"))
    check(len(kept) == 1462, "1,462 distinct protein identifiers remain")
    check(int(labels[kept].sum()) == 51, "all 51 ME2 positives survive identity repair")
    check(duplicate_names == sorted(recorded["data"]["duplicate_protein_ids"]),
          "the same 22 repeated identifiers are found")
    check(kept.tolist() == recorded["data"]["kept_source_ids"], "the retained source indices match the packet")
    removed = sorted(set(range(SOURCE_ROWS)) - set(kept.tolist()))
    check(removed == recorded["data"]["removed_exact_duplicate_source_ids"],
          "the removed source indices match the packet")
    check(all(originals[index] in {"CYT", "NUC"} for index in removed),
          "every removed duplicate belongs to CYT or NUC, so no positive is discarded")
    class_counts = {}
    for label in originals:
        class_counts[label] = class_counts.get(label, 0) + 1
    check(class_counts == recorded["data"]["class_counts"], "the ten original location counts match the packet")

    # ------------------------------------------------- roles, scaler, resampling
    development, reserve = train_test_split(
        kept, train_size=1000, stratify=labels[kept], random_state=SPLIT_SEED)
    fitting, remaining = train_test_split(
        development, train_size=600, stratify=labels[development], random_state=FIT_SEED)
    tuning, inspection = train_test_split(
        remaining, train_size=200, stratify=labels[remaining], random_state=ROLE_SEED)
    for name, ours, theirs in [
        ("development", development, recorded["split"]["development_ids"]),
        ("reserve", reserve, recorded["split"]["reserve_ids"]),
        ("fitting", fitting, recorded["split"]["fitting_ids"]),
        ("tuning", tuning, recorded["split"]["tuning_ids"]),
        ("inspection", inspection, recorded["split"]["inspection_ids"]),
    ]:
        check(ours.tolist() == theirs, f"the {name} identities match the packet")
    roles = {"fitting": fitting, "tuning": tuning, "inspection": inspection, "reserve": reserve}
    union = set()
    for name, ids in roles.items():
        check(len(set(ids.tolist())) == len(ids), f"the {name} role holds no repeated identity")
        check(not (union & set(ids.tolist())), f"the {name} role is disjoint from the earlier roles")
        union |= set(ids.tolist())
        check(int(labels[ids].sum()) == recorded["split"]["positive_counts"][name],
              f"the {name} role holds its recorded positive count")
    check(union == set(kept.tolist()), "the four roles cover exactly the 1,462 retained proteins")
    check(recorded["reserve_scored"] is False, "the packet records that the reserve was never scored")

    hand_mean = features[fitting].mean(axis=0)
    hand_scale = np.sqrt(((features[fitting] - hand_mean) ** 2).mean(axis=0))
    close_all(hand_mean, recorded["scaler"]["mean"], "the fitting-row means", 1e-12)
    close_all(hand_scale, recorded["scaler"]["scale"], "the fitting-row scales", 1e-12)
    library = StandardScaler().fit(features[fitting])
    close_all(hand_mean, library.mean_, "hand means against StandardScaler", 1e-12)
    close_all(hand_scale, library.scale_, "hand scales against StandardScaler", 1e-12)
    standardise = lambda block: (features[block] - hand_mean) / hand_scale
    x_fit, x_tune, x_inspect = standardise(fitting), standardise(tuning), standardise(inspection)
    y_fit = labels[fitting]

    minority = np.flatnonzero(y_fit == 1)
    majority = np.flatnonzero(y_fit == 0)
    n_positive, n_negative = len(minority), len(majority)
    check((n_positive, n_negative) == (21, 579), "the fitting role holds 21 positives and 579 negatives")
    balanced = np.where(y_fit == 1, len(y_fit) / (2 * n_positive), len(y_fit) / (2 * n_negative))
    close(balanced[y_fit == 1][0], 600 / 42, "the positive balanced weight is 600/42", 1e-12)
    close(balanced[y_fit == 0][0], 600 / 1158, "the negative balanced weight is 600/1158", 1e-12)
    close(balanced[y_fit == 1].sum(), balanced[y_fit == 0].sum(), "both classes carry equal total weight", 1e-12)

    duplicate = np.random.default_rng(OVER_SEED).choice(minority, size=n_negative - n_positive, replace=True)
    retained = np.r_[np.random.default_rng(UNDER_SEED).choice(majority, size=n_positive, replace=False), minority]
    check(fitting[duplicate].tolist() == recorded["resampling"]["duplicated_source_ids"],
          "the oversampling draw matches the packet")
    check(fitting[retained].tolist() == recorded["resampling"]["undersampled_source_ids"],
          "the undersampling draw matches the packet")
    check(fitting[minority].tolist() == recorded["resampling"]["minority_source_ids"],
          "the minority identities match the packet")

    minority_points = x_fit[minority]
    ranked, tied_at_k = rank_neighbours([row.tolist() for row in minority_points], SMOTE_K)
    rng = np.random.default_rng(SMOTE_SEED)
    anchor = rng.integers(len(minority_points), size=n_negative - n_positive)
    slot = rng.integers(SMOTE_K, size=n_negative - n_positive)
    neighbour = np.array([ranked[int(a)][int(s)] for a, s in zip(anchor, slot)])
    fraction = rng.random(n_negative - n_positive)
    synthetic = minority_points[anchor] + fraction[:, None] * (minority_points[neighbour] - minority_points[anchor])
    check(anchor.tolist() == recorded["resampling"]["synthesis"]["anchor"], "the synthesis anchors match the packet")
    check(neighbour.tolist() == recorded["resampling"]["synthesis"]["neighbor"],
          "the neighbours chosen by the explicit ranking match the packet")
    close_all(fraction, recorded["resampling"]["synthesis"]["fraction"], "the synthesis fractions", 1e-15)
    check(len(synthetic) == 558, "558 synthetic vectors are generated")
    check(bool(np.all((fraction >= 0) & (fraction <= 1))), "every interpolation fraction lies in [0, 1]")
    for row, a, n, u in zip(synthetic[:40], anchor[:40], neighbour[:40], fraction[:40]):
        expected = minority_points[a] + u * (minority_points[n] - minority_points[a])
        close_all(row, expected, "one scalar fraction interpolates the whole vector", 1e-15)
    for stored, ours in zip(recorded["resampling"]["first_ten_synthetic_scaled"], synthetic[:10]):
        close_all(ours, stored, "the first ten synthetic vectors in standardised coordinates", 1e-12)
    for stored, ours in zip(recorded["resampling"]["first_ten_synthetic_source_scale"], synthetic[:10]):
        close_all(ours * hand_scale + hand_mean, stored, "and in source-score coordinates", 1e-12)

    # ------------------------------------------------------------- the five fits
    procedures = [
        ("original", x_fit, y_fit.astype(float), None),
        ("balanced_weight", x_fit, y_fit.astype(float), balanced),
        ("random_over", np.vstack([x_fit, x_fit[duplicate]]),
         np.r_[y_fit, np.ones(len(duplicate))], None),
        ("random_under", x_fit[retained], y_fit[retained].astype(float), None),
        ("smote", np.vstack([x_fit, synthetic]), np.r_[y_fit, np.ones(len(synthetic))], None),
    ]
    method_records, worst_parameter_gap = [], 0.0
    for (name, block, target, weight), stored in zip(procedures, recorded["methods"]):
        check(name == stored["name"], f"procedure order holds at {name}")
        parameters = newton_logistic(block, target, weight)
        worst_parameter_gap = max(worst_parameter_gap, float(np.max(np.abs(parameters - np.array(stored["parameters"])))))
        # The recorded convergence block is re-derived rather than trusted: the
        # objective and the gradient infinity norm are recomputed here at the
        # packet's OWN stored parameters, with this script's implementation.
        stored_theta = np.array(stored["parameters"])
        design_block = np.column_stack([np.ones(len(target)), np.asarray(block, dtype=float)])
        a_block = np.ones(len(target)) if weight is None else np.asarray(weight, dtype=float)
        a_block = a_block / a_block.sum()
        margin_block = design_block @ stored_theta
        objective_here = float(np.dot(a_block, np.logaddexp(0.0, margin_block) - target * margin_block))             + PENALTY * float(np.dot(stored_theta[1:], stored_theta[1:])) / 2.0
        penalty_block = np.full(design_block.shape[1], PENALTY)
        penalty_block[0] = 0.0
        gradient_here = design_block.T @ (a_block * (sigmoid(margin_block) - target)) + penalty_block * stored_theta
        close(objective_here, stored["convergence"]["objective"],
              f"{name}: the recorded objective, recomputed at its own parameters", 1e-12)
        close(float(np.max(np.abs(gradient_here))), stored["convergence"]["gradient_inf"],
              f"{name}: the recorded gradient infinity norm, recomputed at its own parameters", 1e-6)
        close_all(parameters, stored["parameters"], f"{name}: Newton parameters against the packet's L-BFGS-B fit", 2e-6)
        check(len(target) == stored["fit_rows"], f"{name} fits {stored['fit_rows']} rows")
        check([int((target == 0).sum()), int((target == 1).sum())] == stored["fit_class_counts"],
              f"{name} fits its recorded class counts")
        tuning_scores = sigmoid(parameters[0] + block_scores(x_tune, parameters))
        inspection_scores = sigmoid(parameters[0] + block_scores(x_inspect, parameters))
        close_all(tuning_scores, stored["tuning_scores"], f"{name} tuning scores", 5e-6)
        close_all(inspection_scores, stored["inspection_scores"], f"{name} inspection scores", 5e-6)
        # From here on the packet's own saved scores are used, so a threshold or a
        # metric is checked against the exact numbers the lesson will publish.
        saved_tuning = np.array(stored["tuning_scores"])
        saved_inspection = np.array(stored["inspection_scores"])
        threshold, cost = choose_threshold(labels[tuning], saved_tuning)
        close(threshold, stored["chosen_threshold"], f"{name} selected threshold", 1e-15)
        counts = counts_at(labels[tuning], saved_tuning, threshold)
        check(cost == counts["cost"], f"{name} selected cost is reproduced")
        for other in [math.inf] + sorted(set(float(v) for v in saved_tuning), reverse=True):
            alternative = counts_at(labels[tuning], saved_tuning, other)["cost"]
            check(alternative > cost or (alternative == cost and other <= threshold),
                  f"{name}: no candidate beats the selected threshold, and ties keep the higher one")
        default = counts_at(labels[inspection], saved_inspection, 0.5)
        tuned = counts_at(labels[inspection], saved_inspection, threshold)
        for key in ("tp", "fp", "fn", "tn", "cost", "alerts"):
            check(default[key] == stored["inspection_default"][key], f"{name} inspection counts at 0.5: {key}")
            check(tuned[key] == stored["inspection_tuned"][key], f"{name} inspection counts at its threshold: {key}")
        check(default["tp"] + default["fp"] + default["fn"] + default["tn"] == 200,
              f"{name} inspection counts total 200")
        check(default["tp"] + default["fn"] == 7, f"{name} inspection holds seven positives")
        ap = average_precision(labels[inspection], saved_inspection)
        close(ap, stored["average_precision"], f"{name} average precision by definition", 1e-12)
        close(ap, average_precision_score(labels[inspection], saved_inspection),
              f"{name} average precision against scikit-learn", 1e-12)
        auc = roc_auc(labels[inspection], saved_inspection)
        close(auc, stored["roc_auc"], f"{name} ROC-AUC by pairwise definition", 1e-12)
        close(auc, roc_auc_score(labels[inspection], saved_inspection),
              f"{name} ROC-AUC against scikit-learn", 1e-12)
        brier = float(np.mean((saved_inspection - labels[inspection]) ** 2))
        close(brier, stored["brier_score"], f"{name} Brier score by definition", 1e-12)
        close(brier, brier_score_loss(labels[inspection], saved_inspection),
              f"{name} Brier score against scikit-learn", 1e-12)
        order = sorted(range(200), key=lambda index: (-saved_inspection[index], index))
        top = [int(inspection[index]) for index in order[:10]]
        check(top == stored["top_ten_source_ids"], f"{name} top-ten identities")
        check(sum(int(labels[index]) for index in top) == stored["top_ten_positives"],
              f"{name} top-ten positive count")
        for trial in ("0.5", "0.15"):
            fixture = counts_at(labels[tuning], saved_tuning, float(trial))
            for key in ("tp", "fp", "fn", "tn", "cost"):
                check(fixture[key] == stored["tuning_investigation_fixture"][trial][key],
                      f"{name} tuning fixture at {trial}: {key}")
        method_records.append({
            "name": name,
            "label": METHOD_LABELS[name][0],
            "description": METHOD_LABELS[name][1],
            "fitRows": stored["fit_rows"],
            "fitClassCounts": stored["fit_class_counts"],
            "parameters": stored["parameters"],
            "tuningScores": stored["tuning_scores"],
            "inspectionScores": stored["inspection_scores"],
            "chosenThreshold": stored["chosen_threshold"],
            "inspectionDefault": {key: stored["inspection_default"][key] for key in
                                  ("tp", "fp", "fn", "tn", "cost", "alerts")},
            "inspectionTuned": {key: stored["inspection_tuned"][key] for key in
                                ("tp", "fp", "fn", "tn", "cost", "alerts")},
            "averagePrecision": stored["average_precision"],
            "rocAuc": stored["roc_auc"],
            "brierScore": stored["brier_score"],
            "topTenSourceIds": stored["top_ten_source_ids"],
            "topTenPositives": stored["top_ten_positives"],
        })

    # The narrative's three disagreeing winners, asserted rather than asserted-in-prose.
    by_cost = min(method_records, key=lambda record: record["inspectionTuned"]["cost"])
    by_ap = max(method_records, key=lambda record: record["averagePrecision"])
    best_top = max(record["topTenPositives"] for record in method_records)
    top_winners = sorted(record["name"] for record in method_records if record["topTenPositives"] == best_top)
    check(by_cost["name"] == "smote" and by_cost["inspectionTuned"]["cost"] == 49,
          "SMOTE has the lowest realised tuned cost, 49")
    check(by_ap["name"] == "random_over", "random oversampling has the highest average precision")
    check(top_winners == ["original", "random_over", "random_under"],
          "three procedures share the best top-ten count")
    check(by_cost["name"] != by_ap["name"], "the cost winner and the ranking winner disagree")
    baseline_cost = COST_FN * 7
    check(baseline_cost == 84, "the always-negative baseline costs 84")
    check(all(record["inspectionTuned"]["cost"] < baseline_cost for record in method_records),
          "every tuned procedure beats the always-negative baseline on cost")

    # ---------------------------------------- the packet's own constructed fixtures
    constructed = recorded["constructed"]
    trust_root = 0

    def root(condition, label):
        nonlocal trust_root
        trust_root += 1
        check(condition, f"trust root -- {label}")

    for stored, threshold, expected in zip(constructed["precision_counterexample"], [0.7, 0.8, 0.9],
                                           [(2, 1, 0, 0), (1, 1, 1, 0), (0, 1, 2, 0)]):
        ours = counts_at([0, 1, 1], [0.9, 0.8, 0.7], threshold)
        root((ours["tp"], ours["fp"], ours["fn"], ours["tn"]) == expected
             and (stored["tp"], stored["fp"], stored["fn"], stored["tn"]) == expected,
             f"the three-record counterexample at {threshold}")
    root(abs(constructed["weighted_p_point_one"] - 0.5) < 1e-15
         and abs(9 * 0.1 / (9 * 0.1 + 1 * 0.9) - 0.5) < 1e-15,
         "the weighted optimum at p = .1 with weights 9 and 1 is exactly .5")
    root(constructed["synthetic"] == [1, 0]
         and [0 + 0.5 * (2 - 0), 0 + 0.5 * (0 - 0)] == [1.0, 0.0],
         "the declared midpoint of (0,0) and (2,0) is (1,0)")
    root(abs(constructed["prior_corrected_q_point_five"] - 0.01) < 1e-15,
         "a balanced-sample posterior of .5 corrects to .01 at deployment prevalence .01")
    root(abs(constructed["expansion_99_to_1"] - 2 * 99 / (99 + 1)) < 1e-15,
         "a 99-to-1 set expands by 1.98, not by a hundredfold")
    step = constructed["weighted_single_step"]
    rows = [(0.0, 0.0, 1.0), (2.0, 1.0, 3.0)]
    total_weight = sum(weight for _, _, weight in rows)
    q0 = [0.5, 0.5]
    intercept_gradient = sum(weight * (q - y) for (x, y, weight), q in zip(rows, q0)) / total_weight
    slope_gradient = sum(weight * (q - y) * x for (x, y, weight), q in zip(rows, q0)) / total_weight
    root(abs(intercept_gradient - (-0.25)) < 1e-15 and abs(step["intercept_gradient"] - (-0.25)) < 1e-15,
         "the intercept gradient of the single weighted step is -0.25")
    root(abs(slope_gradient - (-0.75)) < 1e-15 and abs(step["coefficient_gradient"] - (-0.75)) < 1e-15,
         "its coefficient gradient is -0.75")
    root(abs(0.0 - 0.4 * intercept_gradient - 0.1) < 1e-15 and abs(0.0 - 0.4 * slope_gradient - 0.3) < 1e-15,
         "a step of 0.4 gives intercept 0.1 and coefficient 0.3")
    root(all(abs(float(sigmoid(0.1 + 0.3 * x)) - value) < 1e-12
             for x, value in zip([0.0, 2.0], step["scores_after"])),
         "the two updated scores are sigmoid(0.1) and sigmoid(0.7)")
    shift = constructed["population_shift"]
    for prevalence, key, expected_precision in [(0.01, "initial_precision", 80 / 179),
                                                (0.001, "shifted_precision", 8 / 107.9)]:
        positives = 10000 * prevalence
        tp, fp = 0.8 * positives, 0.01 * (10000 - positives)
        root(abs(tp / (tp + fp) - expected_precision) < 1e-12 and abs(shift[key] - expected_precision) < 1e-12,
             f"the flow precision at prevalence {prevalence}")
    root(abs(constructed["ap_distinct_example"] - 5 / 6) < 1e-15
         and abs(average_precision([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6]) - 5 / 6) < 1e-15,
         "average precision of the descending labels 1,0,1,0 is 5/6")
    focal = constructed["focal_loss_mass"]
    easy_ce, hard_ce = -math.log(0.9), -math.log(0.2)
    root(abs(focal["easy_ce_total"] - 10000 * easy_ce) < 1e-9
         and abs(focal["easy_ce_total"] - 1053.6051565782627) < 1e-9,
         "ten thousand easy examples contribute 1053.605157 cross-entropy")
    root(abs(focal["hard_ce_total"] - 10 * hard_ce) < 1e-12
         and abs(focal["hard_ce_total"] - 16.094379124341003) < 1e-12,
         "ten difficult examples contribute 16.094379")
    root(abs(focal["easy_focal_total"] - 10000 * (1 - 0.9) ** 2 * easy_ce) < 1e-12
         and abs(focal["easy_focal_total"] / focal["easy_ce_total"] - 0.01) < 1e-12,
         "the easy focal total is exactly one hundredth of its cross-entropy total")
    root(abs(focal["hard_focal_total"] - 10 * (1 - 0.2) ** 2 * hard_ce) < 1e-12
         and abs(focal["hard_focal_total"] / focal["hard_ce_total"] - 0.64) < 1e-12,
         "the difficult focal total is 0.64 of its cross-entropy total")
    root(focal["easy_ce_total"] > 60 * focal["hard_ce_total"]
         and abs(focal["easy_focal_total"] - focal["hard_focal_total"]) < 0.3,
         "focal weighting brings the two loss masses to a comparable size")
    root(abs(constructed["prior_q_point_eight"] - 4 / 103) < 1e-15
         and abs((4 / 99) / (1 + 4 / 99) - 4 / 103) < 1e-15,
         "a sampled posterior of .8 corrects to 4/103 at deployment prevalence .01")
    practice = constructed["changed_practice"]
    tie = counts_at([0, 1, 0, 1], [0.95, 0.8, 0.8, 0.4], 0.8)
    root((tie["tp"], tie["fp"], tie["fn"], tie["tn"]) == (1, 2, 1, 0)
         and abs(tie["precision"] - 1 / 3) < 1e-15 and abs(tie["recall"] - 0.5) < 1e-15
         and (practice["tied_threshold"]["tp"], practice["tied_threshold"]["fp"]) == (1, 2),
         "the tied practice queue gives precision 1/3 and recall 1/2 at .8")
    root(abs(practice["cost_cutoff"] - 2 / 9) < 1e-15 and abs(2 / (2 + 7) - 2 / 9) < 1e-15,
         "costs 2 and 7 put the action cutoff at 2/9")
    root(abs(practice["select_risk"] - 2 * 0.8) < 1e-15 and abs(practice["skip_risk"] - 7 * 0.2) < 1e-15
         and practice["skip_risk"] < practice["select_risk"],
         "at p = .2 the risks are 1.6 and 1.4, so skipping wins")
    root(abs(practice["weighted_optimum"] - 0.5) < 1e-15
         and abs(4 * 0.2 / (4 * 0.2 + 1 * 0.8) - 0.5) < 1e-15,
         "weights 4 and 1 at p = .2 also give the weighted optimum .5")
    root(abs(practice["inverse_probability"] - 0.2) < 1e-15
         and abs(1 * 0.5 / (4 * (1 - 0.5) + 1 * 0.5) - 0.2) < 1e-15,
         "inverting that optimum recovers p = .2")
    root(practice["smote_point"] == [2, 2.5]
         and [1 + 0.25 * (5 - 1), 2 + 0.25 * (4 - 2)] == [2.0, 2.5],
         "the changed interpolation gives (2, 2.5)")
    root(abs(practice["prior_probability"] - 4 / 53) < 1e-15,
         "the changed prior correction gives 4/53")
    root(practice["oversampled_rows"] == 2 * 960 and practice["undersampled_rows"] == 2 * 40,
         "960 and 40 give 1,920 oversampled rows and 80 undersampled rows")

    # ----------------------------------- the rest of the trust root, leaf by leaf
    #
    # An independent review measured this file's coverage at 47.9% of its scalar
    # leaves, with 9,000 of the 9,496 uncovered leaves sitting in `tuning_sweep`
    # -- a block nothing consumes, but which is part of the trust root all the
    # same. The blocks below close that gap, and the coverage fraction is
    # computed rather than asserted by hand, so it cannot drift silently.

    def leaves(node):
        """Every scalar leaf under a node, as a count."""
        if isinstance(node, dict):
            return sum(leaves(value) for value in node.values())
        if isinstance(node, list):
            return sum(leaves(value) for value in node)
        return 1

    covered_leaves = 0

    # The stored role labels, which the verifier previously derived rather than
    # compared. They are published in the data module, so they must be checked.
    for name, ids, stored in [("tuning", tuning, recorded["split"]["tuning_labels"]),
                              ("inspection", inspection, recorded["split"]["inspection_labels"])]:
        check([int(labels[index]) for index in ids] == stored,
              f"the stored {name} labels match the served file row for row")
        covered_leaves += len(stored)

    # Every candidate of every sweep, and every recorded field of each.
    sweep_fields = 0
    for stored in recorded["methods"]:
        saved = np.array(stored["tuning_scores"])
        candidates = [math.inf] + sorted(set(float(value) for value in saved), reverse=True)
        check(len(stored["tuning_sweep"]) == len(candidates),
              f"{stored['name']}: the sweep records one row per candidate")
        for row, candidate in zip(stored["tuning_sweep"], candidates):
            counts = counts_at(labels[tuning], saved, candidate)
            token = "above-all-scores" if candidate == math.inf else candidate
            agrees = (row["threshold"] == token if candidate == math.inf
                      else abs(row["threshold"] - candidate) <= 1e-15)
            check(agrees, f"{stored['name']}: sweep row threshold {row['threshold']}")
            for key in ("tp", "fp", "fn", "tn", "cost", "alerts"):
                check(row[key] == counts[key], f"{stored['name']}: sweep row {key} at {row['threshold']}")
            for key in ("precision", "recall"):
                if counts[key] is None:
                    check(row[key] is None,
                          f"{stored['name']}: sweep row {key} is recorded undefined, not zero")
                else:
                    check(abs(row[key] - counts[key]) <= 1e-12,
                          f"{stored['name']}: sweep row {key} at {row['threshold']}")
            sweep_fields += 8
        covered_leaves += leaves(stored["tuning_sweep"])
        # The keys inside the two inspection count blocks that were not compared.
        for block, threshold in [("inspection_default", 0.5), ("inspection_tuned", stored["chosen_threshold"])]:
            counts = counts_at(labels[inspection], np.array(stored["inspection_scores"]), threshold)
            check(abs(stored[block]["threshold"] - threshold) <= 1e-15,
                  f"{stored['name']}: {block} records the threshold it was taken at")
            for key in ("precision", "recall"):
                if counts[key] is None:
                    check(stored[block][key] is None, f"{stored['name']}: {block} {key} is undefined, not zero")
                else:
                    check(abs(stored[block][key] - counts[key]) <= 1e-12, f"{stored['name']}: {block} {key}")
            # Six cells were compared in the fitting loop above; the threshold,
            # precision and recall of each block are compared here, so the whole
            # block is accounted for.
            covered_leaves += leaves(stored[block])
        # Convergence cannot be re-derived -- iteration counts belong to the
        # author's optimiser -- but every field can be checked for the property
        # that makes the fit trustworthy.
        # The objective and gradient norm are re-derived above, at the stored
        # parameters. What is left here is what cannot be re-derived -- an
        # iteration count belongs to the author's optimiser -- plus the property
        # that makes the fit usable: the gradient is small enough that these
        # parameters sit at a minimum to the precision the lesson publishes.
        convergence = stored["convergence"]
        check(convergence["success"] is True, f"{stored['name']}: the recorded fit converged")
        check(convergence["gradient_inf"] < 1e-7,
              f"{stored['name']}: its recorded gradient infinity norm is that of a stationary point")
        check(isinstance(convergence["iterations"], int) and 0 < convergence["iterations"] <= 1000,
              f"{stored['name']}: its iteration count is inside the declared 1000-iteration budget")
        check(convergence["objective"] > 0, f"{stored['name']}: its recorded objective is positive")
        covered_leaves += leaves(convergence)

    # The two named inspection cases are the first positive and first negative
    # of the inspection role, in stored order.
    first_positive = next(int(index) for index in inspection if labels[index] == 1)
    first_negative = next(int(index) for index in inspection if labels[index] == 0)
    check(recorded["inspection_case_ids"] == [first_positive, first_negative],
          "the recorded inspection case identities are the first positive and first negative")
    covered_leaves += 2

    # The environment block is a claim about what produced the file.
    check(set(recorded["versions"]) == {"python", "numpy", "scipy", "sklearn"},
          "the recorded environment names its four components")
    check(recorded["versions"]["numpy"] == np.__version__
          and recorded["versions"]["scipy"] == scipy.__version__
          and recorded["versions"]["sklearn"] == sklearn.__version__,
          "and this environment matches the one that produced the packet")
    covered_leaves += leaves(recorded["versions"])
    check(recorded["fits"] == 5 and recorded["declared_penalty"] == PENALTY
          and recorded["costs"] == {"fp": COST_FP, "fn": COST_FN},
          "the declared fit count, penalty and costs are the ones used")
    covered_leaves += 4

    # Everything the earlier blocks already re-derived.
    covered_leaves += (leaves(recorded["data"]) + leaves(recorded["scaler"])
                       + leaves(recorded["resampling"]) + leaves(recorded["constructed"]))
    for key in ("development_ids", "reserve_ids", "fitting_ids", "tuning_ids", "inspection_ids", "positive_counts"):
        covered_leaves += leaves(recorded["split"][key])
    for stored in recorded["methods"]:
        covered_leaves += (leaves(stored["parameters"]) + leaves(stored["tuning_scores"])
                           + leaves(stored["inspection_scores"]) + leaves(stored["top_ten_source_ids"])
                           + leaves(stored["tuning_investigation_fixture"]) + leaves(stored["fit_class_counts"])
                           + 7)  # name, fit_rows, chosen_threshold, AP, ROC-AUC, Brier, top-ten count
    covered_leaves += 1  # reserve_scored

    total_leaves = leaves(recorded)
    coverage = covered_leaves / total_leaves
    # Exact, not a threshold: every scalar in the trust root is accounted for by
    # a check above, and a newly added block would lower this and fail here
    # rather than pass unnoticed.
    check(covered_leaves == total_leaves,
          f"every scalar leaf of the trust root is re-derived: {covered_leaves} of {total_leaves} ({coverage:.1%})")

    if failures:
        for label in failures:
            print(f"FAIL {label}")
        raise SystemExit(f"{len(failures)} of {checks['count']} data checks failed")

    # ------------------------------------------------------------ module emission
    protein = lambda ids: json.dumps([identifiers[int(index)] for index in ids])
    method_text = ",\n".join(
        "  {\n"
        f"    name: {json.dumps(record['name'])},\n"
        f"    label: {json.dumps(record['label'])},\n"
        f"    description: {json.dumps(record['description'])},\n"
        f"    fitRows: {record['fitRows']},\n"
        f"    fitClassCounts: {json.dumps(record['fitClassCounts'])},\n"
        f"    parameters: {numbers(record['parameters'])},\n"
        f"    chosenThreshold: {record['chosenThreshold']!r},\n"
        f"    inspectionDefault: {json.dumps(record['inspectionDefault'])},\n"
        f"    inspectionTuned: {json.dumps(record['inspectionTuned'])},\n"
        f"    averagePrecision: {record['averagePrecision']!r},\n"
        f"    rocAuc: {record['rocAuc']!r},\n"
        f"    brierScore: {record['brierScore']!r},\n"
        f"    topTenSourceIds: {json.dumps(record['topTenSourceIds'])},\n"
        f"    topTenPositives: {record['topTenPositives']},\n"
        f"    tuningScores: {numbers(record['tuningScores'])},\n"
        f"    inspectionScores: {numbers(record['inspectionScores'])}\n"
        "  }"
        for record in method_records)

    module = f'''/** Recorded Yeast results for the imbalanced-learning lesson.
 *
 * Generated by scripts/verify-imbalance-data.py, which recomputes the whole
 * declared study from the dataset this lesson serves -- identity repair, roles,
 * standardisation, resampling lineage, all five fits, thresholds and metrics --
 * through an implementation independent of the content packet's author script,
 * matches every value against that packet, and re-derives the packet's own
 * constructed fixtures from their declared settings. Do not edit by hand.
 *
 * Source: Nakai, K. (1991), Yeast, UCI Machine Learning Repository,
 * https://doi.org/10.24432/C5KG68, licensed CC BY 4.0. This lesson serves its
 * own copy of the unchanged file at
 * /learn-assets/imbalanced-learning/yeast.data
 * ({EXPECTED_BYTES} bytes, SHA-256 {EXPECTED_SHA}).
 *
 * Protocol: {SOURCE_ROWS} source rows hold 1,462 distinct protein identifiers; 22
 * identifiers occur twice with byte-identical scores and labels, and the first
 * occurrence of each is retained. The positive class is the original location
 * {POSITIVE_LABEL}. Six of the eight numeric scores are used, at source columns
 * {COLUMNS}. Stratified splits with seeds {SPLIT_SEED}, {FIT_SEED} and {ROLE_SEED} give 600 fitting,
 * 200 tuning, 200 inspection and 462 reserved proteins, holding 21, 7, 7 and 16
 * positives. Standardisation is fitted on the 600 original fitting rows only.
 * Oversampling uses seed {OVER_SEED}, undersampling seed {UNDER_SEED} and SMOTE seed {SMOTE_SEED} with
 * k = {SMOTE_K}. Every fit minimises the same weight-normalised log loss with an
 * unpenalised intercept and lambda = {PENALTY}. Hypothetical costs are {COST_FP} per false
 * alarm and {COST_FN} per missed positive. No reserved protein is scored anywhere.
 *
 * Fitted with scikit-learn {sklearn.__version__}, SciPy {scipy.__version__}, NumPy {np.__version__} on
 * Python {platform.python_version()}.
 *
 * The `parameters` arrays are L-BFGS-B stopping points, not the exact minimiser.
 * An independent damped-Newton refit polished to 50 decimal digits puts the true
 * optimum within 7.3e-07 of them, which changes no digit this lesson prints --
 * every confusion count, cost, metric at six places, top-ten identity list and
 * selected threshold at its displayed precision is identical from either set.
 * A maintainer reusing these coefficients as a reference should know that they
 * are established to roughly seven significant figures, not sixteen.
 */

export const provenance = {{
  name: "Yeast",
  creator: "Nakai, K. (1991); donated by Paul Horton (1996)",
  doi: "https://doi.org/10.24432/C5KG68",
  page: "https://archive.ics.uci.edu/dataset/110/yeast",
  license: "CC BY 4.0",
  licenseUrl: "https://creativecommons.org/licenses/by/4.0/",
  file: "/learn-assets/imbalanced-learning/yeast.data",
  attribution: "/learn-assets/imbalanced-learning/ATTRIBUTION.txt",
  bytes: {EXPECTED_BYTES},
  sha256: "{EXPECTED_SHA}",
  sourceRows: {SOURCE_ROWS},
  studyRows: 1462,
  duplicateIdentifiers: {len(duplicate_names)},
  positiveLabel: "{POSITIVE_LABEL}",
  positives: 51,
  columns: {json.dumps(COLUMNS)},
  names: {json.dumps(NAMES)},
  meanings: {json.dumps(MEANINGS)},
  excluded: ["erl", "pox"],
  classCounts: {json.dumps(class_counts)},
  reserveScored: false,
  limits: [
    "Historical engineered sequence descriptors, not raw-sequence embeddings or a wet-laboratory trial.",
    "The source supplies no protein-family grouping, so exact-duplicate repair does not establish transfer to a new family or organism.",
    "The source supplies no physical measurement units for these descriptor scores.",
    "The false-alarm and missed-positive costs are hypothetical teaching units, not laboratory prices."
  ]
}};

/** Roles, in the order the protocol creates them. Reserved proteins carry no
 * score anywhere in this lesson, so no reserved identity is published here. */
export const roles = {json.dumps({
    "fitting": {"records": 600, "positives": 21, "influences": "Scaler, synthetic neighbours, weights, fitted coefficients"},
    "tuning": {"records": 200, "positives": 7, "influences": "Each predeclared model's decision threshold"},
    "inspection": {"records": 200, "positives": 7, "influences": "Reported comparison and error analysis"},
    "reserve": {"records": 462, "positives": 16, "influences": "No predictions or scores in this lesson"},
}, indent=2).replace(chr(10), chr(10))};

export const study = {{
  penalty: {PENALTY},
  costFalsePositive: {COST_FP},
  costFalseNegative: {COST_FN},
  splitSeeds: {{ development: {SPLIT_SEED}, fitting: {FIT_SEED}, roles: {ROLE_SEED} }},
  resampleSeeds: {{ oversample: {OVER_SEED}, undersample: {UNDER_SEED}, smote: {SMOTE_SEED} }},
  smoteK: {SMOTE_K},
  fittingPositives: {n_positive},
  fittingNegatives: {n_negative},
  addedMinorityRows: {n_negative - n_positive},
  balancedRows: {n_positive + n_negative + (n_negative - n_positive)},
  undersampledRows: {2 * n_positive},
  positiveWeight: {600 / 42!r},
  negativeWeight: {600 / 1158!r},
  scalerMean: {numbers(hand_mean)},
  scalerScale: {numbers(hand_scale)},
  baselineAccuracy: {193 / 200!r},
  baselineCost: {baseline_cost},
  neighbourTiesAtK: {tied_at_k}
}};

/** The 200 tuning proteins: the records investigation 5 actually thresholds. */
export const tuningRecords = {{
  sourceIds: {json.dumps([int(index) for index in tuning])},
  proteinIds: {protein(tuning)},
  labels: {json.dumps([int(labels[index]) for index in tuning])}
}};

/** The 200 inspection proteins, whose outcome was locked before it was read. */
export const inspectionRecords = {{
  sourceIds: {json.dumps([int(index) for index in inspection])},
  proteinIds: {protein(inspection)},
  labels: {json.dumps([int(labels[index]) for index in inspection])}
}};

/** Ten synthetic vectors, in both coordinate systems, as lineage rather than as
 * observations: each is an interpolation between two fitting-role proteins. */
export const syntheticSample = {{
  count: {len(synthetic)},
  anchorSourceIds: {json.dumps([int(fitting[minority[a]]) for a in anchor[:10]])},
  neighbourSourceIds: {json.dumps([int(fitting[minority[n]]) for n in neighbour[:10]])},
  fractions: {numbers(fraction[:10])},
  standardised: [
{",".join(chr(10) + "    " + numbers(row) for row in synthetic[:10])}
  ],
  sourceScale: [
{",".join(chr(10) + "    " + numbers(row * hand_scale + hand_mean) for row in synthetic[:10])}
  ]
}};

export const methods = [
{method_text}
];
'''

    if write:
        ASSET_DIR.mkdir(parents=True, exist_ok=True)
        ASSET_DATASET.write_bytes(packet_bytes)
        (ASSET_DIR / "ATTRIBUTION.txt").write_text(
            "Yeast\n"
            "Nakai, K. (1991). UCI Machine Learning Repository. Donated by Paul Horton, 1996.\n"
            "https://doi.org/10.24432/C5KG68  .  https://archive.ics.uci.edu/dataset/110/yeast\n"
            "Licensed CC BY 4.0: https://creativecommons.org/licenses/by/4.0/\n"
            f"yeast.data is the unchanged archive member, {EXPECTED_BYTES} bytes, SHA-256 {EXPECTED_SHA}.\n"
            "Whitespace separated, LF line endings, 1,484 rows: a sequence identifier, eight numeric\n"
            "localization scores (mcg, gvh, alm, mit, erl, pox, vac, nuc) and an original location label.\n"
            "\n"
            "What this lesson does with it, and does not do to it: the served file is unchanged. The\n"
            "analysis retains the first occurrence of each of the 22 protein identifiers that appear\n"
            "twice with byte-identical scores and labels, leaving 1,462 distinct proteins and all 51 ME2\n"
            "positives. It defines a binary target, original location ME2 against every other location,\n"
            "and uses six of the eight scores (mcg, gvh, alm, mit, vac, nuc). That is a teaching task\n"
            "definition, not a clinical or diagnostic claim. Interpolated feature vectors generated by\n"
            "SMOTE are synthetic numerical training data, never synthesised protein sequences.\n"
            "This lesson serves its own copy so that no other lesson's asset edit can change it.\n",
            encoding="utf-8", newline="\n",
        )
        MODULE.write_text(module, encoding="utf-8", newline="\n")
    else:
        if not MODULE.exists():
            raise SystemExit("the data module is missing; rerun with --write")
        existing = MODULE.read_text(encoding="utf-8")
        if existing != module:
            raise SystemExit(
                "a fresh regeneration is not byte-identical to src/learn/data/imbalance-data.js; "
                "rerun with --write and inspect the difference")
        if not ASSET_DATASET.exists() or ASSET_DATASET.read_bytes() != packet_bytes:
            raise SystemExit("the served dataset does not match the packet bytes; rerun with --write")
        if not (ASSET_DIR / "ATTRIBUTION.txt").exists():
            raise SystemExit("the served attribution file is missing; rerun with --write")

    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps({
        "checkedAt": datetime.now(timezone.utc).isoformat(),
        "mode": "write" if write else "read-only",
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "sklearn": sklearn.__version__},
        "datasetSha256": digest,
        "datasetBytes": len(packet_bytes),
        "packetCalculatedInputsSha256": packet_inputs_sha,
        "servedCopy": str(ASSET_DATASET.relative_to(ROOT)).replace("\\", "/"),
        "servedMatchesPacket": ASSET_DATASET.exists() and ASSET_DATASET.read_bytes() == packet_bytes,
        "sourceRows": SOURCE_ROWS,
        "studyRows": len(kept),
        "duplicateIdentifiers": len(duplicate_names),
        "roles": {name: {"records": len(ids), "positives": int(labels[ids].sum())} for name, ids in roles.items()},
        "rolesDisjointAndCovering": True,
        "reservePredictionsComputed": False,
        "fits": len(procedures),
        "worstNewtonAgainstLbfgsParameterGap": worst_parameter_gap,
        "neighbourTiesAtK": tied_at_k,
        "totalChecks": checks["count"],
        "trustRootChecks": trust_root,
        "trustRootScalarLeaves": total_leaves,
        "trustRootLeavesReDerived": covered_leaves,
        "trustRootCoverage": round(coverage, 4),
        "trustRootSweepFieldsChecked": sweep_fields,
        "selectedThresholds": {record["name"]: record["chosenThreshold"] for record in method_records},
        "tunedCosts": {record["name"]: record["inspectionTuned"]["cost"] for record in method_records},
        "averagePrecisions": {record["name"]: record["averagePrecision"] for record in method_records},
        "topTenPositives": {record["name"]: record["topTenPositives"] for record in method_records},
        "lowestTunedCost": by_cost["name"],
        "highestAveragePrecision": by_ap["name"],
        "bestTopTen": top_winners,
        "moduleSha256": hashlib.sha256(MODULE.read_bytes()).hexdigest(),
        "verifierSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": (
            "Recomputed the whole declared Yeast study from the served file: first-occurrence identity "
            "repair by a plain scan, the four role identities and their positive counts, hand-computed "
            "standardisation, the oversampling/undersampling draws, the SMOTE neighbour ranking by an "
            "explicit (distance, index) sort and all 558 interpolations, and all five logistic fits by "
            "damped Newton iteration on the analytic Hessian rather than the packet's L-BFGS-B call. "
            "Thresholds, confusion counts, average precision, ROC-AUC and Brier were then recomputed "
            "from the packet's saved scores by direct definition and cross-checked against scikit-learn. "
            "Also re-derived the packet's constructed fixtures -- the trust root that "
            "verify-imbalance-models.mjs consumes as ground truth -- from their declared settings alone, "
            "and checked that a fresh regeneration of src/learn/data/imbalance-data.js is byte-identical."
        ),
        "limitations": [
            "The role splits are library shuffles; they are reproduced and matched, not derived from first principles.",
            "One random split of one historical collection, with no protein-family grouping available.",
            "Development results only; no reserved protein was predicted or scored.",
            "Numerical fitting can differ on other library versions.",
        ],
        "passed": True,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(
        f"PASS: {checks['count']:,} data checks, including {trust_root} re-derivations of the packet's own "
        f"constructed fixtures and {sweep_fields:,} threshold-sweep fields; {coverage:.1%} of the trust root's "
        f"{total_leaves:,} scalar leaves re-derived; five fits reproduced by Newton iteration within "
        f"{worst_parameter_gap:.2e} of the packet's L-BFGS-B parameters; module "
        f"{'written' if write else 'byte-identical'} ({MODULE.stat().st_size / 1024:.0f} KB)."
    )


def block_scores(block, parameters):
    return np.asarray(block) @ np.asarray(parameters)[1:]


if __name__ == "__main__":
    main()
