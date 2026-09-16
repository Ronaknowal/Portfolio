"""Regenerate and verify the observed banknote study for the AutoML & NAS lesson.

Two separate jobs, both read-only unless given `--write`:

1.  **The observed study.** The declared campaign of `docs/teaching/drafts/
    automl-neural-architecture-search-nas/lesson.md` section 5 is recomputed from
    the dataset this lesson serves: 1,348 exact-feature groups, three fixed data
    roles, three group-respecting folds, eleven candidates, 33 fold fits and two
    refits. Every fold accuracy, every out-of-fold prediction, both inspection
    confusion matrices and the seed-75 replay order are matched against the
    content packet's `calculated-inputs.json` before
    `src/learn/data/automl-data.js` is rebuilt. Without `--write` the rebuilt
    module text must be byte-identical to the file already on disk.

2.  **The packet's own trust root.** `calculated-inputs.json` is what
    `scripts/verify-automl-models.mjs` checks the browser models against, so it
    cannot be the only witness for itself. Every value of its `constructed`
    block is re-derived here from the declared settings alone, through
    independent implementations: the normal CDF from `math.erf` rather than
    SciPy's `ndtr`, softmax by hand rather than `scipy.special.softmax`, the
    architecture gradient also by central difference, successive halving by an
    explicit loop, Pareto dominance by pairwise comparison, and the activation
    kernel from Hamming distances. `author-calculations.py` is never imported.

Run:
  scratch/lesson-tools/Scripts/python.exe scripts/verify-automl-data.py --write
  scratch/lesson-tools/Scripts/python.exe scripts/verify-automl-data.py
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import hashlib
import json
import re
import math
import platform
import sys
import warnings
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "docs/teaching/drafts/automl-neural-architecture-search-nas"
PACKET_DATASET = PACKET / "banknote-data.csv"
PACKET_INPUTS = PACKET / "calculated-inputs.json"
MANUSCRIPT = PACKET / "lesson.md"
ASSET_DIR = ROOT / "public/learn-assets/automl-nas"
ASSET_DATASET = ASSET_DIR / "banknote-data.csv"
MODULE = ROOT / "src/learn/data/automl-data.js"
EVIDENCE = ROOT / "docs/teaching/evidence/automl-data.json"

EXPECTED_SHA = "d0539aaed2139ba7a587b3e34fb345ce503ff7d5d33dbf9912d8e195ce425cb9"
EXPECTED_ARCHIVE_SHA = "1e2acd9a2085fadf3d8145c12d3d22af853320d52294a6590c2eaf75fdc05227"
EXPECTED_BYTES = 46400
EXPECTED_ROWS = 1372

DEVELOPMENT_GROUPS = 900
INSPECTION_GROUPS = 200
ROLE_SEED = 71
INSPECTION_SEED = 72
FOLD_SEED = 73
ESTIMATOR_SEED = 74
REPLAY_SEED = 75
BASELINE_INDEX = 3

FEATURE_NAMES = ["variance", "skewness", "kurtosis", "entropy"]
NL = chr(10)

checks: dict[str, int] = {}
failures: list[str] = []


def record(name: str) -> None:
    checks[name] = checks.get(name, 0) + 1


def expect(condition: bool, label: str) -> None:
    record(label.split(" · ")[0])
    if not condition:
        failures.append(label)


def close(actual: float, expected: float, label: str, tolerance: float = 1e-12) -> None:
    expect(abs(actual - expected) <= tolerance * max(1.0, abs(expected)),
           f"{label} · {actual!r} versus {expected!r}")


# --------------------------------------------------------------- declared space

def candidate_registry():
    """The eleven declared configurations, in the registry order that owns ties."""
    entries = []
    for scale in (False, True):
        for penalty in (0.1, 1.0):
            model = LogisticRegression(C=penalty, solver="lbfgs", max_iter=1000, tol=1e-8)
            if scale:
                model = make_pipeline(StandardScaler(), model)
            entries.append({
                "id": f"logistic-{'standard' if scale else 'raw'}-c{penalty:g}",
                "label": f"Logistic, {'standardized' if scale else 'raw'}, C = {penalty:g}",
                "family": "logistic", "scaled": scale,
                "settings": f"C = {penalty:g}, lbfgs, tol 1e-8",
                "parameterCount": None, "model": model,
            })
    for depth in (2, 5):
        entries.append({
            "id": f"tree-depth{depth}", "label": f"Tree, depth {depth}",
            "family": "tree", "scaled": False, "settings": f"maximum depth {depth}",
            "parameterCount": None,
            "model": DecisionTreeClassifier(max_depth=depth, random_state=ESTIMATOR_SEED),
        })
    for neighbors in (3, 9):
        entries.append({
            "id": f"neighbors-standard-k{neighbors}",
            "label": f"Standardized neighbors, k = {neighbors}",
            "family": "neighbors", "scaled": True, "settings": f"k = {neighbors}",
            "parameterCount": None,
            "model": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=neighbors)),
        })
    for widths in ((8,), (16,), (8, 8)):
        shown = ", ".join(str(width) for width in widths)
        entries.append({
            "id": "mlp-tanh-" + "x".join(map(str, widths)),
            "label": f"Standardized tanh network, width{'s' if len(widths) > 1 else ''} {shown}",
            "family": "mlp", "scaled": True,
            "settings": f"tanh, lbfgs, alpha 0.01, widths ({shown})",
            "parameterCount": sum((a + 1) * b for a, b in zip((4,) + widths, widths + (1,))),
            "model": make_pipeline(StandardScaler(), MLPClassifier(
                hidden_layer_sizes=widths, activation="tanh", solver="lbfgs",
                alpha=0.01, max_iter=1000, tol=1e-7, random_state=ESTIMATOR_SEED)),
        })
    return entries


def data_roles(features, labels):
    """Exact-feature groups, three disjoint roles and three group-respecting folds."""
    unique, first, group, counts = np.unique(
        features, axis=0, return_index=True, return_inverse=True, return_counts=True)
    for identifier in np.flatnonzero(counts > 1):
        if len(np.unique(labels[group == identifier])) != 1:
            raise ValueError("Inspect conflicting labels within an identical-feature group.")
    group_labels = labels[first]
    development_groups, rest = train_test_split(
        np.arange(len(unique)), train_size=DEVELOPMENT_GROUPS,
        stratify=group_labels, random_state=ROLE_SEED)
    inspection_groups, reserve_groups = train_test_split(
        rest, train_size=INSPECTION_GROUPS,
        stratify=group_labels[rest], random_state=INSPECTION_SEED)

    def rows(group_ids):
        return np.flatnonzero(np.isin(group, group_ids))

    development, inspection, reserve = map(rows, (development_groups, inspection_groups, reserve_groups))
    splitter = StratifiedKFold(3, shuffle=True, random_state=FOLD_SEED)
    folds = [(rows(development_groups[a]), rows(development_groups[b]))
             for a, b in splitter.split(development_groups, group_labels[development_groups])]
    return group, counts, development, inspection, reserve, folds, development_groups


# ------------------------------------------------- independent constructed maths

def normal_cdf(value: float) -> float:
    """Phi by the error function, independent of SciPy's ndtr."""
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def normal_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def expected_improvement(best: float, mean: float, deviation: float) -> float:
    if deviation == 0:
        return max(best - mean, 0.0)
    z = (best - mean) / deviation
    return (best - mean) * normal_cdf(z) + deviation * normal_pdf(z)


def softmax(logits):
    peak = max(logits)
    weights = [math.exp(value - peak) for value in logits]
    total = sum(weights)
    return [weight / total for weight in weights]


def mixture_state(logits, outputs, target):
    probabilities = softmax(logits)
    mixed = sum(p * o for p, o in zip(probabilities, outputs))
    loss = 0.5 * (mixed - target) ** 2
    gradient = [(mixed - target) * p * (o - mixed) for p, o in zip(probabilities, outputs)]
    return probabilities, mixed, loss, gradient


def halving_schedule(curves, keep_fraction=3):
    """Explicit rung loop: rank on the purchased column only, keep the best third."""
    alive = list(range(len(curves)))
    survivors = [alive.copy()]
    for rung in range(len(curves[0]) - 1):
        alive = sorted(alive, key=lambda index: (curves[index][rung], index))[:len(alive) // keep_fraction]
        survivors.append(alive.copy())
    return survivors


def pareto_front(latency, accuracy):
    """Dominated: another point is no worse in both and strictly better in one."""
    front = []
    for i in range(len(latency)):
        dominated = any(
            latency[j] <= latency[i] and accuracy[j] >= accuracy[i]
            and (latency[j] < latency[i] or accuracy[j] > accuracy[i])
            for j in range(len(latency)) if j != i)
        if not dominated:
            front.append(i)
    return front


def hamming(left: str, right: str) -> int:
    return sum(1 for a, b in zip(left, right) if a != b)


def activation_kernel(codes):
    width = len(codes[0])
    return [[width - hamming(a, b) for b in codes] for a in codes]


def determinant_two(matrix):
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def leaf_paths(node, prefix=""):
    """Every scalar leaf of a nested dict/list, as a dotted path."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from leaf_paths(value, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from leaf_paths(value, f"{prefix}[{index}]")
    else:
        yield prefix


def rederive_trust_root(constructed, covered: set) -> int:
    """Every value of the packet's constructed block, from its declared settings.

    `covered` accumulates the dotted leaf paths this function actually asserts,
    so the evidence can report coverage of the block rather than a hand-kept
    tally of check calls. A tally counted the same three expected-improvement
    leaves twice and credited checks that read no packet key at all.
    """
    count = 0

    def mark(*paths):
        """Record leaf paths the preceding block genuinely pins.

        A path naming a subtree records every leaf beneath it, so a block that
        checks a list element by element gets credit for each element and a
        block that reads one scalar gets credit for one.
        """
        for path in paths:
            node = constructed
            try:
                for part in path.split("."):
                    node = node[part]
            except (KeyError, TypeError):
                raise AssertionError(f"coverage claimed for a path that is not in the packet: {path}")
            for leaf in leaf_paths(node, path):
                covered.add(leaf)

    # Conditional grammar: alternatives add, simultaneously active settings multiply.
    counts = constructed["conditional_counts"]
    expect(counts["logistic"] == 2 * 2, "trust root · logistic branch is 2 scalings by 2 penalties")
    expect(counts["tree"] == 2 and counts["neighbors"] == 2 and counts["mlp"] == 3,
           "trust root · the other three branches")
    expect(counts["total"] == counts["logistic"] + counts["tree"] + counts["neighbors"] + counts["mlp"] == 11,
           "trust root · eleven configurations by addition across branches")
    count += 4
    mark("conditional_counts")

    # Expected improvement, through math.erf rather than SciPy.
    for label, mean, deviation in [("A", 0.35, 0.02), ("B", 0.40, 0.20), ("C", 0.50, 0.0)]:
        close(expected_improvement(0.4, mean, deviation), constructed["ei"][label],
              f"trust root · expected improvement {label}", 1e-12)
        count += 1
    # A σ = 0 candidate that would improve is its deterministic improvement.
    close(expected_improvement(0.4, 0.3, 0.0), 0.1, "trust root · deterministic improvement", 1e-15)
    # Adding a common offset to the incumbent and every mean leaves EI alone.
    for label, mean, deviation in [("A", 0.35, 0.02), ("B", 0.40, 0.20), ("C", 0.50, 0.0)]:
        close(expected_improvement(0.7, mean + 0.3, deviation), constructed["ei"][label],
              f"trust root · common-offset null for {label}", 1e-12)
        count += 1
    count += 1
    mark("ei")

    # The operation mixture, its gradient, and a central-difference cross-check.
    mixture = constructed["mixture"]
    logits, outputs, target = mixture["logits"], mixture["operation_outputs"], mixture["target"]
    probabilities, mixed, loss, gradient = mixture_state(logits, outputs, target)
    for index, value in enumerate(probabilities):
        close(value, mixture["probabilities"][index], f"trust root · mixture probability {index}", 1e-14)
    close(mixed, mixture["output"], "trust root · mixed output", 1e-14)
    close(loss, mixture["half_squared_loss"], "trust root · mixture loss", 1e-14)
    for index, value in enumerate(gradient):
        close(value, mixture["gradient"][index], f"trust root · mixture gradient {index}", 1e-12)
        step = 1e-6
        moved_up = list(logits)
        moved_down = list(logits)
        moved_up[index] += step
        moved_down[index] -= step
        numerical = (mixture_state(moved_up, outputs, target)[2]
                     - mixture_state(moved_down, outputs, target)[2]) / (2 * step)
        close(numerical, value, f"trust root · central difference at logit {index}", 1e-6)
    updated = [value - mixture["step_size"] * slope for value, slope in zip(logits, gradient)]
    for index, value in enumerate(updated):
        close(value, mixture["updated_logits"][index], f"trust root · updated logit {index}", 1e-12)
    close(mixture_state(updated, outputs, target)[1], mixture["updated_output"],
          "trust root · output after one architecture step", 1e-12)
    for index, value in enumerate(softmax([value + 5 for value in logits])):
        close(value, mixture["translation_null_probabilities"][index],
              f"trust root · common-logit null {index}", 1e-14)
    count += 3 * len(gradient) + 8
    mark("mixture.probabilities", "mixture.output", "mixture.half_squared_loss", "mixture.gradient",
         "mixture.updated_logits", "mixture.updated_output", "mixture.translation_null_probabilities",
         "mixture.logits", "mixture.operation_outputs", "mixture.target", "mixture.step_size")

    # Discretization: the mixture can beat either committed operation.
    discrete = constructed["discretization"]
    _, committed_free, mixed_loss, _ = mixture_state([0.0, 0.0], discrete["outputs"], discrete["target"])
    close(mixed_loss, discrete["mixed_loss"], "trust root · the mixture's loss is zero", 1e-15)
    close(committed_free, 0.0, "trust root · and its output is zero", 1e-15)
    for output in discrete["outputs"]:
        close(0.5 * (output - discrete["target"]) ** 2, discrete["selected_loss"],
              "trust root · either committed operation costs two", 1e-15)
    count += 3
    # The discretization block's own probabilities, re-derived rather than echoed.
    discrete_probabilities = softmax([0.0, 0.0])
    for index, value in enumerate(discrete_probabilities):
        close(value, discrete["probabilities"][index],
              f"trust root · discretization probability {index}", 1e-15)
    mark("discretization")

    # The scalar bilevel example: three derivatives of three different functions.
    scalar = constructed["bilevel_scalar"]
    alpha, weight, xi = scalar["alpha"], scalar["w"], scalar["xi"]
    stepped = weight - xi * (weight - alpha)
    close(stepped, scalar["w_prime"], "trust root · one-step weight", 1e-15)
    close(0.0, scalar["first_order_gradient"], "trust root · the direct derivative is zero", 1e-15)
    close((stepped - 1) * xi, scalar["one_step_gradient"], "trust root · one-step outer derivative", 1e-14)
    close(alpha, scalar["exact_inner_optimum"], "trust root · exact inner optimum", 1e-15)
    close(alpha - 1, scalar["exact_outer_gradient"], "trust root · exact outer derivative", 1e-15)
    # A central difference of the one-step outer objective in alpha.
    def one_step_validation(a: float) -> float:
        return 0.5 * ((weight - xi * (weight - a)) - 1) ** 2
    close((one_step_validation(alpha + 1e-6) - one_step_validation(alpha - 1e-6)) / 2e-6,
          scalar["one_step_gradient"], "trust root · one-step derivative by central difference", 1e-7)
    count += 6
    mark("bilevel_scalar")

    # Successive halving on the nine declared curves.
    halving = constructed["halving"]
    curves = halving["curves"]
    survivors = halving_schedule(curves)
    expect(survivors == halving["survivors"], f"trust root · survivor sets {survivors}")
    expect(survivors[1] == [0, 1, 2], "trust root · A, B and C survive the first rung")
    expect(survivors[2] == [2], "trust root · C alone survives the second")
    close(curves[survivors[-1][0]][2], halving["selected_final_loss"], "trust root · selected final loss", 1e-15)
    best_full = min(range(len(curves)), key=lambda index: (curves[index][2], index))
    expect(best_full == halving["full_budget_best"], "trust root · D would win at full resource")
    expect(best_full not in survivors[1], "trust root · and it was eliminated at the first rung")
    resource = halving["resource"]
    sizes = [len(stage) for stage in survivors]
    restart = sum(size * unit for size, unit in zip(sizes, resource))
    resume = sizes[0] * resource[0] + sum(
        size * (unit - previous) for size, unit, previous in zip(sizes[1:], resource[1:], resource[:-1]))
    expect(restart == halving["restart_cost"] == 27, f"trust root · restart work {restart}")
    expect(resume == halving["resume_cost"] == 21, f"trust root · continuation work {resume}")
    expect(len(curves) * resource[-1] == halving["all_full_cost"] == 81, "trust root · all-full work")
    # The slow starter really is rescued by the single edited value.
    rescued = [list(row) for row in curves]
    rescued[3][0] = 0.09
    rescued_survivors = halving_schedule(rescued)
    expect(sorted(rescued_survivors[1]) == [0, 1, 3] and rescued_survivors[2] == [3],
           f"trust root · D survives once its first rung reads 0.09 {rescued_survivors}")
    close(rescued[rescued_survivors[-1][0]][2], 0.02, "trust root · and finishes at 0.02", 1e-15)
    # A uniform offset changes every label and no decision.
    shifted = [[value + 0.05 for value in row] for row in curves]
    expect(halving_schedule(shifted) == survivors, "trust root · a uniform offset is a decision null")
    count += 11
    # `halving.selected` is the index the author's own loop finished on; assert it
    # rather than leaving it the one leaf nothing in the repository reads.
    expect(halving["selected"] == survivors[-1][0],
           f"trust root · the recorded survivor index {halving['selected']}")
    expect(curves[halving["selected"]] == curves[2],
           "trust root · which is candidate C's curve")
    mark("halving.survivors", "halving.selected", "halving.selected_final_loss", "halving.full_budget_best",
         "halving.restart_cost", "halving.resume_cost", "halving.all_full_cost", "halving.resource")

    # Portfolio: the best single default is not the best pair.
    portfolio = constructed["portfolio"]
    old = portfolio["old_task_losses"]
    means = [sum(row) / len(row) for row in old]
    close(min(means), constructed["additional_checks"]["old_task_best_single_loss"],
          "trust root · best single old-task mean", 1e-15)
    expect(means.index(min(means)) == 2, "trust root · configuration C is that default")
    pairs = {frozenset(pair): sum(min(old[i][task] for i in pair) for task in range(2)) / 2
             for pair in combinations(range(3), 2)}
    close(min(pairs.values()), constructed["additional_checks"]["old_task_portfolio_best_loss"],
          "trust root · the complementary pair's mean best loss", 1e-15)
    expect(min(pairs, key=pairs.get) == frozenset({0, 1}), "trust root · and that pair is A with B")
    new = portfolio["new_task_losses"]
    close(min(new[0], new[1]), constructed["additional_checks"]["new_task_portfolio_best_loss"],
          "trust root · the old portfolio's new-task loss", 1e-15)
    close(min(new), constructed["additional_checks"]["new_task_alternative_loss"],
          "trust root · which the excluded configuration beats", 1e-15)
    count += 6
    mark("portfolio")

    # Pareto frontier and the hard cap.
    pareto = constructed["pareto"]
    latency, accuracy, labels = pareto["latency_ms"], pareto["accuracy"], pareto["labels"]
    front = [labels[index] for index in pareto_front(latency, accuracy)]
    expect(front == pareto["nondominated"] == ["A", "B", "C"], f"trust root · frontier {front}")
    feasible = [index for index in range(len(latency)) if latency[index] <= 5]
    best = max(feasible, key=lambda index: (accuracy[index], -latency[index]))
    expect(labels[best] == "B", "trust root · a 5 ms cap selects B")
    tight = [index for index in range(len(latency)) if latency[index] <= 3]
    expect(labels[max(tight, key=lambda index: (accuracy[index], -latency[index]))] == "A",
           "trust root · a 3 ms cap selects A")
    expect([index for index in range(len(latency)) if latency[index] <= 1] == [],
           "trust root · a 1 ms cap leaves nothing feasible")
    count += 4
    mark("pareto.nondominated")

    # The remaining recorded practice and proxy calculations.
    extra = constructed["additional_checks"]
    expect(extra["practice_conditional_count"] == 3 * 2 + 4 + 2 + 2 * 2 == 16,
           "trust root · the changed practice space has sixteen configurations")
    expect(extra["practice_cv_fits"] == 16 * 4 == 64, "trust root · and four-fold CV needs 64 fits")
    practice_curves = [[0.10, 0.09, 0.08], [0.11, 0.08, 0.07], [0.12, 0.07, 0.02], [0.20, 0.18, 0.15]]
    practice_survivors = halving_schedule(practice_curves, keep_fraction=2)
    expect(practice_survivors[1] == [0, 1] and practice_survivors[2] == [1],
           "trust root · practice 3 keeps A and B, then B")
    expect(extra["practice_halving_restart"] == 4 * 1 + 2 * 2 + 1 * 4 == 12, "trust root · practice 3 restart work")
    expect(extra["practice_halving_continue"] == 4 * 1 + 2 * 1 + 1 * 2 == 8, "trust root · practice 3 continuation work")
    expect(extra["practice_neural_parameter_blocks"] == [36, 21, 4]
           and sum(extra["practice_neural_parameter_blocks"]) == 61,
           "trust root · practice 4 parameter blocks total 61")
    expect((5 + 1) * 12 + (12 + 1) == 85, "trust root · and its one-layer alternative holds 85")
    close(expected_improvement(0.3, 0.25, 0.0), extra["practice_ei"][0], "trust root · practice 7 deterministic EI", 1e-15)
    close(expected_improvement(0.3, 0.3, 0.1), extra["practice_ei"][1], "trust root · practice 7 Gaussian EI", 1e-12)
    close(extra["practice_ei"][1], 0.1 / math.sqrt(2 * math.pi), "trust root · which is sigma over root two pi", 1e-12)
    expect(extra["practice_ei"][0] > extra["practice_ei"][1], "trust root · so U wins that comparison")
    probabilities, mixed, _, gradient = mixture_state([0.0, 0.0], [3.0, -1.0], 0.0)
    close(mixed, extra["practice_mixture_output"], "trust root · practice 8 mixed output", 1e-15)
    for index, value in enumerate(gradient):
        close(value, extra["practice_mixture_gradient"][index], f"trust root · practice 8 gradient {index}", 1e-14)
    shifted_probabilities, shifted_mixed, _, shifted_gradient = mixture_state([7.0, 7.0], [3.0, -1.0], 0.0)
    close(shifted_mixed, mixed, "trust root · adding 7 to both logits changes nothing", 1e-14)
    expect(all(abs(a - b) < 1e-14 for a, b in zip(shifted_gradient, gradient)),
           "trust root · nor the practice 8 gradient")
    practice_bilevel = extra["practice_bilevel"]
    close(0.0, practice_bilevel["direct"], "trust root · practice 9 direct derivative", 1e-15)
    close((0.0 - 0.2 * (0.0 - 0.5) - 2) * 0.2, practice_bilevel["one_step"],
          "trust root · practice 9 one-step derivative", 1e-14)
    close(0.5 - 2, practice_bilevel["exact"], "trust root · practice 9 exact derivative", 1e-15)
    close((0.2 - 1) * 0.1, extra["stationary_one_step_gradient"],
          "trust root · a zero training gradient still leaves a nonzero outer derivative", 1e-14)
    kernel = activation_kernel(["110", "101"])
    expect(kernel == extra["naswot_kernel"] == [[3, 1], [1, 3]], f"trust root · activation kernel {kernel}")
    expect(hamming("110", "101") == 2, "trust root · the two codes differ in two positions")
    expect(determinant_two(kernel) == extra["naswot_determinant"] == 8, "trust root · its determinant is 8")
    close(math.log(8), extra["naswot_log_determinant"], "trust root · and its log determinant is ln 8", 1e-15)
    singular = activation_kernel(["110", "110"])
    expect(singular == [[3, 3], [3, 3]] and determinant_two(singular) == extra["identical_code_determinant"] == 0,
           "trust root · identical codes give a singular kernel")
    count += 24
    mark("additional_checks")
    return count


def manuscript_table(text, header_fragment):
    """The data rows of the one markdown table whose header contains a fragment.

    Located by header rather than by row shape: three different tables in this
    manuscript have the shape `| X | n | n | n |`, so a row pattern alone picks
    up the wrong one.
    """
    lines = text.split(NL)
    for index, line in enumerate(lines):
        if line.startswith("|") and header_fragment in line:
            rows = []
            for candidate in lines[index + 2:]:
                if not candidate.startswith("|"):
                    break
                rows.append([cell.strip() for cell in candidate.strip("|").split("|")])
            return rows
    return []


def crosscheck_manuscript(packet, records, roles_rows, covered: set) -> int:
    """Pin the manuscript's own printed numbers against the packet.

    The displayed-program verifier hashes `lesson.md` and extracts its code
    fences; nothing read its prose. So the values this file's other checks call
    "as the manuscript prints it" were hand-transcribed literals, and a
    manuscript number that drifted from the packet would have failed nothing.
    Here the tables are parsed out of the manuscript itself.

    This also closes the other half of the trust-root coverage question: the
    halving curves, the portfolio matrix and the Pareto points are *inputs* to
    the re-derivations above, so those cannot pin them. The manuscript can.
    """
    text = MANUSCRIPT.read_text(encoding="utf-8")
    constructed = packet["constructed"]
    pinned = 0

    def mark(path):
        node = constructed
        for part in path.split("."):
            node = node[part]
        for leaf in leaf_paths(node, path):
            covered.add(leaf)

    # Section 3's nine fidelity curves.
    curves = manuscript_table(text, "Loss at 1 unit")
    expect(len(curves) == 9, f"manuscript · nine declared curves, found {len(curves)}")
    for index, row in enumerate(curves):
        expect(row[0] == "ABCDEFGHI"[index], f"manuscript · halving row {index} is {row[0]}")
        for column in range(3):
            close(float(row[column + 1]), constructed["halving"]["curves"][index][column],
                  f"manuscript · halving {row[0]} at rung {column + 1}", 1e-12)
            pinned += 1
    mark("halving.curves")

    # Section 2's constructed surrogate table.
    improvements = manuscript_table(text, "Predicted mean loss")
    expect(len(improvements) == 3, f"manuscript · three surrogate candidates, found {len(improvements)}")
    for row in improvements:
        close(float(row[3]), constructed["ei"][row[0]], f"manuscript · expected improvement {row[0]}", 1e-5)
        close(expected_improvement(0.4, float(row[1]), float(row[2])), constructed["ei"][row[0]],
              f"manuscript · {row[0]}'s stated mean and deviation reproduce its improvement", 1e-12)
        pinned += 1

    # Section 6's constructed portfolio matrix.
    portfolio = manuscript_table(text, "Old task 1")
    expect(len(portfolio) == 3, f"manuscript · three portfolio configurations, found {len(portfolio)}")
    for index, row in enumerate(portfolio):
        close(float(row[1]), constructed["portfolio"]["old_task_losses"][index][0],
              f"manuscript · portfolio {row[0]} on old task 1", 1e-12)
        close(float(row[2]), constructed["portfolio"]["old_task_losses"][index][1],
              f"manuscript · portfolio {row[0]} on old task 2", 1e-12)
        close(float(row[3]), (float(row[1]) + float(row[2])) / 2,
              f"manuscript · portfolio {row[0]} mean", 1e-12)
        pinned += 2
    new_task = re.search(r"losses A = (0\.\d+), B = (0\.\d+), C = (0\.\d+)", text)
    expect(new_task is not None, "manuscript · the new-task losses are stated")
    if new_task:
        for index in range(3):
            close(float(new_task.group(index + 1)), constructed["portfolio"]["new_task_losses"][index],
                  f"manuscript · new-task loss {index}", 1e-12)
            pinned += 1

    # Section 6's hypothetical deployment points, stated in prose.
    pareto = re.search(
        r"A has (\d+)% accuracy at (\d+) ms; B, (\d+)% at (\d+) ms; C, (\d+)% at (\d+) ms; "
        r"D, (\d+)% at (\d+) ms; E, (\d+)% at (\d+) ms", text)
    expect(pareto is not None, "manuscript · the five deployment points are stated")
    if pareto:
        for index in range(5):
            close(int(pareto.group(index * 2 + 1)) / 100, constructed["pareto"]["accuracy"][index],
                  f"manuscript · Pareto accuracy {index}", 1e-12)
            close(float(pareto.group(index * 2 + 2)), constructed["pareto"]["latency_ms"][index],
                  f"manuscript · Pareto latency {index}", 1e-12)
            expect(constructed["pareto"]["labels"][index] == "ABCDE"[index],
                   f"manuscript · Pareto candidate {index} is named {'ABCDE'[index]}")
            pinned += 3
        mark("pareto")

    # Section 5's eleven observed means, to the six decimals it prints.
    results = manuscript_table(text, "Mean fold accuracy")
    expect(len(results) == 11, f"manuscript · eleven candidate rows, found {len(results)}")
    for index, row in enumerate(results):
        close(float(row[1]), round(records[index]["meanFoldAccuracy"], 6),
              f"manuscript · candidate {index} mean", 1e-12)
        pinned += 1
        if row[2] != "—":
            expect(int(row[2]) == records[index]["parameterCount"],
                   f"manuscript · candidate {index} parameter count")
            pinned += 1

    # Section 5's role table.
    role_rows = manuscript_table(text, "Unique feature groups")
    expect(len(role_rows) == 3, f"manuscript · three roles, found {len(role_rows)}")
    for index, row in enumerate(role_rows):
        expect(int(row[1]) == roles_rows[index]["groups"], f"manuscript · {row[0]} groups")
        expect(int(row[2]) == roles_rows[index]["rows"], f"manuscript · {row[0]} rows")
        expect(int(row[3]) == roles_rows[index]["positives"], f"manuscript · {row[0]} class-1 rows")
        pinned += 3
    return pinned


def check_attribution(counts) -> int:
    """Everything the public attribution file asserts, against the data itself.

    Four verifiers hashed this file and checked it was reachable; nothing
    checked a single number inside it. A changed CSV would have left the
    public-facing attribution silently wrong with every check still green.
    """
    text = (ASSET_DIR / "ATTRIBUTION.txt").read_text(encoding="utf-8")
    served = (ASSET_DIR / "banknote-data.csv").read_bytes()
    pinned = 0
    claims = [
        (str(EXPECTED_BYTES), "the byte count"),
        (EXPECTED_SHA, "the file digest"),
        (EXPECTED_ARCHIVE_SHA, "the archive digest"),
        (f"{EXPECTED_ROWS:,}", "the row count"),
        (f"{len(counts):,}", "the distinct-vector count"),
        (str(int((counts > 1).sum())), "the repeated-vector count"),
        (str(int(counts.sum()) - len(counts)), "the extra-row count"),
        ("CC BY 4.0", "the licence"),
        ("10.24432/C55P57", "the DOI"),
        ("Volker Lohweg", "the attributed author"),
        ("curtosis", "the source spelling note"),
        ("data_banknote_authentication.txt", "the archive member"),
        ("five columns", "the column count"),
        ("CRLF", "the line ending"),
    ]
    for needle, what in claims:
        expect(needle in text, f"attribution · {what} ({needle}) appears in ATTRIBUTION.txt")
        pinned += 1
    expect(len(served) == EXPECTED_BYTES, "attribution · and the stated byte count is the served file's size")
    expect(hashlib.sha256(served).hexdigest() == EXPECTED_SHA,
           "attribution · and the stated digest is the served file's digest")
    return pinned + 2


# ------------------------------------------------------------------ module text

def render_module(payload: dict) -> str:
    def dump(value, indent=2):
        return json.dumps(value, indent=indent, ensure_ascii=False).replace(NL, NL + " " * 0)

    lines = [
        "/** Recorded outcomes of the AutoML & NAS lesson's declared banknote study.",
        " *",
        " * Generated by scripts/verify-automl-data.py, which recomputes the whole",
        " * campaign from the dataset this lesson serves and matches every value",
        " * against the content packet before writing this file. Do not edit by hand.",
        " *",
        " * Source: Volker Lohweg (2012), Banknote Authentication, UCI dataset 267,",
        " * https://doi.org/10.24432/C55P57, licensed CC BY 4.0. This lesson serves its",
        " * own copy of the unchanged comma-separated file at",
        " * /learn-assets/automl-nas/banknote-data.csv",
        f" * ({EXPECTED_BYTES} bytes, SHA-256 {EXPECTED_SHA}).",
        " *",
        " * Protocol: 1,348 exact-feature groups; 900 development, 200 inspection and",
        " * 248 reserved groups split with seeds 71 and 72; three group-stratified folds",
        " * with seed 73 inside development. Every candidate sees the same folds, and a",
        " * pipeline refits its scaler on each fold's fitting rows. Selection maximizes",
        " * the arithmetic mean of the three fold accuracies; an exact tie takes the",
        " * first registry entry. After 33 fold fits, the selected candidate and the",
        " * predeclared standardized logistic C = 1 are refitted on all development rows",
        " * and predicted once on inspection: 35 estimator fits in total. No reserved row",
        " * receives a prediction here or in the packet.",
        " *",
        f" * Fitted with scikit-learn {sklearn.__version__} on Python {platform.python_version()}, NumPy {np.__version__}.",
        " */",
        "",
    ]
    for name, value in payload.items():
        lines.append(f"export const {name} = {dump(value)};")
        lines.append("")
    return NL.join(lines)


# ------------------------------------------------------------------------- main

def main() -> None:
    write = "--write" in sys.argv
    packet = json.loads(PACKET_INPUTS.read_text(encoding="utf-8"))

    served = ASSET_DATASET.read_bytes()
    packet_bytes = PACKET_DATASET.read_bytes()
    expect(hashlib.sha256(served).hexdigest() == EXPECTED_SHA, "dataset · served file hash")
    expect(len(served) == EXPECTED_BYTES, "dataset · served file size")
    expect(served == packet_bytes, "dataset · the served bytes are the packet's bytes")
    source_record = json.loads((PACKET / "data-source.json").read_text(encoding="utf-8"))
    expect(source_record["data_sha256"] == EXPECTED_SHA
           and source_record["archive_sha256"] == EXPECTED_ARCHIVE_SHA,
           "dataset · the packet's own ingestion record agrees")

    raw = np.loadtxt(ASSET_DATASET, delimiter=",")
    features, labels = raw[:, :4], raw[:, 4].astype(int)
    expect(raw.shape == (EXPECTED_ROWS, 5), f"dataset · shape {raw.shape}")
    expect([int((labels == 0).sum()), int((labels == 1).sum())] == source_record["class_counts"],
           "dataset · class counts")

    group, counts, development, inspection, reserve, folds, development_groups = data_roles(features, labels)
    expect(len(counts) == 1348, "roles · 1,348 exact-feature groups")
    expect(int((counts > 1).sum()) == 11, "roles · eleven of them repeat")
    expect(int(counts.sum()) - len(counts) == 24, "roles · accounting for 24 further rows")
    expect(group.tolist() == packet["groups"]["source_row_group"], "roles · group assignment matches the packet")
    expect(counts.tolist() == packet["groups"]["counts"], "roles · and so do the group sizes")
    for name, ids in [("development", development), ("inspection", inspection), ("reserve", reserve)]:
        expect(ids.tolist() == packet["roles"][f"{name}_ids" if name != "reserve" else "reserve_ids"],
               f"roles · {name} row ids match the packet")
        expect(len(ids) == packet["roles"]["row_counts"][name], f"roles · {name} row count")
        expect(int(labels[ids].sum()) == packet["roles"]["positive_counts"][name], f"roles · {name} class-1 count")
    expect(len(set(development.tolist()) & set(inspection.tolist())) == 0
           and len(set(development.tolist()) & set(reserve.tolist())) == 0
           and len(set(inspection.tolist()) & set(reserve.tolist())) == 0,
           "roles · the three roles are row-disjoint")
    expect(sorted(development.tolist() + inspection.tolist() + reserve.tolist()) == list(range(EXPECTED_ROWS)),
           "roles · and together they are every source row")
    role_of = {}
    for name, ids in [("development", development), ("inspection", inspection), ("reserve", reserve)]:
        for row in ids.tolist():
            role_of[row] = name
    expect(all(len({role_of[row] for row in np.flatnonzero(group == identifier).tolist()}) == 1
               for identifier in range(len(counts))),
           "roles · no exact-feature group is split across two roles")

    fold_rows = []
    for index, (fitting, validation) in enumerate(folds):
        expect(fitting.tolist() == packet["roles"]["folds"][index]["fitting_ids"], f"folds · fold {index + 1} fitting ids")
        expect(validation.tolist() == packet["roles"]["folds"][index]["validation_ids"], f"folds · fold {index + 1} validation ids")
        expect(len(set(fitting.tolist()) & set(validation.tolist())) == 0, f"folds · fold {index + 1} is disjoint")
        expect(sorted(fitting.tolist() + validation.tolist()) == sorted(development.tolist()),
               f"folds · fold {index + 1} covers exactly the development pool")
        expect(all(len({("fit" if row in set(fitting.tolist()) else "val")
                        for row in np.flatnonzero(group == identifier).tolist()}) == 1
                   for identifier in np.unique(group[development]).tolist()),
               f"folds · fold {index + 1} keeps every feature group on one side")
        fold_rows.append(len(validation))
    expect(fold_rows == [314, 300, 305], f"folds · validation row counts {fold_rows}")

    registry = candidate_registry()
    expect([entry["id"] for entry in registry] == [entry["id"] for entry in packet["candidates"]],
           "candidates · registry order matches the packet")

    records = []
    caught_warnings: list[str] = []
    for position, entry in enumerate(registry):
        recorded = packet["candidates"][position]
        out_of_fold = np.full(len(labels), -1, dtype=int)
        fold_accuracy, fold_correct = [], []
        for fitting, validation in folds:
            model = clone(entry["model"])
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model.fit(features[fitting], labels[fitting])
            caught_warnings.extend(str(item.message) for item in caught)
            prediction = model.predict(features[validation])
            out_of_fold[validation] = prediction
            fold_accuracy.append(float(accuracy_score(labels[validation], prediction)))
            fold_correct.append(int((prediction == labels[validation]).sum()))
        mean_accuracy = float(np.mean(fold_accuracy))
        pooled = out_of_fold[development]
        expect(pooled.tolist() == recorded["oof_predictions"],
               f"candidates · {entry['id']} reproduces all 919 out-of-fold predictions")
        for index, value in enumerate(fold_accuracy):
            close(value, recorded["fold_accuracy"][index], f"candidates · {entry['id']} fold {index + 1}")
        close(mean_accuracy, recorded["mean_fold_accuracy"], f"candidates · {entry['id']} mean fold accuracy")
        close(float(accuracy_score(labels[development], pooled)), recorded["pooled_oof_accuracy"],
              f"candidates · {entry['id']} pooled out-of-fold accuracy")
        # The mean of fold accuracies and pooled accuracy are different rules.
        for index, correct in enumerate(fold_correct):
            close(correct / fold_rows[index], fold_accuracy[index],
                  f"candidates · {entry['id']} fold {index + 1} numerator over denominator")
        expect(recorded["warnings"] == [], f"candidates · {entry['id']} fitted without a warning")
        errors = [int(row) for row in development[pooled != labels[development]].tolist()]
        records.append({
            "id": entry["id"], "label": entry["label"], "family": entry["family"],
            "scaled": entry["scaled"], "settings": entry["settings"],
            "parameterCount": entry["parameterCount"],
            "foldCorrect": fold_correct, "foldRows": fold_rows,
            "foldAccuracy": fold_accuracy, "meanFoldAccuracy": mean_accuracy,
            "pooledCorrect": int((pooled == labels[development]).sum()),
            "pooledRows": int(len(development)),
            "pooledOofAccuracy": float(accuracy_score(labels[development], pooled)),
            "outOfFoldErrorRows": errors,
        })
    expect(caught_warnings == [], f"candidates · the whole campaign fitted without a warning {caught_warnings}")

    means = [entry["meanFoldAccuracy"] for entry in records]
    selected = max(range(len(records)), key=lambda index: means[index])
    expect(records[selected]["id"] == packet["selected_id"] == "mlp-tanh-16",
           "selection · the width-16 network wins the declared criterion")
    expect(selected == 9, "selection · at registry index 9")
    expect(means[selected] == 1.0, "selection · with mean fold accuracy exactly one")
    tied = [records[index]["id"] for index in range(len(records)) if means[index] == max(means[:selected] + means[selected + 1:])]
    expect(sorted(tied) == ["mlp-tanh-8", "mlp-tanh-8x8", "neighbors-standard-k3"],
           f"selection · three candidates share the runner-up score {tied}")
    for identifier in tied:
        entry = next(item for item in records if item["id"] == identifier)
        expect(entry["outOfFoldErrorRows"] == [349],
               f"selection · {identifier} misses exactly source row 349")
        close(entry["meanFoldAccuracy"], 0.9989071038251366, f"selection · {identifier} mean", 1e-15)
        close(entry["pooledOofAccuracy"], 918 / 919, f"selection · {identifier} pooled accuracy", 1e-15)
    expect(records[10]["meanFoldAccuracy"] < 1.0 and records[10]["parameterCount"] == 121,
           "selection · the deeper 121-parameter network does not win")
    expect([records[8]["parameterCount"], records[9]["parameterCount"]] == [49, 97],
           "selection · the network parameter counts are 49, 97 and 121")

    final = []
    for role, index in (("selected", selected), ("declared_baseline", BASELINE_INDEX)):
        model = clone(registry[index]["model"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(features[development], labels[development])
        expect([str(item.message) for item in caught] == [], f"inspection · {role} refit without a warning")
        prediction = model.predict(features[inspection])
        recorded = next(item for item in packet["inspection"] if item["role"].startswith(role.split("_")[0]))
        expect(prediction.tolist() == recorded["predictions"], f"inspection · {role} reproduces every prediction")
        matrix = confusion_matrix(labels[inspection], prediction, labels=[0, 1]).tolist()
        expect(matrix == recorded["confusion"], f"inspection · {role} confusion matrix {matrix}")
        close(float(accuracy_score(labels[inspection], prediction)), recorded["accuracy"], f"inspection · {role} accuracy")
        errors = [int(row) for row in inspection[prediction != labels[inspection]].tolist()]
        final.append({
            "role": role, "id": registry[index]["id"], "label": registry[index]["label"],
            "correct": int((prediction == labels[inspection]).sum()), "rows": int(len(inspection)),
            "accuracy": float(accuracy_score(labels[inspection], prediction)),
            "confusion": matrix, "errorRows": errors,
        })
    expect(final[0]["confusion"] == [[114, 0], [0, 91]], "inspection · the selected network makes no mistake")
    expect(final[1]["confusion"] == [[111, 3], [0, 91]], "inspection · the baseline misses three class-0 rows")
    expect(final[1]["errorRows"] == [107, 195, 345], f"inspection · at rows {final[1]['errorRows']}")
    expect(sum(final[0]["confusion"][0]) + sum(final[0]["confusion"][1]) == 205, "inspection · over 205 rows")

    majority_class = int(np.argmax(np.bincount(labels[development])))
    majority_correct = int((labels[inspection] == majority_class).sum())
    close(majority_correct / len(inspection), packet["always_majority_inspection_accuracy"],
          "inspection · the always-majority context baseline")
    expect(majority_class == 0 and majority_correct == 114, "inspection · which gets 114 of 205 right")

    replay = np.random.default_rng(REPLAY_SEED).permutation(len(registry)).tolist()
    expect(replay == packet["search_order"] == [7, 10, 6, 9, 3, 2, 1, 4, 8, 0, 5],
           f"replay · the seed-75 reveal order {replay}")
    running = np.maximum.accumulate([means[index] for index in replay]).tolist()
    for index, value in enumerate(running):
        close(value, packet["search_best_so_far"][index], f"replay · best-so-far at prefix {index + 1}")
    # Prefix recommendations use registry order for ties, not arrival order.
    def recommend(prefix):
        available = replay[:prefix]
        best = max(means[index] for index in available)
        return min(index for index in available if means[index] == best)
    expect(recommend(2) == 10 and recommend(3) == 6 and recommend(4) == 9,
           "replay · prefixes 2, 3 and 4 recommend indices 10, 6 and 9")
    close(means[recommend(2)], means[recommend(3)], "replay · and the score is flat across that change", 1e-15)
    expect(means[recommend(4)] > means[recommend(3)], "replay · then rises at prefix 4")

    expect(packet["fits"] == len(registry) * 3 + 2 == 35, "campaign · 35 estimator fits")
    expect(packet["reserve_scored"] is False, "campaign · the packet records no reserved prediction")

    referenced = sorted({row for entry in records for row in entry["outOfFoldErrorRows"]}
                        | {row for entry in final for row in entry["errorRows"]})
    source_rows = {str(row): {
        "line": row + 1,
        "features": [float(value) for value in features[row]],
        "label": int(labels[row]),
        "role": role_of[row],
    } for row in referenced}
    # Every inspectable mistake is a real source row, kept so a learner can read
    # the four descriptors behind it. The weak candidates supply most of them.
    expect(len(referenced) == len(set(referenced)) and all(0 <= row < EXPECTED_ROWS for row in referenced),
           "records · every inspectable row is a distinct source row")
    expect(all(source_rows[str(row)]["role"] in ("development", "inspection") for row in referenced),
           "records · and none of them is a reserved row")
    expect({row for entry in final for row in entry["errorRows"]} == {107, 195, 345},
           "records · the three inspection mistakes belong to the baseline alone")
    expect(all(source_rows[str(row)]["label"] == 0 for row in (107, 195, 345, 349)),
           "records · all four narrated mistakes have true class 0")
    expect([source_rows[str(row)]["line"] for row in (107, 195, 345, 349)] == [108, 196, 346, 350],
           "records · whose file line numbers are 108, 196, 346 and 350")
    expect(min(len(entry["outOfFoldErrorRows"]) for entry in records) == 0
           and max(len(entry["outOfFoldErrorRows"]) for entry in records) == 919 - min(
               entry["pooledCorrect"] for entry in records),
           "records · error lists agree with the pooled numerators")

    trust_root_covered: set = set()
    payload_roles = [
        {"role": "development", "groups": DEVELOPMENT_GROUPS, "rows": int(len(development)),
         "positives": int(labels[development].sum()),
         "use": "Three-fold selection and the final refits."},
        {"role": "inspection", "groups": INSPECTION_GROUPS, "rows": int(len(inspection)),
         "positives": int(labels[inspection].sum()),
         "use": "Compare two fixed fitted procedures once."},
        {"role": "reserved", "groups": int(len(counts)) - DEVELOPMENT_GROUPS - INSPECTION_GROUPS,
         "rows": int(len(reserve)), "positives": int(labels[reserve].sum()),
         "use": "No predictions or model decisions."},
    ]
    manuscript_covered: set = set()
    manuscript_checks = crosscheck_manuscript(packet, records, payload_roles, manuscript_covered)
    attribution_checks = check_attribution(counts)
    # `author_checks` is eighteen self-attested booleans that nothing read.
    # They are the author's own claims, so assert they at least claim what the
    # rest of this run independently establishes.
    author = packet["author_checks"]
    expect(author["python_blocks_parsed"] == 3, "author checks · three displayed Python blocks")
    expect(author["optional_frameworks_executed"] is False, "author checks · no optional framework was executed")
    expect(author["additional_native_fits_for_checks"] == 0, "author checks · no extra fitting campaign")
    expect(all(value is True for key, value in author.items()
               if isinstance(value, bool) and key != "optional_frameworks_executed"),
           "author checks · every other recorded author check is claimed complete")
    expect(packet["tie_rule"] == "first in declared candidate order", "author checks · the recorded tie rule")
    trust_root_checks = rederive_trust_root(packet["constructed"], trust_root_covered)
    trust_root_leaves = sorted(leaf_paths(packet["constructed"], ""))
    trust_root_uncovered = sorted(set(trust_root_leaves) - trust_root_covered - manuscript_covered)
    # Coverage is asserted, not merely reported: a leaf nobody checks is exactly
    # how `halving.selected` sat in the packet unread by anything.
    expect(not trust_root_uncovered,
           f"trust root · every constructed leaf is pinned; unpinned: {trust_root_uncovered[:6]}")

    payload = {
        "provenance": {
            "name": "Banknote Authentication",
            "author": "Volker Lohweg (2012)",
            "doi": "https://doi.org/10.24432/C55P57",
            "page": "https://archive.ics.uci.edu/dataset/267/banknote+authentication",
            "download": "https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip",
            "license": "CC BY 4.0",
            "licenseUrl": "https://creativecommons.org/licenses/by/4.0/",
            "file": "/learn-assets/automl-nas/banknote-data.csv",
            "attribution": "/learn-assets/automl-nas/ATTRIBUTION.txt",
            "member": "data_banknote_authentication.txt",
            "separator": "comma",
            "lineEnding": "CRLF",
            "bytes": EXPECTED_BYTES,
            "sha256": EXPECTED_SHA,
            "archiveSha256": EXPECTED_ARCHIVE_SHA,
            "rows": EXPECTED_ROWS,
            "featureNames": FEATURE_NAMES,
            "uniqueGroups": int(len(counts)),
            "repeatedGroups": int((counts > 1).sum()),
            "repeatedExtraRows": int(counts.sum()) - int(len(counts)),
            "classCounts": [int((labels == 0).sum()), int((labels == 1).sum())],
            "retrieved": "12 September 2026",
        },
        "roles": payload_roles,
        "foldRows": fold_rows,
        "candidates": records,
        "selectedIndex": int(selected),
        "baselineIndex": BASELINE_INDEX,
        "inspection": final,
        "majorityBaseline": {
            "predictedClass": majority_class, "correct": majority_correct,
            "rows": int(len(inspection)), "accuracy": majority_correct / len(inspection),
        },
        "replayOrder": replay,
        "sourceRows": source_rows,
        "estimatorFits": int(len(registry) * 3 + 2),
        "reserveScored": False,
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "scikitLearn": sklearn.__version__,
        },
    }

    text = render_module(payload)
    if write:
        MODULE.write_text(text, encoding="utf-8", newline=NL)
    else:
        current = MODULE.read_text(encoding="utf-8") if MODULE.exists() else ""
        expect(current == text, "module · a fresh regeneration is byte-identical")

    if failures:
        for failure in failures:
            print("FAIL:", failure, file=sys.stderr)
        raise SystemExit(f"{len(failures)} of {sum(checks.values())} checks failed.")

    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps({
        "verifiedAt": datetime.now(timezone.utc).isoformat(),
        "stage": "native regeneration of the observed study and re-derivation of the packet's trust root",
        "verifier": "scripts/verify-automl-data.py",
        "verifierSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "moduleSha256": hashlib.sha256(MODULE.read_bytes()).hexdigest(),
        "moduleWritten": write,
        "datasetSha256": EXPECTED_SHA,
        "datasetBytes": EXPECTED_BYTES,
        "servedMatchesPacket": served == packet_bytes,
        "packetInputsSha256": hashlib.sha256(PACKET_INPUTS.read_bytes()).hexdigest(),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": None, "scikitLearn": sklearn.__version__},
        "rows": EXPECTED_ROWS,
        "uniqueFeatureGroups": int(len(counts)),
        "roleRows": {"development": int(len(development)), "inspection": int(len(inspection)),
                     "reserved": int(len(reserve))},
        "foldValidationRows": fold_rows,
        "estimatorFits": int(len(registry) * 3 + 2),
        "fittingWarnings": caught_warnings,
        "selectedId": records[selected]["id"],
        "inspectionConfusion": {entry["role"]: entry["confusion"] for entry in final},
        "reservedPredictionsComputed": False,
        "groupedChecks": checks,
        "totalChecks": sum(checks.values()),
        "trustRootLeaves": len(trust_root_leaves),
        "trustRootLeavesRederived": len(trust_root_covered & set(trust_root_leaves)),
        "trustRootLeavesPinnedByManuscript": len(manuscript_covered & set(trust_root_leaves)),
        "trustRootLeavesUncovered": trust_root_uncovered,
        "trustRootCoverageNote": (
            "Leaf-path coverage of calculated-inputs.json's constructed block, not a tally of check calls. "
            "'Rederived' leaves are outputs recomputed here from the declared settings by an independent route. "
            "The remainder are declared *inputs* to those re-derivations, which cannot pin themselves; they are "
            "pinned instead against the manuscript's own printed tables, which no check read before."
        ),
        "manuscriptCrossChecks": manuscript_checks,
        "attributionClaimsChecked": attribution_checks,
        "scope": (
            "Recomputed the declared 35-fit banknote campaign from the served CSV: exact-feature grouping, the "
            "three fixed roles, the three group-respecting folds, all eleven candidates' 919 out-of-fold "
            "predictions each, both inspection confusion matrices and the seed-75 replay order, matched value by "
            "value against the content packet's calculated-inputs.json; then re-derived that packet's own trust "
            "root - every value of its constructed block, which verify-automl-models.mjs consumes as ground "
            "truth - from the declared settings alone, using math.erf rather than SciPy's ndtr, a hand-written "
            "softmax, a central difference for both architecture gradients, an explicit halving loop, pairwise "
            "Pareto dominance and Hamming-distance activation kernels, without importing author-calculations.py. "
            "Finally checked that a fresh regeneration of src/learn/data/automl-data.js is byte-identical."
        ),
        "limitations": [
            "One fixed split of one small public collection; exact-feature grouping is not physical specimen identity.",
            "Development and inspection results only; no reserved row was predicted or scored.",
            "Numerical fitting can differ on other library versions.",
            "Displayed program execution and browser rendering are separate checks.",
        ],
        "passed": True,
    }, indent=2) + NL, encoding="utf-8", newline=NL)
    print(f"PASS: {sum(checks.values())} grouped checks across {len(checks)} groups; "
          f"{EXPECTED_ROWS} rows, {len(counts)} feature groups, 35 estimator fits recomputed and matched; "
          f"{len(trust_root_covered & set(trust_root_leaves))} of {len(trust_root_leaves)} trust-root leaves "
          f"re-derived and the other {len(trust_root_leaves) - len(trust_root_covered & set(trust_root_leaves))} "
          f"pinned against the manuscript ({manuscript_checks} manuscript checks, {attribution_checks} attribution claims); "
          f"module {'written' if write else 'byte-identical'} ({MODULE.stat().st_size / 1024:.0f} KB).")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
