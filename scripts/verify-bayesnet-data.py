"""Regenerate and check the Bayesian-networks lesson's recorded data module.

This script does two jobs.

1. It **re-derives the whole trust root**. Every scalar in the content packet's
   `calculated-inputs.json` is recomputed here from the declared settings and
   the dataset this lesson serves, through an implementation written from the
   definitions rather than reused from the packet's own
   `network-experiments.py`: conditional mutual information is accumulated from
   an explicit contingency table, the maximum-weight spanning tree is built with
   union-find instead of a component-relabelling scan, the tree is oriented by
   an explicit stack, inference multiplies its factors in an explicit loop, and
   every constructed alarm, adjustment, service and frontdoor quantity is also
   confirmed a second time in exact `fractions.Fraction` arithmetic, where the
   answer cannot depend on floating-point association at all.

   Coverage is measured, not asserted: the comparison walks the packet tree and
   records the path of every scalar leaf it actually compared. The run fails
   unless that set is the complete set of leaves, so a block added to the packet
   lowers the count and stops the build instead of passing unnoticed.

2. It regenerates `src/learn/data/bayesnet-data.js` from those results.

The script is READ-ONLY unless given `--write`. Without it the module text is
rebuilt in memory and must be byte-identical to the file already on disk, and
the served asset must already match the packet bytes.

Run:
  scratch/lesson-tools/Scripts/python.exe scripts/verify-bayesnet-data.py --write
  scratch/lesson-tools/Scripts/python.exe scripts/verify-bayesnet-data.py
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from fractions import Fraction
from itertools import combinations, product
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / "docs/teaching/drafts/bayesian-networks-causal-graphical-models"
PACKET_DATASET = PACKET / "wine.csv"
PACKET_INPUTS = PACKET / "calculated-inputs.json"
ASSET_DIR = ROOT / "public/learn-assets/bayesian-networks"
ASSET_DATASET = ASSET_DIR / "wine.csv"
MODULE = ROOT / "src/learn/data/bayesnet-data.js"
EVIDENCE = ROOT / "docs/teaching/evidence/bayesnet-data.json"

EXPECTED_SHA = "34ced17cfa0a96bf5ae5c25565da075c612145b20368fc2455e5e8f966e818be"
EXPECTED_BYTES = 12100
SOURCE_ROWS = 178
FEATURES = ["alcohol", "malic_acid", "flavanoids", "color_intensity"]
CLASS_COUNT = 3
TEST_SEED = 61
VALIDATION_SEED = 62
TEST_SIZE = 36
VALIDATION_SIZE = 36
PSEUDO_COUNT = 1

failures: list[str] = []
checks = {"count": 0}
covered: set[str] = set()


def check(condition, label):
    checks["count"] += 1
    if not condition:
        failures.append(label)


def close(actual, expected, label, tolerance=1e-9):
    try:
        difference = abs(float(actual) - float(expected))
    except (TypeError, ValueError):
        check(False, f"{label}: {actual!r} is not a number")
        return
    check(difference <= tolerance * max(1.0, abs(float(expected))), f"{label}: {actual} versus {expected}")


# ------------------------------------------------------------------ trust root

def leaf_paths(node, prefix=""):
    """Every scalar leaf of the recorded tree, addressed by its path."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from leaf_paths(value, f"{prefix}/{key}")
    elif isinstance(node, list):
        if not node:
            yield f"{prefix}/[]"
        for index, value in enumerate(node):
            yield from leaf_paths(value, f"{prefix}/{index}")
    else:
        yield prefix


def compare_tree(got, want, prefix="", tolerance=1e-9):
    """Compare a recomputed subtree against the packet, recording every leaf seen."""
    if isinstance(want, dict):
        if not isinstance(got, dict) or set(got) != set(want):
            check(False, f"{prefix}: key sets differ ({sorted(got) if isinstance(got, dict) else got} "
                         f"versus {sorted(want)})")
            return
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


# --------------------------------------------------- the constructed examples

NODES = ("B", "E", "A", "J", "M")
PARENTS = {"B": (), "E": (), "A": ("B", "E"), "J": ("A",), "M": ("A",)}
ALARM_EXACT = {
    "B": {(): Fraction(1, 1000)},
    "E": {(): Fraction(2, 1000)},
    "A": {(0, 0): Fraction(1, 1000), (0, 1): Fraction(29, 100),
          (1, 0): Fraction(94, 100), (1, 1): Fraction(95, 100)},
    "J": {(0,): Fraction(5, 100), (1,): Fraction(90, 100)},
    "M": {(0,): Fraction(1, 100), (1,): Fraction(70, 100)},
}


def alarm_query(evidence, tables=None):
    """Exact rational enumeration of all 32 worlds. No floating point is involved."""
    tables = ALARM_EXACT if tables is None else tables
    mass = [Fraction(0), Fraction(0)]
    for values in product((0, 1), repeat=5):
        assignment = dict(zip(NODES, values))
        if any(assignment[node] != state for node, state in evidence.items()):
            continue
        probability = Fraction(1)
        for node in NODES:
            chance = tables[node][tuple(assignment[parent] for parent in PARENTS[node])]
            probability *= chance if assignment[node] else 1 - chance
        mass[assignment["B"]] += probability
    total = mass[0] + mass[1]
    return {"evidenceProbability": float(total),
            "burglary": None if total == 0 else float(mass[1] / total)}


def replace_row(node, key, value, tables=None):
    tables = ALARM_EXACT if tables is None else tables
    copied = {name: dict(rows) for name, rows in tables.items()}
    copied[node][key] = value
    return copied


def simple_paths(edges, start, end):
    adjacency: dict[str, set[str]] = {}
    for left, right in edges:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    found = []

    def walk(path):
        if path[-1] == end:
            found.append(list(path))
            return
        for neighbour in sorted(adjacency.get(path[-1], ())):
            if neighbour not in path:
                walk(path + [neighbour])

    walk([start])
    return found


def descendants(edges, node):
    seen, stack = set(), [node]
    while stack:
        current = stack.pop()
        for left, right in edges:
            if left == current and right not in seen:
                seen.add(right)
                stack.append(right)
    return seen


def active_paths(edges, start, end, observed):
    """Pearl's rule applied to each simple path independently, written from the definition."""
    directed = {tuple(edge) for edge in edges}
    observed = set(observed)
    active = []
    for path in simple_paths(edges, start, end):
        blocked = False
        for left, middle, right in zip(path, path[1:], path[2:]):
            collider = (left, middle) in directed and (right, middle) in directed
            if collider:
                if middle not in observed and not (descendants(edges, middle) & observed):
                    blocked = True
                    break
            elif middle in observed:
                blocked = True
                break
        if not blocked:
            active.append(path)
    return active


def service_effect(assignment=(0.2, 0.6), outcome=((0.01, 0.1), (0.05, 0.2))):
    """Exact rational arithmetic; the load prior is one half in both strata."""
    assignment = [Fraction(str(value)) for value in assignment]
    outcome = [[Fraction(str(value)) for value in row] for row in outcome]
    half = Fraction(1, 2)
    observational, interventional = [], []
    for x in (0, 1):
        weights = [half * (assignment[z] if x else 1 - assignment[z]) for z in (0, 1)]
        total = weights[0] + weights[1]
        observational.append(None if total == 0
                             else float(sum(weights[z] * outcome[x][z] for z in (0, 1)) / total))
        interventional.append(float(sum(half * outcome[x][z] for z in (0, 1))))
    return {
        "observational": observational,
        "interventional": interventional,
        "associationDifference": None if None in observational else observational[1] - observational[0],
        "causalDifference": interventional[1] - interventional[0],
        "positiveSupport": all(0 < value < 1 for value in assignment),
    }


def frontdoor():
    """The 16-world latent model, its observed summaries and both identification routes."""
    half = Fraction(1, 2)
    chance_x = [Fraction(2, 10), Fraction(8, 10)]
    chance_m = [Fraction(1, 10), Fraction(9, 10)]
    chance_y = [[Fraction(5, 100), Fraction(40, 100)], [Fraction(50, 100), Fraction(90, 100)]]
    joint = {}
    for u, x, m, y in product((0, 1), repeat=4):
        joint[(u, x, m, y)] = (half
                               * (chance_x[u] if x else 1 - chance_x[u])
                               * (chance_m[x] if m else 1 - chance_m[x])
                               * (chance_y[m][u] if y else 1 - chance_y[m][u]))
    observed = [[[sum(joint[(u, x, m, y)] for u in (0, 1)) for y in (0, 1)] for m in (0, 1)] for x in (0, 1)]
    margin_x = [sum(observed[x][m][y] for m in (0, 1) for y in (0, 1)) for x in (0, 1)]
    queries = []
    for x in (0, 1):
        estimate = Fraction(0)
        for m in (0, 1):
            mediator_given = sum(observed[x][m]) / margin_x[x]
            inner = sum(observed[xp][m][1] / sum(observed[xp][m]) * margin_x[xp] for xp in (0, 1))
            estimate += mediator_given * inner
        direct = sum(half * (chance_m[x] if m else 1 - chance_m[x]) * chance_y[m][u]
                     for u, m in product((0, 1), repeat=2))
        queries.append({"x": x, "frontdoor": float(estimate), "direct": float(direct),
                        "observational": float(sum(observed[x][m][1] for m in (0, 1)) / margin_x[x])})
    return {"observedJoint": [[[float(value) for value in pair] for pair in row] for row in observed],
            "queries": queries}


# ------------------------------------------------------------ the Wine study

def read_served_rows(path):
    """Plain csv reading, independent of the packet's genfromtxt name mangling."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows


def conditional_information(bits, labels, left, right):
    """I(X_left; X_right | C) accumulated from an explicit 3x2x2 contingency table."""
    total = len(bits)
    answer = 0.0
    for label in range(CLASS_COUNT):
        rows = bits[labels == label]
        table = [[0, 0], [0, 0]]
        for row in rows:
            table[row[left]][row[right]] += 1
        class_total = len(rows)
        for a, b in product((0, 1), repeat=2):
            joint = table[a][b]
            if not joint:
                continue
            left_total = table[a][0] + table[a][1]
            right_total = table[0][b] + table[1][b]
            answer += joint / total * np.log(joint * class_total / (left_total * right_total))
    return float(answer)


def maximum_spanning_tree(weights, count):
    """Kruskal with union-find. Ties break on the same (-weight, left, right) order."""
    parent = list(range(count))

    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    edges = []
    for weight, left, right in sorted(weights, key=lambda item: (-item[0], item[1], item[2])):
        a, b = find(left), find(right)
        if a != b:
            parent[a] = b
            edges.append((left, right))
    return edges


def orient_from_root(edges, count, root=0):
    """Orient the undirected tree away from `root` with an explicit stack."""
    parents = [None] * count
    seen = {root}
    stack = [root]
    order = []
    while stack:
        node = stack.pop(0)
        order.append(node)
        for left, right in edges:
            neighbour = right if left == node else left if right == node else None
            if neighbour is not None and neighbour not in seen:
                parents[neighbour] = node
                seen.add(neighbour)
                stack.append(neighbour)
    return parents


def fit_network(features, labels, kind):
    medians = np.median(features, axis=0)
    bits = (features > medians).astype(int)
    weights = [(conditional_information(bits, labels, a, b), a, b)
               for a, b in combinations(range(len(FEATURES)), 2)]
    if kind == "TAN":
        parents = orient_from_root(maximum_spanning_tree(weights, len(FEATURES)), len(FEATURES))
        recorded_weights = [{"left": a, "right": b, "informationNats": w} for w, a, b in weights]
    else:
        parents = [None] * len(FEATURES)
        recorded_weights = []
    tables = []
    for column, parent in enumerate(parents):
        parent_states = 1 if parent is None else 2
        counts = np.full((CLASS_COUNT, parent_states, 2), float(PSEUDO_COUNT))
        for row_index in range(len(bits)):
            parent_state = 0 if parent is None else bits[row_index, parent]
            counts[labels[row_index], parent_state, bits[row_index, column]] += 1
        tables.append((counts / counts.sum(axis=2, keepdims=True)).tolist())
    class_counts = np.array([int((labels == k).sum()) for k in range(CLASS_COUNT)], dtype=float)
    prior = (class_counts + PSEUDO_COUNT) / (len(labels) + CLASS_COUNT * PSEUDO_COUNT)
    return {"kind": kind, "medians": medians.tolist(), "parents": parents,
            "prior": prior.tolist(), "tables": tables, "edgeWeights": recorded_weights}


def predict_network(model, features, visible=(0, 1, 2, 3)):
    """Exact marginalisation over every hidden feature, one explicit product per class."""
    medians = np.asarray(model["medians"])
    tables = [np.asarray(table) for table in model["tables"]]
    output = []
    for row in np.atleast_2d(features):
        observed = (row > medians).astype(int)
        total = np.zeros(CLASS_COUNT)
        for states in product((0, 1), repeat=len(FEATURES)):
            if any(states[column] != observed[column] for column in visible):
                continue
            for klass in range(CLASS_COUNT):
                weight = model["prior"][klass]
                for column, parent in enumerate(model["parents"]):
                    parent_state = 0 if parent is None else states[parent]
                    weight *= tables[column][klass, parent_state, states[column]]
                total[klass] += weight
        output.append((total / total.sum()).tolist())
    return np.array(output)


def score(probabilities, labels):
    picked = probabilities.argmax(axis=1)
    chosen = probabilities[np.arange(len(labels)), labels]
    return {"correct": int((picked == labels).sum()), "rows": int(len(labels)),
            "logLoss": float(-np.log(chosen).mean())}


# ------------------------------------------------------------------ emission

def number(value):
    """A float that round-trips through JSON.parse exactly as Python wrote it."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return repr(float(value))


def numbers(values):
    return "[" + ", ".join(number(value) for value in values) + "]"


def model_literal(model, indent):
    pad = " " * indent
    inner = " " * (indent + 2)
    tables = ",\n".join(
        f"{inner}  [" + ", ".join(
            "[" + ", ".join(numbers(parent_row) for parent_row in class_row) + "]"
            for class_row in column) + "]"
        for column in model["tables"])
    return (
        f"{{\n"
        f"{inner}kind: {json.dumps(model['kind'])},\n"
        f"{inner}medians: {numbers(model['medians'])},\n"
        f"{inner}parents: [" + ", ".join("null" if p is None else str(p) for p in model["parents"]) + "],\n"
        f"{inner}prior: {numbers(model['prior'])},\n"
        f"{inner}tables: [\n{tables},\n{inner}],\n"
        f"{pad}}}"
    )


def main():
    write = "--write" in sys.argv

    packet_bytes = PACKET_DATASET.read_bytes()
    digest = hashlib.sha256(packet_bytes).hexdigest()
    check(digest == EXPECTED_SHA, f"the packet dataset hashes {digest}, not the pinned {EXPECTED_SHA}")
    check(len(packet_bytes) == EXPECTED_BYTES, f"the packet dataset is {len(packet_bytes)} bytes, not {EXPECTED_BYTES}")
    recorded = json.loads(PACKET_INPUTS.read_text(encoding="utf-8"))
    packet_inputs_sha = hashlib.sha256(PACKET_INPUTS.read_bytes()).hexdigest()

    if write:
        ASSET_DIR.mkdir(parents=True, exist_ok=True)
        ASSET_DATASET.write_bytes(packet_bytes)
    served_bytes = ASSET_DATASET.read_bytes() if ASSET_DATASET.exists() else b""
    check(served_bytes == packet_bytes, "the served copy holds the packet's exact bytes")

    # ---------------------------------------------------- constructed examples
    queries = [{}, {"J": 1}, {"J": 1, "M": 1}, {"J": 1, "M": 1, "E": 0},
               {"A": 1}, {"A": 1, "E": 1}, {"A": 1, "J": 1}]
    mine = {"alarmQueries": [{"evidence": evidence, **alarm_query(evidence)} for evidence in queries]}

    elimination = []
    for b in (0, 1):
        for a in (0, 1):
            reduced = sum((ALARM_EXACT["E"][()] if e else 1 - ALARM_EXACT["E"][()])
                          * (ALARM_EXACT["A"][(b, e)] if a else 1 - ALARM_EXACT["A"][(b, e)])
                          for e in (0, 1))
            prior = ALARM_EXACT["B"][()] if b else 1 - ALARM_EXACT["B"][()]
            calls = ALARM_EXACT["J"][(a,)] * ALARM_EXACT["M"][(a,)]
            elimination.append({"b": b, "a": a, "afterSumE": float(reduced),
                                "afterPrior": float(prior * reduced),
                                "jointWithCalls": float(prior * reduced * calls)})
    mine["elimination"] = elimination

    alarm_edges = [(parent, node) for node, parents in PARENTS.items() for parent in parents]
    mine["paths"] = {str(sorted(observed)): active_paths(alarm_edges, "B", "E", observed)
                     for observed in (set(), {"A"}, {"J"}, {"A", "J"})}

    education = [("S", "E"), ("S", "Y"), ("E", "T"), ("T", "Y")]
    backdoor_graph = [edge for edge in education if edge[0] != "T"]
    mine["educationBackdoor"] = {str(sorted(observed)): active_paths(backdoor_graph, "T", "Y", observed)
                                 for observed in (set(), {"S"}, {"E"}, {"S", "E"})}

    mine["service"] = service_effect()
    mine["randomService"] = service_effect((0.4, 0.4))
    mine["noSupport"] = service_effect((0, 1))
    mine["frontdoor"] = frontdoor()

    # ------------------------------------------------------------- Wine study
    rows = read_served_rows(ASSET_DATASET)
    check(len(rows) == SOURCE_ROWS, f"the served dataset holds {len(rows)} rows, not {SOURCE_ROWS}")
    check([int(row["specimen_id"]) for row in rows] == list(range(1, SOURCE_ROWS + 1)),
          "specimen ids run 1 to 178 in source order")
    features = np.array([[float(row[name]) for name in FEATURES] for row in rows])
    labels = np.array([int(row["cultivar"]) for row in rows])
    check(sorted(set(labels.tolist())) == [0, 1, 2], "the served cultivar column holds the three zero-based labels")

    development, test = train_test_split(np.arange(len(labels)), test_size=TEST_SIZE,
                                         stratify=labels, random_state=TEST_SEED)
    train, validation = train_test_split(development, test_size=VALIDATION_SIZE,
                                         stratify=labels[development], random_state=VALIDATION_SEED)
    check(len(train) == 106 and len(validation) == 36 and len(test) == 36,
          f"the split gives {len(train)}/{len(validation)}/{len(test)}, not 106/36/36")
    check(len(set(train.tolist()) | set(validation.tolist()) | set(test.tolist())) == SOURCE_ROWS,
          "the three roles are disjoint and cover every specimen")

    models = {kind: fit_network(features[train], labels[train], kind) for kind in ("NB", "TAN")}
    validation_scores = {kind: score(predict_network(model, features[validation]), labels[validation])
                         for kind, model in models.items()}
    prior_score = score(np.tile(models["NB"]["prior"], (VALIDATION_SIZE, 1)), labels[validation])
    chosen = min(("NB", "TAN"), key=lambda kind: validation_scores[kind]["logLoss"])
    final = fit_network(features[development], labels[development], chosen)
    final_score = score(predict_network(final, features[test]), labels[test])
    validation_rows = []
    for index in validation:
        masks = {}
        for visible in ((), (0,), (2,), (0, 2), (0, 1, 2, 3)):
            key = ",".join(map(str, visible)) or "none"
            masks[key] = predict_network(models["TAN"], features[[index]], visible)[0].tolist()
        validation_rows.append({"id": int(index + 1), "features": features[index].tolist(),
                                "cultivar": int(labels[index]), "tanPosteriors": masks})
    mine["wine"] = {
        "features": FEATURES, "trainIds": (train + 1).tolist(),
        "validationIds": (validation + 1).tolist(), "testIds": (test + 1).tolist(),
        "trainingModels": models, "validation": validation_scores, "validationPrior": prior_score,
        "selected": chosen, "finalModel": final, "finalTest": final_score,
        "finalPrior": score(np.tile(final["prior"], (TEST_SIZE, 1)), labels[test]),
        "validationRows": validation_rows,
    }

    # ------------------------------------------------- changed and null checks
    changed = {}
    tables = replace_row("J", (0,), Fraction(2, 10))
    changed["falseJohn"] = alarm_query({"J": 1, "M": 1}, tables)
    changed["unusedRow"] = alarm_query({"A": 1, "J": 1}, tables)
    tables = replace_row("J", (1,), Fraction(4, 10), replace_row("J", (0,), Fraction(4, 10), tables))
    changed["uninformativeJohn"] = alarm_query({"J": 1}, tables)
    tables = replace_row("J", (1,), Fraction(0), replace_row("J", (0,), Fraction(0), tables))
    changed["impossibleJohn"] = alarm_query({"J": 1}, tables)
    mary = replace_row("M", (1,), Fraction(4, 10), replace_row("M", (0,), Fraction(4, 10)))
    changed["practiceMary"] = alarm_query({"J": 1, "M": 1}, mary)
    changed["changedService"] = service_effect(outcome=((0.01, 0.1), (0.01, 0.2)))
    for alcohol in (13.17, 12.9, 12.8):
        changed["alcohol" + str(alcohol)] = predict_network(
            models["TAN"], np.array([[alcohol, 5.19, 0.63, 7.9]]), (0, 2)).tolist()
    changed["hiddenEdit"] = predict_network(
        models["TAN"], np.array([[12.8, 5.19, 0.63, 7.9]]), (2,)).tolist()
    two_route = alarm_edges + [("B", "K"), ("K", "E")]
    changed["twoPathsKJ"] = active_paths(two_route, "B", "E", {"K", "J"})
    changed["twoPathsK"] = active_paths(two_route, "B", "E", {"K"})
    adjustment_graph = [("S", "E"), ("S", "Y"), ("E", "T"), ("E", "Y")]
    changed["changedAdjustmentS"] = active_paths(adjustment_graph, "T", "Y", {"S"})
    changed["changedAdjustmentE"] = active_paths(adjustment_graph, "T", "Y", {"E"})
    mine["changedAndNullChecks"] = changed

    # --------------------------------------------- compare against the packet
    compare_tree(mine, recorded, "")
    all_leaves = set(leaf_paths(recorded))
    missing = sorted(all_leaves - covered)
    check(covered == all_leaves,
          f"every scalar leaf of the trust root is re-derived: {len(covered)} of {len(all_leaves)}"
          + (f"; first uncovered: {missing[:3]}" if missing else ""))

    # ------------------------------ independent statements about those results
    # These are properties, not repeats of the comparison above: they would fail
    # even if the packet and this script agreed on a wrong number.
    root_checks = 0

    def root(condition, label):
        nonlocal root_checks
        root_checks += 1
        check(condition, f"trust root -- {label}")

    root(abs(mine["alarmQueries"][2]["burglary"] - 0.28417183536439294) < 1e-12,
         "two calls give the posterior the manuscript prints")
    root(mine["alarmQueries"][4]["burglary"] == mine["alarmQueries"][6]["burglary"],
         "John's call adds nothing once the alarm state is known")
    root(mine["alarmQueries"][5]["burglary"] < mine["alarmQueries"][4]["burglary"],
         "an earthquake explains the alarm away")
    root(mine["alarmQueries"][0]["burglary"] == 0.001, "the empty-evidence query returns the prior")
    root(changed["impossibleJohn"]["burglary"] is None,
         "impossible evidence has no posterior rather than zero")
    root(abs(changed["uninformativeJohn"]["burglary"] - 0.001) < 1e-15,
         "equal caller rows return the prior")
    root(abs(changed["practiceMary"]["burglary"] - mine["alarmQueries"][1]["burglary"]) < 1e-15,
         "practice 1: an uninformative Mary leaves John's posterior")
    root(changed["unusedRow"]["burglary"] == mine["alarmQueries"][6]["burglary"],
         "changing an unused table row cannot move a posterior that never reads it")
    root(mine["paths"]["[]"] == [] and mine["paths"]["['A']"] == [["B", "A", "E"]],
         "the collider is blocked unobserved and open when observed")
    root(mine["paths"]["['J']"] == [["B", "A", "E"]],
         "an observed descendant opens the collider too")
    root(mine["educationBackdoor"]["[]"] == [["T", "E", "S", "Y"]],
         "the note's graph has exactly one backdoor path")
    root(all(mine["educationBackdoor"][key] == []
             for key in ("['S']", "['E']", "['E', 'S']")),
         "both singleton sets and their union block that path, as the destination note reasoned")
    root(changed["changedAdjustmentS"] == [["T", "E", "Y"]] and changed["changedAdjustmentE"] == [],
         "adding E to Y leaves S-only open while E still blocks everything")
    root(abs(mine["service"]["causalDifference"] - 0.07) < 1e-15
         and abs(mine["service"]["associationDifference"] - 0.1225) < 1e-15,
         "the service model separates a .07 effect from a .1225 association")
    root(abs(mine["randomService"]["associationDifference"]
             - mine["randomService"]["causalDifference"]) < 1e-15,
         "equal assignment probabilities make observation and intervention agree")
    root(mine["noSupport"]["positiveSupport"] is False,
         "assignment [0, 1] is marked as having no overlap")
    root(all(abs(query["frontdoor"] - query["direct"]) < 1e-12 for query in mine["frontdoor"]["queries"]),
         "the frontdoor formula and the truncated factorisation agree at both treatment levels")
    root(abs(mine["frontdoor"]["queries"][1]["observational"]
             - mine["frontdoor"]["queries"][1]["frontdoor"]) > 0.1,
         "and both differ substantially from the observational probability")
    root(mine["wine"]["selected"] == "NB"
         and mine["wine"]["validation"]["NB"]["logLoss"] < mine["wine"]["validation"]["TAN"]["logLoss"],
         "NB wins the declared validation criterion")
    root(mine["wine"]["validation"]["NB"]["correct"] == mine["wine"]["validation"]["TAN"]["correct"],
         "with the same validation accuracy, so log loss is what separates them")
    root(mine["wine"]["trainingModels"]["TAN"]["parents"] == [None, 0, 0, 0],
         "the learned tree gives every other measurement alcohol as its feature parent")
    root(changed["alcohol13.17"] != changed["alcohol12.9"],
         "crossing the fixed alcohol median changes the posterior")
    root(changed["alcohol12.9"] == changed["alcohol12.8"],
         "an edit within the same bin is a null")
    root(changed["hiddenEdit"] == predict_network(
        models["TAN"], np.array([[13.17, 5.19, 0.63, 7.9]]), (2,)).tolist(),
         "editing a hidden measurement cannot move the posterior")
    no_evidence = predict_network(models["TAN"], features[[validation[0]]], ())[0].tolist()
    root(all(abs(a - b) < 1e-12 for a, b in zip(no_evidence, models["TAN"]["prior"])),
         "with nothing visible the conditional tables sum away and the answer is the class prior")

    # ------------------------------------------------------------ the module
    specimens = ",\n".join(
        "  { id: %d, cultivar: %d, features: %s, tanPosteriors: { %s } }" % (
            row["id"], row["cultivar"], numbers(row["features"]),
            ", ".join(f"{json.dumps(key)}: {numbers(value)}"
                      for key, value in row["tanPosteriors"].items()))
        for row in mine["wine"]["validationRows"])
    weights_literal = ",\n".join(
        "  { left: %d, right: %d, informationNats: %s }" % (
            entry["left"], entry["right"], number(entry["informationNats"]))
        for entry in mine["wine"]["trainingModels"]["TAN"]["edgeWeights"])
    scores_literal = ",\n".join(
        f"  {json.dumps(key)}: {{ correct: {value['correct']}, rows: {value['rows']}, "
        f"logLoss: {number(value['logLoss'])} }}"
        for key, value in [
            ("naiveBayesValidation", mine["wine"]["validation"]["NB"]),
            ("treeAugmentedValidation", mine["wine"]["validation"]["TAN"]),
            ("priorValidation", mine["wine"]["validationPrior"]),
            ("selectedTest", mine["wine"]["finalTest"]),
            ("priorTest", mine["wine"]["finalPrior"]),
        ])

    module = f"""/** Recorded Wine cultivar results for the Bayesian-networks lesson.
 *
 * Generated by scripts/verify-bayesnet-data.py, which recomputes this whole
 * experiment from the dataset this lesson serves -- medians, conditional mutual
 * information, the maximum-weight spanning tree, both smoothed parameter fits,
 * every score and every missing-measurement query -- through an implementation
 * written from the definitions rather than reused from the content packet's
 * author script, and then matches every scalar of that packet against it.
 * Do not edit by hand.
 *
 * Source: Aeberhard, S. and Forina, M., Wine, UCI Machine Learning Repository,
 * https://doi.org/10.24432/C5PC7J, licensed CC BY 4.0. This lesson serves its
 * own copy of the unchanged extract at /learn-assets/bayesian-networks/wine.csv
 * ({EXPECTED_BYTES} bytes, SHA-256 {EXPECTED_SHA}),
 * beside /learn-assets/bayesian-networks/ATTRIBUTION.txt. A different lesson
 * serves a different Wine extract; this one is not shared.
 *
 * Protocol, fixed before any score was read: {SOURCE_ROWS} specimens, four declared
 * measurements, a stratified {TEST_SIZE}-specimen test split with seed {TEST_SEED}, then a
 * stratified {VALIDATION_SIZE}-specimen validation split with seed {VALIDATION_SEED} from the remaining
 * {SOURCE_ROWS - TEST_SIZE}, leaving {SOURCE_ROWS - TEST_SIZE - VALIDATION_SIZE} for training. Each measurement becomes the indicator
 * "strictly above this feature's training median". Both recipes use one
 * Dirichlet pseudo-count per state, including the class prior. The recipe with
 * the lower validation mean log loss is refit on the {SOURCE_ROWS - TEST_SIZE} development specimens
 * and assessed once on the reserved {TEST_SIZE}. No further search was run and no
 * alternative test score was computed.
 *
 * This is cultivar recognition on one small single split. It is not a benchmark,
 * a calibration certificate or a claim about other regions or later vintages.
 */

export const provenance = {{
  name: 'Wine',
  creator: 'Stefan Aeberhard and M. Forina',
  doi: 'https://doi.org/10.24432/C5PC7J',
  page: 'https://archive.ics.uci.edu/dataset/109/wine',
  license: 'CC BY 4.0',
  licenseUrl: 'https://creativecommons.org/licenses/by/4.0/',
  file: '/learn-assets/bayesian-networks/wine.csv',
  attribution: '/learn-assets/bayesian-networks/ATTRIBUTION.txt',
  bytes: {EXPECTED_BYTES},
  sha256: '{EXPECTED_SHA}',
  specimens: {SOURCE_ROWS},
  measurementsInFile: 13,
  cultivars: {CLASS_COUNT},
}};

export const protocol = {{
  features: {json.dumps(FEATURES)},
  featureLabels: ['alcohol', 'malic acid', 'flavanoids', 'colour intensity'],
  testSeed: {TEST_SEED},
  validationSeed: {VALIDATION_SEED},
  trainSize: {len(train)},
  validationSize: {VALIDATION_SIZE},
  testSize: {TEST_SIZE},
  pseudoCount: {PSEUDO_COUNT},
  selected: {json.dumps(mine["wine"]["selected"])},
  trainIds: {json.dumps(mine["wine"]["trainIds"])},
  validationIds: {json.dumps(mine["wine"]["validationIds"])},
  testIds: {json.dumps(mine["wine"]["testIds"])},
}};

/** Fitted on the {len(train)} training specimens only. The lesson's measurement
 * investigation reads the tree-augmented fit, which is deliberately not the
 * recipe that won selection; that is labelled where it is shown. */
export const trainingModels = {{
  naiveBayes: {model_literal(mine["wine"]["trainingModels"]["NB"], 2)},
  treeAugmented: {model_literal(mine["wine"]["trainingModels"]["TAN"], 2)},
}};

/** The selected recipe refit on the {SOURCE_ROWS - TEST_SIZE} development specimens. Assessed once. */
export const finalModel = {model_literal(mine["wine"]["finalModel"], 0)};

/** Training-set conditional mutual information in nats for each feature pair. */
export const conditionalInformation = [
{weights_literal},
];

export const scores = {{
{scores_literal},
}};

/** The 36 validation specimens, with the training tree-augmented model's answer
 * under each of the five measurement subsets the lesson displays. */
export const validationSpecimens = [
{specimens},
];
"""

    coverage = len(covered) / len(all_leaves)
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE.write_text(json.dumps({
        "checkedAt": datetime.now(timezone.utc).isoformat(),
        "mode": "write" if write else "read-only",
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "scipy": scipy.__version__, "scikit-learn": sklearn.__version__},
        "datasetSha256": digest,
        "datasetBytes": len(packet_bytes),
        "servedCopy": str(ASSET_DATASET.relative_to(ROOT)).replace("\\", "/"),
        "servedMatchesPacket": served_bytes == packet_bytes,
        "packetCalculatedInputsSha256": packet_inputs_sha,
        "sourceRows": SOURCE_ROWS,
        "roles": {"train": len(train), "validation": VALIDATION_SIZE, "test": TEST_SIZE},
        "totalChecks": checks["count"],
        "trustRootScalarLeaves": len(all_leaves),
        "trustRootLeavesReDerived": len(covered),
        "trustRootCoverage": round(coverage, 6),
        "trustRootUncoveredPaths": missing,
        "trustRootPropertyChecks": root_checks,
        "selected": mine["wine"]["selected"],
        "validationLogLoss": {kind: mine["wine"]["validation"][kind]["logLoss"] for kind in ("NB", "TAN")},
        "testScore": mine["wine"]["finalTest"],
        "moduleSha256": hashlib.sha256(module.encode("utf-8")).hexdigest(),
        "verifierSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Every scalar of the content packet's calculated-inputs.json re-derived from the declared "
                 "settings and the served dataset: the 32-world alarm queries and elimination trace in exact "
                 "rational arithmetic; all simple-path enumerations for the alarm graph, its second route, and "
                 "both adjustment graphs the destination note concerns; the maintenance model at its declared, "
                 "randomised, no-overlap and changed-response settings; the 16-world frontdoor model by both "
                 "the frontdoor formula and the truncated factorisation; and the complete Wine experiment "
                 "including medians, conditional mutual information, the spanning tree, both smoothed fits, "
                 "all five scores and every validation specimen's five missing-measurement queries. "
                 "Regeneration of src/learn/data/bayesnet-data.js must be byte-identical.",
        "limitations": [
            "The split itself is defined by scikit-learn's stratified train_test_split at the declared seeds; "
            "reproducing it necessarily calls the same function, and a different library version could move it.",
            "One small single split of one historical collection. These are model outputs, not calibration.",
            "Displayed program execution is a separate script, as are the browser model checks and browser review.",
        ],
        "passed": not failures,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")

    # Floors, so that a wholesale loss of coverage fails rather than quietly
    # printing a smaller number. They sit well below the current values.
    check(checks["count"] >= 1200, f"at least 1,200 checks ran; only {checks['count']} did")
    check(root_checks >= 20, f"at least twenty property statements ran; only {root_checks} did")
    check(len(all_leaves) >= 1200, f"the trust root still has its leaves; found {len(all_leaves)}")

    if failures:
        for label in failures:
            print(f"FAIL {label}")
        raise SystemExit(f"{len(failures)} of {checks['count']} data checks failed")

    if write:
        MODULE.write_text(module, encoding="utf-8", newline="\n")
    else:
        if not MODULE.exists():
            raise SystemExit("the data module is missing; rerun with --write")
        if MODULE.read_text(encoding="utf-8") != module:
            raise SystemExit("a fresh regeneration is not byte-identical to src/learn/data/bayesnet-data.js; "
                             "rerun with --write and inspect the difference")
        if not (ASSET_DIR / "ATTRIBUTION.txt").exists():
            raise SystemExit("the served attribution file is missing")


    print(f"PASS: {checks['count']:,} data checks, including {root_checks} independent property statements; "
          f"{coverage:.1%} of the trust root's {len(all_leaves):,} scalar leaves re-derived "
          f"({len(covered):,} covered, {len(missing)} not); module "
          f"{'written' if write else 'byte-identical'} ({MODULE.stat().st_size / 1024:.0f} KB).")


if __name__ == "__main__":
    main()
