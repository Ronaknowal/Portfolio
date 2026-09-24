"""Exact small-network calculations and a fixed, offline Wine comparison.

Run from any directory with Python 3.12, NumPy and scikit-learn:
    python network-experiments.py
Writes calculated-inputs.json beside this script. No network access or tuning.
"""
from itertools import product, combinations
from pathlib import Path
import json
import copy
import numpy as np
from sklearn.model_selection import train_test_split

DIRECTORY = Path(__file__).resolve().parent
NODES = ("B", "E", "A", "J", "M")
PARENTS = {"B": (), "E": (), "A": ("B", "E"), "J": ("A",), "M": ("A",)}
ALARM = {
    "B": {(): .001}, "E": {(): .002},
    "A": {(0, 0): .001, (0, 1): .29, (1, 0): .94, (1, 1): .95},
    "J": {(0,): .05, (1,): .9}, "M": {(0,): .01, (1,): .7},
}


def alarm_joint(assignment, tables=ALARM):
    probability = 1.
    for node in NODES:
        chance = tables[node][tuple(assignment[p] for p in PARENTS[node])]
        probability *= chance if assignment[node] else 1 - chance
    return probability


def alarm_query(evidence, tables=ALARM):
    weights = np.zeros(2)
    for values in product((0, 1), repeat=5):
        assignment = dict(zip(NODES, values))
        if all(assignment[node] == state for node, state in evidence.items()):
            weights[assignment["B"]] += alarm_joint(assignment, tables)
    total = weights.sum()
    return {"evidenceProbability": float(total),
            "burglary": None if total == 0 else float(weights[1] / total)}


def active_paths(edges, start, end, observed):
    """All simple undirected paths; collider descendants are handled explicitly."""
    nodes = set(sum(([a, b] for a, b in edges), []))
    directed = set(map(tuple, edges))
    neighbors = {node: set() for node in nodes}
    ancestors = set(observed)
    for a, b in directed:
        neighbors[a].add(b)
        neighbors[b].add(a)
    while True:
        expanded = ancestors | {a for a, b in directed if b in ancestors}
        if expanded == ancestors:
            break
        ancestors = expanded
    paths = []

    def visit(path):
        if path[-1] == end:
            for left, middle, right in zip(path, path[1:], path[2:]):
                collider = (left, middle) in directed and (right, middle) in directed
                if (collider and middle not in ancestors) or (not collider and middle in observed):
                    return
            paths.append(path)
            return
        for neighbor in sorted(neighbors[path[-1]] - set(path)):
            visit(path + [neighbor])

    visit([start])
    return paths


def service_effect(assignment=(.2, .6), outcome=((.01, .1), (.05, .2))):
    # outcome[x][z]; Z has probability .5 in both strata.
    observational, intervention = [], []
    for x in (0, 1):
        weights = np.array([p if x else 1 - p for p in assignment])
        denominator = weights.sum()
        observational.append(None if denominator == 0 else float(weights @ outcome[x] / denominator))
        intervention.append(float(np.mean(outcome[x])))
    return {"observational": observational, "interventional": intervention,
            "associationDifference": None if None in observational else observational[1] - observational[0],
            "causalDifference": intervention[1] - intervention[0],
            "positiveSupport": all(0 < p < 1 for p in assignment)}


def frontdoor():
    # U -> X, U -> Y, X -> M -> Y. U is hidden in observational summaries.
    joint = np.zeros((2, 2, 2, 2))
    chance_x = [.2, .8]
    chance_m = [.1, .9]
    chance_y = [[.05, .4], [.5, .9]]  # indexed [m][u]
    for u, x, m, y in product((0, 1), repeat=4):
        joint[u, x, m, y] = .5 * (chance_x[u] if x else 1 - chance_x[u]) * (
            chance_m[x] if m else 1 - chance_m[x]) * (chance_y[m][u] if y else 1 - chance_y[m][u])
    observed = joint.sum(axis=0)
    px = observed.sum(axis=(1, 2))
    result = []
    for x in (0, 1):
        estimate = 0.
        for m in (0, 1):
            pm_x = observed[x, m].sum() / px[x]
            inner = sum(observed[xp, m, 1] / observed[xp, m].sum() * px[xp] for xp in (0, 1))
            estimate += pm_x * inner
        direct = sum(.5 * (chance_m[x] if m else 1 - chance_m[x]) * chance_y[m][u]
                     for u, m in product((0, 1), repeat=2))
        result.append({"x": x, "frontdoor": float(estimate), "direct": direct,
                       "observational": float(observed[x, :, 1].sum() / px[x])})
    return {"observedJoint": observed.tolist(), "queries": result}


def conditional_information(bits, labels, left, right):
    answer = 0.
    for label in range(3):
        rows = bits[labels == label]
        counts = np.zeros((2, 2))
        np.add.at(counts, (rows[:, left], rows[:, right]), 1)
        for a, b in product((0, 1), repeat=2):
            if counts[a, b]:
                answer += counts[a, b] / len(bits) * np.log(
                    counts[a, b] * len(rows) / (counts[a].sum() * counts[:, b].sum()))
    return float(answer)


def tree_parents(bits, labels):
    weighted = [(conditional_information(bits, labels, a, b), a, b)
                for a, b in combinations(range(4), 2)]
    # Kruskal: highest weight first; index order breaks exact ties.
    components = list(range(4))
    edges = []
    for weight, a, b in sorted(weighted, key=lambda item: (-item[0], item[1], item[2])):
        if components[a] != components[b]:
            old, new = components[b], components[a]
            components = [new if component == old else component for component in components]
            edges.append((a, b))
    parents = [None] * 4
    seen, queue = {0}, [0]
    while queue:
        node = queue.pop(0)
        for a, b in edges:
            neighbor = b if a == node else a if b == node else None
            if neighbor is not None and neighbor not in seen:
                parents[neighbor] = node
                seen.add(neighbor)
                queue.append(neighbor)
    return parents, [{"left": a, "right": b, "informationNats": w} for w, a, b in weighted]


def fit_network(features, labels, kind):
    medians = np.median(features, axis=0)
    bits = (features > medians).astype(int)
    parents, weights = tree_parents(bits, labels) if kind == "TAN" else ([None] * 4, [])
    tables = []
    for column, parent in enumerate(parents):
        counts = np.ones((3, 1 if parent is None else 2, 2))
        parent_states = np.zeros(len(bits), dtype=int) if parent is None else bits[:, parent]
        np.add.at(counts, (labels, parent_states, bits[:, column]), 1)
        tables.append((counts / counts.sum(axis=2, keepdims=True)).tolist())
    prior = (np.bincount(labels, minlength=3) + 1) / (len(labels) + 3)
    return {"kind": kind, "medians": medians.tolist(), "parents": parents,
            "prior": prior.tolist(), "tables": tables, "edgeWeights": weights}


def predict_network(model, features, visible=(0, 1, 2, 3)):
    probabilities = []
    for row in features:
        observed = (row > model["medians"]).astype(int)
        total = np.zeros(3)
        for states in product((0, 1), repeat=4):
            if any(states[column] != observed[column] for column in visible):
                continue
            weight = np.array(model["prior"])
            for column, parent in enumerate(model["parents"]):
                parent_state = 0 if parent is None else states[parent]
                weight *= np.array(model["tables"][column])[:, parent_state, states[column]]
            total += weight
        probabilities.append(total / total.sum())
    return np.array(probabilities)


def score(probabilities, labels):
    return {"correct": int((probabilities.argmax(axis=1) == labels).sum()),
            "rows": len(labels), "logLoss": float(-np.log(probabilities[np.arange(len(labels)), labels]).mean())}


def wine_experiment():
    data = np.genfromtxt(DIRECTORY / "wine.csv", delimiter=",", names=True)
    names = ["alcohol", "malic_acid", "flavanoids", "color_intensity"]
    features = np.column_stack([data[name] for name in names])
    labels = data["cultivar"].astype(int)
    development, test = train_test_split(np.arange(len(labels)), test_size=36, stratify=labels, random_state=61)
    train, validation = train_test_split(development, test_size=36, stratify=labels[development], random_state=62)
    models = {kind: fit_network(features[train], labels[train], kind) for kind in ("NB", "TAN")}
    validation_scores = {kind: score(predict_network(model, features[validation]), labels[validation])
                         for kind, model in models.items()}
    prior_score = score(np.tile(models["NB"]["prior"], (36, 1)), labels[validation])
    chosen = min(("NB", "TAN"), key=lambda kind: validation_scores[kind]["logLoss"])
    final = fit_network(features[development], labels[development], chosen)
    final_score = score(predict_network(final, features[test]), labels[test])
    rows = []
    for index in validation:
        masks = {}
        for visible in ((), (0,), (2,), (0, 2), (0, 1, 2, 3)):
            masks[",".join(map(str, visible)) or "none"] = predict_network(
                models["TAN"], features[[index]], visible)[0].tolist()
        rows.append({"id": int(index + 1), "features": features[index].tolist(),
                     "cultivar": int(labels[index]), "tanPosteriors": masks})
    return {"features": names, "trainIds": (train + 1).tolist(), "validationIds": (validation + 1).tolist(),
            "testIds": (test + 1).tolist(), "trainingModels": models, "validation": validation_scores,
            "validationPrior": prior_score, "selected": chosen, "finalModel": final,
            "finalTest": final_score, "finalPrior": score(np.tile(final["prior"], (36, 1)), labels[test]),
            "validationRows": rows}


def changed_and_null_checks(wine):
    checks = {}
    tables = copy.deepcopy(ALARM)
    tables["J"][(0,)] = .2
    checks["falseJohn"] = alarm_query({"J": 1, "M": 1}, tables)
    checks["unusedRow"] = alarm_query({"A": 1, "J": 1}, tables)
    tables["J"] = {(0,): .4, (1,): .4}
    checks["uninformativeJohn"] = alarm_query({"J": 1}, tables)
    tables["J"] = {(0,): 0., (1,): 0.}
    checks["impossibleJohn"] = alarm_query({"J": 1}, tables)
    tables = copy.deepcopy(ALARM)
    tables["M"] = {(0,): .4, (1,): .4}
    checks["practiceMary"] = alarm_query({"J": 1, "M": 1}, tables)
    checks["changedService"] = service_effect(outcome=((.01, .1), (.01, .2)))
    model = wine["trainingModels"]["TAN"]
    for alcohol in [13.17, 12.9, 12.8]:
        checks["alcohol" + str(alcohol)] = predict_network(
            model, np.array([[alcohol, 5.19, .63, 7.9]]), (0, 2)).tolist()
    checks["hiddenEdit"] = predict_network(model, np.array([[12.8, 5.19, .63, 7.9]]), (2,)).tolist()
    graph = [("B", "A"), ("E", "A"), ("A", "J"), ("A", "M"), ("B", "K"), ("K", "E")]
    checks["twoPathsKJ"] = active_paths(graph, "B", "E", {"K", "J"})
    checks["twoPathsK"] = active_paths(graph, "B", "E", {"K"})
    adjustment_graph = [("S", "E"), ("S", "Y"), ("E", "T"), ("E", "Y")]
    checks["changedAdjustmentS"] = active_paths(adjustment_graph, "T", "Y", {"S"})
    checks["changedAdjustmentE"] = active_paths(adjustment_graph, "T", "Y", {"E"})
    return checks


def main():
    queries = [{}, {"J": 1}, {"J": 1, "M": 1}, {"J": 1, "M": 1, "E": 0},
               {"A": 1}, {"A": 1, "E": 1}, {"A": 1, "J": 1}]
    ve_trace = []
    for b in (0, 1):
        for a in (0, 1):
            reduced = sum((.002 if e else .998) * (
                ALARM["A"][b, e] if a else 1 - ALARM["A"][b, e]) for e in (0, 1))
            prior = .001 if b else .999
            calls = ALARM["J"][a,] * ALARM["M"][a,]
            ve_trace.append({"b": b, "a": a, "afterSumE": reduced,
                             "afterPrior": prior * reduced, "jointWithCalls": prior * reduced * calls})
    edges = [(parent, node) for node, parents in PARENTS.items() for parent in parents]
    education = [("S", "E"), ("S", "Y"), ("E", "T"), ("T", "Y")]
    backdoor = [edge for edge in education if edge[0] != "T"]
    outputs = {"alarmQueries": [{"evidence": q, **alarm_query(q)} for q in queries],
               "elimination": ve_trace,
               "paths": {str(sorted(obs)): active_paths(edges, "B", "E", obs)
                         for obs in (set(), {"A"}, {"J"}, {"A", "J"})},
               "educationBackdoor": {str(sorted(obs)): active_paths(backdoor, "T", "Y", obs)
                                    for obs in (set(), {"S"}, {"E"}, {"S", "E"})},
               "service": service_effect(), "randomService": service_effect((.4, .4)),
               "noSupport": service_effect((0, 1)), "frontdoor": frontdoor(),
               "wine": wine_experiment()}
    outputs["changedAndNullChecks"] = changed_and_null_checks(outputs["wine"])
    (DIRECTORY / "calculated-inputs.json").write_text(json.dumps(outputs, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in outputs.items() if key != "wine"}, indent=2))
    print(json.dumps({key: outputs["wine"][key] for key in
                      ("validation", "validationPrior", "selected", "finalTest", "finalPrior")}, indent=2))


if __name__ == "__main__":
    main()
