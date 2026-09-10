"""Different algorithms check finite witnesses and bounds, not mirrored traces."""
import itertools
import json
import math
import platform
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linprog

directory = Path("scratch/combinatorial-optimization-review")
fixtures = json.loads((directory / "model-fixtures.json").read_text())
checks = 0


def near(a, b):
    global checks
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-10), (a, b)
    checks += 1


def subsets(count):
    return [tuple(index for index in range(count) if mask & (1 << index)) for mask in range(1 << count)]


for case in fixtures["assignments"]:
    costs = case["costs"]
    best_by_size = {0: 0}
    for size in range(1, min(case["workers"], case["jobs"]) + 1):
        totals = []
        for workers in itertools.combinations(range(case["workers"]), size):
            for jobs in itertools.permutations(range(case["jobs"]), size):
                values = [costs[a][b] for a, b in zip(workers, jobs)]
                if all(value is not None for value in values):
                    totals.append(sum(values))
        if totals:
            best_by_size[size] = min(totals)
    for step in case["trace"]:
        near(step["cost"], best_by_size[step["flow"]])
        assert len({pair["worker"] for pair in step["pairs"]}) == step["flow"]
        assert len({pair["job"] for pair in step["pairs"]}) == step["flow"]
        if step["flow"]:
            near(step["pathCost"], sum(arc["cost"] for arc in step["path"]))
    expected_flow = min(case["required"], max(best_by_size))
    assert case["flow"] == expected_flow
    if case["flow"] < case["required"]:
        assert case["maximumCardinalityProved"]
    for arc in case["residual"]:
        near(arc["reducedCost"], arc["cost"] + case["potentials"][arc["from"]] - case["potentials"][arc["to"]])
        assert arc["reducedCost"] >= 0


for case in fixtures["search"]:
    items, capacity = case["items"], case["capacity"]
    choices = [(mask, sum(items[i]["weight"] for i in chosen), sum(items[i]["value"] for i in chosen)) for mask, chosen in enumerate(subsets(len(items)))]
    feasible = [choice for choice in choices if choice[1] <= capacity]
    optimum = max(value for _, _, value in feasible)
    near(case["oracle"]["value"], optimum)
    for step in case["trace"]:
        incumbent = step["incumbent"]
        assert (incumbent["mask"], incumbent["weight"], incumbent["value"]) in feasible
        assert incumbent["value"] <= optimum + 1e-10 <= step["upper"] + 1e-9
        if step["optimal"]:
            near(incumbent["value"], optimum)
        for node in step["nodes"]:
            if node["status"] == "infeasible":
                assert node["weight"] > capacity
                continue
            prefix = sum(1 << item["original"] for item in case["order"][:node["depth"]])
            descendants = [value for mask, _, value in feasible if mask & prefix == node["mask"]]
            if descendants:
                assert max(descendants) <= node["upper"] + 1e-10
                if node["status"] == "bound-pruned":
                    assert max(descendants) <= incumbent["value"]
            checks += 1


for case in fixtures["scaling"]:
    items, capacity = case["items"], case["capacity"]
    feasible = [chosen for chosen in subsets(len(items)) if sum(items[i]["weight"] for i in chosen) <= capacity]
    optimum = max(sum(items[i]["value"] for i in chosen) for chosen in feasible)
    near(case["oracle"]["value"], optimum)
    selected = case["selected"]
    assert len(set(selected)) == len(selected)
    assert sum(items[i]["weight"] for i in selected) <= capacity
    near(case["value"], sum(items[i]["value"] for i in selected))
    assert case["value"] + 1e-9 >= (1-case["epsilon"]) * optimum
    if not case["active"]:
        assert optimum == 0
        continue
    denominator = round(1 / case["epsilon"])
    count = len(case["active"])
    maximum = max(item["value"] for item in case["active"])
    for item in case["active"]:
        assert item["rounded"] == int(Fraction(item["value"] * count * denominator, maximum))
    for count, row in enumerate(case["rows"]):
        expected = {}
        for subset in subsets(count):
            value = sum(case["active"][i]["rounded"] for i in subset)
            weight = sum(case["active"][i]["weight"] for i in subset)
            expected[value] = min(weight, expected.get(value, math.inf))
        for value, minimum in enumerate(row):
            assert (minimum if minimum is not None else math.inf) == expected.get(value, math.inf)
            checks += 1


for case in fixtures["sets"]:
    sets = case["sets"]
    universe = set(range(case["universeSize"]))
    values = []
    for chosen in subsets(len(sets)):
        union = set().union(*(set(sets[i]["elements"]) for i in chosen))
        values.append((chosen, union, sum(sets[i]["cost"] for i in chosen)))
    budget = case["maximumSelections"]
    if budget is None:
        optima = [cost for _, union, cost in values if union == universe]
        if optima:
            optimum = min(optima)
            near(case["optimumCost"], optimum)
            assert case["feasible"]
            assert case["cost"] <= case["harmonic"] * optimum + 1e-9
            for candidate in sets:
                harmonic = sum(Fraction(1, k) for k in range(1, len(candidate["elements"]) + 1))
                assert sum(case["charges"][i] for i in candidate["elements"]) <= candidate["cost"] * float(harmonic) + 1e-9
        else:
            assert case["optimumCost"] is None and not case["feasible"]
    else:
        optimum = max(len(union) for chosen, union, _ in values if len(chosen) <= budget)
        assert case["optimumCoverage"] == optimum
        if budget:
            assert case["coveredCount"] + 1e-9 >= (1 - (1 - 1/budget)**budget) * optimum
    for step in case["trace"]:
        near(sum(step["charges"]), step["cost"])


for case in fixtures["covers"]:
    costs, edges = case["costs"], case["edges"]
    matrix = np.zeros((len(edges), len(costs)))
    for row, (a, b) in zip(matrix, edges):
        row[a] = row[b] = -1
    result = linprog(costs, A_ub=matrix if edges else None, b_ub=-np.ones(len(edges)) if edges else None, bounds=[(0, 1)]*len(costs), method="highs")
    assert result.success
    near(case["fractionalCost"], result.fun)
    exact = min(sum(costs[i] for i in chosen) for chosen in subsets(len(costs)) if all(a in chosen or b in chosen for a, b in edges))
    near(case["exactCost"], exact)
    rounded = case["rounded"]
    assert all(a in rounded or b in rounded for a, b in edges)
    assert case["roundedCost"] <= 2 * result.fun + 1e-9
    for step in case["trace"]:
        for vertex, cost in enumerate(costs):
            total = sum(load for (a, b), load in zip(edges, step["loads"]) if vertex in (a, b))
            near(step["totals"][vertex], total)
            assert total <= cost
            if vertex in step["selected"]:
                near(total, cost)
        assert step["lower"] <= exact
    assert not case["final"]["uncovered"]
    assert case["final"]["cost"] <= 2 * case["final"]["lower"]


for case in fixtures["matroids"]:
    preset, kind = case["preset"], case["kind"]
    feasible = []
    for mask, chosen in enumerate(subsets(len(preset["names"]))):
        if kind == "uniform":
            valid = len(chosen) <= 2
        elif kind == "intervals":
            valid = all(not (max(preset["intervals"][a][0], preset["intervals"][b][0]) < min(preset["intervals"][a][1], preset["intervals"][b][1])) for a, b in itertools.combinations(chosen, 2))
        else:
            incidence = np.zeros((4, len(chosen)))
            for column, edge in enumerate(chosen):
                a, b = preset["edges"][edge]
                incidence[a, column] = 1
                incidence[b, column] = -1
            valid = not chosen or np.linalg.matrix_rank(incidence) == len(chosen)
        if valid:
            feasible.append(mask)
    assert feasible == case["feasible"]
    optimum = max(sum(case["weights"][i] for i in chosen) for mask, chosen in enumerate(subsets(len(preset["names"]))) if mask in feasible)
    near(case["optimumValue"], optimum)
    if kind != "intervals":
        near(case["greedyValue"], optimum)
    assert case["augmentationHolds"] == (kind != "intervals")

report = {"at": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "numericOrStateComparisons": checks, "method": "Permutation/subset optima, independent LP solves, exact Fraction scaling, incidence matrix rank and actual primal/dual witnesses"}
(directory / "independent-model-results.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
