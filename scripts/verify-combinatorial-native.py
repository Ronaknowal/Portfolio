"""Execute the actual displayed programs and changed contracts against independent oracles."""
import contextlib
import io
import itertools
import json
import math
import platform
import random
import subprocess
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
import scipy

root = Path.cwd()
directory = root / "scratch/combinatorial-optimization-review"
example_json = subprocess.check_output(
    ["node", "--input-type=module", "-e", "import {combinatorialOptimizationExamples as e} from './src/learn/data/combinatorial-optimization-examples.js';console.log(JSON.stringify(e));"],
    text=True, encoding="utf8")
examples = json.loads(example_json)
namespaces = {}
for key, example in examples.items():
    namespace = {"__name__": "__main__"}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(example["code"], key + ".py", "exec"), namespace)
    assert output.getvalue().rstrip() == example["expected"], key
    namespaces[key] = namespace

counts = {"displayedPrograms": len(examples), "assignmentMatricesAndSizes": 0,
          "knapsackSearchStops": 0, "scaledKnapsackCases": 0, "weightedCoverCases": 0,
          "vertexCoverCases": 0, "maximumCoverageCases": 0, "metricTours": 0,
          "changedPracticeContracts": 0, "invalidInputs": 0}
rng = random.Random(32817)

def subsets(count):
    return [tuple(i for i in range(count) if mask & (1 << i)) for mask in range(1 << count)]

assignment = namespaces["assignment"]["min_cost_assignment"]
for entries in itertools.product((-2, 0, 3, None), repeat=4):
    costs = [list(entries[:2]), list(entries[2:])]
    for required in range(3):
        trace, complete = assignment(costs, required)
        optima = {0: 0}
        for size in range(1, required + 1):
            values = [sum(costs[a][b] for a, b in zip(workers, jobs))
                      for workers in itertools.combinations(range(2), size)
                      for jobs in itertools.permutations(range(2), size)
                      if all(costs[a][b] is not None for a, b in zip(workers, jobs))]
            if values:
                optima[size] = min(values)
        assert complete == (required in optima)
        assert trace[-1][0] == max(optima)
        for size, cost, pairs, path in trace:
            assert cost == optima[size]
            assert len(set(a for a, _ in pairs)) == size == len(set(b for _, b in pairs))
        counts["assignmentMatricesAndSizes"] += 1

search = namespaces["boundedKnapsack"]["bounded_knapsack"]
scaled = namespaces["scaledKnapsack"]["scaled_knapsack"]
for case in range(110):
    items = [(rng.randint(1, 12), rng.randint(0, 40)) for _ in range(case % 8)]
    capacity = rng.randint(0, 22)
    feasible = [choice for choice in subsets(len(items)) if sum(items[i][0] for i in choice) <= capacity]
    optimum = max(sum(items[i][1] for i in choice) for choice in feasible)
    for limit in (0, 1, 3, 7, 1000):
        result = search(items, capacity, limit)
        assert tuple(result["items"]) in feasible
        assert result["value"] == sum(items[i][1] for i in result["items"])
        assert result["value"] <= optimum <= result["upper"]
        assert result["proved"] == (result["upper"] == result["value"])
        if limit == 1000:
            assert result["value"] == optimum and result["proved"]
        counts["knapsackSearchStops"] += 1
    for epsilon in (Fraction(1, 2), Fraction(1, 4), Fraction(2, 7), Fraction(1, 10)):
        chosen, weight, value, _ = scaled(items, capacity, epsilon)
        assert tuple(chosen) in feasible
        assert value == sum(items[i][1] for i in chosen) >= (1 - epsilon) * optimum
        assert weight == sum(items[i][0] for i in chosen)
        counts["scaledKnapsackCases"] += 1

cover = namespaces["weightedCover"]["greedy_cover"]
maximum = namespaces["maximumCoverage"]["maximum_coverage"]
for case in range(130):
    universe = set(range(case % 8))
    candidates = [(str(i), {e for e in universe if rng.random() < .5}, rng.randint(0, 8)) for i in range(1 + case % 6)]
    full_union = set().union(*(elements for _, elements, _ in candidates))
    if full_union != universe:
        try:
            cover(universe, candidates)
        except ValueError:
            pass
        else:
            raise AssertionError("Unavailable element was not rejected")
    else:
        chosen, charges = cover(universe, candidates)
        cost = sum(candidates[i][2] for i in chosen)
        optimum = min(sum(candidates[i][2] for i in choice) for choice in subsets(len(candidates))
                      if set().union(*(candidates[i][1] for i in choice)) == universe)
        harmonic = sum((Fraction(1, j) for j in range(1, max(map(lambda row: len(row[1]), candidates)) + 1)), Fraction(0))
        assert sum(charges.values()) == cost <= harmonic * optimum
        for _, elements, price in candidates:
            bound = price * sum((Fraction(1, j) for j in range(1, len(elements) + 1)), Fraction(0))
            assert sum(charges[e] for e in elements) <= bound
    counts["weightedCoverCases"] += 1
    sets = [row[1] for row in candidates]
    for budget in range(len(sets) + 1):
        chosen, covered, _ = maximum(sets, budget)
        optimum = max(len(set().union(*(sets[i] for i in choice))) for choice in subsets(len(sets)) if len(choice) <= budget)
        assert len(chosen) == budget and len(set(chosen)) == budget
        assert covered == set().union(*(sets[i] for i in chosen))
        if budget:
            assert len(covered) >= (1 - (1 - Fraction(1, budget)) ** budget) * optimum
        counts["maximumCoverageCases"] += 1

primal_dual = namespaces["primalDualCover"]["primal_dual_cover"]
round_cover = namespaces["fractionalCover"]["round_cover"]
for case in range(120):
    size = 1 + case % 6
    costs = [rng.randint(0, 12) for _ in range(size)]
    edges = [(a, b) for a in range(size) for b in range(a + 1, size) if rng.random() < .55]
    covers = [choice for choice in subsets(size) if all(a in choice or b in choice for a, b in edges)]
    optimum = min(sum(costs[i] for i in choice) for choice in covers)
    chosen, loads = primal_dual(costs, edges)
    assert tuple(sorted(chosen)) in covers
    assert all(sum(y for y, edge in zip(loads, edges) if vertex in edge) <= costs[vertex] for vertex in range(size))
    assert sum(loads) <= optimum <= sum(costs[i] for i in chosen) <= 2 * sum(loads)
    x, lp, rounded = round_cover(costs, edges)
    assert tuple(rounded) in covers
    assert lp <= optimum + 1e-8 and sum(costs[i] for i in rounded) <= 2 * lp + 1e-7
    # Solve the edge-load maximization, not the displayed vertex-variable LP.
    if edges:
        incidence = np.array([[int(vertex in edge) for edge in edges] for vertex in range(size)], dtype=float)
        dual = linprog(-np.ones(len(edges)), A_ub=incidence, b_ub=costs, bounds=(0, None), method="highs")
        assert dual.success and math.isclose(-dual.fun, lp, abs_tol=1e-8)
    counts["vertexCoverCases"] += 1

metric_routes = namespaces["metricRoutes"]["metric_routes"]
for case in range(36):
    points = [(rng.randint(-5, 5), rng.randint(-5, 5)) for _ in range(2 + case % 5)]
    tree, tree_cost, odd, matching, match_cost, doubled, repaired, optimum = metric_routes(points)
    assert len(odd) % 2 == 0 and len(matching) * 2 == len(odd)
    assert tree_cost <= optimum and 2 * match_cost <= optimum
    assert doubled[1] <= 2 * optimum and 2 * repaired[1] <= 3 * optimum
    counts["metricTours"] += 1

trace, complete = assignment([[-4, 2], [0, None]], 2)
assert complete and [row[1] for row in trace] == [0, -4, 2]
assert scaled([(2, 3), (3, 4), (99, 100000)], 3, Fraction(1, 4))[:3] == ([1], 3, 4)
assert primal_dual([1, 100], [(0, 1)]) == ({0}, [1])
assert np.allclose(round_cover([1, 100], [(0, 1)])[0], [1, 0])
counts["changedPracticeContracts"] = 4
invalid = [
    lambda: assignment([[1], [1, 2]], 1),
    lambda: assignment([[1.5]], 1),
    lambda: assignment([[1]], 2),
    lambda: search([(0, 1)], 1, 1),
    lambda: search([(1, -1)], 1, 1),
    lambda: search([(1, 1)], 1.5, 1),
    lambda: scaled([(1, 1)], 1, Fraction(0)),
    lambda: scaled([(1, 1)], 1, Fraction(1)),
    lambda: scaled([(1, 1)], 1, .5),
    lambda: cover({0}, [("x", {0}, -1)]),
    lambda: primal_dual([-1], []),
    lambda: primal_dual([1], [(0, 2)]),
    lambda: round_cover([1], [(0, 0)]),
    lambda: round_cover([float("nan")], []),
    lambda: round_cover([], []),
    lambda: maximum([{0}], 2),
]
for operation in invalid:
    try:
        operation()
    except ValueError:
        counts["invalidInputs"] += 1
    else:
        raise AssertionError("An invalid input was accepted")
record = {"checkedAt": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
          "numpy": np.__version__, "scipy": scipy.__version__, "counts": counts,
          "actualDisplayedCodeExecuted": True, "allPassed": True}
(directory / "native-contract-results.json").write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
