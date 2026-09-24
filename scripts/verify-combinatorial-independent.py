"""Independent bounded review, separate from the author's algorithm oracles."""
import contextlib
import io
import itertools
import json
import math
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

directory = Path("scratch/combinatorial-independent-review")
data = json.loads((directory / "fixtures.json").read_text(encoding="utf-8"))
namespaces = {}
for name, example in data["examples"].items():
    namespace = {"__name__": "__main__"}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(example["code"], name + ".py", "exec"), namespace)
    assert output.getvalue().rstrip() == example["expected"], name
    namespaces[name] = namespace


def partial_optima(costs):
    workers, jobs = len(costs), len(costs[0])
    result = {0: 0}
    for size in range(1, min(workers, jobs) + 1):
        feasible = []
        for rows in itertools.combinations(range(workers), size):
            for columns in itertools.permutations(range(jobs), size):
                values = [costs[row][column] for row, column in zip(rows, columns)]
                if all(value is not None for value in values):
                    feasible.append(sum(values))
        if feasible:
            result[size] = min(feasible)
    return result


stage_checks = residual_closures = 0
native_assignment = namespaces["assignment"]["min_cost_assignment"]
for state in data["assignments"]:
    costs = state["costs"]
    optimum = partial_optima(costs)
    expected_flow = max(size for size in optimum if size <= state["required"])
    assert state["flow"] == expected_flow
    assert state["feasible"] == (state["required"] in optimum)
    assert state["cost"] == optimum[expected_flow]
    native_trace, complete = native_assignment(costs, state["required"])
    assert complete == state["feasible"]
    for stage in state["trace"]:
        assert stage["cost"] == optimum[stage["flow"]]
        assert len(stage["pairs"]) == stage["flow"]
        assert len({pair["worker"] for pair in stage["pairs"]}) == stage["flow"]
        assert len({pair["job"] for pair in stage["pairs"]}) == stage["flow"]
        stage_checks += 1
    for size, cost, pairs, path in native_trace:
        assert cost == optimum[size]
        assert cost == sum(costs[worker][job] for worker, job in pairs)
    # Build residual edges independently from the final assignment, then use
    # all-pairs closure (not the implementation's Bellman-Ford) for cycles.
    workers, jobs = len(costs), len(costs[0])
    source, sink = workers + jobs, workers + jobs + 1
    vertices = sink + 1
    pairs = {(pair["worker"], pair["job"]) for pair in state["pairs"]}
    used_workers = {worker for worker, _ in pairs}
    used_jobs = {job for _, job in pairs}
    residual = []
    for worker in range(workers):
        residual.append((worker, source, 0) if worker in used_workers else (source, worker, 0))
        for job in range(jobs):
            if costs[worker][job] is not None:
                residual.append((workers + job, worker, -costs[worker][job]) if (worker, job) in pairs else (worker, workers + job, costs[worker][job]))
    for job in range(jobs):
        residual.append((sink, workers + job, 0) if job in used_jobs else (workers + job, sink, 0))
    assert set(residual) == {(edge["from"], edge["to"], edge["cost"]) for edge in state["residual"]}
    distance = [[math.inf] * vertices for _ in range(vertices)]
    for vertex in range(vertices):
        distance[vertex][vertex] = 0
    for start, end, cost in residual:
        distance[start][end] = min(distance[start][end], cost)
        assert cost + state["potentials"][start] - state["potentials"][end] >= 0
    for middle in range(vertices):
        for start in range(vertices):
            for end in range(vertices):
                distance[start][end] = min(distance[start][end], distance[start][middle] + distance[middle][end])
    assert all(distance[vertex][vertex] >= 0 for vertex in range(vertices))
    assert state["maximumCardinalityProved"] == math.isinf(distance[source][sink])
    residual_closures += 1

native_scaling = namespaces["scaledKnapsack"]["scaled_knapsack"]
rounded_score_checks = 0
for state in data["scaling"]:
    items = [(item["weight"], item["value"]) for item in state["items"]]
    capacity = state["capacity"]
    epsilon = Fraction(1, round(1 / state["epsilon"]))
    chosen, weight, value, _ = native_scaling(items, capacity, epsilon)
    active = [(index, w, v) for index, (w, v) in enumerate(items) if w <= capacity and v > 0]
    if not active:
        assert state["value"] == value == 0
        continue
    scale = epsilon * max(v for _, _, v in active) / len(active)
    rounded = {index: Fraction(v) // scale for index, _, v in active}
    assert [row["rounded"] for row in state["active"]] == list(rounded.values())
    scores = []
    true_optimum = 0
    for mask in range(1 << len(items)):
        subset = [index for index in range(len(items)) if mask & (1 << index)]
        if sum(items[index][0] for index in subset) <= capacity:
            true_optimum = max(true_optimum, sum(items[index][1] for index in subset))
            scores.append(sum(rounded.get(index, 0) for index in subset))
    assert sum(rounded[index] for index in state["selected"]) == max(scores)
    assert sum(rounded[index] for index in chosen) == max(scores)
    assert value >= (1 - epsilon) * true_optimum
    assert Fraction(state["value"]) >= (1 - epsilon) * true_optimum
    rounded_score_checks += 1

# Signed greedy forests: use component edge-count identity as an independent
# acyclicity check, rather than the displayed union-find independence routine.
forest_helper = namespaces["matroidForest"]["maximum_weight_forest"]
edge_pairs = list(itertools.combinations(range(4), 2))


def forest_by_components(edges):
    remaining = set(range(4))
    components = 0
    while remaining:
        reached = {remaining.pop()}
        while True:
            expanded = reached | {v for a, b, _ in edges for v, neighbor in [(a, b), (b, a)] if neighbor in reached}
            if expanded == reached:
                break
            reached = expanded
        remaining -= reached
        components += 1
    return len(edges) == 4 - components


for sample in range(40):
    edges = [(a, b, (sample * 7 + index * 11) % 17 - 8) for index, (a, b) in enumerate(edge_pairs)]
    choices = [[edge for index, edge in enumerate(edges) if mask & (1 << index)] for mask in range(64)]
    optimum = max(sum(edge[2] for edge in choice) for choice in choices if forest_by_components(choice))
    selected = forest_helper(4, edges)
    assert forest_by_components(selected)
    assert sum(edge[2] for edge in selected) == optimum

result = {
    "checkedAt": datetime.now(timezone.utc).isoformat(),
    "actualDisplayedPrograms": len(namespaces),
    "rectangularSignedMissingAssignmentStates": len(data["assignments"]),
    "independentlyEnumeratedStageCosts": stage_checks,
    "independentlyRebuiltResidualClosures": residual_closures,
    "changedScalingStates": len(data["scaling"]),
    "exactRoundedObjectiveChecks": rounded_score_checks,
    "signedGreedyForestChecks": 40,
    "allPassed": True,
    "sourceHashes": data["sourceHashes"],
}
(directory / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
