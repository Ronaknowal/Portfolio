"""Complementary review: actual products, bipartite flow duals, digit ranking."""
from bisect import bisect_left, bisect_right
from collections import Counter, deque
import contextlib
from datetime import datetime, timezone
from functools import cache
import hashlib
import io
import json
import math
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / "scratch/dp-state-families-independent/model-fixtures.json").read_text(encoding="utf-8"))
checks = Counter()
environments = {}
for key, example in [("old:" + item["id"], item) for item in data["oldPrograms"]] + list(data["newPrograms"].items()):
    environment, output = {}, io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(example["code"], "actual-display-" + key + ".py", "exec"), environment)
    assert output.getvalue().rstrip() == example["expected"]
    environments[key] = environment
    checks["actual original stdout" if key.startswith("old:") else "actual new stdout"] += 1


class Matrix:
    def __init__(self, values, products=0):
        self.values, self.products = values, products

    def __mul__(self, other):
        rows, inner, columns = len(self.values), len(other.values), len(other.values[0])
        assert len(self.values[0]) == inner
        result = [[sum(self.values[row][k] * other.values[k][column] for k in range(inner))
                   for column in range(columns)] for row in range(rows)]
        return Matrix(result, self.products + other.products + rows * inner * columns)


def all_products(matrices):
    if len(matrices) == 1:
        return [matrices[0]]
    return [left * right for split in range(1, len(matrices))
            for left in all_products(matrices[:split]) for right in all_products(matrices[split:])]


randomizer = random.Random(601923)
for case in data["matrices"]:
    dimensions, plan = case["dimensions"], case["plan"]
    matrices = [Matrix([[randomizer.randrange(-2, 3) for _ in range(columns)] for _ in range(rows)])
                for rows, columns in zip(dimensions, dimensions[1:])]
    expected = all_products(matrices)
    # Actual exact integer products establish associativity for the changed data,
    # while measured elementary-operation counts independently price the witness.
    assert all(result.values == expected[0].values for result in expected)
    native_value, expression, operations = environments["matrix"]["matrix_chain"](dimensions)
    names = {"A" + str(index): matrix for index, matrix in enumerate(matrices)}
    native_product = eval(expression, {"__builtins__": {}}, names)
    javascript_product = eval(plan["expression"].replace("×", "*"), {"__builtins__": {}}, names)
    optimum = min(result.products for result in expected)
    assert native_value == native_product.products == javascript_product.products == plan["result"] == optimum
    assert native_product.values == javascript_product.values == expected[0].values
    checks["changed exact matrix products and priced expressions"] += 1


@cache
def remove_search(live):
    if not live:
        return 0
    return max((live[index-1] if index else 1) * value *
               (live[index+1] if index+1 < len(live) else 1) +
               remove_search(live[:index] + live[index+1:])
               for index, value in enumerate(live))


for case in data["balloons"]:
    values, plan = case["values"], case["plan"]
    optimum, order = environments["balloons"]["balloon_plan"](values)
    assert optimum == plan["result"] == remove_search(tuple(values))
    assert sorted(order) == list(range(len(values)))
    assert sum(environments["balloons"]["replay"](values, order)) == optimum
    checks["changed live-sequence removal search"] += 1
assert remove_search((-2,)) == -2
checks["signed forced-removal changed task"] += 1


def bipartite_optimum(weights, edges, root, forbidden=None):
    """Positive-weight independent set = total positive mass - min vertex cover.

    The cover is a source/sink cut on the tree's bipartition. This uses no
    parent-selected DP recurrence, and also tests cases with arbitrary roots.
    """
    n = len(weights)
    adjacency = [[] for _ in range(n)]
    for first, second in edges:
        adjacency[first].append(second)
        adjacency[second].append(first)
    color = {root: 0}
    pending = [root]
    while pending:
        current = pending.pop()
        for other in adjacency[current]:
            if other not in color:
                color[other] = 1 - color[current]
                pending.append(other)
    assert len(color) == n
    mass = [max(0, value) if index != forbidden else 0 for index, value in enumerate(weights)]
    capacity = [[0] * (n+2) for _ in range(n+2)]
    source, sink, infinity = n, n+1, sum(mass)+1
    for node, value in enumerate(mass):
        if color[node] == 0:
            capacity[source][node] = value
        else:
            capacity[node][sink] = value
    for first, second in edges:
        if color[first] == 1:
            first, second = second, first
        capacity[first][second] = infinity
    flow = 0
    while True:
        parents = {source: None}
        queue = deque([source])
        while queue and sink not in parents:
            first = queue.popleft()
            for second, value in enumerate(capacity[first]):
                if value > 0 and second not in parents:
                    parents[second] = first
                    queue.append(second)
        if sink not in parents:
            break
        amount, node = infinity, sink
        while node != source:
            amount = min(amount, capacity[parents[node]][node])
            node = parents[node]
        node = sink
        while node != source:
            first = parents[node]
            capacity[first][node] -= amount
            capacity[node][first] += amount
            node = first
        flow += amount
    return sum(mass) - flow


for case in data["trees"]:
    weights, edges, root, plan = (case[key] for key in ("weights", "edges", "root", "plan"))
    for query in case["queries"]:
        # Find the subtree independently: remove its one parent edge, then BFS.
        cut = plan["parents"][query["node"]]
        adjacency = [[] for _ in weights]
        for first, second in edges:
            if {first, second} == {query["node"], cut}:
                continue
            adjacency[first].append(second)
            adjacency[second].append(first)
        allowed, pending = {query["node"]}, [query["node"]]
        while pending:
            current = pending.pop()
            for other in adjacency[current]:
                if other not in allowed:
                    allowed.add(other)
                    pending.append(other)
        ids = sorted(allowed)
        remap = {old: new for new, old in enumerate(ids)}
        local_edges = [(remap[a], remap[b]) for a, b in edges if a in allowed and b in allowed]
        local_root = remap[query["node"]]
        expected = bipartite_optimum([weights[node] for node in ids], local_edges, local_root,
                                    local_root if query["parent"] else None)
        selected = query["selected"]
        assert len(selected) == len(set(selected)) and set(selected) <= allowed
        assert not query["parent"] or query["node"] not in selected
        assert not any(a in selected and b in selected for a, b in edges)
        assert expected == query["result"] == sum(weights[node] for node in selected)
        native, witness = environments["tree"]["independent_tree"](
            [weights[node] for node in ids], local_edges, local_root, query["parent"])
        assert native == expected == sum(weights[ids[node]] for node in witness)
        checks["subtree conditional witness versus independent min cut"] += 1
for n in (23, 41, 77):
    weights = [randomizer.randrange(-20, 50) for _ in range(n)]
    edges = [(randomizer.randrange(child), child) for child in range(1, n)]
    for root in (0, n//2, n-1):
        for parent in (False, True):
            optimum, selected = environments["tree"]["independent_tree"](weights, edges, root, parent)
            assert optimum == bipartite_optimum(weights, edges, root, root if parent else None)
            assert optimum == sum(weights[node] for node in selected)
            checks["larger actual native tree versus min cut"] += 1


def ranked_distinct(bound):
    if bound <= 0:
        return 0
    digits = list(map(int, str(bound)))
    length = len(digits)
    total = sum(9 * math.perm(9, size-1) for size in range(1, min(length, 11)))
    if length > 10:
        return total
    used = set()
    for position, digit in enumerate(digits):
        lower = sum(candidate not in used for candidate in range(1 if position == 0 else 0, digit))
        total += lower * math.perm(9-position, length-position-1)
        if digit in used:
            return total
        used.add(digit)
    return total + 1


valid = [value for value in range(1, 1000000) if len(set(str(value))) == len(str(value))]
for case in data["digits"]:
    bound = case["bound"]
    assert case["total"] == bisect_right(valid, bound) == ranked_distinct(bound)
    assert environments["digits"]["count_distinct"](bound) == case["total"]
    for state in case["states"]:
        prefix, length = state["prefix"], len(str(bound))

        def prefix_count(text):
            first = int(text or "0") * 10**(length-len(text))
            last = min(bound, first + 10**(length-len(text)) - 1)
            return max(0, bisect_right(valid, last) - bisect_left(valid, max(1, first)))

        assert state["remaining"] == prefix_count(prefix)
        assert all(branch["count"] == prefix_count(prefix + str(branch["digit"]))
                   for branch in state["branches"])
        if state["remaining"]:
            first = int(prefix or "0") * 10**(length-len(prefix))
            assert state["completion"] == valid[bisect_left(valid, max(1, first))]
        else:
            assert state["completion"] is None
        checks["prefix exact numeric interval and per-digit partition"] += 1
for bound in [10**18, 9876543210, 9876543209, 1000000000, 2345619870, 100200300400,
              *[randomizer.randrange(10**18+1) for _ in range(25)],
              *[randomizer.randrange(10**10) for _ in range(35)]]:
    assert environments["digits"]["count_distinct"](bound) == ranked_distinct(bound)
    checks["large native digit bound versus combinatorial rank"] += 1
for first, last in [(0, 0), (0, 102), (99, 213), (12345, 98765), (9876543200, 9876543210)]:
    assert environments["digits"]["count_range"](first, last) == ranked_distinct(last)-ranked_distinct(first-1)
    checks["changed native inclusive range"] += 1
adjacent = [value for value in range(100, 131) if all(a != b for a, b in zip(str(value), str(value)[1:]))]
distinct = [value for value in range(100, 131) if len(set(str(value))) == len(str(value))]
assert len(adjacent) == 19 and len(distinct) == 17 and set(adjacent)-set(distinct) == {101, 121}
checks["changed adjacent versus all-distinct task"] += 1

for item in data["sourceHashes"]:
    assert hashlib.sha256((ROOT/item["path"]).read_bytes()).hexdigest() == item["sha256"], item["path"]
record = {"checkedAt": datetime.now(timezone.utc).isoformat(), "passed": True,
          "scope": "Complementary reviewer checks; actual values/outputs, not author-test repetition or a proof for every input.",
          "checks": dict(checks), "sourceHashes": data["sourceHashes"],
          "unchangedOriginalSupport": data["preserved"], "preservedPractice": data["preservedPractice"]}
(ROOT/"scratch/dp-state-families-independent/native-results.json").write_text(
    json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(json.dumps(record, indent=2))
