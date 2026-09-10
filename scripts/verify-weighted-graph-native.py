"""Independent finite oracles for the complete displayed Python programs."""
import contextlib
import io
import itertools
import json
import random
import sys
from math import inf

programs = json.load(open(sys.argv[1], encoding="utf-8"))
namespaces = {}
for name, example in programs.items():
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(example["code"], name, "exec"), namespace)
    namespaces[name] = namespace

rng = random.Random(90715)

def path_oracle(n, edges, source):
    reachable = [[u == v for v in range(n)] for u in range(n)]
    for u, v, _ in edges:
        reachable[u][v] = True
    for k in range(n):
        for u in range(n):
            for v in range(n):
                reachable[u][v] |= reachable[u][k] and reachable[k][v]
    answer = [inf] * n
    def paths(u, seen, total):
        answer[u] = min(answer[u], total)
        for a, v, w in edges:
            if a == u and v not in seen:
                paths(v, seen | {v}, total + w)
    paths(source, {source}, 0)
    negative = set()
    for start in range(n):
        def cycles(u, seen, total):
            for a, v, w in edges:
                if a != u:
                    continue
                if v == start and total + w < 0:
                    negative.update(seen)
                elif v not in seen:
                    cycles(v, seen | {v}, total + w)
        cycles(start, {start}, 0)
    for v in range(n):
        if any(reachable[source][k] and reachable[k][v] for k in negative):
            answer[v] = -inf
    return answer

def check_witness(distance, parent, edges, source, route):
    for target, total in enumerate(distance):
        if total in (inf, -inf):
            continue
        vertices, ids = route(parent, source, target)
        assert vertices[0] == source and vertices[-1] == target
        assert len(set(vertices)) == len(vertices)
        assert sum(edges[i][2] for i in ids) == total
        assert all(edges[i][:2] == (vertices[j], vertices[j + 1]) for j, i in enumerate(ids))

path_cases = 0
for _ in range(240):
    n = rng.randrange(1, 6)
    edges = [(u, v, rng.randrange(-5, 8)) for u in range(n) for v in range(n) if rng.randrange(4) == 0]
    if edges and rng.randrange(3) == 0:
        u, v, w = rng.choice(edges)
        edges.append((u, v, w - 1))  # Parallel occurrence IDs remain meaningful.
    all_pairs = namespaces["floyd"]["floyd_warshall"](n, edges)
    if any(all_pairs[v][v] == -inf for v in range(n)):
        try:
            namespaces["johnson"]["johnson"](n, edges)
            raise AssertionError("Johnson accepted a negative cycle")
        except ValueError:
            pass
    else:
        assert namespaces["johnson"]["johnson"](n, edges) == all_pairs
    for source in range(n):
        expected = path_oracle(n, edges, source)
        distance, parent = namespaces["bellman"]["bellman_ford"](n, edges, source)
        assert distance == expected == all_pairs[source]
        check_witness(distance, parent, edges, source, namespaces["bellman"]["route"])
        positive = [(u, v, abs(w)) for u, v, w in edges]
        distance, parent, _ = namespaces["dijkstra"]["dijkstra"](n, positive, source)
        assert distance == path_oracle(n, positive, source)
        check_witness(distance, parent, positive, source, namespaces["dijkstra"]["route"])
        binary = [(u, v, abs(w) % 2) for u, v, w in edges]
        distance, parent = namespaces["zeroOne"]["zero_one_bfs"](n, binary, source)
        assert distance == path_oracle(n, binary, source)
        check_witness(distance, parent, binary, source, namespaces["zeroOne"]["route"])
        limit = rng.randrange(5)
        budget = [inf] * n
        def walks(u, cost, depth):
            budget[u] = min(budget[u], cost)
            if depth < limit:
                for a, v, w in edges:
                    if a == u:
                        walks(v, cost + w, depth + 1)
        walks(source, 0, 0)
        assert namespaces["budget"]["edge_budget"](n, edges, source, limit) == budget
        path_cases += 1

def components(n, edges):
    unseen, result = set(range(n)), []
    while unseen:
        group = {unseen.pop()}
        while True:
            expanded = group | {v for u, v, _ in edges if u in group} | {u for u, v, _ in edges if v in group}
            if expanded == group:
                break
            group = expanded
        unseen -= group
        result.append(frozenset(group))
    return set(result)

def forest_oracle(n, edges):
    original = components(n, edges)
    needed = n - len(original)
    feasible = [sum(w for _, _, w in chosen) for chosen in itertools.combinations(edges, needed) if components(n, chosen) == original]
    return min(feasible), needed, original

def bottleneck_oracle(n, edges, source):
    result = [inf] * n
    def paths(u, seen, bottleneck):
        result[u] = min(result[u], bottleneck)
        for a, b, weight in edges:
            v = b if a == u else a if b == u else None
            if v is not None and v not in seen:
                paths(v, seen | {v}, max(bottleneck, weight))
    paths(source, {source}, -inf)  # Explicit empty-route maximum convention.
    return result

for _ in range(220):
    n = rng.randrange(1, 6)
    edges = [(rng.randrange(n), rng.randrange(n), rng.randrange(-7, 8)) for _ in range(rng.randrange(9))]
    minimum, needed, original = forest_oracle(n, edges)
    for key, name in [("kruskal", "kruskal"), ("prim", "prim_forest")]:
        total, ids, count = namespaces[key][name](n, edges)
        assert total == minimum and len(ids) == needed and count == len(original)
        assert len(set(ids)) == len(ids)
        assert components(n, [edges[i] for i in ids]) == original
        for source in range(n):
            assert bottleneck_oracle(n, [edges[i] for i in ids], source) == bottleneck_oracle(n, edges, source)

for _ in range(160):
    n = rng.randrange(1, 6)
    edges = [(u, v, rng.randrange(-5, 7)) for u in range(n) for v in range(n) if rng.randrange(5) == 0]
    valid = [p for p in itertools.permutations(range(n)) if all(p.index(u) < p.index(v) for u, v, _ in edges)]
    order, cycle = namespaces["topological"]["dfs_order_or_cycle"](n, edges)
    assert (cycle is None) == bool(valid)
    if cycle is not None:
        assert cycle[0] == cycle[-1]
        assert all(any(a == u and b == v for a, b, _ in edges) for u, v in zip(cycle, cycle[1:]))
        try:
            namespaces["topological"]["topological_order"](n, edges)
            raise AssertionError("Cycle was accepted")
        except ValueError:
            pass
    else:
        assert tuple(order) in valid
        assert tuple(namespaces["topological"]["topological_order"](n, edges)) in valid
        for source in range(n):
            distance, parent = namespaces["dag"]["dag_shortest"](n, edges, source)
            assert distance == path_oracle(n, edges, source)
            check_witness(distance, parent, edges, source, namespaces["dag"]["route"])
        durations = [rng.randrange(7) for _ in range(n)]
        start, finish, maximum, chain = namespaces["schedule"]["earliest_schedule"](n, edges, durations)
        lengths = []
        def chains(u, total):
            lengths.append(total)
            for a, v, _ in edges:
                if a == u:
                    chains(v, total + durations[v])
        for u in range(n):
            chains(u, durations[u])
        assert maximum == max(lengths) == sum(durations[u] for u in chain)
        assert all(start[v] >= finish[u] for u, v, _ in edges)

for _ in range(80):
    points = [(rng.randrange(-5, 6), rng.randrange(-5, 6)) for _ in range(rng.randrange(1, 5))]
    edges = [(u, v, abs(a - c) + abs(b - d)) for u, (a, b) in enumerate(points) for v, (c, d) in enumerate(points) if u < v]
    total, selected = namespaces["points"]["connect_points"](points)
    assert total == forest_oracle(len(points), edges)[0]
    assert len(selected) == len(points) - 1

assert namespaces["kruskal"]["kruskal"](0, []) == (0, [], 0)
assert namespaces["prim"]["prim_forest"](0, []) == (0, [], 0)
assert namespaces["topological"]["dfs_order_or_cycle"](0, []) == ([], None)
assert namespaces["schedule"]["earliest_schedule"](0, [], []) == ([], [], 0, [])
assert namespaces["floyd"]["floyd_warshall"](0, []) == []
print(f"Native oracles: {path_cases} signed/binary/nonnegative/budget sources, 220 exhaustive forests, 160 order/cycle/schedule graphs, 80 implicit geometric graphs; witnesses and empty contracts passed.")
