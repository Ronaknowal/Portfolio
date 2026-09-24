"""Independent finite cut, feasible-assignment, matching and label oracles."""
import contextlib
import io
import itertools
import json
import random
import sys
from fractions import Fraction

examples = json.load(open(sys.argv[1], encoding="utf8"))
namespaces = {}
for name, example in examples.items():
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(example["code"], name + ".py", "exec"), namespace)
    namespaces[name] = namespace
ek = namespaces["augment"]["edmonds_karp"]
dinic = namespaces["dinic"]["dinic"]

def brute_cut(n, edges):
    return min(sum(cap for u, v, cap in edges if mask >> u & 1 and not mask >> v & 1)
               for mask in range(1 << n) if mask & 1 and not mask >> (n - 1) & 1)

def feasible(n, edges, flows, source=0, sink=None, limits=None):
    sink = n - 1 if sink is None else sink
    balance = [0] * n
    incoming = [0] * n
    for (u, v, cap), amount in zip(edges, flows):
        if not 0 <= amount <= cap:
            return False, 0
        balance[u] -= amount
        balance[v] += amount
        incoming[v] += amount
    if any(balance[v] for v in range(n) if v not in (source, sink)):
        return False, 0
    if limits and any(incoming[v] > limit for v, limit in limits.items()):
        return False, 0
    return balance[source] + balance[sink] == 0, -balance[source]

def check_network(n, edges):
    expected = brute_cut(n, edges)
    for solve in (ek, dinic):
        result = solve(n, edges, 0, n - 1)
        valid, value = feasible(n, edges, result["flows"])
        assert valid and value == expected == result["value"] == result["cut_capacity"]
        side = set(result["source_side"])
        assert 0 in side and n - 1 not in side
        assert sum(cap for u, v, cap in edges if u in side and v not in side) == value

arcs = [(u, v) for u in range(3) for v in range(3) if u != v]
assignments = 0
for capacities in itertools.product(range(3), repeat=6):
    edges = [(u, v, cap) for (u, v), cap in zip(arcs, capacities)]
    check_network(3, edges)
    best = 0
    for flows in itertools.product(*(range(cap + 1) for cap in capacities)):
        valid, value = feasible(3, edges, flows)
        if valid:
            best = max(best, value)
        assignments += 1
    assert best == brute_cut(3, edges)
rng = random.Random(36190)
for _ in range(350):
    n = rng.randrange(2, 8)
    edges = [(rng.randrange(n), rng.randrange(n), rng.randrange(6))
             for _ in range(rng.randrange(21))]
    check_network(n, edges)
check_network(2, [(0, 1, 10**100), (1, 0, 10**99), (0, 0, 7), (0, 1, 3)])

matching = namespaces["matching"]["bipartite_matching"]
for mask in range(512):
    pairs = [(u, v) for u in range(3) for v in range(3) if mask >> (3*u+v) & 1]
    result = matching(3, 3, pairs + pairs[:1])
    best = max((len(chosen) for size in range(4) for chosen in itertools.combinations(pairs, size)
                if len({u for u, _ in chosen}) == size and len({v for _, v in chosen}) == size), default=0)
    cover = {("L", u) for u in result["cover_left"]} | {("R", v) for v in result["cover_right"]}
    assert len(result["matching"]) == len(cover) == best
    assert all(("L", u) in cover or ("R", v) in cover for u, v in pairs)
    assert all(pair in pairs for pair in result["matching"])
    left = set(result["shortage_left"])
    neighbors = {v for u, v in pairs if u in left}
    assert neighbors == set(result["shortage_neighbors"])
    assert len(left) - len(neighbors) == 3 - best
for left_count, right_count in ((0, 0), (0, 3), (3, 0), (1, 4), (4, 1)):
    result = matching(left_count, right_count, [])
    assert not result["matching"] and result["shortage_left"] == list(range(left_count))

circulation = namespaces["circulation"]["lower_bound_circulation"]
for _ in range(250):
    n = rng.randrange(1, 5)
    edges = []
    for _ in range(rng.randrange(6)):
        lower = rng.randrange(3)
        edges.append((rng.randrange(n), rng.randrange(n), lower, lower + rng.randrange(3)))
    found = False
    for flows in itertools.product(*(range(lower, upper + 1) for _, _, lower, upper in edges)):
        net = [0] * n
        for (u, v, _, _), amount in zip(edges, flows):
            net[u] -= amount
            net[v] += amount
        if not any(net):
            found = True
            break
    result = circulation(n, edges)
    assert (result is not None) == found
    if result is not None:
        net = [0] * n
        for (u, v, lower, upper), amount in zip(edges, result):
            assert lower <= amount <= upper
            net[u] -= amount
            net[v] += amount
        assert not any(net)

vertex_flow = namespaces["vertex"]["vertex_capacity_flow"]
supply_flow = namespaces["supply"]["allocate_supplies"]
for _ in range(160):
    n = 4
    edges = [(u, v, rng.randrange(3)) for u, v in ((0, 1), (0, 2), (1, 2), (2, 1), (1, 3), (2, 3))]
    limits = {1: rng.randrange(4), 2: rng.randrange(4)}
    best = 0
    for flows in itertools.product(*(range(cap + 1) for _, _, cap in edges)):
        valid, value = feasible(n, edges, flows, limits=limits)
        if valid:
            best = max(best, value)
    value, flows = vertex_flow(n, edges, limits, 0, 3)
    assert value == best and feasible(n, edges, flows, limits=limits) == (True, best)
    allocation_edges = [(u, v, rng.randrange(3)) for u in (0, 1) for v in (2, 3)]
    supply = {0: rng.randrange(4), 1: rng.randrange(4)}
    demand = {2: rng.randrange(4), 3: rng.randrange(4)}
    best = 0
    for candidate in itertools.product(*(range(cap + 1) for _, _, cap in allocation_edges)):
        out = [0] * 4
        inside = [0] * 4
        for (u, v, _), amount in zip(allocation_edges, candidate):
            out[u] += amount
            inside[v] += amount
        if all(out[u] <= supply[u] for u in supply) and all(inside[v] <= demand[v] for v in demand):
            best = max(best, sum(candidate))
    result = supply_flow(4, allocation_edges, supply, demand)
    assert result["delivered"] == best
    assert result["all_demands_met"] == (best == sum(demand.values()))
    assert sum(result["unmet"].values()) == sum(demand.values()) - best
    assert sum(result["unused"].values()) == sum(supply.values()) - best

binary_cut = namespaces["pixels"]["binary_label_cut"]
for _ in range(220):
    n = rng.randrange(8)
    background = [rng.randrange(6) for _ in range(n)]
    foreground = [rng.randrange(6) for _ in range(n)]
    neighbors = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.randrange(3) == 0]
    penalty = rng.randrange(6)
    def energy(labels):
        return sum(foreground[i] if label else background[i] for i, label in enumerate(labels)) + penalty * sum(labels[u] != labels[v] for u, v in neighbors)
    optimum = min(energy(labels) for labels in itertools.product((False, True), repeat=n))
    labels, unary, pairwise, value = binary_cut(background, foreground, neighbors, penalty)
    assert energy(labels) == unary + pairwise == value == optimum

trial = namespaces["contraction"]["karger_trial"]
class NeedChoice(Exception):
    def __init__(self, count):
        self.count = count

class ForcedChoices:
    def __init__(self, choices):
        self.choices = iter(choices)
    def choice(self, options):
        try:
            index = next(self.choices)
        except StopIteration:
            raise NeedChoice(len(options))
        return options[index]

def exact_distribution(n, edges, choices=(), probability=Fraction(1)):
    try:
        value, side = trial(n, edges, ForcedChoices(choices))
    except NeedChoice as event:
        outcomes = {}
        for index in range(event.count):
            for key, mass in exact_distribution(n, edges, choices + (index,), probability / event.count).items():
                outcomes[key] = outcomes.get(key, Fraction(0)) + mass
        return outcomes
    side = frozenset(side)
    assert 0 in side and len(side) < n
    assert value == sum((u in side) != (v in side) for u, v in edges)
    return {(value, side): probability}

simple_pairs = list(itertools.combinations(range(4), 2))
for mask in range(64):
    edges = [pair for i, pair in enumerate(simple_pairs) if mask >> i & 1]
    distribution = exact_distribution(4, edges)
    assert sum(distribution.values()) == 1
    cuts = {frozenset(v for v in range(4) if chosen >> v & 1):
            sum(bool(chosen >> u & 1) != bool(chosen >> v & 1) for u, v in edges)
            for chosen in range(1, 15, 2)}
    minimum = min(cuts.values())
    assert all(value >= minimum for value, _ in distribution)
    if minimum == 0:
        assert all(value == 0 for value, _ in distribution)
    else:
        for side, value in cuts.items():
            if value == minimum:
                assert distribution.get((value, side), 0) >= Fraction(1, 6)
parallel_distribution = exact_distribution(3, [(0, 1)] * 5 + [(1, 2), (0, 2), (0, 0)])
assert sum(mass for (value, _), mass in parallel_distribution.items() if value == 2) == Fraction(5, 7)
print(f"Independent native oracles passed: 1080 networks with both solvers, {assignments} feasible-flow candidates, 512 matching/cover cases, 250 circulation cases, 160 vertex and160 supply cases, 220 binary energies, 65 exact contraction distributions.")
