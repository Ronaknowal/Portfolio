"""Independent finite oracles: Boolean truth, subset enumeration, graph cover enumeration.
No tested finite range is presented as a proof of an asymptotic complexity theorem.
"""
import contextlib
import io
import itertools
import json
import random
import sys

programs = json.load(open(sys.argv[1], encoding="utf-8"))
namespaces = {}
for name, example in programs.items():
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(example["code"], name, "exec"), namespace)
    namespaces[name] = namespace

rng = random.Random(81204)

def all_subsets(n):
    return [chosen for k in range(n + 1) for chosen in itertools.combinations(range(n), k)]

def sat_oracle(n, clauses):
    # Independent truth by sets of positive/negative literals made true.
    answers = []
    for chosen in all_subsets(n):
        true_literals = {i + 1 if i in chosen else -(i + 1) for i in range(n)}
        if all(set(clause) & true_literals for clause in clauses):
            answers.append([i in chosen for i in range(n)])
    return answers

formula_cases = [(0, []), (0, [[]]), (1, [[1], [-1]]), (1, [[1, 1], [1]])]
for _ in range(300):
    n = rng.randrange(1, 5)
    formula_cases.append((n, [[rng.choice((-1, 1)) * rng.randrange(1, n + 1)
                              for _ in range(rng.randrange(4))]
                             for _ in range(rng.randrange(4))]))
clique_witnesses = 0
for n, clauses in formula_cases:
    truth = sat_oracle(n, clauses)
    found = namespaces["certificate"]["find_assignment"](n, clauses)
    assert (found is not None) == bool(truth)
    assert found is None or found in truth
    graph, edges, target = namespaces["clique"]["to_clique"](n, clauses)
    cliques = [chosen for chosen in all_subsets(len(graph)) if len(chosen) == target
               and all(pair in edges for pair in itertools.combinations(chosen, 2))]
    assert bool(cliques) == bool(truth)
    for chosen in cliques:
        recovered = namespaces["clique"]["recover_assignment"](n, clauses, list(chosen))
        assert recovered in truth
        clique_witnesses += 1
    def oracle(size, formula):
        return bool(sat_oracle(size, formula))
    recovered, queries = namespaces["selfReduction"]["search_with_oracle"](n, clauses, oracle)
    assert (recovered is not None) == bool(truth)
    assert recovered is None or recovered in truth
    assert queries == (n + 1 if truth else 1)

extension_states = 0
for _ in range(100):
    n = rng.randrange(1, 5)
    # At most one long clause keeps the independent exhaustive extension oracle finite.
    clauses = [[rng.choice((-1, 1)) * rng.randrange(1, n + 1)
                for _ in range(rng.randrange(9))]]
    count, converted = namespaces["threeCnf"]["at_most_three_cnf"](n, clauses)
    assert all(len(clause) <= 3 for clause in converted)
    assert count - n == max(0, len(clauses[0]) - 3)
    converted_answers = sat_oracle(count, converted)
    source_answers = sat_oracle(n, clauses)
    for chosen in all_subsets(n):
        assignment = [i in chosen for i in range(n)]
        assert (assignment in source_answers) == any(answer[:n] == assignment for answer in converted_answers)
        extension_states += 1

graph_count = 0
possible = list(itertools.combinations(range(5), 2))
for mask in range(1 << len(possible)):
    edges = [edge for index, edge in enumerate(possible) if mask & (1 << index)]
    subsets = all_subsets(5)
    covers = [chosen for chosen in subsets if all(set(edge) & set(chosen) for edge in edges)]
    optimum = min(map(len, covers))
    complementary = namespaces["complement"]["complement"](5, edges)
    for chosen in subsets:
        is_clique = all(pair in edges for pair in itertools.combinations(chosen, 2))
        remaining = set(range(5)) - set(chosen)
        is_complement_cover = all(set(edge) & remaining for edge in complementary)
        assert is_clique == is_complement_cover
    for budget in range(7):
        answer, calls = namespaces["parameter"]["bounded_cover"](5, edges, budget)
        assert (answer is not None) == (optimum <= budget)
        assert calls <= 2 ** (min(budget, 5) + 1) - 1
        if answer is not None:
            assert len(answer) == len(set(answer)) <= budget
            assert all(set(edge) & set(answer) for edge in edges)
    cover, matching = namespaces["approximation"]["matching_cover"](5, edges)
    assert len(cover) <= 2 * optimum
    assert all(set(edge) & set(cover) for edge in edges)
    assert len(set(itertools.chain.from_iterable(matching))) == 2 * len(matching)
    assert all(edge in edges for edge in matching)
    assert len(matching) <= optimum
    graph_count += 1

numeric_cases = 0
for _ in range(350):
    values = [rng.randrange(11) for _ in range(rng.randrange(9))]
    target = rng.randrange(35)
    feasible = [chosen for chosen in all_subsets(len(values)) if sum(values[i] for i in chosen) == target]
    answer = namespaces["subset"]["subset_sum"](values, target)
    assert (answer is not None) == bool(feasible)
    if answer is not None:
        assert len(answer) == len(set(answer))
        assert all(0 <= i < len(values) for i in answer)
        assert sum(values[i] for i in answer) == target
    weights = [rng.randrange(-6, 12) for _ in values]
    independent = [chosen for chosen in all_subsets(len(weights))
                   if all(b - a > 1 for a, b in zip(chosen, chosen[1:]))]
    expected = max(sum(weights[i] for i in chosen) for chosen in independent)
    total, chosen = namespaces["path"]["path_independent_set"](weights)
    assert total == expected == sum(weights[i] for i in chosen)
    assert tuple(chosen) in independent
    numeric_cases += 1

assert namespaces["clique"]["recover_assignment"](0, [], []) == []
assert namespaces["parameter"]["bounded_cover"](0, [], 100) == ([], 1)
assert namespaces["approximation"]["matching_cover"](0, []) == ([], [])
assert namespaces["approximation"]["simple_graph"](2, [(0, 1), (1, 0)]) == [(0, 1)]
bad_calls = [
    lambda: namespaces["certificate"]["satisfies"](1, [[2]], [True]),
    lambda: namespaces["certificate"]["satisfies"](1, [[1]], [1]),
    lambda: namespaces["clique"]["to_clique"](1, [[1, 1, 1, 1]]),
    lambda: namespaces["clique"]["recover_assignment"](1, [[1], [-1]], [0, 1]),
    lambda: namespaces["parameter"]["bounded_cover"](2, [(0, 0)], 1),
    lambda: namespaces["parameter"]["bounded_cover"](2, [(0, 1)], -1),
    lambda: namespaces["subset"]["subset_sum"]([1, -1], 0),
]
for call in bad_calls:
    try:
        call()
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid input accepted")
print(f"Independent checks: {len(formula_cases)} formulas, {clique_witnesses} recovered cliques, "
      f"{extension_states} existential-extension assignments, {graph_count} exhaustive graphs "
      f"at seven budgets, {numeric_cases} subset/path cases; empty and invalid contracts pass.")
