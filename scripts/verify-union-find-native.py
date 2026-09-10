"""Independent small-state oracle checks for the actual displayed Python programs."""
import contextlib
import io
import itertools
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    examples = json.load(source)
programs = {}
for name, example in examples.items():
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(example["code"], name, "exec"), namespace)
    programs[name] = namespace

DSU = programs["basic"]["DisjointSet"]

def partition(n, edges):
    """Explicit set union is a different representation from parent forests."""
    groups = [{item} for item in range(n)]
    for a, b in edges:
        left = next(group for group in groups if a in group)
        right = next(group for group in groups if b in group)
        if left is not right:
            groups.remove(left)
            groups.remove(right)
            groups.append(left | right)
    return sorted(sorted(group) for group in groups)

states = 0
edges = list(itertools.combinations_with_replacement(range(4), 2))
for length in range(4):
    for sequence in itertools.product(edges, repeat=length):
        dsu = DSU(4)
        seen = []
        for a, b in sequence:
            before = partition(4, seen)
            merged = not any(a in group and b in group for group in before)
            assert dsu.union(a, b) == merged
            seen.append((a, b))
        expected = partition(4, sequence)
        assert sorted(dsu.groups()) == expected
        assert dsu.count == len(expected)
        roots = [dsu.find(item) for item in range(4)]
        for item in range(4):
            expected_group = next(group for group in expected if item in group)
            assert dsu.component_size(item) == len(expected_group)
            assert dsu.find(item) == roots[item]
        states += 1

for invalid in [-1, 1.5, True, "4"]:
    try:
        DSU(invalid)
        raise AssertionError("invalid constructor accepted")
    except ValueError:
        pass
for invalid in [-1, 4, 1.5, True, "1"]:
    dsu = DSU(4)
    snapshot = dsu.parent[:], dsu.size[:], dsu.count
    try:
        dsu.union(0, invalid)
        raise AssertionError("invalid endpoint accepted")
    except IndexError:
        assert snapshot == (dsu.parent, dsu.size, dsu.count)
assert DSU(0).groups() == []

def flood_count(rows, columns, opened):
    remaining = set(opened)
    count = 0
    while remaining:
        stack = [remaining.pop()]
        count += 1
        while stack:
            row, column = stack.pop()
            for candidate in list(remaining):
                if abs(candidate[0] - row) + abs(candidate[1] - column) == 1:
                    remaining.remove(candidate)
                    stack.append(candidate)
    return count

islands = programs["islands"]["island_counts"]
grid_sequences = 0
for sequence in itertools.product(list(itertools.product(range(2), repeat=2)), repeat=5):
    opened = set()
    expected = []
    for cell in sequence:
        opened.add(cell)
        expected.append(flood_count(2, 2, opened))
    assert islands(2, 2, sequence) == expected
    grid_sequences += 1

equations_possible = programs["equations"]["equations_possible"]
relations = [(a, operator, b) for a in "ab" for b in "ab" for operator in ("==", "!=")]
equation_cases = 0
for sequence in itertools.product(relations, repeat=3):
    possible = False
    for values in itertools.product(range(2), repeat=2):
        assignment = dict(zip("ab", values))
        if all((assignment[a] == assignment[b]) == (operator == "==") for a, operator, b in sequence):
            possible = True
    assert equations_possible(sequence) == possible
    equation_cases += 1

Rollback = programs["rollback"]["RollbackDSU"]
rollback_cases = 0
for checkpoint in [-1, 1, 1.5, True]:
    try:
        Rollback(0).rollback(checkpoint)
        raise AssertionError("invalid rollback checkpoint accepted")
    except ValueError:
        pass
for sequence in itertools.product(edges, repeat=3):
    dsu = Rollback(4)
    snapshots = [(dsu.snapshot(), dsu.parent[:], dsu.size[:], dsu.count)]
    for edge in sequence:
        dsu.union(*edge)
        snapshots.append((dsu.snapshot(), dsu.parent[:], dsu.size[:], dsu.count))
    for checkpoint, parent, size, count in reversed(snapshots):
        dsu.rollback(checkpoint)
        assert (dsu.parent, dsu.size, dsu.count) == (parent, size, count)
    rollback_cases += 1

pair_counts = programs["exercise"]["disconnected_pairs"]
for sequence in itertools.product(edges, repeat=3):
    expected = []
    for end in range(1, len(sequence) + 1):
        groups = partition(4, sequence[:end])
        expected.append(sum(not any(a in group and b in group for group in groups)
                            for a, b in itertools.combinations(range(4), 2)))
    assert pair_counts(4, sequence) == expected

records = programs["identifiers"]["merge_records"]
for record_list in itertools.product([set(), {"x"}, {"y"}, {"x", "y"}], repeat=4):
    overlaps = [(a, b) for a, b in itertools.combinations(range(4), 2) if record_list[a] & record_list[b]]
    assert sorted(records(record_list)) == partition(4, overlaps)

Labeled = programs["metadata"]["LabeledDisjointSet"]
for sequence in itertools.product(edges, repeat=3):
    dsu = Labeled(4)
    for edge in sequence:
        dsu.union(*edge)
    for group in partition(4, sequence):
        for item in group:
            assert dsu.component_label(item) == min(group)
            assert dsu.component_size(item) == len(group)

print(f"Native oracles passed: {states} partitions, {grid_sequences} grid sequences, "
      f"{equation_cases} exhaustively assigned equation cases, {rollback_cases} rollback histories, "
      "1000 disconnected-pair sequences, 256 identifier datasets, 1000 root-metadata sequences.")
