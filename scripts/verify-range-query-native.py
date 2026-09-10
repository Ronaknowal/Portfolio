"""Finite direct-array and exhaustive-subsequence oracles, separate from tree logic."""
import contextlib
import io
import itertools
import json
import random
import sys

programs = json.load(open(sys.argv[1], encoding="utf-8"))
spaces = {}
for name, program in programs.items():
    space = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(program["code"], name, "exec"), space)
    spaces[name] = space

rng = random.Random(71320)
counts = {"tree_arrays": 0, "dynamic_operations": 0, "deque_arrays": 0,
          "weighted_cases": 0, "frequency_cases": 0, "subtrees": 0}
for _ in range(300):
    values = [rng.randrange(-9, 10) for _ in range(rng.randrange(13))]
    adjustments = []
    adjusted = values[:]
    for _ in range(12):
        left, right = sorted([rng.randrange(len(values) + 1), rng.randrange(len(values) + 1)])
        delta = rng.randrange(-8, 9)
        adjustments.append((left, right, delta))
        adjusted[left:right] = [item + delta for item in adjusted[left:right]]
    assert spaces["differences"]["add_ranges"](values, adjustments) == adjusted
    assert spaces["prefixes"]["prefixes"](values) == [sum(values[:end]) for end in range(len(values) + 1)]
    segment = spaces["segment"]["SegmentTree"](values, lambda a, b: a + b, 0)
    ordered = spaces["segment"]["SegmentTree"]([str(x) + ";" for x in values], lambda a, b: a + b, "")
    fenwick = spaces["fenwick"]["Fenwick"](values)
    sparse = spaces["sparse"]["SparseMinimum"](values)
    for left in range(len(values) + 1):
        for right in range(left, len(values) + 1):
            assert segment.query(left, right) == fenwick.range_sum(left, right) == sum(values[left:right])
            assert ordered.query(left, right) == "".join(str(x) + ";" for x in values[left:right])
            if left < right:
                assert sparse.query(left, right) == min(values[left:right])
    lazy = spaces["lazy"]["LazySum"](values)
    range_fenwick = spaces["rangeFenwick"]["RangeAddSum"](values)
    additive = values[:]
    point_values = values[:]
    for _ in range(50):
        left, right = sorted([rng.randrange(len(values) + 1), rng.randrange(len(values) + 1)])
        amount = rng.randrange(-8, 9)
        operation = rng.choice(["add", "set", "query"])
        if operation == "query":
            assert lazy.query(left, right) == sum(values[left:right])
        else:
            getattr(lazy, operation)(left, right, amount)
            values[left:right] = [amount if operation == "set" else item + amount for item in values[left:right]]
        range_fenwick.add(left, right, amount)
        additive[left:right] = [item + amount for item in additive[left:right]]
        query_left, query_right = sorted([rng.randrange(len(values) + 1), rng.randrange(len(values) + 1)])
        assert range_fenwick.range_sum(query_left, query_right) == sum(additive[query_left:query_right])
        assert [lazy.query(i, i + 1) for i in range(len(values))] == values
        assert lazy.query(0, len(values)) == sum(values)
        if values:
            index = rng.randrange(len(values))
            replacement = rng.randrange(-9, 10)
            segment.set(index, replacement)
            fenwick.set(index, replacement)
            point_values[index] = replacement
            assert segment.query(0, len(values)) == fenwick.range_sum(0, len(values)) == sum(point_values)
        counts["dynamic_operations"] += 1
    counts["tree_arrays"] += 1

for length in range(7):
    for items in itertools.product([-2, 0, 3], repeat=length):
        values = list(items)
        for width in range(1, 9):
            expected = []
            for left in range(length - width + 1):
                maximum = max(values[left:left + width])
                index = max(i for i in range(left, left + width) if values[i] == maximum)
                expected.append((maximum, index))
            assert spaces["movingMaximum"]["moving_maxima"](values, width) == expected
        for target in [1, 3, 5, 9]:
            candidates = [(right - left, (left, right)) for left in range(length)
                          for right in range(left + 1, length + 1)
                          if sum(values[left:right]) >= target]
            expected = min(candidates, default=None)
            assert spaces["signedShortest"]["shortest_at_least"](values, target) == expected
        counts["deque_arrays"] += 1

for _ in range(800):
    values = [rng.randrange(-3, 4) for _ in range(rng.randrange(10))]
    weights = [rng.randrange(-6, 8) for _ in values]
    best = 0
    for bits in itertools.product([False, True], repeat=len(values)):
        indices = [index for index, selected in enumerate(bits) if selected]
        if all(values[a] < values[b] for a, b in zip(indices, indices[1:])):
            best = max(best, sum(weights[index] for index in indices))
    score, witness = spaces["weightedIncreasing"]["weighted_increasing"](values, weights)
    assert score == best == sum(weights[index] for index in witness)
    assert all(a < b and values[a] < values[b] for a, b in zip(witness, witness[1:]))
    counts["weighted_cases"] += 1

for _ in range(400):
    frequencies = [rng.randrange(5) for _ in range(rng.randrange(1, 15))]
    tree = spaces["frequencySelection"]["Frequencies"](frequencies)
    for _ in range(10):
        expanded = [index for index, count in enumerate(frequencies) for _ in range(count)]
        assert [tree.kth(rank) for rank in range(1, len(expanded) + 1)] == expanded
        index = rng.randrange(len(frequencies))
        replacement = rng.randrange(5)
        tree.set(index, replacement)
        frequencies[index] = replacement
    counts["frequency_cases"] += 1

for _ in range(300):
    length = rng.randrange(1, 20)
    children = [[] for _ in range(length)]
    for child in range(1, length):
        children[rng.randrange(child)].append(child)
    enter, leave, order = spaces["subtree"]["flatten_tree"](children, 0)
    for root in range(length):
        descendants = {root}
        frontier = [root]
        while frontier:
            vertex = frontier.pop()
            descendants.update(children[vertex])
            frontier.extend(children[vertex])
        assert set(order[enter[root]:leave[root]]) == descendants
        counts["subtrees"] += 1

merge = spaces["richerSummary"]["merge"]
for _ in range(300):
    values = [rng.randrange(-5, 6) for _ in range(rng.randrange(10))]
    tree = spaces["richerSummary"]["SegmentTree"]([(x, x, x, x) for x in values], merge, None)
    for left in range(len(values) + 1):
        for right in range(left, len(values) + 1):
            answer = tree.query(left, right)
            if left == right:
                assert answer is None
            else:
                expected = max(sum(values[a:b]) for a in range(left, right) for b in range(a + 1, right + 1))
                assert answer[3] == expected
print(json.dumps(counts, sort_keys=True))
