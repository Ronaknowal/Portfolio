"""Independent small-instance oracles for the exact displayed Python programs."""
import contextlib
import functools
import io
import itertools
import json
import random
import sys
import types
from fractions import Fraction

examples = json.load(open(sys.argv[1], encoding="utf-8"))
programs = {}
for name, example in examples.items():
    module = types.ModuleType("greedy_example_" + name)
    sys.modules[module.__name__] = module
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(example["code"], name, "exec"), module.__dict__)
    programs[name] = module.__dict__

rng = random.Random(924)
counts = {name: 0 for name in ["intervals", "deadlines", "fractional", "huffman", "reachability", "stabbing", "matching", "coin_comparisons"]}

def subsets(items):
    for length in range(len(items) + 1):
        yield from itertools.combinations(items, length)

def interval_feasible(records):
    ordered = sorted(records, key=lambda row: row[1])
    return all(first[2] <= second[1] for first, second in zip(ordered, ordered[1:]))

for _ in range(1200):
    requests = []
    for index in range(rng.randrange(7)):
        start = rng.randrange(-4, 7)
        requests.append((chr(65 + index), start, start + rng.randrange(1, 5)))
    original = requests.copy()
    selected = programs["intervals"]["select_appointments"](requests)
    selected_rows = [row for row in requests if row[0] in selected]
    assert len(set(selected)) == len(selected)
    assert interval_feasible(selected_rows)
    assert len(selected) == max(len(choice) for choice in subsets(requests) if interval_feasible(choice))
    rooms = programs["rooms"]["minimum_rooms"](requests)
    assert sorted(itertools.chain.from_iterable(rooms)) == sorted(row[0] for row in requests)
    for room in rooms:
        assert interval_feasible([row for row in requests if row[0] in room])
    # Exact endpoint sweep, with ends before starts at equal times.
    events = sorted([(start, 1) for _, start, _ in requests] + [(finish, -1) for _, _, finish in requests])
    depth = active = 0
    for _, delta in events:
        active += delta
        depth = max(depth, active)
    assert len(rooms) == depth
    assert requests == original
    counts["intervals"] += 1

for _ in range(450):
    jobs = [(chr(65 + index), rng.randrange(1, 6), rng.randrange(-2, 18)) for index in range(rng.randrange(6))]
    schedule, maximum = programs["deadlines"]["deadline_schedule"](jobs)
    if jobs:
        candidates = []
        for order in itertools.permutations(jobs):
            time = 0
            latenesses = []
            for _, duration, deadline in order:
                time += duration
                latenesses.append(time - deadline)
            candidates.append(max(latenesses))
        assert maximum == min(candidates)
        assert all(first[2] == second[1] for first, second in zip(schedule, schedule[1:]))
    else:
        assert (schedule, maximum) == ([], None)
    counts["deadlines"] += 1

def fractional_vertices(items, capacity):
    best = Fraction(0)
    for whole in subsets(items):
        weight = sum(item[1] for item in whole)
        value = Fraction(sum(item[2] for item in whole))
        if weight > capacity:
            continue
        best = max(best, value)
        for item in items:
            if item in whole:
                continue
            amount = min(capacity - weight, item[1])
            best = max(best, value + Fraction(amount * item[2], item[1]))
    return best

for _ in range(1000):
    items = [(chr(65 + index), rng.randrange(1, 11), rng.randrange(31)) for index in range(rng.randrange(6))]
    capacity = rng.randrange(30)
    selected, total, remaining = programs["fractional"]["density_allocation"](items, capacity)
    lookup = {item[0]: item for item in items}
    assert total == fractional_vertices(items, capacity)
    assert total == sum(lookup[identity][2] * fraction for identity, fraction in selected)
    assert sum(lookup[identity][1] * fraction for identity, fraction in selected) + remaining == capacity
    assert all(0 < fraction <= 1 for _, fraction in selected)
    whole, whole_value, unused = programs["fractional"]["density_allocation"](items, capacity, False)
    best_whole = max(sum(item[2] for item in choice) for choice in subsets(items) if sum(item[1] for item in choice) <= capacity)
    assert whole_value <= best_whole
    assert all(fraction == 1 for _, fraction in whole)
    counts["fractional"] += 1

@functools.lru_cache(None)
def best_merge(weights):
    if len(weights) < 2:
        return 0
    best = float("inf")
    for first, second in itertools.combinations(range(len(weights)), 2):
        joined = weights[first] + weights[second]
        rest = [weight for index, weight in enumerate(weights) if index not in (first, second)]
        best = min(best, joined + best_merge(tuple(sorted(rest + [joined]))))
    return best

for _ in range(350):
    weights = [rng.randrange(1, 10) for _ in range(rng.randrange(7))]
    frequencies = {chr(65 + index): value for index, value in enumerate(weights)}
    root, codes, merges = programs["huffman"]["huffman"](frequencies)
    weighted_bits = sum(frequencies[symbol] * len(code) for symbol, code in codes.items())
    assert weighted_bits == sum(merges) == best_merge(tuple(sorted(weights)))
    assert all(first == second or not second.startswith(first) for first in codes.values() for second in codes.values())
    assert len(set(codes.values())) == len(codes)
    if frequencies:
        message = "".join(rng.choice(list(frequencies)) for _ in range(12))
        bits = "".join(codes[character] for character in message)
        assert programs["huffman"]["decode"](bits, root, len(message)) == message
    else:
        assert programs["huffman"]["decode"]("", root, 0) == ""
    counts["huffman"] += 1

for length in range(7):
    for values in itertools.product(range(4), repeat=length):
        visited = {0} if values else set()
        frontier = list(visited)
        while frontier:
            source = frontier.pop()
            for target in range(source + 1, min(len(values), source + values[source] + 1)):
                if target not in visited:
                    visited.add(target)
                    frontier.append(target)
        assert programs["reachability"]["can_reach_last"](values) == (len(values) - 1 in visited)
        counts["reachability"] += 1

for _ in range(600):
    intervals = []
    for index in range(rng.randrange(7)):
        start = rng.randrange(-3, 6)
        intervals.append((start, start + rng.randrange(5)))
    points = programs["stabbing"]["covering_points"](intervals)
    assert all(any(start <= point <= finish for point in points) for start, finish in intervals)
    candidates = sorted({finish for _, finish in intervals})
    # A covering point can shift right to the minimum endpoint of what it covers.
    best = min(len(choice) for choice in subsets(candidates) if all(any(start <= point <= finish for point in choice) for start, finish in intervals))
    assert len(points) == best
    counts["stabbing"] += 1

def matching_oracle(requirements, supplies):
    if not requirements:
        return 0
    best = matching_oracle(requirements[1:], supplies)
    for index, amount in enumerate(supplies):
        if amount >= requirements[0]:
            rest = supplies[:index] + supplies[index + 1:]
            best = max(best, 1 + matching_oracle(requirements[1:], rest))
    return best

for _ in range(600):
    requirements = [rng.randrange(6) for _ in range(rng.randrange(6))]
    supplies = [rng.randrange(6) for _ in range(rng.randrange(6))]
    selected = programs["matching"]["match_thresholds"](requirements, supplies)
    assert len(selected) == matching_oracle(requirements, supplies)
    assert len({pair[0] for pair in selected}) == len(selected)
    assert len({pair[1] for pair in selected}) == len(selected)
    assert all(supplies[supply] >= requirements[person] for person, supply in selected)
    counts["matching"] += 1

suboptimal_coins = 0
for denominations in subsets([1, 2, 3, 4, 5]):
    for amount in range(16):
        selected, remainder = programs["coins"]["largest_first"](denominations, amount)
        assert sum(coin * count for coin, count in selected) + remainder == amount
        solutions = [sum(multiplicities) for multiplicities in itertools.product(range(amount + 1), repeat=len(denominations)) if sum(coin * count for coin, count in zip(denominations, multiplicities)) == amount]
        optimum = min(solutions) if solutions else None
        greedy_count = sum(count for _, count in selected) if remainder == 0 else None
        if optimum is not None and (greedy_count is None or greedy_count > optimum):
            suboptimal_coins += 1
        counts["coin_comparisons"] += 1
assert suboptimal_coins > 0

# Exact exercise values, independently computed from completed job times.
assert max(10 - 100, 11 - 2) == 9
assert max(2 - 2, 12 - 100) == 0
exercise_tree, exercise_codes, exercise_merges = programs["huffman"]["huffman"]({"A": 1, "B": 2, "C": 4, "D": 8})
assert exercise_merges == [3, 7, 15]
assert [len(exercise_codes[symbol]) for symbol in "ABCD"] == [3, 3, 2, 1]
assert sum(frequency * len(exercise_codes[symbol]) for symbol, frequency in zip("ABCD", [1, 2, 4, 8])) == 25

invalid_calls = [
    (programs["coins"]["largest_first"], ([0], 6)),
    (programs["intervals"]["select_appointments"], ([("A", 1, 1)],)),
    (programs["rooms"]["minimum_rooms"], ([("A", 2, 1)],)),
    (programs["deadlines"]["deadline_schedule"], ([("A", 0, 1)],)),
    (programs["fractional"]["density_allocation"], ([("A", 0, 1)], 5)),
    (programs["huffman"]["huffman"], ({"A": 0},)),
    (programs["huffman"]["huffman"], ({"AB": 2},)),
    (programs["reachability"]["can_reach_last"], ([-1],)),
    (programs["stabbing"]["covering_points"], ([(3, 2)],)),
    (programs["matching"]["match_thresholds"], ([-1], [1])),
]
for function, arguments in invalid_calls:
    try:
        function(*arguments)
        raise AssertionError("invalid input accepted")
    except ValueError:
        pass
print("Native independent oracle cases:", counts, "coin counterexamples:", suboptimal_coins)
