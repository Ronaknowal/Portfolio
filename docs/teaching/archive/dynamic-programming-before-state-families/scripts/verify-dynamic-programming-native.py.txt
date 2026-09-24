"""Independent exhaustive oracles; no recurrence-shaped oracle for the same task."""
import contextlib
import io
import itertools
import json
import random
import runpy
from collections import deque
from pathlib import Path

directory = Path("scratch/dynamic-programming-verification")
names = ["memo", "rewardWitness", "state", "weightedIntervals", "grid", "gridCount", "lcs", "edit", "knapsack", "compression", "subsetSum", "coins", "masks", "route", "lis", "segment"]
functions = {}
for name in names:
    with contextlib.redirect_stdout(io.StringIO()):
        functions.update(runpy.run_path(str(directory / f"{name}.py")))

def arrays(choices, maximum):
    for length in range(maximum + 1):
        yield from itertools.product(choices, repeat=length)

def subsets(values):
    for mask in range(1 << len(values)):
        yield [index for index in range(len(values)) if mask & (1 << index)]

def reward_oracle(values, blocked=False):
    return max((sum(values[index] for index in chosen) for chosen in subsets(values)
                if not any(right - left == 1 for left, right in zip(chosen, chosen[1:]))
                and not (blocked and 0 in chosen)), default=0)

reward_cases = 0
for values in arrays([-2, 0, 3], 7):
    expected = reward_oracle(values)
    assert functions["best_reward"](values) == expected
    score, selected = functions["reward_plan"](values)
    assert score == expected == sum(values[index] for index in selected)
    assert all(right - left > 1 for left, right in zip(selected, selected[1:]))
    for blocked in [False, True]:
        assert functions["constrained_reward"](values, blocked) == reward_oracle(values, blocked)
    reward_cases += 1

def all_grid_paths(grid, end=None):
    rows, columns = len(grid), len(grid[0])
    end_row, end_column = end if end is not None else (rows - 1, columns - 1)
    paths = []
    # Every right/down path is uniquely determined by its down-move positions.
    for downs in itertools.combinations(range(end_row + end_column), end_row):
        row = column = 0
        path = [(0, 0)]
        for step in range(end_row + end_column):
            if step in downs:
                row += 1
            else:
                column += 1
            path.append((row, column))
        if all(grid[row][column] is not None for row, column in path):
            paths.append((sum(grid[row][column] for row, column in path), path))
    return paths

generator = random.Random(98431)
weighted_cases = 0
for _ in range(300):
    jobs = []
    for index in range(generator.randint(0, 8)):
        start = generator.randint(0, 8)
        jobs.append((start, start + generator.randint(1, 4), generator.randint(-5, 12)))
    choices = []
    for chosen in subsets(jobs):
        if all(jobs[left][1] <= jobs[right][0] or jobs[right][1] <= jobs[left][0]
               for left, right in itertools.combinations(chosen, 2)):
            choices.append((sum(jobs[index][2] for index in chosen), chosen))
    expected = max(score for score, chosen in choices)
    score, witness = functions["weighted_schedule"](jobs)
    assert score == expected and (score, sorted(witness)) in choices
    assert all(jobs[left][1] <= jobs[right][0] for left, right in zip(witness, witness[1:]))
    weighted_cases += 1
grid_cases = 0
for _ in range(250):
    rows, columns = generator.randint(1, 4), generator.randint(1, 4)
    grid = [[None if generator.randrange(5) == 0 else generator.randint(-5, 8) for _ in range(columns)] for _ in range(rows)]
    paths = all_grid_paths(grid)
    score, path = functions["grid_plan"](grid)
    assert score == (min(item[0] for item in paths) if paths else None)
    assert not paths or (score, path) in paths
    assert functions["count_paths"]([[cell is None for cell in row] for row in grid]) == len(paths)
    grid_cases += 1

def subsequences(text):
    return {"".join(text[index] for index in chosen) for chosen in subsets(text)}

def edit_oracle(first, second):
    # Breadth-first legal edits on a bounded binary alphabet, not a DP recurrence.
    frontier = deque([(first, 0)])
    seen = {first}
    maximum_length = max(len(first), len(second))
    while frontier:
        text, cost = frontier.popleft()
        if text == second:
            return cost
        candidates = {text[:index] + text[index + 1:] for index in range(len(text))}
        candidates.update(text[:index] + char + text[index + 1:] for index in range(len(text)) for char in "AB")
        if len(text) < maximum_length:
            candidates.update(text[:index] + char + text[index:] for index in range(len(text) + 1) for char in "AB")
        for candidate in candidates - seen:
            seen.add(candidate)
            frontier.append((candidate, cost + 1))

sequence_cases = edit_cases = 0
strings = ["".join(values) for values in arrays("AB", 4)]
for first in strings:
    for second in strings:
        common = subsequences(first) & subsequences(second)
        expected = max(map(len, common))
        score, witness = functions["longest_common_subsequence"](first, second)
        assert score == expected and witness in common and len(witness) == expected
        sequence_cases += 1
        if len(first) <= 3 and len(second) <= 3:
            assert functions["edit_distance"](first, second) == edit_oracle(first, second)
            edit_cases += 1

def capacity_oracle(items, capacity, reusable=False):
    limits = [range(capacity // weight + 1) if reusable else range(2) for weight, value in items]
    return max(sum(amount * item[1] for amount, item in zip(counts, items))
               for counts in itertools.product(*limits)
               if sum(amount * item[0] for amount, item in zip(counts, items)) <= capacity)

capacity_cases = 0
for _ in range(450):
    items = [(generator.randint(1, 5), generator.randint(-3, 8)) for _ in range(generator.randint(0, 5))]
    capacity = generator.randint(0, 10)
    expected = capacity_oracle(items, capacity)
    score, witness = functions["knapsack_plan"](items, capacity)
    assert score == expected and len(witness) == len(set(witness))
    assert sum(items[index][0] for index in witness) <= capacity
    assert sum(items[index][1] for index in witness) == expected
    assert functions["capacity_value"](items, capacity) == expected
    assert functions["capacity_value"](items, capacity, True) == capacity_oracle(items, capacity, True)
    capacity_cases += 1

subset_cases = 0
for values in arrays([0, 2, 3], 5):
    sums = {sum(values[index] for index in chosen) for chosen in subsets(values)}
    for target in range(12):
        assert functions["subset_sum"](values, target) == (target in sums)
        subset_cases += 1

coin_cases = 0
for kinds in subsets([1, 2, 3, 4]):
    coins = [index + 1 for index in kinds]
    for amount in range(9):
        multisets = [counts for counts in itertools.product(*(range(amount // coin + 1) for coin in coins)) if sum(count * coin for count, coin in zip(counts, coins)) == amount]
        minimum = min(map(sum, multisets)) if multisets else None
        ordered = 0
        for length in range(amount + 1):
            ordered += sum(sum(sequence) == amount for sequence in itertools.product(coins, repeat=length))
        assert functions["coin_answers"](coins, amount) == (minimum, ordered, len(multisets))
        coin_cases += 1

lis_cases = 0
for values in arrays([0, 1, 2], 6):
    expected = max(len(chosen) for chosen in subsets(values) if all(values[left] < values[right] for left, right in zip(chosen, chosen[1:])))
    assert functions["lis_quadratic"](values) == functions["lis_length"](values) == expected
    lis_cases += 1

def route_states(costs):
    count = len(costs)
    results = {}
    for length in range(count):
        for suffix in itertools.permutations(range(1, count), length):
            path = (0,) + suffix
            if any(costs[left][right] is None for left, right in zip(path, path[1:])):
                continue
            score = sum(costs[left][right] for left, right in zip(path, path[1:]))
            key = (sum(1 << index for index in path), path[-1])
            results[key] = min(results.get(key, score), score)
    return results

fixtures = json.loads((directory / "models.json").read_text())
for case in fixtures["rewards"]:
    assert case["result"] == reward_oracle(case["values"])
    for index, value in enumerate(case["cache"]):
        assert value == reward_oracle(case["values"][index:])
for case in fixtures["grids"]:
    grid = [row[:] for row in case["grid"]]
    for blocked in case["blocked"]:
        row, column = map(int, blocked.split(','))
        grid[row][column] = None
    assert list(functions["grid_plan"](grid)[0:1]) == [case["result"]]
    for row in range(len(grid)):
        for column in range(len(grid[0])):
            paths = all_grid_paths(grid, (row, column))
            assert case["costs"][row][column] == (min(item[0] for item in paths) if paths else None)
            assert case["ways"][row][column] == len(paths)
    assert [list(cell) for cell in functions["grid_plan"](grid)[1]] == case["path"]
for case in fixtures["sequences"]:
    assert list(functions["longest_common_subsequence"](case["first"], case["second"])) == [case["result"], case["witness"]]
    for row, values in enumerate(case["lengths"]):
        for column, value in enumerate(values):
            assert value == max(map(len, subsequences(case["first"][:row]) & subsequences(case["second"][:column])))
for case in fixtures["capacities"]:
    items = [(item["weight"], item["value"]) for item in case["items"]]
    assert case["result"] == capacity_oracle(items, case["capacity"], case["direction"] == "ascending")
for case in fixtures["routes"]:
    states = route_states(case["costs"])
    for mask, row in enumerate(case["best"]):
        for endpoint, value in enumerate(row):
            assert value == states.get((mask, endpoint))
    score, path = functions["visit_once"](case["costs"])
    assert score == case["result"] and path == case["path"]

mask_cases = 0
for width in range(1, 9):
    full = (1 << width) - 1
    for mask in range(full + 1):
        members = {index for index in range(width) if mask & (1 << index)}
        for index in range(width):
            added = mask | (1 << index)
            removed = mask & ~(1 << index)
            assert {bit for bit in range(width) if added & (1 << bit)} == members | {index}
            assert {bit for bit in range(width) if removed & (1 << bit)} == members - {index}
        assert {bit for bit in range(width) if (full & ~mask) & (1 << bit)} == set(range(width)) - members
        submask = mask
        visited = []
        while True:
            visited.append(submask)
            if not submask:
                break
            submask = (submask - 1) & mask
        assert len(set(visited)) == 2 ** len(members)
        assert all(value & ~mask == 0 for value in visited)
        mask_cases += 1

segment_cases = 0
for values in arrays("AB", 6):
    text = "".join(values)
    for words in [["A", "BB"], ["AB", "BA"], ["A", "B", "AB"], []]:
        possibilities = []
        if not text:
            possibilities.append([])
        else:
            for cuts in subsets(range(len(text) - 1)):
                boundaries = [0] + [index + 1 for index in cuts] + [len(text)]
                pieces = [text[start:end] for start, end in zip(boundaries, boundaries[1:])]
                if all(piece in words for piece in pieces):
                    possibilities.append(pieces)
        result = functions["command_plan"](text, words)
        assert (result is None) == (not possibilities)
        assert result is None or result in possibilities
        segment_cases += 1

report = dict(reward_cases=reward_cases, weighted_cases=weighted_cases, grid_cases=grid_cases, sequence_cases=sequence_cases, edit_cases=edit_cases,
              capacity_cases=capacity_cases, subset_cases=subset_cases, coin_cases=coin_cases, lis_cases=lis_cases,
              mask_cases=mask_cases, segment_cases=segment_cases, model_cases={key: len(value) for key, value in fixtures.items()})
print(json.dumps(report))
