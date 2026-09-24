"""Independent brute-force/bisect/sorted oracles for ordered-pattern teaching."""
import contextlib
import io
import itertools
import json
from pathlib import Path
import random
import sys
from bisect import bisect_left, bisect_right

directory = Path(sys.argv[1])
programs = json.loads((directory / "programs.json").read_text(encoding="utf-8"))
models = json.loads((directory / "models.json").read_text(encoding="utf-8"))
functions = {}
for name, example in programs.items():
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(example["code"], namespace)
    functions[name] = namespace

def all_arrays(choices, largest):
    for size in range(largest + 1):
        yield from itertools.product(choices, repeat=size)

array_count = 0
for values_tuple in all_arrays([-2, -1, 0, 1, 2], 5):
    values = list(values_tuple)
    array_count += 1
    ordered = sorted(values)
    for name in ["insertion_sort", "merge_sort"]:
        assert functions["stableSorting"][name](values) == ordered
        tagged = [(value, index) for index, value in enumerate(values)]
        assert functions["stableSorting"][name](tagged, key=lambda row: row[0]) == sorted(tagged, key=lambda row: row[0])
    assert functions["partitionSort"]["quicksort_three_way"](values) == ordered
    assert functions["countingSort"]["counting_sort"](values) == ordered
    heap = values.copy()
    functions["heapSort"]["heapsort_in_place"](heap)
    assert heap == ordered
    unique = ordered.copy()
    count = functions["compactMerge"]["unique_prefix"](unique)
    assert unique[:count] == sorted(set(values))
    expected_triples = sorted({tuple(sorted(values[index] for index in selection)) for selection in itertools.combinations(range(len(values)), 3) if sum(values[index] for index in selection) == 0})
    assert functions["triples"]["zero_triples"](values) == expected_triples
    for target in range(-3, 4):
        assert functions["boundaries"]["lower_bound"](ordered, target) == bisect_left(ordered, target)
        assert functions["boundaries"]["upper_bound"](ordered, target) == bisect_right(ordered, target)
        expected_pairs = [(left, right) for left in range(len(values)) for right in range(left + 1, len(values)) if ordered[left] + ordered[right] == target]
        answer = functions["pairSum"]["sorted_pair"](ordered, target)
        assert (answer in expected_pairs) if expected_pairs else answer is None
        expected_count = sum(sum(values[left:right]) == target for right in range(1, len(values) + 1) for left in range(right))
        assert functions["signedCounts"]["count_sum"](values, target) == expected_count
    for width in range(1, len(values) + 1):
        candidates = [(sum(values[start:start + width]), start) for start in range(len(values) - width + 1)]
        expected = max(candidates, key=lambda item: (item[0], -item[1]))
        assert functions["fixedWindow"]["maximum_fixed_sum"](values, width) == expected
    outside = []
    for excluded in range(len(values)):
        product = 1
        for index, value in enumerate(values):
            if index != excluded:
                product *= value
        outside.append(product)
    assert functions["prefixSuffix"]["outside_products"](values) == outside

window_count = 0
for values_tuple in all_arrays([0, 1, 3], 6):
    values = list(values_tuple)
    for target in range(1, 9):
        window_count += 1
        candidates = [(left, right) for right in range(1, len(values) + 1) for left in range(right) if sum(values[left:right]) >= target]
        expected = min(candidates, key=lambda item: (item[1] - item[0], item[0]), default=None)
        assert functions["positiveWindow"]["shortest_at_least"](values, target) == expected

text_count = 0
for characters in all_arrays("abc", 7):
    text = "".join(characters)
    candidates = [(left, right) for left in range(len(text) + 1) for right in range(left, len(text) + 1) if len(set(text[left:right])) == right - left]
    expected = max(candidates, key=lambda item: (item[1] - item[0], -item[0]))
    assert functions["distinctWindow"]["longest_unique"](text) == expected
    text_count += 1

random_source = random.Random(5821)
for _ in range(400):
    left = sorted(random_source.choices(range(-5, 6), k=random_source.randrange(10)))
    right = sorted(random_source.choices(range(-5, 6), k=random_source.randrange(10)))
    destination = left + [None] * len(right)
    functions["compactMerge"]["merge_into_tail"](destination, len(left), right)
    assert destination == sorted(left + right)
    intervals = [sorted(random_source.sample(range(-6, 7), 2)) for _ in range(random_source.randrange(10))]
    merged = functions["intervals"]["merge_closed"](intervals)
    assert all(merged[index - 1][1] < merged[index][0] for index in range(1, len(merged)))
    for point in [index / 2 for index in range(-14, 15)]:
        assert any(start <= point <= end for start, end in intervals) == any(start <= point <= end for start, end in merged)
    records = [(random_source.randrange(-3, 8), random_source.randrange(-5, 6)) for _ in range(random_source.randrange(15))]
    times, prefix = functions["rangeReport"]["prepare_report"](records)
    for start, stop in [(-10, 10), (0, 0), (2, 6), (7, 10)]:
        selected = [amount for time, amount in records if start <= time < stop]
        assert functions["rangeReport"]["query_report"](times, prefix, start, stop) == (len(selected), sum(selected))

rate_count = 0
for jobs_tuple in all_arrays([1, 2, 4], 4):
    if not jobs_tuple:
        continue
    jobs = list(jobs_tuple)
    for budget in range(0, 12):
        expected = next((speed for speed in range(1, max(jobs) + 1) if sum((job + speed - 1) // speed for job in jobs) <= budget), None)
        assert functions["answerSearch"]["minimum_rate"](jobs, budget) == expected
        rate_count += 1

for trace in models["boundaries"]:
    reference = bisect_left if trace["side"] == "left" else bisect_right
    assert trace["boundary"] == reference(trace["values"], trace["target"])
for trace in models["pairs"]:
    values, target = trace["values"], trace["target"]
    solutions = [(left, right) for left in range(len(values)) for right in range(left + 1, len(values)) if values[left] + values[right] == target]
    assert tuple(trace["answer"]) in solutions if solutions else trace["answer"] is None
    for step in trace["steps"]:
        assert all(step["left"] <= left < right <= step["right"] for left, right in solutions)
for trace in models["windows"]:
    values, target = trace["values"], trace["target"]
    candidates = [(left, right) for right in range(1, len(values) + 1) for left in range(right) if sum(values[left:right]) >= target]
    expected = min(candidates, key=lambda item: (item[1] - item[0], item[0]), default=None)
    assert (tuple(trace["best"]) if trace["best"] else None) == expected
    for step in trace["steps"]:
        assert step["total"] == sum(values[step["left"]:step["right"]])
        if step["best"]:
            assert sum(values[step["best"][0]:step["best"][1]]) >= target
for trace in models["rates"]:
    flags = [option["feasible"] for option in trace["options"]]
    assert flags == sorted(flags)
    for option in trace["options"]:
        assert option["slots"] == [-(-job // option["speed"]) for job in [3, 6, 7]]
        assert option["feasible"] == (sum(option["slots"]) <= option["budget"])

for action in [lambda: functions["positiveWindow"]["shortest_at_least"]([1, -1, 5], 5), lambda: functions["fixedWindow"]["maximum_fixed_sum"]([], 1), lambda: functions["intervals"]["merge_closed"]([(2, 1)]), lambda: functions["answerSearch"]["minimum_rate"]([0], 3), lambda: functions["rangeReport"]["query_report"]([], [0], 3, 2)]:
    try:
        action()
        raise AssertionError("invalid input accepted")
    except ValueError:
        pass

print(json.dumps({"python": sys.version.split()[0], "exhaustiveArrays": array_count, "nonnegativeWindowCases": window_count, "uniqueTexts": text_count, "randomMergesAndIntervalUnions": 400, "rangeReports": 1600, "rateCases": rate_count, "crossRuntimeTraces": {name: len(values) for name, values in models.items()}}))
