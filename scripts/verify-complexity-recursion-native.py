import itertools
import json
import math
from pathlib import Path
import sys

directory = Path(sys.argv[1])
examples = json.loads((directory / "examples.json").read_text())
modules = {}
for name, example in examples.items():
    namespace = {"__name__": "verification"}
    exec(compile(example["code"], name, "exec"), namespace)
    modules[name] = namespace

counts = {"arrays": 0, "thresholds": 0, "powers": 0, "searches": 0,
          "recurrences": 0, "append_sequences": 0, "native_frame_traces": 0}
for length in range(7):
    for items in itertools.product((-2, 0, 3), repeat=length):
        values = list(items)
        expected = sum(values)
        assert modules["suffixSum"]["suffix_sum"](values) == expected
        total, calls, peak = modules["balancedSum"]["balanced_sum"](values)
        assert total == expected
        assert calls == max(1, 2 * length - 1)
        assert peak == (math.ceil(math.log2(length)) + 1 if length else 1)
        reversed_values = values.copy()
        swaps = modules["reverse"]["reverse_in_place"](reversed_values)
        assert reversed_values == list(reversed(values)) and swaps == length // 2
        for threshold in (-3, -2, 0, 1, 3, 4):
            assert modules["thresholdCount"]["count_below"](values, threshold) == len([x for x in values if x < threshold])
            counts["thresholds"] += 1
        for target in (-2, 0, 3, 9):
            matching = [index for index, value in enumerate(values) if value == target]
            index, comparisons = modules["searchCases"]["find_counted"](values, target)
            assert index == (matching[0] if matching else None)
            assert comparisons == (index + 1 if index is not None else length)
            counts["searches"] += 1
        assert values == list(items)
        counts["arrays"] += 1

for base in range(-6, 7):
    for exponent in range(65):
        result, calls, multiplications = modules["power"]["power_counted"](base, exponent)
        assert result == pow(base, exponent)
        assert calls == exponent.bit_length() + 1
        assert multiplications == exponent.bit_length() + exponent.bit_count()
        counts["powers"] += 1

for n in range(129):
    expected = (n * n, math.comb(n, 2), n * ((n - 1).bit_length() if n else 0))
    assert modules["loopCounts"]["work_counts"](n) == expected

for exponent in range(9):
    n = 2 ** exponent
    expected = {"chain_constant": n, "chain_linear": n * (n + 1) // 2,
                "half_constant": exponent + 1, "half_linear": 2 * n - 1,
                "two_half_linear": n * (exponent + 1)}
    for kind, answer in expected.items():
        assert modules["recurrences"]["recurrence_cost"](kind, n) == answer
        counts["recurrences"] += 1

for count in range(513):
    costs, copied, capacity = modules["appendBudget"]["doubling_budget"](count)
    expected_copies = sum(2 ** power for power in range(count.bit_length()) if 2 ** power < count)
    assert copied == expected_copies
    assert sum(costs) == count + copied
    assert count <= capacity and (count == 0 or sum(costs) < 3 * count)
    counts["append_sequences"] += 1

fibonacci = [0, 1]
for _ in range(21):
    fibonacci.append(sum(fibonacci[-2:]))
for n in range(21):
    first, second, third, calls, computed = modules["fibonacci"]["fibonacci_comparison"](n)
    assert first == second == third == fibonacci[n]
    assert calls == 2 * fibonacci[n + 1] - 1
    assert computed == max(0, n - 1)

traces = json.loads((directory / "model-traces.json").read_text())
for case in traces:
    native = modules["suffixSum"]["suffix_trace"](case["values"])
    assert native == case["states"]
    counts["native_frame_traces"] += 1

for name, function, bad_values in [
    ("loopCounts", "work_counts", [-1, 1.5, True]),
    ("appendBudget", "doubling_budget", [-1, 1.5, True]),
    ("fibonacci", "fibonacci_comparison", [-1, 21, True]),
]:
    for value in bad_values:
        try:
            modules[name][function](value)
        except ValueError:
            pass
        else:
            raise AssertionError((name, value))

(directory / "native-results.json").write_text(json.dumps({"python": sys.version, "counts": counts, "invalid_boundaries": "passed"}, indent=2))
print(json.dumps(counts))
