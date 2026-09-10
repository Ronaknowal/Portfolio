import contextlib
import io
import itertools
import json
import math
from pathlib import Path
import runpy
import subprocess
import sys

directory = Path(sys.argv[1])
cases = json.loads((directory / "model-cases.json").read_text())


def load_example(key):
    with contextlib.redirect_stdout(io.StringIO()):
        return runpy.run_path(str(directory / f"{key}.py"))


first_index = load_example("search")["first_index"]
compact = load_example("compaction")["remove_stably"]
zero_fill = load_example("zeroFill")["move_zeroes"]
partition = load_example("partition")["partition_three"]
gcd = load_example("gcd")["euclid"]
rectangle = load_example("lexicographic")["visit_rectangle"]
powers = load_example("powers")
modular_power = load_example("modularPower")["modular_power"]
certificate = load_example("certificate")["is_sorted_permutation"]
square_root = load_example("squareRoot")["integer_square_root"]
counts = {}

for case in cases["searches"]:
    values, target, skip = case["values"], case["target"], case["skip"]
    expected = values.index(target) if target in values else -1
    before = list(values)
    assert first_index(values, target) == expected
    assert values == before
    states = case["states"]
    for state in states:
        index = state["index"]
        assert state["prefixValid"] == (0 <= index <= len(values) and target not in values[:index])
        assert state["variant"] == len(values) - index
    if not skip:
        assert states[-1]["result"] == expected
        assert all(state["prefixValid"] for state in states)
    for previous, current in zip(states, states[1:]):
        index = previous["index"]
        if current["finished"]:
            assert current["index"] == index
            assert current["result"] == (index if index < len(values) else -1)
        else:
            assert values[index] != target
            assert current["index"] == min(index + (2 if skip else 1), len(values))
    # Independently enumerate logical obligations for each admitted boundary.
    for claim, reported in case["obligations"].items():
        def predicate(index):
            if not 0 <= index <= len(values):
                return False
            region = range(index) if claim == "prefix" else range(len(values)) if claim == "whole" else []
            return all(values[position] != target for position in region)
        expected_failures = {"initialization": not predicate(0), "preservation": False, "matchExit": False, "absentExit": False}
        for index in range(len(values) + 1):
            if not predicate(index):
                continue
            if index == len(values):
                expected_failures["absentExit"] |= expected != -1
            elif values[index] == target:
                expected_failures["matchExit"] |= index != expected
            else:
                after = min(index + (2 if skip else 1), len(values))
                expected_failures["preservation"] |= not predicate(after)
        for key, failed in expected_failures.items():
            assert (reported[key] is not None) == failed, (case, claim, key)
counts["searchModelTraces"] = len(cases["searches"])

for case in cases["compactions"]:
    values, removed = case["values"], case["removed"]
    working = list(values)
    length = compact(working, removed)
    expected = [value for value in values if value != removed]
    assert length == len(expected) and working[:length] == expected
    assert len(working) == len(values)
    for state in case["trace"]["states"]:
        read, write = state["read"], state["write"]
        expected_origins = [position for position in range(read) if values[position] != removed]
        actual = state["working"]
        assert [item["origin"] for item in actual[:write]] == expected_origins
        assert [item["value"] for item in actual[:write]] == [values[position] for position in expected_origins]
        assert actual[read:] == [{"origin": position, "value": values[position]} for position in range(read, len(values))]
        assert state["boundsValid"] and state["prefixValid"] and state["unreadValid"]
        assert state["variant"] == len(values) - read
    assert [item["value"] for item in case["trace"]["states"][-1]["working"]] == working
counts["compactionModelTraces"] = len(cases["compactions"])

for case in cases["partitions"]:
    original = case["values"]
    working = list(original)
    partition(working)
    assert working == sorted(original)
    for state in case["states"]:
        items = state["working"]
        values = [item["value"] for item in items]
        low, middle, high = state["low"], state["middle"], state["high"]
        expected_checks = {
            "bounds": 0 <= low <= middle <= high <= len(values),
            "zero": all(value == 0 for value in values[:low]),
            "one": all(value == 1 for value in values[low:middle]),
            "two": all(value == 2 for value in values[high:]),
            "occurrences": sorted(item["origin"] for item in items) == list(range(len(original))),
        }
        assert state["checks"] == expected_checks
        assert sorted(values) == sorted(original)
        assert all(item["value"] == original[item["origin"]] for item in items)
        assert state["variant"] == high - middle
        if not case["skipIncoming"]:
            assert all(expected_checks.values())
    if not case["skipIncoming"]:
        assert len(case["states"]) == len(original) + 1
        assert [item["value"] for item in case["states"][-1]["working"]] == working
        assert [state["variant"] for state in case["states"]] == list(range(len(original), -1, -1))
counts["partitionModelTraces"] = len(cases["partitions"])

for case in cases["euclids"]:
    expected = math.gcd(case["first"], case["second"])
    expected_divisors = [value for value in range(1, expected + 1) if expected % value == 0]
    for state in case["states"]:
        a, b = state["a"], state["b"]
        assert state["commonDivisors"] == expected_divisors
        assert math.gcd(a, b) == expected
        if b:
            assert divmod(a, b) == (state["quotient"], state["remainder"])
            assert 0 <= state["remainder"] < b
        else:
            assert state["terminal"] and a == expected
    for old, new in zip(case["states"], case["states"][1:]):
        assert (new["a"], new["b"]) == (old["b"], old["remainder"])
counts["euclidModelTraces"] = len(cases["euclids"])

counts["nativeGcdPairs"] = 0
for a in range(97):
    for b in range(97):
        assert gcd(a, b) == math.gcd(a, b)
        counts["nativeGcdPairs"] += 1
counts["nativePowerPairs"] = 0
for base in range(-4, 5):
    for exponent in range(33):
        expected = pow(base, exponent)
        assert powers["recursive_power"](base, exponent) == expected
        assert powers["iterative_power"](base, exponent) == expected
        counts["nativePowerPairs"] += 1
counts["nativeModularCases"] = 0
for base, exponent, modulus in itertools.product(range(-5, 6), range(12), range(1, 13)):
    assert modular_power(base, exponent, modulus) == pow(base, exponent, modulus)
    counts["nativeModularCases"] += 1
counts["nativeRectangles"] = 0
for rows, columns in itertools.product(range(9), repeat=2):
    visited, measures = rectangle(rows, columns)
    assert visited == list(itertools.product(range(rows), range(columns)))
    assert len(measures) == rows * (columns + 1) + 1
    assert all(after < before for before, after in zip(measures, measures[1:]))
    assert all(all(value >= 0 for value in pair) for pair in measures)
    counts["nativeRectangles"] += 1

arrays = [list(values) for length in range(5) for values in itertools.product((-1, 0, 1), repeat=length)]
counts["nativeCertificates"] = 0
for original, candidate in itertools.product(arrays, repeat=2):
    assert certificate(original, candidate) == (candidate == sorted(original))
    counts["nativeCertificates"] += 1
for original in arrays:
    values = list(original)
    expected = [value for value in original if value != 0] + [0] * original.count(0)
    zero_fill(values)
    assert values == expected
counts["nativeZeroFillArrays"] = len(arrays)

numbers = list(range(10001))
for boundary in [10**6, 10**10, 2**53 + 1]:
    numbers.extend([boundary * boundary - 1, boundary * boundary, boundary * boundary + 1])
for number in numbers:
    actual = square_root(number)
    assert actual == math.isqrt(number)
    assert actual * actual <= number < (actual + 1) * (actual + 1)
counts["nativeSquareRoots"] = len(numbers)

invalid_calls = [
    lambda: gcd(-1, 3), lambda: gcd(1.2, 3),
    lambda: powers["recursive_power"](2, -1), lambda: powers["iterative_power"](2, 1.5),
    lambda: modular_power(2, 4, 0), lambda: modular_power(2, -1, 3),
    lambda: rectangle(-1, 3), lambda: square_root(-1), lambda: square_root(0.5),
]
for call in invalid_calls:
    try:
        call()
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid precondition was not rejected")
invalid_list = [2, 3, 0]
try:
    partition(invalid_list)
except ValueError:
    assert invalid_list == [2, 3, 0]
else:
    raise AssertionError("Invalid partition was accepted")
optimized = subprocess.run([sys.executable, "-I", "-O", str(directory / "assertions.py")], capture_output=True, text=True, check=True)
assert optimized.stdout.strip() == "checked: 1"
counts["nativeInvalidGroups"] = len(invalid_calls) + 1
counts["optimizedAssertionsOmitted"] = True
print(json.dumps({"python": sys.version.split()[0], **counts}))
