"""Independent dict, sorting, exhaustive tuples and actual-storage cost oracles."""
import contextlib
import io
import itertools
import json
import random
import runpy
import sys
from fractions import Fraction
from pathlib import Path

directory = Path(sys.argv[1])
cases = json.loads((directory / "model-cases.json").read_text(encoding="utf-8"))


def load(name):
    with contextlib.redirect_stdout(io.StringIO()):
        return runpy.run_path(str(directory / (name + ".py")))


probe_module = load("probeMap")
ProbeMap = probe_module["ProbeMap"]
EMPTY, DELETED = probe_module["EMPTY"], probe_module["DELETED"]
ChainedMap = load("chainedMap")["ChainedMap"]
ListMap = load("listMap")["ListMap"]
counts = {"probePrograms": 0, "probeCommits": 0, "probeStates": 0}
for case in cases["probes"]:
    table, reference = ProbeMap(case["capacity"]), {}
    commits = [state for state in case["trace"]["states"] if state["phase"] == "commit"]
    operations = case["trace"]["operations"]
    for operation, state in zip(operations, commits, strict=True):
        kind = operation["kind"]
        if kind == "put":
            key, value = operation["key"], operation["value"]
            if key not in reference and len(reference) == len(table.slots):
                try:
                    table.put(key, value)
                    raise AssertionError("full insertion accepted")
                except OverflowError:
                    result = "full"
            else:
                result = key not in reference
                assert table.put(key, value) == result
                reference[key] = value
        elif kind == "get":
            key = operation["key"]
            expected = (key in reference, reference.get(key))
            assert table.find(key) == expected
            result = {"found": expected[0]}
            if expected[0]:
                result["value"] = expected[1]
        elif kind == "del":
            key = operation["key"]
            result = key in reference
            assert table.remove(key) == result
            reference.pop(key, None)
        else:
            capacity = operation["capacity"]
            before = list(table.slots)
            if capacity < len(reference):
                try:
                    table.rebuild(capacity)
                    raise AssertionError("inadequate rebuild accepted")
                except ValueError:
                    result = "rejected"
                    assert table.slots == before
            else:
                table.rebuild(capacity)
                result = [list(pair) for pair in sorted(reference.items())]
        assert state["result"] == result, (operation, state, result)
        pairs = [list(pair) for pair in sorted(reference.items())]
        assert state["logicalPairs"] == pairs
        assert sorted(table.items()) == sorted(reference.items())
        actual_slots = [None if item is EMPTY else {"deleted": True} if item is DELETED
                        else {"key": item[0], "value": item[1]} for item in table.slots]
        assert state["slots"] == actual_slots
        assert not state["issues"] and state["agrees"]
        assert table.size == len(reference)
        counts["probeCommits"] += 1
    for state in case["trace"]["states"]:
        assert not state["issues"]
        slots = state["slots"]
        live = [entry for entry in slots if entry and "key" in entry]
        assert state["live"] == len(live)
        assert state["deleted"] == sum(bool(entry and entry.get("deleted")) for entry in slots)
        assert len({entry["key"] for entry in live}) == len(live)
        assert len(state["visited"]) <= len(slots)
        assert len(set(state["visited"])) == len(state["visited"])
        # Independently ask whether the path before each position contains a hole.
        for destination, entry in enumerate(slots):
            if not entry or "key" not in entry:
                continue
            path = [(entry["key"] + offset) % len(slots) for offset in range(len(slots))]
            assert all(slots[position] is not None for position in path[:path.index(destination)])
        counts["probeStates"] += 1
    counts["probePrograms"] += 1

rng = random.Random(1717)
counts["nativeMapOperations"] = 0
counts["nativeRejectedInputs"] = 0
for capacity in (0, -1, 1.5, True):
    try:
        ProbeMap(capacity)
        raise AssertionError("invalid capacity accepted")
    except ValueError:
        counts["nativeRejectedInputs"] += 1
for MapType in (ProbeMap, ChainedMap):
    table = MapType()
    for key in (True, None, "1", 1.5):
        for method in (lambda: table.put(key, 0), lambda: table.find(key), lambda: table.remove(key)):
            try:
                method()
                raise AssertionError("key outside the declared contract accepted")
            except TypeError:
                counts["nativeRejectedInputs"] += 1
for MapType in (ProbeMap, ChainedMap, ListMap):
    for trial in range(20):
        table = MapType(4) if MapType is ProbeMap else MapType()
        reference = {}
        for step in range(250):
            key = rng.randrange(-20, 21) * 16 + 1
            choice = rng.randrange(4)
            if choice < 2:
                value = None if step % 7 == 0 else rng.randrange(-100, 101)
                if MapType is ProbeMap and key not in reference and len(reference) == len(table.slots):
                    table.rebuild(2 * len(table.slots))
                assert table.put(key, value) == (key not in reference)
                reference[key] = value
            elif choice == 2:
                assert table.remove(key) == (key in reference)
                reference.pop(key, None)
            else:
                assert table.find(key) == (key in reference, reference.get(key))
            for stored, value in reference.items():
                assert table.find(stored) == (True, value)
            if MapType is not ListMap:
                assert table.size == len(reference)
            counts["nativeMapOperations"] += 1
        for key in list(reference):
            assert table.remove(key)
        if MapType is ChainedMap:
            assert len(table.buckets) == 4

counts["hashFamilyCases"] = len(cases["families"])
for state in cases["families"]:
    keys, query = state["keys"], state["query"]
    lengths = []
    for a, b in itertools.product(range(1, 17), range(17)):
        residues = {key: (a * key + b) % 17 for key in keys}
        length = sum(residues[key] % 4 == residues[query] % 4 for key in keys)
        lengths.append(length)
    assert state["totalLength"] == sum(lengths)
    assert state["distribution"] == [{"length": length, "count": lengths.count(length)}
                                     for length in range(1, len(keys) + 1)]
    assert Fraction(sum(lengths), 272) == 1 + (len(keys) - 1) * Fraction(7, 34)
    assert Fraction(sum(lengths), 272) <= Fraction(4 + len(keys) - 1, 4)
    for pair in state["pairCounts"]:
        assert pair["count"] == 56
    selected = (state["multiplier"] - 1) * 17 + state["offset"]
    assert state["selectedLength"] == lengths[selected]

counts["resizeSequences"] = len(cases["resizes"])
counts["resizeStates"] = 0
for case in cases["resizes"]:
    storage, live, total = [None], [], 0
    previous_potential = 0.5
    for number, (operation, state) in enumerate(zip(case["operations"], case["states"][1:], strict=True), 1):
        copied = []
        cost = 0
        if operation == "+":
            if len(live) == len(storage):
                copied = list(live)
                storage = [None] * (2 * len(storage) if case["growth"] == "double" else len(storage) + 1)
            live.append(number)
            cost = 1
        elif live:
            live.pop()
            cost = 1
            ratio = {"half": 2, "quarter": 4}.get(case["policy"])
            if ratio and len(storage) > 1 and ratio * len(live) <= len(storage):
                copied = list(live)
                storage = [None] * max(1, len(storage) // 2)
        total += cost + len(copied)
        assert (state["length"], state["capacity"], state["copies"], state["cost"], state["total"]) == (len(live), len(storage), len(copied), cost + len(copied), total)
        assert state["credits"] == 3 * number - total
        if case["growth"] == "double" and case["policy"] == "quarter":
            assert state["potential"] >= 0
            assert state["cost"] + state["potential"] - previous_potential <= 3
            assert total <= 3 * number + 0.5
        if case["policy"] == "never" and set(case["operations"]) <= {"+"}:
            if case["growth"] == "double":
                potential = 2 * len(live) - len(storage) + 1
                assert potential >= 0 and total == 3 * number - potential
            else:
                assert total == number * (number + 1) // 2
        previous_potential = state["potential"]
        counts["resizeStates"] += 1

for case in cases["dense"]:
    final = case["states"][-1]
    assert set(final["items"]) == set(case["values"]) - {case["removed"]}
    assert len(final["items"]) == len(set(final["items"]))
    assert dict(final["positions"]) == {value: index for index, value in enumerate(final["items"])}
counts["denseModelCases"] = len(cases["dense"])
random_set_module = load("denseRandomSet")
RandomSet = random_set_module["DenseRandomSet"]
try:
    RandomSet().sample()
    raise AssertionError("empty sampling accepted")
except ValueError:
    counts["nativeRejectedInputs"] += 1
counts["nativeDenseOperations"] = 0
table, reference = RandomSet(seed=11), set()
for step in range(2000):
    key = rng.randrange(-20, 21)
    if rng.randrange(2):
        assert table.add(key) == (key not in reference)
        reference.add(key)
    else:
        assert table.remove(key) == (key in reference)
        reference.discard(key)
    assert set(table.items) == reference
    assert table.positions == {value: index for index, value in enumerate(table.items)}
    if reference:
        assert table.sample() in reference
    counts["nativeDenseOperations"] += 1

longest = load("consecutiveRuns")["longest_consecutive"]
counts["consecutiveCases"] = 0
for length in range(7):
    for values in itertools.product((-2, -1, 0, 2), repeat=length):
        ordered = sorted(set(values))
        maximum = run = 0
        for index, value in enumerate(ordered):
            run = run + 1 if index and value == ordered[index - 1] + 1 else 1
            maximum = max(maximum, run)
        actual, checks = longest(values)
        assert actual == maximum
        assert checks <= 3 * len(ordered)
        counts["consecutiveCases"] += 1

count_four = load("pairCounts")["count_quadruples"]
counts["pairCountCases"] = 0
groups = [(), (0,), (1,), (-1, 1), (0, 0)]
for four_groups in itertools.product(groups, repeat=4):
    for target in (-1, 0, 1):
        expected = sum(sum(quadruple) == target for quadruple in itertools.product(*four_groups))
        assert count_four(*four_groups, target=target) == expected
        counts["pairCountCases"] += 1
print(json.dumps(counts))
