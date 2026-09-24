"""Complement actual bitwise programs with signed digit and byte-conversion oracles."""

import contextlib
import io
import itertools
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "scratch/bitwise-independent-review"
payload = json.loads((DIRECTORY / "native-input.json").read_text(encoding="utf-8"))
namespaces = {}
counts = Counter()
for key, example in payload["examples"].items():
    namespace = {"__name__": "__main__"}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(
            compile(example["code"], example.get("filename", key + ".py"), "exec"),
            namespace,
        )
    assert output.getvalue().strip() == example["output"].strip(), key
    namespaces[key] = namespace
    counts["actualCompletePrograms"] += 1


def parity_by_columns(values):
    width = 2 + max(abs(value).bit_length() for value in values)
    words = [format(value % (2**width), f"0{width}b") for value in values]
    text = "".join(str(sum(map(int, column)) % 2) for column in zip(*words))
    return int(text, 2) - (2 ** width if text[0] == "1" else 0)


rng = random.Random(91832)
for width in [1, 3, 7, 65, 127, 257, 1024]:
    bound = 2 ** (width - 1)
    values = [-bound, -bound + 1, -1, 0, bound - 1]
    values += [rng.randrange(-bound, bound) for _ in range(8)]
    for value in values:
        word = namespaces["words"]["encode_signed"](value, width)
        assert namespaces["words"]["decode_signed"](word, width) == value
        # The independent standard byte-conversion API fixes sign extension.
        byte_width = (width + 7) // 8
        encoded_bytes = value.to_bytes(byte_width, "big", signed=True)
        expected = int.from_bytes(encoded_bytes, "big", signed=False) % (2**width)
        assert word == expected
        counts["changedSignedWordByteOracles"] += 1

for run in range(160):
    width = [2, 31, 64, 129, 1024, 2050][run % 6]
    values = [rng.randrange(-(2**width), 2**width) for _ in range(1 + run % 13)]
    expected = parity_by_columns(values)
    assert namespaces["parity"]["xor_fold"](values) == expected
    assert namespaces["parity"]["xor_fold"](list(reversed(values))) == expected
    counts["signedArbitrarySizeParityOracles"] += 1

for run in range(120):
    width = [2, 31, 100, 511, 2048][run % 5]
    candidates = list(
        dict.fromkeys([0, -1, -(2**width), 2**width - 1, 2**width + 7, -17])
    )
    first, second = rng.sample(candidates, 2)
    pairs = [value for value in candidates if value not in (first, second)]
    values = [first, second] + list(
        itertools.chain.from_iterable((value, value) for value in pairs)
    )
    rng.shuffle(values)
    before = list(values)
    assert sorted(namespaces["twoSingletons"]["two_singletons"](values)) == sorted(
        [first, second]
    )
    assert values == before
    counts["signedLargeTwoSingletonOracles"] += 1

for width in [1, 7, 64, 257, 1024, 4096]:
    for value in [0, 2**width - 1, 2 ** (width - 1), rng.randrange(2**width)]:
        assert namespaces["population"]["population_count"](value) == format(
            value, "b"
        ).count("1")
        expected_power = value > 0 and format(value, "b").count("1") == 1
        assert namespaces["population"]["is_power_of_two"](value) == expected_power
        other = rng.randrange(2**width)
        expected_distance = sum(
            a != b
            for a, b in zip(format(value, f"0{width}b"), format(other, f"0{width}b"))
        )
        assert (
            namespaces["population"]["differing_bits"](value, other)
            == expected_distance
        )
        counts["arbitrarySizeCountPowerDistanceOracles"] += 1

for width in [1, 5, 63, 257, 1024]:
    for _ in range(12):
        left = [rng.randrange(width) for _ in range(20)]
        right = [rng.randrange(width) for _ in range(20)]
        a = namespaces["bitsets"]["pack"](left, width)
        b = namespaces["bitsets"]["pack"](right, width)
        assert a == sum(2**member for member in set(left))
        assert namespaces["bitsets"]["unpack"](a & b, width) == sorted(
            set(left).intersection(right)
        )
        assert namespaces["bitsets"]["unpack"](a ^ b, width) == sorted(
            set(left).symmetric_difference(right)
        )
        counts["changedLargeFiniteSetQueries"] += 1

for values in [[6, 6, 6, 2, 2], [1, 2, 3], [0, 0, 0], [-7, -7, -7, 3, 3]]:
    assert not namespaces["parity"]["audit_promise"](values)
    assert namespaces["parity"]["xor_fold"](values) == parity_by_columns(values)
    counts["invalidPromiseCounterexamples"] += 1
for call in [
    lambda: namespaces["words"]["encode_signed"](1, 1),
    lambda: namespaces["words"]["encode_signed"](True, 8),
    lambda: namespaces["words"]["decode_signed"](2**1024, 1024),
    lambda: namespaces["words"]["decode_signed"](0, 1025),
    lambda: namespaces["population"]["population_count"](-1),
    lambda: namespaces["population"]["differing_bits"](-1, 0),
    lambda: namespaces["parity"]["xor_fold"](iter([1, 2])),
    lambda: namespaces["twoSingletons"]["two_singletons"](iter([0, 1])),
    lambda: namespaces["twoSingletons"]["two_singletons"]([0, 0]),
    lambda: namespaces["bitsets"]["pack"]([True], 8),
]:
    try:
        call()
    except ValueError:
        counts["rejectedNativeCalls"] += 1
    else:
        raise AssertionError("Expected documented rejection")

record = {
    "checkedAt": datetime.now(timezone.utc).isoformat(),
    "status": "passed",
    "sources": payload["sources"],
    "counts": dict(counts),
    "modelCounts": payload["counts"],
    "limits": "Complementary finite cases and source proofs, not arbitrary-input formal verification or timing evidence.",
}
(DIRECTORY / "native-results.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({key: value for key, value in record.items() if key != "sources"}))
