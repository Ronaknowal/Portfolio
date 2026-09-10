export const apiExamples = {
  hints: {
    code: `def double(value: float) -> float:
    return value * 2

print(double(3.0))
# Intentionally invalid for a type checker; Python still executes multiplication.
print(double("ha"))`,
    output: "6.0\nhaha",
  },
  contract: {
    code: `from collections.abc import Sequence
from math import isfinite
import doctest

def mean_ms(values: Sequence[float], *, offset_ms: float = 0.0) -> float:
    """Return the mean measurement minus an offset, in milliseconds.

    Inputs must be finite; values must be nonempty. Neither input is mutated.
    Raises ValueError for empty input or non-finite numbers.
    Raises TypeError for values other than built-in int/float (bool is rejected).

    >>> mean_ms([10.0, 20.0, 30.0], offset_ms=2.0)
    18.0
    >>> mean_ms([5.0])
    5.0
    """
    if len(values) == 0:
        raise ValueError("at least one measurement is required")
    for value in [*values, offset_ms]:
        if type(value) not in (int, float):
            raise TypeError("measurements must be built-in int or float, not bool")
        if not isfinite(value):
            raise ValueError("measurements must be finite")
    return sum(values) / len(values) - offset_ms

if __name__ == "__main__":
    print(mean_ms([10.0, 20.0, 30.0], offset_ms=2.0))
    result = doctest.testmod()
    print("doctests:", result.attempted, "failures:", result.failed)
    for bad in [[], [float("nan")], [True]]:
        try:
            mean_ms(bad)
        except (ValueError, TypeError) as error:
            print(type(error).__name__ + ": " + str(error))`,
    output: "18.0\ndoctests: 2 failures: 0\nValueError: at least one measurement is required\nValueError: measurements must be finite\nTypeError: measurements must be built-in int or float, not bool",
  },
  optional: {
    code: `from collections.abc import Mapping

def find_score(scores: Mapping[str, float], name: str) -> float | None:
    return scores.get(name)

scores = {"baseline": 0.0}
for name in ["baseline", "missing"]:
    score = find_score(scores, name)
    if score is None:
        print(name + ": not found")
    else:
        print(f"{name}: {score:.1f}")`,
    output: "baseline: 0.0\nmissing: not found",
  },
  defaults: {
    code: `def add_tag(tag: str, tags: list[str] | None = None) -> list[str]:
    result = [] if tags is None else list(tags)
    result.append(tag)
    return result

original = ["raw"]
print(add_tag("checked", original))
print("original:", original)
print(add_tag("first"), add_tag("second"))`,
    output: "['raw', 'checked']\noriginal: ['raw']\n['first'] ['second']",
  },
  result: {
    code: `from dataclasses import dataclass, FrozenInstanceError
from typing import Protocol

class Reader(Protocol):
    def read(self) -> str: ...

@dataclass(frozen=True)
class MeasurementSummary:
    count: int
    mean_ms: float

def read_count(source: Reader) -> int:
    return len(source.read().splitlines())

if __name__ == "__main__":
    from io import StringIO
    summary = MeasurementSummary(count=3, mean_ms=18.0)
    print(summary)
    print("lines:", read_count(StringIO("first\\nsecond\\n")))
    try:
        summary.count = 4
    except FrozenInstanceError:
        print("field reassignment rejected")`,
    output: "MeasurementSummary(count=3, mean_ms=18.0)\nlines: 2\nfield reassignment rejected",
  },
  loader: {
    code: `from pathlib import Path
from math import isfinite
import json
import tempfile

def load_scores(path: str | Path, *, minimum: float = 0.0) -> dict[str, float]:
    """Read a JSON object of named scores in [0, 1]; return qualifying scores.

    minimum is inclusive and must be finite and in [0, 1].
    Validate the whole file, even entries that would be filtered out.
    OSError (including FileNotFoundError), UnicodeDecodeError and
    JSONDecodeError propagate. ValueError reports an invalid score schema.
    """
    if type(minimum) not in (int, float) or not 0 <= minimum <= 1 or not isfinite(minimum):
        raise ValueError("minimum must be a finite number in [0, 1]")
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("expected a JSON object")
    scores: dict[str, float] = {}
    for name, value in raw.items():
        if not name.strip() or type(value) not in (int, float):
            raise ValueError("score names must be nonblank and values numeric")
        if not 0 <= value <= 1 or not isfinite(value):
            raise ValueError("scores must be finite and in [0, 1]")
        scores[name] = float(value)
    return {name: score for name, score in scores.items() if score >= minimum}

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "scores.json"
        path.write_text('{"baseline": 0.81, "larger": 0.86}', encoding="utf-8")
        print(load_scores(path, minimum=0.85))
        path.write_text('{"baseline": true}', encoding="utf-8")
        try:
            load_scores(path)
        except ValueError as error:
            print(type(error).__name__ + ": " + str(error))
        try:
            load_scores(Path(folder) / "missing.json")
        except FileNotFoundError:
            print("missing file: FileNotFoundError")`,
    output: "{'larger': 0.86}\nValueError: score names must be nonblank and values numeric\nmissing file: FileNotFoundError",
  },
};
