// Independently runnable apiPractice examples; verified code/output pairs.
export const apiPracticeExamples = {
ownership:{code:`def shared(tag, tags=[]):
    tags.append(tag)
    return tags

first = shared("first")
second = shared("second")
print("same object:", first is second)
print("first now:", first)

def copied(tag, tags=None):
    result = [] if tags is None else list(tags)
    result.append(tag)
    return result

original = ["raw"]
result = copied("checked", original)
print("same object:", result is original)
print("original:", original, "result:", result)
nested = [[1]]
outer_copy = list(nested)
outer_copy[0].append(2)
print("nested still shared:", nested)`,output:"same object: True\nfirst now: ['first', 'second']\nsame object: False\noriginal: ['raw'] result: ['raw', 'checked']\nnested still shared: [[1, 2]]"},
apiTransfer:{filename:'duration_api.py',code:`from collections.abc import Iterable
from math import isfinite
import doctest

def durations_ms(values: Iterable[float], *, input_unit: str = "ms") -> list[float]:
    """Return new millisecond durations, consuming values once.

    Accept built-in int/float values that are finite and nonnegative; reject bool.
    input_unit must be 's' or 'ms'. Empty input returns an empty list.
    Raise ValueError for unknown units, negative/nonfinite values or overflow;
    TypeError for a nonmeasurement value. Do not mutate an input collection.
    A one-shot iterator is consumed, including a prefix if validation fails.

    >>> durations_ms([0, 0.25, 2], input_unit="s")
    [0.0, 250.0, 2000.0]
    >>> durations_ms([])
    []
    """
    if input_unit not in ("s", "ms"):
        raise ValueError("input_unit must be s or ms")
    factor = 1000.0 if input_unit == "s" else 1.0
    result = []
    for value in values:
        if type(value) not in (int, float):
            raise TypeError("duration must be a built-in int or float, not bool")
        try:
            numeric = float(value)
        except OverflowError as error:
            raise ValueError("duration outside finite range") from error
        if not isfinite(numeric) or numeric < 0:
            raise ValueError("duration must be finite and nonnegative")
        converted = numeric * factor
        if not isfinite(converted):
            raise ValueError("converted duration outside finite range")
        result.append(converted)
    return result

if __name__ == "__main__":
    original = [0, 0.25, 2]
    print(durations_ms(original, input_unit="s"))
    print("original:", original)
    source = (value for value in [10, 20])
    print("generator:", durations_ms(source), "remaining:", list(source))
    result = doctest.testmod()
    print("doctests:", result.attempted, "failures:", result.failed)
    for bad in [[True], [-1], [float("inf")]]:
        try:
            durations_ms(bad)
        except (TypeError, ValueError) as error:
            print(type(error).__name__)
    raise SystemExit(result.failed != 0)`,output:'[0.0, 250.0, 2000.0]\noriginal: [0, 0.25, 2]\ngenerator: [10.0, 20.0] remaining: []\ndoctests: 2 failures: 0\nTypeError\nValueError\nValueError'}
};
