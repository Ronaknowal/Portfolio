// Independently runnable pythonCore examples; verified code/output pairs.
export const pythonCoreExamples = {
basicsTypes: {
    code: `raw = " 24.5 "
temperature = float(raw.strip())
city = "Pune"
print(type(raw).__name__, type(temperature).__name__)
print(f"{city}: {temperature:.1f} C")
print(7 / 2, 7 // 2, 7 % 2, 2 ** 3)
print(-7 // 2)
print(0.1 + 0.2 == 0.3)
from math import isclose
print(isclose(0.1 + 0.2, 0.3))`,
    output: `str float
Pune: 24.5 C
3.5 3 1 8
-4
False
True`,
  },
basicsCollections: {
    code: `readings = [18, 21, 24]
readings.append(27)
print(readings[0], readings[-1], readings[1:3])
print(readings.pop(), readings)
record = {"city": "Pune", "temperature": 24}
record["unit"] = "C"
print(record.get("humidity", "unknown"))
for key, value in record.items():
    print(key, value)
point = (3, 4)
x, y = point
print(x + y)
labels = {"warm", "cold", "warm"}
print(sorted(labels), "warm" in labels)`,
    output: `18 27 [21, 24]
27 [18, 21, 24]
unknown
city Pune
temperature 24
unit C
7
['cold', 'warm'] True`,
  },
basicsAliasing: {
    code: `a = [18, 21]
b = a
c = a.copy()
b.append(24)
print(a, b, c)
b = [0]
print(a, b)
print(a == c, a is c)`,
    output: `[18, 21, 24] [18, 21, 24] [18, 21]
[18, 21, 24] [0]
False False`,
  },
basicsBranches: {
    code: `readings = [18, None, 25, 31]
for position, value in enumerate(readings, start=1):
    if value is None:
        continue
    if value >= 30:
        label = "hot"
    elif value >= 20:
        label = "comfortable"
    else:
        label = "cool"
    print(position, label)
print(list(range(1, 6, 2)))
print(bool([]), bool(0), bool("0"))`,
    output: `1 cool
3 comfortable
4 hot
[1, 3, 5]
False False True`,
  },
basicsLoops: {
    code: `remaining = 3
while remaining > 0:
    print("attempt", 4 - remaining)
    remaining -= 1
    if remaining == 1:
        break
values = [18, 25, 31]
warm = [value for value in values if value >= 20]
print(warm)
print([round(value * 9 / 5 + 32, 1) for value in warm])
print(list(zip(["Mon", "Tue"], [18, 25])))`,
    output: `attempt 1
attempt 2
[25, 31]
[77.0, 87.8]
[('Mon', 18), ('Tue', 25)]`,
  },
basicsFunctions: {
    code: `def mean(values: list[float], *, digits: int = 1) -> float:
    """Return a rounded mean; reject an empty collection."""
    if not values:
        raise ValueError("at least one reading is required")
    return round(sum(values) / len(values), digits)

print(mean([18, 21, 25]))
print(mean([18, 21, 25], digits=2))

def announce(message):
    print(message)

result = announce("Ready")
print(result)
try:
    mean([])
except ValueError as error:
    print(type(error).__name__, str(error))`,
    output: `21.3
21.33
Ready
None
ValueError at least one reading is required`,
  },
basicsDefaults: {
    code: `def add_reading(value, readings=None):
    if readings is None:
        readings = []
    readings.append(value)
    return readings

print(add_reading(18))
print(add_reading(21))
shared = []
print(add_reading(25, shared), shared)

def total(*values, **options):
    return round(sum(values), options.get("digits", 1))

print(total(1.23, 2.34, digits=2))`,
    output: `[18]
[21]
[25] [25]
3.57`,
  },
basicsProject: {
    files: {
      "readings.py": `def summarize(raw_values):
    values = []
    for text in raw_values:
        text = text.strip()
        if not text:
            continue
        try:
            values.append(float(text))
        except ValueError:
            raise ValueError(f"invalid reading: {text!r}") from None
    if not values:
        raise ValueError("no readings")
    return {"count": len(values), "mean": sum(values) / len(values)}
`,
    },
    filename: "report.py",
    code: `from readings import summarize

def main():
    result = summarize(["18", " ", "21", "24"])
    print(f"count={result['count']}, mean={result['mean']:.1f} C")
    assert result == {"count": 3, "mean": 21.0}
    try:
        summarize(["bad"])
    except ValueError as error:
        print(error)

if __name__ == "__main__":
    main()`,
    output: `count=3, mean=21.0 C
invalid reading: 'bad'`,
  },
basicsPractice: {
    code: `def best_model(scores):
    if not scores:
        raise ValueError("scores must not be empty")
    winner = None
    best_score = float("-inf")
    for name, score in scores.items():
        if winner is None or score > best_score:
            winner, best_score = name, score
    return winner

print(best_model({"baseline": 0.8, "new": 0.9}))
print(best_model({"first": -2, "second": -2}))
try:
    best_model({})
except ValueError as error:
    print(error)`,
    output: `new
first
scores must not be empty`,
  }
};
