// Rendered examples; the owned verifier executes these exact sources independently.
export const pythonFoundationsExamples = {
  first: { code: `first = 18
second = 24
total = first + second
mean = total / 2
print("Mean temperature:", mean, "C")`, output: "Mean temperature: 21.0 C" },
  conversion: { code: `raw = " 24.5 "
clean = raw.strip()
temperature = float(clean)
print(repr(raw), repr(clean))
print(type(raw).__name__, type(temperature).__name__)
print("24" + "1")
print(int("24") + 1)
print(f"Temperature: {temperature:.1f} C")`, output: `' 24.5 ' '24.5'
str float
241
25
Temperature: 24.5 C` },
  indexing: { code: `readings = [18, 21, 24, 27]
print(readings[0], readings[-1])
print(readings[1:3])
print(readings[:2], readings[2:])
readings[1] = 22
readings.append(30)
print(readings)
print(len(readings), sum(readings))`, output: `18 27
[21, 24]
[18, 21] [24, 27]
[18, 22, 24, 27, 30]
5 121` },
  nested: { code: `original = [[18, 21], [24, 27]]
backup = original.copy()
print(original is backup, original[0] is backup[0])
backup[0].append(30)
print(original)
backup[1] = [0]
print(original)
print(backup)`, output: `False True
[[18, 21, 30], [24, 27]]
[[18, 21, 30], [24, 27]]
[[18, 21, 30], [0]]` },
  decisions: { code: `temperature = 31
if temperature >= 30:
    label = "hot"
elif temperature >= 20:
    label = "comfortable"
else:
    label = "cool"
print(label)
print(temperature >= 20, temperature < 30)
print(temperature >= 20 and temperature < 30)`, output: `hot
True False
False` },
  filter: { code: `readings = [18, None, 25, 31, 0]
selected = []
for value in readings:
    if value is None:
        continue
    if value >= 20:
        selected.append(value)
print(selected)
print(readings)`, output: `[25, 31]
[18, None, 25, 31, 0]` },
  mean: { code: `readings = [18, 21, 24]
total = 0
count = 0
for value in readings:
    total = total + value
    count = count + 1
    print("after", value, "total", total, "count", count)
print("mean", total / count)`, output: `after 18 total 18 count 1
after 21 total 39 count 2
after 24 total 63 count 3
mean 21.0` },
  functions: { code: `def convert(celsius):
    fahrenheit = celsius * 9 / 5 + 32
    return fahrenheit

result = convert(20)
print(result)
print(convert(0), convert(100))
print(result + 1)`, output: `68.0
32.0 212.0
69.0` },
  printReturn: { code: `def announce_temperature(celsius):
    print(celsius * 9 / 5 + 32)

result = announce_temperature(20)
print(result)`, output: `68.0
None` },
  parse: { code: `raw = "twenty"
try:
    value = float(raw)
except ValueError:
    print("Invalid number:", repr(raw))
else:
    print("Parsed:", value)
print("Finished this attempt")`, output: `Invalid number: 'twenty'
Finished this attempt` },
  project: {
    files: { "readings.py": `from math import isfinite

def parse_readings(raw_values):
    """Accept a list of strings; skip blanks, reject invalid numbers."""
    values = []
    for raw in raw_values:
        text = raw.strip()
        if text == "":
            continue
        try:
            value = float(text)
        except ValueError:
            raise ValueError(f"invalid reading: {text!r}") from None
        if not isfinite(value):
            raise ValueError(f"reading must be finite: {text!r}")
        values.append(value)
    return values

def summarize(raw_values):
    """Return count and mean in Celsius; reject an empty result."""
    values = parse_readings(raw_values)
    if not values:
        raise ValueError("no readings")
    return {"count": len(values), "mean": sum(values) / len(values)}
` },
    filename: "report.py",
    code: `from readings import summarize

def main():
    raw = ["18", " ", "21", "24"]
    result = summarize(raw)
    print(f"count={result['count']}, mean={result['mean']:.1f} C")
    print("input unchanged:", raw)
    for sample in [["bad"], ["nan"], [" "]]:
        try:
            summarize(sample)
        except ValueError as error:
            print(error)

if __name__ == "__main__":
    main()`,
    output: `count=3, mean=21.0 C
input unchanged: ['18', ' ', '21', '24']
invalid reading: 'bad'
reading must be finite: 'nan'
no readings`,
  },
  moduleCheck: { filename: "check_import.py", code: `import report
from readings import summarize

print("Imported without printing a report")
print(summarize(["0", "24"]))`, output: `Imported without printing a report
{'count': 2, 'mean': 12.0}` },
  projectChange: { filename: "changed_report.py", code: `from readings import summarize

print(summarize(["-6", "0", " ", "12"]))
try:
    summarize(["", " "])
except ValueError as error:
    print(error)`, output: `{'count': 3, 'mean': 2.0}
no readings` },
  mutateParameter: { code: `def add_marker(values):
    values.append(99)
    values = [0]
    return values

readings = [18, 21]
result = add_marker(readings)
print(readings)
print(result)`, output: `[18, 21, 99]
[0]` },
};
