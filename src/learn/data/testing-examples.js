// Independently runnable testing examples; verified code/output pairs.
export const testingExamples = {
testingRed: {
    code: `def wrong_mean(values):
    total = 0
    for value in values:
        total += value
        return total / len(values)  # wrong indentation

actual = wrong_mean([18, 24])
expected = 21.0
try:
    assert actual == expected, f"expected {expected}, got {actual}"
except AssertionError as error:
    print("FAIL:", error)
print("first iteration:", 18 / 2)`,
    output: `FAIL: expected 21.0, got 9.0
first iteration: 9.0`,
  },
testingSuite: {
    files: {
      "metrics.py": `from math import fsum, isfinite

def mean(values):
    if not values:
        raise ValueError("no readings")
    if any(not isfinite(value) for value in values):
        raise ValueError("readings must be finite")
    return fsum(values) / len(values)
`,
    },
    filename: "test_metrics.py",
    code: `import unittest
from io import StringIO
from metrics import mean

class MeanTests(unittest.TestCase):
    def test_ordinary(self):
        self.assertAlmostEqual(mean([18, 24]), 21.0)

    def test_single_and_zero(self):
        for values, expected in [([0], 0), ([-2], -2), ([-2, 2], 0)]:
            with self.subTest(values=values):
                self.assertEqual(mean(values), expected)

    def test_empty(self):
        with self.assertRaisesRegex(ValueError, "no readings"):
            mean([])

    def test_nonfinite(self):
        for value in [float("nan"), float("inf")]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    mean([value])

if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(MeanTests)
    result = unittest.TextTestRunner(stream=StringIO()).run(suite)
    print("tests:", result.testsRun)
    print("failures:", len(result.failures), "errors:", len(result.errors))
    print("passed:", result.wasSuccessful())
    raise SystemExit(not result.wasSuccessful())`,
    output: `tests: 4
failures: 0 errors: 0
passed: True`,
  },
testingIsolation: {
    code: `from unittest.mock import Mock

def load_mean(read_text):
    values = [float(line) for line in read_text().splitlines() if line.strip()]
    if not values:
        raise ValueError("no readings")
    return sum(values) / len(values)

reader = Mock(return_value="18\\n24\\n")
print(load_mean(reader))
reader.assert_called_once_with()
reader.side_effect = OSError("unavailable")
try:
    load_mean(reader)
except OSError as error:
    print(type(error).__name__, str(error))`,
    output: `21.0
OSError unavailable`,
  },
testingIntegration: {
    code: `from pathlib import Path
from tempfile import TemporaryDirectory

def load_readings(path):
    with open(path, encoding="utf-8") as file:
        return [float(line) for line in file if line.strip()]

with TemporaryDirectory() as directory:
    path = Path(directory) / "readings.txt"
    path.write_text("18\\n\\n24\\n", encoding="utf-8")
    values = load_readings(path)
    assert values == [18.0, 24.0]
    print(values)
    print("mean:", sum(values) / len(values))
print("temporary directory removed:", not Path(directory).exists())`,
    output: `[18.0, 24.0]
mean: 21.0
temporary directory removed: True`,
  },
testingTolerance: {
    code: `from math import isclose

def fahrenheit(celsius):
    return celsius * 9 / 5 + 32

for celsius, expected in [(0, 32), (100, 212), (-40, -40)]:
    assert isclose(fahrenheit(celsius), expected, rel_tol=0, abs_tol=1e-9)
print("3 reference cases passed")
for celsius in [-20, 0, 30]:
    delta = fahrenheit(celsius + 10) - fahrenheit(celsius)
    assert isclose(delta, 18, rel_tol=0, abs_tol=1e-9)
print("translation property passed")
print(0.1 + 0.2 == 0.3, isclose(0.1 + 0.2, 0.3))`,
    output: `3 reference cases passed
translation property passed
False True`,
  }
};
