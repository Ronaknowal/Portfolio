// Independently runnable testingPractice examples; verified code/output pairs.
export const testingPracticeExamples = {
testingRepair:{filename:'test_positive_readings.py',code:`import io
import math
import unittest

def positive_mean(text):
    """Mean of strictly positive finite readings; blanks skipped.

    Reject invalid/nonfinite fields. Reject when no positive reading remains.
    Zero and negative readings are valid input but do not enter the mean.
    """
    positive = []
    for line in text.splitlines():
        if not line.strip():
            continue
        value = float(line)
        if not math.isfinite(value):
            raise ValueError("finite readings required")
        if value > 0:
            positive.append(value)
    if not positive:
        raise ValueError("at least one positive reading required")
    return math.fsum(positive) / len(positive)

class PositiveMeanTests(unittest.TestCase):
    def test_mixed_values(self):
        self.assertEqual(positive_mean("-3\\n0\\n6\\n12"), 9.0)

    def test_one_positive_and_blanks(self):
        self.assertEqual(positive_mean("\\n4\\n  "), 4.0)

    def test_invalid_even_when_other_values_are_valid(self):
        for bad in ["6\\nnoise", "6\\nnan", "6\\ninf"]:
            with self.subTest(text=bad), self.assertRaises(ValueError):
                positive_mean(bad)

    def test_no_positive_values(self):
        for text in ["", "0", "-2\\n0"]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                positive_mean(text)

if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PositiveMeanTests)
    result = unittest.TextTestRunner(stream=io.StringIO()).run(suite)
    print("test methods:", result.testsRun, "passed:", result.wasSuccessful())
    print("changed input:", positive_mean("2\\n-4\\n8"))
    raise SystemExit(not result.wasSuccessful())`,output:'test methods: 4 passed: True\nchanged input: 5.0'}
};
