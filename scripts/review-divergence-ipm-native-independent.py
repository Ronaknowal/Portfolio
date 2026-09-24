"""Independent execution and changed exact-test calibration from displayed code."""
import contextlib
import io
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

directory = Path("scratch/divergence-ipm-independent")
examples = json.loads((directory / "displayed-examples.json").read_text(encoding="utf8"))
namespaces = {}
for key, example in examples.items():
    namespace = {"__name__": "__main__"}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(example["code"], key + ".py", "exec"), namespace)
    assert output.getvalue().rstrip() == example["expected"].rstrip(), key
    namespaces[key] = namespace

exact_test = namespaces["project"]["exact_test"]
checked = 0
for pool in ([-1, 0, 0, 1, 2], [-1, -.5, 0, .5, 1, 2],
             [-2, -1, -.5, 0, .5, 1, 2]):
    for first_size in range(2, len(pool) - 1):
        for widths in ((.75,), (.4, .8, 1.6)):
            positions = np.asarray(pool)
            grams = [np.exp(-.5 * ((positions[:, None] - positions[None, :]) / sigma) ** 2) for sigma in widths]
            allocations = list(itertools.combinations(range(len(pool)), first_size))
            scores = []
            for allocation in allocations:
                coefficients = np.full(len(pool), -1 / (len(pool) - first_size))
                coefficients[list(allocation)] = 1 / first_size
                scores.append(max(float(coefficients @ gram @ coefficients) for gram in grams))
            observed, tail, count, p_value = exact_test(pool[:first_size], pool[first_size:], widths)
            assert math.isclose(observed, scores[0], abs_tol=1e-12)
            expected_tail = sum(score >= scores[0] - 1e-12 for score in scores)
            assert count == len(allocations) and tail == expected_tail and p_value == tail / count
            swapped = exact_test(pool[first_size:], pool[:first_size], widths)
            assert swapped[1:] == (tail, count, p_value)
            for level in (.05, .1, .25, .5):
                p_values = [sum(score >= observed - 1e-12 for score in scores) / count for observed in scores]
                assert sum(p <= level for p in p_values) / count <= level + 1e-12
            checked += 1
record = {"checkedAt": datetime.now(timezone.utc).isoformat(), "displayedPrograms": len(examples),
          "changedExactTests": checked, "independentMethod": "Gaussian Gram quadratic forms, unequal sample sizes, ties, complete bandwidth-selection rules and inclusive-tail rank calibration",
          "allPassed": True}
(directory / "native-review.json").write_text(json.dumps(record, indent=2))
print(json.dumps(record, indent=2))
