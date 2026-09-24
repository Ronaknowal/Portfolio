"""Author-owned complete examples. Run each actual formatted program before capture."""

import ast
import contextlib
import io
import json
from pathlib import Path

import black


EXAMPLES = [
    ("estimands", "Choose what one unit of weight represents", "Do equal-class and equal-student averages answer the same question?", '''
from fractions import Fraction as F

classes = [(2, F(4)), (6, F(8))]  # (number of students, class mean score)
class_average = sum(mean for size, mean in classes) / len(classes)
student_average = sum(size * mean for size, mean in classes) / sum(size for size, mean in classes)
print("equal-class mean:", class_average)
print("equal-student mean:", student_average)
print("The target determines the weights; neither row count defines it automatically.")
'''),
    ("finite-samples", "Enumerate a sampling distribution without Monte Carlo", "What still changes when the frame itself is incomplete?", '''
from fractions import Fraction as F
from itertools import combinations


def describe_sampling(population, frame, n):
    if not population or len(set(frame)) != len(frame):
        raise ValueError("Use a nonempty population and unique frame indices")
    if any(type(i) is not int or i < 0 or i >= len(population) for i in frame):
        raise ValueError("Frame indices must refer to the population")
    if type(n) is not int or not 1 <= n <= len(frame):
        raise ValueError("n must be an integer within the frame size")
    values = [F(value) for value in population]
    target = sum(values) / len(values)
    means = [sum(values[i] for i in subset) / n for subset in combinations(frame, n)]
    expectation = sum(means) / len(means)
    variance = sum((mean - expectation) ** 2 for mean in means) / len(means)
    mse = sum((mean - target) ** 2 for mean in means) / len(means)
    return len(means), expectation, variance, expectation - target, mse


population = list(range(2, 17, 2))
for frame, n in [(list(range(8)), 2), (list(range(8)), 4), (list(range(8)), 8), (list(range(4)), 4)]:
    count, expected, variance, bias, mse = describe_sampling(population, frame, n)
    print(f"frame={len(frame)}, n={n}: {count} subsets; E={expected}, Var={variance}, bias={bias}, MSE={mse}")
try:
    describe_sampling(population, [0, 0], 1)
except ValueError as error:
    print("rejected:", error)
'''),
    ("inclusion-weights", "Derive weights from the actual sample probabilities", "Which denominator is fixed before the sample is selected?", '''
from fractions import Fraction as F
from itertools import combinations


def compare_estimators(values, sample_weights):
    samples = list(combinations(range(len(values)), 2))
    if len(sample_weights) != len(samples) or any(w < 0 for w in sample_weights) or sum(sample_weights) <= 0:
        raise ValueError("Supply nonnegative weights for every size-two subset")
    probabilities = [F(w) / sum(sample_weights) for w in sample_weights]
    inclusion = [sum(p for sample, p in zip(samples, probabilities) if i in sample) for i in range(len(values))]
    if any(pi == 0 for pi in inclusion):
        raise ValueError("A target unit has zero inclusion probability")
    rows = []
    for sample, probability in zip(samples, probabilities):
        numerator = sum(F(values[i]) / inclusion[i] for i in sample)
        ht = numerator / len(values)
        ratio = numerator / sum(1 / inclusion[i] for i in sample)
        raw = sum(F(values[i]) for i in sample) / len(sample)
        rows.append((probability, raw, ht, ratio))
    expectations = [sum(row[0] * row[column] for row in rows) for column in [1, 2, 3]]
    return inclusion, expectations


inclusion, expected = compare_estimators([2, 4, 8, 10], [4, 2, 1, 1, 1, 1])
print("inclusion:", ", ".join(map(str, inclusion)))
print("expected raw / fixed-N HT / normalized ratio:", ", ".join(map(str, expected)))
print("target: 6")
try:
    compare_estimators([2, 4, 8, 10], [1, 0, 0, 0, 0, 0])
except ValueError as error:
    print("rejected:", error)
'''),
    ("replacement", "Keep per-draw and at-least-once probabilities separate", "What happens to a unit that appears twice in the same draw sequence?", '''
from fractions import Fraction as F
from itertools import product

values = [F(2), F(8)]
per_draw = [F(3, 4), F(1, 4)]
n = 2
inclusion = [1 - (1 - p) ** n for p in per_draw]
expected_hh = expected_ht = F(0)
for draws in product(range(2), repeat=n):
    probability = per_draw[draws[0]] * per_draw[draws[1]]
    # Hansen-Hurwitz: average a total contribution for EACH draw.
    hh_total = sum(values[i] / per_draw[i] for i in draws) / n
    # Horvitz-Thompson: count each DISTINCT observed unit once.
    ht_total = sum(values[i] / inclusion[i] for i in set(draws))
    expected_hh += probability * hh_total
    expected_ht += probability * ht_total
    print(f"draws={draws}, probability={probability}, HH total={hh_total}, HT total={ht_total}")
print("at-least-once inclusion:", ", ".join(map(str, inclusion)))
print("expected totals:", expected_hh, expected_ht, "true total:", sum(values))
'''),
    ("grouped-samples", "Compare what the grouping permits", "Can equal inclusion probabilities coexist with very different uncertainty?", '''
from fractions import Fraction as F
from itertools import combinations, product


def moments(values):
    mean = sum(values) / len(values)
    return mean, sum((value - mean) ** 2 for value in values) / len(values)


population = list(map(F, range(2, 17, 2)))
stratified = [(sum(a) / 2 + sum(b) / 2) / 2 for a, b in product(combinations(population[:4], 2), combinations(population[4:], 2))]
srs = [sum(s) / 4 for s in combinations(population, 4)]
print("SRS n=4 E,Var:", *moments(srs))
print("two per half E,Var:", *moments(stratified))
periodic = list(map(F, [0, 10] * 4))
systematic = [sum(periodic[start::2]) / 4 for start in [0, 1]]
periodic_srs = [sum(s) / 4 for s in combinations(periodic, 4)]
print("random-start systematic E,Var:", *moments(systematic))
print("periodic SRS E,Var:", *moments(periodic_srs))
# Unequal population shares; equal sample counts do not change the target shares.
stratum_sizes, stratum_means = [2, 6], [F(4), F(8)]
weighted = sum(size * mean for size, mean in zip(stratum_sizes, stratum_means)) / sum(stratum_sizes)
print("population-weighted stratum mean:", weighted, "unweighted:", sum(stratum_means) / 2)
'''),
    ("shared-errors", "Carry shared variation through the actual joint outcomes", "When does an average preserve a target that a difference would remove?", '''
from fractions import Fraction as F
from itertools import product

outcomes = [(F(10 + shared + ea), F(20 + shared + eb)) for shared, ea, eb in product([-2, 2], [-1, 1], [-1, 1])]


def moments(values):
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, variance


for label, values in [
    ("A", [a for a, b in outcomes]),
    ("B", [b for a, b in outcomes]),
    ("average", [(a + b) / 2 for a, b in outcomes]),
    ("difference B-A", [b - a for a, b in outcomes]),
    ("calibrated common baseline average", [((a - 10) + (b - 20)) / 2 for a, b in outcomes]),
]:
    print(label, "mean,variance:", *moments(values))
for units, repeats in [(1, 16), (4, 4), (16, 1)]:
    variance = F(4, units) + F(1, units * repeats)
    print(f"G={units}, m={repeats}: variance={variance}; with fixed bias 2, MSE={variance+4}")
print("A difference targets the offset contrast; it does not estimate a common additive signal.")
'''),
    ("assignments", "Enumerate actual observed outcomes under every allowed assignment", "Does unbiasedness require the effect to be identical for all enrolled units?", '''
from fractions import Fraction as F
from itertools import combinations


def assignment_distribution(y0, y1, n1):
    if len(y0) != len(y1) or type(n1) is not int or not 0 < n1 < len(y0):
        raise ValueError("Aligned potential outcomes and two nonempty arms are required")
    y0, y1 = list(map(F, y0)), list(map(F, y1))
    n, n0 = len(y0), len(y0) - n1
    estimates = []
    for treated in combinations(range(n), n1):
        observed = [y1[i] if i in treated else y0[i] for i in range(n)]
        estimate = sum(observed[i] for i in treated) / n1 - sum(observed[i] for i in range(n) if i not in treated) / n0
        estimates.append(estimate)
    expected = sum(estimates) / len(estimates)
    variance = sum((x - expected) ** 2 for x in estimates) / len(estimates)
    target = sum(b - a for a, b in zip(y0, y1)) / n
    return len(estimates), target, expected, variance


y0 = [0, 1, 5, 6, 10, 11]
for effects in [[2] * 6, [0, 2, 4, 0, 2, 4]]:
    y1 = [base + effect for base, effect in zip(y0, effects)]
    count, target, expected, variance = assignment_distribution(y0, y1, 3)
    print(f"effects={effects}: {count} assignments; target={target}, E={expected}, Var={variance}")
print("The complete potential-outcome table is known here only because it is synthetic.")
'''),
    ("blocked-assignment", "Change the allowed allocations while holding outcomes fixed", "Do all ways of making pairs improve precision?", '''
from fractions import Fraction as F
from itertools import product

y0 = list(map(F, [0, 1, 5, 6, 10, 11]))
y1 = [value + 2 for value in y0]
for name, pairs in [
    ("similar", [(0, 1), (2, 3), (4, 5)]),
    ("unlike", [(0, 5), (1, 4), (2, 3)]),
]:
    estimates = []
    for treated in product(*pairs):
        within = [y1[t] - y0[b if t == a else a] for t, (a, b) in zip(treated, pairs)]
        estimates.append(sum(within) / len(pairs))
    expected = sum(estimates) / len(estimates)
    variance = sum((x - expected) ** 2 for x in estimates) / len(estimates)
    print(f"{name} pairs: {len(estimates)} assignments; E={expected}, Var={variance}")
print("Blocking changes the randomization distribution; use the matching analysis.")
'''),
    ("factorial", "Ask for the effect at a declared setting", "Which interaction values remain possible if the fourth cell is missing?", '''
from fractions import Fraction as F


def effects(cells, high_b_share):
    low = cells[0][1] - cells[0][0]
    high = cells[1][1] - cells[1][0]
    return low, high, high - low, (1 - high_b_share) * low + high_b_share * high


for fourth in [11, 15]:
    cells = [[F(10), F(12)], [F(9), F(fourth)]]
    print(f"fourth={fourth}: low-B, high-B, interaction, equal-mixture effect:", *effects(cells, F(1, 2)))
cells = [[F(10), F(12)], [F(9), F(15)]]
print("high-B share 1/4; averaged A effect:", effects(cells, F(1, 4))[-1])
print("These cells are specified means; one noisy measurement per cell cannot estimate repeatability.")
'''),
    ("precision-budget", "Specify the uncertainty target and the collection cost", "What assumptions does a minimum sample size calculation leave untouched?", '''
from fractions import Fraction as F

# Exact design SD <= 1, given fixed-population N=8 and S^2=24.
N, S2, variance_target = 8, F(24), F(1)
eligible = [(n, (1 - F(n, N)) * S2 / n) for n in range(1, N + 1)]
valid = [(n, variance) for n, variance in eligible if variance <= variance_target]
print("smallest n with design SD <= 1:", valid[0][0], "variance:", valid[0][1])
# A new independent unit costs 4; each reading costs 1. Budget = 32.
plans = []
for G in range(1, 17):
    for m in range(1, 17):
        cost = G * (4 + m)
        if cost <= 32:
            plans.append((F(4, G) + F(1, G * m), cost, G, m))
variance, cost, G, m = min(plans)
print(f"best declared-cost plan: G={G}, m={m}, cost={cost}, variance={variance}")
print("Changing costs or variance assumptions can change this answer; fixed bias is unaffected.")
'''),
    ("attrition", "Bound the missing assigned outcomes instead of turning them into zeros", "Can the observed contrast determine the full assigned-group contrast's sign?", '''
from fractions import Fraction as F


def mean_bounds(values, lower=0, upper=10):
    if not values or lower > upper:
        raise ValueError("Use nonempty assigned outcomes and ordered bounds")
    observed = [F(value) for value in values if value is not None]
    if any(not lower <= value <= upper for value in observed):
        raise ValueError("A recorded outcome is outside the declared bounds")
    missing = len(values) - len(observed)
    return ((sum(observed) + missing * lower) / len(values), (sum(observed) + missing * upper) / len(values))


A, B = [8, 9, None, None], [4, 5, 6, 7]
a, b = mean_bounds(A), mean_bounds(B)
print("A full mean bounds:", *a)
print("B full mean bounds:", *b)
print("assigned-group contrast bounds:", a[0] - b[1], a[1] - b[0])
print("observed-only contrast:", F(17, 2) - F(22, 4))
print("These bound missing assigned outcomes, not each unit's unobserved counterfactual.")
'''),
    ("protocol", "Generate an auditable synthetic blocked assignment", "Can every recorded outcome be traced to an enrolled unit and its original assignment?", '''
import csv
import io
import random
from fractions import Fraction as F

# Synthetic classrooms: assignment unit is the class; two readings per class.
# Each pair is fixed from pretreatment information. A means the new method.
pairs = [("C1", "C2"), ("C3", "C4"), ("C5", "C6")]
rng = random.Random(49)
assignment = {}
for pair in pairs:
    treated = rng.choice(pair)
    assignment.update({unit: ("A" if unit == treated else "B") for unit in pair})
baseline = {f"C{i+1}": value for i, value in enumerate([40, 42, 60, 62, 80, 82])}
output = io.StringIO()
writer = csv.writer(output, lineterminator="\\n")
writer.writerow(["class_id", "assigned_method", "reading_id", "retention_score"])
class_means = {}
for unit, method in assignment.items():
    readings = [baseline[unit] + (3 if method == "A" else 0) + error for error in [-1, 1]]
    class_means[unit] = F(sum(readings), len(readings))
    for index, score in enumerate(readings, 1):
        writer.writerow([unit, method, index, score])
print(output.getvalue().strip())
contrasts = []
for pair in pairs:
    a = next(unit for unit in pair if assignment[unit] == "A")
    b = next(unit for unit in pair if assignment[unit] == "B")
    contrasts.append(class_means[a] - class_means[b])
print("within-pair class contrasts:", ", ".join(map(str, contrasts)))
print("equal-class estimate:", sum(contrasts) / len(contrasts))
print("6 assigned classes, 3 independently randomized pairs, 12 recorded rows.")
print("Synthetic outcomes and a reproducible allocation are not evidence of real teaching effectiveness.")
'''),
]


def main():
    output = []
    for identifier, title, question, code in EXAMPLES:
        code = code.strip() + "\n"
        formatted = black.format_str(code, mode=black.Mode(line_length=88))
        assert ast.dump(ast.parse(code)) == ast.dump(ast.parse(formatted))
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            exec(compile(formatted, identifier + ".py", "exec"), {"__name__": "__main__"})
        output.append({"id": identifier, "title": title, "question": question, "code": formatted, "expected": capture.getvalue().strip()})
    destination = Path("src/learn/data/sampling-measurement-examples.js")
    destination.write_text("// Complete Python programs, formatted without AST changes and executed before capture.\nexport const samplingMeasurementExamples = " + json.dumps(output, ensure_ascii=False, indent=2) + ";\n", encoding="utf-8")
    print(json.dumps({"programsExecuted": len(output), "outputs": {example["id"]: example["expected"] for example in output}}, indent=2))


if __name__ == "__main__":
    main()
