"""Generate complete standalone programs and record their real standard-library output."""
import json
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'scratch/divergence-ipm-native-verification'
DIRECTORY.mkdir(parents=True, exist_ok=True)
examples = {}


def add(name, title, question, code):
    code = dedent(code).strip() + '\n'
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True, timeout=30)
    assert not result.stderr
    examples[name] = {'title': title, 'question': question, 'code': code, 'expected': result.stdout.strip()}
    (DIRECTORY / f'{name}.py').write_text(code, encoding='utf-8')
    print(name, result.stdout.strip(), sep='\n')


LAWS = dedent('''\
import math


def validate(p, q):
    if len(p) != len(q) or not p:
        raise ValueError("Use matching nonempty alphabets.")
    for law in (p, q):
        if any(not math.isfinite(x) or x < 0 for x in law):
            raise ValueError("Masses must be finite and nonnegative.")
        if not math.isclose(math.fsum(law), 1, abs_tol=1e-12):
            raise ValueError("Masses must sum to one.")


def kl(p, q):
    validate(p, q)
    if any(a > 0 and b == 0 for a, b in zip(p, q)):
        return math.inf
    return math.fsum(a * (math.log(a) - math.log(b)) for a, b in zip(p, q) if a)


def comparisons(p, q):
    validate(p, q)
    middle = [(a + b) / 2 for a, b in zip(p, q)]
    chi = math.inf if any(a > 0 and b == 0 for a, b in zip(p, q)) else math.fsum(
        (a - b) ** 2 / b for a, b in zip(p, q) if b
    )
    return {
        "KL": kl(p, q),
        "reverse KL": kl(q, p),
        "JS": (kl(p, middle) + kl(q, middle)) / 2,
        "H squared": math.fsum((math.sqrt(a) - math.sqrt(b)) ** 2 for a, b in zip(p, q)) / 2,
        "chi squared": chi,
        "TV": math.fsum(abs(a - b) for a, b in zip(p, q)) / 2,
    }


''')

MMD = dedent('''\
import math


def mmd_squared(x, y, bandwidth=1):
    if min(len(x), len(y)) < 2:
        raise ValueError("Both groups need at least two observations.")
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("Bandwidth must be positive and finite.")
    if any(not math.isfinite(value) for value in list(x) + list(y)):
        raise ValueError("Observations must be finite.")
    def kernel(a, b):
        return math.exp(-0.5 * ((a - b) / bandwidth) ** 2)
    xx = math.fsum(kernel(a, b) for a in x for b in x)
    yy = math.fsum(kernel(a, b) for a in y for b in y)
    xy = math.fsum(kernel(a, b) for a in x for b in y)
    n, m = len(x), len(y)
    biased = xx / n**2 + yy / m**2 - 2 * xy / (n * m)
    # Every diagonal RBF similarity is exactly one. All cross pairs stay.
    unbiased = (xx - n) / (n * (n - 1)) + (yy - m) / (m * (m - 1)) - 2 * xy / (n * m)
    return biased, unbiased


''')

add('original', 'The original categorical calculation',
    'Which mismatch does each of the three numbers describe, and are their scales directly comparable?', '''
import math

p = [0.7, 0.2, 0.1]
q = [0.4, 0.5, 0.1]
m = [(a + b) / 2 for a, b in zip(p, q)]

forward_kl = sum(a * math.log(a / b) for a, b in zip(p, q))
js = 0.5 * sum(a * math.log(a / b) for a, b in zip(p, m))
js += 0.5 * sum(a * math.log(a / b) for a, b in zip(q, m))
tv = 0.5 * sum(abs(a - b) for a, b in zip(p, q))

print(round(forward_kl, 3), round(js, 3), round(tv, 3))
''')

add('support', 'Keep zero support and metric conventions explicit',
    'Compare overlapping and disjoint laws. Then test whether JS itself obeys the triangle inequality.', LAWS + '''
for p, q in [([0.7, 0.2, 0.1], [0.4, 0.5, 0.1]), ([1, 0], [0, 1])]:
    print("P, Q:", p, q)
    print({name: round(value, 6) for name, value in comparisons(p, q).items()})
left, middle, right = [1, 0], [0.5, 0.5], [0, 1]
direct = comparisons(left, right)["JS"]
leg = comparisons(left, middle)["JS"]
print("JS direct / two legs:", round(direct, 6), round(2 * leg, 6))
print("sqrt(JS) direct / two legs:", round(math.sqrt(direct), 6), round(2 * math.sqrt(leg), 6))
''')

add('events', 'Find the event that separates two sources',
    'Enumerate every event, calculate optimal equal-prior classification accuracy, and see what merging outcomes loses.', LAWS + '''
from itertools import product

p, q = [0.7, 0.2, 0.1, 0], [0.4, 0.5, 0.1, 0]
events = list(product([0, 1], repeat=4))
gap = lambda mask: math.fsum((a - b) * take for a, b, take in zip(p, q, mask))
best = max(events, key=gap)
tv = comparisons(p, q)["TV"]
accuracy = 0.5 * math.fsum(a if a >= b else b for a, b in zip(p, q))
print("Best event mask:", best)
print("Event gap / TV:", round(gap(best), 6), round(tv, 6))
print("Best equal-prior accuracy:", round(accuracy, 6))
for mapping in ([0, 0, 1, 1], [0, 1, 1, 1]):
    pushed_p = [sum(p[i] for i in range(4) if mapping[i] == group) for group in [0, 1]]
    pushed_q = [sum(q[i] for i in range(4) if mapping[i] == group) for group in [0, 1]]
    values = comparisons(pushed_p, pushed_q)
    print("Map:", mapping, "KL / TV:", round(values["KL"], 6), round(values["TV"], 6))
print("Pinsker lower bound / KL:", round(2 * tv**2, 6), round(kl(p, q), 6))
''')

add('observers', 'Construct an optimal one-dimensional Lipschitz critic',
    'Two laws have the same mean. Can an event test or a slope-limited nonlinear function still distinguish them?', '''
import math

positions = [-3, -1, 1, 3]
p, q = [0.4, 0.1, 0.1, 0.4], [0.1, 0.4, 0.4, 0.1]
difference = [a - b for a, b in zip(p, q)]
prefix, running = [], 0
scores = [0]
for i in range(len(positions) - 1):
    running += difference[i]
    prefix.append(running)
    sign = (running > 0) - (running < 0)
    scores.append(scores[-1] - sign * (positions[i + 1] - positions[i]))
w1 = math.fsum(abs(amount) * (positions[i + 1] - positions[i]) for i, amount in enumerate(prefix))
critic_gap = math.fsum(score * mass for score, mass in zip(scores, difference))
linear = abs(math.fsum(position * mass for position, mass in zip(positions, difference)))
tv = math.fsum(abs(mass) for mass in difference) / 2
assert math.isclose(w1, critic_gap)
assert all(abs(scores[j] - scores[i]) <= abs(positions[j] - positions[i]) + 1e-12
           for i in range(4) for j in range(4))
print("Cumulative imbalances:", [round(value, 6) for value in prefix])
print("Optimal critic scores:", scores)
print("Linear / TV / W1:", round(linear, 6), round(tv, 6), round(w1, 6))
print("Move delta0 to delta0.01: W1=0.01, TV=1, JS=ln(2), KL=infinity")
''')

add('kernel', 'Calculate kernel pair similarities and the witness',
    'Detect a variance change despite equal means, then inspect a valid negative unbiased squared estimate.', MMD + '''
x, y = [-2, -2, 2, 2], [-1, -1, 1, 1]
print("Raw means:", sum(x) / len(x), sum(y) / len(y))
for bandwidth in [0.5, 1, 3]:
    biased, unbiased = mmd_squared(x, y, bandwidth)
    print("Bandwidth:", bandwidth, "biased / unbiased squared:", round(biased, 6), round(unbiased, 6))
def witness(t):
    return sum(math.exp(-0.5 * (a - t)**2) for a in x) / len(x) - sum(math.exp(-0.5 * (b - t)**2) for b in y) / len(y)
gap = sum(witness(t) for t in x) / len(x) - sum(witness(t) for t in y) / len(y)
print("Witness expectation gap:", round(gap, 6))
print("Identical empirical laws:", [round(v, 6) for v in mmd_squared([-1, 1], [-1, 1])])
print("Explicit phi=(x,x^2): mean phi for +/-1 is (0,1); for 0 it is (0,0). MMD=1.")
''')

add('permutation', 'Enumerate the entire finite permutation reference',
    'How unusual is the observed split among all 70 label allocations? If selecting among bandwidths is part of the statistic, repeat that selection for every allocation.', MMD + '''
from itertools import combinations

pool = [-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2]
allocations = list(combinations(range(8), 4))
for bandwidths in [(1,), (0.3, 1, 3)]:
    statistics = []
    for allocation in allocations:
        x = [pool[i] for i in allocation]
        y = [pool[i] for i in range(8) if i not in allocation]
        # The complete rule, including maximization, is repeated for every split.
        statistic = max(mmd_squared(x, y, bandwidth)[0] for bandwidth in bandwidths)
        statistics.append(statistic)
    observed = statistics[0]
    tail = sum(value >= observed - 1e-12 for value in statistics)
    print("Bandwidth candidates:", bandwidths)
    print("Observed statistic:", round(observed, 6))
    print("Tail / allocations:", tail, len(statistics))
    print("Exact permutation p-value:", round(tail / len(statistics), 6))
''')

add('variational', 'Measure the gap between a critic and exact KL',
    'Change a known optimal critic while holding both laws fixed. Then verify the optimal discriminator/JS identity without training a neural network.', LAWS + '''
p, q = [0.7, 0.2, 0.1], [0.4, 0.5, 0.1]
truth = kl(p, q)
for scale, offset in [(0, 0), (1, 0), (1, 0.5), (0.5, -1)]:
    scores = [1 + scale * math.log(a / b) + offset for a, b in zip(p, q)]
    bound = math.fsum(a * t - b * math.exp(t - 1) for a, b, t in zip(p, q, scores))
    print("Scale / offset:", scale, offset, "bound / gap:", round(bound, 6), round(truth - bound, 6))
    assert bound <= truth + 1e-12
discriminator = [a / (a + b) for a, b in zip(p, q)]
objective = math.fsum(a * math.log(d) + b * math.log(1 - d) for a, b, d in zip(p, q, discriminator))
reference = -math.log(4) + 2 * comparisons(p, q)["JS"]
print("Optimal discriminator:", [round(d, 6) for d in discriminator])
print("GAN log objective / -ln4+2JS:", round(objective, 6), round(reference, 6))
assert math.isclose(objective, reference, abs_tol=1e-12)
''')

add('weights', 'Use chi-square to diagnose an importance proposal',
    'A proposal underrepresents a rare outcome. What happens to its likelihood-ratio weights and their variance?', LAWS + '''
p = [0.9, 0.1]
for q in ([0.9, 0.1], [0.99, 0.01], [0.999, 0.001]):
    weights = [a / b for a, b in zip(p, q)]
    mean = math.fsum(b * w for b, w in zip(q, weights))
    variance = math.fsum(b * (w - mean)**2 for b, w in zip(q, weights))
    print("Proposal:", q, "weights:", [round(w, 6) for w in weights])
    print("Mean / variance / chi-squared:", round(mean, 6), round(variance, 6), round(comparisons(p, q)["chi squared"], 6))
    assert math.isclose(variance, comparisons(p, q)["chi squared"], abs_tol=1e-12)
q = [0.5, 0.3, 0.2]
direction = [1, -1, 0]
quadratic = sum(v*v / mass for v, mass in zip(direction, q))
for epsilon in [0.02, 0.01, 0.001]:
    nearby = [mass + epsilon * v for mass, v in zip(q, direction)]
    ratio = kl(nearby, q) / (epsilon**2 * quadratic / 2)
    print("Local KL / half chi-square at epsilon:", epsilon, round(ratio, 6))
''')

add('stress', 'Stress-test a score against a missing rare outcome',
    'A model omits a 1% event entirely. Which comparisons remain small, and how often does a sample of 20 miss that event even under the true law?', LAWS + '''
p, q = [0.99, 0.01], [1, 0]
values = comparisons(p, q)
print("Missing rare event:", {name: round(value, 6) for name, value in values.items()})
print("Chance 20 true-law observations contain no rare event:", round(0.99**20, 6))
loss = [0, 100]
risk_p = sum(mass * cost for mass, cost in zip(p, loss))
risk_q = sum(mass * cost for mass, cost in zip(q, loss))
print("Expected cost P / Q:", risk_p, risk_q)
print("Bound from loss range times TV:", round(100 * values["TV"], 6))
print("A small bounded discrepancy need not mean small application-specific harm.")
''')

add('project', 'A complete changed-sample permutation comparison',
    'Build and check a small reusable exact test. What resolution does a three-versus-three sample permit?', MMD + '''
from itertools import combinations


def exact_test(x, y, bandwidths=(1,)):
    if min(len(x), len(y)) < 2 or len(x) + len(y) > 10:
        raise ValueError("Use at least two per group and at most ten total for enumeration.")
    if not bandwidths:
        raise ValueError("Specify the complete bandwidth selection rule.")
    pool, n = list(x) + list(y), len(x)
    statistics = []
    for allocation in combinations(range(len(pool)), n):
        group_x = [pool[i] for i in allocation]
        group_y = [pool[i] for i in range(len(pool)) if i not in allocation]
        statistics.append(max(mmd_squared(group_x, group_y, sigma)[0] for sigma in bandwidths))
    observed = statistics[0]
    tail = sum(value >= observed - 1e-12 for value in statistics)
    return observed, tail, len(statistics), tail / len(statistics)


x, y = [-1, -0.5, 0], [0.5, 1, 1.5]
score, tail, count, p_value = exact_test(x, y, (0.75,))
print("Statistic:", round(score, 6))
print("Tail / allocations / p-value:", tail, count, round(p_value, 6))
print("Reject at level 0.05:", p_value <= 0.05)
assert count == 20 and tail == 2
assert exact_test(y, x, (0.75,))[-1] == p_value
assert exact_test(x, y, (0.75,)) == exact_test(x, y, (0.75,))
print("Swap invariance and deterministic rerun: passed")
''')

destination = ROOT / 'src/learn/data/divergence-ipm-examples.js'
destination.write_text('export const divergenceIpmExamples = ' + json.dumps(examples, indent=2, ensure_ascii=False) + ';\n', encoding='utf-8')
print(f'Wrote {len(examples)} executed programs.')
