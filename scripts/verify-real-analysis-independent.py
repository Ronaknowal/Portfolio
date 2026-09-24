"""Complementary final-review oracles, independent of the prior design suite."""
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path
import contextlib
import io
import json
import math
import mpmath as mp

mp.mp.dps = 70
directory = Path('scratch/real-analysis-independent')
data = json.loads((directory / 'payload.json').read_text(encoding='utf-8'))
checks = Counter()


def check(name, condition):
    assert condition, name
    checks[name] += 1


def close(actual, reference):
    return math.isclose(actual, float(reference), rel_tol=8e-13, abs_tol=3e-15)


native = {}
for example in data['examples']:
    output, namespace = io.StringIO(), {}
    with contextlib.redirect_stdout(output):
        exec(compile(example['code'], example['id'], 'exec'), namespace)
    check('actual-program-stdout', output.getvalue().rstrip() == example['expected'])
    native[example['id']] = namespace

for row in data['tails']:
    epsilon = F(row['p'], row['q'])
    n = row['minimumIndex']
    check('strict-tail-minimality', F(1, n + 1) < epsilon and (n == 1 or F(1, n) >= epsilon))
    check('changed-native-tail', native['tail-certificate']['tail_start'](epsilon) == n)
    check('proposed-tail-inequality', row['certifiesTail'] == (F(1, row['index'] + 1) < epsilon))

for row in data['brackets']:
    left, right = F(row['left']), F(row['right'])
    target, scale = row['target'], 2 ** row['steps']
    # An integer square root after clearing denominators determines the enclosure
    # without following the author's bisection decisions.
    floor_numerator = math.isqrt(target * scale * scale)
    check('integer-root-independent-enclosure', left == F(floor_numerator, scale) and right == F(floor_numerator + 1, scale))
    check('actual-bracket-call', native['exact-bracket']['square_root_bracket'](target, row['steps'], 2 if target == 5 else 1, 3 if target == 5 else 2) == (left, right))

for row in data['blocks']:
    n = row['n']
    # Digamma differences are independent of term summation for harmonic blocks.
    reference = mp.digamma(2 * n + 1) - mp.digamma(n + 1) if row['family'] == 'harmonic' else F(n, (n + 1) * (2 * n + 1))
    check('independent-block-value', close(row['blockTotal'], reference))
    check('actual-changed-block', close(float(native['cauchy-blocks']['block_gap'](n, row['family'])), reference))

for row in data['triangles']:
    n = row['n']
    height = F(n) if row['scaling'] == 'unit-area' else F(1, n) if row['scaling'] == 'shrinking-height' else F(1)
    # Exact degree-two Bernstein shape integration on each side, with variable
    # substitution already eliminated, also checks a first spatial moment.
    triangle = native['triangle-grid']['triangle']
    for moment in [0, 1]:
        total = F(0)
        for a, b in [(F(0), F(1, n)), (F(1, n), F(2, n))]:
            midpoint = (a + b) / 2
            total += (b - a) * sum(weight * x ** moment * triangle(x, n, height) for weight, x in [(1, a), (4, midpoint), (1, b)]) / 6
        check('triangle-mass-and-centroid', total == height / n ** (moment + 1))
    check('triangle-js-area', close(row['integral'], height / n))
    check('triangle-js-l2', close(row['squaredL2Error'], 2 * height ** 2 / (3 * n)))
    for sample in row['samples']:
        check('actual-triangle-samples', close(sample['y'], triangle(F(sample['x']), n, height)))

for row in data['series']:
    n, x = row['n'], F(row['x'])
    value, derivative = native['series-endpoint']['inverse_square_sums'](n, x)
    check('actual-series-finite-sums', close(row['sum'], value) and close(row['derivativeSum'], derivative))
    exact_limit = mp.polylog(2, mp.mpf(x.numerator) / x.denominator)
    check('polylog-function-tail', abs(mp.mpf(value.numerator) / value.denominator - exact_limit) <= mp.mpf(1) / n)
    if x < 1:
        derivative_limit = 1 if x == 0 else -mp.log1p(-mp.mpf(x.numerator) / x.denominator) / (mp.mpf(x.numerator) / x.denominator)
        bound = F(1, n + 1) if x == -1 else abs(x) ** n / ((n + 1) * (1 - abs(x)))
        check('separately-bounded-derivative-tail', abs(mp.mpf(derivative.numerator) / derivative.denominator - derivative_limit) <= mp.mpf(bound.numerator) / bound.denominator + mp.mpf('1e-60'))

for row in data['bernstein']:
    n, x, corner, slope = row['n'], F(row['x']), F(row['corner']), F(row['slope'])
    # de Casteljau's repeated affine interpolation evaluates the same polynomial
    # without building any binomial weights or using the production recurrence.
    layer = [slope * abs(F(k, n) - corner) for k in range(n + 1)]
    while len(layer) > 1:
        layer = [(1 - x) * a + x * b for a, b in zip(layer, layer[1:])]
    exact = layer[0]
    check('de-casteljau-value', close(row['approximation'], exact))
    check('convex-jensen-and-lipschitz', exact >= slope * abs(x - corner) and (exact - slope * abs(x - corner)) ** 2 <= slope ** 2 * x * (1 - x) / n)
    weights = native['bernstein-weights']['weights'](n, x)
    third_moment = sum(weight * (F(k, n) - x) ** 3 for k, weight in enumerate(weights))
    check('changed-native-third-moment', third_moment == x * (1 - x) * (1 - 2 * x) / n ** 2)
    check('every-production-weight', all(close(a['weight'], b) for a, b in zip(row['nodes'], weights)))

for row in data['writers']:
    observer, block, position = F(row['observer']), row['block'], row['position']
    # A bit shift gives the observer's integer prefix; the half-open convention
    # assigns dyadic boundaries to exactly that prefix.
    prefix = (observer.numerator << block) // observer.denominator
    check('dyadic-binary-prefix', row['observerVisitPosition'] == prefix and row['observed'] == int(position == prefix))
    check('actual-changed-visit', native['typewriter']['visit'](block, position, observer) == (position == prefix))

report = native['approximation-report']['approximation_report']
for n in [3, 7, 19, 53, 100]:
    for c in [F(0), F(2, 7), F(3, 5), F(1)]:
        slope, tolerance = F(7, 4), F(7, 40)
        result = report(n, c, slope, tolerance)
        j = (n * c.numerator) // c.denominator
        # Two arithmetic progressions sum |k/n-c| without enumerating nodes
        # or expanding/integrating the basis polynomials.
        left = (j + 1) * c - F(j * (j + 1), 2 * n)
        right = F(n * (n + 1) - j * (j + 1), 2 * n) - (n - j) * c
        integral = slope * (left + right) / (n + 1)
        exact_target = slope * (c ** 2 + (1 - c) ** 2) / 2
        check('closed-progressions-capstone', result == (integral, exact_target, abs(integral - exact_target), n >= 25))

for ratio in [F(-7, 8), F(-1, 5), F(0), F(3, 7)]:
    for n in [0, 3, 19]:
        partial, limit, bound = native['series-bounds']['geometric_partial'](ratio, n)
        check('changed-native-geometric', partial == (1 - ratio ** (n + 1)) / (1 - ratio) and abs(limit - partial) <= bound)

result = {
    'completedAt': datetime.now(timezone.utc).isoformat(), 'passed': True,
    'checks': dict(checks), 'sources': data['sources'],
    'scope': 'Independent final source review: actual13 programs, changed helpers, integer-square-root enclosure, digamma blocks, polylog/log derivative tails, de Casteljau polynomial evaluation, exact third moments, dyadic binary prefixes and closed arithmetic-progression integrals. Not the earlier design suite and not a proof by numerical enumeration.',
}
(directory / 'results.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2))
