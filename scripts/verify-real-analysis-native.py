"""Author numerical checks, extending the preserved independent design fixtures."""
import contextlib
import hashlib
import io
import json
import math
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'scratch/real-analysis-verification'
data = json.loads((DIRECTORY / 'payload.json').read_text(encoding='utf-8'))
checks = {}
def check(group, condition):
    assert condition, group
    checks[group] = checks.get(group, 0) + 1

native = {}
for example in data['examples']:
    namespace = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(example['code'], example['id'] + '.py', 'exec'), namespace)
    check('actual-program-stdout', output.getvalue().rstrip() == example['expected'])
    native[example['id']] = namespace

for row in data['brackets']:
    left, right, target = F(row['left']), F(row['right']), F(row['target'])
    check('exact-bracket-invariant', left * left <= target <= right * right)
    check('exact-bracket-width', right - left == F(1, 2 ** row['steps']))
    actual = native['exact-bracket']['square_root_bracket'](target, row['steps'], 2 if target == 5 else 1, 3 if target == 5 else 2)
    check('actual-native-bracket', actual == (left, right))

for row in data['tails']:
    epsilon = F(row['p'], row['q'])
    reference = next(index for index in range(1, 2001) if F(1, index + 1) < epsilon)
    check('enumerated-strict-tail', reference == row['state']['minimumIndex'])
    check('actual-strict-tail', reference == native['tail-certificate']['tail_start'](epsilon))
    check('chosen-tail-decision', (F(1, row['index'] + 1) < epsilon) == row['state']['certifiesTail'])

for row in data['triangles']:
    n, height = row['n'], F(row['height'])
    # Use exact intended height for the reciprocal family, independent of its rounded JS display.
    if row['scaling'] == 'shrinking-height': height = F(1, n)
    triangle = native['triangle-grid']['triangle']
    integral = square_integral = F(0)
    for a, b in [(F(0), F(1, n)), (F(1, n), F(2, n))]:
        mid = (a + b) / 2
        values = [triangle(t, n, height) for t in [a, mid, b]]
        # Simpson is exact for each polynomial of degree <= 3, including each squared linear side.
        integral += (b - a) * (values[0] + 4 * values[1] + values[2]) / 6
        square_integral += (b - a) * (values[0] ** 2 + 4 * values[1] ** 2 + values[2] ** 2) / 6
    check('piecewise-exact-triangle-integral', math.isclose(float(integral), row['integral'], rel_tol=1e-14))
    check('piecewise-exact-triangle-square', math.isclose(float(square_integral), row['squaredL2Error'], rel_tol=1e-14))

for row in data['bernstein']:
    n, x, corner, slope = row['n'], F(row['x']), F(7, 16), F(3)
    weights = [F(math.comb(n, k)) * x ** k * (1 - x) ** (n - k) for k in range(n + 1)]
    exact = sum(weight * slope * abs(F(k, n) - corner) for k, weight in enumerate(weights))
    check('bernstein-changed-value', math.isclose(float(exact), row['approximation'], rel_tol=4e-14, abs_tol=1e-15))
    check('bernstein-first-moment', sum(w * F(k, n) for k, w in enumerate(weights)) == x)
    check('bernstein-quadratic-reproduction', sum(w * F(k, n) ** 2 for k, w in enumerate(weights)) == x ** 2 + x * (1 - x) / n)
    check('bernstein-lipschitz-certificate', (exact - slope * abs(x - corner)) ** 2 <= slope ** 2 * x * (1 - x) / n)

t = sp.Symbol('t')
report = native['approximation-report']['approximation_report']
for n in [2, 5, 9]:
    for corner in [F(1, 7), F(4, 9)]:
        polynomial = sum(sp.binomial(n, k) * t ** k * (1 - t) ** (n - k) * sp.Rational(abs(F(k, n) - corner)) for k in range(n + 1))
        # Integrate expanded coefficients directly, not the equal-basis-integral identity used by the report.
        coefficients = sp.Poly(sp.expand(polynomial), t)
        integral = sum(c / (power[0] + 1) for power, c in coefficients.terms())
        actual = report(n, corner, 1, F(1, 8))
        check('expanded-polynomial-integral', F(integral) == actual[0])
        check('report-exact-error', abs(F(integral) - (corner ** 2 + (1 - corner) ** 2) / 2) == actual[2])

visit = native['typewriter']['visit']
for block in range(1, 6):
    count = 2 ** block
    for j in range(count):
        left, midpoint, right = F(j, count), F(2 * j + 1, 2 * count), F(j + 1, count)
        check('dyadic-left-boundary', visit(block, j, left))
        check('dyadic-midpoint', visit(block, j, midpoint))
        if right < 1: check('dyadic-right-excluded', not visit(block, j, right))

import mpmath as mp
mp.mp.dps = 100
for row in data['cauchy']:
    n = row['n']
    exact = sum(F(1, k if row['family'] == 'harmonic' else k * (k + 1)) for k in range(n + 1, 2 * n + 1))
    check('cauchy-block-exact-sum', math.isclose(float(exact), row['blockTotal'], rel_tol=2e-14))
    if row['family'] == 'harmonic': check('harmonic-block-lower-bound', exact >= F(1, 2))
    else: check('telescoping-exact-block', exact == F(1, n + 1) - F(1, 2 * n + 1))

for row in data['powers']:
    n, x = row['n'], F(row['fixedPoint'])
    limit = int(x == 1 and row['domain'] == 'closed-unit')
    check('power-fixed-point', math.isclose(float(x ** n), row['fixedValue'], rel_tol=2e-14, abs_tol=1e-100))
    check('power-limit', row['limitValue'] == limit)
    bound = F(row['end']) ** n if row['domain'] == 'compact-subinterval' else F(1)
    check('power-domain-supremum', math.isclose(float(bound), row['supremumError'], rel_tol=2e-14))
    witness = mp.power(2, -mp.mpf(1) / n)
    check('power-moving-witness', abs(witness ** n - mp.mpf('0.5')) < mp.mpf('1e-90'))

for row in data['derivatives']:
    n, power, x = row['n'], row['amplitudePower'], mp.mpf(row['point'])
    function = lambda t: mp.sin(n * t) / n ** power
    derivative = mp.diff(function, x)
    check('derivative-independent-differentiation', math.isclose(float(derivative), row['derivative'], rel_tol=3e-13, abs_tol=3e-14))
    check('function-high-precision', math.isclose(float(function(x)), row['value'], rel_tol=3e-13, abs_tol=3e-14))
    actual = native['derivative-interchange']['value_and_derivative'](n, power, float(x))
    check('actual-native-derivative', math.isclose(actual[1], float(derivative), rel_tol=3e-13, abs_tol=3e-14))

for row in data['series']:
    n, x = row['n'], F(row['x'])
    exact = native['series-endpoint']['inverse_square_sums'](n, x)
    check('series-native-exact-function', math.isclose(float(exact[0]), row['sum'], rel_tol=2e-14, abs_tol=1e-15))
    check('series-native-exact-derivative', math.isclose(float(exact[1]), row['derivativeSum'], rel_tol=2e-14, abs_tol=1e-15))
    check('series-uniform-tail', row['uniformTailBound'] == 1 / n)

for row in data['typewriter']:
    block, position, observer = row['block'], row['position'], F(row['observer'])
    hit = F(position, 2 ** block) <= observer < F(position + 1, 2 ** block)
    check('typewriter-shared-observer', row['observed'] == int(hit))
    check('typewriter-global-index', row['n'] == 2 ** block + position)
    check('typewriter-probability', F(row['probability']) == F(1, 2 ** block))
    check('typewriter-actual-native', visit(block, position, observer) == hit)

for args in [(-1, 0, F(1, 2)), (1.5, 0, F(1, 2)), ('2', 0, F(1, 2)), (2, 0.5, F(1, 2)), (2, 4, F(1, 2)), (2, 0, 1)]:
    try:
        visit(*args)
        raise AssertionError('Native typewriter accepted invalid input')
    except ValueError:
        check('native-typewriter-invalid', True)

tiny = data['tinyBound']
reference = mp.sqrt(mp.mpf(float.fromhex('0x0.0000000000001p-1022'))) / 16
check('positive-bound-underflow-repaired', tiny['pointwiseBound'] > 0 and abs(mp.mpf(tiny['pointwiseBound']) / reference - 1) < mp.mpf('1e-15'))

# Changed local practice: derive strict boundaries and exact capstone answers.
for tolerance, first in [(F(1, 7), 11), (F(1, 20), 30)]:
    check('changed-strict-tail', F(3, 2 * first + 1) < tolerance <= F(3, 2 * (first - 1) + 1))
check('changed-geometric-boundary', F(1, 3 * 4 ** 5) < F(1, 1000) < F(1, 3 * 4 ** 4))
check('changed-alternating-sufficient', F(1, 11 ** 3) < F(1, 1000) == F(1, 10 ** 3))
check('capstone-original-25', report(25, F(3, 10), 1, F(1, 20)) == (F(97, 325), F(29, 100), F(11, 1300), False))
check('capstone-original-100', report(100, F(3, 10), 1, F(1, 20)) == (F(59, 202), F(29, 100), F(21, 10100), True))
check('capstone-changed', report(100, F(2, 5), 2, F(1, 10)) == (F(53, 101), F(13, 25), F(12, 2525), True))

files = ['src/learn/data/topics/real-analysis-sequences-modes-of-convergence.jsx', 'src/learn/data/curriculum/blueprints/real-analysis-sequences-modes-of-convergence.js', 'src/learn/data/real-analysis-models.js', 'src/learn/data/real-analysis-examples.js', 'src/learn/components/lesson-labs/RealAnalysisLabs.jsx', 'src/learn/components/lesson-labs/real-analysis-labs.css']
result = {'completedAt': datetime.now(timezone.utc).isoformat(), 'scope': 'Author verification of actual published programs and bounded models; independent design fixtures retained and extended, separate final independent review pending', 'checks': checks, 'invalidInputsRejected': data['invalidInputsRejected'], 'programs': len(data['examples']), 'python': __import__('sys').version, 'mpmath': mp.__version__, 'sympy': sp.__version__, 'sourceHashes': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}}
(DIRECTORY / 'results.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2))

