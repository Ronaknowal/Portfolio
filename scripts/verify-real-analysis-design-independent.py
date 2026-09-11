"""Bounded review of actual draft helpers; not an author/browser certificate."""
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
DIRECTORY = ROOT / 'scratch/real-analysis-design-independent'
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

tiny = data['tinyBound']
observations = [{
    'kind': 'numerical-range-qualification',
    'call': 'bernsteinApproximation(256, Number.MIN_VALUE, 0.3, 1)',
    'pointwiseBoundReturned': tiny['pointwiseBound'],
    'mathematicalBoundStrictlyPositive': True,
    'scope': 'Outside ordinary slider increments; the interior x*(1-x)/n underflows before sqrt. Reorder sqrt or restrict supported arithmetic range; do not display this as an exact zero certificate.'
}]
files = ['docs/teaching/REAL-ANALYSIS-LESSON-DESIGN.md', 'src/learn/data/curriculum/blueprints/real-analysis-sequences-modes-of-convergence.js', 'src/learn/data/real-analysis-models.js', 'src/learn/data/real-analysis-examples.js']
result = {'completedAt': datetime.now(timezone.utc).isoformat(), 'scope': 'Bounded design/draft-model/native review; no body/browser/freeze claim', 'checks': checks, 'observations': observations,
          'sourceHashes': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}}
(DIRECTORY / 'results.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2))
