from pathlib import Path
from fractions import Fraction
from decimal import Decimal, localcontext
from datetime import datetime, timezone
import contextlib
import io
import json
import math

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'scratch/algebra-functions-verification'
data = json.loads((DIRECTORY / 'model-cases.json').read_text())
counts = {}

def close(actual, expected, tolerance=2e-12):
    assert math.isclose(actual, float(expected), rel_tol=tolerance, abs_tol=tolerance), (actual, expected)

for case in data['equations']:
    a, b, c = [Fraction(case[key]) for key in ('a', 'b', 'c')]
    if not a:
        assert case['kind'] == ('all' if b == c else 'none') and case['solution'] is None
    else:
        expected = (c-b)/a
        close(case['solution'], expected)
        assert a*expected+b == c and case['kind'] == 'one'
counts['exactEquationCases'] = len(data['equations'])
for case in data['functions']:
    x = Fraction(case['x'])
    allowed = not ((case['kind'] == 'reciprocal' and x == 0) or (case['kind'] == 'square' and case['restricted'] and x < 0))
    assert case['allowed'] == allowed
    if not allowed:
        assert case['y'] is None and case['preimages'] == []
        continue
    expected = {'affine': lambda: 2*x+1, 'square': lambda: x*x, 'reciprocal': lambda: 1/x}[case['kind']]()
    close(case['y'], expected)
    expected_preimages = sorted({-abs(x), abs(x)}) if case['kind'] == 'square' and not case['restricted'] else [x]
    assert [Fraction(value) for value in case['preimages']] == expected_preimages
counts['functionDomainCases'] = len(data['functions'])
for case in data['compositions']:
    x = Fraction(case['x'])
    close(case['squareAfterAffine'], 4*x*x+4*x+1)
    close(case['affineAfterSquare'], 2*x*x+1)
    close(case['recovered'], x)
counts['expandedCompositionCases'] = len(data['compositions'])
for case in data['quadratics']:
    h, k = Fraction(case['h']), Fraction(case['k'])
    b, c = -2*h, h*h+k
    discriminant = b*b-4*c
    close(case['b'], b); close(case['c'], c); close(case['discriminant'], discriminant)
    expected = [] if discriminant < 0 else [-float(b)/2] if discriminant == 0 else [(-float(b)-math.sqrt(discriminant))/2, (-float(b)+math.sqrt(discriminant))/2]
    assert len(case['roots']) == len(expected)
    for actual, reference in zip(case['roots'], expected):
        close(actual, reference)
        close(actual*actual+float(b)*actual+float(c), 0)
counts['quadraticDiscriminantCases'] = len(data['quadratics'])
with localcontext() as context:
    context.prec = 65
    for case in data['growth']:
        r = Decimal(str(case['rate']))
        value = Decimal(100)
        for row in case['rows']:
            close(row['growth'], value)
            close(row['additive'], 100+20*row['t'])
            if row['change'] is not None:
                close(row['change'], value - value/(1+r))
            value *= 1+r
        factor = Decimal(str(case['factor']))
        if r == 0:
            assert case['crossing'] == (0 if factor == 1 else None)
        else:
            close(case['crossing'], factor.ln()/(1+r).ln())
        assert case['active'] == case['rows'][case['step']]
    for case in data['logs']:
        reference = (Decimal(str(case['base'])).ln()*Decimal(str(case['exponent']))).exp()
        close(case['value'], reference)
counts['growthDecimalCases'] = len(data['growth'])
counts['logDecimalCases'] = len(data['logs'])
namespaces = {}
for example in data['examples']:
    namespace = {'__name__': '__main__'}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(example['code'], example['id']+'.py', 'exec'), namespace)
    assert output.getvalue().rstrip() == example['expected'], example['id']
    namespaces[example['id']] = namespace
counts['actualNativePrograms'] = len(namespaces)
linear = namespaces['equations']['solve_linear']
for numerator in range(-9, 10):
    a, b, c = Fraction(numerator, 3), Fraction(5, 7), Fraction(2, 9)
    answer = linear(a, b, c)
    if a:
        assert a*answer+b == c
    else:
        assert answer == 'no solution'
counts['actualChangedFractionSolves'] = 19
root = namespaces['roots']['vertex_roots']
for h in (-10, -1, 0, 8):
    for k in (-64, -7, 0, 9):
        result = root(h, k)
        assert len(result) == (0 if k > 0 else 1 if k == 0 else 2)
        for x in result:
            close(x*x-2*h*x+h*h+k, 0)
counts['actualChangedRootCases'] = 16
doubling = namespaces['doubling']['first_doubling']
for initial in (1, 3, 7):
    for target in list(range(1, 201)) + [2**200, 2**200+1]:
        steps, value = doubling(initial, target)
        reference = next(n for n in range(205) if initial*2**n >= target)
        assert steps == reference and value == initial*2**reference
counts['actualIntegerThresholdCases'] = 606
for invalid in (0, -1, True, 1.5):
    try:
        doubling(invalid, 10)
        raise AssertionError('Expected invalid integer rejection')
    except ValueError:
        pass
counts['modelRejectionCases'] = data['rejectionCases']
practice = {}
practice['mixture'] = str(Fraction(2,3)+Fraction(5,6))
practice['rate'] = str((Fraction(2,3)+Fraction(5,6))/3)
practice['grouping'] = -(2**2)+3*(4-6)
practice['wholePurchases'] = [n for n in range(20) if 7+4*n <= 30]
assert practice['mixture'] == '3/2' and practice['rate'] == '1/2' and practice['grouping'] == -10
assert practice['wholePurchases'] == list(range(6))
for x in (Fraction(n,4) for n in range(-32,33)):
    assert (-3*x+5 < 14) == (x > -3)
    assert (x*x+4*x-5 <= 0) == (-5 <= x <= 1)
    assert x*x+4*x-5 == (x+2)**2-9
    if x >= 2:
        output = math.sqrt(float(3*x-6))
        close((output*output+6)/3,x)
practice['percentageUndo'] = str(Fraction(5,4)*Fraction(4,5))
practice['undoFortyPercent'] = str(1-Fraction(5,7))
assert practice['percentageUndo'] == '1' and practice['undoFortyPercent'] == '2/7'
close(math.log(math.sqrt(13)-2,3)+math.log(math.sqrt(13)+2,3),2)
practice['logCandidate'] = math.sqrt(13)
practice['decayCrossing'] = math.log(.25)/math.log(.75)
practice['decayNeighbors'] = [str(Fraction(80)*Fraction(3,4)**n) for n in (4,5)]
assert Fraction(80)*Fraction(3,4)**4 > 20 >= Fraction(80)*Fraction(3,4)**5
practice['logAxisMidpoint'] = math.sqrt(10*1000)
assert practice['logAxisMidpoint'] == 100
practice['changedCapacity'] = doubling(3,100)
practice['sixDelays'] = sum(2*2**n for n in range(6))
assert practice['changedCapacity'] == (6,192) and practice['sixDelays'] == 126
practice['forecastDayThree'] = [95,str(Fraction(50)*Fraction(13,10)**3)]
practice['growthThresholdNeighbors'] = [str(Fraction(50)*Fraction(13,10)**n) for n in (5,6)]
assert Fraction(50)*Fraction(13,10)**5 < 200 <= Fraction(50)*Fraction(13,10)**6
close(math.log(4)/math.log(1.3),5.28385359162228)
counts['independentPracticeChecks'] = 12
result = {'checkedAt': datetime.now(timezone.utc).isoformat(), 'status': 'passed', 'counts': counts, 'practiceCalculations': practice, 'limits': 'Finite independent arithmetic/domain checks do not prove continuum function or exponential-limit theorems. No learner study or integration claim.'}
(DIRECTORY / 'results.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
