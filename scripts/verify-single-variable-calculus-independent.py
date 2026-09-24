"""Complementary exact/coordinate-transformed checks of the installed lesson."""
import contextlib
import io
import json
import math
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path
import mpmath as mp

mp.mp.dps = 90
directory = Path('scratch/single-variable-calculus-independent-review')
data = json.loads((directory / 'fixtures.json').read_text(encoding='utf-8'))
counts = Counter()
contexts = {}
largest = 0.0
largest_relative = 0.0
largest_case = None

def close(actual, expected, tolerance=2e-12):
    global largest, largest_relative, largest_case
    expected = float(expected)
    assert math.isclose(actual, expected, abs_tol=tolerance, rel_tol=tolerance), (actual, expected)
    difference = abs(actual - expected)
    if difference > largest:
        largest = difference
        largest_case = {'actual': actual, 'expected': expected, 'relativeDifference': difference/abs(expected) if expected else None}
    if expected:
        largest_relative = max(largest_relative, difference/abs(expected))

for example in data['examples']:
    context = {}
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(compile(example['code'], example['id'], 'exec'), context)
    assert output.getvalue().rstrip() == example['expected'], example['id']
    contexts[example['id']] = context
    counts['actualDisplayedPrograms'] += 1

for state in data['limits']:
    delta, epsilon = F(str(state['delta'])), F(str(state['epsilon']))
    jump = 2 if state['kind'] == 'jump' else 0
    # Compare the positive branch endpoint exactly, and the negative branch
    # error independently. Open endpoints make equality a successful guarantee.
    right_error = (2 + delta)**2 - 4 + jump
    left_error = 4 - (2 - delta)**2
    assert right_error >= left_error
    assert state['guaranteed'] == (right_error <= epsilon)
    if not state['guaranteed']:
        witness = state['witness']
        x = F(witness['inputExact'])
        assert 2 < x < 2 + delta
        assert F(witness['displacementExact']) == x - 2
        assert F(witness['errorExact']) == x*x - 4 + jump >= epsilon
        counts['strictInteriorRationalWitnesses'] += 1
    counts['adjacentThresholdLimitStates'] += 1

def function(kind, x):
    if kind == 'motion':
        return x**3 - 6*x*x + 9*x
    if kind == 'cusp':
        return abs(x-2)
    return (x-2)**3

for state in data['extrema']:
    kind = state['kind']
    values = [function(kind, F(str(candidate['input']))) for candidate in state['candidates']]
    for candidate, value in zip(state['candidates'], values):
        assert candidate['isMinimum'] == (value == min(values)), (kind, state['left'], state['right'], 'minimum', candidate)
        assert candidate['isMaximum'] == (value == max(values)), (kind, state['left'], state['right'], 'maximum', candidate)
    for interval in state['intervals']:
        a, b = F(str(interval['start'])), F(str(interval['end']))
        # Two distinct rational interior values establish the sign for these
        # already critical-partitioned polynomial/absolute branches. No float midpoint.
        difference = function(kind, (a+3*b)/4) - function(kind, (3*a+b)/4)
        sign = (difference > 0) - (difference < 0)
        assert interval['sign'] == sign, state
    counts['adjacentFloatAndChangedExtremaIntervals'] += 1

for state in data['accumulations']:
    upper, n = F(str(state['upper'])), state['panelCount']
    width = upper/n
    offset = {'left': F(0), 'midpoint': F(1, 2), 'right': F(1)}[state['method']]
    # Closed sums of powers in j supply the signed total without summing panels.
    sum1 = F(n*(n-1), 2)
    sum2 = F(n*(n-1)*(2*n-1), 6)
    signed = width*(3*width*width*(sum2+2*offset*sum1+n*offset*offset)-12*width*(sum1+n*offset)+9*n)
    close(state['signedEstimate'], signed)
    U = mp.mpf(str(state['upper']))
    cuts = [mp.mpf(0)] + [mp.mpf(x) for x in [1, 3] if x < U] + [U]
    distance = sum(mp.quad(lambda t: abs(3*t*t-12*t+9), [a, b]) for a, b in zip(cuts, cuts[1:]))
    close(state['exactDistance'], distance)
    counts['closedPowerSumAndAbsoluteQuadratureCases'] += 1

roundoff_dominates = []
for state in data['taylors']:
    x, n = mp.mpf(str(state['input'])), state['degree']
    exponential = state['kind'] == 'exp'
    f = mp.exp if exponential else lambda t: mp.log1p(t)
    if exponential:
        polynomial = sum(x**j/mp.factorial(j) for j in range(n+1))
        derivative = mp.exp
    else:
        polynomial = sum((-1)**(j+1)*x**j/j for j in range(1, n+1))
        derivative = lambda t: (-1)**n * mp.factorial(n)/(1+t)**(n+1)
    # Fixed [0,1] integral independently handles x<0 orientation and powers.
    integral_remainder = x**(n+1)/mp.factorial(n)*mp.quad(lambda u: (1-u)**n*derivative(x*u), [0, 1])
    assert abs(f(x)-polynomial-integral_remainder) < mp.mpf('1e-70')
    bound = mp.exp(max(0, x))*abs(x)**(n+1)/mp.factorial(n+1) if exponential else abs(x)**(n+1)/((n+1)*min(1, 1+x)**(n+1))
    assert abs(integral_remainder) <= bound*(1+mp.mpf('1e-70'))
    close(state['remainderBound'], bound)
    close(state['polynomial'], polynomial)
    if abs(state['signedError']) > state['remainderBound']:
        roundoff_dominates.append({'kind': state['kind'], 'degree': n, 'input': state['input'], 'computedDifference': state['signedError'], 'analyticRemainder': str(integral_remainder), 'analyticBound': str(bound)})
    counts['fixedSegmentIntegralRemainderCases'] += 1

for state in data['improper']:
    # This helper uses Number-valued parameters (unlike the explicitly decimal
    # epsilon/extrema contract). Preserve the exact binary input near p=1.
    p, q = mp.mpf(state['power']), mp.mpf(state['cutoffExponent'])
    extent = q*mp.log(10)
    sign = 1 if state['kind'] == 'tail' else -1
    # x=exp(±u) moves the finite integral to a regular u segment near p=1.
    integral = mp.quad(lambda u: mp.exp(sign*(1-p)*u), [0, extent])
    close(state['truncated'], integral)
    if state['converges']:
        residual = mp.exp(-abs(1-p)*extent)/abs(1-p)
        close(state['missing'], residual)
    counts['logCoordinateImproperQuadratures'] += 1

for state in data['growth']:
    k, duration = mp.mpf(str(state['rate'])), mp.mpf(str(state['period']))
    accumulated_rate = mp.quad(lambda t: k*10*mp.exp(k*t), [2, 2+duration])
    close(state['intervalAverageRate'], accumulated_rate/duration)
    close(state['futureAmount']-state['currentAmount'], accumulated_rate)
    counts['integratedGrowthRateChecks'] += 1

for numerator in range(-6, 7):
    x = F(numerator, 5)
    primitive = contexts['moving-bounds']['accumulated']
    expected = (x*x + x**6/3) - (x + x**3/3)
    assert primitive(x) == expected
    changed = contexts['changed-motion']['position'](x)
    assert changed == x*x*(x-3)
    counts['changedActualNativeHelperChecks'] += 2

report = {'checkedAt': datetime.now(timezone.utc).isoformat(), 'status': 'passed', 'counts': dict(counts),
          'largestAbsoluteFloatDifference': largest, 'largestDifferenceCase': largest_case,
          'largestRelativeFloatDifference': largest_relative, 'precision': mp.mp.dps,
          'roundoffDominatedTaylorCases': roundoff_dominates,
          'modelSource': data['modelSource'],
          'scope': 'Complementary finite and exact checks; general proof and visible numeric-label qualifications reviewed separately.'}
(directory / 'results.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
print(json.dumps({key: value for key, value in report.items() if key != 'roundoffDominatedTaylorCases'}, indent=2))
