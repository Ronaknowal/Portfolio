"""Complementary review: actual programs, moments, Lambert-W and retained signs.

This finite suite does not promote a floating-point estimate to a certificate.
"""
import contextlib
import hashlib
import io
import json
import math
import subprocess
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path

import scipy
from scipy.integrate import quad
from scipy.special import lambertw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'scratch/numerical-methods-independent-review'
subprocess.run(['node', 'scripts/verify-numerical-methods-independent.mjs'], cwd=ROOT, check=True)
fixtures = json.loads((OUT / 'fixtures.json').read_text(encoding='utf8'))
freeze = json.loads((ROOT / 'docs/teaching/evidence/numerical-methods-author-review.json').read_text(encoding='utf8'))
sources = freeze.get('production', freeze.get('sources'))
assert sources, freeze.keys()
for row in sources:
    assert hashlib.sha256((ROOT / row['path']).read_bytes()).hexdigest() == row['sha256']

comparisons = 0
maximum_absolute = 0.0
def close(actual, reference, atol=3e-12, rtol=3e-12):
    global comparisons, maximum_absolute
    assert math.isfinite(actual) and math.isfinite(reference)
    error = abs(actual - reference)
    assert error <= atol + rtol * abs(reference), (actual, reference, error)
    maximum_absolute = max(maximum_absolute, error)
    comparisons += 1

scopes = {}
for key, example in fixtures['examples'].items():
    scope, stream = {}, io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(example['code'], f'<reviewed-actual-{key}>', 'exec'), scope)
    assert stream.getvalue().rstrip() == example['expected'], key
    scopes[key] = scope
original = json.loads((ROOT / 'docs/teaching/evidence/numerical-methods-original-content.json').read_text(encoding='utf8'))
assert fixtures['examples']['original']['code'] == original['blocks'][0]['text']
assert fixtures['examples']['original']['expected'] == original['blocks'][1]['text']

# Richardson-corrected Simpson equals the integrated five-node interpolant
# (Boole's rule); degree<=5 therefore gives an independent exact moment oracle.
adaptive_leaves = 0
for case in fixtures['adaptive']:
    state = case['result']
    a, b = F(case['a']), F(case['b'])
    if case.get('blind'):
        reference = F(5, 1419264) * F(case['scale'])
        assert state['estimate'] == 0 and state['estimatedError'] == 0
        assert len(state['samples']) == 5 and reference != 0
        f = lambda x, scale=case['scale']: scale * (x*(x-.25)*(x-.5)*(x-.75)*(x-1))**2
        native = scopes['adaptive']['adaptive_simpson'](f, 0, 1, 1e-12)
        assert native[:2] == (0, 0)
        close(quad(f, 0, 1, epsabs=1e-14)[0], float(reference), atol=1e-14)
        continue
    degree, shift = case['degree'], F(case['shift'])
    exact = ((b-shift)**(degree+1)-(a-shift)**(degree+1))/(degree+1)
    close(state['estimate'], float(exact))
    f = lambda x, p=degree: (x-.125)**p
    native = scopes['adaptive']['adaptive_simpson'](f, float(a), float(b), 1e-7)
    close(native[0], float(exact))
    endpoints = [F(leaf['lower']) for leaf in state['leaves']] + [F(state['leaves'][-1]['upper'])]
    assert endpoints[0] == a and endpoints[-1] == b
    for i, leaf in enumerate(state['leaves']):
        left, right = F(leaf['lower']), F(leaf['upper'])
        assert right == endpoints[i+1]
        width = right-left
        values = [(left+F(j,4)*width-shift)**degree for j in range(5)]
        integrated = width * sum(w*y for w,y in zip([7,32,12,32,7], values)) / 90
        close(leaf['value'], float(integrated))
        adaptive_leaves += 1
    close(sum(leaf['budget'] for leaf in state['leaves']), 1e-7, atol=1e-20)
for case in fixtures['workFailures']:
    state = case['result']
    assert state['status'] == 'evaluation limit reached'
    assert state['estimate'] is None and state['estimatedError'] is None
    assert len(state['samples']) <= case['maximumEvaluations']

# The same dimensionless root is retained across sign/scaling changes.
for case in fixtures['brackets']:
    root = case['root']
    for row in case['result']['steps']:
        assert row['lower']-1e-14 <= root <= row['upper']+1e-14
        assert abs(row['midpoint']-root) <= row['radius']+1e-14
        close(row['width'], 2 / 2**row['iteration'])
    f = lambda x, scale=case['scale'], r=root: scale*math.expm1(x-r)
    native = scopes['bracket']['bisect'](f, root-.75, root+1.25, 1e-8)
    assert abs(native[0]-root) <= native[2]+1e-14
for case in fixtures['newton']:
    root = case['root']
    rows = case['result']['steps']
    for i, row in enumerate(rows):
        close(row['proposal'], row['current'] - (1 - math.exp(root-row['current'])))
        assert row['lower'] <= row['next'] <= row['upper']
        if i+1 < len(rows):
            following = rows[i+1]
            assert following['upper']-following['lower'] <= .9*(row['upper']-row['lower'])+1e-14
            assert following['lower']-1e-14 <= root <= following['upper']+1e-14

# Analytic trigonometric identities avoid the same subtractive sin stencil.
for case in fixtures['derivatives']:
    x, h, mode = case['x'], case['h'], case['stencil']
    if mode == 'central': expected = math.cos(x)*math.sin(h)/h
    elif mode == 'forward': expected = 2*math.sin(h/2)*math.cos(x+h/2)/h
    elif mode == 'second': expected = -4*math.sin(x)*math.sin(h/2)**2/h**2
    else:
        expected = (4*math.sin(h/2)*math.cos(x+h/2)-math.sin(h)*math.cos(x+h))/h
    close(case['result']['estimate'], expected, atol=2e-9)

# Analytic inverse uses the negative Lambert-W branch, not either bisection
# implementation. Irregular panel budgets exercise the final capped refinement.
def true_time(target):
    return float(-1-lambertw(-(1-target)/math.e, -1).real)
for case in fixtures['calibration']:
    state = case['result']
    exact = quad(lambda t:t*math.exp(-t), 0, case['candidate'], epsabs=1e-13)[0]
    assert abs(state['estimate']-exact) <= state['integralBound'] + 1e-14
    assert abs(case['candidate']-true_time(case['target'])) <= state['timeBound'] + 2e-13
    if state['sign'] == 'above target': assert exact > case['target']
    if state['sign'] == 'below target': assert exact < case['target']

search_cases = 0
successes = 0
unresolved = 0
for target in [.6, .625, .7, .775, .825, .88, .9]:
    root = true_time(target)
    for tolerance in [.1, .007, .0003]:
        for panels in [8, 9, 17, 33, 65, 129]:
            value, bound, status, trace = scopes['nestedCalibration']['locate_target'](target, tolerance, panels)
            search_cases += 1
            assert abs(value-root) <= bound+5e-13
            if 'tolerance' in status:
                assert bound <= tolerance
                successes += 1
            elif status == 'unresolved integral sign':
                unresolved += 1
                last = trace[-1]
                assert last[3] == panels and abs(last[4]-target) <= last[5]
            else: raise AssertionError(status)
            for lower, upper, candidate, n, estimate, error_bound in trace:
                assert lower-1e-13 <= root <= upper+1e-13
                assert 8 <= n <= panels
                exact = 1-(1+candidate)*math.exp(-candidate)
                assert abs(estimate-exact) <= error_bound+1e-14

# Boundary conditions and unit transfer of the two-node actual helper.
gauss_cases = 0
for a,b in [(-2,.5), (.5,2.25), (2,5)]:
    for degree in range(6):
        estimate = scopes['gaussian']['gauss_two'](lambda x,p=degree:x**p,a,b)
        if degree <= 3:
            close(estimate, (b**(degree+1)-a**(degree+1))/(degree+1))
        else:
            assert not math.isclose(estimate, (b**(degree+1)-a**(degree+1))/(degree+1), abs_tol=1e-8)
        gauss_cases += 1

result = dict(checkedAt=datetime.now(timezone.utc).isoformat(), passed=True,
    actualPrograms=len(scopes), originalCodeOutputPreserved=True,
    adaptivePolynomialCases=18, independentlyIntegratedLeaves=adaptive_leaves,
    blindScaleCases=3, workBudgetFailures=4,
    changedBrackets=len(fixtures['brackets']), changedSafeguards=len(fixtures['newton']),
    trigonometricStencilCases=len(fixtures['derivatives']),
    candidateCalibrationCases=len(fixtures['calibration']),
    nestedSearchCases=search_cases, nestedSuccesses=successes, nestedUnresolved=unresolved,
    mappedGaussianCases=gauss_cases, numericalComparisons=comparisons,
    maximumAbsoluteDiscrepancy=maximum_absolute, scipy=scipy.__version__, sourceHashes=sources,
    limitations=['Finite complementary evidence, not a universal error certificate.',
      'Underflow/extreme-range interval arithmetic is not asserted by the lesson or this review.'])
(OUT / 'results.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if k != 'sourceHashes'},indent=2))
