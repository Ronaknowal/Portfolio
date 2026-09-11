"""Independent exact/NumPy/SciPy checks of actual models and displayed helpers."""
import contextlib
import io
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'scratch/numerical-methods-verification'
subprocess.run(['node', 'scripts/verify-numerical-methods-models.mjs'], cwd=ROOT, check=True)
fixtures = json.loads((DATA/'model-fixtures.json').read_text())
examples = json.loads(subprocess.check_output(['node', '--input-type=module', '-e',
    "import {numericalMethodsExamples as e} from './src/learn/data/numerical-methods-examples.js';console.log(JSON.stringify(e))"], cwd=ROOT, text=True, encoding='utf8'))
scopes = {}
for key, record in examples.items():
    scope, stream = {}, io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(record['code'], f'<actual-{key}>', 'exec'), scope)
    assert stream.getvalue().rstrip() == record['expected'], key
    scopes[key] = scope
original = json.loads((ROOT/'docs/teaching/evidence/numerical-methods-original-content.json').read_text())
assert examples['original']['code'] == original['blocks'][0]['text']
assert examples['original']['expected'] == original['blocks'][1]['text']

for case in fixtures['brackets']:
    target, state = case['target'], case['state']
    reference = brentq(lambda x:x*x-target, 0, 10, xtol=5e-15)
    for row in state['steps']:
        assert row['lower']-1e-14 <= reference <= row['upper']+1e-14
        assert abs(row['midpoint']-reference) <= row['radius']+1e-14
    native = scopes['bracket']['bisect'](lambda x:x*x-target, 0., 10., case['tolerance'])
    assert abs(native[0]-reference) <= native[2]+1e-14

def polynomial(key, x):
    if key == 'square': return x*x-2, 2*x
    if key == 'cycle': return x*x*x-2*x+2, 3*x*x-2
    return (x-1)**2, 2*(x-1)

for case in fixtures['newton']:
    for row in case['state']['steps']:
        value, slope = polynomial(case['key'], F(row['current']))
        np.testing.assert_allclose(float(value), row['value'], atol=2e-13, rtol=2e-13)
        np.testing.assert_allclose(float(slope), row['slope'], atol=2e-13, rtol=2e-13)
        if row['accepted']:
            np.testing.assert_allclose(float(F(row['current'])-value/slope), row['next'], atol=3e-12, rtol=3e-12)

quadrature_max_error = 0.
for case in fixtures['quadrature']:
    a, b, n, degree = F(case['lower']), F(case['upper']), case['panels'], case['power']
    h = (b-a)/n
    values = [(a+i*h)**degree for i in range(n+1)]
    if case['method']=='trapezoid':
        expected = h*(values[0]/2+sum(values[1:-1])+values[-1]/2)
    else:
        expected = h/F(3)*(values[0]+values[-1]+sum((4 if i%2 else 2)*values[i] for i in range(1,n)))
    error = abs(float(expected)-case['state']['estimate'])
    quadrature_max_error = max(quadrature_max_error,error)
    np.testing.assert_allclose(float(expected),case['state']['estimate'],atol=1e-11,rtol=1e-13)
    exact = (b**(degree+1)-a**(degree+1))/(degree+1)
    if (case['method']=='trapezoid' and degree<=1) or (case['method']=='simpson' and degree<=3): assert expected==exact

functions = {'polynomial':lambda x:x*x, 'sine':lambda x:math.sin(math.pi*x),
             'peak':lambda x:1/(1+400*(x-.35)**2),
             'blind':lambda x:(x*(x-.25)*(x-.5)*(x-.75)*(x-1))**2}
adaptive_error = 0.
for case in fixtures['adaptive']:
    state, f = case['state'], functions[case['key']]
    reference = quad(f,0,1,epsabs=1e-13,epsrel=1e-13)[0]
    # Independent integral over each interpolating local polynomial.
    for leaf in state['leaves']:
        a,b=leaf['lower'],leaf['upper']; m=(a+b)/2
        coarse=(b-a)/6*(f(a)+4*f(m)+f(b))
        fine=(b-a)/12*(f(a)+4*f((a+m)/2)+2*f(m)+4*f((m+b)/2)+f(b))
        np.testing.assert_allclose(leaf['value'],(16*fine-coarse)/15,atol=2e-15)
    if case['key']=='blind':
        assert state['estimate']==0 and state['estimatedError']==0
        np.testing.assert_allclose(reference,float(F(5,1419264)),rtol=1e-13)
    elif state['status']=='estimated tolerance reached':
        adaptive_error=max(adaptive_error,abs(reference-state['estimate']))
        # This is a checked fixture property, NOT a general estimator theorem.
        assert abs(reference-state['estimate']) < max(5*case['tolerance'],1e-12)
    native=scopes['adaptive']['adaptive_simpson'](f,0.,1.,case['tolerance'],case['maximumDepth'])
    np.testing.assert_allclose(native[0],state['estimate'],atol=1e-14)
    assert native[2]==len(state['samples'])

for case in fixtures['derivatives']:
    x,h,stencil=case['x'],case['h'],case['stencil']
    # Analytic trigonometric identities give exact-arithmetic stencil values
    # without subtracting two nearby sine evaluations.
    if stencil=='central': expected=math.cos(x)*math.sin(h)/h
    elif stencil=='forward': expected=2*math.sin(h/2)*math.cos(x+h/2)/h
    elif stencil=='second': expected=-4*math.sin(x)*math.sin(h/2)**2/(h*h)
    else: expected=(4*math.sin(h/2)*math.cos(x+h/2)-math.sin(h)*math.cos(x+h))/h
    # Value-level roundoff is amplified by the stated weights and h power.
    denominator=h*h if stencil=='second' else h
    np.testing.assert_allclose(case['state']['estimate'],expected,atol=3e-15/denominator,rtol=2e-13)
for case in fixtures['samples']:
    a,b=case['xs'][0],case['xs'][-1]
    expected=b*b+3*b-a*a-3*a
    np.testing.assert_allclose(case['state']['integral'],expected,atol=2e-12)

for case in fixtures['calibration']:
    target,time=case['target'],case['candidate']
    reference=quad(lambda t:t*math.exp(-t),0,time,epsabs=1e-13)[0]
    root=brentq(lambda t:1-(1+t)*math.exp(-t)-target,2,4,xtol=1e-14)
    assert abs(case['estimate']-reference) <= case['integralBound']+1e-14
    assert abs(time-root) <= case['timeBound']+1e-13
    if case['sign']=='above target': assert reference>target
    if case['sign']=='below target': assert reference<target
    np.testing.assert_allclose(case['volume'],reference,atol=3e-15)

nested_cases = 0
for target in [.61,.7,.8,.89]:
    reference = brentq(lambda t:1-(1+t)*math.exp(-t)-target,2,4,xtol=1e-14)
    for tolerance in [.1,.01,.001]:
        for max_panels in [8,32,128,512]:
            candidate,bound,status,trace = scopes['nestedCalibration']['locate_target'](target,tolerance,max_panels)
            nested_cases += 1
            if status in ['conditional time tolerance','bracket time tolerance']:
                assert abs(candidate-reference)<=bound+1e-13 and bound<=tolerance
            else:
                assert status=='unresolved integral sign'
            for lower,upper,time,panels,estimate,integral_bound in trace:
                assert lower-1e-13 <= reference <= upper+1e-13
                actual=quad(lambda t:t*math.exp(-t),0,time,epsabs=1e-13)[0]
                assert abs(estimate-actual)<=integral_bound+1e-13

# Exact independent checks of changed hand-practice values and oscillator bridge.
assert F(2,2**7)>F(1,100) and F(2,2**8)<=F(1,100)
assert F(1000)*F(5,1419264)==F(625,177408)
assert F(2,9)-F(2,5)==F(-8,45)
assert [float(h*h+F(1,10**6)/h) for h in [F(1,10),F(1,100),F(1,1000)]]==[.01001,.0002,.001001]
assert scopes['adaptive']['adaptive_simpson'](functions['peak'],0.,1.,max_evaluations=5)[0] is None
for h in [F(1,10),F(1,2),F(3,2)]:
    for q,p in [(F(2),F(3)),(F(-1),F(4))]:
        next_p=p-h*q; next_q=q+h*next_p
        assert next_q**2+next_p**2-h*next_q*next_p==q*q+p*p-h*q*p
        assert (q+h*p)**2+(p-h*q)**2==(1+h*h)*(q*q+p*p)

result={'checkedAt':datetime.now(timezone.utc).isoformat(),'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
        'actualPrograms':len(examples),'originalCodeAndOutputConserved':True,'nestedSearchCases':nested_cases,
        'cases':{key:len(value) for key,value in fixtures.items()},'maximumExactQuadratureArithmeticError':quadrature_max_error,
        'maximumAcceptedAdaptiveFixtureError':adaptive_error,'handPracticeAndOscillatorIdentities':'passed',
        'limits':'Finite tests; estimator acceptance is not a theorem for arbitrary functions. Float bounds do not enclose all arithmetic.'}
(DATA/'native-results.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
