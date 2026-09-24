"""Exact and high precision oracles for the actual bounded models and programs."""
import contextlib
import io
import json
import math
import platform
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path

import mpmath as mp
import sympy as sp

mp.mp.dps = 65
directory = Path('scratch/single-variable-calculus-verification')
data = json.loads((directory / 'cases.json').read_text(encoding='utf-8'))


def close(actual, expected, tolerance=3e-12):
    assert math.isclose(actual, float(expected), rel_tol=tolerance, abs_tol=tolerance), (actual, expected)


def fraction(value):
    return F(str(value))


def position(t):
    return t**3 - 6*t**2 + 9*t


for s in data['rates']:
    t, h = fraction(s['baseTime']), fraction(s['increment'])
    change = position(t+h)-position(t)
    derivative = 3*t*t-12*t+9
    for field, expected in [('position', position(t)), ('nextPosition', position(t+h)), ('velocity', derivative), ('secant', change/h), ('acceleration', 6*t-12), ('predictedChange', derivative*h), ('exactChange', change), ('remainder', change-derivative*h)]:
        close(s[field], expected)

for s in data['limits']:
    epsilon, delta = fraction(s['epsilon']), fraction(s['delta'])
    jump = 2 if s['kind'] == 'jump' else 0
    supremum = (2+delta)**2-4+jump
    assert s['guaranteed'] == (supremum <= epsilon)
    close(s['supremum'], supremum)
    if s['guaranteed']:
        assert s['witness'] is None
    else:
        w = s['witness']
        x = F(w['inputExact'])
        assert 0 < x-2 < delta
        assert F(w['displacementExact']) == x-2
        assert F(w['errorExact']) == x*x-4+jump >= epsilon
        if w['resolvedInFloat']:
            assert 0 < abs(w['input']-2) < float(delta) and w['error'] >= float(epsilon)

for s in data['compositions']:
    x, h = fraction(s['input']), fraction(s['increment'])
    f = lambda z: 9*z*z+6*z+1
    derivative = 18*x+6
    for field, expected in [('output', f(x)), ('nextOutput', f(x+h)), ('intermediate', 3*x+1), ('intermediateChange', 3*h), ('outerDerivative', 6*x+2), ('derivative', derivative), ('predictedChange', derivative*h), ('exactChange', f(x+h)-f(x)), ('remainder', 9*h*h)]:
        close(s[field], expected)

t = sp.Symbol('t', real=True)
polynomials = {'motion': t**3-6*t**2+9*t, 'inflection': (t-2)**3}
for s in data['extrema']:
    a, b = fraction(s['left']), fraction(s['right'])
    if s['kind'] == 'cusp':
        critical = [F(2)]
        f = lambda x: abs(x-2)
        derivative = lambda x: None if x == 2 else (-1 if x < 2 else 1)
    else:
        expression = polynomials[s['kind']]
        differentiated = sp.diff(expression, t)
        critical = [F(str(root)) for root in sp.solve(differentiated, t)]
        f = lambda x: F(expression.subs(t, sp.Rational(x.numerator, x.denominator)))
        derivative = lambda x: F(differentiated.subs(t, sp.Rational(x.numerator, x.denominator)))
    candidates = sorted(set([a, b]+[x for x in critical if a <= x <= b]))
    assert [fraction(c['input']) for c in s['candidates']] == candidates
    values = [f(x) for x in candidates]
    close(s['minimum'], min(values)); close(s['maximum'], max(values))
    for c, x, value in zip(s['candidates'], candidates, values):
        close(c['value'], value)
        assert F(c['valueExact']) == value
        assert c['isMinimum'] == (value == min(values))
        assert c['isMaximum'] == (value == max(values))
        if derivative(x) is None: assert c['derivative'] is None
        else: close(c['derivative'], derivative(x))
    for interval in s['intervals']:
        midpoint = (fraction(interval['start'])+fraction(interval['end']))/2
        d = derivative(midpoint)
        assert interval['sign'] == (1 if d > 0 else -1 if d < 0 else 0)

for s in data['accumulations']:
    upper, n = fraction(s['upper']), s['panelCount']
    width = upper/n
    offset = {'left': F(0), 'midpoint': F(1,2), 'right': F(1)}[s['method']]
    terms = []
    for index, panel in enumerate(s['panels']):
        x = (index+offset)*width
        height = 3*x*x-12*x+9
        terms.append(width*height)
        for field, expected in [('start', index*width), ('end', (index+1)*width), ('sample', x), ('height', height), ('signedContribution', width*height)]:
            close(panel[field], expected)
    close(s['signedEstimate'], sum(terms)); close(s['distanceEstimate'], sum(map(abs, terms)))
    close(s['exactDisplacement'], position(upper))
    # Integrate positive/negative polynomial branches symbolically, splitting at roots.
    cuts = [F(0)]+[F(x) for x in (1,3) if x < upper]+[upper]
    distance = sum(abs(position(b)-position(a)) for a,b in zip(cuts,cuts[1:]))
    close(s['exactDistance'], distance)
    close(s['signedError'], sum(terms)-position(upper))
    close(s['endpointVelocity'], 3*upper**2-12*upper+9)

for s in data['growth']:
    k, period = mp.mpf(str(s['rate'])), mp.mpf(str(s['period']))
    current, future = 10*mp.exp(2*k), 10*mp.exp((2+period)*k)
    for field, expected in [('currentAmount', current), ('futureAmount', future), ('instantaneousRate', k*current), ('intervalAverageRate', (future-current)/period), ('perPeriodFraction', mp.exp(k*period)-1), ('averageRelativeRate', (mp.exp(k*period)-1)/period), ('tangentPrediction', current+k*current*period)]:
        close(s[field], expected)

for s in data['taylors']:
    x, degree = mp.mpf(str(s['input'])), s['degree']
    f = mp.exp if s['kind'] == 'exp' else lambda z: mp.log(1+z)
    coefficients = mp.taylor(f, 0, degree)
    polynomial = mp.polyval(list(reversed(coefficients)), x)
    close(s['polynomial'], polynomial); close(s['exactValue'], f(x))
    close(s['signedError'], polynomial-f(x))
    maximum = mp.exp(max(0,x)) if s['kind'] == 'exp' else mp.factorial(degree)/min(1,1+x)**(degree+1)
    bound = maximum*abs(x)**(degree+1)/mp.factorial(degree+1)
    close(s['remainderBound'], bound)
    assert abs(polynomial-f(x)) <= bound*(1+mp.mpf('1e-55'))
    if degree in (0,2,6) and x in (mp.mpf('-.9'),mp.mpf('.5'),mp.mpf('1.5')):
        remainder = mp.quad(lambda u: mp.diff(f,u,degree+1)*(x-u)**degree/mp.factorial(degree), [0,x])
        assert abs(remainder-(f(x)-polynomial)) < mp.mpf('1e-48')

for s in data['improper']:
    # Near p=1, preserve the binary float input exactly. Its distance to1 matters.
    power, q = mp.mpf(s['power']), mp.mpf(s['cutoffExponent'])
    log_cutoff = q*mp.log(10)
    sign = 1 if s['kind'] == 'tail' else -1
    cutoff = mp.exp(sign*log_cutoff)
    # x=exp(u) converts the definite integral to an exponential integrand.
    a,b = (0,log_cutoff) if sign == 1 else (-log_cutoff,0)
    truncated = mp.quad(lambda u: mp.exp((1-power)*u), [a,b]) if a != b else 0
    close(s['cutoff'], cutoff); close(s['truncated'], truncated)
    converges = power > 1 if sign == 1 else power < 1
    assert s['converges'] == converges
    if converges:
        total = 1/abs(power-1)
        close(s['total'], total); close(s['missing'], total-truncated)
    else: assert s['total'] is None and s['missing'] is None

programs, namespaces = [], {}
for example in data['examples']:
    capture, namespace = io.StringIO(), {'__name__': '__main__'}
    with contextlib.redirect_stdout(capture):
        exec(compile(example['code'], example['id']+'.py', 'exec'), namespace)
    assert capture.getvalue().strip() == example['expected'].strip(), example['id']
    programs.append(example['id']); namespaces[example['id']] = namespace
for x in [F(-3,2),F(1,3),F(5,2),F(7)]:
    assert namespaces['motion-secants']['position'](x) == position(x)
    assert namespaces['interval-extrema']['position'](x) == position(x)
    assert namespaces['signed-rectangles']['velocity'](x) == 3*x*x-12*x+9
    assert namespaces['moving-bounds']['accumulated'](x) == (x*x+(x*x)**3/F(3))-(x+x**3/F(3))

# Changed practice and displayed application values, independent symbolic operations.
x = sp.Symbol('x', real=True)
assert sp.diff((2*x-1)**2,x).subs(x,0) == -4
assert sp.Rational(4,5)**2-1 == -sp.Rational(9,25)
assert sp.integrate(2*x-4,(x,0,4)) == 0
assert -sp.integrate(2*x-4,(x,0,2))+sp.integrate(2*x-4,(x,2,4)) == 8
assert sp.simplify(sp.integrate(2*x/(1+x*x),(x,1,3))-sp.log(5)) == 0
assert sp.integrate(x*sp.sin(x),(x,0,sp.pi)) == sp.pi
assert sp.integrate(3+2*x,(x,0,2)) == 10
assert sp.integrate(2+x,(x,0,2)) == 6
assert sp.integrate(x*(2+x),(x,0,2)) == sp.Rational(20,3)
assert sp.integrate(sp.pi*x,(x,0,3)) == 9*sp.pi/2
assert sp.integrate(sp.sqrt(1+x),(x,0,3)) == sp.Rational(14,3)
assert sp.integrate(x*sp.cos(x),(x,0,1)) == sp.sin(1)+sp.cos(1)-1
close(math.log(.8)/3,-.0743812,1e-7)
close(math.log(.5)/(math.log(.8)/3),9.318851,1e-7)
assert 0 < mp.mpf('3.05')-mp.sqrt(mp.mpf('9.3')) < mp.mpf(1)/2400
close(mp.mpf(2)/mp.sqrt(100), .2)
result = {'at': datetime.now(timezone.utc).isoformat(), 'passed': True, 'python': platform.python_version(), 'mpmathPrecision': mp.mp.dps,
          'cases': {key:len(value) for key,value in data.items() if isinstance(value,list) and key not in ('sources','examples')}, 'actualPrograms': programs, 'changedNativeHelpers': 16, 'practiceAndApplicationChecks': 16, 'syntax':data['syntax'], 'sources':data['sources'],
          'limits':['Finite and symbolic checks support the implementations; they do not replace the written real-analysis proofs.', 'Browser geometry, readable rendering and production loading have separate records.']}
(directory/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
