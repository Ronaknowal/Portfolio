"""Execute every complete lesson program and store its actual stdout."""
import contextlib
import ast
import io
import json
from pathlib import Path
from textwrap import dedent
import black

ROOT = Path(__file__).resolve().parents[1]
examples = {}


def example(key, title, question, code, explanation):
    code = dedent(code).strip()
    if key != 'original':
        before = ast.dump(ast.parse(code), include_attributes=False)
        code = black.format_str(code, mode=black.Mode(line_length=72)).strip()
        assert ast.dump(ast.parse(code), include_attributes=False) == before
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        exec(compile(code, f'<{key}>', 'exec'), {})
    examples[key] = dict(title=title, question=question, code=code,
                         expected=stream.getvalue().rstrip(), explanation=explanation)


original = json.loads((ROOT / 'docs/teaching/evidence/numerical-methods-original-content.json').read_text())
example('original', 'Three approximations, three references',
        'Which two comparisons have a visible approximation error at six decimal places?',
        original['blocks'][0]['text'],
        'The root agrees after rounding, but that alone is not a proof of its accuracy. The derivative uses a finite secant and the area uses four trapezoids; each differs from its analytic reference. The next sections explain how to control those differences.')
assert examples['original']['code'] == original['blocks'][0]['text']
assert examples['original']['expected'] == original['blocks'][1]['text']

example('bracket', 'A bracket with explicit stopping states',
        'How many halvings guarantee a midpoint within 0.001 when the initial width is two?', r'''
import math

def bisect(f, lower, upper, tolerance=1e-3, limit=100):
    if not (math.isfinite(lower) and math.isfinite(upper)
            and lower < upper and math.isfinite(upper-lower)):
        raise ValueError('Use a finite increasing interval')
    if not math.isfinite(tolerance) or tolerance <= 0 or limit < 0:
        raise ValueError('Invalid tolerance or work limit')
    fa, fb = f(lower), f(upper)
    if not (math.isfinite(fa) and math.isfinite(fb)):
        return None, 0, None, 'nonfinite evaluation'
    if fa == 0 or fb == 0:
        return (lower if fa == 0 else upper), 0, 0.0, 'endpoint root'
    if (fa < 0) == (fb < 0):
        return None, 0, None, 'no opposite signs'
    for halves in range(limit+1):
        midpoint = lower + (upper-lower)/2
        radius = (upper-lower)/2
        fm = f(midpoint)
        if not math.isfinite(fm):
            return midpoint, halves, radius, 'nonfinite evaluation'
        if fm == 0:
            return midpoint, halves, radius, 'computed zero'
        if radius <= tolerance:
            return midpoint, halves, radius, 'bracket tolerance'
        if midpoint == lower or midpoint == upper:
            return midpoint, halves, radius, 'arithmetic stagnation'
        if halves == limit:
            return midpoint, halves, radius, 'step limit'
        if (fa < 0) != (fm < 0):
            upper = midpoint
        else:
            lower, fa = midpoint, fm

root, halves, radius, status = bisect(lambda x:x*x-2, 0., 2.)
print(status, halves, round(root, 6), round(radius, 6))
print(abs(root-math.sqrt(2)) <= radius)
print(bisect(lambda x:(x-1)**2, 0., 2.)[-1])
''', 'Ten halvings give radius 1/1024. A touching root fails this initial sign test even though it exists. A returned computed zero is a statement about the evaluated function; with an inaccurate function oracle, more care is needed.')

example('newton', 'A cycle and a safeguarded escape',
        'Newton starts at zero on x³−2x+2. Where do its first two tangents send it?', r'''
import math

def f(x): return x**3-2*x+2
def derivative(x): return 3*x*x-2

x = 0.0
plain = [x]
for _ in range(4):
    x -= f(x)/derivative(x)
    plain.append(x)
print('plain:', plain)

lower, upper, x = -2.0, 0.0, 0.0
for step in range(5):
    slope = derivative(x)
    proposal = x-f(x)/slope if slope != 0 else math.nan
    width = upper-lower
    accept = (math.isfinite(proposal)
              and lower+.1*width <= proposal <= upper-.1*width)
    next_x = proposal if accept else lower+width/2
    value = f(next_x)
    print(step+1, 'tangent' if accept else 'bisect', round(next_x, 6))
    if value == 0:
        break
    if (f(lower) < 0) != (value < 0):
        upper = next_x
    else:
        lower = next_x
    x = next_x
print('bracket:', round(lower, 6), round(upper, 6))
''', 'The safeguard rejects a tangent proposal outside the central part of the bracket. Its progress comes from preserving the interval; a five-step demonstration is not a claim that the requested root accuracy has already been reached.')

example('differences', 'Watch actual arithmetic, not an ideal error curve',
        'Will every reduction of h improve the central difference of sin at x=1?', r'''
import math

x = 1.0
reference = math.cos(x)
for exponent in range(1, 13):
    h = 10.0**(-exponent)
    estimate = (math.sin(x+h)-math.sin(x-h))/(2*h)
    print(f'h={h:.0e}  error={abs(estimate-reference):.3e}')

f = lambda value: 1e12+value
h = 1e-5
print('large offset:', (f(x+h)-f(x-h))/(2*h), 'true derivative:', 1)
print('coordinates coincide:', x+1e-20 == x)
''', 'The curve eventually becomes irregular as cancellation exposes evaluation rounding. Adding a constant leaves the mathematical derivative unchanged but can ruin this numerical difference. Last-bit errors can vary with the platform math library.')

example('stencil', 'Derive a boundary stencil and check a direction',
        'Why do the boundary weights −3/2, 2, −1/2 eliminate constants and quadratic error terms?', r'''
from fractions import Fraction as F

weights = [F(-3, 2), F(2), F(-1, 2)]
moments = [sum(w*k**degree for k,w in enumerate(weights))
           for degree in range(4)]
print('moments:', moments)
# For x^3 at x=1 this stencil has derivative 3-2h^2.
for h in [F(1, 2), F(1, 4)]:
    estimate = sum(w*(1+k*h)**3 for k,w in enumerate(weights))/h
    print('h:', h, 'estimate:', estimate, 'error:', estimate-3)

def loss(x, y): return x*x+3*x*y+2*y*y
x, y = 1., 2.
vx, vy = 2., -1.
h = 1e-4
difference = (loss(x+h*vx,y+h*vy)-loss(x-h*vx,y-h*vy))/(2*h)
gradient_dot_direction = (2*x+3*y)*vx+(3*x+4*y)*vy
print('direction:', round(difference, 6), gradient_dot_direction)
''', 'The zeroth moment is zero, the first is one, and the second is zero. The cubic term remains. The directional check compares a scalar perturbation along (2,−1) with the locally derived gradient dotted with that same vector; it does not verify all possible inputs.')

example('quadrature', 'Weights are areas of local interpolants',
        'How do a quadratic curve and an irregular measurement record change the calculation?', r'''
def composite(f, a, b, n, method):
    if not (a < b and n >= 1 and isinstance(n, int)):
        raise ValueError('Use an increasing interval and positive integer n')
    if method not in ('trapezoid', 'simpson') or (method == 'simpson' and n%2):
        raise ValueError('Classic Simpson needs even n')
    h = (b-a)/n
    if method == 'trapezoid':
        return h*(f(a)/2+sum(f(a+i*h) for i in range(1,n))+f(b)/2)
    return h/3*(f(a)+f(b)+sum((4 if i%2 else 2)*f(a+i*h)
                            for i in range(1,n)))

for n in [2, 4, 8]:
    print(n, round(composite(lambda x:x*x,0.,1.,n,'trapezoid'),6),
          round(composite(lambda x:x*x,0.,1.,n,'simpson'),6))
times, rates = [0., .5, 2.], [0., 2., 2.]
areas = [(times[i+1]-times[i])*(rates[i]+rates[i+1])/2
         for i in range(len(times)-1)]
print('measured areas:', areas, 'total:', sum(areas))
''', 'Trapezoids overestimate this convex quadratic, while Simpson is exact in exact arithmetic. The measured record gives areas 0.5 and 3, not the result of pretending the times are equally spaced. Between measured points, the line is an assumption.')

example('adaptive', 'An adaptive estimator with a finite budget',
        'Where will extra samples be requested for a peak near 0.35?', r'''
import math

def adaptive_simpson(f, a, b, tolerance=1e-6, max_depth=12, max_evaluations=4097):
    if not (math.isfinite(a) and math.isfinite(b) and a < b
            and math.isfinite(b-a) and math.isfinite(tolerance) and tolerance > 0):
        raise ValueError('Invalid interval or tolerance')
    if not isinstance(max_depth, int) or max_depth < 0 or max_evaluations < 1:
        raise ValueError('Invalid work limits')
    samples, leaves = {}, []
    failed = False

    def evaluate(x):
        if x not in samples:
            if len(samples) >= max_evaluations:
                raise RuntimeError('evaluation limit')
            value = f(x)
            if not math.isfinite(value):
                raise RuntimeError('nonfinite evaluation')
            samples[x] = value
        return samples[x]

    def panel(left, right, fl, fm, fr):
        return (right-left)/6*(fl+4*fm+fr)

    def recurse(left, right, budget, depth):
        nonlocal failed
        middle = left+(right-left)/2
        lm, rm = left+(middle-left)/2, middle+(right-middle)/2
        if len({left,lm,middle,rm,right}) != 5:
            raise RuntimeError('arithmetic stagnation')
        fl, fm, fr = evaluate(left), evaluate(middle), evaluate(right)
        coarse = panel(left,right,fl,fm,fr)
        fine = (panel(left,middle,fl,evaluate(lm),fm)
                +panel(middle,right,fm,evaluate(rm),fr))
        error = abs(fine-coarse)/15
        corrected = fine+(fine-coarse)/15
        if not (math.isfinite(error) and math.isfinite(corrected)):
            raise RuntimeError('nonfinite arithmetic')
        if error <= budget or depth == max_depth:
            failed |= error > budget
            leaves.append((left,right,error,budget))
            return corrected
        return (recurse(left,middle,budget/2,depth+1)
                +recurse(middle,right,budget/2,depth+1))

    try:
        value = recurse(a,b,tolerance,0)
    except RuntimeError as error:
        return None, None, len(samples), str(error)
    status = 'depth limit' if failed else 'estimated tolerance'
    return value, sum(row[2] for row in leaves), len(samples), status

peak = lambda x:1/(1+400*(x-.35)**2)
reference = (math.atan(13)+math.atan(7))/20
value, estimate, evaluations, status = adaptive_simpson(peak,0.,1.)
print(status, evaluations, round(value,9))
print('estimated error:', f'{estimate:.2e}', 'actual error:', f'{abs(value-reference):.2e}')
print(adaptive_simpson(peak,0.,1.,max_evaluations=5)[-1])
''', 'The error estimate guides work under a smoothness model. Exceeding a limit returns a failure state. This standalone teaching program is adaptive Simpson, not SciPy QUADPACK.')

example('blind', 'A smooth function that initial samples cannot see',
        'Can five zero observations prove that a smooth nonnegative function has zero area?', r'''
from fractions import Fraction as F

nodes = [F(0), F(1,4), F(1,2), F(3,4), F(1)]
# Ascending-power coefficients of the product, then its square.
coefficients = [F(1)]
for root in nodes:
    expanded = [F(0)]*(len(coefficients)+1)
    for power, value in enumerate(coefficients):
        expanded[power] -= root*value
        expanded[power+1] += value
    coefficients = expanded
squared = [F(0)]*(2*len(coefficients)-1)
for i, first in enumerate(coefficients):
    for j, second in enumerate(coefficients):
        squared[i+j] += first*second
def g(x): return sum(value*x**power for power,value in enumerate(squared))
area = sum(value/F(power+1) for power,value in enumerate(squared))
print('samples:', [g(x) for x in nodes])
print('true integral:', area, f'{float(area):.9e}')
print('lost area below 1e-6 for x^(-1/2):', 2*(1e-6)**.5)
print('after x=u^2: integral of 2 on [0,1] =', 2)
''', 'The polynomial is positive between its roots, but the initial Simpson estimates both vanish. Exact coefficient integration supplies an independent reference. The singularity example shows why changing variables can be better than silently omitting a troublesome endpoint.')

example('gaussian', 'Choose two nodes by matching moments',
        'What happens just beyond the two-node rule\'s exact polynomial degree?', r'''
import math

node = 1/math.sqrt(3)
for power in range(5):
    estimate = (-node)**power+node**power
    exact = 0. if power%2 else 2/(power+1)
    print(power, round(estimate,6), round(exact,6))
def gauss_two(f, a, b):
    midpoint, radius = (a+b)/2, (b-a)/2
    return radius*(f(midpoint-radius*node)+f(midpoint+radius*node))
print('x^2 on [2,4]:', round(gauss_two(lambda x:x*x,2.,4.),6))
''', 'The first four monomials match. For x⁴ the rule gives 2/9 instead of 2/5, so smoothness alone does not make it exact. Moving from [−1,1] to [2,4] requires both relocated nodes and the interval scale factor.')

example('calibration', 'Translate volume error into time error',
        'Is the approximation at three minutes enough to locate the 0.8-litre crossing within 0.01 minutes?', r'''
import math

def rate(t): return t*math.exp(-t)
def volume(t): return -math.expm1(-t)-t*math.exp(-t)
def estimate_volume(t, n):
    h = t/n
    return h*(rate(0)/2+sum(rate(i*h) for i in range(1,n))+rate(t)/2)

target, candidate = .8, 3.
minimum_slope = 4*math.exp(-4)
for n in [8, 32, 128]:
    estimate = estimate_volume(candidate,n)
    integral_bound = candidate**3/(6*n*n)
    time_bound = (abs(estimate-target)+integral_bound)/minimum_slope
    sign = ('above' if estimate-target > integral_bound else
            'below' if estimate-target < -integral_bound else 'unresolved')
    print(n, round(estimate,7), sign, 'time bound:', round(time_bound,6))

# Independent analytic volume is used only to validate the exercise.
lower, upper = 2., 4.
for _ in range(50):
    mid = lower+(upper-lower)/2
    if volume(mid) < target: lower = mid
    else: upper = mid
reference = lower+(upper-lower)/2
print('reference time:', round(reference,9))
print('candidate time error:', round(abs(candidate-reference),9))
''', 'The conditional bound concerns trapezoid discretization plus residual and uses a known positive minimum slope. Ordinary float evaluation adds an un-enclosed rounding contribution. Even an accurate integral cannot compensate for choosing a candidate too far from the target.')

example('nestedCalibration', 'Run the integral-to-target search',
        'What should an outer bisection do when its numerical integral cannot resolve the sign?', r'''
import math

def rate(t): return t*math.exp(-t)
def volume(t): return -math.expm1(-t)-t*math.exp(-t)

def locate_target(target, time_tolerance=.01, max_panels=128):
    if not (volume(2) < target < volume(4)):
        raise ValueError('Target must be strictly inside the known bracket')
    if not math.isfinite(time_tolerance) or time_tolerance <= 0:
        raise ValueError('Use a positive finite time tolerance')
    if not isinstance(max_panels, int) or max_panels < 8:
        raise ValueError('Allow at least eight panels')
    lower, upper = 2., 4.
    minimum_slope = 4*math.exp(-4)
    trace = []
    for _ in range(60):
        candidate = lower+(upper-lower)/2
        radius = (upper-lower)/2
        if radius <= time_tolerance:
            return candidate, radius, 'bracket time tolerance', trace
        if candidate == lower or candidate == upper:
            return candidate, radius, 'arithmetic stagnation', trace
        panels = 8
        while True:
            h = candidate/panels
            estimate = h*(rate(0)/2+math.fsum(rate(i*h) for i in range(1,panels))
                          +rate(candidate)/2)
            error_bound = candidate**3/(6*panels**2)
            residual = estimate-target
            time_bound = (abs(residual)+error_bound)/minimum_slope
            trace.append((lower,upper,candidate,panels,estimate,error_bound))
            if time_bound <= time_tolerance:
                return candidate,time_bound,'conditional time tolerance',trace
            if residual > error_bound:
                upper = candidate
                break
            if residual < -error_bound:
                lower = candidate
                break
            if panels == max_panels:
                return candidate,time_bound,'unresolved integral sign',trace
            panels = min(2*panels,max_panels)
    return candidate,radius,'outer step limit',trace

for budget in [8, 128]:
    time, bound, status, trace = locate_target(.8,max_panels=budget)
    print('max panels:',budget,status)
    print('time:',round(time,6),'bound:',round(bound,6),'integral calls:',len(trace))
    print('first candidate panel counts:',[row[3] for row in trace if row[2]==3.])
''', 'The eight-panel search stops with unresolved sign instead of discarding a half-bracket. With enough inner work it can report a conditional time bound. These guarantees use the mathematical trapezoid bound and reliable arithmetic; the program is not an outward-rounded interval-arithmetic solver.')

example('transfer', 'Residuals and time steps need a scale',
        'Can a tiny residual hide a unit error, and can stable-looking time samples describe growing numerical dynamics?', r'''
import math

# A=diag(1,1e-6), b=(1,1e-6), exact solution=(1,1).
approximate = [1., 0.]
residual = [approximate[0]-1, 1e-6*approximate[1]-1e-6]
print('residual norm:', math.hypot(*residual), 'solution error:', 1.)

k, horizon = 2., 3.
for steps in [3, 6, 12, 24]:
    h = horizon/steps
    computed = (1-k*h)**steps
    exact = math.exp(-k*horizon)
    print('h:', h, 'Euler:', round(computed,8), 'exact:', round(exact,8))

from scipy.optimize import brentq
from scipy.integrate import quad
root, status = brentq(lambda x:x*x-2,0.,2.,full_output=True)
area, error_estimate = quad(lambda x:math.exp(-x*x),0.,1.)
print('SciPy root:', round(root,9), 'converged:', status.converged)
print('SciPy area:', round(area,9), 'estimated error:', f'{error_estimate:.2e}')
''', 'The diagonal solve amplifies a residual in its small-coefficient direction. The Euler comparisons all end at the same physical time. The SciPy lines report real library status and an estimated quadrature error, rather than declaring all errors controlled by one flag.')

example('changedCalibration', 'Accepted changed calibration check',
        'After attempting the task, compare your changed target and candidate with this complete calculation.', r'''
import math
from scipy.optimize import brentq

rate = lambda t: t*math.exp(-t)
volume = lambda t: -math.expm1(-t)-t*math.exp(-t)
target, candidate, n = .7, 2.44, 32
h = candidate/n
estimate = h*(rate(0)/2+sum(rate(i*h) for i in range(1,n))+rate(candidate)/2)
bound = candidate**3/(6*n*n)
time_bound = (abs(estimate-target)+bound)/(4*math.exp(-4))
reference = brentq(lambda t: volume(t)-target, 2., 4.)
print(f'residual={estimate-target:.6f}, integral_bound={bound:.6f}')
print(f'time_bound={time_bound:.6f}, actual_error={abs(candidate-reference):.6f}')
''', 'The residual interval straddles zero, so this bound cannot resolve the sign. The conditional time bound exceeds the actual error, as intended. An inconclusive bound does not mean that the candidate must be poor.')

target = ROOT / 'src/learn/data/numerical-methods-examples.js'
target.write_text('// Generated from actually executed standalone Python programs.\nexport const numericalMethodsExamples = '
                  + json.dumps(examples, ensure_ascii=False, indent=2)+';\n', encoding='utf8')
print(f'Executed and saved {len(examples)} complete programs.')
