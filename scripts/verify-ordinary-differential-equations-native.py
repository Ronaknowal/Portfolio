"""Independent native oracles for the actual ODE model exports and displayed helpers."""
import contextlib
import io
import json
import math
import platform
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import numpy as np
import scipy
from scipy.integrate import quad, solve_ivp
from scipy.linalg import expm

root = Path(__file__).resolve().parents[1]
directory = root / 'scratch/ordinary-differential-equations-verification'
fixtures = json.loads((directory / 'model-fixtures.json').read_text(encoding='utf-8'))
examples = json.loads((directory / 'examples.json').read_text(encoding='utf-8'))
mp.mp.dps = 90
counts = {}
maximum_matrix_error = 0.0


def close(actual, expected, atol=2e-11, rtol=2e-11):
    assert np.allclose(actual, expected, atol=atol, rtol=rtol), (actual, expected)


namespaces = {}
for key, record in examples.items():
    output = io.StringIO()
    namespace = {}
    with contextlib.redirect_stdout(output):
        exec(compile(record['code'], key, 'exec'), namespace)
    assert output.getvalue().rstrip() == record['expected'], key
    namespaces[key] = namespace
counts['complete_programs_exact_stdout'] = len(examples)

for case in fixtures['cooling']:
    time, rate, initial = (mp.mpf(case[key]) for key in ('time', 'rate', 'initial'))
    value = 10 + (initial - 10) * mp.exp(-rate * time)
    close(case['result']['state'], float(value))
    close(case['result']['derivative'], float(-rate * (value - 10)))
counts['cooling_high_precision'] = len(fixtures['cooling'])

for case in fixtures['logistic']:
    time, rate, initial, capacity = (mp.mpf(case[key]) for key in ('time', 'rate', 'initial', 'capacity'))
    value = mp.mpf(0) if initial == 0 else capacity * initial / (initial + (capacity - initial) * mp.exp(-rate * time))
    expected = float(value)
    actual = case['result']['state']
    tolerance = max(3 * math.ulp(expected), 3e-12 * abs(expected))
    assert abs(actual - expected) <= tolerance, (case, expected, tolerance)
counts['logistic_high_precision_including_subnormal'] = len(fixtures['logistic'])

for case in fixtures['waiting']:
    result = case['result']
    close(result['derivative'],2*math.sqrt(abs(result['state'])))
    if case['time'] <= case['departure']:
        assert result['state'] == result['derivative'] == 0
    else:
        close(math.sqrt(result['state']),case['time']-case['departure'])
counts['waiting_family_equation_and_join_checks'] = len(fixtures['waiting'])

for case in fixtures['schedules']:
    upper = np.array([[0.,1.],[0.,0.]])
    lower = upper.T
    first,second = (upper,lower) if case['first'] == 'upper' else (lower,upper)
    intermediate = expm(first)@case['initial']
    close(case['result']['intermediate'],intermediate)
    close(case['result']['final'],expm(second)@intermediate)
counts['ordered_stage_matrix_checks'] = len(fixtures['schedules'])

for case in fixtures['matrices']:
    reference = expm(np.asarray(case['matrix']) * case['time'])
    actual = np.asarray(case['result'])
    close(actual, reference)
    maximum_matrix_error = max(maximum_matrix_error, float(np.max(np.abs(actual-reference))))
counts['independent_scipy_exponentials'] = len(fixtures['matrices'])

for case in fixtures['oscillators']:
    matrix = np.array([[0,1],[-4,-case['damping']]], dtype=float)
    position, velocity = expm(case['time'] * matrix) @ [1,case['velocity']]
    result = case['result']
    close(result['state'], [position,velocity])
    close(result['energy'], (velocity**2+4*position**2)/2)
    close(result['energyRate'], -case['damping']*velocity**2)
counts['oscillator_near_critical_energy'] = len(fixtures['oscillators'])

for case in fixtures['forcing']:
    time, switch = case['time'], case['switchTime']
    first_end = min(time,switch)
    first = quad(lambda entered: math.exp(-0.2*(time-entered))*0.1*case['firstPower'],0,first_end)[0]
    second = quad(lambda entered: math.exp(-0.2*(time-entered))*0.1*case['secondPower'],switch,time)[0] if time > switch else 0
    initial = 24*math.exp(-0.2*time)
    close([case['result'][key] for key in ('initialContribution','firstContribution','secondContribution','state')], [initial,first,second,initial+first+second])
counts['independent_forcing_quadratures'] = len(fixtures['forcing'])

for case in fixtures['steps']:
    z = -case['rate']*case['step']
    degree = {'euler':1,'midpoint':2,'rk4':4}[case['method']]
    reference = 40*sum(z**power/math.factorial(power) for power in range(degree+1))
    close(case['result']['state'], reference)
counts['exact_linear_stability_polynomials'] = len(fixtures['steps'])

for case in fixtures['integrations']:
    result = case['result']
    assert result['status'] == 'reached the requested horizon'
    assert result['endTime'] == 5
    history = result['history']
    reference = 40
    degree = {'euler':1,'midpoint':2,'rk4':4}[case['method']]
    for previous, current in zip(history,history[1:]):
        width = current['time']-previous['time']
        z = -0.2*width
        reference *= sum(z**power/math.factorial(power) for power in range(degree+1))
        close(current['state'], reference)
    close(result['signedError'],reference-40/math.e)
counts['same_horizon_partial_step_products'] = len(fixtures['integrations'])

changed = 0
fixed = namespaces['fixedSteps']['integrate']
for method in ('euler','midpoint','rk4'):
    for width in (0.13,0.37,0.8):
        history,status = fixed(method,lambda t,y: t-y,1.0,2.0,width)
        assert status == 'reached horizon' and history[-1][0] == 2
        # Independent high-accuracy ODE solve for a nonautonomous rate law.
        reference = 1+2*math.exp(-2)
        error = abs(history[-1][1]-reference)
        finer,_ = fixed(method,lambda t,y: t-y,1.0,2.0,width/2)
        assert abs(finer[-1][1]-reference) < error
        changed += 1
counts['actual_nonautonomous_refinement_pairs'] = changed

bound_cases = 0
for rate in (0.2,1,3):
    for initial in (1,4):
        for count in (8,17,40):
            width = 1/count
            history,status = fixed('euler',lambda t,y:-rate*y,initial,1.0,width)
            exact = initial*math.exp(-rate)
            error = abs(history[-1][1]-exact)
            curvature = rate**2*initial
            bound = curvature*width*(math.exp(rate)-1)/(2*rate)
            assert error <= bound*(1+1e-12)
            bound_cases += 1
counts['actual_euler_error_bound_cases'] = bound_cases

held = namespaces['heldInput']['held_input_matrices']
held_cases = 0
for generator in (np.zeros((2,2)),np.array([[0,1],[0,0]]),np.array([[-1,2],[0,-1]]),np.array([[0,-2],[2,0]])):
    for duration in (0,1e-8,0.3,2.0):
        for input_vector in (np.array([[0.0],[1.0]]),np.array([[2.0],[-1.0]])):
            transition,response = held(generator,input_vector,duration)
            close(transition,expm(generator*duration))
            reference = np.array([quad(lambda tau: (expm(generator*tau)@input_vector)[row,0],0,duration)[0] for row in range(2)])
            close(response.ravel(),reference)
            held_cases += 1
counts['actual_held_input_changed_quadratures'] = held_cases

series_cases = 0
for start in ((1,0),(0,1),(2,-3)):
    coefficients = namespaces['series']['coefficients'](*start,18)
    for power in range(17):
        assert (power+2)*(power+1)*coefficients[power+2] == (coefficients[power-1] if power else 0)
        series_cases += 1
    reference = solve_ivp(lambda t,state:[state[1],t*state[0]],(0,1.2),start,method='DOP853',rtol=2e-13,atol=2e-14)
    close(namespaces['series']['evaluate'](coefficients,1.2),reference.y[0,-1],atol=3e-8)
counts['actual_exact_fraction_coefficient_identities'] = series_cases

for damping in (4-1e-8,4,4+1e-8):
    for time in (0,0.7,2):
        actual = namespaces['oscillator']['motion'](time,1,damping,4,[1,-1])
        reference = solve_ivp(lambda t,state:[state[1],-4*state[0]-damping*state[1]],(0,max(time,1e-9)),[1,-1],method='DOP853',rtol=1e-12,atol=1e-14)
        if time == 0: close(actual[:2],[1,-1])
        else: close(actual[:2],reference.y[:,-1])
counts['actual_near_critical_independent_integrations'] = 9

result = dict(checkedAt=datetime.now(timezone.utc).isoformat(),status='passed',counts=counts,maximumMatrixAbsoluteError=maximum_matrix_error,versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,mpmath=mp.__version__))
(directory / 'native-results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
