"""Complementary exact/high-precision checks against final production exports."""
import contextlib
import hashlib
import io
import json
import math
import platform
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path

import mpmath as mp
import numpy as np
import scipy
from scipy.integrate import quad, solve_ivp

directory = Path('scratch/ordinary-differential-equations-independent')
data = json.loads((directory / 'fixtures.json').read_text(encoding='utf-8'))
mp.mp.dps = 420
counts = {}


def checked(kind):
    counts[kind] = counts.get(kind, 0) + 1


def close(actual, reference, relative=2e-12, absolute=0):
    rounded = float(reference)
    assert math.isfinite(actual) and math.isfinite(rounded)
    assert abs(actual - rounded) <= absolute + relative * abs(rounded), (actual, rounded)


for source in data['sources']:
    assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() == source['sha256']

for row in data['fixtures']['cooling']:
    initial, equilibrium, time, rate = (mp.mpf(row[key]) for key in ('initial', 'equilibrium', 'time', 'rate'))
    exact = equilibrium + (initial - equilibrium) * mp.exp(-rate * time)
    close(row['result']['state'], exact)
    close(row['result']['derivative'], -rate * (exact - equilibrium), absolute=3e-10)
    checked('highPrecisionCooling')

for row in data['fixtures']['logistic']:
    initial, capacity, time, rate = (mp.mpf(row[key]) for key in ('initial', 'capacity', 'time', 'rate'))
    exact = capacity * initial / (initial + (capacity - initial) * mp.exp(-rate * time))
    close(row['result']['state'], exact, relative=3e-12, absolute=math.ulp(0.0))
    checked('highPrecisionLogistic')

for row in data['fixtures']['matrices']:
    reference = mp.expm(mp.matrix(row['matrix']) * mp.mpf(row['time']))
    for i in range(2):
        for j in range(2):
            close(row['result'][i][j], reference[i, j], absolute=2e-12)
    checked('independentMatrixExponential')

for row in data['fixtures']['forcing']:
    time, switch = row['time'], row['switchTime']
    first = quad(lambda s: 0.1 * row['firstPower'] * math.exp(-0.2 * (time - s)), 0, min(time, switch), epsabs=1e-12)[0]
    second = quad(lambda s: 0.1 * row['secondPower'] * math.exp(-0.2 * (time - s)), min(time, switch), time, epsabs=1e-12)[0]
    close(row['result']['state'], row['initial'] * math.exp(-0.2 * time) + first + second)
    close(row['result']['firstContribution'], first)
    close(row['result']['secondContribution'], second)
    checked('forcingQuadrature')


def exact_step(method, time, state, width):
    # Rational Butcher table evaluation with dyadic input, separately from JS control flow.
    table = {
        'euler': ([F(0)], [[]], [F(1)]),
        'midpoint': ([F(0), F(1, 2)], [[], [F(1, 2)]], [F(0), F(1)]),
        'rk4': ([F(0), F(1, 2), F(1, 2), F(1)], [[], [F(1, 2)], [F(0), F(1, 2)], [F(0), F(0), F(1)]], [F(1, 6), F(1, 3), F(1, 3), F(1, 6)]),
    }
    c, a, b = table[method]
    slopes, stages = [], []
    for index, offset in enumerate(c):
        stage_time = time + offset * width
        stage_state = state + width * sum((coefficient * slopes[j] for j, coefficient in enumerate(a[index])), F(0))
        slope = stage_time - 2 * stage_state
        slopes.append(slope)
        stages.append((stage_time, stage_state, slope))
    return state + width * sum((weight * slope for weight, slope in zip(b, slopes)), F(0)), stages


for row in data['fixtures']['stages']:
    exact, stages = exact_step(row['method'], F(row['time']), F(row['state']), F(row['step']))
    close(row['result']['state'], exact, absolute=2e-15)
    for actual, expected in zip(row['result']['stages'], stages):
        for key, value in zip(('time', 'state', 'slope'), expected):
            assert actual[key] == float(value), (key, actual, expected)
    checked('exactNonautonomousStages')

errors = {}
for row in data['fixtures']['integrations']:
    result = row['result']
    assert result['status'] == 'reached the requested horizon'
    assert result['endTime'] == 1.7
    reference = 1.7 / 2 - 0.25 + 1.75 * math.exp(-3.4)
    error = abs(result['history'][-1]['state'] - reference)
    errors.setdefault(row['method'], []).append(error)
    for previous, current in zip(result['history'], result['history'][1:]):
        exact, _ = exact_step(row['method'], F(previous['time']), F(previous['state']), F(current['step']))
        close(current['state'], exact, absolute=4e-15)
    checked('nonautonomousEqualHorizon')
for values in errors.values():
    assert values[2] < values[1] < values[0]

namespaces = {}
for name, example in data['examples'].items():
    output = io.StringIO()
    namespace = {}
    with contextlib.redirect_stdout(output):
        exec(compile(example['code'], f'actual-ode-{name}.py', 'exec'), namespace)
    assert output.getvalue().strip() == example['expected'].strip(), (name, output.getvalue(), example['expected'])
    namespaces[name] = namespace
    checked('exactDisplayedPrograms')

# Changed calls through the displayed helpers, not merely separate copies.
for duration in (0, 0.125, 1.25):
    for mass in (0.5, 2):
        for initial in ([2, -1], [-1, 3]):
            actual = namespaces['oscillator']['motion'](duration, mass, 0, 8, initial)
            frequency = math.sqrt(8 / mass)
            q = initial[0] * math.cos(frequency * duration) + initial[1] * math.sin(frequency * duration) / frequency
            v = -initial[0] * frequency * math.sin(frequency * duration) + initial[1] * math.cos(frequency * duration)
            np.testing.assert_allclose(actual, [q, v, (mass * v * v + 8 * q * q) / 2, 0], rtol=3e-13, atol=2e-13)
            checked('changedNativeMotion')
for duration in (0, 2**-20, 0.625, 2):
    for matrix in (np.zeros((2, 2)), np.array([[0, 2], [0, 0]]), np.diag([-0.5, 0])):
        inputs = np.array([[1, 0], [-2, 3]], dtype=float)
        transition, response = namespaces['heldInput']['held_input_matrices'](matrix, inputs, duration)
        reference = mp.expm(mp.matrix(matrix.tolist()) * duration)
        np.testing.assert_allclose(transition, np.array(reference.tolist(), dtype=float), rtol=2e-12, atol=1e-14)
        for i in range(2):
            for j in range(2):
                integral = quad(lambda t: float((mp.expm(mp.matrix(matrix.tolist()) * t) * mp.matrix(inputs.tolist()))[i, j]), 0, duration, epsabs=1e-12)[0]
                close(float(response[i, j]), integral, absolute=2e-14)
        checked('changedNativeHeldInput')
for initial in ((-2, 3), (1, -4), (0, 2)):
    coefficients = namespaces['series']['coefficients'](*initial, 24)
    # Airy basis is independent of the power-coefficient recurrence.
    basis = mp.matrix([[mp.airyai(0), mp.airybi(0)], [mp.airyai(0, 1), mp.airybi(0, 1)]])
    weights = mp.lu_solve(basis, mp.matrix(initial))
    for time in (-0.75, 0.375, 0.75):
        reference = weights[0] * mp.airyai(time) + weights[1] * mp.airybi(time)
        close(namespaces['series']['evaluate'](coefficients, time), reference, absolute=2e-13)
        checked('changedNativeSeriesAiry')
for method in ('euler', 'midpoint', 'rk4'):
    history, status = namespaces['fixedSteps']['integrate'](method, lambda t, y: t - 2 * y, 1.5, 1.7, 0.15)
    matching = next(row for row in data['fixtures']['integrations'] if row['method'] == method and row['step'] == 0.15)
    assert status == 'reached horizon'
    np.testing.assert_allclose(history, [[row['time'], row['state']] for row in matching['result']['history']], rtol=2e-13, atol=2e-14)
    checked('changedNativeStepHelper')
threshold = namespaces['events']['threshold']
result = solve_ivp(lambda t, y: -0.4 * y, (0, 10), [80.0], events=threshold, rtol=1e-10, atol=1e-12)
assert result.status == 1 and result.success
close(result.t_events[0][0], math.log(8) / 0.4, relative=2e-9)
assert result.t[-1] < 10
checked('changedNativeEvent')

result = {'checkedAt': datetime.now(timezone.utc).isoformat(), 'passed': True,
          'sources': data['sources'], 'counts': counts, 'conservedAuthorStates': data['conservedStates'],
          'comparedFields': data['comparedFields'], 'maximumAbsoluteConservationChange': data['maximumAbsoluteChange'],
          'invalidInputsRejected': data['rejected'], 'refinementErrors': errors,
          'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'mpmath': mp.__version__}}
(directory / 'results.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2))
