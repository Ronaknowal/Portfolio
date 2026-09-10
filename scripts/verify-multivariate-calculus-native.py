"""Independent polynomial, exact-rational, matrix and derivative checks."""
import ast
import contextlib
from datetime import datetime, timezone
from fractions import Fraction
import io
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

root = Path(__file__).resolve().parents[1]
directory = root / 'scratch/multivariate-verification'
fixtures = json.loads((directory / 'fixtures.json').read_text(encoding='utf-8'))
programs = json.loads((directory / 'programs.json').read_text(encoding='utf-8'))
for key, example in programs.items():
    run = subprocess.run([sys.executable, '-c', example['code']], capture_output=True, text=True)
    assert run.returncode == 0, (key, run.stderr)
    assert run.stdout.strip() == example['expected'], (key, run.stdout)

H = np.diag([2., 4.])
for item in fixtures['local']:
    x, y, angle, step = item['input']
    state = item['state']
    point = np.array([x, y])
    u = np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
    displacement = step * u
    f = lambda p: .5 * p @ H @ p
    np.testing.assert_allclose(state['gradient'], H @ point, atol=1e-12)
    np.testing.assert_allclose(state['rate'], point @ H @ u, atol=1e-12)
    np.testing.assert_allclose(state['remainder'], .5*displacement @ H @ displacement, atol=1e-12)
    np.testing.assert_allclose(state['actual'], f(point+displacement), atol=1e-12)
    assert abs(state['rate']) <= np.linalg.norm(H @ point) + 1e-12
    for row in state['slices'][::10]:
        np.testing.assert_allclose(row['actual'], f(point + row['distance']*u), atol=1e-12)
        np.testing.assert_allclose(row['linear'], f(point) + row['distance']*point @ H @ u, atol=1e-12)

for item in fixtures['paths']:
    kind, coefficient = item['input']
    coefficient = Fraction(str(coefficient))
    state = item['state']
    for index, row in enumerate(state['samples']):
        x = Fraction(1, 10**index)
        y = 0 if kind == 'axis' else coefficient * x**(2 if kind == 'parabola' else 1)
        expected = x*x*y / (x**4+y*y)
        np.testing.assert_allclose(row['value'], float(expected), atol=1e-12)
        np.testing.assert_allclose(row['y'], float(y), atol=1e-12)
    limit = coefficient / (1+coefficient**2) if kind == 'parabola' else 0
    np.testing.assert_allclose(state['limit'], float(limit), atol=1e-12)

for state in fixtures['circle']:
    theta = np.deg2rad(state['angle'])
    objective = lambda t: 2*np.cos(t)+np.sin(t)
    direct_rate = (objective(theta+1e-5)-objective(theta-1e-5))/2e-5
    np.testing.assert_allclose(state['rate'], direct_rate, atol=1e-9)
    np.testing.assert_allclose(state['value'], objective(theta), atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(state['point']), 1, atol=1e-12)
    np.testing.assert_allclose(np.dot(state['point'], state['tangent']), 0, atol=1e-12)
    assert abs(state['value']) <= np.sqrt(5) + 1e-12
    projected = np.array([2, 1]) - np.dot([2, 1], state['point']) * np.array(state['point'])
    np.testing.assert_allclose(state['tangentGradient'], projected, atol=1e-12)

# Each polynomial is represented by monomial coefficients; derivatives are generated.
polynomials = {
    'bowl': {(2, 0): 1, (0, 2): 2},
    'tilted': {(2, 0): 1, (1, 1): 2, (0, 2): 2},
    'maximum': {(2, 0): -1, (0, 2): -2},
    'saddle': {(2, 0): 1, (0, 2): -1},
    'flatMinimum': {(4, 0): 1, (0, 4): 1},
    'flatSaddle': {(4, 0): 1, (0, 4): -1},
}
def evaluate(poly, point):
    return sum(coefficient * point[0]**powers[0] * point[1]**powers[1] for powers, coefficient in poly.items())

def derivative(poly, axis):
    derived = {}
    for powers, coefficient in poly.items():
        if powers[axis]:
            exponent = list(powers)
            exponent[axis] -= 1
            derived[tuple(exponent)] = coefficient * powers[axis]
    return derived

for state in fixtures['curvature']:
    poly = polynomials[state['preset']]
    hessian = np.array([[evaluate(derivative(derivative(poly, i), j), [0., 0.]) for j in range(2)] for i in range(2)])
    u = np.array([np.cos(np.deg2rad(state['angle'])), np.sin(np.deg2rad(state['angle']))])
    np.testing.assert_allclose(state['hessian'], hessian, atol=1e-12)
    np.testing.assert_allclose(state['eigenvalues'], np.linalg.eigvalsh(hessian), atol=1e-12)
    np.testing.assert_allclose(state['curvature'], u @ hessian @ u, atol=1e-12)
    for row in state['samples']:
        np.testing.assert_allclose(row['actual'], evaluate(poly, row['t']*u), atol=1e-12)
        np.testing.assert_allclose(row['quadratic'], .5*row['t']**2*u @ hessian @ u, atol=1e-12)

for state in fixtures['descent']:
    operator = np.eye(2) - state['rate']*H
    for row in state['history']:
        point = np.linalg.matrix_power(operator, row['step']) @ np.array([3., -2.])
        np.testing.assert_allclose(row['point'], point, atol=1e-11, rtol=1e-12)
        np.testing.assert_allclose(row['loss'], .5*point @ H @ point, atol=1e-10, rtol=1e-12)
    spectrum = np.linalg.eigvalsh(operator)
    assert state['convergesFromEveryPoint'] == bool(np.max(np.abs(spectrum)) < 1)

tree = ast.parse(programs['automaticDerivative']['code'])
nodes = [node for node in tree.body if isinstance(node, (ast.Import, ast.ClassDef))]
scope = {}
exec(compile(ast.Module(body=nodes, type_ignores=[]), '<actual lesson Dual>', 'exec'), scope)
Dual = scope['Dual']
rng = np.random.default_rng(510)
for x_value, y_value in rng.uniform(-3, 3, size=(100, 2)):
    x = Dual(x_value, [1, 0])
    y = Dual(y_value, [0, 1])
    result = x*x*y+x.sin()
    expected = [2*x_value*y_value+np.cos(x_value), x_value*x_value]
    np.testing.assert_allclose(result.derivative, expected, atol=1e-12)
    # Constants on both sides exercise lifting and reverse operators.
    other = 2*x + x*3 + 4
    np.testing.assert_allclose(other.derivative, [5, 0], atol=1e-12)

# New practice values and the inline graph-normal's orthogonality.
practice_scope = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(programs['practice']['code'], practice_scope)
new_polynomial = {(1, 1): 1, (0, 2): 1}
new_point = np.array([2., -1.])
new_change = np.array([.03, .02])
new_gradient = [evaluate(derivative(new_polynomial, axis), new_point) for axis in range(2)]
np.testing.assert_allclose(practice_scope['gradient'], new_gradient)
np.testing.assert_allclose(practice_scope['gradient'] @ new_change, -.03)
np.testing.assert_allclose(practice_scope['f'](new_point+new_change)-practice_scope['f'](new_point), -.029)
np.testing.assert_allclose(practice_scope['gradient'] @ practice_scope['direction'], .6)
for rate in (.01, .1, .2, 1/3, .4, .8):
    new_operator = np.eye(2)-rate*np.diag([4., 6.])
    assert bool(np.max(np.abs(np.linalg.eigvalsh(new_operator))) < 1) == (0 < rate < 1/3)
assert Fraction(201, 100)**2*Fraction(298, 100)-12 == Fraction(39498, 1000000)
assert Fraction(101, 10)**2/Fraction(51, 10)-20 == Fraction(1, 510)
np.testing.assert_allclose(np.dot([-2, -4, 1], [1, 0, 2]), 0)
np.testing.assert_allclose(np.dot([-2, -4, 1], [0, 1, 4]), 0)
solution = 2*np.array([3., -4.])/5
np.testing.assert_allclose(solution @ solution, 4)
np.testing.assert_allclose(np.dot([3., -4.], solution), 10)
np.testing.assert_allclose((5/4)*2*solution, [3., -4.])
assert evaluate({(2, 0): 1, (0, 4): -1}, [.01, 0]) > 0
assert evaluate({(2, 0): 1, (0, 4): -1}, [0, .01]) < 0

result = {'status': 'passed', 'checkedAt': datetime.now(timezone.utc).isoformat(),
          'python': sys.version.split()[0], 'numpy': np.__version__, 'standalonePrograms': len(programs),
          **{key+'States': len(value) for key, value in fixtures.items()},
          'unseenActualDualCases': 100, 'independentPracticeGroups': 6,
          'oracleMethods': ['quadratic forms and matrix powers', 'exact rational path and physical-unit calculations',
                            'generated polynomial derivatives and NumPy symmetric eigenvalues', 'independent central circle derivative',
                            'actual lesson Dual class with unseen inputs and analytic gradient']}
(directory / 'results.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
