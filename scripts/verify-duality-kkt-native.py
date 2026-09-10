"""Independent library/optimization oracles for the actual learner helpers and visual states."""
import contextlib
import io
import json
from pathlib import Path
import sys

import cvxpy as cp
import numpy as np
import scipy
from scipy.optimize import minimize_scalar

folder = Path(sys.argv[1])
cases = json.loads((folder / 'model-cases.json').read_text())
examples = json.loads((folder / 'examples.json').read_text())
solver_options = dict(solver='CLARABEL', tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)


def near(actual, expected, tolerance=1e-8):
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)


def load_program(name):
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(examples[name]['code'], namespace)
    return namespace


projection = load_program('projection')
resource = load_program('resource')
rng = np.random.default_rng(9102026)
library_projection = 0
for dimension in [1, 2, 3, 7]:
    for repetition in range(16):
        target = rng.normal(size=dimension) * 3
        normal = rng.normal(size=dimension)
        # Include slack, exact-boundary and violated target states.
        budget = float(normal @ target + [-3, -0.5, 0, 2][repetition % 4])
        point, multiplier = projection['project_halfspace'](target, normal, budget)
        variable = cp.Variable(dimension)
        constraint = normal @ variable <= budget
        problem = cp.Problem(cp.Minimize(cp.sum_squares(variable - target)), [constraint])
        problem.solve(**solver_options)
        assert problem.status == cp.OPTIMAL
        near(np.sum((point - target)**2), problem.value, 3e-7)
        near(point, variable.value, 2e-5)
        assert normal @ point <= budget + 1e-10
        near(2 * (point - target) + multiplier * normal, np.zeros(dimension))
        near(projection['dual_value'](target, normal, budget, multiplier), problem.value, 3e-7)
        for _ in range(4):
            feasible = point + rng.normal(size=dimension)
            if normal @ feasible > budget:
                feasible -= ((normal @ feasible - budget) / (normal @ normal) + 0.1) * normal
            assert np.sum((feasible - target)**2) >= np.sum((point - target)**2) - 1e-9
        library_projection += 1

# Independent linear-system solution of the quadratic L minimization.
for state in cases['projection']:
    candidate = np.array(state['candidate'])
    target = np.array([3., 4.])
    multiplier = state['multiplier']
    minimum = np.linalg.solve(2 * np.eye(2), 2 * target - multiplier * np.ones(2))
    value = np.sum((minimum - target)**2) + multiplier * (minimum.sum() - state['budget'])
    near(state['minimizer'], minimum)
    near(state['dualValue'], value)
    near(state['lagrangian'], np.sum((candidate - target)**2) + multiplier * (candidate.sum() - state['budget']))
    near(state['formalDifference'], state['minimizationGap'] + state['complementarityGap'])
    valid = candidate.sum() <= state['budget'] and multiplier >= 0
    assert (state['certifiedGap'] is not None) == valid
    if valid:
        assert state['dualValue'] <= state['optimalValue'] + 1e-10
        assert state['objective'] >= state['optimalValue'] - 1e-10
        assert state['certifiedGap'] >= -1e-10

# Scalar convex minimization uses a numerical bounded search plus boundary comparison.
scalar_optima = {}
for state in cases['scalar']:
    center = state['center']
    if center not in scalar_optima:
        objective = lambda x: (x - center)**2
        result = minimize_scalar(objective, bounds=(0, 8), method='bounded', options={'xatol': 1e-12})
        best = min([(result.fun, result.x), (objective(0), 0)])
        scalar_optima[center] = best
    value, point = scalar_optima[center]
    near(state['optimalValue'], value)
    near(state['optimum'], point, 2e-7)
    x, price = state['candidate'], state['multiplier']
    checks = [x >= 0, price >= 0, 2 * (x - center) == price, price == 0 or x == 0]
    assert state['allConditions'] == all(checks)
    if all(checks):
        near(state['objective'], value)
        near(state['dualValue'], value)

# Changed scalar programs are independently solved as convex QPs/LPs at representative inputs.
value_cache = {}
for state in cases['sensitivity']:
    for argument, field in [(state['base'], 'originalValue'), (state['base'] + state['change'], 'changedValue')]:
        key = (state['mode'], argument)
        if key not in value_cache:
            if state['mode'] == 'quadratic':
                x = cp.Variable(2)
                problem = cp.Problem(cp.Minimize(cp.sum_squares(x - [3., 4.])), [cp.sum(x) <= argument])
            else:
                x = cp.Variable()
                problem = cp.Problem(cp.Minimize(x), [x >= 0, x >= -argument])
            problem.solve(**solver_options)
            assert problem.status == cp.OPTIMAL
            value_cache[key] = problem.value
        near(state[field], value_cache[key], 3e-7)
    assert state['changedValue'] >= state['supportingValue'] - 1e-10
    if state['mode'] == 'kink' and state['base'] == 0:
        assert state['derivative'] is None
        assert state['leftDerivative'] == -1 and state['rightDerivative'] == 0

resource_optima = {}
local_cache = {}
for state in cases['resource']:
    budget = state['budget']
    if budget not in resource_optima:
        x = cp.Variable(2)
        constraints = [x >= 0, cp.sum(x) <= budget]
        problem = cp.Problem(cp.Minimize(cp.square(x[0] - 3) + 2 * cp.square(x[1] - 4)), constraints)
        problem.solve(**solver_options)
        assert problem.status == cp.OPTIMAL
        resource_optima[budget] = problem.value
    optimum = resource_optima[budget]
    near(state['optimum']['objective'], optimum, 3e-7)
    native_rows = resource['price_iterations'](budget, state['rate'], state['updates'], state['initialPrice'])
    for frame, row in zip(state['frames'], native_rows):
        price = frame['price']
        near(price, row[0])
        near(frame['allocation'], row[1])
        near(frame['dualValue'], row[3])
        near(frame['repaired'], row[4])
        # A library scalar minimization, not the clamp expression, checks each local response.
        if price not in local_cache:
            choices = []
            for weight, target in [(1, 3), (2, 4)]:
                objective = lambda x: weight * (x - target)**2 + price * x
                result = minimize_scalar(objective, bounds=(0, 10), method='bounded', options={'xatol': 1e-12})
                choices.append(min([(result.fun, result.x), (objective(0), 0)]))
            local_cache[price] = choices
        local = local_cache[price]
        near(frame['allocation'], [choice[1] for choice in local], 3e-7)
        near(frame['dualValue'], sum(choice[0] for choice in local) - price * budget)
        assert frame['dualValue'] <= optimum + 1e-7
        assert sum(frame['repaired']) <= budget + 1e-12
        assert min(frame['repaired']) >= 0
        assert frame['repairedObjective'] >= optimum - 1e-7
        near(frame['certificateGap'], frame['repairedObjective'] - frame['dualValue'])
        assert frame['certificateGap'] >= 0
        near(frame['certificateGap'], resource['stable_repair_gap'](price, budget, row[1], row[4]))

# Cancellation regression: preserve small positive mathematical gaps instead of clamping a subtraction.
from decimal import Decimal, localcontext
cancellation_cases = 0
for state in cases['resource']:
    if state['rate'] != 1.75 or state['initialPrice'] != 0 or state['budget'] not in [2.75, 3.25, 4]:
        continue
    for frame in state['frames']:
        if frame['subtractedGap'] >= 0:
            continue
        with localcontext() as context:
            context.prec = 70
            price, budget = Decimal.from_float(frame['price']), Decimal.from_float(state['budget'])
            a = [max(Decimal(0), Decimal(3) - price/2), max(Decimal(0), Decimal(4) - price/4)]
            repaired = [min(a[0], budget), Decimal(0)]
            repaired[1] = min(a[1], budget - repaired[0])
            objective = (repaired[0]-3)**2 + 2*(repaired[1]-4)**2
            dual = (a[0]-3)**2 + 2*(a[1]-4)**2 + price*(sum(a)-budget)
            assert objective - dual >= 0
        assert frame['certificateGap'] >= 0
        cancellation_cases += 1

# Exact rational transfer checks, independently substituted into objective/constraints.
from fractions import Fraction as F
x, y, price = F(8, 5), F(1, 5), F(4, 5)
assert x + 2*y == 2
assert (x - 2)**2 + (y - 1)**2 == F(4, 5)
assert 2*(x - 2) + price == 0 and 2*(y - 1) + 2*price == 0
assert F(1)**2 + F(-1)**2 == 2
assert F(3, 2)**2 / 2 == F(9, 8)
assert F(5, 2)**2 / 2 == F(25, 8)
assert (F(2)-3)**2 + 2*(F(7, 2)-4)**2 + 2*F(1, 2) == F(5, 2)
assert (F(2)-3)**2 + 2*(F(3)-4)**2 == 3

invalid = 0
for arguments in [([], [], 0), ([1, 2], [1], 0), ([1], [0], 0), ([float('nan')], [1], 0), ([1], [1], float('inf'))]:
    try:
        projection['project_halfspace'](*arguments)
    except (ValueError, TypeError):
        invalid += 1
    else:
        raise AssertionError('invalid projection accepted')
for arguments in [(5, -1, 1), (5, 1, True), (5, 1, 1.5), (5, 1, 1, -1), (-1, 1, 1), (5, float('nan'), 1)]:
    try:
        resource['price_iterations'](*arguments)
    except (ValueError, TypeError):
        invalid += 1
    else:
        raise AssertionError('invalid iteration accepted')

print(json.dumps({'python': sys.version.split()[0], 'numpy': np.__version__, 'scipy': scipy.__version__,
                  'cvxpy': cp.__version__, 'generalProjectionSolverComparisons': library_projection,
                  'sensitivitySolverPrograms': len(value_cache), 'resourcePrimalPrograms': len(resource_optima),
                  'localScalarMinimizations': 2 * len(local_cache), 'invalidNativeInputs': invalid,
                  'practiceContracts': 6, 'negativeSubtractionRegressions': cancellation_cases}))
