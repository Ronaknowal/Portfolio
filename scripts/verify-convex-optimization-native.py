"""Independent geometry, matrix-power, scalar candidate and solver oracles."""
import json
from pathlib import Path
import numpy as np

fixtures = json.loads(Path('scratch/convex-optimization-review/model-fixtures.json').read_text())
check = np.testing.assert_allclose

for state in fixtures['chords']:
    left, right, theta = state['left'], state['right'], state['fraction']
    x = (1 - theta) * left + theta * right
    functions = {
        'quadratic': lambda value: value**2,
        'absolute': abs,
        'quartic': lambda value: value**4 / 4,
        'doubleWell': lambda value: value**4 / 4 - value**2,
    }
    f = functions[state['preset']]
    check(state['value'], f(x), atol=1e-12)
    check(state['gap'], (1 - theta) * f(left) + theta * f(right) - f(x), atol=1e-12)
    if state['preset'] == 'quadratic':
        check(state['gap'], theta * (1 - theta) * (right - left)**2, atol=1e-12)


def project_segment(point, first, second):
    displacement = second - first
    length_squared = displacement @ displacement
    if length_squared == 0:
        return first.copy()
    fraction = np.clip((point - first) @ displacement / length_squared, 0, 1)
    return first + fraction * displacement


for state in fixtures['allocations']:
    target = np.array([4., 3.])
    budget = state['budget']
    vertices = np.array([[0., 0.], [budget, 0.], [0., budget]])
    candidates = [project_segment(target, first, second) for first, second in zip(vertices, np.roll(vertices, -1, axis=0))]
    if target.sum() <= budget:
        candidates.append(target)
    optimum = min(candidates, key=lambda point: np.sum((point - target)**2))
    check(state['optimum'], optimum, atol=1e-12)
    point = np.array(state['candidate'])
    gradient = point - target
    value = np.sum(gradient**2) / 2
    lower = value + np.min((vertices - point) @ gradient)
    check(state['lowerBound'], lower, atol=1e-12)
    optimum_value = np.sum((optimum - target)**2) / 2
    check(state['optimumValue'], optimum_value, atol=1e-12)
    assert lower <= optimum_value + 1e-12
    if state['feasible']:
        assert state['gap'] + 1e-12 >= value - optimum_value
    else:
        assert state['gap'] is None

maximum_contour_error = 0.0
for state in fixtures['ridge']:
    X = np.array(state['rows'])
    y = np.array([1., 2., 2.])
    penalty = state['penalty']
    H = 2 * (X.T @ X + penalty * np.eye(2))
    eigenvalues = np.linalg.eigvalsh(H)
    check([state['smallestCurvature'], state['largestCurvature']], eigenvalues, atol=1e-12)
    augmented = np.vstack([X, np.sqrt(penalty) * np.eye(2)])
    optimum = np.linalg.lstsq(augmented, np.concatenate([y, np.zeros(2)]), rcond=None)[0]
    check(state['optimum'], optimum, atol=1e-12)
    def objective(point):
        return np.sum((X @ point - y)**2) + penalty * np.sum(point**2)
    best = objective(optimum)
    operator = np.eye(2) - state['step'] * H
    for frame in state['path']:
        # Independent closed recurrence via a matrix power, not the model loop.
        point = optimum + np.linalg.matrix_power(operator, frame['iteration']) @ (np.array([-1., 2.]) - optimum)
        check(frame['weights'], point, atol=1e-9, rtol=1e-10)
        check(frame['error'], max(0., objective(point) - best), atol=1e-8, rtol=1e-10)
    if state['unique']:
        assert state['lowerBound'] <= best + 1e-9
    for contour in state['contours']:
        for point in contour['points']:
            error = abs(objective(np.array(point)) - best - contour['excess'])
            maximum_contour_error = max(maximum_contour_error, error)
            assert error < 1e-10

for state in fixtures['thresholds']:
    value, penalty = state['input'], state['penalty']
    # Minimize separately on each half-line, including their shared boundary.
    candidates = [0., max(0., value - penalty), min(0., value + penalty)]
    objective = lambda point: (point - value)**2 / 2 + penalty * abs(point)
    optimum = min(candidates, key=objective)
    check(state['optimum'], optimum, atol=1e-12)
    check(state['optimumValue'], objective(optimum), atol=1e-12)
    check(state['value'], objective(state['candidate']), atol=1e-12)
    assert state['stationary'] == (abs(state['candidate'] - optimum) < 1e-10)

print(json.dumps({key: len(value) for key, value in fixtures.items()} | {'maximum_contour_error': maximum_contour_error, 'numpy': np.__version__}))
