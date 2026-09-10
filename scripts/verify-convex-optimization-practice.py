"""Independent finite certificates and changed-input practice checks."""
from fractions import Fraction
import itertools
import json
from pathlib import Path
import numpy as np
import cvxpy as cp

options = dict(solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
checks = {}

# Weighted projection: independent generic constrained solve, then an exact
# rational supporting-plane certificate at every triangle vertex.
w = cp.Variable(2)
problem = cp.Problem(cp.Minimize(cp.square(w[0] - 3) + 4 * cp.square(w[1] - 1)), [w >= 0, cp.sum(w) <= 2])
problem.solve(**options)
assert problem.status == cp.OPTIMAL
np.testing.assert_allclose(w.value, [1.4, 0.6], atol=1e-7, rtol=0)
point = [Fraction(7, 5), Fraction(3, 5)]
gradient = [2 * (point[0] - 3), 8 * (point[1] - 1)]
assert (point[0] - 3) ** 2 + 4 * (point[1] - 1) ** 2 == Fraction(16, 5)
for vertex in [(0, 0), (2, 0), (0, 2)]:
    assert sum(g * (v - x) for g, v, x in zip(gradient, vertex, point)) >= 0
checks['weighted_allocation'] = True

# Matrix powers independently verify the analytical coordinate recurrence.
H = np.diag([1., 100.])
for eta in [0.01, 0.021, 1e-15]:
    for k in range(25):
        actual = np.linalg.matrix_power(np.eye(2) - eta * H, k) @ np.ones(2)
        np.testing.assert_allclose(actual, [(1-eta) ** k, (1-100*eta) ** k], rtol=1e-12, atol=1e-12)
        value = actual @ H @ actual / 2
        lower = value - np.linalg.norm(H @ actual) ** 2 / 2
        assert lower <= 1e-10
checks['curvature_powers'] = 75

# Enumerate boundary, sign and curvature cases through an independent convex
# solver instead of calling the lesson's scalar formula as both computation
# and oracle.
threshold_checks = 0
for curvature, penalty, input_value in itertools.product([0.5, 1., 4.], [0., 0.5, 2.], [-3., -0.5, 0., 0.5, 3.]):
    x = cp.Variable()
    problem = cp.Problem(cp.Minimize(curvature / 2 * cp.square(x - input_value) + penalty * cp.abs(x)))
    problem.solve(**options)
    assert problem.status == cp.OPTIMAL
    expected = np.sign(input_value) * max(abs(input_value) - penalty / curvature, 0)
    np.testing.assert_allclose(x.value, expected, atol=3e-5, rtol=0)
    threshold_checks += 1
checks['weighted_thresholds'] = threshold_checks

for limit, expected in [(None, [6, 4]), (5, [5, 5])]:
    c, ceiling = cp.Variable(), cp.Variable()
    observed = np.array([2., 3., 10.])
    constraints = [c - observed <= ceiling, observed - c <= ceiling]
    if limit is not None:
        constraints.append(c <= limit)
    problem = cp.Problem(cp.Minimize(ceiling), constraints)
    problem.solve(**options)
    assert problem.status == cp.OPTIMAL
    np.testing.assert_allclose([c.value, ceiling.value], expected, atol=1e-7, rtol=0)
    np.testing.assert_allclose(np.max(np.abs(float(c.value) - observed)), expected[1], atol=1e-7)
checks['changed_robust_models'] = 2

# Full-space TV certificates, computed with exact fractions. This checks the
# plotted six-sample figure and the independent four-sample exercise.
tv_cases = [
    ([Fraction(1, 5), Fraction(-1, 10), Fraction(1, 10), 3, Fraction(16, 5), Fraction(14, 5)],
     [Fraction(1, 6)] * 3 + [Fraction(29, 10)] * 3,
     [Fraction(-1, 9), Fraction(7, 9), 1, Fraction(2, 3), Fraction(-1, 3)], Fraction(3, 10), Fraction(137, 150)),
    ([0, 0, 4, 4], [Fraction(1, 4)] * 2 + [Fraction(15, 4)] * 2,
     [Fraction(1, 2), 1, Fraction(1, 2)], Fraction(1, 2), Fraction(15, 8)),
]
for observed, fitted, q, penalty, expected_cost in tv_cases:
    differences = [b - a for a, b in zip(fitted, fitted[1:])]
    assert all(-1 <= value <= 1 for value in q)
    assert all(value == (1 if difference > 0 else -1) for difference, value in zip(differences, q) if difference)
    transpose_action = [-q[0]] + [a - b for a, b in zip(q, q[1:])] + [q[-1]]
    assert all(x - y + penalty * derivative == 0 for x, y, derivative in zip(fitted, observed, transpose_action))
    cost = sum((x - y) ** 2 for x, y in zip(fitted, observed)) / 2 + penalty * sum(map(abs, differences))
    assert cost == expected_cost
    x = cp.Variable(len(observed))
    problem = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(x - np.array(observed, dtype=float)) + float(penalty) * cp.norm1(cp.diff(x))))
    problem.solve(**options)
    np.testing.assert_allclose(x.value, np.array(fitted, dtype=float), atol=1e-7, rtol=0)
checks['exact_tv_certificates'] = 2
Path('scratch/convex-optimization-review/practice-results.json').write_text(json.dumps(checks, indent=2))
print(json.dumps(checks))
