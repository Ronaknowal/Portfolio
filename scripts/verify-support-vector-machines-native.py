"""Independent projections, constrained optimization and changed actual solver inputs."""
import contextlib
import io
import json
import runpy
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from sklearn.svm import SVC, SVR

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'scratch/support-vector-machines-verification'
cases = json.loads((OUT / 'model-cases.json').read_text())
counts = Counter()
maximum = 0.


def close(actual, expected, label, tolerance=2e-8):
    global maximum
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    difference = float(np.max(np.abs(a-b), initial=0) / max(1., float(np.max(np.abs(b), initial=0))))
    maximum = max(maximum, difference)
    assert difference <= tolerance, (label, a, b, difference)


for case in cases['geometry']:
    result = case['result']
    weights, bias = np.array(result['w']), result['b']
    for row in result['rows']:
        point = np.array(row['x'])
        # Constrained Euclidean projection, independent of the displayed projection formula.
        answer = minimize(lambda z: .5*np.sum((z-point)**2), point,
                          jac=lambda z: z-point,
                          constraints={'type': 'eq', 'fun': lambda z: weights@z+bias,
                                       'jac': lambda z: weights}, method='SLSQP', options={'ftol': 1e-13})
        assert answer.success
        close(row['projection'], answer.x, 'perpendicular projection')
        close(abs(row['signedDistance']), np.linalg.norm(point-answer.x), 'distance')
    counts['geometry'] += 1

for case in cases['support']:
    rows = case['result']['rows']
    fitted = SVC(kernel='linear', C=1000, tol=1e-10).fit(np.array([row['x'] for row in rows])[:, None], [row['y'] for row in rows])
    close([case['result']['w'], case['result']['b']], [fitted.coef_[0, 0], fitted.intercept_[0]], 'support movement', 1e-7)
    close(case['result']['primal'], case['result']['dual'], 'support certificate')
    counts['support'] += 1

for case in cases['soft']:
    result, c = case['result'], case['c']
    x = np.array([row['x'] for row in result['rows']])
    y = np.array([-1, 1])
    objective = lambda z: .5*z[0]**2 + c*np.maximum(0, 1-y*(x*z[0]+z[1])).sum()
    # Epigraph quadratic program with explicit slack constraints, including free bias.
    solution = minimize(lambda z: .5*z[0]**2+c*(z[2]+z[3]), [0., 0., 1., 1.],
                        bounds=[(None, None), (None, None), (0, None), (0, None)],
                        constraints={'type': 'ineq', 'fun': lambda z: y*(x*z[0]+z[1])+z[2:]-1},
                        method='SLSQP', options={'ftol': 1e-12, 'maxiter': 500})
    assert solution.success
    close(result['primal'], solution.fun, 'soft epigraph optimum')
    close(result['primal'], objective([result['w'], result['b']]), 'selected bias cost')
    close(result['dual'], result['primal'], 'soft dual')
    counts['soft'] += 1

for case in cases['kernels']:
    result, c = case['result'], case['c']
    gram = np.array(result['gram'])
    labels = np.array([1, -1, -1, 1])
    q = gram*labels[:, None]*labels[None, :]
    solution = minimize(lambda a: .5*a@q@a-a.sum(), np.zeros(4), jac=lambda a: q@a-1,
                        bounds=[(0, c)]*4, constraints={'type': 'eq', 'fun': lambda a: a@labels,
                        'jac': lambda a: labels}, method='SLSQP', options={'ftol': 1e-12, 'maxiter': 500})
    assert solution.success
    close(result['dual'], -solution.fun, 'XOR unconstrained-coordinate optimizer')
    assert np.linalg.eigvalsh(gram).min() >= -1e-10
    close(result['primal'], result['dual'], 'XOR certificate')
    if case['kind'] == 'poly':
        phi = lambda point: np.array([point[0]**2, np.sqrt(2)*point[0]*point[1], point[1]**2])
        points = [row['point'] for row in result['contributions']]
        w = sum(result['alpha']*label*phi(point) for label, point in zip(labels, points))
        close(result['score'], w@phi(case['query']), 'explicit feature query')
    counts['kernels'] += 1

for result in cases['pairs']:
    points, labels = np.array(result['points']), np.array(result['labels'])
    old, new = np.array(result['alpha'], dtype=float), np.array(result['after'], dtype=float)
    gram = points@points.T
    dual = lambda a: a.sum()-.5*(a*labels)@gram@(a*labels)
    direction = new-old
    close(new@labels, 0, 'pair balance')
    close(result['afterDual'], dual(new), 'pair objective')
    i, j, c = result['i'], result['j'], result['c']
    # Independent 2-D constrained QP, without using the one-dimensional optimum formula.
    def changed(v):
        a = old.copy()
        a[i], a[j] = v
        return a
    solution = minimize(lambda v: -dual(changed(v)), old[[i, j]], bounds=[(0, c)]*2,
                        constraints={'type': 'eq', 'fun': lambda v: changed(v)@labels},
                        method='SLSQP', options={'ftol': 1e-12, 'maxiter': 300})
    assert solution.success, (solution.message, result)
    close(result['afterDual'], -solution.fun, 'pair constrained optimum', 1e-7)
    counts['pairs'] += 1

for case in cases['tubes']:
    x = np.array([[-1.], [0.], [1.]])
    target = case['amplitude']*x[:, 0]
    fitted = SVR(kernel='linear', epsilon=case['epsilon'], C=case['c'], tol=1e-10).fit(x, target)
    close(case['result']['slope'], fitted.coef_[0, 0], 'SVR slope', 1e-7)
    counts['tubes'] += 1
for case in cases['sequences']:
    expected = Counter(case['word'][i:i+case['size']] for i in range(max(0, len(case['word'])-case['size']+1)))
    assert case['result']['counts'] == expected
    counts['sequences'] += 1

namespace = runpy.run_path(str(ROOT / 'scripts/support-vector-machines-programs/pair-coordinate-solver.py'), run_name='verification')
PairSVM = namespace['PairSVM']
rng = np.random.default_rng(927)
for index in range(36):
    n = 4 + index % 5
    x = rng.normal(size=(n, 2))
    if index % 4 == 0:
        x[1] = x[0]
    y = np.where(np.arange(n) % 2, 1., -1.)
    c, gamma = [.07, .5, 2][index % 3], [.1, .8, 3][index % 3]
    kernel = 'linear' if index % 2 else 'rbf'
    fitted = PairSVM(c=c, kernel=kernel, gamma=gamma, tolerance=1e-7).fit(x, y)
    assert fitted.status == 'gap passed', (index, fitted.status, fitted.primal-fitted.dual)
    gram = fitted.matrix(x, x)
    q = gram*y[:, None]*y[None, :]
    answer = minimize(lambda a: .5*a@q@a-a.sum(), np.zeros(n), jac=lambda a: q@a-1,
                      bounds=[(0, c)]*n, constraints={'type': 'eq', 'fun': lambda a: a@y,
                      'jac': lambda a: y}, method='SLSQP', options={'ftol': 1e-12, 'maxiter': 1000})
    assert answer.success
    close(fitted.dual, -answer.fun, 'changed native dual', 2e-6)
    assert abs(fitted.alpha@y) < 1e-9 and (fitted.alpha >= 0).all() and (fitted.alpha <= c).all()
    counts['changedNativeSolvers'] += 1
repeated = PairSVM(kernel='linear').fit([[-1.], [1.]], [-1., 1.])
repeated.fit([[0.], [0.]], [-1., 1.])
close(repeated.alpha, [1, 1], 'refit and zero curvature')
stopped = PairSVM(max_steps=1).fit([[0, 0], [1, 2], [-1, 1], [2, 1]], [-1, 1, -1, 1])
assert stopped.status == 'step limit' and stopped.primal-stopped.dual > 0
counts['refitAndCap'] = 2

with contextlib.redirect_stdout(io.StringIO()):
    selection = runpy.run_path(str(ROOT / 'scripts/support-vector-machines-programs/moons-validation.py'))
stored = cases['validation']
for (c, gamma, fitted, train, valid), saved in zip(selection['fitted'], stored['models']):
    query = (selection['x'][selection['valid']]-np.array(stored['scaler']['mean']))/np.array(stored['scaler']['scale'])
    distance = np.sum((query[:, None, :]-np.array(saved['support'])[None, :, :])**2, axis=2)
    score = np.exp(-gamma*distance)@np.array(saved['coefficients'])+saved['bias']
    close(score, fitted.decision_function(query), 'saved actual browser predictor')
    close([saved['training'], saved['validation']], [train, valid], 'saved held-out values')
    counts['savedFittedModels'] += 1
assert not set(row['id'] for row in stored['train']) & set(row['id'] for row in stored['validation'])
counts['invalidBrowserInputs'] = cases['invalidCount']
record = {'executedAt': datetime.now(timezone.utc).isoformat(), 'counts': dict(counts),
          'maximumScaledDifference': maximum, 'scope': 'Finite independently optimized teaching states and changed actual native helpers; no general solver runtime or statistical accuracy guarantee.'}
(OUT / 'model-results.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record))
