"""Check proposed SVM fixtures independently before authoring their visuals."""
import hashlib
import itertools
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'scratch/support-vector-machines-verification'
OUT.mkdir(parents=True, exist_ok=True)
counts = {'support_motion': 0, 'soft_pairs': 0, 'xor_kernels': 0, 'svr_tubes': 0}
largest = 0.0


def close(actual, expected, tolerance=2e-6):
    global largest
    error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
    largest = max(largest, error)
    assert error <= tolerance, (actual, expected, error)


for moved in [.25, .5, .75, 1., 1.5, 2., 3.]:
    x = np.array([-2., -1., 1., moved])[:, None]
    y = np.array([-1., -1., 1., 1.])
    nearest = min(moved, 1.)
    w = 2 / (nearest + 1)
    b = (1 - nearest) / (nearest + 1)
    fitted = SVC(kernel='linear', C=1000, tol=1e-10).fit(x, y)
    close([fitted.coef_[0, 0], fitted.intercept_[0]], [w, b])
    assert np.min(y * (x[:, 0] * w + b)) >= 1 - 1e-12
    counts['support_motion'] += 1

for c, gap in itertools.product([.1, .25, .5, 1., 2.], [0., 1.]):
    x = np.array([-gap, gap])
    y = np.array([-1., 1.])
    alpha = c if gap == 0 else min(c, .5)
    w = 2 * gap * alpha
    bias_limit = max(0., 1 - gap * w)
    dual = 2 * alpha - .5 * w ** 2
    for b in [-bias_limit, 0., bias_limit]:
        primal = .5 * w ** 2 + c * np.maximum(0, 1 - y * (w * x + b)).sum()
        close(primal, dual, 1e-12)
        # The independent primal optimization includes both intercept and weight.
        fit = minimize(lambda v: .5 * v[0] ** 2 + c * np.maximum(0, 1 - y * (v[0] * x + v[1])).sum(),
                       [w, b], method='Powell', options={'xtol': 1e-11, 'ftol': 1e-11})
        close(fit.fun, primal)
        counts['soft_pairs'] += 1

x = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
y = np.prod(x, axis=1)
for c, gamma, kernel in itertools.product([.05, .25, 1., 4.], [.1, .5, 2.], ['linear', 'poly', 'rbf']):
    eigenvalue = 0 if kernel == 'linear' else 8 if kernel == 'poly' else (-np.expm1(-4 * gamma)) ** 2
    alpha = c if eigenvalue == 0 else min(c, 1 / eigenvalue)
    if kernel == 'linear':
        gram = x @ x.T
    elif kernel == 'poly':
        gram = (x @ x.T) ** 2
    else:
        gram = np.exp(-gamma * np.sum((x[:, None] - x[None, :]) ** 2, axis=2))
    q = gram * y[:, None] * y[None, :]
    expected_dual = 4 * alpha - 2 * eigenvalue * alpha ** 2
    fit = minimize(lambda a: .5 * a @ q @ a - a.sum(), np.zeros(4),
                   jac=lambda a: q @ a - 1,
                   bounds=[(0, c)] * 4,
                   constraints={'type': 'eq', 'fun': lambda a: a @ y, 'jac': lambda a: y},
                   method='SLSQP', options={'ftol': 1e-12, 'maxiter': 500})
    assert fit.success, fit.message
    close(-fit.fun, expected_dual, 1e-8)
    close(gram @ (fit.x * y), gram @ (alpha * y), 2e-6)
    counts['xor_kernels'] += 1

for amplitude, epsilon, c in itertools.product([1., 2., 3.], [0., .25, 1., 2.], [.1, .5, 1., 2.]):
    slope = min(2 * c, max(amplitude - epsilon, 0.))
    objective = lambda w: .5 * w ** 2 + 2 * c * max(0., abs(amplitude - w) - epsilon)
    oracle = minimize_scalar(objective, bounds=(-1, 5), method='bounded', options={'xatol': 1e-11})
    close(objective(slope), oracle.fun, 5e-7)
    counts['svr_tubes'] += 1

archive = ROOT / 'docs/teaching/archive/support-vector-machines-before-rewrite'
original_runs = []
for name, snippets in [('original_smo_demos', [1, 2]), ('original_library_fragments', [3, 4, 5])]:
    source = '\n\n'.join((archive / f'original-snippet-{i}.py').read_text(encoding='utf8') for i in snippets)
    try:
        run = subprocess.run([sys.executable, '-c', source], capture_output=True, text=True, timeout=45)
        original_runs.append({'name': name, 'snippetIndices': snippets, 'sha256': hashlib.sha256(source.encode()).hexdigest(),
                              'exitCode': run.returncode, 'stdout': run.stdout, 'stderr': run.stderr})
    except subprocess.TimeoutExpired:
        original_runs.append({'name': name, 'status': '45-second external timeout; no unbounded original loop allowed'})

result = {'checkedAt': datetime.now(timezone.utc).isoformat(), 'counts': counts,
          'maximumAbsoluteError': largest, 'originalRuns': original_runs,
          'scope': 'Design fixtures and original fragment executions only; no claim about future implementation or browser.'}
(OUT / 'design-fixtures.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
print(json.dumps(result, indent=2))
