"""Complementary changed contracts, executing definitions from the actual lesson bundle."""
import ast
import contextlib
import io
import json
import sys
from fractions import Fraction
import numpy as np

examples = json.load(sys.stdin)
counts = {}


def definitions(key):
    tree = ast.parse(examples[key]['code'])
    selected = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))]
    namespace = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), key, 'exec'), namespace)
    return namespace


leaf = definitions('regularized-split')['leaf']
for gradients, hessians, lam, alpha in [([3, -1, 2], [0, 1, 2], 2, 1), ([-3, 1], [1, 4], 0, 3), ([2, 2], [0, 0], 2, 0), ([-4, 1], [2, 1], 1, 0)]:
    weight, improvement = leaf(gradients, hessians, lam, alpha)
    G, A = Fraction(sum(gradients)), Fraction(sum(hessians) + lam)
    candidates = [Fraction(0), -(G + alpha) / A, -(G - alpha) / A]
    objective = lambda w: G * w + A * w * w / 2 + alpha * abs(w)
    optimum = min(candidates, key=objective)
    assert abs(weight - float(optimum)) < 1e-12
    assert abs(improvement + float(objective(optimum))) < 1e-12
counts['actual leaf, exact candidate minimization'] = 4

prefix = definitions('ordered-categories')['prefix_encode']
categories, targets, order = ['K', 'L', 'K', 'K', 'L'], [2, -1, 4, 8, 3], [3, 1, 4, 0, 2]
first = prefix(categories, targets, order, prior=.25, strength=2)[0]
for selected in range(5):
    changed = targets.copy()
    changed[selected] += 17
    second = prefix(categories, changed, order, prior=.25, strength=2)[0]
    assert second[selected] == first[selected]
    for row in order[:order.index(selected)]:
        assert second[row] == first[row]
counts['actual prefix causal influence boundary'] = 5

classes = definitions('recursive-regression')
X = np.array([[-2, 1], [-2, 1], [0, 0], [1, 2], [3, 1]], dtype=float)
y = np.array([2, -1, 5, 0, 7], dtype=float)
for depth in [0, 1, 3]:
    tree = classes['RegressionTree'](depth).fit(X, y)
    prediction = tree.predict(X)
    # Orthogonal leaf projection: residual is orthogonal to its fitted vector and the constant.
    assert abs(np.sum(y - prediction)) < 1e-12
    assert abs(np.dot(y - prediction, prediction)) < 1e-12
    with contextlib.redirect_stdout(io.StringIO()):
        fit = classes['GradientBoostingScratch'](3, .4, depth).fit(X, y)
        expected = fit.predict(X)
        fit.fit(X, y)
    np.testing.assert_array_equal(fit.predict(X), expected)
counts['actual recursive projection and refit'] = 3

# Changed multiclass task independently checks that ranges count rounds, not raw trees.
import xgboost as xgb
data = np.array([[a, b] for a in [-2., -1., 0., 1., 2.] for b in [-1., 0., 1.]])
label = np.array([0 if a < 0 else 1 if b <= 0 else 2 for a, b in data])
dtrain = xgb.DMatrix(data, label=label)
booster = xgb.train(dict(objective='multi:softprob', num_class=3, max_depth=2, eta=.3, nthread=1, seed=23), dtrain, num_boost_round=7)
for count in [1, 3, 6]:
    expected = booster[:count].predict(dtrain, output_margin=True)
    actual = booster.predict(dtrain, output_margin=True, iteration_range=(0, count))
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    assert len(booster[:count].get_dump()) == 3 * count
counts['actual XGBoost multiclass round and slice contract'] = 3
print(json.dumps({'counts': counts, 'xgboost': xgb.__version__, 'scope': 'Actual stored Python definitions; changed data and exact/orthogonal/slice oracles. Author stdout suite reused, not repeated.'}))
