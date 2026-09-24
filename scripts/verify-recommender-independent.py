"""Finite complementary source-program review; no complete author-suite replay."""
import ast
import contextlib
import io
import json
import sys
import warnings
from fractions import Fraction as F
import numpy as np

examples = json.load(sys.stdin)


def namespace(key, definitions_only=False):
    tree = ast.parse(examples[key]['code'])
    kinds = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)
    if not definitions_only:
        kinds += (ast.Assign, ast.AnnAssign)
    selected = [node for node in tree.body if isinstance(node, kinds)]
    values = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), key, 'exec'), values)
    return values


neighbors = namespace('neighbors')
fallback_cases = []
for matrix, wanted in [([[None, None, None], [0, 2, 4], [1, None, 5]], 12 / 5), ([[None, None], [None, None]], 3.0)]:
    neighbors['ratings'] = np.array(matrix, dtype=float)
    neighbors['observed'] = np.isfinite(neighbors['ratings'])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        actual, selected = neighbors['predict'](0, 1)
    fallback_cases.append({'actual': actual if np.isfinite(actual) else str(actual), 'expected': wanted, 'warnings': [str(w.message) for w in caught], 'passed': bool(np.isfinite(actual) and abs(actual - wanted) < 1e-12)})

ranking = namespace('ranking', True)['metrics']
changed = ranking([3, 1], [2, 0, 1, 3], 5)
assert changed['precision'] == .2 and changed['recall'] == 1 / 3 and changed['average_precision'] == 1 / 3
ideal = 7 + 3 / np.log2(3) + 1 / 2
assert abs(changed['ndcg'] - 7 / ideal) < 1e-12
assert changed['fill_rate'] == .4 and changed['candidate_recall_ceiling'] == 1 / 3
empty = ranking([], [1, 0, 3], 2)
assert empty['recall'] == empty['ndcg'] == empty['fill_rate'] == 0

observed = namespace('observedAls', True)['als']
matrix = np.array([[np.nan, np.nan, np.nan], [2., -1., np.nan], [0., 3., np.nan]])
for rank in [1, 2]:
    p, q, history = observed(matrix, rank=rank, penalty=.7, sweeps=3, seed=17)
    np.testing.assert_array_equal(p[0], np.zeros(rank))
    np.testing.assert_array_equal(q[2], np.zeros(rank))
    assert np.all(np.diff(history) <= 1e-9)

# Change only held-out labels. Every training call and selection must remain identical.
experiment = namespace('evaluation')
original_fit = experiment['fit_mf']
calls = []


def recording_fit(records, *args, **kwargs):
    result = original_fit(records, *args, **kwargs)
    calls.append({'records': records.copy(), 'kwargs': kwargs.copy(), 'p': result['p'].copy(), 'q': result['q'].copy()})
    return result


experiment['fit_mf'] = recording_fit
with contextlib.redirect_stdout(io.StringIO()) as first_output:
    experiment['evaluate'](cutoff=19)
first = calls.copy()
calls.clear()
experiment['events'] = [(u, i, -8.0 if time >= 26 else rating, time) for u, i, rating, time in experiment['events']]
with contextlib.redirect_stdout(io.StringIO()) as second_output:
    experiment['evaluate'](cutoff=19)
assert len(first) == len(calls) == 7
assert first_output.getvalue().splitlines()[0] == second_output.getvalue().splitlines()[0]
for before, after in zip(first, calls):
    assert before['records'] == after['records'] and before['kwargs'] == after['kwargs']
    np.testing.assert_array_equal(before['p'], after['p'])
    np.testing.assert_array_equal(before['q'], after['q'])

print(json.dumps({'counts': {'changed actual graded/empty metrics': 2, 'actual observed ALS empty rows/columns': 2, 'changed-cutoff test-label isolation': 1}, 'fallbackCases': fallback_cases, 'isolation': {'fitsCompared': 7, 'selected': first_output.getvalue().splitlines()[0], 'testRatingsChangedOnly': True}, 'limits': 'Actual stored helpers; two changed cutoff experiments, not the complete author benchmark/library suite.'}))
