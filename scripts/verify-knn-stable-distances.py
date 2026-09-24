"""Check the actual amended displayed helpers at small/large finite scales."""
import ast
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
directory = ROOT / 'scratch/tree-knn-input-contracts'
programs = json.loads((directory / 'knn-programs.json').read_text(encoding='utf-8'))
definitions = {}
for program in programs:
    tree = ast.parse(program['code'])
    selected = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))]
    namespace = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), program['id'], 'exec'), namespace)
    definitions[program['id']] = namespace

LocalNeighbors = definitions['scratch-estimator']['LocalNeighbors']
build_tree = definitions['kd-tree']['build_tree']
nearest = definitions['kd-tree']['nearest']
cases = []
for scale in [1e-300, 1e-200, 1.0, 1e150, 1e200, 1e300]:
    X = np.array([[0., 0.], [scale, 0.], [0., 2 * scale]])
    query = np.array([scale, 0.])
    fitted = LocalNeighbors(k=1).fit(X, np.array(['A', 'B', 'C']))
    assert fitted.predict([query]).tolist() == ['B']
    result, _ = nearest(X, build_tree(X), query)
    assert result == (0., 1)
    intermediate = np.array([.75 * scale, .125 * scale])
    oracle = min((math.hypot(*(point - intermediate)), index) for index, point in enumerate(X))
    result, _ = nearest(X, build_tree(X), intermediate)
    assert result[1] == oracle[1]
    assert math.isclose(result[0], oracle[0], rel_tol=2e-15, abs_tol=0)
    for task, targets in [('classification', ['A', 'B', 'C']), ('regression', [10., 20., 40.])]:
        weighted = LocalNeighbors(k=3, weights='distance', task=task).fit(X, targets)
        indices, weights = next(weighted._query([intermediate]))
        distances = np.array([math.hypot(*(X[index] - intermediate)) for index in indices])
        expected = distances.min() / distances
        expected /= expected.sum()
        assert np.allclose(weights, expected, rtol=2e-15, atol=0)
        assert np.isfinite(weighted.predict_proba([intermediate]) if task == 'classification' else weighted.predict([intermediate])).all()
    cases.append({'kind': 'finite-scale exact match, nearest identity and normalized inverse weights', 'scale': scale})

for name, operation in [
    ('estimator', lambda: LocalNeighbors(k=1).fit([[-1e308]], ['A']).predict([[1e308]])),
    ('KD query', lambda: nearest(np.array([[-1e308]]), build_tree(np.array([[-1e308]])), np.array([1e308]))),
]:
    try:
        operation()
    except ValueError as error:
        assert 'overflow' in str(error)
    else:
        raise AssertionError(f'{name} silently accepted an unrepresentable distance')
    cases.append({'kind': 'unrepresentable distance rejected explicitly', 'helper': name})

previous = json.loads((ROOT / 'docs/teaching/archive/tree-knn-input-amendment/example-runs.json').read_text(encoding='utf-8'))
current = json.loads((ROOT / 'scratch/knn-native/example-runs.json').read_text(encoding='utf-8'))
old_runs = {row['id']: row for row in previous['runs']}
changed = [row['id'] for row in current['runs'] if row['codeSha256'] != old_runs[row['id']]['codeSha256']]
assert sorted(changed) == ['kd-tree', 'scratch-estimator']
for row in current['runs']:
    assert row['expected'] == old_runs[row['id']]['expected']
    if row['id'] not in changed:
        assert row == old_runs[row['id']]

record = {
    'checkedAt': datetime.now(timezone.utc).isoformat(),
    'cases': cases,
    'affectedProgramsActuallyReexecuted': changed,
    'unaffectedExecutionsReused': 9,
    'allDisplayedOutputsUnchanged': True,
    'sourceSha256': hashlib.sha256((ROOT / 'src/learn/data/knn-examples.js').read_bytes()).hexdigest(),
    'oracle': 'Python math.hypot and explicit finite-scale checks on the actual displayed helper ASTs',
}
(directory / 'stable-distances.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print('PASS: six finite scales, two explicit overflow rejections; only two programs reexecuted.')
