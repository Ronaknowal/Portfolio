"""Run the exact downloadable/displayed CRF program and retain model-bound outputs.

Run from the repository root with the existing scratch/lesson-tools interpreter.
No package installation or prepared-packet mutation occurs.
"""
from pathlib import Path
from contextlib import redirect_stdout
import hashlib
import io
import itertools
import json
import os
import platform
import runpy
import numpy as np
import scipy
import sklearn

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'public/learn-assets/crf'
output = io.StringIO()
original_directory = Path.cwd()
try:
    os.chdir(ASSETS)
    with redirect_stdout(output):
        program = runpy.run_path(str(ASSETS / 'crf_example.py'))
finally:
    os.chdir(original_directory)

expected = ('30.0\n[[0.9      0.1     ]\n [0.133333 0.866667]]\n[0, 1]\n'
            'independent True [0.788856 0.275   ]\nchain True [0.824047 0.3     ]\n'
            'selected chain [0.791892 0.175   ]\n')
assert output.getvalue() == expected
inference = program['infer']
# Different algorithm: explicit path enumeration with direct products, no messages.
cases = 0
for length in [2, 3, 4]:
    for offset in range(8):
        factors = np.array([[1 + (3*t + j + offset) % 7 for j in range(2)] for t in range(length)], dtype=float)
        pairs = np.array([[1 + offset % 3, 2.], [3., 1 + offset % 4]])
        paths = list(itertools.product(range(2), repeat=length))
        masses = np.array([np.prod([factors[t, path[t]] for t in range(length)]) * np.prod([pairs[path[t-1], path[t]] for t in range(1, length)]) for path in paths])
        logz, nodes, edges, best = inference(np.log(factors), np.log(pairs))
        assert np.isclose(np.exp(logz), masses.sum())
        assert np.isclose(masses[paths.index(tuple(best))], masses.max())
        for t in range(length):
            for label in range(2):
                assert np.isclose(nodes[t, label], sum(mass for path, mass in zip(paths, masses) if path[t] == label)/masses.sum())
        for t in range(length-1):
            for i in range(2):
                for j in range(2):
                    assert np.isclose(edges[t, i, j], sum(mass for path, mass in zip(paths, masses) if path[t:t+2] == (i,j))/masses.sum())
        cases += 1

rows = program['rows']
model = program['models']['chain']
weights, transitions, fit = model
development = []
for row in [row for row in rows if row['split'] == 'dev']:
    features = program['vectorizer'].transform(program['features'](row['tokens']))
    logz, nodes, _, prediction = inference(features @ weights, transitions)
    development.append({**row, 'gold': [program['tag_index'](tag) for tag in row['upos']], 'prediction': prediction, 'marginals': nodes.tolist()})
packet = json.loads((ROOT/'docs/teaching/drafts/conditional-random-fields-crf/checked-results.json').read_text())
for current, prepared in zip(development, packet['real']['chain']['dev']['outputs']):
    assert current['id'] == prepared['id'] and current['prediction'] == prepared['prediction']
    assert np.allclose(current['marginals'], prepared['marginals'], atol=1e-12)
test_confusion = np.zeros((3, 3), dtype=int)
for features, gold in program['groups']['test']:
    _, _, _, prediction = inference(features @ weights, transitions)
    np.add.at(test_confusion, (gold, np.array(prediction)), 1)
assert test_confusion.tolist() == [[49, 1, 26], [8, 42, 23], [17, 2, 202]]
assert int(test_confusion.sum()) == 370 and int(np.trace(test_confusion)) == 293

data = {'development': development, 'transitions': transitions.tolist(),
        'environment': {'Python': platform.python_version(), 'NumPy': np.__version__, 'SciPy': scipy.__version__, 'scikit-learn': sklearn.__version__},
        'stdout': output.getvalue(), 'iterations': {name: int(result[2].nit) for name, result in program['models'].items()}}
(ROOT/'src/learn/data/crf-data.json').write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':'))+'\n', encoding='utf-8')
evidence = {'status': 'passed', 'scope': 'Exact complete native program, independent direct path enumeration, and all 40 development records versus prepared calculations',
            'enumeratedCases': cases, 'environment': data['environment'], 'stdout': output.getvalue(), 'optimizerIterations': data['iterations'], 'testConfusionTrueRows': test_confusion.tolist(),
            'files': {str(path.relative_to(ROOT)).replace('\\','/'): hashlib.sha256(path.read_bytes()).hexdigest() for path in [ASSETS/'crf_example.py', ASSETS/'ewt-sequences.json', Path(__file__)]},
            'optionalReferences': 'PyTorch and CRFsuite are annotated external alternatives; neither optional program is embedded, executed or claimed as this result.'}
(ROOT/'docs/teaching/evidence/crf-native.json').write_text(json.dumps(evidence, indent=2)+'\n', encoding='utf-8')
print(json.dumps({'status': 'passed', 'enumeratedCases': cases, 'developmentRecords': len(development), 'environment': data['environment']}))
