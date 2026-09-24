"""Execute all four complete displayed manifold programs in an isolated runtime.

--write initially extracts the four Python blocks from the retained manuscript,
records actual output, and writes the topic example module. Subsequent runs
execute that exact production code and compare output. Derived UMAP coordinates
are actual program output, with CSV identity retained; never synthesized.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMBA_NUM_THREADS', '1')
import contextlib
import hashlib
import importlib.metadata
import inspect
import io
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import umap
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / 'docs/teaching/drafts/t-sne-umap-manifold-learning'
ASSETS = ROOT / 'public/learn-assets/manifold-learning'
WORK = ROOT / 'scratch/manifold-learning-examples'
MODULE = ROOT / 'src/learn/data/manifold-examples.js'
WRITE = '--write' in sys.argv
ASSETS.mkdir(parents=True, exist_ok=True)
WORK.mkdir(parents=True, exist_ok=True)
shutil.copyfile(PACKET / 'digits-300.csv', ASSETS / 'digits-300.csv')
shutil.copyfile(ASSETS / 'digits-300.csv', WORK / 'digits-300.csv')
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
metadata = [
    ('digitsAudit', 'inspect_digits.py', 'Audit the complete optical-digits collection', 'Will a two-dimensional nonlinear map retain more original pixel neighbors than PCA?'),
    ('umapFit', 'inspect_umap.py', 'Fit actual UMAP coordinates', 'How many ten-neighbor selections survive this fitted UMAP map?'),
    ('heldoutTransform', 'compare_transforms.py', 'Transform held-out images in a fitted reference', 'Will ten UMAP coordinates improve this split’s classifier over the pixel and PCA baselines?'),
    ('tinyTsne', 'tiny_tsne.py', 'Follow a tiny exact t-SNE optimizer', 'Which way will A move on the first update, and will the final normalized KL decrease?'),
]
if WRITE and not MODULE.exists():
    blocks = re.findall(r'```python\n(.*?)\n```', (PACKET / 'lesson.md').read_text(encoding='utf-8'), flags=re.S)
    assert len(blocks) == 4, len(blocks)
    programs = {key: dict(title=title, question=question, file=file, code=code, language='python') for (key, file, title, question), code in zip(metadata, blocks)}
else:
    programs = json.loads(MODULE.read_text(encoding='utf-8').split('export const manifoldExamples = ', 1)[1].strip().removesuffix(';'))

records, namespaces = {}, {}
for key, program in programs.items():
    namespace = {'__file__': str(WORK / program['file']), '__name__': '__main__'}
    code = program['code']
    (WORK / program['file']).write_text(code + '\n', encoding='utf-8')
    # The saved t-SNE authoring run used this host's 12-thread BLAS/OpenMP
    # configuration. Its n_jobs=1 controls neighbor search, not these pools.
    with contextlib.chdir(WORK), threadpool_limits(limits=12 if key == 'digitsAudit' else 1), contextlib.redirect_stdout(io.StringIO()) as stream:
        exec(compile(code, program['file'], 'exec'), namespace)
    output = stream.getvalue().strip()
    if not WRITE:
        assert output == program['expected'], f'{key} output changed:\n{output}\nexpected:\n{program["expected"]}'
    records[key] = {**program, 'expected': output}
    namespaces[key] = namespace
    print(f'{key}:\n{output}', flush=True)

saved = json.loads((PACKET / 'calculated-inputs.json').read_text(encoding='utf-8'))
data = namespaces['digitsAudit']['data']
assert data.shape == (300, 66)
assert np.array_equal(data[:, 0], saved['rows'])
assert digest(ASSETS / 'digits-300.csv') == saved['csv_sha256']
for display, key in [('PCA', 'pca'), ('t-SNE p=5', 'tsne-p5-s7'), ('t-SNE p=30', 'tsne-p30-s7'), ('t-SNE p=80', 'tsne-p80-s7')]:
    actual = namespaces['digitsAudit']['layouts'][display]
    expected = np.asarray(saved['layouts'][key]['coordinates'])
    assert np.allclose(actual, expected, atol=1e-8, rtol=1e-7), f'Original native layout drift: {key}'

fit = namespaces['umapFit']
assert fit['Y'].shape == (300, 2) and np.isfinite(fit['Y']).all()
assert np.array_equal(fit['reducer']._raw_data, fit['X'].astype(np.float32))
assert not fit['reducer']._supervised
heldout = namespaces['heldoutTransform']
assert heldout['X_train'].shape == (225, 64) and heldout['X_valid'].shape == (75, 64)
assert heldout['representations']['UMAP-10'][0].shape == (225, 10)
assert heldout['representations']['UMAP-10'][1].shape == (75, 10)
assert all(np.isfinite(part).all() for pair in heldout['representations'].values() for part in pair)
assert np.array_equal(heldout['reducer']._raw_data, heldout['X_train'].astype(np.float32))
assert np.array_equal(heldout['reducer'].embedding_, heldout['representations']['UMAP-10'][0]), 'transform changed fitted training map'
assert not heldout['reducer']._supervised
tiny = namespaces['tinyTsne']
assert np.allclose(tiny['P'], saved['fixtures']['t_sne_tiny']['P'], atol=1e-12)
assert np.allclose(tiny['Y'], saved['fixtures']['t_sne_tiny']['final_Y'], atol=1e-8)

from umap import umap_ as implementation
from umap import layouts as layout_implementation
source_checks = {
    'smooth_knn_dist': inspect.getsource(implementation.smooth_knn_dist),
    'fuzzy_simplicial_set': inspect.getsource(implementation.fuzzy_simplicial_set),
    'compute_membership_strengths': inspect.getsource(implementation.compute_membership_strengths),
    '_optimize_layout_euclidean_single_epoch': inspect.getsource(layout_implementation._optimize_layout_euclidean_single_epoch),
    'UMAP.transform': inspect.getsource(implementation.UMAP.transform),
}
assert 'np.log2(k)' in source_checks['smooth_knn_dist']
assert 'range(1, distances.shape[1])' in source_checks['smooth_knn_dist']
assert 'transpose - prod_matrix' in source_checks['fuzzy_simplicial_set']
assert 'rho' in source_checks['compute_membership_strengths']
assert 'tau_rand_int' in source_checks['_optimize_layout_euclidean_single_epoch']
assert 'self.embedding_' in source_checks['UMAP.transform']
# Installed-version calibration, independent of the JavaScript local sandbox.
calibration = np.array([[0, 1, 1 + np.log(2), 1 + np.log(2)]], dtype=np.float32)
sigma, rho = implementation.smooth_knn_dist(calibration, 4.0)
assert np.allclose(sigma, [1], atol=1e-4) and np.allclose(rho, [1], atol=1e-7)

versions = {name: importlib.metadata.version(name) for name in ['numpy', 'scipy', 'scikit-learn', 'umap-learn', 'numba', 'llvmlite', 'pynndescent', 'threadpoolctl']}
versions['python'] = sys.version.split()[0]
if WRITE:
    MODULE.write_text('// Actual executed programs; regenerate with scripts/verify-manifold-examples.py --write.\nexport const manifoldExamples = ' + json.dumps(records, ensure_ascii=False, indent=2) + ';\n', encoding='utf-8')
    shutil.copyfile(WORK / 'umap-digits.csv', ASSETS / 'umap-digits.csv')
    (ASSETS / 'runtime-versions.json').write_text(json.dumps(versions, indent=2) + '\n', encoding='utf-8')
    for key, program in records.items():
        (ASSETS / program['file']).write_text(program['code'] + '\n', encoding='utf-8')
else:
    assert digest(WORK / 'umap-digits.csv') == digest(ASSETS / 'umap-digits.csv'), 'UMAP coordinate bytes changed'

evidence = {
    'status': 'passed', 'generatedAt': datetime.now(timezone.utc).isoformat(),
    'command': 'scratch/manifold-learning-runtime/Scripts/python.exe scripts/verify-manifold-examples.py' + (' --write' if WRITE else ''),
    'versions': versions, 'threadPools': {'digitsAudit': 12, 'otherPrograms': 1}, 'datasetSha256': digest(ASSETS / 'digits-300.csv'),
    'examples': {key: {'file': value['file'], 'codeSha256': hashlib.sha256(value['code'].encode()).hexdigest(), 'output': value['expected']} for key, value in records.items()},
    'checks': ['all four exact displayed Python programs executed', 'original PCA/t-SNE coordinates agree with saved arrays', 'source IDs and CSV bytes preserved', 'UMAP fit uses all 300 feature-only rows and returns finite 300×2 coordinates', 'stratified train/validation is 225/75', 'UMAP transform produces finite 75×10 coordinates without modifying the fitted reference', 'tiny t-SNE agrees with independently calculated P and final Y', 'installed smooth-kNN self-column convention checked on exact calibration fixture'],
    'installedSourceSha256': {key: hashlib.sha256(value.encode()).hexdigest() for key, value in source_checks.items()},
    'sourceInterpretation': {'smoothKnn': 'target log2(k); summation excludes first self column', 'union': 'forward + reverse - product', 'optimization': 'positive-edge scheduling and randomly sampled vertex repulsion; not an unbiased full-pair BCE trace', 'transform': 'uses the fitted reference embedding'},
    'sourceHashes': {str(path.relative_to(ROOT)).replace('\\', '/'): digest(path) for path in [Path(__file__), MODULE, ASSETS / 'digits-300.csv', ASSETS / 'umap-digits.csv', ASSETS / 'runtime-versions.json', *[ASSETS / item['file'] for item in records.values()]]},
}
(ROOT / 'docs/teaching/evidence/manifold-native.json').write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
print('PASS: 4 programs, preserved native layouts, actual UMAP fit/held-out transform, installed-source checks.')
