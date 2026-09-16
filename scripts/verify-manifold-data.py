"""Verify and generate compact offline manifold fixtures and observation data.

Run after verify-manifold-examples.py. --write publishes exact retained PCA/
t-SNE arrays plus actual UMAP output. No fitting occurs in this data generator.
Without --write the generated module and downloadable arrays must be current.
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances

ROOT = Path(__file__).resolve().parents[1]
PACKET = ROOT / 'docs/teaching/drafts/t-sne-umap-manifold-learning'
ASSETS = ROOT / 'public/learn-assets/manifold-learning'
MODULE = ROOT / 'src/learn/data/manifold-data.js'
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
saved = json.loads((PACKET / 'calculated-inputs.json').read_text(encoding='utf-8'))
csv = np.loadtxt(ASSETS / 'digits-300.csv', delimiter=',', skiprows=1)
ids, labels, pixels = csv[:, 0].astype(int), csv[:, 1].astype(int), csv[:, 2:]
X = pixels / 16
assert csv.shape == (300, 66) and len(set(ids)) == 300
assert np.array_equal(ids, saved['rows']) and np.all(ids[:-1] < ids[1:])
assert np.array_equal(np.bincount(labels), [30] * 10)
assert np.isfinite(pixels).all() and np.all((pixels >= 0) & (pixels <= 16)) and np.array_equal(pixels, pixels.astype(int))
assert digest(ASSETS / 'digits-300.csv') == saved['csv_sha256']
native = load_digits()
assert np.array_equal(native.data[ids], pixels) and np.array_equal(native.target[ids], labels)
selection = np.sort(np.concatenate([np.flatnonzero(native.target == label)[:30] for label in range(10)]))
assert np.array_equal(ids, selection)

def order(points):
    # Direct subtraction independently avoids pairwise_distances cancellation.
    distance = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(distance, np.inf)
    return np.argsort(distance, axis=1, kind='stable')

input_order = order(X)
def metrics(points, k):
    mapped = order(points)
    retained = np.mean([len(set(a[:k]) & set(b[:k])) / k for a, b in zip(input_order, mapped)])
    return dict(retention=float(retained), trustworthiness=float(trustworthiness(X, points, n_neighbors=k)), continuity=float(trustworthiness(points, X, n_neighbors=k)))

layouts = []
for key, item in saved['layouts'].items():
    coordinates = np.asarray(item['coordinates'])
    assert coordinates.shape == (300, 2) and np.isfinite(coordinates).all()
    for k in [5, 10, 20]:
        actual = metrics(coordinates, k)
        for name, value in actual.items():
            assert abs(value - item['metrics'][str(k)][name]) < 1e-12, (key, k, name, value, item['metrics'][str(k)][name])
    if key == 'pca': title = 'PCA · 2 components'
    else:
        parts = key.split('-')
        title = f't-SNE · perplexity {parts[1][1:]} · {"random" if "random" in parts else "PCA"} init · seed {parts[-1][1:]}'
    layouts.append(dict(key=key, title=title, coordinates=item['coordinates'], metrics=item['metrics'], **({'kl': item['kl']} if 'kl' in item else {})))

umap_csv = np.loadtxt(ASSETS / 'umap-digits.csv', delimiter=',')
assert umap_csv.shape == (300, 4) and np.array_equal(umap_csv[:, :2], csv[:, :2])
assert np.isfinite(umap_csv).all()
umap_coordinates = umap_csv[:, 2:]
layouts.append(dict(key='umap-n15-d01-s7', title='UMAP · 15 neighbors · min_dist 0.1 · seed 7', coordinates=umap_coordinates.tolist(), metrics={str(k): metrics(umap_coordinates, k) for k in [5, 10, 20]}))
by_key = {item['key']: item for item in layouts}
assert all(by_key[f'tsne-p{p}-s7']['coordinates'] == by_key[f'tsne-p{p}-s19']['coordinates'] for p in [5, 30, 80])
assert by_key['tsne-p30-random-s7']['coordinates'] != by_key['tsne-p30-random-s19']['coordinates']
q = np.flatnonzero(ids == 30)[0]
query = {}
for key in ['pca', 'tsne-p30-s7']:
    map_order = order(np.asarray(by_key[key]['coordinates']))[q, :10]
    query[key] = dict(input=ids[input_order[q, :10]].tolist(), map=ids[map_order].tolist(), count=len(set(input_order[q, :10]) & set(map_order)))
assert query['pca']['count'] == 6 and query['tsne-p30-s7']['count'] == 5
assert query['pca']['input'] == [0, 166, 229, 160, 36, 276, 140, 266, 178, 79]

fixtures = saved['fixtures']
P = np.asarray(fixtures['t_sne_tiny']['P'])
Y = np.array([[-1.5], [-0.5], [0.5], [1.5]])
def objective(points):
    differences = points[:, None, :] - points[None, :, :]
    kernel = 1 / (1 + (differences ** 2).sum(axis=2))
    np.fill_diagonal(kernel, 0)
    Q = kernel / kernel.sum()
    mask = P > 0
    cost = float((P[mask] * np.log(P[mask] / Q[mask])).sum())
    gradient = 4 * (((P - Q) * kernel)[:, :, None] * differences).sum(axis=1)
    return cost, gradient
trace = []
for step in range(201):
    cost, gradient = objective(Y)
    if step in [0, 1, 10, 50, 100, 200]: trace.append(dict(step=step, Y=Y.tolist(), cost=cost))
    if step < 200:
        Y -= 0.5 * gradient
        Y -= Y.mean(axis=0)
assert np.allclose(Y, fixtures['t_sne_tiny']['final_Y'], atol=1e-12)
fixtures['t_sne_tiny']['initial_Y'] = trace[0]['Y']
fixtures['t_sne_tiny']['trace'] = trace
fixtures['classical_mds'].update(distances=[[0, 2, 5], [2, 0, 3], [5, 3, 0]], coordinates=[-7/3, -1/3, 8/3])
fixtures['lle'] = dict(input=[0, 1, 3], weights=[2/3, 1/3], output=[0, 2, 6])
fixtures['square_rips'].update(points=[[0, 0], [1, 0], [1, 1], [0, 1]], projected=[[0], [1], [1], [0]])
versions = json.loads((ASSETS / 'runtime-versions.json').read_text(encoding='utf-8'))
digits = dict(hash=saved['csv_sha256'], rows=[dict(sourceRow=int(id), digit=int(label), pixels=row.astype(int).tolist()) for id, label, row in zip(ids, labels, pixels)], layouts=layouts, selection=saved['selection'], versions=versions, originalLayoutVersions=saved['versions'], metric='Euclidean distance on the 64 block counts divided by 16', tieRule='ascending source-row ID for retention and visible lists; scikit-learn native sorting for T and continuity')
module = '/** Native optical-digits arrays and exact constructed fixtures. Generated by scripts/verify-manifold-data.py.\n * Original data: E. Alpaydin and C. Kaynak (1998), UCI Optical Recognition of Handwritten Digits, CC BY 4.0.\n * Selection, transformations, versions and attribution: /learn-assets/manifold-learning/data-provenance.md.\n * Every coordinate row matches rows in order; labels never enter an embedding fit.\n */\n'
module += 'export const MANIFOLD_DIGITS = ' + json.dumps(digits, ensure_ascii=False, separators=(',', ':')) + ';\n'
module += 'export const MANIFOLD_FIXTURES = ' + json.dumps(fixtures, ensure_ascii=False, separators=(',', ':')) + ';\n'
public = dict(datasetSha256=saved['csv_sha256'], rows=saved['rows'], layouts={item['key']: {key: value for key, value in item.items() if key not in ['key', 'title']} for item in layouts}, versions=versions, originalLayoutVersions=saved['versions'])
public_text = json.dumps(public, ensure_ascii=False, separators=(',', ':')) + '\n'
if '--write' in sys.argv:
    MODULE.write_text(module, encoding='utf-8')
    (ASSETS / 'calculated-inputs.json').write_text(public_text, encoding='utf-8')
else:
    assert MODULE.read_text(encoding='utf-8') == module, 'Stale manifold data module'
    assert (ASSETS / 'calculated-inputs.json').read_text(encoding='utf-8') == public_text, 'Stale downloadable coordinates'
evidence = dict(status='passed', generatedAt=datetime.now(timezone.utc).isoformat(), command='scratch/manifold-learning-runtime/Scripts/python.exe scripts/verify-manifold-data.py' + (' --write' if '--write' in sys.argv else ''), observations=300, features=64, nativeLayouts=len(layouts), metricsChecked=90, checks=['CSV hash/selection/source data/labels and features match original bundled collection', 'nine saved layouts are byte-preserved coordinate arrays', 'all coordinate metrics independently recomputed at k=5/10/20', 'actual UMAP exported coordinates preserve every source ID', 'query30 exposes six versus five retained neighbors', 'PCA-init seed equality and random-init seed contrast retained', 'offline 200-step tiny trace uses fixed native P'], query30=query, metrics10={item['key']: item['metrics']['10'] for item in layouts}, sourceHashes={str(path.relative_to(ROOT)).replace('\\', '/'): digest(path) for path in [Path(__file__), MODULE, PACKET / 'calculated-inputs.json', ASSETS / 'digits-300.csv', ASSETS / 'umap-digits.csv', ASSETS / 'calculated-inputs.json', ASSETS / 'runtime-versions.json', ASSETS / 'data-provenance.md']})
(ROOT / 'docs/teaching/evidence/manifold-data.json').write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
print(f'PASS: 300 exact images; {len(layouts)} native layouts; 90 diagnostics; source30 PCA=6, t-SNE=5. UMAP R10={layouts[-1]["metrics"]["10"]["retention"]:.6f}')
