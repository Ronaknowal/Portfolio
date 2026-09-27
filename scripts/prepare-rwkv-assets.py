"""Export only validation data and frozen seed-17 weights; validate native continuation."""
from pathlib import Path
import importlib.util
import json
import platform
import shutil
import sys
import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
topic = 'rwkv-linear-attention-models'
packet = root / 'docs/teaching/drafts' / topic
sys.path.insert(0, str(packet))
from trajectory_memory_models import prepare, TrajectoryMemoryClassifier
from linear_memory_mechanisms import positive_matrix, positive_recurrent, rwkv_stable, rwkv_direct, additive_and_delta

torch.set_num_threads(2)
assets = root / 'public/learn-assets' / topic
downloads = root / 'public/learn-code' / topic
evidence = root / 'docs/teaching/deep-learning-completion' / topic
for folder in (assets, downloads, evidence):
    folder.mkdir(parents=True, exist_ok=True)
learner_files = ('author_calculations.py', 'data-provenance.md', 'investigation-checks.json',
                 'linear_memory_mechanisms.py', 'mechanism-results.json', 'movement_libras.data',
                 'movement_libras.names', 'rwkv_checkpoint_state.py', 'trajectory_memory_models.py',
                 'trajectory-memory-fits.npz', 'trajectory-results.json')
for name in learner_files:
    shutil.copyfile(packet / name, downloads / name)
x, labels, roles, _ = prepare(packet)
arrays = np.load(packet / 'trajectory-memory-fits.npz', allow_pickle=False)
export = {'source': 'UCI Libras Movement, CC BY 4.0; validation rows only', 'models': {},
          'specimens': [{'sourceRow': int(row + 1), 'label': int(labels[row] + 1),
                         'points': ((x[row].numpy() + 1) / 2).tolist()} for row in roles[1]]}
fixtures = []
for kind in ('rwkv4', 'positive_kernel'):
    model = TrajectoryMemoryClassifier(kind).eval()
    prefix = f'{kind}_seed17::'
    state = {name: torch.from_numpy(arrays[prefix + name]) for name in model.state_dict()}
    model.load_state_dict(state)
    export['models'][kind] = {name: value.tolist() for name, value in state.items()}
    for source, cut in ((20, 15), (7, 22)):
        original = x[source - 1:source].clone()
        for variant in ('original', 'reset', 'reflect', 'reverse'):
            row = original.clone()
            if variant == 'reflect': row[0, cut, 0] *= -1
            if variant == 'reverse': row = row.flip(1)
            with torch.no_grad():
                full = model(row)
                first, saved = model.features(row[:, :cut])
                second, _ = model.features(row[:, cut:], None if variant == 'reset' else saved)
                logits = model.classifier(torch.cat([first, second], 1).mean(1))
                if variant != 'reset': torch.testing.assert_close(logits, full, atol=1e-5, rtol=1e-5)
            fixtures.append({'kind': kind, 'sourceRow': source, 'cut': cut, 'variant': variant,
                             'logits': logits[0].tolist(), 'probabilities': logits.softmax(1)[0].tolist()})
    with torch.no_grad():
        current = model(x[roles[1]])
        original = torch.from_numpy(arrays[prefix + 'logits'][roles[1]])
        torch.testing.assert_close(current, original, atol=2e-5, rtol=2e-5)
q = np.array([[1., 1.], [2., 1.], [1., 2.], [3., 1.]])
k = np.array([[1., 0.], [0., 1.], [1., 1.], [2., 1.]])
v = np.array([[3.], [-2.], [7.], [1.]])
for size in (1, 2, 3, 4, 8):
    np.testing.assert_allclose(positive_matrix(q, k, v), positive_recurrent(q, k, v, size)[0], atol=1e-12)
keys = np.log(np.array([[1.], [2.], [1.], [4.]]))
for retention in (.01, .5, 1):
    for offset in (-1000, 0, 1000):
        stable, _ = rwkv_stable(keys + offset, v, np.array([np.log(retention)]), np.array([.7]))
        direct = rwkv_direct(keys + offset, v, np.array([np.log(retention)]), np.array([.7]))
        np.testing.assert_allclose(stable, direct, atol=1e-10)
(assets / 'trajectory-models.json').write_text(json.dumps(export, separators=(',', ':')) + '\n')
(root / 'src/learn/data/rwkv-trajectory-example.json').write_text(json.dumps(next(row for row in export['specimens'] if row['sourceRow'] == 7), separators=(',', ':')) + '\n')
(evidence / 'native-fixtures.json').write_text(json.dumps(fixtures, indent=2) + '\n')
(evidence / 'native-checks.json').write_text(json.dumps({
    'passed': True, 'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__},
    'checks': ['50 validation rows × 2 frozen models reproduce saved logits', '16 full/carry/reset/reflect/reverse fixtures',
               'five chunk sizes including short remainder', 'stable versus enumerated RWKV with retention boundary and ±1000 offsets'],
    'notExecuted': ['Optional released checkpoint program: local weights/tokenizer not supplied', 'New training: original frozen fits preserved'],
}, indent=2) + '\n')
print('PASS: frozen models, native continuation, 16 port fixtures and bounded operator checks')
