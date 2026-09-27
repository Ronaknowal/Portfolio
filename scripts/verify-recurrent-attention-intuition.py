"""Focused revision-4 checks. Do not overwrite revision-3 native receipts."""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import math
import runpy
from datetime import datetime, timezone

import numpy as np
import torch

root = Path(__file__).resolve().parents[1]
revision = root / 'docs/teaching/revisions/attention-mechanism-bahdanau-luong/4'
receipt = revision / 'author-checks.json'
report = {'passed': False, 'date': datetime.now(timezone.utc).isoformat(), 'checks': []}
receipt.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

with contextlib.redirect_stdout(io.StringIO()) as printed:
    module = runpy.run_path(str(root / 'public/learn-code/attention-mechanism-bahdanau-luong/attention-read.py'))
read = module['attention_read']
keys, values = module['keys'], module['values']
query = np.array([1., 0.])
valid = np.array([True, True, True])
weights, context = read(query, keys, values, valid)

# Independent scalar expression, including unequal/signed values.
denominator = math.e + 1 + 1 / math.e
expected_weights = np.array([math.e, 1, 1 / math.e]) / denominator
expected_context = np.array([2 * math.e - 1 / math.e, 2 + 1 / math.e]) / denominator
np.testing.assert_allclose(weights, expected_weights, atol=1e-14)
np.testing.assert_allclose(context, expected_context, atol=1e-14)
report['checks'].append({'name': 'Displayed core executed against independent scalar formula', 'output': printed.getvalue().strip()})

# A masked large value cannot steal mass; forgetting the mask yields the teaching counterexample.
one_key = np.zeros((3, 1))
scalar_values = np.array([[2.], [4.], [0.]])
np.testing.assert_allclose(read(np.ones(1), one_key, scalar_values, np.array([True, True, False]))[1], [3.])
np.testing.assert_allclose(read(np.ones(1), one_key, scalar_values, valid)[1], [2.])
scalar_values[2] = 1e6
np.testing.assert_allclose(read(np.ones(1), one_key, scalar_values, np.array([True, True, False]))[1], [3.])
try:
    read(query, keys, values, np.zeros(3, dtype=bool))
except ValueError:
    pass
else:
    raise AssertionError('All-masked input must be rejected')
report['checks'].append({'name': 'Mask-before-normalization and zero-PAD counterexample', 'passed': True})

for q in ([1., 0.], [.3, -.4], [-2., 1.]):
    for mask in ([True, True, True], [True, False, True], [False, True, False]):
        nw, nc = read(np.array(q), keys, values, np.array(mask))
        scores = torch.tensor(keys, dtype=torch.float64) @ torch.tensor(q, dtype=torch.float64)
        tw = scores.masked_fill(~torch.tensor(mask), -torch.inf).softmax(-1)
        tc = tw @ torch.tensor(values, dtype=torch.float64)
        np.testing.assert_allclose(nw, tw.numpy(), atol=1e-14)
        np.testing.assert_allclose(nc, tc.numpy(), atol=1e-14)
report['checks'].append({'name': 'Nine fresh NumPy/Torch read and mask comparisons', 'passed': True})

assert sum(t * w for t, w in zip([18, 24, 21], [.25, .75, 0])) == 22.5
assert sum([18, 24, 21]) / 3 == 21
assert round(2 * math.tanh(1.5), 6) == 1.810297
assert round(float(context[1] - context[0]), 6) == -0.660964
report['checks'].append({'name': 'Constructed lookup, additive score and learning-margin figures', 'passed': True})

baseline = json.loads((root / 'docs/teaching/evidence/attention-memory-intuition-baseline.json').read_text(encoding='utf-8'))
old = baseline['originalEntries']['attention-mechanism-bahdanau-luong']
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
reuse = [
    name for name in old['implementation']['reviewedFiles']
    if name.startswith('public/learn-code/attention-mechanism-bahdanau-luong/')
    or name in ['src/learn/data/recurrent-attention-models.js', 'src/learn/data/recurrent-attention-measurements.json']
]
for name in reuse:
    assert digest(root / name) == old['implementation']['reviewedFiles'][name], name
for name, expected in old['content']['files'].items():
    assert digest(root / name) == expected, name
report['reusedNumericalEvidence'] = {
    'unchangedRuntimeSources': reuse,
    'receipts': ['docs/teaching/evidence/recurrent-attention-models.json', 'docs/teaching/evidence/recurrent-attention-native.json'],
    'scope': 'Identical numerical engine, weights, native programs, measured data and original packet. New prose/figures/UI are assessed separately.'
}
source_files = [
    'scripts/verify-recurrent-attention-intuition.py',
    'scripts/generate-recurrent-attention-lesson.mjs',
    'public/learn-code/attention-mechanism-bahdanau-luong/attention-read.py',
    'src/learn/components/lesson-labs/RecurrentAttentionIntuition.jsx',
    'src/learn/components/lesson-labs/recurrent-attention-intuition.css',
    'src/learn/components/lesson-labs/RecurrentAttentionLabs.jsx',
    'src/learn/data/topics/attention.jsx',
    'docs/teaching/revisions/attention-mechanism-bahdanau-luong/4/lesson.md',
]
report['sourceHashes'] = {name: digest(root / name) for name in source_files}
report['passed'] = True
receipt.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print('Attention revision 4: new core, 9 native comparisons, 6 constructed teaching cases, unchanged scientific evidence verified.')
