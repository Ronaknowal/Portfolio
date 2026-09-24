"""Verify complete retained protocol, masks, real saved-model outputs and manual traces."""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import platform
import runpy
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
ID = 'sequence-to-sequence-encoder-decoder'
ASSETS = ROOT / 'public/learn-code' / ID
PACKET = ROOT / 'docs/teaching/drafts' / ID
REPORT = ROOT / 'docs/teaching/evidence/seq2seq-native.json'
REPORT.write_text(json.dumps({'passed': False, 'status': 'running'}), encoding='utf-8')
torch.set_num_threads(1)
groups = []
for file in ['inflection-seq2seq.py', 'sequence-mechanics.py', 'english-inflections.csv', 'calculated-inputs.json']:
    assert (ASSETS / file).read_bytes() == (PACKET / file).read_bytes()
groups.append('Programs, protected split and all three final measured runs equal prepared source bytes')
api = runpy.run_path(str(ASSETS / 'inflection-seq2seq.py'))
train, development = api['load_records']()
record = json.loads((ASSETS / 'calculated-inputs.json').read_text())
model = api['Inflector']().eval()
weights = record['runs'][0]['weights']
model.load_state_dict({key: torch.tensor(value, dtype=torch.bool if key == 'invalid_output' else torch.float32) for key, value in weights.items()})
assert sum(parameter.numel() for parameter in model.parameters()) == 37408
for rows in [train, development]:
    source, lengths, decoder_input, target = api['batch'](rows)
    assert (decoder_input[:, 0] == api['BOS']).all()
    assert torch.equal(decoder_input[:, 1:], target[:, :-1])
    assert lengths.tolist() == [len(row['lemma']) + 2 for row in rows]
for run in record['runs']:
    measured = api['summarize'](development, run['development_predictions'])
    for key, value in measured.items():
        assert run['development'][key] == value
    assert measured['reference_characters'] == 3490 and measured['no_eos'] == 0
groups.append('All source lengths and shifted batches, 37408 parameters and all three final metric denominators reconcile')
queries = [{'lemma': row['lemma'], 'feature': row['feature']} for row in development]
predictions = api['greedy'](model, queries)
assert [item['tokens'] for item in predictions] == [item['tokens'] for item in record['runs'][0]['development_predictions']]
groups.append('Source-only native inference reproduces every one of 447 saved development token sequences')
beam_fixtures = []
for query in queries[:8] + [{'lemma': 'lactate', 'feature': 'past'}]:
    greedy = api['greedy'](model, [query])[0]
    for width in [1, 2, 3]:
        output = api['beam'](model, query, width=width)
        if width == 1:
            assert output['tokens'] == greedy['tokens']
        beam_fixtures.append({'query': query, 'width': width, **output})
groups.append('Genuine candidate-state beam1 parity and 27 width1/2/3 native fixtures for JS bridge')
mechanics = runpy.run_path(str(ASSETS / 'sequence-mechanics.py'))
with contextlib.redirect_stdout(io.StringIO()):
    mechanics['main']()
actual = json.loads((ASSETS / 'mechanics-results.json').read_text())
assert actual == json.loads((PACKET / 'mechanics-results.json').read_text())
groups.append('Complete reexecution of scalar gradient, manual GRU traces, interventions, mask/future-target/detach nulls and data slices reproduces every saved value')
capture = io.StringIO()
with contextlib.redirect_stdout(capture):
    runpy.run_path(str(ASSETS / 'saved-inflection.py'), run_name='__main__')
assert capture.getvalue().splitlines() == ['3 lac False', '16 lactated True']
groups.append('Standalone learner saved-inference program prints exact cap/EOS output without fitting')
# The changed-code task checks variable ending times and intact already-ended outputs.
batch_output = api['greedy'](model, [queries[0], queries[1], {'lemma': 'lactate', 'feature': 'past'}], max_output=3)
for query, output in zip([queries[0], queries[1], {'lemma': 'lactate', 'feature': 'past'}], batch_output):
    single = api['greedy'](model, [query], max_output=3)[0]
    assert output['tokens'] == single['tokens'] and output['ended_with_eos'] == single['ended_with_eos']
groups.append('Batched and separate capped generation preserve source/output ownership')
report = {'passed': True, 'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'torch': torch.__version__}, 'groups': groups, 'beamFixtures': beam_fixtures, 'sourceHashes': {str(file.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(file.read_bytes()).hexdigest() for file in ASSETS.iterdir() if file.is_file()}, 'limits': 'Original final-split fits reused unchanged; current execution covers inference, protocol and differentiable mechanisms, not refitting.'}
REPORT.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'passed': True, 'groups': len(groups), 'beamFixtures': len(beam_fixtures)}))
