"""Independent packed-batch and padded teacher-forcing oracle; no fits."""
from pathlib import Path
import hashlib
import importlib.util
import json
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'public/learn-code/sequence-to-sequence-encoder-decoder'
OUT = ROOT / 'docs/teaching/evidence/seq2seq-independent-native.json'
OUT.write_text(json.dumps({'passed': False, 'status': 'running'}))
torch.set_num_threads(1)
spec = importlib.util.spec_from_file_location('published_inflection', ASSETS / 'inflection-seq2seq.py')
published = importlib.util.module_from_spec(spec)
spec.loader.exec_module(published)
weights = json.loads((ASSETS / 'seed-one-inference.json').read_text())['weights']
embedding = nn.Embedding(32, 24, padding_idx=0).double()
embedding.weight.data.copy_(torch.tensor(weights['embedding.weight'], dtype=torch.float64))
encoder, decoder = (nn.GRU(24, 64, batch_first=True).double() for _ in range(2))
for name, module in [('encoder', encoder), ('decoder', decoder)]:
    module.load_state_dict({key.removeprefix(name + '.'): torch.tensor(value, dtype=torch.float64)
                            for key, value in weights.items() if key.startswith(name + '.')})
head = nn.Linear(64, 32).double()
head.load_state_dict({key.removeprefix('readout.'): torch.tensor(value, dtype=torch.float64)
                     for key, value in weights.items() if key.startswith('readout.')})
records = [
    {'lemma': 'a', 'feature': 'past', 'form': 'a'},
    {'lemma': 'zzzzzzzzzzzz', 'feature': 'third_person', 'form': 'ab'},
    {'lemma': 'walk', 'feature': 'participle', 'form': 'walking'},
    {'lemma': 'azerty', 'feature': 'past', 'form': 'azerty'},
]
source, lengths, inputs, targets = published.batch(records)
with torch.no_grad():
    vectors = embedding(source)
    _, context = encoder(pack_padded_sequence(vectors, lengths, batch_first=True, enforce_sorted=False))
    disturbed = vectors.clone()
    for row, length in enumerate(lengths):
        disturbed[row, int(length):] = 123.0
    _, storage_context = encoder(pack_padded_sequence(disturbed, lengths, batch_first=True, enforce_sorted=False))
    torch.testing.assert_close(storage_context, context, rtol=0, atol=0)
    decoded, _ = decoder(embedding(inputs), context)
    logits = head(decoded)
    logits[..., [0, 1, 3, 4, 5]] = -torch.inf
    probabilities = logits.softmax(-1)
    cases = []
    for index, record in enumerate(records):
        cases.append({**record, 'padding': int(targets.shape[1] - len(record['form']) - 1),
                      'inputs': [published.TOKENS[value] for value in inputs[index].tolist()],
                      'targets': [published.TOKENS[value] for value in targets[index].tolist()],
                      'context': context[0, index].tolist(), 'states': decoded[index].tolist(),
                      'probabilities': probabilities[index].tolist()})
sources = ['scripts/verify-seq2seq-independent.py',
           'public/learn-code/sequence-to-sequence-encoder-decoder/inflection-seq2seq.py',
           'public/learn-code/sequence-to-sequence-encoder-decoder/seed-one-inference.json']
report = {'passed': True, 'torch': torch.__version__, 'cases': cases,
          'checks': ['Fresh unsorted source lengths 3/14/6/8 use native packed batch and separate encoder/decoder modules',
                     'Target shifts come from the actual published batch function, including EOS at the first ignored input',
                     'Changing only packed-away source storage to123 leaves all native context coordinates exactly unchanged'],
          'sources': {file: hashlib.sha256((ROOT / file).read_bytes()).hexdigest() for file in sources}}
OUT.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'passed': True, 'batchCases': len(cases), 'decoderPositions': sum(len(case['states']) for case in cases)}))
