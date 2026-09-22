"""Independent complementary checks for the typed-decision project revision.

Run from the repository root with torch/transformers available. This checks
local tiny fixtures only; it neither downloads nor trains a pretrained model.
"""
import copy
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

os.environ['HF_HUB_OFFLINE'] = '1'
ROOT = Path(__file__).resolve().parents[4]
PROGRAMS = ROOT / 'public/learn-projects/typed-decision-model'
sys.path.insert(0, str(PROGRAMS))
import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
from transformers import AutoModel, BertConfig, PreTrainedTokenizerFast
import typed_decision as core
import research_tools as research
import pretrained_decision as adapter


def browser_model_checks():
    vocabulary = core.make_vocabulary(core.fixture_data()['train'])
    cases = []
    for state in ['please help with my refund', 'money left my bank twice', 'REFUND 234 refund!',
                  'Café refund', 'refund ' * 100]:
        for options in [core.OPTIONS, core.OPTIONS[1:] + core.OPTIONS[:1], core.OPTIONS[:2]]:
            cases.append({'question': core.QUESTION, 'state': state, 'options': options})
    javascript = """
import fs from 'node:fs';
import { encodeRequestTrace, scoringHeadStep, attentionMixture } from './src/learn/data/projects/typed-decision-model/mechanism-models.js';
const { cases, vocabulary } = JSON.parse(fs.readFileSync(0, 'utf8'));
const encodings = cases.map(row => encodeRequestTrace(row, vocabulary));
let maximumFiniteDifferenceError = 0;
let gradientCases = 0;
for (const w of [[2,1], [-3,4], [0,0], [7,-5]]) {
  for(let target = 0; target < 3; target++) {
    const value = scoringHeadStep(w,target,0.4);
    for(let axis=0; axis<2; axis++) {
      const plus=[...w], minus=[...w]; plus[axis]+=1e-5; minus[axis]-=1e-5;
      const derivative = (scoringHeadStep(plus,target,0).loss - scoringHeadStep(minus,target,0).loss)/2e-5;
      const error=Math.abs(derivative-value.gradient[axis]);
      maximumFiniteDifferenceError=Math.max(maximumFiniteDifferenceError,error);
      if(error>1e-8) throw new Error('Shared head gradient failed finite differences');
      gradientCases++;
    }
    const identity=scoringHeadStep(w,target,0);
    if(JSON.stringify(identity.nextWeights)!==JSON.stringify(w)) throw new Error('Zero step moved weights');
  }
}
const attention=[];
for(let i=-20; i<=20; i++) {
  for(const mask of [false,true]) attention.push({query:i/10,mask,...attentionMixture(i/10,mask)});
}
process.stdout.write(JSON.stringify({encodings,attention,gradientCases,maximumFiniteDifferenceError}));
"""
    result = json.loads(subprocess.run(['node', '--input-type=module', '-e', javascript], cwd=ROOT,
        input=json.dumps({'cases': cases, 'vocabulary': vocabulary}), text=True,
        capture_output=True, check=True).stdout)
    for row, encoded in zip(cases, result['encodings']):
        try:
            ids, markers = core.encode(row, vocabulary)
        except ValueError:
            assert encoded.get('error'), 'Browser did not reject native over-budget request'
        else:
            assert ids == [cell['id'] for cell in encoded['cells']]
            assert markers == encoded['markers']
            assert row['options'][encoded['target']]['id'] == 'billing'
    for item in result['attention']:
        # Reconstruct with torch matrix arithmetic rather than JS weighted loops.
        query = torch.tensor([[item['query'], 0]], dtype=torch.float64)
        keys = torch.tensor([[1, 0], [0, 1], [-1, 0], [4, 0]], dtype=torch.float64)
        values = torch.tensor([[1, 0], [0, 2], [1, 1], [10, 10]], dtype=torch.float64)
        scores = query @ keys.T / math.sqrt(2)
        if item['mask']:
            scores[0, 3] = -torch.inf
        probabilities = scores.softmax(-1)
        torch.testing.assert_close(probabilities[0], torch.tensor(item['probabilities'], dtype=torch.float64), atol=1e-14, rtol=1e-14)
        torch.testing.assert_close((probabilities @ values)[0], torch.tensor(item['output'], dtype=torch.float64), atol=1e-14, rtol=1e-14)
        if item['mask']:
            assert item['probabilities'][3] == 0
    return {'encoding_cases': len(cases), 'attention_states': len(result['attention']),
            'finite_difference_coordinates': result['gradientCases'],
            'maximum_gradient_error': result['maximumFiniteDifferenceError']}


def diagnostic_checks():
    probabilities = torch.tensor([[0.99, 0.01], [0.90, 0.1], [0.8, 0.2],
                                   [0.4, 0.6], [0.5, 0.5], [1.0, 0.0]], dtype=torch.float64)
    targets = torch.tensor([0, 1, 0, 0, 0, 1])
    confidences, choices = probabilities.max(-1)
    for entry in research.policy_sweep(probabilities, targets):
        # Independent vectorized expected-cost formula, not expected_cost_decision().
        cost = 10 * (1 - confidences)
        tolerance = 1e-12 * torch.maximum(torch.ones_like(cost), torch.maximum(cost.abs(), torch.full_like(cost, entry['review_cost'])))
        act = cost < entry['review_cost'] - tolerance
        wrong = choices != targets
        assert entry['coverage'] == act.double().mean().item()
        assert entry['acted'] == int(act.sum())
        expected_risk = float(wrong[act].double().mean()) if act.any() else None
        assert entry['selective_risk'] == expected_risk
        realized = torch.where(act, wrong * 10.0, entry['review_cost']).double().mean().item()
        assert math.isclose(entry['realized_mean_cost'], realized, abs_tol=1e-12)
    binned = research.reliability(probabilities, targets)
    assert sum(entry['count'] for entry in binned['bins']) == 6
    assert binned['bins'][-1]['count'] == 4  # includes exactly .8 and 1.0.
    manual_ece = sum(entry['count'] * abs(entry['accuracy'] - entry['mean_confidence'])
                     for entry in binned['bins'] if entry['count']) / 6
    assert abs(binned['ece'] - manual_ece) < 1e-12
    assert all(entry['accuracy'] is None and entry['mean_confidence'] is None
               for entry in binned['bins'] if not entry['count'])
    return {'policy_points': 7, 'boundary_confidences': [0.5, 0.6, 0.8, 0.9, 0.99, 1.0],
            'risk_denominator': 'acted only; null when none', 'reliability_bin_mass': 6}


def adapter_checks():
    torch.manual_seed(91)
    vocab = core.make_vocabulary(core.fixture_data()['train'])
    backend = Tokenizer(models.WordLevel(vocab, unk_token='<unk>'))
    backend.normalizer = normalizers.Lowercase()
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='<unk>',
                                       pad_token='<pad>', cls_token='<cls>', sep_token='<sep>')
    encoder = AutoModel.from_config(BertConfig(vocab_size=len(tokenizer), hidden_size=16,
        num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
        max_position_embeddings=128, hidden_dropout_prob=0.2, attention_probs_dropout_prob=0.2))
    adapter.register_marker(tokenizer, encoder)
    marker_id = tokenizer.convert_tokens_to_ids(adapter.MARKER)
    before = encoder.get_input_embeddings().weight.detach().clone()
    adapter.register_marker(tokenizer, encoder)
    torch.testing.assert_close(before, encoder.get_input_embeddings().weight, atol=0, rtol=0)
    model = adapter.EncoderDecisionModel(encoder, freeze_encoder=True).train()
    rows = [core.fixture_data()['train'][0], {**core.fixture_data()['train'][1], 'options': core.OPTIONS[:2]}]
    batch = adapter.batch_requests(rows, tokenizer)
    first = model(**batch)
    torch.testing.assert_close(first, model(**batch), atol=0, rtol=0)  # .2 dropout must remain off.
    first.softmax(-1).sum().backward()
    assert all(parameter.grad is None for parameter in encoder.parameters())
    renamed = copy.deepcopy(rows)
    for row in renamed:
        for index, option in enumerate(row['options']):
            option['id'] = f'opaque-{index}'
    torch.testing.assert_close(first, model(**adapter.batch_requests(renamed, tokenizer)), atol=0, rtol=0)
    # Mixed-batch padding must not change the short request's valid output scores.
    alone = model(**adapter.batch_requests([rows[1]], tokenizer))[0]
    torch.testing.assert_close(first[1, :2], alone, atol=2e-6, rtol=2e-5)
    with tempfile.TemporaryDirectory(prefix='typed-decision-independent-') as directory:
        adapter.save_artifact(directory, model, tokenizer, 1.7, {'checkpoint': 'offline-random-review'})
        restored, loaded, config = adapter.load_artifact(directory)
        assert loaded.convert_tokens_to_ids(adapter.MARKER) == marker_id
        assert config['temperature'] == 1.7
        torch.testing.assert_close(first, restored(**adapter.batch_requests(rows, loaded)), atol=0, rtol=0)
        broken = json.loads((Path(directory) / 'decision-config.json').read_text())
        broken['temperature'] = 0
        (Path(directory) / 'decision-config.json').write_text(json.dumps(broken))
        try:
            adapter.load_artifact(directory)
        except ValueError:
            pass
        else:
            raise AssertionError('Zero artifact temperature accepted')
    rejected = 0
    for change in [{'state': adapter.MARKER}, {'options': rows[0]['options'][:1]},
                   {'options': [core.OPTIONS[0], core.OPTIONS[0]]}, {'state': 'refund ' * 150}]:
        try:
            adapter.encode_request({**rows[0], **change}, tokenizer)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError(f'Invalid request accepted: {change}')
    return {'scope': 'Offline random BERT; no pretrained quality claim',
            'marker_registration': 'idempotent, preserved across save-load',
            'dropout': 'frozen encoder stays deterministic with nonzero configured dropout',
            'identity': 'renamed IDs preserve outputs; mixed-batch padding preserves valid scores',
            'invalid_requests_rejected': rejected, 'invalid_temperature_rejected': True}


def retained_checkpoint_checks(directory):
    source = Path(directory)
    config = json.loads((source / 'config.json').read_text())
    dataset = json.loads((source / 'fixture-data.json').read_text())
    model = core.TypedDecisionModel(len(config['vocabulary'])).eval()
    model.load_state_dict(torch.load(source / 'weights.pt', weights_only=True))
    targets = torch.tensor([row['target'] for row in dataset['test']])
    with torch.inference_mode():
        scores = model(*core.batch_rows(dataset['test'], config['vocabulary']))
    expected = json.loads((PROGRAMS / 'verified-diagnostics.json').read_text())
    assert hashlib.sha256((source / 'weights.pt').read_bytes()).hexdigest() == expected['weights_sha256']
    result = {}
    for name, temperature in [('raw', 1.0), ('calibrated', config['temperature'])]:
        probabilities = (scores.double() / temperature).softmax(-1)
        confidence, choices = probabilities.max(-1)
        true_correct = choices == targets
        # A vectorized bincount reconstructs confusion, independent of summarize's loop.
        confusion = torch.bincount(targets * 3 + choices, minlength=9).reshape(3, 3)
        assert confusion.tolist() == expected['splits']['test'][name]['confusion']['counts']
        act = confidence > 0.9 + 1e-12
        error_count = int((act & ~true_correct).sum())
        reviewed = int((~act).sum())
        realized_cost = (10 * error_count + reviewed) / 18
        entry = expected['splits']['test'][name]['policy_sweep'][3]
        assert entry['acted'] == int(act.sum())
        assert entry['realized_mean_cost'] == realized_cost
        bin_mask = confidence >= 0.8
        recorded = expected['splits']['test'][name]['reliability']['bins'][-1]
        assert recorded['count'] == int(bin_mask.sum())
        assert abs(recorded['mean_confidence'] - confidence[bin_mask].mean().item()) < 1e-12
        result[name] = {'confusion': confusion.tolist(), 'acted': int(act.sum()),
                        'errors_acted': error_count, 'reviewed': reviewed,
                        'cost_per_request': realized_cost,
                        'highest_bin_count': int(bin_mask.sum()),
                        'highest_bin_mean_confidence': confidence[bin_mask].mean().item()}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact', help='Optional retained 600-step core artifact to audit displayed claims')
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = ROOT / 'docs/teaching/projects/evidence/typed-decision-depth-independent.json'
    output.write_text(json.dumps({'passed': False, 'state': 'running'}) + '\n')
    report = {'browser_math': browser_model_checks(), 'diagnostics': diagnostic_checks(),
              'library_adapter': adapter_checks(), 'torch_version': torch.__version__}
    if args.artifact:
        report['retained_checkpoint'] = retained_checkpoint_checks(args.artifact)
    files = ['mechanism-models.js', 'mechanism-labs.jsx', 'mechanism-depth.jsx',
             'research-depth.jsx', 'project-elements.jsx', 'content.jsx']
    paths = [ROOT / 'src/learn/data/projects/typed-decision-model' / name for name in files]
    paths += [PROGRAMS / name for name in ['typed_decision.py', 'research_tools.py', 'pretrained_decision.py', 'trace-fixture.json']]
    report['source_sha256'] = {str(path.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    report['passed'] = True
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'source_sha256'}, indent=2))


if __name__ == '__main__':
    main()
