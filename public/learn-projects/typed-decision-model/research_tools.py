"""Inspect and evaluate the canonical tiny decision model without changing it.

Run next to typed_decision.py. Trace uses seeded random weights, not a trained
model. Evaluate only reads an existing artifact; it never tunes on test labels.
"""
import argparse
import copy
import hashlib
import itertools
import json
import math
import statistics
import time
from pathlib import Path

import torch
from torch.nn import functional as F

import typed_decision as core


def write_report(report, output=None):
    text = json.dumps(report, indent=2, allow_nan=False)
    if output:
        Path(output).write_text(text + '\n', encoding='utf-8')
    else:
        print(text)


# BEGIN trace
def trace():
    torch.manual_seed(11)
    vocabulary = core.make_vocabulary(core.fixture_data()['train'])
    row = {'question': core.QUESTION, 'state': 'please help with my refund',
           'options': core.OPTIONS, 'target': 0}
    shorter = {**row, 'state': 'refund', 'options': row['options'][:2]}
    ids, markers = core.encode(row, vocabulary)
    batch = core.batch_rows([row, shorter], vocabulary)
    token_ids, marker_positions, candidate_mask = batch
    model = core.TypedDecisionModel(len(vocabulary)).double().eval()
    inverse_vocabulary = {value: key for key, value in vocabulary.items()}
    with torch.no_grad():
        hidden = model.token(token_ids) + model.position(torch.arange(token_ids.shape[1]))[None]
        block = model.blocks[0]
        projected = block.qkv(block.attention_norm(hidden))
        query, key, value = projected.view(2, len(ids), 3, 4, 8).permute(2, 0, 3, 1, 4).unbind(0)
        attention = (query @ key.transpose(-2, -1) / math.sqrt(8)).masked_fill(
            ~token_ids.ne(0)[:, None, None, :], float('-inf')).softmax(-1)
    logits = model(*batch)
    targets = torch.tensor([0, 0])
    loss = core.negative_log_likelihood(logits, targets)
    loss.backward()
    before_weight = model.scorer[1].weight.detach().clone()
    scorer_gradient = model.scorer[1].weight.grad.detach().clone()
    torch.optim.SGD(model.parameters(), lr=0.01).step()
    after_logits = model(*batch).detach()
    finite_logits = lambda tensor: [[None if not math.isfinite(x) else x for x in row]
                                   for row in tensor.detach().tolist()]
    analytic = torch.tensor([[2.0, 1.0, 0.0]], dtype=torch.float64, requires_grad=True)
    analytic_loss = core.negative_log_likelihood(analytic, torch.tensor([0]))
    analytic_loss.backward()
    expected = analytic.detach().softmax(-1) - F.one_hot(torch.tensor([0]), 3)
    torch.testing.assert_close(analytic.grad, expected)
    torch.testing.assert_close(model.scorer[1].weight, before_weight - 0.01 * scorer_gradient)
    return {
        'scope': 'Seed-11 random weights; one illustrative SGD update, not the trained author result.',
        'vocabulary': vocabulary, 'request': row, 'tokens': [inverse_vocabulary[i] for i in ids],
        'ids': ids, 'markers': markers,
        'batch': {'input_ids': token_ids.tolist(), 'attention_mask': token_ids.ne(0).tolist(),
                  'marker_positions': marker_positions.tolist(), 'candidate_mask': candidate_mask.tolist()},
        'shapes': {'tokens': list(token_ids.shape), 'hidden': list(hidden.shape),
                   'qkv_projection': list(projected.shape), 'query': list(query.shape),
                   'attention': list(attention.shape), 'candidate_hidden': [2, 3, 32],
                   'logits': list(logits.shape)},
        'first_head_refund_candidate_attention': attention[0, 0, markers[0]].tolist(),
        'one_model_update': {'learning_rate': 0.01, 'loss_before': loss.item(),
                             'logits_before': finite_logits(logits),
                             'scorer_gradient_norm': scorer_gradient.norm().item(),
                             'scorer_weight_delta_norm': (model.scorer[1].weight - before_weight).norm().item(),
                             'logits_after': finite_logits(after_logits),
                             'loss_after': core.negative_log_likelihood(after_logits, targets).item()},
        'analytic_logit_update': {'logits': analytic.detach()[0].tolist(), 'target': 0,
                                  'probabilities': analytic.detach().softmax(-1)[0].tolist(),
                                  'loss': analytic_loss.item(), 'gradient': analytic.grad[0].tolist(),
                                  'learning_rate': 0.1,
                                  'updated_logits': (analytic.detach() - 0.1 * analytic.grad)[0].tolist()},
    }
# END trace


# BEGIN diagnostics
def reliability(probabilities, targets, count=5):
    confidence, choice = probabilities.max(-1)
    correct = choice.eq(targets).double()
    bins = []
    ece = 0.0
    for index in range(count):
        low, high = index / count, (index + 1) / count
        selected = (confidence >= low) & ((confidence < high) if index < count - 1 else (confidence <= high))
        size = int(selected.sum())
        average = float(confidence[selected].mean()) if size else None
        accuracy = float(correct[selected].mean()) if size else None
        if size:
            ece += size / len(targets) * abs(average - accuracy)
        bins.append({'lower_inclusive': low, 'upper': high, 'upper_inclusive': index == count - 1,
                     'count': size, 'mean_confidence': average, 'accuracy': accuracy})
    return {'definition': 'Top-choice confidence versus top-choice accuracy; equal-width bins; empty bins null.',
            'ece': ece, 'bins': bins}


def policy_sweep(probabilities, targets, wrong_cost=10.0):
    result = []
    for review_cost in [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]:
        acted = errors = 0
        realized = estimated = 0.0
        for distribution, target in zip(probabilities.tolist(), targets.tolist()):
            best, action, act_cost = core.expected_cost_decision(distribution, wrong_cost, review_cost)
            if action == 'act':
                acted += 1
                errors += best != target
                realized += wrong_cost * (best != target)
                estimated += act_cost
            else:
                realized += review_cost
                estimated += review_cost
        result.append({'wrong_cost': wrong_cost, 'review_cost': review_cost,
                       'acted': acted, 'reviewed': len(targets) - acted,
                       'coverage': acted / len(targets), 'selective_risk': errors / acted if acted else None,
                       'realized_mean_cost': realized / len(targets),
                       'estimated_mean_cost': estimated / len(targets)})
    return result


def summarize(logits, rows, temperature, vocabulary):
    targets = torch.tensor([row['target'] for row in rows])
    probabilities = (logits.double() / temperature).softmax(-1)
    confusion = torch.zeros((logits.shape[1], logits.shape[1]), dtype=torch.long)
    for target, choice in zip(targets, probabilities.argmax(-1)):
        confusion[target, choice] += 1
    examples = []
    for row, distribution in zip(rows, probabilities.tolist()):
        chosen = max(range(len(distribution)), key=distribution.__getitem__)
        state_words = core.words(row['state'])
        examples.append({'id': row['id'], 'state': row['state'],
                         'target_id': row['options'][row['target']]['id'],
                         'chosen_id': row['options'][chosen]['id'],
                         'probabilities': {o['id']: p for o, p in zip(row['options'], distribution)},
                         'state_word_count': len(state_words),
                         'state_unknown_word_count': sum(word not in vocabulary for word in state_words)})
    return {**core.metrics(logits, rows, temperature), 'examples': examples,
            'confusion': {'rows': 'true', 'columns': 'chosen', 'labels': [o['id'] for o in core.OPTIONS],
                          'counts': confusion.tolist()},
            'reliability': reliability(probabilities, targets),
            'policy_sweep': policy_sweep(probabilities, targets)}


def perturbation_checks(model, vocabulary, rows):
    largest_change = 0.0
    changed = total = 0
    for row in rows:
        base = model(*core.batch_rows([row], vocabulary))[0].softmax(-1)
        for order in itertools.permutations(range(len(row['options']))):
            permuted = {**row, 'options': [row['options'][index] for index in order]}
            scores = model(*core.batch_rows([permuted], vocabulary))[0].softmax(-1)
            restored = scores[torch.tensor([order.index(index) for index in range(len(order))])]
            largest_change = max(largest_change, (restored - base).abs().max().item())
            changed += int(order[scores.argmax().item()] != base.argmax().item())
            total += 1
    renamed = copy.deepcopy(rows)
    for row in renamed:
        for index, option in enumerate(row['options']):
            option['id'] = f'opaque-identifier-{index}'
    base_logits = model(*core.batch_rows(rows, vocabulary))
    renamed_logits = model(*core.batch_rows(renamed, vocabulary))
    torch.testing.assert_close(base_logits, renamed_logits, atol=0, rtol=0)
    return {'option_order': {'permutations_including_identity': total,
                            'max_probability_change': largest_change,
                            'choice_change_fraction': changed / total,
                            'temperature': 1.0, 'claim': 'Measured sensitivity, not guaranteed permutation invariance.'},
            'id_renaming': {'max_logit_change': (base_logits - renamed_logits).abs().max().item(),
                            'claim': 'IDs are outside encoded text; renaming must preserve logits.'}}
# END diagnostics


def evaluate(artifact, latency_runs=0):
    directory = Path(artifact)
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    temperature = config['temperature']
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('artifact temperature must be finite and positive')
    dataset = json.loads((directory / 'fixture-data.json').read_text(encoding='utf-8'))
    model = core.TypedDecisionModel(len(config['vocabulary'])).eval()
    model.load_state_dict(torch.load(directory / 'weights.pt', map_location='cpu', weights_only=True))
    with torch.inference_mode():
        report = {'scope': 'Post-hoc diagnostics of fixed constructed fixtures; no model selection or temperature refit.',
                  'torch_version': torch.__version__, 'temperature': temperature,
                  'weights_sha256': hashlib.sha256((directory / 'weights.pt').read_bytes()).hexdigest(),
                  'definitions': {'coverage': 'Fraction acted on rather than reviewed.',
                                  'selective_risk': 'Observed error fraction among acted cases; null if none.',
                                  'cost': 'Wrong action costs 10, correct action zero; review assumed perfect at its fixed cost.',
                                  'sweep': 'Diagnostic counterfactuals on the fixed test set, not threshold tuning recommendations.'},
                  'splits': {}}
        for split in ['test', 'stress']:
            rows = dataset[split]
            logits = model(*core.batch_rows(rows, config['vocabulary']))
            report['splits'][split] = {'raw': summarize(logits, rows, 1.0, config['vocabulary']),
                                       'calibrated': summarize(logits, rows, temperature, config['vocabulary']),
                                       'lexical': core.metrics(core.lexical_logits(rows), rows),
                                       **perturbation_checks(model, config['vocabulary'], rows)}
        if latency_runs:
            # The CPU transfer and tokenizer are outside this narrowly defined timing.
            batch = core.batch_rows([dataset['test'][0]], config['vocabulary'])
            for _ in range(10):
                model(*batch, library_attention=True)
            timings = []
            for _ in range(latency_runs):
                started = time.perf_counter()
                model(*batch, library_attention=True)
                timings.append((time.perf_counter() - started) * 1000)
            report['latency'] = {'runs': latency_runs, 'warmups': 10, 'batch_size': 1,
                                 'cpu_threads': torch.get_num_threads(), 'median_ms': statistics.median(timings),
                                 'p95_nearest_rank_ms': sorted(timings)[math.ceil(0.95 * latency_runs) - 1],
                                 'scope': 'Warm CPU forward only; excludes tokenization, loading, JSON, network and queuing. Not a service benchmark.'}
    return report


def verify():
    fixture = trace()
    assert len(fixture['ids']) == 35 and fixture['markers'] == [8, 15, 22]
    assert fixture['batch']['candidate_mask'] == [[True, True, True], [True, True, False]]
    probabilities = torch.tensor([[0.9, 0.1], [0.4, 0.6]], dtype=torch.float64)
    targets = torch.tensor([0, 0])
    bins = reliability(probabilities, targets)
    assert sum(item['count'] for item in bins['bins']) == 2
    sweep = policy_sweep(probabilities, targets)
    assert sweep[0]['coverage'] == 0 and sweep[0]['selective_risk'] is None
    assert sweep[3]['coverage'] == 0  # p=.9 is a tie, therefore review.
    assert sweep[-1]['coverage'] == 1 and sweep[-1]['selective_risk'] == 0.5
    assert sweep[-1]['realized_mean_cost'] == 5
    return {'passed': True, 'checks': ['exact encoding', 'variable-candidate masking', 'analytic CE gradient',
                                      'one model parameter update', 'reliability accounting', 'review cost/tie/null risk']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for command in ['trace', 'verify']:
        commands.add_parser(command).add_argument('--output')
    diagnostic = commands.add_parser('evaluate')
    diagnostic.add_argument('--artifact', required=True)
    diagnostic.add_argument('--latency-runs', type=int, default=0)
    diagnostic.add_argument('--output')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == 'evaluate' and not 0 <= args.latency_runs <= 10000:
        parser.error('--latency-runs must be between 0 and 10000')
    report = evaluate(args.artifact, args.latency_runs) if args.command == 'evaluate' else globals()[args.command]()
    write_report(report, args.output)


if __name__ == '__main__':
    main()
