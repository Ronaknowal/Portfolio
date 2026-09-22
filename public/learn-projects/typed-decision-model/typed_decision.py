"""Small typed decision model: transparent attention, calibration, JSON inference.

Python 3.10+, PyTorch 2.6+. No network, pretrained weights, or external dataset.
Run `python typed_decision.py --help`. Data are deliberately constructed fixtures.
"""
import argparse
import json
import math
import random
import re
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

SPECIAL = ['<pad>', '<unk>', '<cls>', '<sep>', '<option>']
OPTIONS = [
    {'id': 'billing', 'description': 'payment invoice charge refund billing'},
    {'id': 'access', 'description': 'login password account access locked'},
    {'id': 'delivery', 'description': 'shipping delivery parcel tracking delayed'},
]
QUESTION = 'Which team should handle this request?'
MAX_TOKENS = 128


def words(text):
    return re.findall(r"[a-z]+", text.lower())


def fixture_data():
    """Disjoint sentence families; known task vocabulary intentionally overlaps."""
    patterns = {
        'train': ['please help with my {}', 'there is a problem with the {}',
                  'i need an update about {}', 'can someone check the {}'],
        'calibration': ['i am contacting you regarding {}', 'who can resolve this {} issue'],
        'test': ['this request concerns {}', 'my unresolved case is about {}'],
    }
    anchors = [['invoice', 'refund', 'payment'], ['login', 'password', 'account'],
               ['parcel', 'tracking', 'delivery']]
    result = {}
    for split, templates in patterns.items():
        result[split] = [
            {'id': f'{split}-{label}-{template_index}-{anchor_index}',
             'question': QUESTION, 'state': template.format(anchor),
             'options': OPTIONS, 'target': label}
            for label, group in enumerate(anchors)
            for anchor_index, anchor in enumerate(group)
            for template_index, template in enumerate(templates)
        ]
    result['stress'] = [
        {'id': 'stress-money', 'question': QUESTION,
         'state': 'money left my bank twice', 'options': OPTIONS, 'target': 0},
        {'id': 'stress-access', 'question': QUESTION,
         'state': 'the site will not let me sign in', 'options': OPTIONS, 'target': 1},
        {'id': 'stress-delivery', 'question': QUESTION,
         'state': 'the box never arrived at my door', 'options': OPTIONS, 'target': 2},
    ]
    return result


def validate_request(row):
    if not isinstance(row, dict):
        raise ValueError('request must be a JSON object')
    for field in ['question', 'state']:
        if not isinstance(row.get(field), str) or not row[field].strip():
            raise ValueError(f'{field} must be a nonempty string')
        if len(row[field]) > 4000:
            raise ValueError(f'{field} exceeds the 4000-character input limit')
        if not words(row[field]):
            raise ValueError(f'{field} needs an ASCII word; this toy tokenizer only reads a-z')
    options = row.get('options')
    if not isinstance(options, list) or not 2 <= len(options) <= 8:
        raise ValueError('provide between two and eight candidate options')
    identifiers = []
    for option in options:
        if not isinstance(option, dict):
            raise ValueError('each option must contain id and description strings')
        for key in ['id', 'description']:
            if not isinstance(option.get(key), str) or not option[key].strip():
                raise ValueError(f'each option needs a nonempty {key}')
            if len(option[key]) > 500:
                raise ValueError('option fields may contain at most 500 characters')
        identifiers.append(option['id'])
        if not words(option['description']):
            raise ValueError('candidate descriptions need an ASCII word; only a-z is modeled')
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('candidate IDs must be unique')


def make_vocabulary(training_rows):
    tokens = set()
    for row in training_rows:
        tokens.update(words(row['question']) + words(row['state']))
        for option in row['options']:
            tokens.update(words(option['description']))
    return {word: index for index, word in enumerate(SPECIAL + sorted(tokens))}


def encode(row, vocabulary):
    validate_request(row)
    lookup = lambda text: [vocabulary.get(word, 1) for word in words(text)]
    ids = [2] + lookup(row['question']) + [3]
    markers = []
    for option in row['options']:
        markers.append(len(ids))
        ids += [4] + lookup(option['description']) + [3]
    ids += lookup(row['state']) + [3]
    if len(ids) > MAX_TOKENS:
        raise ValueError(f'request needs {len(ids)} tokens; maximum is {MAX_TOKENS}')
    return ids, markers


def batch_rows(rows, vocabulary):
    encoded = [encode(row, vocabulary) for row in rows]
    width = max(len(ids) for ids, _ in encoded)
    candidates = max(len(markers) for _, markers in encoded)
    ids = torch.zeros((len(rows), width), dtype=torch.long)
    marker_positions = torch.zeros((len(rows), candidates), dtype=torch.long)
    candidate_mask = torch.zeros((len(rows), candidates), dtype=torch.bool)
    for index, (tokens, markers) in enumerate(encoded):
        ids[index, :len(tokens)] = torch.tensor(tokens)
        marker_positions[index, :len(markers)] = torch.tensor(markers)
        candidate_mask[index, :len(markers)] = True
    return ids, marker_positions, candidate_mask


class AttentionBlock(nn.Module):
    def __init__(self, dimension=32, heads=4):
        super().__init__()
        self.heads = heads
        self.attention_norm = nn.LayerNorm(dimension)
        self.qkv = nn.Linear(dimension, 3 * dimension)
        self.output = nn.Linear(dimension, dimension)
        self.feed_norm = nn.LayerNorm(dimension)
        self.feed = nn.Sequential(nn.Linear(dimension, 4 * dimension), nn.GELU(),
                                  nn.Linear(4 * dimension, dimension))

    def forward(self, hidden, token_mask, library_attention=False):
        batch, length, dimension = hidden.shape
        projected = self.qkv(self.attention_norm(hidden))
        projected = projected.view(batch, length, 3, self.heads, dimension // self.heads)
        query, key, value = projected.permute(2, 0, 3, 1, 4).unbind(0)
        keep = token_mask[:, None, None, :]
        if library_attention:
            attended = F.scaled_dot_product_attention(query, key, value, attn_mask=keep)
        else:
            scores = (query @ key.transpose(-2, -1)) / math.sqrt(dimension // self.heads)
            attention = scores.masked_fill(~keep, float('-inf')).softmax(-1)
            attended = attention @ value
        attended = attended.transpose(1, 2).reshape(batch, length, dimension)
        hidden = hidden + self.output(attended)
        return hidden + self.feed(self.feed_norm(hidden))


class TypedDecisionModel(nn.Module):
    """Original tiny teaching model; not a reproduction of Jev or Laya."""
    def __init__(self, vocabulary_size, dimension=32):
        super().__init__()
        self.token = nn.Embedding(vocabulary_size, dimension, padding_idx=0)
        self.position = nn.Embedding(MAX_TOKENS, dimension)
        self.blocks = nn.ModuleList([AttentionBlock(dimension) for _ in range(2)])
        self.scorer = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 1))

    def forward(self, ids, marker_positions, candidate_mask, library_attention=False):
        position = torch.arange(ids.shape[1], device=ids.device)
        hidden = self.token(ids) + self.position(position)[None]
        for block in self.blocks:
            hidden = block(hidden, ids != 0, library_attention)
        index = marker_positions[..., None].expand(-1, -1, hidden.shape[-1])
        candidate_hidden = hidden.gather(1, index)
        logits = self.scorer(candidate_hidden).squeeze(-1)
        return logits.masked_fill(~candidate_mask, float('-inf'))


def negative_log_likelihood(logits, targets):
    """Stable scratch cross-entropy; includes the candidate mask in logits."""
    selected = logits.gather(1, targets[:, None]).squeeze(1)
    return (torch.logsumexp(logits, dim=-1) - selected).mean()


def lexical_logits(rows):
    return torch.tensor([
        [len(set(words(row['state'])) & set(words(option['description'])))
         for option in row['options']] for row in rows
    ], dtype=torch.float32)


def metrics(logits, rows, temperature=1.0):
    targets = torch.tensor([row['target'] for row in rows])
    probabilities = (logits / temperature).softmax(-1)
    one_hot = F.one_hot(targets, logits.shape[1]).float()
    return {
        'n': len(rows),
        'accuracy': round((probabilities.argmax(-1) == targets).float().mean().item(), 6),
        'nll': round(negative_log_likelihood(logits / temperature, targets).item(), 6),
        'brier_sum': round(((probabilities - one_hot) ** 2).sum(-1).mean().item(), 6),
    }


def fit_temperature(logits, targets):
    """Bounded log grid, including T=1; inspect boundary optima in the report."""
    candidates = sorted(set([1.0] + [math.exp(-3 + index * 6 / 120) for index in range(121)]))
    losses = [negative_log_likelihood(logits / temperature, targets).item()
              for temperature in candidates]
    best = min(range(len(losses)), key=losses.__getitem__)
    return candidates[best], best in (0, len(candidates) - 1)


def expected_cost_decision(probabilities, wrong_cost, review_cost):
    if not math.isfinite(wrong_cost) or wrong_cost <= 0:
        raise ValueError('wrong-cost must be finite and positive')
    if not math.isfinite(review_cost) or review_cost < 0:
        raise ValueError('review-cost must be finite and nonnegative')
    best = max(range(len(probabilities)), key=probabilities.__getitem__)
    act_cost = wrong_cost * (1 - probabilities[best])
    tolerance = 1e-12 * max(1.0, abs(act_cost), abs(review_cost))
    return best, ('act' if act_cost < review_cost - tolerance else 'review'), act_cost


def permute_candidates(rows, rng):
    result = []
    for row in rows:
        order = list(range(len(row['options'])))
        rng.shuffle(order)
        result.append({**row, 'options': [row['options'][index] for index in order],
                       'target': order.index(row['target'])})
    return result


def train(args):
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    dataset = fixture_data()
    vocabulary = make_vocabulary(dataset['train'])
    model = TypedDecisionModel(len(vocabulary))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.01)
    trace = []
    model.train()
    for step in range(args.steps):
        rows = permute_candidates(dataset['train'], rng)
        logits = model(*batch_rows(rows, vocabulary), library_attention=args.library_attention)
        targets = torch.tensor([row['target'] for row in rows])
        loss = negative_log_likelihood(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 20 == 0 or step + 1 == args.steps:
            trace.append({'step': step + 1, 'train_nll': round(loss.item(), 6)})
    model.eval()
    with torch.inference_mode():
        logits = {split: model(*batch_rows(rows, vocabulary)).detach()
                  for split, rows in dataset.items()}
        targets = torch.tensor([row['target'] for row in dataset['calibration']])
        temperature, boundary = fit_temperature(logits['calibration'], targets)
        report = {
            'data': 'constructed vocabulary-matching fixtures, not a real task benchmark',
            'seed': args.seed, 'steps': args.steps, 'torch_version': torch.__version__,
            'parameter_count': sum(parameter.numel() for parameter in model.parameters()),
            'temperature': temperature, 'temperature_at_search_boundary': boundary,
            'training_trace': trace,
            'test_lexical': metrics(lexical_logits(dataset['test']), dataset['test']),
            'test_model_raw': metrics(logits['test'], dataset['test']),
            'test_model_calibrated': metrics(logits['test'], dataset['test'], temperature),
            'stress_model_calibrated': metrics(logits['stress'], dataset['stress'], temperature),
            'stress_lexical': metrics(lexical_logits(dataset['stress']), dataset['stress']),
        }
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), destination / 'weights.pt')
    (destination / 'config.json').write_text(json.dumps({
        'vocabulary': vocabulary, 'temperature': temperature,
        'architecture': 'tiny-option-marker-transformer-v1', 'max_tokens': MAX_TOKENS,
        'question_type': 'choice', 'seed': args.seed,
    }, indent=2), encoding='utf-8')
    (destination / 'fixture-data.json').write_text(json.dumps(dataset, indent=2), encoding='utf-8')
    (destination / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


def predict(args):
    directory = Path(args.artifact)
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    model = TypedDecisionModel(len(config['vocabulary']))
    model.load_state_dict(torch.load(directory / 'weights.pt', map_location='cpu', weights_only=True))
    model.eval()
    row = json.loads(Path(args.input).read_text(encoding='utf-8')) if args.input else json.load(sys.stdin)
    with torch.inference_mode():
        logits = model(*batch_rows([row], config['vocabulary']), library_attention=True)
        probabilities = (logits[0] / config['temperature']).softmax(-1).tolist()
    best, action, cost = expected_cost_decision(probabilities, args.wrong_cost, args.review_cost)
    output = {
        'choice': row['options'][best]['id'],
        'probabilities': {option['id']: value for option, value in zip(row['options'], probabilities)},
        'decision': action, 'estimated_act_cost': cost, 'review_cost': args.review_cost,
        'temperature': config['temperature'], 'artifact': config['architecture'],
    }
    print(json.dumps(output, indent=2))


def verify():
    torch.manual_seed(11)
    dataset = fixture_data()
    vocabulary = make_vocabulary(dataset['train'])
    model = TypedDecisionModel(len(vocabulary)).eval()
    rows = dataset['train'][:2]
    shortened = {**rows[0], 'options': rows[0]['options'][:2]}
    batch = batch_rows([rows[0], shortened], vocabulary)
    manual = model(*batch)
    library = model(*batch, library_attention=True)
    torch.testing.assert_close(manual, library, atol=2e-6, rtol=2e-5)
    assert manual[1, 2].isneginf() and manual.softmax(-1)[1, 2] == 0
    targets = torch.tensor([0, 1])
    torch.testing.assert_close(negative_log_likelihood(manual, targets), F.cross_entropy(manual, targets))
    padded = (F.pad(batch[0], (0, 5)), batch[1], batch[2])
    torch.testing.assert_close(manual, model(*padded), atol=2e-6, rtol=2e-5)
    probabilities = torch.tensor([[0.2, 0.3, 0.5]])
    torch.testing.assert_close(probabilities.log().softmax(-1), probabilities)
    shifted = manual + 1000
    torch.testing.assert_close(manual.softmax(-1), shifted.softmax(-1), atol=3e-5, rtol=3e-5)
    negative_log_likelihood(manual, targets).backward()
    assert all(parameter.grad is None or parameter.grad.isfinite().all() for parameter in model.parameters())
    assert expected_cost_decision([0.8, 0.2], 10, 1)[1] == 'review'
    assert expected_cost_decision([0.99, 0.01], 10, 1)[1] == 'act'
    assert expected_cost_decision([0.9, 0.1], 10, 1)[1] == 'review'
    assert expected_cost_decision([0.900001, 0.099999], 10, 1)[1] == 'act'
    assert expected_cost_decision([0.899999, 0.100001], 10, 1)[1] == 'review'
    assert not (set(row['state'] for row in dataset['train']) &
                set(row['state'] for row in dataset['test']))
    try:
        encode({**rows[0], 'state': 'word ' * 150}, vocabulary)
    except ValueError:
        pass
    else:
        raise AssertionError('over-budget request was accepted')
    for unsupported in ['12345', '\u4f60\u597d']:
        for bad_row in [{**rows[0], 'state': unsupported},
                        {**rows[0], 'question': unsupported},
                        {**rows[0], 'options': [
                            {'id': 'a', 'description': unsupported}, OPTIONS[1]]}]:
            try:
                validate_request(bad_row)
            except ValueError:
                pass
            else:
                raise AssertionError('unsupported tokenizer input was accepted')
    print('PASS: attention parity, variable candidates, padding, loss parity, stable softmax, gradients, policy, split, input limits')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    training = commands.add_parser('train')
    training.add_argument('--steps', type=int, default=600)
    training.add_argument('--seed', type=int, default=7)
    training.add_argument('--output', default='decision-artifact')
    training.add_argument('--library-attention', action='store_true')
    inference = commands.add_parser('predict')
    inference.add_argument('--artifact', default='decision-artifact')
    inference.add_argument('--input', help='JSON request file; omit to read standard input')
    inference.add_argument('--wrong-cost', type=float, default=10)
    inference.add_argument('--review-cost', type=float, default=1)
    commands.add_parser('verify')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == 'train' and args.steps < 1:
        parser.error('--steps must be positive')
    try:
        {'train': train, 'predict': predict, 'verify': lambda _: verify()}[args.command](args)
    except (ValueError, json.JSONDecodeError) as error:
        parser.exit(2, f'Input error: {error}\n')


if __name__ == '__main__':
    main()
