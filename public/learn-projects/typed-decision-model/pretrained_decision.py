"""A practical Transformers encoder adapter for the project's choice task.

Python 3.10+, torch>=2.6,<3, transformers>=4.48,<5. Run `verify` completely
offline with a tiny random BERT and local tokenizer. Real pretrained weights
are read locally by default; --allow-download explicitly enables Hub access.
This is not Laya's checkpoint format and is not a Jev reproduction.
"""
import argparse
import json
import math
import random
import tempfile
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
import transformers
from transformers import AutoModel, AutoTokenizer

import typed_decision as core

MARKER = '<decision_option>'


# BEGIN batch
def encode_request(row, tokenizer, max_tokens=128):
    """Construct by token ID; marker positions cannot drift during tokenization."""
    if not isinstance(row, dict):
        raise ValueError('request must be a JSON object')
    options = row.get('options')
    if not isinstance(options, list) or not 2 <= len(options) <= 8:
        raise ValueError('provide two to eight candidates')
    if any(not isinstance(option, dict) for option in options):
        raise ValueError('candidates must be objects')
    texts = [row.get('question'), row.get('state')] + [o.get('description') for o in options]
    identifiers = [o.get('id') for o in options]
    if any(not isinstance(value, str) or not value.strip() or len(value) > 4000 for value in texts + identifiers):
        raise ValueError('question, state, candidate IDs and descriptions need 1–4000 characters')
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('candidate IDs must be unique')
    if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError('this adapter needs a BERT-style tokenizer with CLS, SEP and PAD')
    marker_id = tokenizer.convert_tokens_to_ids(MARKER)
    if marker_id == tokenizer.unk_token_id or MARKER not in tokenizer.get_vocab():
        raise ValueError('register the decision marker before encoding')

    def tokens(text):
        result = tokenizer.encode(text, add_special_tokens=False)
        if not result or marker_id in result:
            raise ValueError('text must tokenize and must not contain the reserved candidate marker')
        return result

    ids = [tokenizer.cls_token_id] + tokens(row['question']) + [tokenizer.sep_token_id]
    positions = []
    for option in options:
        positions.append(len(ids))
        ids += [marker_id] + tokens(option['description']) + [tokenizer.sep_token_id]
    ids += tokens(row['state']) + [tokenizer.sep_token_id]
    if len(ids) > max_tokens:
        raise ValueError(f'{len(ids)} tokens exceed limit {max_tokens}; no silent truncation')
    return ids, positions


def batch_requests(rows, tokenizer, max_tokens=128):
    if not rows:
        raise ValueError('a batch needs at least one request')
    encoded = [encode_request(row, tokenizer, max_tokens) for row in rows]
    length = max(len(ids) for ids, _ in encoded)
    candidates = max(len(positions) for _, positions in encoded)
    ids = torch.full((len(rows), length), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros_like(ids)
    positions = torch.zeros((len(rows), candidates), dtype=torch.long)
    candidate_mask = torch.zeros_like(positions, dtype=torch.bool)
    for index, (tokens, markers) in enumerate(encoded):
        ids[index, :len(tokens)] = torch.tensor(tokens)
        attention_mask[index, :len(tokens)] = 1
        positions[index, :len(markers)] = torch.tensor(markers)
        candidate_mask[index, :len(markers)] = True
    return {'input_ids': ids, 'attention_mask': attention_mask,
            'marker_positions': positions, 'candidate_mask': candidate_mask}
# END batch


# BEGIN adapter
class EncoderDecisionModel(nn.Module):
    def __init__(self, encoder, freeze_encoder=False):
        super().__init__()
        if getattr(encoder.config, 'is_decoder', False) or getattr(encoder.config, 'is_encoder_decoder', False):
            raise ValueError('use a bidirectional encoder, such as BERT or ModernBERT')
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        dimension = encoder.config.hidden_size
        self.scorer = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 1))
        for parameter in encoder.parameters():
            parameter.requires_grad_(not freeze_encoder)
        if freeze_encoder:
            encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        # Head-only training should not randomly change frozen features via dropout.
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, input_ids, attention_mask, marker_positions, candidate_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        index = marker_positions[..., None].expand(-1, -1, hidden.shape[-1])
        candidate_hidden = hidden.gather(1, index)
        logits = self.scorer(candidate_hidden).squeeze(-1)
        return logits.masked_fill(~candidate_mask, float('-inf'))


def register_marker(tokenizer, encoder):
    if MARKER not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({'additional_special_tokens': [*tokenizer.additional_special_tokens, MARKER]})
    if encoder.get_input_embeddings().num_embeddings != len(tokenizer):
        encoder.resize_token_embeddings(len(tokenizer), mean_resizing=False)


def load_backbone(checkpoint, revision, allow_download, freeze_encoder=False):
    options = {'local_files_only': not allow_download, 'trust_remote_code': False}
    if revision:
        options['revision'] = revision
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, **options)
    encoder = AutoModel.from_pretrained(checkpoint, **options)
    register_marker(tokenizer, encoder)
    return EncoderDecisionModel(encoder, freeze_encoder), tokenizer
# END adapter


# BEGIN save_load
def save_artifact(directory, model, tokenizer, temperature, provenance):
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(destination / 'encoder')
    tokenizer.save_pretrained(destination / 'tokenizer')
    torch.save(model.scorer.state_dict(), destination / 'scorer.pt')
    configuration = {'format': 'encoder-option-marker-v1', 'marker': MARKER,
                     'max_tokens': 128, 'temperature': temperature,
                     'freeze_encoder': model.freeze_encoder,
                     'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
                     **provenance}
    (destination / 'decision-config.json').write_text(json.dumps(configuration, indent=2), encoding='utf-8')


def load_artifact(directory):
    source = Path(directory)
    configuration = json.loads((source / 'decision-config.json').read_text(encoding='utf-8'))
    if configuration.get('format') != 'encoder-option-marker-v1' or configuration.get('marker') != MARKER:
        raise ValueError('unsupported decision artifact')
    if not math.isfinite(configuration['temperature']) or configuration['temperature'] <= 0:
        raise ValueError('invalid saved temperature')
    tokenizer = AutoTokenizer.from_pretrained(source / 'tokenizer', local_files_only=True, trust_remote_code=False)
    encoder = AutoModel.from_pretrained(source / 'encoder', local_files_only=True, trust_remote_code=False)
    if encoder.get_input_embeddings().num_embeddings != len(tokenizer):
        raise ValueError('saved tokenizer and embedding sizes differ')
    model = EncoderDecisionModel(encoder, configuration['freeze_encoder'])
    model.scorer.load_state_dict(torch.load(source / 'scorer.pt', map_location='cpu', weights_only=True))
    return model.eval(), tokenizer, configuration
# END save_load


# BEGIN train
def train(args):
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    model, tokenizer = load_backbone(args.checkpoint, args.revision, args.allow_download, args.freeze_encoder)
    dataset = core.fixture_data()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW([
        {'params': [p for p in trainable if p.ndim > 1], 'weight_decay': 0.01},
        {'params': [p for p in trainable if p.ndim <= 1], 'weight_decay': 0.0},
    ], lr=args.learning_rate)
    trace = []
    model.train()
    for step in range(args.steps):
        rows = core.permute_candidates(dataset['train'], rng)
        batch = batch_requests(rows, tokenizer)
        targets = torch.tensor([row['target'] for row in rows])
        loss = F.cross_entropy(model(**batch), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % 10 == 0 or step + 1 == args.steps:
            trace.append({'step': step + 1, 'train_nll': loss.item()})
    model.eval()
    with torch.inference_mode():
        logits = {split: model(**batch_requests(rows, tokenizer)) for split, rows in dataset.items()}
        temperature, boundary = core.fit_temperature(logits['calibration'],
                                                     torch.tensor([r['target'] for r in dataset['calibration']]))
        report = {'scope': 'Constructed routing fixture only; no claim of pretrained model performance.',
                  'steps': args.steps, 'seed': args.seed, 'learning_rate': args.learning_rate,
                  'freeze_encoder': args.freeze_encoder, 'temperature': temperature,
                  'temperature_at_search_boundary': boundary, 'training_trace': trace,
                  'test_raw': core.metrics(logits['test'], dataset['test']),
                  'test_calibrated': core.metrics(logits['test'], dataset['test'], temperature),
                  'stress_calibrated': core.metrics(logits['stress'], dataset['stress'], temperature),
                  'test_lexical': core.metrics(core.lexical_logits(dataset['test']), dataset['test'])}
    save_artifact(args.output, model, tokenizer, temperature,
                  {'checkpoint': args.checkpoint, 'revision': args.revision,
                   'resolved_commit': getattr(model.encoder.config, '_commit_hash', None),
                   'seed': args.seed, 'data': 'canonical constructed fixture_data'})
    (Path(args.output) / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
# END train


def predict(args):
    model, tokenizer, config = load_artifact(args.artifact)
    row = json.loads(Path(args.input).read_text(encoding='utf-8'))
    with torch.inference_mode():
        logits = model(**batch_requests([row], tokenizer, config['max_tokens']))
        probabilities = (logits[0] / config['temperature']).softmax(-1).tolist()
    best, action, cost = core.expected_cost_decision(probabilities, args.wrong_cost, args.review_cost)
    return {'choice': row['options'][best]['id'], 'decision': action,
            'probabilities': {option['id']: p for option, p in zip(row['options'], probabilities)},
            'estimated_act_cost': cost, 'temperature': config['temperature']}


def verify():
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
    from transformers import BertConfig, PreTrainedTokenizerFast
    torch.manual_seed(19)
    vocabulary = core.make_vocabulary(core.fixture_data()['train'])
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token='<unk>'))
    backend.normalizer = normalizers.Lowercase()
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='<unk>',
                                       pad_token='<pad>', cls_token='<cls>', sep_token='<sep>')
    config = BertConfig(vocab_size=len(tokenizer), hidden_size=32, num_hidden_layers=2,
                        num_attention_heads=4, intermediate_size=64, max_position_embeddings=128,
                        hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
    encoder = AutoModel.from_config(config)
    old_size = encoder.get_input_embeddings().num_embeddings
    register_marker(tokenizer, encoder)
    assert len(tokenizer) == old_size + 1
    model = EncoderDecisionModel(encoder).eval()
    first = core.fixture_data()['train'][0]
    second = {**first, 'state': 'refund', 'options': first['options'][:2]}
    batch = batch_requests([first, second], tokenizer)
    logits = model(**batch)
    marker_id = tokenizer.convert_tokens_to_ids(MARKER)
    gathered = batch['input_ids'].gather(1, batch['marker_positions'])
    assert (gathered[batch['candidate_mask']] == marker_id).all()
    assert logits[1, 2].isneginf() and logits.softmax(-1)[1, 2] == 0
    padded = {**batch, 'input_ids': F.pad(batch['input_ids'], (0, 3), value=tokenizer.pad_token_id),
              'attention_mask': F.pad(batch['attention_mask'], (0, 3))}
    torch.testing.assert_close(logits, model(**padded), atol=2e-6, rtol=2e-5)
    loss = F.cross_entropy(logits, torch.tensor([0, 1]))
    loss.backward()
    assert model.scorer[1].weight.grad.norm() > 0
    marker_gradient = model.encoder.get_input_embeddings().weight.grad[marker_id].norm().item()
    assert marker_gradient > 0 and math.isfinite(marker_gradient)
    with tempfile.TemporaryDirectory(prefix='typed-decision-adapter-') as temporary:
        # Test a real optimizer update before serialization; no downloaded checkpoint.
        torch.optim.AdamW(model.parameters(), lr=0.001).step()
        expected = model(**batch).detach()
        save_artifact(temporary, model, tokenizer, 1.25, {'checkpoint': 'offline-random-tiny-bert'})
        restored, restored_tokenizer, saved = load_artifact(temporary)
        torch.testing.assert_close(expected, restored(**batch_requests([first, second], restored_tokenizer)),
                                   atol=0, rtol=0)
        assert saved['temperature'] == 1.25
    frozen = EncoderDecisionModel(encoder, freeze_encoder=True).train()
    assert not frozen.encoder.training
    assert not any(parameter.requires_grad for parameter in frozen.encoder.parameters())
    assert all(parameter.requires_grad for parameter in frozen.scorer.parameters())
    frozen.zero_grad(set_to_none=True)
    F.cross_entropy(frozen(**batch), torch.tensor([0, 1])).backward()
    assert all(parameter.grad is None for parameter in frozen.encoder.parameters())
    assert frozen.scorer[1].weight.grad.norm() > 0
    for bad in [{**first, 'state': MARKER}, {**first, 'state': 'refund ' * 150}]:
        try:
            batch_requests([bad], tokenizer)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid marker or overlong text was accepted')
    return {'passed': True, 'torch_version': torch.__version__, 'transformers_version': transformers.__version__,
            'scope': 'Offline random tiny BERT and local WordLevel tokenizer; no pretrained weights downloaded or fine-tuned.',
            'marker_embedding_gradient_norm': marker_gradient,
            'checks': ['marker registration and embedding resize', 'marker gather alignment',
                       'variable candidates and padding', 'full-encoder gradients', 'optimizer step',
                       'exact local tokenizer/model/head save-load parity', 'frozen encoder and trainable head',
                       'reserved marker and length rejection']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('verify')
    check.add_argument('--output')
    training = commands.add_parser('train')
    training.add_argument('--checkpoint', required=True)
    training.add_argument('--revision', help='Pin a Hub commit for reproducibility')
    training.add_argument('--allow-download', action='store_true')
    training.add_argument('--freeze-encoder', action='store_true')
    training.add_argument('--learning-rate', type=float, default=2e-5)
    training.add_argument('--steps', type=int, default=60)
    training.add_argument('--seed', type=int, default=7)
    training.add_argument('--output', default='pretrained-artifact')
    inference = commands.add_parser('predict')
    inference.add_argument('--artifact', required=True)
    inference.add_argument('--input', required=True)
    inference.add_argument('--wrong-cost', type=float, default=10)
    inference.add_argument('--review-cost', type=float, default=1)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.command == 'train' and (args.steps < 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error('steps and learning rate must be positive and finite')
    result = verify() if args.command == 'verify' else globals()[args.command](args)
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.command == 'verify' and args.output:
        Path(args.output).write_text(text + '\n', encoding='utf-8')
    else:
        print(text)


if __name__ == '__main__':
    main()
