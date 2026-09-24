"""Retained bounded native checks; run from the repository root.

Requires torch and transformers as described by pretrained_decision.py. Creates
and removes a temporary local checkpoint. Never downloads model weights.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ['HF_HUB_OFFLINE'] = '1'
ROOT = Path(__file__).resolve().parents[4]
PROGRAMS = ROOT / 'public/learn-projects/typed-decision-model'
sys.path.insert(0, str(PROGRAMS))
import torch
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers
from transformers import AutoModel, BertConfig, ModernBertConfig, PreTrainedTokenizerFast

import pretrained_decision as adapter
import research_tools as research
import typed_decision as core


def main():
    torch.set_num_threads(1)
    report = {'research': research.verify(), 'adapter': adapter.verify()}
    torch.manual_seed(23)
    vocabulary = core.make_vocabulary(core.fixture_data()['train'])
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token='<unk>'))
    backend.normalizer = normalizers.Lowercase()
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='<unk>',
                                       pad_token='<pad>', cls_token='<cls>', sep_token='<sep>')
    encoder = AutoModel.from_config(BertConfig(vocab_size=len(tokenizer), hidden_size=32,
                                              num_hidden_layers=1, num_attention_heads=4,
                                              intermediate_size=64, max_position_embeddings=128,
                                              hidden_dropout_prob=0, attention_probs_dropout_prob=0))
    with tempfile.TemporaryDirectory(prefix='typed-decision-depth-flow-') as temporary:
        base = Path(temporary) / 'random-local-encoder'
        tokenizer.save_pretrained(base)
        encoder.save_pretrained(base)
        outcomes = {}
        for frozen in [False, True]:
            output = Path(temporary) / ('frozen' if frozen else 'full')
            args = argparse.Namespace(checkpoint=str(base), revision=None, allow_download=False,
                                      freeze_encoder=frozen, learning_rate=0.001, steps=3, seed=7,
                                      output=str(output))
            result = adapter.train(args)
            assert len(result['training_trace']) == 2
            restored, loaded_tokenizer, config = adapter.load_artifact(output)
            assert config['freeze_encoder'] == frozen
            prediction = adapter.predict(argparse.Namespace(artifact=str(output),
                                         input=str(PROGRAMS / 'request.json'), wrong_cost=10, review_cost=1))
            assert set(prediction['probabilities']) == {'billing', 'access', 'delivery'}
            assert abs(sum(prediction['probabilities'].values()) - 1) < 1e-6
            assert prediction['choice'] in prediction['probabilities']
            outcomes['frozen' if frozen else 'full'] = {'steps': 3, 'train_nll': result['training_trace'],
                                                       'saved_reload_prediction': prediction,
                                                       'scope': 'Offline execution smoke test, not evidence of useful training.'}
        report['offline_training_flow'] = outcomes
    modern = AutoModel.from_config(ModernBertConfig(vocab_size=len(tokenizer), hidden_size=32,
                                                   num_hidden_layers=2, num_attention_heads=4,
                                                   intermediate_size=64, max_position_embeddings=128,
                                                   local_attention=64, global_attn_every_n_layers=2,
                                                   pad_token_id=tokenizer.pad_token_id,
                                                   cls_token_id=tokenizer.cls_token_id,
                                                   sep_token_id=tokenizer.sep_token_id))
    adapter.register_marker(tokenizer, modern)
    modern_model = adapter.EncoderDecisionModel(modern).eval()
    first = core.fixture_data()['train'][0]
    batch = adapter.batch_requests([first, {**first, 'options': first['options'][:2]}], tokenizer)
    logits = modern_model(**batch)
    assert logits[1, 2].isneginf()
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1]))
    loss.backward()
    assert modern_model.scorer[1].weight.grad.norm() > 0
    assert all(p.grad is None or p.grad.isfinite().all() for p in modern_model.parameters())
    report['modernbert'] = {'scope': 'Offline random tiny ModernBERT; no pretrained checkpoint.',
                            'loss': loss.item(), 'candidate_mask': batch['candidate_mask'].tolist(),
                            'passed': ['AutoModel construction', 'marker resize', 'variable candidates',
                                       'forward and finite encoder/head gradients']}
    probabilities = torch.tensor([[0.2, 0.8], [0.9, 0.1], [0.6, 0.4]], dtype=torch.float64)
    labels = torch.tensor([1, 1, 0])
    for item in research.policy_sweep(probabilities, labels):
        assert item['acted'] + item['reviewed'] == 3
        assert item['coverage'] == item['acted'] / 3
        assert item['realized_mean_cost'] >= 0
        assert (item['selective_risk'] is None) == (item['acted'] == 0)
    report['source_sha256'] = {
        name: hashlib.sha256((PROGRAMS / name).read_bytes()).hexdigest()
        for name in ['typed_decision.py', 'research_tools.py', 'pretrained_decision.py', 'trace-fixture.json']
    }
    report['scope'] = 'Actual bounded native execution. No remote checkpoint, pretrained quality claim, GPU, service or latency guarantee.'
    path = ROOT / 'docs/teaching/projects/evidence/typed-decision-native-depth.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('PASS: trace, masks, gradients, save/load, frozen/full training flows and policy accounting')


if __name__ == '__main__':
    main()
