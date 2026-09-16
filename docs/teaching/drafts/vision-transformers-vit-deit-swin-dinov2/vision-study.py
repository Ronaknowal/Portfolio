"""Small, offline patch/class-token/distillation study. Run this file to reproduce.

Requires Python 3.12, NumPy, PyTorch and scikit-learn. No network or GPU access.
See experiment-contract.md and provenance.md for the declared protocol.
"""
from pathlib import Path
import copy
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parent


class AttentionBlock(nn.Module):
    def __init__(self, width=32, heads=4):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.output = nn.Linear(width, width)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, width))

    def forward(self, tokens, capture=False):
        batch, length, width = tokens.shape
        query, key, value = self.qkv(self.norm1(tokens)).reshape(
            batch, length, 3, self.heads, width // self.heads
        ).permute(2, 0, 3, 1, 4).unbind(0)
        weights = ((query @ key.transpose(-1, -2)) / (width // self.heads) ** .5).softmax(-1)
        mixed = (weights @ value).transpose(1, 2).reshape(batch, length, width)
        tokens = tokens + self.output(mixed)
        tokens = tokens + self.ffn(self.norm2(tokens))
        return (tokens, weights) if capture else tokens


class TinyVisionTransformer(nn.Module):
    def __init__(self, distilled=False):
        super().__init__()
        self.distilled = distilled
        self.patch = nn.Conv2d(1, 32, 2, stride=2)
        self.cls = nn.Parameter(torch.zeros(1, 1, 32))
        self.cls_position = nn.Parameter(torch.zeros(1, 1, 32))
        self.patch_position = nn.Parameter(torch.randn(1, 16, 32) * .02)
        self.blocks = nn.ModuleList([AttentionBlock(), AttentionBlock()])
        self.norm = nn.LayerNorm(32)
        self.head = nn.Linear(32, 10)
        if distilled:
            self.dist = nn.Parameter(torch.zeros(1, 1, 32))
            self.dist_position = nn.Parameter(torch.zeros(1, 1, 32))
            self.dist_head = nn.Linear(32, 10)

    def encode(self, images, patch_order=None, move_positions=False, capture=False):
        patches = self.patch(images).flatten(2).transpose(1, 2)
        position = self.patch_position
        if patch_order is not None:
            patches = patches[:, patch_order]
            if move_positions:
                position = position[:, patch_order]
        patches = patches + position
        prefixes = [(self.cls + self.cls_position).expand(len(images), -1, -1)]
        if self.distilled:
            prefixes.append((self.dist + self.dist_position).expand(len(images), -1, -1))
        tokens = torch.cat(prefixes + [patches], dim=1)
        attention = []
        for block in self.blocks:
            if capture:
                tokens, weights = block(tokens, True)
                attention.append(weights)
            else:
                tokens = block(tokens)
        return self.norm(tokens), attention

    def forward(self, images, return_heads=False):
        tokens, _ = self.encode(images)
        logits = self.head(tokens[:, 0])
        if self.distilled:
            dist_logits = self.dist_head(tokens[:, 1])
            return (logits, dist_logits) if return_heads else (logits + dist_logits) / 2
        return logits

    def features(self, images):
        return self.encode(images)[0][:, 0]


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 12, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(12, 24, 3, padding=1), nn.ReLU(), nn.Flatten(),
            nn.Linear(384, 32), nn.ReLU(),
        )
        self.head = nn.Linear(32, 10)

    def features(self, images):
        return self.encoder(images)

    def forward(self, images):
        return self.head(self.features(images))


def load_data():
    original_train = np.loadtxt(ROOT / 'optdigits.tra', delimiter=',', dtype=int)
    original_test = np.loadtxt(ROOT / 'optdigits.tes', delimiter=',', dtype=int)
    vectors_train = set(map(tuple, original_train[:, :-1]))
    vectors_test = set(map(tuple, original_test[:, :-1]))
    assert len(vectors_train) == 3823 and len(vectors_test) == 1797
    assert not vectors_train.intersection(vectors_test)
    generator = np.random.default_rng(173)
    training, validation = [], []
    for digit in range(10):
        indices = generator.permutation(np.flatnonzero(original_train[:, -1] == digit))
        training.extend(indices[:120].tolist())
        validation.extend(indices[120:150].tolist())
    datasets = {}
    for name, rows, ids, prefix in [
        ('train', original_train[training], training, 'tra'),
        ('validation', original_train[validation], validation, 'tra'),
        ('test', original_test, range(len(original_test)), 'tes'),
    ]:
        images = torch.tensor(rows[:, :-1].reshape(-1, 1, 8, 8), dtype=torch.float32) / 16
        labels = torch.tensor(rows[:, -1], dtype=torch.long)
        datasets[name] = (images, labels)
    identifiers = {'train': [i + 1 for i in training], 'validation': [i + 1 for i in validation],
                   'test': list(range(1, 1798))}
    return datasets, identifiers


@torch.no_grad()
def evaluate(model, images, labels):
    model.eval()
    logits = torch.cat([model(chunk) for chunk in images.split(128)])
    return {'cross_entropy': F.cross_entropy(logits, labels).item(),
            'correct': int((logits.argmax(-1) == labels).sum()), 'count': len(labels),
            'predictions': logits.argmax(-1).tolist()}


def train(model, datasets, teacher_targets=None):
    images, labels = datasets['train']
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
    generator = torch.Generator().manual_seed(301)
    best_loss = float('inf')
    best_state = None
    trace = []
    best_step = 0
    for step in range(601):
        if step % 50 == 0:
            metric = evaluate(model, *datasets['validation'])
            trace.append({'step': step, 'cross_entropy': metric['cross_entropy'],
                          'correct': metric['correct']})
            if metric['cross_entropy'] < best_loss:
                best_loss = metric['cross_entropy']
                best_state = copy.deepcopy(model.state_dict())
                best_step = step
        if step == 600:
            break
        model.train()
        batch = torch.randint(len(images), (128,), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        if teacher_targets is None:
            loss = F.cross_entropy(model(images[batch]), labels[batch])
        else:
            cls_logits, dist_logits = model(images[batch], return_heads=True)
            loss = .5 * F.cross_entropy(cls_logits, labels[batch]) + .5 * F.cross_entropy(
                dist_logits, teacher_targets[batch])
        loss.backward()
        optimizer.step()
    model.load_state_dict(best_state)
    model.eval()
    return {'chosen_step': best_step, 'trace': trace,
            'parameters': sum(parameter.numel() for parameter in model.parameters()),
            'metrics': {name: evaluate(model, *data) for name, data in datasets.items()}}


@torch.no_grad()
def probe(model, datasets):
    features = {}
    for name, (images, labels) in datasets.items():
        features[name] = images.flatten(1).numpy() if model is None else torch.cat(
            [model.features(chunk) for chunk in images.split(128)]).numpy()
    readout = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000))
    readout.fit(features['train'], datasets['train'][1].numpy())
    return {name: {'correct': int((readout.predict(features[name]) == labels.numpy()).sum()),
                   'count': len(labels), 'cross_entropy': float(log_loss(
                       labels.numpy(), readout.predict_proba(features[name]), labels=range(10)))}
            for name, (_, labels) in datasets.items()}


def encode_state(model):
    return {key: value.detach().tolist() for key, value in model.state_dict().items()}


def reload_models():
    saved = json.loads((ROOT / 'vision-models.json').read_text())
    models = {'cnn': SmallCNN(), 'vit': TinyVisionTransformer(),
              'distilled': TinyVisionTransformer(True)}
    for name, model in models.items():
        model.load_state_dict({key: torch.tensor(value) for key, value in saved[name].items()})
        model.eval()
    return models


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    datasets, identifiers = load_data()
    torch.manual_seed(211)
    cnn = SmallCNN()
    cnn_result = train(cnn, datasets)
    print('CNN', cnn_result['chosen_step'], cnn_result['metrics']['test']['correct'], flush=True)
    with torch.no_grad():
        teacher_targets = cnn(datasets['train'][0]).argmax(-1)
    torch.manual_seed(223)
    vit = TinyVisionTransformer()
    initial = copy.deepcopy(vit.state_dict())
    distilled = TinyVisionTransformer(True)
    distilled.load_state_dict(initial, strict=False)
    distilled.dist_head.load_state_dict(vit.head.state_dict())
    assert all(torch.equal(initial[key], distilled.state_dict()[key]) for key in initial)
    vit_result = train(vit, datasets)
    print('ViT', vit_result['chosen_step'], vit_result['metrics']['test']['correct'], flush=True)
    distilled_result = train(distilled, datasets, teacher_targets)
    print('Distilled', distilled_result['chosen_step'], distilled_result['metrics']['test']['correct'], flush=True)
    models = {'cnn': cnn, 'vit': vit, 'distilled': distilled}
    results = {'versions': {'numpy': np.__version__, 'torch': torch.__version__},
               'split_ids': identifiers, 'teacher_training_disagreements': int(
                   (teacher_targets != datasets['train'][1]).sum()),
               'cnn': cnn_result, 'vit': vit_result, 'distilled': distilled_result,
               'probes': {name: probe(model, datasets) for name, model in {'raw_pixels': None, **models}.items()}}
    (ROOT / 'study-results.json').write_text(json.dumps(results, indent=2) + '\n')
    (ROOT / 'vision-models.json').write_text(json.dumps(
        {name: encode_state(model) for name, model in models.items()}, separators=(',', ':')) + '\n')
    print(json.dumps({'probes': results['probes'], 'teacher_disagreements': results['teacher_training_disagreements']}, indent=2))


if __name__ == '__main__':
    main()
