"""Offline graph-structure study. This is a small teaching model, not Graphormer/GPS."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def structural_inputs(adjacency):
    n = len(adjacency)
    degree = adjacency.sum(-1)
    transition = adjacency / degree[:, None].clamp_min(1)
    power = torch.eye(n)
    returns = []
    for _ in range(4):
        power = power @ transition
        returns.append(power.diagonal())
    distance = torch.full((n, n), float('inf'))
    distance.fill_diagonal_(0)
    distance[adjacency > 0] = 1
    for middle in range(n):
        distance = torch.minimum(distance, distance[:, middle:middle+1] + distance[middle:middle+1, :])
    distance = torch.where(distance.isfinite(), distance, n).long()
    closed = adjacency + torch.eye(n)
    inv = closed.sum(-1).rsqrt()
    symmetric = inv[:, None] * closed * inv[None, :]
    return torch.stack(returns, -1), distance, symmetric


class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(16), nn.LayerNorm(16)
        self.qkv = nn.Linear(16, 48)
        self.out = nn.Linear(16, 16)
        self.ff = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 16))

    def forward(self, hidden, bias):
        n = len(hidden)
        q, k, value = self.qkv(self.norm1(hidden)).reshape(n, 3, 2, 8).unbind(1)
        scores = torch.einsum('ihd,jhd->hij', q, k) / np.sqrt(8)
        weights = (scores + bias).softmax(-1)
        mixed = torch.einsum('hij,jhd->ihd', weights, value).reshape(n, 16)
        hidden = hidden + self.out(mixed)
        return hidden + self.ff(self.norm2(hidden)), weights


class Model(nn.Module):
    def __init__(self, kind, n):
        super().__init__()
        self.kind = kind
        self.embed = nn.Linear(7 if kind == 'walk' else 3, 16)
        self.head = nn.Linear(16, 2)
        if kind != 'gcn':
            self.blocks = nn.ModuleList([AttentionBlock(), AttentionBlock()])
        if kind == 'distance':
            self.bias = nn.Embedding(n + 1, 2)
            nn.init.zeros_(self.bias.weight)

    def forward(self, features, structure):
        returns, distance, symmetric = structure
        x = torch.cat([features, returns], -1) if self.kind == 'walk' else features
        if self.kind == 'gcn':
            hidden = (symmetric @ (x @ self.embed.weight.T) + self.embed.bias).relu()
            return symmetric @ (hidden @ self.head.weight.T) + self.head.bias, None
        hidden = self.embed(x)
        bias = self.bias(distance).permute(2, 0, 1) if self.kind == 'distance' else 0
        for block in self.blocks:
            hidden, weights = block(hidden, bias)
        return self.head(hidden), weights


def run():
    raw = json.loads((HERE / 'karate-club.json').read_text())
    n = len(raw['nodes'])
    adjacency = torch.zeros(n, n)
    for left, right, _ in raw['edges']:
        adjacency[left, right] = adjacency[right, left] = 1
    features = torch.tensor([[1., row['degree'] / 33., row['clustering']] for row in raw['nodes']])
    labels = torch.tensor([row['club'] for row in raw['nodes']])
    rng = np.random.default_rng(133)
    roles = {key: [] for key in ['fit', 'development', 'assessment']}
    for label in range(2):
        ids = rng.permutation(np.flatnonzero(labels.numpy() == label))
        for key, part in zip(roles, [ids[:5], ids[5:8], ids[8:]]):
            roles[key].extend(int(i) for i in part)
    roles = {key: sorted(value) for key, value in roles.items()}
    assert len(set(sum(roles.values(), []))) == n
    structure = structural_inputs(adjacency)
    permutation = torch.tensor(np.random.default_rng(8).permutation(n))
    permuted_structure = structural_inputs(adjacency[permutation][:, permutation])
    edited = adjacency.clone(); edited[0, 1] = edited[1, 0] = 0
    edited_structure = structural_inputs(edited)
    records = []
    for kind in ['gcn', 'set', 'distance', 'walk']:
        for seed in [11, 29, 47]:
            torch.manual_seed(seed)
            model = Model(kind, n)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
            history = []
            for epoch in range(1, 301):
                optimizer.zero_grad()
                logits, _ = model(features, structure)
                loss = nn.functional.cross_entropy(logits[roles['fit']], labels[roles['fit']])
                loss.backward(); optimizer.step()
                if epoch in [1, 10, 50, 100, 300]:
                    history.append({'epoch': epoch, 'pre_update_fit_loss': float(loss.detach())})
            with torch.no_grad():
                logits, weights = model(features, structure)
                permuted, _ = model(features[permutation], permuted_structure)
                error = float((permuted - logits[permutation]).abs().max())
                assert error < 1e-4
                changed, _ = model(features, edited_structure)
                probability, altered = logits.softmax(-1), changed.softmax(-1)
                record = {'kind': kind, 'seed': seed, 'parameters': sum(p.numel() for p in model.parameters()),
                          'correct': {key: int((logits[ids].argmax(-1) == labels[ids]).sum()) for key, ids in roles.items()},
                          'probabilities': probability.tolist(), 'edge_removed_probabilities': altered.tolist(),
                          'permutation_logit_error': error, 'history': history}
                if seed == 11:
                    record['state_dict'] = {key: value.tolist() for key, value in model.state_dict().items()}
                    record['last_attention'] = None if weights is None else weights.tolist()
                records.append(record)
    result = {'protocol': {'epochs': 300, 'learning_rate': .003, 'weight_decay': .01,
                          'seeds': [11, 29, 47], 'edge_edit': [0, 1], 'structural_features_fixed_on_edit': True,
                          'torch': torch.__version__, 'numpy': np.__version__}, 'roles': roles,
              'features': features.tolist(), 'walk_returns': structure[0].tolist(), 'distances': structure[1].tolist(),
              'records': records}
    (HERE / 'calculated-inputs.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps([{key: row[key] for key in ['kind', 'seed', 'parameters', 'correct', 'permutation_logit_error']} for row in records], indent=2))


if __name__ == '__main__':
    run()
