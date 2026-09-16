"""Small, fully specified sequence experiment; not a pretrained HyenaDNA model."""
from pathlib import Path
import copy
import hashlib
import json
import re
from collections import Counter, defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).parent
ALPHABET = "ACGTDNRS"
CLASSES = ["EI", "IE", "N"]


def load_data():
    rows = [tuple(field.strip() for field in line.split(","))
            for line in (ROOT / "splice.data").read_text().splitlines()]
    by_sequence = defaultdict(list)
    for index, row in enumerate(rows):
        by_sequence[row[2]].append(index)
    conflict = {i for group in by_sequence.values()
                if len({rows[i][0] for i in group}) > 1 for i in group}
    retained = [group[0] for group in by_sequence.values() if group[0] not in conflict]
    # Union source-record prefixes linked by an identical sequence, before splitting.
    parent = {}
    def find(value):
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value
    prefix = [re.split(r"-(?:DONOR|ACCEPTOR|NEG)-", row[1])[0] for row in rows]
    for group in by_sequence.values():
        for index in group:
            parent[find(prefix[index])] = find(prefix[group[0]])
    groups = np.array([find(prefix[i]) for i in retained])
    tokens = np.array([[ALPHABET.index(base) for base in rows[i][2]] for i in retained])
    labels = np.array([CLASSES.index(rows[i][0]) for i in retained])
    fit, rest = next(GroupShuffleSplit(n_splits=1, test_size=.30, random_state=1991)
                     .split(tokens, labels, groups))
    val_local, test_local = next(GroupShuffleSplit(n_splits=1, test_size=.5, random_state=1992)
                                 .split(tokens[rest], labels[rest], groups[rest]))
    roles = {"fit": fit, "validation": rest[val_local], "assessment": rest[test_local]}
    record = {
        "raw_rows": len(rows), "retained_rows": len(retained),
        "conflicting_source_ids": [i + 1 for i in sorted(conflict)],
        "duplicate_groups": [[i + 1 for i in group] for group in by_sequence.values() if len(group) > 1],
        "source_ids": [i + 1 for i in retained], "groups": groups.tolist(),
        "class_counts": dict(Counter(rows[i][0] for i in retained)),
        "role_source_ids": {name: [retained[i] + 1 for i in indices] for name, indices in roles.items()},
        "role_class_counts": {name: np.bincount(labels[indices], minlength=3).tolist() for name, indices in roles.items()},
        "sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ["splice.data", "splice.names"]},
    }
    return torch.tensor(tokens), torch.tensor(labels), roles, record


def causal_convolution(values, kernel):
    """values B,L,D; kernel L,D; fixed lag convention h[0] multiplies current input."""
    length = values.shape[1]
    size = 1 << (2 * length - 2).bit_length()
    spectrum = torch.fft.rfft(values, n=size, dim=1)
    filter_spectrum = torch.fft.rfft(kernel, n=size, dim=0)
    return torch.fft.irfft(spectrum * filter_spectrum[None], n=size, dim=1)[:, :length]


class ImplicitFilter(nn.Module):
    def __init__(self, width=16, reference_length=60):
        super().__init__()
        time = torch.arange(reference_length, dtype=torch.float32) / (reference_length - 1)
        angle = 2 * torch.pi * time[:, None] * torch.tensor([1., 2., 4., 8.])
        self.register_buffer("positions", torch.cat([time[:, None], angle.sin(), angle.cos()], dim=1))
        self.register_buffer("time", time)
        self.first = nn.Linear(9, 32)
        self.last = nn.Linear(32, width)
        self.decay = nn.Parameter(torch.linspace(-1., 2., width))

    def forward(self, length):
        shaped = self.last(torch.sin(self.first(self.positions[:length])))
        return shaped * torch.exp(-self.time[:length, None] * F.softplus(self.decay))


class SequenceBlock(nn.Module):
    def __init__(self, gated=True, width=16):
        super().__init__()
        self.gated = gated
        self.norm = nn.LayerNorm(width)
        self.project = nn.Linear(width, 3 * width)
        self.short = nn.Conv1d(3 * width, 3 * width, 3, groups=3 * width, padding=2)
        self.filter = ImplicitFilter(width)
        self.skip = nn.Parameter(torch.ones(width))
        self.output = nn.Linear(width, width)
        self.feed_norm = nn.LayerNorm(width)
        self.feed = nn.Sequential(nn.Linear(width, 32), nn.GELU(), nn.Linear(32, width))

    def forward(self, hidden, kernel_limit=None, gates_off=False):
        length = hidden.shape[1]
        projected = self.project(self.norm(hidden)).transpose(1, 2)
        q, k, value = self.short(projected)[:, :, :length].transpose(1, 2).chunk(3, dim=-1)
        if not self.gated or gates_off:
            q, k = torch.ones_like(q), torch.ones_like(k)
        kernel = self.filter(length)
        if kernel_limit is not None:
            kernel = kernel * (torch.arange(length)[:, None] < kernel_limit)
        transmitted = k * value
        mixed = q * (causal_convolution(transmitted, kernel) + self.skip * transmitted)
        hidden = hidden + self.output(mixed)
        return hidden + self.feed(self.feed_norm(hidden))


class SpliceReader(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        if kind == "linear":
            self.head = nn.Linear(60 * len(ALPHABET), 3)
        else:
            self.embedding = nn.Embedding(len(ALPHABET), 16)
            self.blocks = nn.ModuleList([SequenceBlock(kind == "gated") for _ in range(2)])
            self.head = nn.Linear(16, 3)

    def forward(self, tokens, kernel_limit=None, gates_off=False, return_hidden=False):
        if self.kind == "linear":
            return self.head(F.one_hot(tokens, len(ALPHABET)).float().flatten(1))
        hidden = self.embedding(tokens)
        for block in self.blocks:
            hidden = block(hidden, kernel_limit, gates_off)
        return hidden if return_hidden else self.head(hidden[:, -1])


def metrics(logits, target):
    predicted = logits.argmax(-1)
    confusion = torch.bincount(3 * target + predicted, minlength=9).reshape(3, 3)
    return {"cross_entropy": F.cross_entropy(logits, target).item(),
            "errors": int((predicted != target).sum()), "count": len(target),
            "confusion": confusion.tolist()}


def main():
    torch.set_num_threads(2)
    tokens, labels, roles, record = load_data()
    report = {"data": record, "protocol": {"epochs": 80, "batch_size": 128,
              "learning_rate": .003, "optimizer": "Adam", "gradient_clip": 1.,
              "checkpoint": "minimum validation cross entropy, earliest exact tie"}, "fits": []}
    arrays = {}
    for kind, seed in [("linear", 29), ("gated", 29), ("ungated", 29), ("gated", 71)]:
        torch.manual_seed(seed)
        model = SpliceReader(kind)
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        generator = torch.Generator().manual_seed(seed + 1000)
        best_loss, best_state, best_epoch = float("inf"), None, None
        history = []
        fit_ids = torch.tensor(roles["fit"])
        for epoch in range(1, 81):
            model.train()
            for selection in torch.randperm(len(fit_ids), generator=generator).split(128):
                ids = fit_ids[selection]
                optimizer.zero_grad()
                loss = F.cross_entropy(model(tokens[ids]), labels[ids])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                validation = metrics(model(tokens[roles["validation"]]), labels[roles["validation"]])
            history.append([epoch, validation["cross_entropy"], validation["errors"]])
            if validation["cross_entropy"] < best_loss:
                best_loss, best_state, best_epoch = validation["cross_entropy"], copy.deepcopy(model.state_dict()), epoch
        model.load_state_dict(best_state)
        model.eval()
        key = f"{kind}_{seed}"
        result = {"kind": kind, "seed": seed, "selected_epoch": best_epoch,
                  "parameters": sum(p.numel() for p in model.parameters()), "history": history}
        with torch.no_grad():
            for role in ["fit", "validation", "assessment"]:
                ids = roles[role]
                logits = model(tokens[ids])
                result[role] = metrics(logits, labels[ids])
                arrays[f"{key}__{role}_logits"] = logits.numpy()
            if kind == "gated":
                for name, options in [("lag_0_to_4", {"kernel_limit": 5}), ("gates_off", {"gates_off": True})]:
                    result[name] = metrics(model(tokens[roles["assessment"]], **options), labels[roles["assessment"]])
            for name, tensor in best_state.items():
                arrays[f"{key}__state__{name}"] = tensor.numpy()
        report["fits"].append(result)
        print(key, best_epoch, result["validation"], result["assessment"], flush=True)
    np.savez_compressed(ROOT / "splice-fits.npz", **arrays)
    (ROOT / "splice-results.json").write_text(json.dumps(report, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
