"""Reproduce the declared small positional-encoding study on CPU.

Run beside the original movement_libras.data file. Requires numpy, torch,
scikit-learn. No network, GPU, pretrained checkpoint or browser training.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np
import sklearn
from sklearn.metrics import f1_score, confusion_matrix
import torch
from torch import nn
from torch.nn import functional as F

PACKET = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def sinusoidal(positions, width, base=10000.0):
    if width % 2 or width < 2:
        raise ValueError("Use an even width of at least two.")
    frequency = base ** (-torch.arange(0, width, 2, device=positions.device,
                                     dtype=torch.float64) / width)
    angles = positions.to(torch.float64)[..., None] * frequency
    return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)


def rotate(values, positions, base=10000.0):
    width = values.shape[-1]
    if width % 2:
        raise ValueError("The rotated width must be even.")
    frequency = base ** (-torch.arange(0, width, 2, device=values.device,
                                     dtype=torch.float64) / width)
    angles = positions.to(torch.float64)[..., None] * frequency
    cosine, sine = angles.cos().to(values.dtype), angles.sin().to(values.dtype)
    even, odd = values[..., 0::2], values[..., 1::2]
    return torch.stack((even * cosine - odd * sine,
                        even * sine + odd * cosine), dim=-1).flatten(-2)


def alibi_slopes(heads):
    """Original author's schedule, including non-power-of-two head counts."""
    def power_of_two(count):
        start = 2 ** (-2 ** -(math.log2(count) - 3))
        return [start ** (index + 1) for index in range(count)]
    if heads < 1:
        raise ValueError("A head count must be positive.")
    closest = 2 ** int(math.floor(math.log2(heads)))
    if closest == heads:
        return power_of_two(heads)
    return power_of_two(closest) + power_of_two(2 * closest)[::2][:heads - closest]


class PositionClassifier(nn.Module):
    def __init__(self, mode):
        super().__init__()
        if mode not in ("none", "sinusoidal", "learned", "rope", "alibi"):
            raise ValueError("Unknown encoding mode.")
        self.mode = mode
        self.stem = nn.Linear(2, 24)
        self.norm_attention = nn.LayerNorm(24)
        self.query_key_value = nn.Linear(24, 72)
        self.attention_output = nn.Linear(24, 24)
        self.norm_feedforward = nn.LayerNorm(24)
        self.feedforward = nn.Sequential(nn.Linear(24, 48), nn.GELU(), nn.Linear(48, 24))
        self.final_norm = nn.LayerNorm(24)
        self.classifier = nn.Linear(24, 15)
        if mode == "learned":
            generator = torch.Generator().manual_seed(303)
            self.position_table = nn.Parameter(torch.randn(45, 24, generator=generator) * .02)

    def forward(self, points, positions=None, padding=None, trace=False):
        batch, length, _ = points.shape
        if positions is None:
            positions = torch.arange(length, device=points.device)
        if positions.shape != (length,):
            raise ValueError("This teaching batch uses one shared position vector.")
        state = self.stem(2 * points - 1)
        if self.mode == "sinusoidal":
            state = state + sinusoidal(positions, 24).to(state.dtype)
        elif self.mode == "learned":
            if positions.dtype not in (torch.int32, torch.int64) or positions.min() < 0 or positions.max() >= 45:
                raise ValueError("Learned position indices must be integers in [0,44].")
            state = state + self.position_table[positions]
        normalized = self.norm_attention(state)
        packed = self.query_key_value(normalized).reshape(batch, length, 3, 2, 12)
        query, key, value = packed.permute(2, 0, 3, 1, 4).unbind(0)
        raw_query, raw_key = query, key
        if self.mode == "rope":
            query, key = rotate(query, positions), rotate(key, positions)
        logits = query @ key.transpose(-2, -1) / math.sqrt(12)
        content_logits = logits.clone()
        if self.mode == "alibi":
            distance = (positions[:, None] - positions[None, :]).abs()
            slopes = torch.tensor(alibi_slopes(2), dtype=state.dtype, device=points.device)
            logits = logits - slopes[None, :, None, None] * distance
        if padding is not None:
            if padding.all(dim=1).any():
                raise ValueError("Each record needs a valid point.")
            logits = logits.masked_fill(padding[:, None, None, :], -torch.inf)
        attention = logits.softmax(-1)
        mixed = (attention @ value).transpose(1, 2).reshape(batch, length, 24)
        state = state + self.attention_output(mixed)
        state = state + self.feedforward(self.norm_feedforward(state))
        normalized = self.final_norm(state)
        if padding is None:
            pooled = normalized.mean(1)
        else:
            valid = ~padding
            pooled = (normalized * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)
        output = self.classifier(pooled)
        if trace:
            return output, {"raw_query": raw_query, "raw_key": raw_key,
                            "query": query, "key": key, "value": value,
                            "content_logits": content_logits, "attention": attention,
                            "pooled": pooled}
        return output


def load_data():
    source = PACKET / "movement_libras.data"
    raw = np.loadtxt(source, delimiter=",")
    assert raw.shape == (360, 91) and np.isfinite(raw).all()
    coordinates, labels = raw[:, :90], raw[:, 90].astype(int) - 1
    _, first, inverse, counts = np.unique(coordinates, axis=0, return_index=True,
                                        return_inverse=True, return_counts=True)
    for group in range(len(first)):
        assert len(set(labels[inverse == group])) == 1
    unique = np.sort(first)
    generator = np.random.default_rng(73)
    splits = {name: [] for name in ("train", "validation", "test")}
    for label in range(15):
        rows = generator.permutation(unique[labels[unique] == label])
        count = 2 * len(rows) // 3
        splits["train"].extend(rows[:count].tolist())
        splits["validation"].extend(rows[count:-4].tolist())
        splits["test"].extend(rows[-4:].tolist())
    assert [len(rows) for rows in splits.values()] == [220, 50, 60]
    contract = {"sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "split_rows_one_based": {name: [row + 1 for row in rows] for name, rows in splits.items()},
                "duplicate_groups_one_based": [[int(i + 1) for i in np.flatnonzero(inverse == group)]
                                                for group in range(len(first)) if counts[group] > 1]}
    return torch.tensor(coordinates.reshape(-1, 45, 2), dtype=torch.float32), torch.tensor(labels), splits, contract


def score(model, points, labels):
    model.eval()
    with torch.no_grad():
        logits = model(points)
    predicted = logits.argmax(-1)
    return {"loss": float(F.cross_entropy(logits, labels)),
            "correct": int((predicted == labels).sum()), "total": len(labels),
            "macro_f1": float(f1_score(labels.numpy(), predicted.numpy(), labels=np.arange(15),
                                       average="macro", zero_division=0))}


def real_study():
    points, labels, splits, contract = load_data()
    example_row = next(row for row in splits["test"] if int(labels[row]) == 3)
    records, models = [], {}
    initial_shared = None
    for mode in ("none", "sinusoidal", "learned", "rope", "alibi"):
        torch.manual_seed(101)
        model = PositionClassifier(mode)
        shared = {name: value.clone() for name, value in model.state_dict().items() if name != "position_table"}
        if initial_shared is None:
            initial_shared = shared
        assert all(torch.equal(value, initial_shared[name]) for name, value in shared.items())
        optimizer = torch.optim.Adam(model.parameters(), lr=.003)
        best_key, best_state, best_epoch = None, None, None
        history = []
        for epoch in range(1, 181):
            model.train()
            optimizer.zero_grad()
            loss = F.cross_entropy(model(points[splits["train"]]), labels[splits["train"]])
            loss.backward()
            optimizer.step()
            validation = score(model, points[splits["validation"]], labels[splits["validation"]])
            key = (validation["macro_f1"], -validation["loss"])
            if best_key is None or key > best_key:
                best_key, best_state, best_epoch = key, deepcopy(model.state_dict()), epoch
            history.append({"epoch": epoch, "training_loss_before_update": float(loss.detach()),
                            "validation": validation})
        model.load_state_dict(best_state)
        record = {"mode": mode, "seed": 101, "parameters": sum(p.numel() for p in model.parameters()),
                  "selected_epoch": best_epoch, "history": history,
                  **{part: score(model, points[rows], labels[rows]) for part, rows in splits.items()}}
        records.append(record)
        print(mode, best_epoch, record["test"], flush=True)
        example = points[example_row:example_row + 1]
        positions = torch.arange(45)
        permutation = torch.arange(44, -1, -1)
        with torch.no_grad():
            baseline, trace = model(example, trace=True)
            paired = model(example[:, permutation], positions=positions[permutation])
            reversed_points = model(example[:, permutation], positions=positions)
            edited = example.clone()
            edited[0, 22, 0] = 1 - edited[0, 22, 0]
            edit_output = model(edited)
            extended = torch.cat((example, torch.full((1, 5, 2), .75)), dim=1)
            extended_positions = torch.cat((positions, torch.zeros(5, dtype=torch.long)))
            padding = torch.tensor([[False] * 45 + [True] * 5])
            masked = model(extended, positions=extended_positions, padding=padding)
            unmasked = model(extended, positions=extended_positions)
            all_logits = model(points[splits["test"]])
        def output(value):
            return {"logits": value[0].tolist(), "probabilities": value.softmax(-1)[0].tolist(),
                    "max_logit_change": float((value - baseline).abs().max())}
        models[mode] = {"state_dict": {name: value.tolist() for name, value in model.state_dict().items()},
                        "source_row_one_based": example_row + 1, "true_class_one_based": 4,
                        "points": example[0].tolist(), "positions": positions.tolist(),
                        "baseline": output(baseline), "paired_permutation": output(paired),
                        "reversed_points": output(reversed_points), "point_edit": output(edit_output),
                        "masked_padding": output(masked), "unmasked_padding": output(unmasked),
                        "trace": {name: value[0].tolist() for name, value in trace.items()},
                        "confusion": confusion_matrix(labels[splits["test"]].numpy(),
                                                       all_logits.argmax(-1).numpy(), labels=np.arange(15)).tolist()}
    (PACKET / "position-models.json").write_text(json.dumps(models, indent=2), encoding="utf-8")
    return {"data": contract, "fits": records}


if __name__ == "__main__":
    results = real_study()
    results["environment"] = {"python": platform.python_version(), "numpy": np.__version__,
                              "torch": torch.__version__, "sklearn": sklearn.__version__, "threads": 1}
    (PACKET / "author-results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
