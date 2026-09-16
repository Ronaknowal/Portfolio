"""Reproduce this lesson's small CPU experiment and exact attention fixtures.

Run: python author-calculations.py
Inputs: movement_libras.data alongside this file. No network or GPU is used.
Dependencies: numpy, torch, scikit-learn. Recorded versions are in author-results.json.
The fixed protocol is documented in design.md; do not tune it against the test rows.
"""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix, f1_score
import torch
from torch import nn
from torch.nn import functional as F


PACKET = Path(__file__).resolve().parent
SEEDS = (101, 102, 103)
EPOCHS = 200
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def attention(query, key, value, allowed=None):
    """Inputs (..., query/key positions, feature); True means an allowed edge.

    Every computed query must have a legal key. The lesson handles unused padded
    query outputs separately; it never asks softmax to normalize an empty set.
    """
    scores = query @ key.transpose(-2, -1) / math.sqrt(query.shape[-1])
    if allowed is not None:
        if not allowed.any(dim=-1).all():
            raise ValueError("Every query needs at least one allowed key")
        scores = scores.masked_fill(~allowed, -torch.inf)
    weights = scores.softmax(dim=-1)
    return weights @ value, weights


class SelfAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.width = width
        self.heads = heads
        self.head_width = width // heads
        self.query = nn.Linear(width, width, bias=False)
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.output = nn.Linear(width, width, bias=False)

    def split_heads(self, tensor):
        batch, length, _ = tensor.shape
        return tensor.reshape(batch, length, self.heads, self.head_width).transpose(1, 2)

    def forward(self, inputs, allowed=None):
        query = self.split_heads(self.query(inputs))
        key = self.split_heads(self.key(inputs))
        value = self.split_heads(self.value(inputs))
        values, weights = attention(query, key, value, allowed)
        merged = values.transpose(1, 2).contiguous().reshape(inputs.shape)
        return self.output(merged), weights


class TrajectoryClassifier(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        if kind == "ordered-linear":
            self.classifier = nn.Linear(90, 15)
        else:
            self.stem = nn.Linear(2, 24)
            self.mixer = SelfAttention(24, int(kind[-1])) if kind.startswith("attention") else None
            self.classifier = nn.Linear(24, 15)

    def forward(self, points, valid=None, return_weights=False):
        if self.kind == "ordered-linear":
            return self.classifier((2 * points - 1).flatten(1))
        features = torch.tanh(self.stem(2 * points - 1))
        weights = None
        if self.mixer is not None:
            allowed = None if valid is None else valid[:, None, None, :]
            mixed, weights = self.mixer(features, allowed)
            features = features + mixed
        if valid is None:
            pooled = features.mean(dim=1)
        else:
            pooled = (features * valid[..., None]).sum(dim=1) / valid.sum(dim=1, keepdim=True)
        logits = self.classifier(pooled)
        return (logits, weights) if return_weights else logits


def load_data():
    source = PACKET / "movement_libras.data"
    raw = np.loadtxt(source, delimiter=",")
    assert raw.shape == (360, 91) and np.isfinite(raw).all()
    features, labels = raw[:, :90], raw[:, 90].astype(int) - 1
    _, first, inverse, counts = np.unique(features, axis=0, return_index=True, return_inverse=True, return_counts=True)
    for group in range(len(first)):
        assert len(set(labels[inverse == group])) == 1
    # Keep the first source occurrence of an exactly repeated trajectory.
    unique_rows = np.sort(first)
    generator = np.random.default_rng(73)
    splits = {name: [] for name in ("train", "validation", "test")}
    for label in range(15):
        ids = generator.permutation(unique_rows[labels[unique_rows] == label])
        training_count = 2 * len(ids) // 3
        validation_count = len(ids) - training_count - 4
        splits["train"].extend(ids[:training_count].tolist())
        splits["validation"].extend(ids[training_count:training_count + validation_count].tolist())
        splits["test"].extend(ids[-4:].tolist())
    assert [len(splits[name]) for name in splits] == [220, 50, 60]
    assert len(set(sum(splits.values(), []))) == 330
    repeated = [[int(index + 1) for index in np.flatnonzero(inverse == group)]
                for group in range(len(first)) if counts[group] > 1]
    contract = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_rows": 360, "unique_trajectories": 330,
        "duplicate_source_row_groups_one_based": repeated,
        "split_seed": 73,
        "source_rows_one_based": {name: [row + 1 for row in ids] for name, ids in splits.items()},
        "transformation": "2 * source_coordinate - 1; no learned preprocessing",
    }
    return torch.tensor(features.reshape(-1, 45, 2), dtype=torch.float32), torch.tensor(labels), splits, contract


def metrics(model, points, labels):
    model.eval()
    with torch.no_grad():
        logits = model(points)
        predictions = logits.argmax(dim=-1)
        return {
            "cross_entropy": float(F.cross_entropy(logits, labels)),
            "correct": int((predictions == labels).sum()),
            "total": len(labels),
            "macro_f1": float(f1_score(labels.numpy(), predictions.numpy(), labels=np.arange(15), average="macro", zero_division=0)),
        }


def fit(kind, seed, points, labels, splits):
    torch.manual_seed(seed)
    model = TrajectoryClassifier(kind)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    training = splits["train"]
    validation = splits["validation"]
    best_key, best_epoch, best_state = (-math.inf, -math.inf), None, None
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(points[training]), labels[training])
        loss.backward()
        optimizer.step()
        validation_metrics = metrics(model, points[validation], labels[validation])
        key = (validation_metrics["macro_f1"], -validation_metrics["cross_entropy"])
        if key > best_key:
            best_key, best_epoch, best_state = key, epoch, deepcopy(model.state_dict())
        if epoch == 1 or epoch % 10 == 0:
            history.append({"epoch": epoch, "training": metrics(model, points[training], labels[training]), "validation": validation_metrics})
    model.load_state_dict(best_state)
    model.eval()
    report = {
        "model": kind, "seed": seed, "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "selected_epoch": best_epoch, "history": history,
        **{name: metrics(model, points[ids], labels[ids]) for name, ids in splits.items()},
    }
    return model, report


def mathematical_fixtures():
    # Hand-built arithmetic, not learned semantic embeddings.
    query = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], dtype=torch.float64)
    key = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], dtype=torch.float64)
    value = torch.tensor([[2., 0.], [0., 2.], [1., 1.]], dtype=torch.float64)
    output, weights = attention(query, key, value)
    causal_output, causal_weights = attention(query, key, value, torch.ones(3, 3, dtype=torch.bool).tril())
    shifted = (query @ key.T / math.sqrt(2) + 1000).softmax(-1)
    same_value = torch.tensor([[3., -2.], [3., -2.]], dtype=torch.float64)
    counterexample = torch.tensor([[.9, .1], [.1, .9]], dtype=torch.float64) @ same_value
    # Analytic reverse pass for a single row, verified against autodiff.
    q = torch.tensor([1., 0.], dtype=torch.float64, requires_grad=True)
    k = key.detach().clone().requires_grad_(True)
    v = value.detach().clone().requires_grad_(True)
    a = (q @ k.T / math.sqrt(2)).softmax(-1)
    z = a @ v
    upstream = torch.tensor([1., -.5], dtype=torch.float64)
    (z @ upstream).backward()
    score_gradient = a.detach() * ((v.detach() - z.detach()) @ upstream)
    calculated_q_gradient = score_gradient @ k.detach() / math.sqrt(2)
    assert torch.allclose(q.grad, calculated_q_gradient, atol=1e-12, rtol=1e-12)
    # One decode query at absolute position 2 may use keys at positions 0,1,2.
    decode_q = query[2:3][None, None]
    decode_k, decode_v = key[None, None], value[None, None]
    decode_allowed = torch.arange(3)[None, :] <= torch.tensor([2])[:, None]
    decode_correct = F.scaled_dot_product_attention(decode_q, decode_k, decode_v, attn_mask=decode_allowed, dropout_p=0.)
    decode_upper_left = F.scaled_dot_product_attention(decode_q, decode_k, decode_v, is_causal=True, dropout_p=0.)
    assert torch.allclose(decode_correct[0, 0, 0], output[2])
    # Map precisely the same four learned maps into the public MHA module.
    torch.manual_seed(19)
    manual = SelfAttention(8, 2).double().eval()
    reference = nn.MultiheadAttention(8, 2, bias=False, dropout=0., batch_first=True).double().eval()
    with torch.no_grad():
        reference.in_proj_weight.copy_(torch.cat([manual.query.weight, manual.key.weight, manual.value.weight]))
        reference.out_proj.weight.copy_(manual.output.weight)
    inputs = torch.randn(2, 4, 8, dtype=torch.float64)
    allowed = torch.ones(4, 4, dtype=torch.bool).tril()
    manual_output, manual_weights = manual(inputs, allowed)
    reference_output, reference_weights = reference(inputs, inputs, inputs, attn_mask=~allowed, need_weights=True, average_attn_weights=False)
    assert torch.allclose(manual_output, reference_output, atol=1e-12, rtol=1e-12)
    assert torch.allclose(manual_weights, reference_weights, atol=1e-12, rtol=1e-12)
    return {
        "query": query.tolist(), "key": key.tolist(), "value": value.tolist(),
        "weights": weights.tolist(), "output": output.tolist(),
        "causal_weights": causal_weights.tolist(), "causal_output": causal_output.tolist(),
        "row_shift_max_difference": float((weights - shifted).abs().max()),
        "identical_values_different_attention_outputs": counterexample.tolist(),
        "gradient": {"upstream": upstream.tolist(), "scores": score_gradient.tolist(), "query": q.grad.tolist(),
                     "keys": k.grad.tolist(), "values": v.grad.tolist()},
        "decode_correct": decode_correct.flatten().tolist(), "decode_upper_left": decode_upper_left.flatten().tolist(),
        "manual_mha_output_max_error": float((manual_output - reference_output).abs().max().detach()),
        "copy_loss_entropy_floor_8_tokens_31_symbols": 7 / 16 * math.log(31),
        "causal_uniform_mean_entropy_17": math.lgamma(18) / 17,
    }


def main():
    points, labels, splits, contract = load_data()
    reports = []
    illustrated_model = None
    for kind in ("mean-pooling", "attention-1", "attention-2", "ordered-linear"):
        for seed in SEEDS:
            model, report = fit(kind, seed, points, labels, splits)
            reports.append(report)
            print(kind, seed, "epoch", report["selected_epoch"], "test", report["test"]["correct"], "/60", flush=True)
            if kind == "attention-2" and seed == 101:
                illustrated_model = model
    # Fixed before fitting: first test trajectory in source class 4, seed101 H2.
    row = next(index for index in splits["test"] if int(labels[index]) == 3)
    selected = points[row:row + 1]
    model = illustrated_model
    with torch.no_grad():
        logits, weights = model(selected, return_weights=True)
        reverse_logits, reverse_weights = model(selected.flip(1), return_weights=True)
        edited = selected.clone()
        edited[0, 22, 0] = 1.0 - edited[0, 22, 0]
        edited_logits = model(edited)
        padded = torch.cat([selected, torch.full((1, 5, 2), .75)], dim=1)
        valid = torch.arange(50)[None] < 45
        padded_logits = model(padded, valid=valid)
        unmasked_logits = model(padded)
        test_logits = model(points[splits["test"]])
        probes = {
            "source_row_one_based": row + 1, "source_class_one_based": int(labels[row]) + 1,
            "points": selected[0].tolist(), "probabilities": logits.softmax(-1)[0].tolist(),
            "attention_weights": weights[0].tolist(),
            "reverse_max_logit_error": float((reverse_logits - logits).abs().max()),
            "reverse_attention_equivariance_error": float((reverse_weights - weights.flip(-1).flip(-2)).abs().max()),
            "masked_padding_max_logit_error": float((padded_logits - logits).abs().max()),
            "unmasked_padding_max_logit_change": float((unmasked_logits - logits).abs().max()),
            "edit": {"frame_one_based": 23, "new_x": float(edited[0, 22, 0]), "max_logit_change": float((edited_logits - logits).abs().max()),
                     "probabilities": edited_logits.softmax(-1)[0].tolist()},
            "test_confusion_matrix": confusion_matrix(labels[splits["test"]], test_logits.argmax(-1), labels=np.arange(15)).tolist(),
        }
        assert probes["reverse_max_logit_error"] < 1e-4
        assert probes["masked_padding_max_logit_error"] < 1e-4
        assert probes["edit"]["max_logit_change"] > .001
        assert probes["unmasked_padding_max_logit_change"] > .001
    model_packet = {"model": "attention-2", "seed": 101, "width": 24, "heads": 2,
                    "state_dict": {name: tensor.tolist() for name, tensor in model.state_dict().items()}, "probes": probes}
    output = {"versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__, "sklearn": sklearn.__version__},
              "protocol": {"epochs": EPOCHS, "optimizer": "Adam", "learning_rate": .01, "weight_decay": 0., "batch": "full training set",
                           "selection": "largest validation macro F1; ties lower validation cross entropy; exact ties earlier epoch",
                           "seeds": list(SEEDS)}, "data": contract, "fits": reports, "fixtures": mathematical_fixtures()}
    (PACKET / "author-results.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (PACKET / "attention-model.json").write_text(json.dumps(model_packet, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    print("Saved author-results.json and attention-model.json", flush=True)


if __name__ == "__main__":
    main()
