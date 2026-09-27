"""Reproduce the small query-conditioned digits study and exact attention fixtures.

Run beside digits-400.csv with Python, NumPy and CPU PyTorch. No downloads.
This trains small classifiers from scratch, not pretrained vision-language models.
"""
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def attention(query, key, value, allowed=None):
    scores = query @ key.T / math.sqrt(query.shape[-1])
    if allowed is not None:
        if not np.all(allowed.any(axis=-1)):
            raise ValueError("Every query must have at least one allowed memory slot")
        scores = np.where(allowed, scores, -np.inf)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ value, weights


class QuestionClassifier(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.query = nn.Embedding(2, 24)
        if mode == "flat":
            self.flat = nn.Sequential(nn.Linear(88, 48), nn.ReLU(), nn.Linear(48, 12))
        else:
            self.patch = nn.Linear(4, 24)
            self.position = nn.Parameter(torch.randn(1, 16, 24) * .02)
            self.read = nn.MultiheadAttention(24, 3, dropout=0, batch_first=True)
            self.feed = nn.Sequential(nn.LayerNorm(24), nn.Linear(24, 48), nn.GELU(), nn.Linear(48, 24))
            self.head = nn.Linear(24, 12)
            if mode == "gated":
                self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, images, questions, capture=False):
        query = self.query(questions)
        if self.mode == "flat":
            return self.flat(torch.cat((images.flatten(1), query), dim=-1)), None
        patches = images.reshape(-1, 4, 2, 4, 2).permute(0, 1, 3, 2, 4).reshape(-1, 16, 4)
        memory = self.patch(patches) + self.position
        query = query[:, None, :]
        update, weights = self.read(query, memory, memory, need_weights=capture, average_attn_weights=False)
        if self.mode == "gated":
            update = self.gate.tanh() * update
        hidden = query + update
        hidden = hidden + self.feed(hidden)
        return self.head(hidden[:, 0]), weights


def main():
    with (HERE / "digits-400.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    pixels = np.array([[int(row[f"pixel_{j}"]) for j in range(64)] for row in rows])
    labels = np.array([int(row["digit"]) for row in rows])
    source_ids = [int(row["source_id"]) for row in rows]
    # Preserve image groups when deriving two related questions per image.
    rng = np.random.default_rng(912)
    train, development, assessment = [], [], []
    for digit in range(10):
        order = rng.permutation(np.flatnonzero(labels == digit))
        train.extend(order[:24]); development.extend(order[24:32]); assessment.extend(order[32:])
    split = {"fit": sorted(train), "development": sorted(development), "assessment": sorted(assessment)}
    signatures = {}
    for i, row in enumerate(pixels):
        signatures.setdefault(tuple(row.tolist()), []).append(source_ids[i])
    duplicates = [ids for ids in signatures.values() if len(ids) > 1]
    assert len(rows) == 400 and len(set(source_ids)) == 400 and not duplicates
    images = torch.tensor(np.repeat(pixels.reshape(-1, 8, 8) / 16., 2, axis=0), dtype=torch.float32)
    questions = torch.tensor(np.tile([0, 1], 400))
    targets = torch.tensor(np.column_stack((labels, 10 + labels % 2)).reshape(-1))
    indices = {name: torch.tensor(np.array(ids)[:, None] * 2 + [0, 1]).flatten() for name, ids in split.items()}
    measured = []
    for mode in ("flat", "cross", "gated"):
        for seed in (11, 29, 47):
            torch.manual_seed(seed)
            model = QuestionClassifier(mode)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
            history = []
            for epoch in range(160):
                model.train()
                for batch in indices["fit"][torch.randperm(len(indices["fit"]))].split(64):
                    logits, _ = model(images[batch], questions[batch])
                    loss = nn.functional.cross_entropy(logits, targets[batch])
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
                if epoch in (0, 9, 39, 79, 159):
                    model.eval()
                    with torch.no_grad():
                        fit_logits, _ = model(images[indices["fit"]], questions[indices["fit"]])
                        dev_logits, _ = model(images[indices["development"]], questions[indices["development"]])
                        history.append({"epoch": epoch + 1, "fit_ce": nn.functional.cross_entropy(fit_logits, targets[indices["fit"]]).item(), "development_ce": nn.functional.cross_entropy(dev_logits, targets[indices["development"]]).item()})
            model.eval()
            record = {"mode": mode, "seed": seed, "parameters": sum(p.numel() for p in model.parameters()), "history": history}
            with torch.no_grad():
                # Assessment is displayed evidence, so it is not a future untouched holdout.
                for role in ("development", "assessment"):
                    ids = indices[role]
                    logits, _ = model(images[ids], questions[ids])
                    correct = logits.argmax(-1) == targets[ids]
                    record[role] = {"correct": int(correct.sum()), "total": len(ids), "digit_correct": int(correct[::2].sum()), "parity_correct": int(correct[1::2].sum())}
                ids = indices["assessment"]
                grouped_images = images[ids].reshape(-1, 2, 8, 8)
                cyclic_images = grouped_images.roll(1, 0).reshape(-1, 8, 8)
                logits, _ = model(cyclic_images, questions[ids])
                record["cyclic_image_correct"] = int((logits.argmax(-1) == targets[ids]).sum())
                # Sorted-by-label rows make a one-place roll a weak mismatch test.
                permutation = np.random.default_rng(5129).permutation(len(grouped_images))
                wrong_images = grouped_images[permutation].reshape(-1, 8, 8)
                logits, _ = model(wrong_images, questions[ids])
                record["mismatched_image_correct"] = int((logits.argmax(-1) == targets[ids]).sum())
                if mode != "flat":
                    examples = indices["assessment"][:8]
                    logits, weights = model(images[examples], questions[examples], capture=True)
                    record["examples"] = [{"source_id": source_ids[int(i) // 2], "question": int(questions[i]), "target": int(targets[i]), "prediction": int(logits[j].argmax()), "head_weights": weights[j, :, 0].tolist()} for j, i in enumerate(examples)]
                if mode == "gated":
                    record["final_gate"] = float(model.gate.tanh())
            measured.append(record)
    q = np.array([[1., 0.], [0., 1.]]) * math.sqrt(2)
    k = np.array([[math.log(2), 0.], [0., math.log(2)], [0., 0.]])
    v = np.array([[2., 0.], [0., 4.], [2., 2.]])
    output, weights = attention(q, k, v)
    allowed = np.array([[True, True, False], [True, True, True]])
    masked, masked_weights = attention(q, k, v, allowed)
    changed = v.copy(); changed[2] = [100, -100]
    changed_output, _ = attention(q, k, changed, allowed)
    permutation = [2, 0, 1]
    permuted, _ = attention(q, k[permutation], v[permutation])
    assert np.allclose(output, [[1.5, 1.5], [1., 2.5]])
    assert np.allclose(masked[0], changed_output[0]) and np.allclose(output, permuted)
    assessment_labels = labels[split["assessment"]]
    shuffled_labels = assessment_labels[np.random.default_rng(5129).permutation(80)]
    perturbations = {"shuffle_source_ids": [source_ids[split["assessment"][i]] for i in np.random.default_rng(5129).permutation(80)], "shuffle_digit_matches": int((assessment_labels == shuffled_labels).sum()), "shuffle_parity_matches": int((assessment_labels % 2 == shuffled_labels % 2).sum()), "cyclic_digit_matches": int((assessment_labels == np.roll(assessment_labels, 1)).sum()), "cyclic_parity_matches": int((assessment_labels % 2 == np.roll(assessment_labels, 1) % 2).sum())}
    result = {"evidence": "Exact fixtures plus measured CPU classifier study; not a VLM benchmark", "versions": {"torch": torch.__version__, "numpy": np.__version__}, "splits": {role: [source_ids[i] for i in ids] for role, ids in split.items()}, "duplicates": duplicates, "perturbations": perturbations, "question_only_correct": {"development": 48, "assessment": 48, "total_each": 160}, "measurements": measured, "exact": {"query": q.tolist(), "key": k.tolist(), "value": v.tolist(), "weights": weights.tolist(), "output": output.tolist(), "masked_output": masked.tolist(), "masked_weights": masked_weights.tolist(), "changed_masked_output": changed_output.tolist()}}
    (HERE / "calculated-inputs.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{"mode": r["mode"], "seed": r["seed"], "assessment": r["assessment"], "mismatch": r["mismatched_image_correct"]} for r in measured], indent=2))


if __name__ == "__main__":
    main()
