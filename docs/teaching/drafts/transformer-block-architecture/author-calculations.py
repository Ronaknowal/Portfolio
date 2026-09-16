"""Bounded CPU reproductions for Transformer Block Architecture.

Run python author-calculations.py beside the two original Libras files.
Dependencies: numpy, torch, scikit-learn. No network or GPU required.
The six fits and all interventions are declared in design.md before results.
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


class TransformerBlock(nn.Module):
    def __init__(self, width=24, heads=2, hidden=48, pre_norm=True):
        super().__init__()
        self.pre_norm = pre_norm
        self.attention = nn.MultiheadAttention(width, heads, dropout=0, batch_first=True)
        self.norm_attention = nn.LayerNorm(width)
        self.norm_feedforward = nn.LayerNorm(width)
        self.feedforward = nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width))

    def forward(self, inputs, blocked=None, padding=None, return_trace=False):
        attention_inputs = self.norm_attention(inputs) if self.pre_norm else inputs
        update, weights = self.attention(attention_inputs, attention_inputs, attention_inputs,
                                         attn_mask=blocked, key_padding_mask=padding,
                                         need_weights=return_trace, average_attn_weights=False)
        residual = inputs + update
        context = residual if self.pre_norm else self.norm_attention(residual)
        feedforward_inputs = self.norm_feedforward(context) if self.pre_norm else context
        feedforward_update = self.feedforward(feedforward_inputs)
        output = context + feedforward_update
        if not self.pre_norm:
            output = self.norm_feedforward(output)
        if return_trace:
            return output, {"input": inputs, "attention_input": attention_inputs,
                            "attention_update": update, "attention_residual": residual,
                            "context": context, "feedforward_input": feedforward_inputs,
                            "feedforward_hidden": self.feedforward[1](self.feedforward[0](feedforward_inputs)),
                            "feedforward_update": feedforward_update, "output": output,
                            "attention_weights": weights}
        return output


class MovementClassifier(nn.Module):
    def __init__(self, pre_norm):
        super().__init__()
        self.stem = nn.Linear(3, 24)
        self.blocks = nn.ModuleList([TransformerBlock(pre_norm=pre_norm) for _ in range(2)])
        # Shared across the controlled comparison, including the post-norm model.
        self.final_norm = nn.LayerNorm(24)
        self.classifier = nn.Linear(24, 15)

    def forward(self, points, times=None, padding=None, return_trace=False):
        if times is None:
            times = torch.linspace(-1, 1, points.shape[1], dtype=points.dtype, device=points.device)
        times = times.expand(points.shape[0], -1)[..., None]
        state = self.stem(torch.cat((2 * points - 1, times), dim=-1))
        traces = []
        for block in self.blocks:
            if return_trace:
                state, trace = block(state, padding=padding, return_trace=True)
                traces.append(trace)
            else:
                state = block(state, padding=padding)
        normalized = self.final_norm(state)
        if padding is None:
            pooled = normalized.mean(dim=1)
        else:
            valid = ~padding
            pooled = (normalized * valid[..., None]).sum(dim=1) / valid.sum(dim=1, keepdim=True)
        logits = self.classifier(pooled)
        return (logits, traces) if return_trace else logits


def load_data():
    source = PACKET / "movement_libras.data"
    raw = np.loadtxt(source, delimiter=",")
    assert raw.shape == (360, 91) and np.isfinite(raw).all()
    coordinates, labels = raw[:, :90], raw[:, 90].astype(int) - 1
    _, first, inverse, counts = np.unique(coordinates, axis=0, return_index=True, return_inverse=True, return_counts=True)
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
                "unique_trajectories": 330,
                "split_rows_one_based": {name: [row + 1 for row in rows] for name, rows in splits.items()},
                "duplicate_groups_one_based": [[int(i + 1) for i in np.flatnonzero(inverse == group)]
                                                for group in range(len(first)) if counts[group] > 1]}
    return torch.tensor(coordinates.reshape(-1, 45, 2), dtype=torch.float32), torch.tensor(labels), splits, contract


def score(model, points, labels):
    model.eval()
    with torch.no_grad():
        logits = model(points)
    predicted = logits.argmax(dim=-1)
    return {"loss": float(F.cross_entropy(logits, labels)),
            "correct": int((predicted == labels).sum()), "total": len(labels),
            "macro_f1": float(f1_score(labels.numpy(), predicted.numpy(), labels=np.arange(15), average="macro", zero_division=0))}


def real_study():
    points, labels, splits, contract = load_data()
    selected_row = next(row for row in splits["test"] if int(labels[row]) == 3)
    records, models = [], {}
    for pre_norm in (True, False):
        name = "pre-norm" if pre_norm else "post-norm"
        for seed in (101, 102, 103):
            torch.manual_seed(seed)
            model = MovementClassifier(pre_norm)
            optimizer = torch.optim.Adam(model.parameters(), lr=.003)
            best_key, best_state, best_epoch = None, None, None
            history = []
            for epoch in range(1, 181):
                model.train()
                optimizer.zero_grad()
                loss = F.cross_entropy(model(points[splits["train"]]), labels[splits["train"]])
                loss.backward()
                optimizer.step()
                train = score(model, points[splits["train"]], labels[splits["train"]])
                validation = score(model, points[splits["validation"]], labels[splits["validation"]])
                key = (validation["macro_f1"], -validation["loss"])
                if best_key is None or key > best_key:
                    best_key, best_state, best_epoch = key, deepcopy(model.state_dict()), epoch
                history.append({"epoch": epoch, "train": train, "validation": validation})
            model.load_state_dict(best_state)
            record = {"placement": name, "seed": seed, "parameters": sum(p.numel() for p in model.parameters()),
                      "selected_epoch": best_epoch, "history": history,
                      **{part: score(model, points[rows], labels[rows]) for part, rows in splits.items()}}
            records.append(record)
            print(name, seed, best_epoch, record["test"], flush=True)
            if seed == 101:
                example = points[selected_row:selected_row + 1]
                times = torch.linspace(-1, 1, 45)
                with torch.no_grad():
                    logits, trace = model(example, return_trace=True)
                    permutation = torch.arange(44, -1, -1)
                    paired = model(example[:, permutation], times=times[permutation])
                    reverse = model(example[:, permutation], times=times)
                    edited = example.clone()
                    edited[0, 22, 0] = 1 - edited[0, 22, 0]
                    edit_logits = model(edited)
                    extended = torch.cat((example, torch.full((1, 5, 2), .75)), dim=1)
                    extended_times = torch.cat((times, torch.zeros(5)))
                    padding = torch.tensor([[False] * 45 + [True] * 5])
                    masked = model(extended, times=extended_times, padding=padding)
                    unmasked = model(extended, times=extended_times)
                    all_logits = model(points[splits["test"]])
                def output(value):
                    return {"logits": value[0].tolist(), "probabilities": value.softmax(-1)[0].tolist(),
                            "max_logit_change": float((value - logits).abs().max())}
                models[name] = {"seed": seed, "selected_epoch": best_epoch,
                               "state_dict": {key: value.tolist() for key, value in model.state_dict().items()},
                               "source_row_one_based": selected_row + 1, "source_class": int(labels[selected_row]) + 1,
                               "points": example[0].tolist(), "times": times.tolist(), "original": output(logits),
                               "paired_permutation": output(paired), "reverse_coordinates_fixed_time": output(reverse),
                               "frame23_x_reflection": output(edit_logits), "masked_padding": output(masked),
                               "unmasked_padding": output(unmasked),
                               "trace": [{key: value[0].tolist() for key, value in layer.items()} for layer in trace],
                               "confusion": confusion_matrix(labels[splits["test"]].numpy(), all_logits.argmax(-1).numpy(), labels=np.arange(15)).tolist()}
    return contract, records, models


def exact_fixtures():
    # Analytic LN/RMS invariances, including the all-ones output-probe null.
    inputs = torch.tensor([1., 2., 5., 8.], dtype=torch.float64, requires_grad=True)
    normalized = F.layer_norm(inputs, (4,), eps=1e-5)
    sum_gradient = torch.autograd.grad(normalized.sum(), inputs, retain_graph=True)[0]
    probe = torch.tensor([1., -1., 0., 0.], dtype=torch.float64)
    probe_gradient = torch.autograd.grad(normalized @ probe, inputs)[0]
    rms = inputs.detach() / torch.sqrt(inputs.detach().square().mean() + 1e-5)
    shift = inputs.detach() + 5
    shift_rms = shift / torch.sqrt(shift.square().mean() + 1e-5)
    normalization = {"x": inputs.detach().tolist(), "ln": normalized.detach().tolist(), "rms": rms.tolist(),
                     "shifted_ln": F.layer_norm(shift, (4,), eps=1e-5).tolist(), "shifted_rms": shift_rms.tolist(),
                     "sum_probe_gradient": sum_gradient.tolist(), "contrast_probe_gradient": probe_gradient.tolist(),
                     "ln_l2": float(normalized.detach().norm()), "ln_rms": float(normalized.detach().square().mean().sqrt())}
    # Compute the same whole block with a reference API; dropout is zero everywhere.
    torch.manual_seed(31)
    own = TransformerBlock(width=8, heads=2, hidden=16).double()
    source = torch.randn(2, 4, 8, dtype=torch.float64)
    blocked = torch.ones(4, 4, dtype=torch.bool).triu(1)
    reference_errors = {}
    for pre_norm in (True, False):
        own.pre_norm = pre_norm
        reference = nn.TransformerEncoderLayer(8, 2, 16, dropout=0, activation="gelu", batch_first=True, norm_first=pre_norm).double()
        reference.self_attn.load_state_dict(own.attention.state_dict())
        reference.norm1.load_state_dict(own.norm_attention.state_dict())
        reference.norm2.load_state_dict(own.norm_feedforward.state_dict())
        reference.linear1.load_state_dict(own.feedforward[0].state_dict())
        reference.linear2.load_state_dict(own.feedforward[2].state_dict())
        difference = float((own(source, blocked) - reference(source, src_mask=blocked)).detach().abs().max())
        assert difference < 1e-12
        reference_errors[str(pre_norm)] = difference
    # Small initialized depth diagnostics, with a declared normalized random probe.
    gradient_records = []
    for depth in (1, 4, 12):
        for pre_norm in (True, False):
            torch.manual_seed(53)
            blocks = nn.ModuleList([TransformerBlock(8, 2, 16, pre_norm).double() for _ in range(depth)])
            generator = torch.Generator().manual_seed(97)
            x = torch.randn(1, 3, 8, dtype=torch.float64, generator=generator, requires_grad=True)
            probe = torch.randn(1, 3, 8, dtype=torch.float64, generator=generator)
            probe = probe / probe.norm()
            states = [x]
            for block in blocks:
                states.append(block(states[-1]))
            # Final affine-free LN is shared; loss probes a nonconstant direction.
            final = F.layer_norm(states[-1], (8,), eps=1e-5)
            gradients = torch.autograd.grad((final * probe).sum(), states, retain_graph=True)
            null = torch.autograd.grad(final.sum(), x)[0]
            gradient_records.append({"depth": depth, "pre_norm": pre_norm,
                                     "state_gradient_l2": [float(g.norm()) for g in gradients],
                                     "state_rms": [float(s.detach().square().mean().sqrt()) for s in states],
                                     "sum_probe_input_gradient_l2": float(null.norm())})
    # Hand trace uses genuine one-head attention on 2 positions / 4 features.
    x = torch.tensor([[1., 2., 5., 8.], [3., 0., 2., 1.]], dtype=torch.float64)
    up = torch.tensor([[1., 0., 1.], [0., 1., 1.], [-1., 0., 0.], [0., -1., 0.]], dtype=torch.float64)
    down = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., .5, -.5]], dtype=torch.float64)
    def trace(value, pre_norm, branch_scale=1):
        attention_input = F.layer_norm(value, (4,), eps=1e-5) if pre_norm else value
        weights = (attention_input @ attention_input.T / 2).softmax(-1)
        update = branch_scale * (weights @ attention_input) / 4
        context = value + update
        if not pre_norm:
            context = F.layer_norm(context, (4,), eps=1e-5)
        ffn_input = F.layer_norm(context, (4,), eps=1e-5) if pre_norm else context
        hidden = torch.relu(ffn_input @ up)
        ffn_update = branch_scale * hidden @ down
        output = context + ffn_update
        if not pre_norm:
            output = F.layer_norm(output, (4,), eps=1e-5)
        return {"attention_input": attention_input.tolist(), "weights": weights.tolist(), "attention_update": update.tolist(),
                "context": context.tolist(), "ffn_input": ffn_input.tolist(), "hidden": hidden.tolist(),
                "ffn_update": ffn_update.tolist(), "output": output.tolist()}
    changed = x.clone(); changed[1, 0] += 2
    fixture = {"x": x.tolist(), "up": up.tolist(), "down": down.tolist(),
               "pre": trace(x, True), "post": trace(x, False), "edited_pre": trace(changed, True),
               "zero_pre": trace(x, True, 0), "zero_post": trace(x, False, 0)}
    return {"normalization": normalization, "reference_max_abs_errors": reference_errors,
            "gradient_depth": gradient_records, "trace": fixture,
            "copy_loss_floor": 8 / 18 * math.log(10)}


if __name__ == "__main__":
    fixture = exact_fixtures()
    contract, records, models = real_study()
    results = {"versions": {"python": platform.python_version(), "numpy": np.__version__,
                             "torch": torch.__version__, "sklearn": sklearn.__version__},
               "protocol": {"seeds": [101, 102, 103], "epochs": 180, "optimizer": "Adam", "learning_rate": .003,
                            "full_training_batch": 220, "weight_decay": 0, "dropout": 0,
                            "selection": "validation macro F1; lower validation CE; earliest exact tie",
                            "position": "append fixed time -1 to 1 before shared 3-to-24 stem",
                            "final_norm": "LayerNorm(24) in BOTH models for controlled placement comparison"},
               "data": contract, "fits": records, "fixtures": fixture}
    (PACKET / "author-results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    (PACKET / "block-models.json").write_text(json.dumps(models, separators=(",", ":")) + "\n", encoding="utf-8")
