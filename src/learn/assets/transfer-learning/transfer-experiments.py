"""Offline source training, target adaptation and one selected target test.

Run beside digits-400.csv using Python 3.12, torch 2.14 and NumPy 2.3.5.
The default writes calculated-inputs.json. No network downloads are needed.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

torch.set_num_threads(1)
METHODS = ("scratch", "probe", "full", "discriminative", "lora2", "adapter4")


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.lower = nn.Linear(64, 32)
        self.upper = nn.Linear(32, 16)

    def forward(self, pixels):
        return torch.tanh(self.upper(torch.tanh(self.lower(pixels))))


class LowRankLinear(nn.Module):
    def __init__(self, base, rank=2, alpha=2.):
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be positive")
        self.base = copy.deepcopy(base).requires_grad_(False)
        self.A = nn.Parameter(torch.empty(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.normal_(self.A, std=.1)
        self.scale = alpha / rank

    def forward(self, values):
        return self.base(values) + self.scale * F.linear(F.linear(values, self.A), self.B)

    def merged(self):
        result = copy.deepcopy(self.base)
        with torch.no_grad():
            result.weight.add_(self.scale * self.B @ self.A)
        return result


class BottleneckAdapter(nn.Module):
    def __init__(self, dimension=16, bottleneck=4):
        super().__init__()
        self.down = nn.Linear(dimension, bottleneck)
        self.up = nn.Linear(bottleneck, dimension)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features):
        return features + self.up(torch.tanh(self.down(features)))


class Classifier(nn.Module):
    def __init__(self, backbone, head, adapter=None):
        super().__init__()
        self.backbone = backbone
        self.adapter = nn.Identity() if adapter is None else adapter
        self.head = head

    def features(self, pixels):
        return self.adapter(self.backbone(pixels))

    def forward(self, pixels):
        return self.head(self.features(pixels))


def score(model, pixels, labels):
    model.eval()
    with torch.no_grad():
        logits = model(pixels)
        return {"ce": F.cross_entropy(logits, labels).item(),
                "correct": int((logits.argmax(1) == labels).sum()),
                "count": len(labels)}


def fit(model, pixels, labels, optimizer, steps, validation=None):
    trace = []
    for step in range(steps + 1):
        if step in {0, 1, 10, 100, steps}:
            row = {"step": step, "train": score(model, pixels, labels)}
            if validation is not None:
                row["validation"] = score(model, *validation)
            trace.append(row)
        if step == steps:
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(model(pixels), labels).backward()
        optimizer.step()
    return trace


def build_target(method, pretrained, initial_backbone, initial_head):
    backbone = copy.deepcopy(initial_backbone if method == "scratch" else pretrained)
    head = copy.deepcopy(initial_head)
    adapter = None
    if method in {"probe", "lora2", "adapter4"}:
        backbone.requires_grad_(False)
    if method == "lora2":
        backbone.lower = LowRankLinear(backbone.lower)
        backbone.upper = LowRankLinear(backbone.upper)
    elif method == "adapter4":
        adapter = BottleneckAdapter()
    model = Classifier(backbone, head, adapter)
    groups = [{"params": model.head.parameters(), "lr": .01}]
    if method in {"scratch", "full", "discriminative"}:
        groups += [{"params": model.backbone.lower.parameters(),
                    "lr": .0001 if method == "discriminative" else .001},
                   {"params": model.backbone.upper.parameters(), "lr": .001}]
    elif method == "lora2":
        groups.append({"params": [p for p in model.backbone.parameters()
                                  if p.requires_grad], "lr": .01})
    elif method == "adapter4":
        groups.append({"params": model.adapter.parameters(), "lr": .01})
    return model, torch.optim.Adam(groups)


def mechanisms():
    dtype = torch.float64
    base = nn.Linear(2, 2, bias=False, dtype=dtype)
    with torch.no_grad():
        base.weight.copy_(torch.eye(2, dtype=dtype))
    layer = LowRankLinear(base, rank=1, alpha=1.).to(dtype)
    with torch.no_grad():
        layer.A.copy_(torch.tensor([[1., -1.]], dtype=dtype))
    inputs = torch.tensor([[2., 1.]], dtype=dtype)
    loss = layer(inputs).square().mean()
    loss.backward()
    first = {"loss": loss.item(), "A_gradient": layer.A.grad.tolist(),
             "B_gradient": layer.B.grad.tolist()}
    with torch.no_grad():
        layer.B.add_(layer.B.grad, alpha=-.1)
    first["after_output"] = layer(inputs).detach().tolist()
    first["after_loss"] = layer(inputs).square().mean().item()
    first["merge_max_error"] = (layer(inputs) - layer.merged()(inputs)).abs().max().item()
    layer.zero_grad(set_to_none=True)
    with torch.no_grad():
        layer.A.zero_()
        layer.B.zero_()
    layer(inputs).square().mean().backward()
    first["both_zero_gradient_max"] = max(layer.A.grad.abs().max().item(),
                                         layer.B.grad.abs().max().item())
    # A frozen parameter can still transmit a derivative to an upstream input.
    frozen_input = inputs.clone().requires_grad_(True)
    layer.base(frozen_input).square().mean().backward()
    first["actual_frozen_input_gradient"] = frozen_input.grad.tolist()
    first["actual_frozen_weight_has_gradient"] = layer.base.weight.grad is not None
    changed = {}
    for name, input_values in (("changed_input", [1., 3.]), ("null_measurement", [1., 1.])):
        with torch.no_grad():
            layer.A.copy_(torch.tensor([[1., -1.]], dtype=dtype))
            layer.B.zero_()
        layer.zero_grad(set_to_none=True)
        item = torch.tensor([input_values], dtype=dtype)
        layer(item).square().mean().backward()
        gradient = layer.B.grad.tolist()
        with torch.no_grad():
            layer.B.add_(layer.B.grad, alpha=-.1)
        changed[name] = {"B_gradient": gradient, "after_output": layer(item).detach().tolist(),
                         "after_loss": layer(item).square().mean().item()}
    first["input_contrasts"] = changed
    norm = nn.BatchNorm1d(1, momentum=.1, dtype=dtype).requires_grad_(False)
    with torch.no_grad():
        norm(torch.tensor([[1.], [3.]], dtype=dtype))
    first["frozen_bn_training_no_grad"] = {
        "mean": norm.running_mean.tolist(), "variance": norm.running_var.tolist()}
    norm.eval()
    with torch.no_grad():
        norm(torch.tensor([[11.], [13.]], dtype=dtype))
    first["frozen_bn_eval_no_grad"] = {
        "mean": norm.running_mean.tolist(), "variance": norm.running_var.tolist()}
    return first


def run():
    directory = Path(__file__).resolve().parent
    data_path = directory / "digits-400.csv"
    data = np.genfromtxt(data_path, delimiter=",", names=True, dtype=np.int64)
    pixels = torch.tensor(np.column_stack([data[f"pixel_{i}"] for i in range(64)]),
                          dtype=torch.float32) / 16
    digits = torch.tensor(data["digit"], dtype=torch.long)
    groups = {int(d): np.flatnonzero(data["digit"] == d) for d in range(10)}
    indices = {
        "source_train": np.concatenate([groups[d][:30] for d in range(5)]),
        "source_holdout": np.concatenate([groups[d][30:] for d in range(5)]),
        "target_train": np.concatenate([groups[d][:8] for d in range(5, 10)]),
        "target_validation": np.concatenate([groups[d][8:20] for d in range(5, 10)]),
        "target_test": np.concatenate([groups[d][20:] for d in range(5, 10)]),
    }
    def subset(name):
        rows = indices[name]
        labels = digits[rows] - (5 if name.startswith("target") else 0)
        return pixels[rows], labels

    results, selected = [], None
    for seed in (1, 2, 3):
        torch.manual_seed(seed)
        original_backbone = Backbone()
        source = Classifier(copy.deepcopy(original_backbone), nn.Linear(16, 5))
        source_trace = fit(source, *subset("source_train"),
                           torch.optim.Adam(source.parameters(), lr=.01), steps=400)
        source_before = score(source, *subset("source_holdout"))
        torch.manual_seed(100 + seed)
        target_head = nn.Linear(16, 5)
        candidates = []
        for method in METHODS:
            # Shared head and saved backbone are unaffected by method creation order.
            torch.manual_seed(200 + seed)
            model, optimizer = build_target(method, source.backbone,
                                             original_backbone, target_head)
            trace = fit(model, *subset("target_train"), optimizer, steps=300,
                        validation=subset("target_validation"))
            after_source = Classifier(copy.deepcopy(model.backbone),
                                       copy.deepcopy(source.head), copy.deepcopy(model.adapter))
            record = {"seed": seed, "method": method,
                      "trainable_parameters": sum(p.numel() for p in model.parameters()
                                                  if p.requires_grad),
                      "total_parameters": sum(p.numel() for p in model.parameters()),
                      "train": trace[-1]["train"], "validation": trace[-1]["validation"],
                      "source_before": source_before,
                      "source_after": None if method == "scratch" else
                          score(after_source, *subset("source_holdout")),
                      "trace": trace}
            if method == "lora2":
                merged = copy.deepcopy(model)
                merged.backbone.lower = model.backbone.lower.merged()
                merged.backbone.upper = model.backbone.upper.merged()
                with torch.no_grad():
                    record["merge_max_logit_error"] = (
                        merged(subset("target_validation")[0]) -
                        model(subset("target_validation")[0])).abs().max().item()
            results.append(record)
            if seed == 1:
                candidates.append((record["validation"]["ce"], model, record))
        if seed == 1:
            # Python min keeps the first candidate on an exact tie: METHODS order.
            _, selected_model, selected_record = min(candidates, key=lambda item: item[0])
            selected = (copy.deepcopy(selected_model), copy.deepcopy(selected_record))
        results.append({"seed": seed, "source_pretraining_trace": source_trace})
    selected_model, selected_record = selected
    selected_model.eval()
    test_pixels, test_labels = subset("target_test")
    with torch.no_grad():
        probabilities = selected_model(test_pixels).softmax(-1)
    # Check a configuration-preserving state round trip without retaining a binary.
    reload_model = copy.deepcopy(selected_model)
    portable_state = {name: value.detach().tolist()
                      for name, value in selected_model.state_dict().items()}
    reload_model.load_state_dict({name: torch.tensor(portable_state[name], dtype=value.dtype)
        for name, value in selected_model.state_dict().items()})
    with torch.no_grad():
        replay_error = (reload_model(test_pixels).softmax(-1) - probabilities).abs().max().item()
    output = {
        "versions": {"torch": torch.__version__, "numpy": np.__version__},
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "splits_source_ids": {name: data["source_id"][rows].tolist()
                             for name, rows in indices.items()},
        "mechanisms": mechanisms(), "runs": results,
        "selection": {"rule": "lowest final validation CE among seed 1; METHODS order breaks ties",
                      "seed": selected_record["seed"], "method": selected_record["method"],
                      "test": score(selected_model, test_pixels, test_labels),
                      "test_probabilities": probabilities.tolist(),
                      "test_labels": test_labels.tolist(),
                      "state_roundtrip_max_probability_error": replay_error},
    }
    (directory / "calculated-inputs.json").write_text(json.dumps(output, indent=2) + "\n",
                                                     encoding="utf-8")
    for row in results:
        if "method" in row:
            print(row["seed"], row["method"], row["trainable_parameters"],
                  round(row["validation"]["ce"], 6), row["validation"]["correct"],
                  None if row["source_after"] is None else row["source_after"]["correct"])
    print("selected", selected_record["method"], output["selection"]["test"])


if __name__ == "__main__":
    run()
