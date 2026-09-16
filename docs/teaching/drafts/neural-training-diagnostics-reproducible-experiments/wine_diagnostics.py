"""Offline controlled teaching runs; writes wine-results.json beside this file."""

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import torch
from torch import nn

PACKET = Path(__file__).resolve().parent
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.set_default_dtype(torch.float64)


def load_inputs():
    rows = np.loadtxt(PACKET / "wine.csv", delimiter=",", skiprows=1)
    split = json.loads((PACKET / "split.json").read_text())
    features = torch.tensor(rows[:, 2:])
    labels = torch.tensor(rows[:, 1], dtype=torch.long)
    training_features = features[split["train"]]
    mean = training_features.mean(dim=0)
    scale = training_features.std(dim=0, correction=0)
    return (features - mean) / scale, labels, split, mean, scale


def make_model(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(13, 32), nn.Tanh(), nn.Linear(32, 3))


def scores(model, features, labels):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        return {
            "loss": nn.functional.cross_entropy(logits, labels).item(),
            "accuracy": (logits.argmax(dim=1) == labels).double().mean().item(),
        }


def fit(features, labels, split, seed, treatment, updates):
    ids = split["tiny"] if treatment in {"tiny", "omitted_update"} else split["train"]
    training_features = features[ids]
    original_targets = labels[ids]
    supplied_targets = original_targets.clone()
    if treatment == "shuffled_labels":
        order = np.random.default_rng(991).permutation(len(ids))
        supplied_targets = supplied_targets[order]
    model = make_model(seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    trace = []
    for update in range(updates + 1):
        trace.append({
            "update": update,
            "train_supplied": scores(model, training_features, supplied_targets),
            "train_original": scores(model, training_features, original_targets),
            "validation": scores(model, features[split["validation"]], labels[split["validation"]]),
        })
        if update == updates:
            break
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(training_features)
        loss = nn.functional.cross_entropy(logits, supplied_targets)
        loss.backward()
        gradient_norm = torch.sqrt(sum(p.grad.square().sum() for p in model.parameters())).item()
        before = [p.detach().clone() for p in model.parameters()]
        if treatment != "omitted_update":
            optimizer.step()
        update_norm = torch.sqrt(sum((p.detach() - old).square().sum() for p, old in zip(model.parameters(), before))).item()
        trace[-1]["next_update_gradient_norm"] = gradient_norm
        trace[-1]["next_update_parameter_change_norm"] = update_norm
    final_predictions = {}
    with torch.no_grad():
        for role, row_ids, targets in [
            ("training", ids, supplied_targets),
            ("validation", split["validation"], labels[split["validation"]]),
        ]:
            probabilities = model(features[row_ids]).softmax(dim=1)
            final_predictions[role] = [
                {"row_id": row_id, "original_target": labels[row_id].item(),
                 "supplied_target": targets[position].item(),
                 "predicted_class": probabilities[position].argmax().item(),
                 "probabilities": probabilities[position].tolist()}
                for position, row_id in enumerate(row_ids)
            ]
    return {"seed": seed, "treatment": treatment, "training_row_ids": ids,
            "supplied_targets": supplied_targets.tolist(), "trace": trace,
            "final_predictions": final_predictions}


def activation_probe(features):
    model = make_model(7)
    result = []
    with torch.no_grad():
        for factor in [1.0, 100.0]:
            activation = model[1](model[0](factor * features))
            result.append({
                "input_multiplier": factor,
                "absolute_activation_quantiles_10_50_90": torch.quantile(activation.abs(), torch.tensor([0.1, 0.5, 0.9])).tolist(),
                "mean_tanh_derivative": (1 - activation.square()).mean().item(),
            })
    return result


def main():
    features, labels, split, mean, scale = load_inputs()
    runs = [fit(features, labels, split, 7, treatment, 120)
            for treatment in ["tiny", "omitted_update"]]
    for seed in [3, 7, 19]:
        for treatment in ["clean", "shuffled_labels"]:
            runs.append(fit(features, labels, split, seed, treatment, 400))
    result = {
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "torch": torch.__version__, "platform": platform.platform(),
                        "processor": platform.processor(), "device": "CPU", "threads": 1,
                        "dtype": "float64", "deterministic_algorithms": True},
        "input_sha256": {name: hashlib.sha256((PACKET / name).read_bytes()).hexdigest()
                         for name in ["wine.csv", "split.json", "experiment-protocol.md"]},
        "training_mean": mean.tolist(), "training_scale": scale.tolist(),
        "activation_probe": activation_probe(features[split["train"]]),
        "runs": runs,
    }
    (PACKET / "wine-results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for run in runs:
        last = run["trace"][-1]
        print(run["treatment"], run["seed"],
              f"train={last['train_supplied']['loss']:.6f}/{last['train_supplied']['accuracy']:.6f}",
              f"validation={last['validation']['loss']:.6f}/{last['validation']['accuracy']:.6f}")
    print(json.dumps(result["activation_probe"]))


if __name__ == "__main__":
    main()
