"""Train small real-digit models, then factor their fixed spatial kernels.

Run beside the attributed digits-400.csv. No downloads or pretrained weights.
The development split is an investigation, not an untouched final test.
"""
from pathlib import Path
import copy
import json
import platform
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
torch.set_num_threads(1)


class DigitClassifier(nn.Module):
    def __init__(self, dilation=1):
        super().__init__()
        self.stem = nn.Conv2d(1, 8, 3, padding=1)
        self.spatial = nn.Conv2d(8, 12, 3, padding=dilation,
                                 dilation=dilation)
        self.head = nn.Linear(12 * 4 * 4, 10)

    def forward(self, images):
        features = F.relu(self.spatial(F.relu(self.stem(images))))
        pooled = F.avg_pool2d(features, 2)
        return self.head(pooled.flatten(1))


def factor_spatial(layer, multiplier):
    """Per-input-channel truncated SVD; no intermediate nonlinearity."""
    outputs, inputs, height, width = layer.weight.shape
    rank_limit = min(outputs, height * width)
    if not 1 <= multiplier <= rank_limit:
        raise ValueError("multiplier outside matrix rank bound")
    depthwise = nn.Conv2d(inputs, inputs * multiplier, (height, width),
                         padding=layer.padding, dilation=layer.dilation,
                         stride=layer.stride, groups=inputs, bias=False)
    pointwise = nn.Conv2d(inputs * multiplier, outputs, 1,
                         bias=layer.bias is not None)
    depthwise = depthwise.to(layer.weight)
    pointwise = pointwise.to(layer.weight)
    singular_values = []
    with torch.no_grad():
        for channel in range(inputs):
            matrix = layer.weight[:, channel].reshape(outputs, -1)
            left, values, right = torch.linalg.svd(matrix, full_matrices=False)
            singular_values.append(values.tolist())
            start = channel * multiplier
            depthwise.weight[start:start + multiplier, 0] = (
                right[:multiplier].reshape(multiplier, height, width))
            pointwise.weight[:, start:start + multiplier, 0, 0] = (
                left[:, :multiplier] * values[:multiplier])
        if layer.bias is not None:
            pointwise.bias.copy_(layer.bias)
    return nn.Sequential(depthwise, pointwise), singular_values


def reconstructed_weight(factorized):
    depthwise, pointwise = factorized
    inputs = depthwise.in_channels
    multiplier = depthwise.out_channels // inputs
    return torch.einsum(
        "ocm,cmhw->ochw",
        pointwise.weight[:, :, 0, 0].reshape(-1, inputs, multiplier),
        depthwise.weight[:, 0].reshape(inputs, multiplier, *depthwise.kernel_size))


def score(model, images, labels):
    model.eval()
    with torch.no_grad():
        logits = model(images)
        predictions = logits.argmax(1)
        return {
            "correct": int((predictions == labels).sum()), "count": len(labels),
            "cross_entropy": float(F.cross_entropy(logits, labels)),
            "predictions": predictions.tolist(),
            "per_class_correct": [
                int(((predictions == labels) & (labels == digit)).sum())
                for digit in range(10)],
        }


def compact_state(model):
    return {name: value.detach().tolist() for name, value in model.state_dict().items()}


def main():
    rows = np.genfromtxt(HERE / "digits-400.csv", delimiter=",", names=True)
    pixels = np.column_stack([rows[f"pixel_{i}"] for i in range(64)])
    ids = rows["source_id"].astype(int)
    labels = torch.tensor(rows["digit"].astype(int))
    images = torch.tensor(pixels / 16, dtype=torch.float32).reshape(-1, 1, 8, 8)
    if len(np.unique(pixels, axis=0)) != 400 or len(np.unique(ids)) != 400:
        raise ValueError("duplicate pixels/IDs require a revised split")
    training, development = train_test_split(
        np.arange(400), test_size=.3, random_state=22, stratify=labels.numpy())
    result = {
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "torch": torch.__version__, "sklearn": sklearn.__version__,
                        "torch_threads": 1},
        "data_audit": {"rows":400, "unique_images":400, "unique_source_ids":400},
        "training_source_ids":ids[training].tolist(),
        "development_source_ids":ids[development].tolist(),
        "development_labels":labels[development].tolist(),
        "runs":[],
    }
    for seed in (1, 2, 3):
        for dilation in (1, 2):
            torch.manual_seed(seed)
            model = DigitClassifier(dilation)
            optimizer = torch.optim.Adam(model.parameters(), lr=.003)
            trace = []
            for step in range(401):
                if step in (0, 1, 25, 100, 200, 400):
                    trace.append({
                        "step":step,
                        "training":score(model, images[training], labels[training]),
                        "development":score(model, images[development], labels[development]),
                    })
                if step == 400:
                    break
                model.train()
                optimizer.zero_grad()
                loss = F.cross_entropy(model(images[training]), labels[training])
                loss.backward()
                optimizer.step()
            run = {
                "seed":seed, "dilation":dilation,
                "parameters":sum(value.numel() for value in model.parameters()),
                "convolution_linear_macs_per_image":64*8*9 + 64*12*8*9 + 192*10,
                "trace":trace,
                "dense":score(model, images[development], labels[development]),
                "factorizations":[],
            }
            with torch.no_grad():
                dense_logits = model(images[development])
            for multiplier in (1, 2, 4, 9):
                compressed = copy.deepcopy(model)
                compressed.spatial, singular_values = factor_spatial(model.spatial, multiplier)
                approximation = reconstructed_weight(compressed.spatial)
                with torch.no_grad():
                    logits = compressed(images[development])
                    relative_error = float(torch.linalg.vector_norm(approximation-model.spatial.weight)
                                           / torch.linalg.vector_norm(model.spatial.weight))
                item = {
                    "multiplier":multiplier,
                    "parameters":sum(value.numel() for value in compressed.parameters()),
                    "convolution_linear_macs_per_image":
                        64*8*9 + 64*8*multiplier*(9+12) + 192*10,
                    "relative_weight_frobenius_error":relative_error,
                    "max_abs_logit_change":float((logits-dense_logits).abs().max()),
                    "development":score(compressed, images[development], labels[development]),
                }
                if seed == 1:
                    item["factorized_spatial_state"] = compact_state(compressed.spatial)
                run["factorizations"].append(item)
            if seed == 1:
                run["dense_state"] = compact_state(model)
                run["spatial_singular_values_by_input_channel"] = singular_values
                disagreements = [
                    index for index, (dense_label, rank_one_label) in enumerate(zip(
                        run["dense"]["predictions"],
                        run["factorizations"][0]["development"]["predictions"]))
                    if dense_label != rank_one_label]
                selected = sorted(set([0] + disagreements[:1]))
                run["examples"] = [
                    {"development_index":index, "source_id":int(ids[development[index]]),
                     "actual":int(labels[development[index]]),
                     "input":images[development[index], 0].tolist(),
                     "dense_logits":dense_logits[index].tolist()}
                    for index in selected]
            result["runs"].append(run)
            print(seed, dilation, run["dense"]["correct"],
                  [(x["multiplier"], x["development"]["correct"])
                   for x in run["factorizations"]], flush=True)
    (HERE / "calculated-inputs.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
