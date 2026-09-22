"""CPU initialization experiments; run beside digits-400.csv.

Python 3.12, torch 2.14, NumPy 2.3.5, scikit-learn 1.9.1.
No pretrained model or additional package is downloaded.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split

torch.set_num_threads(1)
SCHEMES = ("zero", "small", "xavier", "kaiming", "orthogonal", "large")


def fill_weight(weight, scheme, generator):
    fan_out, fan_in = weight.shape
    if scheme == "zero":
        nn.init.zeros_(weight)
    elif scheme == "small":
        nn.init.normal_(weight, std=.01, generator=generator)
    elif scheme == "large":
        nn.init.normal_(weight, std=.5, generator=generator)
    elif scheme == "xavier":
        nn.init.normal_(weight, std=math.sqrt(2 / (fan_in + fan_out)), generator=generator)
    elif scheme == "kaiming":
        nn.init.normal_(weight, std=math.sqrt(2 / fan_in), generator=generator)
    elif scheme == "orthogonal":
        nn.init.orthogonal_(weight, gain=math.sqrt(2), generator=generator)
    else:
        raise ValueError("unknown initialization scheme")


def summary(values):
    detached = values.detach()
    return {"mean": detached.mean().item(),
            "second_moment": detached.square().mean().item(),
            "variance": detached.var(correction=0).item(),
            "zero_fraction": (detached == 0).double().mean().item()}


def signal_propagation():
    width, depth, batch = 64, 20, 128
    generator = torch.Generator().manual_seed(71)
    inputs = torch.randn(batch, width, dtype=torch.float64, generator=generator)
    cotangent = torch.randn(batch, width, dtype=torch.float64, generator=generator)
    records = []
    for seed in (1, 2, 3):
        for scheme in SCHEMES:
            generator = torch.Generator().manual_seed(seed)
            current = inputs.clone().requires_grad_(True)
            states = [current]
            for _ in range(depth):
                weight = torch.empty(width, width, dtype=torch.float64)
                fill_weight(weight, scheme, generator)
                current = F.relu(F.linear(current, weight))
                current.retain_grad()
                states.append(current)
            (current * cotangent).sum().backward()
            records.append({"seed": seed, "scheme": scheme,
                "layers": [{"layer": i, **summary(state),
                            "gradient_rms": state.grad.square().mean().sqrt().item()}
                           for i, state in enumerate(states)]})
    return {"width": width, "depth": depth, "batch": batch,
            "input_seed": 71, "dtype": "float64",
            "cotangent_rms": cotangent.square().mean().sqrt().item(), "records": records}


class DigitMLP(nn.Module):
    def __init__(self, scheme, seed):
        super().__init__()
        self.hidden = nn.ModuleList([nn.Linear(64, 32), nn.Linear(32, 32),
                                    nn.Linear(32, 32), nn.Linear(32, 32)])
        self.head = nn.Linear(32, 10)
        generator = torch.Generator().manual_seed(seed)
        for layer in self.hidden:
            fill_weight(layer.weight, scheme, generator)
            nn.init.zeros_(layer.bias)
        # All schemes have exactly the same random output head for this seed.
        nn.init.xavier_normal_(self.head.weight, generator=torch.Generator().manual_seed(100+seed))
        nn.init.zeros_(self.head.bias)

    def forward(self, pixels):
        for layer in self.hidden:
            pixels = F.relu(layer(pixels))
        return self.head(pixels)


def score(model, pixels, labels):
    model.eval()
    with torch.no_grad():
        scores = model(pixels)
        return {"ce": F.cross_entropy(scores, labels).item(),
                "correct": int((scores.argmax(-1) == labels).sum()),
                "count": len(labels)}


def digit_fits(train_pixels, train_labels, validation_pixels, validation_labels):
    records = []
    for seed in (1, 2, 3):
        for scheme in SCHEMES:
            model = DigitMLP(scheme, seed)
            optimizer = torch.optim.Adam(model.parameters(), lr=.003)
            trace = []
            for step in range(301):
                if step in {0, 1, 10, 100, 300}:
                    trace.append({"step": step, "train": score(model, train_pixels, train_labels),
                        "validation": score(model, validation_pixels, validation_labels)})
                if step == 300:
                    break
                optimizer.zero_grad(set_to_none=True)
                model.train()
                loss = F.cross_entropy(model(train_pixels), train_labels)
                if not torch.isfinite(loss):
                    raise ValueError("non-finite loss; preserve and investigate this configuration")
                loss.backward()
                optimizer.step()
            records.append({"seed": seed, "scheme": scheme, "trace": trace})
    return records


class WidthMLP(nn.Module):
    """Restricted bias-free MLP: fixed input/output sizes, two equally wide ReLU layers.

For mu mode, raw readout initialization is width-independent and its forward
input is divided by width/base_width. Hidden Adam learning rate divides by
the same ratio. This is the MuReadout/MuAdam convention for this exact case,
not a general replacement for the mup package's shape machinery.
"""
    def __init__(self, width, mode, seed, base_width=32):
        super().__init__()
        if mode not in {"standard", "mu"}:
            raise ValueError("mode must be standard or mu")
        self.lower = nn.Linear(64, width, bias=False)
        self.upper = nn.Linear(width, width, bias=False)
        self.readout = nn.Linear(width, 10, bias=False)
        self.multiplier = width / base_width if mode == "mu" else 1.
        generator = torch.Generator().manual_seed(seed)
        nn.init.normal_(self.lower.weight, std=math.sqrt(2/64), generator=generator)
        nn.init.normal_(self.upper.weight, std=math.sqrt(2/width), generator=generator)
        nn.init.normal_(self.readout.weight,
                        std=1/math.sqrt(base_width if mode == "mu" else width),
                        generator=generator)

    def coordinates(self, pixels):
        lower = self.lower(pixels)
        hidden_lower = F.relu(lower)
        upper = self.upper(hidden_lower)
        hidden_upper = F.relu(upper)
        output = self.readout(hidden_upper / self.multiplier)
        return lower, hidden_lower, upper, hidden_upper, output

    def forward(self, pixels):
        return self.coordinates(pixels)[-1]

    def optimizer(self, rate):
        return torch.optim.Adam([
            {"params": self.lower.parameters(), "lr": rate},
            {"params": self.upper.parameters(), "lr": rate/self.multiplier},
            {"params": self.readout.parameters(), "lr": rate},
        ], eps=1e-8)


def width_fits(train_pixels, train_labels, validation_pixels, validation_labels):
    records = []
    for seed in (1, 2, 3):
        for width in (32, 64, 128):
            for mode in ("standard", "mu"):
                for rate in (.001, .003, .01):
                    model = WidthMLP(width, mode, seed)
                    optimizer = model.optimizer(rate)
                    trace = []
                    for step in range(151):
                        if step in {0, 1, 2, 5, 150}:
                            with torch.no_grad():
                                coordinates = model.coordinates(train_pixels[:32])
                                magnitudes = [coordinate.abs().mean().item()
                                              for coordinate in coordinates]
                            trace.append({"step": step, "coordinate_mean_abs": magnitudes,
                                "train": score(model, train_pixels, train_labels),
                                "validation": score(model, validation_pixels, validation_labels)})
                        if step == 150:
                            break
                        optimizer.zero_grad(set_to_none=True)
                        model.train()
                        F.cross_entropy(model(train_pixels), train_labels).backward()
                        optimizer.step()
                    records.append({"seed": seed, "width": width, "mode": mode, "rate": rate,
                                    "trace": trace})
    return records


def exact_fixtures():
    dtype = torch.float64
    values = torch.tensor([-2., -1., 1., 2.], dtype=dtype)
    relu = F.relu(values)
    matrix = torch.diag(torch.sqrt(torch.tensor([1.9, .1], dtype=dtype)))
    generator = torch.Generator().manual_seed(19)
    gaussian = torch.randn(64, 64, generator=generator, dtype=dtype)/8
    q, _ = torch.linalg.qr(gaussian)
    spectrum = torch.linalg.svdvals(gaussian)
    inputs = torch.tensor([-1., 1.], dtype=dtype)
    gated = torch.func.jacrev(lambda item: F.relu(math.sqrt(2)*item))(inputs)
    orthogonal_error = (q.T @ q - torch.eye(64, dtype=dtype)).abs().max().item()
    # Symmetric hidden units versus deliberately distinct deterministic units.
    symmetry = []
    for name, initial_rows in (("identical", [.2,.2]), ("distinct", [.1,.3])):
        weight = torch.tensor(initial_rows, dtype=dtype, requires_grad=True)
        outgoing = torch.tensor([.3,.3], dtype=dtype, requires_grad=True)
        prediction = (outgoing * torch.tanh(weight)).sum()
        (.5*(prediction-1).square()).backward()
        symmetry.append({"name": name, "weight": initial_rows,
                         "prediction": prediction.item(), "weight_gradient": weight.grad.tolist()})
    # Zero readout with distinct features has a learning signal at the readout.
    weight = torch.tensor([.1,.3], dtype=dtype, requires_grad=True)
    outgoing = torch.zeros(2, dtype=dtype, requires_grad=True)
    (.5*((outgoing*torch.tanh(weight)).sum()-1).square()).backward()
    zero_readout = {"hidden_gradient": weight.grad.tolist(), "readout_gradient": outgoing.grad.tolist()}
    truncation = {}
    for name, bounds in (("default_absolute", (-2.,2.)), ("two_sigma", (-.04,.04))):
        sample = torch.empty(100000, dtype=dtype)
        nn.init.trunc_normal_(sample, std=.02, a=bounds[0], b=bounds[1],
                              generator=torch.Generator().manual_seed(13))
        truncation[name] = {"std": sample.std(correction=0).item(),
                            "max_abs": sample.abs().max().item(), "bounds": bounds}
    precision = torch.tensor([1e-5,1e-6,1e-8], dtype=torch.float32)
    changed = torch.tensor([-3., -1., 1., 3.], dtype=dtype)
    intervention = []
    for smaller_gain in (.2, math.sqrt(.1), 1.):
        larger_gain = math.sqrt(2-smaller_gain**2)
        intervention.append({"smaller_gain": smaller_gain, "larger_gain": larger_gain,
                             "average_squared_gain": (smaller_gain**2+larger_gain**2)/2,
                             "five_layer_smaller_gain": smaller_gain**5})
    width_checks = []
    for width in (32,128,256):
        multiplier = width/32
        rate = .004 if width==256 else .003
        width_checks.append({"width": width, "base_width": 32, "base_rate": rate,
                             "input_rate": rate, "hidden_rate": rate/multiplier,
                             "readout_rate": rate, "readout_input_multiplier": 1/multiplier,
                             "raw_readout_std": 1/math.sqrt(32)})
    return {"relu_input": summary(values), "relu_output": summary(relu),
            "changed_relu_input": summary(changed), "changed_relu_output": summary(F.relu(changed)),
            "geometry_interventions": intervention, "width_checks": width_checks,
            "average_gain_matrix": matrix.tolist(),
            "average_squared_singular_value": matrix.square().sum().item()/2,
            "singular_values": torch.linalg.svdvals(matrix).tolist(),
            "gaussian_seed": 19, "gaussian_singular_values": spectrum.tolist(),
            "orthogonal_gram_max_error": orthogonal_error,
            "relu_scaled_identity_jacobian": gated.tolist(),
            "relu_scaled_identity_singular_values": torch.linalg.svdvals(gated).tolist(),
            "symmetry": symmetry, "zero_readout": zero_readout,
            "truncation": truncation,
            "precision": {"float32": precision.tolist(), "bfloat16_back_to_float32":
                          precision.to(torch.bfloat16).float().tolist(),
                          "float16_back_to_float32": precision.half().float().tolist()}}


def run():
    directory = Path(__file__).resolve().parent
    data = np.genfromtxt(directory/"digits-400.csv", delimiter=",", names=True, dtype=np.int64)
    pixels = torch.tensor(np.column_stack([data[f"pixel_{i}"] for i in range(64)]),
                          dtype=torch.float32)/16
    labels = torch.tensor(data["digit"], dtype=torch.long)
    training, validation = train_test_split(np.arange(400), test_size=.3,
                                            stratify=data["digit"], random_state=22)
    arguments = pixels[training], labels[training], pixels[validation], labels[validation]
    output = {"versions": {"torch": torch.__version__, "numpy": np.__version__},
              "training_source_ids": data["source_id"][training].tolist(),
              "validation_source_ids": data["source_id"][validation].tolist(),
              "fixtures": exact_fixtures(), "propagation": signal_propagation(),
              "digit_fits": digit_fits(*arguments), "width_fits": width_fits(*arguments)}
    (directory/"calculated-inputs.json").write_text(json.dumps(output, indent=2)+"\n", encoding="utf-8")
    for row in output["digit_fits"]:
        final = row["trace"][-1]
        print(row["seed"], row["scheme"], round(final["train"]["ce"],6),
              round(final["validation"]["ce"],6), final["validation"]["correct"])
    for mode in ("standard", "mu"):
        for width in (32,64,128):
            means = {rate: np.mean([row["trace"][-1]["validation"]["ce"]
                                    for row in output["width_fits"]
                                    if row["mode"]==mode and row["width"]==width and row["rate"]==rate])
                     for rate in (.001,.003,.01)}
            print(mode, width, means)


if __name__ == "__main__":
    run()
