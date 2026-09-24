"""Complete CPU comparison of dense, local and normalized feature attention.

Run beside the original UCI Libras files with Python, NumPy and PyTorch.
No network, GPU, optional tokenizer or missing training implementation.
"""
from pathlib import Path
import copy
import hashlib
import json
import math
import platform
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent


def load_records():
    source = ROOT / "movement_libras.data"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        "97ebdaa6a9b28ab4a2cdd84b14f19a95a7456a46137c362b65a0669eca3c3c4d")
    raw = np.loadtxt(source, delimiter=",")
    groups = {}
    for row, features in enumerate(raw[:, :90]):
        groups.setdefault(tuple(features), []).append(row)
    for rows in groups.values():
        assert len(set(raw[rows, 90])) == 1
    retained = np.array([rows[0] for rows in groups.values()])
    generator = np.random.default_rng(73)
    split = {name: [] for name in ("train", "validation", "test")}
    for label in range(1, 16):
        rows = generator.permutation(retained[raw[retained, 90] == label])
        boundary = len(rows) * 2 // 3
        split["train"].extend(rows[:boundary].tolist())
        split["validation"].extend(rows[boundary:-4].tolist())
        split["test"].extend(rows[-4:].tolist())
    assert [len(split[name]) for name in split] == [220, 50, 60]
    points = torch.tensor(raw[:, :90].reshape(-1, 45, 2)*2-1, dtype=torch.float32)
    pairs = {name: (points[rows, :-1], points[rows, 1:]) for name, rows in split.items()}
    return points, pairs, split, [rows for rows in groups.values() if len(rows) > 1]


def positive_features(values):
    # Numerically avoid subtracting 1 and adding it back on the negative branch.
    return torch.where(values >= 0, values + 1, torch.exp(values))


class AttentionForecaster(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode, self.heads, self.head_width, self.window = mode, 3, 8, 5
        self.stem = nn.Linear(2, 24)
        self.norm_attention = nn.LayerNorm(24, eps=1e-5)
        self.qkv = nn.Linear(24, 72, bias=False)
        self.output = nn.Linear(24, 24, bias=False)
        self.norm_feedforward = nn.LayerNorm(24, eps=1e-5)
        self.feedforward = nn.Sequential(nn.Linear(24, 48), nn.GELU(), nn.Linear(48, 24))
        self.norm_final = nn.LayerNorm(24, eps=1e-5)
        self.forecast = nn.Linear(24, 2)

    def forward(self, points, positions=None, cache=None, reference=False, trace=False):
        batch, length, _ = points.shape
        if positions is None:
            positions = torch.arange(length, device=points.device)
        frequency = 10000. ** (-torch.arange(0, 24, 2, dtype=points.dtype,
                                            device=points.device)/24)
        angles = positions.to(points.dtype)[:, None]*frequency
        encoding = torch.stack((angles.sin(), angles.cos()), -1).flatten(-2)
        state = self.stem(points)+encoding[None]
        q, k, v = self.qkv(self.norm_attention(state)).reshape(
            batch, length, 3, self.heads, self.head_width).permute(2, 0, 3, 1, 4)
        details = {"query": q, "key": k, "value": v}
        if self.mode == "kernel":
            if reference and cache is not None:
                raise ValueError("The quadratic reference consumes a full prefix.")
            q_features = positive_features(q/self.head_width**.25)
            k_features = positive_features(k/self.head_width**.25)
            outer = k_features[..., :, None]*v[..., None, :]
            prefix_matrix = outer.cumsum(2)
            prefix_normalizer = k_features.cumsum(2)
            if cache is not None:
                prefix_matrix = prefix_matrix+cache[0][:, :, None]
                prefix_normalizer = prefix_normalizer+cache[1][:, :, None]
            denominator = torch.einsum("bhtm,bhtm->bht", q_features, prefix_normalizer)
            if reference:
                similarities = q_features@k_features.transpose(-1, -2)
                legal = positions[:, None] >= positions[None, :]
                similarities = similarities.masked_fill(~legal, 0.)
                weights = similarities/similarities.sum(-1, keepdim=True).clamp_min(1e-9)
                attended = weights@v
                details["weights"] = weights
            else:
                attended = torch.einsum("bhtm,bhtmv->bhtv", q_features, prefix_matrix)
                attended = attended/denominator.clamp_min(1e-9)[..., None]
            new_cache = (prefix_matrix[:, :, -1], prefix_normalizer[:, :, -1])
            details.update({"q_features": q_features, "k_features": k_features,
                "prefix_matrix": prefix_matrix, "prefix_normalizer": prefix_normalizer,
                "denominator": denominator})
        else:
            if cache is None:
                memory_positions = positions
            else:
                k = torch.cat((cache[0], k), 2)
                v = torch.cat((cache[1], v), 2)
                memory_positions = torch.cat((cache[2], positions))
            legal = positions[:, None] >= memory_positions[None, :]
            if self.mode == "window":
                legal &= positions[:, None]-memory_positions[None, :] < self.window
            logits = q@k.transpose(-1, -2)/math.sqrt(self.head_width)
            weights = logits.masked_fill(~legal, -torch.inf).softmax(-1)
            attended = weights@v
            retained = self.window if self.mode == "window" else len(memory_positions)
            new_cache = (k[:, :, -retained:], v[:, :, -retained:], memory_positions[-retained:])
            details.update({"weights": weights, "legal": legal, "logits": logits})
        state = state+self.output(attended.transpose(1, 2).reshape(batch, length, 24))
        state = state+self.feedforward(self.norm_feedforward(state))
        prediction = self.forecast(self.norm_final(state))
        details["attention_head_output"] = attended
        return (prediction, new_cache, details) if trace else (prediction, new_cache)


@torch.no_grad()
def evaluate(model, pairs):
    return {name: float((model(inputs)[0]-targets).square().mean().sqrt())/2
            for name, (inputs, targets) in pairs.items()}


def train_model(mode, pairs):
    torch.manual_seed(137)
    model = AttentionForecaster(mode)
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    best, selected, history, best_weights = math.inf, None, [], None
    for update in range(1, 161):
        model.train()
        prediction, _ = model(pairs["train"][0])
        loss = (prediction-pairs["train"][1]).square().mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = float((model(pairs["validation"][0])[0]-pairs["validation"][1]).square().mean())
        history.append({"update": update, "pre_update_training_mse": float(loss.detach()),
                        "post_update_validation_mse": validation})
        if validation < best:
            best, selected = validation, update
            best_weights = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_weights)
    model.eval()
    return model, {"selected_update": selected, "history": history, "metrics_rmse_original": evaluate(model, pairs)}


def serialize(value):
    if isinstance(value, torch.Tensor): return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k: serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [serialize(v) for v in value]
    return value


def gradient_check(model, input_points):
    first, second = copy.deepcopy(model).double(), copy.deepcopy(model).double()
    a, _ = first(input_points.double(), reference=True)
    b, _ = second(input_points.double(), reference=False)
    a.square().sum().backward(); b.square().sum().backward()
    errors = {}
    for (name, p), (other, q) in zip(first.named_parameters(), second.named_parameters()):
        assert name == other and p.grad is not None and q.grad is not None
        errors[name] = float((p.grad-q.grad).abs().max())
        assert torch.allclose(p.grad, q.grad, atol=1e-9, rtol=1e-9), name
    return {"output_max_error": float((a-b).detach().abs().max()),
            "all_parameter_gradient_max_error": max(errors.values()), "parameter_errors": errors}


@torch.no_grad()
def inspect(model, points, prefix_length=32, edit_position=23, edit_coordinate=0):
    prefix = points[76:77, :prefix_length]
    output, cache, details = model(prefix, trace=True)
    incremental, running = [], None
    for position in range(prefix_length):
        value, running = model(prefix[:, position:position+1], torch.tensor([position]), running)
        incremental.append(value)
    incremental = torch.cat(incremental, 1)
    changed = prefix.clone(); changed[:, edit_position, edit_coordinate] *= -1
    changed_output, _ = model(changed)
    assert torch.equal(output[:, :edit_position], changed_output[:, :edit_position])
    assert torch.allclose(output, incremental, atol=3e-6, rtol=3e-6)
    reference_error = None
    if model.mode == "kernel":
        reference_error = float((model(prefix, reference=True)[0]-output).abs().max())
        assert reference_error < 3e-6
    return {"source_row": 77, "prefix_length": prefix_length, "observed_points": (prefix+1)/2,
        "true_next_point": (points[76, prefix_length]+1)/2,
        "forecast": (output[0, -1]+1)/2, "edited_forecast": (changed_output[0, -1]+1)/2,
        "edit_position": edit_position, "edit_coordinate": edit_coordinate,
        "earlier_output_max_error": float((output[:, :edit_position]-changed_output[:, :edit_position]).abs().max()),
        "incremental_max_error": float((output-incremental).abs().max()),
        "kernel_reference_max_error": reference_error,
        "retained_numeric_payload_bytes": sum(t.numel()*t.element_size() for t in cache[:2]),
        "cache_shapes": [list(t.shape) for t in cache[:2]], "trace": details}


def random_feature_study(dense_trace):
    query = np.asarray(dense_trace["query"])[0, 0].astype(np.float64)/8**.25
    key = np.asarray(dense_trace["key"])[0, 0].astype(np.float64)/8**.25
    value = np.asarray(dense_trace["value"])[0, 0].astype(np.float64)
    legal = np.tri(len(query), dtype=bool)
    scores = query@key.T
    logits = np.where(legal, scores, -np.inf)
    expected = np.exp(logits-logits.max(-1, keepdims=True))
    expected /= expected.sum(-1, keepdims=True)
    reference = expected@value
    records = []
    for seed in range(8):
        projection = np.random.default_rng(seed).normal(size=(256, 8))
        for count in (16, 64, 256):
            q_log = query@projection[:count].T-(query**2).sum(-1, keepdims=True)/2
            k_log = key@projection[:count].T-(key**2).sum(-1, keepdims=True)/2
            # Query-wise scalar cancels; one scalar shared by all keys also cancels.
            q_phi = np.exp(q_log-q_log.max(-1, keepdims=True))/math.sqrt(count)
            k_phi = np.exp(k_log-k_log.max())/math.sqrt(count)
            kernel = np.where(legal, q_phi@k_phi.T, 0.)
            weights = kernel/kernel.sum(-1, keepdims=True)
            output = weights@value
            records.append({"seed": seed, "features": count,
                "relative_output_l2_error": float(np.linalg.norm(output-reference)/np.linalg.norm(reference)),
                "last_weights": weights[-1], "last_output": output[-1]})
    return {"query_scaled": query, "key_scaled": key, "value": value,
            "softmax_weights": expected, "softmax_output": reference, "trials": records}


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    points, pairs, split, duplicates = load_records()
    train_x, train_y = pairs["train"]
    matrix = np.column_stack((train_x.numpy().reshape(-1, 2), np.ones(220*44)))
    affine = np.linalg.lstsq(matrix, train_y.numpy().reshape(-1, 2), rcond=None)[0]
    baselines = {}
    for name, (x, y) in pairs.items():
        fitted = np.column_stack((x.numpy().reshape(-1, 2), np.ones(len(x)*44)))@affine
        baselines[name] = {"persistence_rmse": float((x-y).square().mean().sqrt())/2,
            "affine_rmse": float(np.sqrt(((fitted-y.numpy().reshape(-1, 2))**2).mean()))/2}
    report, model_assets = {}, {}
    for mode in ("dense", "window", "kernel"):
        model, record = train_model(mode, pairs)
        record["parameters"] = sum(p.numel() for p in model.parameters())
        report[mode] = record
        model_assets[mode] = {"weights": model.state_dict(), "worked_example": inspect(model, points),
            "fresh_gated_example": inspect(model, points, 27, 19, 1)}
        if mode == "kernel": report[mode]["gradient_equivalence"] = gradient_check(model, points[76:78, :7])
        print(mode, record["selected_update"], record["metrics_rmse_original"], flush=True)
    random_features = random_feature_study(serialize(model_assets["dense"]["worked_example"]["trace"]))
    summary = {"environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "torch": torch.__version__, "threads": 1},
        "split_source_rows": {k: [row+1 for row in rows] for k, rows in split.items()},
        "duplicate_source_rows": [[row+1 for row in rows] for rows in duplicates],
        "affine_weights_transformed": affine, "baselines": baselines, "models": report}
    for filename, contents in (("author-results.json", summary), ("forecast-models.json", model_assets),
                               ("random-feature-results.json", random_features)):
        (ROOT/filename).write_text(json.dumps(serialize(contents), separators=(",", ":")), encoding="utf-8")
    print("PASS: three declared fits, saved actual forecasts, causal/incremental controls, kernel derivatives, 24 random-feature trials.")


if __name__ == "__main__":
    main()
