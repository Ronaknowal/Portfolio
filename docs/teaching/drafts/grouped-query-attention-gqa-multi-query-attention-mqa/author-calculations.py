"""Complete CPU causal forecasting and MHA -> GQA/MQA conversion study.

Run beside movement_libras.data using Python, NumPy and PyTorch.
Writes author-results.json and forecast-models.json. No network or GPU required.
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
    assert 76 == next(row for row in split["test"] if raw[row, 90] == 4)
    points = torch.tensor(raw[:, :90].reshape(-1, 45, 2) * 2 - 1,
                          dtype=torch.float32)
    pairs = {name: (points[rows, :-1], points[rows, 1:])
             for name, rows in split.items()}
    return raw, points, pairs, split, [rows for rows in groups.values() if len(rows) > 1]


def rotate(values, positions):
    width = values.shape[-1]
    frequencies = 10000.0 ** (-torch.arange(0, width, 2,
                           dtype=values.dtype, device=values.device) / width)
    angle = positions.to(values.dtype)[None, None, :, None] * frequencies
    even, odd = values[..., 0::2], values[..., 1::2]
    return torch.stack((even * angle.cos() - odd * angle.sin(),
                        even * angle.sin() + odd * angle.cos()), -1).flatten(-2)


class CausalForecaster(nn.Module):
    def __init__(self, kv_heads=4):
        super().__init__()
        if kv_heads not in (1, 2, 4):
            raise ValueError("This four-query-head study supports 1, 2 or 4 KV heads.")
        self.kv_heads, self.query_heads, self.head_width = kv_heads, 4, 6
        self.stem = nn.Linear(2, 24)
        self.norm_attention = nn.LayerNorm(24, eps=1e-5)
        self.query = nn.Linear(24, 24)
        self.key = nn.Linear(24, kv_heads * 6)
        self.value = nn.Linear(24, kv_heads * 6)
        self.output = nn.Linear(24, 24)
        self.norm_feedforward = nn.LayerNorm(24, eps=1e-5)
        self.feedforward = nn.Sequential(nn.Linear(24, 48), nn.GELU(), nn.Linear(48, 24))
        self.norm_final = nn.LayerNorm(24, eps=1e-5)
        self.forecast = nn.Linear(24, 2)

    def forward(self, points, positions=None, cache=None, return_trace=False):
        batch, length, _ = points.shape
        if length < 1:
            raise ValueError("Supply at least one observed point.")
        if positions is None:
            start = 0 if cache is None else int(cache[2][-1]) + 1
            positions = torch.arange(start, start + length, device=points.device)
        if positions.ndim != 1 or len(positions) != length:
            raise ValueError("Use one logical position for each input point.")
        state = self.stem(points)
        normalized = self.norm_attention(state)
        query = self.query(normalized).view(batch, length, 4, 6).transpose(1, 2)
        keys = self.key(normalized).view(batch, length, self.kv_heads, 6).transpose(1, 2)
        values = self.value(normalized).view(batch, length, self.kv_heads, 6).transpose(1, 2)
        raw_query, raw_keys = query, keys
        query, keys = rotate(query, positions), rotate(keys, positions)
        key_positions = positions
        if cache is not None:
            old_keys, old_values, old_positions = cache
            keys = torch.cat((old_keys, keys), dim=2)
            values = torch.cat((old_values, values), dim=2)
            key_positions = torch.cat((old_positions, positions))
        next_cache = (keys, values, key_positions)
        group_size = self.query_heads // self.kv_heads
        grouped_query = query.reshape(batch, self.kv_heads, group_size, length, 6)
        scores = torch.einsum("bgrtd,bgud->bgrtu", grouped_query, keys) / math.sqrt(6)
        legal = key_positions[None, :] <= positions[:, None]
        if not bool(legal.any(-1).all()):
            raise ValueError("Every query must have a legal key.")
        scores = scores.masked_fill(~legal[None, None, None], -torch.inf)
        attention = torch.softmax(scores, -1)
        heads = torch.einsum("bgrtu,bgud->bgrtd", attention, values)
        mixed = heads.reshape(batch, 4, length, 6).transpose(1, 2).reshape(batch, length, 24)
        state = state + self.output(mixed)
        state = state + self.feedforward(self.norm_feedforward(state))
        prediction = self.forecast(self.norm_final(state))
        if return_trace:
            trace = {"raw_query": raw_query, "raw_keys_new": raw_keys,
                     "rotated_query": query, "cached_keys": keys, "cached_values": values,
                     "attention": attention.reshape(batch, 4, length, -1),
                     "head_outputs": heads.reshape(batch, 4, length, 6),
                     "prediction_transformed": prediction}
            return prediction, next_cache, trace
        return prediction, next_cache


def convert(parent, kv_heads):
    if parent.kv_heads != parent.query_heads:
        raise ValueError("Conversion reference must be the MHA parent.")
    if kv_heads == parent.kv_heads:
        return copy.deepcopy(parent)
    result = CausalForecaster(kv_heads)
    source = parent.state_dict()
    target = result.state_dict()
    group_size = parent.query_heads // kv_heads
    for name in target:
        if name in ("key.weight", "value.weight"):
            target[name] = source[name].reshape(kv_heads, group_size, 6, 24).mean(1).reshape(kv_heads * 6, 24)
        elif name in ("key.bias", "value.bias"):
            target[name] = source[name].reshape(kv_heads, group_size, 6).mean(1).reshape(kv_heads * 6)
        else:
            target[name] = source[name].clone()
    result.load_state_dict(target)
    return result


@torch.no_grad()
def metrics(model, pairs):
    model.eval()
    result = {}
    for name, (inputs, targets) in pairs.items():
        predicted, _ = model(inputs)
        mse = float((predicted - targets).square().mean())
        result[name] = {"mse_transformed": mse, "rmse_original_coordinate": math.sqrt(mse) / 2}
    return result


def train(model, pairs, steps, learning_rate, include_initial):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    training_inputs, training_targets = pairs["train"]
    validation_inputs, validation_targets = pairs["validation"]
    best_loss, best_state, best_step, history = math.inf, None, None, []
    for step in range(steps + 1):
        if step > 0:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            predictions, _ = model(training_inputs)
            loss = (predictions - training_targets).square().mean()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            predicted, _ = model(validation_inputs)
            validation_loss = float((predicted - validation_targets).square().mean())
        history.append({"step": step, "validation_mse_transformed": validation_loss})
        if (include_initial or step > 0) and validation_loss < best_loss:
            best_loss, best_step = validation_loss, step
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return {"selected_step": best_step, "history": history, "metrics": metrics(model, pairs)}


def as_lists(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: as_lists(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [as_lists(item) for item in value]
    return value


@torch.no_grad()
def inspect_forecast(model, points):
    original = points[76:77, :32]
    baseline, cache, trace = model(original, return_trace=True)
    incremental, step_cache = [], None
    for position in range(32):
        output, step_cache = model(original[:, position:position+1],
                                   torch.tensor([position]), step_cache)
        incremental.append(output)
    incremental = torch.cat(incremental, 1)
    shifted, _ = model(original, torch.arange(100, 132))
    changed = original.clone()
    changed[:, 23, 0] *= -1  # Equivalent to x -> 1-x in original coordinates.
    edited, _ = model(changed)
    # Feed predictions back; do not read future ground-truth points.
    rollout = [baseline[:, -1:]]
    rollout_cache = cache
    for index in range(4):
        predicted, rollout_cache = model(rollout[-1], torch.tensor([32 + index]), rollout_cache)
        rollout.append(predicted)
    actual_cache_bytes = (cache[0].numel() + cache[1].numel()) * cache[0].element_size()
    return as_lists({
        "source_row": 77, "observed_prefix_original_coordinates": (original + 1) / 2,
        "next_true_point_original_coordinates": (points[76, 32] + 1) / 2,
        "next_prediction_original_coordinates": (baseline[0, -1] + 1) / 2,
        "edited_prediction_original_coordinates": (edited[0, -1] + 1) / 2,
        "edited_max_prefix_prediction_change": (edited-baseline).abs().max(),
        "incremental_max_error": (incremental-baseline).abs().max(),
        "common_position_shift_max_error": (shifted-baseline).abs().max(),
        "rollout_original_coordinates": (torch.cat(rollout, 1) + 1) / 2,
        "cache_shape": list(cache[0].shape), "cache_float32_bytes": actual_cache_bytes,
        "trace": trace})


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(101)
    raw, points, pairs, split, duplicates = load_records()
    training_inputs, training_targets = pairs["train"]
    design_matrix = np.column_stack((training_inputs.numpy().reshape(-1, 2), np.ones(220 * 44)))
    affine_weights = np.linalg.lstsq(design_matrix, training_targets.numpy().reshape(-1, 2), rcond=None)[0]
    baselines = {}
    for name, (inputs, targets) in pairs.items():
        affine = np.column_stack((inputs.numpy().reshape(-1, 2), np.ones(inputs.shape[0] * 44))) @ affine_weights
        baselines[name] = {
            "persistence_rmse_original_coordinate": float((inputs-targets).square().mean().sqrt())/2,
            "affine_rmse_original_coordinate": float(np.sqrt(np.mean((affine-targets.numpy().reshape(-1, 2))**2)))/2}
    parent = CausalForecaster(4)
    parent_record = train(parent, pairs, 180, .003, False)
    print("parent", parent_record["selected_step"], parent_record["metrics"]["test"], flush=True)
    records, models = {}, {}
    for kv_heads in (4, 2, 1):
        branch = convert(parent, kv_heads)
        initial = metrics(branch, pairs)
        record = train(branch, pairs, 45, .001, True)
        record.update({"zero_update_metrics": initial,
                       "parameters": sum(parameter.numel() for parameter in branch.parameters())})
        records[str(kv_heads)] = record
        models[str(kv_heads)] = {"weights": as_lists(branch.state_dict()),
                                "investigation": inspect_forecast(branch, points)}
        print("kv_heads", kv_heads, "selected", record["selected_step"],
              "test", record["metrics"]["test"], flush=True)
    report = {"environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "torch": torch.__version__, "cpu_threads": 1},
              "split_source_rows": {name: [row + 1 for row in rows] for name, rows in split.items()},
              "duplicate_source_rows": [[row + 1 for row in rows] for rows in duplicates],
              "baselines": baselines, "affine_weights_transformed": affine_weights.tolist(),
              "parent": parent_record, "branches": records}
    (ROOT/"author-results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (ROOT/"forecast-models.json").write_text(json.dumps(models, separators=(",", ":")), encoding="utf-8")


if __name__ == "__main__":
    main()
