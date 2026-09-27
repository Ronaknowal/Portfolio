"""Complete CPU MLA forecasting, algebraic equivalence and cache-rank study.

Run beside movement_libras.data with Python, NumPy and PyTorch.
Writes author-results.json and forecast-model.json; no network or GPU is needed.
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
    duplicates = [rows for rows in groups.values() if len(rows) > 1]
    return points, pairs, split, duplicates


def rotate(values, positions):
    """Input [B,H,T,R], adjacent coordinate pairs, ordinary RoPE."""
    width = values.shape[-1]
    if width < 2 or width % 2:
        raise ValueError("Rotary width must be a positive even integer.")
    frequency = 10000.0 ** (-torch.arange(0, width, 2, dtype=values.dtype,
                                          device=values.device) / width)
    angle = positions.to(values.dtype)[None, None, :, None] * frequency
    even, odd = values[..., 0::2], values[..., 1::2]
    return torch.stack((even * angle.cos() - odd * angle.sin(),
                        even * angle.sin() + odd * angle.cos()), -1).flatten(-2)


class LatentRMSNorm(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(width))

    def forward(self, values):
        return values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-6) * self.scale


class LatentForecaster(nn.Module):
    """One complete pre-norm block; no hidden model download or runtime."""
    def __init__(self):
        super().__init__()
        self.heads, self.content_width, self.value_width = 4, 4, 4
        self.rotary_width, self.latent_width, self.query_latent_width = 2, 8, 12
        self.stem = nn.Linear(2, 24)
        self.norm_attention = nn.LayerNorm(24, eps=1e-5)
        self.kv_down = nn.Linear(24, 8, bias=False)
        self.kv_norm = LatentRMSNorm(8)
        self.key_up = nn.Linear(8, 16, bias=False)
        self.value_up = nn.Linear(8, 16, bias=False)
        self.rotary_key = nn.Linear(24, 2, bias=False)
        self.query_down = nn.Linear(24, 12, bias=False)
        self.query_norm = LatentRMSNorm(12)
        self.query_content = nn.Linear(12, 16, bias=False)
        self.query_rotary = nn.Linear(12, 8, bias=False)
        self.output = nn.Linear(16, 24, bias=False)
        self.norm_feedforward = nn.LayerNorm(24, eps=1e-5)
        self.feedforward = nn.Sequential(nn.Linear(24, 48), nn.GELU(), nn.Linear(48, 24))
        self.norm_final = nn.LayerNorm(24, eps=1e-5)
        self.forecast = nn.Linear(24, 2)

    def forward(self, points, positions=None, cache=None, mode="absorbed",
                basis=None, wrong_scale=False, return_trace=False):
        """basis [8,r] is applied AFTER the original latent RMSNorm.

        Cache is (projected normalized latent, rotated shared key, logical IDs).
        Its validity includes model, basis, input prefix and position convention.
        """
        if mode not in ("absorbed", "expanded"):
            raise ValueError("Select expanded or absorbed attention.")
        batch, length, _ = points.shape
        if length < 1:
            raise ValueError("Supply at least one observed point.")
        if positions is None:
            start = 0 if cache is None else int(cache[2][-1]) + 1
            positions = torch.arange(start, start + length, device=points.device)
        if positions.ndim != 1 or len(positions) != length:
            raise ValueError("Use one logical position per input point.")
        state = self.stem(points)
        normalized = self.norm_attention(state)
        full_latent = self.kv_norm(self.kv_down(normalized))
        latent = full_latent if basis is None else full_latent @ basis
        query_latent = self.query_norm(self.query_down(normalized))
        content_query = self.query_content(query_latent).view(batch, length, 4, 4).transpose(1, 2)
        rotary_query_raw = self.query_rotary(query_latent).view(batch, length, 4, 2).transpose(1, 2)
        rotary_query = rotate(rotary_query_raw, positions)
        rotary_key_raw = self.rotary_key(normalized)
        rotary_key = rotate(rotary_key_raw[:, None], positions)[:, 0]
        key_up = self.key_up.weight.view(4, 4, 8)
        value_up = self.value_up.weight.view(4, 4, 8)
        if basis is not None:
            key_up = key_up @ basis
            value_up = value_up @ basis
        key_positions = positions
        if cache is not None:
            latent = torch.cat((cache[0], latent), 1)
            rotary_key = torch.cat((cache[1], rotary_key), 1)
            key_positions = torch.cat((cache[2], positions))
        next_cache = (latent, rotary_key, key_positions)
        effective_query = torch.einsum("bhtd,hdc->bhtc", content_query, key_up)
        if mode == "expanded":
            content_keys = torch.einsum("bsc,hdc->bhsd", latent, key_up)
            values = torch.einsum("bsc,hdc->bhsd", latent, value_up)
            content_scores = torch.einsum("bhtd,bhsd->bhts", content_query, content_keys)
        else:
            content_scores = torch.einsum("bhtc,bsc->bhts", effective_query, latent)
        rotary_scores = torch.einsum("bhtr,bsr->bhts", rotary_query, rotary_key)
        divisor = math.sqrt((latent.shape[-1] if wrong_scale else 4) + 2)
        legal = key_positions[None, :] <= positions[:, None]
        if not bool(legal.any(-1).all()):
            raise ValueError("Every query must have at least one legal key.")
        scores = ((content_scores + rotary_scores) / divisor).masked_fill(
            ~legal[None, None], -torch.inf)
        attention = torch.softmax(scores, -1)
        latent_context = torch.einsum("bhts,bsc->bhtc", attention, latent)
        if mode == "expanded":
            heads = torch.einsum("bhts,bhsd->bhtd", attention, values)
        else:
            heads = torch.einsum("bhtc,hdc->bhtd", latent_context, value_up)
        mixed = heads.transpose(1, 2).reshape(batch, length, 16)
        state = state + self.output(mixed)
        state = state + self.feedforward(self.norm_feedforward(state))
        prediction = self.forecast(self.norm_final(state))
        if return_trace:
            trace = {"full_normalized_latent_new": full_latent, "cached_latent": latent,
                     "query_latent": query_latent, "content_query": content_query,
                     "effective_query": effective_query, "rotary_query": rotary_query,
                     "cached_rotary_key": rotary_key, "content_scores": content_scores,
                     "rotary_scores": rotary_scores, "attention": attention,
                     "latent_context": latent_context, "head_outputs": heads,
                     "prediction_transformed": prediction}
            return prediction, next_cache, trace
        return prediction, next_cache


@torch.no_grad()
def metrics(model, pairs, **options):
    result = {}
    for name, (inputs, targets) in pairs.items():
        predicted, _ = model(inputs, **options)
        mse = float((predicted-targets).square().mean())
        result[name] = {"mse_transformed": mse, "rmse_original_coordinate": math.sqrt(mse)/2}
    return result


def train(model, pairs):
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    inputs, targets = pairs["train"]
    validation_inputs, validation_targets = pairs["validation"]
    best_loss, best_state, best_step, history = math.inf, None, None, []
    for step in range(1, 201):
        optimizer.zero_grad(set_to_none=True)
        prediction, _ = model(inputs, mode="expanded")
        loss = (prediction-targets).square().mean()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            validation, _ = model(validation_inputs, mode="expanded")
            validation_loss = float((validation-validation_targets).square().mean())
        history.append({"step": step, "validation_mse_transformed": validation_loss})
        if validation_loss < best_loss:
            best_loss, best_step = validation_loss, step
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return {"selected_step": best_step, "history": history}


def as_lists(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: as_lists(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_lists(item) for item in value]
    return value


def gradient_equivalence(model, sample):
    reference = copy.deepcopy(model).double()
    absorbed = copy.deepcopy(model).double()
    expanded_output, _ = reference(sample.double(), mode="expanded")
    absorbed_output, _ = absorbed(sample.double(), mode="absorbed")
    expanded_output.square().sum().backward()
    absorbed_output.square().sum().backward()
    gradient_errors = {}
    for (name, first), (other_name, second) in zip(reference.named_parameters(), absorbed.named_parameters()):
        assert name == other_name
        assert first.grad is not None and second.grad is not None
        error = float((first.grad-second.grad).abs().max())
        gradient_errors[name] = error
        assert torch.allclose(first.grad, second.grad, atol=1e-9, rtol=1e-9), name
    return {"output_max_error": float((expanded_output-absorbed_output).detach().abs().max()),
            "all_parameter_gradient_max_error": max(gradient_errors.values()),
            "parameter_gradient_errors": gradient_errors}


@torch.no_grad()
def inspect(model, points, basis=None):
    prefix = points[76:77, :32]
    original, cache, trace = model(prefix, basis=basis, return_trace=True)
    expanded, _ = model(prefix, mode="expanded", basis=basis)
    incremental, incremental_cache = [], None
    for position in range(32):
        prediction, incremental_cache = model(prefix[:, position:position+1],
            torch.tensor([position]), incremental_cache, basis=basis)
        incremental.append(prediction)
    incremental = torch.cat(incremental, 1)
    shifted, _ = model(prefix, torch.arange(100, 132), basis=basis)
    changed = prefix.clone()
    changed[:, 23, 0] *= -1
    edited, _ = model(changed, basis=basis)
    wrong, _ = model(prefix, basis=basis, wrong_scale=True)
    assert torch.allclose(original, expanded, atol=2e-6, rtol=2e-6)
    assert torch.allclose(original, incremental, atol=2e-6, rtol=2e-6)
    assert torch.equal(original[:, :23], edited[:, :23])
    return as_lists({"source_row": 77, "observed_points": (prefix+1)/2,
        "true_next_point": (points[76, 32]+1)/2,
        "next_prediction": (original[0, -1]+1)/2,
        "edited_next_prediction": (edited[0, -1]+1)/2,
        "wrong_scale_next_prediction": (wrong[0, -1]+1)/2,
        "wrong_scale_max_error_transformed": (wrong-original).abs().max(),
        "expanded_absorbed_max_error_transformed": (expanded-original).abs().max(),
        "incremental_max_error_transformed": (incremental-original).abs().max(),
        "shift_max_error_transformed": (shifted-original).abs().max(),
        "earlier_than_edit_max_error": (edited[:, :23]-original[:, :23]).abs().max(),
        "cache_shapes": [list(cache[0].shape), list(cache[1].shape)],
        "cache_float32_bytes": (cache[0].numel()+cache[1].numel())*4,
        "trace": trace})


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(131)
    points, pairs, split, duplicates = load_records()
    train_inputs, train_targets = pairs["train"]
    design_matrix = np.column_stack((train_inputs.numpy().reshape(-1, 2), np.ones(220*44)))
    affine_weights = np.linalg.lstsq(design_matrix, train_targets.numpy().reshape(-1, 2), rcond=None)[0]
    baselines = {}
    for name, (inputs, targets) in pairs.items():
        affine = np.column_stack((inputs.numpy().reshape(-1, 2), np.ones(inputs.shape[0]*44))) @ affine_weights
        baselines[name] = {
            "persistence_rmse_original_coordinate": float((inputs-targets).square().mean().sqrt())/2,
            "affine_rmse_original_coordinate": float(np.sqrt(np.mean((affine-targets.numpy().reshape(-1, 2))**2)))/2}
    model = LatentForecaster()
    training = train(model, pairs)
    model.eval()
    joint = torch.cat((model.key_up.weight.detach(), model.value_up.weight.detach()), 0)
    _, singular, right_transpose = torch.linalg.svd(joint, full_matrices=False)
    complete_basis, reduced_basis = right_transpose.T, right_transpose[:4].T
    with torch.no_grad():
        prediction, _ = model(pairs["validation"][0])
        relabeled, _ = model(pairs["validation"][0], basis=complete_basis)
    assert torch.allclose(prediction, relabeled, atol=3e-6, rtol=3e-6)
    reconstruction = joint @ reduced_basis @ reduced_basis.T
    derivatives = gradient_equivalence(model, points[76:78, :7])
    report = {"environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "torch": torch.__version__, "cpu_threads": 1},
        "split_source_rows": {name: [row+1 for row in rows] for name, rows in split.items()},
        "duplicate_source_rows": [[row+1 for row in rows] for rows in duplicates],
        "baselines": baselines, "affine_weights_transformed": affine_weights,
        "training": training, "parameters": sum(p.numel() for p in model.parameters()),
        "full_metrics": metrics(model, pairs), "rank4_intervention_metrics": metrics(model, pairs, basis=reduced_basis),
        "singular_values_joint_up_projection": singular,
        "rank4_frobenius_squared_error": (joint-reconstruction).square().sum(),
        "discarded_singular_squared_sum": singular[4:].square().sum(),
        "full_basis_validation_max_error_transformed": (prediction-relabeled).abs().max(),
        "gradient_equivalence": derivatives}
    models = {"weights": model.state_dict(), "complete_basis": complete_basis,
              "rank4_basis": reduced_basis, "full_investigation": inspect(model, points),
              "rank4_investigation": inspect(model, points, reduced_basis)}
    (ROOT/"author-results.json").write_text(json.dumps(as_lists(report), indent=2), encoding="utf-8")
    (ROOT/"forecast-model.json").write_text(json.dumps(as_lists(models), separators=(",", ":")), encoding="utf-8")
    print(json.dumps(as_lists({"selected_step": training["selected_step"], "parameters": report["parameters"],
        "full_test": report["full_metrics"]["test"], "rank4_test": report["rank4_intervention_metrics"]["test"],
        "gradient_equivalence": {k:v for k,v in derivatives.items() if k != "parameter_gradient_errors"}}), indent=2))


if __name__ == "__main__":
    main()
