"""Exact teaching oracles for causal convolution, gates and compact recurrences."""
from pathlib import Path
import json
import numpy as np
import torch
from scipy.linalg import hankel

ROOT = Path(__file__).parent


def direct(values, kernel):
    return np.convolve(values, kernel)[:len(values)]


def fft_convolution(values, kernel, size=None):
    if size is None:
        size = 1 << (len(values) + len(kernel) - 2).bit_length()
    return np.fft.irfft(np.fft.rfft(values, size) * np.fft.rfft(kernel, size), size)[:len(values)]


def toeplitz(kernel, length):
    return np.array([[kernel[t - j] if 0 <= t - j < len(kernel) else 0.
                      for j in range(length)] for t in range(length)])


def gated(values, kernel, query, key):
    matrix = np.diag(query) @ toeplitz(kernel, len(values)) @ np.diag(key)
    return matrix, matrix @ values


def modal(values, residues, poles, initial=None):
    state = np.zeros(len(poles)) if initial is None else np.array(initial, dtype=float)
    outputs, states = [], []
    for value in values:
        state = poles * state + value
        outputs.append(float(residues @ state))
        states.append(state.copy())
    return np.array(outputs), np.array(states)


def overlap_add(values, kernel, block_size):
    result = np.zeros(len(values) + len(kernel) - 1)
    for start in range(0, len(values), block_size):
        part = np.convolve(values[start:start + block_size], kernel)
        result[start:start + len(part)] += part
    return result[:len(values)]


def main():
    report = {}
    for name, values, kernel in [("worked", [1, 2, 3, 4], [1, .5, .25, .125]),
                                 ("fresh", [2, -1, 3, 0, 1], [.5, 1, -.25])]:
        values, kernel = np.array(values, float), np.array(kernel, float)
        changed = values.copy(); changed[-1] += 4
        report[name] = {"values": values.tolist(), "kernel": kernel.tolist(),
                        "direct": direct(values, kernel).tolist(),
                        "fft": fft_convolution(values, kernel).tolist(),
                        "circular": fft_convolution(values, kernel, len(values)).tolist(),
                        "changed_last": direct(changed, kernel).tolist(),
                        "changed_last_circular": fft_convolution(changed, kernel, len(values)).tolist(),
                        "matrix": toeplitz(kernel, len(values)).tolist()}
        assert np.allclose(direct(values, kernel), fft_convolution(values, kernel), atol=1e-12)
        assert np.allclose(direct(values, kernel), overlap_add(values, kernel, 2), atol=1e-12)
        assert np.allclose(direct(values, [1]), values)
    for name, values, kernel, query, key in [
            ("gates_worked", [1, 2, 3, 4], [1, .5, .25, .125], [1, 2, -1, .5], [1, 0, -1, 2]),
            ("gates_fresh", [2, -1, 3, 1], [1, -.25, .5, 0], [1, -.5, 2, 1], [.5, 1, 0, -1])]:
        matrix, result = gated(values, kernel, query, key)
        opened = np.array(key, float); opened[2] = 1
        changed_matrix, changed = gated(values, kernel, query, opened)
        phi = [1, -.5, 0, 0]
        hierarchy = np.diag(query) @ toeplitz(kernel, 4) @ np.diag(key) @ toeplitz(phi, 4)
        report[name] = {"values": values, "kernel": kernel, "query": query, "key": key,
                       "matrix": matrix.tolist(), "output": result.tolist(),
                       "opened_key_2": changed.tolist(), "hierarchy_phi": phi,
                       "hierarchy_matrix": hierarchy.tolist(), "hierarchy_output": (hierarchy @ values).tolist()}
        assert np.allclose(gated(np.zeros(4), kernel, query, key)[1], 0)
        assert np.allclose(gated(values, kernel, np.ones(4), np.ones(4))[1], direct(values, kernel))
    # One visible loss and update, independently checked by autograd.
    kernel = torch.tensor([.5, .25], dtype=torch.float64, requires_grad=True)
    prediction = 2 * kernel[0] + kernel[1]
    loss = .5 * (prediction - 2) ** 2
    loss.backward()
    new_kernel = kernel.detach() - .1 * kernel.grad
    report["gradient"] = {"prediction": prediction.item(), "loss": loss.item(),
                           "gradient": kernel.grad.tolist(), "new_kernel": new_kernel.tolist(),
                           "new_prediction": (2 * new_kernel[0] + new_kernel[1]).item()}
    # Fixed coordinates retain the filter; rescaling by the active prefix does not.
    position = np.arange(8)
    fixed = np.exp(-position / 7) * np.cos(2 * np.pi * position / 7)
    bad_prefix = np.exp(-np.arange(4) / 3) * np.cos(2 * np.pi * np.arange(4) / 3)
    report["coordinates"] = {"reference_8_filter": fixed.tolist(), "bad_length_4_filter": bad_prefix.tolist(),
                             "first_four_error": float(np.max(np.abs(fixed[:4] - bad_prefix)))}
    values = np.array([1, -2, .5, 3, -1, 2.])
    residues, poles = np.array([.6, .4]), np.array([.5, -.25])
    kernel = (residues[:, None] * poles[:, None] ** np.arange(len(values))).sum(axis=0)
    outputs, states = modal(values, residues, poles)
    full_error = np.max(np.abs(outputs - direct(values, kernel)))
    prefix, prefix_states = modal(values[:3], residues, poles)
    suffix, _ = modal(values[3:], residues, poles, prefix_states[-1])
    reset_suffix, _ = modal(values[3:], residues, poles)
    truncated = kernel.copy(); truncated[2:] = 0
    error_bound = np.abs(values).max() * np.abs(kernel - truncated).sum()
    fresh = np.array([2, 0, -1, 3, 1, -.5])
    fresh_out, fresh_states = modal(fresh, residues, poles)
    fresh_reset, _ = modal(fresh[3:], residues, poles)
    long_kernel = (residues[:, None] * poles[:, None] ** np.arange(12)).sum(axis=0)
    singular_values = np.linalg.svd(hankel(long_kernel[:6], long_kernel[5:11]), compute_uv=False)
    report["stream"] = {"values": values.tolist(), "residues": residues.tolist(), "poles": poles.tolist(),
                         "kernel": kernel.tolist(), "outputs": outputs.tolist(), "states": states.tolist(),
                         "convolution_error": float(full_error), "carried_suffix": suffix.tolist(),
                         "reset_suffix": reset_suffix.tolist(), "truncated_output": direct(values, truncated).tolist(),
                         "observed_truncation_error": float(np.abs(outputs - direct(values, truncated)).max()),
                         "error_bound": float(error_bound), "hankel_singular_values": singular_values.tolist(),
                         "fresh_values": fresh.tolist(), "fresh_outputs": fresh_out.tolist(),
                         "fresh_states": fresh_states.tolist(), "fresh_reset_suffix": fresh_reset.tolist()}
    assert full_error < 1e-12
    assert np.allclose(np.r_[prefix, suffix], outputs)
    assert np.abs(outputs - direct(values, truncated)).max() <= error_bound + 1e-12
    assert np.allclose(modal(np.zeros(6), residues, poles)[0], 0)
    report["copy"] = {"input": [4, 7, 2, 0, 0, 0], "filter": [0, 0, 0, 1],
                       "output": direct([4, 7, 2, 0, 0, 0], [0, 0, 0, 1]).tolist()}
    (ROOT / "mechanism-results.json").write_text(json.dumps(report, indent=2), encoding="utf8")
    print(json.dumps({key: report[key] for key in ["worked", "fresh", "gates_fresh", "gradient", "coordinates", "stream"]}, indent=2))


if __name__ == "__main__":
    main()
