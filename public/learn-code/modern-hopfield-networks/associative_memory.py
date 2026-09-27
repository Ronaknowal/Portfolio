"""Small exact Hopfield mechanisms; run directly to reproduce the lesson numbers."""
from pathlib import Path
import json
import numpy as np


def softmax(scores):
    exponentials = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def store_binary(patterns):
    weights = patterns.T @ patterns / patterns.shape[1]
    np.fill_diagonal(weights, 0)
    return weights


def binary_energy(weights, state):
    return float(-0.5 * state @ weights @ state)


def binary_recall(weights, cue, order=None, max_sweeps=20):
    state = np.array(cue, dtype=float)
    order = list(range(len(state))) if order is None else order
    trace = [{"state": state.tolist(), "energy": binary_energy(weights, state)}]
    for sweep in range(max_sweeps):
        changed = False
        for index in order:
            field = weights[index] @ state
            new_value = np.sign(field) if field != 0 else state[index]
            changed |= bool(new_value != state[index])
            state[index] = new_value
            trace.append({"sweep": sweep + 1, "coordinate": index + 1,
                          "field": float(field), "state": state.tolist(),
                          "energy": binary_energy(weights, state)})
        if not changed:
            break
    return state, trace


def retrieve(patterns, cue, beta):
    weights = softmax(beta * (patterns @ cue))
    return weights @ patterns, weights


def energy(patterns, cue, beta):
    logits = beta * (patterns @ cue)
    peak = logits.max()
    log_sum = peak + np.log(np.exp(logits - peak).sum())
    return float(0.5 * cue @ cue - log_sum / beta)


def iterate(patterns, cue, beta, steps=12):
    state = np.asarray(cue, dtype=float)
    trace = [{"state": state.tolist(), "energy": energy(patterns, state, beta)}]
    for _ in range(steps):
        new, weights = retrieve(patterns, state, beta)
        trace.append({"state": new.tolist(), "weights": weights.tolist(),
                      "energy": energy(patterns, new, beta),
                      "step_norm": float(np.linalg.norm(new - state))})
        state = new
    return trace


def main():
    pattern = np.array([[1, 1, -1, -1]], dtype=float)
    weights = store_binary(pattern)
    results = {"binary_weights": weights.tolist(), "binary": {}}
    for name, cue, order in [
        ("worked", [1, -1, -1, -1], None),
        ("fresh", [-1, 1, -1, -1], None),
        ("fixed_null", [1, 1, -1, -1], None),
        ("ambiguous_forward", [-1, -1, -1, -1], [0, 1, 2, 3]),
        ("ambiguous_reverse", [-1, -1, -1, -1], [3, 2, 1, 0]),
    ]:
        final, trace = binary_recall(weights, cue, order)
        results["binary"][name] = {"final": final.tolist(), "trace": trace}
        assert np.max(np.diff([step["energy"] for step in trace])) <= 1e-12
    pair = np.array([[1., 0.], [-1., 0.]])
    results["modern"] = {}
    for name, cue, beta in [
        ("worked_sharp", [0.2, 0.4], 2.),
        ("worked_broad", [0.2, 0.4], .5),
        ("fresh_sharp", [-0.35, 0.6], 2.),
        ("fresh_broad", [-0.35, 0.6], .5),
        ("symmetry_null", [0., .6], 2.),
        ("wrong_side", [-.2, 0.4], 8.),
    ]:
        trace = iterate(pair, cue, beta)
        results["modern"][name] = trace
        assert max(np.diff([step["energy"] for step in trace])) <= 1e-12
    duplicate_bank = np.array([[1., 0.], [1., 0.], [-1., 0.]])
    results["duplicate"] = {"output": retrieve(duplicate_bank, np.array([0., 0.]), 2.)[0].tolist(),
                            "weights": retrieve(duplicate_bank, np.array([0., 0.]), 2.)[1].tolist()}
    results["norm_trap"] = {"scores": (np.array([[1., 0.], [3., 1.]]) @ np.array([1., 0.])).tolist(),
                            "distances": [0., float(np.sqrt(5))]}
    results["mass_margin"] = {"bank_size": 100, "gap": 3., "beta": 2.,
                              "target_mass": float(1/(1+99*np.exp(-6))),
                              "error_bound_M1": float(2*99*np.exp(-6)/(1+99*np.exp(-6)))}
    # A distinct-value read has the same weights, but is not this energy's state update.
    scores = np.array([0.6, -0.2])
    attention_weights = softmax(scores)
    results["key_value"] = {"weights": attention_weights.tolist(),
                             "label_mass": attention_weights.tolist(),
                             "value_read": float(attention_weights @ np.array([10., -2.]))}
    # One gradient step learns an association; target is the second of two memories.
    q, key_matrix, target, rate = np.array([.2, -.1]), np.eye(2), 1, .1
    probabilities = softmax(key_matrix @ q)
    gradient = key_matrix.T @ (probabilities - np.eye(2)[target])
    updated = q - rate*gradient
    results["gradient"] = {"before": q.tolist(), "probabilities": probabilities.tolist(),
                           "loss": float(-np.log(probabilities[target])), "gradient": gradient.tolist(),
                           "after": updated.tolist(),
                           "loss_after": float(-np.log(softmax(key_matrix @ updated)[target]))}
    # Full-matrix attention parity (same mathematical operator, no bitwise claim).
    import torch
    import torch.nn.functional as functional
    torch.set_num_threads(1)
    bank = torch.tensor([[1., 0.], [0., 1.], [-1., .5]], dtype=torch.float64)
    query = torch.tensor([[[.3, -.2]]], dtype=torch.float64)
    manual = torch.softmax(query @ bank.T / np.sqrt(2), dim=-1) @ bank
    native = functional.scaled_dot_product_attention(query, bank[None], bank[None], dropout_p=0.)
    results["attention_parity"] = {"manual": manual.tolist(),
                                  "maximum_error": float((manual-native).abs().max())}
    # Higher-order energy on an original parity memory: clamped inputs, unknown output.
    triples = np.array([[-1,-1,-1],[-1,1,1],[1,-1,1],[1,1,-1]], float)
    results["polynomial_parity"] = {str(power): [{"inputs": [a,b],
        "energy_output_minus": float(-np.sum((triples@np.array([a,b,-1]))**power)),
        "energy_output_plus": float(-np.sum((triples@np.array([a,b,1]))**power))}
        for a,b in [(-1,-1),(-1,1),(1,-1),(1,1)]] for power in [2,3]}
    # Two different finite-memory criteria, not a capacity theorem fit.
    rng = np.random.default_rng(37)
    results["finite_scan"] = []
    for count in [2, 6, 10, 16, 24]:
        exact_fixed = recalled = trials = 0
        for _ in range(8):
            patterns = rng.choice([-1.,1.], (count,64))
            matrix = store_binary(patterns)
            for row in patterns:
                fields = matrix @ row
                exact_fixed += int(np.all(fields * row >= 0))
                noisy = row.copy()
                noisy[rng.choice(64,6,replace=False)] *= -1
                result, _ = binary_recall(matrix, noisy)
                recalled += int(np.array_equal(result, row))
                trials += 1
        results["finite_scan"].append({"patterns": count, "trials": trials,
                                       "exact_fixed": exact_fixed, "exact_recalled": recalled})
    Path(__file__).with_name("mechanism-results.json").write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps({key: results[key] for key in ["mass_margin","gradient","attention_parity","finite_scan"]}, indent=2))


if __name__ == "__main__":
    main()
