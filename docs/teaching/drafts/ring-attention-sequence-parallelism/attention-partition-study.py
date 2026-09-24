"""Bounded author calculations: exact partition identities, gradients and frozen real input.
No fitting, GPU communication, runtime/memory measurement or performance benchmark.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import numpy as np
import torch

DIRECTORY = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ring_reference", DIRECTORY / "ring-attention-reference.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def maximum_error(first, second):
    return float(np.abs(first - second).max())


def model_inference(points, packet, ranks=4, direction=1):
    weights = {key: np.asarray(value, dtype=np.float64) for key, value in packet["state_dict"].items()}
    features = np.tanh((2 * points - 1) @ weights["stem.weight"].T + weights["stem.bias"])
    length = len(points)
    projections = [(features @ weights[f"mixer.{name}.weight"].T).reshape(length, 2, 12).transpose(1, 0, 2)
                   for name in ("query", "key", "value")]
    allowed = np.ones((length, length), dtype=bool)
    dense, attention = reference.dense_attention(*projections, allowed)
    partition = list(np.array_split(np.arange(length), ranks))
    ring, _, trace = reference.ring_attention(*projections, partition, allowed, direction)
    probabilities = []
    logits = []
    for values in (dense, ring):
        mixed = values.transpose(1, 0, 2).reshape(length, 24) @ weights["mixer.output.weight"].T
        score = (features + mixed).mean(axis=0) @ weights["classifier.weight"].T + weights["classifier.bias"]
        probability = np.exp(score - score.max()); probability /= probability.sum()
        probabilities.append(probability)
        logits.append(score)
    return {"points": points, "query": projections[0], "key": projections[1], "value": projections[2],
            "dense_output": dense, "attention_weights": attention, "probabilities": probabilities[0],
            "predicted_class_one_based": int(np.argmax(probabilities[0]) + 1),
            "output_error": maximum_error(dense, ring), "logit_error": maximum_error(*logits),
            "probability_error": maximum_error(*probabilities), "ownership": partition, "trace": trace}


def scalar_online(scores, values, blocks):
    maximum, total, numerator = -np.inf, 0.0, 0.0
    states = []
    for ids in blocks:
        new_maximum = max(maximum, max(scores[i] for i in ids))
        correction = np.exp(maximum - new_maximum)
        exponentials = np.exp(np.asarray(scores)[ids] - new_maximum)
        total = correction * total + exponentials.sum()
        numerator = correction * numerator + exponentials @ np.asarray(values)[ids]
        maximum = new_maximum
        states.append({"maximum": maximum, "total": total, "numerator": numerator,
                       "correction": correction, "output": numerator / total})
    return states


def main():
    torch.set_num_threads(1)
    rng = np.random.default_rng(91)
    arrays = [rng.normal(size=(2, 7, 3)) for _ in range(3)]
    partitions = list(np.array_split(np.arange(7), 3))
    positions = np.arange(7)
    causal = positions[None, :] <= positions[:, None]
    document = np.array([0, 0, 0, 1, 1, 1, 1])
    packed = causal & (document[:, None] == document[None, :])
    masks = {"dense": np.ones((7, 7), bool), "causal": causal, "packed_causal": packed,
             "empty_query": packed.copy()}
    masks["empty_query"][3] = False
    identities = []
    gradient_checks = []
    upstream = rng.normal(size=(2, 7, 3))
    for name, mask in masks.items():
        dense, probability = reference.dense_attention(*arrays, mask)
        ring, lse, _ = reference.ring_attention(*arrays, partitions, mask)
        reverse, _, _ = reference.ring_attention(*arrays, partitions, mask, -1)
        renamed, _, _ = reference.ring_attention(*arrays, [partitions[i] for i in [2, 0, 1]], mask)
        np.testing.assert_allclose(ring, dense, atol=1e-12, rtol=1e-12)
        identities.append({"mask": name, "ring_error": maximum_error(ring, dense),
                           "reverse_error": maximum_error(reverse, dense), "renamed_rank_error": maximum_error(renamed, dense)})
        gradients = reference.blockwise_backward(*arrays, partitions, mask, ring, lse, upstream)
        # Direct dense derivative and a separately implemented Torch automatic derivative.
        direct_score = probability * (upstream @ arrays[2].swapaxes(-1, -2) -
                                     (upstream * dense).sum(axis=-1, keepdims=True))
        direct = [direct_score @ arrays[1] / np.sqrt(3),
                  direct_score.swapaxes(-1, -2) @ arrays[0] / np.sqrt(3),
                  probability.swapaxes(-1, -2) @ upstream]
        finite_difference = []
        for which, coordinate in enumerate([(0, 5, 1), (1, 2, 0), (0, 6, 2)]):
            shifted = [a.copy() for a in arrays]
            shifted[which][coordinate] += 1e-5
            plus = np.sum(reference.dense_attention(*shifted, mask)[0] * upstream)
            shifted[which][coordinate] -= 2e-5
            minus = np.sum(reference.dense_attention(*shifted, mask)[0] * upstream)
            finite_difference.append(abs((plus - minus) / 2e-5 - gradients[which][coordinate]))
        record = {"mask": name, "dense_gradient_errors": [maximum_error(x, y) for x, y in zip(gradients, direct)],
                  "finite_difference_errors": finite_difference}
        if name != "empty_query":
            tensors = [torch.tensor(a, dtype=torch.float64, requires_grad=True) for a in arrays]
            scores = tensors[0] @ tensors[1].transpose(-1, -2) / np.sqrt(3)
            result = scores.masked_fill(~torch.tensor(mask)[None], -torch.inf).softmax(-1) @ tensors[2]
            (result * torch.tensor(upstream)).sum().backward()
            record["torch_autograd_errors"] = [maximum_error(g, t.grad.numpy()) for g, t in zip(gradients, tensors)]
        gradient_checks.append(record)
    # Fresh asymmetric content; repair metadata is a null, discarding document IDs is not.
    packed_output = reference.dense_attention(*arrays, packed)[0]
    wrong_output = reference.dense_attention(*arrays, causal)[0]
    changed = [a.copy() for a in arrays]; changed[2][0, 1, 0] += 1.25
    fresh_output = reference.ring_attention(*changed, partitions, packed)[0]
    missing_gradient = reference.blockwise_backward(*arrays, [partitions[0]], packed,
        reference.dense_attention(*arrays, packed)[0],
        # LSE from the complete ring: the bug below drops all other pairs, not normalization.
        reference.ring_attention(*arrays, partitions, packed)[1], upstream)[1]
    correct_gradient = reference.blockwise_backward(*arrays, partitions, packed,
        reference.ring_attention(*arrays, partitions, packed)[0],
        reference.ring_attention(*arrays, partitions, packed)[1], upstream)[1]
    scalar_scores = [0, np.log(2), np.log(4), 0]
    scalar_values = [2, 8, 1, -2]
    scalar = scalar_online(scalar_scores, scalar_values, [[0, 1], [2, 3]])
    scalar_shift = scalar_online(np.asarray(scalar_scores) + 1000, scalar_values, [[0, 1], [2, 3]])
    fresh_scalar = scalar_online([np.log(3), 0, np.log(2), 0], [4, -1, 7, 2], [[0, 1], [2, 3]])
    model = json.loads((DIRECTORY / "movement-attention-model.json").read_text())
    data = np.loadtxt(DIRECTORY / "movement_libras.data", delimiter=",")
    examples = []
    for source_id, frame_id, delta in [(77, 22, .1), (20, 9, -.15)]:
        points = data[source_id - 1, :90].reshape(45, 2)
        before = model_inference(points, model)
        edited = points.copy(); edited[frame_id, 0] = np.clip(edited[frame_id, 0] + delta, 0, 1)
        after = model_inference(edited, model)
        reverse = model_inference(points, model, direction=-1)
        examples.append({"source_id": source_id, "actual_class": int(data[source_id - 1, 90]),
                         "before": before, "edited": after, "frame_zero_based": frame_id,
                         "reverse_output_error": reverse["output_error"],
                         "edit_output_change": maximum_error(before["dense_output"], after["dense_output"]),
                         "edit_probability_change": maximum_error(before["probabilities"], after["probabilities"])})
    result = {"environment": {"numpy": np.__version__, "torch": torch.__version__},
              "source_data_sha256": hashlib.sha256((DIRECTORY / "movement_libras.data").read_bytes()).hexdigest(),
              "identities": identities, "gradients": gradient_checks,
              "packed_mask_missing_metadata_change": maximum_error(packed_output, wrong_output),
              "fresh_value_edit_change": maximum_error(packed_output, fresh_output),
              "missing_remote_key_gradient_change": maximum_error(missing_gradient, correct_gradient),
              "scalar": scalar, "scalar_offset_1000": scalar_shift, "fresh_scalar": fresh_scalar,
              "scalar_score_gradient": (np.array([1, 2, 4, 1]) / 8 * (np.array(scalar_values) - 2.5)),
              "real_examples": examples}
    def convert(value):
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, np.generic): return value.item()
        raise TypeError(type(value).__name__)
    (DIRECTORY / "partition-results.json").write_text(json.dumps(result, default=convert, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"identities": identities, "gradients": gradient_checks,
                      "scalars": [scalar[-1], fresh_scalar[-1]],
                      "real": [{"source": e["source_id"], "actual": e["actual_class"],
                                "predicted": e["before"]["predicted_class_one_based"],
                                "error": e["before"]["output_error"], "edit_change": e["edit_probability_change"]}
                               for e in examples]}, default=convert))


if __name__ == "__main__":
    main()
