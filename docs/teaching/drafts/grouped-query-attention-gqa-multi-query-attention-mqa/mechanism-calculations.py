"""Small independent NumPy GQA fixtures and CPU PyTorch operator checks.
Run with Python, NumPy and PyTorch; writes mechanism-fixtures.json.
Constructed arrays are teaching inputs, not learned model activations.
"""
from pathlib import Path
import json
import math

import numpy as np
import torch
from torch.nn import functional as functional


def grouped_attention(query, keys, values, query_positions, key_positions,
                      mapping=None, causal=True):
    # Shapes: Q [B,Hq,T,dk], K [B,Hkv,S,dk], V [B,Hkv,S,dv].
    batch, query_heads, query_length, key_width = query.shape
    kv_heads = keys.shape[1]
    if query_heads % kv_heads or keys.shape[:3] != values.shape[:3]:
        raise ValueError("Use equal-size query groups and matching K/V head/slot counts.")
    if keys.shape[0] != batch or keys.shape[-1] != key_width:
        raise ValueError("Batch and Q/K coordinate widths must agree.")
    if mapping is None:
        mapping = np.arange(query_heads) // (query_heads // kv_heads)
    mapping = np.asarray(mapping)
    if mapping.shape != (query_heads,) or np.any(mapping < 0) or np.any(mapping >= kv_heads):
        raise ValueError("Each query head must name a valid KV head.")
    legal = np.ones((query_length, keys.shape[2]), dtype=bool)
    if causal:
        legal = np.asarray(key_positions)[None, :] <= np.asarray(query_positions)[:, None]
    if not legal.any(-1).all():
        raise ValueError("A query has no legal key.")
    outputs, weights = [], []
    for head, memory_head in enumerate(mapping):
        scores = query[:, head] @ keys[:, memory_head].swapaxes(-1, -2) / math.sqrt(key_width)
        scores = np.where(legal, scores, -np.inf)
        probabilities = np.exp(scores - scores.max(-1, keepdims=True))
        probabilities /= probabilities.sum(-1, keepdims=True)
        outputs.append(probabilities @ values[:, memory_head])
        weights.append(probabilities)
    return np.stack(outputs, 1), np.stack(weights, 1)


def cache_bytes(batch, layers, kv_heads, length, key_width, value_width, bytes_per_element):
    return batch * layers * kv_heads * length * (key_width + value_width) * bytes_per_element


def main():
    root = Path(__file__).resolve().parent
    query = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)[None, :, None, :] * math.sqrt(2)
    keys = np.array([[[1, 0], [0, 1], [1, 1]], [[1, 1], [-1, 0], [0, -1]]], dtype=float)[None]
    values = np.array([[[2, 0], [0, 4], [2, 2]], [[1, 3], [-1, 2], [3, 0]]], dtype=float)[None]
    baseline, probabilities = grouped_attention(query, keys, values, [2], [0, 1, 2])
    changed_value = values.copy()
    changed_value[0, 0, 0] += [1, -1]
    edited_value, edited_value_probabilities = grouped_attention(query, keys, changed_value, [2], [0, 1, 2])
    changed_key = keys.copy()
    changed_key[0, 0, 0] += [1, 0]
    edited_key, _ = grouped_attention(query, changed_key, values, [2], [0, 1, 2])
    changed_query = query.copy()
    changed_query[0, 0, 0] = [math.sqrt(2), math.sqrt(2)]
    edited_query, _ = grouped_attention(changed_query, keys, values, [2], [0, 1, 2])
    wrong, _ = grouped_attention(query, keys, values, [2], [0, 1, 2], [0, 1, 0, 1])
    permuted, _ = grouped_attention(query, keys[:, ::-1], values[:, ::-1], [2], [0, 1, 2], [1, 1, 0, 0])
    expanded, _ = grouped_attention(query, np.repeat(keys, 2, axis=1),
                                    np.repeat(values, 2, axis=1), [2], [0, 1, 2])
    assert np.allclose(baseline, permuted) and np.allclose(baseline, expanded)
    assert np.array_equal(probabilities, edited_value_probabilities)
    assert np.array_equal(baseline[:, 2:], edited_value[:, 2:])
    assert np.array_equal(baseline[:, 1:], edited_query[:, 1:])
    assert np.array_equal(baseline[:, 1], edited_key[:, 1])

    # A nonlinearity counterexample: averaging heads is not averaging outputs.
    conversion_query = np.array([1., 2.])[None, :, None, None]
    conversion_keys = np.array([[2., 0.], [0., 2.]])[None, :, :, None]
    conversion_values = np.array([[1., 3.], [5., -1.]])[None, :, :, None]
    original, old_weights = grouped_attention(conversion_query, conversion_keys,
                                               conversion_values, [1], [0, 1])
    averaged, new_weights = grouped_attention(conversion_query, conversion_keys.mean(1, keepdims=True),
                                               conversion_values.mean(1, keepdims=True), [1], [0, 1])

    # Native checks use small deterministic float64 tensors, no training campaign.
    q = torch.tensor(query.tolist(), dtype=torch.float64, requires_grad=True)
    k = torch.tensor(keys.tolist(), dtype=torch.float64, requires_grad=True)
    v = torch.tensor(values.tolist(), dtype=torch.float64, requires_grad=True)
    direct_scores = torch.einsum("bgrtd,bgud->bgrtu", q.reshape(1, 2, 2, 1, 2), k) / math.sqrt(2)
    direct = torch.einsum("bgrtu,bgud->bgrtd", direct_scores.softmax(-1), v).reshape(1, 4, 1, 2)
    native = functional.scaled_dot_product_attention(q, k, v, dropout_p=0., enable_gqa=True)
    assert torch.allclose(direct, native, atol=1e-12, rtol=1e-12)
    repeated_keys = k.repeat_interleave(2, dim=1)
    repeated_values = v.repeat_interleave(2, dim=1)
    separate_values = repeated_values.detach().clone().requires_grad_()
    separate = (q @ repeated_keys.transpose(-1, -2) / math.sqrt(2)).softmax(-1) @ separate_values
    direct.square().sum().backward(retain_graph=True)
    separate.square().sum().backward()
    expected_shared_gradient = separate_values.grad.reshape(1, 2, 2, 3, 2).sum(2)
    assert torch.allclose(v.grad, expected_shared_gradient, atol=1e-12)
    assert repeated_keys.untyped_storage().data_ptr() != k.untyped_storage().data_ptr()
    upper_left = functional.scaled_dot_product_attention(q.detach(), k.detach(), v.detach(),
                    is_causal=True, dropout_p=0., enable_gqa=True)
    explicit = functional.scaled_dot_product_attention(q.detach(), k.detach(), v.detach(),
                    attn_mask=torch.ones(1, 3, dtype=torch.bool), dropout_p=0., enable_gqa=True)
    assert torch.allclose(explicit, native)
    assert not torch.allclose(upper_left, native)

    counts = []
    for length in (2048, 4096, 8192, 16384, 32768, 65536, 131072):
        counts.append({"length": length, "bytes_by_kv_heads": {
            str(heads): cache_bytes(1, 80, heads, length, 128, 128, 2) for heads in (64, 8, 1)}})
    payload = {
        "manual_inputs": {"query": query.tolist(), "keys": keys.tolist(), "values": values.tolist()},
        "manual_outputs": {"baseline": baseline.tolist(), "weights": probabilities.tolist(),
            "shared_value_edit": edited_value.tolist(), "shared_key_edit": edited_key.tolist(),
            "one_query_edit": edited_query.tolist(), "wrong_interleaved_mapping": wrong.tolist(),
            "consistent_group_permutation_max_error": float(abs(permuted-baseline).max()),
            "tied_mha_max_error": float(abs(expanded-baseline).max())},
        "conversion": {"old_outputs": original.tolist(), "new_outputs": averaged.tolist(),
                       "old_weights": old_weights.tolist(), "new_weights": new_weights.tolist()},
        "native_checks": {"torch": torch.__version__, "native_vs_grouped_max_error": float((native-direct).abs().max().detach()),
            "repeated_k_allocates_distinct_storage": True,
            "compact_k_bytes": k.numel()*k.element_size(),
            "expanded_k_bytes": repeated_keys.numel()*repeated_keys.element_size(),
            "shared_value_gradient": v.grad.tolist(),
            "sum_of_per_read_gradients": expected_shared_gradient.tolist(),
            "wrong_upper_left_causal_output": upper_left.tolist(),
            "explicit_all_legal_output": explicit.tolist()},
        "large_cache_counts": counts,
        "practice": {"cache_2batch_12layer_3heads_1024length_dk64_dv32_fp16": cache_bytes(2, 12, 3, 1024, 64, 32, 2),
                     "amdahl_fraction_0_6_reduction_8": 1/(.4+.6/8)}}
    (root/"mechanism-fixtures.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("weights", probabilities[0, :, 0])
    print("outputs", baseline[0, :, 0])
    print("conversion", original.ravel(), "->", averaged.ravel())
    print("checks", payload["native_checks"]["native_vs_grouped_max_error"], "gradient agreement passed")


if __name__ == "__main__":
    main()
