"""Independent NumPy MLA fixtures: geometry, absorption, positions and costs.

Run with Python and NumPy. All arrays are constructed teaching inputs, not
pretrained activations. Writes mechanism-fixtures.json beside this file.
"""
from pathlib import Path
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parent


def rotate_rows(rows, positions, radians_per_position):
    angle = np.asarray(positions) * radians_per_position
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.stack((rows[..., 0]*cosine-rows[..., 1]*sine,
                     rows[..., 0]*sine+rows[..., 1]*cosine), -1)


def attention(content_queries, latents, key_up, value_up, rotary_queries,
              rotary_keys, scale, legal=None, expanded=False):
    """Q [H,P], C [S,C], U_K [H,P,C], U_V [H,V,C], qR [H,R]."""
    if expanded:
        keys = np.einsum("sc,hpc->hsp", latents, key_up)
        values = np.einsum("sc,hvc->hsv", latents, value_up)
        content_scores = np.einsum("hp,hsp->hs", content_queries, keys)
    else:
        effective_queries = np.einsum("hp,hpc->hc", content_queries, key_up)
        content_scores = effective_queries @ latents.T
    positional_scores = rotary_queries @ rotary_keys.T
    scores = (content_scores+positional_scores)*scale
    if legal is not None:
        scores = np.where(legal, scores, -np.inf)
    if not np.isfinite(scores).any(-1).all():
        raise ValueError("No legal finite key for a query.")
    weights = np.exp(scores-scores.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    latent_mixtures = weights @ latents
    outputs = (np.einsum("hs,hsv->hv", weights, values) if expanded
               else np.einsum("hc,hvc->hv", latent_mixtures, value_up))
    return {"content_scores": content_scores, "positional_scores": positional_scores,
            "scaled_scores": scores, "weights": weights,
            "latent_mixtures": latent_mixtures, "head_outputs": outputs}


def lists(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {name: lists(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [lists(item) for item in value]
    return value


def main():
    content_queries = np.array([[1., 0.], [1., 1.]])
    latents = np.array([[1., 0.], [0., 1.], [1., 1.]])
    key_up = np.array([[[1., 0.], [0., 2.]], [[1., 1.], [1., -1.]]])
    value_up = np.array([np.eye(2), [[2., 0.], [1., -1.]]])
    raw_rotary_queries = np.eye(2)
    raw_rotary_keys = np.array([[1., 0.]]*3)
    # pi/2 is a deliberately convenient hand-example frequency, not the
    # ordinary base-10000 two-coordinate frequency used by the real model.
    rotary_queries = rotate_rows(raw_rotary_queries, [2, 2], math.pi/2)
    rotary_keys = rotate_rows(raw_rotary_keys, [0, 1, 2], math.pi/2)
    arguments = (content_queries, latents, key_up, value_up, rotary_queries, rotary_keys, .5)
    base = attention(*arguments)
    expanded = attention(*arguments, expanded=True)
    assert np.allclose(base["head_outputs"], expanded["head_outputs"], atol=1e-12)
    output_map = np.array([[1., 0., .5, 0.], [0., 1., 0., .5]])
    final = output_map @ base["head_outputs"].reshape(-1)
    basis = np.diag([2., .5])
    inverse = np.linalg.inv(basis)
    relabeled = attention(content_queries, latents @ basis.T,
        key_up @ inverse, value_up @ inverse, rotary_queries, rotary_keys, .5)
    assert np.allclose(relabeled["head_outputs"], base["head_outputs"], atol=1e-12)
    changed_latents = latents.copy()
    changed_latents[0] += [0., 1.]
    changed = attention(content_queries, changed_latents, key_up, value_up,
                        rotary_queries, rotary_keys, .5)
    shifted = attention(content_queries, latents, key_up, value_up,
        rotate_rows(raw_rotary_queries, [7, 7], math.pi/2),
        rotate_rows(raw_rotary_keys, [5, 6, 7], math.pi/2), .5)
    stale_position = attention(content_queries, latents, key_up, value_up,
        rotate_rows(raw_rotary_queries, [3, 3], math.pi/2), rotary_keys, .5)
    changed_value_up = value_up.copy()
    changed_value_up[0, 0, 0] += 1
    value_only = attention(content_queries, latents, key_up, changed_value_up,
                           rotary_queries, rotary_keys, .5)
    assert np.array_equal(value_only["weights"], base["weights"])
    assert np.array_equal(value_only["head_outputs"][1], base["head_outputs"][1])
    assert np.allclose(shifted["head_outputs"], base["head_outputs"], atol=1e-12)
    # A rotation at each key location blocks one key-position-independent
    # effective query. At j=0,1 the required queries below differ.
    rotation = np.array([[0., -1.], [1., 0.]])
    up = np.diag([2., 1.])
    query = np.array([1., 0.])
    effective_by_key = [up.T @ query, up.T @ rotation.T @ query]
    noncommutation = rotation @ up-up @ rotation
    scores_rank_one = np.outer([0., 1., 2.], [0., 1., 2.])
    probabilities = np.exp(scores_rank_one-scores_rank_one.max(-1, keepdims=True))
    probabilities /= probabilities.sum(-1, keepdims=True)
    assert np.linalg.matrix_rank(scores_rank_one) == 1
    assert np.linalg.matrix_rank(probabilities) == 3
    # One discarded direction matters for an input outside the large-singular
    # direction: parameter-Frobenius optimal is not prediction optimal.
    joint = np.diag([10., 1.])
    rank_one = np.diag([10., 0.])
    input_vector = np.array([0., 10.])
    # Keeping all factors visible avoids collapsing unlike cache baselines.
    head_count, content_width, rotary_width, value_width, latent_width = 128, 128, 64, 128, 512
    entries = {"plain_MHA_qk128_v128": head_count*(content_width+value_width),
        "GQA8_qk128_v128": 8*(content_width+value_width),
        "MQA_qk128_v128": content_width+value_width,
        "same_MLA_function_expanded_K192_V128": head_count*(content_width+rotary_width+value_width),
        "same_MLA_function_expanded_content_shared_rotary": head_count*(content_width+value_width)+rotary_width,
        "MLA_compact": latent_width+rotary_width}
    sizes = [{"length": length, "bytes": {name: count*length*60*2 for name, count in entries.items()}}
             for length in (1024, 4096, 16384, 32768, 65536, 131072)]
    length = 32768
    expanded_core = 2*head_count*length*(content_width+rotary_width+value_width)
    absorbed_core = 2*head_count*length*(2*latent_width+rotary_width)
    reconstruction_cost = 2*length*head_count*latent_width*(content_width+value_width)
    output = {"manual_inputs": {"content_queries": content_queries, "latents": latents,
        "key_up": key_up, "value_up": value_up, "raw_rotary_queries": raw_rotary_queries,
        "raw_rotary_keys": raw_rotary_keys, "query_position": 2, "key_positions": [0,1,2],
        "radians_per_position": math.pi/2, "score_scale": .5, "output_map": output_map},
        "baseline": base, "final_output": final,
        "expanded_absorbed_error": np.max(np.abs(expanded["head_outputs"]-base["head_outputs"])),
        "basis_change": {"matrix": basis, "outputs": relabeled["head_outputs"]},
        "changed_first_latent_by_0_1": changed,
        "value_up_only_edit": value_only,
        "common_shift_error": np.max(np.abs(shifted["head_outputs"]-base["head_outputs"])),
        "query_shift_without_cache_update": stale_position,
        "rotary_absorption_counterexample": {"up": up, "rotation": rotation,
            "commutator": noncommutation, "effective_queries_by_key_position": effective_by_key},
        "rank_one_logits_full_rank_probabilities": {"logits": scores_rank_one,
            "probabilities": probabilities, "determinant": np.linalg.det(probabilities)},
        "rank_truncation_input_counterexample": {"joint": joint, "rank_one": rank_one,
            "input": input_vector, "full_output": joint@input_vector,
            "truncated_output": rank_one@input_vector, "parameter_squared_error": 1.},
        "nonlinear_value_counterexample": {"latents": [-1., 1.], "weights": [.5,.5],
            "mix_relu_values": .5, "relu_mixed_latent": 0.},
        "entries_per_token_per_layer": entries, "sixty_layer_two_byte_payloads": sizes,
        "decode_operation_model": {"length": length, "batch": 1, "query_length": 1,
            "multiply_add_operations": 2, "expanded_core": expanded_core,
            "absorbed_core": absorbed_core, "absorbed_to_expanded_core_ratio": absorbed_core/expanded_core,
            "reconstruct_all_prefix_content_KV": reconstruction_cost,
            "new_query_absorption": 2*head_count*content_width*latent_width,
            "latent_output_up_projection": 2*head_count*value_width*latent_width},
        "practice": {"cache_B2_N12_L2048_C24_R8_bytes2": 2*12*2048*(24+8)*2,
            "C12_P4_R4_wrong_scale_ratio": math.sqrt(8/16)}}
    (ROOT/"mechanism-fixtures.json").write_text(json.dumps(lists(output), indent=2), encoding="utf-8")
    print(json.dumps(lists({"hand_outputs": base["head_outputs"], "final_output": final,
        "basis_null_error": np.max(np.abs(relabeled["head_outputs"]-base["head_outputs"])),
        "full_rank_probability_determinant": np.linalg.det(probabilities),
        "decode_core_operations": output["decode_operation_model"]}), indent=2))


if __name__ == "__main__":
    main()
