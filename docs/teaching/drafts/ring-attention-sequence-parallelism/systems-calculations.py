"""Exact ownership/pair counts and an explicitly hypothetical hardware cost model."""
from pathlib import Path
import json
import numpy as np


def layout(length, ranks, kind):
    if kind == "contiguous": return list(np.array_split(np.arange(length), ranks))
    if kind == "striped": return [np.arange(i, length, ranks) for i in range(ranks)]
    if length % (2 * ranks): raise ValueError("This equal zigzag layout needs L divisible by 2P")
    pieces = np.split(np.arange(length), 2 * ranks)
    return [np.concatenate([pieces[i], pieces[2 * ranks - 1 - i]]) for i in range(ranks)]


def work_counts(length, ranks, kind, tile):
    owners = layout(length, ranks, kind)
    pairs, executed = np.zeros((ranks, ranks), int), np.zeros((ranks, ranks), int)
    for i, queries in enumerate(owners):
        for j, keys in enumerate(owners):
            mask = keys[None, :] <= queries[:, None]
            pairs[i, j] = mask.sum()
            for row in range(0, len(queries), tile):
                for column in range(0, len(keys), tile):
                    part = mask[row:row + tile, column:column + tile]
                    if part.any(): executed[i, j] += part.size
    rounds = np.array([[pairs[rank, (rank - step) % ranks] for rank in range(ranks)] for step in range(ranks)])
    tile_rounds = np.array([[executed[rank, (rank - step) % ranks] for rank in range(ranks)] for step in range(ranks)])
    return {"length": length, "ranks": ranks, "layout": kind, "tile": tile,
            "owners": owners, "legal_pairs": pairs, "row_totals": pairs.sum(axis=1),
            "rounds": rounds, "ideal_round_critical_pairs": rounds.max(axis=1).sum(),
            "executed_tile_cells": executed, "tile_round_critical_cells": tile_rounds.max(axis=1).sum()}


def cost(local_length, ranks, query_heads=8, kv_heads=2, head_width=64, element_bytes=2,
         effective_flops=1e14, effective_bytes_per_second=5e10, latency=2e-6):
    # Batch=1. Two dense matrix products only; no causal skipping, softmax or projections.
    flops = 4 * local_length**2 * query_heads * head_width
    payload = 2 * local_length * kv_heads * head_width * element_bytes
    compute = flops / effective_flops
    communication = latency + payload / effective_bytes_per_second
    query_bytes = local_length * query_heads * head_width * element_bytes
    local_kv = payload
    receive_kv = payload
    numerator = local_length * query_heads * head_width * 4
    statistics = 2 * local_length * query_heads * 4
    output = query_bytes
    tile_workspace = query_heads * min(local_length, 128)**2 * 4
    inventory = {"query": query_bytes, "local_or_current_kv": local_kv, "next_receive_kv": receive_kv,
                 "fp32_numerator": numerator, "fp32_maximum_and_sum": statistics,
                 "output_if_distinct": output, "one_fp32_128_square_score_tile_per_head": tile_workspace}
    return {"local_length": local_length, "ranks": ranks, "full_length": local_length * ranks,
            "two_gemm_flops_per_round": flops, "one_direction_bytes_per_transfer": payload,
            "forward_send_bytes_per_rank": (ranks - 1) * payload,
            "compute_us": compute * 1e6, "transfer_us": communication * 1e6,
            "synchronous_us": (ranks * compute + (ranks - 1) * communication) * 1e6,
            "ideal_overlap_us": (compute + (ranks - 1) * max(compute, communication)) * 1e6,
            "compute_only_us": ranks * compute * 1e6, "memory_bytes": inventory,
            "listed_memory_total_bytes": sum(inventory.values())}


def main():
    examples = [work_counts(length, ranks, kind, tile) for length, ranks in [(16, 4), (12, 3)]
                for kind in ["contiguous", "striped", "zigzag"] for tile in [1, 2, 4]]
    costs = [cost(c, 4) for c in [128, 1024, 4096]]
    strong = [cost(4096 // ranks, ranks) for ranks in [1, 2, 4, 8, 16]]
    weak = [cost(1024, ranks) for ranks in [1, 2, 4, 8, 16]]
    tensor = np.arange(8 * 4).reshape(8, 4)
    sequence_shards = list(np.split(tensor, 2, axis=0))
    head_shards = list(np.split(tensor, 2, axis=1))
    restored = np.concatenate(head_shards, axis=1)
    assert np.array_equal(restored, tensor)
    label_inputs = [10, 11, 12, 13, 14, 15]
    result = {"work": examples, "hypothetical_cost": costs, "strong_scaling_fixed_length_4096": strong,
              "weak_scaling_fixed_local_length_1024": weak,
              "ulysses": {"global_token_head_labels": tensor, "sequence_shards": sequence_shards,
                          "head_shards": head_shards, "restored": restored,
                          "per_tensor_elements_per_rank_before_and_after": 16,
                          "per_rank_nonlocal_elements_one_reshard": 8},
              "labels": {"input": label_inputs, "correct_global_next_targets": [11, 12, 13, 14, 15, -100],
                         "wrong_shift_after_split": [11, 12, -100, 14, 15, -100]},
              "loss_averaging": {"valid_tokens": [3, 1], "mean_losses": [2, 6], "correct": 3, "wrong": 4},
              "rope_pair": {"query_global_position": 5, "key_global_position": 1,
                            "angular_frequency": 1, "unrotated_vectors": [1, 0],
                            "correct_unscaled_dot": np.cos(5 - 1),
                            "wrong_chunk_reset_dot": np.cos(0 - 1)},
              "one_million_mha_qkvo_bytes": 4 * 1_000_000 * 8192 * 2,
              "fixed_dataset_relative_attention_cost_L_doubled": 2,
              "fixed_sequences_relative_attention_cost_L_doubled": 4}
    def convert(value):
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, np.generic): return value.item()
        raise TypeError(type(value).__name__)
    target = Path(__file__).resolve().parent / "systems-results.json"
    target.write_text(json.dumps(result, indent=2, default=convert, allow_nan=False) + "\n")
    print(json.dumps({"work": [x for x in examples if x['tile'] == 1], "costs": costs}, default=convert))


if __name__ == "__main__":
    main()
