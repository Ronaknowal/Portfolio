"""Explicit CPU/Gloo ring attention and owner-returning manual backward.

Launch: torchrun --standalone --nproc-per-node=3 distributed_ring.py
PyTorch 2.14.0 target; float64, one sequence, equal Q/K/V head width,
contiguous possibly uneven nonempty shards, global causal mask, no dropout.
This readable synchronous protocol makes no overlap or throughput claim.
"""
from datetime import timedelta
import torch
import torch.distributed as dist
from torch.nn import functional as F


def rotate(packet):
    """Send owned bytes onward; receive into distinct storage before reuse."""
    world, rank = dist.get_world_size(), dist.get_rank()
    if world == 1:
        return packet
    received = torch.empty_like(packet)
    requests = dist.batch_isend_irecv([
        dist.P2POp(dist.isend, packet, (rank+1) % world),
        dist.P2POp(dist.irecv, received, (rank-1) % world),
    ])
    for request in requests:
        request.wait()
    return received


def layout(local_length):
    sizes = [torch.empty(1, dtype=torch.int64) for _ in range(dist.get_world_size())]
    dist.all_gather(sizes, torch.tensor([local_length], dtype=torch.int64))
    counts = [int(size.item()) for size in sizes]
    if min(counts) < 1:
        raise ValueError("Every rank needs a nonempty shard")
    starts = [sum(counts[:rank]) for rank in range(len(counts))]
    return counts, starts


def block_scores(query, keys, query_start, key_start):
    scores = query @ keys.transpose(-1, -2) / query.shape[-1]**.5
    q_positions = query_start + torch.arange(query.shape[1])
    k_positions = key_start + torch.arange(keys.shape[1])
    return scores.masked_fill(k_positions[None, :] > q_positions[:, None], -torch.inf)


def pack(key, value, maximum, with_gradients=False):
    packet = key.new_zeros((4 if with_gradients else 2, key.shape[0], maximum, key.shape[-1]))
    packet[0, :, :key.shape[1]] = key
    packet[1, :, :value.shape[1]] = value
    return packet


@torch.no_grad()
def ring_forward(query, key, value, counts, starts):
    rank, world = dist.get_rank(), dist.get_world_size()
    maximum = query.new_full(query.shape[:-1], -torch.inf)
    mass = torch.zeros_like(maximum)
    numerator = torch.zeros_like(query)
    packet = pack(key, value, max(counts))
    for step in range(world):
        owner = (rank-step) % world
        keys, values = packet[:2, :, :counts[owner]]
        scores = block_scores(query, keys, starts[rank], starts[owner])
        updated = torch.maximum(maximum, scores.amax(-1))
        safe = torch.where(torch.isfinite(updated), updated, 0.)
        correction = torch.exp(maximum-safe)
        probabilities = torch.exp(scores-safe[..., None])
        numerator = correction[..., None]*numerator + probabilities @ values
        mass = correction*mass + probabilities.sum(-1)
        maximum = updated
        if step+1 < world:
            packet = rotate(packet)
    # The global causal contract gives each query at least its own key.
    output = numerator/mass[..., None]
    return output, maximum+mass.log()


@torch.no_grad()
def ring_backward(query, key, value, output, logsumexp, upstream, counts, starts):
    rank, world = dist.get_rank(), dist.get_world_size()
    query_gradient = torch.zeros_like(query)
    packet = pack(key, value, max(counts), with_gradients=True)
    correction = (upstream*output).sum(-1, keepdim=True)
    for step in range(world):
        owner = (rank-step) % world
        length = counts[owner]
        keys, values = packet[:2, :, :length]
        scores = block_scores(query, keys, starts[rank], starts[owner])
        probabilities = torch.exp(scores-logsumexp[..., None])
        score_gradient = probabilities*(upstream @ values.transpose(-1, -2)-correction)
        query_gradient += score_gradient @ keys / query.shape[-1]**.5
        packet[2, :, :length] += score_gradient.transpose(-1, -2) @ query / query.shape[-1]**.5
        packet[3, :, :length] += probabilities.transpose(-1, -2) @ upstream
        # P transfers, not P-1: complete sums must return to their original owner.
        packet = rotate(packet)
    return query_gradient, packet[2, :, :key.shape[1]], packet[3, :, :value.shape[1]]


def main():
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.set_num_threads(1)
        # A small full oracle is created solely for validation, not in either ring routine.
        generator = torch.Generator().manual_seed(71)
        length, heads, width = 2*world+1, 2, 3
        full = [torch.randn(heads, length, width, generator=generator, dtype=torch.float64)
                for _ in range(4)]
        partitions = torch.tensor_split(torch.arange(length), world)
        indices = partitions[rank]
        query, key, value, upstream = [tensor[:, indices].contiguous() for tensor in full]
        counts, starts = layout(len(indices))
        actual, lse = ring_forward(query, key, value, counts, starts)
        gradients = ring_backward(query, key, value, actual, lse, upstream, counts, starts)
        oracle_inputs = [tensor.clone().requires_grad_() for tensor in full[:3]]
        oracle = F.scaled_dot_product_attention(*oracle_inputs, is_causal=True, dropout_p=0.)
        expected_gradients = torch.autograd.grad((oracle*full[3]).sum(), oracle_inputs)
        torch.testing.assert_close(actual, oracle[:, indices], rtol=1e-11, atol=1e-11)
        for actual_gradient, expected in zip(gradients, expected_gradients):
            torch.testing.assert_close(actual_gradient, expected[:, indices], rtol=1e-11, atol=1e-11)
        print("rank", rank, "positions", indices.tolist(), "forward maximum error",
              (actual-oracle[:, indices]).abs().max().item(), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
