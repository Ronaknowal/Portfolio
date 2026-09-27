"""Executable CPU teaching reference, not distributed/FlashAttention code.

Inputs are [heads, positions, channels]; Q and K have equal channels.
One sequence, MHA, no dropout or extra positional bias. A boolean mask
is in GLOBAL position order. Empty rows produce zero, with logsumexp -inf.
The study driver separately verifies gradients and real frozen attention.
"""
import numpy as np


def dense_attention(query, key, value, allowed):
    scores = query @ key.swapaxes(-1, -2) / np.sqrt(query.shape[-1])
    scores = np.where(allowed[None], scores, -np.inf)
    maxima = scores.max(axis=-1, keepdims=True)
    safe_maxima = np.where(np.isfinite(maxima), maxima, 0)
    weights = np.exp(scores - safe_maxima)
    totals = weights.sum(axis=-1, keepdims=True)
    weights = np.divide(weights, totals, out=np.zeros_like(weights), where=totals > 0)
    return weights @ value, weights


def ring_attention(query, key, value, ownership, allowed, direction=1):
    """Simulate P rounds; keep queries local, visit each labeled KV shard once.

    ownership is an exact partition of global positions, possibly uneven.
    direction changes visitation, not sequence positions. CPU arrays already
    exist in one process: this proves arithmetic, not distributed peak memory.
    """
    length = query.shape[1]
    if not ownership:
        raise ValueError("At least one owner is required")
    positions = np.concatenate(ownership)
    if (direction not in (-1, 1) or
            any(len(indices) == 0 for indices in ownership) or
            not np.array_equal(np.sort(positions), np.arange(length)) or
            allowed.shape != (length, length)):
        raise ValueError("Need nonempty exact ownership and a global square mask")
    heads = query.shape[0]
    output = np.zeros((heads, length, value.shape[-1]), dtype=query.dtype)
    logsumexp = np.full((heads, length), -np.inf, dtype=query.dtype)
    trace = []
    ranks = len(ownership)
    for rank, query_ids in enumerate(ownership):
        maxima = np.full((heads, len(query_ids)), -np.inf, dtype=query.dtype)
        totals = np.zeros_like(maxima)
        numerator = np.zeros((heads, len(query_ids), value.shape[-1]), dtype=query.dtype)
        for step in range(ranks):
            owner = (rank - direction * step) % ranks
            key_ids = ownership[owner]
            scores = query[:, query_ids] @ key[:, key_ids].swapaxes(-1, -2)
            scores /= np.sqrt(query.shape[-1])
            scores = np.where(allowed[np.ix_(query_ids, key_ids)][None], scores, -np.inf)
            block_maxima = scores.max(axis=-1)
            new_maxima = np.maximum(maxima, block_maxima)
            safe_maxima = np.where(np.isfinite(new_maxima), new_maxima, 0)
            correction = np.exp(maxima - safe_maxima)
            exponentials = np.exp(scores - safe_maxima[..., None])
            numerator = correction[..., None] * numerator + exponentials @ value[:, key_ids]
            totals = correction * totals + exponentials.sum(axis=-1)
            maxima = new_maxima
            trace.append({"query_owner": rank, "step": step, "kv_owner": owner,
                          "query_ids": query_ids.tolist(), "key_ids": key_ids.tolist()})
        output[:, query_ids] = np.divide(numerator, totals[..., None],
                                        out=np.zeros_like(numerator), where=totals[..., None] > 0)
        safe_totals = np.where(totals > 0, totals, 1)
        logsumexp[:, query_ids] = maxima + np.log(safe_totals)
    return output, logsumexp, trace


def blockwise_backward(query, key, value, ownership, allowed, output, logsumexp, upstream):
    """Recompute probabilities from global LSE; accumulate every KV owner's gradients.

    Central CPU accumulation models the required reduction, not its transport.
    """
    query_gradient, key_gradient, value_gradient = [np.zeros_like(x) for x in (query, key, value)]
    scale = np.sqrt(query.shape[-1])
    for query_ids in ownership:
        row_dot = (upstream[:, query_ids] * output[:, query_ids]).sum(axis=-1, keepdims=True)
        for key_ids in ownership:
            scores = query[:, query_ids] @ key[:, key_ids].swapaxes(-1, -2) / scale
            valid = allowed[np.ix_(query_ids, key_ids)][None]
            lse = logsumexp[:, query_ids, None]
            scores = np.where(valid, scores, -np.inf)
            probabilities = np.exp(scores - np.where(np.isfinite(lse), lse, 0))
            probability_gradient = upstream[:, query_ids] @ value[:, key_ids].swapaxes(-1, -2)
            score_gradient = probabilities * (probability_gradient - row_dot)
            query_gradient[:, query_ids] += score_gradient @ key[:, key_ids] / scale
            key_gradient[:, key_ids] += score_gradient.swapaxes(-1, -2) @ query[:, query_ids] / scale
            value_gradient[:, key_ids] += probabilities.swapaxes(-1, -2) @ upstream[:, query_ids]
    return query_gradient, key_gradient, value_gradient


if __name__ == "__main__":
    generator = np.random.default_rng(91)
    query, key, value = [generator.normal(size=(2, 7, 3)) for _ in range(3)]
    ownership = list(np.array_split(np.arange(7), 3))
    allowed = np.arange(7)[None, :] <= np.arange(7)[:, None]
    reference, _ = dense_attention(query, key, value, allowed)
    distributed, _, _ = ring_attention(query, key, value, ownership, allowed)
    reverse, _, _ = ring_attention(query, key, value, ownership, allowed, -1)
    np.testing.assert_allclose(distributed, reference, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(reverse, reference, atol=1e-12, rtol=1e-12)
    print("Dense/ring maximum absolute difference:", np.abs(distributed - reference).max())
    print("Both directions preserve global causal attention.")
