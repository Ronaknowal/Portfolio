"""Executed, offline float64 mechanism fixtures; no training or GPU benchmark.

Run: python mechanism-calculations.py. Requires NumPy. The JSON includes
author-only expected answers; a learner UI must hide these until prediction.
"""
from pathlib import Path
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parent


def row_softmax(scores):
    values = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    return values / values.sum(axis=-1, keepdims=True)


def causal_window(length, total_keys, global_positions=()):
    query, key = np.indices((length, length))
    legal = key <= query
    window = (query - key) < total_keys
    hubs = np.isin(query, global_positions) | np.isin(key, global_positions)
    return legal & (window | hubs)


def reachable(mask, layers):
    # Rows are receiving positions, columns are contributing input positions.
    relation = np.eye(len(mask), dtype=bool)
    step = mask | np.eye(len(mask), dtype=bool)
    for _ in range(layers):
        relation = (step.astype(np.int64) @ relation.astype(np.int64)) > 0
    return relation


def selected_attention(query, keys, values, indices):
    scores = np.asarray(keys)[indices] @ np.asarray(query) / math.sqrt(len(query))
    return row_softmax(scores) @ np.asarray(values)[indices]


def feature_read(query_features, key_features, values):
    query_features = np.asarray(query_features, float)
    key_features = np.asarray(key_features, float)
    values = np.asarray(values, float)
    state = key_features.T @ values
    normalizer = key_features.sum(0)
    denominator = query_features @ normalizer
    if denominator <= 0:
        raise ValueError("This normalized read needs a positive denominator.")
    weights = query_features @ key_features.T / denominator
    output = query_features @ state / denominator
    assert np.allclose(output, weights @ values, atol=1e-14)
    return dict(state=state, normalizer=normalizer, denominator=denominator,
                weights=weights, output=output)


def gaussian_orthogonal_rows(generator, count, width):
    """Independent blocks; each row marginal is N(0,I), rows not independent."""
    blocks = []
    while sum(len(block) for block in blocks) < count:
        orthogonal, triangular = np.linalg.qr(generator.normal(size=(width, width)))
        signs = np.where(np.diag(triangular) < 0, -1., 1.)
        directions = (orthogonal * signs[None, :]).T
        # Independent chi_d radii retain Gaussian marginals, unlike sqrt(d).
        radii = np.linalg.norm(generator.normal(size=(width, width)), axis=1)
        blocks.append(directions * radii[:, None])
    return np.concatenate(blocks)[:count]


def positive_random_attention(query, key, value, projections, causal=True):
    """Q/K already include d**(-1/4). Fixed projections for the whole call."""
    query, key, value = map(lambda x: np.asarray(x, float), (query, key, value))
    q_log = query @ projections.T - (query * query).sum(-1, keepdims=True) / 2
    k_log = key @ projections.T - (key * key).sum(-1, keepdims=True) / 2
    # One scalar per query and ONE COMMON scalar across keys cancel in ratio.
    q_phi = np.exp(q_log - q_log.max(-1, keepdims=True))
    k_phi = np.exp(k_log - k_log.max())
    kernel = q_phi @ k_phi.T
    if causal:
        kernel = np.where(np.tri(len(query), len(key), dtype=bool), kernel, 0.)
    weights = kernel / kernel.sum(-1, keepdims=True)
    return weights @ value, weights


def occupied_tiles(mask, block=2):
    assert mask.shape[0] % block == mask.shape[1] % block == 0
    tiles = mask.reshape(mask.shape[0] // block, block,
                         mask.shape[1] // block, block).any(axis=(1, 3))
    return dict(edges=int(mask.sum()), occupied_tiles=int(tiles.sum()),
                candidate_pairs=int(tiles.sum()) * block * block, tiles=tiles)


def convert(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k: convert(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [convert(v) for v in value]
    return value


def main():
    results = {}
    # A worked probability example, not a graded default.
    weights = np.array([.5, .25, .25]); values = np.array([2., -1., 4.])
    keep = np.array([0, 1]); reduced = weights[keep] / weights[keep].sum()
    results['removed_mass_worked'] = dict(weights=weights, values=values,
        original=float(weights @ values), selected=float(reduced @ values[keep]),
        removed_mass=.25, bound=2 * 4 * .25)
    # Fresh I1 numerical case: original scores log(1),log(3),log(2),log(4).
    fresh_weights = np.array([1., 3., 2., 4.]) / 10
    fresh_values = np.array([[-2., 1.], [1., 2.], [3., -1.], [0., 4.]])
    fresh_keep = [0, 2, 3]
    results['removed_mass_fresh'] = dict(weights=fresh_weights, values=fresh_values,
        keep=fresh_keep, original=fresh_weights @ fresh_values,
        selected=(fresh_weights[fresh_keep] @ fresh_values[fresh_keep]) / .7,
        removed_mass=.3, constant_value_null=[2., -1.])
    graph = {}
    for name, length, window, hub, source, target in (
        ('worked_local', 12, 3, (), 2, 11),
        ('worked_hub6', 12, 3, (6,), 2, 11),
        ('worked_hub0', 12, 3, (0,), 2, 11),
        ('fresh_local', 10, 2, (), 1, 9),
        ('fresh_hub4', 10, 2, (4,), 1, 9),
        ('fresh_hub0', 10, 2, (0,), 1, 9)):
        mask = causal_window(length, window, hub)
        graph[name] = dict(mask=mask, edges=int(mask.sum()), source=source,
                           target=target, reaches_after_two=bool(reachable(mask, 2)[target, source]))
    assert graph['worked_local']['edges'] == 33
    assert graph['worked_hub6']['reaches_after_two'] and not graph['worked_hub0']['reaches_after_two']
    assert graph['fresh_hub4']['reaches_after_two'] and not graph['fresh_hub0']['reaches_after_two']
    results['graphs'] = graph
    generator = np.random.default_rng(23)
    query, key, value = generator.normal(size=(3, 7, 3))
    mask = causal_window(7, 3)
    scores = query @ key.T / math.sqrt(3)
    masked = row_softmax(np.where(mask, scores, -np.inf)) @ value
    gathered = np.stack([selected_attention(query[i], key, value, np.flatnonzero(mask[i])) for i in range(7)])
    results['gather_mask_max_error'] = float(np.max(np.abs(masked-gathered)))
    assert np.allclose(masked, gathered, atol=1e-14)
    results['feature_worked'] = feature_read([2, 1], [[1, 0], [0, 1], [1, 1]], [[2], [-1], [3]])
    k = np.array([[1., 2.], [2., 1.], [1., 1.]])
    v = np.array([[1., -1.], [0., 2.], [3., 1.]])
    fresh = feature_read([1, 3], k, v)
    changed = v.copy(); changed[2] = [-1, 3]
    results['feature_fresh'] = dict(query=[1, 3], keys=k, values=v, baseline=fresh,
        changed_values=changed, changed=feature_read([1, 3], k, changed),
        constant_null=feature_read([1, 3], k, np.tile([2., -2.], (3, 1))),
        evict_first=feature_read([1, 3], k[1:], v[1:]))
    results['collision'] = [feature_read([1], [[1], [1]], values)
                            for values in ([[1], [3]], [[2], [2]])]
    # One learned sequence summary: future values leak if formed from all rows.
    results['projection_worked'] = dict(coefficients=[.5, 0, .5], values=[1, 2, 9],
        full_output=5., prefix0_output=.5, changed_future_output=1.)
    coefficients = np.array([.25, .5, 0, .25]); values = np.array([4., -2., 3., 8.])
    changed = values.copy(); changed[3] = -4
    results['projection_fresh'] = dict(coefficients=coefficients, values=values,
        changed_values=changed, full_output=float(coefficients @ values),
        changed_full_output=float(coefficients @ changed), prefix1_output=float(coefficients[:2] @ values[:2]))
    diagonal = np.eye(8, dtype=bool)
    scattered = np.zeros((8, 8), bool); scattered[np.arange(8), 2*np.arange(8) % 8] = True
    results['tiles_worked'] = dict(diagonal=occupied_tiles(diagonal), scattered=occupied_tiles(scattered))
    fresh_a = np.zeros((8, 8), bool); fresh_a[:4, :2] = True
    fresh_b = np.zeros((8, 8), bool); fresh_b[np.arange(8), (3*np.arange(8)) % 8] = True
    results['tiles_fresh'] = dict(clustered=occupied_tiles(fresh_a), dispersed=occupied_tiles(fresh_b))
    # Small Nyström example; the pseudoinverse also supports singular matrices.
    q = np.array([[1., 0.], [0., 1.], [1., 1.], [-1., 0.]])
    k = np.array([[1., 1.], [2., 0.], [0., -1.], [-1., 1.]])
    v = np.array([[1.], [-1.], [2.], [0.]])
    q_land = q.reshape(2, 2, 2).mean(1); k_land = k.reshape(2, 2, 2).mean(1)
    front = row_softmax(q @ k_land.T / math.sqrt(2))
    middle = row_softmax(q_land @ k_land.T / math.sqrt(2))
    back = row_softmax(q_land @ k.T / math.sqrt(2))
    approximate = front @ np.linalg.pinv(middle) @ back
    results['nystrom_worked'] = dict(query=q, key=k, value=v, query_landmarks=q_land,
        key_landmarks=k_land, front=front, middle=middle, back=back,
        approximate_weights=approximate, approximate_output=approximate @ v,
        full_output=row_softmax(q @ k.T / math.sqrt(2)) @ v)
    # Gaussian-marginal sampler check: orthogonality is exact within each block.
    orthogonal = gaussian_orthogonal_rows(np.random.default_rng(31), 12, 4)
    for block in orthogonal.reshape(3, 4, 4):
        unit = block / np.linalg.norm(block, axis=1, keepdims=True)
        assert np.allclose(unit @ unit.T, np.eye(4), atol=1e-14)
    results['random_features'] = dict(orthogonal_projection=orthogonal,
        fixed_radius_1d_expectation=math.exp(-1)*math.cosh(2), gaussian_1d_expectation=math.e,
        iid_variance_x1_y1_m64=math.exp(2)*math.expm1(4)/64)
    # Fresh gated random-feature operator, not the manuscript's trained-head data.
    query = np.array([[.1, .4], [.6, -.2], [-.3, .7], [.2, .5]])
    key = np.array([[.4, -.1], [.2, .8], [-.5, .3], [.6, .2]])
    value = np.array([[1., 0.], [-1., 2.], [3., 1.], [0., -2.]])
    exact = row_softmax(np.where(np.tri(4, dtype=bool), query @ key.T, -np.inf)) @ value
    projections = np.random.default_rng(5).normal(size=(64, 2))
    changed_key = key.copy(); changed_key[1] = [-.4, 1.2]
    results['random_features_fresh'] = dict(query_scaled=query, key_scaled=key, value=value,
        projections=projections, reference_output=exact,
        m8=positive_random_attention(query, key, value, projections[:8])[0],
        m64=positive_random_attention(query, key, value, projections)[0],
        changed_key=changed_key,
        changed_m8=positive_random_attention(query, changed_key, value, projections[:8])[0],
        constant_value_null=positive_random_attention(query, key, np.tile([2., -1.], (4, 1)), projections[:8])[0])
    # Independent practice answers, different from both explanations and labs.
    results['practice'] = dict(window_L20_W4_edges=int(causal_window(20, 4).sum()),
        feature=feature_read([2, 1], [[1, 2], [3, 1]], [[-2], [4]]),
        cache_dense_L2048_H4_d16_bytes=2048*4*16*2*4,
        cache_kernel_H4_m24_dv16_bytes=4*24*17*4,
        cache_window_H4_W32_d16_bytes=4*32*16*2*4)
    (ROOT / 'mechanism-results.json').write_text(json.dumps(convert(results), separators=(',', ':')), encoding='utf-8')
    print('PASS: gathered sparse equality, causal graph reach, state reads and controls, projection leakage, tile counts, Gaussian-marginal sampler and fresh gated fixtures.')
    print(json.dumps(convert({key: results[key] for key in ('feature_fresh','projection_fresh','tiles_fresh','practice')}), indent=2))


if __name__ == '__main__':
    main()
