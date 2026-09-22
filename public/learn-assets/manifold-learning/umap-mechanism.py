"""Exact small-neighborhood UMAP graph; explicit, bounded layout mechanisms.

Python 3.12; numpy 2.3.5, scipy 1.18.1, umap-learn 0.5.12.
The full-pair step is a declared teaching objective, not UMAP's sampled trainer.
"""
import numpy as np
from scipy.sparse import csr_matrix


def exact_neighbors(x, k):
    """Self first; distance ties by row ID. O(n^2 d + nk log k), O(nd + nk)."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all() or not 2 <= k <= len(x):
        raise ValueError("Use finite rows and 2 <= k <= n")
    ids = np.empty((len(x), k), dtype=int)
    distances = np.empty((len(x), k), dtype=np.float32)
    try:
        with np.errstate(over="raise", under="raise", invalid="raise"):
            for row in range(len(x)):
                squared = np.sum((x - x[row]) ** 2, axis=1)
                squared[row] = np.inf
                boundary = np.partition(squared, k - 2)[k - 2]
                nearer = np.flatnonzero(squared < boundary)
                tied = np.flatnonzero(squared == boundary)[:k - 1 - len(nearer)]
                chosen = np.concatenate((nearer, tied))
                chosen = chosen[np.lexsort((chosen, squared[chosen]))]
                ids[row] = np.r_[row, chosen]
                distances[row] = np.r_[0.0, np.sqrt(squared[chosen])]
    except FloatingPointError as error:
        raise ValueError("Rescale coordinates: squared or float32 distances underflow or overflow") from error
    return ids, distances


def local_scales(distances, tolerance=1e-5):
    """local_connectivity=1, bandwidth=1, self counted in k; 64 search steps."""
    distances = np.asarray(distances, dtype=float)
    target = np.log2(distances.shape[1])
    sigma, rho = np.empty(len(distances)), np.zeros(len(distances))
    finite = distances[np.isfinite(distances)]
    global_mean = finite.mean() if finite.size else 0.0
    for row, values in enumerate(distances):
        finite_row = values[np.isfinite(values)]
        positive = finite_row[finite_row > 0]
        rho[row] = positive.min() if positive.size else 0.0
        offsets = np.maximum(values[1:] - rho[row], 0)
        low, high, scale = 0.0, np.inf, 1.0
        for _ in range(64):
            mass = np.exp(-offsets / scale).sum()
            if abs(mass - target) < tolerance:
                break
            if mass > target:
                high = scale
            else:
                low = scale
            scale = 2 * scale if np.isinf(high) else (low + high) / 2
        mean = finite_row.mean() if rho[row] > 0 else global_mean
        sigma[row] = max(scale, 1e-3 * mean)
    return sigma, rho


def fuzzy_graph(ids, distances):
    """Use -1/inf for removed neighbors; keep storage proportional to nk."""
    sigma, rho = local_scales(distances)
    rows = np.repeat(np.arange(len(ids)), ids.shape[1])
    columns = ids.ravel()
    valid = (columns >= 0) & (rows != columns) & np.isfinite(distances.ravel())
    rows, columns = rows[valid], columns[valid]
    offset = np.maximum(distances.ravel()[valid] - rho[rows], 0)
    strengths = np.exp(-offset / sigma[rows])
    directed = csr_matrix((strengths, (rows, columns)), shape=(len(ids), len(ids)))
    graph = directed + directed.T - directed.multiply(directed.T)
    graph.eliminate_zeros()
    return graph.tocsr(), directed, sigma, rho


def full_pair_loss_gradient(y, graph):
    """Ideal Bernoulli pair cost with a=b=1, each unordered pair once.

    O(n^2 q) work, O(nq + graph.nnz) storage; small distinct coordinates only.
    """
    gradient = np.zeros_like(y, dtype=float)
    loss = 0.0
    for first in range(len(y) - 1):
        delta = y[first] - y[first + 1:]
        squared = np.sum(delta * delta, axis=1)
        if np.any(squared <= 1e-12):
            raise ValueError("This smooth pair calculation needs separated coordinates")
        weights = graph.getrow(first).toarray()[0, first + 1:]
        loss += np.sum(weights * np.log1p(squared)
                       + (1 - weights) * np.log1p(1 / squared))
        pair_gradient = 2 * (weights - (1 - weights) / squared)[:, None] * delta / (1 + squared[:, None])
        gradient[first] += pair_gradient.sum(axis=0)
        gradient[first + 1:] -= pair_gradient
    return float(loss), gradient


def sampled_edge_step(y, head, tail, negatives, rate):
    """One explicit UMAP-style a=b=gamma=1 update; caller supplies schedule.

    Positive endpoints both move; each negative moves the head only. The 0.001
    repulsion stabilizer and per-coordinate clipping follow the package's rule.
    """
    updated = np.array(y, dtype=float, copy=True)
    delta = updated[head] - updated[tail]
    squared = delta @ delta
    attraction = np.clip(-2 * delta / (1 + squared), -4, 4)
    updated[head] += rate * attraction
    updated[tail] -= rate * attraction
    for other in negatives:
        if other == head:
            continue
        delta = updated[head] - updated[other]
        squared = delta @ delta
        if squared > 0:
            repulsion = np.clip(2 * delta / ((0.001 + squared) * (1 + squared)), -4, 4)
            updated[head] += rate * repulsion
    return updated


def main():
    from umap import UMAP
    from umap.umap_ import fuzzy_simplicial_set
    x = np.array([[0.0], [0.7], [1.9], [4.2], [8.6], [11.3]])
    ids, distances = exact_neighbors(x, 4)
    graph, directed, sigma, rho = fuzzy_graph(ids, distances)
    library, library_sigma, library_rho = fuzzy_simplicial_set(
        x, 4, np.random.RandomState(7), "euclidean",
        knn_indices=ids, knn_dists=distances,
        local_connectivity=1.0, set_op_mix_ratio=1.0)
    np.testing.assert_allclose(rho, library_rho, atol=2e-6)
    np.testing.assert_allclose(sigma, library_sigma, rtol=5e-5, atol=2e-6)
    np.testing.assert_allclose(graph.toarray(), library.toarray(), atol=3e-6)
    reducer = UMAP(n_neighbors=4, local_connectivity=1.0,
                   set_op_mix_ratio=1.0, init="random", n_epochs=100,
                   random_state=7, n_jobs=1).fit(x)
    np.testing.assert_allclose(graph.toarray(), reducer.graph_.toarray(), atol=3e-6)
    print("rho", np.round(rho, 4).tolist())
    print("sigma", np.round(sigma, 4).tolist())
    print("directed_row_0", np.round(directed.toarray()[0], 4).tolist())
    print("fuzzy_graph_max_error", f"{np.max(np.abs(graph.toarray() - library.toarray())):.2e}")
    y = np.array([[-1., 0.2], [-0.4, -0.3], [0.2, 0.1], [0.9, -0.2], [1.5, 0.4], [2.1, -0.1]])
    loss, gradient = full_pair_loss_gradient(y, graph)
    after, _ = full_pair_loss_gradient(y - 0.01 * gradient, graph)
    print("ideal_pair_loss_before_after", round(loss, 6), round(after, 6))
    print("sampled_head_after", np.round(sampled_edge_step(y, 0, 1, [4, 5], 0.01)[0], 6).tolist())


if __name__ == "__main__":
    main()
