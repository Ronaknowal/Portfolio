"""Sparse weighted graph operators and explicit library conventions.

Run: python laplacian-library-bridge.py
Dependencies: numpy==2.3.5, scipy==1.18.1, networkx==3.6.1.
Undirected nonnegative edge weights; repeated edges add; a loop enters A once.
CSR storage and matrix actions use O(V+E) space/work after sparse assembly.
COO->CSR may sort indices. Sparse factorization can have fill; it is not O(E).
"""
import numpy as np
from scipy.sparse import coo_array, diags
from scipy.sparse.csgraph import connected_components, laplacian
from scipy.sparse.linalg import spsolve
import networkx as nx


def adjacency(node_count, edges):
    rows, cols, values = [], [], []
    for i, j, weight in edges:
        if not (0 <= i < node_count and 0 <= j < node_count) or not np.isfinite(weight) or weight < 0:
            raise ValueError("Finite nonnegative weights and in-range node indices required")
        rows.append(i); cols.append(j); values.append(weight)
        if i != j:
            rows.append(j); cols.append(i); values.append(weight)
    matrix = coo_array((values, (rows, cols)), shape=(node_count, node_count), dtype=float).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()  # A zero-weight edge is not a connectivity edge.
    return matrix


def operators(matrix):
    degree = np.asarray(matrix.sum(axis=1)).ravel()
    ordinary = diags(degree, format="csr") - matrix
    inverse_root = np.zeros_like(degree)
    np.divide(1., np.sqrt(degree), out=inverse_root, where=degree > 0)
    scaling = diags(inverse_root, format="csr")
    return ordinary.tocsr(), (scaling @ ordinary @ scaling).tocsr()


def harmonic_extension(matrix, anchors, values):
    """Solve L_uu*u=-L_ub*b only when every component has a fixed anchor."""
    anchors, values = np.asarray(anchors, int), np.asarray(values, float)
    if (anchors.ndim != 1 or values.shape != anchors.shape
            or len(np.unique(anchors)) != len(anchors)
            or np.any(anchors < 0) or np.any(anchors >= matrix.shape[0])
            or not np.isfinite(values).all()):
        raise ValueError("Use distinct valid anchors and matching finite values")
    count, labels = connected_components(matrix, directed=False)
    if len(np.unique(labels[anchors])) != count:
        raise ValueError("Every component, including an isolate, needs an anchor")
    unknown = np.setdiff1d(np.arange(matrix.shape[0]), anchors)
    result = np.empty(matrix.shape[0]); result[anchors] = values
    ordinary, _ = operators(matrix)
    if len(unknown):
        block = ordinary[unknown][:, unknown]
        rhs = -ordinary[unknown][:, anchors] @ values
        result[unknown] = spsolve(block, rhs)
    return result


def main():
    edges = [(0, 1, 1.), (0, 1, 1.), (1, 2, 1.)]
    matrix = adjacency(4, edges)
    graph = nx.MultiGraph(); graph.add_nodes_from(range(4)); graph.add_weighted_edges_from(edges)
    reference = nx.to_scipy_sparse_array(graph, nodelist=list(range(4)), format="csr")
    np.testing.assert_allclose(matrix.toarray(), reference.toarray())
    ordinary, normalized = operators(matrix)
    np.testing.assert_allclose(ordinary.toarray(), laplacian(matrix).toarray())
    np.testing.assert_allclose(normalized.toarray(), laplacian(matrix, normed=True).toarray())
    values = harmonic_extension(matrix, [0, 2, 3], [0., 3., 7.])
    np.testing.assert_allclose(values, [0, 1, 3, 7])
    print("component count", connected_components(matrix, directed=False)[0])
    print("harmonic values", values.tolist())
    looped = adjacency(4, edges + [(1, 1, 4.)])
    loop_l, loop_normalized = operators(looped)
    scipy_normalized = laplacian(looped, normed=True).toarray()
    np.testing.assert_allclose(loop_l.toarray(), ordinary.toarray())
    print("loop leaves ordinary L unchanged", True)
    print("looped normalized center: row-sum / SciPy", round(float(loop_normalized[1, 1]), 6), round(float(scipy_normalized[1, 1]), 6))
    print("isolated normalized row", loop_normalized.toarray()[3].tolist())
    try:
        harmonic_extension(matrix, [0, 2], [0., 3.])
    except ValueError:
        print("unanchored isolate rejected", True)


if __name__ == "__main__":
    main()
