"""Dense mathematics, sparse eigensolver and SpectralClustering on one graph.
Install: python -m pip install numpy==2.3.5 scipy==1.18.1 scikit-learn==1.9.1
Run: python spectral_library.py
"""
import numpy as np
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans, SpectralClustering
from threadpoolctl import threadpool_limits


def laplacian(affinity):
    affinity = csr_matrix(affinity, dtype=np.float64)
    if affinity.shape[0] != affinity.shape[1] or affinity.shape[0] < 2:
        raise ValueError("Expected a square graph with at least two vertices")
    if not np.isfinite(affinity.data).all() or (affinity.data < 0).any():
        raise ValueError("Affinity must be finite and nonnegative")
    difference = affinity - affinity.T
    if difference.nnz and np.max(np.abs(difference.data)) > 1e-12:
        raise ValueError("Expected a symmetric affinity")
    if np.any(affinity.diagonal() != 0):
        raise ValueError("This contract has no self-loops")
    degrees = np.asarray(affinity.sum(axis=1)).ravel()
    if np.any(degrees <= 0):
        raise ValueError("Handle isolated vertices explicitly before this pipeline")
    scale = diags(1 / np.sqrt(degrees))
    return eye(len(degrees), format="csr") - scale @ affinity @ scale, degrees


def same_partition(left, right):
    return np.array_equal(left[:, None] == left, right[:, None] == right)


def normalized_cut(affinity, labels):
    degree = affinity.sum(axis=1)
    return sum(affinity[np.ix_(labels == group, labels != group)].sum()
               / degree[labels == group].sum() for group in np.unique(labels))


def main():
    affinity = np.zeros((9, 9))
    for start in (0, 3, 6):
        affinity[start:start + 3, start:start + 3] = 1 - np.eye(3)
    affinity[2, 3] = affinity[3, 2] = .08
    affinity[5, 6] = affinity[6, 5] = .08
    operator, degree = laplacian(affinity)
    dense_values, dense_vectors = np.linalg.eigh(operator.toarray())
    values, vectors = eigsh(operator, k=3, which="SM", tol=1e-12,
                            v0=np.random.default_rng(12).normal(size=9))
    order = np.argsort(values)
    values, vectors = values[order], vectors[:, order]
    assert np.allclose(values, dense_values[:3], atol=1e-12)
    residual = np.linalg.norm(operator @ vectors - vectors * values)
    projector_error = np.linalg.norm(vectors @ vectors.T - dense_vectors[:, :3] @ dense_vectors[:, :3].T)
    assert residual < 1e-10 and projector_error < 1e-10
    # This is scikit-learn's generalized-eigenvector scaling, not the earlier
    # Ng-Jordan-Weiss row-unit normalization. Both are compared honestly.
    embedding = vectors / np.sqrt(degree[:, None])
    unit_rows = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    with threadpool_limits(limits=1):
        labels = KMeans(n_clusters=3, n_init=20, random_state=12).fit_predict(embedding)
        unit_labels = KMeans(n_clusters=3, n_init=20, random_state=12).fit_predict(unit_rows)
        tool = SpectralClustering(n_clusters=3, n_components=3, affinity="precomputed",
                                 eigen_solver="arpack", eigen_tol=0., assign_labels="kmeans",
                                 n_init=20, random_state=12).fit_predict(csr_matrix(affinity))
    expected = np.repeat(np.arange(3), 3)
    assert same_partition(labels, tool) and same_partition(labels, expected)
    assert same_partition(unit_labels, expected)  # fixture agreement, not universal equivalence
    assert np.isclose(normalized_cut(affinity, labels), normalized_cut(affinity, tool))
    print(f"Small eigenvalues: {np.round(values, 6).tolist()}")
    print("Dense/sparse eigenspace and residual errors < 1e-10")
    print(f"All routes group ABC / DEF / GHI; normalized cut {normalized_cut(affinity, tool):.6f}")
    # Changing the constraint to k=1 means all vertices form one cluster.
    with threadpool_limits(limits=1):
        one = SpectralClustering(n_clusters=1, n_components=1, affinity="precomputed",
                                 random_state=12, n_init=20).fit_predict(affinity)
    assert len(np.unique(one)) == 1 and normalized_cut(affinity, one) == 0
    try:
        laplacian(np.pad(affinity, ((0, 1), (0, 1))))
    except ValueError:
        print("Changed k=1 has cut 0; isolated vertex rejected by explicit policy")
    else:
        raise AssertionError("Isolated vertex was accepted")


if __name__ == "__main__":
    main()
