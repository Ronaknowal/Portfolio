"""Match a range-finder mechanism to scikit-learn. CPU, float64, no timing claim.
Install: python -m pip install numpy==2.3.5 scipy==1.18.1 scikit-learn==1.9.1
Run: python randomized_svd_library.py
"""
import numpy as np
from sklearn.utils.extmath import randomized_svd


def range_svd(matrix, rank, oversamples, iterations, seed):
    """QR-stabilized counterpart of the earlier SVD-orthogonalized range finder.

    This bridge accepts finite, real, nonempty tall matrices, 1 <= k <= n,
    0 <= p <= n-k and q >= 0. It intentionally does not transpose wide input.
    """
    if np.iscomplexobj(matrix):
        raise ValueError("Expected real entries")
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or not all(matrix.shape) or matrix.shape[0] < matrix.shape[1]:
        raise ValueError("Expected a nonempty tall matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("Expected finite entries")
    if any(type(value) is not int for value in (rank, oversamples, iterations)):
        raise ValueError("Rank, oversamples and iterations must be integers")
    if not 1 <= rank <= matrix.shape[1] or not 0 <= oversamples <= matrix.shape[1] - rank or iterations < 0:
        raise ValueError("Invalid rank, oversampling or power budget")
    # RandomState deliberately matches this library's random-number contract.
    probes = np.random.RandomState(seed).normal(size=(matrix.shape[1], rank + oversamples))
    basis = probes
    for _ in range(iterations):
        basis, _ = np.linalg.qr(matrix @ basis, mode="reduced")
        basis, _ = np.linalg.qr(matrix.T @ basis, mode="reduced")
    basis, _ = np.linalg.qr(matrix @ basis, mode="reduced")
    left, values, right = np.linalg.svd(basis.T @ matrix, full_matrices=False)
    return basis @ left[:, :rank], values[:rank], right[:rank]


def check(matrix, rank, oversamples, iterations):
    own = range_svd(matrix, rank, oversamples, iterations, 7)
    tool = randomized_svd(matrix, n_components=rank, n_oversamples=oversamples,
                          n_iter=iterations, power_iteration_normalizer="QR",
                          transpose=False, flip_sign=False, random_state=7)
    own_fit = (own[0] * own[1]) @ own[2]
    tool_fit = (tool[0] * tool[1]) @ tool[2]
    assert np.allclose(own_fit, tool_fit, atol=1e-10, rtol=1e-10)
    assert np.allclose(own[1], tool[1], atol=1e-10, rtol=1e-10)
    assert np.allclose(own[0].T @ own[0], np.eye(rank), atol=1e-12)
    error = np.linalg.norm(matrix - tool_fit)
    optimum = np.linalg.norm(np.linalg.svd(matrix, compute_uv=False)[rank:])
    assert error + 1e-10 >= optimum
    return error, optimum, np.linalg.norm(own_fit - tool_fit)


def main():
    rng = np.random.default_rng(19)
    left, _ = np.linalg.qr(rng.normal(size=(100, 40)))
    right, _ = np.linalg.qr(rng.normal(size=(40, 40)))
    matrix = (left * np.geomspace(10, .01, 40)) @ right.T
    for iterations in (0, 2):
        error, optimum, difference = check(matrix, 5, 4, iterations)
        print(f"q={iterations}: residual {error:.6f}; best rank-5 {optimum:.6f}; route difference {difference:.2e}")
    # A zero matrix and exact low rank are meaningful algorithmic edges.
    for edge in (np.zeros((8, 4)), np.outer(np.arange(8.), np.arange(4.))):
        check(edge, 2, 2, 1)
    for invalid in (np.array([[np.nan]]), np.ones((2, 3)), np.array([[1j]])):
        try:
            range_svd(invalid, 1, 0, 0, 7)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid input was accepted")
    print("Matched reconstructions, singular values, orthogonality, rank bound and edge contracts: passed")


if __name__ == "__main__":
    main()
