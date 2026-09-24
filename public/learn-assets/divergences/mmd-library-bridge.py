"""Blockwise RBF MMD, Gram-matrix oracle, and an exact SciPy permutation test.

Run: python mmd-library-bridge.py
Dependencies: numpy==2.3.5, scipy==1.18.1, scikit-learn==1.9.1.
The production-style sum uses O(B^2) kernel scratch memory, not an N^2 Gram
matrix; time remains quadratic in observations times feature dimension.
The tiny exact permutation example intentionally caches its six-point Gram.
"""
import math
import numpy as np
from scipy.spatial.distance import cdist, jensenshannon
from scipy.special import rel_entr
from scipy.stats import permutation_test
from sklearn.metrics.pairwise import rbf_kernel


def kernel_sum(x, y, sigma, block_size=256, exclude_diagonal=False):
    """Accumulate positive block sums; diagonal exclusion is for the same sample."""
    def blocks():
        for i in range(0, len(x), block_size):
            for j in range(0, len(y), block_size):
                # Direct differences avoid cancellation in ||x||²+||y||²-2x.y.
                kernel = np.exp(-cdist(x[i:i+block_size], y[j:j+block_size], "sqeuclidean") / (2*sigma*sigma))
                if exclude_diagonal and i == j:
                    np.fill_diagonal(kernel, 0.)
                yield float(kernel.sum())
    return math.fsum(blocks())


def mmd2(x, y, sigma=1., block_size=256, unbiased=False):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if (x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]
            or min(len(x), len(y)) < (2 if unbiased else 1)
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or not np.isfinite(sigma) or sigma <= 0
            or not isinstance(block_size, int) or block_size < 1):
        raise ValueError("Finite row samples with matching features, positive bandwidth and block size required")
    n, m = len(x), len(y)
    xx = kernel_sum(x, x, sigma, block_size, unbiased)
    yy = kernel_sum(y, y, sigma, block_size, unbiased)
    xy = kernel_sum(x, y, sigma, block_size)
    return xx / (n*(n-1) if unbiased else n*n) + yy / (m*(m-1) if unbiased else m*m) - 2*xy/(n*m)


def gram_statistic(gram, left, right, unbiased=False):
    left, right = np.asarray(left, int), np.asarray(right, int)
    xx, yy = gram[np.ix_(left, left)], gram[np.ix_(right, right)]
    n, m = len(left), len(right)
    if unbiased:
        xx_sum, yy_sum = xx.sum()-np.trace(xx), yy.sum()-np.trace(yy)
    else:
        xx_sum, yy_sum = xx.sum(), yy.sum()
    return xx_sum/(n*(n-1) if unbiased else n*n) + yy_sum/(m*(m-1) if unbiased else m*m) - 2*gram[np.ix_(left, right)].mean()


def main():
    x, y = np.array([[0.], [.2], [.4]]), np.array([[1.2], [1.4], [1.6]])
    pooled = np.vstack((x, y))
    gram = rbf_kernel(pooled, gamma=1/(2*.5**2))
    left, right = np.arange(3), np.arange(3, 6)
    for unbiased in (False, True):
        value = mmd2(x, y, .5, block_size=2, unbiased=unbiased)
        oracle = gram_statistic(gram, left, right, unbiased)
        np.testing.assert_allclose(value, oracle, atol=1e-14)
        print("unbiased", unbiased, "MMD squared", round(value, 9))
    statistic = lambda a, b: gram_statistic(gram, a, b)
    test = permutation_test((left, right), statistic, permutation_type="independent",
                            vectorized=False, n_resamples=np.inf, alternative="greater")
    print("exact allocation count", len(test.null_distribution), "right-tail p", round(float(test.pvalue), 6))
    p, q = np.array([.75, .25]), np.array([.5, .5])
    midpoint = (p+q)/2
    js = (rel_entr(p, midpoint).sum() + rel_entr(q, midpoint).sum()) / 2
    np.testing.assert_allclose(jensenshannon(p, q)**2, js, atol=1e-15)
    print("JS nats / SciPy distance squared", np.round([js, jensenshannon(p, q)**2], 9).tolist())


if __name__ == "__main__":
    main()
