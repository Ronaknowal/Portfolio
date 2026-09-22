"""Finite count MI, its units, and scikit-learn's discrete feature estimator.

Run: python mutual-information-library-bridge.py
Dependencies: numpy==2.3.5, scikit-learn==1.9.1.
This owns the finite discrete/count convention. Continuous nearest-neighbor MI
is a different estimator; the Feature Selection lesson owns fitted-data workflow.
"""
import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif


def count_mi(counts):
    """Plug-in MI in nats, with empty cells and margins handled without log(0)."""
    counts = np.asarray(counts)
    if (counts.ndim != 2 or not counts.size or not np.issubdtype(counts.dtype, np.integer)
            or np.any(counts < 0)):
        raise ValueError("Pass a nonnegative integer count table, not normalized probabilities")
    total = float(counts.astype(float).sum())
    if total <= 0:
        raise ValueError("At least one observation is required")
    p = counts.astype(float) / total
    rows, cols = p.sum(axis=1), p.sum(axis=0)
    i, j = np.nonzero(counts)
    # Subtracted logs avoid forming a tiny marginal product or its reciprocal.
    return float(np.sum(p[i, j] * (np.log(p[i, j]) - np.log(rows[i]) - np.log(cols[j]))))


def expand_counts(counts):
    i, j = np.indices(counts.shape)
    return np.repeat(i.ravel(), counts.ravel()), np.repeat(j.ravel(), counts.ravel())


def main():
    counts = np.array([[30, 10], [10, 30]])
    x, y = expand_counts(counts)
    manual = count_mi(counts)
    library = mutual_info_score(None, None, contingency=counts)
    feature = mutual_info_classif(x[:, None], y, discrete_features=True, random_state=0)[0]
    np.testing.assert_allclose([library, feature], manual, atol=1e-14)
    print("manual/library/feature nats", np.round([manual, library, feature], 9).tolist())
    print("same MI in bits", round(manual / np.log(2), 9))
    duplicated = count_mi(2 * counts)
    relabeled = count_mi(counts[::-1, ::-1])
    np.testing.assert_allclose([duplicated, relabeled], manual, atol=1e-14)
    print("duplicate rows and relabel invariants", True)
    print("independent table nats", round(count_mi(np.array([[2, 4], [3, 6]])), 9))
    # Unique identifiers memorize the realized sample and produce the entropy of Y.
    identifier_mi = mutual_info_classif(np.arange(len(y))[:, None], y, discrete_features=True)[0]
    print("unique identifier empirical MI in bits", round(identifier_mi / np.log(2), 6))
    print("identifier population relevance established", False)


if __name__ == "__main__":
    main()
