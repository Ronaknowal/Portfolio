"""Same fitted probabilities, explicit entropy, and one paid label at a time.

Install numpy, scikit-learn and scikit-activeml. The invented one-dimensional
pool tests API/state contracts; it is not an accuracy experiment.
"""
import numpy as np
from scipy.special import xlogy
from sklearn.linear_model import LogisticRegression
from skactiveml.classifier import SklearnClassifier
from skactiveml.pool import UncertaintySampling


def entropy(probabilities):
    return -np.sum(xlogy(probabilities, probabilities), axis=1)


def acquire_one(x, known, classifier, strategy, oracle, remaining):
    """The caller supplies acquired labels only. IDs index the original pool."""
    eligible = np.flatnonzero(np.isnan(known))
    if remaining <= 0 or eligible.size == 0:
        return known.copy(), remaining, None
    probabilities = classifier.predict_proba(x[eligible])
    scratch = entropy(probabilities)
    selected, utilities = strategy.query(
        X=x, y=known, clf=classifier, fit_clf=False,
        candidates=eligible, batch_size=1, return_utilities=True)
    np.testing.assert_allclose(utilities[0, eligible], scratch, atol=1e-12)
    chosen = int(selected[0])
    best = eligible[np.isclose(scratch, scratch.max(), rtol=0, atol=1e-12)]
    assert chosen in best
    updated = known.copy()
    answer = oracle(chosen)
    if answer not in classifier.classes_:
        raise ValueError("Oracle label outside the declared class vocabulary")
    updated[chosen] = answer
    classifier.fit(x, updated)
    return updated, remaining - 1, (chosen, float(utilities[0, chosen]))


def main():
    x = np.array([-3., -2., -1., 0.1, 1.1, 1.8, 3., 4.])[:, None]
    known = np.array([0., 0., np.nan, np.nan, np.nan, np.nan, 1., 1.])
    classifier = SklearnClassifier(
        LogisticRegression(C=1.0, solver="lbfgs", max_iter=300, tol=1e-10),
        classes=[0, 1], missing_label=np.nan)
    classifier.fit(x, known)
    strategy = UncertaintySampling(method="entropy", missing_label=np.nan, random_state=19)
    calls = []
    def oracle(row):
        calls.append(row)
        return int(x[row, 0] >= 0.5)
    remaining = 3
    while remaining:
        known, remaining, result = acquire_one(x, known, classifier, strategy, oracle, remaining)
        print("row_utility_labels_remaining", result[0], round(result[1], 6), int(np.isfinite(known).sum()), remaining)
    count = len(calls)
    _, _, result = acquire_one(x, known, classifier, strategy, oracle, 0)
    assert result is None and len(calls) == count
    complete = np.where(np.isnan(known), 0.0, known)
    _, _, result = acquire_one(x, complete, classifier, strategy, oracle, 1)
    assert result is None and len(calls) == count
    print("oracle_calls", calls)
    print("final_probabilities", np.round(classifier.predict_proba(x)[:, 1], 4).tolist())


if __name__ == "__main__":
    main()
