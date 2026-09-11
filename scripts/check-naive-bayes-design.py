"""Proposed fixture calculations, not verification of a production implementation."""
from fractions import Fraction as F
from pathlib import Path
import json
import math
import numpy as np
from scipy.special import logsumexp
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB

X = np.array([[2, 1, 0, 0, 0], [1, 1, 1, 0, 0], [0, 0, 0, 1, 1]])
y = np.array([1, 1, 0])
rows = {}
for kind in (MultinomialNB, BernoulliNB, ComplementNB):
    model = kind(alpha=1).fit(X, y)
    rows[kind.__name__] = {
        "word_parameters": model.feature_log_prob_.tolist(),
        "queries": model.predict_proba([[1, 0, 0, 0, 0], [0, 0, 0, 0, 0], [2, 0, 0, 0, 0]]).tolist(),
    }
joint = [F(1, 3) * F(1, 7), F(2, 3) * F(4, 11)]
assert joint[1] / sum(joint) == F(56, 67)
duplicates = []
for copies in range(1, 6):
    p_plus = F(1, 5) * F(4, 5) + F(4, 5) * F(2, 5)
    actual = F(1, 3)
    reported = F(2**copies, 4 + 2**copies)
    accuracy = F(4, 5) if copies <= 2 else F(16, 25)
    duplicates.append({"copies": copies, "true_positive_posterior": str(actual), "nb_positive_posterior": str(reported), "accuracy_ties_normal": str(accuracy), "positive_mass": str(p_plus)})
results = {
    "status": "design fixtures only; not a production pass",
    "three_document_query_free": {"exact_spam_posterior": "56/67", "decimal": float(F(56, 67)), "log_odds": math.log(float(F(56, 11)))},
    "sklearn_fixture_cross_check": rows,
    "duplicates": duplicates,
    "equal_mean_gaussian_crossings": [-math.sqrt(8 * math.log(2) / 3), math.sqrt(8 * math.log(2) / 3)],
    "old_cluster_analytic_boundary": {"center": [-10/3, -10/3], "radius": math.sqrt(200/9 - (6-math.log(4))/.75)},
    "extreme_scores_normalization": np.exp(np.array([-1001, -1000]) - logsumexp([-1001, -1000])).tolist(),
}
target = Path("scratch/naive-bayes-design-fixtures.json")
target.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
print(json.dumps(results, indent=2))
