"""The split-conformal rank, threshold, prediction sets and rotation, from section 5.

All four functions are copied verbatim from calibration_calculations.py. The rank uses
exact rational arithmetic on alpha and selects an indexed order statistic; it is not a
percentile call, and the two NumPy quantile conventions printed below return two other
numbers on the same data. When the requested rank exceeds the calibration set, the
threshold is infinity, which means every candidate answer is included.

Needs numpy. Run: python conformal_rank.py
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np


def conformal_rank(n, alpha):
    if n < 1:
        raise ValueError("At least one calibration score is required")
    a = Fraction(str(alpha))
    if not 0 < a < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    return math.ceil((n+1)*(1-a))


def conformal_threshold(scores, alpha):
    scores = np.asarray(scores, float)
    if scores.ndim != 1 or not len(scores) or np.any(~np.isfinite(scores)):
        raise ValueError("Need a nonempty vector of finite scores")
    k = conformal_rank(len(scores), alpha)
    return float(np.partition(scores, k-1)[k-1]) if k <= len(scores) else math.inf


def class_sets(probabilities, q):
    return (1 - np.asarray(probabilities, float) <= q).tolist()


def rank_rotation(scores, alpha):
    """Hold each exchangeable position out once; exact conditional rank check."""
    rows = []
    for i, test in enumerate(scores):
        calibration = scores[:i] + scores[i+1:]
        q = conformal_threshold(calibration, alpha)
        rows.append({"test": test, "q": q if math.isfinite(q) else "infinity", "covered": test <= q})
    return {"rows": rows, "covered": sum(r["covered"] for r in rows), "total": len(rows)}


if __name__ == "__main__":
    calibration = [.05, .10, .15, .20, .25, .30, .40, .60, .90]
    n = len(calibration)
    for alpha in (.2, .05):
        k = conformal_rank(n, alpha)
        q = conformal_threshold(calibration, alpha)
        print(f"alpha {alpha}: n {n}, k = ceil({n + 1} * {1 - alpha:.2f}) = {k}, q = {q}")
    q = conformal_threshold(calibration, .2)
    print(f"linear quantile at 8/9 {np.quantile(calibration, 8 / 9):.6f}")
    print(f"higher quantile at 8/9 {np.quantile(calibration, 8 / 9, method='higher'):.6f}")
    print(f"eighth order statistic {q:.6f}")
    for probabilities in ([.80, .15, .05], [.45, .40, .15], [.34, .33, .33]):
        included = class_sets(probabilities, q)
        names = [name for name, keep in zip("ABC", included) if keep]
        shown = "{" + ", ".join(names) + "}" if names else "the empty set"
        print("probabilities", probabilities, "scores",
              [round(1 - value, 2) for value in probabilities], "set", shown)
    rotation = rank_rotation(calibration + [.95], .2)
    print(f"rotation: {rotation['covered']} of {rotation['total']} held-out scores covered")
    tied = rank_rotation([.2] * 10, .2)
    print(f"every score tied at .2: {tied['covered']} of {tied['total']} covered")
