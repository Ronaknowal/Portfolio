"""Isotonic calibration by pool-adjacent-violators, from section 3.

`fit_pav` is copied verbatim from calibration_calculations.py. Equal scores are grouped
before any merging, and a merge takes the count-weighted mean of two blocks rather than
the average of their two means. The final line prints both so the difference is visible.
The function checks its own result against scikit-learn's independent implementation.

Needs numpy and scikit-learn. Run: python monotone_map.py
"""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def fit_pav(scores, labels):
    """Group equal scores first; stack PAV is linear after sorting."""
    s, y = np.asarray(scores, float), np.asarray(labels, float)
    if s.ndim != 1 or y.ndim != 1 or not len(s) or len(s) != len(y):
        raise ValueError("Need equally sized nonempty score and label vectors")
    if np.any(~np.isfinite(s)) or np.any((y != 0) & (y != 1)):
        raise ValueError("Scores must be finite and labels binary")
    order = np.argsort(s, kind="stable")
    knots, first, counts = np.unique(s[order], return_index=True, return_counts=True)
    sums = np.add.reduceat(y[order], first)
    blocks, trace = [], []
    for j, (total, count) in enumerate(zip(sums, counts)):
        blocks.append([j, j, float(total), int(count)])
        while len(blocks) >= 2 and blocks[-2][2]/blocks[-2][3] > blocks[-1][2]/blocks[-1][3]:
            right, left = blocks.pop(), blocks.pop()
            merged = [left[0], right[1], left[2]+right[2], left[3]+right[3]]
            blocks.append(merged)
            trace.append({"left": left, "right": right, "merged": merged})
    fitted = np.empty(len(knots))
    for start, end, total, count in blocks:
        fitted[start:end+1] = total/count
    expected = IsotonicRegression(out_of_bounds="clip").fit(s, y).predict(knots)
    np.testing.assert_allclose(fitted, expected)
    return {"knots": knots.tolist(), "fitted": fitted.tolist(), "blocks": blocks, "merges": trace}


if __name__ == "__main__":
    scores = [-3, -2, -1, 0, 1, 2, 3, 4]
    labels = [0, 1, 0, 0, 1, 0, 1, 1]
    fit = fit_pav(scores, labels)
    print("scores", fit["knots"])
    print("labels", [float(label) for label in labels])
    print("fitted", [round(value, 6) for value in fit["fitted"]])
    for step, merge in enumerate(fit["merges"], start=1):
        left, right, merged = merge["left"], merge["right"], merge["merged"]
        print(f"merge {step}: blocks {left[0]}-{left[1]} at {left[2] / left[3]:.6f}"
              f" and {right[0]}-{right[1]} at {right[2] / right[3]:.6f}"
              f" pool to {merged[2] / merged[3]:.6f} over {merged[3]} observations")
    tied = fit_pav([-2, -2, 0, 1], [1, 0, 0, 1])
    print("tied scores", tied["knots"])
    print("tied fitted", [round(value, 6) for value in tied["fitted"]])
    print(f"count weighted {tied['fitted'][0]:.6f}, not the average of block means {(0.5 + 0.0) / 2:.6f}")
