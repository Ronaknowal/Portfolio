"""Reliability bins and the binned ECE summary, from section 2.

The `reliability` function below is copied verbatim from the lesson's complete program,
calibration_calculations.py. Its guard block is that program's own: an internal boundary
belongs to the bin on its right and a forecast of exactly 1 belongs to the last bin, and
an edge list that does not span [0, 1] would silently drop observations.

Needs numpy. Run: python reliability_bins.py
"""
from __future__ import annotations

import numpy as np


def reliability(probabilities, outcomes, edges):
    p, y, edges = np.asarray(probabilities, float), np.asarray(outcomes, float), np.asarray(edges, float)
    if p.ndim != 1 or y.ndim != 1 or len(p) != len(y) or not len(p):
        raise ValueError("Need equally sized nonempty prediction and outcome vectors")
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)) or np.any((y != 0) & (y != 1)):
        raise ValueError("Probabilities must be finite in [0,1] and outcomes binary")
    if edges.ndim != 1 or len(edges) < 2 or np.any(~np.isfinite(edges)) or edges[0] != 0 or edges[-1] != 1 or np.any(np.diff(edges) <= 0):
        raise ValueError("Increasing bin boundaries must span [0,1]")
    # Internal boundary belongs to the bin on its right; p=1 belongs to last.
    indices = np.minimum(np.searchsorted(edges, p, side="right") - 1, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        selected = indices == b
        count = int(selected.sum())
        if not count:
            rows.append({"bin": b, "count": 0, "mean_p": None, "fraction_positive": None})
            continue
        positive = int(y[selected].sum())
        rows.append({"bin": b, "count": count, "positive": positive,
                     "mean_p": float(p[selected].mean()), "fraction_positive": positive / count,
                     "source_indices": np.flatnonzero(selected).tolist()})
    ece = sum(r["count"] * abs(r["mean_p"] - r["fraction_positive"]) for r in rows if r["count"]) / len(p)
    return {"bins": rows, "ece": ece, "count": len(p)}


if __name__ == "__main__":
    forecasts = [.2] * 5 + [.8] * 5
    outcomes = [1, 1, 0, 0, 0, 1, 1, 1, 0, 0]
    repaired = [.4] * 5 + [.6] * 5
    settings = [
        ("two bins", forecasts, [0, .5, 1]),
        ("one bin", forecasts, [0, 1]),
        ("two bins, forecasts replaced by the observed fractions", repaired, [0, .5, 1]),
    ]
    for name, probabilities, edges in settings:
        report = reliability(probabilities, outcomes, edges)
        print(f"{name}: ECE {report['ece']:.6f} over {report['count']} cards")
        for row in report["bins"]:
            if row["count"]:
                print(f"    bin {row['bin']}: {row['count']} cards, {row['positive']} positive,"
                      f" mean forecast {row['mean_p']:.3f}, observed fraction {row['fraction_positive']:.3f}")
            else:
                print(f"    bin {row['bin']}: empty, so it has no observed fraction")
    boundary = reliability([0, .5, 1], [0, 1, 1], [0, .5, 1])
    counts = [row["count"] for row in boundary["bins"]]
    print(f"boundary fixture: counts {counts}, ECE {boundary['ece']:.6f}")
