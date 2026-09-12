"""Bounded content-stage checks for added shape and unit-counterexample fixtures."""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score

angles_inner = 2 * np.pi * np.arange(12) / 12
angles_outer = 2 * np.pi * np.arange(36) / 36
inner = np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
outer = 3 * np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
X = np.vstack([inner, outer])
ring = np.r_[np.zeros(12, dtype=int), np.ones(36, dtype=int)]
db = DBSCAN(eps=.6, min_samples=3).fit(X)
km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)


def record(points, eps):
    fit = DBSCAN(eps=eps, min_samples=2).fit(points)
    distances = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    return {"labels": fit.labels_.tolist(), "neighbors": (distances <= eps).astype(int).tolist()}


corners = np.array([[0., 0], [1, 0], [0, 2], [1, 2]])
extended = np.vstack([corners, [3, 0]])
result = {
    "checked_at": datetime.now(timezone.utc).isoformat(),
    "scope": "Small author fixture calculations only; no implementation or independent verification phase.",
    "shape": {
        "coordinates": X.tolist(), "ring_ids": ring.tolist(),
        "dbscan_labels": db.labels_.tolist(), "dbscan_core_count": len(db.core_sample_indices_),
        "dbscan_ari": adjusted_rand_score(ring, db.labels_),
        "kmeans_labels": km.labels_.tolist(), "kmeans_centers": km.cluster_centers_.tolist(),
        "kmeans_ari": adjusted_rand_score(ring, km.labels_),
    },
    "unit_four_corner_null": {"original": record(corners, 1), "one_axis_and_radius_times100": record(corners * [1, 100], 100)},
    "unit_five_row_contrast": {
        "original": record(extended, 1), "one_axis_and_radius_times100": record(extended * [1, 100], 100),
        "uniform_conversion_null": record(extended * 100, 100),
        "one_axis_restored_metric_null": record((extended * [1, 100]) * [1, .01], 1),
    },
    "stability": {"parent": 6 * (3 - 1), "children_exit6": 6 * (6 - 3), "children_exit4": 6 * (4 - 3), "tie_exit5": 6 * (5 - 3)},
}
Path(__file__).with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps({"shape": {k:v for k,v in result['shape'].items() if k.endswith('ari') or k.endswith('count')}, "unit_contrast": result['unit_five_row_contrast'], "stability": result['stability']}, indent=2))
