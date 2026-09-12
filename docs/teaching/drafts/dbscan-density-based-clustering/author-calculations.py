"""Small stage-1/2 author probes, not a production/native verification suite.

Run from this directory using the existing lesson-tools Python environment.
Creates the independently runnable offline Iris input and compact fixture evidence.
"""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS, cluster_optics_dbscan
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
STREET = np.array([-1.75, -1.5, -1.25, -1, 1, 1.25, 1.5, 1.75, 0, 4])[:, None]
VARIED = np.array([0, .125, .25, .375, .75, .875, 1, 1.125, 5, 5.75, 6.5, 7.25])[:, None]


def report(points, eps, minimum):
    model = DBSCAN(eps=eps, min_samples=minimum).fit(points)
    labels = model.labels_
    core = np.zeros(len(points), dtype=bool)
    core[model.core_sample_indices_] = True
    assigned = labels >= 0
    clusters = sorted(set(labels) - {-1})
    return {
        "eps": eps, "min_samples": minimum,
        "labels": labels.tolist(),
        "core_ids": np.flatnonzero(core).tolist(),
        "border_ids": np.flatnonzero(assigned & ~core).tolist(),
        "noise_ids": np.flatnonzero(~assigned).tolist(),
        "clusters": len(clusters),
        "sizes": [int(np.sum(labels == group)) for group in clusters],
    }


iris = load_iris()
csv_path = HERE / "iris.csv"
with csv_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["row_id", "sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm", "species"])
    writer.writerows([index, *[f"{value:.1f}" for value in row], int(iris.target[index])] for index, row in enumerate(iris.data))

street = [report(STREET, radius, 4) for radius in [.125, .5, .75, 1, 1.25]]
distance = np.abs(STREET - STREET.T)
reverse = report(STREET[::-1], 1, 4)
reverse["labels_in_original_order"] = reverse["labels"][::-1]
real = []
standardized = StandardScaler().fit_transform(iris.data)
for representation, points, settings in [
    ("four original cm features", iris.data, [(.5, 5)]),
    ("four standardized features", standardized, [(.3, 5), (.5, 5), (.8, 5), (1, 5), (.5, 3), (.5, 10)]),
]:
    for eps, minimum in settings:
        result = report(points, eps, minimum)
        labels = np.array(result["labels"])
        assigned = labels >= 0
        result.update({
            "representation": representation,
            "coverage": float(assigned.mean()),
            "silhouette_assigned": float(silhouette_score(points[assigned], labels[assigned])) if 2 <= result["clusters"] < assigned.sum() else None,
            "ari_all_rows_noise_is_one_label": float(adjusted_rand_score(iris.target, labels)),
            "ari_assigned": float(adjusted_rand_score(iris.target[assigned], labels[assigned])) if assigned.any() else None,
        })
        real.append(result)

opt = OPTICS(min_samples=4, max_eps=2).fit(STREET)
optics = {
    "ordering": opt.ordering_.tolist(),
    "core_distances": [None if not np.isfinite(v) else float(v) for v in opt.core_distances_],
    "ordered_reachability": [None if not np.isfinite(v) else float(v) for v in opt.reachability_[opt.ordering_]],
    "eps1_labels": cluster_optics_dbscan(reachability=opt.reachability_, core_distances=opt.core_distances_, ordering=opt.ordering_, eps=1).tolist(),
}
core_distance = np.sort(distance, axis=1)[:, 3]
mutual = np.maximum(np.maximum(core_distance[:, None], core_distance[None, :]), distance)
np.fill_diagonal(mutual, 0)
hierarchy = HDBSCAN(min_cluster_size=4, min_samples=4, copy=True).fit(STREET)

result = {
    "checked_at": datetime.now(timezone.utc).isoformat(),
    "scope": "Elementary manuscript arithmetic and fixture contrasts; no phase-two implementation, full output campaign or independent review.",
    "versions": {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
    "iris_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
    "street": street,
    "street_neighbor_counts_eps1": (distance <= 1).sum(axis=1).tolist(),
    "street_core_distance_m4": core_distance.tolist(),
    "reverse_order": reverse,
    "varied": [report(VARIED, radius, 3) for radius in [.25, .375, .75]],
    "varied_null": report(np.r_[VARIED[:8, 0], [5, 5.125, 5.25, 5.375]][:, None], .25, 3),
    "duplicate_null": report(np.array([[2.], [2.], [2.], [2.]]), .125, 4),
    "m1_null": report(STREET, .125, 1),
    "optics": optics,
    "mutual_selected": {"D_I": float(mutual[3, 8]), "I_E": float(mutual[8, 4]), "A_B": float(mutual[0, 1])},
    "hdbscan_street": {"labels": hierarchy.labels_.tolist(), "probabilities": hierarchy.probabilities_.tolist()},
    "iris": real,
}
(HERE / "author-calculations.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps({"versions": result["versions"], "street": street, "varied": result["varied"], "optics": optics, "iris": [{key: value for key, value in row.items() if key not in {"labels", "core_ids", "border_ids", "noise_ids"}} for row in real]}, indent=2))
