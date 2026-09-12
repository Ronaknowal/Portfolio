"""Small content-authoring calculations, not an implementation test suite.

Run with the pinned NumPy/scikit-learn environment in data-provenance.md.
This records the figure geometry and directly compares aligned memberships.
It deliberately does not execute the six manuscript programs or verify a UI.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


folder = Path(__file__).parent
data = np.genfromtxt(folder / "iris.csv", delimiter=",", skip_header=1)
x = data[:, 1:5]
scaler = StandardScaler().fit(x)
z = scaler.transform(x)
pca = PCA(2, svd_solver="full").fit(z)
projected = pca.transform(z)
with threadpool_limits(limits=1):
    labels = KMeans(3, n_init=20, random_state=17).fit_predict(z)
    projected_labels = KMeans(3, n_init=20, random_state=17).fit_predict(projected)
ari = adjusted_rand_score(labels, projected_labels)
same_pairs = np.array_equal(labels[:, None] == labels,
                            projected_labels[:, None] == projected_labels)

angles = 2 * np.pi * (np.arange(16) + .25) / 16
rings = np.concatenate([radius * np.column_stack([np.cos(angles), np.sin(angles)])
                        for radius in [1., 2.]])
ring_labels = np.repeat([0, 1], 16)
slice_labels = (rings[:, 0] >= 0).astype(int)
ring_s = silhouette_samples(rings, ring_labels)
slice_s = silhouette_samples(rings, slice_labels)

weights = {}
for name, w in {
    "unit": [1, 1, 1, 1],
    "petal_width_quarter": [1, 1, 1, .25],
    "petal_width_four": [1, 1, 1, 4],
    "common_four": [4, 4, 4, 4],
}.items():
    values = silhouette_samples(z * np.sqrt(w), labels)
    weights[name] = {"weights": w, "mean": float(values.mean()),
                     "negative": int((values < 0).sum()), "values": values.tolist()}

record = {
    "scope": "Content-stage figure inputs and direct identity check; no browser or complete-program campaign.",
    "data": "iris.csv; IDs remain in original 0..149 order",
    "iris_partition_identity": {"scaled4_labels": labels.tolist(),
                                "pca2_labels": projected_labels.tolist(),
                                "ARI_between_partitions": float(ari),
                                "all_pair_memberships_identical": bool(same_pairs)},
    "iris_view": {"scaler_mean": scaler.mean_.tolist(),
                  "scaler_scale": scaler.scale_.tolist(),
                  "pca_components": pca.components_.tolist(),
                  "unwhitened_pca2_coordinates": projected.tolist()},
    "fixed_partition_rescores": weights,
    "ring_contrast": {"radii": [1., 2.], "points_per_ring": 16,
                      "angle_formula": "2*pi*(j+0.25)/16, j=0..15",
                      "coordinates": rings.tolist(), "ring_labels": ring_labels.tolist(),
                      "slice_labels": slice_labels.tolist(),
                      "ring_silhouettes": ring_s.tolist(), "slice_silhouettes": slice_s.tolist(),
                      "ring_mean": float(ring_s.mean()), "slice_mean": float(slice_s.mean())},
}
(folder / "visual-input-calculations.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"partition_ARI": float(ari), "same_pairs": bool(same_pairs),
                  "ring_mean": float(ring_s.mean()), "slice_mean": float(slice_s.mean()),
                  "rescore_means": {k: v["mean"] for k, v in weights.items()}}, indent=2))
