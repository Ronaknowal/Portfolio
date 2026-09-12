"""Generate/verify complete clustering programs and changed native oracles.

Run with the isolated lesson Python; --write captures the actual programs/output
into the topic module. With no flag, execute and verify its exact recorded output.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import ast
import base64
import contextlib
import hashlib
import importlib.metadata
import io
import itertools
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
from fractions import Fraction

import black
import numpy as np
from scipy.cluster.hierarchy import linkage
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits

PROGRAMS = {
    "lloyd": {
        "title": "Implement vanilla D² seeding and Lloyd’s algorithm from scratch",
        "question": "If the last mean update changes the nearest-center assignments, has the algorithm reached a fixed point? What should it return when its iteration budget ends there?",
        "code": r"""
import numpy as np
from sklearn.datasets import make_blobs


def check_points(values):
    # Accept a finite numeric row-by-feature matrix and nothing else.
    points = np.asarray(values, dtype=float)
    if points.ndim != 2 or not np.isfinite(points).all():
        raise ValueError("Use a finite numeric row-by-feature matrix.")
    return points


def squared_distances(points, centers):
    # Return the n × k matrix of squared Euclidean distances.
    return ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)


def d2_seed(points, k, seed=0):
    # Vanilla k-means++: a uniform first row, then D²-proportional draws.
    points = check_points(points)
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(points)))]
    while len(chosen) < k:
        nearest = squared_distances(points, points[chosen]).min(axis=1)
        if nearest.max() == 0:  # every distinct location is already a center
            chosen.append(next(i for i in range(len(points)) if i not in chosen))
        else:
            chosen.append(int(rng.choice(len(points), p=nearest / nearest.sum())))
    return points[chosen].copy(), chosen


def lloyd(points, k=3, seed=0, max_iter=100, initial=None):
    points = check_points(points)
    if type(k) is not int or not 1 <= k <= len(points):
        raise ValueError("Choose an integer k between 1 and the number of rows.")
    if initial is None:
        centers, seed_rows = d2_seed(points, k, seed)
    else:
        centers, seed_rows = check_points(initial).copy(), None
        if centers.shape != (k, points.shape[1]):
            raise ValueError("Initial centers must have shape (k, number of features).")
    history = []
    for iteration in range(1, max_iter + 1):
        distances = squared_distances(points, centers)
        labels_for_means = distances.argmin(axis=1)  # lowest index wins an exact tie
        before = float(distances[np.arange(len(points)), labels_for_means].sum())
        updated = centers.copy()
        for cluster in range(k):
            members = points[labels_for_means == cluster]
            if len(members):  # an empty slot keeps its previous center
                updated[cluster] = members.mean(axis=0)
        updated_distances = squared_distances(points, updated)
        after_means = float(updated_distances[np.arange(len(points)), labels_for_means].sum())
        labels = updated_distances.argmin(axis=1)
        inertia = float(updated_distances[np.arange(len(points)), labels].sum())
        converged = bool(np.array_equal(labels, labels_for_means))
        history.append((before, after_means, inertia, int(np.sum(labels != labels_for_means))))
        centers = updated
        if converged:
            break
    return {"centers": centers, "labels": labels, "inertia": inertia,
            "converged": converged, "iterations": iteration, "history": history,
            "seed_rows": seed_rows}


points, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.8, random_state=42)
result = lloyd(points)
print("data:", points.shape, "seed rows:", result["seed_rows"])
print("SSE before update / after moving means / after reassigning / labels changed")
for before, after, reassigned, changed in result["history"]:
    print(f"{before:.6f} / {after:.6f} / {reassigned:.6f} / {changed}")
print("fixed point:", result["converged"], "after", result["iterations"], "mean update(s)")
for cluster in np.argsort(result["centers"][:, 0]):
    print("center:", np.round(result["centers"][cluster], 4).tolist(),
          "size:", int(np.sum(result["labels"] == cluster)))
print("inertia:", round(result["inertia"], 6))

duplicates = lloyd(np.ones((4, 2)), k=3)
print("four identical rows, k=3: occupied clusters =", len(np.unique(duplicates["labels"])))
changed = np.array([[0.0], [1.0], [4.0], [5.0], [10.0]])
limited = lloyd(changed, k=2, initial=[[0.0], [1.0]], max_iter=1)
print("budget of one update: fixed point =", limited["converged"],
      "returned labels =", limited["labels"].tolist())
for initial in [[[0.0], [4.0]], [[0.0], [10.0]]]:
    local = lloyd(changed, k=2, initial=initial)
    print("start", np.ravel(initial).tolist(), "-> fixed point", local["converged"],
          "inertia", round(local["inertia"], 6))
""",
    },
    "ward": {
        "title": "Calculate Ward's merge cost, then check the dendrogram height",
        "question": "Two merges have the same height. Can a horizontal height cut recover every cluster count that a binary merge-order cut can?",
        "code": r"""
import numpy as np
from scipy.cluster.hierarchy import linkage, cut_tree, fcluster
from sklearn.metrics import adjusted_rand_score

toy = np.array([[1.0, 0.0], [1.5, 0.5], [3.0, 2.0],
                [3.5, 2.0], [7.0, 5.0], [7.5, 5.5]])


def ward_merges(points):
    # Agglomerate by the smallest increase in within-cluster squared error.
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or not np.isfinite(points).all():
        raise ValueError("Use a finite numeric row-by-feature matrix.")
    clusters = {i: (1, row.copy(), [i]) for i, row in enumerate(points)}  # size, mean, rows
    rows, costs = [], []
    for new_id in range(len(points), 2 * len(points) - 1):
        candidates = []
        active = sorted(clusters)
        for position, left in enumerate(active):
            for right in active[position + 1:]:
                na, ma, _ = clusters[left]
                nb, mb, _ = clusters[right]
                delta = na * nb / (na + nb) * float(((ma - mb) ** 2).sum())
                candidates.append((delta, left, right))
        delta, left, right = min(candidates)  # the lowest ID pair breaks an exact tie
        na, ma, members_a = clusters.pop(left)
        nb, mb, members_b = clusters.pop(right)
        clusters[new_id] = (na + nb, (na * ma + nb * mb) / (na + nb), members_a + members_b)
        rows.append([left, right, np.sqrt(2 * delta), na + nb])  # SciPy's height convention
        costs.append(delta)
    return np.array(rows), np.array(costs)


def count_cut(tree, k):
    # Apply exactly the first n − k merges.
    n = len(tree) + 1
    active = {index: [index] for index in range(n)}
    for step, row in enumerate(tree[:n - k]):
        left, right = map(int, row[:2])
        active[n + step] = active.pop(left) + active.pop(right)
    labels = np.empty(n, dtype=int)
    for label, members in enumerate(active.values()):
        labels[members] = label
    return labels


ours, delta = ward_merges(toy)
reference = linkage(toy, method="ward", metric="euclidean")
print("left right delta_SSE scipy_height size")
for own, cost, other in zip(ours, delta, reference):
    print(int(own[0]), int(own[1]), f"{cost:.6f}", f"{other[2]:.6f}", int(own[3]))
print("all heights obey h² = 2Δ:", bool(np.allclose(reference[:, 2] ** 2, 2 * delta)))
print("sum of merge costs / one-cluster SSE:", round(float(delta.sum()), 6),
      round(float(np.sum((toy - toy.mean(axis=0)) ** 2)), 6))
for k in [2, 3, 4]:
    direct = count_cut(ours, k)
    scipy_labels = cut_tree(reference, n_clusters=[k]).ravel()
    print("count cut", k, "agrees with SciPy cut_tree (ARI):",
          round(adjusted_rand_score(direct, scipy_labels), 6))
equal_height = reference[1, 2]
below = fcluster(reference, np.nextafter(equal_height, -np.inf), criterion="distance")
through = fcluster(reference, equal_height, criterion="distance")
max_four = fcluster(reference, 4, criterion="maxclust")
print("clusters from a height cut just below / at the tied height:",
      len(np.unique(below)), len(np.unique(through)))
print("fcluster maxclust=4 returns at most 4; actual:", len(np.unique(max_four)))
""",
    },
    "production": {
        "title": "Fit Old Faithful with scikit-learn, then compare three library estimators",
        "question": "Which k does the inertia curve favor once the axis is read proportionally, and does the silhouette agree? Does MiniBatchKMeans(n_init=10) mean ten complete mini-batch fits?",
        "code": r"""
import numpy as np
import sklearn
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

# Old Faithful geyser, Yellowstone: eruption duration and waiting time to the next
# eruption, both in minutes. 272 rows, the R `faithful` version (Härdle 1991).
FAITHFUL = (
    "3.6 79 1.8 54 3.333 74 2.283 62 4.533 85 2.883 55 4.7 88 3.6 85 1.95 51 4.35 85 "
    "1.833 54 3.917 84 4.2 78 1.75 47 4.7 83 2.167 52 1.75 62 4.8 84 1.6 52 4.25 79 "
    "1.8 51 1.75 47 3.45 78 3.067 69 4.533 74 3.6 83 1.967 55 4.083 76 3.85 78 4.433 79 "
    "4.3 73 4.467 77 3.367 66 4.033 80 3.833 74 2.017 52 1.867 48 4.833 80 1.833 59 "
    "4.783 90 4.35 80 1.883 58 4.567 84 1.75 58 4.533 73 3.317 83 3.833 64 2.1 53 "
    "4.633 82 2 59 4.8 75 4.716 90 1.833 54 4.833 80 1.733 54 4.883 83 3.717 71 1.667 64 "
    "4.567 77 4.317 81 2.233 59 4.5 84 1.75 48 4.8 82 1.817 60 4.4 92 4.167 78 4.7 78 "
    "2.067 65 4.7 73 4.033 82 1.967 56 4.5 79 4 71 1.983 62 5.067 76 2.017 60 4.567 78 "
    "3.883 76 3.6 83 4.133 75 4.333 82 4.1 70 2.633 65 4.067 73 4.933 88 3.95 76 4.517 80 "
    "2.167 48 4 86 2.2 60 4.333 90 1.867 50 4.817 78 1.833 63 4.3 72 4.667 84 3.75 75 "
    "1.867 51 4.9 82 2.483 62 4.367 88 2.1 49 4.5 83 4.05 81 1.867 47 4.7 84 1.783 52 "
    "4.85 86 3.683 81 4.733 75 2.3 59 4.9 89 4.417 79 1.7 59 4.633 81 2.317 50 4.6 85 "
    "1.817 59 4.417 87 2.617 53 4.067 69 4.25 77 1.967 56 4.6 88 3.767 81 1.917 45 4.5 82 "
    "2.267 55 4.65 90 1.867 45 4.167 83 2.8 56 4.333 89 1.833 46 4.383 82 1.883 51 "
    "4.933 86 2.033 53 3.733 79 4.233 81 2.233 60 4.533 82 4.817 77 4.333 76 1.983 59 "
    "4.633 80 2.017 49 5.1 96 1.8 53 5.033 77 4 77 2.4 65 4.6 81 3.567 71 4 70 4.5 81 "
    "4.083 93 1.8 53 3.967 89 2.2 45 4.15 86 2 58 3.833 78 3.5 66 4.583 76 2.367 63 5 88 "
    "1.933 52 4.617 93 1.917 49 2.083 57 4.583 77 3.333 68 4.167 81 4.333 81 4.5 73 "
    "2.417 50 4 85 4.167 74 1.883 55 4.583 77 4.25 83 3.767 83 2.033 51 4.433 78 4.083 84 "
    "1.833 46 4.417 83 2.183 55 4.8 81 1.833 57 4.8 76 4.1 84 3.966 77 4.233 81 3.5 87 "
    "4.366 77 2.25 51 4.667 78 2.1 60 4.35 82 4.133 91 1.867 53 4.6 78 1.783 46 4.367 77 "
    "3.85 84 1.933 49 4.5 83 2.383 71 4.7 80 1.867 49 3.833 75 3.417 64 4.233 76 2.4 53 "
    "4.8 94 2 55 4.15 76 1.867 50 4.267 82 1.75 54 4.483 75 4 78 4.117 79 4.083 78 "
    "4.267 78 3.917 70 4.55 79 4.083 70 2.417 54 4.183 86 2.217 50 4.45 90 1.883 54 "
    "1.85 54 4.283 77 3.95 79 2.333 64 4.15 75 2.35 47 4.933 86 2.9 63 4.583 85 3.833 82 "
    "2.083 57 4.367 82 2.133 67 4.35 74 2.2 54 4.45 83 3.567 73 4.5 73 4.15 88 3.817 80 "
    "3.917 71 4.45 83 2 56 4.283 79 4.767 78 4.533 84 1.85 58 4.25 83 1.983 43 2.25 60 "
    "4.75 75 4.117 81 2.15 46 4.417 90 1.817 46 4.467 74 "
)
faithful = np.array(FAITHFUL.split(), dtype=float).reshape(-1, 2)
print("scikit-learn:", sklearn.__version__, "Old Faithful rows:", faithful.shape)
scaler = StandardScaler().fit(faithful)
standardized = scaler.transform(faithful)
two = KMeans(n_clusters=2, n_init=10, random_state=0).fit(standardized)
raw_two = KMeans(n_clusters=2, n_init=10, random_state=0).fit(faithful)
print("standardized k=2 centers (minutes):",
      np.round(scaler.inverse_transform(two.cluster_centers_), 3).tolist(),
      "sizes:", np.bincount(two.labels_).tolist())
print("raw-unit k=2 centers (minutes):", np.round(raw_two.cluster_centers_, 3).tolist(),
      "sizes:", np.bincount(raw_two.labels_).tolist())
print("agreement of the two geometries (ARI):",
      round(adjusted_rand_score(two.labels_, raw_two.labels_), 6))
print("k / inertia over five single starts / median silhouette, standardized")
for k in range(1, 9):
    runs = [KMeans(n_clusters=k, n_init=1, random_state=seed).fit(standardized)
            for seed in range(5)]
    inertias = [model.inertia_ for model in runs]
    if k == 1:
        print(k, f"{min(inertias):.6f}..{max(inertias):.6f}", "silhouette undefined")
        continue
    silhouettes = [silhouette_score(standardized, model.labels_) for model in runs]
    print(k, f"{min(inertias):.6f}..{max(inertias):.6f}", f"{np.median(silhouettes):.6f}")

# A synthetic comparison retained from the earlier lesson: three generated blobs.
points, _ = make_blobs(n_samples=150, centers=3, cluster_std=0.8, random_state=42)
full = KMeans(n_clusters=3, init="k-means++", n_init=10,
              max_iter=300, algorithm="lloyd", random_state=42).fit(points)
mini = MiniBatchKMeans(n_clusters=3, init="k-means++", n_init=10,
                      batch_size=64, random_state=42).fit(points)
ward = AgglomerativeClustering(n_clusters=3, linkage="ward", metric="euclidean").fit(points)
print("blobs:", points.shape)
for name, fitted in [("KMeans", full), ("MiniBatchKMeans", mini)]:
    print(name, "inertia:", round(fitted.inertia_, 6),
          "silhouette:", round(silhouette_score(points, fitted.labels_), 6))
print("Ward silhouette:", round(silhouette_score(points, ward.labels_), 6))
print("KMeans versus Ward agreement (ARI):",
      round(adjusted_rand_score(full.labels_, ward.labels_), 6))
""",
    },
    "scaling": {
        "title": "Change units deliberately, then freeze a training-only transformation",
        "question": "Multiplying one coordinate by 100 changes its squared-distance weight by what factor? How can a unit conversion preserve the intended geometry?",
        "code": r"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score


def groups(labels):
    return sorted([np.flatnonzero(labels == label).tolist() for label in np.unique(labels)])


rectangle = np.array([[0.0, 0.0], [0.0, 10.0], [2.0, 0.0], [2.0, 10.0]])
def fit_labels(points):
    return KMeans(n_clusters=2, n_init=20, random_state=7).fit_predict(points)

original = fit_labels(rectangle)
converted = rectangle * [100.0, 1.0]
uncorrected = fit_labels(converted)
corrected = fit_labels(converted * np.sqrt([1 / 100**2, 1.0]))
print("original coordinate groups:", groups(original))
print("first coordinate converted without metric correction:", groups(uncorrected))
print("compensated squared-distance weights:", groups(corrected))
print("compensated ARI:", adjusted_rand_score(original, corrected))

# Rows are constructed devices: temperature (°C), vibration (mm/s).
training = np.array([[1.0, 10.0], [2.0, 12.0], [1.5, 9.0],
                     [8.0, 30.0], [9.0, 29.0], [8.5, 32.0]])
held_out = np.array([[2.5, 11.0], [8.0, 31.0], [20.0, 100.0]])
pipeline = make_pipeline(StandardScaler(), KMeans(n_clusters=2, n_init=10, random_state=7))
pipeline.fit(training)
mean_before = pipeline[0].mean_.copy()
centers_before = pipeline[1].cluster_centers_.copy()
labels = pipeline.predict(held_out)
distances = pipeline.transform(held_out).min(axis=1)
print("training-only coordinate means:", np.round(mean_before, 6).tolist())
print("held-out nearest standardized distances:", np.round(distances, 6).tolist())
print("predict changed scaler or centers:",
      not (np.array_equal(mean_before, pipeline[0].mean_) and
           np.array_equal(centers_before, pipeline[1].cluster_centers_)))
""",
    },
    "quantization": {
        "title": "Fit weighted colors and decode a real finite palette",
        "question": "Can eight distinct RGB rows represent 128 pixels exactly during fitting? Which distortion changes when floating-point centers become an 8-bit palette?",
        "code": r"""
import math
import numpy as np
from sklearn.cluster import KMeans

colors = np.array([[240, 20, 20], [250, 30, 15], [230, 15, 25],
                   [20, 230, 20], [30, 240, 15], [15, 220, 30],
                   [20, 20, 230], [30, 25, 240]], dtype=float)
counts = np.array([30, 12, 10, 25, 8, 10, 20, 13])
pixels = np.repeat(colors, counts, axis=0)
initial = colors[[0, 3, 6]].copy()  # Same starting centers in both representations.
weighted = KMeans(n_clusters=3, init=initial, n_init=1, tol=0,
                  algorithm="lloyd", random_state=0).fit(colors, sample_weight=counts)
repeated = KMeans(n_clusters=3, init=initial, n_init=1, tol=0,
                  algorithm="lloyd", random_state=0).fit(pixels)
print("pixels / distinct rows:", len(pixels), len(colors))
print("weighted versus repeated SSE:", round(weighted.inertia_, 6), round(repeated.inertia_, 6))
print("same floating palette:", bool(np.allclose(weighted.cluster_centers_, repeated.cluster_centers_)))

palette = np.clip(np.rint(weighted.cluster_centers_), 0, 255).astype(np.uint8)
# Subtract in floating point: uint8 subtraction would wrap.
distances = ((pixels[:, None, :] - palette.astype(float)[None, :, :]) ** 2).sum(axis=2)
indices = distances.argmin(axis=1)
bits_per_index = math.ceil(math.log2(len(palette)))
bitstream = "".join(format(int(index), f"0{bits_per_index}b") for index in indices)
decoded_indices = np.array([int(bitstream[start:start + bits_per_index], 2)
                            for start in range(0, len(bitstream), bits_per_index)])
decoded = palette[decoded_indices].astype(float)
deployed_sse = float(np.sum((pixels - decoded) ** 2))
raw_bits = len(pixels) * 3 * 8
palette_bits = palette.size * 8
payload_bits = palette_bits + len(bitstream)
print("integer RGB palette:", palette.tolist())
print("decoded matches palette assignment:", bool(np.array_equal(decoded_indices, indices)))
print("floating / decoded squared RGB error per pixel:",
      round(weighted.inertia_ / len(pixels), 6), round(deployed_sse / len(pixels), 6))
print("raw / palette / index / total payload bits:", raw_bits, palette_bits, len(bitstream), payload_bits)
""",
    },
    "capstone": {
        "title": "Choose a clustering protocol before inspecting the held-out report",
        "question": "Which choices use training rows, which use validation rows, and which are frozen before the final profile report? How should an unstable split change your conclusion?",
        "code": r"""
import itertools
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import silhouette_score, adjusted_rand_score


def fit_model(training, k, seed):
    return make_pipeline(StandardScaler(), KMeans(n_clusters=k, n_init=10, random_state=seed)).fit(training)


def report(data_seed=17, split_seed=23):
    latent, _ = make_blobs(n_samples=360, centers=[[-3, 1], [0, -3], [4, 2]],
                          cluster_std=[1.0, 1.5, 0.8], random_state=data_seed)
    # Constructed devices, not measurements from an actual installation.
    points = latent * [3.0, 0.5] + [40.0, 5.0]
    order = np.random.default_rng(split_seed).permutation(len(points))
    training, validation, test = points[order[:240]], points[order[240:300]], points[order[300:]]
    # Declared rule: eligible if each training cluster has >=20 rows and the
    # median bootstrap ARI on the SAME validation rows is >=0.8. Among eligible
    # k in {2,3,4}, choose largest validation silhouette; smaller k breaks a tie.
    bootstrap_rows = [np.random.default_rng(seed).integers(len(training), size=len(training))
                      for seed in range(30, 38)]
    candidates = []
    for k in [2, 3, 4]:
        fitted = fit_model(training, k, 7)
        validation_labels = fitted.predict(validation)
        transformed_validation = fitted[0].transform(validation)
        count = len(np.unique(validation_labels))
        silhouette = silhouette_score(transformed_validation, validation_labels) if 1 < count < len(validation) else -1
        stability = [adjusted_rand_score(validation_labels, fit_model(training[rows], k, 7).predict(validation))
                     for rows in bootstrap_rows]
        minimum_size = int(np.bincount(fitted[1].labels_, minlength=k).min())
        median_stability = float(np.median(stability))
        validation_error = float(np.mean(fitted.transform(validation).min(axis=1) ** 2))
        eligible = minimum_size >= 20 and median_stability >= 0.8 and count > 1
        print("candidate k / validation silhouette / mean squared distance / median ARI / min train size / eligible:",
              k, round(silhouette, 6), round(validation_error, 6), round(median_stability, 6), minimum_size, eligible)
        if eligible:
            candidates.append((silhouette, -k, fitted, k))
    if not candidates:
        raise RuntimeError("No candidate meets the predeclared requirements; revise the plan without using test results.")
    _, _, fitted, k = max(candidates, key=lambda row: row[:2])
    # Test data first enter here. No refit or changed k after reading this report.
    labels = fitted.predict(test)
    baseline = fit_model(training, 1, 7)
    error = float(np.mean(fitted.transform(test).min(axis=1) ** 2))
    baseline_error = float(np.mean(baseline.transform(test).min(axis=1) ** 2))
    centers = fitted[0].inverse_transform(fitted[1].cluster_centers_)
    print("selected k:", k)
    print("test / training-centroid baseline mean squared standardized distance:", round(error, 6), round(baseline_error, 6))
    print("profiles ordered by fitted center temperature: count / mean temperature / mean vibration")
    for cluster in np.argsort(centers[:, 0]):
        members = test[labels == cluster]
        if len(members):
            print(len(members), *np.round(members.mean(axis=0), 6).tolist())
        else:
            print(0, "no held-out members")
    # A transparent resource calculation, not a measured runtime prediction.
    future_n = 100000
    condensed_gib = (future_n * (future_n - 1) // 2) * 8 / 2**30
    print("100000 rows: float64 condensed distance buffer alone (GiB):", round(condensed_gib, 3))
    print("A 16 GiB machine cannot hold that dense hierarchy plan; the reasoning belongs before fitting.")
    return {"selected_k": k, "fitted": fitted, "training": training,
            "validation": validation, "test": test, "labels": labels, "error": error,
            "baseline_error": baseline_error}


result = report()
""",
    },
}


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


module_path = Path("src/learn/data/k-means-hierarchical-examples.js")
evidence_path = Path("docs/teaching/evidence/k-means-hierarchical-native.json")
write = "--write" in sys.argv
if not write:
    stored_text = module_path.read_text(encoding="utf-8")
    stored = json.loads(
        stored_text.split("export const clusteringExamples =", 1)[1]
        .strip()
        .removesuffix(";")
    )
else:
    stored = None

if "--ward-amendment" in sys.argv:
    # Preserve the original full verification and bytes; do not rerun the five
    # unchanged displayed programs or relabel their original evidence as fresh.
    original_bytes = module_path.read_bytes()
    previous = json.loads(evidence_path.read_text(encoding="utf-8"))
    assert previous["sourceHash"] == digest_bytes(original_bytes)
    item = PROGRAMS["ward"]
    code = black.format_str(item["code"].strip() + "\n", mode=black.Mode()).strip()
    assert code != stored["ward"]["code"], "The Ward amendment is already installed."
    scope, stream = {"__name__": "__main__"}, io.StringIO()
    with threadpool_limits(limits=1), contextlib.redirect_stdout(stream):
        exec(compile(code, "clustering-example:ward", "exec"), scope)
    output = stream.getvalue().rstrip()
    assert (
        output == stored["ward"]["expected"]
    ), "Original Ward stdout must be conserved."
    boundary_cases = []
    for separation in [1e-200, 2.3e-162]:
        try:
            scope["ward_merges"]([[0.0], [separation]])
        except ValueError as error:
            boundary_cases.append(
                {
                    "separation": separation,
                    "result": "explicit rejection",
                    "message": str(error),
                }
            )
        else:
            raise AssertionError("Positive separation silently underflowed.")
    for separation in [1e-150, 0.001]:
        tree, costs = scope["ward_merges"]([[0.0], [separation]])
        assert costs[0] > 0 and tree[0, 2] > 0
        np.testing.assert_allclose(tree[0, 2] / separation, 1, rtol=1e-14, atol=0)
        boundary_cases.append(
            {
                "separation": separation,
                "height": float(tree[0, 2]),
                "delta": float(costs[0]),
                "result": "positive representable result",
            }
        )
    tree, costs = scope["ward_merges"]([[1.0], [1.0]])
    assert costs[0] == 0 and tree[0, 2] == 0
    boundary_cases.append({"separation": 0, "result": "exact zero retained"})
    for values in [[0, 1, 4, 10], [-4, -1, 2, 7, 9], [0, 0, 2, 2, 5]]:
        tree, _ = scope["ward_merges"](np.array(values)[:, None])
        np.testing.assert_allclose(
            tree[:, 2],
            linkage(np.array(values)[:, None], method="ward")[:, 2],
            rtol=1e-12,
            atol=1e-12,
        )
    revised = {**stored, "ward": {**stored["ward"], "code": code}}
    for key in stored:
        if key != "ward":
            assert revised[key] == stored[key]
    module_path.write_text(
        "// Complete standalone programs, executed by the semantic native verifier.\nexport const clusteringExamples = "
        + json.dumps(revised, ensure_ascii=False, indent=2)
        + ";\n",
        encoding="utf-8",
    )
    amended_at = datetime.now(timezone.utc).isoformat()
    updated = json.loads(json.dumps(previous))
    updated["originalVerification"] = previous.get("originalVerification", previous)
    updated["sourceHash"] = digest_bytes(module_path.read_bytes())
    updated["verifierHash"] = digest_bytes(Path(__file__).read_bytes())
    updated["amendedAt"] = amended_at
    updated["programs"]["ward"]["codeHash"] = digest_bytes(code.encode())
    updated["amendments"] = previous.get("amendments", []) + [
        {
            "verifiedAt": amended_at,
            "reason": "Independent review found a nonzero admitted centroid separation whose squared distance silently became zero.",
            "sourceHashBefore": previous["sourceHash"],
            "sourceHashAfter": updated["sourceHash"],
            "originalModuleBase64": base64.b64encode(original_bytes).decode(),
            "executedPrograms": ["ward"],
            "originalWardStdoutUnchanged": True,
            "unchangedCompletePrograms": [key for key in stored if key != "ward"],
            "boundaryCases": boundary_cases,
            "changedWardLibraryFixturesRepeated": 3,
            "scope": "Only the affected Ward program and focused cases reran. Original six-run, 49-oracle, five-rejection evidence retains its original timestamp.",
        }
    ]
    evidence_path.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
    print(
        "PASS: affected Ward program, five arithmetic boundaries and three changed library fixtures; original evidence preserved."
    )
    sys.exit(0)

records, scopes = {}, {}
with threadpool_limits(limits=1):
    for key, item in PROGRAMS.items():
        code = black.format_str(item["code"].strip() + "\n", mode=black.Mode()).strip()
        assert ast.dump(ast.parse(code)) == ast.dump(ast.parse(item["code"])), key
        stream = io.StringIO()
        scope = {"__name__": "__main__"}
        with contextlib.redirect_stdout(stream):
            exec(compile(code, f"clustering-example:{key}", "exec"), scope)
        output = stream.getvalue().rstrip()
        record = {
            "title": item["title"],
            "question": item["question"],
            "code": code,
            "expected": output,
            "language": "python",
        }
        if stored is not None:
            assert record == stored[key], f"Displayed program/output changed: {key}"
        records[key], scopes[key] = record, scope
        print(
            f"Executed {key}: {len(code.splitlines())} lines, {len(output.splitlines())} output lines"
        )

checks = []


def equal(actual, expected, context, tolerance=1e-10):
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float),
        np.asarray(expected, dtype=float),
        atol=tolerance,
        rtol=tolerance,
        err_msg=context,
    )


def partitions(n, k):
    """Canonical restricted-growth label strings enumerate each partition once."""

    def extend(labels):
        if len(labels) == n:
            if max(labels) + 1 == k:
                yield tuple(labels)
            return
        for label in range(min(k - 1, max(labels) + 1) + 1):
            yield from extend(labels + [label])

    yield from extend([0])


def exact_partition_sse(values, labels):
    result = Fraction(0)
    for label in set(labels):
        members = [
            Fraction(value)
            for value, assigned in zip(values, labels)
            if assigned == label
        ]
        # Pairwise identity, avoiding a second implementation of centroid SSE.
        result += sum(
            (
                (first - second) ** 2
                for first, second in itertools.combinations(members, 2)
            ),
            Fraction(0),
        ) / len(members)
    return result


lloyd = scopes["lloyd"]["lloyd"]
for values in [[0, 1, 4, 5, 10], [-3, -3, 0, 2, 8], [0, 0, 0, 1, 1], [1, 1, 1, 1]]:
    for k in [1, 2, 3]:
        all_costs = [
            exact_partition_sse(values, labels) for labels in partitions(len(values), k)
        ]
        optimum = min(all_costs)
        for seed in [0, 3, 9]:
            result = lloyd(np.array(values)[:, None], k=k, seed=seed)
            equal(
                result["inertia"],
                exact_partition_sse(values, result["labels"]),
                "fixed-point partition identity",
            )
            assert result["converged"]
            assert result["inertia"] >= float(optimum) - 1e-10
            for before, after, reassigned, _ in result["history"]:
                assert before + 1e-10 >= after >= reassigned - 1e-10
            # The returned state, not a prior assignment, defines its inertia.
            distances = (
                (np.array(values)[:, None, None] - result["centers"][None, :, :]) ** 2
            ).sum(axis=2)
            assert np.array_equal(result["labels"], distances.argmin(axis=1))
            checks.append(
                {
                    "kind": "exact partition lower bound and Lloyd state",
                    "values": values,
                    "k": k,
                    "seed": seed,
                    "globalMinimum": str(optimum),
                    "returnedSSE": result["inertia"],
                }
            )

local_values = [0, 1, 4, 5, 10]
local = lloyd(np.array(local_values)[:, None], k=2, initial=[[0], [4]])
global_minimum = min(
    exact_partition_sse(local_values, labels) for labels in partitions(5, 2)
)
assert local["inertia"] > float(global_minimum)
equal(global_minimum, 17, "exact five-row global optimum")
for initial in [[[0], [1]], [[0], [4]], [[0], [10]]]:
    result = lloyd(np.array(local_values)[:, None], k=2, initial=initial, max_iter=1)
    nearest = (
        ((np.array(local_values)[:, None, None] - result["centers"][None, :, :]) ** 2)
        .sum(axis=2)
        .argmin(axis=1)
    )
    assert np.array_equal(result["labels"], nearest)
    original_labels = (
        ((np.array(local_values)[:, None, None] - np.array(initial)[None, :, :]) ** 2)
        .sum(axis=2)
        .argmin(axis=1)
    )
    assert result["converged"] == bool(np.array_equal(original_labels, nearest))
    checks.append(
        {
            "kind": "budget return contract",
            "initial": initial,
            "converged": result["converged"],
        }
    )

ward = scopes["ward"]["ward_merges"]
equal(
    scopes["ward"]["delta"],
    [
        Fraction(1, 8),
        Fraction(1, 4),
        Fraction(1, 4),
        Fraction(113, 16),
        Fraction(2689, 48),
    ],
    "original six-point hand calculation",
)
for values in [[0, 1, 4, 10], [-4, -1, 2, 7, 9], [0, 0, 2, 2, 5]]:
    points = np.array(values)[:, None]
    tree, costs = ward(points)
    library = linkage(points, method="ward")
    equal(tree[:, 2], library[:, 2], "Ward heights versus independent SciPy")
    active = {index: [index] for index in range(len(values))}
    old_total = Fraction(0)
    for step, row in enumerate(tree):
        left, right = map(int, row[:2])
        active[len(values) + step] = active.pop(left) + active.pop(right)
        labels = np.empty(len(values), dtype=int)
        for label, members in enumerate(active.values()):
            labels[members] = label
        new_total = exact_partition_sse(values, labels)
        equal(costs[step], new_total - old_total, "Ward exact global SSE increment")
        old_total = new_total
    checks.append(
        {"kind": "Ward exact partition increment and library heights", "values": values}
    )

for pixels_count in [5, 9, 17]:
    original = np.array([[0.0, 0.0, 0.0], [30.0, 20.0, 10.0], [200.0, 210.0, 220.0]])
    counts = np.array([pixels_count, 2, 3])
    palette = np.array([[10.0, 8.0, 3.0], [201.0, 208.0, 218.0]])
    weighted_cost = float(
        (
            ((original[:, None] - palette[None]) ** 2).sum(axis=2).min(axis=1) * counts
        ).sum()
    )
    repeated = np.repeat(original, counts, axis=0)
    repeated_cost = float(
        ((repeated[:, None] - palette[None]) ** 2).sum(axis=2).min(axis=1).sum()
    )
    equal(weighted_cost, repeated_cost, "weighted unique versus repeated objective")
    checks.append(
        {
            "kind": "changed palette weights exact repeated objective",
            "counts": counts.tolist(),
            "SSE": weighted_cost,
        }
    )

for malformed in [([], 1), ([[np.nan]], 1), ([[1]], 0), ([[1]], True), ([[1]], 2)]:
    try:
        lloyd(malformed[0], k=malformed[1])
    except ValueError:
        pass
    else:
        raise AssertionError(("expected explicit input rejection", malformed))
for points, k in [([[1.0], [1.0], [1.0], [1.0]], 3), ([[0.0], [0.0], [5.0], [5.0]], 4)]:
    centers, chosen = scopes["lloyd"]["d2_seed"](points, k, seed=3)
    assert len(chosen) == len(set(chosen)) == k
    assert np.array_equal(centers, np.array(points)[chosen])
    # A duplicate-location choice is used only once no location remains uncovered.
    for step in range(1, k):
        previous = np.array(points)[chosen[:step]]
        remaining = (
            ((np.array(points)[:, None, :] - previous[None, :, :]) ** 2)
            .sum(axis=2)
            .min(axis=1)
        )
        assert remaining[chosen[step]] > 0 or np.all(remaining == 0)
    checks.append(
        {
            "kind": "D² duplicate/all-zero policy",
            "points": points,
            "k": k,
            "chosenRows": chosen,
        }
    )
empty = lloyd(
    [[0.0], [1.0], [9.0], [10.0]], k=3, initial=[[0.0], [0.0], [10.0]], max_iter=1
)
equal(empty["centers"][1], [0], "empty center retains its value")
assert empty["converged"] is False
checks.append(
    {
        "kind": "retained empty center can receive reassigned points",
        "centers": empty["centers"].tolist(),
        "labels": empty["labels"].tolist(),
    }
)
limited = scopes["lloyd"]["limited"]
assert limited["converged"] is False
quantization = scopes["quantization"]
equal(
    quantization["weighted"].inertia_,
    quantization["repeated"].inertia_,
    "actual weighted/repeated KMeans objective",
)
equal(
    quantization["weighted"].cluster_centers_,
    quantization["repeated"].cluster_centers_,
    "actual weighted/repeated KMeans means",
)
assert quantization["payload_bits"] == 328 and quantization["raw_bits"] == 3072
assert np.array_equal(quantization["decoded_indices"], quantization["indices"])
equal(
    scopes["scaling"]["pipeline"][0].mean_,
    scopes["scaling"]["training"].mean(axis=0),
    "training-only scaler",
)

with threadpool_limits(limits=1), contextlib.redirect_stdout(
    io.StringIO()
) as changed_output:
    changed_report = scopes["capstone"]["report"](data_seed=29, split_seed=31)
assert np.isfinite(changed_report["error"])
assert (
    adjusted_rand_score(
        changed_report["labels"],
        (changed_report["labels"] + 1) % changed_report["selected_k"],
    )
    == 1
)
checks.append(
    {
        "kind": "changed complete held-out capstone",
        "dataSeed": 29,
        "splitSeed": 31,
        "selectedK": changed_report["selected_k"],
        "stdout": changed_output.getvalue().strip(),
    }
)

if write:
    module_path.write_text(
        "// Complete standalone programs, executed by the semantic native verifier.\nexport const clusteringExamples = "
        + json.dumps(records, ensure_ascii=False, indent=2)
        + ";\n",
        encoding="utf-8",
    )

evidence = {
    "verifiedAt": datetime.now(timezone.utc).isoformat(),
    "stage": "native author verification; complete lesson/browser/independent review remain separate",
    "source": str(module_path).replace("\\", "/"),
    "sourceHash": digest_bytes(module_path.read_bytes()),
    "verifier": "scripts/verify-k-means-hierarchical-examples.py",
    "verifierHash": digest_bytes(Path(__file__).read_bytes()),
    "versions": {
        name: importlib.metadata.version(name)
        for name in ["numpy", "scipy", "scikit-learn"]
    },
    "programs": {
        key: {
            "codeHash": digest_bytes(record["code"].encode()),
            "stdoutHash": digest_bytes(record["expected"].encode()),
            "stdout": record["expected"],
        }
        for key, record in records.items()
    },
    "oracles": checks,
    "explicitRejections": 5,
    "research": [
        {
            "url": "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html",
            "inspected": "1.9.1 init, greedy versus vanilla seeding, explicit n_init and current fit behavior; actual current library executions",
        },
        {
            "url": "https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html",
            "inspected": "1.9.1 n_init initializations versus one complete run, inertia/compute_labels, sample_weight",
        },
        {
            "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html",
            "inspected": "Ward update and tied-minimum warning; checked h squared equals twice actual SSE increment",
        },
        {
            "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.cut_tree.html",
            "inspected": "count and height cut contracts, verified actual tied toy",
        },
    ],
    "limits": [
        "No timing benchmark, deployment claim or global optimum claim for the 150-point data.",
        "Displayed scratch implementations use their explicit small numeric-input contracts.",
        "Original 150-point and six-point datasets are retained where relevant; root owns exact preservation of the four historical programs.",
        "Palette payload calculation excludes headers/dimensions/padding and does not claim perceptual optimality.",
        "Capstone uses synthetic devices; validation criteria and bootstrap stability do not identify real-world categories.",
    ],
}
evidence_path.parent.mkdir(parents=True, exist_ok=True)
if evidence_path.exists():
    previous_evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    for history_key in ["originalVerification", "amendments", "amendedAt"]:
        if history_key in previous_evidence:
            evidence[history_key] = previous_evidence[history_key]
evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
print(
    f"PASS: {len(records)} complete programs, {len(checks)} changed oracle cases, 5 rejection cases."
)
