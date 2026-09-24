"""Execute the DBSCAN lesson's displayed Python programs and record their output.

Run with the isolated lesson Python. `--write` regenerates
src/learn/data/dbscan-examples.js; otherwise the recorded output must match a
fresh execution. The Iris program reads iris.csv beside its own file, so it is
executed as if saved at public/learn-assets/dbscan/iris_dbscan.py.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import contextlib
import hashlib
import importlib.metadata
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

PROGRAMS = {
    "trail": {
        "title": "Program 1: a complete small DBSCAN on the trail",
        "question": "Can we recover the two trail groups while preventing I from transmitting expansion?",
        "code": r"""
from math import dist


def dbscan(points, eps, minimum):
    neighbors = [
        [j for j, other in enumerate(points) if dist(point, other) <= eps]
        for point in points
    ]
    core = [len(row) >= minimum for row in neighbors]
    labels = [-1] * len(points)
    cluster = 0

    for seed in range(len(points)):
        if not core[seed] or labels[seed] != -1:
            continue
        labels[seed] = cluster
        pending = [seed]
        while pending:
            current = pending.pop()
            for other in neighbors[current]:
                if labels[other] != -1:
                    continue
                labels[other] = cluster
                if core[other]:
                    pending.append(other)
        cluster += 1

    types = [
        "core" if core[i] else "border" if labels[i] != -1 else "noise"
        for i in range(len(points))
    ]
    return labels, types


positions = [-1.75, -1.5, -1.25, -1, 1, 1.25, 1.5, 1.75, 0, 4]
points = [(x, 0) for x in positions]
labels, types = dbscan(points, eps=1, minimum=4)
for name, label, kind in zip("ABCDEFGHIJ", labels, types):
    print(name, label, kind)
""",
    },
    "coreRadius": {
        "title": "Program 2: the radius at which each row becomes core",
        "question": "At which exact radii can the trail's core set change?",
        "code": r"""
import numpy as np
from sklearn.neighbors import NearestNeighbors

X = np.array([-1.75, -1.5, -1.25, -1, 1, 1.25, 1.5, 1.75, 0, 4])[:, None]
m = 4
search = NearestNeighbors(n_neighbors=m).fit(X)
distances, _ = search.kneighbors(X)
core_radius = distances[:, m - 1]
print(core_radius.tolist())
print(np.sort(core_radius).tolist())
print(int(np.sum(core_radius <= 1)))
""",
    },
    "geographic": {
        "title": "A radius in kilometres on latitude and longitude",
        "question": "Can the first three locations connect through short links even though their endpoints exceed the chosen radius?",
        "code": r"""
import numpy as np
from sklearn.cluster import DBSCAN

latitude_longitude_degrees = np.array([[0, 0], [0, .01], [0, .02], [0, 1]])
X = np.deg2rad(latitude_longitude_degrees)
labels = DBSCAN(eps=2 / 6371, min_samples=2,
                metric="haversine", algorithm="ball_tree").fit_predict(X)
print(labels.tolist())
""",
    },
    "iris": {
        "title": "Program 3: an offline Iris report, including the rejected rows",
        "question": "Which settings retain most of the 150 flowers, and what happens to the silhouette as coverage rises?",
        "code": r"""
from pathlib import Path
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

data = np.genfromtxt(Path(__file__).with_name("iris.csv"), delimiter=",", skip_header=1)
row_ids = data[:, 0].astype(int)
features = data[:, 1:5]
species = data[:, 5].astype(int)
X = StandardScaler().fit_transform(features)

print("eps clusters core border noise coverage silhouette ARI_all")
for eps in [.3, .5, .8, 1.0]:
    model = DBSCAN(eps=eps, min_samples=5).fit(X)
    labels = model.labels_
    assigned = labels >= 0
    groups = np.unique(labels[assigned])
    core = len(model.core_sample_indices_)
    border = int(assigned.sum()) - core
    noise = int((~assigned).sum())
    silhouette = (silhouette_score(X[assigned], labels[assigned])
                  if 2 <= len(groups) < assigned.sum() else float("nan"))
    ari_all = adjusted_rand_score(species, labels)
    print(f"{eps:.1f} {len(groups)} {core} {border} {noise} "
          f"{assigned.mean():.3f} {silhouette:.3f} {ari_all:.3f}")
    print("noise_ids", row_ids[~assigned].tolist())
""",
    },
    "rings": {
        "title": "Two grouping rules on two concentric rings",
        "question": "Do these two rules recover the constructed ring identities?",
        "code": r"""
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score

inner_angle = 2 * np.pi * np.arange(12) / 12
outer_angle = 2 * np.pi * np.arange(36) / 36
inner = np.column_stack([np.cos(inner_angle), np.sin(inner_angle)])
outer = 3 * np.column_stack([np.cos(outer_angle), np.sin(outer_angle)])
X = np.vstack([inner, outer])
ring = np.r_[np.zeros(12, dtype=int), np.ones(36, dtype=int)]
density = DBSCAN(eps=.6, min_samples=3).fit(X)
centers = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
print("core rows", len(density.core_sample_indices_))
print("DBSCAN ring ARI", round(adjusted_rand_score(ring, density.labels_), 3))
print("KMeans ring ARI", round(adjusted_rand_score(ring, centers.labels_), 3))
""",
    },
    "optics": {
        "title": "Program 4: from an OPTICS ordering to labels at one radius",
        "question": "How does the density ordering become labels at a chosen radius?",
        "code": r"""
import numpy as np
from sklearn.cluster import OPTICS, cluster_optics_dbscan

X = np.array([-1.75, -1.5, -1.25, -1, 1, 1.25, 1.5, 1.75, 0, 4])[:, None]
model = OPTICS(min_samples=4, max_eps=2).fit(X)
labels = cluster_optics_dbscan(
    reachability=model.reachability_,
    core_distances=model.core_distances_,
    ordering=model.ordering_, eps=1,
)
print(model.ordering_.tolist())
print(model.reachability_[model.ordering_].tolist())
print(labels.tolist())
""",
    },
    "mst": {
        "title": "Mutual reachability and a minimum spanning tree by hand",
        "question": "Which local radius delays the border bridge, and how many edges retain the threshold connectivity?",
        "code": r"""
import numpy as np

X = np.array([-1.75, -1.5, -1.25, -1, 1, 1.25, 1.5, 1.75, 0, 4])[:, None]
D = np.abs(X - X.T)
core = np.sort(D, axis=1)[:, 3]  # fourth entry, including self
W = np.maximum(np.maximum(core[:, None], core[None, :]), D)
np.fill_diagonal(W, 0)

inside = {0}
edges = []
while len(inside) < len(X):
    weight, source, target = min(
        (W[i, j], i, j)
        for i in inside for j in range(len(X)) if j not in inside
    )
    edges.append((source, target, float(weight)))
    inside.add(target)

print(core.tolist())
print(len(edges))
print(float(W[0, 1]), float(W[3, 8]))
""",
    },
    "hdbscan": {
        "title": "HDBSCAN on the trail with the self-counting convention",
        "question": "Which trail rows receive a selected cluster, and how do their membership strengths differ?",
        "code": r"""
import numpy as np
from sklearn.cluster import HDBSCAN

X = np.array([-1.75, -1.5, -1.25, -1, 1, 1.25, 1.5, 1.75, 0, 4])[:, None]
model = HDBSCAN(min_cluster_size=4, min_samples=4,
                cluster_selection_method="eom", copy=True).fit(X)
print(model.labels_.tolist())
print(model.probabilities_.round(3).tolist())
""",
    },
    "duplicates": {
        "title": "Weighted duplicates keep their density",
        "question": "Can a smaller input preserve three repeated observations without changing their neighborhood count?",
        "code": r"""
import numpy as np
from sklearn.cluster import DBSCAN

X = np.array([[0.], [0.], [0.], [2.]])
unique, inverse, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
model = DBSCAN(eps=.25, min_samples=3).fit(unique, sample_weight=counts)
print(model.labels_[inverse].tolist())
print(DBSCAN(eps=.25, min_samples=3).fit_predict(unique).tolist())
""",
    },
}

module_path = Path("src/learn/data/dbscan-examples.js")
evidence_path = Path("docs/teaching/evidence/dbscan-native.json")
asset_dir = Path("public/learn-assets/dbscan").resolve()
write = "--write" in sys.argv


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def execute(key):
    code = PROGRAMS[key]["code"].strip() + "\n"
    namespace = {"__file__": str(asset_dir / "iris_dbscan.py")}
    with threadpool_limits(limits=1), contextlib.redirect_stdout(io.StringIO()) as output:
        exec(compile(code, f"dbscan-example:{key}", "exec"), namespace)
    return code, output.getvalue().strip(), namespace


records, scopes = {}, {}
for key in PROGRAMS:
    code, stdout, namespace = execute(key)
    scopes[key] = namespace
    records[key] = {"title": PROGRAMS[key]["title"], "question": PROGRAMS[key]["question"], "code": code.rstrip("\n"), "expected": stdout, "language": "python"}
    print(f"Executed {key}: {len(code.splitlines())} lines, {len(stdout.splitlines())} output lines")

if not write:
    existing = module_path.read_text(encoding="utf-8")
    prefix = "export const dbscanExamples = "
    recorded = json.loads(existing[existing.index(prefix) + len(prefix):].rstrip().rstrip(";"))
    for key, record in records.items():
        assert recorded[key]["code"] == record["code"], f"{key}: displayed code changed"
        assert recorded[key]["expected"] == record["expected"], f"{key}: output changed:\n{recorded[key]['expected']}\n---\n{record['expected']}"

# Oracles tied to the manuscript's stated values and the author probes.
author = json.loads(Path("docs/teaching/drafts/dbscan-density-based-clustering/author-calculations.json").read_text(encoding="utf-8"))
trail = scopes["trail"]
assert trail["labels"] == [0, 0, 0, 0, 1, 1, 1, 1, 0, -1] and trail["types"] == ["core"] * 8 + ["border", "noise"]
reverse_labels, reverse_types = trail["dbscan"](trail["points"][::-1], 1, 4)
assert reverse_types[::-1] == trail["types"] and reverse_labels[::-1][8] != trail["labels"][8] or reverse_labels[::-1][8] == reverse_labels[::-1][4]
assert reverse_labels[::-1][8] == reverse_labels[::-1][4], "reversed order attaches I to E's component"
five = trail["dbscan"]([(x, 0) for x in [0, .25, .5, .75, 2]], .25, 3)
assert five[1] == ["border", "core", "core", "border", "noise"]
assert trail["dbscan"]([(x, 0) for x in [0, .25, .5, .75, 2]], .25, 4)[1] == ["noise"] * 5
moved = trail["dbscan"]([(x, 0) for x in [-1.75, -1.5, -1.25, -1, 1, 1.25, 1.5, 1.75]] + [(0, .125), (4, 0)], 1, 4)
assert moved[1][8] == "noise", "I moved off the line is noise"
assert scopes["coreRadius"]["core_radius"].tolist() == author["street_core_distance_m4"]
assert scopes["geographic"]["labels"].tolist() == [0, 0, 0, -1]
iris_out = records["iris"]["expected"].splitlines()
assert iris_out[1].startswith("0.3 3 19 11 120 0.200 0.630 0.088") and iris_out[3].startswith("0.5 2 93 23 34 0.773 0.656 0.442")
assert iris_out[5].startswith("0.8 2 138 8 4 0.973 0.598 0.552") and iris_out[7].startswith("1.0 2 142 5 3 0.980 0.595 0.554")
assert records["rings"]["expected"].splitlines() == ["core rows 48", "DBSCAN ring ARI 1.0", "KMeans ring ARI -0.016"]
optics = scopes["optics"]
assert optics["model"].ordering_.tolist() == author["optics"]["ordering"] and optics["labels"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 0, -1]
assert scopes["mst"]["core"].tolist() == author["street_core_distance_m4"] and len(scopes["mst"]["edges"]) == 9
assert scopes["hdbscan"]["model"].labels_.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 0, -1]
assert records["duplicates"]["expected"].splitlines() == ["[0, 0, 0, -1]", "[-1, -1]"]
oracle_count = 14

if write:
    module_path.write_text(
        "// Complete displayed programs for the DBSCAN lesson, executed by scripts/verify-dbscan-examples.py.\n"
        "// The Iris program is executed as if saved beside public/learn-assets/dbscan/iris.csv.\n"
        "export const dbscanExamples = " + json.dumps(records, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8",
    )

evidence = {
    "verifiedAt": datetime.now(timezone.utc).isoformat(),
    "stage": "native author verification of displayed programs; browser, independent and integration review are separate",
    "source": str(module_path).replace("\\", "/"),
    "sourceHash": hashlib.sha256(module_path.read_bytes()).hexdigest(),
    "verifier": "scripts/verify-dbscan-examples.py",
    "verifierHash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "irisCsvHash": hashlib.sha256((asset_dir / "iris.csv").read_bytes()).hexdigest(),
    "versions": {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "scikit-learn"]},
    "programs": {key: {"codeHash": digest(record["code"]), "stdoutHash": digest(record["expected"]), "stdout": record["expected"]} for key, record in records.items()},
    "oracles": oracle_count,
    "limits": [
        "Iris values are the corrected scikit-learn bundled copy; the CSV beside the program is served from public/learn-assets/dbscan/.",
        "Program outputs are single-threaded executions with the recorded versions; no timing is measured.",
        "The geographic program is a unit calculation on constructed coordinates, not a geographic benchmark.",
    ],
}
evidence_path.parent.mkdir(parents=True, exist_ok=True)
evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
print(f"PASS: {len(records)} displayed programs executed, {oracle_count} oracle assertions.")
