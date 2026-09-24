// Complete displayed programs for the DBSCAN lesson, executed by scripts/verify-dbscan-examples.py.
// The Iris program is executed as if saved beside public/learn-assets/dbscan/iris.csv.
export const dbscanExamples = {
  "trail": {
    "title": "Program 1: a complete small DBSCAN on the trail",
    "question": "Can we recover the two trail groups while preventing I from transmitting expansion?",
    "code": "from math import dist\n\n\ndef dbscan(points, eps, minimum):\n    neighbors = [\n        [j for j, other in enumerate(points) if dist(point, other) <= eps]\n        for point in points\n    ]\n    core = [len(row) >= minimum for row in neighbors]\n    labels = [-1] * len(points)\n    cluster = 0\n\n    for seed in range(len(points)):\n        if not core[seed] or labels[seed] != -1:\n            continue\n        labels[seed] = cluster\n        pending = [seed]\n        while pending:\n            current = pending.pop()\n            for other in neighbors[current]:\n                if labels[other] != -1:\n                    continue\n                labels[other] = cluster\n                if core[other]:\n                    pending.append(other)\n        cluster += 1\n\n    types = [\n        \"core\" if core[i] else \"border\" if labels[i] != -1 else \"noise\"\n        for i in range(len(points))\n    ]\n    return labels, types\n\n\npositions = [-1.75, -1.5, -1.25, -1, 1, 1.25, 1.5, 1.75, 0, 4]\npoints = [(x, 0) for x in positions]\nlabels, types = dbscan(points, eps=1, minimum=4)\nfor name, label, kind in zip(\"ABCDEFGHIJ\", labels, types):\n    print(name, label, kind)",
    "expected": "A 0 core\nB 0 core\nC 0 core\nD 0 core\nE 1 core\nF 1 core\nG 1 core\nH 1 core\nI 0 border\nJ -1 noise",
    "language": "python"
  },
  "coreRadius": {
    "title": "Program 2: the radius at which each row becomes core",
    "question": "At which exact radii can the trail's core set change?",
    "code": "import numpy as np\nfrom sklearn.neighbors import NearestNeighbors\n\nX = np.array([-1.75, -1.5, -1.25, -1, 1, 1.25, 1.5, 1.75, 0, 4])[:, None]\nm = 4\nsearch = NearestNeighbors(n_neighbors=m).fit(X)\ndistances, _ = search.kneighbors(X)\ncore_radius = distances[:, m - 1]\nprint(core_radius.tolist())\nprint(np.sort(core_radius).tolist())\nprint(int(np.sum(core_radius <= 1)))",
    "expected": "[0.75, 0.5, 0.5, 0.75, 0.75, 0.5, 0.5, 0.75, 1.25, 2.75]\n[0.5, 0.5, 0.5, 0.5, 0.75, 0.75, 0.75, 0.75, 1.25, 2.75]\n8",
    "language": "python"
  },
  "geographic": {
    "title": "A radius in kilometres on latitude and longitude",
    "question": "Can the first three locations connect through short links even though their endpoints exceed the chosen radius?",
    "code": "import numpy as np\nfrom sklearn.cluster import DBSCAN\n\nlatitude_longitude_degrees = np.array([[0, 0], [0, .01], [0, .02], [0, 1]])\nX = np.deg2rad(latitude_longitude_degrees)\nlabels = DBSCAN(eps=2 / 6371, min_samples=2,\n                metric=\"haversine\", algorithm=\"ball_tree\").fit_predict(X)\nprint(labels.tolist())",
    "expected": "[0, 0, 0, -1]",
    "language": "python"
  },
  "iris": {
    "title": "Program 3: an offline Iris report, including the rejected rows",
    "question": "Which settings retain most of the 150 flowers, and what happens to the silhouette as coverage rises?",
    "code": "from pathlib import Path\nimport numpy as np\nfrom sklearn.cluster import DBSCAN\nfrom sklearn.metrics import adjusted_rand_score, silhouette_score\nfrom sklearn.preprocessing import StandardScaler\n\ndata = np.genfromtxt(Path(__file__).with_name(\"iris.csv\"), delimiter=\",\", skip_header=1)\nrow_ids = data[:, 0].astype(int)\nfeatures = data[:, 1:5]\nspecies = data[:, 5].astype(int)\nX = StandardScaler().fit_transform(features)\n\nprint(\"eps clusters core border noise coverage silhouette ARI_all\")\nfor eps in [.3, .5, .8, 1.0]:\n    model = DBSCAN(eps=eps, min_samples=5).fit(X)\n    labels = model.labels_\n    assigned = labels >= 0\n    groups = np.unique(labels[assigned])\n    core = len(model.core_sample_indices_)\n    border = int(assigned.sum()) - core\n    noise = int((~assigned).sum())\n    silhouette = (silhouette_score(X[assigned], labels[assigned])\n                  if 2 <= len(groups) < assigned.sum() else float(\"nan\"))\n    ari_all = adjusted_rand_score(species, labels)\n    print(f\"{eps:.1f} {len(groups)} {core} {border} {noise} \"\n          f\"{assigned.mean():.3f} {silhouette:.3f} {ari_all:.3f}\")\n    print(\"noise_ids\", row_ids[~assigned].tolist())",
    "expected": "eps clusters core border noise coverage silhouette ARI_all\n0.3 3 19 11 120 0.200 0.630 0.088\nnoise_ids [5, 6, 8, 10, 13, 14, 15, 16, 18, 19, 21, 22, 23, 24, 31, 32, 33, 35, 36, 38, 41, 43, 44, 46, 48, 50, 51, 52, 53, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 89, 90, 91, 92, 93, 94, 97, 98, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149]\n0.5 2 93 23 34 0.773 0.656 0.442\nnoise_ids [14, 15, 32, 33, 41, 56, 57, 59, 60, 62, 68, 72, 85, 87, 93, 98, 100, 105, 106, 107, 108, 109, 114, 117, 118, 119, 122, 125, 129, 130, 131, 135, 136, 148]\n0.8 2 138 8 4 0.973 0.598 0.552\nnoise_ids [41, 109, 117, 131]\n1.0 2 142 5 3 0.980 0.595 0.554\nnoise_ids [41, 117, 131]",
    "language": "python"
  },
  "rings": {
    "title": "Two grouping rules on two concentric rings",
    "question": "Do these two rules recover the constructed ring identities?",
    "code": "import numpy as np\nfrom sklearn.cluster import DBSCAN, KMeans\nfrom sklearn.metrics import adjusted_rand_score\n\ninner_angle = 2 * np.pi * np.arange(12) / 12\nouter_angle = 2 * np.pi * np.arange(36) / 36\ninner = np.column_stack([np.cos(inner_angle), np.sin(inner_angle)])\nouter = 3 * np.column_stack([np.cos(outer_angle), np.sin(outer_angle)])\nX = np.vstack([inner, outer])\nring = np.r_[np.zeros(12, dtype=int), np.ones(36, dtype=int)]\ndensity = DBSCAN(eps=.6, min_samples=3).fit(X)\ncenters = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)\nprint(\"core rows\", len(density.core_sample_indices_))\nprint(\"DBSCAN ring ARI\", round(adjusted_rand_score(ring, density.labels_), 3))\nprint(\"KMeans ring ARI\", round(adjusted_rand_score(ring, centers.labels_), 3))",
    "expected": "core rows 48\nDBSCAN ring ARI 1.0\nKMeans ring ARI -0.016",
    "language": "python"
  },
  "optics": {
    "title": "Program 4: from an OPTICS ordering to labels at one radius",
    "question": "How does the density ordering become labels at a chosen radius?",
    "code": "import numpy as np\nfrom sklearn.cluster import OPTICS, cluster_optics_dbscan\n\nX = np.array([-1.75, -1.5, -1.25, -1, 1, 1.25, 1.5, 1.75, 0, 4])[:, None]\nmodel = OPTICS(min_samples=4, max_eps=2).fit(X)\nlabels = cluster_optics_dbscan(\n    reachability=model.reachability_,\n    core_distances=model.core_distances_,\n    ordering=model.ordering_, eps=1,\n)\nprint(model.ordering_.tolist())\nprint(model.reachability_[model.ordering_].tolist())\nprint(labels.tolist())",
    "expected": "[0, 1, 2, 3, 8, 4, 5, 6, 7, 9]\n[inf, 0.75, 0.5, 0.5, 1.0, 1.25, 0.75, 0.5, 0.5, inf]\n[0, 0, 0, 0, 1, 1, 1, 1, 0, -1]",
    "language": "python"
  },
  "mst": {
    "title": "Mutual reachability and a minimum spanning tree by hand",
    "question": "Which local radius delays the border bridge, and how many edges retain the threshold connectivity?",
    "code": "import numpy as np\n\nX = np.array([-1.75, -1.5, -1.25, -1, 1, 1.25, 1.5, 1.75, 0, 4])[:, None]\nD = np.abs(X - X.T)\ncore = np.sort(D, axis=1)[:, 3]  # fourth entry, including self\nW = np.maximum(np.maximum(core[:, None], core[None, :]), D)\nnp.fill_diagonal(W, 0)\n\ninside = {0}\nedges = []\nwhile len(inside) < len(X):\n    weight, source, target = min(\n        (W[i, j], i, j)\n        for i in inside for j in range(len(X)) if j not in inside\n    )\n    edges.append((source, target, float(weight)))\n    inside.add(target)\n\nprint(core.tolist())\nprint(len(edges))\nprint(float(W[0, 1]), float(W[3, 8]))",
    "expected": "[0.75, 0.5, 0.5, 0.75, 0.75, 0.5, 0.5, 0.75, 1.25, 2.75]\n9\n0.75 1.25",
    "language": "python"
  },
  "hdbscan": {
    "title": "HDBSCAN on the trail with the self-counting convention",
    "question": "Which trail rows receive a selected cluster, and how do their membership strengths differ?",
    "code": "import numpy as np\nfrom sklearn.cluster import HDBSCAN\n\nX = np.array([-1.75, -1.5, -1.25, -1, 1, 1.25, 1.5, 1.75, 0, 4])[:, None]\nmodel = HDBSCAN(min_cluster_size=4, min_samples=4,\n                cluster_selection_method=\"eom\", copy=True).fit(X)\nprint(model.labels_.tolist())\nprint(model.probabilities_.round(3).tolist())",
    "expected": "[0, 0, 0, 0, 1, 1, 1, 1, 0, -1]\n[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.6, 0.0]",
    "language": "python"
  },
  "duplicates": {
    "title": "Weighted duplicates keep their density",
    "question": "Can a smaller input preserve three repeated observations without changing their neighborhood count?",
    "code": "import numpy as np\nfrom sklearn.cluster import DBSCAN\n\nX = np.array([[0.], [0.], [0.], [2.]])\nunique, inverse, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)\nmodel = DBSCAN(eps=.25, min_samples=3).fit(unique, sample_weight=counts)\nprint(model.labels_[inverse].tolist())\nprint(DBSCAN(eps=.25, min_samples=3).fit_predict(unique).tolist())",
    "expected": "[0, 0, 0, -1]\n[-1, -1]",
    "language": "python"
  }
};
