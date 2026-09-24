"""Calculate the manuscript's frozen-report fixture once during authoring.

This records intended numbers and row provenance; it does not execute the
displayed program, test a production implementation or perform browser QA.
"""

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


folder = Path(__file__).parent
data = np.genfromtxt(folder / "iris.csv", delimiter=",", skip_header=1)
x, species = data[:, 1:5], data[:, 5].astype(int)
order = np.random.default_rng(23).permutation(150)
fit_ids, selection_ids, report_ids = order[:90], order[90:120], order[120:]
scaler = StandardScaler().fit(x[fit_ids])
train = scaler.transform(x[fit_ids])
selection = scaler.transform(x[selection_ids])
report = scaler.transform(x[report_ids])
candidates = []
with threadpool_limits(limits=1):
    for k in [2, 3, 4]:
        model = KMeans(k, n_init=20, random_state=17).fit(train)
        assigned = model.predict(selection)
        counts = np.bincount(model.labels_, minlength=k)
        eligible = counts.min() >= 10 and 2 <= len(set(assigned)) < len(assigned)
        candidates.append({"k": k, "sizes": counts.tolist(), "eligible": bool(eligible),
                           "silhouette": float(silhouette_score(selection, assigned)),
                           "model": model})
selected = max((row for row in candidates if row["eligible"]),
               key=lambda row: (row["silhouette"], -row["k"]))
model = selected["model"]
labels = model.predict(report)
error = np.mean(np.min(np.sum((report[:, None] - model.cluster_centers_)**2, axis=2), axis=1))
baseline_error = np.mean(np.sum((report - train.mean(axis=0))**2, axis=1))
record = {"scope": "Small author calculation of a prespecified fixture, not complete-program verification.",
          "fit_ids": fit_ids.tolist(), "selection_ids": selection_ids.tolist(),
          "report_ids": report_ids.tolist(),
          "candidates": [{key: val for key, val in row.items() if key != "model"}
                         for row in candidates],
          "selected_k": selected["k"], "selection_silhouette": selected["silhouette"],
          "report_distortion": float(error), "report_baseline_distortion": float(baseline_error),
          "report_species_ARI": float(adjusted_rand_score(species[report_ids], labels))}
(folder / "report-author-calculation.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: v for k, v in record.items() if not k.endswith("_ids")}, indent=2))
