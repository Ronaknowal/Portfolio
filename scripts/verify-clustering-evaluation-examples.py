"""Execute the clustering-evaluation lesson's six displayed programs and record
their output. `--write` regenerates src/learn/data/clustering-evaluation-examples.js;
otherwise the recorded output must match. Programs that read iris.csv beside
themselves run in a temporary folder holding the lesson's exact CSV.
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
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

CSV = Path("public/learn-assets/clustering-evaluation/iris.csv")
PROGRAMS = {
    "silhouette": {
        "title": "Program 1 — calculate the bars",
        "question": "After C moves from L to R, which two averages exchange roles in its score, and why must every other bar be recomputed too?",
        "code": r"""
import numpy as np
from sklearn.metrics import silhouette_samples


def silhouettes(points, labels):
    points = np.asarray(points, dtype=float)
    labels = np.asarray(labels)
    groups = np.unique(labels)
    if not 2 <= len(groups) < len(points):
        raise ValueError("Use 2 to n-1 nonempty groups.")
    distance = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    result = np.zeros(len(points))
    for i, group in enumerate(labels):
        own = labels == group
        own[i] = False
        if not own.any():
            continue
        a = distance[i, own].mean()
        b = min(distance[i, labels == other].mean()
                for other in groups if other != group)
        result[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0
    return result


x = np.array([0, 1, 2, 7, 8, 9.0])[:, None]
groups = np.array([0, 0, 0, 1, 1, 1])
scores = silhouettes(x, groups)
print("scores:", np.round(scores, 6).tolist())
print("mean:", round(float(scores.mean()), 6))
print("matches library:", np.allclose(scores, silhouette_samples(x, groups)))
print("common scale unchanged:", np.allclose(scores, silhouettes(10*x, groups)))
changed = np.array([0, 0, 1, 1, 1, 1])
print("changed C:", round(float(silhouettes(x, changed)[2]), 6))
print("changed mean:", round(float(silhouettes(x, changed).mean()), 6))
""",
    },
    "pairs": {
        "title": "Program 2 — pair counting, then the compact formula",
        "question": "Why do the direct 28-pair enumeration and the contingency-cell shortcut have to produce the same Rand index?",
        "code": r"""
from collections import Counter
from itertools import combinations
from math import comb
from sklearn.metrics import adjusted_rand_score, rand_score

u = [0, 0, 0, 0, 1, 1, 1, 1]
v = [0, 0, 0, 1, 0, 1, 1, 1]
cells = Counter(zip(u, v))
rows, columns = Counter(u), Counter(v)
pairs = comb(len(u), 2)
together_both = sum(comb(n, 2) for n in cells.values())
together_u = sum(comb(n, 2) for n in rows.values())
together_v = sum(comb(n, 2) for n in columns.values())
expected = together_u * together_v / pairs
upper = (together_u + together_v) / 2
ari = (together_both - expected) / (upper - expected)
agreements = sum((u[i] == u[j]) == (v[i] == v[j])
                 for i, j in combinations(range(len(u)), 2))
print("S A B M:", together_both, together_u, together_v, pairs)
print("RI:", round(agreements / pairs, 6))
print("ARI:", round(ari, 6))
print("library:", round(rand_score(u, v), 6),
      round(adjusted_rand_score(u, v), 6))
renamed = [9 if label == 0 else 4 for label in v]
print("renamed ARI:", round(adjusted_rand_score(u, renamed), 6))
""",
    },
    "chance": {
        "title": "Program 3 — enumerate the chance experiment",
        "question": "Seventy assignments keep both margins at four and four. Which overlap is the most common, and what NMI does it give?",
        "code": r"""
from itertools import combinations
import numpy as np
from sklearn.metrics import (adjusted_rand_score,
                             normalized_mutual_info_score,
                             adjusted_mutual_info_score)

reference = np.array([0, 0, 0, 0, 1, 1, 1, 1])
results = []
overlaps = np.zeros(5, dtype=int)
for chosen in combinations(range(8), 4):
    candidate = np.ones(8, dtype=int)
    candidate[list(chosen)] = 0
    overlaps[np.sum(candidate[:4] == 0)] += 1
    results.append([
        adjusted_rand_score(reference, candidate),
        normalized_mutual_info_score(reference, candidate, average_method="arithmetic"),
        adjusted_mutual_info_score(reference, candidate, average_method="arithmetic")
    ])
means = np.mean(results, axis=0)
print("assignments:", len(results))
print("overlap counts:", overlaps.tolist())
print("mean NMI:", round(float(means[1]), 6))
print("adjusted means close to zero:", np.allclose(means[[0, 2]], 0, atol=1e-12))
""",
    },
    "iris": {
        "title": "Program 4 — reproduce the disagreements",
        "question": "Which representation and k do you expect to win on silhouette, and which on species agreement? Decide before reading the eight lines.",
        "needsCsv": True,
        "code": r"""
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score, silhouette_samples,
                             adjusted_rand_score, adjusted_mutual_info_score)

data = np.genfromtxt(Path(__file__).with_name("iris.csv"),
                     delimiter=",", skip_header=1)
x, species = data[:, 1:5], data[:, 5].astype(int)
scaled = StandardScaler().fit_transform(x)
representations = {
    "raw4": x,
    "scaled4": scaled,
    "pca2": PCA(2, svd_solver="full").fit_transform(scaled),
    "white2": PCA(2, whiten=True, svd_solver="full").fit_transform(scaled),
}
for name, z in representations.items():
    for k in [2, 3]:
        labels = KMeans(n_clusters=k, n_init=20, random_state=17).fit_predict(z)
        values = silhouette_samples(z, labels)
        print(name, k, f"sil={values.mean():.6f}",
              f"ARI={adjusted_rand_score(species, labels):.6f}",
              f"AMI={adjusted_mutual_info_score(species, labels):.6f}",
              f"negative={np.sum(values < 0)}")
""",
    },
    "probes": {
        "title": "Program 5 — compare the same probes",
        "question": "Both fits see the same six locations. What differs between them, and why is comparing predictions on the fixed probes the right comparison?",
        "code": r"""
from itertools import combinations
import numpy as np
from sklearn.metrics import adjusted_rand_score


def exact_line_centers(x, weight, k):
    # x is sorted; this small example has strictly positive weights.
    best = None
    for cuts in combinations(range(1, len(x)), k-1):
        groups = np.split(np.arange(len(x)), cuts)
        centers = tuple(np.average(x[g], weights=weight[g]) for g in groups)
        cost = sum(np.sum(weight[g] * (x[g]-center)**2)
                   for g, center in zip(groups, centers))
        candidate = (float(cost), centers)
        if best is None or candidate < best:
            best = candidate
    return np.array(best[1])


x = np.array([0, 1, 4, 5, 8, 9.0])
weights = [np.array([3, 3, 1, 1, 1, 1]), np.array([1, 1, 1, 1, 3, 3])]
predictions = []
for weight in weights:
    centers = exact_line_centers(x, weight, 2)
    labels = np.abs(x[:, None] - centers).argmin(axis=1)
    predictions.append(labels)
    print("centers:", centers.tolist(), "probe labels:", labels.tolist())
print("probe ARI:", round(adjusted_rand_score(*predictions), 6))
print("one-group ARI:", adjusted_rand_score(np.zeros(6), np.zeros(6)))
""",
    },
    "report": {
        "title": "Program 6 — freeze the decision before opening the report",
        "question": "Which rows fit the scaler, which rows choose k, and which rows are touched exactly once? Trace each print statement to its population.",
        "needsCsv": True,
        "code": r"""
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

data = np.genfromtxt(Path(__file__).with_name("iris.csv"), delimiter=",", skip_header=1)
x, species = data[:, 1:5], data[:, 5].astype(int)
order = np.random.default_rng(23).permutation(len(x))
fit_ids, select_ids, report_ids = order[:90], order[90:120], order[120:]
candidates = []
for k in [2, 3, 4]:
    model = make_pipeline(StandardScaler(),
                          KMeans(n_clusters=k, n_init=20, random_state=17))
    model.fit(x[fit_ids])
    selected_labels = model.predict(x[select_ids])
    group_count = len(np.unique(selected_labels))
    sizes = np.bincount(model[-1].labels_, minlength=k)
    if sizes.min() < 10 or not 2 <= group_count < len(select_ids):
        continue
    score = silhouette_score(model[0].transform(x[select_ids]), selected_labels)
    candidates.append((score, -k, model))
if not candidates:
    raise ValueError("No eligible candidate under the development rule.")
selection_score, negative_k, model = max(candidates, key=lambda row: row[:2])

# All choices above are now frozen.
report_labels = model.predict(x[report_ids])
baseline = make_pipeline(StandardScaler(),
                         KMeans(n_clusters=1, n_init=1, random_state=17))
baseline.fit(x[fit_ids])
error = np.mean(model.transform(x[report_ids]).min(axis=1)**2)
base_error = np.mean(baseline.transform(x[report_ids]).min(axis=1)**2)
print("split sizes:", len(fit_ids), len(select_ids), len(report_ids))
print("selected k:", -negative_k, "selection silhouette:", round(selection_score, 6))
print("report distortion / baseline:", round(error, 6), round(base_error, 6))
print("report species ARI:", round(adjusted_rand_score(species[report_ids], report_labels), 6))
""",
    },
}

module_path = Path("src/learn/data/clustering-evaluation-examples.js")
evidence_path = Path("docs/teaching/evidence/clustering-evaluation-native.json")
write = "--write" in sys.argv
digest = lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()

workdir = Path(tempfile.mkdtemp(prefix="clustering-evaluation-programs-"))
shutil.copy(CSV, workdir / "iris.csv")
records = {}
namespaces = {}
try:
    for key, item in PROGRAMS.items():
        code = item["code"].strip() + "\n"
        namespace = {"__file__": str(workdir / f"{key}.py"), "__name__": "__main__"}
        with threadpool_limits(limits=1), contextlib.redirect_stdout(io.StringIO()) as output:
            exec(compile(code, f"clustering-evaluation:{key}", "exec"), namespace)
        namespaces[key] = namespace
        records[key] = {"title": item["title"], "question": item["question"], "code": code.rstrip("\n"), "expected": output.getvalue().strip(), "language": "python"}
        print(f"Executed {key}: {len(code.splitlines())} lines, {len(records[key]['expected'].splitlines())} output lines")
finally:
    shutil.rmtree(workdir, ignore_errors=True)

if not write:
    existing = module_path.read_text(encoding="utf-8")
    prefix = "export const clusteringEvaluationExamples = "
    recorded = json.loads(existing[existing.index(prefix) + len(prefix):].rstrip().rstrip(";"))
    for key, record in records.items():
        assert recorded[key]["code"] == record["code"], f"{key}: displayed code changed"
        assert recorded[key]["expected"] == record["expected"], f"{key}: output changed:\n{recorded[key]['expected']}\n---\n{record['expected']}"

# Oracle assertions against manuscript values.
s = namespaces["silhouette"]
assert np.allclose(s["scores"], [0.8125, 0.857143, 0.75, 0.75, 0.857143, 0.8125], atol=1e-6)
assert abs(s["silhouettes"](s["x"], s["changed"])[2] + 0.75) < 1e-12
p = namespaces["pairs"]
assert (p["together_both"], p["together_u"], p["together_v"], p["pairs"]) == (6, 12, 12, 28) and abs(p["ari"] - 0.125) < 1e-12
c = namespaces["chance"]
assert c["overlaps"].tolist() == [1, 16, 36, 16, 1] and abs(c["means"][1] - 0.114844) < 5e-6
author = json.load(open("docs/teaching/drafts/clustering-evaluation-validation-silhouette-ari-nmi/author-calculations.json", encoding="utf-8"))
for line in records["iris"]["expected"].splitlines():
    name, k, sil, ari, ami, negative = line.split()
    match = next(row for row in author["iris"] if row["representation"] == name and row["k"] == int(k))
    assert abs(float(sil[4:]) - match["silhouette"]) < 5e-7 and abs(float(ari[4:]) - match["ARI"]) < 5e-7 and abs(float(ami[4:]) - match["AMI"]) < 5e-7 and int(negative[9:]) == match["negative"], line
pr = namespaces["probes"]
assert [row.tolist() for row in pr["predictions"]] == [[0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1]]
r = namespaces["report"]
report = json.load(open("docs/teaching/drafts/clustering-evaluation-validation-silhouette-ari-nmi/report-author-calculation.json", encoding="utf-8"))
assert -r["negative_k"] == report["selected_k"] and abs(r["selection_score"] - report["selection_silhouette"]) < 1e-9 and abs(r["error"] - report["report_distortion"]) < 1e-9 and abs(r["base_error"] - report["report_baseline_distortion"]) < 1e-9
assert sorted(r["fit_ids"].tolist()) == sorted(report["fit_ids"]) and sorted(r["report_ids"].tolist()) == sorted(report["report_ids"])
oracle_count = 11

if write:
    module_path.write_text("// Complete displayed programs for the clustering-evaluation lesson, executed by scripts/verify-clustering-evaluation-examples.py.\n// Programs 4 and 6 read the lesson's iris.csv from the same folder they are saved in.\nexport const clusteringEvaluationExamples = " + json.dumps(records, ensure_ascii=False, indent=2) + ";\n", encoding="utf-8")

evidence = {
    "verifiedAt": datetime.now(timezone.utc).isoformat(),
    "stage": "native author verification of displayed programs; browser, independent and integration review are separate",
    "source": str(module_path).replace("\\", "/"),
    "sourceHash": hashlib.sha256(module_path.read_bytes()).hexdigest(),
    "verifier": "scripts/verify-clustering-evaluation-examples.py",
    "verifierHash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "csvHash": hashlib.sha256(CSV.read_bytes()).hexdigest(),
    "versions": {name: importlib.metadata.version(name) for name in ["numpy", "scipy", "scikit-learn"]},
    "programs": {key: {"codeHash": digest(record["code"]), "stdoutHash": digest(record["expected"]), "stdout": record["expected"]} for key, record in records.items()},
    "oracles": oracle_count,
    "limits": ["Iris values are the scikit-learn bundled corrected copy, exported to the lesson CSV; no UCI download.", "Single-threaded BLAS and OMP; fold-free programs, so no timing claims.", "Manuscript-value oracles compare six-decimal printed values within 5e-7."],
}
evidence_path.parent.mkdir(parents=True, exist_ok=True)
evidence_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
print(f"PASS: {len(records)} displayed programs executed, {oracle_count} oracle assertions.")
