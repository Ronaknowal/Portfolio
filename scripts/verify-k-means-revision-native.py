"""Read-only focused native/data oracle for the September K-Means revision.

Called by verify-k-means-revision-models.mjs with current exports on stdin.
Old evidence is never rewritten; only the two changed complete programs execute.
The new Faithful figure bindings are independently fitted here, including every
retained diagnostic, label, subsample index and linkage row.
"""

import os

for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[name] = "1"

import ast
import contextlib
import csv
import hashlib
import io
import json
import sys
from importlib.metadata import version
from pathlib import Path

import black
import numpy as np
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


packet = json.load(sys.stdin)
examples, faithful = packet["examples"], packet["faithful"]
checks = []
record = lambda name, **values: checks.append({"name": name, **values})


def close(actual, expected, tolerance=1e-10):
    assert np.allclose(actual, expected, rtol=0, atol=tolerance), (actual, expected)


with threadpool_limits(limits=1):
    scopes = {}
    for name in ["lloyd", "ward"]:
        scope, stream = {"__name__": "__main__"}, io.StringIO()
        with contextlib.redirect_stdout(stream):
            exec(compile(examples[name]["code"], f"revision:{name}", "exec"), scope)
        actual = stream.getvalue().rstrip()
        assert actual == examples[name]["expected"], name
        scopes[name] = scope
        record(f"complete changed {name} program preserves stdout", stdoutSha256=hashlib.sha256(actual.encode()).hexdigest())

    for name, call in [
        ("zero iteration budget", lambda: scopes["lloyd"]["lloyd"]([[0], [1]], k=1, max_iter=0)),
        ("negative iteration budget", lambda: scopes["lloyd"]["lloyd"]([[0], [1]], k=1, max_iter=-1)),
        ("fractional iteration budget", lambda: scopes["lloyd"]["lloyd"]([[0], [1]], k=1, max_iter=1.5)),
        ("boolean iteration budget", lambda: scopes["lloyd"]["lloyd"]([[0], [1]], k=1, max_iter=True)),
        ("Lloyd zero-column matrix", lambda: scopes["lloyd"]["lloyd"]([[], []], k=1)),
        ("Lloyd zero-row matrix", lambda: scopes["lloyd"]["lloyd"](np.empty((0, 2)), k=1)),
        ("Ward zero-column matrix", lambda: scopes["ward"]["ward_merges"]([[], []])),
        ("Ward zero-row matrix", lambda: scopes["ward"]["ward_merges"](np.empty((0, 2)))),
    ]:
        try:
            call()
        except ValueError:
            record(name)
        else:
            raise AssertionError(name)
    one = scopes["lloyd"]["lloyd"]([[0], [1], [4], [5], [10]], k=2, initial=[[0], [1]], max_iter=1)
    assert not one["converged"] and one["iterations"] == 1
    assert one["labels"].tolist() == [0, 0, 1, 1, 1]
    record("one-update budget keeps coherent nearest-center labels")

    # Parse the generator instead of importing it: importing executes its old
    # full campaign and overwrites the historical evidence that we must retain.
    with open("scripts/verify-k-means-hierarchical-examples.py", encoding="utf-8") as handle:
        generator = ast.parse(handle.read())
    generated_programs = ast.literal_eval(next(node.value for node in generator.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "PROGRAMS" for target in node.targets)))
    for name, item in generated_programs.items():
        assert black.format_str(item["code"].strip() + "\n", mode=black.Mode()).strip() == examples[name]["code"], name
    record("all six displayed programs match their generation source")

    # Independent bytes from the public mirror named by the lesson, not the
    # inline Python program or the JS array being checked.
    source_url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/faithful.csv"
    # Retained independent download from source_url, fetched for this review on
    # 21 September 2026. Keep future reproducibility independent of networking.
    source_bytes = Path("docs/teaching/evidence/k-means-revision-faithful-source.csv").read_bytes()
    rows = list(csv.DictReader(io.StringIO(source_bytes.decode())))
    source_data = np.array([[float(row["eruptions"]), float(row["waiting"])] for row in rows])
    data = np.array(faithful["faithfulPoints"])
    assert data.shape == (272, 2) and np.array_equal(data, source_data)
    record("all 272 data rows match the R datasets mirror", source=source_url, sha256=hashlib.sha256(source_bytes).hexdigest(), rows=len(rows))

    # Extract the data string without running the unchanged full production
    # program. Its recorded stdout is reused by exact code and output digests.
    program_ast = ast.parse(examples["production"]["code"])
    inline = ast.literal_eval(next(node.value for node in program_ast.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "FAITHFUL" for target in node.targets)))
    assert np.array_equal(data, np.array(inline.split(), dtype=float).reshape(-1, 2))
    record("all 272 displayed Python rows match figure data")

    scaler = StandardScaler().fit(data)
    scaled = scaler.transform(data)
    close(scaler.mean_, faithful["faithfulStandardization"]["mean"])
    close(scaler.scale_, faithful["faithfulStandardization"]["scale"])
    two = KMeans(n_clusters=2, n_init=10, random_state=0).fit(scaled)
    close(np.round(scaler.inverse_transform(two.cluster_centers_), 4), faithful["faithfulTwoGroups"]["centers"])
    assert two.labels_.tolist() == faithful["faithfulTwoGroups"]["labels"]
    assert np.bincount(two.labels_).tolist() == faithful["faithfulTwoGroups"]["sizes"]
    record("population standardization and all 272 two-center labels and summaries")
    diagnostics = []
    for item in faithful["faithfulDiagnostics"]:
        k = item["k"]
        models = [KMeans(n_clusters=k, n_init=1, random_state=seed).fit(scaled) for seed in range(5)]
        values = [model.inertia_ for model in models]
        close(values, item["inertias"], 5.01e-7)
        score = None if k == 1 else float(np.median([silhouette_score(scaled, model.labels_) for model in models]))
        if score is None:
            assert item["medianSilhouette"] is None
        else:
            close(score, item["medianSilhouette"], 5.01e-7)
        diagnostics.append({"k": k, "inertias": values, "medianSilhouette": score})
        record(f"all five exact diagnostics at k={k}, including median silhouette")

    sample = np.sort(np.random.default_rng(7).choice(len(data), 40, replace=False))
    assert sample.tolist() == faithful["faithfulSubsample"]
    tree = linkage(scaled[sample], method="ward")
    stored_tree = np.array(faithful["faithfulWardLinkage"])
    assert np.array_equal(tree[:, [0, 1, 3]], stored_tree[:, [0, 1, 3]])
    close(tree[:, 2], stored_tree[:, 2], 5.01e-7)
    record("40 subsample indices and all 39 SciPy linkage rows")

    # Complement SciPy reproduction with direct SSE increments for the exact
    # membership at every merge; do not trust the linkage's own height formula.
    groups = {index: [index] for index in range(40)}
    subset = scaled[sample]
    def sse(members):
        values = subset[members]
        return float(np.sum((values - values.mean(axis=0)) ** 2))
    for index, (left, right, height, size) in enumerate(tree):
        first, second = groups[int(left)], groups[int(right)]
        merged = first + second
        close(height * height / 2, sse(merged) - sse(first) - sse(second), 1e-11)
        assert len(merged) == size
        groups[40 + index] = merged
    record("all 39 Ward heights independently match direct union-minus-parts SSE")
    root_children = [groups[int(child)] for child in tree[-1, :2]]
    assert all(len(set(two.labels_[sample[members]])) == 1 for members in root_children)
    record("two top dendrogram groups coincide with the full-data two-center colors")

    raw = KMeans(n_clusters=2, n_init=10, random_state=0).fit(data)
    seconds = KMeans(n_clusters=2, n_init=10, random_state=0).fit(data * [1, 60])
    seconds_centers = seconds.cluster_centers_ / [1, 60]
    overall_rmse = float(np.std(data[:, 1]))
    cluster_rmse = float(np.sqrt(np.mean((data[:, 1] - scaler.inverse_transform(two.cluster_centers_)[two.labels_, 1]) ** 2)))
    interpretive = {
        "rawCentersMinutes": raw.cluster_centers_.tolist(),
        "secondsCentersConvertedToMinutes": seconds_centers.tolist(),
        "secondsVsRawARI": adjusted_rand_score(seconds.labels_, raw.labels_),
        "rawVsStandardizedARI": adjusted_rand_score(raw.labels_, two.labels_),
        "secondsVsStandardizedARI": adjusted_rand_score(seconds.labels_, two.labels_),
        "singleMeanWaitRMSE": overall_rmse,
        "twoGroupDescriptiveWaitRMSE": cluster_rmse,
        "fractionWaitSquaredErrorReduced": 1 - cluster_rmse ** 2 / overall_rmse ** 2,
        "limitation": "Both coordinates, including the waiting outcome, selected these fitted labels. These are descriptive in-sample errors, not a duration-only forecasting evaluation.",
    }
    record("changed waiting-unit and descriptive-error oracles", **interpretive)

print(json.dumps({"passed": True, "checks": checks, "versions": {name: version(name) for name in ["numpy", "scipy", "scikit-learn"]}, "diagnostics": diagnostics, "interpretive": interpretive}, allow_nan=False))
