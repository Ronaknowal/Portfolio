"""Complementary finite checks for Spectral31; independent of its author suite."""
import contextlib
import hashlib
import io
import json
import subprocess
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scratch/spectral-independent-review"
OUT.mkdir(parents=True, exist_ok=True)


def laplacian(n, edges):
    matrix = [[F(0) for _ in range(n)] for _ in range(n)]
    degree = [F(0) for _ in range(n)]
    for a, b, w in edges:
        w = F(w)
        matrix[a][a] += w
        matrix[b][b] += w
        matrix[a][b] -= w
        matrix[b][a] -= w
        degree[a] += w
        degree[b] += w
    return matrix, degree


def exact_solve(matrix, rhs):
    n = len(rhs)
    augmented = [list(row) + [value] for row, value in zip(matrix, rhs)]
    for col in range(n):
        pivot = next(row for row in range(col, n) if augmented[row][col])
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        pivot_value = augmented[col][col]
        augmented[col] = [x / pivot_value for x in augmented[col]]
        for row in range(n):
            if row != col:
                factor = augmented[row][col]
                augmented[row] = [x - factor * y for x, y in zip(augmented[row], augmented[col])]
    return [row[-1] for row in augmented]


def energy(edges, vector):
    return sum(F(w) * (vector[a] - vector[b]) ** 2 for a, b, w in edges)


graphs = [
    (4, [(0, 1, 2), (1, 2, 1), (2, 3, 3)]),
    (4, [(0, 1, 2), (0, 2, 3), (1, 2, 5), (2, 3, 7)]),
    (4, [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 0, 1)]),
]
threshold_records = []
for n, edges in graphs:
    _, degree = laplacian(n, edges)
    for raw in [[0, 0, 0, 1], [2, -1, 4, -3], [1, 1, -2, -2]]:
        mean = sum(d * x for d, x in zip(degree, raw)) / sum(degree)
        centered = [F(x) - mean for x in raw]
        order = sorted(range(n), key=lambda i: centered[i])
        total = F(0)
        for i in order:
            total += degree[i]
            if total * 2 >= sum(degree):
                median = centered[i]
                break
        z = [x - median for x in centered]
        norm = min(z) ** 2 + max(z) ** 2
        cuts = sorted(set(z + [F(0)]))
        expected_cut = F(0)
        expected_volume = F(0)
        membership = [F(0)] * n
        actual_phis = []
        mass_sum = F(0)
        for left, right in zip(cuts, cuts[1:]):
            t = (left + right) / 2
            mass = (right * abs(right) - left * abs(left)) / norm
            mass_sum += mass
            small = {i for i in range(n) if (z[i] > t if t > 0 else z[i] < t)}
            volume = sum(degree[i] for i in small)
            assert 2 * volume <= sum(degree)
            crossing = sum(F(w) for a, b, w in edges if (a in small) != (b in small))
            expected_volume += mass * volume
            expected_cut += mass * crossing
            for i in small:
                membership[i] += mass
            if volume:
                actual_phis.append(crossing / volume)
        dz = sum(d * x * x for d, x in zip(degree, z))
        dy = sum(d * x * x for d, x in zip(degree, centered))
        rho = energy(edges, centered) / dy
        assert mass_sum == 1
        assert membership == [x * x / norm for x in z]
        assert expected_volume == dz / norm
        assert expected_cut ** 2 <= 2 * energy(edges, z) * dz / norm ** 2
        assert min(actual_phis) ** 2 <= 2 * rho
        threshold_records.append({
            "edges": edges, "raw": raw, "median": str(median),
            "expectedVolume": str(expected_volume), "expectedCut": str(expected_cut),
            "smallSideProbabilities": [str(value) for value in membership],
            "bestThresholdConductance": str(min(actual_phis)),
        })

node_program = """
import fs from 'node:fs';
import {graphFromEdges,symmetricSpectrum,filterState,clusteringState} from './src/learn/data/spectral-graph-models.js';
import {spectralGraphExamples} from './src/learn/data/spectral-graph-examples.js';
const fixtures=JSON.parse(fs.readFileSync(0,'utf8'));
console.log(JSON.stringify({
  fixtures:fixtures.map(([n,edges])=>{const graph=graphFromEdges(n,edges);return {...graph,spectrum:symmetricSpectrum(graph.laplacian),normalizedSpectrum:symmetricSpectrum(graph.normalized)};}),
  filters:[0,.17,1.3].flatMap(w=>['heat','ridge'].flatMap(kind=>[0,.3,3].map(amount=>({w,kind,amount,state:filterState(w,'spike',kind,amount)})))),
  clusters:[0,.12,1.3].map(weight=>clusteringState(weight,'nearby')),
  examples:spectralGraphExamples
}));
"""
response = subprocess.run(
    ["node", "--input-type=module", "-e", node_program], cwd=ROOT,
    input=json.dumps(graphs), text=True, capture_output=True, check=True
)
native = json.loads(response.stdout)
from scipy.linalg import expm

spectral_cases = 0
for (n, edges), result in zip(graphs, native["fixtures"]):
    exact_l, degree = laplacian(n, edges)
    L = np.array(exact_l, dtype=float)
    U = np.array(result["spectrum"]["vectors"]).T
    values = np.array(result["spectrum"]["values"])
    np.testing.assert_allclose(L @ U, U * values, atol=1e-11)
    np.testing.assert_allclose(U @ U.T, np.eye(n), atol=1e-12)
    normal = np.array(result["normalized"])
    V = np.array(result["normalizedSpectrum"]["vectors"]).T
    y = V[:, 1] / np.sqrt(np.array(degree, dtype=float))
    nu = result["normalizedSpectrum"]["values"][1]
    np.testing.assert_allclose(L @ y, nu * np.array(degree, dtype=float) * y, atol=1e-11)
    np.testing.assert_allclose(y @ np.array(degree, dtype=float), 0, atol=1e-11)
    np.testing.assert_allclose(normal @ V[:, 1], nu * V[:, 1], atol=1e-11)
    spectral_cases += 1

for result in native["filters"]:
    state = result["state"]
    L = np.array(state["laplacian"])
    x = np.array(state["signal"])
    expected = expm(-result["amount"] * L) @ x if result["kind"] == "heat" else np.linalg.solve(np.eye(6) + result["amount"] * L, x)
    np.testing.assert_allclose(state["output"], expected, atol=2e-11)
    np.testing.assert_allclose(state["discarded"], np.sum((expected - x) ** 2), atol=2e-11)
    assert expected.min() >= -2e-12 and expected.max() <= 1 + 2e-12
    np.testing.assert_allclose(expected.sum(), 1, atol=2e-12)

namespaces = {}
for example in native["examples"]:
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(example["code"], example["id"] + ".py", "exec"), namespace)
    namespaces[example["id"]] = namespace

# Exact changed circuits, not the author's two-triangle fixture.
circuit_records = []
for n, edges in graphs:
    L, _ = laplacian(n, edges)
    for a, b in [(0, 1), (0, 3), (1, 2)]:
        keep = [i for i in range(n) if i != b]
        rhs = [F(int(i == a)) for i in keep]
        solution = exact_solve([[L[i][j] for j in keep] for i in keep], rhs)
        exact = solution[keep.index(a)]
        actual = namespaces["resistance"]["effective_resistance"](np.array(L, dtype=float), a, b)
        np.testing.assert_allclose(actual, float(exact), atol=1e-12)
        circuit_records.append({"edges": edges, "endpoints": [a, b], "exact": str(exact)})

# Disconnected unequal-degree components: node-row geometry survives label and basis changes.
A = np.zeros((7, 7))
for i, j, weight in [(0, 1, 2), (1, 2, 3), (3, 4, 1), (4, 5, 4), (3, 5, 2)]:
    A[i, j] = A[j, i] = weight
# Vertex 6 tests the explicit isolate rejection before it is removed.
try:
    namespaces["clusters"]["spectral_rows"](A, 2)
    raise AssertionError("isolated row was accepted")
except ValueError:
    pass
A = A[:6, :6]
_, rows = namespaces["clusters"]["spectral_rows"](A, 2)
expected_distances = np.array([[0 if (i < 3) == (j < 3) else 2 for j in range(6)] for i in range(6)])
np.testing.assert_allclose(((rows[:, None] - rows[None]) ** 2).sum(axis=2), expected_distances, atol=1e-12)
permutation = [5, 1, 3, 0, 4, 2]
_, permuted = namespaces["clusters"]["spectral_rows"](A[np.ix_(permutation, permutation)], 2)
np.testing.assert_allclose(((permuted[:, None] - permuted[None]) ** 2).sum(axis=2), expected_distances[np.ix_(permutation, permutation)], atol=1e-12)
rotation = np.array([[F(3, 5), F(-4, 5)], [F(4, 5), F(3, 5)]], dtype=float)
np.testing.assert_allclose(rows @ rows.T, (rows @ rotation) @ (rows @ rotation).T, atol=1e-12)
for cluster in native["clusters"]:
    points = np.array(cluster["points"])
    previous_loss = float("inf")
    for frame in cluster["frames"][1:]:
        centers = np.array(frame["centroids"])
        labels = np.array(frame["labels"])
        distances = ((points[:, None] - centers[None]) ** 2).sum(axis=2)
        assert frame["loss"] <= previous_loss + 1e-12
        np.testing.assert_allclose(frame["loss"], distances[np.arange(9), labels].sum(), atol=1e-12)
        if frame["phase"] == "Move to means":
            for group in range(3):
                if (labels == group).any():
                    np.testing.assert_allclose(centers[group], points[labels == group].mean(axis=0), atol=1e-12)
        previous_loss = frame["loss"]

# Repeated eigenvalue: choose an analytic square-cycle basis to avoid solver matching.
u0 = np.ones(4) / 2
B = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float) / np.sqrt(2)
rotated = B @ rotation
np.testing.assert_allclose(B @ B.T, rotated @ rotated.T, atol=1e-15)
assert np.linalg.norm(np.outer(B[:, 0], B[:, 0]) - np.outer(rotated[:, 0], rotated[:, 0])) > .5
L, _ = laplacian(*graphs[2])
L = np.array(L, dtype=float)
u4 = np.array([1, -1, 1, -1]) / 2
ridge = np.outer(u0, u0) + (B @ B.T) / 3 + np.outer(u4, u4) / 5
np.testing.assert_allclose(ridge, np.linalg.solve(np.eye(4) + L, np.eye(4)), atol=1e-15)
record = {
    "checkedAt": datetime.now(timezone.utc).isoformat(), "allPassed": True,
    "weightedMedianExactThresholds": threshold_records,
    "changedSpectralCoordinateCases": spectral_cases,
    "heatAndRidgeIndependentMatrixFunctions": len(native["filters"]),
    "exactChangedCircuits": circuit_records,
    "embeddingChecks": "Unequal-degree disconnected components; explicit isolate rejection; vertex permutation; orthogonal zero-subspace rotation; three actual Lloyd traces and occupied center means.",
    "repeatedBlockChecks": "Analytic square-cycle basis; whole projector and ridge invariant; one retained direction changes.",
    "reviewedModelSha256": hashlib.sha256((ROOT / "src/learn/data/spectral-graph-models.js").read_bytes()).hexdigest(),
}
(OUT / "results.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(json.dumps({key: value for key, value in record.items() if key not in ["weightedMedianExactThresholds", "exactChangedCircuits"]}, indent=2))
