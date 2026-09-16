"""Bounded content calculations; not a production/browser verifier.

Run with Python 3.12, NumPy and scikit-learn from this directory.
The checked-in CSV and JSON are offline teaching inputs for phase two.
"""
from pathlib import Path
import csv
import hashlib
import json
import platform
import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import pairwise_distances
from scipy.sparse.csgraph import shortest_path

ROOT = Path(__file__).resolve().parent


def neighbor_order(points):
    distance = pairwise_distances(points)
    np.fill_diagonal(distance, np.inf)
    return np.argsort(distance, axis=1, kind="stable")


def retention(original, layout, k):
    before, after = neighbor_order(original), neighbor_order(layout)
    return np.mean([len(set(a[:k]) & set(b[:k])) / k
                    for a, b in zip(before, after)])


def conditional_row(squared_distances, perplexity):
    lower, upper, beta = 0.0, np.inf, 1.0
    distances = np.asarray(squared_distances, dtype=float)
    for _ in range(80):
        weights = np.exp(-beta * (distances - distances.min()))
        probabilities = weights / weights.sum()
        positive = probabilities > 0
        entropy = -np.sum(probabilities[positive] * np.log(probabilities[positive]))
        if abs(entropy - np.log(perplexity)) < 1e-10:
            break
        if entropy > np.log(perplexity):
            lower = beta
            beta = 2 * beta if np.isinf(upper) else (lower + upper) / 2
        else:
            upper = beta
            beta = (lower + upper) / 2
    return probabilities, beta


def joint_probabilities(points, perplexity):
    squared = pairwise_distances(points, squared=True)
    n = len(points)
    conditional = np.zeros((n, n))
    for i in range(n):
        others = np.arange(n) != i
        conditional[i, others], _ = conditional_row(squared[i, others], perplexity)
    return (conditional + conditional.T) / (2 * n)


def objective_gradient(P, Y):
    offsets = Y[:, None, :] - Y[None, :, :]
    kernel = 1 / (1 + np.sum(offsets ** 2, axis=2))
    np.fill_diagonal(kernel, 0)
    Q = kernel / kernel.sum()
    mask = P > 0
    cost = np.sum(P[mask] * np.log(P[mask] / Q[mask]))
    gradient = 4 * np.sum(((P - Q) * kernel)[:, :, None] * offsets, axis=1)
    return float(cost), gradient


def main():
    digits = load_digits()
    selected = np.sort(np.concatenate([np.flatnonzero(digits.target == label)[:30]
                                      for label in range(10)]))
    with (ROOT / "digits-300.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["source_row", "digit"] + [f"pixel_{i}" for i in range(64)])
        for row in selected:
            writer.writerow([row, digits.target[row], *digits.data[row].astype(int)])
    X = digits.data[selected] / 16.0
    layouts = {"pca": PCA(n_components=2, svd_solver="full").fit_transform(X)}
    fits = {}
    for perplexity in [5, 30, 80]:
        for seed in [7, 19]:
            key = f"tsne-p{perplexity}-s{seed}"
            estimator = TSNE(n_components=2, perplexity=perplexity, init="pca",
                             learning_rate="auto", max_iter=1000, random_state=seed,
                             method="barnes_hut", angle=0.5, n_jobs=1)
            layouts[key] = estimator.fit_transform(X)
            fits[key] = {"kl": float(estimator.kl_divergence_)}
    for seed in [7, 19]:
        key = f"tsne-p30-random-s{seed}"
        estimator = TSNE(n_components=2, perplexity=30, init="random",
                         learning_rate="auto", max_iter=1000, random_state=seed,
                         method="barnes_hut", angle=0.5, n_jobs=1)
        layouts[key] = estimator.fit_transform(X)
        fits[key] = {"kl": float(estimator.kl_divergence_)}
    layout_results = {}
    for name, Y in layouts.items():
        layout_results[name] = {
            "coordinates": Y.tolist(),
            "metrics": {str(k): {"retention": float(retention(X, Y, k)),
                                 "trustworthiness": float(trustworthiness(X, Y, n_neighbors=k)),
                                 "continuity": float(trustworthiness(Y, X, n_neighbors=k))}
                        for k in [5, 10, 20]},
            **fits.get(name, {})}
    toy_X = np.array([[0], [1], [3], [7]], dtype=float)
    P = joint_probabilities(toy_X, 2.0)
    Y = np.array([[-1.5], [-0.5], [0.5], [1.5]])
    initial_cost, gradient = objective_gradient(P, Y)
    finite_difference = np.zeros_like(Y)
    for i in range(len(Y)):
        delta = np.zeros_like(Y)
        delta[i, 0] = 1e-6
        finite_difference[i] = (objective_gradient(P, Y + delta)[0] -
                                objective_gradient(P, Y - delta)[0]) / 2e-6
    before = Y.copy()
    for _ in range(200):
        _, gradient = objective_gradient(P, Y)
        Y -= 0.5 * gradient
        Y -= Y.mean(axis=0)
    graph_points = np.array([[0, 0], [0, 1], [0, 2], [1, 2], [2, 2], [2, 1], [2, 0]])
    distance = pairwise_distances(graph_points)
    graph_results = {}
    for radius in [0.75, 1.0, 1.5, 2.0]:
        graph = np.where((distance <= radius) & (distance > 0), distance, 0)
        geodesic = shortest_path(graph, directed=False)[0, -1]
        graph_results[str(radius)] = None if np.isinf(geodesic) else float(geodesic)
    rows = {}
    for label, distances in {"unequal": [1, 4, 9], "equal": [4, 4, 4]}.items():
        rows[label] = {}
        for sigma in [0.5, 1, 2]:
            weights = np.exp(-np.asarray(distances) / (2 * sigma * sigma))
            probabilities = weights / weights.sum()
            rows[label][str(sigma)] = {"probabilities": probabilities.tolist(),
                "perplexity": float(np.exp(-np.sum(probabilities * np.log(probabilities))))}
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    projected = square[:, :1]
    J = np.eye(3) - np.ones((3, 3)) / 3
    D = np.array([[0, 2, 5], [2, 0, 3], [5, 3, 0]], dtype=float)
    B = -0.5 * J @ D**2 @ J
    record = {
        "scope": "Content probes and offline inputs; no UMAP execution or phase-two verification",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "sklearn": sklearn.__version__},
        "csv_sha256": hashlib.sha256((ROOT / "digits-300.csv").read_bytes()).hexdigest(),
        "selection": "First 30 source rows for each target 0..9, merged in ascending source-row order",
        "rows": selected.tolist(), "digit_counts": [30] * 10,
        "layouts": layout_results,
        "fixtures": {
            "u_chain_endpoint_distances": graph_results,
            "probability_rows": rows,
            "t_sne_tiny": {"P": P.tolist(), "initial_cost": initial_cost,
                            "initial_gradient": objective_gradient(P, before)[1].tolist(),
                            "gradient_max_error": float(np.max(abs(objective_gradient(P, before)[1] - finite_difference))),
                            "final_Y": Y.tolist(), "final_cost": objective_gradient(P, Y)[0]},
            "fuzzy_union": {"0.5,0.25": 0.5 + 0.25 - 0.5 * 0.25,
                            "0.5,0": 0.5, "0,0": 0, "1,1": 1},
            "pair_objective_w_0.625": {str(r): float(-0.625*np.log(1/(1+r*r)) -
                                   0.375*np.log(1-1/(1+r*r))) for r in [0.5, 1, 2]},
            "classical_mds": {"B": B.tolist(), "eigenvalues": np.linalg.eigvalsh(B).tolist()},
            "square_rips": {"input_side": 1, "input_diagonal": float(np.sqrt(2)),
                            "projection_duplicate_pairs": [[0, 3], [1, 2]]},
            "false_neighbors": {"X": toy_X.tolist(), "Y": [[0], [5], [1], [11]],
                                "retention1": float(retention(toy_X, np.array([[0], [5], [1], [11]]), 1)),
                                "trustworthiness1": float(trustworthiness(toy_X, np.array([[0], [5], [1], [11]]), n_neighbors=1))},
            "null_layout": {"retention1": float(retention(toy_X, toy_X, 1)),
                            "trustworthiness1": float(trustworthiness(toy_X, toy_X, n_neighbors=1))},
            "rigid_layout": {"retention10": float(retention(X, -X + 3, 10))}}
    }
    (ROOT / "calculated-inputs.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"versions": record["versions"], "csv_sha256": record["csv_sha256"],
                      "metrics10": {name: obj["metrics"]["10"] for name, obj in layout_results.items()},
                      "fixtures": record["fixtures"]}, indent=2))


if __name__ == "__main__":
    main()
