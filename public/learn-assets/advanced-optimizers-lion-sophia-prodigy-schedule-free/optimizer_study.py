"""Reproduce the declared 24-fit classroom comparison, not an LLM benchmark."""
from pathlib import Path
import hashlib
import json
import numpy as np
from optimizer_rules import Optimizer, probabilities, gradient, metrics

DIRECTORY = Path(__file__).resolve().parent
METHODS = {"adamw": [.01, .03], "adamw_cosine": [.01, .03],
           "lion": [.003, .01], "sophia_g": [.01, .03],
           "prodigy": [.3, 1.0], "schedule_free": [.01, .03]}
SEEDS = [11, 29]


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def load_data():
    raw = np.loadtxt(DIRECTORY / "digits-400.csv", delimiter=",", skiprows=1)
    features = np.column_stack((raw[:, 1:65] / 16, np.ones(len(raw))))
    labels = raw[:, -1].astype(int)
    roles = {name: [] for name in ("fit", "validation", "assessment")}
    generator = np.random.default_rng(925)
    for label in range(10):
        indices = generator.permutation(np.flatnonzero(labels == label))
        for role, subset in zip(roles, (indices[:24], indices[24:32], indices[32:])):
            roles[role].extend(subset.tolist())
    return raw, features, labels, roles


def main():
    raw, features, labels, roles = load_data()
    runs = []
    for method, rates in METHODS.items():
        for rate in rates:
            for seed in SEEDS:
                initial = np.random.default_rng(seed).normal(0, .01, (65, 10))
                optimizer = Optimizer(initial, method, rate)
                batches = np.random.default_rng(700 + seed)
                pseudo_labels = np.random.default_rng(1700 + seed)
                curve = []
                for step in range(1, 401):
                    indices = batches.choice(roles["fit"], size=64, replace=True)
                    inputs, targets = features[indices], labels[indices]
                    current_gradient = gradient(inputs, targets, optimizer.parameters)
                    curvature = None
                    if method == "sophia_g" and (step - 1) % 10 == 0:
                        distribution = probabilities(inputs, optimizer.parameters)
                        draws = pseudo_labels.random(64)
                        sampled = (draws[:, None] > np.cumsum(distribution, axis=1)).sum(axis=1)
                        sampled = np.minimum(sampled, 9)
                        sampled_gradient = gradient(inputs, sampled, optimizer.parameters)
                        curvature = 64 * sampled_gradient**2
                    diagnostics = optimizer.step(current_gradient, curvature)
                    if step in (1, 10) or step % 20 == 0:
                        curve.append({"step": step, **diagnostics,
                            "fit": metrics(features[roles["fit"]], labels[roles["fit"]],
                                           optimizer.evaluation_parameters()),
                            "validation": metrics(features[roles["validation"]], labels[roles["validation"]],
                                                  optimizer.evaluation_parameters()),
                            "training_iterate_validation": metrics(features[roles["validation"]],
                                labels[roles["validation"]], optimizer.parameters)})
                runs.append({"method": method, "rate": rate, "seed": seed, "curve": curve,
                    "validation": curve[-1]["validation"],
                    "parameters": optimizer.parameters.copy(), "state": optimizer.state})
    selected = {}
    for method, rates in METHODS.items():
        selected[method] = min(rates, key=lambda rate: np.mean([
            run["validation"]["cross_entropy"] for run in runs
            if run["method"] == method and run["rate"] == rate]))
    models, reports = [], []
    for run in runs:
        chosen = run["rate"] == selected[run["method"]]
        report = {key: value for key, value in run.items() if key not in ("parameters", "state")}
        report["selected_by_validation"] = chosen
        if chosen:
            evaluation = run["state"].get("average", run["parameters"])
            report["assessment"] = metrics(features[roles["assessment"]], labels[roles["assessment"]], evaluation)
            models.append({key: run[key] for key in ("method", "rate", "seed", "parameters", "state")})
        reports.append(report)
    results = {"dataset_sha256": hashlib.sha256((DIRECTORY / "digits-400.csv").read_bytes()).hexdigest(),
        "roles": {role: raw[indices, 0].astype(int).tolist() for role, indices in roles.items()},
        "parameters": 650, "updates_per_fit": 400, "batch_size": 64, "weight_decay": 0,
        "warmup_updates": 20, "seeds": SEEDS, "candidates": METHODS, "selected": selected,
        "curvature_refreshes_per_sophia_fit": 40, "majority_assessment_correct": 8,
        "runs": reports}
    (DIRECTORY / "study-results.json").write_text(json.dumps(results, indent=2, default=serializable, allow_nan=False) + "\n")
    (DIRECTORY / "fitted-optimizer-states.json").write_text(json.dumps(models, default=serializable, allow_nan=False) + "\n")
    for run in reports:
        if run["selected_by_validation"]:
            print(run["method"], run["rate"], run["seed"], run["assessment"])


if __name__ == "__main__":
    main()
