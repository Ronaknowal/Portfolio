"""Author-only SSL calculations; no runtime implementation or browser tests."""
from pathlib import Path
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import LabelSpreading, SelfTrainingClassifier

HERE = Path(__file__).parent


def graph_fixtures():
    weights = np.zeros((4, 4))
    for i in range(3):
        weights[i, i + 1] = weights[i + 1, i] = 1
    known = np.array([0, 3])
    unknown = np.array([1, 2])
    values = np.array([0.0, 1.0])

    def hard(w):
        laplacian = np.diag(w.sum(axis=1)) - w
        solution = np.linalg.solve(
            laplacian[np.ix_(unknown, unknown)],
            w[np.ix_(unknown, known)] @ values,
        )
        state = np.array([0.0, 0.5, 0.5, 1.0])
        trace = [state.tolist()]
        transition = w / w.sum(axis=1)[:, None]
        for step in range(50):
            state = transition @ state
            state[known] = values
            if step < 5:
                trace.append(state.tolist())
        assert np.allclose(state[unknown], solution, atol=1e-12)
        return {"unlabeled": solution.tolist(), "trace": trace}

    bridge = weights.copy()
    bridge[1, 3] = bridge[3, 1] = 2
    separated = weights.copy()
    separated[1, 2] = separated[2, 1] = 0
    result = {"line": hard(weights), "bridge": hard(bridge), "separated": hard(separated)}
    degree = weights.sum(axis=1)
    symmetric = weights / np.sqrt(degree[:, None] * degree[None, :])
    seeds = np.zeros((4, 2))
    seeds[0, 0] = seeds[3, 1] = 1
    result["spreading"] = {}
    for alpha in [0.0, 0.2, 0.8]:
        direct = np.linalg.solve(np.eye(4) - alpha * symmetric, (1 - alpha) * seeds)
        state = seeds.copy()
        trace = [state.tolist()]
        for step in range(200):
            state = alpha * symmetric @ state + (1 - alpha) * seeds
            if step < 4:
                trace.append(state.tolist())
        assert np.allclose(state, direct, atol=1e-12)
        totals = direct.sum(axis=1, keepdims=True)
        normalized = np.divide(direct, totals, out=np.zeros_like(direct), where=totals > 0)
        result["spreading"][str(alpha)] = {
            "scores": direct.tolist(), "normalized": normalized.tolist(),
            "has_signal": (totals[:, 0] > 0).tolist(), "trace": trace,
        }
    result["symmetric_row_sums"] = symmetric.sum(axis=1).tolist()
    result["contraction"] = {"alpha": 0.99, "after_100": .99**100, "after_500": .99**500,
                             "steps_below_1e_6": int(np.ceil(np.log(1e-6)/np.log(.99)))}
    # A disconnected unlabeled two-node component has zero forcing under spreading.
    isolated = np.array([[0., 1.], [1., 0.]])
    assert np.array_equal(np.linalg.solve(np.eye(2)-.8*isolated, np.zeros((2, 2))), np.zeros((2, 2)))
    result["unanchored_component"] = {"spreading_scores": [[0,0],[0,0]], "classification": "unknown",
                                    "hard_solution": "not uniquely determined"}
    return result


def train_self(x_pool, initial_labels, threshold, audit_truth):
    labels = initial_labels.copy()
    rounds = []
    for iteration in range(1, 11):
        model = LogisticRegression(C=1, max_iter=500).fit(x_pool[labels >= 0], labels[labels >= 0])
        remaining = np.flatnonzero(labels < 0)
        if len(remaining) == 0:
            break
        probabilities = model.predict_proba(x_pool[remaining])
        confidence = probabilities.max(axis=1)
        accepted = confidence >= threshold
        rows = remaining[accepted]
        predicted = model.classes_[probabilities[accepted].argmax(axis=1)]
        rounds.append({"iteration": iteration, "accepted_rows": rows.tolist(),
                       "predicted": predicted.tolist(), "confidence": confidence[accepted].tolist(),
                       "wrong_offline_audit": int(np.sum(predicted != audit_truth[rows]))})
        if len(rows) == 0:
            break
        labels[rows] = predicted
    final = LogisticRegression(C=1, max_iter=500).fit(x_pool[labels >= 0], labels[labels >= 0])
    return final, labels, rounds


def real_data():
    data = np.genfromtxt(HERE/"banknote-subset.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
    raw = np.column_stack([data[name] for name in ["variance", "skewness", "curtosis", "entropy"]])
    truth = data["class"]
    pool = np.arange(320)
    development = np.arange(320, 400)
    test = np.arange(400, 480)
    # Shared label-free preprocessing; future inputs are excluded.
    scaler = StandardScaler().fit(raw[pool])
    features = scaler.transform(raw)
    seed_rows = np.concatenate([np.flatnonzero(truth[pool] == c)[:3] for c in [0, 1]])
    labels = np.full(320, -1)
    labels[seed_rows] = truth[seed_rows]
    models = {
        "supervised_lr": LogisticRegression(C=1, max_iter=500).fit(features[seed_rows], truth[seed_rows]),
        "supervised_knn": KNeighborsClassifier(n_neighbors=3).fit(features[seed_rows], truth[seed_rows]),
    }
    result = {"seed_rows": seed_rows.tolist(), "seed_source_rows": data["source_row"][seed_rows].tolist(),
              "scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(),
              "development": {}, "self_training_rounds": {}}
    for gamma in [.25, 1., 4.]:
        models[f"spreading_{gamma}"] = LabelSpreading(
            kernel="rbf", gamma=gamma, alpha=.2, max_iter=1000, tol=1e-6,
        ).fit(features[pool], labels)
    for threshold in [.8, .95]:
        model, final_labels, rounds = train_self(features[pool], labels, threshold, truth[pool])
        models[f"self_{threshold}"] = model
        result["self_training_rounds"][str(threshold)] = {
            "rounds": rounds, "final_count": int(np.sum(final_labels >= 0)),
            "pseudo_wrong": int(np.sum(final_labels[(labels < 0) & (final_labels >= 0)] != truth[pool][(labels < 0) & (final_labels >= 0)])),
            "unlabeled_remaining": int(np.sum(final_labels < 0)),
        }
        reference = SelfTrainingClassifier(
            LogisticRegression(C=1,max_iter=500), threshold=threshold, max_iter=10,
        ).fit(features[pool], labels)
        assert np.array_equal(model.predict(features[development]), reference.predict(features[development]))
    for name, model in models.items():
        predictions = model.predict(features[development])
        result["development"][name] = {"correct": int(np.sum(predictions == truth[development])),
                                      "total":80, "accuracy": float(np.mean(predictions == truth[development])),
                                      "predictions": predictions.tolist()}
    # Predeclared tie order prefers simpler supervised baseline in insertion order.
    selected = max(result["development"], key=lambda name:result["development"][name]["accuracy"])
    predictions = models[selected].predict(features[test])
    result["selected"] = selected
    result["final_test"] = {"correct":int(np.sum(predictions==truth[test])), "total":80,
                            "accuracy":float(np.mean(predictions==truth[test])),
                            "predictions":predictions.tolist(), "truth":truth[test].tolist()}
    return result


def teaching_fixtures():
    import runpy
    module = runpy.run_path(str(HERE/"cotrain-categories.py"))
    rows, initial, cotrain = module["ROWS"], module["INITIAL"], module["cotrain"]
    changed = list(rows)
    changed[2] = ("blue", "triangle")
    co = {"baseline": cotrain(rows, initial),
          "changed_bridge": cotrain(changed, initial),
          "duplicate_views": cotrain([(left, left) for left, _ in rows], initial),
          "no_anchors": cotrain(rows, [None]*len(rows))}
    assert co["baseline"]["rules"][0]["green"] == 0
    assert co["changed_bridge"]["rules"][0]["green"] == 1
    assert co["duplicate_views"]["labels"][0][3:6] == [None, None, None]
    assert co["no_anchors"]["history"][0]["offers"] == []

    def centroid(unlabeled):
        labeled_x, labeled_y = [-2., 2.], [0, 1]
        remaining = list(enumerate(unlabeled))
        trace = []
        for iteration in range(8):
            means = [float(np.mean([x for x,y in zip(labeled_x,labeled_y) if y==c])) for c in [0,1]]
            offers = []
            for index, value in remaining:
                logits = -np.square(np.array([value-means[0], value-means[1]]))
                weights = np.exp(logits-logits.max())
                p = weights/weights.sum()
                if p.max() >= .8:
                    offers.append({"index":index, "x":value, "label":int(np.argmax(p)), "confidence":float(p.max())})
            trace.append({"means":means, "boundary":sum(means)/2, "offers":offers})
            if not offers:
                break
            accepted = {offer["index"] for offer in offers}
            for offer in offers:
                labeled_x.append(offer["x"])
                labeled_y.append(offer["label"])
            remaining = [(index,value) for index,value in remaining if index not in accepted]
        return trace
    self_demo = {"baseline":centroid([-1.,0.,1.,3.]),
                 "changed_far_point":centroid([-1.,0.,1.,9.]),
                 "no_confident_points":centroid([0.,0.])}
    assert self_demo["baseline"][-1]["boundary"] == .5
    assert self_demo["changed_far_point"][-1]["boundary"] == 1.5
    assert len(self_demo["no_confident_points"]) == 1
    return {"co_training":co, "centroid_self_training":self_demo}


if __name__ == "__main__":
    result = {"graph": graph_fixtures(), "banknotes": real_data(), "teaching": teaching_fixtures()}
    (HERE/"checked-results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "graph": result["graph"],
        "development": {name:{k:v for k,v in values.items() if k!="predictions"}
                        for name,values in result["banknotes"]["development"].items()},
        "selected":result["banknotes"]["selected"],
        "final_test":{k:v for k,v in result["banknotes"]["final_test"].items() if k not in ["predictions","truth"]},
        "self_rounds":{name:{"counts":[len(r["accepted_rows"]) for r in value["rounds"]],
                            "wrong":[r["wrong_offline_audit"] for r in value["rounds"]]}
                       for name,value in result["banknotes"]["self_training_rounds"].items()},
    }, indent=2))
