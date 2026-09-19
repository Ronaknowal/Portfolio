import numpy as np

def infer(emissions, transitions):
    length, labels = emissions.shape
    forward = np.empty_like(emissions)
    backward = np.zeros_like(emissions)
    forward[0] = emissions[0]
    for t in range(1, length):
        forward[t] = emissions[t] + np.logaddexp.reduce(
            forward[t-1, :, None] + transitions, axis=0)
    log_partition = np.logaddexp.reduce(forward[-1])
    for t in range(length-2, -1, -1):
        backward[t] = np.logaddexp.reduce(
            transitions + emissions[t+1] + backward[t+1], axis=1)
    nodes = np.exp(forward + backward - log_partition)
    edges = np.array([
        np.exp(forward[t-1, :, None] + transitions
               + emissions[t] + backward[t] - log_partition)
        for t in range(1, length)])
    best = emissions[0].copy()
    parents = []
    for t in range(1, length):
        candidates = best[:, None] + transitions
        parents.append(candidates.argmax(axis=0))
        best = candidates.max(axis=0) + emissions[t]
    path = [int(best.argmax())]
    for parent in reversed(parents):
        path.append(int(parent[path[-1]]))
    return float(log_partition), nodes, edges, path[::-1]

emissions = np.log([[3., 1.], [1., 2.]])
transitions = np.log([[1., 4.], [1., 1.]])
logz, nodes, edges, path = infer(emissions, transitions)
print(round(np.exp(logz), 6))
print(np.round(nodes, 6))
print(path)

import json
from pathlib import Path
from scipy.optimize import minimize
from sklearn.feature_extraction import DictVectorizer

rows = json.loads(Path("ewt-sequences.json").read_text(encoding="utf-8"))

def tag_index(tag):
    return 0 if tag in ("NOUN", "PROPN") else 1 if tag in ("VERB", "AUX") else 2

def features(words):
    return [{"bias": 1., "word": word.lower(), "suffix": word.lower()[-2:],
             "capital": float(word.istitle()),
             "previous": words[i-1].lower() if i else "<START>",
             "next": words[i+1].lower() if i+1 < len(words) else "<END>"}
            for i, word in enumerate(words)]

vectorizer = DictVectorizer(sparse=False)
vectorizer.fit([f for row in rows if row["split"] == "train"
                for f in features(row["tokens"])])
groups = {split: [(vectorizer.transform(features(row["tokens"])),
                   np.array([tag_index(t) for t in row["upos"]]))
                  for row in rows if row["split"] == split]
          for split in ("train", "dev", "test")}
feature_count = len(vectorizer.feature_names_)

def fit_model(structured):
    def objective(theta):
        weights = theta[:feature_count*3].reshape(feature_count, 3)
        transitions = (theta[feature_count*3:].reshape(3, 3)
                       if structured else np.zeros((3, 3)))
        loss = 0.; grad_w = np.zeros_like(weights); grad_a = np.zeros((3, 3))
        for x, gold in groups["train"]:
            emissions = x @ weights
            logz, nodes, edges, _ = infer(emissions, transitions)
            loss += (logz - emissions[np.arange(len(gold)), gold].sum()
                     - transitions[gold[:-1], gold[1:]].sum())
            residual = nodes.copy()
            residual[np.arange(len(gold)), gold] -= 1
            grad_w += x.T @ residual
            if structured:
                grad_a += edges.sum(axis=0)
                np.add.at(grad_a, (gold[:-1], gold[1:]), -1.)
        gradient = (np.r_[grad_w.ravel(), grad_a.ravel()]
                    if structured else grad_w.ravel())
        count = len(groups["train"])
        return loss/count + .05*np.dot(theta, theta), gradient/count + .1*theta
    initial = np.zeros(feature_count*3 + (9 if structured else 0))
    result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                      options={"maxiter": 150, "ftol": 1e-11, "gtol": 1e-7})
    weights = result.x[:feature_count*3].reshape(feature_count, 3)
    transitions = (result.x[feature_count*3:].reshape(3, 3)
                   if structured else np.zeros((3, 3)))
    return weights, transitions, result

def evaluate(model, split):
    weights, transitions, _ = model
    correct = total = whole = 0
    for x, gold in groups[split]:
        _, _, _, path = infer(x @ weights, transitions)
        correct += (gold == path).sum()
        total += len(gold)
        whole += np.array_equal(gold, path)
    return float(correct/total), float(whole/len(groups[split]))

models = {"independent": fit_model(False), "chain": fit_model(True)}
for name, model in models.items():
    print(name, model[2].success, np.round(evaluate(model, "dev"), 6))
selected = max(models, key=lambda name: evaluate(models[name], "dev")[0])
print("selected", selected, np.round(evaluate(models[selected], "test"), 6))
