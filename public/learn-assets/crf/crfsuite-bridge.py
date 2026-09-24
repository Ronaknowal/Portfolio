"""Fit a small CRFsuite chain, then reconstruct its inference from its weights.

Install numpy and sklearn-crfsuite. Numerical dictionaries avoid implicit string
feature expansion. This is a contract check, not a text-tagging benchmark.
"""
import numpy as np
from sklearn_crfsuite import CRF


def features(words, include_suffix=True):
    return [dict(bias=1.0, capital=float(word[0].isupper()),
                 suffix_ing=float(include_suffix and word.lower().endswith("ing")),
                 BOS=float(i == 0), EOS=float(i == len(words) - 1))
            for i, word in enumerate(words)]


def infer(emissions, transitions):
    """The earlier log-space recurrence, adapted to expose the API bridge."""
    length, labels = emissions.shape
    forward, backward = np.empty_like(emissions), np.zeros_like(emissions)
    forward[0] = emissions[0]
    for t in range(1, length):
        forward[t] = emissions[t] + np.logaddexp.reduce(forward[t-1, :, None] + transitions, axis=0)
    logz = np.logaddexp.reduce(forward[-1])
    for t in range(length - 2, -1, -1):
        backward[t] = np.logaddexp.reduce(transitions + emissions[t+1] + backward[t+1], axis=1)
    nodes = np.exp(forward + backward - logz)
    best, parents = emissions[0].copy(), []
    for t in range(1, length):
        candidates = best[:, None] + transitions
        parents.append(candidates.argmax(axis=0))
        best = candidates.max(axis=0) + emissions[t]
    path = [int(best.argmax())]
    for parent in reversed(parents):
        path.append(int(parent[path[-1]]))
    return float(logz), nodes, path[::-1]


def chain_arrays(model, items):
    labels = list(model.classes_)
    emissions = np.array([[sum(value * model.state_features_.get((name, label), 0.0)
                               for name, value in item.items())
                           for label in labels] for item in items])
    transitions = np.array([[model.transition_features_.get((left, right), 0.0)
                             for right in labels] for left in labels])
    return labels, emissions, transitions


def fit(include_suffix=True):
    sentences = [["Maya", "is", "running"], ["Chen", "keeps", "walking"],
                 ["birds", "are", "singing"], ["Ron", "likes", "swimming"],
                 ["Walking", "helps", "Maya"], ["Swimming", "helps", "Chen"]]
    tags = [["N", "V", "V"]] * 4 + [["N", "V", "N"]] * 2
    model = CRF(algorithm="lbfgs", c1=0.0, c2=0.1, max_iterations=200,
                all_possible_states=True, all_possible_transitions=True)
    model.fit([features(words, include_suffix) for words in sentences], tags)
    return model


def main():
    model = fit()
    items = features(["Taylor", "enjoys", "running"])
    labels, emissions, transitions = chain_arrays(model, items)
    logz, nodes, path = infer(emissions, transitions)
    library_nodes = np.array([[row[label] for label in labels]
                              for row in model.predict_marginals_single(items)])
    library_path = model.predict_single(items)
    np.testing.assert_allclose(nodes, library_nodes, atol=3e-6)
    assert [labels[i] for i in path] == library_path
    score = sum(emissions[t, label] for t, label in enumerate(path))
    score += sum(transitions[left, right] for left, right in zip(path, path[1:]))
    model.tagger_.set(items)
    np.testing.assert_allclose(np.exp(score - logz), model.tagger_.probability(library_path), atol=3e-6)
    print("labels", labels)
    print("decoded", library_path)
    print("marginals", np.round(nodes, 6).tolist())
    print("marginal_max_error", f"{np.max(np.abs(nodes - library_nodes)):.2e}")
    unseen = [dict(item, never_seen=1.0) for item in items]
    np.testing.assert_allclose(
        [[r[label] for label in labels] for r in model.predict_marginals_single(unseen)], library_nodes)
    print("unseen_feature_max_change", 0.0)
    changed = fit(include_suffix=False)
    print("without_suffix", changed.predict_single(features(["Taylor", "enjoys", "running"], False)))


if __name__ == "__main__":
    main()
