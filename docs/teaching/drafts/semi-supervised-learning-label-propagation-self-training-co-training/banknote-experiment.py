from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading

def self_train(x, initial, threshold):
    labels = initial.copy()
    history = []
    for round_number in range(1, 11):
        known = labels >= 0
        model = LogisticRegression(C=1, max_iter=500).fit(x[known], labels[known])
        remaining = np.flatnonzero(~known)
        if remaining.size == 0:
            break
        probability = model.predict_proba(x[remaining])
        accept = probability.max(axis=1) >= threshold
        rows = remaining[accept]
        guessed = model.classes_[probability[accept].argmax(axis=1)]
        history.append((rows.copy(), guessed.copy()))
        if rows.size == 0:
            break
        labels[rows] = guessed
    known = labels >= 0
    model = LogisticRegression(C=1, max_iter=500).fit(x[known], labels[known])
    return model, history

table = np.genfromtxt(
    Path(__file__).with_name("banknote-subset.csv"),
    delimiter=",", names=True, dtype=None, encoding="utf-8",
)
raw = np.column_stack([table[name] for name in
                       ["variance", "skewness", "curtosis", "entropy"]])
truth = table["class"]
assert len(table) == 480
pool, dev, test = np.arange(320), np.arange(320, 400), np.arange(400, 480)
x = StandardScaler().fit(raw[pool]).transform(raw)
seed = np.concatenate([np.flatnonzero(truth[pool] == c)[:3] for c in [0, 1]])
initial = np.full(320, -1)
initial[seed] = truth[seed]
models = {
    "supervised_lr": LogisticRegression(C=1, max_iter=500).fit(x[seed], truth[seed]),
    "supervised_knn": KNeighborsClassifier(3).fit(x[seed], truth[seed]),
}
histories = {}
for gamma in [.25, 1., 4.]:
    models[f"spreading_{gamma}"] = LabelSpreading(
        kernel="rbf", gamma=gamma, alpha=.2, max_iter=1000, tol=1e-6
    ).fit(x[pool], initial)
for threshold in [.8, .95]:
    model, history = self_train(x[pool], initial, threshold)
    models[f"self_{threshold}"] = model
    histories[threshold] = history

correct = {}
for name, model in models.items():
    correct[name] = int(np.sum(model.predict(x[dev]) == truth[dev]))
    print(name, correct[name], "/ 80")
# Dictionary order resolves ties in favor of the simpler baseline, as predeclared.
selected = max(correct, key=correct.get)
test_correct = int(np.sum(models[selected].predict(x[test]) == truth[test]))
print("selected", selected, "test", test_correct, "/ 80")
# Offline explanation only: these targets never enter self_train.
for threshold, history in histories.items():
    print("threshold", threshold, "accepted", [len(rows) for rows, _ in history])
    print("wrong in offline audit",
          [int(np.sum(guessed != truth[rows])) for rows, guessed in history])
