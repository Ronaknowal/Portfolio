from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, log_loss, confusion_matrix
)

data = np.genfromtxt(Path(__file__).with_name("wine.csv"),
                     delimiter=",", names=True)
two = np.column_stack([data["alcohol"], data["color_intensity"]])
three = np.column_stack([two, data["flavanoids"]])
y = data["cultivar"].astype(int)
ids = np.arange(len(y))
development, test = train_test_split(
    ids, test_size=36, stratify=y, random_state=21
)
train, valid = train_test_split(
    development, test_size=36, stratify=y[development], random_state=22
)

def linear_model():
    return make_pipeline(
        StandardScaler(), LogisticRegression(C=1, max_iter=2000)
    )

candidates = {
    "majority": (DummyClassifier(strategy="prior"), two),
    "linear_two": (linear_model(), two),
    "forest_two": (RandomForestClassifier(
        n_estimators=100, max_depth=4, min_samples_leaf=3,
        random_state=21, n_jobs=1
    ), two),
    "linear_three": (linear_model(), three),
}
validation_predictions = {}
scores = {}
print("split sizes", len(train), len(valid), len(test))
for name, (model, matrix) in candidates.items():
    model.fit(matrix[train], y[train])
    pred = model.predict(matrix[valid])
    prob = model.predict_proba(matrix[valid])
    validation_predictions[name] = pred
    scores[name] = balanced_accuracy_score(y[valid], pred)
    print(name, round(scores[name], 6),
          round(accuracy_score(y[valid], pred), 6),
          round(log_loss(y[valid], prob), 6))

reference = validation_predictions["linear_two"]
for name in ["forest_two", "linear_three"]:
    pred = validation_predictions[name]
    fixed = np.sum((reference != y[valid]) & (pred == y[valid]))
    broken = np.sum((reference == y[valid]) & (pred != y[valid]))
    print("paired", name, int(fixed), int(broken))
for name in ["linear_two", "linear_three"]:
    wrong = validation_predictions[name] != y[valid]
    for lower in [True, False]:
        mask = (two[valid, 1] < 4) if lower else (two[valid, 1] >= 4)
        print("slice", name, "color<4" if lower else "color>=4",
              int(mask.sum()), int(wrong[mask].sum()))

eligible = ["linear_two", "forest_two", "linear_three"]
selected = max(eligible, key=lambda name: scores[name])
model, matrix = candidates[selected]
final_prediction = model.predict(matrix[test])
final_probability = model.predict_proba(matrix[test])
print("selected", selected)
print("test", round(balanced_accuracy_score(y[test], final_prediction), 6),
      round(accuracy_score(y[test], final_prediction), 6),
      round(log_loss(y[test], final_probability), 6))
print(confusion_matrix(y[test], final_prediction))
