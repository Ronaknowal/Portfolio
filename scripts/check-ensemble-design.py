"""Evaluate proposed teaching fixtures; this is not production verification."""
import itertools
import json
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

directory = Path("scratch/ensemble-methods")
directory.mkdir(exist_ok=True)
result = {"checkedAt": datetime.now(timezone.utc).isoformat(), "status": "design fixtures only"}

# A finite law: two independent fair signals; Y's success probability is their mean.
calibration = []
for a, b in itertools.product([0, 1], repeat=2):
    truth = F(a + b, 2)
    pa, pb = F(1 + 2 * a, 4), F(1 + 2 * b, 4)
    calibration.append({"signals": [a, b], "trueChance": str(truth), "pA": str(pa), "pB": str(pb), "mean": str((pa + pb) / 2)})
result["calibratedBaseCounterexample"] = calibration
result["bootstrapMissing"] = {"n6m6": str(F(5, 6) ** 6), "n6m3": str(F(5, 6) ** 3)}
result["threeIndependentError03"] = str(3 * F(3, 10) ** 2 * F(7, 10) + F(3, 10) ** 3)

# Different held-out prediction errors, not an assumed correlation curve.
ra = np.array([-3., -1., 2., 2.])
rb = np.array([1., 1., -1., -3.])
weight = np.dot(rb, rb - ra) / np.dot(ra - rb, ra - rb)
brute = minimize_scalar(lambda w: np.mean((w * ra + (1 - w) * rb) ** 2), bounds=(0, 1), method="bounded")
assert abs(weight - brute.x) < 1e-10
result["residualBlend"] = {"a": ra.tolist(), "b": rb.tolist(), "weightA": float(weight), "mseA": float(np.mean(ra ** 2)), "mseB": float(np.mean(rb ** 2)), "mseBlend": float(brute.fun)}

# OOF row ownership: a memorizing nearest-neighbor rule and a line.
x = np.arange(6., dtype=float).reshape(-1, 1)
y = np.array([1., 2., 2., 6., 7., 8.])
folds = [np.array([0, 3]), np.array([1, 4]), np.array([2, 5])]
oof = np.zeros((6, 2))
for held in folds:
    train = np.setdiff1d(np.arange(6), held)
    for j, estimator in enumerate([KNeighborsRegressor(1), LinearRegression()]):
        oof[held, j] = estimator.fit(x[train], y[train]).predict(x[held])
r1, r2 = oof[:, 0] - y, oof[:, 1] - y
weight = np.clip(np.dot(r2, r2 - r1) / np.dot(r1 - r2, r1 - r2), 0, 1)
query = np.array([.5, 2.5, 4.5]).reshape(-1, 1)
predictions = np.column_stack([estimator.fit(x, y).predict(query) for estimator in [KNeighborsRegressor(1), LinearRegression()]])
result["oof"] = {"matrix": oof.tolist(), "weightNearest": float(weight), "query": query[:, 0].tolist(), "fullRefitPredictions": predictions.tolist(), "stackPrediction": (predictions @ np.array([weight, 1 - weight])).tolist()}

# Actual current discrete SAMME uses twice the usual binary signed-vote alpha.
labels = np.array([0, 0, 1, 1, 0, 1])
ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, random_state=7), n_estimators=3, random_state=7).fit(x, labels)
result["samme"] = {"errors": ada.estimator_errors_[:3].tolist(), "weights": ada.estimator_weights_[:3].tolist()}
assert abs(ada.estimator_errors_[0] - 1 / 6) < 1e-15
assert abs(ada.estimator_weights_[0] - np.log(5)) < 1e-14

# Actual mixed probability/score meta-features and a nonpartition CV rejection.
xx = np.array([[-3., 0.], [-2., 1.], [-1., 0.], [0., 1.], [1., 0.], [2., 1.], [3., 0.], [4., 1.]])
yy = np.array([0, 0, 1, 0, 1, 1, 0, 1])
stack = StackingClassifier(estimators=[("bayes", GaussianNB()), ("margin", make_pipeline(StandardScaler(), LinearSVC()))], final_estimator=LogisticRegression(), cv=2).fit(xx, yy)
assert stack.stack_method_ == ["predict_proba", "decision_function"]
assert stack.transform(xx).shape == (8, 2)
result["mixedStack"] = {"methods": stack.stack_method_, "shape": list(stack.transform(xx).shape)}
try:
    StackingClassifier(estimators=[("line", LogisticRegression())], cv=TimeSeriesSplit(2)).fit(np.arange(12.).reshape(-1, 1), np.tile([0, 1], 6))
except ValueError as error:
    result["timeSeriesStackRejection"] = str(error)
else:
    raise AssertionError("Expected cross_val_predict partition restriction")
(directory / "design-fixtures.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2))
