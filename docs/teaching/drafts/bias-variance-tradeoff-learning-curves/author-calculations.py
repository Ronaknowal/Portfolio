"""Bounded content calculations; no production or browser implementation."""
from itertools import product
import json
from pathlib import Path
import numpy as np
import sklearn
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split, learning_curve, validation_curve

folder = Path(__file__).parent

def finite_experiment(train_x, degree, curvature, sigma, grid):
    signs = np.array(list(product([-1., 1.], repeat=len(train_x))))
    targets = 1 + train_x + curvature * train_x**2 + sigma * signs
    design = np.vander(train_x, degree + 1, increasing=True)
    coefficients = np.linalg.lstsq(design, targets.T, rcond=None)[0]
    predictions = (np.vander(grid, degree + 1, increasing=True) @ coefficients).T
    truth = 1 + grid + curvature * grid**2
    mean = predictions.mean(axis=0)
    bias_squared = (mean - truth)**2
    variance = predictions.var(axis=0)
    excess = ((predictions - truth)**2).mean(axis=0)
    return {"trainX": train_x.tolist(), "degree": degree, "curvature": curvature,
            "sigma": sigma, "grid": grid.tolist(), "truth": truth.tolist(),
            "targets": targets.tolist(), "predictions": predictions.tolist(),
            "mean": mean.tolist(), "biasSquared": bias_squared.tolist(),
            "variance": variance.tolist(), "expectedError": (excess + sigma**2).tolist(),
            "identityMaxError": float(np.max(abs(excess - bias_squared - variance)))}

grid = np.linspace(-1.5, 1.5, 61)
finite = [finite_experiment(np.array([-1., 0., 1.]), degree, curvature, sigma, grid)
          for curvature in (0., 1.) for sigma in (0., .5, 1.) for degree in (0, 1, 2)]
finite += [finite_experiment(np.array([-1., -.5, 0., .5, 1.]), degree, 1., .5, grid)
           for degree in (0, 1, 2)]
data = np.loadtxt(folder / "airfoil-self-noise.dat")
x, y = data[:, :5], data[:, 5]
development, untouched = train_test_split(np.arange(len(y)), train_size=1200, random_state=41)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
sizes = [60, 120, 240, 480, 900]
models = {
    "mean": DummyRegressor(strategy="mean"),
    "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.)),
    "tree_leaf1": DecisionTreeRegressor(min_samples_leaf=1, random_state=43),
    "tree_leaf20": DecisionTreeRegressor(min_samples_leaf=20, random_state=43),
}
learning = {}
for name, model in models.items():
    actual_sizes, train, valid = learning_curve(model, x[development], y[development],
        train_sizes=sizes, cv=cv, scoring="neg_mean_squared_error", shuffle=True,
        random_state=44, n_jobs=1)
    learning[name] = {"sizes": actual_sizes.tolist(), "trainMse": (-train).tolist(),
                      "validationMse": (-valid).tolist()}
leaves = [1, 2, 5, 10, 20, 40]
train, valid = validation_curve(DecisionTreeRegressor(random_state=43), x[development],
    y[development], param_name="min_samples_leaf", param_range=leaves, cv=cv,
    scoring="neg_mean_squared_error", n_jobs=1)
stage_train, stage_valid = train_test_split(development, test_size=240, random_state=45)
boosted = GradientBoostingRegressor(n_estimators=120, max_depth=2, learning_rate=.1,
                                   random_state=46).fit(x[stage_train], y[stage_train])
stage = {"trainIds": stage_train.tolist(), "validationIds": stage_valid.tolist(),
         "trainMse": [float(np.mean((y[stage_train] - p)**2)) for p in boosted.staged_predict(x[stage_train])],
         "validationMse": [float(np.mean((y[stage_valid] - p)**2)) for p in boosted.staged_predict(x[stage_valid])]}
double_descent = []
for ratio in (.1, .25, .5, .75, .8, .9, .99, 1.01, 1.1, 1.5, 2., 3.):
    bias = (1 - ratio)**2 if ratio < 1 else 0.
    variance = ratio*(1-ratio) + .04*ratio/(1-ratio) if ratio < 1 else .04/(ratio-1)
    double_descent.append({"ratio": ratio, "biasSquared": bias, "varianceApprox": variance,
                           "riskApprox": bias + variance + .04})
result = {"environment": {"numpy": np.__version__, "sklearn": sklearn.__version__},
          "finiteExperiments": finite,
          "real": {"developmentIds": development.tolist(), "untouchedIds": untouched.tolist(),
                   "learning": learning, "validation": {"minSamplesLeaf": leaves,
                      "trainMse": (-train).tolist(), "validationMse": (-valid).tolist()},
                   "stage": stage}, "doubleDescentApproximation": double_descent}
folder.joinpath("calculated-inputs.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
print("finite identity max", max(row["identityMaxError"] for row in finite))
for name, rows in learning.items():
    print(name, [(n, round(float(np.mean(tr)), 4), round(float(np.mean(va)), 4))
                 for n, tr, va in zip(rows["sizes"], rows["trainMse"], rows["validationMse"])])
print("validation", [(n, round(float(np.mean(tr)), 4), round(float(np.mean(va)), 4))
                     for n, tr, va in zip(leaves, -train, -valid)])
print("stages", [(i+1, round(stage["trainMse"][i], 4), round(stage["validationMse"][i], 4))
                 for i in (0, 9, 29, 59, 119)])
print("best inspected stage", 1+int(np.argmin(stage["validationMse"])))
