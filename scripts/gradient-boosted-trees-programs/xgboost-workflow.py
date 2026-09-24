import tempfile
from pathlib import Path
import numpy as np
import xgboost as xgb
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=1000, n_features=10, n_informative=10,
                       noise=20, random_state=42)
X_development, X_test, y_development, y_test = train_test_split(X, y, test_size=.15, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_development, y_development, test_size=150, random_state=43)
assert (len(y_train), len(y_valid), len(y_test)) == (700, 150, 150)
rmse = lambda target, prediction: np.sqrt(np.mean((target - prediction) ** 2))
base = float(np.mean(y_train))
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)
dtest = xgb.DMatrix(X_test)  # No test labels enter training or selection.
params = dict(objective="reg:squarederror", eval_metric="rmse", max_depth=3,
              learning_rate=.1, subsample=.8, colsample_bytree=.8,
              reg_lambda=1., reg_alpha=0., tree_method="hist", seed=42,
              nthread=1, base_score=base)
history = {}
native = xgb.train(params, dtrain, num_boost_round=200,
                   evals=[(dtrain, "train"), (dvalid, "valid")],
                   early_stopping_rounds=12, evals_result=history, verbose_eval=False)
# Native predict uses the full stored ensemble unless an iteration range is given.
count = native.best_iteration + 1  # best_iteration is zero-based; upper bound exclusive.
prediction = native.predict(dtest, iteration_range=(0, count))
assert prediction.shape == y_test.shape
wrapper = xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=.1,
                          subsample=.8, colsample_bytree=.8, reg_lambda=1., reg_alpha=0.,
                          tree_method="hist", random_state=42, n_jobs=1,
                          base_score=base, eval_metric="rmse", early_stopping_rounds=12)
wrapper.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
assert wrapper.best_iteration == native.best_iteration
assert np.allclose(wrapper.predict(X_test), prediction, atol=1e-6, rtol=1e-6)
print("XGBoost", xgb.__version__, "; CPU; one thread; rows=700/150/150")
print(f"best zero-based round={native.best_iteration}; prediction rounds={count}; stored={native.num_boosted_rounds()}")
print(f"baseline test RMSE={rmse(y_test,np.full(len(y_test),base)):.6f}")
print(f"selected validation RMSE={history['valid']['rmse'][native.best_iteration]:.6f}")
print(f"locked-model test RMSE={rmse(y_test,prediction):.6f}; native/wrapper match=True")
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "regression.json"
    native.save_model(path)
    restored = xgb.Booster()
    restored.load_model(path)
    restored_prediction = restored.predict(dtest, iteration_range=(0, count))
    print(f"save/reload maximum prediction difference={np.max(np.abs(prediction-restored_prediction)):.1f}")
# Domain condition for this constructed data: increasing x0 should not reduce y.
monotone = xgb.XGBRegressor(n_estimators=count, max_depth=3, learning_rate=.1,
                           tree_method="hist", n_jobs=1, random_state=42,
                           monotone_constraints=(1,0,0,0,0,0,0,0,0,0))
monotone.fit(X_train, y_train)
grid = np.zeros((101, 10))
grid[:, 0] = np.linspace(-3, 3, 101)
print("specified monotone slice verified:", bool(np.all(np.diff(monotone.predict(grid)) >= -1e-7)))
