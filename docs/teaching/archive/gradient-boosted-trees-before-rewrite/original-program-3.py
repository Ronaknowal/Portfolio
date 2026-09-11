import numpy as np
import xgboost as xgb
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(42)
X, y = make_regression(n_samples=1000, n_features=10, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Native DMatrix API ---
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest  = xgb.DMatrix(X_test,  label=y_test)

params = {
    "objective":        "reg:squarederror",
    "max_depth":        4,
    "learning_rate":    0.1,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "lambda":           1.0,   # L2 on leaf weights
    "alpha":            0.0,   # L1 on leaf weights
    "tree_method":      "hist",
    # "device": "cuda",        # uncomment for GPU
    "seed":             42,
}
evals_result = {}
model_xgb = xgb.train(
    params, dtrain, num_boost_round=100,
    evals=[(dtrain, "train"), (dtest, "test")],
    early_stopping_rounds=10,
    evals_result=evals_result,
    verbose_eval=False,
)
preds = model_xgb.predict(dtest)
rmse  = np.sqrt(mean_squared_error(y_test, preds))
print(f"XGBoost (DMatrix)  best_iter={model_xgb.best_iteration}  test_RMSE={rmse:.4f}")
print(f"  train RMSE: {evals_result['train']['rmse'][model_xgb.best_iteration]:.4f}")
print(f"  test  RMSE: {evals_result['test']['rmse'][model_xgb.best_iteration]:.4f}")

# --- sklearn wrapper ---
from xgboost import XGBRegressor
xgb_sk = XGBRegressor(
    n_estimators=100, max_depth=4, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    tree_method="hist", early_stopping_rounds=10,
    random_state=42, eval_metric="rmse",
)
xgb_sk.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
sk_rmse = np.sqrt(mean_squared_error(y_test, xgb_sk.predict(X_test)))
print(f"\nXGBRegressor (sklearn)  best_iter={xgb_sk.best_iteration}  test_RMSE={sk_rmse:.4f}")

# --- Monotone constraints ---
xgb_mono = XGBRegressor(
    n_estimators=100, max_depth=4, learning_rate=0.1,
    tree_method="hist", random_state=42,
    monotone_constraints=(1,0,0,0,0,0,0,0,0,0),  # feature 0 forced increasing
    eval_metric="rmse", early_stopping_rounds=10,
)
xgb_mono.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
mono_rmse = np.sqrt(mean_squared_error(y_test, xgb_mono.predict(X_test)))
print(f"XGBRegressor (monotone)  best_iter={xgb_mono.best_iteration}  test_RMSE={mono_rmse:.4f}")