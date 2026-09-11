import numpy as np
import lightgbm as lgb
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(42)
X, y = make_regression(n_samples=1000, n_features=10, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_data = lgb.Dataset(X_train, label=y_train)
test_data  = lgb.Dataset(X_test,  label=y_test, reference=train_data)

params = {
    "objective":        "regression",
    "metric":           "rmse",
    "num_leaves":       31,         # controls leaf-wise max leaves per tree
    "learning_rate":    0.1,
    "feature_fraction": 0.8,        # colsample equivalent
    "bagging_fraction": 0.8,        # subsample equivalent
    "bagging_freq":     5,
    "lambda_l2":        1.0,
    # "device": "gpu",             # uncomment for GPU
    "verbose":          -1,
    "seed":             42,
}

callbacks = [
    lgb.early_stopping(stopping_rounds=10, verbose=False),
    lgb.log_evaluation(period=-1),   # suppress per-round output
]

model_lgb = lgb.train(
    params, train_data, num_boost_round=100,
    valid_sets=[train_data, test_data],
    valid_names=["train", "test"],
    callbacks=callbacks,
)
preds = model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration)
rmse  = np.sqrt(mean_squared_error(y_test, preds))
print(f"LightGBM  best_iter={model_lgb.best_iteration}  test_RMSE={rmse:.4f}")
print(f"  num_trees={model_lgb.num_trees()}")
print(f"  top-3 features by split: {sorted(enumerate(model_lgb.feature_importance()), key=lambda x: -x[1])[:3]}")