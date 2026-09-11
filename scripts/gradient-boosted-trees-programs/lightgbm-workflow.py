import numpy as np
import lightgbm as lgb
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=1000, n_features=10, n_informative=10,
                       noise=20, random_state=42)
X_development, X_test, y_development, y_test = train_test_split(X, y, test_size=.15, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_development, y_development, test_size=150, random_state=43)
train = lgb.Dataset(X_train, label=y_train)
valid = lgb.Dataset(X_valid, label=y_valid, reference=train)
params = dict(objective="regression", metric="rmse", num_leaves=15,
              learning_rate=.1, feature_fraction=.8, bagging_fraction=.8,
              bagging_freq=1, lambda_l2=1., seed=42, verbosity=-1,
              num_threads=1, deterministic=True, force_col_wise=True)
history = {}
model = lgb.train(params, train, num_boost_round=200,
                  valid_sets=[valid], valid_names=["valid"],
                  callbacks=[lgb.early_stopping(12, verbose=False),
                             lgb.record_evaluation(history), lgb.log_evaluation(0)])
# LightGBM best_iteration is a count, already suitable for num_iteration.
prediction = model.predict(X_test, num_iteration=model.best_iteration)
assert prediction.shape == y_test.shape
rmse = lambda target, fitted: np.sqrt(np.mean((target - fitted) ** 2))
print("LightGBM", lgb.__version__, "; CPU; one thread; rows=700/150/150")
print(f"best iteration count={model.best_iteration}; stored trees={model.num_trees()}")
print(f"baseline test RMSE={rmse(y_test,np.full(len(y_test),np.mean(y_train))):.6f}")
print(f"selected validation RMSE={history['valid']['rmse'][model.best_iteration-1]:.6f}")
print(f"locked-model test RMSE={rmse(y_test,prediction):.6f}")
# Split count and gain ask different questions; neither is a causal effect.
split_count = model.feature_importance(importance_type="split")
gain = model.feature_importance(importance_type="gain")
print("feature 0:", f"split count={split_count[0]}; summed gain={gain[0]:.3f}")
