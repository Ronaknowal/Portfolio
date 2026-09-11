import numpy as np
import catboost
from catboost import CatBoostRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=1000, n_features=10, n_informative=10,
                       noise=20, random_state=42)
X_development, X_test, y_development, y_test = train_test_split(X, y, test_size=.15, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_development, y_development, test_size=150, random_state=43)
model = CatBoostRegressor(iterations=200, depth=4, learning_rate=.1,
                          l2_leaf_reg=1., loss_function="RMSE", eval_metric="RMSE",
                          boosting_type="Ordered", early_stopping_rounds=12,
                          random_seed=42, thread_count=1, verbose=False,
                          allow_writing_files=False)
model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)
prediction = np.asarray(model.predict(X_test)).reshape(-1)
assert prediction.shape == y_test.shape
assert model.tree_count_ == model.get_best_iteration() + 1
rmse = lambda target, fitted: np.sqrt(np.mean((target - fitted) ** 2))
print("CatBoost", catboost.__version__, "; CPU; Ordered; one thread; rows=700/150/150")
print(f"best zero-based round={model.get_best_iteration()}; retained trees={model.tree_count_}")
print(f"baseline test RMSE={rmse(y_test,np.full(len(y_test),np.mean(y_train))):.6f}")
print(f"selected validation RMSE={model.get_best_score()['validation']['RMSE']:.6f}")
print(f"locked-model test RMSE={rmse(y_test,prediction):.6f}")
print("This explicit Ordered run does not establish which library wins on other data.")
