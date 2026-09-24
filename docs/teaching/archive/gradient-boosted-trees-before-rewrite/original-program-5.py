import numpy as np
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(42)
X, y = make_regression(n_samples=1000, n_features=10, noise=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Regression ---
model_cat = CatBoostRegressor(
    iterations=100, depth=4, learning_rate=0.1,
    l2_leaf_reg=1.0, loss_function="RMSE", eval_metric="RMSE",
    early_stopping_rounds=10, random_seed=42, verbose=False,
)
model_cat.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)
preds = model_cat.predict(X_test)
rmse  = np.sqrt(mean_squared_error(y_test, preds))
print(f"CatBoost (regression)  best_iter={model_cat.best_iteration_}  test_RMSE={rmse:.4f}")
print(f"  feature importances (top 3): {sorted(enumerate(model_cat.get_feature_importance()), key=lambda x: -x[1])[:3]}")

# --- Classification with categorical features (ordered target statistics) ---
n = 500
cat1 = np.random.choice(['A', 'B', 'C'], n)
cat2 = np.random.choice(['X', 'Y'], n)
num  = np.random.randn(n, 3)
X_mixed = np.column_stack([cat1, cat2, num.astype(str)])
y_mixed = ((num[:, 0] + (cat1 == 'A').astype(float)) > 0).astype(int)

train_pool = Pool(data=X_mixed[:400], label=y_mixed[:400], cat_features=[0, 1])
test_pool  = Pool(data=X_mixed[400:], label=y_mixed[400:], cat_features=[0, 1])

clf = CatBoostClassifier(
    iterations=50, depth=4, learning_rate=0.1,
    loss_function='Logloss', eval_metric='Accuracy',
    random_seed=42, verbose=False,
)
clf.fit(train_pool, eval_set=test_pool, use_best_model=True)
acc = np.mean(clf.predict(test_pool) == y_mixed[400:])
print(f"\nCatBoost (classifier, cat_features=[0,1])  best_iter={clf.best_iteration_}  accuracy={acc:.4f}")