import numpy as np
from catboost import CatBoostClassifier, Pool

rng = np.random.default_rng(42)
n = 600
category = rng.choice(["A", "B", "C"], n)
number = rng.normal(size=n)
target = (number + .8 * (category == "A") - .4 * (category == "C") > .2).astype(int)
data = [[str(category[i]), float(number[i])] for i in range(n)]
# Independent synthetic rows: a fixed row split is appropriate in this fixture.
train = Pool(data[:400], label=target[:400], cat_features=[0])
valid = Pool(data[400:500], label=target[400:500], cat_features=[0])
test = Pool(data[500:], cat_features=[0])
model = CatBoostClassifier(iterations=150, depth=3, learning_rate=.1,
                           loss_function="Logloss", eval_metric="Logloss",
                           boosting_type="Ordered", one_hot_max_size=1,
                           early_stopping_rounds=10, random_seed=42,
                           thread_count=1, verbose=False, allow_writing_files=False)
model.fit(train, eval_set=valid, use_best_model=True)
raw = np.asarray(model.predict(test))
prediction = raw.reshape(-1).astype(int)
probability = np.asarray(model.predict_proba(test))[:, 1]
truth = target[500:]
assert prediction.shape == probability.shape == truth.shape == (100,)
loss = -np.mean(truth * np.log(np.clip(probability,1e-15,1)) + (1-truth)*np.log(np.clip(1-probability,1e-15,1)))
print(f"raw class shape={raw.shape}; aligned shape={prediction.shape}")
print(f"best zero-based round={model.get_best_iteration()}; retained trees={model.tree_count_}")
print(f"locked test accuracy={np.mean(prediction==truth):.6f}; log-loss={loss:.6f}")
unseen = Pool([["NEW", 0.0]], cat_features=[0])
print("unseen-category probability is finite:", bool(np.isfinite(model.predict_proba(unseen)).all()))
# A broadcasting mistake can silently score every prediction against every label.
bad_comparison = prediction.reshape(-1, 1) == truth
print(f"wrong comparison shape={bad_comparison.shape}; wrong accuracy={bad_comparison.mean():.6f}")
