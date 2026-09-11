import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb

# Both libraries can receive declared categorical features. Codes alone are not enough.
dtype = pd.CategoricalDtype(categories=["A", "B", "C"])
training = pd.DataFrame({"group": pd.Series(["A","B","C"] * 12, dtype=dtype),
                         "load": np.tile([0., 1., 2.], 12)})
target = np.tile([0., 4., 1.], 12)
future = pd.DataFrame({"group": pd.Series(["C","A","B"], dtype=dtype), "load": [2.,0.,1.]})
xmodel = xgb.XGBRegressor(n_estimators=12, max_depth=2, learning_rate=.3,
                          tree_method="hist", enable_categorical=True,
                          n_jobs=1, random_state=42)
xmodel.fit(training, target)
lmodel = lgb.LGBMRegressor(n_estimators=12, num_leaves=3, learning_rate=.3,
                           min_child_samples=1, min_data_in_bin=1, verbosity=-1,
                           n_jobs=1, deterministic=True, force_col_wise=True)
lmodel.fit(training, target, categorical_feature=["group"])
print("declared category order:", list(dtype.categories))
print("XGBoost finite predictions:", bool(np.isfinite(xmodel.predict(future)).all()))
print("LightGBM finite predictions:", bool(np.isfinite(lmodel.predict(future)).all()))
print("category=C, A, B predictions:")
print("XGBoost", np.round(xmodel.predict(future), 4))
print("LightGBM", np.round(lmodel.predict(future), 4))
print("Keep the training schema; this toy does not validate arbitrary unseen-category pipelines.")
