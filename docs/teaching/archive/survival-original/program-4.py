from lifelines import WeibullAFTFitter

# Accelerated Failure Time model: log(T) = beta^T x + sigma * epsilon
# Parametric — assumes Weibull distribution for baseline
aft = WeibullAFTFitter()
aft.fit(rossi, duration_col="week", event_col="arrest")
print(f"AFT concordance: {aft.concordance_index_:.4f}")
# AFT concordance: 0.6405

# scikit-survival: sklearn-compatible API, adds RSF and GBSA
# pip install scikit-survival
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
import pandas as pd

# Build structured array required by sksurv
y = Surv.from_dataframe("arrest", "week", rossi)
X_rsf = rossi.drop(columns=["week", "arrest"])

rsf = RandomSurvivalForest(n_estimators=100, min_samples_leaf=10, random_state=42)
rsf.fit(X_rsf, y)
print(f"RSF concordance: {rsf.score(X_rsf, y):.4f}")
# RSF concordance (training): 0.7089
# Note: evaluate on held-out set in practice — training concordance is optimistic