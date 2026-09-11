from sklearn.svm import SVC, LinearSVC
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import numpy as np

np.random.seed(42)
X, y = make_classification(n_samples=500, n_features=20,
                            n_informative=10, n_redundant=5,
                            random_state=42)

# CRITICAL: always scale before SVC with RBF kernel
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

# Standard RBF SVC
svc = SVC(kernel="rbf",
          C=1.0,
          gamma="scale",          # gamma = 1/(n_features * X.var())
          probability=True,       # Platt scaling; adds ~5x train cost
          class_weight=None,      # set "balanced" for imbalanced data
          random_state=42)
svc.fit(X_sc, y)

print(f"SVC(kernel=rbf) train accuracy : {svc.score(X_sc, y):.3f}")
# SVC(kernel=rbf) train accuracy : 0.980
print(f"n_support_vectors              : {svc.support_.shape[0]}")
# n_support_vectors              : 253
print(f"predict_proba (first 3)        : {svc.predict_proba(X_sc[:3]).round(3).tolist()}")
# predict_proba (first 3)        : [[0.023, 0.977], [0.998, 0.002], [0.023, 0.977]]
print(f"decision_function (first 3)    : {svc.decision_function(X_sc[:3]).round(3).tolist()}")
# decision_function (first 3)    : [1.0, -1.374, 1.0]