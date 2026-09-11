from sklearn.svm import SVR
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

X_r, y_r = make_regression(n_samples=200, n_features=5,
                             noise=5, random_state=42)
X_r_sc = StandardScaler().fit_transform(X_r)

# epsilon-insensitive tube: no penalty if |prediction - target| < epsilon
svr = SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")
svr.fit(X_r_sc, y_r)
print(f"SVR R^2 on train: {svr.score(X_r_sc, y_r):.3f}")
# SVR R^2 on train: 0.775
print(f"SVR n_support_vectors: {len(svr.support_)}")
# SVR n_support_vectors: 199