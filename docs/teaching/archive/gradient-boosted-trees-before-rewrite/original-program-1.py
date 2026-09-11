import numpy as np

class DecisionStump:
    """Shallow regression tree for use as a weak learner."""
    def __init__(self, max_depth=2):
        self.max_depth = max_depth
        self.tree = None

    def _best_split(self, X, r):
        m, n = X.shape
        best_mse, best_feat, best_thr = float('inf'), 0, 0
        best_left, best_right = None, None
        for feat in range(n):
            for thr in np.unique(X[:, feat]):
                left  = r[X[:, feat] <= thr]
                right = r[X[:, feat] >  thr]
                if len(left) == 0 or len(right) == 0:
                    continue
                mse = (np.var(left)*len(left) + np.var(right)*len(right)) / m
                if mse < best_mse:
                    best_mse = mse
                    best_feat, best_thr = feat, thr
                    best_left  = X[:, feat] <= thr
                    best_right = X[:, feat] >  thr
        return best_feat, best_thr, best_left, best_right

    def _build(self, X, r, depth):
        if depth == 0 or len(r) <= 1 or np.var(r) < 1e-8:
            return {'leaf': True, 'value': np.mean(r)}
        feat, thr, lm, rm = self._best_split(X, r)
        return {'leaf': False, 'feat': feat, 'thr': thr,
                'left':  self._build(X[lm], r[lm], depth-1),
                'right': self._build(X[rm], r[rm], depth-1)}

    def fit(self, X, r):
        self.tree = self._build(X, r, self.max_depth); return self

    def _pred1(self, node, x):
        if node['leaf']: return node['value']
        return self._pred1(node['left']  if x[node['feat']] <= node['thr']
                           else node['right'], x)

    def predict(self, X):
        return np.array([self._pred1(self.tree, x) for x in X])


class GradientBoostingScratch:
    def __init__(self, n_estimators=5, learning_rate=0.3, max_depth=2):
        self.n, self.lr, self.md = n_estimators, learning_rate, max_depth
        self.F0, self.trees = None, []

    def fit(self, X, y):
        self.F0 = np.mean(y)
        F = np.full(len(y), self.F0)
        print(f"{'Round':>5}  {'MSE':>9}  {'Mean pseudo-residual':>22}")
        print("-" * 44)
        for m in range(self.n):
            pseudo_resid = y - F          # gradient of MSE = -(y - F)
            mse = np.mean((y - F) ** 2)
            print(f"{m:>5}  {mse:>9.4f}  {np.mean(pseudo_resid):>22.6f}")
            tree = DecisionStump(max_depth=self.md).fit(X, pseudo_resid)
            F += self.lr * tree.predict(X)
            self.trees.append(tree)
        print(f"{'done':>5}  {np.mean((y - F)**2):>9.4f}")
        return self

    def predict(self, X):
        F = np.full(X.shape[0], self.F0)
        for tree in self.trees:
            F += self.lr * tree.predict(X)
        return F


np.random.seed(42)
X = np.random.randn(100, 2)
y = 3*X[:, 0] - 2*X[:, 1] + np.random.randn(100)*0.5

print("=== Gradient Boosting from Scratch (MSE loss) ===")
gb = GradientBoostingScratch(n_estimators=5, learning_rate=0.3, max_depth=2)
gb.fit(X, y)
# Predictions on 3 test points
X_test = np.array([[1.0, -1.0], [0.0, 0.0], [-1.0, 1.0]])
print("\nPredictions:", np.round(gb.predict(X_test), 3))
print("True values: ", np.round(3*X_test[:, 0] - 2*X_test[:, 1], 3))