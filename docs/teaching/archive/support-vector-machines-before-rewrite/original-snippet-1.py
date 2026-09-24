import numpy as np

class SimpleSMO:
    """Simplified Sequential Minimal Optimization for binary SVM.

    Platt 1998 (MSR-TR-98-14) — random working set selection for clarity.
    Labels must be +1 / -1.
    """

    def __init__(self, C=1.0, kernel="rbf", gamma=0.5,
                 max_passes=100, tol=1e-3):
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.max_passes = max_passes
        self.tol = tol

    def _k(self, x1, x2):
        if self.kernel == "rbf":
            diff = x1 - x2
            return np.exp(-self.gamma * np.dot(diff, diff))
        elif self.kernel == "linear":
            return np.dot(x1, x2)
        raise ValueError(f"Unknown kernel: ${self.kernel}")

    def fit(self, X, y):
        n = len(y)
        self.alpha = np.zeros(n)
        b = 0.0

        # Precompute full kernel matrix: O(n^2) space and time
        K = np.array([[self._k(X[i], X[j])
                       for j in range(n)] for i in range(n)])

        passes = 0
        while passes < self.max_passes:
            num_changed = 0
            for i in range(n):
                # Prediction error for point i
                Ei = (self.alpha * y) @ K[i] + b - y[i]

                # Check KKT violation
                violates = (
                    (y[i] * Ei < -self.tol and self.alpha[i] < self.C) or
                    (y[i] * Ei >  self.tol and self.alpha[i] > 0)
                )
                if not violates:
                    continue

                # Pick j != i uniformly at random
                j = np.random.choice([k for k in range(n) if k != i])
                Ej = (self.alpha * y) @ K[j] + b - y[j]

                ai_old, aj_old = self.alpha[i], self.alpha[j]

                # Compute clipping bounds
                if y[i] != y[j]:
                    L = max(0.0, aj_old - ai_old)
                    H = min(self.C, self.C + aj_old - ai_old)
                else:
                    L = max(0.0, ai_old + aj_old - self.C)
                    H = min(self.C, ai_old + aj_old)
                if L >= H:
                    continue

                # Second-order step size
                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue

                # Update alpha_j, clip, then update alpha_i
                self.alpha[j] -= y[j] * (Ei - Ej) / eta
                self.alpha[j] = np.clip(self.alpha[j], L, H)
                if abs(self.alpha[j] - aj_old) < 1e-5:
                    continue
                self.alpha[i] += y[i] * y[j] * (aj_old - self.alpha[j])

                # Update bias
                b1 = (b - Ei
                      - y[i] * (self.alpha[i] - ai_old) * K[i, i]
                      - y[j] * (self.alpha[j] - aj_old) * K[i, j])
                b2 = (b - Ej
                      - y[i] * (self.alpha[i] - ai_old) * K[i, j]
                      - y[j] * (self.alpha[j] - aj_old) * K[j, j])
                if 0 < self.alpha[i] < self.C:
                    b = b1
                elif 0 < self.alpha[j] < self.C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0

                num_changed += 1

            passes = 0 if num_changed > 0 else passes + 1

        self.b = b
        self.X_train = X
        self.y_train = y
        self.n_support = int((self.alpha > 1e-5).sum())

    def decision_function(self, X):
        return np.array([
            sum(self.alpha[i] * self.y_train[i] * self._k(self.X_train[i], x)
                for i in range(len(self.y_train))) + self.b
            for x in X
        ])

    def predict(self, X):
        return np.sign(self.decision_function(X))