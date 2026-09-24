"""Bounded educational pair-coordinate solver; not the production LIBSVM algorithm."""
import numpy as np


class PairSVM:
    def __init__(self, c=1., kernel='rbf', gamma=.5, tolerance=1e-7, max_steps=20000):
        self.c, self.kernel, self.gamma = c, kernel, gamma
        self.tolerance, self.max_steps = tolerance, max_steps

    def matrix(self, left, right):
        if self.kernel == 'linear':
            return left @ right.T
        squared = np.sum((left[:, None, :] - right[None, :, :]) ** 2, axis=2)
        return np.exp(-self.gamma * squared)

    def certificate(self, gram, alpha, y):
        raw = gram @ (alpha * y)
        # For fixed w, the hinge sum has slope -n_positive at b=-infinity.
        # Every breakpoint y_i-raw_i increases that slope by one.
        breaks = np.sort(y - raw)
        positive = int(np.sum(y == 1))
        bias = (breaks[positive - 1] + breaks[positive]) / 2
        norm_squared = float((alpha * y) @ raw)
        primal = .5 * norm_squared + self.c * np.maximum(0., 1 - y * (raw + bias)).sum()
        dual = alpha.sum() - .5 * norm_squared
        return bias, float(primal), float(dual)

    def fit(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if x.ndim != 2 or not (2 <= len(x) <= 100) or not (1 <= x.shape[1] <= 20):
            raise ValueError('Use 2..100 rows and 1..20 numeric features.')
        if y.shape != (len(x),) or set(y.tolist()) != {-1., 1.}:
            raise ValueError('Supply both -1 and +1 labels, one per row.')
        if not np.isfinite(x).all() or np.max(np.abs(x)) > 1000:
            raise ValueError('Finite feature magnitudes must be at most1000.')
        if self.kernel not in ['linear', 'rbf'] or not (.001 <= self.c <= 100):
            raise ValueError('Choose linear/rbf and C in [.001,100].')
        if not (1e-4 <= self.gamma <= 10) or not (1e-10 <= self.tolerance <= .01):
            raise ValueError('Gamma or gap tolerance outside teaching range.')
        if not isinstance(self.max_steps, int) or not (1 <= self.max_steps <= 50000):
            raise ValueError('Use 1..50000 update steps.')
        self.x, self.y = x.copy(), y.copy()
        gram = self.matrix(x, x)
        alpha = np.zeros(len(y))  # Refit starts from a fresh feasible vector.
        i, j = np.triu_indices(len(y), 1)
        sign = y[i] * y[j]
        squared = np.sum((x[i] - x[j]) ** 2, axis=1)
        curvature = squared if self.kernel == 'linear' else -2 * np.expm1(-self.gamma * squared)
        self.status = 'step limit'
        for step in range(self.max_steps + 1):
            bias, primal, dual = self.certificate(gram, alpha, y)
            if primal - dual <= self.tolerance * max(1., abs(primal)):
                self.status = 'gap passed'
                break
            if step == self.max_steps:
                break
            gradient = 1 - y * (gram @ (alpha * y))
            slope = gradient[j] - sign * gradient[i]
            lower = np.maximum(-alpha[j], np.where(sign == 1, alpha[i] - self.c, -alpha[i]))
            upper = np.minimum(self.c - alpha[j], np.where(sign == 1, alpha[i], self.c - alpha[i]))
            candidate = np.divide(slope, curvature, out=np.zeros_like(slope), where=curvature > 0)
            candidate = np.clip(candidate, lower, upper)
            endpoint = np.where(slope * upper > slope * lower, upper, lower)
            candidate = np.where(curvature == 0, endpoint, candidate)
            gain = slope * candidate - .5 * curvature * candidate ** 2
            chosen = int(np.argmax(gain))
            if gain[chosen] <= 1e-15:
                self.status = 'stalled before gap criterion'
                break
            delta = candidate[chosen]
            left, right = int(i[chosen]), int(j[chosen])
            # Compute both values before committing; clipping removes endpoint roundoff only.
            new_left = np.clip(alpha[left] - sign[chosen] * delta, 0., self.c)
            new_right = np.clip(alpha[right] + delta, 0., self.c)
            alpha[left], alpha[right] = new_left, new_right
            if abs(alpha @ y) > 1e-9:
                raise ArithmeticError('The update lost label balance.')
        self.alpha, self.bias = alpha, bias
        self.primal, self.dual, self.steps = primal, dual, step
        return self

    def decision_function(self, x):
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.x.shape[1] or not np.isfinite(x).all() or np.max(np.abs(x), initial=0) > 1000:
            raise ValueError('Prediction rows must match training shape/range.')
        return self.matrix(x, self.x) @ (self.alpha * self.y) + self.bias

    def predict(self, x):
        # This lesson chooses +1 on an exact zero score; libraries may choose a different tie.
        return np.where(self.decision_function(x) >= 0, 1., -1.)


if __name__ == '__main__':
    np.random.seed(42)
    blobs = np.vstack([np.random.randn(20, 2) + [2., 2.], np.random.randn(20, 2) - [2., 2.]])
    blob_labels = np.r_[np.ones(20), -np.ones(20)]
    np.random.seed(7)
    xor = np.random.randn(40, 2)
    xor_labels = np.where(np.prod(xor, axis=1) > 0, 1., -1.)
    for name, x, y, gamma in [('blobs', blobs, blob_labels, .5), ('XOR', xor, xor_labels, 1.)]:
        model = PairSVM(gamma=gamma).fit(x, y)
        print(f'{name}: {model.status}; train accuracy={np.mean(model.predict(x)==y):.3f}')
        print(f'primal={model.primal:.6f} dual={model.dual:.6f} gap={model.primal-model.dual:.6f}')
        print('positive coefficients:', int(np.sum(model.alpha > 1e-8)))
    duplicate = PairSVM(kernel='linear').fit([[0.], [0.]], [-1., 1.])
    print('opposite duplicate:', duplicate.status, 'alpha=', duplicate.alpha.tolist())
