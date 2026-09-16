import numpy as np

X = np.array([0., 1., 3., 7.])[:, None]
n, perplexity = len(X), 2.0
conditional = np.zeros((n, n))
for i in range(n):
    others = np.arange(n) != i
    distances = np.sum((X[others] - X[i])**2, axis=1)
    lower, upper, beta = 0.0, np.inf, 1.0
    for _ in range(80):
        weights = np.exp(-beta * (distances - distances.min()))
        p = weights / weights.sum()
        positive = p > 0
        entropy = -np.sum(p[positive] * np.log(p[positive]))
        if abs(entropy - np.log(perplexity)) < 1e-10:
            break
        if entropy > np.log(perplexity):
            lower = beta
            beta = 2 * beta if np.isinf(upper) else (lower + upper) / 2
        else:
            upper = beta
            beta = (lower + upper) / 2
    conditional[i, others] = p
P = (conditional + conditional.T) / (2 * n)

def cost_and_gradient(Y):
    offsets = Y[:, None, :] - Y[None, :, :]
    kernel = 1 / (1 + np.sum(offsets**2, axis=2))
    np.fill_diagonal(kernel, 0)
    Q = kernel / kernel.sum()
    mask = P > 0
    cost = np.sum(P[mask] * np.log(P[mask] / Q[mask]))
    gradient = 4 * np.sum(((P - Q) * kernel)[:, :, None] * offsets, axis=1)
    return cost, gradient

Y = np.array([-1.5, -0.5, 0.5, 1.5])[:, None]
print(f"initial KL={cost_and_gradient(Y)[0]:.6f}")
for _ in range(200):
    _, gradient = cost_and_gradient(Y)
    Y -= 0.5 * gradient
    Y -= Y.mean(axis=0)
print(f"final KL={cost_and_gradient(Y)[0]:.6f}")
print(np.round(Y.ravel(), 6))
