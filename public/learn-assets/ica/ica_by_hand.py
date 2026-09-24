import numpy as np

S = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
A = np.array([[2., 1.], [1., 2.]])
X = S @ A.T
mean = X.mean(axis=0)
centered = X - mean
values, vectors = np.linalg.eigh(centered.T @ centered / len(X))
K = (vectors / np.sqrt(values)).T
Z = centered @ K.T

def separate(Z, seed=12, max_iter=200, tol=1e-10):
    rng = np.random.default_rng(seed)
    W = np.zeros((Z.shape[1], Z.shape[1]))
    for j in range(len(W)):
        w = rng.normal(size=Z.shape[1])
        w -= W[:j].T @ (W[:j] @ w)
        w /= np.linalg.norm(w)
        for _ in range(max_iter):
            y = Z @ w
            r = Z.T @ (y**3) / len(Z) - (3*y*y).mean() * w
            r -= W[:j].T @ (W[:j] @ r)
            new = r / np.linalg.norm(r)
            change = 1 - abs(new @ w)
            w = new
            if change < tol:
                break
        else:
            raise RuntimeError("Iteration limit reached")
        W[j] = w
    return W

W = separate(Z)
estimated = Z @ W.T
B = W @ K
C = np.corrcoef(S.T, estimated.T)[:2, 2:]
reconstructed = estimated @ np.linalg.inv(B).T + mean
print(np.round(Z.T @ Z / len(Z), 6))
print(np.round(np.max(np.abs(C), axis=1), 6))
print(np.max(np.abs(reconstructed - X)) < 1e-10)
