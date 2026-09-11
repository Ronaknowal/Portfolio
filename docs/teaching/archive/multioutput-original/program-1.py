def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def logistic_train(X, y, lr=0.05, n_iter=300):
    """
    Mini logistic regression via gradient descent.
    Gradient of log-loss = (1/n) * X^T (sigma(Xw) - y)
    """
    n, d = X.shape
    w = np.zeros(d)
    for _ in range(n_iter):
        grad = (1 / n) * (X.T @ (sigmoid(X @ w) - y))
        w -= lr * grad
    return w