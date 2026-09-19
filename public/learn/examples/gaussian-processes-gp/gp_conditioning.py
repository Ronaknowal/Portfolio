import numpy as np
from scipy.linalg import solve_triangular

def rbf(x, z, length=1.0):
    distances = np.asarray(x)[:, None] - np.asarray(z)[None, :]
    return np.exp(-0.5 * (distances / length) ** 2)

def predict(x, y, targets, noise_variance=0.25):
    covariance = rbf(x, x) + noise_variance * np.eye(len(x))
    lower = np.linalg.cholesky(covariance)
    intermediate = solve_triangular(lower, y, lower=True)
    weights = solve_triangular(lower.T, intermediate, lower=False)
    cross = rbf(x, targets)
    solved_cross = solve_triangular(lower, cross, lower=True)
    mean = cross.T @ weights
    latent_covariance = rbf(targets, targets) - solved_cross.T @ solved_cross
    log_marginal = (
        -0.5 * y @ weights
        - np.log(np.diag(lower)).sum()
        - 0.5 * len(x) * np.log(2 * np.pi)
    )
    return mean, latent_covariance, log_marginal

x = np.array([0.0, 2.0])
y = np.array([1.0, -1.0])
targets = np.array([0.0, 1.0, 2.0, 4.0])
mean, covariance, log_marginal = predict(x, y, targets)
print(np.round(mean, 6))
print(np.round(np.diag(covariance), 6))
print(round(float(log_marginal), 6))
changed = predict(x, np.array([3.0, -2.0]), targets)
print(np.allclose(covariance, changed[1]))
