"""The lesson's average-loss RKHS objective matched to KernelRidge.
Install: python -m pip install numpy==2.3.5 scipy==1.18.1 scikit-learn==1.9.1
Run: python kernel_ridge_library.py
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.kernel_ridge import KernelRidge


def kernel(left, right, gamma):
    return np.exp(-gamma * (left[:, None] - right[None, :])**2)


def main():
    x = np.array([0., .5, .5, 1.])
    y = np.array([0., .5, 1.5, 0.])
    query = np.array([.3, .5, .8])
    gamma = 4.
    gram = kernel(x, x, gamma)
    query_gram = kernel(query, x, gamma)
    for penalty in (.05, .1):
        shift = len(x) * penalty
        system = gram + shift * np.eye(len(x))
        coefficients = cho_solve(cho_factor(system, lower=True), y)
        # No intercept and no centering in either route. alpha is a shift, not
        # the average-loss lambda and not this lesson's coefficient vector.
        model = KernelRidge(alpha=shift, kernel="rbf", gamma=gamma).fit(x[:, None], y)
        precomputed = KernelRidge(alpha=shift, kernel="precomputed").fit(gram, y)
        own = query_gram @ coefficients
        assert np.allclose(model.dual_coef_, coefficients, atol=1e-12)
        assert np.allclose(model.predict(query[:, None]), own, atol=1e-12)
        assert np.allclose(precomputed.predict(query_gram), own, atol=1e-12)
        relative_residual = np.linalg.norm(system @ coefficients - y) / np.linalg.norm(y)
        assert relative_residual < 1e-12
        objective = np.mean((gram @ coefficients - y)**2) + penalty * coefficients @ gram @ coefficients
        tool_objective = np.mean((model.predict(x[:, None]) - y)**2) + penalty * model.dual_coef_ @ gram @ model.dual_coef_
        assert np.isclose(objective, tool_objective, atol=1e-12)
        assert np.isclose((gram @ coefficients)[1], (gram @ coefficients)[2])
        print(f"lambda={penalty:.2f}; library alpha={shift:.2f}; predictions {np.round(own, 6).tolist()}; objective {objective:.6f}; residual < 1e-12")
    assert np.linalg.matrix_rank(gram) == 3
    # A zero target checks linearity; duplicate inputs test positive shifting.
    zero = KernelRidge(alpha=.2, kernel="rbf", gamma=gamma).fit(x[:, None], np.zeros(4))
    assert np.array_equal(zero.predict(query[:, None]), np.zeros(3))
    print("Coefficient, prediction, objective, duplicate-input and zero-target checks: passed")


if __name__ == "__main__":
    main()
