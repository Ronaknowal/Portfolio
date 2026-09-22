"""Mean-field ADVI against an exact correlated Gaussian target.
Install: python -m pip install pymc==6.3.2 numpy==2.3.5 scipy==1.18.1
Run: python pymc_variational.py
Normalized target with no observations: isolates inference error from modeling.
The fixed optimization budget is explicit; it is not a convergence certificate.
"""
import numpy as np
import pymc as pm
from scipy.linalg import cho_factor, cho_solve


def gaussian_kl(mean, variance, target_mean, covariance):
    factor = cho_factor(covariance, lower=True)
    delta = mean - target_mean
    precision_diagonal = np.diag(cho_solve(factor, np.eye(len(mean))))
    logdet = 2 * np.log(np.diag(factor[0])).sum()
    return .5 * (precision_diagonal @ variance + delta @ cho_solve(factor, delta)
                 - len(mean) + logdet - np.log(variance).sum())


def fit_case(correlation):
    target_mean = np.array([1., -1.])
    covariance = np.array([[1., correlation], [correlation, 1.]])
    with pm.Model():
        pm.MvNormal("theta", mu=target_mean, cov=covariance, shape=2)
        approximation = pm.fit(n=15000, method="advi", random_seed=51,
                               obj_n_mc=20, obj_optimizer=pm.adam(learning_rate=.01),
                               progressbar=False)
    mean = np.asarray(approximation.mean.eval())
    variance = np.asarray(approximation.std.eval())**2
    assert np.isfinite(mean).all() and np.all(np.isfinite(variance) & (variance > 0))
    optimal_variance = np.full(2, 1 - correlation**2)
    family_gap = gaussian_kl(target_mean, optimal_variance, target_mean, covariance)
    achieved_kl = gaussian_kl(mean, variance, target_mean, covariance)
    optimization_gap = achieved_kl - family_gap
    assert optimization_gap >= -1e-10
    assert optimization_gap < .03  # declared finite-budget tolerance, not fitted after the run
    assert np.max(np.abs(mean - target_mean)) < .15
    assert np.max(np.abs(variance - optimal_variance)) < .15
    print(f"rho={correlation:.1f}; fitted mean {np.round(mean, 4).tolist()}; fitted variances {np.round(variance, 4).tolist()}")
    print(f"  exact mean-field variances {np.round(optimal_variance, 4).tolist()}; true marginal variances [1, 1]")
    print(f"  achieved KL {achieved_kl:.6f} = family gap {family_gap:.6f} + optimization gap {optimization_gap:.6f}")


def main():
    # Identity target supplies an exact zero-KL edge for the analytic oracle.
    assert abs(gaussian_kl(np.zeros(2), np.ones(2), np.zeros(2), np.eye(2))) < 1e-12
    for correlation in (.8, 0.):
        fit_case(correlation)
    print("Analytic KL, fitted-parameter and changed-correlation checks: passed; no q sampling error in these comparisons")


if __name__ == "__main__":
    main()
