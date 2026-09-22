"""From identifiable cell contrasts to fitted GLMs and the AIPW score.

Run: python causal-estimation-bridge.py
Dependencies: numpy==2.3.5, statsmodels==0.15.0 (with its normal dependencies).
This constructed table illustrates estimation under a supplied valid backdoor
set; a fitted regression cannot establish the causal assumptions. The saturated
four-cell model is small enough to check exactly. Flexible nuisance models need
appropriate sample splitting/cross-fitting, not this in-sample demonstration.
"""
import numpy as np
import statsmodels.api as sm


def expand_cells(sizes, successes):
    """Order: (Z,T)=(0,0),(0,1),(1,0),(1,1); binary outcomes."""
    sizes, successes = np.asarray(sizes), np.asarray(successes)
    if (sizes.shape != (4,) or successes.shape != (4,)
            or not np.issubdtype(sizes.dtype, np.integer)
            or not np.issubdtype(successes.dtype, np.integer)
            or np.any(sizes <= 0) or np.any(successes <= 0)
            or np.any(successes >= sizes)):
        raise ValueError("Use positive integer cell sizes with both outcomes in every cell")
    z = np.repeat([0., 0., 1., 1.], sizes)
    treatment = np.repeat([0., 1., 0., 1.], sizes)
    outcome = np.concatenate([np.r_[np.ones(y), np.zeros(n-y)]
                              for n, y in zip(sizes, successes)])
    return z, treatment, outcome


def scores(treatment, outcome, propensity, m0, m1):
    """Empirical target population; ordinary ATE weights, without clipping."""
    if np.any(propensity <= 0) or np.any(propensity >= 1):
        raise ValueError("Overlap is required; clipping would change the estimator")
    regression = m1 - m0
    weighted = treatment * outcome / propensity - (1-treatment) * outcome / (1-propensity)
    augmented = regression + treatment * (outcome-m1) / propensity - (1-treatment) * (outcome-m0) / (1-propensity)
    return np.array([regression.mean(), weighted.mean(), augmented.mean()])


def design(z, treatment):
    return np.column_stack((np.ones(len(z)), treatment, z, treatment*z))


def estimate(sizes, successes):
    z, treatment, outcome = expand_cells(sizes, successes)
    cell = (2*z + treatment).astype(int)
    counts = np.bincount(cell, minlength=4).reshape(2, 2)
    means = (np.bincount(cell, weights=outcome, minlength=4) / counts.ravel()).reshape(2, 2)
    strata = z.astype(int)
    propensity = (counts[:, 1] / counts.sum(axis=1))[strata]
    manual = scores(treatment, outcome, propensity, means[strata, 0], means[strata, 1])

    propensity_fit = sm.GLM(treatment, sm.add_constant(z), family=sm.families.Binomial()).fit(tol=1e-12)
    outcome_fit = sm.GLM(outcome, design(z, treatment), family=sm.families.Binomial()).fit(tol=1e-12)
    if not propensity_fit.converged or not outcome_fit.converged:
        raise RuntimeError("The GLM iteration did not converge")
    fitted_e = propensity_fit.predict(sm.add_constant(z))
    fitted_m0 = outcome_fit.predict(design(z, np.zeros_like(z)))
    fitted_m1 = outcome_fit.predict(design(z, np.ones_like(z)))
    fitted = scores(treatment, outcome, fitted_e, fitted_m0, fitted_m1)
    np.testing.assert_allclose(fitted_e, propensity, atol=1e-10)
    np.testing.assert_allclose(fitted_m0, means[strata, 0], atol=1e-10)
    np.testing.assert_allclose(fitted_m1, means[strata, 1], atol=1e-10)
    return manual, fitted, float(outcome[treatment == 1].mean() - outcome[treatment == 0].mean())


def cell_bootstrap(sizes, successes, repetitions=2000, seed=41):
    """Conditional on four fixed cell sizes; not an unconditional cohort bootstrap."""
    sizes, successes = np.asarray(sizes), np.asarray(successes)
    means = successes / sizes
    z_weights = sizes.reshape(2, 2).sum(axis=1) / sizes.sum()
    draws = np.random.default_rng(seed).binomial(sizes, means, size=(repetitions, 4)) / sizes
    contrasts = (draws[:, [1, 3]] - draws[:, [0, 2]]) @ z_weights
    return np.quantile(contrasts, [.025, .975])


def main():
    sizes, successes = [80, 20, 20, 80], [8, 4, 6, 32]
    manual, fitted, raw = estimate(sizes, successes)
    np.testing.assert_allclose(manual, [.1, .1, .1], atol=1e-12)
    np.testing.assert_allclose(fitted, manual, atol=1e-10)
    print("unadjusted difference", round(raw, 6))
    print("manual g-computation/IPW/AIPW", np.round(manual, 6).tolist())
    print("fitted GLM g-computation/IPW/AIPW", np.round(fitted, 6).tolist())
    print("conditional cell-bootstrap 95% interval", np.round(cell_bootstrap(sizes, successes), 6).tolist())
    changed, library, _ = estimate([60, 30, 40, 70], [6, 9, 12, 35])
    np.testing.assert_allclose(changed, library, atol=1e-10)
    print("changed target-population estimates", np.round(changed, 6).tolist())


if __name__ == "__main__":
    main()
