"""The lesson's Beta(10,4) posterior, using maintained NUTS and diagnostics.
Install: python -m pip install pymc==6.3.2 arviz==1.3.0 scipy==1.18.1
Run: python pymc_posterior.py
Four sequential CPU chains keep the example portable on Windows.
"""
import numpy as np
import pymc as pm
import arviz_stats as azs
from scipy.stats import beta


def main():
    with pm.Model():
        theta = pm.Beta("theta", alpha=2, beta=2)
        pm.Binomial("observed_successes", n=10, p=theta, observed=8)
        trace = pm.sample(draws=2000, tune=1000, chains=4, cores=1,
                          random_seed=214, nuts_sampler="pymc",
                          nuts={"target_accept": .9}, progressbar=False,
                          compute_convergence_checks=True, return_inferencedata=True)
    draws = np.asarray(trace.posterior["theta"])
    assert draws.shape == (4, 2000) and np.isfinite(draws).all()
    assert np.all((draws > 0) & (draws < 1))
    # Do not flatten before diagnostics: chain identity matters.
    rhat = float(azs.rhat(draws, method="rank"))
    bulk = float(azs.ess(draws, method="bulk"))
    tail = float(azs.ess(draws, method="tail", prob=(.05, .95)))
    divergences = int(np.asarray(trace.sample_stats["diverging"]).sum())
    print(f"4 chains x 2000 retained; 1000 tuning per chain; divergences {divergences}")
    print(f"rank R-hat {rhat:.4f}; bulk ESS {bulk:.1f}; tail ESS {tail:.1f}")
    assert rhat < 1.01 and bulk > 400 and tail > 400 and divergences == 0
    for name, values, exact in (("mean", draws, 10 / 14),
                                ("P(theta > .70)", (draws > .70).astype(float), beta.sf(.70, 10, 4))):
        estimate = values.mean()
        mcse = float(azs.mcse(values, method="mean"))
        assert np.isfinite(mcse) and mcse > 0
        # A finite-run diagnostic tolerance, not a deterministic guarantee of
        # correctness or a confidence statement about all future runs.
        assert abs(estimate - exact) < 5 * mcse
        print(f"{name}: estimate {estimate:.6f}; exact {exact:.6f}; MCSE {mcse:.6f}; error/MCSE {abs(estimate-exact)/mcse:.2f}")
    # Changed-constraint reference for the learner's second run: 1 success,
    # 5 failures with the same Beta(2,2) prior gives Beta(3,7).
    print(f"Changed data (n=6, successes=1): exact mean {3/10:.6f}; exact tail {beta.sf(.70, 3, 7):.9f}")
    print("Support, finite diagnostics, precision and exact-target checks: passed")


if __name__ == "__main__":
    main()
