"""Vectorized log-domain balancing and POT on identical positive marginals.

Run: python sinkhorn-library-bridge.py
Dependencies: numpy==2.3.5, scipy==1.18.1, POT==0.9.7.
Dense costs/plan require O(n*m) memory; K updates require O(K*n*m) arithmetic.
Finite moderate-scale costs, epsilon>0, strictly positive probability weights.
Zero-mass rows must be removed and reinserted by the caller. No epsilon scaling,
implicit plan, unbalanced mass, differentiation, or acceleration is hidden here.
"""
import numpy as np
from scipy.special import logsumexp, xlogy
import ot


def sinkhorn_log(a, b, cost, epsilon, tolerance=1e-10, max_steps=10000):
    a, b, cost = np.asarray(a, float), np.asarray(b, float), np.asarray(cost, float)
    if (a.ndim != 1 or b.ndim != 1 or not a.size or not b.size
            or cost.shape != (a.size, b.size) or not np.isfinite(cost).all()
            or not np.isfinite(a).all() or not np.isfinite(b).all()
            or np.any(a <= 0) or np.any(b <= 0)
            or abs(a.sum() - 1) > 1e-12 or abs(b.sum() - 1) > 1e-12
            or not np.isfinite(epsilon) or epsilon <= 0
            or tolerance <= 0 or max_steps < 1):
        raise ValueError("Require positive probability marginals, finite costs, and valid solver settings")
    log_kernel = -(cost - cost.min()) / epsilon
    if not np.isfinite(log_kernel).all():
        raise ValueError("Cost range divided by epsilon exceeds this floating-point contract")
    log_a, log_b = np.log(a), np.log(b)
    log_u, log_v = np.zeros_like(a), np.zeros_like(b)
    for iteration in range(1, max_steps + 1):
        log_u = log_a - logsumexp(log_kernel + log_v[None, :], axis=1)
        log_v = log_b - logsumexp(log_kernel + log_u[:, None], axis=0)
        gauge = log_v.mean()
        log_v -= gauge
        log_u += gauge
        plan = np.exp(log_kernel + log_u[:, None] + log_v[None, :])
        residual = max(np.max(np.abs(plan.sum(axis=1) - a)),
                       np.max(np.abs(plan.sum(axis=0) - b)))
        if residual <= tolerance:
            break
    return plan, {"converged": bool(residual <= tolerance),
                  "iterations": iteration, "residual": float(residual)}


def objectives(plan, cost, epsilon):
    linear = float(np.sum(plan * cost))
    entropy = float(np.sum(xlogy(plan, plan)))  # Defines 0*log(0)=0.
    return linear, linear + epsilon * (entropy - plan.sum()), linear + epsilon * entropy


def main():
    cases = [([.5, .5], [.5, .5], [[0., 1.], [1., 0.]], 1.),
             ([.2, .3, .5], [.6, .4], [[0., 2.], [1., .5], [3., 0.]], .7)]
    for a, b, cost, epsilon in cases:
        a, b, cost = np.array(a), np.array(b), np.array(cost)
        plan, status = sinkhorn_log(a, b, cost, epsilon)
        reference = ot.sinkhorn(a, b, cost, epsilon, method="sinkhorn_log",
                                numItermax=10000, stopThr=1e-12)
        assert status["converged"]
        np.testing.assert_allclose(plan, reference, atol=1e-9)
        linear, full, without_constant = objectives(plan, cost, epsilon)
        print("plan", np.round(plan, 6).tolist())
        print("residual", f"{status['residual']:.2e}", "POT max_error", f"{np.max(abs(plan-reference)):.2e}")
        print("linear/full/no-constant", np.round([linear, full, without_constant], 6).tolist())
        shifted, _ = sinkhorn_log(a, b, cost + 1000, epsilon)
        np.testing.assert_allclose(shifted, plan, atol=1e-12)
    _, stopped = sinkhorn_log([.6, .4], [.4, .6], [[0, 1], [1, 0]], .01, max_steps=5)
    print("sharp five-step budget converged", stopped["converged"])


if __name__ == "__main__":
    main()
