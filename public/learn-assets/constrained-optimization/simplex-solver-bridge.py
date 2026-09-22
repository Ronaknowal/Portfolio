"""Simplex projection, projected descent, and SciPy SLSQP on one objective.

Run: python simplex-solver-bridge.py
Dependencies: numpy==2.3.5, scipy==1.18.1.
Contract: finite moderate-scale vectors; positive diagonal curvature and budget.
Sorting projection: O(d log d) time, O(d) storage. Descent: O(K d log d).
SLSQP solves a more general problem; its stopping flag is not our certificate.
"""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize


def project_simplex(vector, budget=1.):
    """Euclidean projection onto x>=0, sum(x)=budget via a shared threshold."""
    vector = np.asarray(vector, dtype=float)
    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
        raise ValueError("Expected a nonempty finite vector")
    if not np.isfinite(budget) or budget <= 0:
        raise ValueError("budget must be positive and finite")
    # Translation invariance reduces cancellation from a large common offset.
    shifted = vector - vector.max()
    ordered = np.sort(shifted)[::-1]
    thresholds = (np.cumsum(ordered) - budget) / np.arange(1, len(vector) + 1)
    last = np.flatnonzero(ordered > thresholds)[-1]
    return np.maximum(shifted - thresholds[last], 0.)


def solve_projected(center, curvature, budget=1., tolerance=1e-10, max_steps=10000):
    """Minimize .5*sum(curvature*(x-center)^2); return the measured certificate."""
    center, curvature = np.asarray(center, float), np.asarray(curvature, float)
    if (center.ndim != 1 or curvature.shape != center.shape or not center.size
            or not np.isfinite(center).all() or not np.isfinite(curvature).all()
            or np.any(curvature <= 0) or tolerance <= 0 or max_steps < 0):
        raise ValueError("Require matching finite vectors, positive curvature and valid stop settings")
    step = 1. / curvature.max()
    x = project_simplex(center, budget)
    for iteration in range(max_steps + 1):
        proposed = project_simplex(x - step * curvature * (x - center), budget)
        residual = np.linalg.norm((x - proposed) / step, ord=np.inf)
        if residual <= tolerance or iteration == max_steps:
            return x, {"converged": bool(residual <= tolerance),
                       "iterations": iteration, "mapping": float(residual)}
        x = proposed


def solve_library(center, curvature, budget=1.):
    center, curvature = np.asarray(center, float), np.asarray(curvature, float)
    objective = lambda x: .5 * np.dot(curvature, (x - center) ** 2)
    gradient = lambda x: curvature * (x - center)
    result = minimize(objective, np.full(len(center), budget / len(center)),
                      jac=gradient, method="SLSQP", bounds=Bounds(0., np.inf),
                      constraints=LinearConstraint(np.ones((1, len(center))), budget, budget),
                      options={"ftol": 1e-13, "maxiter": 500})
    if not result.success:
        raise RuntimeError(result.message)
    return result


def main():
    for center, curvature, budget in [([2., 1.4], [1., 1.], 1.),
                                       ([2., -.2, 1., .5], [1., 3., 2., 4.], 1.5)]:
        x, certificate = solve_projected(center, curvature, budget)
        result = solve_library(center, curvature, budget)
        gradient = np.asarray(curvature) * (result.x - center)
        step = 1 / max(curvature)
        library_mapping = np.max(np.abs(result.x - project_simplex(result.x - step * gradient, budget))) / step
        assert certificate["converged"] and library_mapping < 1e-6
        assert abs(x.sum() - budget) < 1e-12 and x.min() >= 0
        np.testing.assert_allclose(x, result.x, atol=1e-7)
        print("budget", budget, "solution", np.round(x, 6).tolist())
        print("scratch mapping", f"{certificate['mapping']:.2e}",
              "SLSQP mapping", f"{library_mapping:.2e}")
    _, stopped = solve_projected([2., -.2, 1., .5], [1., 3., 2., 4.], 1.5, max_steps=1)
    print("one-step budget converged", stopped["converged"])


if __name__ == "__main__":
    main()
