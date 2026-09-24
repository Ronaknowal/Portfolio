// Complete separately executable programs; stdout captured with the recorded runtime.
export const convexOptimizationExamples = {
  allocationCertificate: {
    title: "Certify a feasible allocation against every point in its triangle",
    code: String.raw`import numpy as np

target = np.array([4.0, 3.0])
budget = 4.0
vertices = np.array([[0.0, 0.0], [budget, 0.0], [0.0, budget]])
for candidate in (np.array([2.0, 1.0]), np.array([2.5, 1.5]), target):
    gradient = candidate - target
    value = 0.5 * gradient @ gradient
    feasible = bool(np.all(candidate >= 0) and candidate.sum() <= budget)
    # A linear function on this triangle attains a minimum at a vertex.
    lower = value + np.min((vertices - candidate) @ gradient)
    gap = round(float(value - lower), 6) if feasible else None
    print("candidate:", candidate.tolist(), "feasible:", feasible)
    print("objective:", round(float(value), 6), "lower bound:", round(float(lower), 6), "gap:", gap)`,
    expected: "candidate: [2.0, 1.0] feasible: True\nobjective: 4.0 lower bound: 2.0 gap: 2.0\ncandidate: [2.5, 1.5] feasible: True\nobjective: 2.25 lower bound: 2.25 gap: 0.0\ncandidate: [4.0, 3.0] feasible: False\nobjective: 0.0 lower bound: 0.0 gap: None",
  },
  ridgeReference: {
    title: "Run the original ridge fit and check its stationary equation",
    code: String.raw`import numpy as np

X = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
y = np.array([1.0, 2.0, 2.0])
ridge = 0.5
# This example penalizes BOTH coordinates, including the intercept.
w = np.linalg.solve(X.T @ X + ridge * np.eye(2), X.T @ y)
prediction = X @ w
residual_squared = np.sum((prediction - y) ** 2)
penalty = ridge * (w @ w)
gradient = 2 * X.T @ (prediction - y) + 2 * ridge * w
hessian = 2 * (X.T @ X + ridge * np.eye(2))
print("weights:", np.round(w, 6).tolist())
print("predictions:", np.round(prediction, 6).tolist())
print("fit error squared:", round(float(residual_squared), 6))
print("penalty:", round(float(penalty), 6))
print("total objective:", round(float(residual_squared + penalty), 6))
print("stationary:", bool(np.linalg.norm(gradient) < 1e-12))
print("Hessian eigenvalues:", np.round(np.linalg.eigvalsh(hessian), 6).tolist())`,
    expected: "weights: [0.926829, 0.585366]\npredictions: [0.926829, 1.512195, 2.097561]\nfit error squared: 0.252826\npenalty: 0.600833\ntotal objective: 0.853659\nstationary: True\nHessian eigenvalues: [2.675445, 15.324555]",
  },
  curvatureSteps: {
    title: "Separate changing the objective from choosing a stable numerical step",
    code: String.raw`import numpy as np

y = np.array([1.0, 2.0, 2.0])
for label, X in [("full", np.array([[1., 0.], [1., 1.], [1., 2.]])),
                 ("duplicate", np.ones((3, 2)))]:
    for ridge in (0.0, 0.5):
        H = 2 * (X.T @ X + ridge * np.eye(2))
        mu, L = np.linalg.eigvalsh(H)
        # Reference minimizes ||Xw-y||² + ridge ||w||²; lstsq handles rank loss.
        augmented = np.vstack([X, np.sqrt(ridge) * np.eye(2)])
        response = np.concatenate([y, np.zeros(2)])
        optimum = np.linalg.lstsq(augmented, response, rcond=None)[0]
        def objective(w):
            return float(np.sum((X @ w - y) ** 2) + ridge * (w @ w))
        best = objective(optimum)
        print(label, "ridge", ridge, "curvatures", np.round([mu, L], 6).tolist())
        for factor in (1.0, 2.1):
            w = np.array([-1.0, 2.0])
            errors = [objective(w) - best]
            for _ in range(24):
                gradient = 2 * X.T @ (X @ w - y) + 2 * ridge * w
                w -= (factor / L) * gradient
                errors.append(objective(w) - best)
            print("  step*L", factor, "initial/final gaps", np.round([errors[0], errors[-1]], 6).tolist(),
                  "monotone:", bool(np.all(np.diff(errors) <= 1e-10)))`,
    expected: "full ridge 0.0 curvatures [1.675445, 14.324555]\n  step*L 1.0 initial/final gaps [5.833333, 0.014845] monotone: True\n  step*L 2.1 initial/final gaps [5.833333, 1.737794] monotone: False\nfull ridge 0.5 curvatures [2.675445, 15.324555]\n  step*L 1.0 initial/final gaps [7.646341, 0.000765] monotone: True\n  step*L 2.1 initial/final gaps [7.646341, 0.327016] monotone: False\nduplicate ridge 0.0 curvatures [0.0, 12.0]\n  step*L 1.0 initial/final gaps [1.333333, 0.0] monotone: True\n  step*L 2.1 initial/final gaps [1.333333, 129.356312] monotone: False\nduplicate ridge 0.5 curvatures [1.0, 13.0]\n  step*L 1.0 initial/final gaps [3.192308, 0.048262] monotone: True\n  step*L 2.1 initial/final gaps [3.192308, 91.420564] monotone: False",
  },
  softThreshold: {
    title: "Derive and check a signed proximal solution at and beyond the kink",
    code: String.raw`import numpy as np

def soft_threshold(values, penalty):
    if penalty < 0:
        raise ValueError("The convex penalty must be nonnegative")
    values = np.asarray(values, dtype=float)
    return np.sign(values) * np.maximum(np.abs(values) - penalty, 0.0)

inputs = np.array([-3.0, -1.0, -0.2, 0.0, 1.0, 3.0])
penalty = 1.0
solution = soft_threshold(inputs, penalty)
for value, result in zip(inputs, solution):
    # Optimality: 0 belongs to result-value + penalty * subgradient(|result|).
    if result == 0:
        residual = max(0.0, abs(value) - penalty)
    else:
        residual = abs(result - value + penalty * np.sign(result))
    print("input", float(value), "solution", float(result), "optimality residual", float(residual))
print("zero penalty:", soft_threshold(inputs, 0.0).tolist())`,
    expected: "input -3.0 solution -2.0 optimality residual 0.0\ninput -1.0 solution -0.0 optimality residual 0.0\ninput -0.2 solution -0.0 optimality residual 0.0\ninput 0.0 solution 0.0 optimality residual 0.0\ninput 1.0 solution 0.0 optimality residual 0.0\ninput 3.0 solution 2.0 optimality residual 0.0\nzero penalty: [-3.0, -1.0, -0.2, 0.0, 1.0, 3.0]",
  },
  proximalLasso: {
    title: "Combine a smooth fit step with a nonsmooth penalty step",
    code: String.raw`import numpy as np

A = np.array([[1., 0., 1.], [0., 1., 1.], [1., 1., 0.], [2., 0., 1.]])
b = np.array([1., 0., 1., 2.])
penalty = 0.2
L = np.linalg.norm(A, 2) ** 2
step = 1.0 / L
w = np.zeros(A.shape[1])
for iteration in range(1, 5001):
    gradient = A.T @ (A @ w - b)
    forward = w - step * gradient
    updated = np.sign(forward) * np.maximum(np.abs(forward) - step * penalty, 0.0)
    mapping_norm = np.linalg.norm((w - updated) / step)
    w = updated
    if mapping_norm < 1e-10:
        break
else:
    raise RuntimeError("The stated stopping tolerance was not reached")

gradient = A.T @ (A @ w - b)
residuals = np.where(np.abs(w) > 1e-8,
                     np.abs(gradient + penalty * np.sign(w)),
                     np.maximum(np.abs(gradient) - penalty, 0.0))
objective = 0.5 * np.sum((A @ w - b) ** 2) + penalty * np.sum(np.abs(w))
print("iterations:", iteration)
print("weights:", np.round(w, 6).tolist())
print("objective:", round(float(objective), 6))
print("stationarity within 1e-8:", bool(np.max(residuals) < 1e-8))
print("positive definite fit Hessian:", bool(np.min(np.linalg.eigvalsh(A.T @ A)) > 0))`,
    expected: "iterations: 30\nweights: [0.966667, 0.0, 0.0]\nobjective: 0.196667\nstationarity within 1e-8: True\npositive definite fit Hessian: True",
  },
  cvxpyAllocation: {
    title: "Solve the same allocation and independently inspect its certificate",
    code: String.raw`import cvxpy as cp
import numpy as np

target = np.array([4.0, 3.0])
budget = 4.0
allocation = cp.Variable(2)
problem = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(allocation - target)),
                     [allocation >= 0, cp.sum(allocation) <= budget])
problem.solve(solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
print("DCP:", problem.is_dcp(), "status:", problem.status)
if problem.status != cp.OPTIMAL or allocation.value is None:
    raise RuntimeError("No solution at the requested accuracy was returned")
w = allocation.value
feasibility_residual = max(float(np.max(-w)), float(w.sum() - budget), 0.0)
gradient = w - target
value = float(0.5 * gradient @ gradient)
vertices = np.array([[0., 0.], [budget, 0.], [0., budget]])
lower_bound = value + float(np.min((vertices - w) @ gradient))
gap = value - lower_bound
print("allocation:", np.round(w, 6).tolist())
print("objective:", round(value, 6))
print("feasibility residual < 1e-8:", feasibility_residual < 1e-8)
print("supporting-plane gap < 1e-8:", abs(gap) < 1e-8)
print("matches geometric solution:", bool(np.allclose(w, [2.5, 1.5], atol=1e-8)))`,
    expected: "DCP: True status: optimal\nallocation: [2.5, 1.5]\nobjective: 2.25\nfeasibility residual < 1e-8: True\nsupporting-plane gap < 1e-8: True\nmatches geometric solution: True",
  },
  dcpAndStatus: {
    title: "Keep a convexity check separate from a successful optimization result",
    code: String.raw`import cvxpy as cp

x = cp.Variable()
# Both represent sqrt(4 + (x-1)²), but only one follows DCP composition rules.
first = cp.sqrt(4 + cp.square(x - 1))
second = cp.norm(cp.hstack([2, x - 1]), 2)
print("sqrt expression:", first.curvature, first.is_dcp())
print("norm expression:", second.curvature, second.is_dcp())

infeasible = cp.Problem(cp.Minimize(cp.square(x)), [x >= 3, x <= 1])
unbounded = cp.Problem(cp.Minimize(-2 * x))
for name, problem in [("contradictory constraints", infeasible), ("unbounded objective", unbounded)]:
    problem.solve(solver=cp.CLARABEL)
    print(name, "DCP:", problem.is_dcp(), "status:", problem.status)
    # Do not use a variable value merely because the expression was accepted.
    if problem.status != cp.OPTIMAL:
        print("  no certified finite minimizer to report")`,
    expected: "sqrt expression: QUASICONVEX False\nnorm expression: CONVEX True\ncontradictory constraints DCP: True status: infeasible\n  no certified finite minimizer to report\nunbounded objective DCP: True status: unbounded\n  no certified finite minimizer to report",
  },
  robustEpigraph: {
    title: "Choose a constant calibration that minimizes its worst absolute error",
    code: String.raw`import cvxpy as cp
import numpy as np

observations = np.array([1.0, 2.0, 6.0])
level = cp.Variable()
worst = cp.Variable()
problem = cp.Problem(cp.Minimize(worst),
                     [level - observations <= worst, observations - level <= worst])
problem.solve(solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
if problem.status != cp.OPTIMAL:
    raise RuntimeError(problem.status)
solution = float(level.value)
actual_error = float(np.max(np.abs(solution - observations)))
# Any level is at least half the endpoint distance from one of the two extremes.
lower_bound = float((observations.max() - observations.min()) / 2)
mean = float(observations.mean())
print("status:", problem.status)
print("minimax level:", round(solution, 6), "worst error:", round(actual_error, 6))
print("endpoint lower bound:", lower_bound)
print("bound attained within 1e-8:", abs(actual_error - lower_bound) < 1e-8)
print("least-squares level:", mean, "its worst error:", float(np.max(np.abs(mean - observations))))`,
    expected: "status: optimal\nminimax level: 3.5 worst error: 2.5\nendpoint lower bound: 2.5\nbound attained within 1e-8: True\nleast-squares level: 3.0 its worst error: 3.0",
  },
  totalVariation: {
    title: "Denoise neighboring readings while allowing a real jump",
    code: String.raw`import cvxpy as cp
import numpy as np

readings = np.array([0.2, -0.1, 0.1, 3.0, 3.2, 2.8])
penalty = 0.3
signal = cp.Variable(len(readings))
differences = signal[1:] - signal[:-1]
problem = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(signal - readings)
                                + penalty * cp.norm(differences, 1)))
problem.solve(solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
if problem.status != cp.OPTIMAL:
    raise RuntimeError(problem.status)
estimate = signal.value
fit = float(0.5 * np.sum((estimate - readings) ** 2))
roughness = float(np.sum(np.abs(np.diff(estimate))))
print("status:", problem.status)
print("readings:", readings.tolist())
print("estimate:", np.round(estimate, 6).tolist())
print("half squared fit:", round(fit, 6))
print("total variation:", round(roughness, 6))
print("objective:", round(fit + penalty * roughness, 6))`,
    expected: "status: optimal\nreadings: [0.2, -0.1, 0.1, 3.0, 3.2, 2.8]\nestimate: [0.166667, 0.166667, 0.166667, 2.9, 2.9, 2.9]\nhalf squared fit: 0.093333\ntotal variation: 2.733333\nobjective: 0.913333",
  },
};
