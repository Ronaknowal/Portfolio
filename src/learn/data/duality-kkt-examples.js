const projectionDefinition = `import numpy as np

def project_halfspace(target, normal, budget):
    target = np.asarray(target, dtype=float)
    normal = np.asarray(normal, dtype=float)
    if target.ndim != 1 or target.size == 0 or normal.shape != target.shape:
        raise ValueError("Use matching nonempty vectors")
    if not np.isfinite(target).all() or not np.isfinite(normal).all() or not np.isfinite(budget):
        raise ValueError("Inputs must be finite")
    squared_norm = float(normal @ normal)
    if squared_norm == 0:
        raise ValueError("This half-space example requires a nonzero normal")
    excess = max(0.0, float(normal @ target) - budget)
    point = target - (excess / squared_norm) * normal
    multiplier = 2 * excess / squared_norm
    return point, multiplier

def dual_value(target, normal, budget, multiplier):
    return (multiplier * (normal @ target - budget)
            - multiplier**2 * (normal @ normal) / 4)`;

const resourceDefinition = `import numpy as np

def allocation_at_price(price, budget):
    if not np.isfinite(price) or price < 0 or not np.isfinite(budget) or budget < 0:
        raise ValueError("Use finite nonnegative price and budget")
    allocation = np.maximum(0.0, np.array([3.0, 4.0]) - price / np.array([2.0, 4.0]))
    objective = (allocation[0] - 3)**2 + 2 * (allocation[1] - 4)**2
    violation = float(allocation.sum() - budget)
    dual = float(objective + price * violation)
    # Keep the first allocation if possible; give the second the remaining budget.
    # This is a feasible repair, not the optimal Euclidean projection.
    repaired = np.array([min(allocation[0], budget), 0.0])
    repaired[1] = min(allocation[1], max(0.0, budget - repaired[0]))
    upper = float((repaired[0] - 3)**2 + 2 * (repaired[1] - 4)**2)
    return allocation, violation, dual, repaired, upper

def stable_repair_gap(price, budget, allocation, repaired):
    weights, targets = np.array([1.0, 2.0]), np.array([3.0, 4.0])
    displacement = repaired - allocation
    boundary_gradient = np.where(allocation == 0, np.maximum(0, price - 2 * weights * targets), 0)
    # A priority repair computes the second allocation from this remaining budget.
    slack = (budget - repaired[0]) - repaired[1]
    local_gap = float(np.sum(weights * displacement**2 + boundary_gradient * displacement))
    return local_gap + price * slack

def price_iterations(budget, rate, updates, initial_price=0.0):
    if type(updates) is not int or updates < 0:
        raise ValueError("Use a nonnegative integer number of updates")
    if not np.isfinite(rate) or rate < 0:
        raise ValueError("Use a finite nonnegative rate")
    price = float(initial_price)
    rows = []
    for step in range(updates + 1):
        allocation, violation, dual, repaired, upper = allocation_at_price(price, budget)
        rows.append((price, allocation, violation, dual, repaired, upper))
        if step < updates:
            price = max(0.0, price + rate * violation)
    return rows`;

export const dualityKktExamples = {
  projection: {
    title: 'Compute the closest feasible point and its multiplier',
    code: `${projectionDefinition}

target, normal = np.array([3.0, 4.0]), np.ones(2)
for budget in [5.0, 7.0, 9.0]:
    point, multiplier = project_halfspace(target, normal, budget)
    objective = float(np.sum((point - target)**2))
    print("budget", budget, "point", point.tolist(),
          "objective", objective, "multiplier", multiplier)
    assert np.allclose(2 * (point - target) + multiplier * normal, 0)
    assert np.isclose(objective, dual_value(target, normal, budget, multiplier))`,
    expected: "budget 5.0 point [2.0, 3.0] objective 2.0 multiplier 2.0\r\nbudget 7.0 point [3.0, 4.0] objective 0.0 multiplier 0.0\r\nbudget 9.0 point [3.0, 4.0] objective 0.0 multiplier 0.0",
  },
  bounds: {
    title: 'Check the bound chain before interpreting a gap',
    code: `${projectionDefinition}

target, normal, budget = np.array([3.0, 4.0]), np.ones(2), 5.0
for point, multiplier in [(np.array([2.0, 2.0]), 1.0),
                          (np.array([4.0, 4.0]), 2.0),
                          (np.array([2.0, 3.0]), -1.0)]:
    cost = float(np.sum((point - target)**2))
    residual = float(normal @ point - budget)
    lower = float(dual_value(target, normal, budget, multiplier))
    lagrangian = cost + multiplier * residual
    minimizer = target - multiplier * normal / 2
    decomposition = (-multiplier * residual + np.sum((point - minimizer)**2))
    assert np.isclose(cost - lower, decomposition)
    valid = residual <= 0 and multiplier >= 0
    print("x", point.tolist(), "lambda", multiplier, "L", lagrangian, "q", lower)
    print("valid certificate", valid, "gap", cost - lower if valid else None)`,
    expected: "x [2.0, 2.0] lambda 1.0 L 4.0 q 1.5\r\nvalid certificate True gap 3.5\r\nx [4.0, 4.0] lambda 2.0 L 7.0 q 2.0\r\nvalid certificate False gap None\r\nx [2.0, 3.0] lambda -1.0 L 2.0 q -2.5\r\nvalid certificate False gap None",
  },
  equality: {
    title: 'An equality multiplier can be negative',
    code: `import numpy as np

# Minimize ||x||^2 with A x = b. Here A has independent rows.
A = np.array([[1.0, 1.0]])
b = np.array([2.0])
n, m = A.shape[1], A.shape[0]
K = np.block([[2 * np.eye(n), A.T], [A, np.zeros((m, m))]])
solution = np.linalg.solve(K, np.concatenate([np.zeros(n), b]))
x, nu = solution[:n], solution[n:]
dual = float(-0.25 * nu @ A @ A.T @ nu - b @ nu)
print("equality point", x.tolist(), "nu", nu.tolist(), "value", float(x @ x))
print("dual", dual, "stationarity", np.allclose(2 * x + A.T @ nu, 0))

# Changed contract: x1+x2 >= 2 is g(x)=2-x1-x2 <= 0.
lam = 2.0
print("inequality lambda", lam,
      "stationarity", np.allclose(2 * x - lam * np.ones(2), 0),
      "slack", float(x.sum() - 2))`,
    expected: "equality point [1.0, 1.0] nu [-2.0] value 2.0\r\ndual 2.0 stationarity True\r\ninequality lambda 2.0 stationarity True slack 0.0",
  },
  conditions: {
    title: 'Diagnose each KKT condition separately',
    code: `def scalar_kkt(center, point, multiplier):
    constraint = -point
    stationarity = 2 * (point - center) - multiplier
    product = multiplier * constraint
    return (constraint <= 0, multiplier >= 0, stationarity == 0, product == 0)

# Order: primal feasibility, dual sign, stationarity, complementarity.
for center, point, multiplier in [(-1, 0, 2), (0, 0, 0), (1, 1, 0),
                                  (-1, 0, 0), (-1, 1, 4), (-1, -1, 0)]:
    print((center, point, multiplier), scalar_kkt(center, point, multiplier))

# Duplicate constraints -x <= 0 and -2x <= 0 at the unique optimum x=0.
# Stationarity for f=(x+1)^2 requires lambda1+2*lambda2=2.
for first, second in [(2, 0), (1, 0.5), (0, 1)]:
    print("duplicate prices", (first, second),
          "same certificate", 2 - first - 2 * second == 0)`,
    expected: "(-1, 0, 2) (True, True, True, True)\r\n(0, 0, 0) (True, True, True, True)\r\n(1, 1, 0) (True, True, True, True)\r\n(-1, 0, 0) (True, True, False, True)\r\n(-1, 1, 4) (True, True, True, False)\r\n(-1, -1, 0) (False, True, True, True)\r\nduplicate prices (2, 0) same certificate True\r\nduplicate prices (1, 0.5) same certificate True\r\nduplicate prices (0, 1) same certificate True",
  },
  domain: {
    title: 'A nonnegative multiplier can still give only minus infinity',
    code: `from math import inf

# Minimize x with 1-x <= 0, retaining domain x in all real numbers.
def dual(multiplier):
    if multiplier < 0:
        raise ValueError("Inequality multipliers must be nonnegative")
    return 1.0 if multiplier == 1 else -inf

for multiplier in [0.0, 0.5, 1.0, 2.0]:
    slope = 1 - multiplier
    print("lambda", multiplier, "slope in x", slope, "q", dual(multiplier))
print("primal optimum", 1.0, "attained dual maximum", dual(1.0))`,
    expected: "lambda 0.0 slope in x 1.0 q -inf\r\nlambda 0.5 slope in x 0.5 q -inf\r\nlambda 1.0 slope in x 0.0 q 1.0\r\nlambda 2.0 slope in x -1.0 q -inf\r\nprimal optimum 1.0 attained dual maximum 1.0",
  },
  qualification: {
    title: 'Separate equal values, attained multipliers and a nonconvex gap',
    code: `from math import inf

# Convex but degenerate: minimize x subject to x*x <= 0.
def degenerate_dual(multiplier):
    if multiplier < 0:
        raise ValueError("Use a nonnegative multiplier")
    return -1 / (4 * multiplier) if multiplier > 0 else -inf

for multiplier in [0, 1, 10, 100]:
    print("lambda", multiplier, "q", degenerate_dual(multiplier))
print("primal at x=0", 0, "Lagrangian derivative there", 1)
print("affine reformulation x=0: nu=-1 makes L identically zero")

# Nonconvex equality: minimize x with -x<=0 and x*x=1. Only x=1 is feasible.
def nonconvex_dual(lam, nu):
    if lam < 0:
        raise ValueError("lambda must be nonnegative")
    if nu > 0:
        return -nu - (1 - lam)**2 / (4 * nu)
    return 0.0 if nu == 0 and lam == 1 else -inf

print("nonconvex primal", 1.0, "dual optimum", nonconvex_dual(1, 0), "gap", 1.0)`,
    expected: "lambda 0 q -inf\r\nlambda 1 q -0.25\r\nlambda 10 q -0.025\r\nlambda 100 q -0.0025\r\nprimal at x=0 0 Lagrangian derivative there 1\r\naffine reformulation x=0: nu=-1 makes L identically zero\r\nnonconvex primal 1.0 dual optimum 0.0 gap 1.0",
  },
  sensitivity: {
    title: 'Compare actual reoptimized cost with a supporting price',
    code: `def value(budget):
    return max(0.0, 7 - budget)**2 / 2

for budget, change in [(5.0, 0.1), (5.0, -0.1), (7.0, -0.1)]:
    price = max(0.0, 7 - budget)
    actual = value(budget + change) - value(budget)
    linear = -price * change
    print("budget", budget, "change", change, "price", price,
          "actual cost change", round(actual, 6), "linear", round(linear, 6))
    assert actual >= linear - 1e-12

# min t with -t<=0 and -t<=u has value max(0,-u).
# At u=0 every price in [0,1] is dual optimal; the derivative is not unique.
for price in [0.0, 0.5, 1.0]:
    for change in [-0.2, 0.2]:
        actual = max(0.0, -change)
        supporting = -price * change
        print("kink price", price, "change", change,
              "value", actual, "lower line", supporting)
        assert actual >= supporting

print("constraint scaled by 10: price", 2.0 / 10, "same contribution", (2.0 / 10) * 10)`,
    expected: "budget 5.0 change 0.1 price 2.0 actual cost change -0.195 linear -0.2\r\nbudget 5.0 change -0.1 price 2.0 actual cost change 0.205 linear 0.2\r\nbudget 7.0 change -0.1 price 0.0 actual cost change 0.005 linear 0.0\r\nkink price 0.0 change -0.2 value 0.2 lower line 0.0\r\nkink price 0.0 change 0.2 value 0.0 lower line -0.0\r\nkink price 0.5 change -0.2 value 0.2 lower line 0.1\r\nkink price 0.5 change 0.2 value 0.0 lower line -0.1\r\nkink price 1.0 change -0.2 value 0.2 lower line 0.2\r\nkink price 1.0 change 0.2 value 0.0 lower line -0.2\r\nconstraint scaled by 10: price 0.2 same contribution 2.0",
  },
  resource: {
    title: 'Let local allocations respond to a shared price',
    code: `${resourceDefinition}

for rate in [1.0, 3.0, 4.0]:
    rows = price_iterations(5.0, rate, 8)
    print("rate", rate, "first prices", [round(row[0], 6) for row in rows[:5]])
    price, allocation, violation, lower, repaired, upper = rows[-1]
    print("final allocation", np.round(allocation, 6).tolist(),
          "violation", round(violation, 6))
    print("lower", round(lower, 6), "repaired upper", round(upper, 6),
          "stable gap", format(stable_repair_gap(price, 5.0, allocation, repaired), ".6g"))
    for _, _, _, lower, repaired, upper in rows:
        assert np.all(repaired >= 0) and repaired.sum() <= 5 + 1e-12
        assert lower <= 8/3 + 1e-10 and upper >= 8/3 - 1e-10
print("analytic optimum", [5/3, 10/3], "price", 8/3, "value", 8/3)`,
    expected: "rate 1.0 first prices [0.0, 2.0, 2.5, 2.625, 2.65625]\r\nfinal allocation [1.666687, 3.333344] violation 3.1e-05\r\nlower 2.666667 repaired upper 2.666667 stable gap 1.86265e-09\r\nrate 3.0 first prices [0.0, 6.0, 0.0, 6.0, 0.0]\r\nfinal allocation [3.0, 4.0] violation 2.0\r\nlower 0.0 repaired upper 8.0 stable gap 8\r\nrate 4.0 first prices [0.0, 8.0, 0.0, 8.0, 0.0]\r\nfinal allocation [3.0, 4.0] violation 2.0\r\nlower 0.0 repaired upper 8.0 stable gap 8\r\nanalytic optimum [1.6666666666666667, 3.3333333333333335] price 2.6666666666666665 value 2.6666666666666665",
  },
  solver: {
    title: 'Check solver multipliers in the original coordinates',
    code: `import cvxpy as cp
import numpy as np

target, normal, budget = np.array([3.0, 4.0]), np.ones(2), 5.0
x = cp.Variable(2)
limit = normal @ x <= budget
problem = cp.Problem(cp.Minimize(cp.sum_squares(x - target)), [limit])
problem.solve(solver="CLARABEL", tol_gap_abs=1e-10,
              tol_gap_rel=1e-10, tol_feas=1e-10)
if problem.status != cp.OPTIMAL:
    raise RuntimeError(f"Inspect solver status: {problem.status}")
point, multiplier = x.value, float(limit.dual_value)
primal_residual = max(0.0, float(normal @ point - budget))
stationarity = np.linalg.norm(2 * (point - target) + multiplier * normal)
complementarity = abs(multiplier * float(normal @ point - budget))
print("status", problem.status)
print("point", np.round(point, 6).tolist(), "lambda", round(multiplier, 6))
print("residuals below 1e-7", bool(max(primal_residual, stationarity, complementarity) < 1e-7))

# Validate a known exact feasible reconstruction separately from floating residuals.
feasible = np.array([2.0, 3.0])
lam = max(0.0, multiplier)
upper = float(np.sum((feasible - target)**2))
# Completed-square form avoids cancellation around the optimum lambda=2.
lower = 2.0 - 0.5 * (lam - 2.0)**2
print("reconstructed feasible", bool(normal @ feasible <= budget))
print("upper", upper, "lower", round(lower, 8), "gap", round(upper - lower, 8))
assert upper - lower >= 0 and np.allclose(point, feasible, atol=1e-7)`,
    expected: "status optimal\r\npoint [2.0, 3.0] lambda 2.0\r\nresiduals below 1e-7 True\r\nreconstructed feasible True\r\nupper 2.0 lower 2.0 gap 0.0",
  },
  svm: {
    title: 'Recover a separating weight from the dual coefficients',
    code: `import cvxpy as cp
import numpy as np

X = np.array([[-2.0, 0.0], [-1.0, 1.0], [1.0, 1.0], [2.0, 0.0]])
y = np.array([-1.0, -1.0, 1.0, 1.0])
w, offset = cp.Variable(2), cp.Variable()
margin = cp.multiply(y, X @ w + offset) >= 1
primal = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(w)), [margin])
options = dict(solver="CLARABEL", tol_gap_abs=1e-10,
               tol_gap_rel=1e-10, tol_feas=1e-10)
primal.solve(**options)

alpha = cp.Variable(len(y))
recovered = X.T @ cp.multiply(y, alpha)
dual = cp.Problem(cp.Maximize(cp.sum(alpha) - 0.5 * cp.sum_squares(recovered)),
                  [alpha >= 0, y @ alpha == 0])
dual.solve(**options)
if primal.status != cp.OPTIMAL or dual.status != cp.OPTIMAL:
    raise RuntimeError("Both problems must report an inspected optimal result")
weights_from_dual = X.T @ (y * alpha.value)
slack = y * (X @ w.value + offset.value) - 1
def display(values):
    result = np.round(np.asarray(values), 6)
    result[result == 0] = 0  # Normalize only displayed signed zero.
    return result.tolist()
print("weight", display(w.value), "offset", round(float(offset.value), 6))
print("dual coefficients", display(alpha.value))
print("margin slack", display(slack))
print("primal and dual values", round(primal.value, 6), round(dual.value, 6))
print("weight reconstruction", bool(np.allclose(w.value, weights_from_dual, atol=1e-7)))
print("complementarity", bool(np.max(np.abs(alpha.value * slack)) < 1e-7))
assert np.max(-slack) < 1e-7 and np.min(alpha.value) > -1e-7
assert abs(float(y @ alpha.value)) < 1e-7
assert np.allclose(w.value, [1.0, 0.0], atol=1e-7)`,
    expected: "weight [1.0, 0.0] offset 0.0\r\ndual coefficients [0.0, 0.5, 0.5, 0.0]\r\nmargin slack [1.0, 0.0, 0.0, 1.0]\r\nprimal and dual values 0.5 0.5\r\nweight reconstruction True\r\ncomplementarity True",
  },
};
