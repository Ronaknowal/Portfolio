export const matrixDecompositionExamples = {
  squareSolve: {
    title: 'Solve the original two-equation system and inspect the residual',
    code: `import numpy as np

A = np.array([[3.0, 1.0], [1.0, 2.0]])
b = np.array([9.0, 8.0])
x = np.linalg.solve(A, b)
print("solution:", x.tolist())
print("reconstructed target:", (A @ x).tolist())
print("residual norm:", f"{np.linalg.norm(b - A @ x):.6f}")`,
    expected: `solution: [2.0, 3.0]
reconstructed target: [9.0, 8.0]
residual norm: 0.000000`,
  },
  pivotedLu: {
    title: 'Store partial-pivoting LU factors and reuse them for another target',
    code: `import numpy as np

def factor_lu(matrix):
    if np.iscomplexobj(matrix):
        raise ValueError("This teaching implementation uses real coefficients")
    upper = np.array(matrix, dtype=float, copy=True)
    if (upper.ndim != 2 or upper.shape[0] != upper.shape[1]
            or upper.shape[0] == 0 or not np.isfinite(upper).all()):
        raise ValueError("Use a nonempty, finite, square real matrix")
    size = upper.shape[0]
    lower = np.eye(size)
    ordering = np.arange(size)
    for column in range(size):
        pivot = column + int(np.argmax(np.abs(upper[column:, column])))
        if upper[pivot, column] == 0:
            raise np.linalg.LinAlgError("Zero available pivot")
        if pivot != column:
            upper[[column, pivot], :] = upper[[pivot, column], :]
            ordering[[column, pivot]] = ordering[[pivot, column]]
            # Earlier multipliers belong to the equations being reordered.
            lower[[column, pivot], :column] = lower[[pivot, column], :column]
        for row in range(column + 1, size):
            multiplier = upper[row, column] / upper[column, column]
            lower[row, column] = multiplier
            upper[row, column:] -= multiplier * upper[column, column:]
            upper[row, column] = 0.0
    return ordering, lower, upper

def solve_factored(ordering, lower, upper, target):
    target = np.asarray(target, dtype=float)
    size = len(ordering)
    if target.shape != (size,) or not np.isfinite(target).all():
        raise ValueError("Use one finite target per equation")
    reordered = target[ordering]
    intermediate = np.empty(size)
    for row in range(size):
        intermediate[row] = reordered[row] - lower[row, :row] @ intermediate[:row]
    solution = np.empty(size)
    for row in range(size - 1, -1, -1):
        known = upper[row, row + 1:] @ solution[row + 1:]
        solution[row] = (intermediate[row] - known) / upper[row, row]
    return solution

A = np.array([[0., 2., 1.], [1., 1., 0.], [2., 0., 1.]])
ordering, L, U = factor_lu(A)
print("row order:", ordering.tolist())
print("PA equals LU:", bool(np.allclose(A[ordering], L @ U)))
for b in [np.array([7., 3., 5.]), np.array([1., 0., 3.])]:
    answer = solve_factored(ordering, L, U, b)
    print("solution:", np.round(answer, 6).tolist())
    print("satisfies original:", bool(np.allclose(A @ answer, b)))`,
    expected: `row order: [2, 0, 1]
PA equals LU: True
solution: [1.0, 2.0, 3.0]
satisfies original: True
solution: [0.5, -0.5, 2.0]
satisfies original: True`,
  },
  qrFit: {
    title: 'Fit the original noisy line using an explicit QR solve',
    code: `import numpy as np

time = np.array([0., 1., 2., 3.])
measurement = np.array([0.1, 1.0, 2.2, 2.9])
A = np.column_stack([time, np.ones_like(time)])
Q, R = np.linalg.qr(A, mode="reduced")
coefficients = np.linalg.solve(R, Q.T @ measurement)
prediction = A @ coefficients
residual = measurement - prediction
reference = np.linalg.lstsq(A, measurement, rcond=None)[0]
print("slope, intercept:", np.round(coefficients, 6).tolist())
print("predictions:", np.round(prediction, 6).tolist())
print("squared residual:", f"{residual @ residual:.6f}")
print("residual orthogonal:", bool(np.allclose(A.T @ residual, 0)))
print("agrees with lstsq:", bool(np.allclose(coefficients, reference)))`,
    expected: `slope, intercept: [0.96, 0.11]
predictions: [0.11, 1.07, 2.03, 2.99]
squared residual: 0.042000
residual orthogonal: True
agrees with lstsq: True`,
  },
  householder: {
    title: 'A reflection can turn a column into a triangular direction',
    code: `import numpy as np

a = np.array([3., 4.])
# Choose the opposite sign to avoid subtracting two nearly equal numbers.
alpha = -np.copysign(np.linalg.norm(a), a[0])
v = a.copy()
v[0] -= alpha
v /= np.linalg.norm(v)
H = np.eye(2) - 2 * np.outer(v, v)
print("reflected column:", np.round(H @ a, 6).tolist())
print("length preserved:", bool(np.allclose(np.linalg.norm(H @ a), 5)))
print("orthogonal:", bool(np.allclose(H.T @ H, np.eye(2))))
print("reflection determinant:", f"{np.linalg.det(H):.0f}")`,
    expected: `reflected column: [-5.0, 0.0]
length preserved: True
orthogonal: True
reflection determinant: -1`,
  },
  covariance: {
    title: 'Derive a covariance factor, sample, and whiten with a solve',
    code: `import numpy as np

C = np.array([[1., 0.8], [0.8, 1.]])
if not np.allclose(C, C.T):
    raise ValueError("Covariance must be symmetric")
L = np.linalg.cholesky(C)
rng = np.random.default_rng(7)
independent = rng.normal(size=(10_000, 2))
correlated = independent @ L.T
# Each sample is stored as a row; solve column-oriented systems via transpose.
whitened = np.linalg.solve(L, correlated.T).T
print("L:", np.round(L, 6).tolist())
print("LL.T equals C:", bool(np.allclose(L @ L.T, C)))
print("sample covariance:", np.round(np.cov(correlated, rowvar=False), 4).tolist())
print("whitening reverses transform:", bool(np.allclose(whitened, independent)))
print("log determinant:", f"{2 * np.log(np.diag(L)).sum():.6f}")`,
    expected: `L: [[1.0, 0.0], [0.8, 0.6]]
LL.T equals C: True
sample covariance: [[0.9952, 0.8017], [0.8017, 0.9966]]
whitening reverses transform: True
log determinant: -1.021651`,
  },
  covarianceBoundary: {
    title: 'Check symmetry separately and distinguish positive semidefinite inputs',
    code: `import numpy as np

cases = {
    "positive definite": np.array([[4., 2.], [2., 2.]]),
    "semidefinite": np.array([[1., 1.], [1., 1.]]),
    "indefinite": np.array([[1., 2.], [2., 1.]]),
    "asymmetric": np.array([[1., 9.], [0., 1.]]),
}
for name, matrix in cases.items():
    if not np.allclose(matrix, matrix.T):
        print(name + ": rejected before Cholesky")
        continue
    try:
        lower = np.linalg.cholesky(matrix)
        print(name + ": factor verified = " + str(bool(np.allclose(lower @ lower.T, matrix))))
    except np.linalg.LinAlgError:
        print(name + ": no strictly positive Cholesky factor")`,
    expected: `positive definite: factor verified = True
semidefinite: no strictly positive Cholesky factor
indefinite: no strictly positive Cholesky factor
asymmetric: rejected before Cholesky`,
  },
  ratingsSvd: {
    title: 'Retain two directions of the original dense ratings fixture',
    code: `import numpy as np

# Every entry, including 0, is treated as an observed numerical value here.
ratings = np.array([[5., 4., 0.], [4., 5., 1.], [1., 0., 5.]])
U, singular_values, Vt = np.linalg.svd(ratings, full_matrices=False)
rank_budget = 2
approximation = (U[:, :rank_budget] * singular_values[:rank_budget]) @ Vt[:rank_budget]
error = np.linalg.norm(ratings - approximation, "fro")
discarded_error = np.linalg.norm(singular_values[rank_budget:])
print("singular values:", np.round(singular_values, 6).tolist())
print("rank-two matrix:", np.round(approximation, 6).tolist())
print("Frobenius error:", f"{error:.6f}")
print("discarded values predict error:", bool(np.allclose(error, discarded_error)))
print("factor entries / original entries:", rank_budget * (3 + 3 + 1), "/", ratings.size)`,
    expected: `singular values: [9.122749, 4.960125, 1.082874]
rank-two matrix: [[4.465971, 4.525322, 0.087792], [4.54288, 4.465971, 0.910753], [0.910753, 0.087792, 5.014672]]
Frobenius error: 1.082874
discarded values predict error: True
factor entries / original entries: 14 / 9`,
  },
  minimumNorm: {
    title: 'A rank-deficient system can have infinitely many equally good fits',
    code: `import numpy as np

A = np.array([[1., 1.], [2., 2.], [3., 3.]])
b = np.array([2., 4., 6.])
x, residual_summary, rank, singular_values = np.linalg.lstsq(A, b, rcond=None)
alternative = np.array([2., 0.])
print("minimum-norm solution:", np.round(x, 6).tolist())
print("numerical rank:", int(rank))
print("returned residual entries:", residual_summary.size)
print("actual residual norm:", f"{np.linalg.norm(b - A @ x):.6f}")
print("alternative is also exact:", bool(np.allclose(A @ alternative, b)))
print("norms:", f"{np.linalg.norm(x):.6f}", f"{np.linalg.norm(alternative):.6f}")`,
    expected: `minimum-norm solution: [1.0, 1.0]
numerical rank: 1
returned residual entries: 0
actual residual norm: 0.000000
alternative is also exact: True
norms: 1.414214 2.000000`,
  },
  sensitivity: {
    title: 'A tiny residual does not settle sensitivity to the measured target',
    code: `import numpy as np

A = np.diag([1., 1e-8])
b = np.array([1., 1e-8])
changed_b = np.array([1., 2e-8])
x = np.linalg.solve(A, b)
changed_x = np.linalg.solve(A, changed_b)
print("original solution:", x.tolist())
print("changed solution:", changed_x.tolist())
print("relative target change:", f"{np.linalg.norm(changed_b-b)/np.linalg.norm(b):.2e}")
print("relative solution change:", f"{np.linalg.norm(changed_x-x)/np.linalg.norm(x):.6f}")
print("both residuals tiny:", bool(np.linalg.norm(A@x-b) < 1e-15 and np.linalg.norm(A@changed_x-changed_b) < 1e-15))
print("condition numbers:", f"{np.linalg.cond(A):.0e}", f"{np.linalg.cond(A.T@A):.0e}")`,
    expected: `original solution: [1.0, 1.0]
changed solution: [1.0, 2.0]
relative target change: 1.00e-08
relative solution change: 0.707107
both residuals tiny: True
condition numbers: 1e+08 1e+16`,
  },
  independentFit: {
    title: 'Independent calibration task: solve and verify a changed dataset',
    code: `import numpy as np

input_level = np.array([-1., 0., 1.])
output_level = np.array([1., 1., 4.])
A = np.column_stack([input_level, np.ones_like(input_level)])
Q, R = np.linalg.qr(A)
x = np.linalg.solve(R, Q.T @ output_level)
residual = output_level - A @ x
print("slope, intercept:", np.round(x, 6).tolist())
print("prediction:", np.round(A @ x, 6).tolist())
print("residual:", np.round(residual, 6).tolist())
print("squared error:", f"{residual @ residual:.6f}")
print("orthogonal residual:", bool(np.allclose(A.T @ residual, 0)))`,
    expected: `slope, intercept: [1.5, 2.0]
prediction: [0.5, 2.0, 3.5]
residual: [0.5, -1.0, 0.5]
squared error: 1.500000
orthogonal residual: True`,
  },
};
