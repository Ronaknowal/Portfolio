export const eigenvalueExamples = {
  symmetricPairs: {
    title: 'Check the original symmetric transformation without relying on vector signs',
    code: `import numpy as np

A = np.array([[2.0, 1.0], [1.0, 2.0]])
values, vectors = np.linalg.eigh(A)
v = np.array([1.0, 1.0])
print("eigenvalues:", values.round(6).tolist())
print("A @ [1, 1]:", (A @ v).tolist())
print("3 * [1, 1]:", (3 * v).tolist())
print("all returned pairs:", np.allclose(A @ vectors, vectors * values))
print("orthonormal columns:", np.allclose(vectors.T @ vectors, np.eye(2)))
print("trace and eigenvalue sum:", np.trace(A), values.sum())
print("determinant and product:", f"{np.linalg.det(A):.6f}", f"{values.prod():.6f}")`,
    expected: "eigenvalues: [1.0, 3.0]\nA @ [1, 1]: [3.0, 3.0]\n3 * [1, 1]: [3.0, 3.0]\nall returned pairs: True\northonormal columns: True\ntrace and eigenvalue sum: 4.0 4.0\ndeterminant and product: 3.000000 3.000000"
  },
  diagonalizedPowers: {
    title: 'Compute repeated updates through an eigenbasis',
    code: `import numpy as np

A = np.array([[2.0, 1.0], [1.0, 2.0]])
S = np.array([[1.0, 1.0], [1.0, -1.0]])
values = np.array([3.0, 1.0])
x = np.array([2.0, 0.0])
coordinates = np.linalg.solve(S, x)
print("basis coordinates:", coordinates.tolist())
print("A S = S Lambda:", np.allclose(A @ S, S * values))
for k in range(4):
    predicted = S @ (values ** k * coordinates)
    actual = np.linalg.matrix_power(A, k) @ x
    print("step", k, predicted.tolist(), "agrees:", np.allclose(predicted, actual))

reflection = np.diag([1.0, -1.0])
diagonal = np.array([1.0, 1.0])
print("reflection squared is identity:", np.allclose(reflection @ reflection, np.eye(2)))
print("diagonal fixed after two steps:", np.allclose(reflection @ reflection @ diagonal, diagonal))
print("diagonal fixed after one step:", np.allclose(reflection @ diagonal, diagonal))`,
    expected: "basis coordinates: [1.0, 1.0]\nA S = S Lambda: True\nstep 0 [2.0, 0.0] agrees: True\nstep 1 [4.0, 2.0] agrees: True\nstep 2 [10.0, 8.0] agrees: True\nstep 3 [28.0, 26.0] agrees: True\nreflection squared is identity: True\ndiagonal fixed after two steps: True\ndiagonal fixed after one step: False"
  },
  missingDirections: {
    title: 'Distinguish repeated eigenvalues, missing directions and complex pairs',
    code: `import numpy as np

identity = np.eye(2)
shear = np.array([[1.0, 1.0], [0.0, 1.0]])
for name, matrix in [("identity", identity), ("shear", shear)]:
    eigenspace_dimension = 2 - np.linalg.matrix_rank(matrix - np.eye(2))
    print(name, "dimension for eigenvalue 1:", eigenspace_dimension)

rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
v = np.array([1.0, -1.0j])
print("complex pair A v = i v:", np.allclose(rotation @ v, 1j * v))
print("conjugate pair:", np.allclose(rotation @ v.conj(), -1j * v.conj()))
print("four quarter-turns:", np.linalg.matrix_power(rotation, 4).tolist())`,
    expected: "identity dimension for eigenvalue 1: 2\nshear dimension for eigenvalue 1: 1\ncomplex pair A v = i v: True\nconjugate pair: True\nfour quarter-turns: [[1.0, 0.0], [0.0, 1.0]]"
  },
  nonorthogonalBasis: {
    title: 'A general eigenbasis needs its inverse, not automatically its transpose',
    code: `import numpy as np

A = np.array([[2.0, 1.0], [0.0, 1.0]])
S = np.array([[1.0, -1.0], [0.0, 1.0]])
values = np.array([2.0, 1.0])
x = np.array([0.0, 1.0])
coordinates = np.linalg.solve(S, x)
print("coordinates:", coordinates.tolist())
print("correct output:", (S @ (values * coordinates)).tolist())
print("transpose-based output:", (S @ (values * (S.T @ x))).tolist())
print("basis columns orthogonal:", np.isclose(S[:, 0] @ S[:, 1], 0))
print("pair residuals:", np.linalg.norm(A @ S - S * values, axis=0).tolist())`,
    expected: "coordinates: [1.0, 1.0]\ncorrect output: [1.0, 1.0]\ntranspose-based output: [-1.0, 1.0]\nbasis columns orthogonal: False\npair residuals: [0.0, 0.0]"
  },
  pcaProjection: {
    title: 'Complete the original PCA idea with data, projections and an SVD check',
    code: `import numpy as np

observations = np.array([[-2.0, -1.0], [-1.0, -2.0], [1.0, 2.0], [2.0, 1.0]])
X = observations - observations.mean(axis=0)
C = X.T @ X / (len(X) - 1)
values, vectors = np.linalg.eigh(C)
order = np.argsort(values)[::-1]
values, vectors = values[order], vectors[:, order]
direction = vectors[:, 0]
if direction[0] < 0:  # only fixes this fixture's presentation sign
    direction = -direction
scores = X @ direction
reconstruction = np.outer(scores, direction)
_, singular_values, Vt = np.linalg.svd(X, full_matrices=False)
print("covariance:", C.round(6).tolist())
print("eigenvalues:", values.round(6).tolist())
print("scores:", scores.round(6).tolist())
print("projected points:", reconstruction.round(6).tolist())
print("retained fraction:", f"{values[0] / values.sum():.6f}")
print("squared reconstruction error:", f"{np.sum((X - reconstruction) ** 2):.6f}")
print("SVD variances agree:", np.allclose(singular_values ** 2 / (len(X) - 1), values))
print("SVD line agrees:", np.allclose(np.outer(direction, direction), np.outer(Vt[0], Vt[0])))`,
    expected: "covariance: [[3.333333, 2.666667], [2.666667, 3.333333]]\neigenvalues: [6.0, 0.666667]\nscores: [-2.12132, -2.12132, 2.12132, 2.12132]\nprojected points: [[-1.5, -1.5], [-1.5, -1.5], [1.5, 1.5], [1.5, 1.5]]\nretained fraction: 0.900000\nsquared reconstruction error: 2.000000\nSVD variances agree: True\nSVD line agrees: True"
  },
  transientGrowth: {
    title: 'A stable spectrum can still permit temporary amplification',
    code: `import numpy as np

A = np.array([[0.6, 2.0], [0.0, 0.6]])
x0 = np.array([0.0, 1.0])
print("eigenvalues:", np.linalg.eigvals(A).tolist())
for k in [0, 1, 2, 3, 10, 30]:
    actual = np.linalg.matrix_power(A, k) @ x0
    predicted = np.array([0.0, 1.0]) if k == 0 else np.array([2 * k * 0.6 ** (k - 1), 0.6 ** k])
    print("step", k, "norm:", f"{np.linalg.norm(actual):.6f}", "formula agrees:", np.allclose(actual, predicted))`,
    expected: "eigenvalues: [0.6, 0.6]\nstep 0 norm: 1.000000 formula agrees: True\nstep 1 norm: 2.088061 formula agrees: True\nstep 2 norm: 2.426850 formula agrees: True\nstep 3 norm: 2.170773 formula agrees: True\nstep 10 norm: 0.201645 formula agrees: True\nstep 30 norm: 0.000022 formula agrees: True"
  },
  twoCompartmentMixing: {
    title: 'Predict the equilibrium and the decaying imbalance of two compartments',
    code: `import numpy as np

# Each column describes where one compartment's current amount goes.
P = np.array([[0.8, 0.3], [0.2, 0.7]])
initial = np.array([1000.0, 0.0])
equilibrium = np.array([600.0, 400.0])
imbalance = np.array([400.0, -400.0])
print("column sums:", P.sum(axis=0).tolist())
print("stationary amount:", np.allclose(P @ equilibrium, equilibrium))
print("imbalance factor:", np.allclose(P @ imbalance, 0.5 * imbalance))
for k in range(4):
    actual = np.linalg.matrix_power(P, k) @ initial
    predicted = equilibrium + 0.5 ** k * imbalance
    print("step", k, actual.round(6).tolist(), "agrees:", np.allclose(actual, predicted))`,
    expected: "column sums: [1.0, 1.0]\nstationary amount: True\nimbalance factor: True\nstep 0 [1000.0, 0.0] agrees: True\nstep 1 [800.0, 200.0] agrees: True\nstep 2 [700.0, 300.0] agrees: True\nstep 3 [650.0, 350.0] agrees: True"
  },
  powerIteration: {
    title: 'Normalize a power iteration and test a misleading zero-residual start',
    code: `import numpy as np

def power_iteration(matrix, start, tolerance=1e-10, maximum_steps=100):
    if np.iscomplexobj(matrix) or np.iscomplexobj(start):
        raise ValueError("This teaching implementation uses real inputs")
    matrix = np.asarray(matrix, dtype=float)
    vector = np.asarray(start, dtype=float)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
            or vector.shape != (matrix.shape[0],)
            or not np.isfinite(matrix).all() or not np.isfinite(vector).all()
            or np.linalg.norm(vector) == 0):
        raise ValueError("Use a finite square real matrix and a nonzero matching vector")
    if (not np.isfinite(tolerance) or tolerance <= 0
            or not isinstance(maximum_steps, int) or maximum_steps < 1):
        raise ValueError("Use a positive tolerance and iteration limit")
    vector = vector / np.linalg.norm(vector)
    for step in range(1, maximum_steps + 1):
        product = matrix @ vector
        length = np.linalg.norm(product)
        if length == 0:
            return vector, 0.0, 0.0, step
        vector = product / length
        value = float(vector @ matrix @ vector)
        residual = float(np.linalg.norm(matrix @ vector - value * vector))
        if residual <= tolerance:
            return vector, value, residual, step
    raise RuntimeError("No small residual within the step budget")

A = np.array([[2.0, 1.0], [1.0, 2.0]])
for label, start in [("mixed start", [1.0, 0.0]), ("exact minor direction", [1.0, -1.0])]:
    vector, value, residual, steps = power_iteration(A, start)
    print(label, "value:", f"{value:.6f}", "residual:", f"{residual:.6f}")
print("A's largest eigenvalue:", np.linalg.eigvalsh(A)[-1])
# A small residual certifies a pair, not that it is the requested dominant pair.`,
    expected: "mixed start value: 3.000000 residual: 0.000000\nexact minor direction value: 1.000000 residual: 0.000000\nA's largest eigenvalue: 3.0"
  },
  coupledModes: {
    title: 'Separate two ideal coupled oscillation modes',
    code: `import numpy as np

# Ideal equal masses m=1 kg, spring constant k=1 N/m; fixed outer walls.
mass = 1.0
K = np.array([[2.0, -1.0], [-1.0, 2.0]])
values, modes = np.linalg.eigh(K / mass)
frequencies = np.sqrt(values)
amplitudes = np.array([0.2, 0.1])  # metres in the orthonormal mode coordinates
times = np.linspace(0.0, 2.0, 9)
positions = np.array([modes @ (amplitudes * np.cos(frequencies * t)) for t in times])
accelerations = np.array([modes @ (-values * amplitudes * np.cos(frequencies * t)) for t in times])
print("squared angular frequencies:", values.round(6).tolist())
print("angular frequencies (rad/s):", frequencies.round(6).tolist())
print("mass times acceleration equals minus K q:", np.allclose(mass * accelerations, -positions @ K.T))
print("mode columns orthonormal:", np.allclose(modes.T @ modes, np.eye(2)))`,
    expected: "squared angular frequencies: [1.0, 3.0]\nangular frequencies (rad/s): [1.0, 1.732051]\nmass times acceleration equals minus K q: True\nmode columns orthonormal: True"
  },
  independentSpectrum: {
    title: 'Check a different spectrum and an inferred rank-one approximation',
    code: `import numpy as np

A = np.array([[5.0, 2.0], [2.0, 2.0]])
q1 = np.array([2.0, 1.0]) / np.sqrt(5)
q2 = np.array([1.0, -2.0]) / np.sqrt(5)
rank_one = 6 * np.outer(q1, q1)
print("dominant pair:", np.allclose(A @ q1, 6 * q1))
print("other pair:", np.allclose(A @ q2, q2))
print("rank-one matrix:", rank_one.round(6).tolist())
print("spectral error:", f"{np.linalg.norm(A - rank_one, ord=2):.6f}")
print("Frobenius error:", f"{np.linalg.norm(A - rank_one):.6f}")
print("eigenvalue sum and trace:", 7.0, np.trace(A))`,
    expected: "dominant pair: True\nother pair: True\nrank-one matrix: [[4.8, 2.4], [2.4, 1.2]]\nspectral error: 1.000000\nFrobenius error: 1.000000\neigenvalue sum and trace: 7.0 7.0"
  }
};
