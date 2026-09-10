// Complete standalone NumPy programs with captured stdout; imports are lesson-local.
export const randomizedLinearAlgebraExamples = {
  weightedColumns: {
    title: "See a successful probe and an exact cancellation",
    code: `import numpy as np

A = np.array([[3., 0., 3.], [0., 1., 1.]])
for weights in ([1., 1., 0.], [1., 1., -1.]):
    y = A @ weights
    if np.linalg.norm(y):
        q = y / np.linalg.norm(y)
        approximation = np.outer(q, q) @ A
    else:
        approximation = np.zeros_like(A)
    print("weights:", weights, "probe:", y.tolist())
    print("squared error:", round(float(np.sum((A - approximation)**2)), 6))
print("A is zero:", bool(np.all(A == 0)))`,
    expected: "weights: [1.0, 1.0, 0.0] probe: [3.0, 1.0]\nsquared error: 1.8\nweights: [1.0, 1.0, -1.0] probe: [0.0, 0.0]\nsquared error: 20.0\nA is zero: False"
  },
  gaussianBaseline: {
    title: "Run the complete 100-by-40 randomized SVD",
    code: `import numpy as np

def orth(Y):
    # Factor only the skinny sketch; drop numerical null directions.
    U, s, _ = np.linalg.svd(Y, full_matrices=False)
    tolerance = np.finfo(float).eps * max(Y.shape) * (s[0] if s.size else 0.)
    return U[:, s > tolerance]

def randomized_svd(A, k, p, q, rng):
    if np.iscomplexobj(A):
        raise ValueError("this teaching implementation accepts real matrices")
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or not A.size or not np.isfinite(A).all():
        raise ValueError("A must be a nonempty finite real matrix")
    if not all(isinstance(v, int) for v in (k, p, q)):
        raise ValueError("k, p and q must be integers")
    if not 1 <= k <= min(A.shape) or p < 0 or q < 0:
        raise ValueError("require 1 <= k <= min(shape), p >= 0, q >= 0")
    width = min(k + p, min(A.shape))
    Q = orth(A @ rng.normal(size=(A.shape[1], width)))
    for _ in range(q):
        if not Q.shape[1]:
            break
        Z = orth(A.T @ Q)
        Q = orth(A @ Z)
    B = Q.T @ A
    small_U, s, Vt = np.linalg.svd(B, full_matrices=False)
    return (Q @ small_U[:, :k]), s[:k], Vt[:k], Q, B

rng = np.random.default_rng(7)
A = rng.normal(size=(100, 40))
U, s, Vt, Q, B = randomized_svd(A, 5, 4, 0, rng)
approximation = (U * s) @ Vt
exact_s = np.linalg.svd(A, compute_uv=False)
print("Q, B, approximation:", Q.shape, B.shape, approximation.shape)
print("relative error:", f"{np.linalg.norm(A - approximation) / np.linalg.norm(A):.6f}")
print("best rank-5 relative error:", f"{np.linalg.norm(exact_s[5:]) / np.linalg.norm(exact_s):.6f}")
print("orthonormal U:", np.allclose(U.T @ U, np.eye(5)))`,
    expected: "Q, B, approximation: (100, 9) (9, 40) (100, 40)\nrelative error: 0.897937\nbest rank-5 relative error: 0.852888\northonormal U: True"
  },
  spectrumBudget: {
    title: "Separate rank limits from the quality of the sketch",
    code: `import numpy as np

def orth(Y):
    # Factor only the skinny sketch; drop numerical null directions.
    U, s, _ = np.linalg.svd(Y, full_matrices=False)
    tolerance = np.finfo(float).eps * max(Y.shape) * (s[0] if s.size else 0.)
    return U[:, s > tolerance]

def randomized_svd(A, k, p, q, rng):
    if np.iscomplexobj(A):
        raise ValueError("this teaching implementation accepts real matrices")
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or not A.size or not np.isfinite(A).all():
        raise ValueError("A must be a nonempty finite real matrix")
    if not all(isinstance(v, int) for v in (k, p, q)):
        raise ValueError("k, p and q must be integers")
    if not 1 <= k <= min(A.shape) or p < 0 or q < 0:
        raise ValueError("require 1 <= k <= min(shape), p >= 0, q >= 0")
    width = min(k + p, min(A.shape))
    Q = orth(A @ rng.normal(size=(A.shape[1], width)))
    for _ in range(q):
        if not Q.shape[1]:
            break
        Z = orth(A.T @ Q)
        Q = orth(A @ Z)
    B = Q.T @ A
    small_U, s, Vt = np.linalg.svd(B, full_matrices=False)
    return (Q @ small_U[:, :k]), s[:k], Vt[:k], Q, B

rng = np.random.default_rng(19)
left, _ = np.linalg.qr(rng.normal(size=(30, 12)))
right, _ = np.linalg.qr(rng.normal(size=(12, 12)))
spectra = {
    "fast": np.array([20., 8., 3., 1., .5, .2, .1, .05, .02, .01, .005, .002]),
    "flat": np.ones(12),
}
for name, spectrum in spectra.items():
    A = (left * spectrum) @ right.T
    best = np.linalg.norm(spectrum[3:]) / np.linalg.norm(spectrum)
    print(name, "best rank-3:", f"{best:.6f}")
    for p, q in [(0, 0), (3, 0), (3, 1), (3, 2)]:
        U, s, Vt, _, _ = randomized_svd(A, 3, p, q, np.random.default_rng(7))
        relative = np.linalg.norm(A - (U * s) @ Vt) / np.linalg.norm(A)
        print("p, q, passes:", p, q, 2 + 2*q, "error:", f"{relative:.6f}")`,
    expected: "fast best rank-3: 0.052414\np, q, passes: 0 0 2 error: 0.136814\np, q, passes: 3 0 2 error: 0.053323\np, q, passes: 3 1 4 error: 0.052414\np, q, passes: 3 2 6 error: 0.052414\nflat best rank-3: 0.866025\np, q, passes: 0 0 2 error: 0.866025\np, q, passes: 3 0 2 error: 0.866025\np, q, passes: 3 1 4 error: 0.866025\np, q, passes: 3 2 6 error: 0.866025"
  },
  errorDecomposition: {
    title: "Account for both sources of reconstruction error",
    code: `import numpy as np

def orth(Y):
    # Factor only the skinny sketch; drop numerical null directions.
    U, s, _ = np.linalg.svd(Y, full_matrices=False)
    tolerance = np.finfo(float).eps * max(Y.shape) * (s[0] if s.size else 0.)
    return U[:, s > tolerance]

def randomized_svd(A, k, p, q, rng):
    if np.iscomplexobj(A):
        raise ValueError("this teaching implementation accepts real matrices")
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or not A.size or not np.isfinite(A).all():
        raise ValueError("A must be a nonempty finite real matrix")
    if not all(isinstance(v, int) for v in (k, p, q)):
        raise ValueError("k, p and q must be integers")
    if not 1 <= k <= min(A.shape) or p < 0 or q < 0:
        raise ValueError("require 1 <= k <= min(shape), p >= 0, q >= 0")
    width = min(k + p, min(A.shape))
    Q = orth(A @ rng.normal(size=(A.shape[1], width)))
    for _ in range(q):
        if not Q.shape[1]:
            break
        Z = orth(A.T @ Q)
        Q = orth(A @ Z)
    B = Q.T @ A
    small_U, s, Vt = np.linalg.svd(B, full_matrices=False)
    return (Q @ small_U[:, :k]), s[:k], Vt[:k], Q, B

A = np.diag([9., 4., 2., 1., .5, .2])
U, s, Vt, Q, B = randomized_svd(A, 2, 1, 0, np.random.default_rng(13))
approximation = (U * s) @ Vt
projected = Q @ B
range_error = np.sum((A - projected)**2)
truncation_error = np.sum((projected - approximation)**2)
total_error = np.sum((A - approximation)**2)
print("range squared:", f"{range_error:.6f}")
print("truncation squared:", f"{truncation_error:.6f}")
print("total squared:", f"{total_error:.6f}")
print("Pythagorean identity:", np.isclose(range_error + truncation_error, total_error))
for name, matrix in [("rank one", np.ones((5, 3))), ("zero", np.zeros((5, 3)))]:
    U, s, Vt, Q, _ = randomized_svd(matrix, 2, 1, 2, np.random.default_rng(2))
    print(name, "observed directions:", Q.shape[1], "reconstructed:", np.allclose(matrix, (U*s) @ Vt))`,
    expected: "range squared: 1.892159\ntruncation squared: 3.819055\ntotal squared: 5.711214\nPythagorean identity: True\nrank one observed directions: 1 reconstructed: True\nzero observed directions: 0 reconstructed: True"
  },
  blockedProducts: {
    title: "Form the same two-pass sketch from row blocks",
    code: `import numpy as np

rng = np.random.default_rng(23)
A = rng.normal(size=(20, 12))
omega = rng.normal(size=(12, 5))
# A is in memory only to verify this small demonstration.
# A real reader must supply the same rows again on pass two.
Y = np.vstack([A[start:start+4] @ omega for start in range(0, 20, 4)])
Q, _ = np.linalg.qr(Y, mode="reduced")
B = np.zeros((5, 12))
for start in range(0, 20, 4):
    B += Q[start:start+4].T @ A[start:start+4]
print("blocked Y agrees:", np.allclose(Y, A @ omega))
print("blocked B agrees:", np.allclose(B, Q.T @ A))
print("stored Q and B:", Q.shape, B.shape)
print("input passes:", 2)`,
    expected: "blocked Y agrees: True\nblocked B agrees: True\nstored Q and B: (20, 5) (5, 12)\ninput passes: 2"
  },
  rowSketches: {
    title: "Judge the fitted line on all original observations",
    code: `import numpy as np

x = np.array([0., 1., 2., 3., 4., 5., 6., 20.])
A = np.column_stack([np.ones(8), x])
b = 2 + .5*x + np.array([0., .2, -.1, .4, -.3, .1, -.2, 3.])
rng = np.random.default_rng(7)
sketches = {
    "all rows": np.eye(8),
    "first four": np.eye(8)[:4],
    "two endpoints": np.eye(8)[[0, 7]],
    "Gaussian four": rng.normal(size=(4, 8)) / np.sqrt(4),
}
for name, S in sketches.items():
    coefficients, _, rank, _ = np.linalg.lstsq(S @ A, S @ b, rcond=None)
    print(name, "rank:", rank, "coefficients:", coefficients.round(6).tolist())
    print("sketch SSE:", f"{np.linalg.norm(S @ (A @ coefficients - b))**2:.6f}",
          "original SSE:", f"{np.linalg.norm(A @ coefficients - b)**2:.6f}")`,
    expected: "all rows rank: 2 coefficients: [1.595372, 0.654562]\nsketch SSE: 1.438843 original SSE: 1.438843\nfirst four rank: 2 coefficients: [1.99, 0.59]\nsketch SSE: 0.107000 original SSE: 2.642100\ntwo endpoints rank: 2 coefficients: [2.0, 0.65]\nsketch SSE: 0.000000 original SSE: 2.607500\nGaussian four rank: 2 coefficients: [1.431522, 0.669254]\nsketch SSE: 0.063598 original SSE: 1.562205"
  },
  leverageSampling: {
    title: "Rescale nonuniform row samples so their Gram matrix is unbiased",
    code: `import numpy as np

x = np.array([0., 1., 2., 3., 4., 5., 6., 20.])
A = np.column_stack([np.ones(8), x])
b = 2 + .5*x + np.array([0., .2, -.1, .4, -.3, .1, -.2, 3.])
Q, _ = np.linalg.qr(A, mode="reduced")
leverage = np.sum(Q**2, axis=1)
probabilities = leverage / Q.shape[1]
print("leverage sums to column rank:", round(float(leverage.sum()), 6))
print("last-row sampling probability:", f"{probabilities[-1]:.6f}")
# Enumerate the expectation for one independently sampled, rescaled row.
expected_gram = sum(probabilities[i] * np.outer(A[i], A[i]) / probabilities[i] for i in range(8))
print("expected Gram equals original:", np.allclose(expected_gram, A.T @ A))
sample_count = 6
indices = np.random.default_rng(11).choice(8, size=sample_count, replace=True, p=probabilities)
weights = 1 / np.sqrt(sample_count * probabilities[indices])
coefficients, _, rank, _ = np.linalg.lstsq(A[indices] * weights[:, None], b[indices] * weights, rcond=None)
print("sampled indices:", indices.tolist(), "rank:", rank)
print("original SSE:", f"{np.linalg.norm(A @ coefficients - b)**2:.6f}")`,
    expected: "leverage sums to column rank: 2.0\nlast-row sampling probability: 0.456386\nexpected Gram equals original: True\nsampled indices: [1, 6, 7, 0, 1, 7] rank: 2\noriginal SSE: 1.688986"
  },
  embeddingCheck: {
    title: "Test every direction in a known small residual subspace",
    code: `import numpy as np

rng = np.random.default_rng(5)
A = rng.normal(size=(100, 2))
b = A @ np.array([2., -1.]) + rng.normal(size=100)
Q, _ = np.linalg.qr(np.column_stack([A, b]), mode="reduced")
S = rng.normal(size=(40, 100)) / np.sqrt(40)
stretches_squared = np.linalg.svd(S @ Q, compute_uv=False)**2
epsilon = np.max(np.abs(stretches_squared - 1))
exact = np.linalg.lstsq(A, b, rcond=None)[0]
sketched = np.linalg.lstsq(S @ A, S @ b, rcond=None)[0]
ratio = np.linalg.norm(A @ sketched - b) / np.linalg.norm(A @ exact - b)
print("squared stretches:", np.sort(stretches_squared).round(6).tolist())
print("actual residual ratio:", f"{ratio:.6f}")
if epsilon < 1:
    bound = np.sqrt((1 + epsilon) / (1 - epsilon))
    print("verified-subspace bound:", f"{bound:.6f}", "holds:", bool(ratio <= bound))
else:
    print("this epsilon supplies no positive lower bound")
print("checking a subspace costs work; this is an oracle, not a free certificate")`,
    expected: "squared stretches: [0.486929, 0.74155, 1.146711]\nactual residual ratio: 1.022000\nverified-subspace bound: 1.762776 holds: True\nchecking a subspace costs work; this is an oracle, not a free certificate"
  },
  traceEstimation: {
    title: "Derive an unbiased trace estimate by enumerating the signs",
    code: `import itertools
import numpy as np

signs = np.array(list(itertools.product([-1., 1.], repeat=2)))
for name, A in [("diagonal", np.diag([4., 2.])),
                ("coupled", np.array([[4., 1.], [1., 2.]])),
                ("zero trace", np.array([[1., 2.], [2., -1.]]))]:
    values = np.array([z @ A @ z for z in signs])
    print(name, "all values:", values.tolist())
    print("expectation:", values.mean(), "trace:", np.trace(A), "variance:", values.var())
probes = np.random.default_rng(7).choice([-1., 1.], size=(8, 2))
A = np.array([[4., 1.], [1., 2.]])
values = np.array([z @ A @ z for z in probes])
print("running means:", (values.cumsum() / np.arange(1, 9)).round(3).tolist())`,
    expected: "diagonal all values: [6.0, 6.0, 6.0, 6.0]\nexpectation: 6.0 trace: 6.0 variance: 0.0\ncoupled all values: [8.0, 4.0, 4.0, 8.0]\nexpectation: 6.0 trace: 6.0 variance: 4.0\nzero trace all values: [4.0, -4.0, -4.0, 4.0]\nexpectation: 0.0 trace: 0.0 variance: 16.0\nrunning means: [8.0, 8.0, 8.0, 7.0, 7.2, 6.667, 6.286, 6.0]"
  },
  freshResidualProbes: {
    title: "Why validation must use fresh probes",
    code: `import itertools
import numpy as np

A = np.diag([5., 2., 1.])
training = np.array([1., 1., 1.])
q = A @ training
q /= np.linalg.norm(q)
R = A - np.outer(q, q) @ A
print("training-probe squared residual:", round(float(np.linalg.norm(R @ training)**2), 6))
signs = np.array(list(itertools.product([-1., 1.], repeat=3)))
fresh_values = np.array([np.linalg.norm(R @ z)**2 for z in signs])
print("all independent-sign expectation:", f"{fresh_values.mean():.6f}")
print("actual Frobenius squared error:", f"{np.sum(R**2):.6f}")
print("individual probe values:", fresh_values.round(3).tolist())`,
    expected: "training-probe squared residual: 0.0\nall independent-sign expectation: 8.600000\nactual Frobenius squared error: 8.600000\nindividual probe values: [0.0, 3.867, 13.867, 16.667, 16.667, 13.867, 3.867, 0.0]"
  },
  latentStructurePractice: {
    title: "Check a new three-factor signal plus noise",
    code: `import numpy as np

def orth(Y):
    # Factor only the skinny sketch; drop numerical null directions.
    U, s, _ = np.linalg.svd(Y, full_matrices=False)
    tolerance = np.finfo(float).eps * max(Y.shape) * (s[0] if s.size else 0.)
    return U[:, s > tolerance]

def randomized_svd(A, k, p, q, rng):
    if np.iscomplexobj(A):
        raise ValueError("this teaching implementation accepts real matrices")
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or not A.size or not np.isfinite(A).all():
        raise ValueError("A must be a nonempty finite real matrix")
    if not all(isinstance(v, int) for v in (k, p, q)):
        raise ValueError("k, p and q must be integers")
    if not 1 <= k <= min(A.shape) or p < 0 or q < 0:
        raise ValueError("require 1 <= k <= min(shape), p >= 0, q >= 0")
    width = min(k + p, min(A.shape))
    Q = orth(A @ rng.normal(size=(A.shape[1], width)))
    for _ in range(q):
        if not Q.shape[1]:
            break
        Z = orth(A.T @ Q)
        Q = orth(A @ Z)
    B = Q.T @ A
    small_U, s, Vt = np.linalg.svd(B, full_matrices=False)
    return (Q @ small_U[:, :k]), s[:k], Vt[:k], Q, B

rng = np.random.default_rng(41)
signal = rng.normal(size=(80, 3)) @ rng.normal(size=(3, 30))
A = signal + .05 * rng.normal(size=(80, 30))
exact_s = np.linalg.svd(A, compute_uv=False)
print("best rank-3 relative error:", f"{np.linalg.norm(exact_s[3:]) / np.linalg.norm(A):.6f}")
for p, q in [(0, 0), (5, 0), (5, 1)]:
    U, s, Vt, _, _ = randomized_svd(A, 3, p, q, np.random.default_rng(9))
    error = np.linalg.norm(A - (U*s) @ Vt) / np.linalg.norm(A)
    print("p, q:", p, q, "relative error:", f"{error:.6f}")
print("small observed singular directions may still contain meaningful information")`,
    expected: "best rank-3 relative error: 0.029018\np, q: 0 0 relative error: 0.507439\np, q: 5 0 relative error: 0.036773\np, q: 5 1 relative error: 0.029018\nsmall observed singular directions may still contain meaningful information"
  }
};
