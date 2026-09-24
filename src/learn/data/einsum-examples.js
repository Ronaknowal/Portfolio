export const einsumExamples = {
  patterns: {
    title: 'Read the original small arrays through explicit outputs',
    code: `import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print("matrix product:", np.einsum("ik,kj->ij", A, B).tolist())
print("diagonal:", np.einsum("ii->i", A).tolist())
print("trace:", int(np.einsum("ii->", A)))
print("column sums:", np.einsum("ij->j", A).tolist())
print("outer:", np.einsum("i,j->ij", [1, 2], [3, 4]).tolist())
print("elementwise:", np.einsum("ij,ij->ij", A, B).tolist())
print("implicit ji:", np.einsum("ji", A).tolist())
print("explicit ji->ji:", np.einsum("ji->ji", A).tolist())`,
    expected: "matrix product: [[19, 22], [43, 50]]\ndiagonal: [1, 4]\ntrace: 5\ncolumn sums: [4, 6]\nouter: [[3, 4], [6, 8]]\nelementwise: [[5, 12], [21, 32]]\nimplicit ji: [[1, 3], [2, 4]]\nexplicit ji->ji: [[1, 2], [3, 4]]"
  },
  loops: {
    title: 'Check a non-square contraction with ordinary loops',
    code: `import numpy as np

A = np.array([[1, 2, 3], [-1, 0, 2]])
B = np.array([[2, 1], [0, 4], [1, -1]])
C = np.zeros((A.shape[0], B.shape[1]), dtype=int)
for i in range(A.shape[0]):
    for j in range(B.shape[1]):
        for k in range(A.shape[1]):
            C[i, j] += A[i, k] * B[k, j]
result = np.einsum("ik,kj->ij", A, B)
print("result:", result.tolist())
print("loops agree:", np.array_equal(result, C))
print("matrix product agrees:", np.array_equal(result, A @ B))
print("reordered result:", np.einsum("ik,kj->ji", A, B).tolist())
print("selected products:", (A[1, :] * B[:, 1]).tolist())`,
    expected: "result: [[5, 6], [0, -3]]\nloops agree: True\nmatrix product agrees: True\nreordered result: [[5, 0], [6, -3]]\nselected products: [-1, 0, -2]"
  },
  batches: {
    title: 'Distinguish paired batches, all pairs and broadcast groups',
    code: `import numpy as np

X = np.array([[1, 2], [3, 4]])
Y = np.array([[10, 20], [30, 40]])
paired = np.einsum("bi,bi->b", X, Y)
all_pairs = np.einsum("bi,ci->bc", X, Y)
print("paired:", paired.tolist())
print("all pairs:", all_pairs.tolist())
print("same-batch diagonal:", np.diag(all_pairs).tolist())
# Two separate broadcast groups: shape (2,1,3) and (4,3).
left = np.arange(6).reshape(2, 1, 3)
right = np.arange(12).reshape(4, 3)
combined = np.einsum("...i,...i->...", left, right)
print("broadcast result shape:", combined.shape)
print("broadcast reference agrees:", np.array_equal(combined, (left * right).sum(axis=-1)))
# An explicitly labelled singleton axis can also expand across operands.
print("singleton b:", np.einsum("bi,bi->b", np.ones((1, 2), dtype=int), Y).tolist())`,
    expected: "paired: [50, 250]\nall pairs: [[50, 110], [110, 250]]\nsame-batch diagonal: [50, 250]\nbroadcast result shape: (2, 4)\nbroadcast reference agrees: True\nsingleton b: [30, 70]"
  },
  covariance: {
    title: 'Compute a covariance for each batch and check the estimator',
    code: `import numpy as np

# Axes: batch, observation, feature. Both features use the same arbitrary units.
X = np.array([[[1., 2.], [3., 0.], [5., 4.]],
              [[2., 1.], [4., 3.], [6., 5.]]])
n = X.shape[1]
if n < 2:
    raise ValueError("Sample covariance needs at least two observations")
centered = X - X.mean(axis=1, keepdims=True)
covariance = np.einsum("bni,bnj->bij", centered, centered) / (n - 1)
reference = np.stack([np.cov(batch, rowvar=False, ddof=1) for batch in X])
print("means:", X.mean(axis=1).tolist())
print("covariances:", covariance.tolist())
print("library agrees:", np.allclose(covariance, reference))
print("uncentered second moment:", (np.einsum("bni,bnj->bij", X, X) / n)[0].round(6).tolist())
print("translation invariant:", np.allclose(
    covariance, np.stack([np.cov(batch + [100, -30], rowvar=False) for batch in X])))`,
    expected: "means: [[3.0, 2.0], [4.0, 3.0]]\ncovariances: [[[4.0, 2.0], [2.0, 4.0]], [[4.0, 4.0], [4.0, 4.0]]]\nlibrary agrees: True\nuncentered second moment: [[11.666667, 7.333333], [7.333333, 6.666667]]\ntranslation invariant: True"
  },
  attention: {
    title: 'Run masked scaled dot-product attention from complete inputs',
    code: `import numpy as np

def scaled_attention(q, k, v, allowed):
    q, k, v = [np.asarray(a, dtype=float) for a in (q, k, v)]
    allowed = np.asarray(allowed, dtype=bool)
    if (q.ndim != 3 or k.ndim != 3 or v.ndim != 3
            or q.shape[0] != k.shape[0] or k.shape[:2] != v.shape[:2]
            or q.shape[2] != k.shape[2] or min(q.shape + k.shape + v.shape) < 1):
        raise ValueError("Use matching batch/key/query-feature dimensions")
    if not all(np.isfinite(a).all() for a in (q, k, v)):
        raise ValueError("Use finite inputs")
    if allowed.shape != (q.shape[0], q.shape[1], k.shape[1]) or not allowed.any(axis=-1).all():
        raise ValueError("Give each query at least one allowed key")
    scores = np.einsum("btd,bsd->bts", q, k) / np.sqrt(q.shape[-1])
    masked = np.where(allowed, scores, -np.inf)
    shifted = masked - masked.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    weights = exponentials / exponentials.sum(axis=-1, keepdims=True)
    context = np.einsum("bts,bsv->btv", weights, v)
    return weights, context

Q = np.array([[[1., 0.], [0., 1.]]])
K = np.array([[[1., 0.], [0., 1.], [1., 1.]]])
V = np.array([[[2., 0.], [0., 4.], [2., 2.]]])
allowed = np.array([[[True, False, True], [True, True, True]]])
weights, context = scaled_attention(Q, K, V, allowed)
print("weights:", weights.round(6).tolist())
print("context:", context.round(6).tolist())
print("row sums:", weights.sum(axis=-1).tolist())
print("matrix reference:", np.allclose(context, weights @ V))
try:
    scaled_attention(Q, K, V, np.zeros_like(allowed))
except ValueError as error:
    print("all masked:", str(error))`,
    expected: "weights: [[[0.5, 0.0, 0.5], [0.197776, 0.401112, 0.401112]]]\ncontext: [[[2.0, 1.0], [1.197776, 2.406673]]]\nrow sums: [[1.0, 1.0]]\nmatrix reference: True\nall masked: Give each query at least one allowed key"
  },
  order: {
    title: 'Choose a contraction path and verify the same result',
    code: `import numpy as np

rng = np.random.default_rng(7)
A = rng.integers(-2, 3, size=(5, 40))
B = rng.integers(-2, 3, size=(40, 2))
C = rng.integers(-2, 3, size=(2, 30))
left = (A @ B) @ C
right = A @ (B @ C)
path, report = np.einsum_path("ab,bc,cd->ad", A, B, C, optimize="greedy")
result = np.einsum("ab,bc,cd->ad", A, B, C, optimize=path)
print("shape:", result.shape)
print("both matrix orders agree:", np.array_equal(left, right))
print("planned einsum agrees:", np.array_equal(result, left))
a, b, c, d = 5, 40, 2, 30
print("left multiplications:", a*b*c + a*c*d)
print("right multiplications:", b*c*d + a*b*d)
print("first intermediate elements:", a*c, b*d)
# Inspect report locally; its formatting/path details may vary by NumPy version.
print("plan is reusable:", path[0] == "einsum_path")`,
    expected: "shape: (5, 30)\nboth matrix orders agree: True\nplanned einsum agrees: True\nleft multiplications: 700\nright multiplications: 8400\nfirst intermediate elements: 10 1200\nplan is reusable: True"
  },
  basis: {
    title: 'Check the same vector, measurement and length in a sheared basis',
    code: `import numpy as np

S = np.array([[1., 1.], [0., 1.]])  # New basis columns in old coordinates.
v = np.array([3., 2.])
functional = np.array([2., -1.])
A = np.array([[2., 1.], [0., 1.]])
new_v = np.linalg.solve(S, v)
new_f = S.T @ functional
new_A = np.linalg.solve(S, A @ S)
new_G = S.T @ S  # Old coordinates were orthonormal.
print("new vector coordinates:", new_v.tolist())
print("reconstructed vector:", (S @ new_v).tolist())
print("new functional:", new_f.tolist())
print("measurements:", float(functional @ v), float(new_f @ new_v))
print("new map:", new_A.tolist())
print("same mapped vector:", np.allclose(S @ (new_A @ new_v), A @ v))
print("naive coordinate squares:", float(new_v @ new_v))
print("metric:", new_G.tolist())
print("true squared lengths:", float(v @ v), float(np.einsum("i,ij,j->", new_v, new_G, new_v)))`,
    expected: "new vector coordinates: [1.0, 2.0]\nreconstructed vector: [3.0, 2.0]\nnew functional: [2.0, 1.0]\nmeasurements: 4.0 4.0\nnew map: [[2.0, 2.0], [0.0, 1.0]]\nsame mapped vector: True\nnaive coordinate squares: 5.0\nmetric: [[1.0, 1.0], [1.0, 2.0]]\ntrue squared lengths: 13.0 13.0"
  },
  energy: {
    title: 'Sum kinetic energy over bodies while keeping time',
    code: `import numpy as np

# mass: kilograms; velocity axes: body, time, Cartesian component, in m/s.
mass = np.array([2., 3.])
velocity = np.array([[[1., 0.], [0., 2.], [1., 1.]],
                     [[0., 2.], [1., 0.], [2., 0.]]])
energy = 0.5 * np.einsum("n,ntd,ntd->t", mass, velocity, velocity)
reference = np.zeros(velocity.shape[1])
for n in range(velocity.shape[0]):
    for t in range(velocity.shape[1]):
        reference[t] += 0.5 * mass[n] * np.dot(velocity[n, t], velocity[n, t])
print("energy at each time, joules:", energy.tolist())
print("body/time loop agrees:", np.allclose(energy, reference))
print("retained axis size:", energy.shape)`,
    expected: "energy at each time, joules: [7.0, 5.5, 8.0]\nbody/time loop agrees: True\nretained axis size: (3,)"
  },
  numerical: {
    title: 'Check aliasing and complex products explicitly',
    code: `import numpy as np

A = np.array([[1., 2.], [3., 4.]])
diagonal = np.einsum("ii->i", A)
diagonal[0] = 9
print("diagonal is a view:", np.shares_memory(A, diagonal))
print("A after diagonal edit:", A.tolist())
independent = np.einsum("ij->ji", A).copy()
independent[0, 0] = -1
print("copy leaves A alone:", float(A[0, 0]))
z = np.array([1 + 2j, 3 - 1j])
print("bilinear sum:", np.einsum("i,i->", z, z))
print("Hermitian squared length:", np.einsum("i,i->", z.conj(), z))
print("vdot agrees:", np.allclose(np.einsum("i,i->", z.conj(), z), np.vdot(z, z)))`,
    expected: "diagonal is a view: True\nA after diagonal edit: [[9.0, 2.0], [3.0, 4.0]]\ncopy leaves A alone: 9.0\nbilinear sum: (5-2j)\nHermitian squared length: (15+0j)\nvdot agrees: True"
  },
  higher: {
    title: 'Contract two axes without confusing them with the output axes',
    code: `import numpy as np

A = np.arange(12).reshape(2, 3, 2)
B = (np.arange(12) - 3).reshape(3, 2, 2)
result = np.einsum("ijk,jil->kl", A, B)
reference = np.tensordot(A, B, axes=([0, 1], [1, 0]))
print("result:", result.tolist())
print("output shape:", result.shape)
print("tensordot agrees:", np.array_equal(result, reference))
print("one element via explicit sum:", sum(
    A[i, j, 1] * B[j, i, 0] for i in range(2) for j in range(3)))`,
    expected: "result: [[110, 140], [122, 158]]\noutput shape: (2, 2)\ntensordot agrees: True\none element via explicit sum: 122"
  },
};

