// Independently runnable numpyReference examples; verified code/output pairs.
export const numpyReferenceExamples = {
numpyCreate: {
    code: `import numpy as np

X = np.array([[18, 40], [24, 50], [30, 60]], dtype=np.float64)
print(X.tolist())
print("shape:", X.shape, "ndim:", X.ndim, "size:", X.size)
print("dtype:", X.dtype, "bytes:", X.nbytes)
print(np.arange(0, 6, 2).tolist())
print(np.linspace(0, 1, 3).tolist())
print(np.zeros((1, 2)).tolist(), np.full((1, 2), 7).tolist())
print(np.eye(2).tolist())`,
    output: `[[18.0, 40.0], [24.0, 50.0], [30.0, 60.0]]
shape: (3, 2) ndim: 2 size: 6
dtype: float64 bytes: 48
[0, 2, 4]
[0.0, 0.5, 1.0]
[[0.0, 0.0]] [[7, 7]]
[[1.0, 0.0], [0.0, 1.0]]`,
  },
numpyDtype: {
    code: `import numpy as np

small = np.array([120], dtype=np.int8)
print((small + small).tolist())
print((small.astype(np.int64) + small).tolist())
values = np.array([1, 2], dtype=np.int64)
try:
    values /= 2
except TypeError:
    print("in-place float result cannot be stored as int64")
print((values / 2).tolist())
print(np.asarray(values) is values)`,
    output: `[-16]
[240]
in-place float result cannot be stored as int64
[0.5, 1.0]
True`,
  },
numpyIndex: {
    code: `import numpy as np

X = np.array([[18, 40], [24, 50], [30, 60]])
column = X[:, 0]
picked = X[[0, 2]]
column[0] = 19
picked[0, 0] = -1
print(X.tolist())
print(np.shares_memory(X, column), np.shares_memory(X, picked))
print(X[:, 0].shape, X[:, :1].shape)
print(X[[0, 2], [0, 1]].tolist())
print(X[np.ix_([0, 2], [0, 1])].tolist())
X[[0, 2], 0] = 0
print(X[:, 0].tolist())`,
    output: `[[19, 40], [24, 50], [30, 60]]
True False
(3,) (3, 1)
[19, 60]
[[19, 40], [30, 60]]
[0, 24, 0]`,
  },
numpyMask: {
    code: `import numpy as np

X = np.array([[18, 40], [24, 50], [30, 60]])
mask = (X[:, 0] >= 24) & (X[:, 1] < 60)
print(mask.tolist())
print(X[mask].tolist())
print(X[X >= 30].tolist())
print(np.where(X[:, 0] >= 24, "warm", "cool").tolist())
print(bool(np.any(mask)), bool(np.all(mask)))
try:
    X[np.array([[True], [False], [True]])]
except IndexError:
    print("boolean indexing does not broadcast a (3, 1) mask to (3, 2)")`,
    output: `[False, True, False]
[[24, 50]]
[40, 50, 30, 60]
['cool', 'warm', 'warm']
True False
boolean indexing does not broadcast a (3, 1) mask to (3, 2)`,
  },
numpyShapes: {
    code: `import numpy as np

a = np.arange(6).reshape(3, 2)
print(a.tolist())
print(a.T.shape, a.reshape(2, 3).tolist())
v = np.array([1, 2, 3])
print(v.T.shape, v[:, None].shape, v[None, :].shape)
print(np.stack([v, v], axis=0).shape)
print(np.concatenate([v, v]).shape)
print([part.tolist() for part in np.array_split(v, 2)])
print(np.moveaxis(np.zeros((2, 3, 4)), 0, -1).shape)
print(np.squeeze(v[:, None], axis=1).shape)`,
    output: `[[0, 1], [2, 3], [4, 5]]
(2, 3) [[0, 1, 2], [3, 4, 5]]
(3,) (3, 1) (1, 3)
(2, 3)
(6,)
[[1, 2], [3]]
(3, 4, 2)
(3,)`,
  },
numpyBroadcast: {
    code: `import numpy as np

X = np.array([[18, 40], [24, 50], [30, 60]])
print((X - np.array([20, 50])).tolist())
print((X - np.array([1, 2, 3])[:, None]).tolist())
try:
    X - np.array([1, 2, 3])
except ValueError:
    print("(3, 2) and (3,) are incompatible")
column = np.array([1, 2, 3])[:, None]
row = np.array([10, 20, 30])
print((column + row).shape)`,
    output: `[[-2, -10], [4, 0], [10, 10]]
[[17, 39], [22, 48], [27, 57]]
(3, 2) and (3,) are incompatible
(3, 3)`,
  },
numpyReduce: {
    code: `import numpy as np

X = np.array([[18., 40.], [24., 50.], [30., 60.]])
print(X.mean(axis=0).tolist())
print(X.mean(axis=1).tolist())
print(X.mean(axis=0, keepdims=True).shape)
print(X.sum(), X.argmax(axis=0).tolist())
v = np.array([1., 2., 3.])
print(round(float(v.var(ddof=0)), 3), float(v.var(ddof=1)))
print(np.cumsum(v).tolist(), np.diff(v).tolist())
print(np.gradient(v, 0.5).tolist())`,
    output: `[24.0, 50.0]
[29.0, 37.0, 45.0]
(1, 2)
222.0 [2, 2]
0.667 1.0
[1.0, 3.0, 6.0] [1.0, 1.0]
[2.0, 2.0, 2.0]`,
  },
numpyMissing: {
    code: `import numpy as np

values = np.array([1., np.nan, 3.])
print(np.isfinite(values).tolist())
print(np.isnan(values).tolist())
print(float(np.nanmean(values)))
numerator = np.array([10., 20., 30.])
denominator = np.array([2., 0., 5.])
result = np.full_like(numerator, np.nan)
np.divide(numerator, denominator, out=result, where=denominator != 0)
print(result.tolist())
print(np.clip(np.array([-2., 0., 5.]), 0, 3).tolist())
with np.errstate(divide="raise"):
    try:
        np.log(np.array([0.]))
    except FloatingPointError:
        print("log(0) is not a finite real value")`,
    output: `[True, False, True]
[False, True, False]
2.0
[5.0, nan, 6.0]
[0.0, 0.0, 3.0]
log(0) is not a finite real value`,
  },
numpySort: {
    code: `import numpy as np

scores = np.array([0.3, 0.9, 0.6])
order = np.argsort(scores)
print(order.tolist(), scores[order].tolist())
print(order[-2:][::-1].tolist())
print(np.unique([2, 1, 2], return_counts=True)[1].tolist())
print(np.isin([1, 2, 3], [1, 3]).tolist())
print(np.nonzero(np.array([0, 7, 0, 8]))[0].tolist())
print(np.partition([4, 1, 3, 2], 2)[2])`,
    output: `[0, 2, 1] [0.3, 0.6, 0.9]
[1, 2]
[1, 2]
[True, False, True]
[1, 3]
3`,
  },
numpyAlgebra: {
    code: `import numpy as np

X = np.array([[1., 2.], [3., 4.]])
w = np.array([2., -1.])
print((X * w).tolist())
print((X @ w).tolist())
print((X @ X.T).tolist())
print(np.einsum("ij,j->i", X, w).tolist())
A = np.array([[2., 0.], [0., 4.]])
b = np.array([6., 8.])
solution = np.linalg.solve(A, b)
print(solution.tolist(), bool(np.allclose(A @ solution, b)))
U, s, Vh = np.linalg.svd(A, full_matrices=False)
print("shapes:", U.shape, s.shape, Vh.shape)
print("singular values:", s.tolist())
print("reconstructed:", bool(np.allclose((U * s) @ Vh, A)))
rank_one = (U[:, :1] * s[:1]) @ Vh[:1, :]
print("rank-one:", rank_one.tolist())
design = np.array([[1., 0.], [1., 1.], [1., 2.]])
coef, residuals, rank, singular = np.linalg.lstsq(design, [1., 3., 5.], rcond=None)
print("least squares:", np.round(coef, 3).tolist(), "rank:", rank)
print("eigenvalues:", np.linalg.eigvalsh(A).tolist())
print("Frobenius norm:", round(float(np.linalg.norm(X)), 3))`,
    output: `[[2.0, -2.0], [6.0, -4.0]]
[0.0, 2.0]
[[5.0, 11.0], [11.0, 25.0]]
[0.0, 2.0]
[3.0, 2.0] True
shapes: (2, 2) (2,) (2, 2)
singular values: [4.0, 2.0]
reconstructed: True
rank-one: [[0.0, 0.0], [0.0, 4.0]]
least squares: [1.0, 2.0] rank: 2
eigenvalues: [2.0, 4.0]
Frobenius norm: 5.477`,
  },
numpyRandomIO: {
    code: `import numpy as np
from io import BytesIO

rng = np.random.default_rng(7)
same = np.random.default_rng(7)
draw = rng.integers(0, 10, size=(2, 3))
print("shape:", draw.shape)
print("same initial seed:", bool(np.array_equal(draw, same.integers(0, 10, size=(2, 3)))))
print("normal sample shape:", rng.normal(size=(3, 2)).shape)
print("permutation covers input:", sorted(rng.permutation(4).tolist()))
buffer = BytesIO()
np.save(buffer, draw)
buffer.seek(0)
restored = np.load(buffer, allow_pickle=False)
print("round trip:", bool(np.array_equal(draw, restored)), restored.dtype == draw.dtype)`,
    output: `shape: (2, 3)
same initial seed: True
normal sample shape: (3, 2)
permutation covers input: [0, 1, 2, 3]
round trip: True True`,
  },
numpyProject: {
    code: `import numpy as np

train = np.array([[1., 10., 5.], [2., 20., 5.], [3., 30., 5.]])
test = np.array([[4., 40., 5.]])
mean = train.mean(axis=0, keepdims=True)
scale = train.std(axis=0, keepdims=True)
safe_scale = np.where(scale == 0, 1, scale)
Z = (train - mean) / safe_scale
test_Z = (test - mean) / safe_scale
selected = Z[Z[:, 0] > 0]
gram = selected @ selected.T
print("mean:", mean.tolist())
print("standardized:", np.round(Z, 3).tolist())
print("test:", np.round(test_Z, 3).tolist())
print("selected shape:", selected.shape)
print("Gram:", np.round(gram, 3).tolist())
np.testing.assert_allclose(Z.mean(axis=0), 0, atol=1e-12)
np.testing.assert_allclose(Z.std(axis=0), [1, 1, 0], atol=1e-12)
print("checks passed")`,
    output: `mean: [[2.0, 20.0, 5.0]]
standardized: [[-1.225, -1.225, 0.0], [0.0, 0.0, 0.0], [1.225, 1.225, 0.0]]
test: [[2.449, 2.449, 0.0]]
selected shape: (1, 3)
Gram: [[3.0]]
checks passed`,
  }
};
