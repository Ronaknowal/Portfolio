export const vectorTensorExamples = {
  projection: {
    title: 'Check a projection and its perpendicular remainder',
    code: `import numpy as np

v = np.array([3.0, 2.0])
u = np.array([2.0, 1.0])
coefficient = (v @ u) / (u @ u)  # u must be nonzero
projection = coefficient * u
residual = v - projection
print("dot:", float(v @ u))
print("length:", round(float(np.linalg.norm(v)), 4))
print("projection:", projection.tolist())
print("residual:", np.round(residual, 3).tolist())
print("perpendicular:", bool(np.isclose(residual @ u, 0.0)))
print("reconstructs:", bool(np.allclose(projection + residual, v)))`,
    expected: `dot: 8.0
length: 3.6056
projection: [3.2, 1.6]
residual: [-0.2, 0.4]
perpendicular: True
reconstructs: True`,
  },
  maps: {
    title: 'Compute column images and change the composition order',
    code: `import numpy as np

x = np.array([1, 1])
stretch = np.array([[2, 0], [0, 1]])
rotate = np.array([[0, -1], [1, 0]])
print("stretch then rotate:", (rotate @ stretch @ x).tolist())
print("rotate then stretch:", (stretch @ rotate @ x).tolist())
shear = np.array([[1, 1], [0, 1]])
v = np.array([2, 1])
columns = v[0] * shear[:, 0] + v[1] * shear[:, 1]
print("weighted columns:", columns.tolist())
print("row calculation:", (shear @ v).tolist())
print("identity:", (np.eye(2, dtype=int) @ v).tolist())`,
    expected: `stretch then rotate: [-1, 2]
rotate then stretch: [-2, 1]
weighted columns: [3, 1]
row calculation: [3, 1]
identity: [2, 1]`,
  },
  resources: {
    title: 'Turn three orders into resource requirements',
    code: `import numpy as np

# Columns: small packs, large packs. Rows: three separate orders.
orders = np.array([[2, 1], [0, 3], [1, 2]])
# Rows: small, large. Columns: panels, bolts per pack.
recipe = np.array([[3, 2], [5, 4]])
resources = orders @ recipe
print("resources:", resources.tolist())
print("first panel count:", int(2 * 3 + 1 * 5))
print("with one spare panel per order:", (resources + [1, 0]).tolist())
print("total resources:", resources.sum(axis=0).tolist())
print("large-only orders:", resources[orders[:, 0] == 0].tolist())`,
    expected: `resources: [[11, 8], [15, 12], [13, 10]]
first panel count: 11
with one spare panel per order: [[12, 8], [16, 12], [14, 10]]
total resources: [39, 30]
large-only orders: [[15, 12]]`,
  },
  affine: {
    title: 'Retained example: four observations through an affine layer',
    code: `import numpy as np

X = np.array([[1., 2., 3.], [2., 1., 0.],
              [0., 1., 2.], [3., 2., 1.]])
W = np.array([[0.2, -0.1], [0.4, 0.3], [0.1, 0.5]])
b = np.array([0.01, -0.02])
scores = X @ W + b
feature_mean = X.mean(axis=0, keepdims=True)
centered = X - feature_mean
print("shapes:", X.shape, W.shape, scores.shape)
print("scores:", np.round(scores, 2).tolist())
print("feature mean:", feature_mean.tolist())
print("centered column sums:", centered.sum(axis=0).tolist())`,
    expected: `shapes: (4, 3) (3, 2) (4, 2)
scores: [[1.31, 1.98], [0.81, 0.08], [0.61, 1.28], [1.51, 0.78]]
feature mean: [[1.5, 1.5, 1.5]]
centered column sums: [0.0, 0.0, 0.0]`,
  },
  reduction: {
    title: 'Reduce named axes, then keep one for subtraction',
    code: `import numpy as np

# session, time, channel; invented measurements in one common unit
readings = np.arange(12, dtype=float).reshape(2, 2, 3)
print("mean over time:", readings.mean(axis=1).tolist())
print("mean over session:", readings.mean(axis=0).tolist())
print("mean over channel:", readings.mean(axis=2).tolist())
baseline = readings.mean(axis=1, keepdims=True)
centered = readings - baseline
print("kept shape:", baseline.shape)
print("time-centered means:", centered.mean(axis=1).tolist())`,
    expected: `mean over time: [[1.5, 2.5, 3.5], [7.5, 8.5, 9.5]]
mean over session: [[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]]
mean over channel: [[1.0, 4.0], [7.0, 10.0]]
kept shape: (2, 1, 3)
time-centered means: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]`,
  },
  broadcasting: {
    title: 'A legal broadcast that answers the wrong question',
    code: `import numpy as np

X = np.array([[2., 10.], [4., 14.]])
row_means = X.mean(axis=1)
wrong = X - row_means
right = X - X.mean(axis=1, keepdims=True)
print("row means:", row_means.tolist())
print("wrong:", wrong.tolist())
print("right:", right.tolist())
print("right row means:", right.mean(axis=1).tolist())
column = np.array([[1], [2], [3]])
row = np.array([10, 20, 30])
print("outer addition shape:", (column + row).shape)
print("outer addition:", (column + row).tolist())`,
    expected: `row means: [6.0, 9.0]
wrong: [[-4.0, 1.0], [-2.0, 5.0]]
right: [[-4.0, 4.0], [-5.0, 5.0]]
right row means: [0.0, 0.0]
outer addition shape: (3, 3)
outer addition: [[11, 21, 31], [12, 22, 32], [13, 23, 33]]`,
  },
  reindex: {
    title: 'Follow values through transpose, reshape and a new axis',
    code: `import numpy as np

A = np.arange(6).reshape(2, 3)
print("transpose:", A.T.tolist())
print("reshape:", A.reshape(3, 2).tolist())
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print("stack:", np.stack([a, b], axis=0).tolist())
print("concatenate:", np.concatenate([a, b], axis=0).tolist())
print("vector/row/column:", a.shape, a[None, :].shape, a[:, None].shape)
print("1-D transpose:", a.T.shape)
print("selected:", A[:, [0, 2]].tolist())`,
    expected: `transpose: [[0, 3], [1, 4], [2, 5]]
reshape: [[0, 1], [2, 3], [4, 5]]
stack: [[1, 2, 3], [4, 5, 6]]
concatenate: [1, 2, 3, 4, 5, 6]
vector/row/column: (3,) (1, 3) (3, 1)
1-D transpose: (3,)
selected: [[0, 2], [3, 5]]`,
  },
  images: {
    title: 'Retained image shapes, with values and round-trip checks',
    code: `import numpy as np

# Each value is an index marker, not an actual image intensity.
images = np.arange(8 * 28 * 28).reshape(8, 28, 28, 1)
flat = images.reshape(8, -1)
channels_first = np.moveaxis(images, -1, 1)
mean_image = images.mean(axis=0)
left, right = np.split(flat, 2, axis=0)
combined = np.concatenate([left, right], axis=0)
print("flattened:", flat.shape)
print("channels first:", channels_first.shape)
print("mean image:", mean_image.shape)
print("split:", left.shape, right.shape)
print("recombined:", bool(np.array_equal(combined, flat)))
print("same indexed value:", int(images[1, 2, 3, 0]), int(channels_first[1, 0, 2, 3]))`,
    expected: `flattened: (8, 784)
channels first: (8, 1, 28, 28)
mean image: (28, 28, 1)
split: (4, 784) (4, 784)
recombined: True
same indexed value: 843 843`,
  },
  tokens: {
    title: 'Project tokens without mixing the examples',
    code: `import numpy as np

# batch, token position, hidden coordinate
embeddings = np.arange(12, dtype=float).reshape(2, 2, 3)
W = np.array([[1., 0.], [0., 1.], [1., -1.]])
scores = embeddings @ W
sequence_mean = embeddings.mean(axis=1)
print("output shape:", scores.shape)
print("token scores:", scores.tolist())
print("one vector per sequence:", sequence_mean.tolist())
print("mean then project:", (sequence_mean @ W).tolist())
print("commutes with this mean:", bool(np.allclose(scores.mean(axis=1), sequence_mean @ W)))`,
    expected: `output shape: (2, 2, 2)
token scores: [[[2.0, -1.0], [8.0, -1.0]], [[14.0, -1.0], [20.0, -1.0]]]
one vector per sequence: [[1.5, 2.5, 3.5], [7.5, 8.5, 9.5]]
mean then project: [[5.0, -1.0], [17.0, -1.0]]
commutes with this mean: True`,
  },
  nullspace: {
    title: 'A lost direction can be exactly the nuisance you want to reject',
    code: `import numpy as np

difference = np.array([-1., 1.])
readings = np.array([11., 15.])
same_offset = np.array([100., 100.])
print("difference:", float(difference @ readings))
print("after equal offset:", float(difference @ (readings + same_offset)))
print("rejected offset:", float(difference @ same_offset))
A = np.array([[1., 1.], [0., 0.]])
print("two inputs, same output:", (A @ [1., 0.]).tolist(), (A @ [0., 1.]).tolist())
print("lost direction:", (A @ [1., -1.]).tolist())
z = np.array([1j])
print("ordinary product:", z @ z)
print("conjugate product:", z.conj() @ z)`,
    expected: `difference: 4.0
after equal offset: 4.0
rejected offset: 0.0
two inputs, same output: [1.0, 0.0] [1.0, 0.0]
lost direction: [0.0, 0.0]
ordinary product: (-1+0j)
conjugate product: (1+0j)`,
  },
};
