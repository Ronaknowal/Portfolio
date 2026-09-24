import numpy as np

weight = np.array([[0., 1., 0., 0.], [1., 0., 1., 0.],
                   [0., 1., 0., 1.], [0., 0., 1., 0.]])
degree = weight.sum(axis=1)
laplacian = np.diag(degree) - weight
known, unknown = [0, 3], [1, 2]
hard = np.linalg.solve(
    laplacian[np.ix_(unknown, unknown)],
    weight[np.ix_(unknown, known)] @ np.array([0., 1.]),
)
normalized = weight / np.sqrt(degree[:, None] * degree[None, :])
evidence = np.zeros((4, 2))
evidence[0, 0] = evidence[3, 1] = 1
alpha = .8
soft = np.linalg.solve(np.eye(4) - alpha * normalized, (1-alpha) * evidence)
row_mass = soft.sum(axis=1, keepdims=True)
readout = np.divide(soft, row_mass, out=np.zeros_like(soft), where=row_mass > 0)
print("hard B,C:", np.round(hard, 6))
print("soft class-1 readout A,B,C,D:", np.round(readout[:, 1], 6))
print("has evidence:", (row_mass[:, 0] > 0).tolist())
