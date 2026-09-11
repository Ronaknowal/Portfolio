import numpy as np

x = np.array([2.0, 1.0])
w = np.array([1.0, 2.0])
b = -1.0
for scale in [1.0, 3.0]:
    weights, bias = scale * w, scale * b
    score = weights @ x + bias
    distance = score / np.linalg.norm(weights)
    projection = x - score * weights / (weights @ weights)
    print(f'scale={scale:.0f} score={score:.1f} distance={distance:.6f}')
    print('projection:', projection.round(6).tolist())
    print(f'boundary residual={weights @ projection + bias:.6f}')
