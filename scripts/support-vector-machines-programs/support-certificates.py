import numpy as np

for moved in [3.0, 0.5]:
    x = np.array([-2.0, -1.0, 1.0, moved])
    y = np.array([-1.0, -1.0, 1.0, 1.0])
    right = min(1.0, moved)
    w, b = 2 / (1 + right), (1 - right) / (1 + right)
    alpha = np.zeros(4)
    alpha[1] = alpha[2 if moved >= 1 else 3] = 2 / (1 + right) ** 2
    margin = y * (w * x + b)
    primal = .5 * w ** 2
    dual = alpha.sum() - .5 * (alpha @ (y * x)) ** 2
    assert np.all(margin >= 1 - 1e-12)
    assert abs(alpha @ y) < 1e-12 and abs(primal - dual) < 1e-12
    print(f'moved={moved:.1f} old feasible={bool(np.all(y * x >= 1))}')
    print(f'w={w:.6f} b={b:.6f} boundary={-b / w:.6f} gap={primal-dual:.6f}')

# The two positive rows are identical. Their combined coefficient is fixed.
x = np.array([-1.0, 1.0, 1.0])
y = np.array([-1.0, 1.0, 1.0])
for split in [0.0, .2, .5]:
    alpha = np.array([.5, split, .5 - split])
    w = alpha @ (y * x)
    print('alpha:', alpha.round(3).tolist(), f'w={w:.1f} dual={alpha.sum() - .5*w*w:.1f}')
