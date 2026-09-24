import numpy as np


def pair_move(x, y, alpha, c, i, j):
    gram = x @ x.T
    gradient = 1 - y * (gram @ (alpha * y))
    sign = y[i] * y[j]
    lower = max(-alpha[j], alpha[i] - c if sign == 1 else -alpha[i])
    upper = min(c - alpha[j], alpha[i] if sign == 1 else c - alpha[i])
    slope = gradient[j] - sign * gradient[i]
    curvature = np.sum((x[i] - x[j]) ** 2)
    gain = lambda delta: slope * delta - .5 * curvature * delta ** 2
    if curvature > 0:
        delta = np.clip(slope / curvature, lower, upper)
    else:
        delta = 0. if slope == 0 else max([lower, upper], key=gain)
    result = alpha.copy()
    result[i], result[j] = alpha[i] - sign * delta, alpha[j] + delta
    return result, lower, upper, curvature, gain(delta)


if __name__ == '__main__':
    x = np.array([[-1., 0.], [.5, 1.], [1., -.5]])
    y, alpha = np.array([-1., 1., 1.]), np.array([.2, .1, .1])
    for i, j in [(0, 1), (1, 2)]:
        after, lower, upper, q, gain = pair_move(x, y, alpha, 1.2, i, j)
        print(f'pair=({i},{j}) delta interval=[{lower:.3f},{upper:.3f}] q={q:.3f}')
        print('after:', after.round(6).tolist(), f'balance={after @ y:.6f} gain={gain:.6f}')
    duplicate = pair_move(np.zeros((2, 1)), np.array([-1., 1.]), np.zeros(2), 1.2, 0, 1)
    print('zero-curvature after:', duplicate[0].tolist(), f'gain={duplicate[-1]:.1f}')
