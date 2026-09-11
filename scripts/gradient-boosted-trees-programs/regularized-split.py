def leaf(gradient, hessian, lam=1.0, alpha=0.0):
    if lam < 0 or alpha < 0 or any(h < 0 for h in hessian):
        raise ValueError("nonnegative regularization and Hessians required")
    G, H = sum(gradient), sum(hessian)
    if H + lam <= 0:
        raise ValueError("positive quadratic curvature required")
    softened = max(G - alpha, 0) - max(-G - alpha, 0)
    weight = -softened / (H + lam)
    improvement = softened ** 2 / (2 * (H + lam))
    return weight, improvement


# This is a chosen CURRENT score, not mean([1,2,3,4,5]), which is 3.
y, score = [1, 2, 3, 4, 5], 2.5
g, h = [score - value for value in y], [1.0] * len(y)
for alpha, gamma in [(0, 0), (0, 3), (2, 0)]:
    parent = leaf(g, h, alpha=alpha)
    left = leaf(g[:2], h[:2], alpha=alpha)
    right = leaf(g[2:], h[2:], alpha=alpha)
    gross = left[1] + right[1] - parent[1]
    net = gross - gamma
    print(f"alpha={alpha}; gamma={gamma}; leaves=({left[0]:.6f},{right[0]:.6f}); net gain={net:.6f}; accept={net>0}")
print("alpha=0, rate=.3 update:", [round(score + .3 * leaf(g[:2] if i<2 else g[2:], h[:2] if i<2 else h[2:])[0],4) for i in range(5)])
