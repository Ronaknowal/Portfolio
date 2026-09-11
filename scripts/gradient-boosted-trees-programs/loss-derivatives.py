import math


def sigmoid(score):
    return 1 / (1 + math.exp(-score)) if score >= 0 else math.exp(score) / (1 + math.exp(score))


def log_loss(y, score):
    return max(score, 0) - y * score + math.log1p(math.exp(-abs(score)))


y, score = 1, -2.0
p = sigmoid(score)
gradient, hessian = p - y, p * (1 - p)
step = 1e-4
numerical_g = (log_loss(y, score + step) - log_loss(y, score - step)) / (2 * step)
numerical_h = (log_loss(y, score + step) - 2 * log_loss(y, score) + log_loss(y, score - step)) / step ** 2
print(f"p={p:.6f}; g={gradient:.6f}; h={hessian:.6f}")
print(f"finite differences: g={numerical_g:.6f}; h={numerical_h:.6f}")
print(f"score -2 plus 1: p={sigmoid(score + 1):.6f}")
print(f"incorrect probability addition: {p + sigmoid(1):.6f}")
print(f"extreme wrong score loss={log_loss(1,-1000):.1f}; correct score loss={log_loss(1,1000):.1f}")
