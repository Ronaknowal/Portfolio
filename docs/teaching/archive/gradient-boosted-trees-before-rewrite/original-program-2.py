import numpy as np

def xgb_leaf_value(g, h, lam=1.0):
    """Optimal leaf weight w* = -sum(g) / (sum(h) + lambda)."""
    return -np.sum(g) / (np.sum(h) + lam)

def xgb_split_gain(G_L, H_L, G_R, H_R, lam=1.0, gamma=0.0):
    """Split gain (Chen & Guestrin 2016, Eq. 7)."""
    return 0.5 * (
        G_L**2 / (H_L + lam) +
        G_R**2 / (H_R + lam) -
        (G_L + G_R)**2 / (H_L + H_R + lam)
    ) - gamma

# --- MSE example ---
y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
F = np.full(5, 2.5)     # initial prediction = mean(y)
g = F - y               # gradient of MSE = F - y
h = np.ones_like(y)     # hessian of MSE = 1 everywhere

print("=== XGBoost second-order update ===")
print(f"Gradients g : {g}")
print(f"Hessians  h : {h}")

# Evaluate one candidate split: left = first 2, right = last 3
G_L, H_L = np.sum(g[:2]), np.sum(h[:2])
G_R, H_R = np.sum(g[2:]), np.sum(h[2:])
gain = xgb_split_gain(G_L, H_L, G_R, H_R, lam=1.0, gamma=0.0)
w_L = xgb_leaf_value(g[:2], h[:2], lam=1.0)
w_R = xgb_leaf_value(g[2:], h[2:], lam=1.0)

print(f"\nSplit: left=[y1,y2]  right=[y3,y4,y5]")
print(f"  G_L={G_L:.2f}  H_L={H_L:.2f}  G_R={G_R:.2f}  H_R={H_R:.2f}")
print(f"  Split Gain = {gain:.4f}")
print(f"  Optimal w_L = {w_L:.4f}  w_R = {w_R:.4f}")

# Apply update (learning_rate = 0.3)
lr = 0.3
F_new = F.copy()
F_new[:2] += lr * w_L
F_new[2:] += lr * w_R
print(f"\nF before: {F}")
print(f"F after : {np.round(F_new, 4)}")