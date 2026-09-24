np.random.seed(42)

# 5 users x 7 items implicit feedback (click counts)
counts = np.array([
    [10,  3,  0,  1,  0, 15,  0],
    [ 8,  0,  5,  1,  2,  0,  4],
    [ 0,  6,  0,  0,  9, 12,  0],
    [ 1,  0,  0, 20,  7,  0,  3],
    [ 0,  1, 14,  8,  0,  0,  6],
], dtype=float)

alpha = 40.0
K_als = 3
lam_als = 0.1
n_als_epochs = 15

# Confidence and binary preference matrices
C = 1.0 + alpha * counts      # conf[u,i] = 1 + 40 * count
Pref = (counts > 0).astype(float)  # 1 if interacted, 0 if not

X = np.random.randn(n_users, K_als) * 0.1  # user factors
Y = np.random.randn(n_items, K_als) * 0.1  # item factors

for epoch in range(n_als_epochs):
    # Fix Y, solve for each user (k x k system per user)
    YTY = Y.T @ Y
    for u in range(n_users):
        Cu = np.diag(C[u])
        A = YTY + Y.T @ (Cu - np.eye(n_items)) @ Y + lam_als * np.eye(K_als)
        b = Y.T @ Cu @ Pref[u]
        X[u] = np.linalg.solve(A, b)
    # Fix X, solve for each item
    XTX = X.T @ X
    for i in range(n_items):
        Ci = np.diag(C[:, i])
        A = XTX + X.T @ (Ci - np.eye(n_users)) @ X + lam_als * np.eye(K_als)
        b = X.T @ Ci @ Pref[:, i]
        Y[i] = np.linalg.solve(A, b)
    # Weighted loss
    pred = X @ Y.T
    loss = sum(C[u,i] * (Pref[u,i] - pred[u,i])**2
               for u in range(n_users) for i in range(n_items))
    loss += lam_als * (np.sum(X**2) + np.sum(Y**2))
    if epoch in (0, 3, 6, 9, 12, 14):
        print(f"Epoch {epoch+1:2d}: weighted loss={loss:.2f}")

# Epoch  1: weighted loss=137.60
# Epoch  4: weighted loss=83.67
# Epoch  7: weighted loss=66.10
# Epoch 10: weighted loss=56.95
# Epoch 13: weighted loss=50.30
# Epoch 15: weighted loss=46.78

pred = X @ Y.T
unseen_0 = [i for i in range(n_items) if counts[0, i] == 0]
top3_imp = sorted(unseen_0, key=lambda i: -pred[0, i])[:3]
print("Top-3 unseen items for user 0 (implicit ALS):")
for rank, i in enumerate(top3_imp, 1):
    print(f"  rank {rank}: item {i}  score={pred[0,i]:.4f}")
# rank 1: item 6  score=0.1300
# rank 2: item 2  score=0.1140
# rank 3: item 4  score=-0.1450