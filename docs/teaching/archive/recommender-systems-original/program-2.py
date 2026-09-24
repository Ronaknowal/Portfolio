np.random.seed(42)
K = 3          # latent factors
lr = 0.01      # learning rate
lam = 0.1      # regularization
n_epochs = 100

P = np.random.randn(n_users, K) * 0.1  # user factors (5 x 3)
Q = np.random.randn(n_items, K) * 0.1  # item factors (7 x 3)
bu = np.zeros(n_users)                  # user biases
bi = np.zeros(n_items)                  # item biases
mu = R[R > 0].mean()                    # global mean = 3.10

for epoch in range(n_epochs):
    total_sq = 0.0
    count = 0
    for u in range(n_users):
        for i in range(n_items):
            if R[u, i] > 0:
                pred = mu + bu[u] + bi[i] + P[u] @ Q[i]
                err = R[u, i] - pred
                bu[u] += lr * (err - lam * bu[u])
                bi[i] += lr * (err - lam * bi[i])
                p_u_old = P[u].copy()
                P[u] += lr * (err * Q[i] - lam * P[u])
                Q[i] += lr * (err * p_u_old - lam * Q[i])
                total_sq += err ** 2
                count += 1
    if epoch in (0, 20, 40, 60, 80, 99):
        print(f"Epoch {epoch+1:3d}: RMSE={np.sqrt(total_sq/count):.4f}")

# Epoch   1: RMSE=1.3438
# Epoch  21: RMSE=1.2217
# Epoch  41: RMSE=1.1226
# Epoch  61: RMSE=0.9049
# Epoch  81: RMSE=0.5471
# Epoch 100: RMSE=0.3645

# Reconstruct full rating matrix
R_pred = mu + bu[:, None] + bi[None, :] + P @ Q.T
np.set_printoptions(precision=2, suppress=True)
print("\nPredicted ratings matrix:")
print(R_pred)
# [[4.79 3.18 4.15 1.32 3.   3.66 3.41]
#  [3.86 2.4  3.7  1.21 2.59 3.02 2.76]
#  [3.6  2.56 4.63 3.04 3.64 3.51 3.21]
#  [1.39 1.13 4.6  4.67 3.84 2.7  2.33]
#  [2.32 1.72 4.67 4.05 3.8  3.03 2.71]]

print("\nTop-3 recs for user 0 (missing items: 2, 4, 6):")
missing_0 = [i for i in range(n_items) if R[0, i] == 0]
top3 = sorted(missing_0, key=lambda i: -R_pred[0, i])[:3]
for rank, i in enumerate(top3, 1):
    print(f"  rank {rank}: item {i}  predicted={R_pred[0,i]:.2f}")
# rank 1: item 2  predicted=4.15
# rank 2: item 6  predicted=3.41
# rank 3: item 4  predicted=3.00

print("\nTop-3 recs for user 2 (missing items: 0, 2, 3, 6):")
missing_2 = [i for i in range(n_items) if R[2, i] == 0]
top3_2 = sorted(missing_2, key=lambda i: -R_pred[2, i])[:3]
for rank, i in enumerate(top3_2, 1):
    print(f"  rank {rank}: item {i}  predicted={R_pred[2,i]:.2f}")
# rank 1: item 2  predicted=4.63
# rank 2: item 0  predicted=3.60
# rank 3: item 6  predicted=3.21