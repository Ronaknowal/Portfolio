import numpy as np

# 5 users x 7 items explicit rating matrix (0 = missing)
R = np.array([
    [5, 3, 0, 1, 0, 4, 0],
    [4, 0, 4, 1, 2, 0, 3],
    [0, 3, 0, 0, 4, 3, 0],
    [1, 0, 0, 5, 4, 0, 2],
    [0, 1, 5, 4, 0, 0, 3],
], dtype=float)

n_users, n_items = R.shape
print(f"Matrix shape: {n_users} users x {n_items} items")
print(f"Observed ratings: {int((R > 0).sum())} / {n_users * n_items}  "
      f"(sparsity {100*(R == 0).mean():.1f}%)")

# Matrix shape: 5 users x 7 items
# Observed ratings: 20 / 35  (sparsity 42.9%)