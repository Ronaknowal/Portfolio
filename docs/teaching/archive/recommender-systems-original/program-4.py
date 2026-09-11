# pip install implicit
# Note: implicit expects item x user sparse matrix format.
# user_factors maps to rows of the input matrix (items when input is item x user).
# Scores: actual_user_scores[u, i] = item_factors[u] . user_factors[i]

import implicit
import scipy.sparse as sp
import numpy as np

# Build 5-user x 7-item click-count matrix
counts = np.array([
    [10,  3,  0,  1,  0, 15,  0],
    [ 8,  0,  5,  1,  2,  0,  4],
    [ 0,  6,  0,  0,  9, 12,  0],
    [ 1,  0,  0, 20,  7,  0,  3],
    [ 0,  1, 14,  8,  0,  0,  6],
], dtype=float)

user_items = sp.csr_matrix(counts)
item_users = user_items.T.tocsr()   # implicit expects (n_items, n_users)

model = implicit.als.AlternatingLeastSquares(
    factors=5,
    regularization=0.1,
    iterations=20,
    random_state=42,
)
model.fit(item_users)

# Retrieve scores manually (item_factors are actually user factors
# when input was item x user — see implicit's confusing naming)
scores = model.item_factors @ model.user_factors.T  # (n_users, n_items)

unseen_0 = [i for i in range(7) if counts[0, i] == 0]
top3 = sorted(unseen_0, key=lambda i: -scores[0, i])[:3]
print("Top-3 for user 0 (implicit ALS library):")
for rank, i in enumerate(top3, 1):
    print(f"  rank {rank}: item {i}  score={scores[0,i]:.4f}")
# rank 1: item 6  score=0.0425
# rank 2: item 2  score=0.0219
# rank 3: item 4  score=0.0152

# BPR variant — same API, different learning objective
bpr_model = implicit.bpr.BayesianPersonalizedRanking(
    factors=5, iterations=100, random_state=42
)
bpr_model.fit(item_users)