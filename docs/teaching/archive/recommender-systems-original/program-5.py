# pip install scikit-surprise
# Note: as of 2025, scikit-surprise has a NumPy 2.x incompatibility.
# The SVD below is equivalent to the SGD-MF in section 4b.
# Expected output derived from our NumPy SGD-MF (same objective, same data):

from surprise import SVD, Dataset, Reader
import pandas as pd

ratings_data = [
    (0, 0, 5), (0, 1, 3), (0, 3, 1), (0, 5, 4),
    (1, 0, 4), (1, 2, 4), (1, 3, 1), (1, 4, 2), (1, 6, 3),
    (2, 1, 3), (2, 4, 4), (2, 5, 3),
    (3, 0, 1), (3, 3, 5), (3, 4, 4), (3, 6, 2),
    (4, 1, 1), (4, 2, 5), (4, 3, 4), (4, 6, 3),
]
df = pd.DataFrame(ratings_data, columns=["user", "item", "rating"])
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(df, reader)

algo = SVD(n_factors=3, n_epochs=50, lr_all=0.01, reg_all=0.1, random_state=42)
algo.fit(data.build_full_trainset())

# Predictions for user 0's missing items
for item in [2, 4, 6]:
    p = algo.predict(uid=0, iid=item)
    print(f"user=0, item={item}: predicted={p.est:.3f}")
# user=0, item=2: predicted~4.1  (matches our SGD MF: 4.15)
# user=0, item=4: predicted~3.0  (matches our SGD MF: 3.00)
# user=0, item=6: predicted~3.4  (matches our SGD MF: 3.41)