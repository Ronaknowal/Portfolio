def cosine_sim(a, b):
    """Cosine similarity over co-rated positions (both nonzero)."""
    mask = (a != 0) & (b != 0)
    if mask.sum() == 0:
        return 0.0
    a_m, b_m = a[mask], b[mask]
    return float(np.dot(a_m, b_m) / (np.linalg.norm(a_m) * np.linalg.norm(b_m) + 1e-10))

# Build item-item similarity matrix
item_sim = np.zeros((n_items, n_items))
for i in range(n_items):
    for j in range(n_items):
        item_sim[i, j] = cosine_sim(R[:, i], R[:, j])

def predict_item_cf(R, item_sim, user, item, k=3):
    """Predict rating for (user, item) using top-k similar items."""
    neighbors = np.argsort(-item_sim[item])
    rated = np.where(R[user] != 0)[0]
    num, den = 0.0, 0.0
    count = 0
    for nb in neighbors:
        if nb != item and nb in rated:
            s = item_sim[item, nb]
            num += s * R[user, nb]
            den += abs(s)
            count += 1
            if count >= k:
                break
    return num / (den + 1e-10) if den > 0 else 0.0

# Predict missing entries for user 0
print("Item-CF predictions for user 0 (missing items: 2, 4, 6):")
for item in [2, 4, 6]:
    pred = predict_item_cf(R, item_sim, user=0, item=item, k=3)
    print(f"  item {item}: predicted={pred:.3f}")

# Item-CF predictions for user 0 (missing items: 2, 4, 6):
#   item 2: predicted=3.063
#   item 4: predicted=3.333
#   item 6: predicted=3.000

# Top-3 recommendation for user 0
missing_0 = [i for i in range(n_items) if R[0, i] == 0]
scored = [(i, predict_item_cf(R, item_sim, 0, i)) for i in missing_0]
scored.sort(key=lambda x: -x[1])
print("Top-3 recs for user 0:", [(i, round(s, 2)) for i, s in scored[:3]])
# Top-3 recs for user 0: [(4, 3.33), (2, 3.06), (6, 3.0)]