def prefix_encode(categories, targets, order, prior=0.5, strength=1.0):
    if len(categories) != len(targets) or sorted(order) != list(range(len(targets))) or strength <= 0:
        raise ValueError("valid equal rows, permutation and positive strength required")
    totals, counts, encoded = {}, {}, [None] * len(targets)
    for index in order:
        category = categories[index]
        encoded[index] = (totals.get(category, 0) + strength * prior) / (counts.get(category, 0) + strength)
        # The target is added AFTER encoding this row.
        totals[category] = totals.get(category, 0) + targets[index]
        counts[category] = counts.get(category, 0) + 1
    return encoded, totals, counts


categories = ["A", "B", "A", "C", "B", "A"]
targets = [0, 1, 1, 1, 0, 1]
order = list(range(6))
for label, changed in [("original", targets), ("flip row 3", [0, 1, 0, 1, 0, 1])]:
    encoded, totals, counts = prefix_encode(categories, changed, order)
    naive_a = (totals["A"] + 0.5) / (counts["A"] + 1)
    print(label, "prefixes:", [round(value, 6) for value in encoded])
    print(f"row 3 prefix={encoded[2]:.6f}; full-training A statistic={naive_a:.6f}")
print("unseen category at inference: external prior=0.500000")
