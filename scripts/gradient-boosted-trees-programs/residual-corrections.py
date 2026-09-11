from statistics import mean


def fit_stump(x, residual):
    if len(x) != len(residual) or not x:
        raise ValueError("equal, nonempty rows required")
    values = sorted(set(x))
    center = mean(residual)
    best = {"cut": None, "left": center, "right": center,
            "sse": sum((r - center) ** 2 for r in residual)}
    for low, high in zip(values, values[1:]):
        midpoint = low + (high - low) / 2
        cut = midpoint if midpoint < high else low
        left = [r for value, r in zip(x, residual) if value <= cut]
        right = [r for value, r in zip(x, residual) if value > cut]
        a, b = mean(left), mean(right)
        cost = sum((r - a) ** 2 for r in left) + sum((r - b) ** 2 for r in right)
        if cost < best["sse"]:
            best = {"cut": cut, "left": a, "right": b, "sse": cost}
    return best


def predict(tree, value):
    return tree["left"] if tree["cut"] is None or value <= tree["cut"] else tree["right"]


x = [1, 2, 3, 4, 5, 6]
y = [2, 2, 3, 7, 8, 8]
base, rate = mean(y), 0.5
prediction = [base] * len(y)
trees = []
print(f"baseline={base:.3f}; MSE={mean((a-b)**2 for a,b in zip(y,prediction)):.6f}")
for step in range(1, 4):
    residual = [target - fitted for target, fitted in zip(y, prediction)]
    tree = fit_stump(x, residual)
    trees.append(tree)
    prediction = [fitted + rate * predict(tree, value) for value, fitted in zip(x, prediction)]
    error = mean((a - b) ** 2 for a, b in zip(y, prediction))
    print(f"round={step}; cut={tree['cut']}; leaves=({tree['left']:.6f},{tree['right']:.6f}); MSE={error:.6f}")
new_x = 4.5
print(f"new x={new_x}; prediction={base + rate * sum(predict(tree,new_x) for tree in trees):.6f}")
