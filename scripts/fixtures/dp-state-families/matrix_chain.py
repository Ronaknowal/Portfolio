def matrix_chain(dimensions):
    if len(dimensions) < 2 or any(type(value) is not int or value <= 0 for value in dimensions):
        raise ValueError("Use at least two positive integer dimensions")
    size = len(dimensions) - 1
    costs = [[None] * (size + 1) for _ in range(size)]
    splits = [[None] * (size + 1) for _ in range(size)]
    for length in range(1, size + 1):
        for left in range(size - length + 1):
            right = left + length
            if length == 1:
                costs[left][right] = 0
            for split in range(left + 1, right):
                candidate = (costs[left][split] + costs[split][right]
                             + dimensions[left] * dimensions[split] * dimensions[right])
                if costs[left][right] is None or candidate < costs[left][right]:
                    costs[left][right], splits[left][right] = candidate, split

    # Stack events emit one expression and child-before-parent operations.
    tokens, operations = [], []
    stack = [("visit", 0, size)]
    while stack:
        kind, first, second = stack.pop()
        if kind == "text":
            tokens.append(first)
        elif kind == "merge":
            left, split, right = first
            operations.append((left, split, right))
        elif second == first + 1:
            tokens.append(f"A{first}")
        else:
            split = splits[first][second]
            stack.extend([("merge", (first, split, second), None),
                          ("text", ")", None), ("visit", split, second),
                          ("text", " * ", None), ("visit", first, split),
                          ("text", "(", None)])
    return costs[0][size], "".join(tokens), operations


for dimensions in [[8, 2, 12, 3, 6], [3, 7, 2, 5], [2, 2, 2, 2], [4, 3]]:
    value, expression, operations = matrix_chain(dimensions)
    print(dimensions, "=>", value, expression)
    print("child-before-parent intervals:", operations)
