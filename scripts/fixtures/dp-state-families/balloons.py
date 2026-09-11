def balloon_plan(values):
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("Use nonnegative integer values")
    padded = [1, *values, 1]
    size = len(padded)
    best = [[0] * size for _ in range(size)]
    last_choice = [[None] * size for _ in range(size)]
    for gap in range(2, size):
        for left in range(size - gap):
            right = left + gap
            winner = None
            for last in range(left + 1, right):
                candidate = (best[left][last] + best[last][right]
                             + padded[left] * padded[last] * padded[right])
                if winner is None or candidate > winner:
                    winner, last_choice[left][right] = candidate, last
            best[left][right] = winner

    order = []
    stack = [(False, 0, size - 1)]
    while stack:
        emit, left, right = stack.pop()
        if emit:
            order.append(left - 1)  # Convert padded position to original ID.
            continue
        last = last_choice[left][right]
        if last is not None:
            stack.extend([(True, last, last), (False, last, right), (False, left, last)])
    return best[0][-1], order


def replay(values, order):
    if sorted(order) != list(range(len(values))):
        raise ValueError("Remove every original object exactly once")
    live, earned = list(range(len(values))), []
    for original in order:
        position = live.index(original)
        left = values[live[position - 1]] if position else 1
        right = values[live[position + 1]] if position + 1 < len(live) else 1
        earned.append(left * values[original] * right)
        live.pop(position)
    return earned


for values in [[2, 4, 3], [2, 0, 3], [1, 1, 1], []]:
    value, order = balloon_plan(values)
    scores = replay(values, order)
    print(values, "=>", value, "original IDs", order, "earned", scores)
    assert sum(scores) == value
