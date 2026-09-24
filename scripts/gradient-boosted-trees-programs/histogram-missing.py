from statistics import mean


def candidates(cuts, missing_target):
    x, y = [1, 2, 3, 4, 5, 6, None], [1, 1, 1, 9, 9, 9, missing_target]
    base = mean(y)
    g = [base - value for value in y]
    result = []
    for cut in cuts:
        for missing in ["left", "right"]:
            left = [i for i, value in enumerate(x) if (missing == "left" if value is None else value <= cut)]
            right = [i for i in range(len(x)) if i not in left]
            score = lambda rows: sum(g[i] for i in rows) ** 2 / (2 * len(rows))
            gain = score(left) + score(right) - score(list(range(len(x))))
            result.append((gain, cut, missing))
    return max(result, key=lambda item: item[0])


for target in [8, 2]:
    for name, cuts in [("fine", [1.5, 2.5, 3.5, 4.5, 5.5]), ("coarse", [2.5, 4.5])]:
        gain, cut, missing = candidates(cuts, target)
        print(f"missing target={target}; {name}: cut={cut}; missing->{missing}; gain={gain:.6f}")
