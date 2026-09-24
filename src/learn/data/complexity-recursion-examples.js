export const complexityRecursionExamples = {
  loopCounts: {
    title: 'Count the body, including an empty input',
    code: `def work_counts(n):
    if type(n) is not int or n < 0:
        raise ValueError("n must be a nonnegative integer")
    square = triangle = doubling = 0
    for i in range(n):
        for j in range(n):
            square += 1
        for j in range(i):
            triangle += 1
        j = 1
        while j < n:
            doubling += 1
            j *= 2
    return square, triangle, doubling

if __name__ == "__main__":
    for size in (0, 1, 6, 12):
        print(size, work_counts(size))`,
    expected: '0 (0, 0, 0)\n1 (1, 0, 0)\n6 (36, 15, 18)\n12 (144, 66, 48)'
  },
  searchCases: {
    title: 'Same length, different search work',
    code: `def find_counted(values, target):
    comparisons = 0
    for index, value in enumerate(values):
        comparisons += 1
        if value == target:
            return index, comparisons
    return None, comparisons

if __name__ == "__main__":
    records = [12, 30, 18, 7]
    for target in (12, 7, 99):
        print(target, find_counted(records, target))
    successful_counts = [find_counted(records, x)[1] for x in records]
    print("uniform successful mean", sum(successful_counts) / len(records))
    print("empty", find_counted([], 12))`,
    expected: '12 (0, 1)\n7 (3, 4)\n99 (None, 4)\nuniform successful mean 2.5\nempty (None, 0)'
  },
  suffixSum: {
    title: 'A complete recursive contract and an observable call trace',
    code: `def suffix_sum(values):
    def visit(index):
        if index == len(values):
            return 0
        return values[index] + visit(index + 1)
    return visit(0)

def suffix_trace(values):
    # Diagnostics deliberately allocate extra storage for snapshots.
    frames, states = [], []
    calls = 0
    def record(event, result=None):
        states.append({"event": event, "frames": [dict(f) for f in frames],
                       "calls": calls, "result": result})
    def visit(index):
        nonlocal calls
        calls += 1
        frame = {"index": index, "phase": "enter", "value": None, "child": None}
        frames.append(frame)
        record("enter")
        if index == len(values):
            frame["phase"], frame["value"] = "return", 0
            record("base")
        else:
            frame["phase"] = "waiting"
            record("call")
            frame["child"] = visit(index + 1)
            frame["phase"] = "return"
            frame["value"] = values[index] + frame["child"]
            record("combine")
        frames.pop()
        return frame["value"]
    result = visit(0)
    record("finished", result)
    return states

if __name__ == "__main__":
    readings = [3, 1, 4]
    print("sum", suffix_sum(readings))
    print("empty", suffix_sum([]))
    for state in suffix_trace(readings):
        if state["event"] in ("base", "combine"):
            current = state["frames"][-1]
            print("return", current["index"], current["value"])
    print("input unchanged", readings)`,
    expected: 'sum 8\nempty 0\nreturn 3 0\nreturn 2 4\nreturn 1 5\nreturn 0 8\ninput unchanged [3, 1, 4]'
  },
  balancedSum: {
    title: 'Two children, but only linear total work',
    code: `def balanced_sum(values):
    calls = 0
    peak_frames = 0
    def visit(start, stop, depth):
        nonlocal calls, peak_frames
        calls += 1
        peak_frames = max(peak_frames, depth)
        size = stop - start
        if size == 0:
            return 0
        if size == 1:
            return values[start]
        middle = start + size // 2
        left = visit(start, middle, depth + 1)
        right = visit(middle, stop, depth + 1)
        return left + right
    result = visit(0, len(values), 1)
    return result, calls, peak_frames

if __name__ == "__main__":
    print("empty", balanced_sum([]))
    print("odd", balanced_sum([3, -1, 4, 1, 5]))
    print("eight", balanced_sum(list(range(1, 9))))`,
    expected: 'empty (0, 1, 1)\nodd (12, 9, 4)\neight (36, 15, 4)'
  },
  power: {
    title: 'Reuse a half-power instead of recomputing it',
    code: `def power_counted(base, exponent):
    if type(exponent) is not int or exponent < 0:
        raise ValueError("exponent must be a nonnegative integer")
    calls = multiplications = 0
    def visit(remaining):
        nonlocal calls, multiplications
        calls += 1
        if remaining == 0:
            return 1
        half = visit(remaining // 2)
        result = half * half
        multiplications += 1
        if remaining % 2:
            result *= base
            multiplications += 1
        return result
    result = visit(exponent)
    return result, calls, multiplications

if __name__ == "__main__":
    for base, exponent in ((2, 13), (-3, 4), (7, 0)):
        print(base, exponent, power_counted(base, exponent))`,
    expected: '2 13 (8192, 5, 7)\n-3 4 (81, 4, 4)\n7 0 (1, 1, 0)'
  },
  recurrences: {
    title: 'Execute the five toy cost recurrences',
    code: `def recurrence_cost(kind, n):
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive power of two")
    probe = n
    while probe > 1 and probe % 2 == 0:
        probe //= 2
    if probe != 1:
        raise ValueError("n must be a positive power of two")
    settings = {
        "chain_constant": (1, False, False),
        "chain_linear": (1, False, True),
        "half_constant": (1, True, False),
        "half_linear": (1, True, True),
        "two_half_linear": (2, True, True),
    }
    if kind not in settings:
        raise ValueError("unknown recurrence")
    children, halve, linear = settings[kind]
    def visit(size):
        if size == 1:
            return 1
        smaller = size // 2 if halve else size - 1
        local = size if linear else 1
        return local + sum(visit(smaller) for _ in range(children))
    return visit(n)

if __name__ == "__main__":
    for kind in ("chain_constant", "chain_linear", "half_constant",
                 "half_linear", "two_half_linear"):
        print(kind, recurrence_cost(kind, 8))`,
    expected: 'chain_constant 8\nchain_linear 36\nhalf_constant 4\nhalf_linear 15\ntwo_half_linear 32'
  },
  fibonacci: {
    title: 'Separate repeated calls from distinct subproblems',
    code: `def fibonacci_comparison(n):
    if type(n) is not int or not 0 <= n <= 20:
        raise ValueError("use 0 <= n <= 20 for this naive comparison")
    naive_calls = 0
    def naive(index):
        nonlocal naive_calls
        naive_calls += 1
        if index < 2:
            return index
        return naive(index - 1) + naive(index - 2)
    cache = {0: 0, 1: 1}
    computed = 0
    def memoized(index):
        nonlocal computed
        if index not in cache:
            cache[index] = memoized(index - 1) + memoized(index - 2)
            computed += 1
        return cache[index]
    first = naive(n)
    second = memoized(n)
    previous, current = 0, 1
    for _ in range(n):
        previous, current = current, previous + current
    return first, second, previous, naive_calls, computed

if __name__ == "__main__":
    for index in (0, 1, 5, 8):
        print(index, fibonacci_comparison(index))`,
    expected: '0 (0, 0, 0, 1, 0)\n1 (1, 1, 1, 1, 0)\n5 (5, 5, 5, 15, 4)\n8 (21, 21, 21, 67, 7)'
  },
  appendBudget: {
    title: 'A costly append can belong to a cheap sequence',
    code: `def doubling_budget(count):
    if type(count) is not int or count < 0:
        raise ValueError("count must be a nonnegative integer")
    capacity, size, copied = 1, 0, 0
    costs = []
    for _ in range(count):
        copy_now = 0
        if size == capacity:
            copy_now = size
            copied += copy_now
            capacity *= 2
        size += 1
        costs.append(1 + copy_now)  # one write, plus old elements copied
    return costs, copied, capacity

if __name__ == "__main__":
    costs, copied, capacity = doubling_budget(9)
    print("per append", costs)
    print("copies", copied, "writes", len(costs), "total", sum(costs))
    print("capacity", capacity)`,
    expected: 'per append [1, 2, 3, 1, 5, 1, 1, 1, 9]\ncopies 15 writes 9 total 24\ncapacity 16'
  },
  reverse: {
    title: 'Removing recursion changes peak storage, not the required swaps',
    code: `def reverse_in_place(values):
    left, right, swaps = 0, len(values) - 1, 0
    while left < right:
        values[left], values[right] = values[right], values[left]
        swaps += 1
        left += 1
        right -= 1
    return swaps

if __name__ == "__main__":
    for values in ([], [9], [2, 4, 6, 8, 10]):
        original = list(values)
        swaps = reverse_in_place(values)
        print(original, "->", values, "swaps", swaps)`,
    expected: '[] -> [] swaps 0\n[9] -> [9] swaps 0\n[2, 4, 6, 8, 10] -> [10, 8, 6, 4, 2] swaps 2'
  },
  thresholdCount: {
    title: 'Independent task — one threshold, arbitrary signed readings',
    code: `def count_below(values, threshold):
    def visit(start, stop):
        if start == stop:
            return 0
        if stop - start == 1:
            return int(values[start] < threshold)
        middle = start + (stop - start) // 2
        return visit(start, middle) + visit(middle, stop)
    return visit(0, len(values))

if __name__ == "__main__":
    print(count_below([], 0))
    print(count_below([4, -2, 4, 9, 0], 4))
    print(count_below([-3, -2, -1], -1))`,
    expected: '0\n2\n2'
  }
};
