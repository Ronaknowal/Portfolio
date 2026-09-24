// Every exported code string is a complete standalone Python program. Shared
// class text below avoids maintaining different copies within this one topic.
const segmentClass = `class SegmentTree:
    def __init__(self, values, combine, identity):
        self.n = len(values)
        self.combine, self.identity = combine, identity
        self.size = 1
        while self.size < self.n:
            self.size *= 2
        self.tree = [identity] * (2 * self.size)
        self.tree[self.size:self.size + self.n] = values
        for node in range(self.size - 1, 0, -1):
            self.tree[node] = combine(self.tree[2 * node], self.tree[2 * node + 1])

    def set(self, index, value):
        if not 0 <= index < self.n:
            raise IndexError("Array index out of range")
        node = self.size + index
        self.tree[node] = value
        while node > 1:
            node //= 2
            self.tree[node] = self.combine(self.tree[2 * node], self.tree[2 * node + 1])

    def query(self, left, right):
        if not 0 <= left <= right <= self.n:
            raise IndexError("Use a half-open range within the array")
        left += self.size
        right += self.size
        before = after = self.identity
        while left < right:
            if left % 2:
                before = self.combine(before, self.tree[left])
                left += 1
            if right % 2:
                right -= 1
                after = self.combine(self.tree[right], after)
            left //= 2
            right //= 2
        return self.combine(before, after)`;

const fenwickClass = `class Fenwick:
    def __init__(self, values):
        self.values = list(values)
        self.n = len(values)
        self.tree = [0] + list(values)
        for index in range(1, self.n + 1):
            parent = index + (index & -index)
            if parent <= self.n:
                self.tree[parent] += self.tree[index]

    def add(self, index, delta):
        if not 0 <= index < self.n:
            raise IndexError("Array index out of range")
        self.values[index] += delta
        internal = index + 1
        while internal <= self.n:
            self.tree[internal] += delta
            internal += internal & -internal

    def set(self, index, value):
        if not 0 <= index < self.n:
            raise IndexError("Array index out of range")
        self.add(index, value - self.values[index])

    def prefix(self, end):
        if not 0 <= end <= self.n:
            raise IndexError("Prefix boundary out of range")
        result = 0
        while end > 0:
            result += self.tree[end]
            end -= end & -end
        return result

    def range_sum(self, left, right):
        if not 0 <= left <= right <= self.n:
            raise IndexError("Use a half-open range within the array")
        return self.prefix(right) - self.prefix(left)`;

export const rangeQueryExamples = {
  prefixes: {
    title: 'Static prefix cancellation',
    code: `def prefixes(values):
    result = [0]
    for value in values:
        result.append(result[-1] + value)
    return result

values = [2, 1, 3, 4, 0, 5, 2, 1]
cached = prefixes(values)
print(cached)
print("sum [1,7):", cached[7] - cached[1])
print("empty [3,3):", cached[3] - cached[3])`,
    expected: '[0, 2, 3, 6, 10, 10, 15, 17, 18]\nsum [1,7): 15\nempty [3,3): 0',
  },
  differences: {
    title: 'Turn a batch of range additions into boundary changes',
    code: `def add_ranges(values, updates):
    # Each update is (left, right, delta), with right excluded.
    size = len(values)
    differences = [0] * (size + 1)
    for left, right, delta in updates:
        if not 0 <= left <= right <= size:
            raise IndexError("Invalid update boundaries")
        differences[left] += delta
        differences[right] -= delta
    changed = []
    running = 0
    for index, value in enumerate(values):
        running += differences[index]
        changed.append(value + running)
    return changed

values = [2, 1, 3, 4, 0, 5, 2, 1]
print(add_ranges(values, [(1, 5, 3), (3, 8, -2)]))
print("original:", values)`,
    expected: '[2, 4, 6, 5, 1, 3, 0, -1]\noriginal: [2, 1, 3, 4, 0, 5, 2, 1]',
  },
  segment: {
    title: 'One interval tree, different ordered summary contracts',
    code: `from math import inf

${segmentClass}

values = [2, 1, 3, 4, 0, 5, 2, 1]
sums = SegmentTree(values, lambda a, b: a + b, 0)
print("sum:", sums.query(1, 7))
sums.set(4, 3)
print("after assignment:", sums.query(1, 7))
minimum = SegmentTree(values, min, inf)
print("minimum:", minimum.query(1, 7), "empty:", minimum.query(3, 3))
letters = SegmentTree(list("ABCDEFGH"), lambda a, b: a + b, "")
print("ordered fold:", letters.query(1, 7))
print("empty string:", repr(letters.query(2, 2)))
print("empty tree:", SegmentTree([], lambda a, b: a + b, 0).query(0, 0))`,
    expected: "sum: 15\nafter assignment: 18\nminimum: 0 empty: inf\nordered fold: BCDEFG\nempty string: ''\nempty tree: 0",
  },
  fenwick: {
    title: 'Build binary prefix blocks, then add or assign correctly',
    code: `${fenwickClass}

tree = Fenwick([2, 1, 3, 4, 0, 5, 2, 1])
print("internal cells:", tree.tree)
print("prefix 7:", tree.prefix(7), "range [1,7):", tree.range_sum(1, 7))
tree.add(4, 3)
print("after add:", tree.range_sum(1, 7))
tree.set(2, -1)
print("after set:", tree.range_sum(1, 7))
print("empty prefix:", Fenwick([]).prefix(0))`,
    expected: 'internal cells: [0, 2, 3, 3, 10, 0, 5, 2, 18]\nprefix 7: 17 range [1,7): 15\nafter add: 18\nafter set: 14\nempty prefix: 0',
  },
  rangeFenwick: {
    title: 'Two difference summaries support range additions and sums',
    code: `${fenwickClass}

class RangeAddSum:
    def __init__(self, values):
        self.n = len(values)
        difference = []
        previous = 0
        for value in values:
            difference.append(value - previous)
            previous = value
        difference.append(-previous)  # Boundary n returns the extension to zero.
        self.changes = Fenwick(difference)
        self.weighted = Fenwick([index * value for index, value in enumerate(difference)])

    def add(self, left, right, delta):
        if not 0 <= left <= right <= self.n:
            raise IndexError("Invalid half-open range")
        for boundary, change in [(left, delta), (right, -delta)]:
            self.changes.add(boundary, change)
            self.weighted.add(boundary, boundary * change)

    def prefix(self, end):
        if not 0 <= end <= self.n:
            raise IndexError("Invalid prefix boundary")
        return end * self.changes.prefix(end) - self.weighted.prefix(end)

    def range_sum(self, left, right):
        if not 0 <= left <= right <= self.n:
            raise IndexError("Invalid half-open range")
        return self.prefix(right) - self.prefix(left)

tree = RangeAddSum([2, 1, 3, 4])
print("initial:", tree.range_sum(0, 4))
tree.add(0, 4, 3)
tree.add(1, 3, -2)
print("values:", [tree.range_sum(i, i + 1) for i in range(4)])
print("whole:", tree.range_sum(0, 4), "middle:", tree.range_sum(1, 3))
print("empty:", RangeAddSum([]).range_sum(0, 0))`,
    expected: 'initial: 10\nvalues: [5, 2, 4, 7]\nwhole: 18 middle: 6\nempty: 0',
  },
  lazy: {
    title: 'Compose range assignment and addition without stale answers',
    code: `class LazySum:
    def __init__(self, values):
        self.n = len(values)
        self.size = 1
        while self.size < self.n:
            self.size *= 2
        self.total = [0] * (2 * self.size)
        self.multiplier = [1] * (2 * self.size)
        self.addition = [0] * (2 * self.size)
        self.total[self.size:self.size + self.n] = values
        for node in range(self.size - 1, 0, -1):
            self.total[node] = self.total[2 * node] + self.total[2 * node + 1]

    def _apply(self, node, length, multiplier, addition):
        self.total[node] = multiplier * self.total[node] + addition * length
        if length > 1:
            # New map AFTER old map: m*(old_m*x + old_b) + b.
            self.multiplier[node] = multiplier * self.multiplier[node]
            self.addition[node] = multiplier * self.addition[node] + addition

    def _push(self, node, length):
        multiplier, addition = self.multiplier[node], self.addition[node]
        if length > 1 and (multiplier != 1 or addition != 0):
            self._apply(2 * node, length // 2, multiplier, addition)
            self._apply(2 * node + 1, length // 2, multiplier, addition)
            self.multiplier[node], self.addition[node] = 1, 0

    def _check(self, left, right):
        if not 0 <= left <= right <= self.n:
            raise IndexError("Invalid half-open range")

    def _change(self, left, right, multiplier, addition):
        self._check(left, right)
        def visit(node, low, high):
            if right <= low or high <= left or left == right:
                return
            if left <= low and high <= right:
                self._apply(node, high - low, multiplier, addition)
                return
            self._push(node, high - low)
            middle = (low + high) // 2
            visit(2 * node, low, middle)
            visit(2 * node + 1, middle, high)
            self.total[node] = self.total[2 * node] + self.total[2 * node + 1]
        visit(1, 0, self.size)

    def add(self, left, right, delta):
        self._change(left, right, 1, delta)

    def set(self, left, right, value):
        self._change(left, right, 0, value)

    def query(self, left, right):
        self._check(left, right)
        def visit(node, low, high):
            if right <= low or high <= left or left == right:
                return 0
            if left <= low and high <= right:
                return self.total[node]
            self._push(node, high - low)
            middle = (low + high) // 2
            return visit(2 * node, low, middle) + visit(2 * node + 1, middle, high)
        return visit(1, 0, self.size)

tree = LazySum([2, 1, 3, 4])
tree.add(0, 4, 3)
tree.set(1, 3, 5)
print("after add then middle set:", [tree.query(i, i + 1) for i in range(4)])
tree.add(2, 4, -2)
print("sum [1,4):", tree.query(1, 4))
first, second = LazySum([2, 1]), LazySum([2, 1])
first.set(0, 2, 4)
first.add(0, 2, 3)
second.add(0, 2, 3)
second.set(0, 2, 4)
print("set then add:", first.query(0, 2), "add then set:", second.query(0, 2))
print("empty:", LazySum([]).query(0, 0))`,
    expected: 'after add then middle set: [5, 5, 5, 7]\nsum [1,4): 13\nset then add: 14 add then set: 8\nempty: 0',
  },
  sparse: {
    title: 'Use overlapping blocks only for the right operation',
    code: `class SparseMinimum:
    def __init__(self, values):
        self.n = len(values)
        self.levels = [list(values)]
        length = 2
        while length <= self.n:
            previous = self.levels[-1]
            half = length // 2
            self.levels.append([min(previous[i], previous[i + half])
                                for i in range(self.n - length + 1)])
            length *= 2

    def query(self, left, right):
        if not 0 <= left < right <= self.n:
            raise IndexError("Minimum requires a nonempty valid range")
        power = (right - left).bit_length() - 1
        length = 1 << power
        return min(self.levels[power][left], self.levels[power][right - length])

values = [2, 1, 3, 4, 0, 5, 2, 1]
table = SparseMinimum(values)
print(table.query(1, 7), table.query(2, 4), table.query(5, 6))
print("true sum [1,7):", sum(values[1:7]))
print("wrong overlapping sum:", sum(values[1:5]) + sum(values[3:7]))`,
    expected: '0 3 5\ntrue sum [1,7): 15\nwrong overlapping sum: 19',
  },
  movingMaximum: {
    title: 'Expire old indices and remove dominated maxima separately',
    code: `from collections import deque

def moving_maxima(values, width):
    if not isinstance(width, int) or width <= 0:
        raise ValueError("Width must be a positive integer")
    candidates = deque()
    answers = []
    for index, value in enumerate(values):
        while candidates and candidates[0] <= index - width:
            candidates.popleft()
        while candidates and values[candidates[-1]] <= value:
            candidates.pop()
        candidates.append(index)
        if index + 1 >= width:
            position = candidates[0]
            answers.append((values[position], position))
    return answers  # (maximum, newest tied original index)

print(moving_maxima([4, 2, 2, 5, 1, 3, 0, 2], 3))
print("equal values:", moving_maxima([2, 2, 2], 2))
print("no full window:", moving_maxima([1, 2], 3))
print("empty:", moving_maxima([], 1))`,
    expected: '[(4, 0), (5, 3), (5, 3), (5, 3), (3, 5), (3, 5)]\nequal values: [(2, 1), (2, 2)]\nno full window: []\nempty: []',
  },
  signedShortest: {
    title: 'A prefix deque repairs shortest ranges with negative entries',
    code: `from collections import deque

def shortest_at_least(values, target):
    if target <= 0:
        raise ValueError("This interface requires a positive target")
    prefixes = [0]
    for value in values:
        prefixes.append(prefixes[-1] + value)
    candidates = deque()
    best = None
    for end, total in enumerate(prefixes):
        while candidates and total - prefixes[candidates[0]] >= target:
            start = candidates.popleft()
            answer = (end - start, (start, end))
            if best is None or answer < best:
                best = answer
        while candidates and prefixes[candidates[-1]] >= total:
            candidates.pop()
        candidates.append(end)
    return best  # None or (length, half-open witness); earliest start on ties.

print(shortest_at_least([1, -1, 5], 5))
print(shortest_at_least([2, -1, 2], 3))
print(shortest_at_least([-1, -2], 1))
print(shortest_at_least([], 1))`,
    expected: '(1, (2, 3))\n(3, (0, 3))\nNone\nNone',
  },
  weightedIncreasing: {
    title: 'Accelerate a weighted increasing-subsequence recurrence and keep its witness',
    code: `${segmentClass}

def weighted_increasing(values, weights):
    if len(values) != len(weights):
        raise ValueError("Every value needs a weight")
    coordinates = sorted(set(values))
    ranks = {value: rank for rank, value in enumerate(coordinates)}
    # A negative endpoint denotes the empty sequence; prefer it on zero ties.
    def better(first, second):
        if first[0] != second[0]:
            return first if first[0] > second[0] else second
        return first if first[1] <= second[1] else second
    tree = SegmentTree([(0, -1)] * len(coordinates), better, (0, -1))
    parents = [-1] * len(values)
    for index, (value, weight) in enumerate(zip(values, weights)):
        rank = ranks[value]
        score, previous = tree.query(0, rank)  # Strictly smaller values only.
        parents[index] = previous
        candidate = (score + weight, index)
        tree.set(rank, better(tree.query(rank, rank + 1), candidate))
    score, endpoint = tree.query(0, len(coordinates))
    witness = []
    while endpoint != -1:
        witness.append(endpoint)
        endpoint = parents[endpoint]
    return score, witness[::-1]

print(weighted_increasing([3, 1, 2, 2, 4], [4, 2, 5, 20, 3]))
print(weighted_increasing([2, 2], [5, 7]))
print(weighted_increasing([1, 2], [-4, -2]))
print(weighted_increasing([], []))`,
    expected: '(25, [1, 3, 4])\n(7, [1])\n(0, [])\n(0, [])',
  },
  frequencySelection: {
    title: 'Find an occurrence rank directly through nonnegative frequency blocks',
    code: `${fenwickClass}

class Frequencies(Fenwick):
    def __init__(self, counts):
        if any(not isinstance(count, int) or count < 0 for count in counts):
            raise ValueError("Counts must be nonnegative integers")
        super().__init__(counts)

    def add(self, index, delta):
        if not 0 <= index < self.n or not isinstance(delta, int) or self.values[index] + delta < 0:
            raise ValueError("Keep counts nonnegative at a valid index")
        super().add(index, delta)

    def kth(self, rank):
        if not isinstance(rank, int) or not 1 <= rank <= self.prefix(self.n):
            raise ValueError("Use a one-based existing occurrence rank")
        # index is a prefix LENGTH known to contain fewer than rank occurrences.
        index = 0
        bit = 1 << (self.n.bit_length() - 1)
        remaining = rank
        while bit:
            candidate = index + bit
            if candidate <= self.n and self.tree[candidate] < remaining:
                remaining -= self.tree[candidate]
                index = candidate
            bit >>= 1
        return index  # External index of the first prefix reaching rank.

table = Frequencies([2, 0, 3, 1])
print("ranks 1 through 6:", [table.kth(rank) for rank in range(1, 7)])
table.add(1, 1)
print("third after update:", table.kth(3))
print("counts:", table.values)`,
    expected: 'ranks 1 through 6: [0, 0, 2, 2, 2, 3]\nthird after update: 1\ncounts: [2, 1, 3, 1]',
  },
  subtree: {
    title: 'Independent application: turn rooted subtrees into contiguous array ranges',
    code: `${fenwickClass}

def flatten_tree(children, root):
    size = len(children)
    if not 0 <= root < size:
        raise ValueError("Choose an existing root")
    enter = [-1] * size
    leave = [-1] * size
    order = []
    stack = [(root, False)]
    while stack:
        vertex, exiting = stack.pop()
        if exiting:
            leave[vertex] = len(order)
            continue
        if not 0 <= vertex < size or enter[vertex] != -1:
            raise ValueError("Children must form a rooted tree, without duplicate visits")
        enter[vertex] = len(order)
        order.append(vertex)
        stack.append((vertex, True))
        stack.extend((child, False) for child in reversed(children[vertex]))
    if len(order) != size:
        raise ValueError("Every vertex must belong to the chosen tree")
    return enter, leave, order

children = [[1, 2], [3, 4], [5], [], [], []]
values = [5, 2, 7, 1, 4, 3]
enter, leave, order = flatten_tree(children, 0)
tree = Fenwick([values[vertex] for vertex in order])
print("DFS order:", order)
print("subtree 1 range:", (enter[1], leave[1]))
print("subtree 1 sum:", tree.range_sum(enter[1], leave[1]))
tree.add(enter[3], 6)
print("after node 3 increases:", tree.range_sum(enter[1], leave[1]))
print("whole tree:", tree.range_sum(enter[0], leave[0]))`,
    expected: 'DFS order: [0, 1, 3, 4, 2, 5]\nsubtree 1 range: (1, 4)\nsubtree 1 sum: 7\nafter node 3 increases: 13\nwhole tree: 28',
  },
  richerSummary: {
    title: 'Independent application: maintain the earlier four-field subarray summary',
    code: `${segmentClass}

def merge(first, second):
    # None is an empty summary; nonempty fields are total, prefix, suffix, best.
    if first is None:
        return second
    if second is None:
        return first
    total_a, prefix_a, suffix_a, best_a = first
    total_b, prefix_b, suffix_b, best_b = second
    return (total_a + total_b,
            max(prefix_a, total_a + prefix_b),
            max(suffix_b, total_b + suffix_a),
            max(best_a, best_b, suffix_a + prefix_b))

values = [-2, 4, -1, 3, -5, 2]
tree = SegmentTree([(value, value, value, value) for value in values], merge, None)
print("initial best:", tree.query(0, 6)[3])
tree.set(4, (2, 2, 2, 2))
print("updated best:", tree.query(0, 6)[3])
print("negative singleton:", tree.query(0, 1)[3])
print("empty summary:", tree.query(3, 3))`,
    expected: 'initial best: 6\nupdated best: 10\nnegative singleton: -2\nempty summary: None',
  },
};
