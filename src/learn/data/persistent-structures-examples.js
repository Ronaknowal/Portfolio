const pointCore = `from dataclasses import dataclass

@dataclass(frozen=True, eq=False)
class Node:
    low: int
    high: int
    total: int
    left: "Node | None" = None
    right: "Node | None" = None

def build(values):
    values = tuple(values)
    if any(type(value) is not int for value in values):
        raise ValueError("Integer payloads required")
    def visit(low, high):
        if low == high:
            return None
        if high - low == 1:
            return Node(low, high, values[low])
        middle = (low + high) // 2
        left, right = visit(low, middle), visit(middle, high)
        return Node(low, high, left.total + right.total, left, right)
    return visit(0, len(values))

def assign(root, index, value):
    if type(index) is not int or root is None or not root.low <= index < root.high:
        raise IndexError("Index outside this array")
    if type(value) is not int:
        raise ValueError("Integer payload required")
    def visit(node):
        if node.high - node.low == 1:
            return node if node.total == value else Node(node.low, node.high, value)
        middle = (node.low + node.high) // 2
        left = visit(node.left) if index < middle else node.left
        right = visit(node.right) if index >= middle else node.right
        if left is node.left and right is node.right:
            return node
        return Node(node.low, node.high, left.total + right.total, left, right)
    return visit(root)

def range_sum(root, low, high):
    size = 0 if root is None else root.high
    if type(low) is not int or type(high) is not int or not 0 <= low <= high <= size:
        raise ValueError("Use a half-open range inside the array")
    def visit(node):
        if node is None or low == high or node.high <= low or high <= node.low:
            return 0
        if low <= node.low and node.high <= high:
            return node.total
        return visit(node.left) + visit(node.right)
    return visit(root)

def identities(roots):
    reached = set()
    def visit(node):
        if node is None or node in reached:
            return
        reached.add(node)
        visit(node.left)
        visit(node.right)
    for root in roots:
        visit(root)
    return reached
`;

const lazyCore = `from dataclasses import dataclass

@dataclass(frozen=True, eq=False)
class LazyNode:
    low: int
    high: int
    total: int
    add: int = 0
    left: "LazyNode | None" = None
    right: "LazyNode | None" = None

def lazy_build(values):
    values = tuple(values)
    if any(type(value) is not int for value in values):
        raise ValueError("Integer payloads required")
    def visit(low, high):
        if low == high:
            return None
        if high - low == 1:
            return LazyNode(low, high, values[low])
        middle = (low + high) // 2
        left, right = visit(low, middle), visit(middle, high)
        return LazyNode(low, high, left.total + right.total, 0, left, right)
    return visit(0, len(values))

def check_range(root, low, high):
    size = 0 if root is None else root.high
    if type(low) is not int or type(high) is not int or not 0 <= low <= high <= size:
        raise ValueError("Invalid half-open range")

def range_add(root, low, high, delta):
    check_range(root, low, high)
    if type(delta) is not int:
        raise ValueError("Integer increment required")
    def visit(node):
        if node is None or low == high or delta == 0 or node.high <= low or high <= node.low:
            return node
        width = node.high - node.low
        if low <= node.low and node.high <= high:
            return LazyNode(node.low, node.high, node.total + delta * width,
                            node.add + delta, node.left, node.right)
        left, right = visit(node.left), visit(node.right)
        total = left.total + right.total + node.add * width
        return LazyNode(node.low, node.high, total, node.add, left, right)
    return visit(root)

def lazy_sum(root, low, high):
    check_range(root, low, high)
    def visit(node, inherited):
        if node is None or low == high or node.high <= low or high <= node.low:
            return 0
        if low <= node.low and node.high <= high:
            return node.total + inherited * (node.high - node.low)
        carry = inherited + node.add
        return visit(node.left, carry) + visit(node.right, carry)
    return visit(root, 0)
`;

export const persistentStructuresExamples = {
  alias: {
    title: 'Two names, a shallow copy, and a preserved value',
    code: `from copy import deepcopy

original = {"counts": [2, 1, 4]}
alias = original
shallow = original.copy()
independent = deepcopy(original)
shallow["counts"][1] = 9
print("same outer:", alias is original, shallow is original)
print("same inner:", shallow["counts"] is original["counts"])
print("original:", original["counts"])
print("deep copy:", independent["counts"])
`,
    expected: 'same outer: True False\nsame inner: True\noriginal: [2, 9, 4]\ndeep copy: [2, 1, 4]',
  },
  stack: {
    title: 'Create and pop shared-tail stack versions',
    code: `from dataclasses import dataclass

@dataclass(frozen=True, eq=False)
class Link:
    value: int
    tail: "Link | None" = None

def push(root, value):
    if type(value) is not int:
        raise ValueError("Integer payload required")
    return Link(value, root)

def pop(root):
    if root is None:
        raise IndexError("Empty stack")
    return root.value, root.tail

def values(root):
    result = []
    while root is not None:
        result.append(root.value)
        root = root.tail
    return result

tail = push(push(None, 1), 3)
a, b = push(tail, 9), push(tail, 7)
print(values(a), values(b), values(tail))
value, restored = pop(a)
print(value, restored is tail, a.tail is b.tail)
`,
    expected: '[9, 3, 1] [7, 3, 1] [3, 1]\n9 True True',
  },
  point: {
    title: 'Build, branch, query and count physical objects',
    code: pointCore + `
v0 = build([2, 1, 4, 3, 5])
v1 = assign(v0, 2, 9)
v2 = assign(v0, 0, -2)
for root in [v0, v1, v2]:
    print([range_sum(root, i, i + 1) for i in range(5)], range_sum(root, 1, 4))
print("new per branch:", len(identities([v1]) - identities([v0])),
      len(identities([v2]) - identities([v0])))
print("shared left:", v1.left is v0.left)
print("no-op root:", assign(v0, 2, 4) is v0)
print("all / only v1:", len(identities([v0, v1, v2])), len(identities([v1])))
print("empty sum:", range_sum(build([]), 0, 0))
`,
    expected: '[2, 1, 4, 3, 5] 8\n[2, 1, 9, 3, 5] 13\n[-2, 1, 4, 3, 5] 8\nnew per branch: 3 3\nshared left: True\nno-op root: True\nall / only v1: 15 9\nempty sum: 0',
  },
  history: {
    title: 'Keep only per-index writes on one snapshot timeline',
    code: `from bisect import bisect_right

class SnapshotArray:
    def __init__(self, length):
        if type(length) is not int or length < 0:
            raise ValueError("Nonnegative length required")
        self.history = [[(0, 0)] for _ in range(length)]
        self.current = 0

    def check_index(self, index):
        if type(index) is not int or not 0 <= index < len(self.history):
            raise IndexError("Index outside this array")

    def set(self, index, value):
        self.check_index(index)
        if type(value) is not int:
            raise ValueError("Integer value required")
        history = self.history[index]
        if history[-1][0] == self.current:
            history[-1] = (self.current, value)
        else:
            history.append((self.current, value))

    def snap(self):
        result = self.current
        self.current += 1
        return result

    def get(self, index, snapshot):
        self.check_index(index)
        if type(snapshot) is not int or not 0 <= snapshot < self.current:
            raise ValueError("Snapshot must already be saved")
        history = self.history[index]
        position = bisect_right(history, snapshot, key=lambda write: write[0]) - 1
        return history[position][1]

array = SnapshotArray(3)
print("snap:", array.snap())
array.set(0, 4)
array.set(0, 5)
print("snap:", array.snap())
array.set(1, 7)
print("snap:", array.snap())
array.set(0, 9)
array.snap()
array.snap()
print("history:", array.history)
print("queries:", array.get(0, 0), array.get(0, 2), array.get(0, 4), array.get(2, 4))
`,
    expected: 'snap: 0\nsnap: 1\nsnap: 2\nhistory: [[(0, 0), (1, 5), (3, 9)], [(0, 0), (2, 7)], [(0, 0)]]\nqueries: 0 5 9 0',
  },
  rank: {
    title: 'Use two persistent prefix roots to select a subarray occurrence',
    code: pointCore + `
class PrefixRanks:
    def __init__(self, values):
        values = tuple(values)
        if any(type(value) is not int for value in values):
            raise ValueError("Integer values required")
        self.alphabet = sorted(set(values))
        self.size = len(values)
        self.roots = [build([0] * len(self.alphabet))]
        coordinate = {value: index for index, value in enumerate(self.alphabet)}
        for value in values:
            old = self.roots[-1]
            index = coordinate[value]
            count = range_sum(old, index, index + 1)
            self.roots.append(assign(old, index, count + 1))

    def kth(self, low, high, rank):
        if any(type(value) is not int for value in [low, high, rank]):
            raise ValueError("Integer endpoints and rank required")
        if not 0 <= low < high <= self.size or not 1 <= rank <= high - low:
            raise ValueError("Nonempty subarray and valid 1-based rank required")
        earlier, later = self.roots[low], self.roots[high]
        while later.high - later.low > 1:
            left_count = later.left.total - earlier.left.total
            if rank <= left_count:
                earlier, later = earlier.left, later.left
            else:
                rank -= left_count
                earlier, later = earlier.right, later.right
        return self.alphabet[later.low]

query = PrefixRanks([5, 1, 4, 1, 3, 5])
print("alphabet:", query.alphabet)
print("ranks [1,5):", [query.kth(1, 5, k) for k in range(1, 5)])
print("ranks [0,6):", [query.kth(0, 6, k) for k in range(1, 7)])
print("prefix totals:", [0 if root is None else root.total for root in query.roots])
`,
    expected: 'alphabet: [1, 3, 4, 5]\nranks [1,5): [1, 1, 3, 4]\nranks [0,6): [1, 1, 3, 4, 5, 5]\nprefix totals: [0, 1, 2, 3, 4, 5, 6]',
  },
  lazy: {
    title: 'Preserve lazy range-add versions without pushing into old children',
    code: lazyCore + `
v0 = lazy_build([2, 1, 4, 3, 5])
v1 = range_add(v0, 0, 5, 10)
v2 = range_add(v1, 1, 4, -3)
v3 = range_add(v0, 2, 5, 2)
for root in [v0, v1, v2, v3]:
    print([lazy_sum(root, i, i + 1) for i in range(5)], lazy_sum(root, 0, 5))
print("full-cover children shared:", v1.left is v0.left, v1.right is v0.right)
print("root tags:", v0.add, v1.add, v2.add, v3.add)
`,
    expected: '[2, 1, 4, 3, 5] 15\n[12, 11, 14, 13, 15] 65\n[12, 8, 11, 10, 15] 56\n[2, 1, 6, 5, 7] 21\nfull-cover children shared: True True\nroot tags: 0 10 10 0',
  },
  queue: {
    title: 'Why repeated branches can repeat a queue reversal',
    code: `# A deliberately naive persistent two-list queue: tuples stand in for lists.
# Count only elements visited by reversals; tuple slicing adds further costs.
def dequeue(front, rear):
    reversed_visits = 0
    if not front:
        front = tuple(reversed(rear))
        reversed_visits = len(rear)
        rear = ()
    if not front:
        raise IndexError("Empty queue")
    return front[0], (front[1:], rear), reversed_visits

old = ((), (4, 3, 2, 1))
work = 0
for _ in range(3):
    value, new_version, visits = dequeue(*old)
    work += visits
    print(value, new_version, visits)
print("reversal visits:", work)
`,
    expected: '1 ((2, 3, 4), ()) 4\n1 ((2, 3, 4), ()) 4\n1 ((2, 3, 4), ()) 4\nreversal visits: 12',
  },
};
