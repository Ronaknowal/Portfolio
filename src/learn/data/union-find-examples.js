const dsuImplementation = `class DisjointSet:
    def __init__(self, n):
        if type(n) is not int or n < 0:
            raise ValueError("n must be a nonnegative integer")
        self.parent = list(range(n))
        self.size = [1] * n
        self.count = n

    def _validate(self, item):
        if type(item) is not int or not 0 <= item < len(self.parent):
            raise IndexError("element outside 0..n-1")

    def find(self, item):
        self._validate(item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        # Save the old next pointer before changing it.
        while self.parent[item] != item:
            next_item = self.parent[item]
            self.parent[item] = root
            item = next_item
        return root

    def union(self, a, b):
        # Validate both before find can mutate a path.
        self._validate(a)
        self._validate(b)
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return False
        if self.size[root_a] < self.size[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        self.size[root_a] += self.size[root_b]
        self.count -= 1
        return True

    def connected(self, a, b):
        self._validate(a)
        self._validate(b)
        return self.find(a) == self.find(b)

    def component_size(self, item):
        return self.size[self.find(item)]

    def groups(self):
        result = {}
        for item in range(len(self.parent)):
            result.setdefault(self.find(item), []).append(item)
        return list(result.values())
`;

export const unionFindExamples = {
  basic: {
    title: 'A complete size-weighted, path-compressing implementation',
    code: `${dsuImplementation}
dsu = DisjointSet(6)
for a, b in [(0, 1), (2, 3), (1, 2), (0, 3)]:
    print((a, b), "merged:", dsu.union(a, b), "components:", dsu.count)
print("0 connected to 3:", dsu.connected(0, 3))
print("0 connected to 4:", dsu.connected(0, 4))
print("size of 3's component:", dsu.component_size(3))
print("groups:", dsu.groups())
print("empty:", DisjointSet(0).count, DisjointSet(0).groups())
try:
    dsu.find(-1)
except IndexError as error:
    print(error)
`,
    expected: '(0, 1) merged: True components: 5\n(2, 3) merged: True components: 4\n(1, 2) merged: True components: 3\n(0, 3) merged: False components: 3\n0 connected to 3: True\n0 connected to 4: False\nsize of 3\'s component: 4\ngroups: [[0, 1, 2, 3], [4], [5]]\nempty: 0 []\nelement outside 0..n-1',
  },
  eager: {
    title: 'A correct baseline: change every member label on a merge',
    code: `def quick_find_union(labels, a, b):
    old, new = labels[a], labels[b]
    if old == new:
        return False
    for item in range(len(labels)):
        if labels[item] == old:
            labels[item] = new
    return True

labels = list(range(5))
for edge in [(0, 1), (2, 3), (1, 2), (0, 3)]:
    changed = quick_find_union(labels, *edge)
    print(edge, changed, labels)
print("0 and 3:", labels[0] == labels[3])
`,
    expected: '(0, 1) True [1, 1, 2, 3, 4]\n(2, 3) True [1, 1, 3, 3, 4]\n(1, 2) True [3, 3, 3, 3, 4]\n(0, 3) False [3, 3, 3, 3, 4]\n0 and 3: True',
  },
  compression: {
    title: 'Inspect the same representative before and after compression',
    code: `${dsuImplementation}
def path(dsu, item):
    route = [item]
    while dsu.parent[item] != item:
        item = dsu.parent[item]
        route.append(item)
    return route

dsu = DisjointSet(8)
for edge in [(0, 1), (2, 3), (0, 2), (4, 5), (6, 7), (4, 6), (0, 4)]:
    dsu.union(*edge)
print("before:", path(dsu, 7))
print("representative:", dsu.find(7))
print("after:", path(dsu, 7))
print("another branch:", path(dsu, 5))
print("root size:", dsu.component_size(7), "components:", dsu.count)
`,
    expected: 'before: [7, 6, 4, 0]\nrepresentative: 0\nafter: [7, 0]\nanother branch: [5, 4, 0]\nroot size: 8 components: 1',
  },
  redundant: {
    title: 'Retain a connecting forest and identify redundant links',
    code: `${dsuImplementation}
def classify_links(n, edges):
    dsu = DisjointSet(n)
    kept, redundant = [], []
    for a, b in edges:
        if dsu.union(a, b):
            kept.append((a, b))
        else:
            redundant.append((a, b))
    return kept, redundant, dsu.count

kept, extra, count = classify_links(6, [(0, 1), (1, 2), (0, 2), (3, 4), (2, 4)])
print("kept:", kept)
print("redundant:", extra)
print("components:", count)
print("minimum new links:", count - 1)
print("cables reusable:", len(extra) >= count - 1)
`,
    expected: 'kept: [(0, 1), (1, 2), (3, 4), (2, 4)]\nredundant: [(0, 2)]\ncomponents: 2\nminimum new links: 1\ncables reusable: True',
  },
  equations: {
    title: 'Process all equalities before testing inequalities',
    code: `${dsuImplementation}
def equations_possible(equations):
    # Contract: triples (name, "==" or "!=", name); names are strings.
    names = sorted({name for a, _, b in equations for name in (a, b)})
    index = {name: i for i, name in enumerate(names)}
    dsu = DisjointSet(len(names))
    for a, operator, b in equations:
        if operator not in ("==", "!="):
            raise ValueError("unsupported relation")
        if operator == "==":
            dsu.union(index[a], index[b])
    return all(operator == "==" or not dsu.connected(index[a], index[b])
               for a, operator, b in equations)

print(equations_possible([("x", "!=", "z"), ("x", "==", "y"), ("y", "==", "z")]))
print(equations_possible([("x", "!=", "z"), ("x", "==", "y")]))
print(equations_possible([("x", "!=", "x")]))
print(equations_possible([]))
`,
    expected: 'False\nTrue\nFalse\nTrue',
  },
  identifiers: {
    title: 'Group records through shared identifiers, including transitive matches',
    code: `${dsuImplementation}
def merge_records(records):
    # Each record is an iterable of identifiers; output contains record indices.
    dsu = DisjointSet(len(records))
    first_owner = {}
    for record_id, identifiers in enumerate(records):
        for identifier in identifiers:
            if identifier in first_owner:
                dsu.union(record_id, first_owner[identifier])
            else:
                first_owner[identifier] = record_id
    return dsu.groups()

records = [{"cal-17", "camera-a"}, {"camera-a", "session-9"},
           {"session-9"}, {"cal-18"}, set()]
print(merge_records(records))
print(merge_records([]))
`,
    expected: '[[0, 1, 2], [3], [4]]\n[]',
  },
  islands: {
    title: 'Count active islands as cells open',
    code: `${dsuImplementation}
def island_counts(rows, columns, openings):
    if type(rows) is not int or type(columns) is not int or rows < 0 or columns < 0:
        raise ValueError("grid dimensions must be nonnegative integers")
    dsu = DisjointSet(rows * columns)
    active = set()
    count = 0
    answers = []
    for row, column in openings:
        if type(row) is not int or type(column) is not int or not (0 <= row < rows and 0 <= column < columns):
            raise IndexError("cell outside grid")
        cell = row * columns + column
        if cell not in active:
            active.add(cell)
            count += 1
            for dr, dc in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
                r, c = row + dr, column + dc
                neighbor = r * columns + c
                if 0 <= r < rows and 0 <= c < columns and neighbor in active:
                    if dsu.union(cell, neighbor):
                        count -= 1
        answers.append(count)
    return answers

print(island_counts(3, 3, [(0, 0), (0, 2), (0, 1), (1, 1), (0, 1), (2, 2)]))
ring = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]
print("ring then center:", island_counts(3, 3, ring + [(1, 1)])[-2:])
print("empty grid:", island_counts(0, 0, []))
`,
    expected: '[1, 2, 1, 1, 1, 2]\nring then center: [1, 1]\nempty grid: []',
  },
  metadata: {
    title: 'Keep a meaningful component label separately from its root',
    code: `${dsuImplementation}
class LabeledDisjointSet(DisjointSet):
    def __init__(self, n):
        super().__init__(n)
        self.minimum = list(range(n))

    def union(self, a, b):
        self._validate(a)
        self._validate(b)
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return False
        label = min(self.minimum[root_a], self.minimum[root_b])
        super().union(root_a, root_b)
        self.minimum[self.find(root_a)] = label
        return True

    def component_label(self, item):
        return self.minimum[self.find(item)]

dsu = LabeledDisjointSet(5)
dsu.union(3, 4)
dsu.union(3, 0)
print("representative:", dsu.find(0))
print("minimum label:", dsu.component_label(4))
print("duplicate merge:", dsu.union(4, 0))
print("size:", dsu.component_size(0))
`,
    expected: 'representative: 3\nminimum label: 0\nduplicate merge: False\nsize: 3',
  },
  rollback: {
    title: 'Optional: restore an earlier merge state with a change log',
    code: `class RollbackDSU:
    def __init__(self, n):
        if type(n) is not int or n < 0:
            raise ValueError("n must be a nonnegative integer")
        self.parent = list(range(n))
        self.size = [1] * n
        self.count = n
        self.history = []

    def find(self, item):
        if type(item) is not int or not 0 <= item < len(self.parent):
            raise IndexError("element outside 0..n-1")
        while self.parent[item] != item:
            item = self.parent[item]  # deliberately no compression
        return item

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.history.append((b, a, self.size[a]))
        self.parent[b] = a
        self.size[a] += self.size[b]
        self.count -= 1
        return True

    def snapshot(self):
        return len(self.history)

    def rollback(self, checkpoint):
        # A token is a depth in the current history, not a permanent version ID.
        if type(checkpoint) is not int or not 0 <= checkpoint <= len(self.history):
            raise ValueError("checkpoint outside current history")
        while len(self.history) > checkpoint:
            child, parent, old_size = self.history.pop()
            self.parent[child] = child
            self.size[parent] = old_size
            self.count += 1

dsu = RollbackDSU(4)
dsu.union(0, 1)
checkpoint = dsu.snapshot()
dsu.union(1, 2)
dsu.union(0, 2)  # no structural change to log
print("temporary:", dsu.count, dsu.find(0) == dsu.find(2))
dsu.rollback(checkpoint)
print("restored:", dsu.count, dsu.find(0) == dsu.find(2))
print("original merge retained:", dsu.find(0) == dsu.find(1))
dsu.rollback(0)
print("singletons:", dsu.parent, dsu.count)
`,
    expected: 'temporary: 2 True\nrestored: 3 False\noriginal merge retained: True\nsingletons: [0, 1, 2, 3] 4',
  },
  exercise: {
    title: 'Exercise solution: count disconnected unordered pairs after every link',
    code: `${dsuImplementation}
def disconnected_pairs(n, links):
    dsu = DisjointSet(n)
    remaining = n * (n - 1) // 2
    answers = []
    for a, b in links:
        root_a, root_b = dsu.find(a), dsu.find(b)
        if root_a != root_b:
            remaining -= dsu.size[root_a] * dsu.size[root_b]
            dsu.union(root_a, root_b)
        answers.append(remaining)
    return answers

print(disconnected_pairs(5, [(0, 1), (2, 3), (1, 2), (0, 3), (3, 4)]))
print(disconnected_pairs(0, []))
`,
    expected: '[9, 8, 4, 4, 0]\n[]',
  },
};
