const treeSetup = `class Node:
    def __init__(self, key, left=None, right=None):
        self.key, self.left, self.right = key, left, right

def insert(root, key):
    if root is None:
        return Node(key)
    current = root
    while True:
        if key == current.key:
            return root  # This is a set: ignore an existing key.
        side = "left" if key < current.key else "right"
        child = getattr(current, side)
        if child is None:
            setattr(current, side, Node(key))
            return root
        current = child

def build(keys):
    root = None
    for key in keys:
        root = insert(root, key)
    return root

def inorder(root):
    result, stack, current = [], [], root
    while current is not None or stack:
        while current is not None:
            stack.append(current)
            current = current.left
        current = stack.pop()
        result.append(current.key)
        current = current.right
    return result

KEYS = [8, 3, 10, 1, 6, 14, 4, 7, 13]
`;

export const treeExamples = {
  search: {
    title: "Build an ordered set and explain a search path",
    code: `${treeSetup}
def search(root, key):
    path = []
    while root is not None:
        path.append(root.key)
        if key == root.key:
            return True, path
        root = root.left if key < root.key else root.right
    return False, path

root = build(KEYS)
print("find 7:", search(root, 7))
print("find 5:", search(root, 5))
root = insert(root, 6)
print("after duplicate:", inorder(root))
print("empty:", search(None, 7))
`,
    expected: "find 7: (True, [8, 3, 6, 7])\nfind 5: (False, [8, 3, 6, 4])\nafter duplicate: [1, 3, 4, 6, 7, 8, 10, 13, 14]\nempty: (False, [])",
  },
  traversals: {
    title: "One tree, four orders, and an explicit stack",
    code: `${treeSetup}
from collections import deque

def depth_first(root, order):
    result = []
    def visit(node):
        if node is None:
            return
        if order == "pre": result.append(node.key)
        visit(node.left)
        if order == "in": result.append(node.key)
        visit(node.right)
        if order == "post": result.append(node.key)
    visit(root)
    return result

def levels(root):
    if root is None:
        return []
    result, queue = [], deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.key)
            if node.left is not None: queue.append(node.left)
            if node.right is not None: queue.append(node.right)
        result.append(level)
    return result

root = build(KEYS)
for order in ["pre", "in", "post"]:
    print(order, depth_first(root, order))
print("levels", levels(root))
print("iterative matches", inorder(root) == depth_first(root, "in"))
`,
    expected: "pre [8, 3, 1, 6, 4, 7, 10, 14, 13]\nin [1, 3, 4, 6, 7, 8, 10, 13, 14]\npost [1, 4, 7, 6, 3, 13, 14, 10, 8]\nlevels [[8], [3, 10], [1, 6, 14], [4, 7, 13]]\niterative matches True",
  },
  deletion: {
    title: "Delete a key and reconnect the surviving subtree",
    code: `${treeSetup}
def delete(root, key):
    if root is None:
        return None
    if key < root.key:
        root.left = delete(root.left, key)
    elif key > root.key:
        root.right = delete(root.right, key)
    else:
        if root.left is None:
            return root.right
        if root.right is None:
            return root.left
        successor = root.right
        while successor.left is not None:
            successor = successor.left
        root.key = successor.key
        root.right = delete(root.right, successor.key)
    return root

root = build(KEYS)
original_root = root
root = delete(root, 8)
print("delete root:", inorder(root))
print("same object, new key:", root is original_root, root.key)
root = delete(root, 999)
print("missing key:", inorder(root))
# Here the successor (30) has a right child (35).
other = delete(build([20, 10, 40, 30, 50, 35]), 20)
print("successor child retained:", inorder(other))
print("singleton:", delete(Node(1), 1))
`,
    expected: "delete root: [1, 3, 4, 6, 7, 10, 13, 14]\nsame object, new key: True 10\nmissing key: [1, 3, 4, 6, 7, 10, 13, 14]\nsuccessor child retained: [10, 30, 35, 40, 50]\nsingleton: None",
  },
  validation: {
    title: "Carry all ancestor restrictions into a subtree",
    code: `${treeSetup}
def is_bst(root, lower=None, upper=None):
    if root is None:
        return True
    if lower is not None and root.key <= lower:
        return False
    if upper is not None and root.key >= upper:
        return False
    return (is_bst(root.left, lower, root.key)
            and is_bst(root.right, root.key, upper))

bad = Node(10, Node(5, right=Node(12)))
duplicate = Node(10, Node(10))
print("valid:", is_bst(build(KEYS)))
print("parent-only trap:", is_bst(bad))
print("duplicate:", is_bst(duplicate))
print("empty:", is_bst(None))
`,
    expected: "valid: True\nparent-only trap: False\nduplicate: False\nempty: True",
  },
  orderedQueries: {
    title: "Find a compatible version and prune an inclusive range",
    code: `${treeSetup}
def floor_key(root, target):
    candidate = None
    while root is not None:
        if root.key == target:
            return root.key
        if root.key < target:
            candidate = root.key
            root = root.right
        else:
            root = root.left
    return candidate

def range_keys(root, low, high):
    if low > high:
        raise ValueError("low must not exceed high")
    result = []
    def visit(node):
        if node is None: return
        if low < node.key: visit(node.left)
        if low <= node.key <= high: result.append(node.key)
        if node.key < high: visit(node.right)
    visit(root)
    return result

root = build(KEYS)
print("floor 5:", floor_key(root, 5))
print("floor 0:", floor_key(root, 0))
print("range [4, 10]:", range_keys(root, 4, 10))
print("range [9, 9]:", range_keys(root, 9, 9))
`,
    expected: "floor 5: 4\nfloor 0: None\nrange [4, 10]: [4, 6, 7, 8, 10]\nrange [9, 9]: []",
  },
  expression: {
    title: "An expression tree preserves grouping without BST ordering",
    code: `${treeSetup}
def evaluate(node):
    if node is None:
        raise ValueError("missing expression")
    if node.left is None and node.right is None:
        if not isinstance(node.key, (int, float)):
            raise ValueError("leaf must be numeric")
        return node.key
    left = evaluate(node.left)
    right = evaluate(node.right)
    if node.key == "+": return left + right
    if node.key == "-": return left - right
    if node.key == "*": return left * right
    raise ValueError("unsupported operator")

expression = Node("*", Node("+", Node(2), Node(3)),
                  Node("-", Node(9), Node(4)))
print("(2 + 3) * (9 - 4) =", evaluate(expression))
`,
    expected: "(2 + 3) * (9 - 4) = 25",
  },
  summaries: {
    title: "Return a subtree result: height, diameter and common ancestor",
    code: `${treeSetup}
def summarize(root):
    if root is None:
        return -1, 0  # height in edges, diameter in edges
    left_height, left_diameter = summarize(root.left)
    right_height, right_diameter = summarize(root.right)
    height = 1 + max(left_height, right_height)
    through_root = left_height + right_height + 2
    return height, max(left_diameter, right_diameter, through_root)

def common_ancestor(root, first, second):
    # General binary tree; both supplied node identities must be in the tree.
    if root is None or root is first or root is second:
        return root
    left = common_ancestor(root.left, first, second)
    right = common_ancestor(root.right, first, second)
    return root if left is not None and right is not None else left or right

root = build(KEYS)
print("height, diameter:", summarize(root))
print("empty:", summarize(None))
print("ancestor of 4 and 7:", common_ancestor(root, root.left.right.left,
                                              root.left.right.right).key)
`,
    expected: "height, diameter: (3, 6)\nempty: (-1, 0)\nancestor of 4 and 7: 6",
  },
  balance: {
    title: "Separate a sorted sequence from a shape; rotate without changing order",
    code: `${treeSetup}
def height(node):
    return -1 if node is None else 1 + max(height(node.left), height(node.right))

def from_sorted(keys):
    def make(low, high):  # Half-open index interval; no list slices.
        if low >= high: return None
        middle = (low + high) // 2
        return Node(keys[middle], make(low, middle), make(middle + 1, high))
    return make(0, len(keys))

def rotate_right(root):
    if root is None or root.left is None:
        raise ValueError("right rotation needs a left child")
    pivot = root.left
    root.left = pivot.right
    pivot.right = root
    return pivot

keys = list(range(1, 8))
skewed = build(keys)
balanced = from_sorted(keys)
print("same order:", inorder(skewed) == inorder(balanced))
print("heights:", height(skewed), height(balanced))
tree = build([30, 20, 40, 10, 25])
before = inorder(tree)
tree = rotate_right(tree)
print("rotation:", tree.key, inorder(tree) == before, tree.right.left.key)
`,
    expected: "same order: True\nheights: 6 2\nrotation: 20 True 25",
  },
  ceilingPractice: {
    title: "Solution: the smallest available key at least as large as a request",
    code: `${treeSetup}
def ceiling_key(root, target):
    candidate = None
    while root is not None:
        if root.key == target: return root.key
        if root.key > target:
            candidate = root.key
            root = root.left
        else:
            root = root.right
    return candidate

root = build(KEYS)
for target in [0, 5, 6, 15]:
    print(target, ceiling_key(root, target))
assert ceiling_key(None, 5) is None
assert ceiling_key(build([2]), 2) == 2
`,
    expected: "0 1\n5 6\n6 6\n15 None",
  },
  serialization: {
    title: "Record missing children so a binary tree can be rebuilt",
    code: `${treeSetup}
def encode(root):
    tokens = []
    def visit(node):
        if node is None:
            tokens.append(None)
            return
        tokens.append(node.key)
        visit(node.left)
        visit(node.right)
    visit(root)
    return tokens

def decode(tokens):
    position = 0
    def read():
        nonlocal position
        if position == len(tokens): raise ValueError("truncated tree")
        key = tokens[position]
        position += 1
        return None if key is None else Node(key, read(), read())
    result = read()
    if position != len(tokens): raise ValueError("trailing tokens")
    return result

tree = Node(2, right=Node(1))  # A binary tree, deliberately not a BST.
tokens = encode(tree)
print(tokens)
print("same shape and values:", encode(decode(tokens)) == tokens)
print("empty:", encode(None))
`,
    expected: "[2, None, 1, None, None]\nsame shape and values: True\nempty: [None]",
  },
};
