"""Independent contracts for the exact displayed Trees programs.

The oracles use finite ordered sets, path maps, ancestor intersections and graph
distances. They do not reuse the lesson's recursive summaries or validation.
Run through verify-tree-examples.mjs to create the exact source manifest.
"""
import bisect
import contextlib
import io
import itertools
import json
import math
import random
import sys
from collections import Counter, deque
from pathlib import Path

examples = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
namespaces = {}
for name, example in examples.items():
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(example["code"], f"displayed-{name}.py", "exec"), namespace)
    namespaces[name] = namespace
counts = Counter()


def inspect(root):
    """Path map independently checks finite single-parent shape and positions."""
    paths, identities = {}, set()
    pending = deque([(root, "")]) if root is not None else deque()
    while pending:
        node, path = pending.popleft()
        assert id(node) not in identities, "cycle or shared child"
        identities.add(id(node))
        paths[path] = node
        if node.left is not None:
            pending.append((node.left, path + "L"))
        if node.right is not None:
            pending.append((node.right, path + "R"))
    return paths


def ordered_values(root):
    paths = inspect(root)
    # Ordering all ancestors is intentionally separate from bound propagation.
    for path, node in paths.items():
        for depth, side in enumerate(path):
            ancestor = paths[path[:depth]]
            assert (node.key < ancestor.key) if side == "L" else (node.key > ancestor.key)
    return sorted(node.key for node in paths.values())


def raises_value_error(function, *args):
    try:
        function(*args)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


insert_orders = [()]
for size in range(1, 7):
    insert_orders.extend(itertools.permutations(range(size)))
insert_orders.extend([
    (8, 3, 10, 1, 6, 14, 4, 7, 13),
    (20, 10, 40, 30, 50, 35),
    (-10**100, 0, 10**100, -5, 5),
    (4, 4, 2, 2, 6, 6),
])

search_namespace = namespaces["search"]
deletion_namespace = namespaces["deletion"]
query_namespace = namespaces["orderedQueries"]
ceiling_namespace = namespaces["ceilingPractice"]
traversal_namespace = namespaces["traversals"]
for insertion_order in insert_orders:
    root = search_namespace["build"](insertion_order)
    values = sorted(set(insertion_order))
    assert ordered_values(root) == values
    assert search_namespace["inorder"](root) == values
    paths = inspect(root)
    for key in values:
        previous_ids = {id(node) for node in paths.values()}
        assert search_namespace["insert"](root, key) is root
        assert {id(node) for node in inspect(root).values()} == previous_ids
    queries = sorted(set(range(-2, 9)) | set(values))
    for target in queries:
        found, path = search_namespace["search"](root, target)
        assert found == (target in values)
        current = root
        for index, key in enumerate(path):
            assert current is not None and current.key == key
            if target == key:
                assert found and index == len(path) - 1
            else:
                current = current.left if target < key else current.right
        if not found:
            assert current is None
        floor_index = bisect.bisect_right(values, target) - 1
        ceiling_index = bisect.bisect_left(values, target)
        assert query_namespace["floor_key"](root, target) == (values[floor_index] if floor_index >= 0 else None)
        assert ceiling_namespace["ceiling_key"](root, target) == (values[ceiling_index] if ceiling_index < len(values) else None)
        counts["search_floor_ceiling_queries"] += 1
    for low, high in itertools.combinations_with_replacement((-2, 0, 2, 4, 8), 2):
        assert query_namespace["range_keys"](root, low, high) == [key for key in values if low <= key <= high]
        counts["range_queries"] += 1
    raises_value_error(query_namespace["range_keys"], root, 2, 1)
    for target in values + [-999, 999]:
        candidate = deletion_namespace["build"](insertion_order)
        before = inspect(candidate)
        before_ids = {id(node) for node in before.values()}
        target_node = next((node for node in before.values() if node.key == target), None)
        removed_node = target_node
        two_children = target_node is not None and target_node.left is not None and target_node.right is not None
        if two_children:
            # The set successor identifies the object that copy-delete unlinks.
            successor_key = values[bisect.bisect_right(values, target)]
            removed_node = next(node for node in before.values() if node.key == successor_key)
        result = deletion_namespace["delete"](candidate, target)
        after = inspect(result)
        assert ordered_values(result) == [key for key in values if key != target]
        expected_ids = before_ids - ({id(removed_node)} if removed_node is not None else set())
        assert {id(node) for node in after.values()} == expected_ids
        if two_children:
            assert id(target_node) in expected_ids and target_node.key == successor_key
        if target_node is None:
            assert result is candidate
        counts["deletions_with_identity_checks"] += 1
    # Lexicographic path markers generate the three emission orders without recursion.
    for order, marker in [("pre", "A"), ("in", "M"), ("post", "Z")]:
        expected = [node.key for path, node in sorted(paths.items(), key=lambda item: item[0] + marker)]
        assert traversal_namespace["depth_first"](root, order) == expected
    expected_levels = [[node.key for path, node in paths.items() if len(path) == depth]
                       for depth in range(max(map(len, paths), default=-1) + 1)]
    assert traversal_namespace["levels"](root) == expected_levels
    counts["insertion_shapes_and_traversals"] += 1

# General binary trees deliberately include duplicate and unordered keys.
randomizer = random.Random(70491)
Node = namespaces["summaries"]["Node"]


def random_tree(size):
    if size == 0:
        return None
    root = Node(randomizer.randrange(-3, 4))
    available = [(root, "left"), (root, "right")]
    for _ in range(size - 1):
        position = randomizer.randrange(len(available))
        parent, side = available.pop(position)
        child = Node(randomizer.randrange(-3, 4))
        setattr(parent, side, child)
        available.extend([(child, "left"), (child, "right")])
    return root


general_trees = [random_tree(size) for size in range(25) for _ in range(8)]
general_trees.extend([
    Node(10, Node(5, right=Node(12))),
    Node(10, right=Node(15, Node(8))),
    Node(10, Node(10)),
    Node(0, Node(-10**100), Node(10**100)),
    # Diameter belongs entirely inside the left subtree; height 4, diameter 6.
    Node(20, Node(10, Node(9, Node(8, Node(7))), Node(11, right=Node(12, right=Node(13))))),
])
for root in general_trees:
    paths = inspect(root)
    expected_valid = True
    try:
        ordered_values(root)
    except AssertionError:
        expected_valid = False
    assert namespaces["validation"]["is_bst"](root) == expected_valid
    adjacency = {path: [] for path in paths}
    for path in paths:
        if path:
            adjacency[path].append(path[:-1])
            adjacency[path[:-1]].append(path)
    diameter = 0
    for start in paths:
        queue, visited = deque([(start, 0)]), {start}
        while queue:
            path, distance = queue.popleft()
            diameter = max(diameter, distance)
            for neighbor in adjacency[path]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
    assert namespaces["summaries"]["summarize"](root) == (max(map(len, paths), default=-1), diameter)
    for first_path, second_path in itertools.product(paths, repeat=2):
        shared_length = 0
        for first_side, second_side in zip(first_path, second_path):
            if first_side != second_side:
                break
            shared_length += 1
        expected_ancestor = paths[first_path[:shared_length]]
        assert namespaces["summaries"]["common_ancestor"](root, paths[first_path], paths[second_path]) is expected_ancestor
        counts["lca_identity_pairs"] += 1
    # Independent preorder token oracle includes every missing child position.
    tokens_by_path = {path + "A": node.key for path, node in paths.items()}
    if not paths:
        tokens_by_path["A"] = None
    for path, node in paths.items():
        for side, child in [("L", node.left), ("R", node.right)]:
            if child is None:
                tokens_by_path[path + side + "A"] = None
    expected_tokens = [value for _, value in sorted(tokens_by_path.items())]
    encoded = namespaces["serialization"]["encode"](root)
    assert encoded == expected_tokens
    assert len(encoded) == 2 * len(paths) + 1
    assert encoded.count(None) == len(paths) + 1
    decoded = namespaces["serialization"]["decode"](expected_tokens)
    assert {path: node.key for path, node in inspect(decoded).items()} == {path: node.key for path, node in paths.items()}
    assert not ({id(node) for node in paths.values()} & {id(node) for node in inspect(decoded).values()})
    counts["arbitrary_tree_validation_summary_serialization_cases"] += 1

for tokens in [[], [1], [1, None], [None, None], [1, None, None, 2], [1, 2, None, None]]:
    raises_value_error(namespaces["serialization"]["decode"], tokens)
    counts["malformed_serializations"] += 1

for size in range(130):
    values = list(range(-size, size, 2))
    root = namespaces["balance"]["from_sorted"](values)
    paths = inspect(root)
    assert ordered_values(root) == values
    expected_height = size.bit_length() - 1
    assert max(map(len, paths), default=-1) == expected_height
    for path in paths:
        left_height = max((len(child_path) - len(path) - 1 for child_path in paths if child_path.startswith(path + "L")), default=-1)
        right_height = max((len(child_path) - len(path) - 1 for child_path in paths if child_path.startswith(path + "R")), default=-1)
        assert abs(left_height - right_height) <= 1
    counts["balanced_rebuild_sizes"] += 1

for insertion_order in insert_orders:
    root = namespaces["balance"]["build"](insertion_order)
    if root is None or root.left is None:
        raises_value_error(namespaces["balance"]["rotate_right"], root)
        counts["invalid_rotations"] += 1
        continue
    old_root, pivot, middle = root, root.left, root.left.right
    before_ids = {id(node) for node in inspect(root).values()}
    result = namespaces["balance"]["rotate_right"](root)
    assert result is pivot and result.right is old_root and old_root.left is middle
    assert ordered_values(result) == sorted(set(insertion_order))
    assert {id(node) for node in inspect(result).values()} == before_ids
    counts["rotations_with_identity_checks"] += 1

evaluate = namespaces["expression"]["evaluate"]
for left, right in itertools.product((-7, -0.5, 0, 2, 11), repeat=2):
    for operator, expected in [("+", left + right), ("-", left - right), ("*", left * right)]:
        assert evaluate(Node(operator, Node(left), Node(right))) == expected
        counts["expression_evaluations"] += 1
for malformed in [None, Node("literal"), Node("+", Node(1)), Node("/", Node(1), Node(2)), Node(1, Node(2), Node(3))]:
    raises_value_error(evaluate, malformed)
    counts["malformed_expressions"] += 1

# Deliberately exceed the usual recursion bound only for advertised iterative helpers.
deep = search_namespace["build"](range(1500))
assert search_namespace["search"](deep, 1499)[0]
assert search_namespace["inorder"](deep) == list(range(1500))
assert query_namespace["floor_key"](deep, 1600) == 1499
assert ceiling_namespace["ceiling_key"](deep, -1) == 0
counts["deep_iterative_chain_nodes"] = 1500
print(json.dumps({"independent_native_cases": dict(counts)}, sort_keys=True))
