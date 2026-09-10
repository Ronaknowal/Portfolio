import contextlib
import io
import itertools
import json
import random
import sys

programs = json.load(open(sys.argv[1], encoding="utf-8"))
namespaces = {}
for name, example in programs.items():
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(example["code"], name + ".py", "exec"), namespace)
    namespaces[name] = namespace

rng = random.Random(5643)
point = namespaces["point"]
lazy = namespaces["lazy"]
checks = 0
for size in range(9):
    for _ in range(50):
        initial = [rng.randrange(-10, 11) for _ in range(size)]
        roots = [point["build"](initial)]
        lazy_roots = [lazy["lazy_build"](initial)]
        arrays = [initial]
        lazy_arrays = [initial]
        for operation in range(12):
            parent = rng.randrange(len(roots))
            next_array = arrays[parent].copy()
            if size:
                index, value = rng.randrange(size), rng.randrange(-10, 11)
                old_nodes = point["identities"]([roots[parent]])
                result = point["assign"](roots[parent], index, value)
                new_nodes = point["identities"]([result]) - old_nodes
                path = [node for node in old_nodes if node.low <= index < node.high]
                assert len(new_nodes) == (len(path) if next_array[index] != value else 0)
                next_array[index] = value
            else:
                result = None
            roots.append(result)
            arrays.append(next_array)
            low, high = sorted([rng.randrange(size + 1), rng.randrange(size + 1)])
            delta = rng.randrange(-5, 6)
            lazy_result = lazy["range_add"](lazy_roots[parent], low, high, delta)
            lazy_next = lazy_arrays[parent].copy()
            for index in range(low, high):
                lazy_next[index] += delta
            lazy_roots.append(lazy_result)
            lazy_arrays.append(lazy_next)
            for version in range(len(roots)):
                for begin in range(size + 1):
                    for end in range(begin, size + 1):
                        assert point["range_sum"](roots[version], begin, end) == sum(arrays[version][begin:end])
                        assert lazy["lazy_sum"](lazy_roots[version], begin, end) == sum(lazy_arrays[version][begin:end])
                        checks += 2
                def check_lazy(node):
                    if node is None or node.left is None:
                        return
                    assert node.total == node.left.total + node.right.total + node.add * (node.high - node.low)
                    check_lazy(node.left)
                    check_lazy(node.right)
                check_lazy(lazy_roots[version])
        assert len(point["identities"]([roots[0]])) == max(0, 2 * size - 1)
        # Removing handles must not double-count common subtrees.
        independently_seen = set()
        pending = [root for root in roots[::2] if root is not None]
        while pending:
            node = pending.pop()
            if id(node) in independently_seen:
                continue
            independently_seen.add(id(node))
            pending.extend(child for child in (node.left, node.right) if child is not None)
        assert len(point["identities"](roots[::2])) == len(independently_seen)

ranks = 0
for size in range(6):
    for values in itertools.product((-1, 0, 1), repeat=size):
        query = namespaces["rank"]["PrefixRanks"](iter(values))
        for low in range(size):
            for high in range(low + 1, size + 1):
                expected = sorted(values[low:high])
                for rank, value in enumerate(expected, 1):
                    assert query.kth(low, high, rank) == value
                    ranks += 1

history_checks = 0
for size in range(1, 9):
    for _ in range(50):
        history = namespaces["history"]["SnapshotArray"](size)
        current = [0] * size
        snapshots = []
        for operation in range(100):
            if rng.randrange(3):
                index, value = rng.randrange(size), rng.randrange(-50, 51)
                history.set(index, value)
                current[index] = value
            else:
                assert history.snap() == len(snapshots)
                snapshots.append(current.copy())
            if snapshots:
                version = rng.randrange(len(snapshots))
                for index in range(size):
                    assert history.get(index, version) == snapshots[version][index]
                    history_checks += 1
        for writes in history.history:
            assert all(writes[i][0] < writes[i + 1][0] for i in range(len(writes) - 1))

root = point["build"]([10**60, -(10**60), 3])
assert point["range_sum"](point["assign"](root, 2, 10**80), 0, 3) == 10**80
try:
    root.total = 0
    raise AssertionError("Frozen node was mutable")
except AttributeError:
    pass
for operation in [lambda: point["assign"](None, 0, 1), lambda: point["assign"](root, -1, 1), lambda: point["range_sum"](root, 2, 1), lambda: namespaces["rank"]["PrefixRanks"]([]).kth(0, 0, 1), lambda: lazy["range_add"](None, 0, 1, 3), lambda: namespaces["history"]["SnapshotArray"](1).get(0, 0)]:
    try:
        operation()
        raise AssertionError("Invalid operation was accepted")
    except (IndexError, ValueError):
        pass
print(f"PASS: {checks} historical point/lazy sum checks, {ranks} exhaustive sorted-rank checks, {history_checks} copied-snapshot lookups; identity, allocation, frozen fields, empty/invalid and large-integer contracts.")
