"""Independent finite contracts for the exact displayed heap/trie programs.

References are sorted multisets, insertion-ticket event lists and string sets.
No oracle reimplements sift-up/down or recursive trie suggestion collection.
"""
import contextlib
import heapq
import io
import itertools
import json
import random
import sys
from collections import Counter
from pathlib import Path

examples = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
namespaces = {}
for name, example in examples.items():
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(example["code"], f"displayed-{name}.py", "exec"), namespace)
    namespaces[name] = namespace
counts = Counter()
randomizer = random.Random(35811)


def raises(exception_type, function, *args):
    try:
        function(*args)
    except exception_type:
        return
    raise AssertionError(f"expected {exception_type.__name__}")


def assert_heap(heap, expected):
    assert Counter(heap.items) == Counter(expected), "retain every occurrence"
    for child in range(1, len(heap.items)):
        assert heap.items[(child - 1) // 2] <= heap.items[child]


MinHeap = namespaces["heapOperations"]["MinHeap"]
input_sequences = [list(values) for size in range(7)
                   for values in itertools.product((-2, 0, 2), repeat=size)]
input_sequences.extend([[-10**100, 10**100, 0], [1.25, -0.5, 2.75, 1.25], list(range(60)), list(range(60, -1, -1))])
for values in input_sequences:
    original = list(values)
    heap = MinHeap(values)
    assert heap.items is not values
    assert_heap(heap, values)
    assert values == original
    popped = []
    while heap.items:
        minimum = heap.pop()
        popped.append(minimum)
        assert_heap(heap, sorted(values)[len(popped):])
        counts["heap_pop_states"] += 1
    assert popped == sorted(values)
    raises(IndexError, heap.pop)
    inserted = []
    for value in reversed(values + [3, -3, 0]):
        heap.push(value)
        inserted.append(value)
        assert_heap(heap, inserted)
        counts["heap_push_states"] += 1
    assert [heap.pop() for _ in range(len(heap.items))] == sorted(inserted)
    iterable_heap = MinHeap(iter(values))
    assert_heap(iterable_heap, values)
    sortable = list(values)
    assert namespaces["heapSort"]["heap_sort"](sortable) is None
    assert sortable == sorted(values)
    for k in sorted({1, 2, 3, len(values) + 1}):
        states = list(namespaces["topK"]["top_k_states"](iter(values), k))
        assert len(states) == len(values)
        for consumed, (retained, kth) in enumerate(states, start=1):
            expected = sorted(values[:consumed], reverse=True)[:k]
            assert retained == expected
            assert kth == (expected[-1] if consumed >= k else None)
            counts["largest_k_prefix_states"] += 1
    for k in sorted({0, 1, 2, 3, len(values) + 1}):
        assert namespaces["smallestPractice"]["smallest_k"](iter(values), k) == sorted(values)[:k]
        counts["smallest_k_cases"] += 1
    counts["heap_build_and_sort_sequences"] += 1

for invalid in [0, -1, 1.5, "2", None, True, False]:
    raises(ValueError, lambda: list(namespaces["topK"]["top_k_states"]([2, 1], invalid)))
for invalid in [-1, 1.5, "2", None, True, False]:
    raises(ValueError, namespaces["smallestPractice"]["smallest_k"], [2, 1], invalid)


class CountedIterator:
    def __init__(self, values):
        self.values = iter(values)
        self.consumed = 0

    def __iter__(self):
        return self

    def __next__(self):
        value = next(self.values)
        self.consumed += 1
        return value


unconsumed = CountedIterator([1, 2, 3])
assert namespaces["smallestPractice"]["smallest_k"](unconsumed, 0) == []
assert unconsumed.consumed == 0
stream = CountedIterator([3, 1, 5])
stream_states = namespaces["topK"]["top_k_states"](stream, 2)
assert stream.consumed == 0
assert next(stream_states) == ([3], None) and stream.consumed == 1
assert next(stream_states) == ([3, 1], 1) and stream.consumed == 2
counts["iterator_consumption_contracts"] += 3

# Stable mutable priorities: reference chooses a minimum from current live jobs.
PriorityQueue = namespaces["stablePriorities"]["PriorityQueue"]
jobs = ["alpha", 7, ("tuple", 1), None, object(), object()]
for trial in range(100):
    queue, live, next_ticket = PriorityQueue(), {}, 0
    for step in range(100):
        operation = randomizer.choice(("put", "put", "put", "cancel", "pop", "compact"))
        job = randomizer.choice(jobs)
        if operation == "put":
            priority = randomizer.randrange(-4, 5)
            queue.put(job, priority)
            live[job] = (priority, next_ticket)
            next_ticket += 1
        elif operation == "cancel":
            assert queue.cancel(job) == (job in live)
            live.pop(job, None)
        elif operation == "compact":
            queue.compact()
            assert len(queue.heap) == len(live)
        elif live:
            expected_job = min(live, key=lambda candidate: live[candidate])
            assert queue.pop() == (expected_job, live[expected_job][0])
            del live[expected_job]
        else:
            raises(KeyError, queue.pop)
        assert set(queue.live) == set(live)
        counts["priority_update_cancel_tie_operations"] += 1
    while live:
        expected_job = min(live, key=lambda candidate: live[candidate])
        assert queue.pop() == (expected_job, live[expected_job][0])
        del live[expected_job]
    raises(KeyError, queue.pop)
    queue.compact()
    assert queue.heap == []

merge_sorted = namespaces["mergeStreams"]["merge_sorted"]
for trial in range(300):
    sources = [sorted(randomizer.randrange(-9, 10) for _ in range(randomizer.randrange(12)))
               for _ in range(randomizer.randrange(9))]
    expected = sorted(itertools.chain.from_iterable(sources))
    assert list(merge_sorted(iter(source) for source in sources)) == expected
    counts["merge_multiset_cases"] += 1
tracked_sources = [CountedIterator([1, 4]), CountedIterator([]), CountedIterator([2, 3])]
merged = merge_sorted(tracked_sources)
assert all(source.consumed == 0 for source in tracked_sources)
assert next(merged) == 1 and [source.consumed for source in tracked_sources] == [1, 0, 1]
assert next(merged) == 2 and [source.consumed for source in tracked_sources] == [2, 0, 1]
assert list(merged) == [3, 4]
counts["iterator_consumption_contracts"] += 3

# Schedule admission and the published event loop agree with a sorted event list.
event_namespace = namespaces["eventSchedule"]
for trial in range(100):
    event_namespace["events"] = []
    event_namespace["tickets"] = itertools.count()
    event_namespace["now"] = 0
    expected = []
    next_ticket = 0
    for index in range(randomizer.randrange(1, 15)):
        time = randomizer.randrange(0, 8)
        name = f"event-{index}"
        followups = tuple((randomizer.randrange(4), f"child-{index}-{child}") for child in range(randomizer.randrange(3)))
        event_namespace["schedule"](time, name, followups)
        expected.append((time, next_ticket, name, followups))
        next_ticket += 1
    actual_order, expected_order = [], []
    # Independent reference repeatedly sorts a simple event collection.
    reference = list(expected)
    while reference:
        reference.sort(key=lambda event: (event[0], event[1]))
        time, _, name, followups = reference.pop(0)
        expected_order.append((time, name))
        for delay, child_name in followups:
            reference.append((time + delay, next_ticket, child_name, ()))
            next_ticket += 1
    while event_namespace["events"]:
        time, _, name, followups = heapq.heappop(event_namespace["events"])
        event_namespace["now"] = time
        actual_order.append((time, name))
        for delay, child_name in followups:
            event_namespace["schedule"](time + delay, child_name)
    assert actual_order == expected_order
    raises(ValueError, event_namespace["schedule"], event_namespace["now"] - 1, "past")
    assert not event_namespace["events"]
    counts["stable_event_schedules"] += 1

Trie = namespaces["trieOperations"]["Trie"]
matching_prefix = namespaces["prefixPolicies"]["matching_prefix"]
words = ["", "a", "app", "apple", "applesauce", "car", "cart", "cat", "dog", "map", "é", "e\u0301", "🧠", "🧠lab", "👩\u200d💻", "A", "a/b", "a.b"]
queries = words + ["ca", "ap", "z", "carts", "applecart", "\u200d", "e", "👩"]


def trie_words(trie):
    result, seen = set(), set()
    pending = [(trie.root, "")]
    while pending:
        node, text = pending.pop()
        assert id(node) not in seen, "trie nodes cannot alias or form a cycle"
        seen.add(id(node))
        if node.terminal:
            result.add(text)
        for character, child in node.children.items():
            assert isinstance(character, str) and len(character) == 1
            pending.append((child, text + character))
        if node is not trie.root:
            assert node.terminal or node.children, "unused leaf should have been pruned"
    return result


for trial in range(60):
    trie, expected = Trie(), set()
    for step in range(40):
        word = randomizer.choice(words)
        if randomizer.randrange(3):
            trie.insert(word)
            expected.add(word)
        else:
            assert trie.delete(word) == (word in expected)
            expected.discard(word)
        assert trie_words(trie) == expected
        for query in queries:
            assert trie.contains(query) == (query in expected)
            assert trie.starts_with(query) == (query == "" or any(word.startswith(query) for word in expected))
            matches = sorted(word for word in expected if word.startswith(query))
            for limit in [0, 1, 2, 3, 20]:
                assert trie.suggestions(query, limit) == matches[:limit]
                counts["trie_suggestion_queries"] += 1
            prefix_matches = [word for word in expected if query.startswith(word)]
            assert matching_prefix(trie, query) == (min(prefix_matches, key=len) if prefix_matches else None)
            assert matching_prefix(trie, query, longest=True) == (max(prefix_matches, key=len) if prefix_matches else None)
            counts["trie_exact_prefix_policy_queries"] += 1
        counts["trie_mutations_with_pruning_checks"] += 1
    for invalid in [-1, 2.5, "2", None, True, False]:
        raises(ValueError, trie.suggestions, "", invalid)
        counts["invalid_suggestion_limits"] += 1

trie = Trie()
for word in ["car", "cart", "carton"]:
    trie.insert(word)
assert trie.delete("car") and trie.contains("cart") and trie.contains("carton")
assert trie.delete("carton") and trie.contains("cart") and not trie.starts_with("carto")
assert trie.delete("cart") and trie.root.children == {}
trie.insert("é")
assert not trie.contains("e\u0301"), "Unicode normalization is not implicit"
trie.insert("e\u0301")
assert trie.suggestions("") == ["e\u0301", "é"], "ordering uses code points, not locale collation"
deep_word = "a" * 1500
trie.insert(deep_word)
assert trie.contains(deep_word) and trie.starts_with(deep_word[:-1])
assert trie.delete(deep_word) and not trie.starts_with("a")
counts["deep_iterative_trie_code_points"] = len(deep_word)
print(json.dumps({"independent_native_cases": dict(counts)}, sort_keys=True))
