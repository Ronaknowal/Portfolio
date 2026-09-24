"""Execute complete teaching programs and save their captured output, not guesses."""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'scratch/external-memory-native'
FOLDER.mkdir(parents=True, exist_ok=True)
examples = {}

def add(name, title, code):
    code = textwrap.dedent(code).strip() + '\n'
    path = FOLDER / f'{name}.py'
    path.write_text(code, encoding='utf-8')
    result = subprocess.run([sys.executable, '-X', 'utf8', '-I', str(path)], capture_output=True, text=True, encoding='utf-8', check=True)
    examples[name] = {'title': title, 'code': code.rstrip(), 'expected': result.stdout.rstrip()}

CACHE = '''
from collections import OrderedDict

def buffer_counts(requests, page_size, capacity):
    if page_size < 1 or capacity < 1:
        raise ValueError("positive page size and capacity required")
    resident = OrderedDict()
    reads = writes = hits = 0
    for address, operation in requests:
        page = address // page_size
        if page in resident:
            hits += 1
            dirty = resident.pop(page)
        else:
            reads += 1  # write-allocate also reads the existing page
            dirty = False
            if len(resident) == capacity:
                _, evicted_dirty = resident.popitem(last=False)
                writes += int(evicted_dirty)
        resident[page] = dirty or operation == "write"
    pending = sum(resident.values())
    return reads, writes, pending, hits
'''
add('locality', 'Same records, different page transfers', CACHE + '''
sequential = [(address, "read") for address in range(16)]
strided = [(column + 4 * row, "read")
           for column in range(4) for row in range(4)]
for capacity in [1, 2, 4]:
    print("frames", capacity,
          "sequential reads", buffer_counts(sequential, 4, capacity)[0],
          "strided reads", buffer_counts(strided, 4, capacity)[0])
print("empty:", buffer_counts([], 4, 2))
''')
add('writeback', 'Repeated writes share a dirty page', CACHE + '''
requests = [(0, "write"), (1, "write"), (4, "read"),
            (0, "write"), (8, "write"), (12, "read"), (0, "read")]
for capacity in [1, 2, 4]:
    reads, evictions, pending, hits = buffer_counts(requests, 4, capacity)
    print("frames", capacity, "reads", reads, "dirty evictions", evictions,
          "final flush writes", pending, "total writes", evictions + pending,
          "hits", hits)
''')

BTREE = '''
from bisect import bisect_left

class Page:
    def __init__(self, keys=None, children=None):
        self.keys = [] if keys is None else keys
        self.children = [] if children is None else children

class BTree:
    """Integer set. Each Page models one block; Python objects are not disk pages."""
    def __init__(self, degree=2):
        if degree < 2:
            raise ValueError("minimum degree must be at least two")
        self.t = degree
        self.root = Page()
        self.events = []

    def search(self, key):
        page = self.root
        visits = 0
        while True:
            visits += 1
            index = bisect_left(page.keys, key)
            if index < len(page.keys) and page.keys[index] == key:
                return True, visits
            if not page.children:
                return False, visits
            page = page.children[index]

    def split(self, parent, index):
        child = parent.children[index]
        middle = self.t - 1
        separator = child.keys[middle]
        right = Page(child.keys[middle + 1:], child.children[self.t:])
        child.keys = child.keys[:middle]
        child.children = child.children[:self.t]
        parent.keys.insert(index, separator)
        parent.children.insert(index + 1, right)
        self.events.append("split")

    def insert(self, key):
        if self.search(key)[0]:
            return False
        if len(self.root.keys) == 2 * self.t - 1:
            self.root = Page(children=[self.root])
            self.split(self.root, 0)
        page = self.root
        while page.children:
            index = bisect_left(page.keys, key)
            if len(page.children[index].keys) == 2 * self.t - 1:
                self.split(page, index)
                if key > page.keys[index]:
                    index += 1
            page = page.children[index]
        page.keys.insert(bisect_left(page.keys, key), key)
        return True

    def merge(self, parent, index):
        left, right = parent.children[index:index + 2]
        left.keys += [parent.keys.pop(index)] + right.keys
        left.children += right.children
        parent.children.pop(index + 1)
        self.events.append("merge")
        return left

    def remove(self, key):
        # A precheck gives missing-key deletion an unchanged-shape contract.
        if not self.search(key)[0]:
            return False

        def descend(page, target):
            index = bisect_left(page.keys, target)
            if index < len(page.keys) and page.keys[index] == target:
                if not page.children:
                    page.keys.pop(index)
                elif len(page.children[index].keys) >= self.t:
                    predecessor = page.children[index]
                    while predecessor.children:
                        predecessor = predecessor.children[-1]
                    replacement = predecessor.keys[-1]
                    page.keys[index] = replacement
                    self.events.append("predecessor")
                    descend(page.children[index], replacement)
                elif len(page.children[index + 1].keys) >= self.t:
                    successor = page.children[index + 1]
                    while successor.children:
                        successor = successor.children[0]
                    replacement = successor.keys[0]
                    page.keys[index] = replacement
                    self.events.append("successor")
                    descend(page.children[index + 1], replacement)
                else:
                    descend(self.merge(page, index), target)
                return

            child = page.children[index]
            if len(child.keys) == self.t - 1:
                left = page.children[index - 1] if index else None
                right = page.children[index + 1] if index + 1 < len(page.children) else None
                if left and len(left.keys) >= self.t:
                    child.keys.insert(0, page.keys[index - 1])
                    page.keys[index - 1] = left.keys.pop()
                    if left.children:
                        child.children.insert(0, left.children.pop())
                    self.events.append("borrow-left")
                elif right and len(right.keys) >= self.t:
                    child.keys.append(page.keys[index])
                    page.keys[index] = right.keys.pop(0)
                    if right.children:
                        child.children.append(right.children.pop(0))
                    self.events.append("borrow-right")
                else:
                    if right is None:
                        index -= 1
                    child = self.merge(page, index)
            descend(child, target)

        descend(self.root, key)
        if not self.root.keys and self.root.children:
            self.root = self.root.children[0]
            self.events.append("shrink-root")
        return True

    def ordered(self):
        def walk(page):
            if not page.children:
                return list(page.keys)
            result = []
            for index, child in enumerate(page.children):
                result.extend(walk(child))
                if index < len(page.keys):
                    result.append(page.keys[index])
            return result
        return walk(self.root)
'''
add('btreeUpdates', 'A complete B-tree set: search, split, borrow and merge', BTREE + '''
tree = BTree(2)
for value in [10, 20, 5, 6, 12, 30, 7, 17]:
    tree.insert(value)
print("ordered:", tree.ordered())
print("root keys:", tree.root.keys)
print("find 17:", tree.search(17), "find 99:", tree.search(99))
print("duplicate:", tree.insert(10), "missing delete:", tree.remove(99))
for value in [6, 7, 5, 10, 12, 17, 20, 30]:
    tree.remove(value)
print("after all removals:", tree.ordered())
print("repairs observed:", sorted(set(tree.events)))
''')

BPLUS = '''
from bisect import bisect_right

def balanced_groups(values, capacity):
    count = (len(values) + capacity - 1) // capacity
    if count == 0:
        return []
    small, extra = divmod(len(values), count)
    groups, offset = [], 0
    for index in range(count):
        size = small + (index < extra)
        groups.append(values[offset:offset + size])
        offset += size
    return groups

def bulk_index(records, leaf_capacity=3, fanout=3):
    if leaf_capacity < 2 or fanout < 2:
        raise ValueError("capacities must be at least two")
    if records != sorted(set(records)):
        raise ValueError("bulk input must be sorted unique keys")
    if not records:
        return None, []
    pages = [{"id": f"L{i}", "keys": group, "children": []}
             for i, group in enumerate(balanced_groups(records, leaf_capacity))]
    leaves = pages[:]
    for index, leaf in enumerate(leaves):
        leaf["next"] = leaves[index + 1]["id"] if index + 1 < len(leaves) else None

    def minimum(page):
        while page["children"]:
            page = page["children"][0]
        return page["keys"][0]

    next_id = len(leaves)
    while len(pages) > 1:
        parents = []
        for children in balanced_groups(pages, fanout):
            parents.append({"id": f"I{next_id}", "children": children,
                            "keys": [minimum(child) for child in children[1:]]})
            next_id += 1
        pages = parents
    return pages[0], leaves

def range_query(root, leaves, low, high):
    if root is None or low > high:
        return [], []
    visited, result = [], []
    page = root
    while page["children"]:
        visited.append(page["id"])
        page = page["children"][bisect_right(page["keys"], low)]
    by_id = {leaf["id"]: leaf for leaf in leaves}  # simulated disk page directory
    while page is not None:
        visited.append(page["id"])
        result.extend(value for value in page["keys"] if low <= value <= high)
        if page["keys"][-1] > high:
            break
        page = by_id.get(page["next"])
    return result, visited
'''
add('leafRanges', 'Bulk B+ pages and an inclusive range', BPLUS + '''
records = list(range(2, 36, 3))
root, leaves = bulk_index(records)
for low, high in [(10, 27), (20, 20), (36, 40), (9, 3)]:
    result, visits = range_query(root, leaves, low, high)
    print((low, high), result, "pages", visits)
print("empty index:", range_query(*bulk_index([]), 0, 9))
''')

add('fanout', 'Translate a hypothetical page layout into capacity', '''
from math import ceil

def child_capacity(page_bytes, header_bytes, key_bytes, pointer_bytes):
    # c child pointers and c−1 separators; records are in B+ leaves.
    return (page_bytes - header_bytes + key_bytes) // (key_bytes + pointer_bytes)

capacity = child_capacity(4096, 64, 8, 8)
print("maximum children:", capacity)
print("bytes used:", 64 + capacity * 8 + (capacity - 1) * 8)
print("one more child:", 64 + (capacity + 1) * 8 + capacity * 8)
for degree in [2, 16, 126]:
    # A nonempty minimum-degree B-tree of edge-height h has at least 2*t**h−1 keys.
    height = 0
    while 2 * degree ** (height + 1) - 1 <= 1_000_000:
        height += 1
    print("degree", degree, "maximum edge height at N=1000000:", height,
          "cold path pages at most", height + 1)
''')

EXTERNAL_SORT = '''
import heapq
import struct
import tempfile
from pathlib import Path

def external_sort(values, page_records=4, memory_pages=3):
    """Binary signed-64-bit records; materializes every run/pass, including singleton groups.
    Counts nonempty logical block reads/writes, not physical disk I/O or durable flushes.
    Python object/heap/interpreter overhead is outside the abstract record-page budget.
    """
    if page_records < 1 or memory_pages < 3:
        raise ValueError("positive page size and at least three buffer pages required")
    memory_records = page_records * memory_pages
    fan_in = memory_pages - 1
    counters = {"reads": 0, "writes": 0}
    stages = []
    serial = 0
    with tempfile.TemporaryDirectory(prefix="external-sort-") as directory:
        directory = Path(directory)

        def filename():
            nonlocal serial
            serial += 1
            return directory / f"run-{serial}.bin"

        def write_run(target, records):
            buffer = []
            with target.open("wb") as stream:
                for value in records:
                    buffer.append(value)
                    if len(buffer) == page_records:
                        stream.write(struct.pack("<" + "q" * len(buffer), *buffer))
                        counters["writes"] += 1
                        buffer.clear()
                if buffer:
                    stream.write(struct.pack("<" + "q" * len(buffer), *buffer))
                    counters["writes"] += 1

        def read_run(source):
            with source.open("rb") as stream:
                while True:
                    raw = stream.read(8 * page_records)
                    if not raw:
                        return
                    if len(raw) % 8:
                        raise ValueError("truncated signed-64-bit record")
                    counters["reads"] += 1
                    yield from struct.unpack("<" + "q" * (len(raw) // 8), raw)

        source = filename()
        write_run(source, values)
        # Initial input creation is outside the measured sorting operation.
        counters = {"reads": 0, "writes": 0}
        runs, chunk = [], []
        before = dict(counters)
        for value in read_run(source):
            chunk.append(value)
            if len(chunk) == memory_records:
                target = filename()
                write_run(target, sorted(chunk))
                runs.append(target)
                chunk.clear()
        if chunk:
            target = filename()
            write_run(target, sorted(chunk))
            runs.append(target)
        if runs:
            stages.append((len(runs), counters["reads"], counters["writes"]))
        while len(runs) > 1:
            before = dict(counters)
            output_runs = []
            for offset in range(0, len(runs), fan_in):
                inputs = [read_run(source) for source in runs[offset:offset + fan_in]]
                target = filename()
                write_run(target, heapq.merge(*inputs))
                output_runs.append(target)
            runs = output_runs
            stages.append((len(runs), counters["reads"] - before["reads"],
                           counters["writes"] - before["writes"]))
        sorting_counts = dict(counters)
        # Loading output for this tiny demonstration/oracle is outside sort I/O.
        result = list(read_run(runs[0])) if runs else []
    return result, stages, sorting_counts
'''
add('externalSort', 'Sort real temporary binary files with bounded merge fan-in', EXTERNAL_SORT + '''
values = list(range(31, -1, -1))
result, stages, counts = external_sort(values, 4, 3)
print("pass tuples (runs, reads, writes):", stages)
print("sort transfers:", counts)
print("sorted correctly:", result == sorted(values))
print("partial and duplicates:", external_sort([8, -1, 8, 2, 0, -1, 3], 2, 3)[0])
print("empty:", external_sort([], 4, 3))
''')

add('batchJoin', 'A sorted merge can avoid repeated full scans', '''
def merge_unique_join(left, right):
    """Sorted unique (key,value) records; duplicate join keys need group handling."""
    i = j = 0
    output = []
    while i < len(left) and j < len(right):
        if left[i][0] < right[j][0]:
            i += 1
        elif left[i][0] > right[j][0]:
            j += 1
        else:
            output.append((left[i][0], left[i][1], right[j][1]))
            i += 1
            j += 1
    return output

measurements = [(1, 12), (3, 19), (5, 8), (9, 15)]
calibration = [(2, "B"), (3, "C"), (5, "E"), (8, "H")]
print(merge_unique_join(measurements, calibration))
print("empty:", merge_unique_join([], calibration))
print("one match:", merge_unique_join([(3, 7)], [(3, "C")]))
''')

add('crashRoots', 'Recover a shadow-page root after each interrupted stage', '''
def crash_states(early_root=False):
    pages = {"oldRoot": "oldLeaf", "oldLeaf": 5,
             "newRoot": "newLeaf", "newLeaf": 9}
    durable = {"oldRoot", "oldLeaf"}
    root = "oldRoot"
    result = []

    def observe(label):
        page = root
        while page in durable and isinstance(pages[page], str):
            page = pages[page]
        value = pages[page] if page in durable else "invalid root reachability"
        result.append((label, value))

    observe("initial")
    observe("prepare volatile pages")
    if early_root:
        root = "newRoot"  # assumed atomic durable metadata publication
        observe("publish too early")
    durable.add("newLeaf")
    observe("leaf durable")
    durable.add("newRoot")
    observe("tree root page durable")
    if not early_root:
        root = "newRoot"
        observe("publish metadata")
    observe("acknowledge")
    return result

for early in [False, True]:
    print("early publication:", early)
    for label, value in crash_states(early):
        print(" ", label, "->", value)
''')

target = ROOT / 'src/learn/data/external-memory-examples.js'
target.write_text('export const externalMemoryExamples = ' + json.dumps(examples, indent=2, ensure_ascii=False) + ';\n', encoding='utf-8')
(FOLDER / 'captured.json').write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps({name: example['expected'] for name, example in examples.items()}, indent=2))
