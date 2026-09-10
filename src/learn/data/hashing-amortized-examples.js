export const hashingAmortizedExamples = {
  listMap: {
    title: "Start with a correct map whose costs are easy to see",
    code: String.raw`class ListMap:
    """Integer keys; arbitrary values. The boolean distinguishes absence."""
    def __init__(self):
        self.entries = []

    def put(self, key, value):
        for index, (stored, _) in enumerate(self.entries):
            if stored == key:
                self.entries[index] = (key, value)
                return False
        self.entries.append((key, value))
        return True

    def find(self, key):
        for stored, value in self.entries:
            if stored == key:
                return True, value
        return False, None

    def remove(self, key):
        for index, (stored, _) in enumerate(self.entries):
            if stored == key:
                self.entries.pop(index)
                return True
        return False


table = ListMap()
print("new, replace:", table.put(9, 90), table.put(9, None))
print("stored None:", table.find(9))
print("absent:", table.find(10))
print("remove, repeat:", table.remove(9), table.remove(9))`,
    expected: "new, replace: True False\nstored None: (True, None)\nabsent: (False, None)\nremove, repeat: True False",
  },
  chainedMap: {
    title: "Grow and shrink a chained map without losing key/value associations",
    code: String.raw`class ChainedMap:
    """Integer keys, explicit modulo hash; no expected-time promise for this hash."""
    def __init__(self):
        self.buckets = [[] for _ in range(4)]
        self.size = 0
        self.moved = 0

    def _check_key(self, key):
        if type(key) is not int:
            raise TypeError("keys must be integers")

    def _rebuild(self, capacity):
        fresh = [[] for _ in range(capacity)]
        for bucket in self.buckets:
            for key, value in bucket:
                # Existing keys are already unique: do not search each new chain.
                fresh[key % capacity].append((key, value))
                self.moved += 1
        self.buckets = fresh

    def put(self, key, value):
        self._check_key(key)
        bucket = self.buckets[key % len(self.buckets)]
        for index, (stored, _) in enumerate(bucket):
            if stored == key:
                bucket[index] = (key, value)
                return False
        if self.size == len(self.buckets):
            self._rebuild(2 * len(self.buckets))
        self.buckets[key % len(self.buckets)].append((key, value))
        self.size += 1
        return True

    def find(self, key):
        self._check_key(key)
        for stored, value in self.buckets[key % len(self.buckets)]:
            if stored == key:
                return True, value
        return False, None

    def remove(self, key):
        self._check_key(key)
        bucket = self.buckets[key % len(self.buckets)]
        for index, (stored, _) in enumerate(bucket):
            if stored == key:
                bucket.pop(index)
                self.size -= 1
                if len(self.buckets) > 4 and self.size <= len(self.buckets) // 4:
                    self._rebuild(len(self.buckets) // 2)
                return True
        return False


table = ChainedMap()
for key in (1, 9, 17, 25, 33):
    table.put(key, key * 10)
print("capacity after growth:", len(table.buckets))
print("collision chain:", table.buckets[1])
print("replace:", table.put(17, None), table.find(17), "size:", table.size)
for key in (1, 9, 25):
    table.remove(key)
print("capacity after shrink:", len(table.buckets))
print("survivors:", sorted(entry for bucket in table.buckets for entry in bucket))
print("entries moved in rebuilds:", table.moved)`,
    expected: "capacity after growth: 8\ncollision chain: [(1, 10), (9, 90), (17, 170), (25, 250), (33, 330)]\nreplace: False (True, None) size: 5\ncapacity after shrink: 4\nsurvivors: [(17, None), (33, 330)]\nentries moved in rebuilds: 6",
  },
  probeMap: {
    title: "Implement bounded probing, tombstones and explicit rebuilding",
    code: String.raw`EMPTY = object()
DELETED = object()


class ProbeMap:
    """Fixed capacity, integer keys; rebuild is explicit, not automatic."""
    def __init__(self, capacity=8):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.slots = [EMPTY] * capacity
        self.size = 0

    def _locate(self, key):
        if type(key) is not int:
            raise TypeError("keys must be integers")
        capacity = len(self.slots)
        first_deleted = None
        for offset in range(capacity):
            position = (key % capacity + offset) % capacity
            entry = self.slots[position]
            if entry is EMPTY:
                return None, position if first_deleted is None else first_deleted
            if entry is DELETED:
                if first_deleted is None:
                    first_deleted = position
            elif entry[0] == key:
                return position, None
        return None, first_deleted

    def put(self, key, value):
        found, free = self._locate(key)
        if found is not None:
            self.slots[found] = (key, value)
            return False
        if free is None:
            raise OverflowError("no free slot; rebuild or remove first")
        self.slots[free] = (key, value)
        self.size += 1
        return True

    def find(self, key):
        found, _ = self._locate(key)
        return (False, None) if found is None else (True, self.slots[found][1])

    def remove(self, key):
        found, _ = self._locate(key)
        if found is None:
            return False
        self.slots[found] = DELETED
        self.size -= 1
        return True

    def items(self):
        return [entry for entry in self.slots
                if entry is not EMPTY and entry is not DELETED]

    def rebuild(self, capacity):
        if type(capacity) is not int or capacity < max(1, self.size):
            raise ValueError("new capacity must hold every live entry")
        fresh = ProbeMap(capacity)
        for key, value in self.items():
            fresh.put(key, value)
        # Publish only after the new representation has been constructed.
        self.slots = fresh.slots


table = ProbeMap(8)
for key in (1, 9, 17):
    table.put(key, key * 10)
table.remove(1)
print("lookup past deletion:", table.find(17))
print("replace past deletion:", table.put(17, 171), "size:", table.size)
table.put(25, None)
print("stored None:", table.find(25))
print("before rebuild:", sorted(table.items()))
table.rebuild(16)
print("after rebuild:", sorted(table.items()))
print("positions:", [(index, entry[0]) for index, entry in enumerate(table.slots)
                     if entry is not EMPTY and entry is not DELETED])
full = ProbeMap(2)
full.put(-1, 0)
full.put(1, 10)
print("absent in full table:", full.find(3))
try:
    full.put(3, 30)
except OverflowError as error:
    print("full insertion:", error)
full.remove(-1)
print("reuse without EMPTY:", full.put(3, 30), sorted(full.items()))`,
    expected: "lookup past deletion: (True, 170)\nreplace past deletion: False size: 2\nstored None: (True, None)\nbefore rebuild: [(9, 90), (17, 171), (25, None)]\nafter rebuild: [(9, 90), (17, 171), (25, None)]\npositions: [(1, 17), (9, 25), (10, 9)]\nabsent in full table: (False, None)\nfull insertion: no free slot; rebuild or remove first\nreuse without EMPTY: True [(1, 10), (3, 30)]",
  },
  deletionFailure: {
    title: "Expose an incorrect deletion and a duplicate insertion",
    code: String.raw`EMPTY, DELETED = object(), object()


def find(slots, key):
    for offset in range(len(slots)):
        position = (key + offset) % len(slots)
        entry = slots[position]
        if entry is EMPTY:
            return False
        if entry is not DELETED and entry[0] == key:
            return True
    return False


original = [EMPTY, (1, 10), (9, 90), (17, 170), EMPTY, EMPTY, EMPTY, EMPTY]
bad, good = list(original), list(original)
bad[1] = EMPTY
good[1] = DELETED
print("erase-to-EMPTY lookup of 17:", find(bad, 17))
print("tombstone lookup of 17:", find(good, 17))

# Incorrect insertion that stops at the first reusable slot creates a duplicate.
bad_update = list(good)
bad_update[1] = (17, 171)
print("stored copies of key 17:", sum(entry is not EMPTY and entry is not DELETED
                                    and entry[0] == 17 for entry in bad_update))
print("probe positions with step 2, capacity 8:", [(1 + 2*i) % 8 for i in range(8)])
print("probe positions with step 3, capacity 8:", [(1 + 3*i) % 8 for i in range(8)])`,
    expected: "erase-to-EMPTY lookup of 17: False\ntombstone lookup of 17: True\nstored copies of key 17: 2\nprobe positions with step 2, capacity 8: [1, 3, 5, 7, 1, 3, 5, 7]\nprobe positions with step 3, capacity 8: [1, 4, 7, 2, 5, 0, 3, 6]",
  },
  hashFamily: {
    title: "Enumerate every hash choice for one fixed key set",
    code: String.raw`from collections import Counter
from fractions import Fraction


def home(key, a, b):
    return ((a * key + b) % 17) % 4


keys, query = [1, 5, 9, 13], 9
lengths = []
pair_collisions = Counter()
for a in range(1, 17):
    for b in range(17):
        colliders = [key for key in keys if home(key, a, b) == home(query, a, b)]
        lengths.append(len(colliders))
        for key in colliders:
            if key != query:
                pair_collisions[key] += 1
print("hash functions:", len(lengths))
print("chain length -> number of functions:", sorted(Counter(lengths).items()))
print("mean candidate chain length:", Fraction(sum(lengths), len(lengths)))
print("bound 1+(n-1)/m:", Fraction(1) + Fraction(len(keys)-1, 4))
print("collision probabilities:", [(key, str(Fraction(count, len(lengths))))
                                  for key, count in sorted(pair_collisions.items())])
# The same fixed modulo home sends all four keys to bucket 1.
print("fixed modulo homes:", [key % 4 for key in keys])`,
    expected: "hash functions: 272\nchain length -> number of functions: [(1, 144), (2, 98), (3, 20), (4, 10)]\nmean candidate chain length: 55/34\nbound 1+(n-1)/m: 7/4\ncollision probabilities: [(1, '7/34'), (5, '7/34'), (13, '7/34')]\nfixed modulo homes: [1, 1, 1, 1]",
  },
  resizeAccounting: {
    title: "Count geometric growth and repeated shrink/grow work",
    code: String.raw`def simulate(operations, shrink="quarter", growth="double"):
    length, capacity, total = 0, 1, 0
    records = []
    for operation in operations:
        copies, cost = 0, 1
        if operation == "+":
            if length == capacity:
                copies = length
                capacity = 2*capacity if growth == "double" else capacity+1
            length += 1
        elif operation == "-" and length:
            length -= 1
            threshold = capacity/2 if shrink == "half" else capacity/4
            if shrink != "never" and capacity > 1 and length <= threshold:
                capacity //= 2
                copies = length
        elif operation == "-":
            cost = 0
        else:
            raise ValueError("operations must be + or -")
        total += cost + copies
        records.append((length, capacity, copies, cost+copies, total))
    return records


appends = simulate("+" * 24, shrink="never")
print("doubling: writes, copies, total:", 24, sum(row[2] for row in appends), appends[-1][-1])
print("final potential 2n-C+1:", 2*appends[-1][0]-appends[-1][1]+1)
print("3 charges per append:", 3*len(appends))
one_at_a_time = simulate("+" * 24, shrink="never", growth="one")
print("add-one capacity: copies, total:", sum(row[2] for row in one_at_a_time), one_at_a_time[-1][-1])
mixed = "+" * 8 + "+-" * 8
for rule in ("half", "quarter"):
    records = simulate(mixed, shrink=rule)
    print(rule, "length, capacity, total:", records[-1][0], records[-1][1], records[-1][-1])
drain = simulate("+" * 16 + "-" * 16)
print("empty after growth/drain:", drain[-1][0], "capacity:", drain[-1][1])`,
    expected: "doubling: writes, copies, total: 24 31 55\nfinal potential 2n-C+1: 17\n3 charges per append: 72\nadd-one capacity: copies, total: 276 300\nhalf length, capacity, total: 8 8 159\nquarter length, capacity, total: 8 16 39\nempty after growth/drain: 0 capacity: 1",
  },
  denseRandomSet: {
    title: "Maintain a dense set with a reverse index and uniform sampling",
    code: String.raw`import random


class DenseRandomSet:
    """Unordered integer set. Array positions and reverse lookup must agree."""
    def __init__(self, seed=0):
        self.items = []
        self.positions = {}
        self.generator = random.Random(seed)

    def add(self, value):
        if value in self.positions:
            return False
        self.positions[value] = len(self.items)
        self.items.append(value)
        return True

    def remove(self, value):
        if value not in self.positions:
            return False
        position = self.positions[value]
        last = self.items[-1]
        self.items[position] = last
        self.positions[last] = position
        self.items.pop()
        del self.positions[value]
        return True

    def sample(self):
        if not self.items:
            raise ValueError("cannot sample an empty set")
        return self.generator.choice(self.items)


values = DenseRandomSet(seed=11)
for value in (10, 30, 20):
    print("new:", values.add(value))
print("duplicate:", values.add(30))
values.remove(30)
print("after middle removal:", values.items, sorted(values.positions.items()))
print("sample is a member:", values.sample() in values.positions)
values.remove(20)
print("only remaining outcome:", values.sample())
values.remove(10)
try:
    values.sample()
except ValueError as error:
    print("empty:", error)`,
    expected: "new: True\nnew: True\nnew: True\nduplicate: False\nafter middle removal: [10, 20] [(10, 0), (20, 1)]\nsample is a member: True\nonly remaining outcome: 10\nempty: cannot sample an empty set",
  },
  consecutiveRuns: {
    title: "Charge each distinct value to one maximal run",
    code: String.raw`def longest_consecutive(values):
    remaining = set(values)
    best = 0
    membership_checks = 0
    for start in remaining:
        membership_checks += 1
        if start - 1 in remaining:
            continue
        end = start
        while True:
            membership_checks += 1
            if end not in remaining:
                break
            end += 1
        best = max(best, end - start)
    return best, membership_checks


for values in ([11, 4, 3, 2, 4, 10], [], [0, 0], [-3, -2, -1, 2]):
    length, checks = longest_consecutive(values)
    print("input:", values, "longest:", length, "membership checks:", checks)`,
    expected: "input: [11, 4, 3, 2, 4, 10] longest: 3 membership checks: 12\ninput: [] longest: 0 membership checks: 0\ninput: [0, 0] longest: 1 membership checks: 3\ninput: [-3, -2, -1, 2] longest: 3 membership checks: 10",
  },
  pairCounts: {
    title: "Count index tuples by retaining pair multiplicities",
    code: String.raw`def count_quadruples(first, second, third, fourth, target=0):
    pair_counts = {}
    for left in first:
        for right in second:
            total = left + right
            pair_counts[total] = pair_counts.get(total, 0) + 1
    answer = 0
    for left in third:
        for right in fourth:
            answer += pair_counts.get(target - (left + right), 0)
    return answer


print("two identical choices in each list:", count_quadruples([1, 1], [2, 2], [-1, -1], [-2, -2]))
print("unequal lengths:", count_quadruples([0, 1], [0], [0, -1], [0]))
print("empty group:", count_quadruples([], [0], [0], [0]))`,
    expected: "two identical choices in each list: 16\nunequal lengths: 2\nempty group: 0",
  },
  keyContracts: {
    title: "Keep composite identity separate from its hash",
    code: String.raw`from dataclasses import dataclass


@dataclass(frozen=True)
class EventKey:
    source: str
    sequence: int


events = {EventKey("A", 4): "first"}
events[EventKey("A", 4)] = "replaced"
events[EventKey("B", 4)] = "independent"
print("entry count:", len(events))
print("equal composite key:", events[EventKey("A", 4)])
print("other source:", events[EventKey("B", 4)])
try:
    events[["A", 4]] = "invalid"
except TypeError:
    print("mutable list key: rejected")
# A hash narrows candidates. Delimiter-free string concatenation is not a key schema.
print("ambiguous concatenation:", "ab" + "c" == "a" + "bc")
print("unambiguous tuple:", ("ab", "c") == ("a", "bc"))`,
    expected: "entry count: 2\nequal composite key: replaced\nother source: independent\nmutable list key: rejected\nambiguous concatenation: True\nunambiguous tuple: False",
  }
};
