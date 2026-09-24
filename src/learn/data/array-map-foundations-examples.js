export const arrayMapExamples={
  sequence:{code:`items = ["A", "B", "C"]
print("index 1:", items[1])
items.insert(1, "X")
print("insert:", items)
removed = items.pop(2)
print("removed:", removed)
items.append("D")
print("final:", items)
print("slice:", items[1:3])
assert items == ["A", "X", "C", "D"]`,output:`index 1: B
insert: ['A', 'X', 'B', 'C']
removed: B
final: ['A', 'X', 'C', 'D']
slice: ['X', 'C']`},
  unicode:{code:`import unicodedata

composed = "caf\u00e9"
decomposed = "cafe\u0301"
print("exact equality:", composed == decomposed)
print("code points:", len(composed), len(decomposed))
print("UTF-8 bytes:", len(composed.encode("utf-8")), len(decomposed.encode("utf-8")))
normalized = unicodedata.normalize("NFC", decomposed)
print("after NFC:", normalized == composed)
print("accent retained:", normalized)

# A declared label-matching policy, not a universal identity rule.
def label_key(raw):
    return unicodedata.normalize("NFC", raw.strip().casefold())

counts = {}
for raw in [" Temp ", "temp", "cafe\u0301", "caf\u00e9", "TEMP"]:
    key = label_key(raw)
    counts[key] = counts.get(key, 0) + 1
print("counts:", sorted(counts.items()))`,output:`exact equality: False
code points: 4 5
UTF-8 bytes: 5 6
after NFC: True
accent retained: café
counts: [('café', 2), ('temp', 3)]`},
  keys:{code:`counts = {}
for event_id in [10, 14, 10, 18, 14, 10]:
    counts[event_id] = counts.get(event_id, 0) + 1
print("counts:", sorted(counts.items()))
print("missing lookup:", counts.get(22, 0))
print("22 inserted by get:", 22 in counts)
counts[14] = 99
print("replacement:", counts[14], "keys:", len(counts))

# Equal numeric keys name the same entry, despite different Python types.
numeric = {1: "integer", True: "boolean", 1.0: "float"}
print("equal numeric keys:", len(numeric), numeric[1])
try:
    counts[[10, 14]] = 1
except TypeError:
    print("a mutable list is not a dictionary key")`,output:`counts: [(10, 3), (14, 2), (18, 1)]
missing lookup: 0
22 inserted by get: False
replacement: 99 keys: 3
equal numeric keys: 1 float
a mutable list is not a dictionary key`},
  collisions:{code:`class Key:
    def __init__(self, number):
        self.number = number

    def __hash__(self):
        return 0  # Deliberately poor distribution, for this demonstration only.

    def __eq__(self, other):
        if not isinstance(other, Key):
            return NotImplemented
        return self.number == other.number

table = {Key(10): 2, Key(14): 5, Key(18): 1}
print("distinct colliding keys:", len(table))
print("lookup 18:", table[Key(18)])
table[Key(14)] = 99
print("after equal-key update:", len(table), table[Key(14)])
print("absent key:", Key(22) not in table)`,output:`distinct colliding keys: 3
lookup 18: 1
after equal-key update: 3 99
absent key: True`},
  index:{code:`documents = ["red fox red", "blue fox", "red blue"]
postings = {}
for document_id, text in enumerate(documents):
    for word in text.split():
        postings.setdefault(word, set()).add(document_id)
for word in sorted(postings):
    print(word, "->", sorted(postings[word]))
matches = postings.get("red", set()) & postings.get("fox", set())
print("red AND fox:", sorted(matches))
assert matches == {0}
assert postings["red"] == {0, 2}`,output:`blue -> [1, 2]
fox -> [0, 1]
red -> [0, 2]
red AND fox: [0]`},
  unique:{code:`def first_unique(events):
    counts = {}
    for event in events:
        counts[event] = counts.get(event, 0) + 1
    # Iterate the original sequence to preserve first-occurrence order.
    for event in events:
        if counts[event] == 1:
            return event
    return None

for events in [[7, 2, 7, 9, 2, 4], [], [0, 1, 1], [3, 3]]:
    print(events, "->", first_unique(events))
assert first_unique([7, 2, 7, 9, 2, 4]) == 9`,output:`[7, 2, 7, 9, 2, 4] -> 9
[] -> None
[0, 1, 1] -> 0
[3, 3] -> None`},
};
