// Authoring blueprint: Arrays, Strings & Hash Maps.
// Edit this topic-owned source; registration lives in ./index.js.
export default {
  "summary": "Choose positional, textual and keyed storage by its operations; trace element movement, Unicode units and collision-safe lookup before combining them into useful programs.",
  "outcomes": [
    "Distinguish length, capacity, value identity and position",
    "Count dynamic-array movement and separate individual from amortized costs",
    "Compare code points, visible characters, UTF-8 bytes and a declared normalization policy",
    "Explain hash/equality rules, replacement, collisions, occupancy and expected cost assumptions",
    "Build an inverted index and independently find the first unique integer event"
  ],
  "prerequisites": [
    "Python Basics: Types, Control Flow, Functions & Modules"
  ],
  "sequence": [
    "Start with a reading position, a recorded label and event frequency",
    "Derive base plus index times fixed slot width; distinguish CPython references, NumPy values and virtual contiguity",
    "Run indexing, insertion, deletion, append and half-open slicing with identity caveats",
    "Trace front/middle/end insertions, growth and deletion before introducing O(1)/O(n)",
    "Use geometric copy totals as an optional explanation of amortized append",
    "Compare code points, UTF-8 offsets and canonical normalization; retain raw labels under a declared matching policy",
    "Hash a key to a chained bucket then compare candidate equality; include missing lookup and existing replacement",
    "Run dict counting and equal numeric key/unhashable key cases",
    "Optionally force collisions with a custom class while preserving stable key identity",
    "Compare ordering, membership, cost assumptions and resizing; link formal hashing later",
    "Build word-to-document posting sets and intersect an AND query",
    "Independently count then scan a reusable integer sequence for its earliest unique ID"
  ],
  "visual": {
    "type": "Backing-array cells and source/destination arrows",
    "question": "What must move to preserve order?",
    "interaction": "Choose insertion position or deletion and spare/full capacity; step copying, shifting and insertion with counted writes."
  },
  "visuals": [
    {
      "type": "Text units linked to encoded bytes",
      "question": "Can two identical-looking labels differ in length and equality?",
      "interaction": "Select ASCII, composed/decomposed accented text or emoji, toggle NFC and inspect each code point and UTF-8 span."
    },
    {
      "type": "Bucket and equality trace",
      "question": "Why do colliding keys remain separate entries?",
      "interaction": "Choose bucket count, key and get/set; step candidates, comparison counts, replacement or absence."
    }
  ],
  "practice": {
    "task": "Run six full examples including a tiny inverted index; independently solve first-unique-event selection with empty, repeated, zero and changed-order cases.",
    "success": "Sequences and normalization agree with Python; collisions preserve unequal keys; posting sets count document presence, and the unique result preserves first-occurrence order."
  },
  "misconceptions": [
    "Position is not stable element identity",
    "Spare slots are not live elements",
    "Amortized does not promise each append has low latency",
    "Python len(str) is neither a byte count nor full grapheme count",
    "Equal hash does not mean equal key",
    "Insertion order is not sorted order",
    "A reusable sequence and a one-shot iterator have different two-pass contracts"
  ],
  "sources": [
    "https://docs.python.org/3/tutorial/datastructures.html",
    "https://docs.python.org/3/faq/design.html#how-are-lists-implemented-in-cpython",
    "https://docs.python.org/3/howto/unicode.html",
    "https://docs.python.org/3/library/unicodedata.html",
    "https://docs.python.org/3/library/stdtypes.html#mapping-types-dict",
    "https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/resources/lecture-4-hashing/"
  ],
  "depth": "core",
  "reviewFocus": "Three investigations, six programs; independent list/dict/encoding oracles. Separate-chaining and doubling are explicit models, not CPython layouts. Formal amortization, probing and string matching stay with existing later owners. MIT lecture videos/notes and Python Unicode tutorial provide alternate explanations. Optional custom keys need OOP; the core does not.",
  "designRecord": "docs/teaching/systems-and-structures-design.md"
};
