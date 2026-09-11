// Topic-owned plan; see original and extension design records.
export default {
  "summary": "Choose positional, textual and keyed storage by its operations; trace element movement, Unicode units and collision-safe lookup before combining them into useful programs. A deeper bitwise branch derives finite bitsets, signed-word interpretation, XOR promises, sparse population count and two-singleton partitioning.",
  "outcomes": [
    "Distinguish length, capacity, value identity and position",
    "Count dynamic-array movement and separate individual from amortized costs",
    "Compare code points, visible characters, UTF-8 bytes and a declared normalization policy",
    "Explain hash/equality rules, replacement, collisions, occupancy and expected cost assumptions",
    "Build an inverted index and independently find the first unique integer event",
    "Represent a finite set with bit positions and prove intersection/union/difference/XOR/complement membership",
    "Decode signed and unsigned words; distinguish explicit width from Python integer shifts and magnitude bit_count",
    "Prove one- and two-singleton XOR algorithms under their exact promises and diagnose invalid inputs",
    "Derive lowest-bit clearing, positive power tests and Hamming distance; account for arbitrary-integer bit costs"
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
    "Independently count then scan a reusable integer sequence for its earliest unique ID",
    "After the original independent task, offer a deeper representation branch without changing module order",
    "Derive binary place values and per-bit truth rules; compare bitmap postings with ordinary posting sets",
    "Derive two’s-complement signed weights, sign/zero fill and explicit truncation versus Python integers",
    "Prove the prefix-XOR invariant and parity cancellation; inspect negative/zero and invalid-promise examples",
    "Prove borrow-based lowest-bit clearing and population-count termination; apply XOR to Hamming distance",
    "Optionally isolate a distinguishing bit and recover two singletons in two passes",
    "Solve changed finite-set, word, promise and bitmap-report tasks before guided bitwise practice"
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
    },
    {
      "type": "Aligned finite-set membership columns",
      "question": "Which members satisfy this set query?",
      "interaction": "Toggle two sets, change the declared universe or query and inspect each resulting membership bit."
    },
    {
      "type": "Weighted word and source-position shift map",
      "question": "How can the same bits mean a negative value, and where do shifted bits go?",
      "interaction": "Toggle a four/eight-bit word and compare unsigned/signed sums and zero/sign-filled shift results."
    },
    {
      "type": "Parity stream with separate frequency audit",
      "question": "What does XOR preserve, and which promise turns it into a singleton?",
      "interaction": "Step bounded events, apply custom data, compare one/two/invalid-promise scenarios and inspect extra diagnostic storage."
    },
    {
      "type": "Borrow suffix and disappearing-bit trace",
      "question": "Why does one AND remove exactly one set position?",
      "interaction": "Build a byte and step x, x−1 and their AND while preserving the removed-plus-remaining count invariant."
    },
    {
      "type": "Two-bucket singleton partition figure",
      "question": "Why do pairs stay together while the two survivors separate?",
      "interaction": "A static computed split for the worked example makes the bucket and cancellation proof visible."
    }
  ],
  "practice": {
    "task": "Run six full examples including a tiny inverted index; independently solve first-unique-event selection with empty, repeated, zero and changed-order cases. The deeper branch has five additional complete native programs and four visible changed-contract tasks, plus four core and one optional verified bitwise problem.",
    "success": "Sequences and normalization agree with Python; collisions preserve unequal keys; posting sets count document presence, and the unique result preserves first-occurrence order. Bitmap queries match sets, word shifts match the declared interpretation, XOR answers are used only under valid promises, and count/partition witnesses agree with independent oracles."
  },
  "misconceptions": [
    "Position is not stable element identity",
    "Spare slots are not live elements",
    "Amortized does not promise each append has low latency",
    "Python len(str) is neither a byte count nor full grapheme count",
    "Equal hash does not mean equal key",
    "Insertion order is not sorted order",
    "A reusable sequence and a one-shot iterator have different two-pass contracts",
    "Arithmetic addition is not idempotent membership insertion",
    "Finite complement needs a universe",
    "Signed two’s complement is not sign plus magnitude",
    "XOR retains parity, not exact frequencies or first-occurrence order",
    "A bounded number of Python integers is not a constant bit budget",
    "Zero XOR/nonzero XOR alone does not validate a singleton promise"
  ],
  "sources": [
    "https://docs.python.org/3/tutorial/datastructures.html",
    "https://docs.python.org/3/faq/design.html#how-are-lists-implemented-in-cpython",
    "https://docs.python.org/3/howto/unicode.html",
    "https://docs.python.org/3/library/unicodedata.html",
    "https://docs.python.org/3/library/stdtypes.html#mapping-types-dict",
    "https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/resources/lecture-4-hashing/",
    "https://docs.python.org/3/library/stdtypes.html#bitwise-operations-on-integer-types",
    "https://cses.fi/book/book.pdf"
  ],
  "depth": "core",
  "reviewFocus": "Three investigations, six programs; independent list/dict/encoding oracles. Separate-chaining and doubling are explicit models, not CPython layouts. Formal amortization, probing and string matching stay with existing later owners. MIT lecture videos/notes and Python Unicode tutorial provide alternate explanations. Optional custom keys need OOP; the core does not. The bitwise extension preserves all original teaching and validates actual exported models/native programs against sets, binary-string arithmetic and frequencies; full desktop/narrow/keyboard reading plus independent review are required. Native integer rules follow Python, not the Handbook’s nonportable C++ width/overflow assumptions.",
  "designRecord": "docs/teaching/systems-and-structures-design.md",
  "extensionDesignRecord": "docs/teaching/BITWISE-FOUNDATIONS-EXTENSION-DESIGN.md"
};
