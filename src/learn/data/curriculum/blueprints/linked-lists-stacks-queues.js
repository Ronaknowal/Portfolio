// Authoring blueprint: Linked Lists, Stacks & Queues.
// Edit this topic-owned source; registration lives in ./index.js.
export default {
  "summary": "Separate LIFO/FIFO interfaces from storage; reason about preserved node identity and invariants while rewiring links, matching brackets and reusing bounded queue slots.",
  "outcomes": [
    "State the known-reference condition behind constant-cost local edits",
    "Reverse an acyclic list without losing or duplicating nodes and return its new head",
    "Use an unmatched-prefix stack invariant to reject crossed brackets",
    "Distinguish logical FIFO order from wrapped physical slots and enforce a full-queue policy",
    "Combine map lookup with recency and independently remove a node or maintain a recent numerical window"
  ],
  "prerequisites": [
    "Arrays, Strings & Hash Maps",
    "Object-Oriented Programming in Python"
  ],
  "sequence": [
    "Introduce undo and waiting jobs as LIFO/FIFO interfaces with multiple storage choices",
    "Define node identity, next, head, None, aliasing and traversal",
    "Explain insertion/removal, head/tail cases, singly/doubly/circular variants and conditional costs",
    "Trace correct reversal alongside a lost-successor bug using previous/current/saved references",
    "State initialization, preservation and termination of the reversed-prefix invariant, then run full code",
    "Build bracket matching from newest unmatched opening through rejection and final emptiness",
    "Implement stack behavior using list ends; distinguish a bracket recognizer from a parser",
    "Explain deque versus list front shifts and derive circular head/size/tail indexing",
    "Trace full/empty/wrap boundaries, then run a bounded queue with rejection and legitimate None values",
    "Use OrderedDict for a two-entry LRU trace and distinguish recency from stale-data policy",
    "Independently remove only the first matching node using a sentinel or explicit head case",
    "Transfer FIFO membership into a recent-window mean with width, finite-input and numerical caveats"
  ],
  "visual": {
    "type": "Node arrows and named references",
    "question": "Which reference must be saved before changing next?",
    "interaction": "Step correct/broken reversal for three nodes, one node or empty input; see all identities even when a suffix becomes unreachable."
  },
  "visuals": [
    {
      "type": "Input cursor and unmatched-opening stack",
      "question": "Why are balanced counts insufficient?",
      "interaction": "Step valid, crossed, early-closing, unclosed and empty inputs; inspect matching top, rejection and final acceptance."
    },
    {
      "type": "Physical ring and logical FIFO sequence",
      "question": "Are equal head/tail indices empty or full?",
      "interaction": "Vary capacity three/four, step enqueue/dequeue through wrap, rejected full writes and rejected empty reads."
    }
  ],
  "practice": {
    "task": "Run six complete programs; independently remove a first matching node while preserving remaining identities, then compute recent means at changed widths.",
    "success": "Traversal is acyclic with preserved identity/order; a separate grammar oracle agrees with bracket decisions; every ring operation agrees with deque including full/empty and None; window results agree with direct slice means."
  },
  "misconceptions": [
    "A stack is not necessarily a linked list",
    "Local O(1) deletion omits any search for the predecessor",
    "Ignoring the returned head can hide the reversed list",
    "Matching bracket counts do not establish proper nesting",
    "Physical ring order differs from FIFO order",
    "A bounded deque may evict while this work queue rejects",
    "Unlinking does not eliminate every external reference",
    "LRU does not define expiration or freshness"
  ],
  "sources": [
    "https://cs50.harvard.edu/x/2025/notes/5/",
    "https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-stacks",
    "https://docs.python.org/3/library/collections.html#collections.deque",
    "https://docs.python.org/3/library/collections.html#collections.OrderedDict"
  ],
  "depth": "core",
  "reviewFocus": "Three mechanism-specific investigations, six programs, two independent tasks. No concurrency guarantees from single-threaded storage. Continue to Trees in this module, including its planned entry; prerequisite review links never silently reshuffle module order. CS50 lecture/notes use C; annotate that mismatch and never claim full-video viewing without evidence.",
  "designRecord": "docs/teaching/systems-and-structures-design.md"
};
