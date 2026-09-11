// Topic-owned blueprint; original design remains in docs/teaching/systems-and-structures-design.md.
export default {
  "summary": "Separate LIFO/FIFO interfaces from storage; reason about preserved node identity and invariants while rewiring links, matching brackets and reusing bounded queue slots.",
  "outcomes": [
    "State the known-reference condition behind constant-cost local edits",
    "Reverse an acyclic list without losing or duplicating nodes and return its new head",
    "Use an unmatched-prefix stack invariant to reject crossed brackets",
    "Distinguish logical FIFO order from wrapped physical slots and enforce a full-queue policy",
    "Combine map lookup with recency and independently remove a node or maintain a recent numerical window",
    "Find a cycle entry without changing links using a proved constant-space two-phase traversal",
    "Derive first/second even-middle guards and perform a guarded identity-preserving left-heavy split",
    "Resolve first strictly-greater future distances with a duplicate-aware monotonic stack and total-work proof",
    "Derive both strictly-smaller boundaries and certify a maximum unit-width histogram rectangle"
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
    "Transfer FIFO membership into a recent-window mean with width, finite-input and numerical caveats",
    "Extend visited-identity detection into a positive fast/slow meeting, reset-to-entry proof and cycle-length calculation",
    "Derive both middle conventions and a left-heavy split with a nonmutation-on-cycle rejection contract",
    "Compare nested forward searches with an unresolved-index stack, strict/equal behavior and aggregate pop accounting",
    "Transfer to permanently dominated smaller-boundary candidates, both directional scans and histogram completeness",
    "Independently change cycle prefix/length, split aliases, equality and tied rectangle optima before new official practice"
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
    },
    {
      "type": "Successor path, return edge and two named reference phases",
      "question": "Why can the first positive meeting differ from the entry?",
      "interaction": "Step detection/reset/entry with empty, nullable, head-entry and self-loop fixtures; values can repeat without changing identities."
    },
    {
      "type": "Acyclic chain and explicit cut boundary",
      "question": "Which even middle satisfies the query, and which link yields a left-heavy split?",
      "interaction": "Vary length and first/second policy; inspect the separately labeled cut and both returned heads."
    },
    {
      "type": "Reading bars and unresolved index stack",
      "question": "Which questions can one arrival settle and why are equal values different?",
      "interaction": "Step arrivals/pops/pushes, compare strict/equal requirements and inspect exact distances and operation counts."
    },
    {
      "type": "Candidate stack, smaller blockers and unit-width rectangle",
      "question": "Why can a popped candidate be forgotten, and why does the boundary rectangle reach an optimum?",
      "interaction": "Step each directional scan and vary the limiting bar; inspect equal plateaus, zero/empty inputs and tied maxima."
    }
  ],
  "practice": {
    "task": "Run six complete programs; independently remove a first matching node while preserving remaining identities, then compute recent means at changed widths. Retain those six programs and add complete cycle/middle/future-distance/boundary programs plus changed entry, alias, equality and tied-optimum tasks.",
    "success": "Traversal is acyclic with preserved identity/order; a separate grammar oracle agrees with bracket decisions; every ring operation agrees with deque including full/empty and None; window results agree with direct slice means. A separate first-repeat oracle matches cycle entry without mutation; split outputs partition the original identities; direct forward scans and exhaustive intervals verify distances, boundaries and rectangle witnesses."
  },
  "misconceptions": [
    "A stack is not necessarily a linked list",
    "Local O(1) deletion omits any search for the predecessor",
    "Ignoring the returned head can hide the reversed list",
    "Matching bracket counts do not establish proper nesting",
    "Physical ring order differs from FIFO order",
    "A bounded deque may evict while this work queue rejects",
    "Unlinking does not eliminate every external reference",
    "LRU does not define expiration or freshness",
    "Initial pointer equality does not prove a cycle",
    "A meeting node need not be the cycle entry",
    "Changing the fast speed invalidates the same reset proof",
    "Second middle and a left-heavy split have different contracts",
    "Equal values do not answer a strictly-greater question or form a strictly-smaller blocker",
    "An expensive arrival can coexist with linear total stack work"
  ],
  "sources": [
    "https://cs50.harvard.edu/x/2025/notes/5/",
    "https://docs.python.org/3/tutorial/datastructures.html#using-lists-as-stacks",
    "https://docs.python.org/3/library/collections.html#collections.deque",
    "https://docs.python.org/3/library/collections.html#collections.OrderedDict",
    "https://cses.fi/book/book.pdf",
    "https://leetcode.com/problems/linked-list-cycle-ii/",
    "https://leetcode.com/problems/daily-temperatures/"
  ],
  "depth": "core",
  "reviewFocus": "Preserve all original linked/reversal/bracket/ring/LRU/window coverage and six native programs; extend locally with fast/slow entry and middle proofs plus monotonic future/boundary stacks. Distinguish identity/value, query/mutation, strict/equal policies and invariant roles. Historical three-lab evidence remains unchanged; the extension has its own scoped design and native/browser/independent review. Keep Trees next in module order. No branching graph, concurrency or universal interview mastery claim.",
  "designRecord": "docs/teaching/LINKED-TRAVERSAL-MONOTONIC-EXTENSION-DESIGN.md"
};
