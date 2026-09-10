# Heaps and tries: visual and model contract

10 September 2026. Topic `heaps-priority-queues-tries`; second of the user's three requested DSA implementations. Read with the [individual lesson design](HEAPS-TRIES-LESSON-DESIGN.md), current teaching standard and engineering policy. The formats here address this topic's mechanisms and are not a reusable quota.

## Why these representations

| Difficulty | Visual form | Learning purpose |
| --- | --- | --- |
| Heap shape is complete, but its array is not sorted. | Linked array cells and actual complete-tree geometry, sharing zero-based indices. Select an array position and inspect its parent/children. | Distinguish a fixed position from the value moving through it. The positional formulas are visible relationships rather than isolated definitions. |
| Repairs compare specific relatives, then swap along one path. | `HeapOperationsLab` steps push, pop or bottom-up construction, highlighting compared/swapped positions in both views and reporting remaining violations. | Predict the next repair; observe smaller-child choice, equal priorities, a lone child and intermediate violations. An output-only trace would conceal which relationships were repaired. |
| Bottom-up construction does not give every node the root's height. | `HeapConstructionFigure` groups the actual 15 indices by computed subtree height and displays count × height contributions. | Show the shrinking number of tall subtrees and the resulting downward-work budget. This is a structural count, not an empirical performance graph. |
| A min-heap can retain the largest k, and duplicates count. | `TopKStreamLab` uses an occurrence ribbon, current candidate, retained heap, discarded history and prominent kth boundary. | The decision concerns the smallest retained item; the stream/decision view differs from the mechanical repair trace. Candidate inspection and committed prefix results stay distinct. |
| Character paths and complete-word membership are separate. | `TrieSharingFigure` shows character-labelled edges from left to right, prefix text at nodes and terminal double rings/checkmarks. | `car`, `cart` and `cat` share paths while retaining independent word endings. A static inline picture establishes that vocabulary before controls. |
| Lookup, insertion and deletion follow the same character path but update different state. | `TriePrefixLab` links consumed query characters, sparse prefix geometry, terminal flags, exact tables and stored-word/suggestion results. | Compare exact versus prefix queries; clear a word marker without deleting descendants; inspect safe suffix pruning. |

## Files and pure model API

- [heap-trie-models.js](../../src/learn/data/heap-trie-models.js): pure trace and layout functions; no browser side effects, timers, external engines or learner-code execution.
- [HeapTrieLabs.jsx](../../src/learn/components/lesson-labs/HeapTrieLabs.jsx) and [heap-trie-labs.css](../../src/learn/components/lesson-labs/heap-trie-labs.css): the five named, self-contained lesson exports above; scoped styling only.
- [verify-heap-trie-models.mjs](../../scripts/verify-heap-trie-models.mjs): independent reference checks.

`heapOperationTrace(values, operation, key)` supports `push`, `pop` and `build`. Frames contain copied `heap` values, operation/key, active/compared/swapped indices, cumulative comparison/swap counts, popped value, phase/note and result. `heapViolations` reports existing parent/child violations. `heapLayout` derives complete-tree positions and edges from array indices. `heapConstructionProfile` computes subtree heights and their counts directly from the array shape.

`topKStreamTrace(stream, k)` records heap entries `{value, sourceIndex}` so equal values remain distinct occurrences. Frames distinguish `current` candidate from `processed` committed inputs and preserve discarded identities. `prefixComplete` marks each committed result. `kth` is null until at least k values have arrived. It is the retained minimum thereafter, including during inspection of the next uncommitted candidate.

A trie is `{rootId, nextId, nodes: [{id, prefix, terminal, children}]}` with sparse character-to-child references. `buildTrie`, `trieWords`, `validateTrie`, `trieLayout` and `trieOperationTrace` separate storage, enumeration, integrity, geometry and execution. Supported operations are `exact`, `prefix`, `insert`, `delete`. Frames copy the full state and include consumed query, active/visited identities, optional current edge, matches and removed IDs. Each snapshot is isolated from other snapshots and caller state.

## Heap investigation contracts

**Starting point.** Valid min-heap `[1, 2, 9, 7, 5]`, pushing 0. Predict two swaps: index 5 with 2, then 2 with 0, producing `[0, 2, 1, 7, 5, 9]`. Popping the original heap returns 1 and leaves `[2, 5, 9, 7]`. The unordered preset `[7, 2, 9, 1, 5]` uses bottom-up build and ends at `[1, 2, 9, 7, 5]`.

**Mechanism.** Insertion appends, then compares upward. Pop removes the last cell, replaces the root if nonempty, then compares downward with the smaller existing child. Equal children choose left; equal parent/child priorities stop repair. Build starts at `floor(n/2) − 1` and repairs internal subtrees in reverse index order. Existing children already head valid heaps when their parent is processed. Comparing siblings and comparing the chosen child to its parent are separate frames/counts.

**Input and state.** Draft array, operation and push value apply through Run heap operation. Presets apply their stated trace immediately. Reset restores the default. Running again starts from the input array; completing a trace does not silently change that input. Back selects a prior copied snapshot. Invalid numeric text preserves the active trace. A syntactically valid but unordered input is deliberately rejected for push/pop with a repair-first explanation; Build accepts it. Empty pop is a visible empty result here, explicitly contrasted with Python's IndexError.

**Readability and inspection.** Native SVG value/index sizes are preserved through local horizontal scrolling; labels are not shrunk to unreadable sizes on narrow screens. Array cells are keyboard buttons, and the selected index exposes parent/child links in text. Amber means active; dashed outlines mean compared; thick green outlines mean swapped; a dotted outer selection ring connects a chosen array cell with its tree position. Counts, labels and outlines accompany colors. SVG descriptions enumerate positions, values and relationships.

**Construction figure.** Fifteen positions have 8 nodes at height 0, 4 at height 1, 2 at height 2 and 1 at height 3. The sum of per-node downward height budgets is `8×0 + 4×1 + 2×2 + 1×3 = 11`. This is an upper bound on downward moves across bottom-up repairs, not an assertion of eleven actual swaps or comparisons. A checked level uses at most two priority comparisons in this implementation; actual repairs may stop early. Geometry and colors do not encode invented timing data.

## Top-k investigation contract

Default stream `[5, 1, 9, 3, 9, 2]`, k = 3. Predict the second 9's admission; final retained values are `[5, 9, 9]`, with kth value 5. Occurrence numbers distinguish the two nines. A candidate has not yet entered the committed prefix; pressing the next step commits the entire repair. Individual swap mechanics belong to the preceding lab rather than being repeated in every top-k step.

The current minimum retained value is the replacement boundary. Fill until k entries exist; thereafter replace only for a strictly larger value. Equal candidates are discarded under this trace's tie policy; retaining a different equal occurrence would preserve the same correct ranked values. If fewer than k values have arrived, retain all arrivals but report no kth value. The committed retained and discarded identities partition exactly the processed prefix. The UI shows discarded history for explanation; a production streaming solution need not retain it. It does not promise stable ordering of equal-priority jobs.

Stream and k edits apply through Read this stream; invalid input preserves the active stream. Back/Reset are deterministic. The stream ribbon uses statuses as text in addition to colors, and the retained heap separates positional indices from source occurrence labels. The ending transfer changes to smallest-k and asks which boundary should be exposed.

## Trie investigation contracts

**Shared vocabulary.** `car`, `cart`, `cat`, `dog` create nine nodes including the empty root. Character labels belong to edges; the label below each node is its consumed prefix. Terminal double rings and ✓ mean a complete stored string; a dot means a prefix without stored-word membership. The root always exists as the empty prefix ε; it is terminal only if the empty string was inserted. Missing alphabet characters have no allocated child slot.

**Queries.** Exact `ca` is absent while prefix `ca` yields `car, cart, cat`. Both follow the same character path. A missing edge terminates without mutation. Empty-prefix lookup finds the root even in an empty trie, with no suggestions if no words exist. Suggestions collect terminal descendants in alphabetic order for these ASCII fixtures; they do not rank popularity.

**Mutations.** Insert `ca` by reusing its path and setting its terminal flag. Duplicate insertion leaves set membership unchanged. Delete `car` by clearing its marker while keeping `cart`. Delete `cart` from the original sample by clearing its marker and pruning only its nonterminal childless t suffix; terminal `car` stops pruning. More generally, pruning stops at the first terminal node or node with children. Delete a stored empty word by clearing only the root's flag. Intermediate unused suffix paths can appear between clear-marker and prune steps; the final set and pruned shape are checked independently.

**Input and inspection.** Starting words, query and operation apply with Trace these characters. Each run starts from the word-list input, not from an intermediate mutation snapshot. Blank word list is an empty set; ε explicitly stores an empty word; blank or ε query means the empty string. Invalid input preserves active state. Word nodes retain identities through surviving prefix paths; removed IDs are named when pruning. A linked query strip shows consumed characters. The diagram preserves native character/prefix sizes with local keyboard scrolling; a table provides exact prefixes, terminal flags and child links. No diagram dimension represents physical memory or measured speed.

## Research and model boundaries

Primary references inspected 10 September 2026:

- [Python heapq documentation](https://docs.python.org/3/library/heapq.html): zero-based min-heap invariant and push/pop/build contracts. The current page identifies Python 3.14.7; the lesson's native examples target Python 3.12-compatible APIs. Our educational swap trace is not claimed to reproduce Python heapq's exact internal comparison sequence.
- [Princeton Priority Queues](https://algs4.cs.princeton.edu/24pq/): swim/sink and construction analysis. Its initial max-heap, one-based representation is adapted deliberately to this lesson's min-heap, zero-based convention.
- [Princeton Tries](https://algs4.cs.princeton.edu/52trie/): stored-key versus prefix operations and trie representation choices. This model uses sparse children and terminal flags, not a fixed-radix Java array.

Browser bounds: integer values −99…99, at most 12 heap entries, at most 16 stream occurrences and k from 1 through 8; trie word lists at most 8 entries of 6 lowercase ASCII letters and at most 32 visible nodes. Duplicate heap priorities count, duplicate trie words do not. Input bounds are visible teaching limits, not restrictions on the general structures. The browser performs no case folding, locale collation, Unicode normalization, wildcard search, priority updates/cancellation, memory reclamation or real-time scheduling. The lesson's separate native Python examples supply broader applications and Python Unicode code-point semantics.

Comparison counters refer to this model's priority comparisons. Array scans for rendering, copied frames, sorted display results and retained explanation history do not establish production complexity or runtime benchmarks. There are no generated raster images, external lab dependencies or timers.

## Verification evidence

- `node scripts/verify-heap-trie-models.mjs` passed **1,820 heap operations**, **2,187 streams checked at every prefix**, and **2,072 trie operations**. Inputs include all arrays of lengths 0–5 over three integer values, multiple k values, and all subsets of a small prefix-rich word vocabulary with exact/prefix/insert/delete queries. Expected results use independent sorted multisets/prefixes and a string set, rather than copying the heap or trie implementation.
- Checks cover retained/discarded occurrence conservation, duplicate priorities and terminal words, empty/singleton inputs, negative values, kth unavailable before k arrivals, smaller/equal/lone child choices, source/snapshot isolation, global trie integrity, terminal-marker deletion and pruning, empty string/root policy, stable surviving IDs, input limits and invalid-state handling. Geometry checks compare every rendered edge with its real structural relationship and keep labels within the SVG view box. Construction heights are compared with a separate recursive shape calculation across sizes through 1,023.
- Owned JSX parses with the repository's Babel parser. The new Heap/Graph models, labs and model verifiers were formatted with the installed Babel generator. Before/after normalized ASTs, comments, literal values and JSX text values were conserved, then both complete model suites passed again. Formatting evidence and original sources are retained under `scratch/dsa-format-baseline/`.
- [Integrated native/browser verification](HEAPS-TRIES-VERIFICATION.md) passed at 1440px and 390px. Per width, it checked ten heap scenarios/56 states, five top-k scenarios/52 states, fifteen trie scenarios/65 states, exact array/tree and prefix/terminal correspondence, draft/error/reset behavior, keyboard access, references and no page/lab overflow or page errors. Representative screenshots were opened and inspected. [Browser results](../../scratch/heap-trie-lesson-review/results.json) and [horizontal keyboard-scroll checks](../../scratch/heap-trie-lesson-review/keyboard-scroll.json) retain evidence; the latter reaches the full cart branch in the narrow static trie. No browser-review blocker remains. These checks complement, rather than replace, the pure model and native Python suites.

These are author/runtime checks. User acceptance and an observed beginner learning study have not been claimed.
