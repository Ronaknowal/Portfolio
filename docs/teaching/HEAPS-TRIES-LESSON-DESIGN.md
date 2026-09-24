# Heaps, Priority Queues & Tries: lesson design

10 September 2026. Stable ID `heaps-priority-queues-tries`; fourth topic in DSA module order. Retrieved the topic-plan command: planned, no individual brief, no destination file, empty unresolved inbox. Existing prerequisite is Trees & Binary Search Trees, now implemented. This is the second of the three requested implementations; retain the title because it explicitly names the two different collection contracts.

## Learning contract and scope decisions

A beginner should implement and select a binary heap for changing priorities and a trie for shared prefixes, explain every repair or character transition, then transfer these structures to streaming selection, merging and prefix policies. Python list/dict/class basics and the preceding tree lesson suffice; explain logarithmic/linear cost and recursion locally as needed. Keep two clearly signposted branches in this bundled topic instead of pretending a trie is a kind of heap.

| Coverage decision | Evidence and ownership | Treatment |
| --- | --- | --- |
| Min/max heaps, implicit complete shape, push/pop/build | This entry is the natural foundational owner; absent lesson | Full invariant, termination, runnable implementation and array/tree trace |
| Stable priority entries, updates/cancellation | Priority queue contract rather than just a heap operation | Worked numeric priority + monotonic ticket + payload; stale-entry handling, storage/compaction limits |
| Streaming top-k and k-way merge | Distinct boundary/frontier applications, not another list of uses | Concrete complete examples, output accounting, ties and empty sources |
| Event scheduling | Shows why priorities can change while output is produced | Small deterministic original event simulation with future-time constraint; no real-time guarantee |
| Exact word, prefix, insertion and safe deletion | A prefix path alone does not establish a stored word | Full sparse-child trie with terminal marker, empty-string contract, pruning without deleting longer words |
| Suggestions and shortest/longest prefix policies | Valuable prefix-specific transfer best introduced here | Explain finding candidates versus ranking; full small example and costs including returned strings |
| Heap sort, two heaps, indexed heaps and compressed tries | Extensions of local mechanisms | Optional depth with worked mechanism; specialized variants are not required for the first pass |
| Weighted paths and heap frontier | Existing planned `Shortest Paths, Spanning Trees & Topological Ordering` in cross-domain-expansion | Bridge/link only; preserve full weighted correctness proof and stale-entry implications for destination author |
| Full string matching/automata | Existing planned string matching owner | Distinguish prefix lookup from substring matching; no full algorithm detour |

## Hurdles and representations

| Question | Explanation and example | Representation / learner action | Evidence |
| --- | --- | --- | --- |
| Why isn't a heap array sorted? | Parent order + complete shape, not sibling order or BST bounds | Array indices linked to actual tree edges; select push/pop/build and inspect swaps | Multiset preserved, parent comparisons valid at completed states, result matches sorted oracle |
| Which child repairs a min-heap? | Move smaller child up; choosing arbitrary child can leave another violation | Highlight compared positions and changing repair path | Equal-child tie rule, lone child and empty/singleton cases |
| Why linear heap construction? | Most nodes are leaves or near leaves; sum actual subtree-height work | Static level/work illustration for 15 positions, with cost explanation, no invented benchmark | Height counts computed from index structure |
| Which k values matter? | Retain largest k occurrences; minimum retained value is boundary | Stream ribbon, current candidate, retained heap and discarded values; editable stream/k | Each prefix checked against independently sorted prefix; duplicates and k larger than prefix |
| Why distinguish word from prefix? | `car` versus `cart`, path versus terminal marker | Character-labelled trie edges, terminal glyph, query consumption and selected branch; insertion/deletion | Set and startsWith oracle; deleting prefix retains descendants |
| What does a prefix policy choose? | Shortest match stops at first terminal; longest remembers last terminal | Concrete trace plus suggestions output | Exact consumed-prefix and output-set checks |

Labs use bounded deterministic inputs and event-triggered traces, no timers or unbounded search. Edits apply explicitly and reset step state. Array cells, edges, terminal flags and outputs derive from the same source model. Preserve text alternatives, legible narrow-screen labels and keyboard controls; use local scrolling or enlargement where needed. No artificial diagram/lab count requirement.

## Research ledger

- Python `heapq` documentation, current 3.14.7 page, inspected 10 September: zero-based min-heap interface, pushpop/replace distinction, tuple tie caveat, max-heap APIs added in 3.14. Core examples target and execute on Python 3.12, using portable min-heap APIs. https://docs.python.org/3/library/heapq.html
- Princeton Priority Queues chapter: inspected heap ordering, swim/sink, linear construction and resizing qualification. Its illustrated max-heap/one-based indexing differs from our min-heap/zero-based convention. https://algs4.cs.princeton.edu/24pq/
- MIT OCW lecture 4 resource and its 28-page companion slide PDF inspected for indexing, bottom-up construction and shrinking active heapsort range. This is notes review, not full video viewing. https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/resources/lecture-4-heaps-and-heap-sort/ and https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/8ebfeb1c645b10b3709919603e7d51be_MIT6_006F11_lec04.pdf
- Princeton Tries chapter and lecture slides inspected for terminal values, prefix enumeration, deletion and representation choices. Our examples use sparse dictionary children and original small words; their Java fixed-radix implementation has different storage constants. https://algs4.cs.princeton.edu/52trie/ and https://algs4.cs.princeton.edu/lectures/keynote/52Tries-2x2.pdf

## Practice and acceptance

Local independent tasks: repair a wrong child choice; retain smallest k instead of largest k; delete a prefix while preserving a longer stored word; choose a structure under changed query constraints. Supply hints, complete solutions or reasoned acceptance criteria, and adversarial examples. Curated LeetCode work follows [DSA-PRACTICE-STANDARD.md](DSA-PRACTICE-STANDARD.md), with core and optional extensions, no universal interview guarantee.

Run every complete displayed Python program and compare exact output. Independently check heaps against sorted multisets, top-k against sorted prefixes, trie operations against a set, output/ranking against filtering/sorting, and event timestamps against expected schedule. Browser review covers actual search/update/step/reset behavior, invalid inputs, desktop/mobile, keyboard, anchors and selected-topic loading. Record real results separately; this design is not verification or user acceptance.
