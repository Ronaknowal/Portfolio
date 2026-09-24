# Heaps, Priority Queues & Tries: independent verification

10 September 2026. Scope: the second topic in the authorized sequential DSA implementation, after Trees & Binary Search Trees. Read with [the lesson design](HEAPS-TRIES-LESSON-DESIGN.md), [DSA practice standard](DSA-PRACTICE-STANDARD.md), [teaching standard](../../LESSON-TEACHING-STANDARD.md) and [learning code standard](../engineering/LEARNING-CODE-STANDARD.md). The exact stable-ID topic plan and returned authoring notes were read; no destination note or relevant unresolved routing item existed.

## Teaching and contract review

The topic explicitly teaches two different queries and structures: changing priority selection through a complete binary heap, then prefix-based retrieval through a trie. It uses the previous tree lesson's shape, height, invariant and traversal ideas without suggesting that a trie is a heap or that a heap has BST ordering. Ground-up array-position formulas, complete shape, local repairs and a smaller-child counterexample explain why the operations work.

Reviewed details include duplicate occurrences, equal-child versus stable-payload tie policies, a copied constructor input, empty/singleton removal, zero-based Python conventions, pushpop versus replace, Python 3.14 max-heap API availability versus portable Python 3.12 examples, amortized list-resizing qualifications, linear bottom-up construction and the additional sorting cost of displayed top-k snapshots. Stable priorities distinguish logical live entries from physical stale storage; expensive stale-skipping pops and compaction costs are explicit. The merge bound includes scanning empty sources and iterator state. Event times are simulated, may equal the current time, and never promise wall-clock deadlines or termination under endless immediate rescheduling.

Trie coverage separates an existing path from a terminal word, preserves extensions during deletion, explicitly stores or removes the empty word at the root, and treats empty-prefix existence separately from nonempty results. Sparse dictionary costs, Unicode code points, normalization/case policies, lexical versus popularity order, output/sorting costs and shortest/longest prefix choices are qualified. Recursive suggestions may hit a depth limit even though insertion, lookup and deletion use iterative loops. Ordinary prefix lookup is not arbitrary substring matching; optional compressed, wildcard and two-heap ideas remain scoped extensions with prerequisites.

The review identified one concrete API boundary defect: a fractional suggestion limit such as 2.5 never equaled the emitted-count cutoff and silently returned too many suggestions. The author changed suggestions and both k helpers to require actual integers (excluding bool), with nonnegative/positive policies stated by their respective contracts. Independent boundary checks were added and rerun. The review also prompted a recursion-limit note near suggestions and precise nonempty-prefix node counting.

## Native examples and independent references — passed

Command: `node scripts/verify-heap-trie-examples.mjs`.

The wrapper imports the exact displayed example strings, runs all ten standalone programs in isolated Python processes and compares stdout, then invokes [verify-heap-trie-native.py](../../scripts/verify-heap-trie-native.py). Python: repository `scratch/lesson-tools/Scripts/python.exe` (3.12.14); `LESSON_PYTHON` can override it. Fixture sources and the exact JSON manifest are retained in `scratch/heap-trie-native-verification/`. No package installation or network is required.

| Verified behavior | Actual passing coverage | Independent reference |
| --- | ---: | --- |
| Displayed code and outputs | 10 complete programs | Real isolated Python execution |
| Heap construction and in-place sorting | 1,097 sequences | Every length-0…6 sequence over three signed/tied values, plus extreme, fractional, ascending and descending fixtures; separately sorted multisets |
| Pop and push intermediate states | 6,143 pop; 9,434 push | Occurrence-count conservation and every parent-child comparison after each operation |
| Largest-k prefixes | 24,551 | Each consumed prefix independently sorted, including duplicate values, short prefixes and k larger than input |
| Smallest-k results | 5,472 | Independently sorted input slices, including k=0 and empty input |
| Mutable stable priorities | 10,000 operations | A separate live-job map chooses the minimum (priority, arrival ticket); updates, cancellation, empty pops and compaction include mixed hashable payload identifiers that cannot be mutually ordered |
| Sorted-source merges | 300 multisets | Flatten-and-sort oracle; empty sources, duplicate values and generator sources |
| Lazy consumption | 6 explicit contracts | Counted one-shot iterators verify no eager top-k consumption, zero-k short circuit and one pending merge head per active source |
| Event schedules | 100 | Independent sorted event collection with insertion tickets, equal timestamps, follow-up delays and rejection of the simulated past |
| Trie mutation and pruning | 2,400 | A separate string set plus full reachable-path/terminal reconstruction; no aliased nodes or unused nonterminal leaves |
| Exact, prefix, shortest and longest queries | 62,400 query bundles | String equality/starts-with filtering, independent minimum/maximum matching-prefix length |
| Bounded suggestions | 312,000 | Filtering and sorting the separate stored-word set; 0/1/2/3/large limits, missing prefixes and duplicates |
| Invalid suggestion limits | 360 | Negative, fractional, string, None and boolean limits reject with ValueError |
| Deep iterative trie methods | 1,500 code points | Insert, contains, starts_with and delete; no claim that recursive suggestions support this depth |

The corpus includes distinct NFC/decomposed strings, emoji and a joiner sequence: normalization is not silently applied, terminal membership remains exact, and Python ordering follows stored code points. A dedicated car/cart/carton fixture deletes a prefix and then an extension without losing the remaining word. k helper validation is exercised on invalid numbers/types as well as bool. After the validation edit, all displayed outputs and the complete independent native corpus passed again.

The references do not reimplement sift-up/down or the recursive suggestion collector. They validate meaningful finite cases, not universal correctness or asymptotic guarantees. Source/API review belongs to the coordinating author's research ledger; this reviewer did not watch entire videos or submit platform solutions.

## Browser and visual review — passed

Command: `node scripts/review-heap-trie-lesson.cjs` against the published Vite route `http://127.0.0.1:5173/learn/path/full-curriculum/heaps-priority-queues-tries`. Headless Microsoft Edge via Playwright used fresh 1440×1000 and 390×1000 pages with reduced-motion preference. Actual results are retained at `scratch/heap-trie-lesson-review/results.json` (10 September 2026, 05:21 UTC).

At each width, the suite verified:

- Ten heap scenarios, totaling 56 trace states: push, pop, build, right-child repair, ties, singleton/empty boundaries, invalid starting-heap rejection and browser capacity. Array values and drawn tree positions agree at every intermediate state, with correct final values and returned minima. Selected array indices link to the right tree position; Back, presets, invalid-input preservation and Reset work.
- Five top-k scenarios, totaling 52 states: default duplicates, k larger than available data, signed values with k=1, empty stream and all ties. Each intermediate committed prefix matches a separately sorted prefix. Retained/discarded occurrence counts agree with that prefix; an inspected candidate is not included prematurely. Invalid k values preserve the active stream.
- Fifteen trie scenarios, totaling 65 states: exact/prefix differences, missing keys, prefix and suffix deletion, duplicate/new insertion, stored empty word, empty-prefix suggestions, empty-trie lookup and insertion. Exact terminal tables and visible stored-word results agree at every step. Clearing car's marker retains its outgoing t→cart edge; deleting cart prunes only the unused suffix. Invalid text/node counts preserve the active trie, and Reset restores the sample.
- Ten complete rendered code/output sections, ten official practice links, initial closed hints/optional group, keyboard opening/closing, safe new-tab attributes, nine unique in-page route anchors, guided-practice hash navigation and the actual next Graphs topic.
- Focus and Tab behavior in all three investigations, native controls at least 43px high, readable source links, no whole-page or lab overflow and no page errors.

One initial harness assumption sorted the visible ε glyph as if it were a literal key. The harness was corrected to sort the underlying empty string before displaying ε; the lesson's order was already correct. This was not a lesson failure.

Representative desktop and 390px screenshots were opened and inspected: heap insertion/array selection, right-child removal, duplicate streaming decisions, trie terminal clearing/pruning, both inline figures and the foundation practice stage. Files are retained under the same scratch directory. The heap's fixed indices remain distinct from moving values; occurrence IDs distinguish equal 9s; the trie uses edge characters, prefix labels and terminal double rings consistently. The construction figure shows the actual 15-position height groups and total downward budget 11 without presenting timings.

Narrow trie drawings intentionally scroll locally to preserve native label size. An additional real-keyboard check verified ArrowRight changes scrollLeft in all three investigations and reaches the far-right terminal branch of the static trie figure (188px maximum offset); `keyboard-scroll.json`, `trie-inline-mobile-start.png` and `trie-inline-mobile-end.png` preserve evidence. The full cart terminal, retained car terminal and cat/dog branches remain readable when panned. An initial auxiliary attempt used End, which is not the horizontal movement key; the confirmed check uses ArrowRight. Exact terminal/link tables provide a text alternative. Native horizontal scrolling may animate independently of the application's reduced-motion preference; no teaching animation is required to understand a state.

The separate model owner's [visual/model record](HEAPS-TRIES-VISUAL-MODEL-REVIEW.md) reports 1,820 heap operations, 2,187 streams checked at every prefix and 2,072 trie operations, plus geometry/integrity/limit cases. This is separately attributed model evidence. The native examples and integrated browser checks here do not substitute for those model tests or for actual screenshot inspection.

## Status

- Lesson implementation: complete and published by the coordinating author.
- Computational verification: displayed Python and independent native corpus passed, including revised validation boundaries.
- Browser/visual verification: integrated 1440px/390px checks and actual screenshot inspection passed; no outstanding blocker from this review.
- User review / real beginner walkthrough: not performed.
- Next action: ready for user review; continue to Graphs in the authorized module sequence. The coordinating author retains application-build and curriculum-conservation verification responsibility.
