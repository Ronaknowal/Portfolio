# DSA implementation-depth remediation — author handoff

**Final integration, 22 September 2026:** independent review and production/browser checks are complete for this scope; the ledger records both phases complete. [The final report](REMEDIATION.md) closes the integration work reserved in the dated author checkpoint below.

22 September 2026. Scope: the ten unresolved DSA rows in [FOUNDATIONS.md](FOUNDATIONS.md), under the user's authorized scratch/library completion request. **Author implementation and native/source checks are complete; independent content review, production browser review and integration remain with the coordinating task.** No shared build, development server, publication registry, central ledger, shared policy or aggregate audit was changed by this author.

This is a scoped extension of existing substantial lessons, not a replacement of their scratch algorithms. The earlier native examples remain their semantic owners. [The extractor](../../../scripts/build-dsa-library-bridges.mjs) derives topic-owned Python mechanism downloads from selected complete example strings, preserving definition/decorator text and rejecting inconsistent duplicate definitions. The new comparison imports those modules; it does not maintain a hand-edited second algorithm. The complete earlier examples remain visible on the page. New code views fetch their own downloadable source only when opened. Root added the shared wrapper's `language` prop for the C++ comparison.

## Evidence and source binding

- [Native/source evidence](dsa-remediation-native.json): **99 complete existing Python programs**, **11 new Python programs**, and **one compiled C++17 program** executed with exact recorded-output checks. The C++ range example and Python replay match exactly. All ten bodies and ten bridge components parse. The generated mechanism downloads match their owners; every official AtCoder header matches its recorded SHA-256.
- [Complementary author oracles](dsa-remediation-models.json): 23 named contract/changed-practice groups, including 2,048 directed traversal fixtures, 945 text/pattern pairs with every seam, 100 exhaustive-cut flow comparisons, 50 simple-path/forest-subset comparisons, all historical intervals across 80 branches, exact reservoir choice enumeration and file error/cleanup cases. These are author checks, not independent review.
- The compiled [range oracle](../../../scripts/fixtures/range-library-oracle.cpp) uses the actual lesson adapter and checks **14,000** direct-array ranges after signed set/add changes, plus **21 noncommutative folds**. This checks order, empty intervals and stale children; it does not benchmark upstream internals.
- [Machine-readable topic manifest](dsa-remediation.json) lists stable IDs, bodies, sections, assets, exact hashes and evidence. `author-complete-awaiting-independent-and-integration` is deliberately distinct from final completion. The manifest is the exhaustive changed/new relevant file list for integration; unchanged example owners are identified separately.
- Runtimes: Python **3.12.14**, NumPy **2.3.5**, SciPy **1.18.1**, NetworkX **3.6.1**, immutables **0.21**, the evidence-recorded SQLite version, and the available UCRT64 `g++` with `-std=c++17 -O2`. Installing immutables added only its pinned 35 KB wheel; it did not upgrade existing numerical dependencies. AtCoder **v1.6** required headers and CC0 license are pinned, downloaded from the official repository, bundled in a topic-owned archive and hash recorded.

Commands actually run:

```text
node scripts/build-curriculum-inventory.mjs --topic <each of the ten stable IDs>
node scripts/build-dsa-library-bridges.mjs
node scripts/verify-dsa-library-bridges.mjs --update-output
scratch/lesson-tools/Scripts/python.exe -B scripts/verify-dsa-remediation-models.py
node scripts/verify-dsa-library-bridges.mjs
node scripts/verify-dsa-library-bridges.mjs --reuse-originals
```

Every preflight returned success and a published, historically complete topic with current implementation-depth guidance. Existing destination notes were assessed. Historical extension notes about advanced string indexes, computational geometry families, persistence and durability remain intact; these were not interpreted as new scope. The four 21 September API-gap entries receive the bounded author disposition below.

The final native receipt reuses original-program passes only when the earlier passed receipt's exact owner hash still matches; it reruns the new bridges and compiled oracles after the last source amendment. SQL transaction mode was made explicit as Python 3.12's `autocommit=False`, and its current program still produces the recorded output. Actual prerequisite owners inspected: `sql-relational-data-transactions-for-ml.jsx` §§2, 6–7 with `sql-examples.js` for binding/transaction boundaries; `heaps-priority-queues-tries.jsx` §6 with `heap-trie-examples.js` `mergeStreams` for the lazy k-way frontier. The new sorter reuses the standard heap primitive rather than duplicating that earlier implementation.

## Trees & Binary Search Trees

- **Disposition:** deeper-review row resolved by actual mechanism/API assessment and an executed ordinary-tool route. The core outcome is ordered set operations, traversal, invariants and understanding height/rotation tradeoffs, not a complete production AVL implementation. A standard-library sorted-list alternative is appropriate; no artificial package was added to imitate a tree.
- **Scratch owner:** `src/learn/data/tree-examples.js`, search/deletion/orderedQueries/ceilingPractice/balance; body §§2–6. The original insertion/deletion and query functions are imported in `ordered_set_library.py` through `tree_mechanisms.py`.
- **New learner route:** `TreeLibraryBridge.jsx`, after §6, compares mixed insert/delete states, set duplicate policy, floor/ceiling and inclusive ranges using bisect. The lesson states O(h) tree operations versus O(log n) search/O(n) edits in a sorted list; rebuilding or one rotation does not create a dynamic balance guarantee. Constant-cost comparisons are an explicit model.
- **Practice/checks:** reserve the next eligible slot, then release duplicates; 250 mixed mutations against a Python-set/scanning oracle, including empty/extreme queries. Original tree programs all execute unchanged.
- **Boundary:** unbalanced height and recursive operations' stack limits remain in the original explanation. No universal speed or worst-case logarithmic tree claim is introduced. `bisect` is the ordinary array route, not claimed to be a balanced container.

## Graphs: Representations, BFS & DFS

- **Disposition:** maintained-tool construction/traversal gap repaired locally.
- **Scratch owner:** `graph-traversal-examples.js`, representations/breadthFirst/depthFirst/components; body §§2–5. The complete clone/grid/cycle/color mechanisms and their practice remain intact.
- **New learner route:** `GraphTraversalLibraryBridge.jsx`, after §5, establishes the reusable NetworkX construction boundary: declare vertices including isolates, preserve direction, collapse repeated simple edges, explicitly distinguish a multigraph. The code checks BFS distance/path witnesses, DFS reachability and undirected component sets. Different tied trees need not match.
- **Practice/checks:** reverse directed edges to find incoming reachability, preserve new isolate I; exhaustive three-vertex directed graphs including loops plus an isolated fourth vertex, checked against independently computed distance closure. Duplicate edges are injected. The prose limits all-path materialization to its small witness fixture and points to length-only output for linear search state.
- **Boundary:** graph drawing coordinates are not route cost; custom grid-neighbor generation and identity-aware object cloning remain local mechanisms. Later weighted/flow sections link to this actual new section instead of claiming a planned owner.

## Segment Trees, Fenwick Trees & Range Queries

- **Disposition:** deeper-review row resolved through a compiled official API counterpart and actual algebra checks.
- **Scratch owner:** `range-query-examples.js`, segment/fenwick/lazy, body §§2–5. The linear Fenwick construction, ordered two-sided fold and new-after-old affine tags are preserved. The later sparse/deque/DP/rank mechanisms retain their full original examples.
- **New learner route:** `RangeQueryLibraryBridge.jsx`, after §5, gives C++17 compiler/header setup, a downloadable pinned official AtCoder subset and a complete `range_library.cpp`; `range_library_reference.py` reuses local Python classes on the same values. Segment prod/set, Fenwick delta add and lazy summary/action/composition are mapped explicitly.
- **Practice/checks:** assignment/addition in reversed order, every subrange of a five-element array, empty intervals and non-power-of-two lengths. The C++ oracle checks 14,000 array-derived answers plus noncommutative folds; Python local structures have their own direct-array checks.
- **Boundary:** the bridge is explicitly optional C++ depth, not a disguised Python file. Its straightforward library Fenwick initialization is O(n log n), compared with the locally taught O(n) build. String concatenation is an order illustration whose combine cost is not constant. C++ long-long bounds include intermediate arithmetic; no overflow is asserted safe outside that contract. No language/runtime performance comparison is claimed.

## Shortest Paths, Spanning Trees & Topological Ordering

- **Disposition:** weighted-solver gap repaired, with independent objective-specific witnesses.
- **Scratch owner:** `weighted-graph-examples.js`, dijkstra/bellman/kruskal/topological; body §§2–6. Budgeted routes, 0/1 BFS, Floyd–Warshall, Johnson, Prim, dense implicit distances, signed DAG relaxation and critical schedules retain their complete programs and explanations. All twelve programs execute.
- **New learner route:** `WeightedGraphLibraryBridge.jsx`, before §8. NetworkX MultiDiGraph retains parallel edge IDs and explicit weights; Dijkstra and Bellman–Ford compare source distances. An undirected MultiGraph supplies minimum spanning forests; graphlib provides the normal standard-library dependency API with its predecessor convention.
- **Practice/checks:** zero then signed cheaper parallel edge, all reachable path costs, isolated target; 50 graph fixtures checked by simple-path enumeration and spanning-subset enumeration. Negative-cycle raising is demonstrated separately from the local per-destination negative-infinity contract. Topological outputs are checked by every precedence relation; cycle detection is exercised.
- **Boundary:** graphlib is not a finite-worker scheduler. A maintained solver comparison for each related spelling of all-pairs/priority specialization is not substituted for each owned algorithm's existing implementation: those mechanisms use the already-taught graph, heap/deque, matrix and DP primitives. The new parity claims are specifically source shortest paths, forest objective and dependency order, not all package algorithms or all output representations.

## String Matching, Prefix Functions & Rolling Hashes

- **Disposition:** deeper-review row resolved by a concrete one-shot-versus-streaming decision and matched built-in adapter.
- **Scratch owner:** `string-matching-examples.js`, prefix/kmp/stream; original rolling, substring hash, Unicode, period/palindrome, DNA and Z programs remain complete and executed.
- **New learner route:** `StringSearchLibraryBridge.jsx`, before §11. Python str.find is used for ordinary first occurrence, and an index-based repeated-find adapter retains overlaps without suffix allocation. The empty pattern and code-point coordinate contract match KMP. Unicode normalization and stream state are compared explicitly.
- **Practice/checks:** 945 binary text/pattern pairs plus every stream seam and empty delivery against direct slicing; changed six-symbol chunk exercise covers empty and nonempty patterns. The code prints escaped non-ASCII text for portable Windows stdout while preserving actual Unicode input.
- **Boundary:** repeated find calls are not claimed to inherit KMP's total worst-case bound. Stream memory stays O(pattern length) excluding output; no join is used in the requested implementation. Normalized offsets belong to the normalized string, and chunkwise normalization is not assumed equivalent to whole-string normalization. Deferred advanced string families remain deferred.

## Randomized Algorithms, Sampling & Error Guarantees

- **Disposition:** deeper-review row resolved by actual random API comparison with a deliberate distribution-level contract.
- **Scratch owner:** `randomized-algorithm-examples.js`, rejection/shuffle/weighted/reservoir, with selection/Freivalds/budget/replay branches preserved and all eleven programs executed. Exact mechanisms legitimately reuse Random for pseudorandom bits rather than rebuilding a PRNG.
- **New learner route:** `SamplingLibraryBridge.jsx`, before §9. Standard shuffle/sample/choices expose mutation, occurrence identity, replacement, short-stream policy, floating versus exact integer tickets, and state restoration. Equal seeds across different algorithms are not falsely treated as output parity.
- **Practice/checks:** downsample a four-item reservoir to two without replay; exact six-item combinatorial enumeration verifies uniform pair multiplicities. Reservoir's complete five-record choice tree has uniform subset counts. Integer-ticket boundary tests retain a one-ticket category beside a weight of 2^60, and zero capacity does not consume input.
- **Boundary:** no finite frequency run is called a proof; proofs and exact finite enumeration are distinguished. Sampling an unknown stream by first listing it would violate the taught memory contract. Quickselect/Freivalds are locally taught custom routines over standard array/random primitives, not missing package wrappers.

## Network Flow, Minimum Cuts & Bipartite Matching

- **Disposition:** maintained flow/cut gap repaired; specialized matching route also supplied.
- **Scratch owner:** `network-flow-examples.js`, residual core/augment/matching; body §§2–5. Dinic, lower-bound/supply/node-capacity reductions, binary labeling and random global contraction remain preserved, with all nine programs executed.
- **New learner route:** `FlowLibraryBridge.jsx`, after §5. Explicit capacity attributes and Edmonds–Karp selection; parallel capacities aggregate because NetworkX flow rejects multigraphs, antiparallel edges remain distinct, loops are omitted. Actual maximum-flow and minimum-cut results are checked through conservation, bounds and cut equality. Hopcroft–Karp matching uses tagged partition identities and explicit top_nodes; cover size and coverage are checked.
- **Practice/checks:** changed exits give a capacity-4 certificate on original edge identities. One hundred integer networks with parallel/antiparallel/loop edges are compared with every possible tiny s–t cut; local original-edge flows are separately checked for conservation and capacity bounds.
- **Boundary:** aggregate flow does not identify individual parallel-edge allocations. Cost/lower-bound/mandatory-demand models reuse their existing explicit reductions. No arbitrary float-tolerance robustness or particular tied optimum is claimed.

## Computational Geometry, Robust Predicates & Convex Hulls

- **Disposition:** practical hull-tool gap repaired without weakening exact predicate ownership.
- **Scratch owner:** `computational-geometry-examples.js`, orientation/hull/polygon area; all eight original programs execute, including rational construction and precision counterexamples.
- **New learner route:** `GeometryLibraryBridge.jsx`, before §7. SciPy ConvexHull/Qhull on explicit float64, duplicate normalization, corner-only set comparison, area/perimeter field interpretation and actual collinear QhullError. The exact all-boundary/collinear/singleton policies remain local.
- **Practice/checks:** translated coordinates with origin subtraction before conversion, versus lost distinctions near 2^53; 100 nondegenerate small-integer hulls compare corners and exact area and also verify support maxima independently. This is a bounded matching-coordinate comparison, not a floating oracle for arbitrary exact integers.
- **Boundary:** jitter would change input; no automatic QJ repair is supplied. Sorting/orientation/storage costs and arbitrary-precision bit cost are stated. No reopening of Voronoi, 3D or other deferred specialist families.

## Persistent Data Structures, Structural Sharing & Versioned Queries

- **Disposition:** deeper-review row resolved with an executed persistent collection route and its actual abstraction boundary.
- **Scratch owner:** `persistent-structures-examples.js`, point build/assign/range_sum and identities; body §§2–4. Stack, monotone timeline histories, prefix rank, immutable lazy updates and branching-amortization counterexample remain taught and executed.
- **New learner route:** `PersistentMapLibraryBridge.jsx`, after §4. immutables.Map reuses integer indices as keys and branches from named old versions; a batch mutation context and shared mutable payload are demonstrated. A hash-trie map is not mislabeled an interval tree or a range-sum library.
- **Practice/checks:** two-position historical correction, every interval of every retained version across 80 branches against copied arrays, old-state preservation and explicit root no-op identity only for our own tree.
- **Boundary:** tree updates/queries have logarithmic path cost and allocation in their declared model; map range sums visit keys rather than exploiting cached summaries. Hash/equality/collision behavior and shallow payload immutability are stated. No measured library speed/memory claim or claim to teach HAMT internals from scratch.

## External-Memory Algorithms, B-Trees & I/O Complexity

- **Disposition:** deeper-review row resolved with a real database route and a stronger practical file-sort interface, while preserving the pedagogical I/O model.
- **Scratch owners:** `external-memory-examples.js`, complete B-tree search/split/borrow/merge, B+ routing, file-sort counters, merge join and root-publication protocol; all eight programs execute. `external_sort_stream.py` owns the new file-output/resource-lifetime route.
- **New learner route:** `ExternalMemoryLibraryBridge.jsx` adds streaming sorter after §5 and SQLite index after §7. The sorter keeps output on disk, sorts chunks in place, limits open merge streams, closes them through ExitStack, removes consumed runs and carries singleton groups. The SQLite program uses a named unique index, bound query values, explicit ordering, commit/rollback and reopen, compared to the existing BTree set.
- **Practice/checks:** duplicate-preserving SQL composite index plus bounded-fan-in sorting; empty/singleton/partial/multilevel files, negative and duplicate values, preserved input, truncated-record rejection, existing-output protection and temporary-workspace cleanup. No physical disk-performance measurements were inferred from logical counters or EXPLAIN output.
- **Boundary:** record-buffer bounds also require O(number of runs) metadata and Python overhead. The final copy can leave a partial new destination on failure; it is explicitly not a crash-durable transaction. Normal SQLite reopen/rollback tests are not power-loss tests. Original shadow-root assumptions remain intact.

## Research and learner-experience author assessment

Primary pages actually opened/read on 22 September 2026 are linked in each new lesson section: Python bisect, str.find, random, graphlib, sqlite3, heapq.merge and ExitStack; NetworkX graph/direction/reverse, shortest paths, minimum spanning tree, max flow and Hopcroft–Karp; SciPy ConvexHull; immutables primary repository API; official AtCoder segtree/Fenwick/lazy contracts and v1.6 release; SQLite query planner and EXPLAIN QUERY PLAN. Their moving documentation versions vary; executed versions above are the reproducibility claim. Original examples and worked exercises are independently authored and executed; no copied documentation implementation is presented as our scratch algorithm. The official AtCoder files retain their upstream license/provenance.

The source-level learner-experience pass finds a connected route in each lesson: the original question/mechanism precedes an API choice, variable/state correspondence is explicit, worked outputs are interpreted, and the learner changes a meaningful constraint. Existing diagrams and immediate-output investigations stay intact. Ten added practice prompts keep hints and reasoned solutions in separate initially closed details. No learner-prediction controls were added. New code remains deferred, copyable/downloadable and scoped to one topic. Read-time estimates increase by ten minutes per lesson; optional compiler depth is labeled. No uniform lab quota or package-for-its-own-sake was introduced.

Nine-point assessment: relevance and starting contract present; prerequisite refresh/reuse located; mechanism preserved; complete worked programs executed; representation fit retained; API comparison interprets ties/units/state; independent changed practice present; setup/cost/numerical/resource boundaries explicit; browser reading/keyboard/theme/mobile verification **pending root integration**. This author assessment is not a beginner study or independent review. The review should particularly inspect language-aware code labeling, same-folder helper downloads, long compiler setup wrapping, the new source anchors and preserved module navigation.

## Remaining handoff

Independent source review requested one prose correction in String Matching §11: replace the stale predict-first instruction with running, inspecting state changes and comparing output. The body was reparsed and its binding refreshed; all executable sources and retained native/model results are unchanged. No experiment was rerun for this prose-only correction.

No known numerical or content finding remains in the added scope after author checks. Root owns independent review, any requested corrections, source-bound phase updates, regenerated navigation/search, final production build and browser checks at desktop/390/320. The four API destination-note entries are author-remediated with this evidence; final implementation status must follow that integration. The aggregate foundations audit remains unchanged as the dated triage input.
