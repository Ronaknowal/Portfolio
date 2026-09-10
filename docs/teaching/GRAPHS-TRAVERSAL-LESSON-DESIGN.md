# Graphs: Representations, BFS & DFS — lesson design

10 September 2026. Stable ID `graphs-representations-bfs-dfs`; fifth DSA topic and third requested new implementation. Topic-plan command returned planned/no individual brief/no destination note and an empty routing inbox. Preserve title and module order. Readiness: earlier queues/stacks, dictionaries and references; explain the set/edge and invariant vocabulary locally because the formal sets/proofs prerequisite is in another module.

## Scope and ownership

Finish line: model a relation as a graph, choose an honest representation, implement and justify BFS/DFS, recover a route, handle disconnected components and transfer to implicit-grid and shared-object problems. This is more than two traversal templates. The core must explain which state is needed for discovery, pending work and the requested answer.

| Idea | Present coverage / best owner | Decision |
| --- | --- | --- |
| Vertices/edges, direction, adjacency list/matrix/edge list | This planned entry is foundational owner | Full small same-graph comparison; preserve isolated vertices and define duplicate/self-edge policy |
| BFS layers, mark-on-discovery, parents and equal-cost shortest routes | No body exists; earlier queues supply mechanism | Full implementation, invariant/proof, queue/distance/parent trace and unreachable/source-equals-target cases |
| DFS pending iterators and finish events | Earlier Trees teaches call/return | Explicit frame stack matching recursive depth-first behavior; explain why a bag of discovered vertices is insufficient for ancestor/finish reasoning |
| All components, grids and multiple sources | Local extensions with useful transfer | Full programs, independent task and grid wavefront view; state four/eight-neighbor and distance-unit contracts |
| Clone graph identity | Earlier OOP/references explain aliasing but not cyclic graph copying | Small exact object-copy example; map old object to new before exploring edges |
| Directed cycle, bipartition, all paths | Need richer state beyond one visited set | Optional developed mechanisms with executable cycle/color examples and a path-local transfer explanation; no claim generic DFS answers every question |
| Weighted paths/topological order/spanning trees | Existing planned `Shortest Paths, Spanning Trees & Topological Ordering`, inspected in cross-domain-expansion | Local weighted counterexample and active/finished bridge; save source-specific frontier/state note for fuller destination treatment |
| Low-link/SCC/bridges | More advanced graph structure than core traversal | Name/locate next-owner need only when useful; do not paste unexplained formulas or claim complete coverage |

## Visual contracts

| Hurdle | Chosen representation and learner question | Required model evidence |
| --- | --- | --- |
| Same graph, different storage | Editable small edge set with synchronized node-link, adjacency list and matrix; toggle direction | Every drawn/matrix/list edge agrees, isolated vertices survive, duplicate policy explicit; geometry is layout, not distance |
| When does a vertex become discovered? | BFS graph layers, FIFO frontier and parent/distance table; step edges and compare routes | Enqueue once, shortest edge-count oracle, parent chain uses real edges; tie order is deterministic not unique answer |
| What is unfinished in DFS? | Frame stack showing next neighbor + active path, entry/finish order and graph | Actual recursive reference order for a declared neighbor order, no cycles in parent forest; reachability matches BFS but paths need not be shortest |
| Graphs can be implicit | Editable bounded wall grid with single/multiple sources, distance cells and selected path | Grid neighbor rule and edge counts exact; blockers/unreachable distinguished, multi-source distances match minimum of independent single-source distances |
| A revisit need not be a cycle | Small directed acyclic diamond contrasted with a back edge to active work | Reachability/active-vs-finished evidence, direction arrowheads, no “seen means cycle” rule |
| Copying topology without aliasing original nodes | Original and copied three-node diagrams, stacked on narrow screens; preserve shared C and cycle back to A | Native identity-map checks; prime labels encode fresh object identity, not changed values or extra edges |

Default explicit graph: vertices A–H; undirected edges A–B, A–C, B–D, C–D, D–E, F–G; H isolated. Sorted neighbor order in browser makes traces repeatable. The diamond A–B–D–C–A contains a cycle, while F–G and H demonstrate disconnected/isolated regions. BFS A discovers A,B,C,D,E; DFS with ascending neighbors enters A,B,D,C,E. Algorithms count conceptual vertices/edges, not renderer lookup operations. Layout must preserve readable labels on small screens, with local scroll/enlarge when needed. Models are deterministic, finite and event-triggered; no animation timer or unbounded all-path enumeration.

## Native examples and practice

Complete standalone Python programs for representations, BFS/parents, iterative DFS frames, connected components, grid distances with multiple sources, graph cloning, bipartition, directed-cycle detection and a changed independent grid problem. Test against independent transitive closure/shortest-distance and identity oracles, not only matching final arrays. Include empty/disconnected/self-source/duplicate-edge/invalid-endpoint/long-chain cases and matrix-versus-list costs. Explain recursion limits for optional recursive branches.

Practice progresses from predicting queue/frame states to changed adjacency/distance contracts and diagnosing invalid visited/cycle rules. Independent tasks include an eight-neighbor route with cell-count output, a component counterexample and a directed revisit without a cycle. Give hints and complete or reasoned solutions. Curated LeetCode practice follows [DSA-PRACTICE-STANDARD.md](DSA-PRACTICE-STANDARD.md); optional tasks identify their extra invariants and later owners.

## Sources and review plan

- Princeton Undirected Graphs and Directed Graphs chapters inspected for representation, reachability, BFS path guarantees, DFS/finish and cycle distinctions: https://algs4.cs.princeton.edu/41graph/ and https://algs4.cs.princeton.edu/42digraph/ . Define our own vocabulary/conventions explicitly where textbooks differ.
- MIT OCW lectures 13 and14 resource pages and their seven-page typed companion PDFs read for substantive representation/layer and frame/finish/cycle coverage; this is notes review, not full video viewing. https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/resources/lecture-13-breadth-first-search-bfs/ and https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/resources/lecture-14-depth-first-search-dfs-topological-sort/ . Actual PDF sources: https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/1208e162775f6f5cedfbb9f2b694ede0_MIT6_006F11_lec13.pdf and https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/e59f8a55929028498953691891229a17_MIT6_006F11_lec14.pdf .
- All calculations, traces, fixtures, diagrams and runnable examples are generated by the local algorithms or original teaching setups; none is an empirical runtime benchmark.

Native/model verification, browser/keyboard/mobile review, publication conservation and production load checks are required before completion. Save actual evidence in the topic verification record; this design is not verification, approval or a full-curriculum coverage audit.
