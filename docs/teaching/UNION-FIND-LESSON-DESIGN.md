# Disjoint Sets & Union-Find — individual lesson design

Scope: stable topic `disjoint-sets-union-find`, immediately after Graphs: Representations, BFS & DFS. Reviewed the topic plan and unassigned inbox on 10 September 2026; no destination note exists. The unrelated bit-manipulation ownership note is not folded into this lesson. Preserve the title, ID and module order.

## Learner and promise

The learner can use lists, dictionaries, sets, queues and the preceding graph lesson. They can label components by traversal but have not seen a mutable partition data structure. Bridge “connected groups” to equivalence classes locally. Teach the full ordinary DSU contract, eager versus forest representations, global root invariant, size weighting, rank distinction, full compression, amortized costs, root metadata, incremental connectivity and active-element modeling. Include limitations and a bounded rollback extension; weighted/potential constraints and fully dynamic connectivity remain identified further study, not implied capabilities.

## Reading flow and mechanism decisions

1. Start with links arriving among eight devices; distinguish connectivity from a route and from directed reachability. Introduce disjoint, partition, representative and the three operations with a worked merge table.
2. Show the same partition under different parent forests; parent links are implementation choices, not input edges. Explain eager labels, root self-parent, root-only union and idempotence before optimization.
3. Implement a complete, iterative, validated size-weighted Python class. Trace an ordinary union and explain initialization, preservation, termination and component count. Explain why root size, not a stale child entry, is consulted.
4. Contrast an unweighted chain with size weighting; prove the doubling bound. A separate compression investigation shows unchanged groups, discovered root, individual rewrites and next-query hop count. Distinguish size, rank and height, and amortized sequence cost from an individual guarantee.
5. Teach output enumeration and application models: redundant undirected links, equality classes, shared identifiers and incremental grid islands. Each application explains what is a vertex, what merges mean and what DSU cannot infer.
6. Explain absent deletion/path support and complete rollback-without-compression as optional depth. Link weighted graph algorithms for Kruskal's proof and advanced persistence for historical versions.
7. Independent changed-contract exercises have initially closed hints and full explained solutions, tests and costs. Official LeetCode practice connects direct reconstruction to modeling and output; no fixed difficulty or count quota.

## Visual contracts

- `UnionFindEquivalenceFigure`: two genuinely different forests for the same four-node partition, with original edge relation separately stated; a small inline figure fits narrow screens.
- `UnionFindLab`: bounded eight-element operations, independent editable endpoints and explicit union application, inspection before/after; forest geometry and parent/root-size table derive from each immutable snapshot, component membership shown separately. Balanced construction and chain presets expose different pointer depths. Undo within a trace is a display rewind, not a supported DSU deletion.
- `PathCompressionLab`: start from a computed size-weighted depth-three tree, select any element, follow parents then rewrite the path. Show before/current path and next lookup hops. Querying a root and a direct child remain meaningful cases. Root/group invariants persist.
- `IslandUnionLab`: five-by-five grid starts closed; activate a cell and join side-adjacent active components. Visible component IDs and count come from the model. Repeated activation does nothing; reset is explicitly a new grid, not dynamic edge deletion. Root deduplication/failed union prevents double decrement when two neighbors already connect.
- All controls keyboard-operable; diagrams have accurate text/table alternatives, visible pointer direction and readable local scroll for wide forests. Grid stays directly readable on mobile. Geometry is not a performance plot. No timers, random states or external engines.

## Code and verification

Self-contained Python examples: basic DSU, eager-label contrast, compression trace, redundant edge filter, equality consistency, shared identifier groups, active islands, root aggregate metadata, rollback and independent exercise solutions where complete code materially helps. Exact stdout executed under Python 3.12. Native invariants checked against explicit-set merging and graph flood fill, not the displayed implementation. JS model checked against independent graph closure across exhaustive small edge sequences and grids. Verify no cycles, correct root sizes/counts, immutable old snapshots, valid empty/invalid boundaries, root preservation during compression and rollback restoration.

Before completion: actual lesson registration by root, desktop/mobile/keyboard interactions, opened screenshots, absence of page overflow, useful empty/error states, route anchors, native/model checks, and a verification record containing concrete commands and counts. Source-ready is distinct from verified completion.

## Research ledger

Primary references inspected: Princeton Algorithms §1.5 (`https://algs4.cs.princeton.edu/15uf/`) for variants/doubling/incremental scope; Princeton UF API (`https://algs4.cs.princeton.edu/code/javadoc/edu/princeton/cs/algs4/UF.html`) for rank/halving contract and sequence bound. Course alternate: Princeton Algorithms Part I on Coursera, Union-Find module; the module listing was inspected (five videos from Dynamic Connectivity through Union-Find Applications), not the video playback. Official public statements checked for LeetCode 547, 684, 990, 721 and 1319 on 10 September 2026. Princeton's percolation specification (`https://coursera.cs.princeton.edu/algs4/assignments/percolation/specification.php`) and FAQ (`https://coursera.cs.princeton.edu/algs4/assignments/percolation/faq.php`) were read for the open-site model, one-based API difference and backwash warning. Original lesson examples and annotations, no copied solutions. The course's percolation model is an application, not a source of invented quantitative thresholds.
