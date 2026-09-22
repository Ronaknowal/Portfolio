# Authoring notes: Graphs: Representations, BFS & DFS

Canonical topic ID: `graphs-representations-bfs-dfs`.

## 2026-09-21 — Pair the visible mechanism with an executed maintained-tool route

- Status: resolved — independently reviewed and integrated on 22 September 2026.
- Origin: [Foundation implementation-depth audit](../implementation-depth/FOUNDATIONS.md), a source audit rather than a new full correctness review.
- Source inspected: `src/learn/data/topics/graphs-representations-bfs-dfs.jsx`, §§2–7; graph-traversal-examples.js; the directly imported example file is under `src/learn/data/`.
- Existing coverage: Explicit adjacency construction, deque BFS, DFS frames, components and graph cloning are learner-visible. The source does not provide a runnable maintained graph-tool route on the same input.
- Proposed treatment and closure check: Add one small ordinary graph-library construction and traversal comparison after the explicit BFS. Preserve isolated vertices and declared direction; compare reachable distances and valid paths rather than insisting on an identical tie-dependent traversal order. Ask learners to change direction or add an isolated vertex and explain both results.
- Ownership and boundary: This topic should own reusable graph-library construction; weighted graph and flow lessons can reuse that bridge while teaching their own algorithm contracts.
- Quality requirement: preserve algorithm invariants, explicit supported inputs, appropriate time/space costs, numerical stability and failure behavior. Distinguish a transparent teaching implementation from an efficient maintained implementation; claim optimality only for a stated cost model with evidence. A library import in a verifier does not satisfy the learner-facing route.
- Evidence limits: this finding comes from the local learner-visible source and its complete-program ownership. This audit did not newly research or execute the suggested package; its version, contracts, setup and expected output remain to verify during the scoped repair. Existing reviewed implementation checkpoints are not revoked by this additional depth finding.
- Resolution: `GraphTraversalLibraryBridge.jsx` now pairs the unchanged BFS/DFS/component mechanisms with an executed NetworkX construction and traversal program, preserves isolated vertices/direction, checks distances and route witnesses, and adds reversed-edge practice. The author checks include exhaustive tiny directed graphs and source-bound native replay. See [DSA remediation](../implementation-depth/DSA-REMEDIATION.md#graphs-representations-bfs--dfs) and its [source manifest](../implementation-depth/dsa-remediation.json). This resolves the authoring gap; final reviewed implementation status remains with integration.
- Final implementation resolution, 22 September 2026: the learner-facing mechanism/tool route, contract comparison and changed-constraint practice are implemented; native checks, independent review and affected production checks pass. [Final scope and evidence](../implementation-depth/REMEDIATION.md). Earlier proposal/evidence-limit wording above is historical, not an unresolved task.

