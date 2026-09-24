# Authoring notes: Motion Planning & Path Planning

Canonical topic ID: `motion-planning-path-planning-rrt-a-trajectory-optimization`.

## 2026-09-10 — Connect exact graph routes to heuristic and physical planning

- Status: open.
- Origin: [Weighted graph lesson design](../WEIGHTED-GRAPHS-LESSON-DESIGN.md) and [verification](../WEIGHTED-GRAPHS-VERIFICATION.md), DSA position15.
- Destination and rationale: exact topic-plan CLI verified this existing Embodied Intelligence owner and its Configuration Space/Collision Checking prerequisite. A* and trajectory feasibility belong with the state geometry and constraints that give them meaning. No new topic or module reorder was needed.
- Existing coverage: the weighted graph lesson implements nonnegative Dijkstra with physical stale entries, finite route witnesses, negative-edge alternatives, 0/1 grids and Johnson's telescoping potentials. It does not teach a complete A* heuristic/state-space/physical planner.
- Learning benefit: derive f=g+h from a target-specific lower-bound estimate; distinguish admissibility, consistency, goal extraction and reopening behavior under the actual graph-search variant. Explain why setting a heuristic does not by itself justify correctness, and why SVG coordinates are not physical distances unless an explicit model says so.
- Proposed treatment: a bounded spatial scenario with collision checks and a heuristic whose inequality is actually demonstrated; compare Dijkstra and A* expansion count on the same applied state graph while confirming equal optimal path cost. Separate measured counts from runtime claims. Explicitly define four/eight neighbors, diagonal corner-cutting, units and state sufficiency. With inconsistent heuristics, derive and test reopening rather than reuse a closed-set rule silently.
- Scope boundary: for RRT and trajectory optimization, distinguish feasible from optimal, discrete route from continuous trajectory, obstacle inflation/configuration space, and approximation/stochastic completeness assumptions. These are proposals for the receiving topic's design, not a claim that the whole planning catalogue was audited.
- Evidence: live topic-plan ID/prerequisite; weighted graph's independently verified route/frontier/potential implementations; primary sources must be selected and rechecked by the receiving author for A* and motion-specific claims.
- Resolution: awaiting destination author assessment. Preserve/adapt/reroute with reasons when that scoped topic is implemented.
