# Authoring notes: Computational Geometry, Robust Predicates & Convex Hulls

Canonical topic ID: `computational-geometry-robust-predicates-convex-hulls`.

## 2026-09-10 — Preserve the finish line and reassess specialist continuations

- Status: reasoned deferral for a future scope assessment; current planar lesson is implemented, not a claim of full specialist-field coverage.
- Origin: [lesson design](../COMPUTATIONAL-GEOMETRY-LESSON-DESIGN.md).
- Current ownership: orientation and signed area, segment contact/degeneracy, exact versus rounded inputs and constructions, adaptive-predicate concepts, monotone convex hull with two boundary contracts, polygon area and boundary-first containment, linear support/bounds applications and canonical rational direction grouping. Complete local examples and practice are self-contained.
- Additional families: Voronoi diagrams/Delaunay triangulation and incircle predicates; sweep-line arrangements and intersection reporting; spatial indexing; 3D orientation/hulls; specialized ordered-hull queries and rotating calipers; spherical geometry. The lesson explicitly names these as further study without presenting an unexplained partial implementation.
- Why defer: these require additional predicates, representations, assumptions and proofs. Adding all of them to a first planar-predicate/hull page would make the learning path less coherent and imply more coverage than the implementation provides. A scoped inventory title search did not identify a dedicated owner for these specific families; this was not a full site audit. Keep this topic as the receiving assessment point unless a better existing owner is confirmed.
- Future author action: inspect current catalogue and notes, determine which extension genuinely belongs here, and either deepen a coherent local section or record a justified destination. Do not silently label all computational geometry covered, rename the stable ID, or create a duplicate lesson simply because a resource mentions a technique.
- Routed connection: finite-footprint/continuous collision is assigned for assessment to [Configuration Space, Collision Checking & Feasible Trajectories](configuration-space-collision-checking-feasible-trajectories.md), whose existing brief already owns those mechanisms. Do not duplicate that robotics chapter here.
- Resolution: current scope retained; specialist expansion remains a documented future decision, not an unfinished core exercise.
