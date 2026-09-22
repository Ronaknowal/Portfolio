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

## 2026-09-21 — Pair the visible mechanism with an executed maintained-tool route

- Status: resolved — independently reviewed and integrated on 22 September 2026.
- Origin: [Foundation implementation-depth audit](../implementation-depth/FOUNDATIONS.md), a source audit rather than a new full correctness review.
- Source inspected: `src/learn/data/topics/computational-geometry-robust-predicates-convex-hulls.jsx`, §§3–6; computational-geometry-examples.js; the directly imported example file is under `src/learn/data/`.
- Existing coverage: Integer/rational predicates and monotone-chain hull code are visible; CGAL and adaptive predicates are references, not a runnable library comparison.
- Proposed treatment and closure check: Provide one researched, executed hull route in a maintained geometry library, with explicit coordinate type, collinear-boundary and duplicate policies. Compare the same nondegenerate fixture, then a degeneracy where contracts can differ. Explain why floating Qhull output is not an oracle for arbitrary exact-integer predicates.
- Ownership and boundary: This is a small production-use bridge inside the current planar scope, not reopening the separately deferred Voronoi/3D/sweep-line extensions.
- Quality requirement: preserve algorithm invariants, explicit supported inputs, appropriate time/space costs, numerical stability and failure behavior. Distinguish a transparent teaching implementation from an efficient maintained implementation; claim optimality only for a stated cost model with evidence. A library import in a verifier does not satisfy the learner-facing route.
- Evidence limits: this finding comes from the local learner-visible source and its complete-program ownership. This audit did not newly research or execute the suggested package; its version, contracts, setup and expected output remain to verify during the scoped repair. Existing reviewed implementation checkpoints are not revoked by this additional depth finding.
- Resolution: `GeometryLibraryBridge.jsx` now compares the exact local corner hull with SciPy/Qhull on declared small float64 coordinates, executes a collinear exception case, explains corner/boundary and area/perimeter conventions, and adds translation-before-conversion practice. Exact integer predicates retain their own contract; no floating oracle or specialist-family completion is claimed. See [DSA remediation](../implementation-depth/DSA-REMEDIATION.md#computational-geometry-robust-predicates--convex-hulls) and its [source manifest](../implementation-depth/dsa-remediation.json). Author checks pass; independent review and final integration remain separate.
- Final implementation resolution, 22 September 2026: the learner-facing mechanism/tool route, contract comparison and changed-constraint practice are implemented; native checks, independent review and affected production checks pass. [Final scope and evidence](../implementation-depth/REMEDIATION.md). Earlier proposal/evidence-limit wording above is historical, not an unresolved task.

