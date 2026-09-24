# Authoring note: Configuration Space, Collision Checking & Feasible Trajectories

Canonical topic ID: `configuration-space-collision-checking-feasible-trajectories`.

## 2026-09-10 — Carry geometric contact contracts into physical motion

- Status: open, for destination author assessment.
- Origin: [Computational Geometry lesson design](../COMPUTATIONAL-GEOMETRY-LESSON-DESIGN.md), DSA position 20.
- Destination evidence: the exact topic-plan command identifies this existing, planned Embodied Intelligence topic. Its brief already covers workspace/configuration-space mapping, obstacle inflation, continuous edge checks, joint limits and trajectory timing. This is the more direct home than the downstream Motion Planning topic; no catalogue expansion or prerequisite reordering was performed.
- Already taught at the origin: exact planar orientation and segment membership; closed segment classifications distinguishing proper crossing, touch and overlap; conservative bounding-box rejection; the difference between a concave footprint and its convex hull; an entire point path versus its endpoint checks. Integer arithmetic correctness does not establish measurement certainty.
- Proposed bridge: reuse one small path with both endpoints clear but a colliding interior, then replace the point by a disk or other explicitly defined footprint. State whether boundary contact is permitted and show what changes under that decision. Explain the geometry and assumptions behind configuration-space obstacle inflation, rather than inferring finite-body safety from center-point tests. Retain a counterexample where overlapping bounds are only a candidate and the narrow-phase answer is negative.
- Visual benefit: linked workspace/configuration-space views with the same pose/path, visible footprint and swept region would expose the missing relationship. Use a model with actual collision predicates and stated coordinate units; avoid treating attractive drawing coordinates as a physical guarantee.
- Required reassessment: this is a proposed connection, not a mandate to repeat the origin's determinant chapter or an assertion that existing planned coverage is absent. The receiving author should preserve, adapt or reroute it with reasons, inspect appropriate primary robotics sources, and test continuous-motion/discretization assumptions. Its existing brief may already supply the best treatment.
- Resolution: awaiting implementation of the destination topic.
