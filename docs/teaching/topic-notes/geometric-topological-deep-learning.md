# Authoring notes: Geometric & Topological Deep Learning

Canonical topic ID: geometric-topological-deep-learning

## 2026-09-11 — Teach topology features and topology-aware architectures as separate mechanisms

- Status: open
- Origin: scoped [Topology/TDA design](../TOPOLOGY-TDA-LESSON-DESIGN.md).
- Destination and ownership: this existing frontier entry in the deep-learning catalogue is the best home for trainable persistence representations, differentiating topological objectives and learning on cells/simplices. Basic chain maps and persistence remain in Mathematical Foundations.
- Existing coverage: catalogue title present in §Frontier Architectures; no publication mapping or individual stable-ID brief found in the scoped source search. This is a planned owner, not an implemented teaching claim.
- Idea and benefit: distinguish (1) a fixed persistence descriptor supplied to an ordinary model, (2) trainable weighting/filtration and (3) message passing across vertices, edges and higher-dimensional cells. Learners should know the actual state, operation and invariance of each.
- Proposed treatment: build from TDA's boundary operators and exact small complex. For cell-based learning, name feature dimension, incidence orientation, cochain space and how an orientation change transforms features/operators. For differentiable persistence, identify pairings, ties/nondifferentiable changes and the domain of any gradient claim. Compare a topology-free baseline under the same split and input information.
- Example/visual/practice: a triangle's edge flow can have zero vertex boundary while still be a filled-face boundary. Show edge/face features and the actual operator result; contrast it with a persistence image's pixel features. A changed orientation or new face is an independent invariance/validity test. These are proposed instructional fixtures, not a measured model benefit.
- Prerequisites: Graph Fundamentals, TDA homology/persistence, linear algebra and backpropagation; use actual stable names when designing. Full empirical/architecture research is still required.
- Evidence: [Chazal–Michel2021](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2021.667963/full), §6.3.3 inspected10September2026 UTC for the distinct feature/learning roles; [Adams et al.2017](https://jmlr.org/papers/v18/16-337.html), §4 inspected for fixed persistence-image construction. These sources do not establish the correctness of a future differentiable layer. Investigate its original paper, implementation and gradient limitations when authoring.
- Resolution: pending destination design; do not infer new rollout authorization.
- Implementation/verification links: none yet.
