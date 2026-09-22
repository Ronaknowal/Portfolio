# Authoring notes: Spectral Graph Theory

Canonical topic ID: `spectral-graph-theory`.

## 2026-09-21 — Pair the visible mechanism with an executed maintained-tool route

- Status: resolved — independently reviewed and integrated on 22 September 2026.
- Origin: [Foundation implementation-depth audit](../implementation-depth/FOUNDATIONS.md), a source audit rather than a new full correctness review.
- Source inspected: `src/learn/data/topics/spectral-graph-theory.jsx`, §§5 and 9, especially source lines 146–147; spectral-graph-examples.js; the directly imported example file is under `src/learn/data/`.
- Existing coverage: The complete normalized-eigenvector/Lloyd pipeline is displayed. SciPy eigsh and scikit-learn SpectralClustering are named but their actual library variants are not executed for comparison.
- Proposed treatment and closure check: Add a same-affinity maintained-library clustering example with explicit normalization, assignment strategy and seed; compare partitions modulo label permutations and graph cut/embedding quantities. Include an isolated vertex or repeated eigenvalue discussion so different valid bases are not labeled bugs.
- Ownership and boundary: This topic owns spectral normalization and clustering conventions; reuse Graphs for constructing vertices/edges and Eigenvalues for residual checks.
- Quality requirement: preserve algorithm invariants, explicit supported inputs, appropriate time/space costs, numerical stability and failure behavior. Distinguish a transparent teaching implementation from an efficient maintained implementation; claim optimality only for a stated cost model with evidence. A library import in a verifier does not satisfy the learner-facing route.
- Evidence limits: this finding comes from the local learner-visible source and its complete-program ownership. This audit did not newly research or execute the suggested package; its version, contracts, setup and expected output remain to verify during the scoped repair. Existing reviewed implementation checkpoints are not revoked by this additional depth finding.
- Resolution: the existing mechanism programs are preserved. The lesson now exposes `public/learn-assets/spectral-graph/spectral_library.py` through a lazy complete-program viewer, with explicit API/mechanism mappings, interpreted recorded output, computation limits and changed-constraint practice. [Author report](../implementation-depth/MATH-LIBRARY-REMEDIATION.md); [native receipt](../evidence/spectral-graph-library-native.json). This closes the missing learner-facing route at author level; final review/integration is separately recorded by the integration owner.
- Final implementation resolution, 22 September 2026: the learner-facing mechanism/tool route, contract comparison and changed-constraint practice are implemented; native checks, independent review and affected production checks pass. [Final scope and evidence](../implementation-depth/REMEDIATION.md). Earlier proposal/evidence-limit wording above is historical, not an unresolved task.

