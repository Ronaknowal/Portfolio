# Authoring notes: Randomized Linear Algebra

Canonical topic ID: `randomized-linear-algebra`.

## 2026-09-21 — Pair the visible mechanism with an executed maintained-tool route

- Status: resolved — independently reviewed and integrated on 22 September 2026.
- Origin: [Foundation implementation-depth audit](../implementation-depth/FOUNDATIONS.md), a source audit rather than a new full correctness review.
- Source inspected: `src/learn/data/topics/randomized-linear-algebra.jsx`, §7, especially source lines 121–123; randomized-linear-algebra-examples.js; the directly imported example file is under `src/learn/data/`.
- Existing coverage: The complete range-finder/randomized-SVD helper, block passes and validation probes are displayed. The lesson explicitly says scikit-learn randomized_svd was not executed; the production route is documentation only.
- Proposed treatment and closure check: Execute the maintained randomized_svd route with explicit rank, oversampling, power iterations, normalization and random state, then compare reconstruction/subspace errors on the same prepared matrix. Do not demand equal random factors or singular-vector signs across implementations. Practice should change spectrum/rank and choose a budget from an error requirement.
- Ownership and boundary: This topic owns randomized decomposition options; reuse NumPy and matrix decomposition primitives, without reimplementing BLAS/QR/SVD.
- Quality requirement: preserve algorithm invariants, explicit supported inputs, appropriate time/space costs, numerical stability and failure behavior. Distinguish a transparent teaching implementation from an efficient maintained implementation; claim optimality only for a stated cost model with evidence. A library import in a verifier does not satisfy the learner-facing route.
- Evidence limits: this finding comes from the local learner-visible source and its complete-program ownership. This audit did not newly research or execute the suggested package; its version, contracts, setup and expected output remain to verify during the scoped repair. Existing reviewed implementation checkpoints are not revoked by this additional depth finding.
- Resolution: the existing mechanism programs are preserved. The lesson now exposes `public/learn-assets/randomized-linear-algebra/randomized_svd_library.py` through a lazy complete-program viewer, with explicit API/mechanism mappings, interpreted recorded output, computation limits and changed-constraint practice. [Author report](../implementation-depth/MATH-LIBRARY-REMEDIATION.md); [native receipt](../evidence/randomized-linear-algebra-library-native.json). This closes the missing learner-facing route at author level; final review/integration is separately recorded by the integration owner.
- Final implementation resolution, 22 September 2026: the learner-facing mechanism/tool route, contract comparison and changed-constraint practice are implemented; native checks, independent review and affected production checks pass. [Final scope and evidence](../implementation-depth/REMEDIATION.md). Earlier proposal/evidence-limit wording above is historical, not an unresolved task.

