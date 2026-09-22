# Authoring notes: Topology & Topological Data Analysis (TDA)

Canonical topic ID: `topology-topological-data-analysis-tda`.

## 2026-09-21 — Pair the visible mechanism with an executed maintained-tool route

- Status: resolved — independently reviewed and integrated on 22 September 2026.
- Origin: [Foundation implementation-depth audit](../implementation-depth/FOUNDATIONS.md), a source audit rather than a new full correctness review.
- Source inspected: `src/learn/data/topics/topology-topological-data-analysis-tda.jsx`, §5, especially source line 130, and GUDHI reference near 260; topology-tda-examples.js; the directly imported example file is under `src/learn/data/`.
- Existing coverage: The complete F2 boundary reducer and finite Rips/cubical mechanisms are visible; GUDHI 3.13 documentation is explicitly distinguished from actual execution.
- Proposed treatment and closure check: Add an executed maintained-library Rips persistence example with the same metric, filtration scale, coefficient field, required simplex dimension and cutoff as the square fixture. Compare diagram multisets with multiplicity, finite versus essential/censored bars and tolerance, rather than basis-dependent representative chains. Transfer to changed point spacing or missing faces.
- Ownership and boundary: Retain the small reducer for mechanism learning; do not present its cubic worst-case implementation as the preferred large-data route.
- Quality requirement: preserve algorithm invariants, explicit supported inputs, appropriate time/space costs, numerical stability and failure behavior. Distinguish a transparent teaching implementation from an efficient maintained implementation; claim optimality only for a stated cost model with evidence. A library import in a verifier does not satisfy the learner-facing route.
- Evidence limits: this finding comes from the local learner-visible source and its complete-program ownership. This audit did not newly research or execute the suggested package; its version, contracts, setup and expected output remain to verify during the scoped repair. Existing reviewed implementation checkpoints are not revoked by this additional depth finding.
- Resolution: the existing mechanism programs are preserved. The lesson now exposes `public/learn-assets/topology-tda/gudhi_rips.py` through a lazy complete-program viewer, with explicit API/mechanism mappings, interpreted recorded output, computation limits and changed-constraint practice. [Author report](../implementation-depth/MATH-LIBRARY-REMEDIATION.md); [native receipt](../evidence/topology-tda-library-native.json). This closes the missing learner-facing route at author level; final review/integration is separately recorded by the integration owner.
- Final implementation resolution, 22 September 2026: the learner-facing mechanism/tool route, contract comparison and changed-constraint practice are implemented; native checks, independent review and affected production checks pass. [Final scope and evidence](../implementation-depth/REMEDIATION.md). Earlier proposal/evidence-limit wording above is historical, not an unresolved task.

