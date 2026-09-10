# Authoring notes: Functional Analysis & RKHS

Canonical topic ID: `functional-analysis-rkhs`.

## 2026-09-10 — Explain when a kernel mean represents an entire distribution

- Status: open
- Origin: [f-Divergences & IPMs design](../F-DIVERGENCES-IPMS-LESSON-DESIGN.md), finite feature-mean and MMD bridge.
- Destination and ownership rationale: the existing published RKHS body explains evaluation representers and Gram positivity, but the inspected source does not develop mean-embedding existence or characteristic kernels. This later mathematics topic can supply the abstract conditions after the earlier MMD lesson introduces a finite geometric example and practical pair-sum calculation.
- Idea and learning benefit: distinguish positive-semidefinite kernel validity from injectivity of the map P→E_P k(X,·). A linear kernel compares means; a finite polynomial feature map can match several moments while different distributions remain. Kernel validity alone does not make MMD a separating metric on all laws.
- Proposed treatment: connect the reproducing identity and bounded expectation functional to the kernel mean; explain a sufficient integrability condition E sqrt(k(X,X))<infinity and why a bounded Gaussian kernel satisfies it. Show a changed pair with identical first two moments but different laws, then explain fixed positive-bandwidth Gaussian characteristic behavior on R^d with properly scoped assumptions. More general characteristic/universal relationships should be researched, not treated as synonyms.
- Prerequisites and boundaries: local MMD derivation supplies finite vectors and pair sums; this topic supplies completeness/Riesz/continuity details if appropriate to its actual final scope. Do not force advanced RKHS theory into the beginner MMD introduction.
- Evidence: [Gretton et al. 2012, sections2.2–2.3](https://jmlr.org/papers/volume13/gretton12a/gretton12a.pdf) read 10 September 2026; original mean-existence and characteristic-kernel references remain to assess for deeper treatment.
- Resolution: not yet assessed by the destination author; retain title unless final scoped review supports a change.
- Implementation/verification links: [f-Divergences/IPMs source verification](../F-DIVERGENCES-IPMS-VERIFICATION.md); destination work remains open.
