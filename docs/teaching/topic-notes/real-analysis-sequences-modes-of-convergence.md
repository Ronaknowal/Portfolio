# Authoring notes: Real Analysis, Sequences & Modes of Convergence

Canonical topic ID: `real-analysis-sequences-modes-of-convergence`

## 2026-09-10 — Connect quantifier order to the measure lesson's fixed-point and moving-spike examples

- Status: open
- Origin: [Measure Theory design](../MEASURE-THEORY-PROBABILITY-SPACES-DESIGN.md), section 6 of `measure-theory-probability-spaces`.
- Destination and ownership rationale: exact topic-plan command confirms a planned analysis topic with pointwise/uniform convergence and invalid interchange as explicit outcomes. This is the correct home for the epsilon/N quantifier comparison; keep advanced probability convergence criteria in a clearly optional bridge rather than making measure theory a prerequisite of the introductory core.
- Idea and learning benefit: distinguish one fixed x from the changing choice x=1/(2n). In f_n=n·1_(0,1/n), every fixed point eventually sees zero, although the integral remains one and the supremum does not decrease. Compare x^n on [0,1], which has an exceptional limit value at 1, with uniform convergence on [0,r] for fixed r<1. Tie uniform convergence on a finite-measure domain to |∫f_n−∫f|≤μ(domain) sup|f_n−f|, explaining why that argument needs finite measure.
- Existing coverage: the destination brief (inspected 10 September) already plans pointwise versus uniform convergence and a limit-integral counterexample. The current Measure Theory lesson supplies MCT, Fatou, DCT and the exact fixed-point/integral comparison, but does not teach a complete map of modes of convergence or uniform integrability.
- Proposed treatment: use changed functions and explicit quantifier order in the core; link back to the Measure Theory counterexample only for readers with that context. Assess an optional uniform-integrability bridge after probability convergence has been introduced. Do not equate bounded L1 norms with uniform integrability or claim it follows from one fixed sample plot.
- Evidence: [Axler's author-hosted book](https://measure.axler.net/MIRA.pdf), scoped sections 3.9/3.11 and 3.31 inspected for the measure lesson; finite teaching models independently verified. The stronger optional uniform-integrability theorem has not been researched in this authoring scope and requires its own primary-source/hypothesis review.
- Prerequisites and boundaries: elementary sequences and epsilon definitions for the core; measurable random variables and probability convergence for any optional uniform-integrability statement. No prerequisite cycle should be introduced merely to cross-link an application.
- Resolution: receiving author should merge with the already planned outcomes and decide whether the optional probability branch helps this topic or should be routed to a stochastic-process topic. No destination body or title changed.
- Implementation/verification links: origin `scripts/verify-measure-theory.mjs`, `scratch/measure-theory-verification/native-results.json`, and [Measure Theory verification](../MEASURE-THEORY-PROBABILITY-SPACES-VERIFICATION.md).
