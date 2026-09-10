# PAC Learning & VC Dimension — authoring notes

Canonical topic ID: `pac-learning-vc-dimension`.

## 2026-09-10 — Extend finite-family concentration without losing the selection contract

- Status: open for the destination author's assessment.
- Origin: [Concentration Inequalities design](../CONCENTRATION-INEQUALITIES-LESSON-DESIGN.md), section 7 and changed exercise 7 of `concentration-inequalities-hoeffding-bernstein-chernoff`.
- Ownership: this destination is the natural primary home for going from a finite fixed class to shattering/growth/VC-based generalization. Concentration teaches the individual bounded event and a finite union; MCMC teaches dependent estimation. Do not duplicate the foundational MGF chapter here.
- Existing scope checked: exact catalogue identity and the current source's opening PAC/VC sections were read. They discuss sample complexity, thresholds, intervals and half-planes, but the body is not current-standard reviewed. The finite-family bridge below is taught and independently checked in the origin, not a claim that this destination lacks every such explanation.
- Proposed treatment: start with K predictors fixed before evaluation and independent bounded-loss observations. Union-bound failure at δ/K for each predictor; independence across predictors is unnecessary even when they share the same validation sample. Explain why the simultaneously protected event still covers the selected member. Then show why an infinite or outcome-invented class cannot simply substitute its cardinality into that calculation, motivating the appropriate capacity/uniform-convergence argument.
- Concrete changed anchor: K=25, n=500, losses in [0,1], δ=.05 gives two-sided finite-family radius sqrt(log(1000)/1000)≈.08311291. A family chosen using the same validation outcomes needs additional reasoning; merely counting the final retained candidates does not establish the predeclared-family contract.
- Accuracy reassessment: the existing opening calls a sufficient VC-style sample bound the “minimum data,” and phrases finite-VC learnability broadly beside a polynomial-efficiency PAC definition. During the scoped rewrite, investigate the precise realizable/agnostic, statistical/computational and regularity hypotheses rather than copying those sentences as universal guarantees. This is a narrow source-reading concern, not a completed audit of the destination's historical references.
- Prerequisites: probability/expectation, independent bounded observations, union bounds, finite combinatorial labelling. Introduce the learning setting and quantifiers before any theorem.
- Evidence: Bartlett's Berkeley lecture https://www.stat.berkeley.edu/~bartlett/courses/2014spring-cs281bstat241b/lectures/04-notes.pdf, page 4 finite-family reasoning inspected; the origin's exact family checks and changed numerical calculation pass. A future author must inspect primary PAC/VC sources for the stronger claims and retain meaningful existing figures only after checking them.
- Resolution: not yet implemented in the destination; no title, order or publication change made by this note.
