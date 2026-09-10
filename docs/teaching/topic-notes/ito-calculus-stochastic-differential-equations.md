# Authoring notes: Itô Calculus & Stochastic Differential Equations

Canonical topic ID: ito-calculus-stochastic-differential-equations

## 2026-09-10 — Preserve sampled-path, quadratic-variation and conditional-bridge distinctions

- Status: open.
- Origin: mathematics 33, [Stochastic Processes design](../STOCHASTIC-PROCESSES-LESSON-DESIGN.md). The origin is designed, not yet implemented or verified.
- Destination and ownership rationale: the origin supplies Brownian increments and finite observation limits. Itô owns adapted stochastic integration, the extra chain-rule term, general SDE discretization and numerical convergence.
- Existing coverage inspected: current Itô source sections 2–3 state differential bookkeeping and Itô's formula; section 5 gives a GBM Euler example; sections 6/8 discuss conventions and numerical checks. It has no worked adapted-versus-look-ahead integral comparison, explicit coupled-grid experiment or between-sample crossing treatment.
- Proposed treatment: connect the origin's equal deterministic partition calculation E(sum deltaW²)=T and variance 2T²/n to the second-order Taylor term. Establish the stated mode of convergence instead of applying an ordinary iid law of large numbers across a triangular array or claiming arbitrary path-dependent partitions. Distinguish square-integrable/locally defined integrals and adaptedness at an appropriate level.
- Example and learning benefit: compare the left-point sum for W against a midpoint sum on the same Brownian samples, with their different limits and a clear convention. Couple fine/coarse SDE runs through shared Brownian increments to separate trajectory error from a changed random draw. The origin's conditional midpoint has mean(x+y)/2 and varianceDelta/4.
- Threshold connection: constant-coefficient Brownian motion with endpoints x,y below b has conditional crossing probability exp(-2(b-x)(b-y)/(sigma²Delta)). A coarse plot can miss crossings even when its sampled Brownian values are exact. Extending a local Brownian-bridge approximation to state-dependent SDE coefficients needs a separate accuracy claim, not automatic exactness.
- Prerequisites/boundaries: introduce conditioning, histories, moments and stochastic increment scaling; do not assume a chart proves nowhere differentiability. Keep model error, numerical error and sampling error distinct. Formal existence/uniqueness or solver-order assertions require their actual regularity assumptions.
- Evidence: [Haugh SDE simulation slides](https://www.columbia.edu/~mh2078/MonteCarlo/MCS_SDEs_MasterSlides.pdf), conditional construction slides 23–24 and barrier discussion 28–31; [MIT reflection notes](https://ocw.mit.edu/courses/15-070j-advanced-stochastic-processes-fall-2013/aca1518a09539a09ddd37428ab0d0268_MIT15_070JF13_Lec7.pdf), sections 1–2; [MIT Stochastic Processes II notes](https://ocw.mit.edu/courses/18-s096-topics-in-mathematics-with-applications-in-finance-fall-2013/3b97c6b0c282dd9dc024c4c7ffe3fba8_MIT18_S096F13_lecnote17.pdf), quadratic-variation motivation. Read 10 September 2026; origin plans direct moment and reflection checks. Do not treat every informal source convergence sentence as a sufficiently qualified theorem.
- Resolution: not yet reviewed by destination author; no destination body changed.
- Implementation/verification links: none yet.
