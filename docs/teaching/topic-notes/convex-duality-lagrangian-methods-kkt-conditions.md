# Authoring notes: Convex Duality & Lagrangian Methods (KKT Conditions)

Canonical topic ID: convex-duality-lagrangian-methods-kkt-conditions

## 2026-09-10 — Make certificate, necessity and shadow-price conditions precise

- Status: implemented
- Origin: [Convex Optimization design](../CONVEX-OPTIMIZATION-DESIGN.md), scoped review of its later teaching bridge.
- Destination and rationale: this topic owns multiplier/duality theory. Convex Optimization develops a supporting-plane lower bound and feasible-candidate gap first, so learners can connect a dual bound to an already understood certificate.
- Existing coverage: the current published KKT body says a zero multiplier means a constraint is not limiting, and describes Slater alongside sufficiency. These need exact distinctions in its already-authorized rewrite.
- Proposed treatment: derive weak duality and explain why feasible primal/dual values bound the optimum. For differentiable convex inequality functions and affine equalities, a KKT point is sufficient for global optimality; Slater-type qualification is relevant to necessity/multiplier existence/strong duality, not an extra requirement to validate an existing KKT certificate. An active constraint can have a zero multiplier. Multiplier sensitivity interpretation requires its appropriate differentiability/regularity/value-function qualifications.
- Example and assessment: minimize x² subject to x≥0. At x=0 the constraint is active and its KKT multiplier is0. Compare a shifted quadratic whose same boundary has positive multiplier. Ask what zero slack, zero price and lack of a local effect mean separately.
- Evidence: direct inspection of the current KKT source10 September2026, plus supporting-plane/optimality material in [Boyd/Vandenberghe/Nobel slides](https://web.stanford.edu/~boyd/cvxbook/bv_cvxslides.pdf), pages96–98. Research the actual duality/KKT and sensitivity sections before final teaching; the source's broader chapters were not read in this narrow origin review.
- Resolution: implemented in the 10 September KKT rewrite. Sections 2–4 derive the lower-bound chain and convex sufficiency; section 5 distinguishes Slater's existence/necessity role from validating a supplied certificate, including zero-gap dual nonattainment. The scalar lab and complete example show active zero multipliers. Section 6 derives supporting price estimates, differentiability, a corner with nonunique prices, finite changes and units/scaling. Independent changed-contract practice tests these distinctions.
- Implementation/verification links: [destination design](../CONVEX-DUALITY-KKT-DESIGN.md), [destination verification](../CONVEX-DUALITY-KKT-VERIFICATION.md), and `src/learn/data/topics/convex-duality-lagrangian-methods-kkt-conditions.jsx`. The original discovery and its partial-source-review scope remain above as provenance.
