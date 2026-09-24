# Incoming discovery for Single-Variable Calculus

## 2026-09-11 — From repeated multiplication to a continuous rate

- Status: implemented in the Calculus47 draft; final independent review pending. Proposed during Algebra43 implementation.
- Origin: [Algebra design](../ALGEBRA-FUNCTIONS-LESSON-DESIGN.md).
- Current treatment: the finite subdivision sequence(1+1/m)^m motivates the number e; the convergence claim is stated, not proved by a finite table. A(t)=A0 exp(kt) has dimensionless kt; k is the continuous-rate parameter, distinct from a discrete fraction r where k=ln(1+r) per period. No derivative is assumed.
- Best owner and purpose: this Calculus lesson can now derive the derivative of exp and the instantaneous relative-rate interpretation, using the locally developed limit machinery. Give a complete proof or clearly declared theorem, then distinguish average and instantaneous rates.
- Prerequisites: Algebra logarithm/inverse and positive-base conventions; limits taught at the destination. Reasoned proposal, not a mandate to copy a prior visual.
- Disposition: included in section9 of `src/learn/data/topics/single-variable-calculus-limits-derivatives-integrals.jsx`. The lesson constructs ln from an integral, proves its product law and inverse exponential derivative, bounds n·ln(1+1/n) to prove the compounding limit, and derives the relation between a period's fraction and a continuous rate. The exponential-rate investigation separates the local derivative, finite change and finite average, with a zero/negative-rate comparison. Section12 supplies an integrating-factor uniqueness argument. All15 actual programs and264 high-precision growth states passed in `scratch/single-variable-calculus-verification/results.json`; final browser/independent evidence is maintained in the Calculus verification record. No originating Algebra body was changed.
