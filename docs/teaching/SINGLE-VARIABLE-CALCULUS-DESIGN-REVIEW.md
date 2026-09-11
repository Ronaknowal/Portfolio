# Single-variable calculus: independent design and initial-contract review

Reviewed 11 September 2026 by the independent testing/documentation agent. This is a bounded review of the complete individual design, blueprint, initial pure models and all 15 displayed Python programs. It does **not** certify the later body, lab graphics, browser behavior or production integration.

## Findings and their resolution

1. The original signed-rectangle program asked why an accurate displacement could still misrepresent travel, but its chosen grids all gave the exact travel value 12 while displacement was inaccurate. The author added a three-panel grid and changed the question to distinguish the two limiting quantities and the coincidental exactness of some finite grids. Executed outputs are now 148/9 and 92/9 for the left and midpoint three-panel travel estimates. The four/eight-panel coincidences must remain explicitly identified in the actual explanation.
2. The original floating-point limit decision rejected epsilon = .84, delta = .2 even though the supremum equals .84 and the input strip is open. Its proposed counterexample lay on the excluded boundary. The author now interprets the displayed decimals as exact rational inputs, compares them exactly and supplies a certified strict-interior rational witness when the guarantee fails. The equality case succeeds; epsilon = .83 has witness 1407/640. For epsilon = .8399999999999999 the exact witness is valid although rounded geometry cannot resolve it. The model marks that distinction; the eventual lab must display the exact fraction and explain the rounded figure rather than imply a visible strict separation.

No further actionable design or mathematical-contract defect was found in this finite read. The prerequisite choices (Algebra, Sets/Logic and Geometry) avoid a silent dependency on later numerical or multivariate calculus. The motion example supplies consistent rate, displacement, travel and unit meanings. Rolle/MVT and both FTC directions have explicit hypotheses; defining logarithm by the integral of 1/t before deriving the inverse exponential avoids circular derivative arguments. The design distinguishes a finite Taylor approximation and bound from infinite-series convergence, and a truncated improper integral from its limit.

## Complementary execution

Run `node scripts/verify-single-variable-calculus-design-review.mjs`; it invokes the isolated lesson Python and [the independent oracle](../../scripts/verify-single-variable-calculus-design-review.py). The [result and exact four inspected-source hashes](../../scratch/single-variable-calculus-design-review/results.json), captured at 00:47:38 UTC, record:

- 270 exact rational limit decisions and witness contracts;
- 48 rational secant/remainder cases using the expanded position polynomial;
- 144 independently integrated displacement/travel and exact rational rectangle cases;
- 40 expanded composition checks and 72 high-precision Taylor comparisons;
- execution and exact stdout comparison of all 15 actual stored programs.

The review used independently expressed polynomial calculations, SciPy quadrature and 75-digit mpmath series. It did not substitute sampled numeric evidence for the stated limiting proofs. Source changes after the recorded hashes require appropriately scoped follow-up; final pedagogical/visual verification remains the author's responsibility.
