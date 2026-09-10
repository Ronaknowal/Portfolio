# Convex Optimization — independent source and mathematics review

10 September 2026. A bounded review by the DSA author while the root author completed the Convex lesson's native/browser work. Read the entire current lesson, all nine complete native programs, the full models/labs and [design](CONVEX-OPTIMIZATION-DESIGN.md). This is not another curriculum audit or a replacement for the root's browser/solver verification.

## Findings and resolution

All reported findings are resolved by the root author and independently rechecked at **2026-09-10 10:33:50.041 UTC**. No Convex lesson-owned source was edited by this reviewer.

| Initial finding | Reproducer and implication | Final change and check |
| --- | --- | --- |
| Tiny positive ridge penalties classified as nonunique | `ridgeCurvatureState('duplicate',1e-13,1,1)` used `μ>1e-12`, although every positive penalty gives a positive curvature floor. At 1e-16, subtracting nearly equal eigenvalue terms erased μ entirely. | Semantic uniqueness distinguishes λ=0 from λ>0; the known duplicate fixture uses μ=2λ and L=12+2λ. Independent cases 0,1e-16,1e-13,1e-12,1e-10,1e-8,.1 agree exactly with these formulas. |
| Cancellation in the tiny duplicate ridge solve | At λ=1e-8 the determinant expression returned .8333333481363071 instead of the fixture's stable coefficient `5/(6+λ)=.8333333319444445`. | Both coordinates use the exact fixture expression. Every small-penalty case matches; overflowing contour radii are explicitly rejected instead of emitting invalid geometry. |
| Approximate feasibility presented as exact feasibility | Budget 4 and candidate(2.5+5e-11,1.5) passed the 1e-10 slack, while its objective fell below the true feasible optimum; clamped actual error hid the issue. | Exact model feasibility is false for this point, its positive violation remains visible, and feasible-gap/error fields are null. Numerical solver tolerances remain separately described in the lesson. |
| Near-stationarity presented as zero belonging to a subgradient set | `softThresholdState(0,0,5e-11)` marked stationary even though its subgradient is the singleton{5e-11}, excluding zero. | Exact membership flag is false. Fixed quarter-step UI cases continue to work; no tolerance is silently substituted for the exact mathematical certificate. |

The visible ridge control's fixed choices were not affected by the initial small-penalty defect, but the model accepted every real penalty in[0,2]. Fixing the accepted-input contract prevents future controls or reuse from teaching the wrong uniqueness statement. The same distinction applies to the continuous certificate APIs.

Initial evidence is preserved independently of the recheck: [before](evidence/convex-independent-before.json), **10:31:44.652 UTC**, and [after](evidence/convex-independent-after.json), **10:33:50.041 UTC**. Initial false flags remain in the first record; they are not relabeled as successful evidence.

## Mathematical and pedagogical scope reviewed

No additional mathematical blocker was found in the convex-set/function definitions, line-restriction derivative tests with their domain hypotheses, segment proof that local minima are global, distinctions among existence/strict/strong convexity, constrained first-order optimality, triangle supporting-bound certificate, composition/epigraph construction, DCP versus mathematical convexity, fixed-objective ridge eigenmode recurrence, subgradient and proximal derivations, or the coupled total-variation example.

The reading flow makes the needed distinctions explicit: integer feasibility is separate from a convex relaxation, positive Hessian diagonal entries do not prove PSD, a nonzero boundary gradient can be optimal, a solver stopping is not a proof, intercept regularization changes the model, and applying scalar thresholding to observations does not solve a penalty on adjacent differences. The later duality/splitting/stochastic-optimizer material is routed rather than assumed as a hidden prerequisite. Plot values use stated formulas/fixtures, clipping and axis units; no fabricated empirical speed comparison was introduced. The reviewer checked source contracts, while the root author owns the actual plot/browser render review.

An additional exact-arithmetic check uses Python `Fraction`, not a solver:

- The six-reading total-variation certificate has valid subgradients and exact zero stationarity residual, objective **137/150**.
- The four-reading practice certificate likewise has exact zero stationarity residual, objective **15/8**.
- The retained ridge fixture gives coefficients **38/41,24/41**, fit **425/1681**, penalty **1010/1681**, total **35/41**, and exact zero stationary equation. This checks the actual original data and λ=.5 against direct algebra; it is not a claim to have recovered an unavailable historical Git body.
- The weighted allocation exercise's directional certificate and objective lower bound hold for **6,390** rational feasible points, alongside the general supporting-plane argument. Finite enumeration is supplementary evidence, not the proof itself.

Durable reproducer: `scripts/review-convex-mathematics.mjs`. With `CONVEX_REVIEW_PHASE=final`, it asserts the resolved accepted-input cases and the finite contour guard, then writes `scratch/convex-cross-review/final-results.json`. The exact Fraction program is emitted in that directory and run with the existing isolated Python runtime. Running without that variable writes a new current snapshot in scratch; the durable before/after evidence linked above remains unchanged.

Root production integration and the user's acceptance remain separate. The closed review does not authorize rewriting other mathematics lessons or claim a universal theorem from finite test counts.
