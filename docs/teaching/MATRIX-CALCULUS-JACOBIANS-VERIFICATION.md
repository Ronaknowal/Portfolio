# Matrix Calculus & Jacobians — verification

10 September 2026. Stable ID `matrix-calculus-jacobians`. [Design, coverage and source review](MATRIX-CALCULUS-JACOBIANS-DESIGN.md). Implementation and author review are complete; user acceptance and the parent's final integrated build remain separate.

## Retained and improved learning

Preserved the original two-output polynomial at (2,3), its (0.01,−0.02) input change, Jacobian convention, composition rule, reverse-mode motivation, affine gradient formulas and original 4×3 / 3×2 / 4×2 shape exercise. Those are now part of a self-contained progression with a scalar/partial calculus refresher, exact remainder, visible JVP/VJP distinction, branch accumulation and fully instantiated batch loss/gradients. Explicitly reconciles column mathematical vectors with row-stored batches. Independent practice addresses both numerical answers and explanations, including shape-valid wrong formulas.

Four distinct labs expose local approximation curves, forward/reverse derivative propagation, per-observation parameter contributions and computed finite-difference errors. Four inline representations supply a labelled Jacobian, two influence paths, shared-bias fan-out/gradient sum and matrix-square product-rule terms. Six independent tasks contain optional hints and explained solutions. Ten complete programs cover the core and optional dual arithmetic, matrix square, solve sensitivity, softmax coupling and Hessian branches. The title, stable route and next Tensor Algebra & Einsum module bridge are retained.

## Numerical evidence

Command: `node scripts/verify-matrix-calculus.mjs`; native oracle: `scripts/verify-matrix-calculus-native.py`. Actual pass at 07:57:27 UTC in `scratch/matrix-calculus-verification/results.json`, Python 3.12.14 / NumPy 2.3.5. No framework or dependency installation needed.

| Check | Actual result |
| --- | --- |
| Displayed complete programs | 10 exact stdout comparisons passed |
| Local models | 5,915 point/direction/step cases versus independent complex-step Jacobians and exact polynomial remainders |
| Forward/reverse chain | 4,225 cases versus complex-step derivatives of the directly evaluated vector/scalar function, plus independent transpose pairings |
| Affine gradients | 60 individual X/W/b derivatives versus complex-step perturbations of the complete objective; each highlighted observation contribution also checked separately |
| Finite differences | 120 smooth/kink/step cases; exact polynomial expansions with scale-aware roundoff bounds, left/right kink limits, rounded-away steps |
| Learner dual program | 16 additional unseen inputs/directions versus a separate derivative oracle |
| Matrix operators | 300 seeded square/inverse/solve derivative cases versus complex-step native NumPy calculations |
| Softmax | 100 unseen score fixtures; complex-step Jacobian, shift invariance and null direction |
| Transfer/counterexamples | Four groups: new Jacobian, scalar chain, partial-existence counterexample and Hessian checks |
| Invalid API inputs | 11 grouped finite/shape/fixture/reduction/step guards |

Complex-step is used only for the smooth analytic functions where that reference is valid; absolute value at its kink is checked through one-sided slopes. The finite-difference lab genuinely uses floating arithmetic, while the standalone Decimal example separates exact truncation error from early double-precision cancellation. The initial oracle accidentally allocated an integer gradient buffer for integer input fixtures; changing that reference buffer to floating point corrected the harness. No lesson gradient bug was concealed by that fix.

## Integrated browser evidence

Command: `node scripts/review-matrix-calculus.cjs`. Full checks use Edge headless at **1440×1000 and 390×1000** with reduced motion. The final pass at **08:15:27 UTC** is recorded in `scratch/matrix-calculus-browser/results.json`, after an explicit-value selector correction; at each width:

- 112 local point/direction/step states with rates, changes/remainders, valid curve coordinates and deterministic reset.
- 128 forward/reverse combinations across input, output-weight and tangent seeds; derivative-stage reset, next/previous/complete controls and separately derived scalar-gradient expectations.
- 36 affine fixture/objective/selected-parameter states, exact contribution sums, a single selected parameter and reset; bias cancellation checked separately.
- 80 finite-difference function/point/step states, undefined-derivative messaging at the kink, correct one-sided slopes, valid curves and reset.
- Ten rendered complete programs, ten working unique section anchors, six independent-practice disclosures, source/next-topic links, keyboard traversal and hint opening, native controls at least 44px tall, and no page or displayed-equation overflow at either width.

Final short copy polish removed implementation-history wording from the lesson and shortened a clipped select label. `node scripts/review-matrix-calculus-reading.cjs` passed at **08:14:36 UTC**, verifying that final state at both widths, all nine reference links, formula/page overflow, the reverse trace, and all ten selected-step marker positions. Evidence: `scratch/matrix-calculus-browser/reading-results.json`. Final mobile reverse-trace and reference screenshots were also opened and read. A harness-only ambiguity was corrected: Playwright's string `selectOption('1')` matched the first option's label `1` before the intended value `1`. Both scripts now use explicit `{ value }` selectors and assert the selected value; the marker check also waits for the corresponding rendered state. No learner-control defect was involved.

Actual opened screenshots include desktop/mobile local curves and reverse trace, mobile affine/bias selection, finite-difference chart and kink, matrix-square and shared-bias figures, and ordinary-reading scalar/Jacobian/matrix-gradient/practice sections. The review found and fixed a mobile pseudo-arrow overflow, three unnecessarily wide displayed equations, an ambiguous “row sum” phrase and missing selected-h/rounding explanation. The resulting formulas fit without sideways reading. Model numbers and plots come from declared equations and invented fixtures, not illustrative benchmark rankings.

An unrelated transient curriculum cycle between the parent's Decompositions/Eigenvalues prerequisites briefly prevented the whole route from mounting. Parent repaired that shared edge before the final integrated browser run; this record does not count the failed mount as a lesson pass. Parent registered this topic's individual blueprint and owns final production integration.

## Review limits and reproducibility

Formatting of owned lesson/lab/model source was verified with normalized Babel AST equality, including raw template and JSX text values (`scratch/matrix-calculus-verification/formatting-results.json`). Later changes were small, intentional copy edits verified in the supplemental reading check; no derivative model changed after its native pass.

The research ledger distinguishes substantive source text, video entry/embedded identity and unavailable video fetches. No complete video or transcript review, PyTorch execution, physical sensor validation or observed beginner study is claimed. Existing publication is not user approval. This lesson is ready for the parent's final build and user review within the agreed mathematical scope.
