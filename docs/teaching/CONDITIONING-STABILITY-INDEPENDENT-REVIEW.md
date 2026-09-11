# Conditioning, Stability & Numerical Analysis — independent review

The six production files match the author's **11 September 2026, 06:19:37.973122 UTC** freeze. The [independent packet](evidence/conditioning-stability-independent-review.json) records their exact hashes, the author's packet fingerprint and this review's final time. No production source or author evidence was edited. No unresolved material mathematical, program or visual finding was identified. Parent production integration and user acceptance remain separate.

## Substantive reading

The reviewer read the entire body, all eleven complete programs, the six production files including CSS and brief, the assessed design and author verification. The review traced each local argument rather than treating successful numerical fixtures as its proof: induced infinity norm; normal-result rounding assumptions; scalar first-order sensitivity versus finite linear bounds; nonzero reference norms and the perturbation denominator; attainment of both backward-error minima; zero-entry permissions; row scaling with rescaled uncertainty; input-path rounding factors in summation; the correction-operator recurrence; uniform fixed-time constants; the exact parasitic mode; defect versus algebraic residual; and the capstone's exact stored-data reference.

The distinction between a computed residual and an exact residual is explicit. So are the limits of a normwise perturbation model, the possibility of an exact least-squares optimum with nonzero residual, rank loss in rounded Gram formation, and the separate roles of float32 factors and float64 residuals/iterates. The ten changed practice answers were read against their questions. The final report treats an uncertainty interval as a worst-case guarantee, not an assertion about the unknown actual error.

Primary references revisited were [Higham's backward-error discussion](https://nhigham.com/2020/03/25/what-is-backward-error/), [Driscoll and Braun's zero-stability treatment](https://fncbook.com/zerostability/) and [LAPACK's componentwise-error discussion](https://www.netlib.org/lapack/lug/node79.html). The precise selected scope is recorded in the JSON; embedded source examples and videos were not claimed executed or watched.

## Complementary execution

Run `node scripts/verify-conditioning-independent.mjs`. The driver imports the actual production exports and all actual program strings, verifies the author fingerprints and passes them to [the reviewer Python checks](../../scripts/verify-conditioning-independent.py). The [actual result](../../scratch/conditioning-independent-review/results.json) passed at **06:37:21 UTC**:

- Eleven complete programs executed with their exact saved stdout.
- Thirty-two changed rectangular/dense/zero systems for each backward-error permission, comparing the actual rational diagnostic with a separately assembled constrained linear-programming minimum. The optimization represents perturbations directly; it does not use the quotient being tested as its objective or constraints.
- Forty changed native signed-disturbance cases checked using augmented matrix powers, and six changed odd-step counterexamples checked using a companion matrix power.
- Three hundred actual JavaScript disturbance states and three hundred envelope states compared with independent exact finite operator sums.
- Twenty-four exact square-root reference states verified both by the defining squared rational inequalities and independent 180-digit Decimal evaluation, including zero and the negative endpoint; seven stored binary64 fraction identities include subnormal and extreme values.
- Ninety changed summation/permutation cases check the actual left-fold bound and finite recoveries of the actual Neumaier/fsum helpers. Twenty changed refinement calls, with different correction counts, use an independent symbolic rational matrix inverse for their central reference and error.

The maximum scaled numerical discrepancy was **7.8×10⁻¹⁶**. LP feasibility/optimality is finite floating-point evidence; exact Fraction/SymPy comparisons are separately identified. These checks complement the author's larger bounded sweep. Neither set proves universal library or arbitrary-input behavior.

## Actual browser and image review

Run `node scripts/review-conditioning-independent.cjs`. The [reviewer browser result](../../scratch/conditioning-independent-review/browser-results.json) uses the registered route with actual Space Grotesk fonts at **1440, 390 and 320 pixels**. Sixteen operated states per width cover changed tie rounding, keyboard scrolling/reset, the negative domain endpoint and zero-relative-error state, tiny signed input, a changed measurement intersection, inconsistent/coincident rows, scaled backward permissions, an exact positive addition tree and zero sum, signed amplification, fixed-time refinement and the changed report answer.

All ten anchors and seventeen display equations fit; the document has no horizontal overflow, and actual page errors, React warnings/errors and failed content requests are empty. Range ArrowRight, checkbox Space, reset Enter, disclosure Enter and focused local scrolling were operated. The source fingerprints match before and after the run. The author's separate comprehensive 82-state run and final reading amendment are retained as author evidence, not relabelled reviewer actions.

Nine named final images were actually opened: reference branches, tiny cancellation, the positive addition tree at desktop and phone widths, the scaled backward witness, coincident measurements, signed propagation, ordinary residual prose/equation and the changed report solution. The exact paths and hashes are in the independent packet. Wide arithmetic trees and tables intentionally retain local keyboard-accessible scrolling and visible instructions; the ordinary prose and equations fit the page. Capture counts alone are not claimed as visual inspection.

This is a bounded independent implementation review. It does not claim a beginner user study, user approval, full-video viewing or production build/loading integration. The parent owns the remaining integration decision.
