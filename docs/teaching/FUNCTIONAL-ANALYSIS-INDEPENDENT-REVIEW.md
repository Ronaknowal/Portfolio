# Functional Analysis & RKHS — independent review

The bounded independent mathematical, source and native review covers Mathematics 39, `functional-analysis-rkhs`. Two concrete findings were reported to the author and repaired: an unqualified thresholded Nyström inverse identity and an admitted interpolation input that produced infinite coefficients/energy. The complementary current-source checks pass. The final author freeze at 22:18:16 UTC and all six source fingerprints match the independently checked version. Fourteen specific final images were actually opened and inspected. No unresolved material finding remains. [Durable independent evidence](evidence/functional-analysis-independent-review.json) preserves the initial observations, final regressions and exact identities; completion is not inferred from publication.

## What was read and assessed

Read the complete actual lesson, all eleven complete Python programs and displayed outputs, the pure JavaScript models, all seven investigations and two inline figures, the individual blueprint, detailed design and author native/browser verification scripts. The preserved original was compared directly with its durable archive and executed unchanged. The independent script executes the actual stored programs; it does not infer their behavior from JavaScript counterparts.

The source review checked the definitions and proof transitions: norm versus inner-product geometry; Cauchy completeness versus boundedness; the L² equivalence-class and unbounded-evaluation objections; closed-subspace projection and its midpoint/parallelogram existence argument; Hilbert-space Riesz including the zero functional; and the anchored absolutely continuous integral space, its L² derivative isometry and the min-kernel reproducing calculation. The original function-space coverage is developed rather than replaced by kernel regression alone.

Further checks covered kernel positivity on every finite list, repeated inputs and redundant feature quotients; completion and controlled point values; the representer projection with an existing minimizer and strict/non-strict penalty distinction; average-loss KRR existence and unique function without canceling singular K; coefficient versus function norm; an unpenalized intercept as a different objective; train-only preprocessing and separate validation/test use; and dense/feature computation costs with their assumptions.

The review also assessed measurable expectation functionals and sufficient integrability, distribution separation versus kernel validity/universality, independent-pair MMD conventions, Gaussian characteristic scope, deterministic kernel-quadrature errors and attainability, reference-measure-dependent Mercer modes, compact diagonal inverse instability, KRR/GP mean normalization and Brownian path membership. Finite tests below complement these proof reads; none is presented as a numerical proof of an infinite-dimensional theorem.

The stated compact-domain/full-support Mercer assumptions and norm coordinates were cross-checked against [Kanagawa et al., section 4.1, theorems 4.1–4.2](https://arxiv.org/pdf/1807.02582). The representer penalty scope was checked against [Schölkopf, Herbrich and Smola](https://alex.smola.org/papers/2001/SchHerSmo01.pdf). These were selected source checks on 10 September 2026 UTC, not a claim to have reviewed every page or watched the linked MIT video. The author's broader resource-inspection ledger remains separately attributed.

## Findings and disposition

1. **Exact versus thresholded inverse.** Section 7 originally retained eigenvalues above a numerical cutoff and then used the exact pseudoinverse identity without qualification. For `W=C=diag(1,1e-14)`, retaining only the first direction produces a bottom-right reconstructed entry 0, whereas the exact inverse reconstruction retains 1e-14. The author now defines the inverse on the retained eigenspace and states when it equals the mathematical pseudoinverse. The formula and implemented approximation now have the same contract.
2. **Native interpolation arithmetic range.** Calling the actual displayed `anchored_interpolate([1e-310], [1.0])` originally returned infinite coefficients and energies after passing its finite/increasing-domain checks. The paired `1e-308` case returned finite values near 1e308. The author added explicit coefficient/energy range handling, including underflow/positivity checks. The independent current-source run confirms the overflowing case raises `ValueError` while the paired finite case still works. Eleven displayed outputs and the original bytes are conserved.

3. **Checkable changed-program practice.** Practice H asks for a particular noise change but initially supplied reporting criteria only. The reviewer executed that full changed protocol and an explicitly chosen mostly linear mechanism. The author agreed to add the actual settings and validation/test errors, preserving the point that model selection and a finite result do not establish a universal ranking. The author added these checkable values and verified their final rendering at 1440, 390 and 320 pixels. Both narrow practice screenshots were opened independently and agree with the executed values.

A few missing prose spaces were also reported as reading polish and repaired by the author. No production file was edited by this reviewer. Initial observed results and their exact scope are preserved in the durable evidence rather than overwritten with a passing status; the obsolete whole-source hash was not captured during the initial exploratory call and is not invented.

## Complementary calculations actually run

Command from the repository root:

```text
scratch/lesson-tools/Scripts/python.exe -X utf8 -I scripts/verify-functional-analysis-independent.py
```

The final independent run at 22:18:52 UTC reads current source and records its six fingerprints. The script was conventionally formatted with normalized Python AST conservation. It checks:

- All eleven actual programs/stdout and original-program/output conservation.
- Twelve 70-digit kernel solves and functional-objective remainder identities, including duplicate locations, small regularization and varied bandwidth. The largest RKHS norm difference from the bounded JavaScript model was approximately 2.64e-10.
- Six Gaussian MMD calculations through an independent Fourier-domain integral, instead of duplicating a Gram/pair-sum oracle. Maximum discrepancy difference was approximately 4.44e-16.
- Seven exact Fraction multiple-node projection/attainment calculations, including unsorted nodes, duplicates and anchors; 24 independent exact single-node residual-energy polynomials.
- Twenty-seven changed interval/asymmetric-wiggle calculations proving the cross term zero and energy decomposition exactly, without copying the lesson fixture's constants.
- Eight redundant-feature quotient checks, verifying minimum-norm parameters and agreement with kernel predictions.
- Two actual changed capstone programs: increased noise and a mostly linear generating mechanism. Each retains training-only scaling, independent groups and validation-based selection before test evaluation. Their real stdout is recorded; a prescribed winner is not imposed.
- The concrete discarded-positive-eigenvalue counterexample and the paired native interpolation arithmetic regression.

The script and raw results are retained with hashes. The author has additional broad model, guard and browser suites; those are not attributed to this independent reviewer.

## Visual and final identity closure

All six current production hashes match the final author record and the final independent execution. The body is `3765cacb0054566b3a929d240b186a60a5e42e8185272054336a2ce862d48e01`; the other five exact fingerprints are in the evidence JSON. The initial frozen body and subsequent prose-only practice amendment are distinguished in the author record.

The independent reviewer opened fourteen author-generated images: ordinary measurement reading; completion, spike, evaluation and changed-wiggle mechanisms; the desktop invalid Gram matrix; duplicate kernel prediction and contributions; the narrow quadrature equation and optimum; changed quadrature practice; both numerical-repair contexts; and the final changed-capstone answers at 390 and 320 pixels. Labels, mathematical quantities, declared line/area meanings and the concrete practice values were inspected. The twelve earlier image hashes stayed unchanged after the prose amendment. Wide sequence coordinates, precise tables and source code retain intentional local scrolling. No additional visual blocker was found.

The author’s full actual-font 1440/390/320 interaction run at 22:09:47 and focused final reading at 22:16:30 are separately attributed; this reviewer did not claim a second exhaustive browser run. The final prose addition did not alter models, programs or controls. Parent production integration and user acceptance remain separate.
