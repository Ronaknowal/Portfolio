# Random Matrix Theory — author verification

Author scope: Mathematics34, stable ID `random-matrix-theory`. This record concerns the complete replacement of the existing lesson, preserving its useful coverage and exact original program/output. Shared registration, production integration and the independent mathematical review remain root-owned. User acceptance is separate.

## What changed and why

The original body introduced MP edges, spikes, Wigner matrices and applications, but did not show how sample covariance produces principal directions, distinguish population from empirical centering, include the complete zero atom, or provide substantial independent practice and model-specific visual investigations. The revised body teaches a connected sequence: counted data → Gram/SVD/rank → finite moments → complete MP measure → finite probability and null simulation → spike alignment → inverse gains → symmetric ensembles and exact level gaps → a defensible application workflow.

The original complete program and stdout are unchanged. Its earlier surrounding hard-edge and broad weak-spike language is replaced by scoped claims. Original source bytes, source text, code and stdout remain in [the preservation archive](evidence/random-matrix-original-content.json). There is no promise that one page exhausts RMT: local laws, edge fluctuations, free probability and the circular law are an explicitly optional research map. Useful PCA and initialization extensions are routed to their actual destination notes rather than silently treated as fully taught there.

## Visual and interaction contracts

| Representation and placement | Learning question and computation | Verification / limit |
| --- | --- | --- |
| Counted sample, Gram matrix and two eigenvector directions, section1 | How can independent population coordinates produce correlated observed counts? The eight rows produce S=[[1,.5],[.5,1]]. | Exact arithmetic, NumPy eigensolve and SVD; the displayed eigenvector lines share the coordinate origin and have a3:1 length ratio. Length represents eigenvalue, not a confidence interval. |
| Raw versus centered direction diagram, section2 | Why does subtracting a mean remove one possible row direction? | Rank bounds are maxima; captions do not claim full rank for arbitrary data. Gaussian row independence after rotation is distinguished from the general rank argument. |
| Covariance investigation, section4 | Change shape, entry law, sample centering or seed, then inspect actual sample eigenvalues and integrated reference mass. | All16 UI shape/law/centering combinations per viewport; covariance/SVD and rank checked independently. Numerical zero tolerance is visible. MP continuous bin probabilities plus the separate atom sum to one. |
| Finite-null investigation, section5 | Compare a largest eigenvalue with59 finite null maxima, including a duplicated-feature null. | Six comparisons per viewport;118 null maxima independently recomputed by NumPy. Rank validity is conditional on exchangeability of a fully specified procedure. Vertical dot stacking carries no quantitative meaning; no invented y-axis counts remain. |
| Spike investigation, section6 | Separate population variance, sample eigenvalue and squared direction overlap. | Fixed-noise strength change, both ratios, threshold/endpoint states and new sample/reset. The actual finite curve point is distinct from the qualified Gaussian asymptotic curve. Independent eigensolves and the integrated scalar Schur relation check the values. |
| Inverse-gain figure, section7 | Which perturbation grows, and what does adding αI change? | Exact gains and direct solve checks. Small response differences remain visible at sufficient precision. Conditioning does not assert improved prediction. |
| Symmetric-ensemble investigation, section8 | Mirror entries and change scale: why can eigenvalues be negative and why divide by√d? | Both ensembles and all3 sizes with normalization on/off; actual entries and spectra. Histogram area and transformed density scale correctly; GOE diagonal variance and finite second moments are explicit. |
| Avoided-crossing investigation, section8 | What does coupling do to a two-level gap? | Exact two-by-two eigenvalues, signed parameter cases, crossing/maximum/reset in browser. This is an exact toy matrix, not measured physics or a universal spacing distribution. |

Every program has its actual prediction question visibly before its code. Setup is before the first program. Eight substantial changed-input practice tasks have separate optional hints and explained solutions; seven contain independently checked calculations, while the final sensor scenario is assessed by its assumptions and decision criteria.

## Native and independent numerical evidence

Commands:

```text
scratch/lesson-tools/Scripts/python.exe -X utf8 -I scripts/prepare-random-matrix-examples.py
scratch/lesson-tools/Scripts/python.exe -X utf8 -I scripts/verify-random-matrix-native.py
node scripts/format-random-matrix.cjs
```

The native verifier invokes `scripts/verify-random-matrix-models.mjs`, then uses independent NumPy eigensolvers/SVD/direct solves, independently split SciPy integration, exact fractions and full enumeration where appropriate. It does not validate eigenvalues by calling the same browser solver twice.

Latest native results: [scratch record](../../scratch/random-matrix-verification/native-results.json), also copied into the durable author evidence at freeze. Python3.12.14, NumPy2.3.5 and SciPy1.18.1:

- 11 displayed programs actually executed and exact stdout compared; original code/output preserved.
- 112 covariance/spike configurations checked against NumPy/SVD, including centered, wide, duplicated/repeated and sign-valued cases; maximum eigenvalue discrepancy8.89×10⁻¹⁵.
- 144 independent MP interval integrals across γ=.02,.25,.8,1±10⁻¹⁰,1,1.1,4,20 and variances.01,1,9,100; maximum discrepancy3.56×10⁻¹⁵.
- 118 finite null maxima checked independently; the six rank comparisons conserve their reported exceedance counts.
- 24 symmetric ensemble matrices,48 two-level matrices,12 finite Gaussian bounds, repeated-eigenspace residual/orthogonality fixtures and16 unsupported browser API states.
- Changed learner functions for MP moments, spike limits, finite bounds and two-level eigenvalues actually run; native range guards tested. All16 matrices of the two-row/two-column sign ensemble independently enumerated to check its exact finite second moment.

The Gaussian limiting theorems are supported by the scoped primary-source read and mathematical reasoning, not claimed to be proved by these numerical tests. Reproducible Monte Carlo values are simulated outputs, not empirical benchmarks.

## Browser and ordinary reading

`node scripts/review-random-matrix.cjs` runs Edge/Playwright against the actual Vite route with public Google Fonts permitted. It checks1440,390 and320px viewports, keyboard focus and activation, actual anchor arrival, native question/code/output correspondence, initial hint disclosure, all control presets, crossing and extreme states, redraw/reset calculations, math and SVG label bounds, and normal reading captures. It records page/console errors and verifies loaded Space Grotesk, JetBrains Mono and KaTeX fonts.

The final browser result and exact source hashes are in [the durable evidence](evidence/random-matrix-author-review.json). Ordinary reading screenshots include the introduction and all10 section arrivals, all3 inline figures, each investigation both at its controls and at its plot/readout, changed states and the changed-program practice solution. Selected actual images were opened and inspected, rather than relying only on bounding boxes.

Issues discovered and resolved during authoring:

1. Adaptive numerical bin integration lost roughly5.9×10⁻⁷ mass extremely nearγ=1. The browser now uses the analytic angular antiderivative with `atan2`; an independent integral explicitly resolves the thin endpoint layer. The γ=1 singularity and zero atom remain separate.
2. The initial eigenvector lines did not share the axis origin; their final geometry is centered with equal coordinate scale and a3:1 length ratio. CSS inheritance also made two value labels black; explicit semantic colors repair their contrast.
3. Initial narrow displays overflowed some equations and y-label bounds. Equations now break at meaningful relationships; the finite singular-value bounds name L and U before squaring. Inversion is a readable sequence. Plot labels fit without changing mathematical values or shrinking the body font.
4. The finite null graphic initially implied a quantitative y-axis despite using arbitrary stacking. That axis was removed and the stacking meaning is explained. Small inverse responses use enough digits to show the actual change.
5. The MP section anchor and Queueing destination were corrected to the actual stable IDs. Verification harness assumptions about a `pre` element were corrected to the existing renderer, and keyboard modality is entered before checking `:focus-visible`.

The author evidence distinguishes these resolved findings from final results. There are no remaining known author-level content, model or rendering failures. The bounded independent review and integrated production checks are recorded separately by their owners.

### Independent-review correction: the displayed Python integral

The independent reviewer found an additional numerical defect in the actual
displayed `mp_moments` helper, beyond the earlier JavaScript repair. Its unsplit
x-domain quadrature returned continuous mass about 1.000050005 at γ = 0.9999;
at γ = 1.0001 the atom plus continuous mass was about 1.000049995. A quadrature
call that emitted no warning had still missed the narrow endpoint behavior.

The displayed program now changes variables to an angle and explicitly splits
the narrow layer near zero. It retains the exact γ = 1 limiting integrand and
keeps the zero atom separate. All eleven previously displayed outputs are
conserved. The actual corrected helper passes 55 aspect-ratio/variance cases,
including γ = 1 ± 10⁻⁴, 1 ± 10⁻⁵ and 1 ± 10⁻⁶, with no integration warnings and
maximum scaled moment/total-mass error below 5.1×10⁻¹². It is numerical
quadrature, so small rounding/integration error remains explicit.

The generator and native verifier were rerun. The targeted original-font
browser reading suite verifies the corrected code, prediction, unchanged
stdout and new explanation at 1440/390/320px. The final author evidence includes
this later result alongside the earlier full interaction run; the numerical
browser models, lesson body and lab interactions did not change in this repair.
The independent review retains the pre-fix result and its own final regression.

## Research and remaining limits

The [design](RANDOM-MATRIX-THEORY-DESIGN.md) records exact inspected portions of Paul2007, Vershynin's tutorial, Bandeira/MIT notes, Speicher's notes and recorded course, and Petrov's graduate course outline. A discovered error in an older MP note was not reproduced. The newer companion notes, creator's course page and lecture11 metadata were inspected; no full-video playback is claimed. The lesson provides these as annotated alternate routes and supplies the core mechanisms independently.

Current official NumPy `eigh`/generator and SciPy `quad` pages were read during implementation. Their online “stable” version labels can differ from the actual tested versions recorded here. Browser and Python random generators deliberately differ, so equal seed numbers do not imply matching samples.

The browser uses a bounded Jacobi eigensolver and explicitly displayed numerical-zero tolerance. It is a teaching implementation for the supported dimensions, not a general high-performance eigensolver. Simulated null scores do not validate model choice, and fixed random seeds do not establish independence of real observations. Source-level and browser checks do not amount to user acceptance or a universal generalization guarantee.
