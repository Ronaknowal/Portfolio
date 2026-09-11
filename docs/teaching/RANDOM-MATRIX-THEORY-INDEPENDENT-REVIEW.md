# Random Matrix Theory — bounded independent review

Independent reviewer: `testing_documentation_completion`. Scope: mathematics34, `random-matrix-theory`. The complete current body, pure models, all eleven displayed programs, all lab rendering logic, design, author verification and original preservation archive were read. The review is complementary to the author's numerical/browser suite, not a relabeling or repetition of it. No production file was edited by this reviewer.

**Disposition: closed, with no unresolved material finding in the bounded scope.** The identified native numerical issue is repaired and the complementary execution passes. All six production hashes match the author freeze at **2026-09-10T20:36:52.161044+00:00**. Exact hashes, the preserved initial failure, final numerical results and inspected screenshot identities are in [the independent evidence](evidence/random-matrix-independent-review.json).

## Resolved finding: the displayed integration helper lost the MP endpoint layer

The initially reviewed `mass` program in `src/learn/data/random-matrix-examples.js` called unsplit `scipy.integrate.quad` on the eigenvalue interval. Near aspect ratio 1, the lower endpoint is tiny and a narrow contribution was missed without a warning. The browser's angular antiderivative had already been repaired, but that repair was not reflected in the displayed native helper.

Direct execution with variance 1 gives:

| Accepted aspect ratio | Returned continuous mass | Required continuous mass | Error in total mass |
| --- | ---: | ---: | ---: |
| 0.9999 | 1.000050005000339 | 1 | +0.000050005000339 |
| 1.0001 | 0.9999500049949213 | 0.9999000099990001 | +0.0000499949959213 |

The same problem appears at 1±10⁻⁵ and 1±10⁻⁶. The review script also reproduces it at variance 4. A probability mass greater than one contradicts the central teaching contract. This is a changed-input failure in the complete published program, not a criticism of the correctly scoped MP theorem or the repaired browser integrator.

The author and root received exact inputs and the request to use an angular transformation with an explicitly resolved boundary layer (or a comparably justified stable computation), retain the atom separately, and add those inputs to the actual native-program tests. `scratch/random-matrix-independent-review/pre-fix-results.json` preserves the first run. No production patch was applied by the reviewer.

The author replaced the displayed helper with the angular substitution and explicit integration breaks at multiples of the endpoint-layer width. This reviewer read the repaired actual code and reran the complementary script at **2026-09-10T20:33:25.509224+00:00**. All near-one cases now pass, without quadrature warnings, and all eleven previous program outputs remain conserved. The independent interval integrator uses separately chosen integration breaks and checks the browser's analytic antiderivative; the displayed helper is also checked against the exact continuous mass and first two moments. This resolution preserves the failure record instead of rewriting the initial run as a pass.

## Mathematics and teaching contracts reviewed

- The row/feature orientation, known population mean versus estimated sample centering, denominator change and exact nullspace/rank argument are coherent. The exact n−1 independent-row representation is restricted to Gaussian observations; centered signs are identified as a limiting comparison.
- The MP law keeps the continuous part, zero atom, scaling and conditional-positive mean distinct. Bulk convergence is not promoted into a finite test or an extreme-eigenvalue theorem under only finite variance.
- The real Gaussian one-spike model uses population eigenvalue ℓ, the correct threshold and the correct branch. Alignment is squared and sign-invariant; a leading-eigenvector transition is not called a universal impossibility result. The local Schur elimination is correctly separated from the imported asymptotic theorem.
- Gaussian finite bounds use the simultaneous failure probability and clamp the lower singular bound before squaring. Matched-null rank scores are qualified by exchangeability; data-fitted nulls and selection require separate calibration.
- GOE diagonal/off-diagonal scaling and finite second moments differ correctly from the zero-diagonal sign ensemble. The radial gap law is exact only for the stated unscaled two-by-two ensemble. The lesson does not label the two-by-two surmise an exact all-size spacing law.
- Inverse gains, regularization, fixed-input norm preservation versus an adaptively selected worst direction, and the eight changed practice solutions were read and checked. The counted-row, rank and spectrum representations match the equations and declared units.
- The full original seven-section source was read; its core shape/noise/spike/semicircle/conditioning/model-selection and rank-practice ideas are retained or developed. The actual original program and output are exactly conserved, independently of the author's claim.

The reviewer directly read [Paul's 2007 paper](https://www3.stat.sinica.edu.tw/statistica/oldpdf/A17n418.pdf), its Gaussian setup and Theorems1,2,4, checking the ratio, simple-spike, outlier and overlap conditions. The scalar MP-integral identity also provides an independent route to checking the above-threshold formulas. [Vershynin's tutorial](https://arxiv.org/pdf/1011.3027), Theorem5.32 and Corollary5.35 with the intervening concentration argument, supports the finite expectation/tail distinction and union-bound probability used by the lesson. Neither source's complete asymptotic proof was audited. No independent full-video playback is claimed; the author's limited metadata/companion-note description remains explicit.

## Complementary numerical evidence

Command: `scratch/lesson-tools/Scripts/python.exe -X utf8 scripts/verify-random-matrix-independent.py`.

The final complementary execution at 2026-09-10T20:33:25.509224+00:00 passed all checks below, including the repaired displayed mass helper. The full result is `scratch/random-matrix-independent-review/results.json`; the pre-repair run remains separately preserved.

| Independent formulation | Cases |
| --- | ---: |
| Actual complete programs executed and stdout matched | 11 |
| Changed raw/centered Gaussian/sign covariances against SVD and entry sums | 16 |
| Independently split angular quadrature versus browser MP interval masses | 99 |
| Exact Gauss–Hermite integration of raw and centered Gaussian degree-four moments | 810 entry states |
| Finite Schur-complement scalar roots and eigenvector-normalization overlaps | 8 |
| Actual native and JS spike branches | 12 |
| Independent MP-integral outlier and overlap identities above threshold | 6 |
| Changed finite bounds, including vacuous lower and tiny failure probabilities | 18 |
| Changed signed/shifted two-level matrices against symmetric eigensolves | 27 |
| Independent radial integrations of the GOE gap CDF | 5 |

The maximum MP interval discrepancy was 7.22×10⁻¹⁶. Gaussian quadrature gives raw/centered second moments 2.5/4 for n=p=2 and 2/2.5 for n=3,p=2, independently verifying the finite correction and Gaussian centering claim.

An additional exact enumeration of 351 exchangeable tuples with tied three-valued statistics checks conservativeness at twelve significance levels for B=2,3,4 null replicates. It is saved in `scratch/random-matrix-independent-review/tied-rank-enumeration.json`. This checks the discrete rank argument, not whether a scientific data-generating model is correct.

## Actual visual inspection and bounds of this review

The reviewer opened five existing author captures: `inline-0-390.png`, `covariance-wide-centered-390.png`, `random-spike-initial-1440.png`, `final-finite-bounds-320.png`, and `finite-matched-null-390.png`, all under `scratch/random-matrix-browser/`. They show meaningful counted products/principal directions, matched-bin comparisons, finite versus limiting spike values, readable finite-bound conversion and the finite-null score. These were author-generated captures, independently opened here; this review does not claim to have rerun the author's complete browser suite or opened every author screenshot.

No additional material mathematical or visual defect was found in this finite scope. This is not an exhaustive IEEE-range guarantee, a complete proof audit of random matrix theory, a production benchmark, integrated publication approval or user acceptance. The repaired source is frozen and all six final production identities are independently matched.

## Final repair display and identity closure

The reviewer additionally opened the author's final `final-mass-program-390.png` and `final-mass-output-320.png`. The actual program shows the angular substitution and endpoint-layer breaks; the following readable explanation states why unsplit quadrature can silently miss this layer. Long code/output lines remain locally scrollable rather than forcing page overflow. These two captures supplement the five previously inspected author images. The author's focused actual-font display check at 2026-09-10T20:35:44.178Z covers the question, code, output and explanation at 1440/390/320; that execution is attributed to the author.

The independent repaired-helper run had a maximum absolute error of 5.05×10⁻¹² among the exact mass/first-two-moment comparisons and no quadrature warnings. It is separate from the author's broader 55-helper-case regression. No production source was edited by this reviewer.

| Frozen production file | SHA-256 |
| --- | --- |
| `src/learn/data/topics/random-matrix-theory.jsx` | `226725367e3a93a218a8e67b87349093333a7d7c7c1bca1d53bc5d05b1e96ad2` |
| `src/learn/data/random-matrix-models.js` | `ba75ce3a68e187bfc0f55b861fb9209633ecaa7a2952601754edab742069776f` |
| `src/learn/data/random-matrix-examples.js` | `4386d8e5b5b21886156b81c87fe2f97a844700b3b541a952bbb6ce6f63e59780` |
| `src/learn/components/lesson-labs/RandomMatrixLabs.jsx` | `fc418ffaecfb5d6281b4cd68a17e357b146c4cd4feda3a187d50c13cc5c7d493` |
| `src/learn/components/lesson-labs/random-matrix-labs.css` | `53871c332a5c3e4994238b2ff5d4b57d6e98ee41d9e3f807d1ad15733b989489` |
| `src/learn/data/curriculum/blueprints/random-matrix-theory.js` | `517b37fe8e2666f224738aa5a045d5505f7dd1a0174379744c41ea6af46a0394` |
