# Sequential review — Feature Selection & Importance (23)

Reviewed 16 September 2026, limited to `feature-selection-importance-shap-permutation-mutual-info`. All seven finding groups below are closed on the identified source. The reviewer changed only the two verification scripts and this record; the integration owner repaired production code and operated the final browser.

## Coverage reconciliation

The prepared lesson, visual specifications and design reconcile with the current reader: nine sections, five figures, five investigations, four complete programs, ten closed-solution practices and readiness/reference material. No substantive scope omission found. The implementation preserves the distinction among information, fixed-model reliance, removal and refitting, local allocation and causal claims; count-derived MI and XOR/duplicates; filter/wrapper/embedded selection and fit costs; donor-level permutation and grouping; coalition/background/conditional-game distinctions and class/output units. The Wine workflow retains the 178-row source, 138/40 development/reserve boundary, 100/38 fit/inspection boundary, nine CV fits plus two refits, selected six-feature versus separately declared four-field models, actual tree paths, all sixteen coalitions and twelve explained cases. The deeper branches retain conditional MI, Markov blankets, measurement costs, stability/significance, grouped players, computational costs and the limits of cumulative attribution fractions.

The optional SHAP program's execution, deferred in the packet, is correctly recorded as completed in the runtime with SHAP 0.52.0. Existing independent Wine refits, exact coalition proofs, source-tree comparisons and native programs are reused where their source is unchanged. This pass concentrated on legal near-null inputs, explanatory causes, grading boundaries and actual visual geometry rather than repeating that full campaign.

## Concrete findings and repairs

| ID | Reproduction and consequence | Repair and bounded verification |
| --- | --- | --- |
| R23-1 | User-reported gold bands came from scroll-container gradients, which could look like diagram data and appeared without actual overflow. | The integration owner replaced them with a plain hint shown only for actual overflow, a labeled keyboard-focusable inner scroll region, and ResizeObserver cleanup. Negative-value hatching remains. Browser checks computed backgrounds and pseudo-elements at 1366/390, hint equivalence to actual overflow, and ArrowRight scrolling of an overflowing mobile SVG. Owner inspected desktop/mobile F1 and table captures. |
| R23-2 | I1 counts `[[9999,9998],[10000,9999]]` have determinant 1 and are dependent, but entropy subtraction yielded zero and direct summation a negative value; the grader called them independent. Equal-entropy copy also claimed opposite preferred labels although both rows prefer label 0. | Stable nonnegative KL remainder with count-product independence, retaining direct and entropy-subtraction diagnostics separately with a cancellation caveat. Independent 70-digit evaluation gives MI `4.510225845067146494022681575455e-18` bits; a relative-error regression distinguishes it from zero. Browser commits “some information,” checks the scientific result and excludes the false independence/opposite-label claims. Opposite-label copy now requires opposite sides of one half. |
| R23-3 | I3 rows `(-.0001,-.0001)` twice and `(.0001,.0001)` twice, targets zero, weights `(.0001,-.0001)`, first-column donors `(2,3,0,1)`: the real MSE increase is `4e-16`, but an absolute `1e-12` floor called it exactly unchanged. | Paired squared-residual differences avoid subtracting aggregate losses. Direction uses an eight-ULP scale from the actual losses without a fixed floor; the null option states floating-point precision. Independent model and committed browser regressions retain the positive effect and existing true-null cases. |
| R23-4 | I4 instance `(2,2)`, zero reference, gamma −1 produces zero allocations by cancellation, not by matching the reference. Instance `(0,3)`, zero reference, gamma 1 has equal order increments despite a nonzero interaction term. A single background row also makes mean output equal output at the mean even for a nonlinear model. | Feedback names the actual cause, treats nonlinear mean/output differences as conditional, and explicitly states the `1e-12` tie convention before grading. Browser covers cancellation, the nonzero-gamma order null, and a difference about `−1e-12` whose scientific readout explains the accepted tie. Reset restores the inspected coalition. |
| R23-5 | Shared numeric grading compared directly with its tolerance, and only disclosed the tolerance after commitment, making decimal boundary answers vulnerable to binary rounding. | Tolerance is stated before input and inclusive comparison adds four ULPs. I1 perfect-copy MI 1 accepts `1±.0001` and rejects `1±.000101`; all labs use the repaired shared component. |
| R23-6 | Waterfall rectangles used a minimum width of 1.5 even for exactly zero baseline/reconstruction; tiny ranges also inherited a fixed scale floor. Painted distance therefore contradicted the numeric zero. | Actual widths and actual nonzero spans, with a fallback only for an exactly zero span. Browser checks both zero baseline/reconstruction rectangles have width zero in the cancellation fixture. |
| R23-7 | F4 scaled the repeat-dot plot by mean decreases alone (maximum about .3592). A real flavanoids repeat at .5 fell at x≈365 in a 300-wide SVG and was clipped despite the claim that all twenty repeats were shown. | Scale includes means and all repeat decreases. Browser requires all eighty dots within the plotted x bounds; owner inspected the repaired capture and confirmed all points visible. |

## Evidence and limits

- Reviewer ran the augmented model verifier: **PASS, 2,570 assertions in 209 groups**. The new near-independent MI oracle uses high precision and a relative check; the tiny permutation and cancellation cases have separate analytic expectations. Browser script syntax passed.
- Final production browser run by the integration owner: **PASS, 17 cases and 41 captures**, following a passing build. Two new groups add five captures and preserve the old cases. Coverage includes numeric endpoints, near-null grading, explanatory copy, zero-width bars, all repeat positions, painted scroll backgrounds, overflow hints and mobile keyboard scrolling. Final `selection-browser.json` source hashes were independently compared with the filesystem and all match. The owner visually inspected the repaired gradient/overflow and F4 captures; this reviewer did not independently operate or view the final browser session.
- Native evidence is reused: four programs and **51 oracles**; examples and native-verifier digests match `selection-native.json`. Data module and generator digests match `selection-data.json`. The unchanged 10,782-byte Wine member has SHA256 `6be6b1203f3d51df0b553a70e57b8a723cd405683958204f96d23d7cd6aea659`. The eleven-fit recorded study still leaves the forty reserved rows unpredicted and unscored.
- No external references were re-researched; no full native/data campaign was rerun; no later topic was reviewed. Previous broad independent evidence is reused only for the unchanged source and claims it covers.

## Reviewed source identity (SHA256)

| Source | SHA256 |
| --- | --- |
| packet `lesson.md` | `2e4125594caf47ff5f108fa69beff804b67f55195503958d4ba0041fb017b035` |
| packet `visual-specifications.md` | `499da63eaf6c2b1d9e3f3ef668a3bd679938caf19528ae1fb0a71d8c49bdf19e` |
| packet `design.md` | `50b05ad83c5078cd617934dc3d5b53c6887995379840509070757c16d6f6a95e` |
| packet `calculated-inputs.json` | `f13f6d7a7897872e371f417154465e053a49e903047d161abf68ff205fe39109` |
| `src/learn/data/topics/feature-selection-importance-shap-permutation-mutual-info.jsx` | `5e3fd265e17905591a42781745c1ee148bca68dd29baaea3527b9050066e9823` |
| `src/learn/data/selection-models.js` | `bdbf0c7972d337382a4d85a6ee9cf205822a99855c89ce3c5b7c188cf058d0dc` |
| `src/learn/data/selection-data.js` | `6bae5793a66a6531ba5470607b1bf554837430ddae86dcce6b2cb4dd68e45937` |
| `src/learn/data/selection-examples.js` | `9dd716db1d8972318d41dd1f1acfe6e501f9765dd62b63d31d884f076612a76e` |
| `src/learn/components/lesson-labs/SelectionLabs.jsx` | `25d75fca677f495e5d80393f040f33ebbe1e6433a45cd18aa7cbc3e441bcb163` |
| `src/learn/components/lesson-labs/SelectionShared.jsx` | `297e99b8bb23534514c6a86006e9802e84cd7ba7b69780993039c17a331cf3cd` |
| `src/learn/components/lesson-labs/SelectionFigures.jsx` | `f4ebfa82a60b564729eab5559b9013e4f7d04759afb5ba0dca98864e494d7ed4` |
| `src/learn/components/lesson-labs/selection-labs.css` | `3f30b036ed45643d50c2b34df9dc51ceb633d9eea87069756f5c20ee8357756f` |
| `scripts/verify-selection-models.mjs` | `4381f7a8e35417253fa38d49400b31ba696705eaefb046e0f5e14cf075aa5af6` |
| `scripts/verify-selection-browser.cjs` | `08177281da4a625d41624f2abef47c30f181ebd6f76bf6ad2ce96f98e91dbc01` |
| `scripts/verify-selection-examples.py` | `9d26819e938f4f640176fd5acc65534e76d2e86dee83342b0342262714b2c592` |
| `scripts/verify-selection-data.py` | `dd3e6bbdb92e9612935beb1d62a7a6460d00860d01a55c480e523b34c3fa7668` |
