# K-Means working revision: independent models and native-data review

Reviewed 21 September 2026 by `kmeans_models_review`, independently of the original model/native author. Scope is the pending 12 September revision of `k-means-hierarchical-clustering`, authorized for review by the user. **Pure models, the six displayed Python programs, and the Old Faithful figure data are closed after the focused repairs below.** The root owns actual browser behavior, visual layout, complete lesson integration and the ledger; this report does not substitute for those checks.

The [design and proposed revision](K-MEANS-HIERARCHICAL-LESSON-DESIGN.md), [previous complete review](K-MEANS-HIERARCHICAL-INDEPENDENT-REVIEW.md), [previous model/visual review](K-MEANS-HIERARCHICAL-VISUAL-MODEL-REVIEW.md), actual three data modules and complete native generator were read. The starting identity is preserved in [the root baseline](evidence/k-means-revision-review-baseline.json). No historical receipt, other lesson, publication mapping or delivery checkpoint was overwritten by this sub-review.

## Findings and repairs

1. **A finite sample is not its exact probability.** The original seeding question asked how often the farthest row would appear in the displayed 200 draws but graded against its exact conditional probability. Across all 62 nonempty, nonfull subsets of the six rows, 17 cross the question's half-probability grading boundary. For example, with rows P0 and P2 selected, the exact probability of the farthest row is 0.5579399142 while the seeded sample selects it 99 times: less than half of 200. The root has the complete counterexample list and owns the question/feedback repair. Exact probabilities and the observed frequencies are both numerically correct; it was their instructional comparison that was wrong.

2. **Exhausted seeding returned an infinite uniform comparator.** After all six rows are selected, `seedingFrequencies` calculated `1 / (6 - 6)`. It now returns zero when there is no unselected row. All D² counts remain zero and the farthest row remains `null`; the UI must explain that no draw exists rather than call this a normalized D² distribution. All-identical data can also exhaust D² mass before every row index is chosen. Both states were explicitly checked.

3. **The Lloyd program had an undefined return at a zero iteration budget.** `max_iter=0` skipped the loop and attempted to return unset variables. A compact positive-integer check now rejects zero, negative, fractional and boolean budgets, while a one-update budget retains coherent nearest-center labels and explicitly reports nonconvergence when appropriate.

4. **Zero-column and zero-row native inputs could silently express an empty geometry.** The Lloyd and Ward input checks now require a nonempty finite row-by-feature matrix. They reject both empty shapes with a clear `ValueError`. These are small input-contract repairs, not an attempt to turn the teaching functions into arbitrary-range numerical libraries. The implementation still uses ordinary NumPy squared-distance arithmetic, with the rescaling limitation explained in the lesson. The generator was synchronized with the displayed code.

5. **The maximum palette control is not always lossless.** The model correctly limits the mosaic to six entries and the sky/gradient to eight. Their actual unique-color counts are 6, 37 and 160, and their displayed errors at the maximum controls are respectively 0, 35,945 and 165,886. The root owns the misleading maximum-means-lossless feedback repair. The image calculations themselves needed no change.

6. **The changed-unit exercise and forecast interpretation needed actual-data checks.** On this 272-row fixture, converting only waiting time from minutes to seconds leaves the raw-unit two-center assignments unchanged (ARI 1) and produces the same centers when mapped back to minutes: `[2.09433, 54.75]` and `[4.2979302326, 80.2848837209]`. It does not move the duration centers toward each other as the original exercise asserted. Raw and standardized assignments have ARI 0.9415367538. Also, the standardized two-group fit reduces descriptive in-sample waiting RMSE from 13.5699600176 to 5.8656988428, an 81.31546% squared-error reduction, but the waiting outcome itself helped choose those labels. Those numbers do not establish duration-only forecasting performance. The content reviewer received these verified values and owns both explanations.

## Complementary verification actually executed

The durable verifier is [verify-k-means-revision-models.mjs](../../scripts/verify-k-means-revision-models.mjs), with its focused [native/data oracle](../../scripts/verify-k-means-revision-native.py). It saves a provisional incomplete receipt before work and writes a passing receipt only after all assertions succeed. It never imports the old generator, whose top-level code runs a broad campaign and rewrites prior evidence. Instead it parses the generator to verify all six complete displayed program strings.

[The model receipt](evidence/k-means-revision-model-review.json) records **159 grouped cases**, with actual supported inputs and assertions:

| Scope | Executed evidence |
| --- | --- |
| Every ordered starting-center pair | All 36 pairs on the six-point fixture and all 16 on the rectangle, including coincident centers. Every phase's residual error, monotonicity, nearest-center assignment and mean-optimal SSE checked; all runs reach their declared fixed point. |
| Exhaustive geometry | Every one of seven partitions under all eight unit/weight choices, pairwise-variance objective identity, compensated-unit equality, duplicate rows and the eight-point enumeration boundary. |
| Sequential seeding support | All 63 nonempty selected subsets, positive-mass interval selections, zero-mass cases and exact probability totals. Each 200-draw histogram is independently reconstructed with BigInt 32-bit PRNG arithmetic and integer-mass cross-products rather than the implementation's floating CDF. |
| Hierarchy | Both six-row and chain fixtures, all four linkages, every candidate merge's independently calculated cost, every count cut and every distinct height cut. The chain's five exact unit-height merges enter a height cut together. |
| Palettes | All 22 available image/size settings. Every color count, fixed point, expanded-pixel versus weighted objective, rounded reconstructed pixel and displayed error checked. |

[The native/data receipt](evidence/k-means-revision-native-review.json) records **27 grouped checks**, with NumPy 2.3.5, SciPy 1.18.1 and scikit-learn 1.9.1:

- Executed the changed Lloyd and Ward programs in full; their stdout is byte-for-byte unchanged. Checked the eight invalid-input cases and the coherent one-update return.
- Confirmed all six displayed programs match their generation source. The other four programs' code strings **and** stdout strings match their historical native execution digests, so their complete executions and earlier changed native oracles are reused rather than relabeled as fresh runs. The whole module's historical digest does not match the starting module, so reuse is deliberately at these stronger per-program identities rather than an unsupported whole-file claim.
- Compared all 272 JavaScript data rows with an independently downloaded [Rdatasets CSV](evidence/k-means-revision-faithful-source.csv), and compared every inline Python data row with those same observations.
- Independently fitted population standardization, all 272 two-center labels, both mapped-back centers and group sizes. Recomputed every one of the 40 retained inertia values and all seven defined median silhouettes, with the six-decimal storage tolerance stated in the verifier.
- Reproduced all 40 sampled indices and all 39 SciPy Ward linkage rows, including child IDs/counts and heights. Separately re-derived every height by direct union SSE minus the two constituent SSEs; confirmed the top two tree groups agree with the full-data two-center colors on those rows.
- Computed the changed-unit and descriptive-error findings above.

The current **faithful data module is unchanged**: all data, diagnostics and tree values were correct. No new dataset or synthetic replacement was introduced.

## Provenance and retained evidence

The [R datasets documentation](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/faithful.html) confirms 272 rows, eruption durations and waits to the **next** eruption in minutes, Härdle's source, and heavily rounded originally-second-based durations. The independent CSV was fetched from [the lesson's named Rdatasets mirror](https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/faithful.csv) on 21 September 2026; the retained bytes have SHA-256 `5043db1e2c51c8e8fd67e0868c768ae589770cc76ad0ac0c5b7afd1fca31fc57`. Later verifier runs use this retained snapshot instead of depending on a live network response.

The historical [native receipt](evidence/k-means-hierarchical-native.json) and [211-group model receipt](evidence/k-means-hierarchical-models.json) are preserved, with their digests recorded in the new evidence. The model receipt's starting model/data identity matches the root baseline. This review complements its ordinary-range/rational/library checks; it does not claim to freshly rerun that entire campaign.

Initial verifier attempts failed before a passing receipt: the Windows child Python decoded Unicode stdin with the default code page, then the sandbox blocked the attempted public CSV fetch. The verifier now runs Python explicitly in UTF-8 and uses the independently fetched retained CSV. These were verifier/environment failures, not passing content checks or runtime lesson failures. The final full focused run passed.

## Frozen source identity and limits

| Reviewed runtime source | SHA-256 |
| --- | --- |
| `src/learn/data/k-means-hierarchical-models.js` | `7cc680a3d45229732e493abd344bee4e98dfb9aefa145dbc89a5731e53f8fb99` |
| `src/learn/data/k-means-hierarchical-faithful.js` | `64358723bb94ea232a9b4e5da3f61fedc6464711234690e47a390914aaa60e50` |
| `src/learn/data/k-means-hierarchical-examples.js` | `36cb906138b9822b8a3d0c5a851f358d827d42edf8152cdba800d817ebd2d747` |

These sources are frozen for root integration. The generator, new verifiers, downloaded CSV and native receipt have separate exact hashes in the machine evidence. No browser screenshot, measured performance result, publication/sequence check, formal floating-point proof, universal clustering optimum or user acceptance is claimed here. The root's final revision review closes those applicable integration boundaries.
