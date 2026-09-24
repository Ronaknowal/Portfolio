# f-Divergences & Integral Probability Metrics — author verification

10 September 2026 UTC (11 September locally). Mathematics position 29, stable ID f-divergences-integral-probability-metrics. Complete scoped reimplementation, native/model and actual-browser author checks passed. Parent owns the separate independent review, production integration and goal ledger. Publication and author verification are not user acceptance.

## Scope and teaching decisions

The full original body was inspected and archived at scratch/divergence-ipm-native-verification/original-lesson.jsx. Its useful mass-ratio versus observable framing and original complete three-category calculation were preserved; original output remains 0.208 0.054 0.3. The old body named MMD and applications without enough local construction, computation, practice or visual investigation to teach them independently.

The revised lesson builds probability ratios and support before processing, events and classification. It then constructs a finite Lipschitz observer, separates geometry from exact support, builds feature means and kernel witnesses, and distinguishes empirical norms, unbiased estimates, permutation evidence and variational bounds. Applications relate chi-square to importance weights and stress scores against omitted rare events and task loss. Ten changed-case exercises have hints and explained solutions; the last contains a complete small permutation-test project.

Title and module position are retained. The closing route is Graph Fundamentals. Links to Entropy, Hypothesis Testing, Optimal Transport, RKHS or GANs are review/deeper links, not publication-based navigation shortcuts. No other lesson, registry or shared syllabus source was edited.

Read the [topic design](F-DIVERGENCES-IPMS-LESSON-DESIGN.md). No incoming destination note existed at design time. Two researched discoveries were routed to [Functional Analysis & RKHS](topic-notes/functional-analysis-rkhs.md) and [GAN Fundamentals](topic-notes/generative-adversarial-networks-gan-fundamentals.md); they remain open for their destination authors. Abstract embedding and practical network-training work is not claimed completed here.

## Production ownership and representations

| Source | Role |
| --- | --- |
| src/learn/data/topics/f-divergences-integral-probability-metrics.jsx | Complete narrative, local derivations, visible example questions, practice and annotated references |
| src/learn/data/divergence-ipm-models.js | Pure bounded calculations; copied and frozen snapshots |
| src/learn/data/divergence-ipm-examples.js | Ten executed complete Python programs and exact stdout |
| src/learn/components/lesson-labs/DivergenceIpmLabs.jsx | Six investigations and three inline figures |
| src/learn/components/lesson-labs/divergence-ipm-labs.css | Topic-owned responsive diagrams, ledgers, controls and named scroll regions |
| src/learn/data/curriculum/blueprints/f-divergences-integral-probability-metrics.js | Individual stable-ID plan and review contract |

| Visual | Mechanism and meaningful action | Accuracy contract |
| --- | --- | --- |
| Probability ratio | Aligned masses → ratio → weighted penalty; change generator, apply drafts, swap or remove support | Six formulas share normalized laws. Centered KL rows have cancelling linear corrections. Infinity is symbolic, not a finite bar. |
| Coarsening flow | Opposite discrepancies cancel when A/B are observed jointly | Actual shared deterministic channel; connectors mean grouping, not mass width. |
| Metric triangle | JS direct versus two-leg cost, then its square root | Half-JS convention; layout is not physical geometry or proof of the general root-metric theorem. |
| Observer class | Zero linear gap but positive event/Lipschitz witnesses; change pair/class | Exact event maximum and ordered W1 critic; scores link to expectation contributions. |
| Moving atoms | Shrinking physical displacement with still-disjoint exact support | Analytic values with fixed positive Gaussian bandwidth; stems identify equal-mass atoms, not density heights. |
| Feature means | Map x to (x,x²); raw means agree but vector means differ | Explicit finite features and actual coordinates, not a universal separation claim. |
| Kernel witness | Change samples/kernel/bandwidth; connect witness to signed Gram blocks | Repeats remain samples and unequal sizes work. Empirical squared norm, signed unbiased estimate and unnormalized witness gap are distinguished. |
| Permutation rank | Inspect all 70 allocations and the exact statistic histogram | Observed rank stays fixed while inspecting allocations. Tail uses exact values; bins only summarize. Exchangeability and selection rules are explicit. |
| Variational ledger | Restore/flatten/shift a critic; compare reward and penalty | Exact fixed-law population calculation, correct KL conjugate and offset gap; no fitted-sample population-bound promise. |

Draft edits do not silently replace applied states; invalid inputs preserve the last calculation. Native controls support keyboard operation/reset. Dense Gram tables have named focusable horizontal scroll regions; small inline figures fit naturally on mobile.

## Mathematical and native checks

Run node scripts/verify-divergence-ipm-models.mjs; it invokes scripts/verify-divergence-ipm-native.py with Python 3.12.14. Final results are in [durable evidence](evidence/f-divergences-ipms-verification.json); raw fixtures, programs and results are under scratch/divergence-ipm-native-verification/.

| Independent check | Actual scope |
| --- | --- |
| Executed learner programs | Ten standalone standard-library Python files; exact stdout and empty stderr. Original calculation retained. |
| Known-law comparisons | 441 law pairs; direct SciPy relative entropy/JS, square-root norm, Pearson formula, exhaustive event maximum and Pinsker |
| Near equality | Five 90-digit Decimal KL cases; centered series agrees at sampled scales |
| Shared processing | 89 channels including five safe-range/identity/constant cases; all six divergences checked against independent pushforwards and DPI |
| Observable geometry | 252 states; 168 independent SciPy transport/critic linear programs |
| Kernel quantities | 122 changed states, unequal sizes, explicit Gram/features, PSD and witness checks |
| Sampling expectations | Six fully enumerated finite 2+2 experiments check unbiasedness and the feature-variance bias identity |
| Permutation calibration | 350 allocations over five bandwidths; scores and conditional rank validity at all 70 reference levels |
| Variational objective | 25 scale/offset states, conjugate calculation and independent BFGS optimum |
| Moving atoms | 21 Decimal checks, including displacement 1e−9 and cancellation-safe exponential calculation |
| Inputs and transfer | 26 model rejections; copy/freeze checks; 15 changed native MMD inputs; eight complete changed permutation rules; six invalid project inputs; 14 numerical practice checks |

Final suite: 6,373 numerical comparisons; maximum absolute difference 1.4211e−14. These are finite observations, not universal error guarantees. Counts describe this lesson's evidence, not future quotas.

The convention review checked half-JS and half-squared Hellinger, metric roots, TV versus twice-TV score classes, empirical versus population MMD, normalized versus unnormalized witnesses, finite diagonal moments for bias, GAN boundary log conventions, and population versus fitted Fenchel objectives.

### Independent-review arithmetic repair

The parent reviewer reproduced an accepted-input defect before freeze: processDivergence with weights [100,1e-8] and [1e-8,100], and channel [[Number.MIN_VALUE,1],[0,1]], let a positive product underflow into a false support zero. It could report infinite processed KL/JS despite finite input KL. Pre-repair model SHA256: 7117c02922fc53dad85760877e1840d395f37aaf8811377b03c2b396ebfc933f.

The exported processing calculator now accepts exact-zero channel entries or values in [1e−8,1], retaining the row-sum rule. With at most eight input weights, each at most 100 and each positive weight at least 1e−8, any positive processed mass is at least 1.25e−19. This avoids false support loss and midpoint underflow without clamping positive entries to zero. Inputs below the arithmetic range are rejected explicitly. The mathematical DPI theorem is unchanged.

The exact reviewer case and a just-below-range case are regression rejections. Minimum accepted probabilities, identity and constant maps are checked against independent formulas. Actual browser-loaded calls at all three widths confirm rejection, finite boundary values, preserved exact zeros and zero divergence after a constant map. The visible coarsening figure uses exact 0/1 entries and is unchanged.

## Browser and source evidence

scripts/review-divergence-ipm-lesson.cjs ran actual Edge at 1440, 390 and 320 pixels with the existing Space Grotesk font loaded. HMR WebSockets were isolated against concurrent authoring; no module or font requests were mocked. All widths passed:

- Ten section anchors and ten visible program questions adjacent to their complete source and output.
- Every investigation: changed valid input, rejected drafts preserving state, reset, selectors, keyboard range endpoints/increments, infinite support, zero linear gap, negative unbiased estimate, negative critic bound and exact-rank invariance.
- Keyboard scrolling of the signed Gram matrix when wider than the reading column.
- Both disclosures for all ten exercises; all deeper derivations opened before geometry checks.
- No document/equation overflow, out-of-viewBox or undersized SVG labels, clipped selected labels, sub-44px controls, page exceptions, non-HMR console warnings or failed requests.

Actual-image review found a cramped mobile ratio value and a small atom-axis label margin; both were repaired. Four formulas gained mathematical line breaks at 320px. The kernel selector uses a short descriptive label while its formula remains explained. Full checks were rerun after these fixes.

scripts/review-divergence-ipm-final-reading.cjs additionally checks the final kernel-mean/finite-moment paragraphs and repaired processing arithmetic in the loaded browser at all widths. Final source fingerprints are in durable evidence and scratch/divergence-ipm-native-verification/final-source-hashes.json.

Screenshots under scratch/divergence-ipm-browser/ were actually opened: ordinary readings, 320px coarsening/triangle/features, 390px observers/kernels, 320px moving atoms/ratios/critic, permutation histogram and final assumptions. Large region screenshots were reviewed alongside ordinary viewport captures; resized previews did not replace source geometry checks.

scripts/format-divergence-ipm.cjs verifies normalized JS/JSX AST and strings and CSS meaning before/after formatting. node scripts/verify-curriculum.mjs passes current module/order/schema checks. Parent owns production integration; no deployment or user approval is implied.

## Primary research and alternate learning

Retrieved 10 September 2026. Actual reviewed scope:

- [Polyanskiy/Wu notes](https://people.lids.mit.edu/yp/homepage/data/LN_fdiv.pdf): relevant parts of sections 7.1–7.4 and 7.6, including singular terms, invariance, processing, metric roots and Pinsker. Their unhalved JS/H² factors were converted.
- [Gretton et al. 2012](https://jmlr.org/papers/volume13/gretton12a/gretton12a.pdf): sections 2–3 for embeddings, witnesses and estimators. Our unbiased form retains every cross pair, distinguished from the displayed paired-index alternative; finite expectation checks verify the chosen form.
- [f-GAN](https://arxiv.org/pdf/1606.00709), section 2: conjugacy and variational objectives. KL conjugate/constants were derived independently rather than relying on imperfect PDF table extraction.
- [Original GAN](https://arxiv.org/pdf/1406.2661), section 4.1: discriminator optimum and JS identity; [WGAN](https://proceedings.mlr.press/v70/arjovsky17a/arjovsky17a.pdf), section 2: singular supports and topology. No universal optimization/gradient claim was inherited.
- [Gretton's creator teaching page](https://www.gatsby.ucl.ac.uk/~gretton/teaching.html) verifies the [MLSS 2020 recording target](https://www.youtube.com/watch?v=eANiXrWO1dM) and lecture description. The recording was not watched in full; its large slides exceeded the tool limit. This limit is stated in the lesson. JMLR supplies the substantively reviewed written alternative.

No real-world benchmark data were invented: figures show defined finite models, analytic laws or labeled empirical/permutation quantities. The chapter does not certify fairness, useful generation or calibrated population divergence from fitted critics. No beginner study or screen-reader listening session was conducted; actual keyboard/geometry/reading and numerical evidence are distinguished from unobserved outcomes.
