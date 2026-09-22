# PCA & Dimensionality Reduction — content and delivery record

## Current live-exploration contract — 21 September 2026

This dated UX amendment supersedes earlier prediction-entry, grading, commit-to-reveal and prediction-retirement requirements in this document. There is no learner prediction feature, even optional. Historical evidence below records the earlier interface and remains history; it is not the current acceptance contract.

Rotate or fit a direction on editable points and follow scores, residual segments and squared loss immediately; change units or standardization and inspect the leading direction; move an error budget and inspect the first qualifying count and individual validation wines; vary spreads and label rules and inspect retained-coordinate collisions.

Keep separate independent practice, model predictions, scientific validity checks and training/validation/held-out information boundaries. Meaningful valid control changes must reach the visible calculation and topic-specific diagram together. Natural algorithm Step/Back/Run actions remain where they expose a process; they must never require a learner guess. Reset restores a coherent initial state. A graph or number must not silently describe obsolete inputs; invalid inputs show an error and either clear invalid outputs or explicitly retain the last valid result. See [the current migration evidence](LIVE-EXPLORATION-CLASSICAL-EARLY.md) for implemented checks and limitations.


**Visual-layout follow-up, 14 September 2026:** [the expanded diagram review](LESSON-VISUAL-LAYOUT-REVIEW.md) separates the projection-foot, conservation and residual-alarm labels from data strokes. Point coordinates, scales, examples and prepared content are unchanged. The affected desktop/320px captures were inspected and the [PCA browser suite](evidence/pca-browser.json) passes 14 cases on these sources. This later record supersedes earlier screenshots for the affected figures.

Stable ID: `pca-dimensionality-reduction`. Module: Classical Machine Learning, Unsupervised Learning, position 12 of 39. Requested 12 September 2026: **research and writing only**, including visual/lab specifications. Author: primary agent. No independent review is claimed.

## Current state

Content is complete, subject to the source-bound checkpoint in the central delivery ledger. Implementation is not started. The published PCA lesson remains the older version. The pending K-Means proposal and all unrelated working-tree changes are outside this request. User acceptance and independent implementation review are not claimed.

The deliverables are [the complete manuscript](drafts/pca-dimensionality-reduction/lesson.md), [visual/lab specifications](drafts/pca-dimensionality-reduction/visual-specifications.md), [offline Wine data](drafts/pca-dimensionality-reduction/wine.csv), [data provenance](drafts/pca-dimensionality-reduction/data-provenance.md), and [author calculations](drafts/pca-dimensionality-reduction/author-calculations.json). Website code, rendered diagrams, labs, publication, navigation, independent review and integration are deferred to an explicit phase-two request.

## Learning contract and scope

Keep the existing title: PCA remains the main method, with a practical comparison of dimensionality-reduction alternatives. Additional material clarifies that promise rather than requiring a new title or identity.

Beginner finish line: explain centering, a principal direction, a score and reconstruction; calculate one projection and its loss; run an offline PCA analysis; choose a component count for a stated purpose. Intermediate outcomes: diagnose scaling, leakage and misleading plots; distinguish explained variance from task usefulness; interpret component coefficients. Deeper outcomes: connect covariance eigenvectors to SVD, derive the loss identity, recognize sign/tie/rank issues, understand whitening and choose a computational approach.

Use a four-point exact example before the real Wine dataset. Required skills are means, squared distances, basic Python arrays and the idea of train/validation separation. Refresh dot products, sample variance and matrix shapes locally. The existing Matrix Decompositions blueprint covers orthogonal projection, SVD and truncation; Eigenvalues covers covariance directions. These are optional review links for deeper derivations, not unexplained gates that interrupt the module sequence.

Read the complete old PCA body in sections. Preserve its useful coverage of geometry, covariance/eigenvectors/SVD, executable examples, scree/loadings interpretation, production transforms, solvers, whitening, incremental/kernel PCA, alternatives, scaling, failures and independent practice. Replace its numerical fixtures and unsupported generalizations rather than preserving errors. The original source is unchanged and recoverable at git commit `6316d31cbd58010a633c74b1c5551ad849979a72`, path `src/learn/data/topics/pca-dimensionality-reduction.jsx`, SHA256 `195fd2e37488e7e2bebad54c25010bf243c3e170ffe0c85a3326c44a0fd13a83`.

## Coverage decisions and continuity

The author reread the current AGENTS, handoff, teaching standard, domain playbook, topic-design brief, delivery contract and code standard after the user's edits. Apply their first-pass route, once-stated cautions, real dataset, recorded investigations, visible quantitative contrasts, compact teaching code and separate learning-experience assessment. This request stops at the content checkpoint; quality-first guidance does not authorize building the deferred interface.

| Idea or old coverage | Evidence/issue | Decision and location |
| --- | --- | --- |
| Data is nearly always low-dimensional; leading PCs name latent causes | Old intro asserted these without a data/model-specific basis | Replace with an explicit compression question; §3 gives one clear home for variance-versus-meaning; §10.3 develops sampling. |
| Projection, covariance, eigendecomposition, SVD and reconstruction | Useful mathematical core; some old formulas omitted mean restoration when calling the result original data | Keep and derive using a consistent four-point fixture, §§2–4/10.1. Trace forward and inverse shapes, new observations, total and per-entry errors. |
| Old manual 5-point example and mixed numerical claims | Unnecessary second running fixture with suspect displayed calculations; old scatter and output sample scopes differed | Replace with exact four-point arithmetic and verified code. Preserve the learning operation rather than the incorrect numbers. |
| Matrix-decomposition/eigenvalue prerequisite | Read actual matrix blueprint and Eigenvalues §5: projection, covariance and the same centered four-point values are taught | Use a local refresher and an explicit same-result connection, with deeper review links rather than module reordering. |
| Old Iris heatmap and API demonstration | Bare coefficient colors and unjustified biological “size/shape fact” interpretation; all features similar units obscured scaling | Use Wine with distinct magnitudes; signed coefficients, score/direction/correlation distinction, explicit biplot convention, §§5/7. Iris itself is not a required learning outcome. |
| Choosing k and scree plots | Old blanket signal/noise elbow interpretation and isotropic-sample claim were false | Real curve and purpose-specific selection in §6, explicit mean baseline, 10% budget and changed 6% exercise; finite Gaussian sample in §10.3. |
| Clustering destination note | Read both current PCA notes and previous-topic route; distances are the natural continuation | Incorporate full rotation and truncated-distance identity, erased-pair and erased-label examples in §8; note remains open for implementation. |
| RMT destination note | Correct need to distinguish sample/population/task, but asymptotic spike numbers add unnecessary prerequisites here | Adapt to finite Gaussian sample and held-out reconstruction; link the existing RMT owner for its spike theory. No unverified asymptotic example pasted. |
| Fit/transform, leakage and independent data | Core practical obligation | §§4/6 and complete folded pipeline §10.4. Distinguish full-collection description, train/validation compression and development CV; never claim final test performance. |
| Whitening, signs, degeneracies, constant/missing values | Old whitening population-variance output was wrong; individual signs and ties conflated | §10.2 explains rank bound, zero variance, sign/tied-subspace equivalence, whitening and `ddof`; new numbers were calculated. |
| Computational complexity, randomized and incremental PCA | Old text conflated SVD with explicitly forming covariance, hid input storage and asserted universal defaults/hardware ceilings | Retain distinct solver/cost mechanisms in §10.6; include 40 GB input example; no invented runtimes; batch approximation distinguished from rounding. |
| Kernel PCA, nonlinear methods, random projections, autoencoders | Old text denied UMAP transforms, conflated kernel forward/inverse, overpromised manifold outcomes, required labels/GPU for autoencoders | Correct comparison §10.7; full nonlinear algorithms belong to existing t-SNE/UMAP/manifold owner, not another complete course embedded here. |
| Applications and technical surprises | Need more than an industry list | Mean-restored reconstruction, lost pair distance, noise spectrum, decoder storage, pixel-direction interpretation and sensor residual diagnostic all teach distinct mechanisms; neural bridge shows row-order invariance. |
| Neural follow-on idea | Actual neural blueprint already proposes held-out reconstruction/stability; a noisy-input minimum would choose all dimensions | Save the budget/clean-target distinction and row-orientation reasoning in its destination note. No neural lesson edited. |

Retain the display title and stable identity. Do not expand it to list every application. The exact next module entry is `clustering-evaluation-validation-silhouette-ari-nmi`, position 13. No successor implementation is authorized by finishing this draft.

## Canonical-reference coverage check

Canonical open reference: **Jolliffe & Cadima (2016), “Principal component analysis: a review and recent developments,” §§2–3**. Read the relevant section list and core passages. Its topics include the standard algebraic/geometric definition, an example, covariance/correlation scaling, biplots, number of components, high-dimensional sample effects, functional PCA, simplified components, robust PCA and symbolic-data PCA. Decisions:

- Standard definition, geometry, scaling, biplots, dimensional choice and high-dimensional sampling all receive developed treatment.
- Simplified/sparse, robust, functional and probabilistic variations receive a clearly labeled orientation in §10.7, sufficient to recognize that each changes assumptions/objectives; their detailed derivations would interrupt the stated core outcome.
- Symbolic interval/histogram PCA is deliberately omitted from the learner core because it requires a new data representation and objective, not ordinary numeric-matrix PCA. This is a documented exclusion, not a claim of mastering every PCA research extension.
- The canonical fossil-teeth/atmospheric examples are not copied. Wine, constructed sensor comparisons and image decoding serve this lesson's outcomes with original arithmetic and explicit provenance.

Also reviewed the **official ISL Chapter 12 slide list/content and Python PCA notebook**: scores/loadings, scaling, variance accounting, biplots and SVD all have a local home. An official book PDF request failed in the web tool; do not claim the entire book chapter was read. The authors' resources and accessible chapter material are linked for learners. The survey supplied the canonical section-list audit; the notebook supplied a second practical treatment. Neither source's structure was copied wholesale.

## Hurdles, representations and evidence of learning

| Hurdle / outcome | Explanation and complete example | Representation contract | Practice / evidence |
| --- | --- | --- | --- |
| A direction is not an observation score | Dot product and mean-restored inverse for A; new point (6,4) | F1, L1 and F3 | Sign prediction, exercises1/2, changed coordinates |
| Maximum variance = minimum perpendicular loss | Right triangle, 20=18+2, covariance/SVD identity | F2 and L1 | Rotate to second axis, account for denominators |
| Scale is part of the objective | Rectangle unit change and real Wine raw/standardized results | L2 and F4 | Predict a tie at m=a/b; repair scaling conclusion |
| k needs a decision criterion | Fixed split, mean baseline, explicit error budget | L3, all counts including0/13 | 6% budget self-check selects10, mini-project |
| Coefficients, correlations, scores and arrows differ | Four-point .7071 coefficient versus .9487 correlation; specified biplot dot product | F4/F5 | Explain an observation's recovered feature |
| Preserved variance does not guarantee neighborhoods or labels | Full-distance identity, A/B collapse, 99.01% rectangle | L4 | Change label rule independently of PCA geometry |
| Samples have apparent directions even under population symmetry | Seed23, Gaussian40×20, first-two share .2459 | F6 | Changed-seed interpretation, exercise8 |
| Discarded residuals can be diagnostically valuable | Same-mode versus cross-mode sensor reading | F7 | Explain score0/residual18 without an alarm threshold |
| Practical implementation and resource cost | Native SVD/API, train-only transforms, decoder storage, solver alternatives | F3 plus code/decision table | Fold comparison, storage threshold, leakage diagnosis |

No visual quota was used. Four investigations address distinct manipulable questions; seven figure contracts are placed at the explanation they serve, with F1 allowed to share L1's immediately visible state. F3/F5/F7 may stay static. Required explanation and exact tables remain available without interaction.

## Claim and source ledger

Sources inspected **12 September 2026**. Record below distinguishes material read from executable claims checked. The manuscript is original exposition and calculations, not a reproduction of a source's wording. Moving URLs were checked this date; the local numerical environment was Python3.12.14, NumPy2.3.5, SciPy1.18.1, scikit-learn1.9.1.

| Claim / uncertainty | Source and locator | What was inspected / resolution |
| --- | --- | --- |
| Standard PCA objective; scaling/loading/biplot conventions; variants | [Jolliffe & Cadima](https://pmc.ncbi.nlm.nih.gov/articles/PMC4792409/), §2(a), §2(c)(i–iv), §3 headings and relevant passages | Section-list coverage check; algebraic versus geometric equivalence; conventions. Our numerical derivations and examples are computed separately. Use convention-specific arrow interpretation, not generic biplot-angle claims. |
| A second practical teaching route | [ISL official resources](https://www.statlearning.com/resources-python), [Chapter12 slides](https://web.stanford.edu/~hastie/ISLR2/Slides/Ch12_Unsupervised_Learning.pdf), [Python PCA lab](https://islp.readthedocs.io/en/stable/labs/Ch12-unsup-lab.html#principal-components-analysis) | PCA introduction, normalized directions, score/loadings, scaling, plots and notebook SVD comparison read. No copied USArrests figures/data/code; resource annotation warns about extra packages/version setup. Full book PDF unavailable through attempted web request. |
| Video fit rather than title-only recommendation | [MIT lecture19](https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/resources/lecture-19-video/), [PCA slides](https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/d85e1a9d113142ade8ce5e4f5ef0b4e8_MIT18_650F16_PCA.pdf), multivariate review through PCA algorithm | Video destination and substantive companion slides inspected. Mathematical fit verified; 1/n versus1/(n−1) difference annotated. Did not watch full video or claim timestamps. |
| Wine data origin, license, numeric interpretation | [UCI109](https://archive.ics.uci.edu/dataset/109/wine), dataset information, citation, variables and license; [loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html) | UCI metadata and CC BY4.0 inspected; actual offline observations come from installed scikit-learn. ZIP blocked in shell; no direct-source byte comparison claimed. No missing rows; label codes excluded. Numeric units not invented. |
| Scaling changes real PCA outcomes | [Scikit-learn scaling example](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html), PCA/scaling comparison | Worked source used as an alternate resource; our raw/standardized calculations use explicit full178 fit, and our budget uses separate133/45 split. These protocols are not confused with the source example. |
| SVD orientation and reduced shapes | [NumPy SVD](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html), parameters/returns/notes | Documentation was NumPy2.5; relevant operation also checked in installed2.3.5. `full_matrices=False`, directions as rows, nonnegative singular values and reconstruction verified. |
| PCA API centering, shapes, solvers, variance/whitening | [PCA1.9.1](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html), parameters/attributes/inverse_transform | Current solver set includes covariance_eigh. Local fit/transform/inverse and whitening probe resolves n versus n−1: ddof1=1 and ddof0=.75 on four rows. Do not copy docs' simplified sqrt(n) wording into incorrect printed output. |
| Standardization denominator and constant columns | [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html), scale_ and Notes | ddof0, zero-variance scale1; derived n/(n−1) sample-variance factor. Distinguish this from PCA whitening. |
| Leakage and fold boundaries | [Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage), §§12.1–12.2 | Train-only fitted preprocessing, pipelines inside folds and correct downstream usage read. A complete local 5-fold logistic comparison was calculated; no final test claim. |
| Exact and approximate computational approaches | [Halko et al. manuscript](https://arxiv.org/pdf/0909.4061), §§1.2–1.4 / prototype and StageA/StageB; [decomposition guide](https://scikit-learn.org/stable/modules/decomposition.html), PCA subsections | Two-stage sketch/reduced SVD and pass/shape costs checked. Only asymptotic/byte arithmetic is stated, not measured speedup or universal solver ranking. |
| Incremental approximation and data access | [IncrementalPCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html), description/partial_fit and decomposition guide | Batch-based approximation and memory scope; distinguish incremental truncation from mere numerical roundoff. No implemented streaming example or performance benchmark claimed. |
| Kernel forward transform versus inverse | [KernelPCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html), transform/inverse_transform and fit_inverse_transform | Correct old conflation: new-point projection is supported independently of optional inverse fitting. Detailed nonlinear lesson is a separate owner. |
| UMAP transforms and alternative objectives | [UMAP transform tutorial](https://umap-learn.readthedocs.io/en/latest/transform.html); [random projection guide](https://scikit-learn.org/stable/modules/random_projection.html) | New-data transform and finite-set approximate-distance guarantee inspected. No promise that UMAP always unrolls a manifold or wins universally. |

The finite Gaussian example does not reuse the destination note's restrictive asymptotic spike formulas. Its generator and sample outputs are the evidence for the specific number; the identity population covariance is a property of the specified independent standard-normal construction. Orthogonal-distance, centering/row-permutation, variance/error and storage statements are derived explicitly in the manuscript.

## Author checks completed in the content phase

1. **Arithmetic and executable manuscript outputs.** Computed the four-point covariance, scores, projections, SSE, alternate directions, new observation, Wine full-data ratios/coefficients, exact split and all14 validation component counts. Ran the six displayed linked programs for SVD, library transform, Wine exploration, budget selection, inverse scaling and eigenvalue comparison. The separate Gaussian and five-fold prediction snippets were run during research; their unchanged outputs are retained rather than rerunning a full campaign. Compared displayed output blocks with those results. Corrected a cumulative Wine rounding error from96.16% to96.17%.
2. **Meaningful fixture coverage.** Checked L1 angles0/45/90/135, common translation, collinearity and identical-data null; L2 multipliers1/2/10 under raw/standardized modes plus a changed-width tie; L3 budgets6%/10%/11%, k0 and k13; L4 coordinate/label collision cases and unchanged fit under relabeling. Fixed-basis validation loss decreases by construction; no false denoising U-curve is specified. Input-control/runtime checks remain deferred.
3. **Data and examples.** CSV preserves the bundled178×13 observations and labels with round-trip floating precision; all13 feature columns are finite. Mean/scale/split/versions and calculated outputs are retained. Research shell download failure was replaced by the legitimate bundled dataset, with provenance stated accurately.
4. **Coverage/correctness reading.** Checked old core coverage against the new packet and canonical survey. Repaired scaling and sign universals, denominator ambiguity, covariance/SVD cost confusion, input-memory omission, unsupported biological labels, autoencoder requirements and nonlinear transform misconceptions. This is author review, not independent review.
5. **Handoff checks.** Validate local links, lesson route IDs, visual anchors, fenced examples and the two-phase checkpoint. Run ledger behavior checks and inventory generation at the substantive completion boundary. Record only their actual final outcome; no application build or browser suite is required for this content-only change.

## Learning-experience assessment — author heuristic, not a learner study

| Checklist question | Finding and disposition |
| --- | --- |
| First-pass route and branches | Explicit after the opening, with §§1–6 plus practice1–4/readiness; interpretation, task connection and §10 clearly available as deeper routes. No derivation is needed before the first projection. |
| Cautions have a home | Variance-versus-meaning at §3; scale choice at §5; leakage/evaluation at §6; sampling and denoising in §10.3. Later examples refer to these distinctions instead of repeatedly printing caveats. Code prints results only. A manual voice pass and rough phrase count found no caution-density problem. |
| Real question and data | Return to the original wine question through a55.41% two-score picture and an8-coordinate budget choice; data/provenance included offline. The constructed fixture teaches exact mechanics before real measurements. |
| Labs investigate | Four distinct questions, unset prediction controls with model comparison, continuous/entity controls and null cases are specified. Mathematical fixtures checked; actual interaction/user behavior not yet tested. |
| Figures are perceptible | Equal geometric scales, loss baselines, readable residuals, 0-to1 budget curve and narrow layouts specified individually. **Actual rendered desktop/phone perceptibility remains deferred.** Tables are supplements, not excuses for an unreadable plot. |
| Connections are explicit | Variance/reconstruction, covariance/SVD, scores=AV=US, full rotation/truncated pair distances and neural row-order invariance are explained. Canonical variants are developed or deliberately scoped. |
| Displayed code teaches the mechanism | Short runnable calculations, finite-matrix assumption stated in prose, no printed warnings or defensive scaffolding obscuring projection. Linked snippets specify which preceding setup to rerun; no automatic notebook execution assumed. |
| Independent variation | Collinear new data/new observation, keeping PC2,6% budget, storage break-even, dependence counterexample and changed seed; hints/reveals and exact checks supplied. Mini-project has criteria rather than a claimed automatic mastery grade. |
| Screenshots and integration quality | No screenshots were captured or inspected because visual implementation is deferred. Phase two must examine informative contrast/prediction states on desktop and phone, as well as the default view. Lab implementation, browser accessibility, final independent review and integration remain unknown; these are concrete phase-two requirements, not missing content sections. |

## Phase-two continuation

Current next action: **wait for a request to implement this prepared topic**. A finish request must pass `node scripts/build-curriculum-inventory.mjs --topic pca-dimensionality-reduction --work finish`, then consume this entire packet. Do not restart the research by default, skip the new quality rules, or advance to the next published topic. The user may also request another content-only topic independently.

Intended production destination: existing `src/learn/data/topics/pca-dimensionality-reduction.jsx`; topic-owned blueprint and semantic examples/models/figures/labs as specified. Retain title/ID/module order. The manuscript is the new content authority while pending; the old published body is the baseline for conservation, not a source of current approval. Keep all required draft inputs and their hashes until an equivalent implemented source checkpoint is retained.

During phase two, build the actual diagrams/labs and convert the manuscript into the reader components; wire real data download and references; then complete applicable native/model, browser/keyboard/mobile, author/independent correctness and learning-experience reviews, affected fixes, curriculum/loading/build and integration checks. Reassess any changed API or unresolved implementation question then. Update destination notes to implemented only when that work is real. Formal independent review is deferred, not bypassed.

Do not reopen or overwrite the separate unreviewed K-Means proposal. No new temporary utilities, copied videos, image downloads, full-book archives or screenshots were retained by this content phase. The Wine CSV and compact author calculations are required draft inputs and must not be cleaned up as scratch.

## Content-checkpoint validation — 12 September 2026

- `node scripts/verify-lesson-delivery.mjs` passed all eight behavior groups, with108 tracked topics. PCA is a new content-first entry; the107 historical entries and K-Means pending proposal remain distinct.
- `node scripts/build-curriculum-inventory.mjs` passed:28 modules,1218 unique topics,228 publications and7 guided paths retained. Effective current phase counts are107 content-complete and106 implementation-complete; these differ from historical completion because of the separately pending K-Means source revision.
- The packet's26 local Markdown links resolved; all11 internal lesson route IDs exist. Eight displayed Python programs have matching recorded output blocks; six linked blocks were executed from the manuscript and the two unchanged research programs reuse their recorded runs. CSV round-trip equals the bundled numeric data. The rough voice check found2 listed hedging phrases among220 prose-like paragraphs; the author's actual reading, not that count alone, supplied the voice assessment.
- The original published PCA source still matches its starting SHA256. No production lesson, blueprint, publication mapping or runtime asset was modified. The pending draft's six required inputs are hashed in the central ledger. Authoring inventory generation is not a website build.
- Finish preflight is a read-only continuation check. Passing it means the complete content inputs are present and current; it does not start implementation, endorse an unrendered visual or replace phase-two review.

## Phase two — implementation, verification and integration, 12 September 2026

The finish request passed `--work finish` against the unchanged content checkpoint. The complete manuscript and specifications were consumed; the implementer is the primary agent, not the phase-one author's independent reviewer.

### What was built

- `src/learn/data/topics/pca-dimensionality-reduction.jsx`: the eleven-section manuscript as reader components, with the first-pass route immediately after the introduction, one caution callout in section 3 as the home for variance-versus-meaning, and the eight displayed programs placed where the manuscript placed them. Deeper branches 10.1–10.7 are visible subsections, not collapsed, so their figures stay in the reading flow.
- `src/learn/data/pca-models.js`: pure two-dimensional projection and exact 2×2 eigen models, the rectangle metric, label collisions, and Wine recomputation (scores, validation loss curve, per-row reconstruction) from natively fitted quantities embedded in `src/learn/data/pca-wine-data.js` (178 rows at full precision, full-collection and 133/45 training fits, Gaussian spectrum). The manuscript's symbolic column directions are stored as rows in code, as the specification requires.
- `src/learn/components/lesson-labs/PcaLabs.jsx`: L1 projection workbench (editable points up to twelve within ±10, reference and proposed angles, recorded prediction with residual hidden until compared, stale invalidation, fit-best-direction, Back history, translation), L2 metric lab (pending versus applied parameters, tie handling with four equally good directions drawn), L3 budget lab (curve hidden until the count is committed, budget slider and typed value, per-wine residual table in original units), L4 variance-versus-label lab (spread, label rule and retained-component controls, magnified inset). Every lab records a prediction and compares it with the model.
- `src/learn/components/lesson-labs/PcaFigures.jsx`: F1 shadow-and-score, F2 conserved total, F3 matrix shapes, F4 Wine overview with cultivar overlay off by default, F5 biplot reading, F6 Gaussian spectrum, F7 residual alarm. All values are computed from the models or the embedded native record; none is hand-drawn.
- `src/learn/data/pca-examples.js`, generated by `scripts/verify-pca-examples.py`, which executes all eight displayed programs (three as declared continuations of an earlier program's namespace) and asserts twelve oracle values from the manuscript.
- `public/learn-assets/pca/wine.csv`: the phase-one CSV served as a real download, linked from section 1 with attribution in the references.
- `src/learn/data/curriculum/blueprints/pca-dimensionality-reduction.js`, registered in the blueprint index.

Specification deviations, with reasons: F1 is a separate static figure at the top of section 2 rather than L1's initial state, because the two sit at different places in the explanation and F1 must be visible before the dot product is introduced. The L1 prediction options are less/same/more error rather than radio buttons; a select with a compare button behaves identically for keyboard users. Dragging points was not implemented; coordinates are edited by number field, which the specification allows. The Wine F4 point selector was not built; the figure supports the exploratory reading it was designed for, and the per-wine inspection lives in L3 where it has a decision attached.

### Checks actually run

| Check | Result |
| --- | --- |
| `node scripts/verify-pca-models.mjs` | 33 grouped checks: exhaustive 0.05° angle sweeps and conservation on six clouds, analytic four-point, rectangle and collision fixtures, browser Wine recomputation against the native 133/45 record within 1e-8, storage arithmetic, rejection contracts. Evidence `evidence/pca-models.json`. |
| `scratch/lesson-tools/Scripts/python.exe scripts/verify-pca-examples.py` | 8 programs executed (NumPy 2.3.5, SciPy 1.18.1, scikit-learn 1.9.1); every output block equals the manuscript's; 12 oracle assertions. Evidence `evidence/pca-native.json`. |
| Production build | Passed after each change; final selected-route chunk `pca-dimensionality-reduction-*.js`. |
| `node scripts/verify-pca-browser.cjs` (Edge, Playwright) | 11 cases: visible code/output for all eight programs, anchors, downloadable CSV, all four labs operated by keyboard and pointer including stale-prediction invalidation, 390 px and 320 px layouts without overflow, completion persistence, actual successor route, import/render failure recovery, lazy closure. Evidence `evidence/pca-browser.json` and nine screenshots. |
| `node scripts/verify-curriculum.mjs`, `node scripts/build-curriculum-inventory.mjs`, `node scripts/verify-lesson-delivery.mjs` | Passed; 1,218 IDs, 353 briefs, module order and 228 publications preserved. |

Two data defects were caught and fixed during implementation: one Wine value with seven significant digits was truncated by a `%g` export and made the browser scores disagree with the native fit at 1e-9, so the data module now stores full-precision values; and `<output>` elements inside range labels captured the label association, which the `Field` component now binds explicitly to the control.

### Author learning-experience checklist (heuristic; not a learner study)

1. Route: stated after the introduction; section 10 opens by naming itself the deeper branch. Sections 7 and 8 are labeled in the route as interpretation and connection.
2. Cautions: one callout in section 3; later percentages are stated without repeated qualification. No displayed program prints a cautionary sentence.
3. Real question: the Wine spreadsheet opens the lesson and returns in sections 5, 6, 7 and 10.4; the CSV is downloadable.
4. Labs: all four record a prediction and compare it. Free controls: arbitrary angles and edited points (L1), any a, b, m (L2), any budget and any wine and feature (L3), spreads and label rule (L4). Fixture checks: 0°/45°/90°/135° and collinear/identical/translated clouds; tie at m = a/b and standardized ties; budgets 0.10/0.11/0.06; y-labels versus x-labels. All verified in `verify-pca-models.mjs`.
5. Figures: inspected in the nine screenshots at desktop and 390 px. Fixed during inspection: a ruler spilling outside its plot, an Â label covering point B, a plot title colliding with a tick, oversized budget-curve text, and an inset overlapping a point. The L4 inset caption is slightly wider than its box on phones and remains legible.
6. Connections: variance and loss (conserved total), covariance and SVD (10.1), scores = AV = US, distance identities (section 8), PCA versus regression (10.7). The canonical survey and ISL are linked.
7. Code: the SVD program is seventeen lines with the algorithm in eight; no guards beyond what the manuscript specified.
8. Practice: eight changed exercises plus the mini-project; the 6% budget and the changed-seed Gaussian give exact values, and the budget lab reproduces the 6% answer.
9. Screenshots: captured after a compared prediction, after a revealed crossing, with cultivar labels overlaid, and at 390 px for the conservation, shapes, label, Gaussian and metric views; all opened and inspected.

### Independent review

See `PCA-INDEPENDENT-REVIEW.md` for the reviewer's separate correctness and learning-experience findings and their disposition below.

### Independent review disposition, 12 September 2026

The reviewer (a separate agent that did not author any source) recomputed the four-point fixture in exact rationals, the Wine row-0 chain and every embedded fit against a fresh scikit-learn fit, the label collision and its null, the Gaussian spectrum, the CSV against `load_wine`, the √0.9 correlation, storage and 40 GB arithmetic, whitening `ddof` claims, cumulative-variance statements, practice solutions and seven of the eight displayed program outputs, and ran the nine-item learning-experience checklist. Its report is `PCA-INDEPENDENT-REVIEW.md`; the hashes recorded there identify the sources it read, which precede the fixes below.

| Finding | Severity | Disposition |
| --- | --- | --- |
| Budget-curve axis labels rendered in SVG-default black on the dark panel | Material | Fixed: the curve's text now has an explicit fill; recaptured in `evidence/pca-budget-desktop.png` and inspected. |
| “Before running: Before running:” on the Wine scaling program | Minor | Fixed in the program's question; examples regenerated by the native verifier. |
| Typing a negative coordinate in the projection editor could zero the point | Minor | Fixed: empty and lone-minus drafts are ignored while typing. |
| Tie multiplier a/b unreachable on the 0.25-step slider | Minor | Fixed: a typed multiplier field (0.01 step) was added. A follow-on finding from the browser check: a multiplier typed to four decimals produced variances 5.3333 and 5.3335 and an arbitrary “second axis” verdict; ties are now reported when the two variances agree within 0.01%, and the lab caption says so. `verify-pca-models.mjs` asserts both the near-tie and a clearly non-tied neighbour. |
| Captions restating adjacent prose (F7, F4) | Minor | Trimmed both captions. L1 and L3 captions were kept because they name lab-specific behaviour. |
| Screenshots missed the L2 tie, L4 collision, F5 and F7 states | Minor | The browser verifier now captures all four; inspected. Two further label-size issues in F5 and F7 were found in those captures and fixed. |
| F6 x-axis label clipped at phone width | Minor | Shortened. |
| Wine retrieval date absent from the lesson's provenance | Minor | Added to the reference bullet. |

Every displayed program still reproduces the manuscript's output after these changes; the model verifier passes 33 grouped checks and the browser verifier 11 cases. The ledger's content checkpoint was reconciled for this record alone, which is the only content file that changed; manuscript, specifications, data and calculations remain byte-identical. `verify-lesson-delivery.mjs` currently fails on a missing handoff record for a different topic registered by a concurrent content author; that is outside this increment and was not altered.


## Prepared-content implementation audit — 14 September 2026

On the user's request to verify PCA through GMM, a fresh reviewer read the complete prepared manuscript and visual specifications against the complete implemented PCA lesson, examples, models, figures and labs. The review found no omitted core explanation, derivation, worked program, exercise or resource. The coverage map and exact inspected source versions are in [the PCA/Clustering audit](PCA-CLUSTERING-CONTENT-IMPLEMENTATION-AUDIT.md). This review is separate from the original 12 September review; its focused repairs preserve the prepared packet and all displayed programs/data.

The metric lab accepted b=4 and multiplier=10 but passed coordinates of ±40 into a shared ±20 point guard, throwing before the learner could apply the draft. Its validated rectangle now goes through the same covariance calculation without widening the free-point input contract. A regression checks 27 boundary/interior parameter combinations in raw and standardized modes and confirms that the original free-point bound still rejects 21. The existing model verifier passed **34 grouped checks** on 14 September; native program execution evidence remains valid because its source and verifier hashes are unchanged.

Other local repairs: projection Back restores the selected observation alongside its prior point set (Add→Back previously indexed a removed point); Fit best direction retains its exact angle instead of rounding to whole degrees; feedback distinguishes two class-pure retained locations from four distinct observations; compared predictions cannot be rewritten after the answer is revealed; a changed budget hides its new crossing until a fresh commitment. The first-pass route correctly asks for the three investigations in sections 1–6; the fourth is the optional section-8 connection.

Browser closure, screenshot inspection and source-checkpoint reconciliation for these repairs belong to the integrating root agent. The model check alone does not certify that browser work. All eight learner programs, the Wine CSV and fitted data, the manuscript and specifications remain unchanged. No publication or curriculum order was changed by this focused repair.

The production browser suite initially passed its original 11 cases on the integrating build. Complementary assertions were then added to `verify-pca-browser.cjs` for legal ±40 rectangle coordinates and independent axis variances, Add→Back selection restoration, an exact noninteger fitted angle, two class-pure retained locations, stale budget-curve hiding and frozen/reopened predictions. These extend the suite to 14 case records and require a fresh root execution; this append does not claim they have already passed.

## Coordinator verification closure, 14 September 2026

The prepared-content comparison and subsequent repairs are closed in [the five-topic audit](PCA-THROUGH-GMM-IMPLEMENTATION-AUDIT.md). The coordinator ran 14 passing production browser cases against the corrected topic source, including changed-input and prediction regressions, narrow layouts, module navigation, loading and recovery. Selected informative screenshots were actually inspected; exact captures and the mathematical/native evidence retained for unchanged code are listed in that record. The saved manuscript/specifications and offline author inputs are unchanged by this audit. The appended design and repaired implementation are re-bound in the existing revision's phase ledger; earlier completion dates and independent-review attribution are preserved. User acceptance is separate.
