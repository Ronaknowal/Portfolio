# Regularization: content design and continuation

## Current live-exploration contract — 21 September 2026

This dated UX amendment supersedes earlier prediction-entry, grading, commit-to-reveal and prediction-retirement requirements in this document. There is no learner prediction feature, even optional. Historical evidence below records the earlier interface and remains history; it is not the current acceptance contract.

Edit z, penalty and mixing fraction and follow the soft threshold and exact-zero interval live; edit rows and coordinate order and step actual residual updates; change physical airfoil measurements and inspect linked polynomial terms and fitted output; vary inputs, coefficients and keep probability and inspect all four dropout branches and expected loss immediately.

Keep separate independent practice, model predictions, scientific validity checks and training/validation/held-out information boundaries. Meaningful valid control changes must reach the visible calculation and topic-specific diagram together. Natural algorithm Step/Back/Run actions remain where they expose a process; they must never require a learner guess. Reset restores a coherent initial state. A graph or number must not silently describe obsolete inputs; invalid inputs show an error and either clear invalid outputs or explicitly retain the last valid result. See [the current migration evidence](../../LIVE-EXPLORATION-CLASSICAL-EARLY.md) for implemented checks and limitations.


Stable ID `regularization-l1-l2-elastic-net-dropout`; authorized batch position 4. Author `/root/classical_feature_content`, 12 September 2026. This is a complete research/write packet, with implementation and its formal review still pending. Root owns the central ledger, hashes and integration.

> **Superseded, 14 September 2026.** Everything above this line is the phase-one record and is kept unchanged. Implementation
> is no longer pending: the lesson is implemented and verified (see "Phase two: implementation" below), and an independent
> phase-two review has been completed and dispositioned (see "Disposition of the independent review"). Where a phase-one
> sentence says implementation or review is outstanding, read it as a statement made on 12 September. The central phase
> ledger, the generated inventory and the authoring handoff are the parent's to reconcile, and this record does not speak
> for them.

## Preflight, scope and original conservation

Ran `node scripts/build-curriculum-inventory.mjs --topic regularization-l1-l2-elastic-net-dropout --work content`; read its returned inventory, domain guidance and complete destination note. No existing topic blueprint was supplied, so this is the individual design. Read the entire original `src/learn/data/topics/regularization-l1-l2-elastic-net-dropout.jsx`, including all code, numeric outputs, diagrams, failure cases and exercises. Baseline commit `8c5da59f18516be77c29d5aeeafca3decca4f738`; source SHA-256 `b42e476faa4c993589ac721605fb462865b588987a9c79c162d7f7ee87d001b9`. Source remains untouched.

The title remains accurate. The immediate preceding CV lesson supplies assessment/selection and complete-pipeline boundaries; feature preprocessing was introduced just before it. The next Feature Selection & Importance lesson will own attribution questions. Neural-network layers are later in the sequence, so this manuscript locally defines a hidden unit and activation before masks. Core first-pass route appears immediately after the introduction; SVD, priors, generalized penalties and information criteria are deeper optional branches, not concealed readiness requirements.

| Original material | Conservation and correction |
| --- | --- |
| Motivation, penalty families and history | Retain training-cost/preference distinction and all four named methods; replace unsupported historical priority details, universal ridge risk dominance and fixed winner/rate claims with declared objectives and actual evidence |
| Ridge geometry and formula | Retain geometry, identifiability, singular directions and computational alternatives; correct summed-versus-mean factors, original-coordinate nonmonotonicity and the false claim L2 can never produce zero |
| Lasso geometry and coordinate descent | Preserve thresholding, paths and computational explanation; derive the missing sample-count factor, distinguish zero/nonzero KKT conditions and remove corner/forever-zero/causal-support claims |
| Elastic net and correlated features | Retain grouping motivation, mixed penalty, original paper and coordinate update; prove exact duplicate case and avoid treating mere correlation as guaranteed nonuniqueness or universal equality. Explain historical rescaling versus the current documented estimator |
| Synthetic native demos and path plots | Replace invented/mislabeled results with the full licensed observed Airfoil input, real fold paths, a complete executable protocol and exact small hand traces. Old claimed four-signal/five-zero mismatch is not retained |
| Dropout and network code | Preserve masks, expected-value reasoning and generalization motivation. Derive exact independent-feature squared-loss penalty; old two-layer code evaluated training rows while labeling the score test MSE, so that output and inferred gap are removed. The later deep owner supplies full-network native comparison |
| Internal CV and early stopping | Correct scaling before internal LassoCV folds, Ridge/Lasso alpha mismatch and early stopping on a protected test set. Early stopping is a related spectral preference, not exactly ridge in general |
| Framework/tree snippets | Replace repetitive untethered framework recipes with the complete local regression/dropout programs. The earlier Gradient Boosted Trees lesson owns tree leaf/structure penalties and stopping implementation; a leaf L1 penalty does not automatically delete a whole tree. Detailed adaptive optimizer implementation belongs to its optimizer owner |
| Solvers, scaling and failure modes | Retain practical resource/conditioning discussion; remove arbitrary n=d dispatch, universal SAGA schedules, Dask shortcuts, fixed training-time multipliers and mandatory-normalization claims. State actual assumptions, dimensions, convergence and units |
| Parameterization note and advanced ideas | Fully incorporate the open two-factor whole-objective derivation; add generalized smoothness, MAP normalization and a bounded substantive AIC/BIC/MDL home agreed with root |
| Practice | Replace mostly repeated or unsupported winner assertions with ten changed problems and optional explained solutions, plus four different editable investigations |

The user asks for useful depth, not fidelity to inaccurate old claims or duplicated library snippets. No useful original mechanism is discarded without the disposition above. The source's fabricated/illustrative benchmark shapes do not deserve conservation as measured evidence.

## Canonical coverage and neighboring ownership

The inspected current sklearn linear-model reference covers 1.1.2 Ridge (objective, complexity, CV), 1.1.3 Lasso (objective, coordinate descent, paths, CV and information criteria), 1.1.4 Multi-task Lasso, 1.1.5 Elastic Net, 1.1.6 Multi-task Elastic Net, and 1.1.7–1.1.8 least-angle/lasso-path algorithms. Core objectives, solvers, pipeline assessment and path interpretation are taught locally. Multi-task/group structure receives a defined bounded bridge; a full multi-output optimization course is not a core prerequisite. LARS is a path-computation alternative rather than a second complete solver tutorial. Robust, Bayesian-estimator and classification API branches are not renamed as missing regularization content; their models have separate owners.

Zou–Hastie sections inspected: 1 introduction; 2.1 definition, 2.2 augmented lasso relation, 2.3 grouping effect, 2.4 priors; 3.1 naive-estimate overshrinkage and 3.2 historical corrected estimator. The manuscript retains the mechanisms and convention distinction; it does not claim their specific empirical rankings or full later experiments were independently replicated.

Root supplied the official ESL chapter 7 complete section list from publisher front matter, https://link.springer.com/content/pdf/bfm:978-0-387-84858-7/1, PDF page 15. This packet owns the bounded 7.7 Bayesian/BIC and 7.8 MDL comparison agreed with CV/Bias–Variance, while the root's Bias–Variance owns the complete fixed-smoother optimism/trace derivation. Section 9 links that exact destination. Full inaccessible ESL chapter text was not read or used as a claim source.

Grünwald tutorial contents inspected: chapter 1 introduction/compression/selection/crude-versus-refined/philosophy/Occam/history/summary; chapter 2 codes and information, model/statistical preliminaries, crude codes, universal codes, refined MDL, extensions, beyond-parametric cases, relations to Bayes/CV/Kolmogorov, problems and conclusion. Local scope covers coding cost, declared two-part code, finite-normalizer NML and its relation to evidence/BIC. Prefix/Kraft proofs, minimax-regret proof, infinite-model repairs, prequential coding and coding philosophy remain explicitly deeper source study, not unowned core skills. A full MDL monograph would obscure this lesson's main purpose.

The parallel deep `dropout-droppath-stochastic-depth` author agrees to own unit/channel/branch masks, full train/eval and BatchNorm interactions, stochastic-depth placement and an actual CPU model comparison. Classical content owns the independent-feature expected-square-loss derivation. No need to reproduce it as that later lesson's central activity. The nonconvex destination note is incorporated but stays implementation-open.

## Hurdles, representations and learner evidence

| Learner hurdle | Representation and activity | Evidence of understanding |
| --- | --- | --- |
| A penalty confused with better test accuracy | Scalar objective decomposition plus real development paths | Separate what is optimized from what was assessed |
| Unit changes mistaken for importance | Meter/centimeter product and penalty inset | Track identical prediction with changed parameter cost |
| Sparsity treated as a slogan | I1 editable threshold, coordinate formula and exact geometry | Predict zero/sign under a changed association |
| A solver treated as a black box | I2 actual editable rows, centering, partial residual and KKT trace | Recompute a coordinate and recognize target-shift null |
| Coefficients treated as unique attribution | Duplicate-feature parameter plane and full minimizer segment | Give different coefficient vectors with the same fitted result |
| Raw measurements disconnected from model terms | I3 measurement-to-polynomial-to-signed-contribution trace | Change a raw frequency and track all affected terms |
| Mean output confused with mean loss | I4 weighted four-mask probability tree and nonlinear inset | Explain exact nonnegative extra loss and its limits |
| Regularization confused with an unchanged fit | Factor symmetry and scalar product objective | Find the whole optimum off the zero-loss curve |
| All complexity criteria treated as the same penalty | Likelihood/complexity accounts plus actual bit code | Explain units, assumptions and a real arithmetic disagreement |

All investigations start with an unset prediction bound to the applied entities. Changed/null cases have calculated expected behavior. Distinct geometry, residual traces, contribution diagrams and probability trees replace a generic repeated lab design. Static figures sit beside the explanation they clarify. Visual contracts specify exact equations, state, labels, units, keyboard/mobile/text alternatives and phase-two closure without prescribing decorative uniformity.

## Substantive research inspected

| Primary source / learning resource | Actual body inspected and use |
| --- | --- |
| [sklearn linear models](https://scikit-learn.org/stable/modules/linear_model.html) | Relevant section list and full objective/CV/complexity/multi-task/LARS bodies above; current normalization and criterion assumptions |
| [Zou–Hastie](https://hastie.su.domains/Papers/B67.2%20%282005%29%20301-320%20Zou%20%26%20Hastie.pdf) | Sections 1–3.2, including exact-duplicate lemma and standardized same-sign grouping bound; not full later experiment sections |
| [Tibshirani uniqueness](https://arxiv.org/pdf/1206.0313) | Introduction, section 2.1 equal-fit/KKT lemma, 2.2 general-position conditions and broader-model lemma; no blanket claim all correlated designs are nonunique |
| [Wager and colleagues](https://nlp.stanford.edu/pubs/wager2013dropout.pdf) | Sections 3–4 additive/dropout noising and square-loss versus logistic approximation; local factors independently derived and exactly enumerated |
| [Srivastava and colleagues](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) | Mask definition/training sections 4–5.1, experimental comparison 6.5, rate/model-size discussion 7.3 and averaging discussion 7.5; hypotheses and experiment-specific results kept distinct |
| [PyTorch Dropout](https://docs.pytorch.org/docs/main/generated/torch.nn.Dropout.html) | Current signature, drop-probability parameter, inverted scaling and eval identity; stable URL was empty, main documentation worked. No PyTorch execution claim |
| [AdamW original](https://arxiv.org/pdf/1711.05101) | Section 2, propositions and algorithms for ordinary SGD equivalence versus adaptive decoupling |
| [Grünwald MDL tutorial](https://homepages.cwi.nl/~pdg/ftp/mdlintro.pdf) | Full contents, chapter 1.1–1.4 body, 2.2.3 precision/information, 2.5.3 NML, 2.6.3 evidence approximation and 2.9.2 MDL/BIC distinction; not all eighty pages |
| [Lasso model-selection visual/code example](https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html) | Actual constructed-noise/Diabetes setup, criterion derivation/plots and internal-CV code. Learner annotation explicitly corrects preprocessing placement when adapting that example |
| [Tomography example](https://scikit-learn.org/stable/auto_examples/applications/plot_tomography_l1_reconstruction.html) | Full actual projection-operator builder, synthetic sparse image, noise, fitting and display code; meaningful visual alternative with its favorable sparse-input assumption explained |
| [UCI Airfoil](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise) | Data description, actual units, attribution and license; locally preserved complete observed input |
| [Speed–Yu normal-regression selection](https://www.ism.ac.jp/editsec/aism/pdf/045_1_0035.pdf) | Sections 1–2 conditional/averaged prediction-risk setting, finite-dimensional assumptions and refitting distinction. Context check only; no claim of reading all its later AIC proofs or using universal rankings |

Akaike memorial metadata was not used as a technical-body source. Schwarz's original PDF returned a gate; a PSU page failed, and CMU slides returned a JavaScript shell. No inaccessible content or unwatched video is endorsed. The two inspected visual/code resources provide genuinely different learning routes without a video quota. No book/PDF/media download is retained.

## Author checks and frozen-content boundary

Read the complete new manuscript in ordinary order, then the complete visual contracts, including all code, changed practice and solutions. Actual 60-fit Airfoil calculation had no convergence warnings; all families choose the smallest tested strength, the selected final lasso retains all twenty terms, and large lasso/elastic-net penalties exactly reach the mean baseline. These observed results are preserved even though they do not make an idealized U curve or simple family ranking. No reserved score was computed.

Exact coordinate/duplicate histories, dropout enumeration and factor/criterion examples were checked. A final bounded arithmetic call independently confirmed the changed-sign shrinkage, nonuniform-mask loss, changed AIC/BIC and translated smoothness solutions. The author read corrected a subtle KKT statement: strict subthreshold association forces zero, while equality alone permits zero or nonzero. It also made the denoising objective's local unaveraged convention explicit and added physical baseline/changed inference values to the prose. No broad refitting campaign was needed.

Both complete main teaching programs have setup and honest execution boundaries. Equivalent author calculations ran; separately assembled verbatim program execution, source integration, browser state/rendering/performance and formal independent implementation review remain phase two. This is an author accuracy/learning-experience check, not a substitute for that review. No known material writing gap remains.

## Next action and retention

Root may reconcile this packet as content prepared and link its AIC/BIC/MDL home from CV and Bias–Variance. Keep implementation pending. Phase two implements the manuscript and topic-specific figures/investigations, verifies the displayed programs and arbitrary edited cases, then completes route-local loading, accessibility and formal review. Preserve the seven required packet files and the open destination note; no disposable scratch/cache was created. Next content topic is Feature Selection & Importance.

**Done, 14 September 2026.** Phase two was carried out and independently reviewed; the next action is no longer "keep
implementation pending". The seven packet files are preserved; this design record is the only one of them that changed,
because phase two appends to it by design, so its hash in the ledger's content checkpoint needs refreshing. What remains
outside this record: the ledger entry, the generated inventory and the handoff.

## Phase two: implementation — 14 September 2026

The manuscript is implemented as the rewritten `src/learn/data/topics/regularization-l1-l2-elastic-net-dropout.jsx`:
eleven sections, eight inline figures, four investigations, three executed Python programs and practices 1 to 10.
Every claim, table row, program, caution and practice task in `lesson.md` survives to the page, and the declared
first-pass route (sections 1–6, practice 1–6) is stated at the top.

### What was built

| File | What it holds |
| --- | --- |
| `src/learn/data/regularization-models.js` | The pure models. Soft-thresholding and the scalar solution with its threshold, divisor and one-sided slopes; the mixed penalty measure and its exact level-set radius in closed form; the two-coordinate solution with the budget it attains and the data-loss contour through it; a full coordinate-descent fit with centring, an unpenalized intercept, a per-coordinate trace, a KKT stopping rule and an honest unconverged result; the duplicate-column analysis; exact dropout mask enumeration with its analytic twin; ridge and early-stopping filters; the factor optimum; a 3×3 solver and the generalized-Tikhonov comparison; AIC/BIC; the declared sixteen-bit code; and the saved airfoil model's polynomial expansion and signed contributions. Every entry point refuses non-finite input, a negative strength, a mixing fraction outside [0, 1], a ragged design or an out-of-range measurement rather than substituting a default. |
| `src/learn/data/regularization-data.js` | Generated by `scripts/verify-regularization-data.py`. Provenance with the licence, the SHA-256 and the protocol; the twenty term names; eighteen candidates with their three fold MSEs and nonzero counts; the per-fold coefficient paths; both baselines; the three selected refits; the saved ridge model; and the one development-row inference fixture. `reservedPredictionsComputed: false` is part of the module. |
| `src/learn/data/regularization-examples.js` | Generated by `scripts/verify-regularization-examples.py --write`: the three displayed programs with the output an actual run produced. |
| `src/learn/components/lesson-labs/RegularizationShared.jsx` | Draft/commit investigation state, a number field that publishes only valid values, the prediction block with its optional second numeric commitment, a framed plot and signed contribution bars with an explicit exact-zero mark. |
| `src/learn/components/lesson-labs/RegularizationLabs.jsx` | The four investigations. |
| `src/learn/components/lesson-labs/RegularizationFigures.jsx` | Figures 1 to 8. |
| `src/learn/components/lesson-labs/regularization-labs.css` | The stylesheet, every class prefixed `rg-`. |
| `public/learn-assets/regularization/` | The unchanged 59,984-byte tab-separated `.dat` and an attribution file. |
| `src/learn/data/curriculum/blueprints/…` | The blueprint. Registration in `blueprints/index.js` is the parent's. |

The investigations follow the visual contract: I1 the scalar threshold with its objective curve, its z-to-coefficient map
and the threshold band; I2 the coordinate fit over editable rows, both features, the targets, λ, ρ and the coordinate
order, with the partial residual after every visit and an objective-by-sweep history from the actual computed trace;
I3 the five physical airfoil measurements traced to twenty polynomial terms and twenty signed decibel contributions;
I4 the exact four-branch mask tree with its probability strips, discrete output distribution and the analytic penalty
shown independently. Each starts with nothing selected, grades the prediction against the inputs committed with it,
and retires both the prediction and its feedback when any relevant input changes.

### Departures from the manuscript, and why

1. **The coordinate program's third printed line.** The manuscript recorded `elastic_net [1.666667 0.] 0.0 1` before the
   standalone program had been executed. NumPy actually pads that array to `[1.666667 0.      ]`. The executed output is
   published; the values are identical, and the lesson says the padding is NumPy's own array formatting. The manuscript's
   sentence that the displayed program "awaits phase-two execution" is dropped, because it has now been executed.
2. **One comment in the airfoil program.** A single comment at the `np.loadtxt` call names the tab-separated 1,503 × 6
   layout, so the separator is stated where the program reads it. Nothing else in the displayed code changed.
3. **The dropout enumeration's filename.** The manuscript gives no filename; it is published as `dropout_masks.py`.
4. **A checkpoint after the four-row example.** The manuscript's two investigation-2 contrasts (row 0's target 3.4 → 7.4,
   and adding seven to every target) are also posed as a checkpoint question, because the contrast between them is the
   point of that section. No new claim is introduced; both cases are presets in the investigation.
5. **Section 6's closing paragraph is a callout.** Same words, given the visual weight of a stated-once caution.
6. **One formula split further.** The dropout expectation is set as three lines rather than two so it does not overflow a
   320 px screen. Measured, not guessed.
7. **Two additions to the closing material.** Section 11 gains a readiness-check table mapping each skill to where it was
   taught, and a closing paragraph after the Sources block naming which results are constructed calculations and which are
   calculations on the identified real dataset. Neither is in the manuscript; both restate what the manuscript already
   establishes rather than adding a claim.
8. **One sentence's subject changed.** The manuscript's "The equivalent full calculation ran serially in the author
   environment with no warnings" became "it ran with no warnings", said of the *displayed* program. That is a stronger
   statement than the manuscript made, and it is true: `verify-regularization-data.py` wraps the whole 60-fit refit in
   `warnings.catch_warnings(record=True)` and asserts the list is empty, and the independent reviewer's own run of the
   extracted program emitted nothing on stderr.
9. **Investigation 3's control bounds are wider than the observed columns.** The specification says "use observed column
   ranges as default control ranges". The controls instead accept a wider physical band (frequency 100–25,000 Hz against an
   observed 200–20,000, and similarly for the other four), because the point of the investigation is to change a
   measurement and watch six terms move, and a control pinned to the observed extremes cannot show that at the edges. The
   observed ranges are on the page twice — in a caption and as a table column — beside the standing caveat that an edited
   row is a model scenario. The displacement field's precision was 7 decimals, which could not accept the 9-decimal
   observed minimum it reports; it is now 9.
10. **The keep probability's floor was 0.05 where the contract says 0.1.** Now 0.1, matching the contract. The model's own
   guard, which refuses q ≤ 0 whatever the control allows, is unchanged.
11. **`rawFeatureNames` is exported by the generated data module and used by nothing.** Five short strings; it is the
   generator's record of the names the polynomial expansion was built from. Kept for that, not for the page.

Two departures were found by the independent review and **removed** rather than declared: a "Calculate without recording a
prediction" action in all four investigations, which contradicted a binding contract line, and figure 4's missing per-term
selector. Both are described under the disposition below.

### Checks actually run

| Check | Result |
| --- | --- |
| `node scripts/verify-regularization-models.mjs` | PASS — 113 grouped checks. Every four-row, duplicate, dropout, criteria, factor, smoothness and early-stopping value matched against `calculated-inputs.json`; independent identities (a scanned scalar optimum, recomputed KKT conditions and a local objective grid over eight designs, a scanned two-factor surface, a sampled penalty level set, a scanned constrained contact point, back-substitution into the denoising system, a reconstructed polynomial term order); refused input on every entry point; and the generated data module checked against the packet, including each published mean against its own folds. |
| `…python.exe scripts/verify-regularization-examples.py --write`, then again without `--write` | PASS — three programs executed, 59 oracle assertions, and the recorded output proven to match a fresh run. |
| `…python.exe scripts/verify-regularization-data.py` | PASS — the full 60-fit campaign refitted from the supplied `.dat` with no warnings, every fold score, coefficient vector, nonzero count, intercept, selected refit and the inference fixture matched against the packet, then the module written and the data file served. |
| `npx vite build --outDir dist-reg` | PASS. |
| `DIST_DIR=dist-reg … node scripts/verify-regularization-browser.cjs` | PASS — 13 cases, 25 screenshots, at 1366, 1024, 768, 390 and 320 px. |

Evidence: [model checks](../../evidence/regularization-models.json), [executed programs](../../evidence/regularization-native.json),
[the regenerated campaign](../../evidence/regularization-data.json) and [the browser pass with screenshots](../../evidence/regularization-browser.json).

### What the screenshots made us change

Every assertion passed before these were opened. Reading the images found eleven defects:

1. **Figure 1** — the in-plot legend ran past its own viewBox, and the two total annotations sat on top of the gold curve.
   The legend moved to the caption and the totals to a clear band at the top of the plot.
2. **Figure 2** — each panel's coordinate label overflowed the panel edge. The coordinates moved into the sentence beneath,
   which also names the markers.
3. **Figure 3** — two long in-figure sentences overflowed; both moved to HTML, and the vertical axis gained its `w₂` name.
4. **Figure 4** — the `15` value label and the `0.001` tick label collided, and the full-width coefficient panel magnified a
   340-unit viewBox to about 740 px, rendering 12 px type at roughly 26 px. The y ticks were re-chosen, the tick row moved
   down, every full-width diagram is now capped at 440 px, and the "unpenalized OLS" label moved off the data line.
5. **Figure 5** — the "v₂" arrow was a 12-unit stub beside a much longer grid line, so the grid read as the direction, and its
   label floated away from it. Redrawn as two equal-length unit directions paired with two data-sensitivity bars on one
   shared scale, with the explanation in HTML.
6. **Figure 7** — removing the legend left a large empty band above the plot; the plot was re-seated and each series' connector
   given its own colour so the three are distinguishable.
7. **Figure 8** — the stacked-bar legend overflowed the panel and overlapped both totals; it moved to the panel's caption.
8. **Investigation 1** — the objective plot's tick labels read −1.1, −0.35, 0.4, 1.15, 1.9, and the lowest value label sat on the
   row of tick labels. The span is now rounded to whole numbers and every value label is clamped inside the drawn box.
9. **Investigation 3** — the observed-range suffix on the five measurement controls could not shrink, and it pushed the
   document to 375 px at a 320 px viewport: the only horizontal overflow on the page. The ranges moved into a caption and the
   field label was allowed to wrap. Separately, an unchanged row printed "a change of −2.842 × 10⁻¹⁴ dB", because the browser
   recomputes the saved model rather than reading a stored answer; a difference inside the stated 10⁻⁹ dB agreement now prints
   as no change, and that tolerance is stated on the page.
10. **Investigation 4** — the mask tree was rendered full width, so its 12 px type became about 22 px, and the words
    "· impossible" were clipped at the SVG edge. The tree was redrawn inside the 440 px cap, with the probability printed at the
    right of each branch, a probability strip inside each lane, and a dashed outline as the zero-probability convention. The
    prose also said "the two zero-probability branches" where q = 1 leaves three; corrected.
11. **Everywhere** — printed numbers used a hyphen where the prose uses a typographic minus. Every number the labs and figures
    *format* through `round`, `fixed` or `signed` now uses U+2212. This claim was originally written as "every number the labs
    and figures print", which was false: three practice values went through `Number.prototype.toFixed` and one figure column
    printed a raw property. The independent review caught both (S4); they are fixed and the claim is now stated accurately.

A twelfth finding came from the document-overflow check rather than an image: one KaTeX display overflowed a 320 px screen
by two pixels and is now split across three lines.

### What is still not claimed

- No reserved airfoil row is predicted or scored, by the page, the displayed program or any verifier.
- The airfoil numbers are development comparison and selection records on one row-level split of one collection. They do not
  establish performance on a new airfoil or experimental run, and they are not a contest against the later learning-curves lesson.
- The lesson executes no PyTorch and makes no native dropout-network claim; the deep lesson owns that.
- The airfoil candidates are precomputed native fits. The browser reproduces their recorded outcomes; it does not refit them.
- Formal independent review, integration and user acceptance are separate from this implementation. Nothing is committed, and
  the phase ledger is the parent's to close.

## Disposition of the independent review — 14 September 2026

Reviewed in [REGULARIZATION-INDEPENDENT-REVIEW.md](../../REGULARIZATION-INDEPENDENT-REVIEW.md). The reviewer recomputed
368 constructed values and the whole airfoil campaign from first principles, executed all three displayed programs, and
checked every external URL; **nothing numeric disagreed**, so no numeric claim on the page changed. What follows is every
finding and what was done.

### Blocking

| # | Finding | Disposition |
| --- | --- | --- |
| **B1** | Two project records assert contradictory implementation states; the content checkpoint's design-record hash is stale. | **Partly fixed, partly not mine.** The parts of the record I own are corrected: this design record now carries a dated "Superseded" note under its phase-one header and a "Done" note under its phase-one next action, so neither still says implementation is pending; and the topic note no longer says "closed" — it says the destination content is implemented and verified, and that the central ledger, not the note, is the authoritative statement of delivery state. The ledger entry, the generated inventory, the handoff section and the refreshed design-record hash are the parent's, as the reviewer's fix (a) and (c) say. My edits to this file are now final. |
| **B2** | Figure 5's paragraph said the arrows are drawn to their singular values; they are not, and the figure's own legend and `aria-label` say so. | **Fixed.** The paragraph now says the two direction stems are drawn at the *same* length because both are unit vectors, and that the two bars beside them are each drawn to their own direction's singular value on one shared scale. The picture, the in-figure legend, the paragraph and the `aria-label` now say the same thing. Re-captured and inspected: `regularization-figure-5-desktop.png`, `regularization-figure-5-390.png`. |

### Should-fix

| # | Finding | Disposition |
| --- | --- | --- |
| **S1** | Every investigation offered a one-click way to skip the prediction, which the binding contract does not permit. | **Fixed by removal, the literal reading of the contract.** `explore` is gone from `useInvestigation` and the "Calculate without recording a prediction" button is gone from `Prediction`; there is now exactly one path to an answer in all four investigations, and a line beside the disabled button says the answer appears once a prediction is recorded and that Reset or an edit starts another attempt. The `graded` flag and the `is-plain` verdict style went with it. The browser verifier no longer uses the bypass to render the labs for its layout pass: it records a real prediction in each, and it now asserts that no button matching `/without recording a prediction/` exists anywhere on the page. This also closes **O10**, since there is no longer a path that leaves a selected radio beside an ungraded answer. |
| **S2** | Figure 4's coefficient panel dropped the specification's per-term selector and had one y label. | **Fixed.** A fourth control, "Follow one term", lists the twenty `featureNames`; the chosen path is drawn in gold at 2.4 stroke width while the other nineteen drop to 0.28 opacity, its row is highlighted in the full table, and the caption names its value at λ = 0.001 and at λ = 100. The vertical axis now carries five labels at ±extent, ±extent/2 and 0. Captured as `regularization-figure-4-followed-term-desktop.png` and checked. |
| **S3** | Two table cells that carry the teaching point were clipped at desktop width. | **Fixed, and generalised.** The coordinate-step cell is now "declared 0 · flat" with the explanation moved into the table caption, and F7's headings are "quantity / values / edges". The underlying cause was that every cell was `white-space: nowrap`, so a wide table truncated its last column at the scroll container's edge instead of shrinking; the last column of every `.rg-table-scroll` table may now wrap. Measured after the fix: no `.rg-table-scroll` on the page overflows its container at 1366 px, and the flat cell's right edge is 15 px inside its region. Re-captured `regularization-coordinate-constant-desktop.png` and `regularization-figure-7-desktop.png`. |
| **S4** | Three practice values and one figure column printed an ASCII hyphen, and repair note 11 claimed otherwise. | **Fixed.** Practice 1's three values go through the body's own `num()` helper; F8's log-likelihood and −2ℓ columns go through `round()`. The three range captions added for O2 were audited for the same defect and routed through `round()` too. Repair note 11 is corrected above and now says what is actually true. Verified in `regularization-figure-8-desktop.png`: the table prints −150 and −146 with the same minus sign as §9's table. |
| **S5** | Investigation 3's verdict read "a change of no change from the recorded row". | **Fixed.** The verdict now reads "…dB, unchanged from the recorded row" when the difference is inside the stated 10⁻⁹ dB agreement, and "…dB, a change of −0.836659 dB from the recorded row" otherwise. |
| **S6** | The departures list was materially incomplete. | **Fixed.** The list above goes from six entries to eleven, adding the two closing additions, the changed subject of the "no warnings" sentence, I3's wider control bounds and its precision fix, the keep-probability floor and the unused `rawFeatureNames` export. The two departures the review found and I removed rather than declared — the explore action and the missing term selector — are named at the end of that list and dispositioned here. |
| **S7** | The q ≠ 0.5 dropout state was asserted but never photographed, and its screenshot filename pointed at a different state. | **Fixed.** The capture now happens immediately after the practice fixture is committed and before the target-shift preset is clicked, so `regularization-dropout-practice-desktop.png` shows what it claims: q = 0.75, probabilities 0.0625 / 0.1875 / 0.1875 / 0.5625 with visibly different strips, and expected loss 1.166666667. The null state gets its own file, `regularization-dropout-target-null-desktop.png`. The assertion was also strengthened from "not all equal" to the exact four values. The narrow gaps are closed: the coordinate lab now has a 390 px capture, and so do figures 3, 5, 6, 7 and 8. The evidence set goes from 25 screenshots to 33. |

### Observations

| # | Observation | Disposition |
| --- | --- | --- |
| **O1** | The keep-probability floor was 0.05 against the contract's 0.1. | **Fixed**: now 0.1. |
| **O2** | Control ranges were reachable only by entering an invalid value. | **Fixed**: I1, I2 and I4 each gained a short standing line naming their control ranges; I4's also says why a keep probability of zero is refused. I3 already did this well and is unchanged. |
| **O3** | I3's control bounds are wider than the observed columns, and the displacement field could not accept the 9-decimal minimum it reports. | **Precision fixed** (7 → 9 decimals). **The wider band is kept and now declared** as departure 9: a control pinned to the observed extremes cannot show what the investigation exists to show, and both the observed ranges and the model-scenario caveat are on the page. |
| **O4** | F4's "unpenalized OLS" annotation read as one run with the "20" tick. | **Fixed by removing both in-plot comparator labels.** Moving the label had already put it on the data line once; both dashed horizontals are now named in the panel's caption with their values, which is unambiguous and cannot collide. |
| **O5** | F3's elastic-net marker was the same shape as the lasso minimizers. | **Fixed**: elastic net is now a diamond, and the caption says "the open square is ridge and the pale diamond elastic net, and both of those sit off the segment". |
| **O6** | `round()` strips trailing zeros, so readout decimals vary row to row. | **Not changed, deliberately.** The columns that must align already use `fixed()`; `round()` is for prose and readouts, where a trailing-zero pad reads as false precision. |
| **O7** | Surplus exports. | **Partly acted on.** `rawFeatureNames` is now declared in the departures list. The review's account of `selected` is not accurate for this module: the data module's `selected` export carries four fields per family (family, strength, selectionMse, nonzero) — twelve values in all, not three sets of twenty coefficients plus forty scaler values. The full coefficients ship once, in `ridgeModel`, which the page uses. Nothing was removed: `selected` is what the models verifier checks the selection rule against. |
| **O8** | No page-visible statement that the airfoil fits are version-sensitive. | **Fixed**: §5 now opens its results paragraph with "Every number in this section, and in the figure and investigation below it, is bit-exact in the pinned environment named above; another NumPy or scikit-learn version can move the last digits." |
| **O9** | The Zou–Hastie annotation calls the corrected rescaled estimate "historical". | **Not changed.** The sentence is the manuscript's, word for word, and the standing rule is that manuscript claims survive to the page; changing it here would put the lesson and the packet out of step. The reviewer judges the practical advice right and the disagreement one of emphasis. Recorded for the parent as a manuscript-level item. |
| **O10** | `explore()` left the recorded choice set. | **Fixed** by S1's removal of `explore`. |
| **O11** | F6's stated penalty for the same-prediction alternative was only in the `aria-label`. | **Fixed**: the panel's caption now prints both, "(1, 1) costs 0.5 in penalty and (2, 0.5) costs 1.0625", computed rather than typed. |
| **O12** | One table cell is the hard-coded string `5/3`. | **Not changed.** It is the manuscript's own cell, and the computed 1.666666667 appears in I1's family table and in F2's row. |
| **O13** | The blueprint's `sequence` has twelve entries for eleven sections. | **Not changed.** It is an idea sequence, not a section list; §7 legitimately contributes two entries. |
| **O14** | Reading-time estimates are unverifiable. | **Not changed.** They are estimates and are labelled as such in `readTime`; measuring them is not something this pass can do honestly. |

### Re-verification after the fixes

| Check | Result |
| --- | --- |
| `node scripts/verify-regularization-models.mjs` | PASS — 113 grouped checks, re-run after every source edit so its recorded hashes bind to the final bytes. |
| `…python.exe scripts/verify-regularization-examples.py --write`, then again without `--write` | PASS — 3 programs, 59 oracles. |
| `…python.exe scripts/verify-regularization-data.py` | PASS — 1,503 rows, 60 fits, no warnings. |
| `npx vite build --outDir dist-reg` | PASS. |
| `DIST_DIR=dist-reg … node scripts/verify-regularization-browser.cjs` | PASS — 13 cases, **33** screenshots. New assertions: no bypass button exists anywhere on the page; the four q = 0.75 branch probabilities are exactly 0.0625 / 0.1875 / 0.1875 / 0.5625; the followed term is named in the caption and marked in the full table. |

Screenshots opened and confirmed after the fixes: figure 5 at desktop and 390 px (the paragraph now matches the drawing),
figure 4 at desktop and with one term followed (the gold path, the dimmed nineteen, the five y labels, and no text on the
data line), figure 7 and the constant-column coordinate state (both formerly clipped cells now complete), figure 8 (both
minus signs now identical), the q = 0.75 dropout capture (non-uniform strips and probabilities) and the threshold lab
(one commit action, ranges stated, typographic minus).
