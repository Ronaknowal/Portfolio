# Imbalanced Learning — content design and continuation

Stable ID `imbalanced-learning-smote-cost-sensitive-learning`. Author `/root/classical_feature_content`,12 September2026. Current delivery is **research and writing only**. Manuscript, visual/investigation contracts, observed input, provenance and bounded calculations are the content packet. Implementation, displayed-program execution as a published artifact, independent phase-two review, browser/accessibility/performance checks and integration are pending. Root owns the central source-bound checkpoint.

## Scope, conservation and sequence

The topic inventory was run with `--topic imbalanced-learning-smote-cost-sensitive-learning --work content`. Its one destination note was read completely; there is no current individual blueprint, so this design supplies one. The entire854-line current original source was read, not only an introduction or generated outline:

`src/learn/data/topics/imbalanced-learning-smote-cost-sensitive-learning.jsx`

Original SHA-256 `a20c7ec96488e66709a6d89f90a590885b781b69819d2e646dbe9185e1d323b1`, source baseline commit `8c5da59f18516be77c29d5aeeafca3decca4f738`. The source remains unchanged. Publication before this rewrite is not approval of its claims.

Preserved substantive coverage: class rarity and examples; counts/precision/recall/F measures/AP/ROC; sampling versus weighting versus action thresholds; weighted loss and cost-sensitive decisions; random over/undersampling; SMOTE mechanism and working code; Borderline/ADASYN/categorical variants; cleaning and balanced ensembles; train-fold pipeline placement; focal loss; prior/probability interpretation; real-use scenarios; scaling/cost of training; independent exercises and references. Replaced invented or unsupported outputs with a fixed observed Yeast experiment. A small hand dataset remains for algebra/geometry rather than masquerading as measured generalization.

Proposed display title **Imbalanced Learning: SMOTE, Cost-Sensitive Learning & Rare-Event Decisions** makes the locally necessary decision/probability/queue material visible. This is the best home for those connections between sampling, objective and action; no extra catalogue topic or stable-ID change is needed. Phase two can adopt this display title while preserving all routes, module memberships and progress.

Actual sequence is Bias–Variance & Learning Curves → this topic → AutoML & NAS → HMM. The previous topic supplies variation across training sets and learning-curve interpretation. This topic locally refreshes logistic prediction and training/assessment ownership, then defines confusion counts, posterior/cost notation, weighted objective and SMOTE. It does not presuppose later Evaluation Metrics or Calibration. AutoML receives the complete-procedure objective and training-boundary contract. Later Loss Functions owns full network/focal implementation; here the origin, modulator, loss-mass example and product-rule correction are sufficient substantive preparation. Earlier Multi-Label supplies row/label identity; its incoming note is explicitly resolved in this packet. These scope boundaries avoid forcing a global prerequisite reorder.

## Learner hurdles and chosen treatment

| Hurdle | Local explanation and representation | Independent evidence of understanding |
| --- | --- | --- |
| An accurate classifier can do no useful detection |1,000-case count conservation, separate actual-class bands, always-negative baseline|practice1 constructs equal-accuracy/opposite-recall matrices|
| A threshold is an action on scores |editable named score queue, tied groups, exact precision/recall steps|practice2 handles an exact-budget tie differently|
| Rarity alone determines the right cutoff |two conditional action risks and crossing, cost common-factor null|practice3 derives a changed cutoff|
| Class weighting merely makes probabilities more correct |derive weighted expected-loss optimum/inverse; loss landscape and common-factor test|practice4 recovers original probability from another weighted.5|
| SMOTE manufactures trustworthy independent examples |interpolate an actual vector, show collision with opposite class, majority-move null and metric-dependent neighbor contrast|practice5 computes changed geometry; practice6 refuses unsupported full-label assignment|
| Validation can be balanced just like fitting |role/fold diagram and complete manual train-only program|real roles and threshold/inspection separation|
| One ranking defines the best procedure |actual cost/AP/top10 disagreement on Yeast|practice7 chooses a question and recognizes selection/uncertainty|
| More rows imply more evidence |identity repair, original/synthetic lineage and bounded row counts|practice9 counts unique evidence separately|
| Priors, feature shift and score error are one problem |derive prior-odds formula under fixed class-conditionals|practice8/10 state missing assumptions and new-sensor evaluation|

The first-pass route appears immediately after the introduction: §§1–7 and practice1–7. Advanced sampler families, focal, prior shift and resource analysis are §8, with practice8–10. Essential readiness does not depend on the advanced branches. Solutions/hints are closed by default and use changed inputs, not the same demonstration with a missing final sentence. Real applications include bounded protein follow-up, dense-detection background locations, deliberately enriched materials screening and constrained factory alert queues; each explains a mechanism rather than decorating the page with industry names.

## Accuracy repairs from the old source

- Accuracy is a legitimate equal-error-cost quantity; “meaningless” and universal “never optimize” advice were replaced by an explicit counterexample and task distinction. No arbitrary class-ratio threshold selects an algorithm. A100,000-row dataset at.1% has100 positives, not an unspecified handful.
- Higher empirical threshold can reduce precision. The incoming three-record counterexample, ties, no-selection denominators and top-k distinction are now explicit. ROC-AUC is not mechanically inflated by extra true negatives under fixed class-conditionals; AP is defined by grouped recall increments and distinguished from trapezoidal PR area. Fβ's squared coefficient is not a universal monetary-cost ratio.
- Equal-cost posterior cutoff.5 is valid even for rare events. The cost cutoff, full cost-matrix rule, calibrated-score versus full posterior distinction, weighted-loss optimum and prior-shift assumptions replace unconditional “always tune”/“weighted scores are probabilities” implications. Model fitting, calibration and threshold choice are distinct.
- Ordinary SMOTE uses minority neighbors and one vector-wide scalar interpolation fraction. Neither the synthetic label nor every multi-label component follows from geometry alone. Categorical variants have different semantics. The earlier claimed random/stdout matrices were not retained as evidence. Sampling inside training is required by the protocol; a manual implementation is valid and an ordinary sklearn pipeline does not silently skip samplers as a universal behavior.
- NearMiss variants' retention rules were corrected; cleaning can remove genuine overlap, and balanced counts need not survive cleaning/ADASYN integer allocation. Combining every lever is not inherently optimal. Balanced class weights after a balanced resampler may be all1.
- Weighted normalization and penalty are explicit. Weighting can avoid duplicate storage, but no same-training-time/zero-overhead guarantee is made. Focal loss's modulator also has a derivative; its hard examples can be noise. SMOTE changes class-conditionals, so class-prior correction alone is not a universal calibration repair.
- Complexity includes dimensions/minority-neighbor work and the actual balancing expansion2M/(M+m), not an inverse-prevalence memory explosion. No unmeasured browser/native timing curves survive. Budget allocation/ties use no inspection truth to choose candidates.

## Canonical coverage check

Canonical organizational reference: the full [imbalanced-learn User Guide section list](https://imbalanced-learn.org/stable/user_guide.html), inspected12 September2026. This was an explicit coverage check, not an inference from a search snippet. Main section dispositions:

| Canonical section and subsections | Disposition |
| --- | --- |
|1 Introduction/API/problem|core rarity, operation distinctions and optional package path; package developer conventions not core theory|
|2 Oversampling: naive, SMOTE/ADASYN, ill-posed cases, variants, formulation, multiclass|core duplication/SMOTE/geometry/labels; §8 adaptive variants and multiclass boundaries; current implementation semantics checked|
|3 Undersampling: generation, random/NearMiss, Tomek/ENN/repeatedENN/AllKNN/CondensedNN/OneSided/NeighborhoodCleaning, instance hardness|random/NearMiss/Tomek/ENN taught; iterative cleaning families are extensions of the stated rule, not a mandatory sampler-name catalogue; centroid generation and supervised hardness selection assessed but not needed for this lesson's mastery route|
|4 Combining over/undersampling|§8 generation then cleaning, final-count/overlap implications|
|5 Ensembles: bagging/forest/boosting|§8 mechanism and earlier ensemble bridge; no duplicate full tree/boost implementation|
|6 Miscellaneous: custom sampler/generators, TensorFlow/Keras|manual complete procedure supplied; framework batch-generator APIs defer to deep training/implementation use, no pedagogical omission of resampling ownership|
|7 Metrics: sensitivity/specificity, geometric/index-balanced accuracy, macro-MAE, classification report, value-difference metric|core binary measures/costs/AP/ROC; G-mean/index-balanced score and ordinal macro-MAE not substituted for the declared task; later Evaluation Metrics owns wider aggregation/ordinal choices; VDM noted in categorical reference, not an unexplained distance derivation|
|8 CV: instance hardness, AP/split variability|training-only/CV roles taught; current page inspected but found inconsistent hardness definition and invalid classifier import, so no copied recipe; task-appropriate group/time/stratified protocol remains local core|
|9 Pitfalls/leakage|full protected fitting/tuning/inspection/reserve program and duplicate-ID repair|
|10 Dataset utilities;11 developer guidelines;12 references|real licensed input/provenance and curated sources supplied; synthetic dataset/API development utilities are not missing learner concepts|

This map does not claim every implementation variant's full paper was read. Main substantive guide bodies were inspected as recorded below; unsupported/stale wording was investigated rather than repeated. No future learner must open all canonical subsections to understand the local first-pass procedure.

## Research record: substantive bodies actually inspected

All accessed12 September2026. Primary/official sources only support technical claims; annotations in the learner references offer complementary visual, mathematical and code approaches.

- [SMOTE paper](https://arxiv.org/pdf/1106.1813): actual §4.1/4.2 context, construction, full pseudocode/example and §4.3 combination; §6.1 SMOTE-NC algorithm/unfavorable Adult case and start of nominal extension. Not all37 pages/benchmarks read. Original pseudocode's placement of the random gap can be read coordinate-wise, whereas its text/current guide use a segment; the lesson explicitly chooses one scalar per vector and checks that implementation.
- [Oversampling guide](https://imbalanced-learn.org/stable/over_sampling.html): full substantive body on duplication/shrinkage, SMOTE/ADASYN, variants, mixed/all-categorical and formula. Its boundary-neighborhood phrasing is not copied as a precise theorem. [Undersampling guide](https://imbalanced-learn.org/stable/under_sampling.html): actual prototype/random/NearMiss/Tomek/ENN/iterative-cleaning sections through instance-hardness introduction. Selected deeper APIs, not every example, inspected. An apparent retained-set typo in the condensed-neighbor prose was not propagated.
- [Combined samplers](https://imbalanced-learn.org/stable/combine.html) and [ensembles](https://imbalanced-learn.org/stable/ensemble.html): full substantive bodies. [Metrics](https://imbalanced-learn.org/stable/metrics.html): sensitivity/specificity, geometric/index-balanced, macro-MAE, report and VDM portions. Stale sklearn-capability wording was not treated as current fact.
- [Model selection](https://imbalanced-learn.org/stable/model_selection.html): complete substantive page. It inconsistently defines hardness using true-class versus highest predicted-class probability and contains an invalid `sklearn.ensemble.LogisticRegressionClassifier` import. Reported to root; no manuscript/API dependency on that recipe. This is why canonical references inform coverage without dictating wording blindly.
- [Pitfalls](https://imbalanced-learn.org/stable/common_pitfalls.html): substantive leaky/corrected Adult patterns and nested comparison passages inspected; no claim its exact numerical improvements must occur in another dataset.
- [Visual oversampling comparison](https://imbalanced-learn.org/stable/auto_examples/over-sampling/plot_comparison_over_sampling.html): actual setup/code, point-cloud and decision-boundary examples for duplication/shrinkage, SMOTE/ADASYN and boundary/SVM/KMeans variants, mixed/all-categorical examples. Qualitative visual alternative; not a benchmark reproduced locally.
- [Decision threshold guide](https://scikit-learn.org/stable/modules/classification_threshold.html): actual statistical-versus-decision discussion, custom objective/CV tuning, prefit/fixed-threshold examples. Normative medical language does not define this lesson's hypothetical costs. [AP API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html): substantive formula/noninterpolation/parameters/returns/examples, not the navigation-only initial view.
- [Elkan cost-sensitive paper](https://cseweb.ucsd.edu/~elkan/rescale.pdf): §§1–3 cost accounting, threshold and class-prior rescaling proof; selected §4 learner-family/§5 discussion. Own clean derivations supply local math; no claim of universal finite-model threshold/weight equivalence.
- [Davis–Goadrich author-hosted report](https://mark.goadrich.com/articles/davisgoadrichpr.pdf): actual confusion/ROC/PR definitions, fixed-count relationship and §4 interpolation discussion. Legacy Wisconsin host failed; working author-hosted technical report was read. No invented all-threshold precision monotonicity.
- [Focal-loss paper](https://arxiv.org/pdf/1708.02002): actual §3.1 weightedCE, §3.2 focal definition/plots, §3.3 initialization and §3.4 sampling context. Detection benchmarks were not reproduced. The local loss totals and derivative were calculated independently.
- [UCI Yeast](https://archive.ics.uci.edu/dataset/110/yeast): actual source feature/label/license body and full original `yeast.names`; unchanged small data downloaded with hashes, not catalogue metadata alone. Provenance records authorship/dates and exact observed-data limitations.

No video was inspected, so none is endorsed on title/metadata. The full code/visual example and several mathematically distinct primary articles provide useful alternate learning routes without adding a video quota.

## Author calculations and learning-experience check

Final observed input:1,462 unique protein IDs from1,484 source rows after verifying/removing22 exact repeats;51 positives. This repair occurred after an initial five-fit calculation. Same fixed five methods were rerun once; ten fits total across that correction, five final retained. Full reason, unchanged settings, row roles, seeds, source identities and reserve status are in provenance. No reserved predictions in either pass. Final actual comparison has SMOTE lowest cost49, random oversampling highest AP.278679 and several top10 winners with2 positives. Original probabilities have smaller Brier than weighted/SMOTE scores; this is not renamed proof of calibration. No method was changed after that mixed outcome.

Bounded post-writing checks reused saved fitted scores, with zero new fits: reconstructed all five tuning choices and inspection count/cost results; checked four split sets cover1,462 distinct source IDs; preserved `reserve_scored:false`; recomputed exact hand cases, changed exercises and focal derivative by central difference; parsed all three displayed Python blocks; ran the displayed SMOTE helper on the same bounded input against the author routine. The optional imbalanced-learn program is not executed. These are author evidence for writing, not formal phase-two native/browser review.

I5's real tuning contrast is actually checked: original threshold.5 hasTP1/FP0/FN6/TN193,cost72; threshold.15 hasTP5/FP4/FN2/TN189,cost28. Precision falls despite better recall/cost. The weights/common-cost-factor and majority-move SMOTE nulls are mathematical consequences of the specified input controls. Neighbor-switch geometry is constructed with a fixed metric and actual coordinates; the spec does not secretly refit a scaler in the promised null mode.

Complete manuscript and specification were inspected for continuity, changed practice, formula/units alignment, actual-role labels and reference routes before handoff. Prose number spacing and an incorrect earlier random-forest route were corrected. Essential concepts remain visible; repeated calibration/leakage cautions have a home, rather than appearing after every result. Five investigations have distinct unsolved inputs, recorded predictions and meaningful consequences; six inline figures teach separate mechanisms. Those counts result from the topic's hurdles, not a quota. No lab is a mandatory ceremonial button to reveal a predetermined paragraph.

## Next action

Root reconciles and source-binds the complete content packet and note disposition. No unresolved material writing gap is known. On a later authorized finish request, run the inventory with `--work finish`, consume the full packet, implement the specific figures/investigations and public download with attribution, execute displayed programs, perform independent correctness/learning-experience review and affected browser/accessibility/performance checks, then integrate. Do not mark implementation complete from this packet or score the reserve simply because it exists.

No disposable source archives, images, browser captures or scratch scripts were created. The small dataset, author script and calculated inputs are necessary pending implementation inputs and must be retained. Shared tools and other authors' files were not changed.

## Phase two: implementation record (phase A)

Appended by the implementation phase. The content packet above &mdash; manuscript, visual and
investigation contracts, `author-calculations.py`, `calculated-inputs.json`, `data-provenance.md` and
`yeast.data` &mdash; was consumed unchanged. Nothing in the sections above this heading was edited.
This section records what was built, every departure from the specification with its reason, and
what remains for the browser phase.

### The published lesson replaced a live page

The previous body of `src/learn/data/topics/imbalanced-learning-smote-cost-sensitive-learning.jsx`
was 855 lines, 63,689 bytes, SHA-256 `a20c7ec96488e66709a6d89f90a590885b781b69819d2e646dbe9185e1d323b1`.
That matches the hash this design record already stated and matches `git show HEAD:` byte for byte,
so the file overwritten was the published one and nothing uncommitted was lost. It is preserved at
`scratch/imbalance-preserve/imbalanced-learning-smote-cost-sensitive-learning.ORIGINAL.jsx`.

### Files created

| File | Lines | What it is |
| --- | ---: | --- |
| `src/learn/data/topics/imbalanced-learning-smote-cost-sensitive-learning.jsx` | 411 | The lesson: sections 1&ndash;10, six inline figures, five investigations, three displayed programs, ten practice tasks, one checkpoint, sources |
| `src/learn/data/imbalance-models.js` | 856 | Every quantity a figure or investigation draws, with guards that refuse input rather than substituting a default |
| `src/learn/data/imbalance-data.js` | 249 | Generated: provenance, the four roles, the five procedures' saved tuning and inspection scores, thresholds, metrics, top-ten identities, synthetic lineage |
| `src/learn/data/imbalance-examples.js` | 33 | Generated: the three displayed programs, extracted verbatim from the manuscript and executed |
| `src/learn/components/lesson-labs/ImbalanceShared.jsx` | 415 | The commitment contract, controls, tables, plots, undefined-value formatting |
| `src/learn/components/lesson-labs/ImbalanceLabs.jsx` | 934 | Investigations 1&ndash;5 |
| `src/learn/components/lesson-labs/ImbalanceFigures.jsx` | 530 | Figures 1&ndash;6 |
| `src/learn/components/lesson-labs/imbalance-labs.css` | 293 | Styles |
| `src/learn/data/curriculum/blueprints/imbalanced-learning-smote-cost-sensitive-learning.js` | 101 | Blueprint: 15 outcomes, 20 sequence steps, 29 misconceptions, 12 sources |
| `public/learn-assets/imbalanced-learning/yeast.data` | 1,485 | The unchanged 94,976-byte source member, SHA-256 verified |
| `public/learn-assets/imbalanced-learning/ATTRIBUTION.txt` | 17 | Citation, DOI, licence, hash, and what the analysis does and does not do to the data |
| `scripts/verify-imbalance-data.py` | &mdash; | Regenerates and verifies the data module; re-derives the packet's trust root |
| `scripts/verify-imbalance-examples.py` | &mdash; | Extracts the displayed programs verbatim from the manuscript and executes them |
| `scripts/verify-imbalance-models.mjs` | &mdash; | Independent recomputation of every number the lesson states |

### Verifier results

All three phase-A verifiers pass from a clean checkout and write evidence JSON.

* `scripts/verify-imbalance-data.py` &mdash; **1,442 checks**, including **28 re-derivations of the
  packet's own trust root**. Evidence: `docs/teaching/evidence/imbalance-data.json`.
* `scripts/verify-imbalance-examples.py` &mdash; **3 programs executed, 58 oracle assertions**.
  Evidence: `docs/teaching/evidence/imbalance-native.json`.
* `scripts/verify-imbalance-models.mjs` &mdash; **386 grouped checks across 58 groups**, including
  1,000 swept threshold candidates, 168 weighted-optimum grid points, all 24 display permutations of
  the tied queue and a 24-position majority-move null. Evidence:
  `docs/teaching/evidence/imbalance-models.json`.

The packet's whole 439&nbsp;KB `calculated-inputs.json` was independently reproduced before any code
was written: rerunning `author-calculations.py` in this environment gave **zero differences** at a
1e&minus;12 tolerance across every split identity, synthesis fraction, fitted score, threshold and
metric. The data verifier then recomputes the same study through a *different* implementation &mdash;
first-occurrence identity repair by a plain scan rather than `np.unique`, hand-computed
standardisation, an explicit (distance, index) neighbour sort, and **damped Newton iteration on the
analytic Hessian instead of the author's L-BFGS-B call**. The two optimisers agree to
7.2&times;10&#8315;&#8311; on every coefficient of all five fits. Average precision, ROC-AUC, Brier and the
cost-optimal threshold are computed from their definitions and then cross-checked against
scikit-learn.

### Departures from the specification, and why

1. **The optional `imbalanced-learn` program is now executed.** The content phase deliberately left
   it unrun, and the manuscript says so. Phase two installed **imbalanced-learn 0.14.2** (with its
   dependency **sklearn-compat 0.1.6**) into `scratch/lesson-tools` and ran the program, so the page
   shows real output &mdash; three fold average precisions `0.49462759 0.36721427 0.37700713` and a
   mean of `0.41294966368063185`. Displaying a program with no output, or with invented output,
   were the only alternatives and both are worse. The manuscript's own sentence remains true as
   written (it describes the content phase), and the page's surrounding commentary now states that
   this implementation executed it. The examples verifier asserts that **no fold score coincides
   with any inspection AP in the five-model table**, which is the claim the manuscript makes about
   that program: it is a different protocol, not a reproduction.
2. **Displayed programs are extracted from the manuscript rather than transcribed.**
   `verify-imbalance-examples.py` reads the three ```python fences out of `lesson.md` and pins each
   by the SHA-256 of its fence body (`770e9d29&hellip;`, `fd0bd843&hellip;`, `675b409c&hellip;`). The
   manuscript is stored CRLF and the hashes are over the text-mode (LF) body; the model verifier
   normalises line endings before checking that the published code appears verbatim in the
   manuscript. A changed manuscript fails loudly and must be re-pinned deliberately.
3. **The three programs run as real separate processes**, not `exec` in a shared namespace, because
   the manuscript tells a learner to save three files that import one another and
   `yeast_smote_cv.py` imports `read_study_data` from `yeast_imbalance.py`. Running them any other
   way would not test the instruction actually given.
4. **A retired verdict is kept as labelled history.** The shared contract in the earlier lessons
   erases a verdict when an input changes. Several investigations here turn on an answer *not*
   moving &mdash; the majority-move null, the common-cost-factor null, the doubled-weights null &mdash;
   and erasing the previous result destroys the comparison the null is about. `useInvestigation`
   therefore keeps the last retired result and the `Prediction` component renders it under an
   explicit "a previous attempt, no longer current" heading, with a sentence saying the inputs have
   changed and a new prediction is required. Nothing retired is ever presented as current, and the
   grading still uses only the committed draft.
5. **A `role` badge travels with investigation 5's state.** The spec requires that activating
   inspection exploration changes the claim being made. The badge is part of the investigation
   header and switches between "Role: tuning" and an exploratory warning, rather than being a static
   sentence that would keep saying "untouched" after the learner had touched it.
6. **Investigation 3's doubled-weights null is shown with printed loss values, not a pinned axis.**
   The spec allows either "the same y-axis scale or directly labeled loss values". A pinned axis is
   impossible across the full control range (the loss at the optimum varies by orders of magnitude
   over w &isin; [0.1, 20]), and clipping would misrepresent the curve. The reveal therefore prints the
   loss at the current optimum *and* the loss at the previous applied optimum side by side, so the
   doubling is read as two numbers rather than inferred from a picture whose axis moved. The axis
   maximum is stated in the caption.
7. **Investigation 5's candidate table is sampled for display, not for computation.** The sweep runs
   over all 201 candidates; the rendered table shows the near-optimal ones plus a regular sample of
   the rest, and its footnote says so and gives the full candidate count. Rendering 201 rows of a
   scrollable table on a phone was the alternative.
8. **Figure 2's ROC marker is drawn on a zoomed axis** running 0 to 0.05 in the false-positive rate,
   with the zoom named in the axis label. At full scale a rate of 0.01 is a pixel from the axis. The
   spec explicitly permits "a zoomed labeled ROC inset".
9. **Figure 6 uses two panels with their own stated scales** rather than one shared axis or a log
   axis. Under cross-entropy the difficult group is 1.5% of the loss mass; on one axis its bar
   disappears. Each panel prints its own total in its heading and its bars are shares of that total.
   The spec permits "separate count and per-example-loss rulers" for exactly this reason.
10. **The SMOTE investigation refuses an unconstructible cloud in place.** Reclassifying every point
    as majority is reachable through the class controls and the model correctly raises. Rather than
    letting that become a blank page, the lab catches the refusal, explains it as the algorithm's own
    contract, keeps every other edit, and hides the prediction gate until the cloud is constructible
    again.
11. **Display title.** The proposed expansion, *Imbalanced Learning: SMOTE, Cost-Sensitive Learning &
    Rare-Event Decisions*, is adopted as the page title. Routes, stable ID and module membership are
    untouched.

### Two defects found and fixed during implementation

* **A render-time crash that would have blanked the page.** The topic module imports `Math` from
  `components/content/Math.jsx`, which shadows the global `Math` inside that module. Two
  module-scope calls to `Math.max` and `Math.min` therefore resolved to the KaTeX component and threw
  on first paint. This was caught by a server-side render smoke test, before any browser was opened.
  Both call sites now use explicit `largest`/`smallest` reducers, with a comment naming the hazard.
  Other topic files that import `Math` are worth checking for the same pattern; that is a note for
  the integration owner, not a change made here.
* **A stylesheet rule that would have rendered diagrams at 16&nbsp;px.** The SVG text rule is scoped at
  the lesson root, `.imb-lesson svg text`, and uses `font-size`/`font-family` longhands. The two most
  recent lessons scope the equivalent rule at figure classes (`.ad-plot svg text, .ad-figure svg
  text`) and use the `font` shorthand; both are the exact failure modes recorded in the handoff, and
  a diagram inside an investigation stage wrapper would have missed those selectors entirely.

### Self-checks run in phase A

* **KaTeX.** All **139** `Math`/`MathBlock` expressions on the page were parsed with the repo's own
  KaTeX at the correct display mode. All parse. **None uses an accent or a wide brace**
  (`\widehat`, `\overbrace`, &hellip;), so no expression can inherit an oversized accent SVG that sets
  a minimum width. Long displays are split into `\begin{gathered}` stacks whose longest stripped line
  is 37 characters. Actual 320&nbsp;px overflow remains a browser-phase measurement.
* **Escape auditor.** A scan of this lesson's own files for control characters, TeX commands that
  lost a backslash, and literal `\n` in strings reports clean. Two heuristic false positives
  (`\nabla`, and a lesson-slug URL) were tightened out of the auditor rather than ignored. All files
  containing backslashes were written with the Write/Edit tools; one `node -e` and one bash heredoc
  did eat backslashes during development and both were caught and rewritten.
* **Server-side render.** The whole lesson and each of the eleven figures and investigations render
  through `react-dom/server` without throwing. The rendered first paint contains **five prediction
  gates, six figures, zero verdicts and zero revealed bins**, which is the "nothing revealed on first
  paint" contract checked mechanically. No reserved identity or score list appears; the page does
  state that the reserve is unscored.
* **Production build.** `npx vite build` exits 0 and emits the lesson as its own 233&nbsp;KB chunk with
  a 12.6&nbsp;KB stylesheet. The build directory was deleted afterwards.

### What the investigations actually control

| | Learner-owned inputs | Recorded prediction | Entity-level control | Fixture or null |
| --- | --- | --- | --- | --- |
| I1 score queue | every score, every truth, add/remove records, both gates | direction of precision | named records A&hellip;L, bins by identity | tied-group null over all 24 display permutations; above-all-scores undefined state |
| I2 action costs | posterior, both costs | which action is cheaper | the single case | common-factor null; both-costs-zero has no cutoff |
| I3 weighted score | probability or weighted score, both weights, mode | where the optimum sits | the population parameters | equal-weights contrast; doubled-weights null |
| I4 SMOTE geometry | every coordinate, every class, add/remove points, anchor, k, rank, fraction, both divisors | collision, majority-move or neighbour switch | named points A&hellip;N | majority-move null over 24 positions; k-invalidation refusal; zero-length segment |
| I5 tuning queue | procedure, trial gate, both costs, exploratory mode | direction of realised cost | 200 real proteins by source ID and protein ID | cost-scaling null; ranking-invariance null; discrete-candidate null where the gate does not move |

Every verdict is graded against the committed draft, not live state, and every relevant edit retires
it. No investigation refits a classifier in the browser.

### Still pending

Browser verification (`scripts/verify-imbalance-browser.cjs`) is deliberately not attempted in phase
A, since the blueprint is registered between phases and evidence must be captured against a
registered page. Pending for phase C: a production build previewed at 127.0.0.1:4188; real behaviour
cases at 1366, 390 and 320&nbsp;px; screenshots with unique paths and unique content, **opened and
looked at**; KaTeX display-width measurement on the real page; a curve-through-label sweep using the
`sampleCurvesThroughLabels` routine copied from `scripts/verify-bias-variance-browser.cjs`; and
keyboard and screen-reader operation of every control.

### For the integration owner

* `imbalanced-learn` 0.14.2 and `sklearn-compat` 0.1.6 were installed into `scratch/lesson-tools`.
* No shared component was changed and nothing outside the owned file list was touched.
* The `Math`-shadowing hazard described above is a pattern worth grepping for across other topics.

## Phase two: browser verification record (phase C)

Appended after the blueprint was registered. `scripts/verify-imbalance-browser.cjs` runs against a
production build previewed at `127.0.0.1:4188` and writes
`docs/teaching/evidence/imbalance-browser.json`.

**Result: 20 cases, 46 screenshots, every one with a distinct path and a distinct content hash.**
Checked at 1366, 1024, 768, 390 and 320 px. Microsoft Edge 153.0.4234.32.

### What the verifier covers

Structure and metadata; all three displayed programs present verbatim with their real output; every
recorded outcome row; the served dataset fetched over HTTP and matched byte-for-byte against its
recorded SHA-256, with its attribution; all five investigations including every declared contrast and
every exact null; the grading contract off its happy path; six figures' required content; plotted
coordinates read back from two figures and tested against the model layer; a five-width sweep for
label collisions, curve-through-label crossings and rendered type size; the loading closure; narrow
layout, formula width, table stacking and caption placement; keyboard reach and control bounds;
completion and sequence; and two load-failure recoveries.

`sampleCurvesThroughLabels` was copied into this verifier from
`scripts/verify-bias-variance-browser.cjs`, with this lesson's grid and backplate class names. It
earned its place immediately: see below.

### Defects the verifier found

1. **Investigation 1's reveal described a different comparison from the one it graded.** The verdict
   grades the move from the applied start gate to the applied target gate; the recall sentence beside
   it compared the *previously applied* target with the current one. Fixed to use the same pair.
2. **Label overflow in four figures.** A bar's value label pushed off the viewBox when its bar nearly
   filled the track (figures 5 and 6); three pipeline captions and the ROC axis caption were longer
   than 340 units. Figure 5 now reserves a right-hand label column, figure 6 moves a long bar's value
   inside the bar, and the long captions were split or shortened.
3. **Five curve-through-label crossings that the shared straight-line inspector cannot see.** Both
   risk lines ran through each other's value labels; the loss curve ran along its own clipping note;
   a pipeline connector travelled **34% of its own length** inside the label naming it, and another
   crossed `choose t`. Fixed by removing two redundant in-plot value labels (both risks are printed
   exactly in the strip above), rerouting the connectors down a clear gutter, and placing the two
   remaining annotations by a **corner-clearance search** rather than a fixed offset — the shape of
   both plots changes with the learner's own inputs, so no fixed position can be safe.
4. **The SMOTE segment ran through a point label,** because the default cloud's majority point M
   lies exactly on the segment and its label sat beside it. Point labels are now offset
   *perpendicular to the segment*, on the side the point already lies. The generated point's label
   takes the opposite side, because when G lands on an observed point — the whole purpose of the
   default setup — sharing a side puts the two labels on each other.
5. **Duplicate axis tick labels overlapped in the scatter's corner:** the lowest vertical tick and
   the first horizontal tick are the same number in the same place. The horizontal row keeps it.
6. **SVG type rendered at 7.61 px at 320 px and 8.86 px at 390 px.** A 340-unit viewBox inside a
   padded panel inside a padded card renders at about 0.76 on a phone. Fixed at the cause: the panel
   and the investigation give their inline padding back below 560 px, and both type tiers step up
   there. Smallest rendered primary label is now **10.00 px at 320 px** and 11.61 px at 1366 px;
   smallest secondary caption 8.53 px.
7. **Three KaTeX display formulas overflowed 320 px on the real page** — the precision/recall pair at
   315 px, F-beta at 287 px and the prevalence identity at 298 px, against 280 px available. None
   involved an accent, so this was ordinary width: each was restructured into more `gathered` lines,
   putting the wide fraction on a line of its own.

### Defects found only by opening the screenshots and looking

This is the step that matters, and it produced the worst defect of the phase.

1. **Figure 1's class bands were completely invisible.** All four coloured segments and their
   in-band count labels were painted over by the outline rectangle. The cause: that rectangle carried
   a `fill="none"` **presentation attribute**, which loses to this stylesheet's own `.imb-lane { fill }`
   rule, so the "outline" was a solid opaque box drawn last. Every assertion passed throughout —
   the band coordinates were right, the counts were right, the aria description was right, the
   accompanying table was right. Only the picture was wrong, and the picture is the figure. Fixed
   with an `.imb-lane.is-outline { fill: none }` class. A new check now rejects **any** shape in the
   lesson whose `fill="none"` attribute computes to a solid fill, and asserts that the four band
   parts paint four distinct colours, that the outline is unfilled, and that each in-band count label
   is present.
2. **Investigation 3 printed "the formula gives —".** `exactMinimum` lives on the loss-curve result,
   not on the optimum result, so an `undefined` rendered as the em dash the number formatter returns
   for a missing value — which reads as a formatted number rather than as a hole. A new check
   rejects any revealed text of the form "gives/is/are/equals —".
3. **Figure 5's top-ten identity column was clipped,** because only a table's last column wraps and
   that column sat second-to-last. The identity list is now last, and the counts column precedes it.
4. **Three metric columns dropped trailing zeros** (`0.19355` beside `0.273317`), so a column of
   numbers that must line up did not. They now use the fixed-decimal formatter.
5. **Two table headers truncated**: figure 5's "cost FP + 12 FN" became "realised cost", and figure
   6's eight long headers were shortened with the full names moved into the caption, which was
   clipping the focal-total column the figure exists to show.
6. **A slider and the number field that types the same quantity landed on different grid rows**, so
   the pair read as two unrelated controls. They now share one grid cell.
7. **Editable entities did not align to rows.** A four-column auto-fit grid packed a point's three
   controls across a row boundary, so "A — x, A — y, A — class, B — x" read as one line. The row
   group now takes its column count from the entity it edits: two for a record, three for a point.
8. **Investigation 5 printed a 15-digit threshold** and worded a cost tie as though both tied
   candidates had won. Now six places, with an explicit note that the decision uses the saved score
   exactly and never the rounded display, and the tie sentence names the candidates and then says
   which one is selected.

### Departures added in phase C

* **Two in-plot value labels were removed** from the cost investigation rather than repositioned.
  Both risks are printed exactly, to six places, in the strip directly above the plot; any label at
  the case posterior sits precisely where the two lines are closest to each other. The specification
  asks for the exact numerical risks to be retained rather than relying on pixel separation, which
  this does.
* **In-plot annotations are positioned by a clearance search**, not by an offset from the point they
  describe. With a false-alarm cost of 1 against a missed-positive cost of 12 the selecting line runs
  along the bottom of the box for its whole width, so "just above the axis" is the worst possible
  place; with the costs reversed it is the best. A fixed rule cannot be correct for inputs the
  learner chooses.
* **Type sizes step up below 560 px** and two levels of inline padding are returned to the drawing.
  This is a change to how the figures render on a phone, not to what they claim.
* The manuscript's formulas are **regrouped, not reworded**: the same expressions, broken across more
  lines so each line fits 320 px.

### What could not be fixed, and why

* **The cost crossing is small when the two costs are far apart.** At 1 against 12 on a shared
  vertical axis the crossing sits near the bottom-left corner and the two case markers nearly
  coincide. This is a property of the quantities, not of the drawing: a shared scale is what makes
  "multiplying both costs by three leaves the crossing where it is" visible at all. The
  specification's remedy is the one applied — annotate the crossing and retain the exact numerical
  risks rather than relying on pixel separation — and both risks, the cutoff as a fraction and as a
  decimal, and the chosen action are printed above the plot.
* **Investigation 5's candidate table is long.** 201 candidates are swept; the rendered table shows
  the near-optimal ones plus a regular sample, inside a scroll region, with a footnote giving the
  full candidate count and the minimum. Rendering all 201 rows on a phone would be worse.
* **The pipeline diagram has no arrowheads.** Direction is carried by reading order and by the
  captions. Adding markers risked reintroducing the label collisions just removed, for a gain the
  caption already delivers.

### Final state

All four verifiers pass from a clean checkout:

| Verifier | Result |
| --- | --- |
| `verify-imbalance-data.py` | 1,442 checks, 28 trust-root re-derivations, module byte-identical |
| `verify-imbalance-examples.py` | 3 programs executed, 58 oracles |
| `verify-imbalance-models.mjs` | 386 grouped checks across 58 groups |
| `verify-imbalance-browser.cjs` | 20 cases, 46 screenshots, 5 widths |

`npx vite build` exits 0. The server-side render smoke test still reports five prediction gates, six
figures and zero verdicts on first paint. All 139 KaTeX expressions parse, none uses an accent, and
none overflows 320 px on the real page. The build directory was deleted and the preview stopped.

## Disposition of the independent review

The [independent review](../../IMBALANCED-LEARNING-INDEPENDENT-REVIEW.md) ran roughly 33.3 million
independent constructed checks importing none of this lesson's code: 25,351,101 exact-rational
evaluations of investigation 2's decision rule over its complete enterable grid and 7,920,000 of
investigation 3's, a from-scratch refit of all five weighted logistic procedures polished to 50
decimal digits with `mpmath`, all 1,000 threshold-sweep candidates across 8 recorded fields, both
200-value score arrays per procedure, 73 exact-rational checks of every constructed number, and a
by-hand parse and identity repair of the served bytes. It found **no disagreement with any published
number** anywhere — manuscript, lesson body, figures, practice solutions, generated module or trust
root — and established, rather than assumed, that the 7.2e-07 optimiser gap moves not one digit the
page prints. It raised four blocking findings, four to fix and nine observations.

| Finding | What was wrong | Resolution |
|---|---|---|
| B1 | A preset labelled "Exact null: double both costs" graded the *total declared cost*, which doubling genuinely raises — so a learner reasoning correctly from the common-factor argument was marked wrong, and the browser verifier encoded the contradiction and passed | Investigation 5 now carries the question the null is actually about. A "Quantity to predict" control chooses between the total cost at a trial gate and **the gate the cost sweep selects**; the doubling preset asks the second, where the answer is genuinely "the selected gate does not move", and the reveal explains why every candidate's cost scaling leaves the argmin alone. A companion preset asks the same doubling about the total cost, where it correctly rises. The verifier assertion was rewritten to match |
| B2 | The retained verdict was rendered only while the next prediction was being recorded, so for the three investigations that grade an absolute property it printed the current answer above the radios — and it vanished at the moment the comparison it justified could be drawn | The retired verdict now appears **only beside the new one**, as a two-row comparison that says whether the answer moved. While the gate is open, a neutral held-attempt notice names no outcome at all. Where an investigation's question is itself an input, a `sameQuestion` predicate suppresses the comparison rather than putting answers to two different questions side by side |
| B3 | Investigation 1 printed "the calculation gives **—**, outside 0.000001" for an undefined precision — the exact mistake §2 exists to prevent — and the Phase C guard written to stop it could never fire | The graded value is passed through as `null` rather than `NaN`, and the verdict now reads "there is no number to compare it against here: nothing is selected at that gate, so TP + FP is zero and precision has no value. That is not a near miss, and it is not zero." The browser verifier now fills the numeric guess **against that operating point**, which is the combination that was never exercised. The guard's own regex was also repaired — see below |
| B4 | Figure 4's caption and in-diagram label claimed "no arrow reaches the reserved proteins at all" in a diagram with no arrowheads, where a connector terminates inside the reserved box, and the aria description said something different | The caption now says what is true and what the aria description already said: *no connector carries a sampler into an assessment branch, no connector leaves the reserved proteins into a transform or predict step, and the split itself does reach the reserve — that is how the reserve is created.* The in-diagram label reads "nothing leaves the reserve", the aria text was aligned, and the spine now docks at each role box's top edge instead of running into its middle |
| S1 | Practice 5's preset loaded two minority points, so the lab capped k at 1 while the solution quotes "at most two other minority observations" and three minority points | A third minority point C at (5, −1) was added — distance 5 from the anchor against B's √20, so B remains nearest and the generated point is still exactly (2, 2.5). The lab now caps k at 2 and refuses k = 5 with "Keep it between 1 and 2", the bound the solution quotes |
| S2 | The Phase A "Files created" line counts were stale by up to 112 lines after Phase C | The table is now marked as a Phase A snapshot and current counts are given below |
| S3 | The packet's `data-provenance.md` states EXC 37 where the served file has 35, and its ten counts sum to 1,486 against its own stated 1,484 | **Not edited — frozen packet text.** Recorded here and reported to the integration owner for routing. Nothing a learner sees is affected: `calculated-inputs.json` and the generated module both carry the correct 35, and the data verifier re-derives all ten counts from the served file |
| S4 | Investigation 4's two before-and-after questions graded "unchanged" when nothing had been moved | Apply is now blocked until the input the question is about has actually changed — a `requireChange` predicate over the majority coordinates or the scale divisors, since `pending` alone is satisfied by selecting the question itself — with a note saying which input to change |

### Two defects of my own, found while fixing these

Repairing B3 exposed something worse than the finding. The Phase C em-dash guard's regex had been
written through a shell heredoc that turned `\b` into a **literal backspace byte**, so it matched a
control character and could never fire; a second regex added during this pass was corrupted the same
way. Both are repaired, the file decodes as clean UTF-8 with no control bytes, and the escape auditor
now covers the four verifier scripts as well as the lesson's own sources, reporting the offending
code point and flagging invalid UTF-8. A check that cannot fail is worse than no check, and two of
mine could not.

The new null guard also caught a mislabel of my own: a preset named "Null: tie B and C at .8" is the
*setup* the reorder null runs from, not a null itself. It is now "Setup for the tie null".

### Observations acted on

Trust-root coverage (**O1**) was the most valuable. The review measured 47.9% of 18,239 scalar leaves
re-derived, with 9,000 of the 9,496 uncovered leaves in the unconsumed `tuning_sweep` block. The data
verifier now re-derives **every leaf**: all 1,000 sweep candidates across 8 fields each, the 400
stored role labels against the served file, both inspection count blocks' threshold/precision/recall
keys, the recorded objective and gradient infinity norm recomputed at the packet's own parameters,
the named inspection case identities, and the environment block. Coverage is **computed rather than
asserted by hand** and checked for exact equality, so a newly added block lowers it and fails rather
than passing unnoticed: `100.0% of the trust root's 18,239 scalar leaves`, with the check count rising
from 1,442 to 10,514.

Also acted on: the generated module's header now states that the published `parameters` are L-BFGS-B
stopping points established to roughly seven significant figures, not sixteen (**O2**); all four
blind spots the review named are closed with structural guards — a numeric guess exercised against a
null-valued answer, the history asserted absent-as-an-answer while a gate is open, every figure's
visible text checked against its own arrowhead count and its aria description, and **every preset
labelled "null" required to grade to the no-change branch**, with a dynamic check that no such preset
escapes the table (**O3**); figure 1's overflow label is anchored to the side its own segment sits on,
so "false alarm 18" no longer sits 300 px from the six-unit sliver it names (**O5**); figure 5's two
vertical rules start below the first group heading, removing the axis-through-label overruns
(**O4**); and investigation 1's card rail keeps stable record order until the reveal, so the
score-versus-gate partition is not encoded on a rail whose cards all still say "awaiting the gate"
(**O7**).

### Observations recorded and left

The faint ROC gridlines that pass behind the marker label in figure 2 (**O4**) are background at one
tenth the stroke weight of a data line and obscure no value; the label already carries an opaque
backplate, and moving it would put it further from the marker it names. The inconsistent part order
between figure 1's two bands (**O5**) is deliberate: each band reads in its own natural order —
detected before missed, false alarm before cleared — and forcing a shared order would put the
positive band's error on the left, where the negative band's cleared mass is. `repairCloud`'s silent
clamp of k when an edit lowers the minority count (**S1**, related note) is kept: the repaired value
is what the state strip displays, so nothing is hidden, and the alternative is refusing to render a
cloud the learner is midway through editing. **O6** and **O9** record things that were right and need
no action. **O8** — the topic note's closing sentence still asserting the original body is unchanged
— is outside this lesson's file list and is reported to the integration owner rather than edited
here.

### Current file sizes

The Phase A table above is a snapshot of that phase. Measured with `wc -l` after Phase C and this pass:

| File | Lines |
| --- | ---: |
| `src/learn/data/topics/imbalanced-learning-smote-cost-sensitive-learning.jsx` | 410 |
| `src/learn/data/imbalance-models.js` | 855 |
| `src/learn/data/imbalance-data.js` | 256 |
| `src/learn/components/lesson-labs/ImbalanceShared.jsx` | 461 |
| `src/learn/components/lesson-labs/ImbalanceLabs.jsx` | 1110 |
| `src/learn/components/lesson-labs/ImbalanceFigures.jsx` | 570 |
| `src/learn/components/lesson-labs/imbalance-labs.css` | 322 |
| `src/learn/data/curriculum/blueprints/imbalanced-learning-smote-cost-sensitive-learning.js` | 100 |

### Final verifier state

| Verifier | Result |
| --- | --- |
| `verify-imbalance-data.py` | 10,514 checks; 100.0% of the trust root's 18,239 scalar leaves re-derived; module byte-identical |
| `verify-imbalance-examples.py` | 3 programs executed, 58 oracles |
| `verify-imbalance-models.mjs` | 386 grouped checks across 58 groups |
| `verify-imbalance-browser.cjs` | 24 cases, 47 screenshots, 5 widths |

`npx vite build` exits 0; the server-side render smoke test still reports five prediction gates, six
figures and zero verdicts on first paint; all 139 KaTeX expressions parse with no accents and none
overflows 320 px; the escape auditor is clean across the lesson's sources and its four verifiers. The
build directory was deleted and the preview stopped.
