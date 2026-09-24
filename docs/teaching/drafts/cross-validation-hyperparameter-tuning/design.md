# Cross-validation and tuning: content design and continuation

## Current live-exploration contract — 21 September 2026

This dated UX amendment supersedes earlier prediction-entry, grading, commit-to-reveal and prediction-retirement requirements in this document. There is no learner prediction feature, even optional. Historical evidence below records the earlier interface and remains history; it is not the current acceptance contract.

Edit features, labels and fold assignments and inspect the held-out model outputs immediately; change fair-label cases or candidate rules and follow selected validation accuracy and fresh-label performance; change a nested fixture and trace which training labels can influence the selected neighbour count and protected-row model output; edit loss trajectories, budget and survival factor and inspect the paid-for schedule. Invalid configurations display errors; hindsight remains an explicitly identified diagnostic outside the actual schedule.

Keep separate independent practice, model predictions, scientific validity checks and training/validation/held-out information boundaries. Meaningful valid control changes must reach the visible calculation and topic-specific diagram together. Natural algorithm Step/Back/Run actions remain where they expose a process; they must never require a learner guess. Reset restores a coherent initial state. A graph or number must not silently describe obsolete inputs; invalid inputs show an error and either clear invalid outputs or explicitly retain the last valid result. See [the current migration evidence](../../LIVE-EXPLORATION-CLASSICAL-EARLY.md) for implemented checks and limitations.


Stable ID `cross-validation-hyperparameter-tuning`; Classical ML position21, authorized batch position3. Author `/root/classical_feature_content`, 12 September2026. Content-only research/write packet; shared phase ledger and final hash reconciliation belong to root. No production/browser/formal implementation review is claimed.

## Preflight and original conservation

Ran `node scripts/build-curriculum-inventory.mjs --topic cross-validation-hyperparameter-tuning --work content`; read all returned domain/note material and the individual-design-required state. Full existing JSX was read in order, including programs, outputs, visual data, decision tables and exercises. Baseline commit `8c5da59f18516be77c29d5aeeafca3decca4f738`, original source `src/learn/data/topics/cross-validation-hyperparameter-tuning.jsx`, SHA-256 `6c80a5e38004f269398f93e1a222b2e4a3c5239afae4baf4c186ab96b1bc31a5`. Source remains untouched.

Title remains appropriate. Inventory declares a prerequisite whose catalogue position is later: ML Problem Formulation, Baselines & Data Leakage. The lesson therefore introduces prediction task, row/unit, baseline accuracy, information availability and validation/test roles locally, rather than blocking the actual module sequence or presupposing that later lesson. Previous Feature Scaling's fit/transform boundary is the immediate bridge; next Regularization uses this selection framework. Existing ensemble destination note is consumed and incorporated, with content disposition saved but implementation still open.

| Original material | Conservation and correction |
| --- | --- |
| Motivation and historical overview | Preserve need for assessment/selection and original research references where substantive evidence inspected; replace unsupported universal holdout swings and triumphalist history with an actual task/estimand |
| K-fold, stratified, grouped, time, LOO and repeated variants | Retain all with fit sizes, available information and explicit tasks; remove always/mandatory classifier/time prescriptions and arbitrary row thresholds |
| LOOCV variance and equal-correlation formula | Retain general covariance formula and conditional simplification; correct false correlation=overlap argument with independent-label fixed-rule counterexample and primary variance theorem |
| Nested CV and winner's curse | Retain full procedure and code; correct universal unbiasedness, mandatory nested-only claim, fixed optimism percentages and individual-result ordering. Explicit separate final test is also valid. |
| Scratch splitters, logistic fit and nested loop | Replace remainder-dropping and unused duplicate-fit code with complete tiny splitter/1NN and actual pipeline nested experiment; optimizer internals belong to their existing owners, not an unstable200-step logistic demonstration with claimed outputs |
| Grid/random/halving/TPE implementations | Preserve mechanisms, resource accounting, current API and complete main plus optional Optuna program; exact unexecuted Optuna outputs intentionally omitted. No misleading search-method winner claim from different spaces/budgets |
| Step traces, score heatmaps, trajectories | Replace impossible .90/.85 accuracy on five assessed rows, fabricated grid/trajectory values and unjustified intervals with integer-correct traces, real candidate scores, analytic coverage and constructed disclosed trajectories |
| Search decision matrix and runtime | Replace arbitrary numeric recommendation heatmap, trial-count/seconds cutoffs and wrong arithmetic. Nine-candidate schedule costs270 from scratch or210 with real resumption, not the original invalid geometric formula. |
| Early stopping, parallelism, failure cases and exercises | Preserve practical issues, but place stopping inside protected boundaries, distinguish dependent comparisons from automatic hypothesis tests, and give changed exercises with optional explanations |

The original claimed nested score.90 exceeding selected score.885 was not proof of upward bias on that realization; the new prose explains bias as a repeated-experiment property. No old invented empirical curve is conserved merely for visual similarity.

## Canonical reference coverage and ownership

Primary current reference section lists inspected: sklearn Cross-validation3.1.1 score/multiple metrics/OOF;3.1.2 iid K-fold/repeated/LOO/leave-P-out/shuffle, stratification, predefined, grouped variants and time. Parameter search3.2.1 grid;3.2.2 random;3.2.3 halving and resources;3.2.4 metric/multimetric, nested parameters and development/evaluation; estimator-specific alternatives. Manuscript covers main operations and leaves exhaustive splitter listing to its annotated API, with local examples of each meaningful boundary. Leave-P-groups and GroupShuffleSplit are variants of the locally taught group holdout/resampling principles; no separate API tutorial is necessary to core readiness.

Root successfully inspected the official ESL front matter PDF page15 and provided its full chapter7 section list after this agent's web retriever returned an internal error. Source https://link.springer.com/content/pdf/bfm:978-0-387-84858-7/1. This is a section-list check only, not a claim that the full chapter was read. Disposition:7.1 introduction→task/estimands;7.2–7.3 bias–variance→root's dedicated later lesson with local boundary here;7.4 optimism→exact mean-predictor derivation;7.5–7.6 in-sample/effective-parameter details→Bias–Variance's complete fixed-linear-smoother derivation;7.7 Bayesian/BIC and7.8 MDL→forthcoming Regularization comparison with explicit assumptions and purposes;7.9 VC→dedicated PAC/VC lesson;7.10 allthree CV subbranches→this manuscript's mechanism, wrong/right boundaries and uncertainty;7.11 bootstrap→local distinct-sample/uncertainty bridge, full advanced inferential variants deferred to specialist references;7.12 conditional/expected test error→explicit local definition and independent mean-model calculation. No unsupported full-book endorsement.

Root and this author agreed Regularization will give AIC/BIC/MDL a bounded substantive home rather than leaving the canonical ideas in a vague unowned reference. Do not duplicate root's full linear-smoother proof there. The manuscript links that immediate next topic directly; parent will reconcile its completed coverage when that packet is frozen.

## Hurdles, representations and learner evidence

| Hurdle | Representation | Demonstrated skill |
| --- | --- | --- |
| Model fit versus setting selection | Nested learning loops with protected assessment outside | Identify which rows influence a decision |
| Fold membership and leftovers | I1 editable point/role matrix and computed1NN predictions | Every row assessed once; own-target null; correct pooling |
| Row/group/time mismatch | Two task-specific patient timelines and label-availability cutoff | Choose split from actual future use |
| Selection learns criterion noise | I2 candidate bit patterns and exhaustive fair-label enumeration | Score increase without new predictive information; duplicate null |
| Inner/outer indices confused | F3 real index explorer and I3 sixteen-row actual nested fits | Protected label cannot choose k; training label can |
| Uncertainty confused with overlap/spread | Conditional/expected diagrams, exact mean-risk curves, covariance counterexample | State what is averaged and why fold spread is not a universal CI |
| Search-space coverage confused with score proximity | Grid projections and analytic hit probabilities | Distinguish distribution mass from performance tolerance |
| Early budgets confuse rankings | I4 editable observed/hidden trajectories and exact resource ledger | Identify slow starter and late-value null |

Core route appears immediately after introduction and ends with its own readiness criteria. Deeper expectation/covariance/TPE/Hyperband branches do not become gates. Every lab has unset input-bound recorded predictions, meaningful entity edits, checked contrasts and nulls, and text/mobile requirements. I3 was improved during writing from a hypothetical editable score table to actual nested computation on editable sixteen-row features/labels; real saved experiment remains separately labeled.

## Substantive research record

| Source | Material actually inspected | Use / limit |
| --- | --- | --- |
| [sklearn CV guide](https://scikit-learn.org/stable/modules/cross_validation.html) | Section list,fit/transform,score-vs-OOF,all main split families,stratification caveat,current GroupKFold and time behavior | Current contracts; source's informal LOO explanation not accepted as a general proof |
| [sklearn search guide](https://scikit-learn.org/stable/modules/grid_search.html) | Grid/random/halving resource definitions and actual schedules,cv_results_,metric/refit/nested-parameter/development sections | Correct allocation and API boundaries, no fabricated deployment timings |
| [cross_val_predict API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html) and existing verified ensemble note | Exactly-once prediction partition requirement; note's installed TimeSeriesSplit failure and full manual OOF construction context | Local distinction retained without claiming a second native failure campaign |
| [Cawley–Talbot](https://www.jmlr.org/papers/volume11/cawley10a/cawley10a.pdf) | §2.1 context, §4/4.1 criterion overfitting and expected-vs-single-realization results, §4.2 split-sample variation and figures 4–9 | Selection included in the learned recipe; benchmark effect sizes not reused as universal |
| [Bengio–Grandvalet](https://www.jmlr.org/papers/volume5/grandvalet04a/grandvalet04a.pdf) | Abstract,§§1–2 performance/expected performance distinctions and theorem scope; covariance discussion | Narrow no-universal-unbiased-variance result, not impossibility of all uncertainty methods |
| [Bergstra–Bengio random search](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf) | Introduction/effective dimension figure and§2.1–2.2 validation/test and trial-distribution discussion | Independent probability derivation, no claimed within5%-optimal guarantee or universal dominance |
| [Original TPE paper](https://papers.nips.cc/paper_files/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf) | §§2–4,expected-improvement equation and density ratio derivation,conditional parameter interpretation | Local discrete EI example and true TPE distinction; no made-up trajectory or fixed quadratic overhead |
| [Hyperband](https://www.jmlr.org/papers/volume18/16-558/16-558.pdf) | Introduction,§§3.1–3.2 algorithm and actual bracket table,early-ranking/envelope explanation | Halving versus multiple brackets, no monotonic-curve sufficiency claim; full theorem proof outside manuscript scope |
| [Optuna TPE5.0 docs](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html) | Signature,startup trials,sampler seed,current multivariate/conditional semantics,working API example | Complete supplementary development program, explicitly unexecuted; no package installed |
| [CV indices visual example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html) | Actual constructed class/group data,plot_cv_indices role assignment code and illustrated splitter comparisons | Meaningful alternate visual/code learning resource, not title-only endorsement or runtime claim |
| [Palmer Penguins project](https://allisonhorst.github.io/palmerpenguins/) | Same actual dataset/variable/license evidence from immediate predecessor | Reused full CC0 observed input with explicit familiar-data limitation |

No video watch claimed; the inspected visual code article and primary diagrams provide alternative learning routes without a medium quota. No inaccessible full ESL chapter or stale URL was repeatedly probed after root supplied the working section-list evidence.

## Evidence and author reconciliation

Bounded real pipeline calculation:76 small nested/final pipeline fits and3 majority baselines, serial, no warnings, exact splits and predictions retained. No numeric claim comes from a manually painted accuracy surface. Tiny fold predictions and sixteen fair-label patterns are enumerated, not noisy simulation. Separate small nested-neighbor functions computed base and changed-label cases; all states retained. Practice arithmetic is derived from declared finite quantities; no source performance percentages are invented. Main/tiny displayed programs reproduce equivalent author-script calculations; verbatim full native program campaign is still phase two. Optional Optuna is current-doc checked but unexecuted with no promised stdout.

Complete manuscript and specifications were read in ordinary order before submission to root. Targeted author checks covered remainder preservation, possible integer accuracy values, weighted/unweighted results, TPE-vs-GP density semantics, halving cost/refit accounting, own-outer-label independence and real-versus-constructed boundaries. The previous score-edit lab was replaced by sixteen-row computed entities because altering outcomes would hide the mechanism. The final read aligned F4 with the actual seed-5 sample positions, linked the criteria branch to Regularization and sorted the tiny splitter's training IDs so its distance-tie contract also holds when shuffled. No new real-data fit campaign was needed for these wording/small-fixture corrections. Formal implementation review remains separate.

## Phase-two handoff

Implement prose, inline figures and four topic-specific investigations from these contracts, with source/data/result provenance. Execute displayed programs including the explicitly deferred supplementary Optuna example where authorized; verify actual UI numeric states, changed/null cases, arbitrary entity edits, prediction commitment and readable responsive/text alternatives. Complete formal independent review and production integration before marking implementation complete. Source/runtime/manifest/navigation and central ledger were not edited by this author. Keep required offline inputs/results/calculation; remove the exact disposable Python import cache after writing its numerical evidence. Next authoring topic is Regularization.

## Phase two: implementation

14 September 2026. Implemented from the frozen content packet above: the manuscript, the visual specifications, `data-provenance.md`, `author-calculations.py`, `calculated-inputs.json` and `penguins.csv` were read in full before any file was written, together with the two most recent finished lessons (GMM, t-SNE/manifold) and `GMM-INDEPENDENT-REVIEW.md` as the standard this work expects to be reviewed against. Nothing in the packet was edited; this section is the only write into this directory.

### What was built

| File | Role |
| --- | --- |
| `src/learn/data/validation-models.js` | Pure models. A **fold plan** is the central entity: `foldPlan({rows, folds, groupOf, labelOf, requirePartition})` returns which source row IDs sit in which fold, the complementary training list for each fit, whether any group spans a boundary, how the declared classes fell per fold, and whether the plan is an exactly-once partition. `outOfFoldCoverage` turns that into the `cross_val_predict` contract. Also `consecutiveFolds` (`np.array_split` semantics), `nearestNeighborRun`, `scoreCandidates`/`enumerateSelection`/`futureAccuracy`, `nestedTrace`, `hitProbability`/`drawsForHitProbability`, `meanPredictorRisk`, `fixedRuleAccuracyVariance`, `foldMeanVariance`, `expectedImprovement`, `tpeAcquisition`, `successiveHalving`, `fitBudget`, `bootstrapDistinctShare`. Every entry point validates and raises `RangeError` rather than substituting: a dropped remainder, an empty fold, a duplicate row, an unknown ID, an even neighbour count, a non-binary label, a neighbour count larger than the inner training set, a decreasing budget list, a survival factor below 2, a first comparison at the last budget, and probabilities that do not sum to one are all refused. |
| `src/learn/data/validation-data.js` | Generated. `PENGUIN_SOURCE` (hash, byte count, columns, missing counts, species counts, licence, limits), `NESTED_EXPERIMENT` (per outer fold: training and protected row IDs, inner validation IDs, all six candidates with three inner fold scores and their mean, selected index, selection score, every protected prediction and truth, correct count, majority baseline, and the rows predicted wrongly), `RANDOM_COVERAGE_POINTS` and `RUNTIME_VERSIONS`. |
| `src/learn/data/validation-examples.js` | Generated. The three displayed programs with their executed output. |
| `src/learn/components/lesson-labs/ValidationShared.jsx` | Investigation primitives: the predict → **commit** → apply contract, number and select fields that publish only valid values, role marks, tables, and a 360-unit plot frame. |
| `src/learn/components/lesson-labs/ValidationLabs.jsx` | The four investigations. |
| `src/learn/components/lesson-labs/ValidationFigures.jsx` | F1, F2, F3, F4, F5, F6 and the read-only real-experiment explorer. |
| `src/learn/components/lesson-labs/validation-labs.css` | Stylesheet; every class prefixed `cv-`. |
| `src/learn/data/topics/cross-validation-hyperparameter-tuning.jsx` | The rewritten lesson body: eleven sections, seven inline figures, four investigations, three complete programs, ten practice tasks with separate closed hint and solution disclosures, a readiness table and the annotated references. Default export shape unchanged. |
| `src/learn/data/curriculum/blueprints/cross-validation-hyperparameter-tuning.js` | Blueprint. **Not registered here**: `blueprints/index.js` is the parent's file. |
| `public/learn-assets/cross-validation/` | `penguins.csv` (byte-identical to the packet copy, SHA-256 `f204db2c…`), `data-provenance.md`, and a generated `nested-experiment.json` holding every split, candidate score and prediction so the experiment is inspectable outside the page. This lesson serves its own copy; the concurrent feature-scaling lesson keeps its own. |
| `scripts/verify-validation-{models.mjs,examples.py,data.py,browser.cjs}` | The four verifiers, with evidence at `docs/teaching/evidence/validation-{models,native,data,browser}.json`. |

### Mapping to the specification

F1 is the two-loop diagram with the assessment bundle outside both loops and a labelled text equivalent; F2 carries the two patient panels computed through `foldPlan` with its grouping report, the label-availability timeline, and the forward-versus-partition coverage contrast; F3 is the five-stage nested expansion drawn with outer fold 1's real row IDs, beside the read-only explorer over all three folds and all eighteen candidate scores; F4 is the grid-versus-nine-draws projection with the stored `default_rng(5)` coordinates plus the analytic hit curve; F5 is the conditional/expected pair, the exact mean-predictor curves, the covariance grid with a computed fold-mean-variance table, and the always-zero counterexample; F6 is the discrete-mass improvement picture with its exact contribution table and the TPE density-ratio branch with a computed acquisition table. I1–I4 are the four investigations named in the specification, each with the specification's checked baseline, contrast and null preloaded as declared setups.

### Departures from the manuscript and specification, with reasons

1. **The Optuna program is executed and its output published.** The manuscript says it was "not executed during the content phase" and that exact settings were "intentionally not invented"; the specification allows phase two to execute it. Optuna 5.0.0 was installed into the shared `scratch/lesson-tools` environment (numpy 2.3.5, pandas 3.0.1, scikit-learn 1.9.1 unchanged; only Mako, PyYAML, SQLAlchemy, alembic, colorlog and greenlet were added) and the program ran beside the served CSV. The page keeps the content-phase caveat verbatim and adds that the two printed lines are one reproducible run in a pinned environment rather than a property of TPE, and that the printed score is selection evidence, not an assessment. Its result — selection score 0.9884058 with 1 neighbour, distance power 1, uniform weights — is *lower* than the grid's inner selection score and is published as it came out.
2. **The pooled-risk identity is re-set with a named subexpression.** The manuscript's display is a double sum; at 320 px it overflows. The page defines ℓᵢ in prose and then displays a two-line `gathered` form, which is the same identity. The same treatment was applied to R(m), the mean-predictor expansion, the fold-mean variance and the expected-improvement definition.
3. **Descriptive text moved out of the SVGs.** Every viewBox in this lesson is 360 units wide — the narrowest column a figure is rendered into — and all prose lives in HTML so it reflows. Two long in-figure captions (the covariance grid's and the horizon timeline's) became HTML paragraphs for that reason.
4. **The role table caption in F2 duplicates the panel heading.** Kept deliberately: the table is a `role="region"` with an accessible name, and the duplication is what a screen-reader user hears as context.
5. **Two verified models that the manuscript states only in prose are now also computed on the page** — the equal-correlation fold-mean variance and the TPE acquisition value — rather than shipping as exports nothing consumes. Both tables are labelled as consequences of a stated assumption, never as measurements.
6. **No "calculate without predicting" escape hatch.** The GMM review found an intro promise broken by such a button. Here the Apply action is disabled until a prediction is committed, in all four investigations.
7. **`hasIntegratedGuide: true` is set in the lesson module but the generated navigation still says false.** `src/learn/data/generated/navigation.js` is regenerated by `scripts/generate-learning-artifacts.mjs`, which reads the blueprint registry; that is the parent's step. Until it runs, the legacy `LessonGuide` compass renders above this lesson's own integrated introduction, and the sidebar still shows the old `~50 min` read time and a null depth. See "What is needed from files this agent did not edit".

Nothing else in the manuscript was dropped. All ten practice tasks, the three data-role rows, both worked tables in §2, the sixteen-pattern enumeration table, the recorded-results table, the halving schedule table, every named caution and the closing "none of them is a benchmark" paragraph are on the page.

### Checks actually run, with results

| Check | Command | Result |
| --- | --- | --- |
| Models | `node scripts/verify-validation-models.mjs` | **PASS** — 10 groups, 17,912 assertions. Independent oracles: `array_split` sizes recomputed for every n ≤ 40 and every k; 400 randomised 1-NN runs graded against a brute-force nearest-eligible search; 200 randomised nested runs graded against a directly written stable-sort majority vote; all 16 fair-label patterns enumerated; the fixed-rule leave-one-out variance enumerated over every label pattern to n = 8; 240 (p, T) hit probabilities recomputed by repeated multiplication; and all three retained packet traces (`tiny`, `tiny_nested.trace`, `changed_trace`, `selection_changed_trace`) compared value by value. |
| Displayed programs | `scratch/lesson-tools/Scripts/python.exe scripts/verify-validation-examples.py --write`, then again without `--write` | **PASS** both times — 3 programs executed, 48 oracle assertions. The second run proves the recorded output matches a fresh execution. |
| Real data | `scratch/lesson-tools/Scripts/python.exe scripts/verify-validation-data.py` (and `--write`) | **PASS** — 125 checks, 76 pipeline fits refitted from the served CSV. Every value in `calculated-inputs.json` was re-derived, not copied: all three outer fold index lists, all nine inner validation lists, all 18 candidate rows with their three inner scores, the selected indices, all 344 protected predictions and truths, the baselines, pooled accuracy and the final selection. Selections 11/Standard, 3/Standard, 3/Robust; 114/115, 114/115, 112/114; baselines 51, 51, 50; fold mean 0.9883549, pooled 340/344 = 0.9883721, final selection score 0.9913043. |
| Build | `npx vite build --outDir dist-cv` | **PASS**; the lesson chunk is 172.8 kB raw / 53.9 kB gzip. |
| Browser | `DIST_DIR=dist-cv LEARNING_BASE_URL=http://127.0.0.1:4184 … node scripts/verify-validation-browser.cjs` | **PASS** — 12 behaviour groups, 18 captures. Includes: every displayed program's code and output present verbatim; 23 named manuscript claims and cautions asserted present by phrase; module position 21 of 39 with the correct previous and next; all four investigations opening unset with Apply disabled; I1's baseline, training-label contrast, own-label null, translation null, prediction retirement on an arbitrary edit and refusal of an empty fold; I2's 0.6875 enumeration, duplicate null, sixteen-pattern saturation and the 0,1,0,1 fixture before and after a matching rule; I3's tie-resolved baseline, selection contrast, protected-label null and the role swap; I4's early-ranking baseline with its 150-unit ledger, later-start contrast, unobserved-value null, observed-value contrast and the 270/810/210 nine-candidate ledger, with the hindsight comparison absent from the DOM until revealed; all 18 real candidate means visible and read-only; layout inspected at 1366/1024/768/390/320 with **zero** geometry findings; no display-formula overflow at 320 px with every disclosure open; no SVG label below 9.5 effective pixels at any width; no overflow at 200 % root text; only this lesson's body requested; completion persistence; and recovery from both an import failure and a render failure. |

### What the screenshots changed

All 18 captures were opened and looked at. Seven changes came from that pass, and five of them were invisible to the assertions that had already passed.

1. **F5's risk curves were a staircase.** They were sampled on a continuous axis and then rounded to integer n, which drew 39 steps instead of the formula. Replaced with a polyline through the integer values.
2. **F5's two value labels sat on top of the curves**, and "n=12 → 4.3333" was drawn across the dotted irreducible line. The geometry inspector never saw it, because it only tests `line` elements and that line is a `polyline`. Both labels were removed; the marks stay, and a caption plus the exact-values table name them.
3. **F4's random draws were clipped by their own frame.** Draw 5 at coordinate 2 = 0.9992 and draw 4 at 0.0453 were drawn half outside the unit square. The drawing area is now inset inside the frame.
4. **F1's score-return path ran straight through the "validation rows" label**, and the fitted-model arrow landed in the gap between two boxes. The return path now leaves the score box and runs below both, and two arrows — one from the fitted model, one from the validation rows — now land inside the score box.
5. **F2's cutoff line crossed two timeline labels.** Every word moved out of that SVG into an HTML caption; the drawing keeps only the dots, the measurement span and the cutoff line.
6. **F6's "improvement region" label overlapped candidate A's mass label.** The region is now named in the legend instead, and the legend also says that stem height is the believed probability, which the picture did not previously state.
7. **I1's row IDs and label chips collided**, I4's plot wasted its height on a forced 0.5 ceiling with no intermediate ticks and no axis name, and both labs' dense control rows wrapped their captions to three lines at 390 px. Fixed by separating the stacked marks, scaling I4's axis to the largest authored loss with three ticks and a "loss" label, and giving the fields a short visible label with the full sentence kept as the accessible name.

One wording error was also caught by reading the rendered page rather than by any assertion: the real-experiment explorer said "Five **other** candidates reach the same inner mean" for outer fold 3, where five candidates share the top mean **in total**. Corrected.

Two earlier failures worth recording because they were found by running, not by reading: a preset key falling through to `undefined` crashed the fold lab on the first arbitrary edit, and switching the inspected outer fold in I3 left a stale protected-row ID that made Apply throw silently. Both are now impossible — the preset caption falls back, and the protected row is resolved from the fold in both the prompt and the answer.

### What is *not* claimed

- No independent review. This is the author's own implementation and its own verification.
- The real experiment is one dataset, one seed pair, one candidate grid and one tie rule. The outer score assesses the select-and-refit procedure at that outer training size; it is not an estimate of an oracle-best setting, not new evidence about penguin classification, and not independent validation of the preceding lesson's scaler choice.
- Every penguin number is bit-exact only in the pinned environment, which is also the environment the author used. Determinism was confirmed across repeated runs in that environment; no other library version was tested.
- The Optuna output is one run of a sequential sampler at one seed. A different Optuna or scikit-learn version can move the trial sequence.
- No timing, wall-clock or speedup claim appears anywhere. Fit counts are exact; costs in the halving lab are abstract resource units.
- No screen-reader pass, no reduced-motion pass and no colour-contrast measurement were run. Role encoding was designed not to depend on hue (every role also carries a word and a glyph or dash pattern), but that was not instrumented.
- Six of the 18 captures are at 320 or 390 px; there is no capture at 768 or 1024, where only the assertions ran.
- The phase ledger was not touched, and `blueprints/index.js`, `lesson-manifest.json`, `tracks.js`, `generated/navigation.js` and everything under `docs/curriculum/` were not edited.

### What is needed from files this agent did not edit

1. **Register the blueprint** in `src/learn/data/curriculum/blueprints/index.js`.
2. **Regenerate the learning artifacts** (`node scripts/generate-learning-artifacts.mjs`) afterwards. That is what carries `hasIntegratedGuide: true`, the new `readTime`, and `depth: 'core'` from the blueprint into `src/learn/data/generated/navigation.js`. Until it runs, the legacy compass renders above the lesson's own introduction.
3. **Close the phase-two ledger entry** for this topic with the source hashes below.
4. Optuna 5.0.0 and its six pure-Python dependencies were added to the shared `scratch/lesson-tools` environment. Nothing else in it changed; if that environment is meant to stay pinned, this addition should be recorded or reverted.

### Source versions at completion (SHA-256)

| File | SHA-256 |
| --- | --- |
| `src/learn/data/topics/cross-validation-hyperparameter-tuning.jsx` | `06fc691828739656ef3c53fddecd11c78474b5d2e5caac1ce10dddc9439c1222` |
| `src/learn/data/validation-models.js` | `b38fbc1835d45988a0f0dc7a2f9fa5cfcbb948ace82136286085a4f6dd19d4f4` |
| `src/learn/data/validation-data.js` | `fdc04c6e6cd048f59b26e10152f741ed2284c01f27cee66f570b0d30489c931a` |
| `src/learn/data/validation-examples.js` | `d5c8677e86c0b7513ef6748624b26a910062b94925b53a29b41ba93ce556394b` |
| `src/learn/components/lesson-labs/ValidationShared.jsx` | `2d0f5e8ea0976282d274238be59c1e374c3ca73ec541fc466b4abcec8dc0926a` |
| `src/learn/components/lesson-labs/ValidationLabs.jsx` | `d1cbef8d650dc6e6edd78f8089012f22dba9c94a5caa118ed877e5c3ce57e2c5` |
| `src/learn/components/lesson-labs/ValidationFigures.jsx` | `702fe7d0e1fabc581899cf104ab1c76d55b255817e8b9411b6c20b26b5efb984` |
| `src/learn/components/lesson-labs/validation-labs.css` | `f15b477b3f0679973546d10707980d2ca75d8208287771492f6027787fa31b75` |
| `src/learn/data/curriculum/blueprints/cross-validation-hyperparameter-tuning.js` | `e95b4dc370f106d1503a942baa8b19c1555bc2f8d51112336ea2d2e7af492477` |
| `public/learn-assets/cross-validation/penguins.csv` | `f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93` |
| `scripts/verify-validation-models.mjs` | `71f9a4a9763d70b9750c60651789a4765c9fb60655b1cd4680fbc87c6ccbb272` |
| `scripts/verify-validation-examples.py` | `8b2cb0041cbcecfe2a40454d0b9d4644bc4e95cd722f44b64c57bdc6fc9220ec` |
| `scripts/verify-validation-data.py` | `bd4725042f264ef345863c4909609ff7a009be2b2372d5cab74725fdb8687833` |
| `scripts/verify-validation-browser.cjs` | `26f8008ec9d37a1aee705c9d40c3a8cddd5b5e3b67230bb3b0b4498a1e87646c` |

A hash records identity, not correctness. The four evidence files name exactly these bytes, so their recorded results bind to this version. Nothing is committed, and user acceptance is separate from this completion.

## Disposition of the independent review

14 September 2026. The [independent review](../../CROSS-VALIDATION-INDEPENDENT-REVIEW.md) was read in full before any change. It refitted the whole penguin experiment without `GridSearchCV` and reproduced every selection, correct count, baseline, all 72 inner candidate scores, both summaries and the final score exactly; all three displayed programs reproduced their published output byte for byte; and it confirmed live that none of the four investigations shows or grades anything before the learner commits. No number changed as a result of this pass. Every fix below is a caption, a sentence, a link, a column, a state transition or an added check.

| Finding | Disposition |
| --- | --- |
| **B1** ledger stale | **Not mine.** The coordinator is closing it. The source-hash table at the end of this section is the current one to bind. |
| **S1** previous-trial comparison unreachable | **Fixed.** `useInvestigation.retire` now stashes the outgoing result into `previous` at the moment an edit retires it, instead of `apply` trying to stash at a moment that could never arrive; the dead branch in `apply` is gone. `reset` and `load` still clear it, as the review prescribed. The panel's wording was corrected to match the state it actually appears in — the earlier answer belongs to inputs the learner has since edited, not to a result still on screen. Now exercised and captured: `validation-folds-previous-trial.png`. |
| **S2** Figure 3's inner level showed only counts | **Fixed.** Step 2 now lists the first six original row IDs of each inner validation fold in the same `sample()` style step 1 uses, drawn from `innerSplits[].validationRows`, with a sentence noting that not one protected ID appears among them. The caption's promise is now true. Asserted in the browser pass against the shipped data. |
| **S3** Optuna departure stopped one sentence short | **Fixed, and made a checked fact rather than a recollection.** I reproduced the review's claim independently before publishing it: the study scores on `StratifiedKFold(3, shuffle=True, random_state=73)` over all 344 rows — the same three folds the final grid search uses — and its space contains three neighbours, standard scaling, Euclidean distance and uniform weights, which on those folds scores `0.991304347826087` against the study's `0.9884057971014494`. The page now says so, and says that none of the twenty trials evaluated that point, framed as a fact about one seeded run at one budget rather than about TPE or in favour of grids. The old "changes the search space" guard was rewritten so it no longer implies incomparability. Five new oracles in `verify-validation-examples.py` check the split equality, the point's score, its strict advantage, its membership of the declared space and its absence from the twenty trials, so the sentence fails the verifier if it ever stops being true. |
| **S4** hindsight leaked before its own control | **Fixed.** The post-Apply caption now says only the graded fact — "It is not the lowest loss at the full budget" — and points at the control; the Reveal block is the only place that names the candidate, its loss and the regret. The browser evidence case was reworded to claim what it tests, and a new assertion checks the caption itself rather than only the block's absence. |
| **S5** final all-data tie unstated | **Fixed, both ways the review offered.** `verify-validation-data.py` now records the final search's six candidate rows, its `bestIndex` and its `tiedWith` list, checked against a fresh refit; §6 gains a paragraph naming the tie (`3/StandardScaler and 3/RobustScaler` at `0.9913043478`), the fact that all six collapse into three tied pairs, and the conclusion that the deployed scaler is settled by enumeration order rather than by evidence; and the explorer gains a fourth view, "Final search · all 344 rows", with that candidate table and an explicit "Protected result: none — every row was used to choose". |
| **S6** `nested-experiment.json` linked by nothing | **Fixed.** §6 now links it beside the CSV and the provenance note, describing what it holds. The browser pass asserts the anchor exists. |
| **O1** F4's hit curve kept the sample-and-round construction | **Fixed.** Both curves now use `integerCurve`, the helper written for F5. |
| **O2** F1's terminal arrow had no source | **Fixed.** A "selected, refitted" box now sits between the separator and the protected box, so the arrow leaves something; the `aria-label` was updated to match what is drawn. The 320 px compression the observation also notes is unchanged and still stands as a limit. |
| **O3** §2 checkpoint duplicated practice 2 | **Fixed.** The checkpoint is removed; the manuscript poses the question once, as practice 2. The now-unused `Checkpoint` import went with it, and the browser pass asserts no `.lesson-check` remains. |
| **O4** dropped forward pointer | **Fixed.** §1 now links the evaluation-metrics lesson as well as the imbalanced-learning one. |
| **O6** selected row marked by colour alone | **Fixed.** I1's contributions table gains a `role` column reading "nearest"; I2's candidate table marks the winner in its first cell as "candidate B · selected". The first attempt added a sixth column to I2 and the screenshot showed it pushing the fresh-label figure past the visible width, so the marker moved into the rule cell and the long header was shortened to "on fresh labels" with the caption carrying the full meaning. The table now fits its container exactly (scrollWidth 701 = clientWidth 701 at 1366 px). |
| **O9** pinned versions are behind current releases | **Fixed.** §6 now says the pins are for reproducibility rather than a recommendation to stay behind, and that newer releases can move the digits. |
| **O10** Optuna's real console output is 23 lines | **Fixed in prose.** The paragraph after the program says it also logs one line per trial to standard error and that the two lines shown are the program's own standard output. I did not add `set_verbosity` to the displayed program: it is the manuscript's program, and adding a line to suppress output the lesson then has to explain is a worse trade than one clause. |
| **O11** outstanding-work list two-thirds done | **Recorded.** Blueprint registration and artifact regeneration are done, by the integration owner; departure 7 in the phase-two section above is therefore retired, and I confirmed no legacy compass renders. The Optuna addition to `scratch/lesson-tools` remains recorded only here. |
| **O12** covariance grid stated its point rather than drawing it | **Fixed.** The twenty off-diagonal cells now carry a `C` glyph against the diagonal's `V`, in different colours, and the caption and `aria-label` name both. |
| **O13** `0.9913043` appears as two different quantities | **Fixed by naming it.** §6 now points out that outer folds 1 and 2 each score 114/115 and the final selection score prints the same seven digits, that they are different quantities, and that the equality is arithmetic — 114/115 and 228/230 are the same ratio. `verify-validation-models.mjs` asserts the equality and that outer fold 1's own selection score is a third number again. |
| **O5, O7, O8, O14** | **No change, and the reasons are the review's own.** O5 is a deliberate trim the review agrees improves the intro. O7's 9.82 px is above the project's floor and raising type further would start wrapping labels inside their boxes. O8's two touches are 6–7-unit offsets that read cleanly. O14 is the specification's own compromise: the container is focusable and keyboard-scrollable, and shrinking the text to avoid the scroll is what the specification forbids. |

### One defect the review did not find, caught by re-opening a screenshot

The new previous-trial capture put the fold lab into a state nobody had looked at: the contrast preset with row 3 relabelled, where all three folds reach accuracy 1. The summary sentence there read "They agree here because the folds happen to be equal in size" — and the folds are 3, 2 and 2. The stated reason was false; the two summaries agreed because every fold reached the same accuracy, not because the weightings coincided. The sentence now distinguishes three cases and says which one applies, and the browser pass asserts both the unequal-and-differing wording on the baseline and the equal-accuracy wording on that contrast. This is the same class of defect the screenshots caught the first time round: an assertion can confirm a number and still let a wrong explanation ship beside it.

### Checks re-run after these changes

| Check | Result |
| --- | --- |
| `node scripts/verify-validation-models.mjs` | **PASS** — 10 groups, **17,926** assertions (14 new, covering the final search's candidate table, its two-way tie, the three tied pairs, and the 114/115 coincidence). |
| `verify-validation-examples.py --write`, then again without | **PASS** both — 3 programs, **53** oracles (5 new, all about the Optuna claim). Output still reproduces a fresh run. |
| `verify-validation-data.py` | **PASS** — **129** checks (4 new, on the final search's tie and its three distinct scores), 76 pipeline fits refitted. Every previously recorded value is unchanged. |
| `npx vite build --outDir dist-cv` | **PASS**. |
| Browser pass on port 4184 | **PASS** — **13** behaviour groups, **20** captures. New cases: Figure 3's inner row IDs; the explorer's final-search view with all six means, both tied candidates and the "no protected result" statement; the previous-trial comparison appearing on edit, surviving the next apply and clearing on Reset; the hindsight caption not naming the winner; the split-record anchor; the absence of any checkpoint; and both fold-weighting explanations. Layout inspection at 1366/1024/768/390/320 remains at **zero** geometry findings, no display formula overflows at 320 px, and no SVG label falls below 9.5 effective pixels. |

All twenty captures were re-opened and looked at. Beyond the two fixes above that came directly from them, nothing else needed changing.

### Source versions after the review pass (SHA-256)

| File | SHA-256 |
| --- | --- |
| `src/learn/data/topics/cross-validation-hyperparameter-tuning.jsx` | `c16456eee696b78ff1347c53a088cf3f9a85e627d1781473e50d14bf868d0454` |
| `src/learn/data/validation-models.js` | `b38fbc1835d45988a0f0dc7a2f9fa5cfcbb948ace82136286085a4f6dd19d4f4` |
| `src/learn/data/validation-data.js` | `6a7b1a5d9ef55846fb723967202498ef4f71add328b1e94e1ae8e320b317a0a0` |
| `src/learn/data/validation-examples.js` | `d5c8677e86c0b7513ef6748624b26a910062b94925b53a29b41ba93ce556394b` |
| `src/learn/components/lesson-labs/ValidationShared.jsx` | `6a455af14239e47597e88ffe312cff08affeb0971ae5b6dd2892a035309ff782` |
| `src/learn/components/lesson-labs/ValidationLabs.jsx` | `c4dec956db99e311323d85c4022d433a4bb71786fea98f6f23778141c136a088` |
| `src/learn/components/lesson-labs/ValidationFigures.jsx` | `3a541a8101ceac75212b50fc46372717bf5f29563eab373fa506c74b50e2b47e` |
| `src/learn/components/lesson-labs/validation-labs.css` | `93c191d5f17c6c669d2b0a7e9b42fbc0fa94db896d434574c6a80ee2da048b83` |
| `src/learn/data/curriculum/blueprints/cross-validation-hyperparameter-tuning.js` | `e95b4dc370f106d1503a942baa8b19c1555bc2f8d51112336ea2d2e7af492477` |
| `public/learn-assets/cross-validation/penguins.csv` | `f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93` |
| `public/learn-assets/cross-validation/nested-experiment.json` | `f8a61a361629e29a0f9bb8e42f1b548620910611210c23709da9b095efd4290f` |
| `scripts/verify-validation-models.mjs` | `8cad1f39c2436ee49ac7c6ccf54a2b4690e41a8225330255ffb02ec7ec35ea73` |
| `scripts/verify-validation-examples.py` | `4978c6284032f58f848e36125462713add830b46224a48f1a88041cd392d12be` |
| `scripts/verify-validation-data.py` | `1f42ae523cea1a3ff83f10a1831b1e888c8586a0d9362ddf55169237db6db629` |
| `scripts/verify-validation-browser.cjs` | `a5b88f8d43186e7f4434a54b266960cfc51a90f97a080e2f4f65a57fcbfa1af9` |

These supersede the table in the phase-two section above. The four evidence files record exactly these bytes. My edits to this design record are final; nothing is committed, and user acceptance remains separate.
