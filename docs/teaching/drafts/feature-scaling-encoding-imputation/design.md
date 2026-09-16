# Feature Scaling, Encoding & Imputation: content design

Stable ID `feature-scaling-encoding-imputation`; Classical ML position20, authorized batch position2. Author `/root/classical_feature_content`, 12 September 2026. Research/write only. Root owns central phase status, hashes and reconciliation; this packet does not claim rendered implementation, production publication or formal independent review.

## Preflight, scope and conservation

Inventory preflight `node scripts/build-curriculum-inventory.mjs --topic feature-scaling-encoding-imputation --work content` was run and its current individual-design-required output read. No destination note existed. Full original `src/learn/data/topics/feature-scaling-encoding-imputation.jsx` was read in ordered chunks through its end, at baseline commit `8c5da59f18516be77c29d5aeeafca3decca4f738`; original SHA-256 `88fd2cde110f3b7c34721d19733de299ddf883abc189ac5927b4ab4797b4a09a`. It remains untouched.

Retain the title. Scope is preparation of tabular features, including operational fit/transform boundaries and deeper representation/uncertainty consequences. NMF precedes this lesson, so the signed-centering/nonnegative-input distinction is explicit. CV follows, so the holdout accuracy calculation and internal target-encoding folds are taught locally rather than assuming a later lesson. The next lesson owns resampling strategy, tuning and evaluation of the selection procedure. Root's later Bias–Variance owns repeated-training decomposition and diagnostics, not a prerequisite for this manuscript.

| Original coverage | Disposition |
| --- | --- |
| Standard/minmax/robust/MaxAbs, normalization, outlier comparisons | Retained with exact numbers, feature geometry and sparse constraints; repaired universality, Gaussian-feature and test-range claims |
| Power transforms, quantiles, inverse maps | Full Box–Cox/Yeo–Johnson definitions and limiting cases retained in deeper branch; rank/magnitude distinction replaces invented density plots and blanket invertibility/order-loss claims |
| One-hot/ordinal/rare/unknown categories | Retained with explicit geometry, reference-column and estimator conditions; repaired claims that every least-squares solver requires dropped categories and that one one-hot tree split selects arbitrary subsets |
| Target encoding with smoothing and out-of-fold construction | Retained and corrected: prior computed within each internal training fold; no remainder rows dropped; own-target dependency contrast/null; current1.9 API checked |
| Feature hashing | Retained with signed counts, collision example and expectation assumptions; removed binary-hash simplification |
| Mean/median/mode, KNN, iterative, indicators, missingness mechanisms | Retained with explicit observed-vs-imputed meaning; exact overlap/donor calculation replaces pre-imputed-distance scratch algorithm; MAR/MNAR caveats and multiple-imputation distinctions repaired |
| Mixed-table pipeline and measured comparison | Replaced untraceable scores with actual offline CC0 penguin dataset, full program and retained predictions; pipeline benefit kept without claiming it makes all leakage impossible |
| Operational performance and inverse-transform claims | No invented timings or arbitrary data-size thresholds. Sparse densification, width/collision tradeoffs and future-value behavior remain concrete. Exact complexity depends on solver/storage/queries; no false fixed all-purpose KNN runtime claim. |

Canonical reference section list inspected: scikit-learn preprocessing guide8.3.1 standardization/ranges/sparse/outliers/kernel centering;8.3.2 nonlinear;8.3.3 normalization;8.3.4 categorical including target;8.3.5 discretization/binarization;8.3.6 missing;8.3.7 polynomial/splines;8.3.8 custom transforms. Core covers the preparation mechanisms; deeper branches retain nonlinear, hashing and basis features. Binarization is the one-threshold special case of the locally taught bins. Kernel matrix centering is acknowledged as a specialist extension owned by kernel/PCA material rather than duplicating its Gram-matrix algebra here; the core does not need it. Sparse coding, bins, polynomial/spline bases and custom circular features receive local explanations, not only links. No canonical major preparation family was silently erased.

## Learning hurdle map

| Hurdle | Teaching action and representation | Learner evidence |
| --- | --- | --- |
| Numbers in incompatible units become accidental weights | Exact Q/A/B contribution calculation; I1 editable metric plane and table | Predict neighbor flip and common-scale null |
| Standardization confused with distribution shape | Same five observed identities on actual rulers, outlier zoom; deeper rank/log map | Explain what remains extreme and what is lost |
| Categories treated as ordered numbers | Full one-hot table plus simplex/reference geometry | Calculate changed distances, unknown policy |
| Imputed value treated as discovered measurement | Two possible completions of identical observed data; indicator distinction | Separate estimated input from physical observation |
| Incomplete donors and overlap misunderstood | I2 cell eligibility/overlap table, exact m/q calculation | Select donors after a target cell is removed |
| Fit and transform mixed or refitted later | Boundary diagram and I3 actual fitted-column trace | Transform an edited held-out cell without changing training statistics |
| Target encoding exposes own answer through category or prior | I4 explicit donor graph and fold-specific arithmetic | Predict own-target null and prior-mediated change |
| One completion mistaken for uncertainty analysis | Three analysis branches, exact within/between variance | Pool changed analysis estimates in deeper practice |

The learner can finish the core without target encoding, power-formula memorization, or Rubin pooling. First-pass route is immediately after introduction. Each major caution has a home rather than repeated warning boxes; code prints results, not compliance prose. Core and deeper practice contain changed inputs, optional closed hints/solutions, and explained exact results. Interesting applications have specific representational consequences: spectral shape versus energy, circular direction versus elapsed time, and fixed-memory hashing. They are not disconnected fact badges.

## Research and claim checks

References below were substantively inspected as of12 September2026, not endorsed from titles alone. The manuscript's examples, calculations and narrative are independently developed.

| Evidence | Material inspected | Consequence |
| --- | --- | --- |
| [Preprocessing guide](https://scikit-learn.org/stable/modules/preprocessing.html) | Complete section list; scaling/sparse/normalization/category behavior and nonlinear piecewise equations | Corrected Yeo–Johnson negative branch and distinctions among geometry, rank and marginal shape; documented scope above |
| [Imputation guide](https://scikit-learn.org/stable/modules/impute.html) | Univariate/iterative/single-vs-multiple, nearest-neighbor eligibility and fallback, missing indicators and empty columns | Corrected incomplete-donor algorithm and one-imputation certainty claims |
| [TargetEncoder API](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html) | fit_transform semantics, smoothing/prior, unknowns,1.9 cv splitter/iterable and deprecations | Locally checked manual cross-fitting excludes row targets even through smoothing; current API wording rather than outdated copied snippet |
| [Cross-fitting example](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder_cross_val.html) | Actual near-unique/random category construction, pipeline fit_transform call and fitted-vs-held-out result discussion | Annotated alternative resource; no claim we reran its larger experiment |
| [Outlier scaler example](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html) | Actual feature selection, full/magnified plotting code, Standard/Robust/Quantile/Normalizer descriptions and outputs | Alternate visual learning route; our own exact five-value data replaces any copied empirical plot; no reused runtime measurement |
| [Palmer Penguins project](https://allisonhorst.github.io/palmerpenguins/) and [variable reference](https://allisonhorst.github.io/palmerpenguins/reference/penguins.html) | Collection context,344 rows, variable units and CC0 license; actual public CSV | Full unchanged offline file, explicit original context versus our task, exact split/provenance |
| [Missingness introduction](https://stefvanbuuren.name/fimd/sec-MCAR.html) | Full MCAR/MAR/MNAR conceptual section and measurement examples | Conditional definitions and observed-data identification limit; original examples not copied |
| [mice pooling](https://amices.org/mice/reference/pool.html) | Analysis-then-pool workflow, total variance rule, input estimates/SE and small-sample degrees of freedom | Exact independent scalar example; multiple-imputation inference not equated with default IterativeImputer |
| [Feature hashing paper](https://arxiv.org/pdf/0902.2206) | §2 signed map, Lemma2 inner-product expectation/variance setup | Actual signed-sum mechanism and explicit hash-assumption boundary |

The JSS mice article landing-page abstract was read, but its Paper link failed to open twice; its full algorithm text is not claimed inspected or needed as an unsupported endorsement. The author-written book subsection and current mice pooling body supplied the relevant evidence. No video was claimed watched; the inspected visual/code articles provide meaningful alternate learning modes without a video quota.

## Quantitative evidence and author checks

`author-calculations.py` executed using read-only shared lesson-tools Python3.12.14/NumPy2.3.5/pandas3.0.1/sklearn1.9.1. Four small actual KNN pipelines, majority baseline, fitted statistics/transformed held-out records and confusion matrices retained in calculated-inputs.json. Actual correct counts38,67,84,85,85 out of86, no chosen deployment winner. Scaler five-value outputs, changed target-encoding arrays, and exact KNNImputer output200 were calculated. A JSON serialization issue was repaired before the successful run; no numerical fit failed or reached an iterative budget. Main displayed program is a complete extraction of equivalent operations from the calculation script, not claimed separately executed verbatim. The target-encoding displayed function is the same arithmetic exercised there. Formal native-example and UI campaigns remain phase two.

Checked constructed arithmetic: raw Q/A/B totals10,001 versus10; scaled2 versus9.0001; changed candidate mass4,400→17; common divisor×2→.5 versus2.250025. Missing target donor removal→300; changedD1 target140→220; unselectedD3 edit null. Rank/log change and categorical distances follow exact formulas. Practice changes standardnew8→sqrt6, minmax1.5; pooledvariance13/3; held-out missingmass formula uses saved parameters. Numeric displays use limited decimals while comparisons use unrounded results.

Complete manuscript and full specs inspected in sequence during author self-review: introduction→ruler→category→missingness→fit boundary→real experiment→deeper branches→changed practice→next link. Specific fixes during authoring: own-target prior leak corrected before writing; replaced accidental unit example without a ranking contrast by checked1mm/100g fixture; removed possible implication of KNN donors requiring complete rows; clearly labeled literal displayed-program execution boundary. No unresolved material content gap is being delegated as a heading or TODO.

Author learning-experience checklist: a novice has an explicit route and local model/metric definitions; quantities carry units and shapes; four investigations expose different mechanisms with unset input-bound predictions and edited entities; observed inputs/derived states are traceable; every investigation has a checked changed and unchanged case; important cautions have one appropriate home; first-pass readiness uses only core skills; alternative resources have inspected content and guidance; accessible and mobile behaviors are specified but rendered perceptibility is unverified. The real comparison includes a majority baseline and an unfavorable raw preparation, without claiming one observed-row difference settles model selection.

## Continuation

Phase two should implement lesson prose and the specified figures/investigations, retain source data/provenance, execute displayed programs and validate JS reproductions, check arbitrary changed/null inputs and actual mobile/keyboard readability, complete formal independent review and only then update implementation/publication status. No runtime code, blueprint, manifest, navigation, shared ledger or handoff was changed by this author. No disposable scratch file was created for this topic; necessary CSV/results/calculation stay with the pending packet. The next manuscript is Cross-Validation & Hyperparameter Tuning.

## Phase two: implementation — 14 September 2026

Implemented from this packet on request. The manuscript, visual specifications, `data-provenance.md`, `author-calculations.py`, `calculated-inputs.json` and `penguins.csv` are unchanged inputs; this section is the only edit to the packet.

### What was built

| File | Role |
| --- | --- |
| `src/learn/data/topics/feature-scaling-encoding-imputation.jsx` | The published lesson: eleven sections in the manuscript's order, nine inline figures, four investigations, two executed programs, one checkpoint and practices 1–9. Default export shape unchanged. |
| `src/learn/data/scaling-models.js` | The pure model layer. Every fitting function returns an explicit fitted object (`fitScaler`, `fitMedianImputer`, `fitOneHot`, `fitPreparation`) and every applying function takes one of those objects plus a record (`applyScaler`, `applyMedianImputer`, `encodeCategory`, `transformRecord`). Nothing recomputes a statistic from the row it is transforming. Validation refuses bad input rather than substituting: a zero divisor, an empty training column, a query whose target cell is present, a single non-empty fold, a missing category with no declared fill, and a non-binary target are all `RangeError`s. Also carries the overlap distance, cross-fitted encoding, signed hashing, Box–Cox/Yeo–Johnson, the rank convention, circular coordinates and Rubin pooling. |
| `src/learn/data/scaling-data.js` | Generated. All 344 penguin rows with missing cells as `null`, the stratified split, the fitted medians/centres/scales for the three rulers, the 86 transformed held-out rows, per-method predictions, the comparison counts with confusion matrices, and the three constructed fixtures. |
| `src/learn/data/scaling-examples.js` | Generated. Both displayed programs with the stdout of their own execution. |
| `src/learn/components/lesson-labs/ScalingShared.jsx` | Draft/active/prediction contract, number and select fields that treat an empty field as *not recorded*, tables, a framed plot and a labelled number line whose viewBox width matches its rendered width. |
| `src/learn/components/lesson-labs/ScalingLabs.jsx` | The four investigations. |
| `src/learn/components/lesson-labs/ScalingFigures.jsx` | The nine figures. |
| `src/learn/components/lesson-labs/scaling-labs.css` | All classes prefixed `sc-`. |
| `src/learn/data/curriculum/blueprints/feature-scaling-encoding-imputation.js` | The blueprint. This author did not edit `blueprints/index.js`; the integration owner registered it there on 14 September 2026, after this section was first written. |
| `public/learn-assets/feature-scaling/penguins.csv` | The unchanged 15,241-byte public CSV, served so the displayed program's `Path(__file__).with_name("penguins.csv")` runs as written. |
| `scripts/verify-scaling-{models.mjs,examples.py,data.py,browser.cjs}` | The four verifiers. |

The leakage boundary is enforced in the model layer as well as in the prose. `fitPreparation` is called with the 258 training rows only; `transformRecord` never mutates the fitted object, and `verify-scaling-models.mjs` asserts both that the frozen fit reproduces all 86 recorded held-out coordinate rows and that a fit over all 344 rows would give a *different* mass centre and a different median — the contrast the boundary exists to prevent.

### Departures from the manuscript, with reasons

| Manuscript | Implementation | Reason |
| --- | --- | --- |
| §2 `d^2(Q,A)=(41-40)^2+(4100-4000)^2=10{,}001` on one display line | Split after the first squared term | The single line measured 332 px inside a 280 px column at 320 px. Measured, not guessed. |
| §4 the three donor distances on two lines | Three lines | Same reason: 305 px in a 280 px column. |
| §7 Yeo–Johnson written directly in `x` | The two branches name `v = x + 1` and `u = 1 − x`, stated in the sentence that introduces each | The negative branch overflowed at 320 px. Naming the subexpression is the specification's own suggested remedy; the formula and its special cases are unchanged. |
| §9 `\bar\theta`, `\bar U` and `B` on two lines | Three lines | 283 px in a 280 px column. |
| §6 “The following complete experiment was executed during authoring through an equivalent calculation script” | “The block below is executed verbatim against the same served CSV before this page is published, and the output shown underneath it is that run's own output” | The claim got stronger, not weaker: `verify-scaling-examples.py` executes the displayed block itself and refuses to pass if the recorded stdout differs from a fresh run. |
| §1 one-hot table, §8 six-row table | Rendered inside `CategoryFigure` and `EncodingFigure` rather than as separate prose tables | Both tables are computed live from the model layer, so they cannot drift from the arithmetic beside them. Every row is present. |
| Practice 6's coordinate given as inline display math | Given as ordinary prose text in the section-6 checkpoint | The inline formula was the one element that made the document scroll horizontally at 320 px with the solutions open. The practice-6 solution itself keeps the formula. |
| — | A readiness-check table, a comparison figure with a zero-origin count axis, and a signed-hash table were added | The readiness statement and the hashing example are in the manuscript as prose; the figure and tables make them checkable. |

Nothing was dropped: every table row, program, caution and practice task in the manuscript is on the page, and no honesty caveat was softened. The reserved-set result is published as it came out, including that one extra correct row settles nothing.

### Checks actually run

| Check | Result |
| --- | --- |
| `node scripts/verify-scaling-models.mjs` | **PASS** — 27 grouped checks; evidence `docs/teaching/evidence/scaling-models.json`. |
| `scratch/lesson-tools/Scripts/python.exe scripts/verify-scaling-examples.py --write` then again without `--write` | **PASS** both times — 2 programs, 24 oracle assertions; evidence `docs/teaching/evidence/scaling-native.json`. The second run proves the recorded output matches a fresh execution. |
| `scratch/lesson-tools/Scripts/python.exe scripts/verify-scaling-data.py` | **PASS** — 344 rows, 258/86 split, four preparations refitted and matched against `calculated-inputs.json`; evidence `docs/teaching/evidence/scaling-data.json`. |
| `npx vite build --outDir dist-scaling` | **PASS**. |
| `DIST_DIR=dist-scaling LEARNING_BASE_URL=http://127.0.0.1:4183 … node scripts/verify-scaling-browser.cjs` | **PASS** — 14 cases, 29 screenshots; evidence `docs/teaching/evidence/scaling-browser.json`. |

The browser pass covers: every displayed program's code and output present verbatim; eleven route anchors; the served CSV matching the checkpointed SHA-256 with `NA` still `NA`; all five recorded counts and every fitted statistic visible; each of the four cautions checked by sentence; investigation 1's raw/scaled flip with the 4,400 g contrast and both nulls; investigation 2's two contrasts, the labelled no-overlap fallback and the refusal to invent a value for an absent target column; investigation 3's exact coordinate, the imputed-mass contrast, the unchanged-coordinate null and all three category states; investigation 4's own-target null, the row-1 contrast, practice 7, a refused empty fold and the declared zero-smoothing fallback; label/line geometry and document width at 1366, 1024, 768, 390 and 320 px; enlarged root text; script-closure isolation; completion persistence and the actual successor link; and both controlled load failures.

### What the screenshots changed

Every one of the 29 captures was opened and looked at. Eight defects were found that way, all of which passed their assertions first:

1. **Every `fontSize` attribute in every figure was silently ignored.** `.sc-figure svg text` sets the `font` shorthand, and a CSS rule beats an SVG presentation attribute, so labels written at 8 and 9 units all rendered at 11 and collided. They are now inline styles, which win.
2. **Figure 1's branch boxes overlapped their own labels** once the sizes took effect; both stages were re-laid out and the “absent” connector was given its own style instead of borrowing the forbidden-arrow style, which had read as a prohibition.
3. **Figure 4's “apply, frozen” arrow pointed into empty space** and the final-comparison box was fed by arrows that contradicted its caption. The whole diagram was redrawn: one bundle, two apply arms, a separate answers rail, and the crossed backward arrow clear of both.
4. **Figure 4c ran its axis title into the zero tick** (“correct / 86 0”); the title moved to its own line under the ticks.
5. **Figure 6's variance bars were on three different scales.** Each `.sc-bar-row` was its own grid, so the label column width — and therefore the track width — changed per row. One shared grid with `display: contents` rows fixed it; the total bar now genuinely fills the track that the within bar fills three quarters of.
6. **The number lines rendered their ticks at about 6 px.** Their viewBox was 560 units wide inside a 300 px panel. The default width is now 300, and the magnified interval is marked on the full-range line with a bracket rather than only named in the title.
7. **Figure 3's coordinate plane had its distance line running through the “green (1, 0)” label**, and figure 5's connectors ran through its axis titles and tick labels. The plane was re-laid out; the rank figure now reserves a gutter at each end for its axis values, names the three axes in an HTML list, and keeps every connector inside the data span.
8. **At 320 px the diagram text was about 7 px.** The two small sizes were raised by roughly 20 %, and four boxes and three labels were resized or shortened to hold the larger type. This first pass did not go far enough, and the claim originally made here — that the smallest label then rendered near 9 px — was wrong by about 2 px at the bottom of the distribution; the independent review measured 6.82 px. The disposition section below records the second pass and the measured result.

A ninth finding came from the same pass rather than from a screenshot: three investigations computed their displayed detail from `active` while grading from `draft`, and never promoted `draft` to `active`, so the detail tables showed the initial fixture after every commit. They now promote on commit, and the “inputs changed” notice comes from the draft/active comparison.

### Still not claimed

- No formal independent review. This is the author's own implementation and its own checks.
- No keyboard-only walkthrough, no screen-reader session, no reduced-motion or forced-colors pass. Focus styles, labelled controls, text alternatives for every figure and scrollable tables are implemented but were not operated by an assistive technology.
- The penguin comparison counts are the content phase's fitted results, re-derived by `verify-scaling-data.py` from the same CSV in the same pinned environment. The browser reproduces the fitted preparation and the transformed rows; it does not run the neighbour classifier, and no classification score is recalculated for an edited record.
- Numerical results are bit-exact only in the pinned environment (Python 3.12.14, NumPy 2.3.5, pandas 3.0.1, scikit-learn 1.9.1). No other library version was tested.
- One fixed stratified split of one curated collection. Nothing here establishes a deployment winner among the three rulers, and the page says so.
- The `docs/teaching/lesson-delivery-progress.json` ledger was not edited by this author; closing the phase belongs to the integration owner, who also made the `blueprints/index.js` registration.

### Implemented source versions (SHA-256)

| File | SHA-256 |
| --- | --- |
| `src/learn/data/topics/feature-scaling-encoding-imputation.jsx` | `2fea8f9e8152f764e89f94e464ed7d9132616864e08c1ced3efd0d831511864d` |
| `src/learn/data/scaling-models.js` | `2016b0426dad0a3c76da8a04dd88ccee56a78c9f75bf81b029007552d56c9d2c` |
| `src/learn/data/scaling-data.js` | `1f0bc9e1ae81883aa0bccbd0ee3daa1140ce9f87b5847203f1da82efdbf93c04` |
| `src/learn/data/scaling-examples.js` | `0ef4de579fe7b3a38a15d10dbc8e6f8265081d6f13a82894cd5d91140aefd2ef` |
| `src/learn/components/lesson-labs/ScalingShared.jsx` | `62889bcf4de73be3adea32d55286b88ce73e54a9481f85ab97b82e7f2bf41448` |
| `src/learn/components/lesson-labs/ScalingLabs.jsx` | `7d3dc5f21495294ba730d485e8e4d72eb512990c0aaec310cede1c6c4f9e0516` |
| `src/learn/components/lesson-labs/ScalingFigures.jsx` | `caa34de92c6a0d4340f38580c2e037faaba42506adbbb10fb4b415e6aa2db162` |
| `src/learn/components/lesson-labs/scaling-labs.css` | `757920fad31c092e8f599df8355f4990917177b4e60df55aedd20e2c22354b2b` |
| `src/learn/data/curriculum/blueprints/feature-scaling-encoding-imputation.js` | `8b87cdc8f5f49cdd7036c63f7f56016e97dabfd2006413e77151010b061310b6` |
| `public/learn-assets/feature-scaling/penguins.csv` | `f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93` |
| `scripts/verify-scaling-models.mjs` | `24c9f425d50c935f48148814b1f3a2c5f7ff345c238f5287495d461d80b9d743` |
| `scripts/verify-scaling-examples.py` | `bf074aa2a638213f406e5b4243cc162dc131ac7bb9c7adfbcb8829d4c83cc045` |
| `scripts/verify-scaling-data.py` | `fc17c2c19aea0e36457b9079c6818117c2ea91cd466d46b04a79637758fc55d9` |
| `scripts/verify-scaling-browser.cjs` | `5ee18d31a316b2f4a505c37ed4b0ee479e9ac56b62d411c0e4e4915f904f49c5` |

A hash proves identity, not correctness. The four evidence files under `docs/teaching/evidence/` record the same bytes, so their recorded results bind to these versions. Nothing is committed.

## Disposition of the independent review — 14 September 2026

[The independent review](../../FEATURE-SCALING-INDEPENDENT-REVIEW.md) was read in full and every finding acted on or answered. It reproduced roughly sixty numeric claims from the served CSV and the manuscript's own arithmetic with zero disagreements, including all 602 transformed held-out values at 0.000e+00, and confirmed the leakage claim in both halves and the late staleness repair by driving all four investigations. Nothing below changes a number.

### Blocking

**B1 — the false pointer in practice 1. Fixed.** The added sentence claimed Investigation 1 accepts practice 1's coordinates; six of its eight numbers are outside the lab's ranges, which the reviewer confirmed live. The lab's bounds are deliberate — they keep it at penguin scale — so the sentence was rewritten rather than the bounds widened: it now says the investigation shows the same mechanism but that its fields are bounded to plausible penguin measurements, so those particular coordinates cannot be typed there. `verify-scaling-browser.cjs` now asserts both that the false phrase is absent and that the honest one is present, so the claim cannot come back unnoticed.

**B2 — the record set. The part that is mine is fixed.** Lines 87 and 142 above said `blueprints/index.js` was not edited; the integration owner had registered the blueprint after this section was first written. Both lines now say so. The ledger entry, its `design.md` hash, the generated inventory and the handoff entry belong to the integration owner and are left alone, as the standard puts stage 6 with the increment owner. The build was remade and the browser pass re-run after the registration, so the current evidence describes a `dist-scaling` that contains it.

### Should-fix

| Finding | Disposition |
| --- | --- |
| **S1** Figure 4's mislabelled forbidden flow, orphaned answers rail, and no arrowheads anywhere in the topic | **Fixed.** A shared `FlowDefs` component defines four arrowhead markers, and every flow in Figures 1, 4, 6 and the I4 donor graph now references one, so direction is drawn. The forbidden arrow points backward into the training box and its label sits directly under the crossed marker, clear of the apply arm. The answers rail now leaves the held-out box itself at x = 326, runs down the right margin past the transformed block, is labelled "answers", and lands on the comparison box with an arrowhead. |
| **S2** I4 drew only non-matching donors into the fold prior, so tracing the edges gave a prior of 0 | **Fixed.** Every donor now draws a dashed edge into the prior box, and a same-category donor draws a solid edge into the numerator as well. Tracing the base fixture now reproduces the stated 1/3. The legend reads "dashed: prior (every donor)", and the browser pass asserts three prior edges and one numerator edge. |
| **S3** the ungraded action stayed live after a graded check and destroyed the verdict | **Fixed by disabling it**, the option the sibling lessons took; the regularization lesson's removal of the bypass was not copied, because looking first without committing is worth keeping for a learner who is not ready to predict. `disabled={Boolean(result)}` in all four investigations. The browser pass asserts the button is open before a commit and closed after one. |
| **S4** `.sc-point-id` was inert because the `font` shorthand outranked it and reset the weight | **Fixed at the root.** The shared rule now uses `font-size` and `font-family` longhands instead of the shorthand, so it no longer resets weight, style or line-height for any `<text>` in the topic, and the identifier rule is given matching specificity. The browser pass measures the computed weight and fill of a rendered identifier. Searching for other rules the shorthand was beating found none: `.sc-point-id` was the only class it silently defeated. |
| **S5** 93 of 181 figure text nodes below 9 px at 320 px, smallest 6.82, against a record claiming near 9 px | **Fixed, and the claim corrected twice.** Every 9.5-unit label was raised to 10.5; Figures 1 and 6 were re-laid out to hold the larger type; and the "expanded below" caption, the smallest label in the topic, was moved out of the SVG into the number line's HTML figcaption, where it no longer scales with the viewBox. Figures also reclaim 24 px of the reader's side padding below 620 px. **Measured after the change, and now recorded by the browser pass on every run: 177 figure text nodes, smallest 11.55 px at 390 px and 9.37 px at 320 px, none below 9 px at either width.** The original claim in repair 8 above is marked wrong rather than quietly edited. |
| **S6** figures read 4, 4c, 4b, and Figure 1's caption pointed at the wrong one | **Fixed by removing two numbers rather than renumbering.** The two §6 additions the manuscript does not specify — the controlled comparison and the encoding donor table — now carry unnumbered captions, so the numbered captions read 1, 2, 3, 4, 4b, 5, 6 down the page whatever order the additions sit in. Figure 4b is the specification's own "second figure" of the F4 pair, so it keeps its letter. Figure 1's caption now points at Figure 4b. This also settles the reviewer's separate objection that a §8 figure was named as a variant of §7's. |
| **S7** Figure 3 drew "red" as a blue dot and "blue" as a green dot | **Fixed.** All markers are neutral grey; the dropped reference corner is a hollow square, so role is carried by shape. The panel says so. |

### Observations

| # | Disposition |
| --- | --- |
| **O1** verifiers could not be re-run without writing | **Fixed for all four.** Every verifier is read-only by default and writes only under `--write`. Read-only runs still execute every check: the models verifier asserts the recorded evidence describes the current sources and counts; the examples verifier executes both programs and asserts the recorded code and stdout hashes match; the data verifier recomputes everything and asserts the module on disk is byte-identical to a fresh regeneration; the browser verifier runs all 17 cases and asserts the recorded source hashes and case count. A reviewer can now re-run the set inside a no-write boundary. |
| **O2** Figure 2's min–max panel named the range, not the maximum | **Fixed**; it names minimum, range and maximum. |
| **O3** I3 graded to ±0.05 while asking for four decimals | **Fixed**; the tolerance is 0.005, and the verdict still discloses it. |
| **O4** I2 printed predicted and computed donor lists in different orders | **Fixed**; both are printed sorted. |
| **O5** I3's absent-category clause stuttered | **Fixed**; "not recorded, so the fitted not_recorded category absorbs it". |
| **O6** Figure 4c's bar colours bypassed the stylesheet and the baseline's colour was unexplained | **Fixed**; `.sc-bar` and `.sc-bar.is-baseline` are stylesheet rules, and the paragraph now says why the baseline is drawn differently — it ignores every measurement and always answers Adelie. |
| **O7** §8's execution claim was dropped and not replaced | **Fixed**; the §8 block now says it is executed and its output pinned by the same verifier, which is true. |
| **O8** `θ̂` and `θ̄` did not compose in the SVG font | **Fixed**; Figure 6 reads "estimate 9, U = 4" and "pooled: mean 10". |
| **O9** no provenance file beside the served CSV | **Not done.** The licence, all three creators, the byte count, the SHA-256, the row count and the `NA` convention are on the lesson page in the Sources block, which the reviewer judged adequate and arguably better. Adding a second copy of the provenance under `public/` would create a file that can drift from the packet's own, which is the record of truth. |
| **O10** dead exports, and a hand-written hash table beside a verified `signedHash` | **Fixed.** `useInvestigation`, `curve` and `signed` are removed from `ScalingShared.jsx`, and §8's signed-hash table is now computed by `signedHash` from the constructed map, so it cannot drift from the arithmetic. The remaining model exports that the UI does not reach (`boxCox`, `l2Normalize`, `rankCoordinates`, the guards) are kept deliberately: they are the manuscript's arithmetic, and `verify-scaling-models.mjs` checks the page's numbers against them. |
| **O11** `readTime` did not match the route paragraph | **Fixed**; it now reads "~50 min core reading · 50–80 min code and practice · deeper branches a second sitting", which is what the route says. |
| **O12** I1's caption promised a message the field does not give | **Fixed**; the caption names the bound the field enforces and adds that the measurement fields are bounded for the same reason, which is also what makes B1's replacement sentence true. |
| **O13** `result.key` was stored but never read | **Fixed**; all three hand-rolled investigations now gate their detail block on `result.key === draftKey`, so a future edit path that forgets to clear the result cannot show a stale table. |
| **O14** I1 offers no point dragging | **Not done, deliberately.** The contract says "by number input **or** point movement", and the number inputs satisfy it while also giving keyboard parity, exact entry and explicit range refusal — all of which a drag would have to reproduce. Recorded as a possible later addition, not a gap. |

### Checks re-run after these changes

| Check | Result |
| --- | --- |
| `node scripts/verify-scaling-models.mjs --write`, then read-only | **PASS** both — 27 grouped checks |
| `verify-scaling-examples.py --write`, then read-only | **PASS** both — 2 programs, 24 oracle assertions, output unchanged |
| `verify-scaling-data.py --write`, then read-only | **PASS** both — the module on disk is byte-identical to a fresh regeneration |
| `npx vite build --outDir dist-scaling` | **PASS**, remade after the blueprint registration |
| `verify-scaling-browser.cjs --write`, then read-only | **PASS** both — **17 cases** (13 before), 29 screenshots |

Four browser cases are new: the ungraded path open before a commit and closed after one in all four investigations; three prior edges and one numerator edge in the donor graph; a shared arrowhead defined and referenced with the identifier weight and colour measured on a rendered node; and the small-type distribution measured and recorded at 390 and 320 px on every run rather than asserted against a target.

### What the re-captured screenshots showed

All 29 were re-taken and the changed ones opened. Three further defects were found that way, none of which any assertion caught:

1. **Figure 6's boxes were too tight for the larger type** — "completion 1" sat on "symbolic" and "analysis" on its estimate. The figure was re-laid out on a taller grid with the pooled box beneath the analyses and the crossed shortcut below that.
2. **Figure 1's `sex_not_recorded` sat on its box border** and "training answer, into fitting" overflowed its box on both sides. Both boxes were made taller and the target label split across two lines.
3. **The I4 donor graph's row chips and its closing note ran past the drawing's right edge** and were clipped by the SVG viewport. The chips lost their redundant "row" prefix, and the note moved into the panel's HTML paragraph.

### Still not claimed, after this round

Unchanged from the section above, with one addition: the figure type is now measured rather than asserted, but no screen-reader session, second browser engine or 200 % zoom pass has been run, and the version-independence of the real-data result is still untested on any library version other than the pinned one.

### Implemented source versions after the disposition (SHA-256)

| File | SHA-256 |
| --- | --- |
| `src/learn/data/topics/feature-scaling-encoding-imputation.jsx` | `22b7fc178c0664a7d87d881d42997f908470c67dffbd25eb3e62f5db8cc43dcb` |
| `src/learn/data/scaling-models.js` | `2016b0426dad0a3c76da8a04dd88ccee56a78c9f75bf81b029007552d56c9d2c` |
| `src/learn/data/scaling-data.js` | `1f0bc9e1ae81883aa0bccbd0ee3daa1140ce9f87b5847203f1da82efdbf93c04` |
| `src/learn/data/scaling-examples.js` | `0ef4de579fe7b3a38a15d10dbc8e6f8265081d6f13a82894cd5d91140aefd2ef` |
| `src/learn/components/lesson-labs/ScalingShared.jsx` | `2560e271b1e92d9557b08f174b07989d24efb94903304f12cc4dd939c9de19d6` |
| `src/learn/components/lesson-labs/ScalingLabs.jsx` | `c9682e0a70ccb4e8411e231a2446ee24d2e25dd1b2eb436efbef96d2710e7534` |
| `src/learn/components/lesson-labs/ScalingFigures.jsx` | `3b32bab8f9375e9786301c85c3f6c14ef1ad1f9f2d110e41b80f082aff822723` |
| `src/learn/components/lesson-labs/scaling-labs.css` | `aa9ad22ed790347b211c38f8690416dc91c02afc82e5fc686786abce835b6a22` |
| `src/learn/data/curriculum/blueprints/feature-scaling-encoding-imputation.js` | `8b87cdc8f5f49cdd7036c63f7f56016e97dabfd2006413e77151010b061310b6` |
| `public/learn-assets/feature-scaling/penguins.csv` | `f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93` |
| `scripts/verify-scaling-models.mjs` | `6d21caabdc6858eddfa558b896af09854e7b7dbb4c6e09ccf9431f7ae04f7b18` |
| `scripts/verify-scaling-examples.py` | `55ad5ebc51eaa9b08325ac27e7d73b96f507c55cbede14bafbe2b7be5ab31921` |
| `scripts/verify-scaling-data.py` | `1364b4e0d5232086b08a3dad0d599388111a6ce63493e622e31f246705b7bc3d` |
| `scripts/verify-scaling-browser.cjs` | `af34b2251fb273960c469bd97664c6af548bb321a2179f48b14cc5f4dd19fd36` |

These supersede the table in the phase-two section above. The four evidence files under `docs/teaching/evidence/` were rewritten against these bytes. Nothing is committed.

