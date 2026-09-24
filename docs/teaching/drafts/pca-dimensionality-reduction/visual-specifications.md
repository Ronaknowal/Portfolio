# PCA: visual and investigation specifications

## Current live-exploration contract — 21 September 2026

This dated UX amendment supersedes earlier prediction-entry, grading, commit-to-reveal and prediction-retirement requirements in this document. There is no learner prediction feature, even optional. Historical evidence below records the earlier interface and remains history; it is not the current acceptance contract.

Rotate or fit a direction on editable points and follow scores, residual segments and squared loss immediately; change units or standardization and inspect the leading direction; move an error budget and inspect the first qualifying count and individual validation wines; vary spreads and label rules and inspect retained-coordinate collisions.

Keep separate independent practice, model predictions, scientific validity checks and training/validation/held-out information boundaries. Meaningful valid control changes must reach the visible calculation and topic-specific diagram together. Natural algorithm Step/Back/Run actions remain where they expose a process; they must never require a learner guess. Reset restores a coherent initial state. A graph or number must not silently describe obsolete inputs; invalid inputs show an error and either clear invalid outputs or explicitly retain the last valid result. See [the current migration evidence](../../LIVE-EXPLORATION-CLASSICAL-EARLY.md) for implemented checks and limitations.


Content-first handoff, 12 September 2026. Read with [the manuscript](lesson.md) and [design/research record](../../PCA-LESSON-DESIGN.md). **These are specifications, not implemented or browser-verified components.** All desktop/mobile perceptibility, interaction, keyboard, runtime and performance checks below are phase-two acceptance criteria. No screenshots or rendered assets were produced in phase one.

The representations have different jobs: a ruler exposes projection; a shape/operation diagram exposes array multiplication; a metric experiment exposes scaling; an empirical error curve supports a budget decision; a label strip exposes task-information loss. Do not convert them into one repeated text-output simulator. Reuse accessible controls and typography where useful.

## Data and numerical conventions

- Rows are observations. Store feature-space directions as rows in code, matching scikit-learn `components_`; the manuscript's symbolic `V_k` stores them as columns. Explain the transpose at the code boundary.
- Center using the fitting mean. Sample covariance and reported PC variances divide by `n−1`; `StandardScaler`-equivalent scales divide by `n`. For complete nonconstant columns, this convention changes absolute covariance by a common factor, not the PCA directions or ratios.
- Exact fixtures are specified below. Wine is **real data** in [wine.csv](wine.csv), with full [provenance](data-provenance.md). Its calculated outputs, fold membership and retained numerical reference are in [author-calculations.json](author-calculations.json). The Gaussian example is explicitly simulated. None of these plots is an empirical runtime comparison.
- Do not use rounded plotted numbers as model inputs. Format only at the display boundary. Snap values only when identifying a specified exact tie or suppressing machine-scale negative zero; retain the true model values for comparisons.
- Direction signs may be chosen for stable presentation: for isolated eigenvalues, make the largest-magnitude direction coefficient positive, with first-index tie breaking. Paired scores flip with the direction. At exact repeated eigenvalues, show nonuniqueness rather than implying that the chosen deterministic drawing is the only solution.
- For analytic two-dimensional models, independent references can use the symmetric 2×2 eigenvalue formula and a native SVD. Compare projection matrices or reconstructions rather than signs. A browser model must not import the entire native calculation record, research ledger or another lesson's examples.

## F1 — A point, its shadow and its saved coordinate

**Location:** section 2 immediately after the four observations; before the learner sees a dot product. **Question:** what survives when a two-dimensional observation becomes one number?

Use A=(1,1), B=(2,0), C=(4,4), D=(5,3), centroid μ=(3,2), v₁=(1,1)/√2. Show all original points, the line through μ, their perpendicular feet, and one-dimensional score positions. Keep A selected initially. Draw its residual as a dashed segment with a right-angle marker. Name original A, mean, projected A and score; do not make the learner infer the correspondence from color.

Axes are the two sensor readings in the same arbitrary measurement unit. Equal aspect ratio is essential: Euclidean perpendicularity must appear perpendicular. A centered-coordinate alternative can be a manual step, but the original-coordinate line must pass through (3,2), not zero. Score strip positions reflect actual scores −3/√2 and +3/√2. Stack coincident labels A/B and C/D with a connector to their shared score; do not jitter the numerical positions.

Caption: “A and B have different readings but the same diagonal score. The dashed segment is the difference the one-coordinate representation loses.” Text alternative gives A's center, score, projection and residual; manuscript tables supply all four rows.

**Mobile:** put square plot above score strip. Show A's detail below; keep other point labels short. Never shrink a long horizontal pipeline until labels become unreadable.

**Checks:** mean (3,2), A reconstruction (1.5,.5), each SSE contribution .5; all feet on the line; residual dot v₁=0; score order and coincident labels correct. Phase two must inspect desktop and narrow screenshots at ordinary zoom for the residual/right-angle correspondence.

## L1 — Projection workbench

**Location:** end of section 2. F1 may be the initial visual state of this investigation if it is immediately visible before interaction. Do not add a duplicate diagram merely to meet a count.

**Investigation:** choose a direction, predict whether it loses more or less than a reference direction, and inspect what the choice preserves. It is an actual projection experiment, not a series of prewritten captions.

**Inputs/state:** editable 4-point dataset; reference angle initially 0°; proposed angle initially 45°; selected observation A; centered/original display; optional common translation (0,0). Angles are measured from the first axis, periodic over 180°. Dataset editing applies atomically. Allow at least four and at most twelve points with finite bounded coordinates (e.g. −10..10) and explicit Add/remove controls. Numeric inputs and point selection must support keyboard use; dragging is an additional convenience.

**Record and compare:** provide an unset radio choice “less error / same error / more error” and a “Compare prediction” action. Capture the prediction and both angles against the committed dataset. Reveal reference SSE, proposed SSE and their difference after commitment; compare using a documented numerical tolerance. Keep the two projections side by side or sequentially selectable on a phone. Before commitment, do not expose the proposed SSE in the result readout. After changing data or either angle, visibly invalidate the old comparison and request a new prediction; do not silently grade a previous answer against a new input.

**Entity interaction:** learner enters coordinates or moves a selected point, and chooses arbitrary angles. “Fit best direction” can reveal the true PCA optimum after the prediction comparison; it must compute from the active dataset rather than selecting a scripted preset. Exact ties show a family of equally good directions. Store a small input history so Back restores the prior data/angle state; Reset returns the original dataset, unselected prediction, translation and reference.

**Held fixed in an angle comparison:** points, centering, units, retained dimension=1. Common translation is a separate experiment that changes μ while preserving centered points. Explain that editing one point changes the data, then compare directions on that same edited data; do not compare losses on different point sets as if only the direction changed.

**Linked outputs:** square scatter; selected point's centered vector, score, projection and residual; distribution of scores on a line; retained squared length plus SSE = total. The model emits one state consumed by all panels. Buttons never change the math independently of the points.

**Fixtures/expected contrasts:**

| Dataset and comparison | Expected result |
| --- | --- |
| Original, 0° vs 45° | SSE 10 vs 2; retained totals 10 vs 18 |
| Original, 45° vs 135° | SSE 2 vs 18 |
| Original, 0° vs 90° | SSE 10 vs 10, a genuine equal-error comparison |
| Original plus common translation (2,−1), refit | Same centered covariance/eigenvalues/SSE; μ becomes (5,1) |
| Exactly collinear (−2,−2),(0,0),(2,2) | Diagonal SSE 0; perpendicular SSE 16 |
| All points identical | Mean reconstructs exactly, total=0; show undefined variance fraction, no invented 100% preference |

These analytic states were checked during author calculations. Arbitrary input guards, keyboard behavior, history and rendered consistency are not yet implemented. Phase two must test them, including one asymmetric learner-edited cloud against native SVD.

**Feedback:** “Your proposed direction loses 2 units²; the reference loses 10. It saves 8 units² of squared error on these same four observations.” Pair this with the actual residual lengths. For equality, explain that equal objective values are possible without identical projected coordinates. Do not print a canned caution after each observation.

**Transfer:** translate, refit, and predict what changes; then change a single point and identify the newly preferred direction. Completion is not automatic evidence of mastery.

## F2 — The conserved quantity behind the two objectives

**Location:** section 3, beside the three-direction calculation. Use horizontal (0°), diagonal (45°), perpendicular (135°) views of the same centered points, followed by three stacked bars with equal total length 20. Retained values 10,18,2 and residual values 10,2,18 fill those bars exactly. Give them text labels and distinct fills/patterns.

This representation explains the conservation equation visually; a chart of percentages alone loses the right-triangle mechanism. Keep a single selected point's right triangle visible next to the total accounting. Bars start at zero and share an axis; do not autoscale each bar.

**Mobile:** stack the three comparisons; retain equal plot scale and identical total-bar width. Text alternative supplies all values and says that changing the ruler redistributes a fixed total. Checks: bar sums, squared lengths and relation to sample variance via division by 3. It can share pure geometry with L1 but must not hide the comparison behind an unrelated tab.

## F3 — Shapes and the fitted transform

**Location:** section 4. **Question:** which matrix multiplies which, and which values are reused for a new observation?

Use a small matrix view of the actual 4×2 data; subtract a 1×2 mean; multiply 4×2 centered rows by the 2×1 retained direction to get 4×1 scores. Reconstruct through 4×1 times 1×2, then add the mean. Highlight A's row throughout. “Fit” produces mean and direction; “Transform a new observation” reuses both. A second branch traces (6,4) to (5.5,4.5).

No arbitrary equal-size generic boxes: cell layout, dimensional labels and highlighted row/column expose the operation. This is static or manually stepped; automatic animation is unnecessary. On mobile, place operations vertically with shape labels and preserve the highlighted correspondence. Text alternative describes dot product and outer-product reconstruction. Verify all shapes, orientation, mean restoration and the distinction between a score vector and a direction.

## L2 — Units define a metric

**Location:** section 5, after the exact rectangle. **Question:** can changing only the unit change the answer, and what does standardization do?

**Fixture:** four corners (±a,±b), initially a=2, b=1. A positive multiplier m, initially 1, changes the second input column to ±mb. Allow a,b∈[.25,4] and m∈[.25,10] through labeled sliders plus numeric fields. A raw/standardized mode selector is explicit. Prevent zero widths and nonfinite entries with inline messages. An advanced point edit can reuse the L1 editor but is not required; independent width/height and unit controls already act on actual data rather than switching presets.

**Prediction:** before applying a proposed m, learner selects “first axis / second axis / no preferred axis” for the new PC1. Freeze the current parameters while the prediction is committed; Apply computes and compares the leading covariance eigenvalues. Also display the previous and proposed retained variance fractions. Changing a,b or mode resets the pending result.

**Mechanism view:** draw the raw observation rectangle with true equal coordinate scaling. Standardized mode gets its own square plot labeled in fitted standard deviations. Link a selected point to its centered/scaled row. Coefficient arrows and separate variance bars make the dominant direction visible. A mode switch is a different geometry, never an axis relabel on unchanged point positions. Do not squash an elongated raw rectangle into a square-looking scatter through independent automatic axis scales.

**Required contrasts/nulls:** raw a=2,b=1,m=1 -> first-axis PC, 80%; m=2 -> equal variances (tie); m=10 -> second-axis PC, 400/416=96.1538%. Standardized at all three multipliers -> equal variances; same square (up to roundoff), no unique PC1. Common rescaling of both raw features -> same directions and ratios, scaled absolute variance. These specific alternatives were checked analytically/numerically in phase one.

**Feedback:** explain which squared errors were reweighted; at ties, show an arbitrary valid axis with “any direction is equally good” or show both equal variance bars without picking a winner. A “PCA prefers the diagonal” result at the standardized tie is incorrect.

**Transfer:** choose a different rectangle width, predict the tie multiplier `m=a/b`, then test it. Reset clears prediction and restores a=2,b=1,m=1, raw mode. Mobile puts source data/controls above linked square view and equal-baseline variance bars. Keyboard labels must include “unit multiplier,” not unexplained `m` alone.

## F4 — Real Wine measurements: scale, scores and coefficients

**Location:** section 5 after the real-data outputs. **Question:** why do two valid analyses report such different explained-variance percentages?

Calculate raw and training-on-all-178 standardized PCA from the retained Wine values. This is explicitly descriptive analysis of the full collection. It is not the train/validation experiment in L3. Default view has two small charts: leading raw variance shares and standardized shares, each with a 0–100% baseline. Raw PC1=99.8091%; standardized PC1=36.1988%, PC2=19.2075%. Show a separate standardized two-score scatter, axes naming PC1/PC2 and their percentages. Data attribution remains accessible.

The first view can label “55.41% retained by this picture” without claiming a sharply separated set of three classes. Optional cultivar overlays use three shapes plus colors and a clear legend; hide labels by default until the learner has inspected the unsupervised geometry. No target column enters the PCA model.

The accompanying coefficient view needs all thirteen feature names, signed bars about zero and a text table; do not force long feature names into a tiny heatmap. A selected point's table lists original values, standardized values and coefficients. An optional selection control can connect a point and row; this is exploratory inspection, not another claimed investigation with artificial grading.

**Mobile:** score plot retains readable axes and zero; coefficient chart stacks below. Bars give full labels alongside values. Do not draw all 178 row labels at once; keep point selection and a searchable/step-through observation selector with clear IDs.

**Checks:** no source/label mixup; CSV round-trips to bundled Wine data; all-rows fit distinguished from L3 fit; ratios, total, signed coefficients and scores agree with native calculations; percentages remain readable at 390px. Verify any claimed cultivar contrast from the actual scatter instead of adding promotional text based only on labels.

## L3 — Choose a component budget, then inspect the loss

**Location:** section 6 after the train/validation program. **Question:** what is the smallest stored coordinate count satisfying my budget, and which measurements does it distort?

**Fixed data protocol:** the exact 133/45 split in `author-calculations.json`, matching `train_test_split(...test_size=.25, random_state=42, stratify=wine.target)` in scikit-learn 1.9.1. Fix fitted training mean, `ddof=0` scales and unwhitened orthonormal PCA basis. Freeze all 13 features; avoid adding a feature-subset control unless all recalculation and comparison semantics are implemented and reviewed.

**Prediction and meaningful inputs:** budget slider plus numeric input q∈[.01,.60], initially .10; predicted minimum k is an initially unset integer input 0..13. Learner commits before results reveal. Compare against the smallest k whose exact validation loss ratio is ≤q. Display selected k, preceding failing k and their ratios. The meaningful user-designed variation is the budget, supplemented by selection of a validation observation and its feature to diagnose; the change is not just a named preset. Also allow manual k selection after commitment to explore loss versus size.

**Geometry:** use an empirical line/point plot k=0..13 against validation error as a fraction of the mean-only baseline. Include k=0, ratio1; k=13, ratio~0; horizontal user-budget line; selected crossing and preceding failure. Axes start at zero. A companion table/secondary plot shows training cumulative variance; **do not label validation error reduction as training explained variance**. Separate charts prevent a dual-axis illusion.

For q=.10, k=7 gives .1265121, k=8 gives .0959546, selecting8. For q=.06, k=9 gives .0723539, k=10 gives .0520663, selecting10. At k=2 the ratio=.4280959. Native values are retained in the calculation record. A slider choice of exactly a rounded label must not cause a false crossing: compare full precision and display more digits near the boundary.

**Budget-null case:** q=.10 and q=.11 select the same k=8, although the budget line moves. Explain this stepwise response. Changing a display-selected observation leaves the globally selected count unchanged. At k=13, all validation observations reconstruct to numerical precision because the 13-feature basis is complete; this is not a proof of low-dimensional generalization.

**Residual inspection:** selected observation shows thirteen original-versus-reconstructed values in a aligned feature table with native values and standardized residuals. A signed residual bar chart can supplement it. Do not combine incompatible chemical units on one raw-value error axis. Offer raw units per selected feature or a common standardized-residual scale. Explain inverse standardization with a selected cell. Preserve exact UCI/scikit identifiers and fold membership in any data export.

**State:** new budget invalidates old prediction before a new reveal; manual k exploration is visibly separate from graded minimum-count prediction. Reset restores .10, unset prediction and first validation row. No random resplit control is needed; alternate splits can be native practice. Do not pretend a budget was specified prospectively if it has already been adapted to the observed curve.

**Accuracy/accessibility:** use native PCA references and direct reconstruction SSE, plus the mean-only ratio. Check monotonicity for this fixed nested basis, all 14 counts, endpoint reconstruction, inverse scaling and 10%/6%/11% thresholds. Native numerical checks are recorded; actual component controls, mobile layouts and browser model equivalence are deferred. On phones, keep the budget control, crossing plot and selected answer together, then put per-record inspection underneath. Provide a table alternative and announce result changes without an ARIA update on every drag frame.

## F5 — Reading a coefficient, a correlation and a biplot

**Location:** section 7, optional depth near the biplot discussion. Begin with the four-point data to avoid thirteen crossing arrows. Use two score coordinates, A selected, and the first-feature arrow `(1/√2,1/√2)` in PC coordinates for the stated v₁/v₂ convention. A's dot product with this arrow is −2, the centered first feature; add μ₁=3 to recover1. Show the numerical pair beside the vectors.

This chosen biplot has observation coordinates Z and feature vectors from rows of V. State that scaling. Do not label their angle as an exact feature-correlation formula. A separate numeric comparison shows first-feature coefficient .7071 versus feature/PC1 correlation .9487. If arrows are enlarged, state multiplier and keep the dot-product readout in original coefficient coordinates.

**Mobile:** one feature and one observation at a time, with a persistent text row; no hover-only explanation. Text alternative states the dot product and recovered feature. Check arrow orientation, coordinate sign pairing and use of standardized versus original units. F5 need not become another full lab; its purpose is decoding a standard representation.

## L4 — Variance is not the label

**Location:** section 8 after the 99% example. **Objects:** rectangle observations at (±a,±b), initial a=10,b=1, and binary labels from either sign of first coordinate or sign of second coordinate. Use a∈[2,12], b∈[.25,1.5] so a>b and PC1 is unambiguously first-axis in the core comparison. This deliberately avoids imposing an arbitrary component at a tie. For ties, refer to L2.

**Prediction:** learner chooses retain PC1, PC2 or both and commits “labels distinguishable / conflicting labels share a coordinate.” Compute the answer from grouping the retained coordinates with the active labels, not from preset prose. Reveal two representations: full 2D points with class shapes, and retained coordinate strip with coincident observations shown in stacked labels anchored at their exact shared position. Also show retained variance fraction, separately from label collisions. This is distinguishability of this finite constructed set, **not** a trained classifier's test accuracy.

**Entity controls:** numeric spreads a and b; label rule; retained dimensions. Changing spreads lets users create a new data geometry. Changing label rule keeps the observations and PCA fixed while changing usefulness. Reset restores a=10,b=1, second-coordinate labels and an unset prediction. Any edit invalidates the previous comparison; record configuration and prediction together.

**Contrasts:** at (10,1), PC1 retains100/101 and collides labels defined by y; PC2 retains1/101 and distinguishes them; both retain100% and distinguish. Labels defined by x reverse which one-coordinate choice is useful. Changing y-labels to x-labels leaves eigenvalues/directions unchanged (null effect on fit). Varying a while retaining PC1 cannot repair the y-label collision. Exact fixtures/identities were checked in phase one; browser grouping, invalid inputs and display still need implementation checks.

**Feedback:** name the actual pair that now shares one score and different labels. Do not claim an interview-style universal guarantee or empirical classifier result. Transfer question: which change improved the representation's usefulness without changing its variance objective?

**Mobile:** full geometry above 1D strip; labels use shapes plus text. Plot must maintain actual aspect ratio but can give the narrow vertical separation an annotated magnified inset. Mark the inset's changed scale; never secretly stretch the main plot. Check that ±1 remains perceptible beside ±10 at the chosen rendered width. A small fixed square plot with readable marker separation is preferable to filling the entire page width with elongated axes.

## F6 — Sample spectrum versus population symmetry

**Location:** section 10.3 beside the seeded Gaussian example. Generate `default_rng(23).normal(size=(40,20))`, center, then exact PCA. Draw sample explained-variance bars for ranks1..20; horizontal population-per-direction share=.05. First two sample shares sum .2459146. Caption identifies finite-sample simulation and notes that the population covariance has no preferred direction. The reference is an equal-population-variance benchmark, not a significance threshold or a predicted sample spectrum.

Use a 0-based y-axis sufficient to see .05 and the leading sample values; no implied elbow annotation. Label axes “ordered sample component” and “fraction of sample variance,” with a separate label for the population reference convention. Text alternative gives setup, seed, reference and first-two result. On narrow screens, render every bar but only sensible tick labels; use a table for exact inspection. Phase two checks actual numerical generation and desktop/mobile readability; no confidence interval or inference test is claimed.

## F7 — The residual reveals a different anomaly

**Location:** section 10.5. A constructed normal diagonal mode through zero, observation P=(3,3) on it and Q=(3,−3) across it. The usual direction is fixed v=(1,1)/√2, not refit to P/Q. Show P's score3√2/residual0 and Q's score0/residual squared18. Include a zero baseline in the residual display and use equal geometric scales.

The graphic explains two diagnostics with an exact example. It is not real sensor data, a calibrated detector or a clinical claim. Do not draw a red rejection zone without a justified threshold. Mobile stacks the scatter and the two aligned score/residual rows. Check residual projections, units and the fixed-reference boundary. A static diagram is sufficient; arbitrary animation or another lab would add little.

## Implementation ownership and bounded validation

When phase two is requested, use the existing stable-ID lesson path, a topic-owned blueprint, semantic `pca-models.js`, `pca-examples.js` and `PcaProjectionLab` / `PcaMetricLab` / `PcaBudgetLab` / `PcaTaskInformationLab` components, or an equally clear topic-owned decomposition. These are intended responsibilities, not instructions to create a file for every tiny figure. Share pure 2D projection utilities between this topic's views; do not import another lesson's full models. Keep data/logic separate from rendering. The full Wine dataset is tiny; measure before introducing a heavy eigensolver or worker. Fixed 13-feature Wine outputs may be prepared from documented native inputs, with browser reconstruction checked against them; editable 2D data must be computed live.

Keep this packet out of browser imports. Transfer the approved data to a topic-owned runtime asset only in phase two; keep attribution and offer the CSV download. The manuscript's relative documentation/data links need real production destinations, not links into `docs/` that the website cannot serve. Retain current topic IDs, route/module order and progress identity. Do not change publication merely because this packet exists.

Phase two must perform native/model equivalence, example output checks, fixture contrasts and nulls; author and independent correctness **and** learning-experience reviews; actual desktop/phone/keyboard inspection of every new visual; source-bound fixes and relevant curriculum/build/import/loading checks. Reuse the bounded author calculations as inputs, not as evidence that these unimplemented items passed. Render plots before asserting that the claimed feature is perceptible. The implementer may improve a specification with documented reasoning; preserve the learning outcome and reconcile the checkpoint after material content changes.
