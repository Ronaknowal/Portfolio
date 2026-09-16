# Initialization: visual and investigation specifications

Prepared content only. No React, SVG, browser lab, generated chart asset, or runtime registration is implemented. All numerical inputs named here are in `calculated-inputs.json`; the complete producer is `initialization-experiments.py`. Do not redraw empirical curves from a remembered theory.

## Shared interaction contract

Mount each investigation only at its explanatory home, and defer the optional width activity until requested. Use topic-specific marks rather than repeating a console-like lab. Predictions begin unset and are bound to the entire input/configuration revision. “Reveal” is disabled until a prediction is submitted; show the learner's answer, actual result, and a mechanism-based explanation. Numeric grading uses stated absolute/relative tolerances, not displayed rounded strings. Changes to any contributing input invalidate a previous prediction/result; clearing only a label is insufficient.

Reset returns a meaningful unsolved fixture, an unset prediction, and no successful score. A preset selection is exploration, not successful learner construction. Where a task asks for an edit, require an actual entity change and evaluate the resulting constraint. Keep prediction and construction outcomes distinct. Respect reduced motion, provide buttons/keyboard equivalents for sliders and draggable marks, use labels/shapes as well as color, include a data table or explicit equations, and announce result changes politely. Avoid automatic moving traces.

No browser training, random resampling, large matrix decomposition, or 54-fit sweep. Read compact selected records on demand or implement the tiny explicit two-dimensional formulas. Phase two must check packet parity, keyboard use, input bounds, input invalidation, null/contrast fixtures, mobile flow, legibility, and lazy-loading costs. Author calculations below are not claims that those UI checks already passed.

## 1. Forward and backward layer traces

**Home/question:** immediately after the first signal table. How can modest per-layer scaling accumulate through depth, and does the backward probe tell the same story?

**Representation:** two aligned plots sharing layer index 0–20. Upper: pooled activation second moment; lower: RMS of gradient with respect to that layer's activation, for the fixed scalar probe. Explain the output cotangent at the end of the lower chart. Zero values appear on an explicitly separate zero rail; never substitute an arbitrary epsilon and label it as a measured value. Positive values use log10 scale with human-readable powers-of-ten ticks. Add a table of raw values and seed selector.

**Source:** `propagation.records`, six schemes × three seeds; input/cotangent seed 71; float64, batch 128, width 64. No timing axis. Each series retains the original exact record and rounds only labels. Variable is `second_moment`, not `variance`. Optional statistics selector may expose variance/mean/zero_fraction but must change the title, units, and prediction scope.

**Prediction:** initially seed 1 with Kaiming as a visible reference; candidate `small` is selected but unrevealed. Ask whether its layer-20 second moment is <0.1×, within [0.1×,10×], or >10× its own input value. Submit then grade using actual ratio. Input scheme/seed changes clear the answer and hide the result again. The question does not promise that “within range” establishes good training.

**Checked outcomes:** seed-1 small ratio ≈9.33e−51; large ≈8.49e17; Kaiming ≈0.736. Zero has exact 0. The lower graph starts near 9.54e−26, 9.10e8, and 0.848 respectively at input; its output value is the supplied cotangent RMS ≈0.994535 for every scheme, including zero. Annotate why a zero output can still have a nonzero externally specified gradient with respect to that output.

**Null/contrast:** repeat the same record/prediction without changing input—must give identical results, not a fresh random curve. Small↔large changes direction; switching Kaiming to orthogonal does not imply an improvement label. The orthogonal layer-20 mean square 0.2316 is displayed honestly.

**Learner edit:** an optional local idealized recurrence builder asks the learner to choose a weight variance for width 100 so that `(100*variance/2)^20` stays between 0.9 and 1.1. Starts at variance 0.01, unsolved. Permit numeric variance in [0,0.1]; 0.02 solves, 0.01 contracts, 0.04 grows. Explicitly label this panel an assumption-based calculation, not an additional measured network; do not overlay it indistinguishably on observations.

## 2. Four values: second moment versus variance

**Home/question:** statistics refresher. What changes when ReLU moves negative values to zero?

**Representation:** four persistent identified dots, a zero origin, and a separate mean marker. Before/after rows show values; selectable squared-distance bars measure from zero or from each row's mean. The values and sums remain text-readable. No histogram bins that hide the four exact observations.

**Source/formulas:** `fixtures.relu_input/output` from [-2,-1,1,2]. Mean, mean square, and correction-zero variance. No stochastic sampling or distribution inference.

**Prediction:** choose separately whether second moment halves and whether variance halves; both answers initially unset. Grade two distinct statements. Reveal input (0,2.5,2.5) and output (0.75,1.25,0.6875) in mean/q/variance order. Feedback for “both halve”: clipping shifts the mean, so subtracting mean squared changes variance further.

**Construct:** ask for a symmetric four-value input with larger absolute outer values while retaining the inner pair ±1; start outer magnitude 2, target output variance 1.5, unsolved. Numeric outer magnitude [1,5], step 0.1 or direct edit. Outer magnitude 3 solves within 1e−9. Preserve four entities rather than a formula-only slider.

**Checked contrast/null:** [-3,-1,1,3] has q5→2.5, variance5→1.5, mean0→1. Toggle identity activation: values and both moments unchanged. Changing one positive value independently can break symmetry, so a free-edit mode must compute actual sums rather than apply the “half” rule. That mode has no default automatic correctness badge.

**Mobile:** stack before and after; keep dot IDs and numeric rows aligned. Screen-reader equivalent lists every value and both centers.

## 3. Circle, ellipse, and gated directions

**Home/question:** average preservation versus every-direction preservation.

**Representation:** input unit circle; output ellipse under diagonal M; labeled editable input arrow and corresponding output arrow. Beside it show singular values, average squared singular value, and the gain for this vector. A second view draws local directional arrows through the ReLU gate at a specified base point. Do not treat the finite output vector as the local Jacobian action.

**Computed model:** lower gain `s` in [0.05,1], larger gain `sqrt(2-s*s)`. M is diagonal, no hidden rotation in the initial activity. Input vector components bounded [-2,2]; reject both zero for gain calculation with useful text. Exact norm ratio `norm(Mv)/norm(v)`. Repeat count 1–8 uses diagonal powers. If adding rotation later, rotate input/output consistently and verify the formula.

**Prediction:** base s=sqrt(0.1), v=(0,1), depth1; ask which vector shrinks even though mean squared singular value equals1. Answer initially unset. After result, ask the learner to make the smaller-direction five-layer gain ≤0.001 while preserving the average squared gain. Initial s≈0.316 gives ≈0.003162, unsolved; s0.2 produces 0.00032 and larger1.4. Check both constraints plus a meaningful changed s; a success message explains the direction tradeoff.

**Nulls/contrasts:** s1 gives identity and all gains1. At s0.2, v=(1,0) grows1.4 while v=(0,1) shrinks0.2. At ReLU base point (-1,1), J=diag(0,sqrt2), singular values [sqrt2,0]; at (1,1), J=sqrt2*I. Avoid derivative ambiguity by rejecting zero base coordinates for this task or explicitly explaining the chosen derivative convention.

**Source:** exact `fixtures.geometry_interventions`, `average_gain_matrix`, `relu_scaled_identity_jacobian`. Gaussian spectrum is a separate saved 64-dimensional example, with histogram plus min/max and full text table; no claim that the 2D ellipse depicts that matrix. QR Gram maximum error is a numerical identity check, not an error-rate metric.

## 4. Hidden-unit symmetry and the first gradient

**Home/question:** why two feature slots can remain one feature.

**Representation:** two small neuron paths with equal incoming/outgoing weights, labeled tanh outputs, prediction, and separate incoming gradient arrows. Step only one update at a time. Distinct-row case changes the actual incoming weights; zero-head case distinguishes blocked hidden gradients from nonzero head gradients.

**Source:** `fixtures.symmetry` and `zero_readout`. These are analytic/autograd calculations for input1, target1, half-MSE, not a trained benchmark. In the initial identical case both incoming gradients −0.2541693849; distinct incoming [0.1,0.3] yields [−0.2621811825,−0.2423390161]. Zero head hides no gradient: hidden [0,0], head [−0.0996679946,−0.2913126125].

This is a guided trace, not another graded construction lab. Let learners predict which arrows disappear before revealing the zero-head case, without accumulating a site-wide mastery score from clicking Next. Reset clears the reveal. A text equation and values suffice on mobile; no force-layout graph engine.

## 5. Actual training: probabilities and correct counts

**Home/question:** does starting with a reasonable scale guarantee the best validation result?

**Representation:** distinguish the four-hidden-layer digit architecture from the twenty-layer diagnostic network. A compact image strip uses actual CSV pixel rows only; show integer target labels. Plot saved training/validation CE against exact update checkpoints, with discrete point markers. Use a separate count panel for validation correct/120. Do not interpolate a dense measured curve between five checkpoints.

**Source:** `digit_fits` records. Choose seed and scheme; reveal alternate seeds. Same head within seed and same data across all runs. No synthetic digit assets. Saved CSV IDs provide provenance.

**Interaction:** comparison and interpretation, not “tune until green.” Prompt why tiny training CE can coexist with larger validation CE. Display seed1 small/large cases and the differing correct-count ranges without automatically ranking one initializer as universally best. The all-zero model's equal logits create argmax class0; the 12/120 count follows the balanced validation split. No browser fitting or benchmark claims.

## 6. μP shape and update workbench

**Home/question:** which quantities change when width changes?

**Representation:** three labeled matrices with their actual dimensions (n×64, n×n, 10×n), raw weight standard deviations, optimizer group rates, and an explicit forward division node before readout. Fixed input/output dimensions use a distinct border and text label from changing hidden dimensions. A width selector reveals the corresponding saved coordinate records and validation grid below the graph.

**State:** basewidth32 fixed; targetwidth128 initially; baserate0.003; learner fields for hidden rate and readout input multiplier start blank. Predict whether μP and standard procedures coincide at base width before revealing that comparison. Grade with absolute tolerance1e−10 for rates/multipliers; show both correct formulas and the learner's attempted values.

**Unsolved construction:** require a completed pair (0.00075,0.25) at width128; then changed case width256/baserate0.004 requires (0.0005,0.125). Input/readout rates remainbase. A preset selecting “μP” must not auto-fill the task answer. Changing width or base rate clears both fields and prior result. Width256 is an arithmetic task only; do not invent a trained256 result.

**Null:** width32 yields identical mode traces across all seeds/rates; checked by exact JSON equality. **Contrast:** omit forward division at width128 and the same raw readout's output is four times as large for the same features; clearly an algebraic forward comparison, not a measured learning curve or accuracy. Holding hidden rate at0.003 is four times the intended0.00075. Render factor from current width, never hardcode it.

**Recorded graph:** `width_fits` provides mean absolute coordinates on fixed32trainingrows atsteps0,1,2,5,150. Label coordinate identity and stage. Show the initial readout decay honestly; no “all lines must be flat” rule. Validation grid computes means across three seeds from final CE, highlight minimum only within each row, and show exact candidate-rate range. Expose tiny standard128 difference0.00101 without claiming a reliable ranking. New controls select existing data, not new model runs.

## 7. Truncation and precision reference illustrations

**Home/question:** what operation does a numerical API perform?

Use a normal-density schematic with shaded retained interval, labeled **distribution illustration**, and separate actual sample statistics from `fixtures.truncation`. Show absolute bounds−2/2 beside±0.04 for std0.02. Do not fabricate sample histogram bins; the saved fixture contains summary statistics only. If phase two adds a histogram, generate and retain the actual sample/bin counts and update provenance.

For precision use a small conversion table from `fixtures.precision`; scientific notation with sufficient significant digits distinguishes representation error from becoming zero. A magnified number-line explanation can illustrate relative spacing, but no invented machine-level underflow threshold.

These are explanatory references, not mandatory extra labs. Keep the common readable typography and controls while choosing each representation for its mechanism.
