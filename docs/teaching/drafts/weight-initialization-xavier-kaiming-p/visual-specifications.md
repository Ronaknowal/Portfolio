# Initialization: visual and investigation specifications

## Live exploration contract — 21 September 2026

Open each investigation with its current inputs, intermediate mechanism and complete current output visible. Apply valid edits to meaningful entities immediately and update diagrams, tables, units and causal explanation together. No prediction entry, predicted-answer choices, commitment, prediction grading or answer-unlock feature is part of this packet, even optionally. Model predictions and mathematical masks/gates remain subject matter.

Use the topic-specific controls and checked fixtures below. Pair sliders or direct manipulation with labeled keyboard/numeric controls; keep presets as starting points, not the only editable values. A pinned baseline preserves its inputs, seed, units and outputs while the current case changes. Explain both a meaningful contrast and an unchanged/null result, then connect the observed effect to a practical design decision. Reset restores the stated fixture and current result. Invalid text has a local explanation and a clearly identified last valid result; never silently clamp or pair new inputs with old output.

Step/Back and bounded Run controls advance a real computation or reveal its chronological stages, not permission to view an answer. Show the current state and its result throughout. Keep exact small calculations live. For costly frozen inference, debounce or run bounded work with pending/current state labels and stale-result cancellation; inspect saved measurements without implying fresh training. Respect reduced motion, keep focus stable and avoid announcing every animation frame. Independent written practice and its hints/solutions stay separate.

Phase two must test default results without any action, meaningful edits, quick consecutive edits, valid extremes, null/invalid cases, reset, linked-view agreement, keyboard operation and readable phone layouts. The mathematical/reference checks already specified below remain; these live browser checks have not been performed in this content-only revision.

### Topic-specific live route
**Explore scale and directional transmission.** Inspect saved initialization/seed traces; edit four activation values, singular directions, depth and width/rate scaling.
**See the consequence.** Show forward/backward second moments, means/variance, directional gain and shape/update formulas together. Continuous tiny models are distinct from selectors over measured training records.
**Decision connection.** Choose an initialization/parameterization by the signal and update behavior it preserves, without treating average scale as every-direction stability.


Prepared content only. No React, SVG, browser lab, generated chart asset, or runtime registration is implemented. All numerical inputs named here are in `calculated-inputs.json`; the complete producer is `initialization-experiments.py`. Do not redraw empirical curves from a remembered theory.

## Shared interaction contract

Each investigation follows the live exploration contract above: current results are visible immediately, valid entity edits update all linked views, and comparisons explain the mechanism. Reset restores the declared inputs and recomputes their result. No prediction or answer-submission state is retained.

Reset restores the stated inputs and immediately displays their computed result. A preset selection is exploration, not successful learner construction. Where a task asks for an edit, require an actual entity change and evaluate the resulting constraint. Keep observation and construction outcomes distinct. Respect reduced motion, provide buttons/keyboard equivalents for sliders and draggable marks, use labels/shapes as well as color, include a data table or explicit equations, and announce result changes politely. Avoid automatic moving traces.

No browser training, random resampling, large matrix decomposition, or 54-fit sweep. Read compact selected records on demand or implement the tiny explicit two-dimensional formulas. Phase two must check packet parity, keyboard use, input bounds, input invalidation, null/contrast fixtures, mobile flow, legibility, and lazy-loading costs. Author calculations below are not claims that those UI checks already passed.

## 1. Forward and backward layer traces

**Home/question:** immediately after the first signal table. How can modest per-layer scaling accumulate through depth, and does the backward probe tell the same story?

**Representation:** two aligned plots sharing layer index 0–20. Upper: pooled activation second moment; lower: RMS of gradient with respect to that layer's activation, for the fixed scalar probe. Explain the output cotangent at the end of the lower chart. Zero values appear on an explicitly separate zero rail; never substitute an arbitrary epsilon and label it as a measured value. Positive values use log10 scale with human-readable powers-of-ten ticks. Add a table of raw values and seed selector.

**Source:** `propagation.records`, six schemes × three seeds; input/cotangent seed 71; float64, batch 128, width 64. No timing axis. Each series retains the original exact record and rounds only labels. Variable is `second_moment`, not `variance`. Optional statistics selector may expose variance/mean/zero_fraction but must change the title, units, and live comparison.

**Live observation:** initially show seed 1 Kaiming and candidate `small` together, including each layer-20/input second-moment ratio. Label ratios below .1, between .1 and 10, or above 10 as descriptive ranges. Changing scheme or seed immediately selects the corresponding recorded values. Being within that range is not a certificate of good training.

**Checked outcomes:** seed-1 small ratio ≈9.33e−51; large ≈8.49e17; Kaiming ≈0.736. Zero has exact 0. The lower graph starts near 9.54e−26, 9.10e8, and 0.848 respectively at input; its output value is the supplied cotangent RMS ≈0.994535 for every scheme, including zero. Annotate why a zero output can still have a nonzero externally specified gradient with respect to that output.

**Null/contrast:** select the same recorded model output without changing input—must give identical results, not a fresh random curve. Small↔large changes direction; switching Kaiming to orthogonal does not imply an improvement label. The orthogonal layer-20 mean square 0.2316 is displayed honestly.

**Learner edit:** an optional local idealized recurrence builder asks the learner to choose a weight variance for width 100 so that `(100*variance/2)^20` stays between 0.9 and 1.1. Starts at variance 0.01, unsolved. Permit numeric variance in [0,0.1]; 0.02 solves, 0.01 contracts, 0.04 grows. Explicitly label this panel an assumption-based calculation, not an additional measured network; do not overlay it indistinguishably on observations.

## 2. Four values: second moment versus variance

**Home/question:** statistics refresher. What changes when ReLU moves negative values to zero?

**Representation:** four persistent identified dots, a zero origin, and a separate mean marker. Before/after rows show values; selectable squared-distance bars measure from zero or from each row's mean. The values and sums remain text-readable. No histogram bins that hide the four exact observations.

**Source/formulas:** `fixtures.relu_input/output` from [-2,-1,1,2]. Mean, mean square, and correction-zero variance. No stochastic sampling or distribution inference.

**Live observation:** display the original and changed second moments and variances together; show their ratios and explain why these statistics need not scale alike. Calculate and explain two distinct statements. Reveal input (0,2.5,2.5) and output (0.75,1.25,0.6875) in mean/q/variance order. Feedback for “both halve”: clipping shifts the mean, so subtracting mean squared changes variance further.

**Construct:** ask for a symmetric four-value input with larger absolute outer values while retaining the inner pair ±1; start outer magnitude 2, target output variance 1.5, unsolved. Numeric outer magnitude [1,5], step 0.1 or direct edit. Outer magnitude 3 solves within 1e−9. Preserve four entities rather than a formula-only slider.

**Checked contrast/null:** [-3,-1,1,3] has q5→2.5, variance5→1.5, mean0→1. Toggle identity activation: values and both moments unchanged. Changing one positive value independently can break symmetry, so a free-edit mode must compute actual sums rather than apply the “half” rule. That mode has no default automatic diagnostic readout.

**Mobile:** stack before and after; keep dot IDs and numeric rows aligned. Screen-reader equivalent lists every value and both centers.

## 3. Circle, ellipse, and gated directions

**Home/question:** average preservation versus every-direction preservation.

**Representation:** input unit circle; output ellipse under diagonal M; labeled editable input arrow and corresponding output arrow. Beside it show singular values, average squared singular value, and the gain for this vector. A second view draws local directional arrows through the ReLU gate at a specified base point. Do not treat the finite output vector as the local Jacobian action.

**Computed model:** lower gain `s` in [0.05,1], larger gain `sqrt(2-s*s)`. M is diagonal, no hidden rotation in the initial activity. Input vector components bounded [-2,2]; reject both zero for gain calculation with useful text. Exact norm ratio `norm(Mv)/norm(v)`. Repeat count 1–8 uses diagonal powers. If adding rotation later, rotate input/output consistently and verify the formula.

**Live observation:** base s=sqrt(0.1), v=(0,1), depth1; show whether vector shrinks even though mean squared singular value equals1. The current computed result is visible. After result, ask the learner to make the smaller-direction five-layer gain ≤0.001 while preserving the average squared gain. Initial s≈0.316 gives ≈0.003162, unsolved; s0.2 produces 0.00032 and larger1.4. Check both constraints plus a meaningful changed s; a diagnostic readout explains the direction tradeoff.

**Nulls/contrasts:** s1 gives identity and all gains1. At s0.2, v=(1,0) grows1.4 while v=(0,1) shrinks0.2. At ReLU base point (-1,1), J=diag(0,sqrt2), singular values [sqrt2,0]; at (1,1), J=sqrt2*I. Avoid derivative ambiguity by rejecting zero base coordinates for this task or explicitly explaining the chosen derivative convention.

**Source:** exact `fixtures.geometry_interventions`, `average_gain_matrix`, `relu_scaled_identity_jacobian`. Gaussian spectrum is a separate saved 64-dimensional example, with histogram plus min/max and full text table; no claim that the 2D ellipse depicts that matrix. QR Gram maximum error is a numerical identity check, not an error-rate metric.

## 4. Hidden-unit symmetry and the first gradient

**Home/question:** why two feature slots can remain one feature.

**Representation:** two small neuron paths with equal incoming/outgoing weights, labeled tanh outputs, prediction, and separate incoming gradient arrows. Step only one update at a time. Distinct-row case changes the actual incoming weights; zero-head case distinguishes blocked hidden gradients from nonzero head gradients.

**Source:** `fixtures.symmetry` and `zero_readout`. These are analytic/autograd calculations for input1, target1, half-MSE, not a trained benchmark. In the initial identical case both incoming gradients −0.2541693849; distinct incoming [0.1,0.3] yields [−0.2621811825,−0.2423390161]. Zero head hides no gradient: hidden [0,0], head [−0.0996679946,−0.2913126125].

This is a guided trace, not another construction lab. Let learners inspect which arrows disappear while displaying the zero-head case, without accumulating a site-wide mastery score from clicking Next. Reset restores the stated inputs and immediately displays their computed result. A text equation and values suffice on mobile; no force-layout graph engine.

## 5. Actual training: probabilities and correct counts

**Home/question:** does starting with a reasonable scale guarantee the best validation result?

**Representation:** distinguish the four-hidden-layer digit architecture from the twenty-layer diagnostic network. A compact image strip uses actual CSV pixel rows only; show integer target labels. Plot saved training/validation CE against exact update checkpoints, with discrete point markers. Use a separate count panel for validation correct/120. Do not interpolate a dense measured curve between five checkpoints.

**Source:** `digit_fits` records. Choose seed and scheme; reveal alternate seeds. Same head within seed and same data across all runs. No synthetic digit assets. Saved CSV IDs provide provenance.

**Interaction:** comparison and interpretation, not “tune until green.” Prompt why tiny training CE can coexist with larger validation CE. Display seed1 small/large cases and the differing correct-count ranges without automatically ranking one initializer as universally best. The all-zero model's equal logits create argmax class0; the 12/120 count follows the balanced validation split. No browser fitting or benchmark claims.

## 6. μP shape and update workbench

**Home/question:** which quantities change when width changes?

**Representation:** three labeled matrices with their actual dimensions (n×64, n×n, 10×n), raw weight standard deviations, optimizer group rates, and an explicit forward division node before readout. Fixed input/output dimensions use a distinct border and text label from changing hidden dimensions. A width selector reveals the corresponding saved coordinate records and validation grid below the graph.

**State:** base width 32 fixed; target width 128 initially; base rate 0.003. Width and base rate are editable controls; the hidden rate and readout input multiplier are derived readouts visible immediately. Recompute them and display their formulas on every valid edit. Display whether μP and standard procedures coincide at base width. Use absolute tolerance 1e−10 for numerical comparisons; retain no attempted-answer fields.

**Width/rate exploration:** show hidden rate and readout input multiplier as derived readouts beside the width and base-rate controls. At width 128 the pair is (.00075,.25); width 256/base rate .004 gives (.0005,.125), while input/readout rates remain at base. Comparing standard and μP formulas updates these values immediately. Width 256 is arithmetic only; no trained width-256 result was retained.

**Null:** width32 yields identical mode traces across all seeds/rates; checked by exact JSON equality. **Contrast:** omit forward division at width128 and the same raw readout's output is four times as large for the same features; clearly an algebraic forward comparison, not a measured learning curve or accuracy. Holding hidden rate at0.003 is four times the intended0.00075. Render factor from current width, never hardcode it.

**Recorded graph:** `width_fits` provides mean absolute coordinates on fixed32trainingrows atsteps0,1,2,5,150. Label coordinate identity and stage. Show the initial readout decay honestly; no “all lines must be flat” rule. Validation grid computes means across three seeds from final CE, highlight minimum only within each row, and show exact candidate-rate range. Expose tiny standard128 difference0.00101 without claiming a reliable ranking. New controls select existing data, not new model runs.

## 7. Truncation and precision reference illustrations

**Home/question:** what operation does a numerical API perform?

Use a normal-density schematic with shaded retained interval, labeled **distribution illustration**, and separate actual sample statistics from `fixtures.truncation`. Show absolute bounds−2/2 beside±0.04 for std0.02. Do not fabricate sample histogram bins; the saved fixture contains summary statistics only. If phase two adds a histogram, generate and retain the actual sample/bin counts and update provenance.

For precision use a small conversion table from `fixtures.precision`; scientific notation with sufficient significant digits distinguishes representation error from becoming zero. A magnified number-line explanation can illustrate relative spacing, but no invented machine-level underflow threshold.

These are explanatory references, not mandatory extra labs. Keep the common readable typography and controls while choosing each representation for its mechanism.


## Code-to-mechanism implementation contract — 22 September 2026

Place the manuscript's new implementation route next to its stated concept section. Preserve the named axes, state and algorithm steps when implementing figures; the complete teaching programs are content inputs, not a hidden replacement for learner-visible code. Render long source only on demand with keyboard-scrollable code and wrapping download labels. Keep constructed comparison fixtures separate from recorded training experiments. No browser execution of Python, pretrained-model download or GPU experiment is required to operate a lab.

The ownership map in design.md identifies which operations are implemented here and which actual sources are reused. Both paths must be findable: the transparent mechanism and the ordinary package/tool route, followed by the changed-input practice. There is no guess-entry, prediction submission or answer-unlock state. Current outputs remain visible while the learner edits meaningful inputs; separate written practice can retain hints and solutions.
