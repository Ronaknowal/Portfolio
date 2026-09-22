# Residual connections: visual specifications

## Live exploration contract — 21 September 2026

Open each investigation with its current inputs, intermediate mechanism and complete current output visible. Apply valid edits to meaningful entities immediately and update diagrams, tables, units and causal explanation together. No prediction entry, predicted-answer choices, commitment, prediction grading or answer-unlock feature is part of this packet, even optionally. Model predictions and mathematical masks/gates remain subject matter.

Use the topic-specific controls and checked fixtures below. Pair sliders or direct manipulation with labeled keyboard/numeric controls; keep presets as starting points, not the only editable values. A pinned baseline preserves its inputs, seed, units and outputs while the current case changes. Explain both a meaningful contrast and an unchanged/null result, then connect the observed effect to a practical design decision. Reset restores the stated fixture and current result. Invalid text has a local explanation and a clearly identified last valid result; never silently clamp or pair new inputs with old output.

Step/Back and bounded Run controls advance a real computation or reveal its chronological stages, not permission to view an answer. Show the current state and its result throughout. Keep exact small calculations live. For costly frozen inference, debounce or run bounded work with pending/current state labels and stale-result cancellation; inspect saved measurements without implying fresh training. Respect reduced motion, keep focus stable and avoid announcing every animation frame. Independent written practice and its hints/solutions stay separate.

Phase two must test default results without any action, meaningful edits, quick consecutive edits, valid extremes, null/invalid cases, reset, linked-view agreement, keyboard operation and readable phone layouts. The mathematical/reference checks already specified below remain; these live browser checks have not been performed in this content-only revision.

### Topic-specific live route
**Follow the direct and residual paths.** Edit branch weights, scalar depth/gain, normalization placement, projection entries and Euler step; inspect recorded block omissions.
**See the consequence.** Show correction contributions, current output/loss, both derivative paths and shape compatibility as inputs change. Display every intermediate gain and genuine before/after ablation record.
**Decision connection.** Recognize cancellation, shape mismatch and step-size instability despite the presence of an identity path.


Research/write only. Implementation, rendered diagrams and browser investigations remain pending. Numeric producer: `residual-experiments.py`; recorded inputs/results: `calculated-inputs.json`. Real images come only from attributed `digits-400.csv`.

## Common investigation behavior

Use the representation that exposes each operation: persistent value lanes for addition, signed contributions for gradients, movable operator locations for ordering, sockets for shapes, and actual point traces for fitting. Do not turn all of these into the same text-output box.

Each investigation follows the live exploration contract above: current results are visible immediately, valid entity edits update all linked views, and comparisons explain the mechanism. Reset restores the declared inputs and recomputes their result. No prediction or answer-submission state is retained.

Use labeled arrows and values in addition to color, keyboard/direct numeric alternatives to drag operations, touch-sized buttons, polite result announcements, reduced-motion support, and an equivalent text equation/table. Stack lanes vertically on narrow screens while preserving input/output identity. No animation is required to understand a result.

Bound all live computations to tiny vectors/matrices or scalar chains. Select recorded digit fits rather than training in the browser. No random reruns, full dataset plotting, tensor backend, or graph-layout dependency is necessary. Formal phase-two parity/accessibility/mobile/lazy-loading checks are deferred; author fixture calculations are complete.

## 1. Correction lanes and one actual parameter update

**Home:** first numerical example. **Question:** which values are preserved and which are changed?

Input x=[2,−1], weight [[0.1,0.2],[0,−0.5]], target [1,0]. Display a persistent direct lane carrying the two named coordinates, a matrix multiplication lane with all four products and row sums, then an addition node. A target marker and half-squared loss connect representation change to the training objective. Label two-dimensional vectors as a teaching representation, not an image/accuracy benchmark.

**Source:** `mechanisms.one_update`. Correction [0,0.5], output [2,−0.5], loss0.625; gradient matrix [[2,−1],[−1,0.5]]. SGD0.1 gives weight [[−0.1,0.3],[0.1,−0.55]], output [1.5,−0.25], loss0.15625. The SGD trace is a guided reveal, not a claim that every step lowers arbitrary loss.

**Live observation:** changing W11 from0.1 to−0.1 immediately shows the output-coordinate difference and loss difference against the labeled original. Both The current computed result is visible; compute from the changed calculation, not from the selected control's name.

**Construction:** after resetting to original W, task: reduce loss below0.4 by editing exactly one weight; each weight bounded [−1,1], input/target fixed. Initial0.625 is unsolved. Editing W11 to−0.1 yields [1.6,−0.5], loss0.305 (`one_weight_edit`), a checked solution. Permit other valid edits; Show current outputs with each valid input change. Feedback separates invalid edit count from a valid edit that misses the objective.

**Null/contrast:** zero edit preserves0.625 and is not success; choose W11=0.3 and loss rises to1.105 by direct formula. Changing W21 affects only output coordinate2. Valid input changes recompute every dependent result and explanation.

## 2. Two backward paths, cancellation and depth

**Home:** after the input-gradient derivation. **Representation:** two signed gradient arrows per coordinate, shown separately and then summed. Do not render the identity contribution as a shield that cannot be canceled.

**Source:** `one_update` gives direct g=[1,−0.5], branch Wᵀg=[0.1,0.45], total [1.1,−0.05]. Keep the upstream gradient fixed while tracing these two paths; replacing the forward function and recomputing a different loss would change g and is a different experiment.

**Scalar builder:** each block is x→(1+a)x; a in[−2,1], integer block count1–20. Show sign and magnitude of total derivative separately and a table of intermediate values. No log of zero; display exact zero as such. This is a deterministic local derivative calculation, not a trained network curve.

**Live observation:** at a=−0.5 and ten blocks, show the gain magnitude and its position below, equal to or above1. Reveal1/1024 with per-block0.5. Then construct gain magnitude below0.001 while keeping all identity paths. Start at a=0, unsolved; an edit to−0.5 solves. A follow-up repair asks for every intermediate block to preserve both sign and magnitude; only a=0 meets that stronger criterion in bounds, unlike a=−2 which alternates signs.

**Checked fixtures:** `scalar_derivatives` includes a−1→0,−0.5→0.0009765625,0→1,0.1→2.59374246,1→1024 at depth10. Null a0 preserves input/derivative. Contrast a−1 cancels despite visible skip. Numeric equality tolerance1e−10 for these small scalar calculations.

## 3. Operator-placement map

**Home:** pre/post ordering. **Question:** when correction is zero, is the whole block identity?

Draw the same direct/correction lanes. Place ReLU or LayerNorm after the join, or normalization inside the branch. Let learners move one operator with a select/button alternative. Keep F=0 for the initial comparison so a change comes specifically from location.

**Source:** x=[−2,1]; `zero_branch_order` gives pure output[−2,1]/JacobianI and post-ReLU[0,1]/diag(0,1). `zero_branch_norm` gives approximately[−0.99999778,0.99999778], epsilon1e−5, affine scale1/offset0; local Jacobian has entries±1.481471605e−6, and maps[1,1] tozero.

**Live observation:** display whether output and the common-shift direction are preserved, alongside the actual computed differences. The post-LN task must not equate small output variance with good gradient flow. Calculate and explain output equality to x separately from Jacobian action on[1,1].

**Unsolved edit:** initial ReLU after join fails to preserve the negative coordinate. Move it into the zero correction route to restore identity, and inspect the current result. Compare actual graph operations, not the display name. A plus sign still present is not enough. Null positive input[1,2] hides post-ReLU's difference; show why a negative-coordinate contrast is needed. Different LayerNorm axes are not a control here; this task deliberately normalizes both vector coordinates.

## 4. Shape and correspondence sockets

**Home:** projection explanation. Use labeled feature sockets, vector values, and a matrix whose rows show new coordinates. Identity preserves two sockets; P=[[1,0],[0,1],[1,1]] creates three outputs.

**Actual fixtures:** `projection` maps[2,−1]→[2,−1,1] and upstream[1,2,3]→[4,5]. `changed_projection` maps[1,3]→[1,3,4] and upstream[2,−1,4]→[6,3].

**Live observation:** choose the output size and display the computed third coordinate with its contributions immediately. **Construction:** start a 3-coordinate correction connected to an identity2-coordinate skip, unsolved. Learner supplies/selects the explicit3×2 projection and must enter at least the third row [1,1]; the checker validates both shape and values for x[2,−1] and changed x[1,3]. Numeric tolerance1e−9. An all-zero third row matches size but fails meaning.

Include an intentionally broadcastable correction shaped[batch,1] and show whether it performs three independent corrections. Explain repetition explicitly rather than calling every broadcast illegal. This is a diagnostic card, not an executable tensor exception in the browser. A permutation example can have equal shapes yet different feature labels; do not award correctness from shape alone.

For the optional image inset, show [B,C,H,W] and a spatial grid on both routes, with a 1×1 channel projection/stride marker. Defer convolution arithmetic to its owner; do not imply equal shapes certify spatial alignment.

## 5. Identity-start opening mechanism

**Home:** zero final layer, ReZero and LayerScale. **Representation:** a two-layer correction path, scalar/per-feature scale sockets, and separate gradient indicators for final weight, earlier weight and scale. A zero numeric output and zero gradient must use distinct labels.

**Sources:** `zero_last` and `gates`. The zero final linear case has output[1,2], earlier gradient0, final gradient[[1,2],[2,4]]. Extra final ReLU leaves both weight gradients0. ReZero x[1,2], F[0,0.7], target0, scale0 has scale gradient1.4 and zero branch-weight gradient; SGD0.1 changes scale to−0.14. Scale0.001 starts with small nonzero weight gradients. Scale1 is the contrasting larger-contribution case.

Use a guided first-step trace that shows which parameters have nonzero gradients and why. Reset returns the initial state; Step advances the actual update. A different input/target can make a scalar gradient zero, so the explanation makes no unconditional learning claim. LayerScale is a conceptual per-feature extension, not another trained method in the saved fits.

## 6. Real fitted models and block ablations

**Home:** complete digit experiment and later path discussion.

A small architecture strip shows stem64→32, L blocks each with LN→Linear32→tanh→Linear32, and head32→10. Use actual CSV pixels for a few examples with source IDs. Four return expressions occupy different branch diagrams; do not relabel the LN-containing zero-gate experiment as an exact normalization-free ReZero reproduction.

Plot training/validation CE at saved steps0/1/25/100/250 with point markers. Data: `fits`, three seeds; depth0 stem-only and depths2/6/12 × four modes. Unit is natural-log loss per example. Correct counts have a separate axis/table. No interpolated observations, runtime/timing claims, or automatically generated “degradation cured” headline. The no-block baseline remains visible.

Switch between loss-specific layer diagnostics (mean square and activation gradient RMS on fixed 32 training rows) without calling a larger gradient better. Show stem-weight gradient as a separate quantity from activation gradient. Parameter displacements confirm trained branches moved; initial zero-gate metrics match the stem-only baseline for every seed.

**Ablation inset:** select an already-trained residual model and omit a block using `validation_block_ablations`. Keep its exact unmodified fit as the baseline and show both retained records immediately. Seed 1/depth 6 has 116/120 with all blocks; omissions range from 28 to 114. This is deletion without retraining. Changing fit/block updates the displayed record; Reset selects the full model. Omit-none is the null and returns the unmodified metrics.

Do not offer arbitrary multiple deletions with fabricated combinations; those are not present in the saved data. If phase two extends combinations, compute and preserve new measurements from retained models first.

## 7. Linear path expansion versus nonlinear composition

**Home:** optional path interpretation. A two-block graph highlights the four linear products x, W0x, W1x, W1W0x without suggesting four separately trained models. Ordered arrows matter.

Switch to the explicitly nonlinear scalar F(x)=x². Label initial state **x0=1**, then first state2, final6. The invalid distributed expression gives4, per `path_expansion`. Highlight the operation F(1+1)=4 compared with F(1)+F(1)=2. This is a small comparison diagram, not a new lab quota. Optional x0=0 null makes both expressions0 and demonstrates why one passing fixture is insufficient.

## 8. Practical and numerical extensions

The denoising inset shows observed signal[0.2,0.9,0.4], hypothetical estimated noise[0.1,−0.1,0.2], and subtraction result[0.1,1,0.2], explicitly labeled **illustrative arithmetic, not model output**. Do not generate attractive before/after images and imply they were produced by a trained DnCNN.

The Euler inset follows x0=1 under x←(1−dt)x for ten steps; use an editable step in[0,3] and pause/next controls. `euler` records dt0.1→0.34867844,1→0,2.5→57.665039. live comparison asks whether a continuously decaying equation implies any positive discrete step decays; reject by the2.5contrast. A construction can request a sign-preserving strict contraction:0<dt<1 solves; dt2 alternates with constant magnitude, dt0 identity is the null but not strict contraction. This is an optional numerical connection, not a measured neural-network benchmark.

Memory explanation is a static lifetime sketch: identity value needed at join, branch intermediates optionally saved/recomputed. `pure_addition_saved_tensors` is empty in the author's pure addition/sum backward hook. Do not infer whole-model peak memory from this narrow check. Checkpoint semantics and mask replay are deferred runtime topics.


## Code-to-mechanism implementation contract — 22 September 2026

Place the manuscript's new implementation route next to its stated concept section. Preserve the named axes, state and algorithm steps when implementing figures; the complete teaching programs are content inputs, not a hidden replacement for learner-visible code. Render long source only on demand with keyboard-scrollable code and wrapping download labels. Keep constructed comparison fixtures separate from recorded training experiments. No browser execution of Python, pretrained-model download or GPU experiment is required to operate a lab.

The ownership map in design.md identifies which operations are implemented here and which actual sources are reused. Both paths must be findable: the transparent mechanism and the ordinary package/tool route, followed by the changed-input practice. There is no guess-entry, prediction submission or answer-unlock state. Current outputs remain visible while the learner edits meaningful inputs; separate written practice can retain hints and solutions.
