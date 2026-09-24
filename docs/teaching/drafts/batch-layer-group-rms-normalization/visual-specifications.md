# Normalization visual specifications

## Live exploration contract — 21 September 2026

Open each investigation with its current inputs, intermediate mechanism and complete current output visible. Apply valid edits to meaningful entities immediately and update diagrams, tables, units and causal explanation together. No prediction entry, predicted-answer choices, commitment, prediction grading or answer-unlock feature is part of this packet, even optionally. Model predictions and mathematical masks/gates remain subject matter.

Use the topic-specific controls and checked fixtures below. Pair sliders or direct manipulation with labeled keyboard/numeric controls; keep presets as starting points, not the only editable values. A pinned baseline preserves its inputs, seed, units and outputs while the current case changes. Explain both a meaningful contrast and an unchanged/null result, then connect the observed effect to a practical design decision. Reset restores the stated fixture and current result. Invalid text has a local explanation and a clearly identified last valid result; never silently clamp or pair new inputs with old output.

Step/Back and bounded Run controls advance a real computation or reveal its chronological stages, not permission to view an answer. Show the current state and its result throughout. Keep exact small calculations live. For costly frozen inference, debounce or run bounded work with pending/current state labels and stale-result cancellation; inspect saved measurements without implying fresh training. Respect reduced motion, keep focus stable and avoid announcing every animation frame. Independent written practice and its hints/solutions stay separate.

Phase two must test default results without any action, meaningful edits, quick consecutive edits, valid extremes, null/invalid cases, reset, linked-view agreement, keyboard operation and readable phone layouts. The mathematical/reference checks already specified below remain; these live browser checks have not been performed in this content-only revision.

### Topic-specific live route
**Change the group that defines the ruler.** Edit tensor cells, normalization method/group count, offset/scale, epsilon, affine values, mode and running-statistic parameters.
**See the consequence.** Highlight each statistic membership set and show all affected outputs, means/variances and running buffers immediately. Compare changing another example with changing a member of the same group.
**Decision connection.** Choose a normalizer and mode by its information dependencies, batch sensitivity and inference state.


Content preparation only. These visuals are not implemented or browser-verified. Preserve their different jobs: shared ruler, tensor membership, transform geometry, state timeline, gradient graph, and actual training trajectories.

## Common investigation contract

Each investigation follows the live exploration contract above: current results are visible immediately, valid entity edits update all linked views, and comparisons explain the mechanism. Reset restores the declared inputs and recomputes their result. No prediction or answer-submission state is retained.

Use finite bounded inputs, no automatic animation, no live browser training, and no network data requests. All formula plots come from the declared mathematics; measured learning curves come only from calculated-inputs.json. Display units and source type. Numeric inputs and keyboard controls must provide the same operations as dragging. Coordinate tables and prose must explain the same dependencies as colors. Announce changes and results accessibly. Stack linked views on small screens without losing input/output identity. Keep state local and render bounded groups, not a giant tensor or hundreds of images.

## A. Shared ruler

A static or manually stepped trace follows [1,3,5,7] through mean4, deviations[-3,-1,1,3], squared deviations[9,1,1,9], variance5, and normalization with epsilon1e-5. Exact normalized outputs are in mechanisms.batch_train_eval.train. Label variance after normalization as v/(v+epsilon), not exactly one. A shared affine scale/shift panel may show transformed statistics. Keep each observation's identity across views. Do not imply that centering creates a Gaussian distribution.

## B. Statistics membership and cross-example influence

Use shape(2,4,1,2), values1–16 in the exact JSON order. Display two example cards, four channel rows per card, two spatial cells per row. Selecting a cell highlights the exact coordinates and values in its statistics group. Modes: training BN, LN over(C,H,W), GN with G=1/2/4, and IN. Identity affine and epsilon1e-5.

**Live group comparison:** change sample1/channel0/position0 from 9 to 19 and immediately highlight all methods/cells affected in sample0. Compute differences with tolerance 1e−10. The checked maximum BN difference is .15022057675166323; LN/GN/IN differences are zero. The membership highlights explain the shared statistic responsible; no “none” answer selection is required.

Independent task: under GN2, modify a different channel's cell so sample0/channel0's normalized value changes. Allow each channel to be selected as an input and show its actual effect immediately. Calculate and explain the selected entity and actual effect. Channels0/1 share its group; channels2/3 provide a contrasting null. Another null is translating every member of one centered group by the same offset. Arbitrary cell values are bounded[-50,50]. Recompute every output; do not hand-author a before/after heatmap.

Expose group sizes, means, variances and identities. G must divide4. The GN1/LN and GN4/IN equivalence check uses identity affine and matching epsilon. Distinct LayerNorm elementwise affine and GroupNorm channel affine settings invalidate a full-function equivalence claim.

## C. Offset, scale, and centering

Two-feature vector view with origin, mean point, centered arrow, RMS radius and output coordinates. Inputs[-20,20], epsilon1e-8 to1, identity affine. Epsilon has squared activation units.

Live observation: adding10 to[1,3] changes which outputs? Checked LN stays[-.999995,.999995]; RMS changes[.4472131483,1.3416394449] to[.9135002469,1.0795912009]. Calculate and explain change/no-change with1e-9 tolerance. Independent task: construct a nonzero input where both methods agree, without starting at a solved zero-mean vector. Calculate and explain actual output equality and an explanation of the zero mean. Checked null[-1,1] agrees; contrast[5,5] yields centered zeros versus RMS approximately ones. All-zero input yields zeros with positive epsilon.

Scaling is approximately invariant for positive factors at finite epsilon. Display numerical differences instead of result checks exact equality. Near-zero inputs show the epsilon-dominated denominator. Negative scaling reverses signs. State changes and reset follow the common contract.

## D. BatchNorm state ledger

Keep current batch, affine parameters, running buffers and mode visibly separate. Start one channel with [1,3,5,7], old running mean0/variance1, momentum.1 and epsilon1e-5. Show the next mean/variance immediately as a computed forward-pass preview: .4 and1.566666667. Advance Batch commits that actual buffer transition, without any answer field. Show population variance5 and corrected variance20/3 in distinct rows.

Switching to evaluation uses updated buffers without changing them. First output is -1.341639444861 in training versus .479359747293 in evaluation; all four outputs are stored. Do not claim eval() makes the two functions identical.

Independent task: choose four batch values that make the running mean one after a single step from the default state with momentum.1. Do not provide a solved batch. Calculate and explain actual result and explain that its batch mean must be ten. Contrast momentum0: buffers do not update, but training still uses batch statistics. Null: evaluation with running statistics enabled leaves buffers unchanged for any input batch.

Keep a distinct cardinality card: one image with spatial values[1,3] is legal and produces approximately[-1,1]; exactly one value per channel is rejected in training. Do not erase the spatial dimensions when N=1. Caption that correlated spatial values are not independent examples.

## E. Gradients through statistics

Computation graph: x branches to mean and variance, both feed normalized values, then affine transform and MSE. Use x[1,3], gamma[1,2], beta[0,0], target[0,1], epsilon1e-5. Show accumulation through statistics, not only the direct division path.

The executed affine update holds x fixed and changes gamma/beta with rate.1. Loss .999985000175 becomes .639992000073; output[-.999995,1.999990] becomes[-.799997,1.799993]. A step trace is sufficient here; an extra generic quiz wrapper is unnecessary. The tiny input gradient is explained by the two-feature geometry and epsilon rather than called a failure.

## F. Actual matched training trajectories

Plot digits.records: four normalization choices times three seeds, epochs0/1/5/20/50. Select training-evaluation CE, validation CE, or correct count. CE is in nats; count denominator120. Connect only observed points and label the unrecorded intervals. The training metric explicitly uses evaluation mode, especially for BN.

Show individual seeds or their separately marked lines. Do not infer a population confidence interval from three seeds. Tables must match manuscript and JSON. No browser fitting. No individual predictions were retained for this packet, so do not invent confusion matrices or per-image prediction tooltips. Load the CSV only if showing the input-specimen illustration.

## G. Placement and resource arithmetic

Two explicit graphs: post-norm y=LN(x+F(x)); pre-norm y=x+F(LN(x)). For x[1,3] and F=0, outputs are LN(x) and x. Mark the pre-norm direct identity derivative path while also showing the branch contribution. A second derived case F(u)=.5u may be added after phase-two arithmetic checks. Do not introduce unexplained attention, rotary positions or a purported full LLM block.

Separate footprint calculation: shape8×8192×8192 has536870912 elements; one FP32 read is2GiB, one two-byte read is1GiB. This is memory traffic, not runtime or a claimed training speedup. Prefer a simple readable calculation if a diagram adds no understanding.

## Deferred implementation checks

Independently replay formula/gradient fixtures, state updates and nulls, axis membership including channel-last traps, group divisibility/equivalence conditions, near-zero variance/epsilon, low-precision promotion, and saved real trajectories. Replay the complete program in a clean declared environment. Then verify actual live updates, meaningful edits, invalidation/reset, keyboard/touch/text routes, mobile layout, math rendering, lazy loading and bounded computation. These are phase-two checks, not completed content checks.
