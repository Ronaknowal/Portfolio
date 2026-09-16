# Normalization visual specifications

Content preparation only. These visuals are not implemented or browser-verified. Preserve their different jobs: shared ruler, tensor membership, transform geometry, state timeline, gradient graph, and actual training trajectories.

## Common investigation contract

Start each investigation with an unset prediction. Bind the committed answer to the current input entities, requested comparison, mode, epsilon, affine parameters, and running state. Grade the computed result against the submitted answer. Any input change invalidates the prediction and result immediately; reset restores the declared fixture and clears assessment state. Require a meaningful, unsolved entity edit for the second investigation rather than treating a preset replay as mastery.

Use finite bounded inputs, no automatic animation, no live browser training, and no network data requests. All formula plots come from the declared mathematics; measured learning curves come only from calculated-inputs.json. Display units and source type. Numeric inputs and keyboard controls must provide the same operations as dragging. Coordinate tables and prose must explain the same dependencies as colors. Announce changes and results accessibly. Stack linked views on small screens without losing input/output identity. Keep state local and render bounded groups, not a giant tensor or hundreds of images.

## A. Shared ruler

A static or manually stepped trace follows [1,3,5,7] through mean4, deviations[-3,-1,1,3], squared deviations[9,1,1,9], variance5, and normalization with epsilon1e-5. Exact normalized outputs are in mechanisms.batch_train_eval.train. Label variance after normalization as v/(v+epsilon), not exactly one. A shared affine scale/shift panel may show transformed statistics. Keep each observation's identity across views. Do not imply that centering creates a Gaussian distribution.

## B. Statistics membership and cross-example influence

Use shape(2,4,1,2), values1–16 in the exact JSON order. Display two example cards, four channel rows per card, two spatial cells per row. Selecting a cell highlights the exact coordinates and values in its statistics group. Modes: training BN, LN over(C,H,W), GN with G=1/2/4, and IN. Identity affine and epsilon1e-5.

First prediction: before changing sample1/channel0/position0 from9 to19, identify all methods that change any normalized cell in sample0. Selection starts unset; explicit commit is required even for a prediction of no methods. Grade output differences with tolerance1e-10. Checked outcome: BN maximum first-example difference .15022057675166323; LN/GN/IN differences zero. Show the changed shared group, not just a success badge.

Independent task: under GN2, modify a different channel's cell so sample0/channel0's normalized value changes. Do not preselect the successful channel. Grade the selected entity and actual effect. Channels0/1 share its group; channels2/3 provide a contrasting null. Another null is translating every member of one centered group by the same offset. Arbitrary cell values are bounded[-50,50]. Recompute every output; do not hand-author a before/after heatmap.

Expose group sizes, means, variances and identities. G must divide4. The GN1/LN and GN4/IN equivalence check uses identity affine and matching epsilon. Distinct LayerNorm elementwise affine and GroupNorm channel affine settings invalidate a full-function equivalence claim.

## C. Offset, scale, and centering

Two-feature vector view with origin, mean point, centered arrow, RMS radius and output coordinates. Inputs[-20,20], epsilon1e-8 to1, identity affine. Epsilon has squared activation units.

Prediction: adding10 to[1,3] changes which outputs? Checked LN stays[-.999995,.999995]; RMS changes[.4472131483,1.3416394449] to[.9135002469,1.0795912009]. Grade change/no-change with1e-9 tolerance. Independent task: construct a nonzero input where both methods agree, without starting at a solved zero-mean vector. Grade actual output equality and an explanation of the zero mean. Checked null[-1,1] agrees; contrast[5,5] yields centered zeros versus RMS approximately ones. All-zero input yields zeros with positive epsilon.

Scaling is approximately invariant for positive factors at finite epsilon. Display numerical differences instead of grading exact equality. Near-zero inputs show the epsilon-dominated denominator. Negative scaling reverses signs. State changes and reset follow the common contract.

## D. BatchNorm state ledger

Keep current batch, affine parameters, running buffers and mode visibly separate. Start one channel with [1,3,5,7], old running mean0/variance1, momentum.1 and epsilon1e-5. Ask for next mean/variance before Apply Batch; answer fields start empty. Accept .4 and1.566666667 at the displayed precision. Show population variance5 and corrected variance20/3 in distinct rows.

Switching to evaluation uses updated buffers without changing them. First output is -1.341639444861 in training versus .479359747293 in evaluation; all four outputs are stored. Do not claim eval() makes the two functions identical.

Independent task: choose four batch values that make the running mean one after a single step from the default state with momentum.1. Do not provide a solved batch. Grade actual result and explain that its batch mean must be ten. Contrast momentum0: buffers do not update, but training still uses batch statistics. Null: evaluation with running statistics enabled leaves buffers unchanged for any input batch.

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

Independently replay formula/gradient fixtures, state updates and nulls, axis membership including channel-last traps, group divisibility/equivalence conditions, near-zero variance/epsilon, low-precision promotion, and saved real trajectories. Replay the complete program in a clean declared environment. Then verify actual grading, meaningful edits, invalidation/reset, keyboard/touch/text routes, mobile layout, math rendering, lazy loading and bounded computation. These are phase-two checks, not completed content checks.
