# Depthwise and dilated convolution: visual implementation contract

Content-first specifications, 13 September 2026. No illustration, browser lab or runtime component has been implemented. Follow current semantic ownership/lazy-loading/code standards in phase two; the names below describe teaching purpose, not prescribed exports or a generic lab template.

## Inputs, numerical truth and common interaction rules

Retain lesson.md, convolution-factorization.py, author-checks.py, calculated-inputs.json, author-check-results.json, digits-400.csv and data-provenance.md. Numbers in the small fixtures are constructed exact calculations, not fitted measurements. Real-digit predictions and weights come from the six actual CPU fits; the independent direct-loop reconstructions and edited-input outputs are in author-check-results.json. Report unitless intensities, logits, weights, counts and MACs explicitly. No latency graph exists.

Each investigation initializes prediction to null, with no option preselected and no hidden answer rendered beside it. Bind an attempt to the full relevant input/model/target/step revision. Before revealing the new output, require the learner's prediction and then grade it against a computed result with an explanation of the contributing terms. A plausible-sounding explanation alone is not grading. Preserve the pre-edit comparison state while an edit is staged. Any subsequent input, filter, rank, dilation, selected probe, model or objective change invalidates the existing grade and clears the prediction; selecting explanatory labels alone must not change numbers.

Use three-way increase/decrease/unchanged predictions with a numeric tolerance where that question is appropriate; multi-target rank repair has an actual numeric objective. Never declare success merely because any control moved. Reset restores the exact starting entities, selected target, unsolved objective, comparison snapshot and null prediction. Separate “restore values” from “load contrasting case” so changing scenarios is deliberate.

Keyboard alternatives for every editable cell: labelled numeric inputs and optional arrow-key grid traversal. Provide readable static matrices, current formulas and text result tables; neither colors, hovering, dragging nor animation may carry unique meaning. On narrow screens stack channel lanes, branch maps and result tables in explanatory order; preserve row/column headers and numerical access. Pan a bounded grid if necessary instead of making labels illegible. Announce result changes once after evaluation; respect reduced motion. Use high-contrast positive/negative/zero markers and shape/pattern distinctions. No visual may hide negative weights with a positive-only heatmap.

Validate finite bounded numbers before computing. Provide actionable errors and retain the last valid state without grading an invalid input. Cap grids, channel counts, layers and editable models as specified; recompute only on explicit evaluate or a short input debounce, never continuous fitting. Lazy-load real model weights only when that investigation opens. Browser/native/formal independent checks below are deferred phase two requirements.

## 1. Channel-lane arithmetic and shared-filter repair

### Placement and form

Section 1, immediately after the three-row input/filter/filtered-value table: two horizontal input strips, independent two-tap spatial filter cards, filtered strips, then a small mixing matrix with output strip. A selected output highlights the exact input cells and multiplicative paths that form it. A single motion step should follow real dependencies, not decorative flow.

Section 2's rank explanation adds a second view using two input probes and a displayed effective-filter matrix. It shares the principle but changes the task: construct a function satisfying two probes instead of moving a numerical slider for its own sake.

### Arithmetic state and rules

Default input A=[1,2,3], B=[2,0,1]; filters D_A=[1,-1], D_B=[.5,1]; mixing P=[2,-1]; no bias, no activation, valid stride one. Inputs bounded −4…8 and weights −3…3, steps .5 or free finite decimal within bounds. Allow an output-index selection 0/1. Calculate each channel's valid cross-correlation, then P times filtered channels. Expose units as constructed signal values; channels A/B are examples, not named learned semantic features.

Initial reveal should show the selected original output only after predicting it numerically, or present the worked original −3 from prose and require an unset comparative prediction for a staged new input. Do not simultaneously treat −3 as hidden and visible. Preferred route: original worked result is visible; learner stages B[0]=4, predicts first output direction, then evaluates.

Checked baseline filtered A=[−1,−1], B=[1,1], outputs [−3,−3]. B[0] 2→4 yields first output −4, second −3. B[0] 2→0 yields first output −2, second −3. Editing A[2] 3→5 leaves first output unchanged but changes second by 2·(−1)·2=−4, yielding −7. Setting both mixing weights to zero yields [0,0] for all spatial inputs: teach a true numerical null distinct from an unreachable-input null.

Show the chain-rule overlay on demand for the first output target1: gradient wrt output−4, mixing[4,−4], D_A[−8,−16], D_B[8,0]. “One update” uses η=.01 and exactly simultaneous old-state gradients, giving output−1.9824/loss4.44735488 from author-check-results.json. If rate/target/entities are edited, recompute; never reuse saved gradients. This is optional explanatory depth; no need to animate a whole training campaign. Bound η0… .1 and target−3…3, step1 only. Predict loss direction before stepping; the checked η=.1 contrast gives output5.16/loss8.6528, worse than initial loss8, so do not label every update an improvement. η0 is an exact no-change null.

### Rank task

One input channel, two-tap patch, two output channels. The two probes are x1=[2,0] with target[2,0] and x2=[0,3] with target[0,3]. Display an unsolved starting spatial filter [1,0] and output mixing [1,0], multiplier1. Allow actual filter coefficients and mixing coefficients to be edited, not just a menu of completed solutions. Target residual is the maximum absolute output-target error across both probes. Gate success at error≤1e−9 for exact decimal values; show both residual vectors so fitting only the selected probe cannot succeed. Prediction asks whether the edited candidate will satisfy both before evaluating.

The initial effective matrix[[1,0],[0,0]] matches probe1 and fails probe2. Both outputs must share one spatial pattern at m1; the two-probe identity objective is impossible in this form. Offer a deliberate “add a second spatial filter” action after exploration, beginning new coefficients at zero, not at the answer. At m2 an answer D=[[1,0],[0,1]], P=identity satisfies both. Any invertible change of intermediate basis with inverse mixing may also be valid; grade outputs, not literal coefficient equality. Zeroing all output2 mixing must fail probe2 irrespective of output1. In a separate probe-choice diagnostic, setting probe2 temporarily to[2,0] also changes its identity-function target to[2,0] and invalidates the old attempt. This illustrates why repeated dependent probes cannot certify a kernel; objective reset restores the two independent probes.

Bounds m1/2, filter/mixing coefficients−3…3, exactly two probes. No SVD needed in this tiny lab. Expanded theory names rank, while the main view says “independent spatial patterns.” Static fallback shows the worked identity counterexample and a closed solution to the repair, not a disabled mock interaction.

### Deferred checks

Independent arithmetic of edits and derivative overlay, simultaneous-update ordering, two-probe grading under noncanonical valid factorizations, zero/null cases, undo/reset/new-attempt behavior, keyboard cell editing, signed coefficient legibility and narrow-screen lane ordering. Derivative values and base/step are author checked; UI behavior is not.

## 2. Sampling stencil and serial reachable-site lattice

### Placement and form

Section 3 before formulas: a nine-cell number line with a selected output center and three disconnected weighted taps, an outer span bracket, and explicit padding markers. Next to the receptive-field table, show a layered lattice with highlighted paths from one output to actual input offsets. Switch from 1D to its 2D Cartesian grid for comparison. The lattice should expose holes; filling the whole outline with one color would teach the wrong concept.

### Stencil state and task

Default signal1…9, center index4, three weights[1,1,1], dilation2, stride1, zero padding. Index cells from0 so values cannot be confused with coordinates. Let the learner edit meaningful signal cells, weights and dilation; position0…8, dilation1…8, values−5…20, weights−2…2. Staged edit retains old sum and asks increase/decrease/unchanged before recomputing. Grade direction using |delta|≤1e−10 as unchanged.

Author-checked fixtures: baseline d2 samples2/4/6 and sums15; input6→20 gives28, while input5→20 leaves15. d1 base15 and input5→20 gives29. d8 base5 with two outside taps. All-zero weights give zero for every image, distinct from padding. The arithmetic ramp gives equal d1/d2 sums with different samples; show this as a counterexample to inferring structure from one output.

Result explanation must name selected cells and products: “Index5 was not sampled at dilation2, so its changed value contributed zero here.” For zero coefficients, explain weight zero rather than absence from the stencil. Use real zero-padding cells outside the bounded signal, not invented extrapolated data.

### Lattice construction task

User edits a serial list of 1–4 positive integer dilation rates, each1…9. Start[1,4] with a challenge: preserve a bounding width at least7 while removing all interior unreachable offsets. Predict whether every interior offset becomes reachable before evaluating. Compute exact set recurrence S←{s+u·d: s∈S,u∈{−1,0,1}}; display sorted offsets and holes in [minS,maxS], and width. Valid alternative schedules qualify by the actual bounds/coverage objective, not preset equality. Rates[1,2] is a solution. [1,4] remains unsolved despite gcd1.

Author-checked contrasts: [2,2,2] width13/sites7/2D49of169; [1,2,4]15/sites15/225of225; [1,4]11/sites9/81of121; [1,2,5]17/sites17/289of289; [1,2,9]25/sites21/441of625. Rates[1,3,9] produce27/sites27. Merely reordering rates at stride1 on the unbounded linear structural support leaves the reachable set unchanged; show it as a null for this geometric model, not a guarantee that nonlinear finite networks are identical.

The bounded browser supports at most4 layers with d≤9, so maximum outline73 sites; 2D73²=5329 cells can be rendered as one code-native figure/canvas with textual set access, not thousands of interactive DOM controls. Allow numeric rate editing; the output lattice itself need not be a keyboard-focusable grid of every path.

Add a separate **finite-map** panel for8×8, one3×3 stencil, dilation1/2/4/8. Use author-check-results.json boundary.valid_tap_counts. Selected output position labels how many taps actually address data. At d8 count1 everywhere. The unbounded lattice and finite-map panel must be visibly distinct; otherwise clipping could silently invalidate the lattice theorem.

### Deferred checks

Exact support/set versus brute-force path enumeration, Cartesian2D count, group/sampling distinctions, off-grid clipping, unchanged-rate-order null, changed objective invalidation, all9 input positions and outer-padding edges, visible formula units, and no confusing occupied-cell count with bounding-box area or nonzero gradient.

## 3. Parallel context branches

### Placement and form

Section 5 ASPP subsection: show one source strip/map fanning into several locally sampled branches plus a pooled global branch, then concatenate into an explicit channel vector and multiply by projection weights. Label “small constructed ASPP-style context calculation,” not a trained DeepLab segmentation. Under it a static2D schematic translates to pointwise, rates6/12/18 and global branches at OS16; all outputs share spatial size.

### State and prediction

Use the same9-value strip1…9, center4, branch vector [center, sum(d1), sum(d2), sum(d4), global mean]. Projection weights[0,.1,.2,.3,1]. No activation, bias or normalization in this arithmetic toy. Baseline branch vector[5,15,15,15,5], projection14.

Editable input cells−5…20 and projection coefficients−1…1; branch rates fixed for the primary task to teach common-source parallelism. Stage cell8 from9 to20 and ask which branches change (initial unset checkboxes with a “none” option), plus projection direction. Grade against actual branch deltas and projected delta. Expected: d4 and global mean change; pointwise/d1/d2 do not. Branch d4 becomes26 and mean56/9; projection18.5222222222. Explain that no d1 output feeds d2; all read the original strip.

Null/contrast: swapping cells0/1 preserves global mean5 but changes d4 sum15→16 and projected14→14.3. Setting all projection weights zero makes projected output0 while branch changes remain visible. Editing only branch labels leaves numbers unchanged. A global-mean branch cannot identify a spatial rearrangement by itself.

Author computations retained under parallel in author-check-results.json. For nondefault edited inputs use exact same formulas, not a saved screenshot. After submitting an answer, reveal per-branch read positions and products. Changing projection weights invalidates the projection prediction but still permits showing unchanged source-branch results with explicit current revision; simplest consistent design clears the entire attempt.

### Deferred checks

Parallel versus serial topology, concatenation channel count, global mean denominator9, branchwise and final grading, null with projection zeros, model-text fallback and 2D schematic rate/OS labels. No pretrained DeepLab downloads or mock segmentation masks.

## 4. Real-digit compression and input intervention

### Placement and form

Section 6 after all-seed results: two synchronized digit views for original and chosen factorization, a signed ten-logit comparison and selectable original/reconstructed kernel tiles. This is an inspectable numerical model, not a generic text-output panel. Keep the full recorded experiment table available as context.

### Offline inputs and computation

calculated-inputs.json retains six actual fits. Only seed1/dilation1 and seed1/dilation2 have full dense_state and each rank's factorized_spatial_state, so **editable inference is confined to these two saved models**. Other seeds remain recorded-result rows, not simulated live models. Seed1 selected real inputs: source251/digit4 in both; source299/digit1 for d1 disagreement; source32/digit9 for d2 disagreement. Selecting a different real specimen changes the input; editing its displayed identity alone must not change logits.

No training or SVD in the browser. Use supplied factors for multipliers1/2/4/9, or reconstruct effective spatial kernels by summing P·D. Feed pixel/16 through the exact stem→ReLU→spatial→ReLU→avgpool2→flatten→linear model. Apply bias at the actual locations. Parameter order is NCHW/row-major; avgpool uses nonoverlapping2×2 windows with denominator4 and no padding. Convolution uses cross-correlation, zero padding1 in stem, padding=dilation in second layer. Dense weights and factors are sufficient for true edited-image inference. Quantities are logits, not probabilities; if a probability view is offered, apply numerically stable softmax to all ten logits from the same model.

Use double-precision JS arithmetic and compare with author direct-loop values. Saved float32 PyTorch versus independent NumPy float64 dense predictions differ by at most1.152e−5 on four selected inputs. Do not require bit equality. Target phase-two logit tolerance3e−5 against the saved float32 originals, with closer agreement expected against author-loop results. Check large-magnitude/error relative tolerances deliberately; never hide a class mismatch as rounding.

### Tasks and accurate feedback

First choose a fixed real input/model. Before revealing the rank-one result, ask whether predicted class will match the dense model; show an unset yes/no choice. Grade using computed argmax (lowest index on a tie). Reveal both class scores, true label and whether each is correct. If the class stays the same, show logit/confidence differences rather than calling the models identical.

Then offer a meaningful **pixel intervention**: choose a cell and new intensity0…16, or paint bounded selected cells with keyboard input alternative. Keep the original image for comparison; edited images are constructed corruptions of a real specimen, not newly observed digit samples. Select one class score to inspect, preferably the source's true class. Prediction is increase/decrease/unchanged for that score before evaluation. Grade with tolerance1e−8 for the deterministic double-precision model. The actual label remains a reference to the original specimen, not a claim that every arbitrary edit still depicts that digit.

Checked source251,d1 edit normalized[0,0]0→1 decreases actual-class logit by.8635377838; source299,d1 decreases1.4089149643. Source251,d2 decreases1.4188996912; source32,d2 **increases**1.1825324818. These contrasting signs come from real saved weights, not a rule that adding white pixels raises confidence. All four author edited logits and argmaxes are retained in author-check-results.json. Restoring the original pixel is an exact null versus original inference; editing only the specimen label is another null. For kernel-only view, rank9 reconstruction preserves the learned function within float tolerance while increasing MACs.

An optional challenge sets a narrow edit budget, such as one corner pixel, and asks the learner to find a change that decreases the selected class logit while retaining the original predicted class. Begin unsolved; accept any bounded solution satisfying recomputed score and class constraints, and disclose if no attempted candidate succeeded. Do not grade against an absent target or silently substitute a pre-solved image. This is an optional extension because the two primary prediction tasks already teach distinct concepts.

### Bounds, rendering and deferred work

64 editable pixels, two saved models, four available factorized layers, ten classes. Debounce edits and recompute only the two selected forward paths; at these dimensions arithmetic is bounded, but measure responsiveness during phase two. Keep data lazy and retain a static all-seed result table if loading fails. Release inactive model state when leaving the investigation; no perpetual canvas loop, worker or tensor runtime is required for tiny direct arithmetic.

For filters show dimensions/output-input selection and shared diverging scale within a comparison. Never compare independently auto-scaled originals and reconstructions without a scale label. Show reconstruction residuals numerically for a chosen filter. A large rank error is not a saliency map.

Deferred: independent full-model inference for all saved selected inputs and each factorization; edited-pixel contrasts and nulls; rank9 tolerance; raw-intensity/normalized-intensity conversion; focus preservation, keyboard painting/reset, labels/true-label caveat, lazy network failure, narrow-screen signed-logit layout and actual browser responsiveness. No on-device latency superiority, training recovery or causal explanation is inferred from these interactions.

## 5. Supporting static figures and complete author handoff

- Section2 cost table uses stacked arithmetic areas or aligned bars for depthwise and pointwise weights. Values are exact integers; put zero at bar origin and distinguish weights from MACs. It does not need its own generic slider lab.
- Section3 RF recurrence shows old incoming span and center jump before adding dilated offsets. Include one stride example so jump units survive the picture.
- Section5 MBConv diagram places expansion before downsampling and shows each actual H×W×C shape. A separate tiny hard-swish graph is formula-derived, with marked−3/3 joins and negative portion; hard-sigmoid is not the same curve.
- Section6 real comparison is a grouped result table/dot plot with all six runs, no interpolated training-time performance curve or invented error bars. Its axis is correct/120 or CE with declared units. Seeds are repeated training randomness on one split, not independent datasets.

Author reread of the full manuscript and these contracts is required before checkpoint. The code/input calculations supplied here are bounded author evidence. Phase-two owner must implement the intended pedagogy, assess all material issues discovered during rendering, and complete formal numerical/independent/browser/accessibility/integration work separately. Reuse unchanged data and verified calculations; do not rerun six fits merely because a caption or focus behavior changes.
