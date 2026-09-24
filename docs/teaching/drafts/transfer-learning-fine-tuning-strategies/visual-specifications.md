# Transfer learning: visual and investigation contracts

## Live exploration contract — 21 September 2026

Open each investigation with its current inputs, intermediate mechanism and complete current output visible. Apply valid edits to meaningful entities immediately and update diagrams, tables, units and causal explanation together. No prediction entry, predicted-answer choices, commitment, prediction grading or answer-unlock feature is part of this packet, even optionally. Model predictions and mathematical masks/gates remain subject matter.

Use the topic-specific controls and checked fixtures below. Pair sliders or direct manipulation with labeled keyboard/numeric controls; keep presets as starting points, not the only editable values. A pinned baseline preserves its inputs, seed, units and outputs while the current case changes. Explain both a meaningful contrast and an unchanged/null result, then connect the observed effect to a practical design decision. Reset restores the stated fixture and current result. Invalid text has a local explanation and a clearly identified last valid result; never silently clamp or pair new inputs with old output.

Step/Back and bounded Run controls advance a real computation or reveal its chronological stages, not permission to view an answer. Show the current state and its result throughout. Keep exact small calculations live. For costly frozen inference, debounce or run bounded work with pending/current state labels and stale-result cancellation; inspect saved measurements without implying fresh training. Respect reduced motion, keep focus stable and avoid announcing every animation frame. Independent written practice and its hints/solutions stay separate.

Phase two must test default results without any action, meaningful edits, quick consecutive edits, valid extremes, null/invalid cases, reset, linked-view agreement, keyboard operation and readable phone layouts. The mathematical/reference checks already specified below remain; these live browser checks have not been performed in this content-only revision.

### Topic-specific live route
**Inspect what freezing actually freezes.** Toggle parameter updates, gradient recording and module mode; edit LoRA factors/rate; change parameter budgets over saved validation candidates.
**See the consequence.** Show parameters, buffers, gradients and before/after function values separately. Display eligible candidates and the validation-selected winner live; the one retained test result remains clearly identified as previously observed.
**Decision connection.** Select an adaptation strategy under a real budget and avoid confusing frozen weights with frozen behavior or repeated inspection with a fresh test.


Content-only packet. These specifications are prepared; no components, SVGs, browser models or publication changes have been implemented. Read the whole manuscript and complete CPU program before implementing. Saved arithmetic and fitted observations are in calculated-inputs.json; real pixels are in digits-400.csv.

## Shared behavior and evidence contract

Use computation diagrams, matrix editors, specimen partitions and an evidence-selection workspace for their distinct mechanisms. Do not replace them with a generic slider plus paragraph. Keep numerical claims bound to the formulas or saved runs below. Unit labels, denominators and the selected seed remain visible. Color always has a text, shape or line-style counterpart.

Each investigation follows the live exploration contract above: current results are visible immediately, valid entity edits update all linked views, and comparisons explain the mechanism. Reset restores the declared inputs and recomputes their result. No prediction or answer-submission state is retained.

Text alternatives must express the mathematical relationships, not merely say “chart.” Keyboard controls supplement every drag or matrix edit. Use bounded numerical fields, labeled row/column meaning, visible focus and explicit finite-input errors. Announce a settled result politely, not on every keystroke. At 360 px, stack linked panels in causal order and preserve matrix headers; use a local horizontal scroller only for a genuinely two-dimensional matrix. Respect reduced motion and provide step buttons; no autoplay is required.

Browser execution is small arithmetic or selection over saved records. The 18 neural fits run in the downloadable offline program, never automatically in the browser. Import only this topic's small necessary data on demand; do not eagerly include the entire manuscript's JSON in navigation.

## 1. Reuse a measuring instrument, replace its meaning

Place after §1's 64→32→16→5 description. Draw real 8×8 digit input → two tanh layers → 16 features → five source score sockets. Below, align the reused backbone with a new head, with output labels 5–9. The two heads have identical shape but different label mappings. Use a visible “copied learned weights” connector from backbone to backbone; source head is saved off to the side, not silently deleted.

Inputs are one actual source training specimen and one target training specimen, found by saved source IDs. Pixel intensity uses the known 0–16 scale. Hidden boxes represent dimensions and relationships, not measured activation values unless those are separately generated and saved in phase two. Caption: “Same dimensions do not imply the same label meanings.”

A tap/keyboard choice highlights one corresponding stage and shows its tensor shape: one specimen (64), hidden (32), features (16), logits (5). Batch view is optional and explicitly adds a leading N axis. Explain the layer's weights as out×in. Text alternative lists the two label orders and the shared/replaced parts.

Contrast: selecting the old head on target features retains output meanings 0–4; do not simply relabel old scores as correct target predictions. Selecting a different recorded specimen changes the input while leaving model ownership unchanged. Editing only an identity label must not change the numerical input or prediction. This is an explanatory diagram, not a computed model-evaluation lab.

## 2. Three controls behind a freeze

Place in §2. Show aligned lanes for trainable parameters, optimizer membership, and running state. A small BatchNorm state transition is more useful here than an abstract network-colored heatmap.

Model: one-channel BatchNorm, affine parameters frozen, old running mean 0 and variance 1, momentum 0.1, training input [1,3]. Population batch variance is 1; the unbiased running update uses variance 2. A training forward inside no_grad yields running mean 0.2 and variance 1.1, as recorded by mechanisms.frozen_bn_training_no_grad. Evaluation on [11,13] leaves them 0.2 and 1.1, recorded in frozen_bn_eval_no_grad. Label statistics, weight values and gradient graph separately.

**Live freeze comparison:** with frozen affine weights and no_grad, execute a training forward and show the actual running-statistic transition. Switch to evaluation and show its unchanged stored buffers while outputs remain computable. Toggling no_grad alone does not change which BatchNorm statistics are used or updated. Keep parameter updates, module mode and gradient recording as independent controls.

For the null variant with graph recording enabled, make input require gradients so the graph indicator is meaningful; do not call backward when no graph exists. No random dropout visualization is needed for this activity. A neighboring static derivative edge from frozen W to an upstream input uses the recorded [2,1] gradient and “weight gradient: absent,” connecting freeze policy to LoRA.

Bound input to two finite values in [−20,20] and momentum [0,1]. If allowing edits, recompute mean, biased/unbiased variance and EMA explicitly; at one value, explain the training cardinality error instead of showing NaN. Reset resets the running state as well as the controls. Do not update buffers just because the component rendered again.

## 3. Source and target evidence are different partitions

Place before §3's program. Show five bins containing specimen counts: source train 150, source holdout 50, target train 40, target validation 60, target test 100. Source digits 0–4 and target digits 5–9 have different labels. A compact expandable ID table maps every specimen to exactly one bin.

Forward arrows: source train → pretrained backbone; target train → candidate fitting; target validation → choice; exact chosen artifact → target test. The source holdout arrow leads to a separate retention report, not model selection. Lock-shaped markers on test distinguish “unconsumed” versus “reported,” with text meaning. Selection does not turn a target row into a source row.

Rows derive from splits_source_ids, not a fresh random split in the browser. Verify all 400 IDs are unique across partitions and map to the CSV. Label these fixed row blocks, not writer groups or the official UCI benchmark. The chart must retain the target test 77/100 observation when discussed later; do not improve apparent performance by choosing easier displayed rows.

This diagram is explanatory. A short ungraded check asks why target digit 5 uses training class index 0 in the new five-output head. Its answer points to digit−5 in the program and the target head order, while source index 0 had meant digit 0.

## 4. LoRA factor workshop

Place with §4. Draw a two-path computation: frozen 2×2 W sends x directly to the sum; editable A compresses x to one value, editable B expands it to two values, then scale and sum. Matrices and vectors use aligned cells and animated correspondence only on explicit Step. Show the dimensions near multiplication points.

Default: W identity, A [1,−1], B [0,0]ᵀ, x [2,1]ᵀ, zero target, scale 1, learning rate 0.1, mean squared loss over two outputs. Show the current output and both factor gradients immediately, identifying which gradients are nonzero and why. Compute the derivative formula from §6 and show the computed values and their cause. Saved default: loss 2.5, grad A zero, grad B [2,1], after output [1.8,.9], loss 2.025.

Additional live readout: display the first output after one step; implementation parity tolerance is 1e−6 in ordinary units. “Show one update” reveals the matrix difference, not just a loss number. The live view shows component errors and explains which multiplication determined the missed value.

Meaningful live contrast: keep A, B, W and target at their starting values and change x from [2,1] to [1,1]. Display Ax, both factor gradients and loss throughout. At x [1,1], Ax=0, both gradients are zero and loss is 1. Explain zero loss separately from a bottleneck receiving no signal. Thresholds of 1e−10 for gradient magnitude and 1e−6 for positive loss may support explanatory numerical wording; they are not an answer check, acceptance condition or unlock rule.

Checked contrast x [1,3]: grad B [−2,−6], updated output [.6,1.8], loss 1.8. Checked null: both factors initialized zero, x [2,1], both gradients zero. This must not receive an “optimization succeeded” badge. A separate reset to the source default restores random-factor/zero-factor asymmetry, not a random new numerical problem.

Bound matrix entries and inputs to [−5,5], learning rate to [0,.5], rank to 1 or 2, fixed dimensions 2×2. Recompute on every valid edit; no unbounded training loop. If both factors update on a step, compute all gradients from the same pre-update state. Show both algebraic merged output and floating-point tolerance; do not require bit equality. Rank 2 expands the intermediate vector and dimensions; current outputs update. Provide row-major text tables, exact formulas, keyboard cell inputs and a stacked narrow layout.

## 5. Adapter feature detour and parameter budget

Place with §5. Align h's direct path with down 16→4, tanh, up 4→16, then addition. At initialization the up weight/bias are zero, so highlight that the added vector is zero while h still passes through. Mark the different location from LoRA: this edits features after the backbone, rather than the backbone's linear weights.

Count display is derived: down 16×4+4, up 4×16+16, total adapter 148, head 16×5+5=85, total trainable 233. Toggle “include head” and update the count honestly. Show a small exact byte budget for declared FP32 gradients+two moments (12 bytes per trainable value), clearly excluding other memory. No hardware bars imply actual runtime or allocation measurements.

Optional ungraded dimension editor: d in [2,1024], bottleneck in [1,64], integer; count 2db+b+d. A parallel LoRA count dk versus r(d+k) helps distinguish method counts. Fixtures: d4,k6,r2 →20<24; r3→30>24. Null: changing a displayed decimal GB/GiB unit changes the numerical representation of bytes, not the number of stored values. This is an exact accounting view; do not turn it into a fabricated performance predictor.

## 6. Choose a model from evidence, then report once

Place in §7. Use a validation table plus selectable actual loss traces and source-retention dumbbells, not one generic “accuracy” line. Source before/after results retain denominators 50; target validation denominator 60 is distinct. Axis for loss: mean CE in nats. Axis for training: full-batch update count. Only steps 0,1,10,100,300 exist in the saved target traces; thin connecting segments must identify samples rather than promise interpolated measurements.

Investigation A displays all retained seed-1 validation candidates and highlights the winner under the predeclared minimum-final-validation-CE rule. Scratch wins with CE .123143. Show its one recorded held-out result, 77/100 and CE .757101, as an already observed report. Changing the highlighted method is development inspection and cannot create an unretained test result or a new untouched test. Explain the actual selection rule and distinguish validation evidence from that one test report.

Investigation B is a separate explicitly hypothetical resource constraint applied to the same saved validation observations. Editing a maximum trainable-parameter budget immediately displays the eligible set and its validation winner, without prediction entry or submission. At budget 400, scratch/full/discriminative are excluded; LoRA2 at 373 and CE .369318 wins over probe and adapter. At budget 250, adapter4 at 233 has much worse CE than probe, so probe wins. At budget 84, no candidate is eligible; show “no eligible candidate” rather than silently lifting the constraint. At 5000 the unconstrained scratch winner returns. Budgets 400 and 500 are a null contrast: the eligible set and selected winner are unchanged; compute from the actual records with stable method-order ties.

This constraint exercise does not have a new final test result. Explicit label: “Reinterpret existing validation records; no new model was trained or tested.” Require a current live comparison for each budget revision. A parameter-budget input is a real decision constraint; do not let UI cosmetics masquerade as experimental inputs.

For retention, selecting probe shows exactly unchanged source loss and 49/50 accuracy at seed 1. LoRA2 shows 49→44 with frozen base. Full shows 49→49 but a changed CE. Ask an ungraded interpretation question: which claim does this falsify? Answer: frozen base guarantees unchanged adapted behavior. It does not establish a universal method ranking. Seed selection invalidates any target-selection exercise and clearly identifies seed 2/3 as sensitivity views, not new selection candidates.

On narrow screens retain the metric labels beside each method, then display one selected trace below. Avoid six overplotted curves as the only representation. All values are available as an accessible table.

## 7. Save the function; show the next topic's connection

Place in §8 and link to the initialization transition. Draw a checkpoint package with architecture, base identity, adapter configuration, weights/buffers, pixel order/scale and class order as distinct required fields. The head-label and preprocessing fields must be as prominent as the tensor payload. A local replay fixture shows max probability difference zero for the recorded state round trip; explicitly identify that configuration was held fixed.

A descriptive failure comparison keeps tensor shapes constant while swapping labels [5,6,7,8,9] to [9,8,7,6,5]. Probabilities do not change but their declared meaning does. Do not claim model accuracy under remapping without computing it from actual labels. An input-preprocessing version change invalidates the replay fixture.

The optional schedule diagram in §9 plots the exact teaching formula at t0,10,100 with rates .0003125,.01,.0003125. Label it “specified schedule,” not an observed loss/learning curve or the exact original ULMFiT implementation. A static table is sufficient if an interactive schedule editor would distract from transfer.

## Deferred phase-two acceptance

Implement the semantic topic-owned figures/models and lazy downloads. Execute the final displayed full program and any separate exposed snippet under its shown setup; match the saved split and candidate protocol. Verify factor gradients with an independent calculation, original/merged forward tolerance, actual frozen-state behavior, all selection/budget contrast/null cases, data identity and checkpoint replay. Preserve unfavorable results.

Conduct independent correctness and learning-experience review, then browser, keyboard, screen-reader/text-alternative, narrow-screen, overflow, reduced-motion and input-invalidation checks. Check code/download availability, topic/module route identity and the immediate initialization continuation. Only after these and integration may implementation be marked complete. These specifications and author calculations do not constitute those checks.
