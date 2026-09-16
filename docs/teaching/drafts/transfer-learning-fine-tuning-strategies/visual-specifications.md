# Transfer learning: visual and investigation contracts

Content-only packet. These specifications are prepared; no components, SVGs, browser models or publication changes have been implemented. Read the whole manuscript and complete CPU program before implementing. Saved arithmetic and fitted observations are in calculated-inputs.json; real pixels are in digits-400.csv.

## Shared behavior and evidence contract

Use computation diagrams, matrix editors, specimen partitions and an evidence-selection workspace for their distinct mechanisms. Do not replace them with a generic slider plus paragraph. Keep numerical claims bound to the formulas or saved runs below. Unit labels, denominators and the selected seed remain visible. Color always has a text, shape or line-style counterpart.

Each graded investigation starts with prediction unset. A prediction is an explicit choice or numeric entry submitted for the current complete input revision; there is no selected default. Grade the stated target from actual computed/saved results and explain the difference between the learner's prediction and observation. Any edit to a causal input invalidates prediction, feedback and revealed answer, even if the numerical answer happens to remain unchanged. A reset restores defaults and removes prediction/grade; it does not prefill the correct answer. View-only selection, such as expanding source code, need not invalidate an unrelated task.

Text alternatives must express the mathematical relationships, not merely say “chart.” Keyboard controls supplement every drag or matrix edit. Use bounded numerical fields, labeled row/column meaning, visible focus and explicit finite-input errors. Announce submitted feedback politely once, not on every keystroke. At 360 px, stack linked panels in causal order and preserve matrix headers; use a local horizontal scroller only for a genuinely two-dimensional matrix. Respect reduced motion and provide step buttons; no autoplay is required.

Browser execution is small arithmetic or selection over saved records. The 18 neural fits run in the downloadable offline program, never automatically in the browser. Import only this topic's small necessary data on demand; do not eagerly include the entire manuscript's JSON in navigation.

## 1. Reuse a measuring instrument, replace its meaning

Place after §1's 64→32→16→5 description. Draw real 8×8 digit input → two tanh layers → 16 features → five source score sockets. Below, align the reused backbone with a new head, with output labels 5–9. The two heads have identical shape but different label mappings. Use a visible “copied learned weights” connector from backbone to backbone; source head is saved off to the side, not silently deleted.

Inputs are one actual source training specimen and one target training specimen, found by saved source IDs. Pixel intensity uses the known 0–16 scale. Hidden boxes represent dimensions and relationships, not measured activation values unless those are separately generated and saved in phase two. Caption: “Same dimensions do not imply the same label meanings.”

A tap/keyboard choice highlights one corresponding stage and shows its tensor shape: one specimen (64), hidden (32), features (16), logits (5). Batch view is optional and explicitly adds a leading N axis. Explain the layer's weights as out×in. Text alternative lists the two label orders and the shared/replaced parts.

Contrast: selecting the old head on target features retains output meanings 0–4; do not simply relabel old scores as correct target predictions. Selecting a different recorded specimen changes the input while leaving model ownership unchanged. Editing only an identity label must not change the numerical input or prediction. This is an explanatory diagram, not a graded model-evaluation lab.

## 2. Three controls behind a freeze

Place in §2. Show aligned lanes for trainable parameters, optimizer membership, and running state. A small BatchNorm state transition is more useful here than an abstract network-colored heatmap.

Model: one-channel BatchNorm, affine parameters frozen, old running mean 0 and variance 1, momentum 0.1, training input [1,3]. Population batch variance is 1; the unbiased running update uses variance 2. A training forward inside no_grad yields running mean 0.2 and variance 1.1, as recorded by mechanisms.frozen_bn_training_no_grad. Evaluation on [11,13] leaves them 0.2 and 1.1, recorded in frozen_bn_eval_no_grad. Label statistics, weight values and gradient graph separately.

Graded question: with frozen affine weights and no_grad selected, predict whether a training forward changes running statistics. Prediction yes/no is initially unset. Run reveals both exact state transitions and grades against “yes.” Learner can then change module mode to evaluation and predict again; output remains computable while the stored statistics no longer update. The contrasting change is mode, not an unrelated learning-rate slider. Null: toggling no_grad while preserving mode does not itself switch which BatchNorm statistics are used or updated.

For the null variant with graph recording enabled, make input require gradients so the graph indicator is meaningful; do not call backward when no graph exists. No random dropout visualization is needed for this activity. A neighboring static derivative edge from frozen W to an upstream input uses the recorded [2,1] gradient and “weight gradient: absent,” connecting freeze policy to LoRA.

Bound input to two finite values in [−20,20] and momentum [0,1]. If allowing edits, recompute mean, biased/unbiased variance and EMA explicitly; at one value, explain the training cardinality error instead of showing NaN. Reset resets the running state as well as the controls. Do not update buffers just because the component rendered again.

## 3. Source and target evidence are different partitions

Place before §3's program. Show five bins containing specimen counts: source train 150, source holdout 50, target train 40, target validation 60, target test 100. Source digits 0–4 and target digits 5–9 have different labels. A compact expandable ID table maps every specimen to exactly one bin.

Forward arrows: source train → pretrained backbone; target train → candidate fitting; target validation → choice; exact chosen artifact → target test. The source holdout arrow leads to a separate retention report, not model selection. Lock-shaped markers on test distinguish “unconsumed” versus “reported,” with text meaning. Selection does not turn a target row into a source row.

Rows derive from splits_source_ids, not a fresh random split in the browser. Verify all 400 IDs are unique across partitions and map to the CSV. Label these fixed row blocks, not writer groups or the official UCI benchmark. The chart must retain the target test 77/100 observation when discussed later; do not improve apparent performance by choosing easier displayed rows.

This diagram is explanatory. A short ungraded check asks why target digit 5 uses training class index 0 in the new five-output head. Its answer points to digit−5 in the program and the target head order, while source index 0 had meant digit 0.

## 4. LoRA factor workshop

Place with §4. Draw a two-path computation: frozen 2×2 W sends x directly to the sum; editable A compresses x to one value, editable B expands it to two values, then scale and sum. Matrices and vectors use aligned cells and animated correspondence only on explicit Step. Show the dimensions near multiplication points.

Default: W identity, A [1,−1], B [0,0]ᵀ, x [2,1]ᵀ, zero target, scale 1, learning rate 0.1, mean squared loss over two outputs. Before revealing gradients, learner predicts which factor has a nonzero gradient: A only, B only, both, neither. No default selected. Compute the derivative formula from §6 and grade. Saved default: loss 2.5, grad A zero, grad B [2,1], after output [1.8,.9], loss 2.025.

Separate prediction: enter the first output after one step, tolerance 1e−6 in ordinary units. “Show one update” reveals the matrix difference, not just a loss number. Submission grades component errors and explains which multiplication determined the missed value.

Meaningful unsolved edit task: “Change the input to make both factor gradients zero while the loss remains positive; keep A, B, W and target fixed.” Start from the unsolved default. Accept when both gradient maxima ≤1e−10 and loss >1e−6. Example fixture x [1,1] gives Ax=0, zero gradients, loss 1. The solution is not preloaded. Text feedback distinguishes zero loss from a bottleneck receiving no signal.

Checked contrast x [1,3]: grad B [−2,−6], updated output [.6,1.8], loss 1.8. Checked null: both factors initialized zero, x [2,1], both gradients zero. This must not receive an “optimization succeeded” badge. A separate reset to the source default restores random-factor/zero-factor asymmetry, not a random new numerical problem.

Bound matrix entries and inputs to [−5,5], learning rate to [0,.5], rank to 1 or 2, fixed dimensions 2×2. Recompute on submit; no unbounded training loop. If both factors update on a step, compute all gradients from the same pre-update state. Show both algebraic merged output and floating-point tolerance; do not require bit equality. Rank 2 expands the intermediate vector and dimensions; prediction invalidates. Provide row-major text tables, exact formulas, keyboard cell inputs and a stacked narrow layout.

## 5. Adapter feature detour and parameter budget

Place with §5. Align h's direct path with down 16→4, tanh, up 4→16, then addition. At initialization the up weight/bias are zero, so highlight that the added vector is zero while h still passes through. Mark the different location from LoRA: this edits features after the backbone, rather than the backbone's linear weights.

Count display is derived: down 16×4+4, up 4×16+16, total adapter 148, head 16×5+5=85, total trainable 233. Toggle “include head” and update the count honestly. Show a small exact byte budget for declared FP32 gradients+two moments (12 bytes per trainable value), clearly excluding other memory. No hardware bars imply actual runtime or allocation measurements.

Optional ungraded dimension editor: d in [2,1024], bottleneck in [1,64], integer; count 2db+b+d. A parallel LoRA count dk versus r(d+k) helps distinguish method counts. Fixtures: d4,k6,r2 →20<24; r3→30>24. Null: changing a displayed decimal GB/GiB unit changes the numerical representation of bytes, not the number of stored values. This is an exact accounting view; do not turn it into a fabricated performance predictor.

## 6. Choose a model from evidence, then report once

Place in §7. Use a validation table plus selectable actual loss traces and source-retention dumbbells, not one generic “accuracy” line. Source before/after results retain denominators 50; target validation denominator 60 is distinct. Axis for loss: mean CE in nats. Axis for training: full-batch update count. Only steps 0,1,10,100,300 exist in the saved target traces; thin connecting segments must identify samples rather than promise interpolated measurements.

Investigation A: choose which seed-1 candidate the predeclared minimum-final-validation-CE rule selects. Prediction unset; target test result initially concealed. Submission grades against scratch and explains 0.123143 is the smallest CE. Reveal its one actual held-out result, 77/100 and CE .757101, only after selection. An incorrect choice shows the validation comparison and permits correction; it must not display a fabricated test result for that method. Selecting another candidate afterward is a development inspection, not a new untouched test.

Investigation B is a separate explicitly hypothetical resource constraint applied to the same saved validation observations. Learner edits a maximum trainable-parameter budget, then predicts the eligible winner. At budget 400, scratch/full/discriminative are excluded; LoRA2 at 373 and CE .369318 wins over probe and adapter. At budget 250, adapter4 at 233 has much worse CE than probe, so probe wins. At budget 84, no candidate is eligible; show “no eligible candidate” rather than silently lifting the constraint. At 5000 the unconstrained scratch winner returns. Budgets 400 and 500 are a null contrast: the eligible set and selected winner are unchanged. Grade from the actual records with stable method-order ties.

This constraint exercise does not have a new final test result. Explicit label: “Reinterpret existing validation records; no new model was trained or tested.” Require an initially unset prediction for each budget revision. A parameter-budget input is a real decision constraint; do not let UI cosmetics masquerade as experimental inputs.

For retention, selecting probe shows exactly unchanged source loss and 49/50 accuracy at seed 1. LoRA2 shows 49→44 with frozen base. Full shows 49→49 but a changed CE. Ask an ungraded interpretation question: which claim does this falsify? Answer: frozen base guarantees unchanged adapted behavior. It does not establish a universal method ranking. Seed selection invalidates any target-selection exercise and clearly identifies seed 2/3 as sensitivity views, not new selection candidates.

On narrow screens retain the metric labels beside each method, then display one selected trace below. Avoid six overplotted curves as the only representation. All values are available as an accessible table.

## 7. Save the function; show the next topic's connection

Place in §8 and link to the initialization transition. Draw a checkpoint package with architecture, base identity, adapter configuration, weights/buffers, pixel order/scale and class order as distinct required fields. The head-label and preprocessing fields must be as prominent as the tensor payload. A local replay fixture shows max probability difference zero for the recorded state round trip; explicitly identify that configuration was held fixed.

A descriptive failure comparison keeps tensor shapes constant while swapping labels [5,6,7,8,9] to [9,8,7,6,5]. Probabilities do not change but their declared meaning does. Do not claim model accuracy under remapping without computing it from actual labels. An input-preprocessing version change invalidates the replay fixture.

The optional schedule diagram in §9 plots the exact teaching formula at t0,10,100 with rates .0003125,.01,.0003125. Label it “specified schedule,” not an observed loss/learning curve or the exact original ULMFiT implementation. A static table is sufficient if an interactive schedule editor would distract from transfer.

## Deferred phase-two acceptance

Implement the semantic topic-owned figures/models and lazy downloads. Execute the final displayed full program and any separate exposed snippet under its shown setup; match the saved split and candidate protocol. Verify factor gradients with an independent calculation, original/merged forward tolerance, actual frozen-state behavior, all selection/budget contrast/null cases, data identity and checkpoint replay. Preserve unfavorable results.

Conduct independent correctness and learning-experience review, then browser, keyboard, screen-reader/text-alternative, narrow-screen, overflow, reduced-motion and input-invalidation checks. Check code/download availability, topic/module route identity and the immediate initialization continuation. Only after these and integration may implementation be marked complete. These specifications and author calculations do not constitute those checks.
