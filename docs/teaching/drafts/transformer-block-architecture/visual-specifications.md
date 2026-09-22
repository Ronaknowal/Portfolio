# Transformer Block Architecture — visual and investigation contracts

## Live exploration contract — 21 September 2026

Open each investigation with its current inputs, intermediate mechanism and complete current output visible. Apply valid edits to meaningful entities immediately and update diagrams, tables, units and causal explanation together. No prediction entry, predicted-answer choices, commitment, prediction grading or answer-unlock feature is part of this packet, even optionally. Model predictions and mathematical masks/gates remain subject matter.

Use the topic-specific controls and checked fixtures below. Pair sliders or direct manipulation with labeled keyboard/numeric controls; keep presets as starting points, not the only editable values. A pinned baseline preserves its inputs, seed, units and outputs while the current case changes. Explain both a meaningful contrast and an unchanged/null result, then connect the observed effect to a practical design decision. Reset restores the stated fixture and current result. Invalid text has a local explanation and a clearly identified last valid result; never silently clamp or pair new inputs with old output.

Step/Back and bounded Run controls advance a real computation or reveal its chronological stages, not permission to view an answer. Show the current state and its result throughout. Keep exact small calculations live. For costly frozen inference, debounce or run bounded work with pending/current state labels and stale-result cancellation; inspect saved measurements without implying fresh training. Respect reduced motion, keep focus stable and avoid announcing every animation frame. Independent written practice and its hints/solutions stay separate.

Phase two must test default results without any action, meaningful edits, quick consecutive edits, valid extremes, null/invalid cases, reset, linked-view agreement, keyboard operation and readable phone layouts. The mathematical/reference checks already specified below remain; these live browser checks have not been performed in this content-only revision.

### Topic-specific live route
**Follow a complete block.** Edit features, normalizer offset/scale, FFN matrices, branch multiplier, pre/post placement and supported retained trajectory inputs.
**See the consequence.** Update normalization reference sets, feature writes, residual state, block output and derivative paths together. Show exact zero-branch and common-shift cases.
**Decision connection.** Reason about placement and branch scaling using their actual function and gradient effects rather than an architecture slogan.


Content-first specification, 13 September 2026. No figures, browser controls or runtime have been implemented. The accompanying complete `lesson.md` supplies the learner prose; this file specifies representations, computations and verification states. The source inputs and both actual measured models are retained for a later authorized implementation. Do not replace these mechanisms with a uniform text-output lab or a single generic architecture box.

## Shared presentation, correctness and interaction contract

Use precise code-native matrices, wiring, signed bars and axis geometry, with distinct representations for each mechanism. Current outputs are visible and valid entity edits recompute them. Presets do not exhaust the allowed inputs; new cases use actual formulas or saved weights, never interpolated probabilities.

Label position/feature indices, signed response/update, RMS, L2 norm, attention weight, class probability, logit, nats/target, parameters and MACs accurately. Preserve negative values and full calculation precision. Use color plus text/shape, exact numeric tables, keyboard controls and concise settled-state announcements.

On phones, select a position/stage/feature or model while retaining a labeled baseline; reflow wiring vertically. Do not shrink a 90-value matrix until unreadable. Respect reduced motion and provide direct step selection. Exact fixtures are capped at 6 positions, 8 features, 2 heads and 16 FFN responses. The real model has width 24, 2 heads, 2 blocks, 15 classes and at most 60 valid/padded records. Use bounded debounced inference with stale-result cancellation where needed; no browser fitting. Keep imports/data topic-owned and distinguish saved CPU evidence from future browser verification.

## F1 — communication lanes and the carried state

**Placement:** §1 after the operation/job table. **Question:** which operation can directly read another position, and what interface lets the next block begin?

Draw three input rows as horizontal position lanes, each with a small feature strip, entering a block and leaving with the same feature width. Attention has explicit cross-position query-to-key/value edges; FFN uses local up/activation/down shapes within each lane. Two addition junctions visibly combine an incoming carried state with the corresponding update. Label the sequence direction and feature direction; use `(B,L,d)` at entry/exit and `(B,L,f)` only inside the FFN. Clearly distinguish a “layer” meaning this entire block from a “linear layer” inside its FFN.

The caption explains that each position's FFN is the same learned function even though its contextual inputs differ. Selecting a lane may highlight dependencies, but this figure does not need a prediction form: the concept is static structure, with full table/text equivalents. Verify that the diagram does not depict the FFN as combining positions or show separate independently learned weights per position. Mobile can show one receiver plus labeled donors above it.

## F2 — the two normalization circuits

**Placement:** §2 beside the pre/post equations. Draw exactly these routes:

Pre: `X → fork`; branch `N1 → Attention → +X = Z`; branch from `Z` through `N2 → FFN → +Z = Y`.

Post: `X → Attention → +X → N1 = Z → FFN → +Z → N2 = Y`.

Bypass arrows must originate at the correctly named unmodified tensor. Normalization in pre-norm never moves onto the bypass. Do not draw the FFN reading the original X in the sequential version. Distinguish a junction's outgoing state from a sublayer's branch output. Toggle a zero-update overlay to mute attention/FFN updates and expose pre's identity route versus post's two remaining norms. The four-feature zero fixture is in `author-results.json.fixtures.trace.zero_pre/zero_post`; use exact vectors, with displayed rounded post row 1 `[-1.095440,-.730293,.365147,1.460586]`.

The toggle reveals a computed structural control, but does not claim zero-initializing every branch parameter would yield a useful trainable network. That separate initialization question belongs to the earlier residual lesson. Keyboard focus order follows the equation; text description names each operation in sequence. On mobile stack two labeled circuits, not a side-by-side miniature.

## I1 — choose the normalizer's reference frame

**Placement:** §3 after offset/scale explanation. **Goal:** discover centering versus scaling using actual feature vectors, not an animation of an unexplained formula.

Start with editable `x=[1,2,5,8]`, epsilon `1e-5`, identity gain and zero LayerNorm offset. The primary view is aligned dot plots for raw x, centered deviations, LN output and RMSNorm output. Mark zero, mean and a ruler representing the relevant denominator. A small table shows `mean`, `mean squared deviation`, `mean square`, `RMS` and `L2`; formula terms highlight matching marks. Default widths are four features; allow 2–8. Numeric inputs have reasonable defaults and bounded finite domain; explicitly reject nonfinite entries. Epsilon editable in a documented positive range such as `1e-8`–`.1` with a log-scale slider and numeric alternative.

Controls change a common offset/positive scale and a direct feature value. Show LN and RMSNorm before/after tensors and numerical differences immediately, with a pinned baseline. The optional affine controls expose gain/bias without changing the main identity-affine comparison. Unequal learned affine parameters can change mean/norm; the same gain applies to both methods, and only LN has the displayed offset.

**Checked contrast:** adding 5 to `[1,2,5,8]` leaves LN exactly unchanged in the saved calculation, while RMSNorm changes from `[.206284,.412568,1.031421,1.650274]` to `[.637793,.744092,1.062988,1.381884]`. The L2 of the original LN vector is 1.9999986667 and RMS .9999993333; do not make the length label one. **Checked conceptual null:** any zero-mean input with matching gain, zero offset and epsilon makes the two definitions agree. `[−1,−1,1,1]` is a useful selectable null fixture with its matching result immediately visible; arbitrary edits must recompute the same quantities. Zero input returns zero for both with positive epsilon; a nonzero constant input only becomes zero under LN. Positive scaling is approximate with epsilon, so the feedback uses the actual error, not a blanket “exactly invariant” answer.

**Transfer:** ask the learner to create a vector where the two methods agree, then break agreement by editing one feature. The artifact is the editable vector; do not prefill its solution. Use numerical feedback and the centering explanation. Tolerance for the shown exact invariance may be `1e-10` in Float64 fixtures; approximate scaling reports magnitude instead of applying that same predicate. Preserve the denominator and affine-state labels.

Closed hint: “Make the feature mean zero, then compare the two denominators.” Closed worked solution: `[-1,-1,1,1]` has mean zero and both denominators `sqrt(1+epsilon)`; changing its last entry to 2 gives a nonzero mean and different outputs. Check the learner's actual vector rather than demanding this one answer.

**Phase-two checks:** original/shifted contrast, learner-built zero mean, constant/nonconstant input, near-zero with larger epsilon, gain edit, visible default→edited input→synchronized result state, keyboard edit, reset, numeric table, desktop and narrow screen with readable axes.

## F3 and I2 — feature-write circuit and whole-block workbench

**Placement:** F3 within §4 and I2 within §5. They share compact exact matrix calculations but have distinct explanatory views. **F3 question:** how does an FFN's hidden response become a signed output update? **I2 question:** which exact junction changed after an edit, and why does placement matter?

### Direct FFN view

Use the explicit `W_up`4×3 and `W_down`3×4 in §4 / `fixtures.trace`. Inputs `[[1,2,−1,−2], [−1,0,1,0]]`, biases zero and ReLU produce updates `[[2,4,1.5,−1.5],[0,0,0,0]]`. A matrix-column view explains each up response; a matrix-row view explains each write direction. Signed stacked contribution bars should sum visibly to the update, with a raw table alternative. Hover alone must not be the only way to inspect a contribution.

Let the learner edit either token's actual features, any matrix entry, and choose ReLU/exact GELU/SiLU as optional comparisons with the selected formula shown. For SwiGLU, add a separately editable gate projection and show `u`, gate preactivation, `SiLU(g)` and product before the down projection. The primary gate fixture `u=2,g=1` gives 1.462117; changing `g` to −1 gives −.537883. The multiplier must not be labeled probability or forced to a 0–1 axis. Matrix shape changes use explicit row/column labels; if omitted in the first implementation, retain genuine entry editing rather than adding fake width controls.

**Live observation:** select a token/output coordinate, edit an input or FFN matrix entry and show its signed contribution and new output immediately. Keep the original result and exact difference available. The model uses current entities rather than a named-preset answer.

The learner can build a response that writes a positive change in one feature and a negative change in another. Editing actual matrix entries or vector coordinates updates the signed contributions and resulting sums immediately. This is a separate question from whether attention already contextualized the direct FFN input.

### Whole-block view

Default two-token input `[[1,2,5,8],[3,0,2,1]]`, epsilon 1e−5, identity LayerNorm affine maps, one head, `Q=K=H`, `V=H/4`, output projection identity; same FFN matrices and ReLU. All this is visibly labeled **exact declared fixture; not learned weights**. Use the actual normalized or unnormalized attention input H determined by placement. No random “activation intensity” matrix.

Show named stages: X; attention input; attention probabilities; attention update; first raw residual sum; contextual state Z; FFN input; FFN responses; FFN update; output Y. For post-norm, distinguish the raw sum from normalized Z; for pre-norm those two values coincide. Show the four-feature bars for a selected token at each stage; optional full two-row tables are exact. A paired pre/post view must use identical parameters and input.

**Baseline actual numbers:** pre attention weights approximately `[[.923430,.076570],[.076571,.923429]]`; pre output row 1 `[.772791,1.805723,5.092858,8.328628]`, row 2 `[4.165418,−.323706,2.110232,.924718]`. The FFN update at first token is zero for this fixture; show that meaningful null, not fabricated color variation. The second token's FFN update is `[.876661,0,0,0]`. Post output row 1 approximately `[-1.095440,−.730293,.365147,1.460586]` and row 2 `[.695474,−1.720968,.632590,.392905]`. Stage data are in `author-results.json.fixtures.trace`.

**Meaningful edits:** all input entries and FFN matrix entries; a declared branch multiplier λ; placement switch. Compute the comparison from the complete current inputs. λ multiplies both complete branch outputs after their sublayer calculations; it does not change what normalization means. Set λ=0 for exact pre identity / post normalize-twice control. Reset restores λ=1 and current comparison.

**Checked change:** edit token 2 first input3→5. Under pre-norm, token 1 final feature 1 decreases .77279056→.76206034 even though its own raw input is unchanged, because attention reads token 2; token 2 feature 1 increases 4.16541767→6.96371964. The direct FFN null above helps distinguish indirect context dependence from pointwise processing. All of the changed trace, including a newly active third FFN response at token 2, is retained in `edited_pre`.

**Unsolved construction:** ask the learner to create a case where attention changes the other token but the selected direct FFN output remains zero, or to make a hidden response contribute opposite signs to two features. Keep a reasoned hint/solution initially closed and check the actual numerical property; there can be multiple valid answers. A stepper supports inspection, but completing predetermined steps alone is not the investigation.

Closed hint: “Find a receiver whose three FFN preactivations remain nonpositive while another token changes its attention mixture.” Closed solution: in the saved pre-norm fixture, change token 2's first input from 3 to 5. Token 1's first contextual feature changes `.77279056→.76206034`, but all three of its FFN responses remain zero. For the signed-write version, activate the third response and use its existing `[0,0,.5,-.5]` write direction. Accept other numerically verified constructions.

**Phase-two checks:** exact saved fixtures, each wiring, zero branch, edited cross-position contrast and direct FFN null, arbitrary matrix edit, signed contribution sums, invalid-prediction handling, overflow labels, touch+keyboard table editing and mobile stage selection. No realtime animation of 6×8 matrices is required; prioritize legibility.

## F4 and I3 — real movement through two complete blocks

**Placement:** §6 at results and failure inspection. **Goal:** see how the two-layer classifier uses tagged records, inspect an observed failure and distinguish a changed trajectory from a permutation of equivalent input records.

### F4: observed fit results

Use `author-results.json.fits` in declared order. Separate train/validation cross-entropy curves (nats/example) from a paired test macro-F1 chart. Training-history x-axis is epoch 1–180. Selected epochs 83/146/112 for pre and 125/170/129 for post should be marked using the actual validation-F1 selection rule. Do not infer checkpoint selection from the lowest visible CE when the rule prioritizes F1. Test macro F1 in seed order is pre `.807037,.804974,.843545`, post `.866825,.852011,.781429`. Test correct counts pre 49/49/51 and post 52/51/47 out 60; expose them alongside decimal metrics.

No per-epoch test scores exist in the experiment and none may be invented. Show all six seeds/runs, no confidence interval from an inappropriate independence assumption, and no smoothed curve that hides actual selected points. The explanatory text identifies fixed data split and changed initialization. A compact comparison table remains usable without the chart.

### I3: fixed actual model, editable real entities

Predeclared source row 77 is anticlockwise arc/class 4; seed 101 is the display checkpoint in both placements. Use full original 45 x/y values and tags from `block-models.json`. Display a two-dimensional trajectory with equal coordinate scale, fixed source range 0–1, connected segments and explicit first/last markers. Time tags are normalized index, not seconds. A linked list/table shows each selected point's source index, editable x/y and editable tag. The learned model uses fixed `2*x−1`, concatenates tag, then Linear 3→24; do not apply a dataset-fitted scaler or transform tags as though they were raw coordinates.

The primary visualization overlays a baseline and one edit, with a probability panel for all 15 classes and a detailed selected class/logit view. Use the source metadata's correct class names or retain explicit class IDs if reliable display names are not ported. Keep class 4 attached to the original record only. Each model's original prediction is a failure: pre class 7, post class 3. Pre class 4 probability `.000211211`, post `.200043380`. Show these actual results and allow the learner to inspect them; do not substitute a correctly classified trajectory.

**Live model comparison:** choose placement/checkpoint, edit actual points/tags/padding and inspect class-4 probability, a selected logit and maximum absolute logit difference. Display the top class as an additional readout; values may change without an argmax change. Use a reconciled float32-compatible tolerance such as 1e−4 absolute logits, separate from whether the model predicts the true class.

**Checked contrasts/nulls, exact IDs in `block-models.json`:**

| Intervention | Pre-norm seed 101 | Post-norm seed 101 | Teaching role |
|---|---|---|---|
| `paired_permutation`: reverse point/time records together | max logit difference 4.77e−7 |1.43e−6 | Pooled permutation null |
| `reverse_coordinates_fixed_time` | max difference 4.885242; class 5; p5 .720945 |5.745950; class 5; p5 .594156 | Location/time association changes |
| `frame23_x_reflection` | max difference 3.752769; class 2 |5.955282; class 11 | Genuine coordinate edit, not a stored permutation trick |
| `masked_padding` with original times retained | max difference 5.96e−7 |9.54e−7 | Correct masks at both layers and valid-only pool |
| `unmasked_padding` | max difference 1.923383 |2.423569 | Pads treated as evidence change outputs |

The padding comparison appends five (.75,.75) points with tag 0; original 45 tags remain unchanged. Mask padded keys in both blocks and exclude padded query rows from pooling. Do not recompute original time spacing over the extended 50 rows. For causal modes elsewhere, position tags and masks impose different boundaries; this real classifier is bidirectional. Permit arbitrary 45-point edits and 2–60 records if insert/delete is offered; require at least one valid record. If all are marked padding, show a clear input-state message rather than calling softmax/pooling on an empty valid set or fabricating a distribution.

### Inspect updates without inventing neuron meanings

Below the path, select a token position and block 1/2. Show signed 24-feature bars for incoming carried state, normalized branch input, attention update, contextual state, FFN input, FFN update and output. Offer the 48 FFN hidden responses as a horizontally scrollable/table view, not an illegible global heatmap. The attention matrix is 2 heads×45×45 at default length; one selected receiver row and head is the main view, with exact weights/table. A full matrix is an optional larger-screen view with named axes. Post-norm traces include `attention_residual` separately from normalized `context`. The saved trace has every stage for baseline; new inputs must recompute all stages.

RMS/L2 summaries use explicit per-token feature reductions; learned affine normalizers may not have unit RMS or zero mean. Do not renormalize all stage intensities to make them appear equal. Show difference bars on a meaningful shared scale with exact numbers to reveal small versus large changes. Viewing which feature changed does not identify its semantic meaning; the manuscript explains that limit once in this investigation's home.

### Data, implementation and performance

`block-models.json` contains actual state dictionaries with PyTorch output-by-input dense storage. MHA input projection is packed Q/K/V in that order, with packed bias and an output projection; two heads each width 12. Implement exact GELU or a documented numerically adequate approximation verified against the retained actual fixture. LN epsilon 1e−5, affine gain and bias included at all sites, no dropout. Mean pooling occurs after final LayerNorm; classifier bias is present. All these details matter to the predictions.

The retained JSON includes evidence traces and both model weights (~1.64MB source JSON). Phase two should derive compact topic-owned inference assets from those exact weights, load the selected model on demand, and avoid serving full training histories/source data to every navigation visit. Recompute rather than ship matrices for arbitrary edits. Preserve enough baseline numeric evidence for checks and offer the offline full reproduction files separately. Typed arrays and at most 60 records keep small forward computation bounded; a worker is optional if actual measurements justify it. Never include the native fitting loop in the browser.

**Phase-two checks:** native/reference parity for both actual checkpoints and all above interventions; arbitrary point and time edit; malformed/empty/all-padded input; matrix packing/axes; selected-input live recomputation; reset; keyboard coordinate editor; point selection with readable start/end/time labels; desktop and mobile before/after figures. Test stale async responses if a worker is used and lazy-load error recovery under the code standard. These are deferred implementation checks, not completed claims.

## I4 and F5 — choose a gradient question that can change

**Placement:** §7 after the sum-loss counterexample and before the recorded depth chart. **Goal:** construct an informative and an uninformative output probe, and see which derivative the measurement represents.

I4 starts with `x=[1,2,5,8]`, epsilon 1e−5, no learned affine transform, and editable four-feature probe `p`. A probe is a vector of weights defining scalar `pᵀLN(x)`; display that scalar formula next to editable entries. Display the computed input gradient and whether its norm is numerically zero. Treat probe entries as genuine data to edit, with an all-ones probe as the visible default. Show the computed scalar, each gradient coordinate and contributing terms immediately; changing x, epsilon or p recomputes every linked quantity.

Use the analytic Jacobian in §7 to compute `Jᵀp` and central finite differences with default step 1e−5 as a separately labeled numerical cross-check. Display signed feature-gradient bars on a shared scale and the exact derivative formula. Record both result and max discrepancy; finite-difference approximation must not be presented as mathematical exactness. Allow an advanced perturbation-step edit to teach cancellation/truncation effects without changing the exact analytic result.

**Checked null:** p=`[1,1,1,1]` gives zero analytic gradient (floating residual below 3e−17). **Checked contrast:** p=`[1,−1,0,0]` gives `[.328633364,−.389491304,.012171588,.048686352]`; central-difference max error 8.37e−12. **Checked non-contraction example:** x=`[−.03,−.01,.01,.03]` has Jacobian largest singular value 44.2807443 and two directions with that gain, despite normalized output scale. Singular-value computation may be precomputed for this fixture unless the runtime supplies an exact bounded 4×4 method; never reuse that value for arbitrary edited x. To show arbitrary-input amplification without an SVD dependency, evaluate a learner-entered perturbation direction and its analytic directional gain instead.

**Constructive challenge:** create two distinct nonconstant input vectors for which the same all-ones probe gives no gradient; then replace the probe so it measures a nonzero contrast. Solutions initially closed. Also allow an optional zero-sum probe near constant x to see that input spread affects gain. Define zero/unchanged using consistent tolerance and show the actual gradient magnitude; a nearly radial direction with epsilon has a small nonzero derivative rather than exactly zero by fiat.

Closed hint: “The sum of centered features is zero for any input, not just the default.” Closed solution: `[1,2,5,8]` and `[6,7,10,13]` both give zero gradient for the all-ones probe. The contrast probe `[1,-1,0,0]` gives the nonzero gradient listed above at both inputs because the common shift leaves the normalization derivative unchanged. The implementation checks actual edited vectors and probe, so other valid examples succeed too.

F5 uses the saved `fixtures.gradient_depth` rows. Two charts: carried-state RMS versus state depth and input/state gradient L2 versus state depth for the unit-random-readout experiment. Select depth 1/4/12, placement(s), and show seed 53/97, d8,h2,f16, final shared affine-free LN and probe norm 1. Intermediate gradients are with respect to entire `(1,3,8)` carried states; do not relabel them average parameter gradients or singular values. Each selected depth has its own actual run, with matching inputs/probe; do not interpolate a depth 100 curve. Use linear y scales including zero for these positive values unless an explicit log view helps a separate null; never show log0. The all-ones null can appear as a labeled baseline/table, not a fictitious successful-training curve.

**Phase-two checks:** exact analytic/autograd/finite-difference fixture, all-ones null, changed probe and input, small-spread directional amplification, finite-difference step labeling, no overclaiming arbitrary SVD values, comparison/invalidated/reset states, readable mobile signed bars and complete exact-value table.

## F6 — three task boundaries and target shifting

**Placement:** §8. A complete-observation encoder panel shows all valid input positions communicating. A decoder-only panel shows the diagonal-inclusive causal triangle and aligned input/next-target rows. An encoder–decoder panel shows source states feeding the decoder cross-attention K/V, while Q comes from the decoder's updated state. Keep source padding exclusions visible independently of target causality. Positions are labeled as input indices and predicted next tokens; do not imply the current scored target may be fed at the same input position. A cached-step inset can show stored K/V per layer with absolute positions, but its purpose is conceptual setup; the dedicated cache lessons own detailed capacity/performance calculations.

The copy example uses 19 tokens/18 targets: BOS,8 fresh symbols,SEP,8 repeated symbols,EOS. Highlight the 8 fresh targets and 8 copy targets in different hatch patterns, with the two delimiter targets separately named. Mark input SEP at zero-based9 predicting the first repeated target at target index 9. The entropy denominator must be 18, with 8/18 ln10=1.023371 nats/target. This is a derived expected floor for fresh independent uniform data, not a measured training curve. Practice changes the source length/alphabet independently.

## F7 — real variants as equations and dependencies

**Placement:** §8 variant table. Draw the sequential and parallel block as a small pair where the FFN's input dependency is highlighted. The parallel version is `x+A(N(x))+F(N(x))`; do not accidentally omit the attention update from the sequential output. A second row shows branch-output norm, before/after branch norms and scaled postnorm, each with the exact position of its normalizer. Cite PaLM/Swin V2/Gemma 2/DeepNet next to the relevant version, without plotting unsourced rankings or implying “modern” has one mandatory wiring. These are static diagrams with optional zero-update overlays, not another compulsory lab.

## F8 — parameter and MAC accounting

**Placement:** §9. Inputs `B,L,d,f,H` may be editable if useful, but label the architecture assumptions: ordinary equal-total-width attention, two-map FFN, independent block weights, and dense full-pair arithmetic. Separate parameters, MACs, FLOPs and bytes rather than combining them on one axis. Plot exact linear-map and pairwise-attention MAC formulas from the manuscript. For B1,d512,f2048, saved values are L512:1,610,612,736 and 268,435,456 MACs; L4096:12,884,901,888 and 17,179,869,184. Show the parameter panel staying flat as L varies. Head count changes materialized matrix entries but does not by itself alter these leading total-width parameter/MAC terms when width stays fixed and divisibility holds.

A memory inset separates parameter bytes, materialized attention entries `BH L²`, FFN intermediate entries `BLf` and standard per-layer K/V-cache entries `2BLd`. Bytes require explicit dtype and do not include unlisted allocator/runtime/optimizer overhead. Checkpointing is a dependency/recompute diagram with retained boundaries, not a universal 50% saving bar. Parallelism is a partition of up columns/down rows with a visible sum of partial outputs; communication arrows must not be confused with locally free addition. No invented library latency or GPU speed charts.

## Later verification and handoff

Inspect an edited matrix with signed contributions, a real model failure with nontrivial before/after values, permutation/padding nulls and the degenerate-gradient contrast. Verify visible current outputs, live updates, reset, stale-result protection, keyboard operation and informative desktop/phone states. Independent practice retains its hints and explained solutions.

Preserve genuine learning choices while applying the code standard's lazy imports and bounded state. No additional fit campaign is necessary for these unchanged retained results. If implementation finds a consequential mathematical/content issue, fix its precise source and rebind affected evidence/checkpoints rather than reopening unrelated historical packets.

## Written implementation route and placement — 22 September 2026

Beside the two wiring diagrams, display which norm/attention/FFN state maps to the reference encoder layer. The epsilon-edit exercise belongs beside normalizer controls; output and derivative values update on the same state. A library comparison status is only measured after native execution, not synthesized as a green badge.

Use topic-owned responsive diagrams and local scrolling for code/matrices. Long filenames and links wrap within the reader at 320px. Show source/setup/download dependencies at the relevant explanation; deferred Python programs load only on request. Keep labels outside geometric marks where possible, fixed scale comparisons truthful, and current results visible during edits. No learner prediction field, submit button, answer lock or optional prediction gate is specified. Existing numerical/interaction checks still apply, and optional package/checkpoint routes carry their actual unexecuted status until phase two supplies evidence.
