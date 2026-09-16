# Cross-attention: implementation specifications

Phase: research/write only. Implement these representations on a later finish request. The complete learner copy is in lesson.md; cross-attention-study.py and calculated-inputs.json are author evidence, not browser modules. All exact quantities below must be recomputed from their displayed inputs. Trained evidence is restricted to saved records.

## Shared interaction behavior

For each investigation, the prediction begins unset and refers to the currently visible input version. Record the prediction before Reveal/Run enables; feedback compares the recorded answer with the actual result. Edits invalidate an old prediction/result and retain a visible “input changed; predict again” state. Reset restores the documented starting input and clears answers, reveals and history. Hints and solutions begin closed. Do not preset the correct multiple-choice option or label a control change as successful practice.

Use local labels, units and topic entities. Text tables and numeric inputs must offer the same reasoning path as spatial diagrams. Keyboard focus is visible; draggable items have move controls. Color indicates a role redundantly through labels/patterns. At narrow widths, stack query/memory/output views and keep locally scrollable matrices inside the figure; never shrink text to fit. Only the open investigation mounts optional interactive work. No external model/API call, autoplay training or full-corpus preload.

## 1. Rectangular read — lesson §1

Entities: two query rows, three paired key/value slots, score cells, normalization row, weighted value contributions and two output coordinates per query. Initial Q/K/V are calculated-inputs.json exact.query/key/value. Query labels are “reader 1/2”, never invented words or object regions.

Render a compact static first read inline immediately after the four-step mechanism. An expandable editor exposes Q/K/V coordinates, key/value pair reordering and a per-query allowed-slot toggle. Supported dimensions 2–4 queries,2–6 memory slots, key/value width 2; finite values−8..8 except an explicit stored masked-value contrast 100/−100. Score calculation uses stable softmax. Reject a wholly masked row with an explanation before computing. Do not quietly replace it by zeros or uniform weights.

Ask for output coordinates (tolerance 1e−3) or identify which output changes for the selected edit, then reveal all intermediate products. Default outputs (1.5,1.5),(1,2.5); masked first row (4/3,4/3). Third-value edit 100/−100 leaves first output unchanged and gives second (25.5,−23). Show that normalization operates across S. A paired permutation only reorders columns; a value-only permutation is a contrast. If queries are equal and masks match, outputs match. Every displayed arrow weight and table cell derives from the same current calculation.

## 2. Shape assembly — §2

Static rectangular matrices illustrate dₓ→dₖ, dₘ→dₖ and dₘ→dᵥ before the interactive exercise. Do not use square boxes that obscure T≠S. Compact shape builder lets the learner enter T,S,H,dₖ,dᵥ and reader width, each small positive integer≤16 except widths≤128. Predict weight and concatenated-output dimensions in unset fields. Reveal product compatibility and the output projection needed for a residual addition. Example 4 heads,T7,S11,dₖ12,dᵥ8 produces 4×7×11 weights and 7×32 concatenation; projection 32→48. Changing S affects columns/work, not output length; changing T affects output length. This is arithmetic, not a learned model.

## 3. Input order, operation and availability — §3

Use a token ribbon containing imageA, questionA, answerA, imageB, questionB, answerB. Below it show two distinct operation diagrams: projected visual tokens in a causal stream; separate text self-attention and rectangular visual reads. Block placement has a vertical depth ruler independent of the input ribbon. The reader should see that interleaved input can coexist with cross-attention.

Activity starts with an incomplete visual access mask. Learner assigns available images to each answer span and predicts whether a selected direct edge is permitted. Support “all preceding images” and “most recent preceding image” as explicitly different policies; select the policy before assessment. Show indirect paths through earlier text as dashed labeled paths, without implying a later direct image read. Perturb the placement of imageB to test future-image exclusion. Whole-clip visual encoding is a separate upstream edge: enabling it demonstrates why a final causal mask alone cannot establish online availability.

Adjacent static teacher-forcing ribbon uses answer “red bicycle” with input/target offset and separate attention/loss mask rows. An editable practice case “three birds” requires alignment. Score only the learner’s actual target placement and prohibited access, with explanatory feedback; masking a question’s loss must not erase its attention context. Never expose a target within its own predictor state.

## 4. Compression collision and learned slots — §4

Initial scalar sets (1,3) and (2,2) flow to a single uniform mean slot. Learner predicts whether the downstream state distinguishes the sets; reveal identical 2s. Allow entering two finite values per set and one or two output slots. One-slot mode is explicitly an averaging compressor; two-slot mode keeps ordered identity slots, not a claimed trained resampler. Ask the learner to construct distinct sets with equal mean and unequal max or spread; validate those properties numerically. The maximum distinguishes 3 and 2 in the starting example. A null case identical sets remains indistinguishable even before compression.

Static bridge shows learned latent queries as M anonymous slots reading S visual features, then later text queries reading those M slots. Slots are numbered learned positions; no quadrant labels or fabricated spatial interpretation. Mention repeated reads in the depth view and link the actual long-context owner. Do not make a fictitious accuracy-versus-M plot.

## 5. Gate and gradient routes — §4

Draw two rails, X and tanh (α)F, meeting at Y. Below, show loss residual, ∂L/∂α and ∂L/∂F for the same scalar squared-error fixture. Inputs X,F,target,α,rate bounded to−4..4 and rate 0..1. Learner predicts sign/zero status and next Y before one update. Only α changes in this scalar experiment; label F fixed for this step. Default X1,F2,target 3,α0,rate.1 gives gate derivative−4, α′.4, Y′≈1.759898 and L′≈.768927. Use computed values rather than copying these rounded annotations.

Contrasts: F0 gives no gate gradient; nonzeroα permits ∂L/∂F; rate 0 gives no change; an excessive step need not help. Feedback distinguishes zero output update from a zero gate gradient. A separate computation-graph inset contrasts a frozen parameter (no update marker) with a severed differentiation route (no input derivative). This inset is explanatory, not a mock implementation of autograd.

## 6. Named-model comparison — §5

Four compact routes share notation but use the actual different entities: Flamingo’s resampler and inserted gated blocks; BLIP-2’s Q-Former and projected query prefix; LLaVA’s original linear projection; Idefics 2’s pooled prefix. Mark frozen/trainable modules for the explicitly named stage. Do not reproduce paper figures wholesale. Link each primary source and date adjacent to its route.

BLIP-2 gets a local mask triptych because training objectives are its conceptual hurdle. Contrastive separates query/text, matching allows bidirectional query/text interaction, generation allows text→queries/earlier text but no query→text. Distinguish shared self-attention from visual cross-attention. No quiz on uncited parameter trivia; the meaningful task is following information and gradient routes.

## 7. Real image/query evidence — §6

Load only this packet’s small result subset on expansion. Data: digits-400.csv for actual pixels/labels; calculated-inputs.json for splits, nine fit summaries and selected head maps. Show attribution in the figure caption. Each selected map names source ID, question, model, seed, prediction, target, head and attention normalization. A4×4 overlay corresponds to the sixteen actual 2×2 input patches; all values remain accessible as a table. It is not an attribution proof. Do not label stored maps as resampler latents or imaginary object regions.

Controls select only stored source/question/model/seed/head combinations; unavailable combinations explain that no record was retained. Changing the question on a saved image displays the actual paired record, after an unset prediction of digit versus parity answer class. No arbitrary pixel editor pretends to run an unretained neural model. For arbitrary entity changes use the exact §1 editor; a future expanded neural inference feature requires actual weights and separate implementation verification.

Aggregate chart reports raw correct/count, separate digit/parity results, all three seeds and parameter counts. Its y-axis is accuracy 0–1. Epoch traces plot only the saved epochs 1,10,40,80,160; straight segments are visual connections, not measured intermediate steps. Explicitly identify fit/development curves and keep assessment out of selection.

The mismatch panel distinguishes original pairing, a one-image cyclic shift and a seeded shuffle of whole image groups. The sorted source subset makes the cyclic shift a weak test: expose exact label/parity match counts from perturbations. Learner records whether a chosen transformation substantially changes the relevant targets, then reveals the count and measured scores. This teaches validating a control itself. Assessment labels were inspected and are not a new untouched holdout. Show all seeds; no architecture winner badge or extrapolation to production VLMs.

## 8. Work and cache budget — §7

Two views: (a) attention-pair grids, with dense versus permitted causal counts separately selected; (b) stacked bytes for visual prefix K/V or per-cross-layer projected K/V. Inputs are the named P,J,S,S′,dₖᵥ,b,Bₛ, with reset example in manuscript. Scalar counts use safe JavaScript integers and reject overflow. Bytes display raw and binaryMiB. “Retain raw features” requires an explicit feature width/count; otherwise exclude them visibly. Text caches, weights and temporary tensors remain listed exclusions.

Predict one budget before reveal. Null: equal token/width/layer counts give equal projected K/V bytes. Contrast: reducing latent count changes memory but makes no accuracy promise. “Recompute K/V” removes their persistent storage and labels added projection work; it does not show fabricated runtime savings. A causal triangle counts allowed pairs, not guaranteed executed kernel FLOPs. Responsive fallback is an equally complete formula table.

## Finish acceptance and deferred work

Implement all necessary inline visual homes, not just the expanded labs. Verify keyboard/input/reveal/reset/invalidation and mobile text flow; inspect actual rendered figures. Compare exact editor cases against the adjacent author program and independently derived examples; verify saved tables/maps against the JSON and CSV source IDs. Execute the final displayed program and keep its artifact linkage intact. Perform independent correctness and learning-experience review and the appropriate integration/build/browser checks only under finish authorization. No such checks have been claimed in this content packet.
