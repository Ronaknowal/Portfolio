# Attention: visual and investigation specifications

## Live exploration contract — 21 September 2026

Open each investigation with its current inputs, intermediate mechanism and complete current output visible. Apply valid edits to meaningful entities immediately and update diagrams, tables, units and causal explanation together. No prediction entry, predicted-answer choices, commitment, prediction grading or answer-unlock feature is part of this packet, even optionally. Model predictions and mathematical masks/gates remain subject matter.

Use the topic-specific controls and checked fixtures below. Pair sliders or direct manipulation with labeled keyboard/numeric controls; keep presets as starting points, not the only editable values. A pinned baseline preserves its inputs, seed, units and outputs while the current case changes. Explain both a meaningful contrast and an unchanged/null result, then connect the observed effect to a practical design decision. Reset restores the stated fixture and current result. Invalid text has a local explanation and a clearly identified last valid result; never silently clamp or pair new inputs with old output.

Step/Back and bounded Run controls advance a real computation or reveal its chronological stages, not permission to view an answer. Show the current state and its result throughout. Keep exact small calculations live. For costly frozen inference, debounce or run bounded work with pending/current state labels and stale-result cancellation; inspect saved measurements without implying fresh training. Respect reduced motion, keep focus stable and avoid announcing every animation frame. Independent written practice and its hints/solutions stay separate.

Phase two must test default results without any action, meaningful edits, quick consecutive edits, valid extremes, null/invalid cases, reset, linked-view agreement, keyboard operation and readable phone layouts. The mathematical/reference checks already specified below remain; these live browser checks have not been performed in this content-only revision.

### Topic-specific live route
**Move a memory contribution.** Edit queries, keys/values, scorer parameters, padding validity, local-window placement and supported real source/prefix inputs.
**See the consequence.** Synchronize scores, normalized weights, weighted values, context and decoder output; keep masked rows and missing legal donors explicit. Retain saved observations as records.
**Decision connection.** Choose or diagnose a scorer/read window from the dependencies it creates, without reading attention weight as a complete causal explanation.


Content-first packet,13 September2026. These are implementation inputs, not rendered or independently reviewed components. Stable ID `attention-mechanism-bahdanau-luong`. Read `lesson.md` in full before building. Preserve the fresh-default/worked-example distinction and the numerical source of truth. No fixed lab count is intended.

## Shared implementation contract

Every quantitative object is one of: independently calculated constructed input (`analytic-results.json`), actual saved trained-model trace (`mechanics-results.json`), or measured fit result (`calculated-inputs.json`). Do not draw plausible substitute alignments, historical BLEU curves or implementation-speed benchmarks. Raw hidden coordinates have no assigned linguistic meaning. Label score, source weight, context coordinate and output probability separately.

All investigations below are live explorations. Store an immutable baseline when comparing an intervention and compute the current case from its complete source, model, task and control state. Valid edits update figures and exact readouts together. Inspect/reset/back navigate the actual computation; there is no predicted-answer state.

Use explicit numerical tolerances for live comparisons: constructed weights/context 1e−5, model display 1e−4 while retaining full internal precision, analytic equality/nulls 1e−10 and native float32 1e−6 unless a smaller signal is intentionally inspected. Display the computed difference and its mechanism rather than a correctness verdict.

Use native labeled inputs/buttons and keyboard selection. Provide coordinate labels, position IDs, full text tables and a concise generated description; color is redundant. Focus stays with the action, result announcements are polite, and reduced motion uses immediate state replacement. Never announce every cell during a step. At phone width, keep the source ruler horizontally scrollable with a visible scroll affordance, stack linked panels, maintain tabular numeric alignment and allow a row/column text inspector. No unreadable shrunken matrices.

Bounds: constructed memory 3–6 positions, dimensions 2, values/keys/query finite in[−3,3], learning rate 0–0.2; reject all-invalid masks. Model task inputs lowercase a–z, lengths3–8 in the reader to match the supplied lesson scope (source program supports1–32 but that is not an extrapolation guarantee). Output bound 16tokens including EOS, source request and EOS retained. Optional local ruler length 5, radius 1–3 in 0.5 increments, center 1–5 in 0.5 increments. No external requests or learner training in a lab.

Load only the active topic's chosen model on first use. Default seed 1 is fixed for continuity, not “best selected.” Store production model numbers in a compact topic-owned asset; do not import every six-run JSON array into the initial reader bundle. The full JSON remains a download. Bounded matrix operations can use an on-demand worker if measurement warrants; key/memory caches belong to model+source revision, decoder suffix to model+source+prefix. Dispose listeners/worker buffers on unmount. Do not compromise probability/mask accuracy to reduce payload. Browser/performance/phone checks remain phase two.

## A. Source-memory shelf — visible explanatory figure

**Placement:** section 1 directly after the token table. **Question:** what becomes available when an encoder retains every state?

Worked source `<past> l a c t a t e <eos>`, positions 0–8. One feature column per source position; lower decoder timeline has generated `l a c t a t e d <eos>`. Use a thick final-state→initial-state edge and independent many-source→read→current-context edges. The request and repeated characters retain actual position identity. Shape labels \(H:9\times64,\ c_t:64,\ s_t:64\) describe this model; drawn feature squares are a schematic sampling of coordinates, not their actual magnitudes.

Include a small optional bidirectional inset showing forward and backward dependencies; do not suggest our fitted forward source is bidirectional. Caption states values are learned contextual representations and can still discard information. Accessible prose describes both surviving paths. On phones stack shelf/read/decoder without crossing lines. Phase two verify each token/state correspondence, valid source length 9, and that no target token leaks into its own prediction. No invented prediction gate: this is an explanatory diagram.

## B. Memory-read and learning workbench — one investigation with distinct questions

**Placements:** section 2 shows reading mode; section 5 reopens the same calculation with gradient/update mode, carrying only learner-chosen inputs if they opted to retain them. The section 5 default is a fresh run with a current comparison; earlier reading completion does not auto-complete learning mode.

**Worked figure:** `fixtures.worked` query(1,0), keys[(1,0),(0,1),(−1,0)], values[(2,0),(0,2),(−1,1)]. Display scores[1,0,−1], weights and context beside the triangle. Position A/B/C colors remain stable. Signed vector-contribution segments sum to the context; weights sum 1. Numeric unit: dimensionless model coordinate; layout distances in the plane encode these two coordinates. Score bars and probability bars have distinct axes. No code-only output box.

**Fresh exploration default:** `fixtures.fresh` query(.3,−.4), same initial keys/values. Show its output immediately. The learner edits one actual key/value coordinate via grid or point movement (numeric alternative mandatory) and sets an optional valid mask. Recompute the weight on B, both context coordinates and second-class probability together; keep a labeled baseline so each change is visible. No comparison choice or answer is collected.

Source formulas: \(e=Kq,\alpha=\mathrm{softmax}(e_{\mathrm{valid}}),c=\alpha V,p=\mathrm{softmax}(c)\). Class target is coordinate 2, loss \(-\log p_2\). Keys and values may be independently edited in this pedagogical generic read; explain that the trained model derives both from source memory, so editing a source token is a different operation.

**Executed fresh fixtures:**

|Case|Outcome/check|
|---|---|
|Fresh|α≈(.488902658,.242781875,.268315467),c≈(.709489848,.753879217),L≈.671198778|
|B key changes(0,1)→(2,1)|αB≈.368772142,c≈(.591440883,.961215895),L≈.525254873|
|B value changes(0,2)→(1,2)|`fresh_value_contrast`: weights exactly unchanged, context/output/loss change; consume full calculated numbers|
|B value changes(0,2)→(1,3)|`fresh_value_edit`: both context coordinates increase by αB; output probabilities/loss unchanged because the logits share an additive shift|
|Mask B|αB=0,c≈(.936968919,.354343694),L≈1.026304408|
|Add masked extreme memory|`masked_extra_null`: existing weights/context/probabilities unchanged|
|Equal all values(.5,.5), then change query|Weights change but c=(.5,.5),L=log2 and score/query gradients zero|

The equal-value fixture is explicitly constructed to show a null; do not use it to claim normal model attention never matters. The value-shift null is a different phenomenon: the context changes while output softmax does not.

**Gradient activity:** expose signed gradients and an editable learning rate. Show old/new loss and the actual simultaneous parameter update immediately; an oversized step can increase loss despite a correct derivative. Step the operation for inspection without requiring an answer.

**Checks:** `attention-calculations.py` compares each analytic gradient with double-precision native autograd and centered finite differences. Future model/UI must match all listed contrasts/nulls, reject all-masked/NaN inputs, preserve one probability denominator and both coordinates, and make changes visible at phone width. No browser check done yet.

## C. Scorer algebra and decoder timelines — explanatory figures

Place the cancellation table beside section 2's proof and the two timelines beside section 3. The calculation uses `cancellation`: scalar keys[−.7,.1,1.3] and queries .2/1.1. Compare linear 2q+k with tanh(2q+k). Linear attention rows match; nonlinear rows differ. Adding 37 to every score is also a null. Show the common query term leaving the fraction; concatenation itself is not a nonlinearity.

Timeline objects: previous state and output embedding, memory read, recurrent update, output projection. Bahdanau reads with the prior state before the GRU; Luong updates the GRU before reading with the new state. Input feeding connects the previous combined vector to the next GRU input, initially zero. Do not draw the current predicted token as its own input. Label state/memory 64, embedding 24 and additive hidden width 32. Any schedule/scorer toggles here are exploratory explanations, not additional computed labs.

On phones, use two vertical timelines. Verify all dependency arrows, projections, and the optional input-width change 24→88. The initialized input-feeding check has 62,048 parameters and a second-output probability difference 0.00073138997 when its fed vector is replaced with zero. That checks an information path, not trained quality.

## D. Padding surgery — model investigation

Place it beside section 4's mask explanation. Use a source-token strip with two removable PAD storage slots, its selected attention row, signed context contributions and output probabilities.

**Fresh default:** real development `cash + past`, additive seed 1, target-step index 2 (third generated character), from `traces.additive.fresh`. Two extra storage slots start correctly masked. This differs from worked `lactate`. The task asks for padding attention mass or context change, not a promised word change. Learners change actual slot-validity flags, Show the current computed result and its contributing terms immediately. They can remove/reinsert a slot; inserted storage is invalid by default.

Hold weights, actual source tokens/length 6, encoder packing and all reference labels fixed. Padding values are zero because unpacking missing positions yields zero; their keys are projections of that memory. Broken masking deliberately admits these cells after encoding. Label this fault injection, not valid text.

At index 2, the correct weight on source `s` is 0.965103679 and context coordinate 0 is−0.254118904. With the fault, each PAD receives 0.186533587 (total 0.373067173), `s` receives 0.604137106, and context coordinate 0 becomes−0.158506690. Maximum output-probability difference at that step is 0.010851494. Both complete greedy words remain `cashed`. Use fixed scales that make stolen mass and signed context changes visible.

General seed 1 is an optional inspection, not this default contrast. Its early padding effect is tiny; at index 4 admitted PAD mass is 0.576445529, while its largest probability difference across steps is only 0.000040637. Do not force every architecture to show a dramatic first-step word change. The author replaced an unsuitable first-step assertion with these actual informative steps without repeating fits.

Nulls: correctly masked added slots preserve valid output distributions within float32 tolerance; a display-name change preserves numbers; reinstating the mask restores the original. Source-character edits require reencoding and are taught in G. Reset recomputes current outputs and restores the fresh masked source. A yet-uncomputed generated row cannot consume future reference tokens. Native packing and extra-batch-padding nulls were executed; phase two must retain both encoder-length and attention protections.

## E. Measured learning and outcome plots — explanatory exploration

Place after section 6's result table. Read `runs[*].checkpoints` at updates 0/100/400/800/1200 and `mechanics-results.prior_baseline` for the original fixed-context runs and rules. Plot exact-match fraction 0–1 with a zero baseline against actual updates. Retain each seed's dots/lines and a rule line 407/447. No invented band, best-seed selection, curve smoothing or training-time values. Group toggles change visibility only and are not computed experiments.

Provide the exact-count table and, optionally, a separate NLL plot with its own axis and explicit log-scale label if used. Never combine accuracy, CER and NLL on one unlabeled scale. State differing parameter counts, decoder orders and heads beside the shared data/protocol information. Length slices use `mechanics-results.runs[*].slices` with denominators 168 and 279.

Future desktop/phone inspection must confirm that low fixed-context fractions, late dips and the rule line remain legible. Values are verified; visual perceptibility has not yet been checked.

## F. Worked alignment and nonidentifiability — explanatory figures

Place beside section 7's table. Use the complete `traces.additive.worked` matrix: nine generated rows by nine source positions for `lactate+past`. Fixed 0–1 brightness, explicit request position 0, repeated-letter positions and source EOS 8. Label generated row tokens and step numbers; probabilities displayed 1.0000 are rounded approximations.

An optional general-seed 1 view compares two different fitted networks, not one parameter-toggle ablation. Their outputs match but their EOS attention differs. Selecting a row exposes its saved `query`, scores, keys, context and output distribution. The saved `state` is the post-update state; it is not always the query. Hidden coordinates have numeric indices, not invented semantic meanings.

The nonidentifiability figure uses `nonidentifiability`: three values and two attention distributions, both returning(.5,.5). A fresh optional comparison uses values[(2,1),(0,3),(1,2)] and weights(.3,.3,.4)/(.1,.1,.8), both returning(1,2). This is an explanatory counterexample, not another gated lab. Show different weights converging on the same context point; supply an exact table and text equivalent.

## G. Edit a source or generated prefix — fitted investigation

Place after section 7's interpretation discussion. **Fresh default:** `cash+past`, additive seed 1, generated-prefix mode, no prediction or answer selected. Use editable source characters/request above a growing output strip, then the alignment matrix and linked context/probability inspector. Worked `lactate` stays in F.

At a selected decoder step, display each character's current probability and its change from the original prefix. Edits to a later forced choice leave earlier distributions unchanged under the stated causal model. Recompute from the current source/prefix and explain exactly which dependency changed.

Both seed 1 models were executed on these fixtures:
- `cash+past`→`cashed`;
- source h→k, constructed `cask+past`→`casked`;
- request changed to participle→`cashing`;
- first emitted c forced to b after its distribution was computed→`bashed`;
- unchanged replay, display-name-only edit and correct masked padding leave outputs/numbers unchanged.

For additive scoring, first P(c) is 0.999311854; the source edit yields 0.999419014 and request edit0.999541596. Prefix forcing preserves that first distribution but changes second P(a) from 0.999695452 to 0.997742569. These small early probability changes are not the main visual contrast: show the complete changed suffix and selected context/alignment. An explicitly labeled delta panel can show small differences with scientific notation or up to eight decimal places. Equality result checks uses the shared 1e−6 native tolerance, not the rounding of a four-place display.

A source/request edit rebuilds memory, keys and initial state. A prefix edit reuses source memory and replays from before the affected output. Back cannot retain states from another branch. The first distribution is calculated before forcing its emitted character. Original argmax must remain inspectable.

Independent NumPy traces matched native tokens, weights, contexts and probabilities, with maximum probability error 2.299902445e−7. All six models' 447 development token predictions matched saved JSON from source-only inputs. Attention beam search is not a new lab here; the predecessor owns search, and section 8 explains shared-source/per-candidate-state requirements.

## H. Local-window investigation — optional branch

Place in section 8's local-attention subsection. Source ruler1–5, scores[0,.5,1,−.5,2], values[1,2,3,4,5]. The worked p3,D2 figure is explanatory. **Fresh default is p2.5,D1.5**, with no prediction selected.

Move the actual center, change radius or edit a source score in [−3,3]. Show entering/exiting positions, weight sum and context immediately. A labeled switch compares original multiplication with the renormalized alternative; the method is part of the active computation and cannot silently change between views.

Fresh valid positions 1–4 give sum 0.621783211 and context 1.612772446. Center 3.5 shifts validity to 2–5, sum 0.314289475 and context 1.125883607. Worked p3,D2 gives sum 0.390754832/context 1.254374991; renormalizing gives context 3.210133027. Editing the excluded fifth score in the fresh setting must leave the read unchanged. Repeating an unchanged center is another null.

All center/normalization contrasts were calculated; the excluded-score null is checked in author closure. Show the sum gauge with visible0/1 references and exact text. Unnormalized contexts need not remain in the values' convex hull. This activity edits a constructed center; it does not train the predicted-center network. Reject a zero radius or empty valid window even if normal controls cannot create one.

## I. Copy aggregation and location cues — inline application diagrams

Place at the respective section 8 formulas. The copy diagram uses `pointer`: Ada/met/Ada, two source positions joining one Ada output, vocabulary/copy mixing weights .4/.6 and final probabilities.46/.42/.12. Keep source-position and output-word columns distinct. Exercise6 supplies fresh red/blue/red with OOV red and mixing.2; its answer stays closed.

The location diagram sends the previous attention row through a local convolution, then adds each resulting feature inside its corresponding current score. Repeated acoustic-region boxes are conceptual, not measured audio. Explain that a contextual encoder plus a location cue does not guarantee monotonicity or streaming. An optional backward-encoder inset retains future-audio edges to show why a later narrow window cannot remove those dependencies.

## Deferred completion

Under an authorized finish request: implement topic-owned models, diagrams, labs and lazy assets; validate scalar/native/state correspondence and any changed displayed programs; obtain independent correctness/coverage and learning-experience review; inspect contrast and edited-input states, keyboard use, phone layout, reduced motion, errors, reload and route integration; measure active payload/latency; then publish/register. Preserve all pending inputs. A content checkpoint does not mean those implementation/review stages passed.


## Precise fixture and scope details retained for implementation

**Memory-read gradient.** Compute score gradient α_j gᵀ(v_j−c), query gradient Kᵀ∇_eL and a complete forward calculation after the simultaneous update. The fresh query gradient is (.745201886,−.232125401); rate .1 produces query (.225479811,−.376787460) and loss .612218160. Rate 0 preserves state/loss. Equal values give zero gradient despite nonzero rate. Show signed chain contributions and exact values live. This small scalar activity does not modify the saved fitted model.


## Code-to-mechanism implementation contract — 22 September 2026

Place the manuscript's new implementation route next to its stated concept section. Preserve the named axes, state and algorithm steps when implementing figures; the complete teaching programs are content inputs, not a hidden replacement for learner-visible code. Render long source only on demand with keyboard-scrollable code and wrapping download labels. Keep constructed comparison fixtures separate from recorded training experiments. No browser execution of Python, pretrained-model download or GPU experiment is required to operate a lab.

The ownership map in design.md identifies which operations are implemented here and which actual sources are reused. Both paths must be findable: the transparent mechanism and the ordinary package/tool route, followed by the changed-input practice. There is no guess-entry, prediction submission or answer-unlock state. Current outputs remain visible while the learner edits meaningful inputs; separate written practice can retain hints and solutions.
