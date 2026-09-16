# GQA/MQA — visual and investigation contracts

Research/write only, 13 September 2026. These are actionable phase-two contracts; no browser visuals or production code were implemented. The manuscript's figures belong beside their explanations. Use head wiring, vector mixtures, tensor accounting, conversion and an actual causal forecast rather than forcing every mechanism into the same text-output box.

## Evidence and common interaction contract

`mechanism-calculations.py` / `mechanism-fixtures.json` own hand geometry, routing, conversion counterexample, storage arithmetic and native operator/gradient evidence. `author-calculations.py` / `author-results.json` / `forecast-models.json` and the original two Libras files own the actual forecasting study. The two kinds of inputs must remain visibly distinguished. A hand scalar is not a trained activation; a byte count is not a measured GPU allocation or timing.

Every investigation begins with an **unset prediction** tied to a canonical snapshot of its input and selected model/configuration. Editing a consequential input invalidates the old answer and closes current feedback. Keep the prior prediction and its original input visible only as an explicitly old attempt. Commit a prediction, then reveal. An “Explore without grading” action may reveal directly but earns no practice completion. Reset restores declared inputs, clears the prediction and closes feedback. A display-only choice of already computed query tooltip does not invalidate an unchanged experiment; a new graded comparison does.

The prediction concerns the actual pending change: unchanged/changed or affected heads, estimated output, computed payload or whether a cache is valid. Permit ties and numerically unchanged custom cases. Free-text reasons are checked against an explained rubric, not a fake semantic text classifier. Feedback must report actual computed values and explain the local mechanism; never classify all edits as meaningful merely because a control moved.

All practice/why disclosures initially closed. No autoplay, animations claiming new data, forced simulation speed or unbounded timers. Use native controls and equivalent keyboard/text editing for every drag. Invalid/NaN/infinite input produces inline guidance and retains the last valid state; do not clip or invent a valid answer silently. Group choices enforce equal-size positive head counts where that operator is selected. All-masked rows show an explicit no-legal-key state rather than NaNs or fabricated probabilities.

Use labelled head identities, group IDs, position IDs and units. Probability scales are 0–1, signed vectors/logits use diverging scales, counts have bytes/MiB/GiB/GB labels. Shape axes are semantic labels, not colors alone. Tables/text summaries expose the same information as geometry. On narrow screens, select one head/group to inspect and stack details; allow bounded local table scrolling without widening the whole page. Honor reduced motion, visible keyboard focus and concise live reveal announcements.

## F1: one write, repeated reads

**Place:** §1 after the cache explanation. Three columns represent successive causal steps. At each layer, a new input creates Q and a native-count K/V record. New Q reads the legal cache and is discarded after its output; old K/V remain. Distinguish fresh projection arrows from repeated-read arrows. A separate prompt band shows many known queries processed together before the sequential generation region.

The diagram's time axis is token/sample position, not measured duration. It must not imply one shared cross-layer cache, token strings being cached instead of computed vectors, or a future input being projected before it has been observed or generated. Provide a written step list as the accessible counterpart. A small sketch is sufficient; no whole autoregressive chatbot is needed.

## F2 / I1: readers, shared memory and actual weighted sums

**Place:** wiring figure in §2, full investigation beside §3's hand computation. Keep two visuals distinct: a query-head → KV-head wiring matrix and a query-position → key-position attention distribution. The wiring is fixed for a configuration; the weights depend on data. Query labels Q0…Q3 connect to group labels KV0/KV1, then to three token-position records in that group.

**Baseline entity:** `manual_inputs` in the fixture JSON. Four queries at one legal final position, two KV heads, three memory positions, key/value width 2. Queries sqrt(2)×`[[1,0],[0,1],[-1,0],[0,-1]]`. Group 0 keys `[[1,0],[0,1],[1,1]]`, values `[[2,0],[0,4],[2,2]]`; group 1 keys `[[1,1],[-1,0],[0,-1]]`, values `[[1,3],[-1,2],[3,0]]`. Mapping `[0,0,1,1]`. These are constructed arithmetic inputs, visibly labeled.

**Genuine edits:** every Q/K/V coordinate editable in −4…4; select an arrow/vector and drag its two-dimensional endpoint, or edit its numeric pair. Change key positions within 0…16 and query position 0…16 with explicit legality. Add/remove memory records up to 8. Head-count comparison can use Hq=4/Hkv=4,2,1; changing representation must clearly specify how new fixture heads are initialized, not imply an unchanged trained model. A simple separate schematic switch can teach head counts without deriving outputs until full edited inputs exist.

**Prediction:** identify which heads' attention weights and mixed outputs change after the proposed edit. Select none/specific heads/all with no default. A numeric output estimate for one chosen head is optional. Reveal linked scaled-score bars, 0–1 weights, value-vector contributions and their sum; keep displayed decimals separate from full-precision calculations.

**Exact baseline outputs:** head 0 `[1.68927519,1.46608721]`, head 1 `[1.15536240,2.53391279]`, head 2 `[.15897503,1.60057363]`, head 3 `[1.84102497,.75954866]`. Weights are saved in full precision. All weight rows sum to 1 and all permitted hand inputs are computed, not selected from preset outputs.

**Checked controls and contrasting edits:**

1. Change group 0 position 0 value by `[1,-1]`. Weights unchanged, head 0 output delta `[.4223188,-.4223188]`, head 1 delta `[.1553624,-.1553624]`, heads 2/3 unchanged. This is value-path fan-out.
2. Change group 0 position 0 key by `[1,0]`. Head 0 scores change; head 1 is an exact null because its query's x component is 0; other-group heads unchanged. Recognize the null from actual dot products, not a prewritten “all readers change” rule.
3. Change only Q0 to sqrt(2)×`[1,1]`. Only head 0's local result changes. Shared stored memory is unchanged.
4. Relabel groups with K/V swapped and mapping `[1,1,0,0]`: every output unchanged. Change mapping alone to `[0,1,0,1]`: a different function, with actual saved wrong-mapping outputs. Do not call noncontiguous routing mathematically invalid; it is inconsistent with the original convention if weights are not transformed.
5. Repeat each KV group to form tied MHA heads: output identical within float64 tolerance 1e−12. Show compact storage and repeated transient representations separately. No duplication animation may claim a zero-byte allocation.
6. A new future key is excluded by the causal mask even with a large score. Every-key-illegal state is rejected explicitly. Change a value while keeping scores to isolate output versus attention changes.

**Feedback and access:** draw signed value vectors from a common origin; changing vector length is not probability. A contribution table lists `weight × each value coordinate` and the output sum. Use group line styles/shapes as well as color. Mobile shows one query's details at a time with the group linkage always visible. All coordinate/position edits have native numeric equivalents and focus-preserving commit/reset. Runtime bound: 8 keys, 8 query heads, 4-coordinate optional extension; no full trained model required here.

**Phase-two checks:** independent reference parity for edited arrays; head mapping; exact controls; all-masked/invalid inputs; prediction input binding; visible distinction between wiring and attention. Native source checks already passed but do not substitute for browser parity.

## F3 / I2: build a cache budget from its axes

**Place:** §4 beside the cache growth and count derivations. The main representation is a labelled per-token K/V record, repeated by heads, occupied tokens, layers and requests. Keep K-width and V-width separate. Small block counts illustrate multiplication without rendering one element per real token. A linked table gives exact integer bytes and selected binary/decimal units.

**Inputs:** default B=1, N=80, Hq=64, Hkv=8, L=32768, dk=128, dv=128, bytes=2. User can edit B=1…64, layers 1…256, Hq from 1…128, valid divisor Hkv, L=1…262144, dk/dv=1…256 and payload precision 2/4 bytes with optional 1/0.5-byte idealized quantized payload. If selecting quantized payload, display its intentionally excluded metadata; do not claim complete physical allocation. Exact integer/rational arithmetic should remain safe for bounds or use BigInt for raw payload counts. Do not allocate tensors with those real shapes.

**Prediction:** numeric payload or expected ratio after a proposed change, initially empty. Ask the learner to identify which axis changed. Reveal `B*N*L*Hkv*(dk+dv)*bytes` with actual factors, per-token payload and total. Preserve query-head count when comparing stored heads. Invalid Hq/Hkv divisibility gives a clear choice repair, not a rounded group size.

**Fixtures:** Hkv=64/8/1 with default other inputs at L=32768 gives 85,899,345,920 /10,737,418,240 /1,342,177,280 bytes =80/10/1.25 GiB. Per-token 2,621,440 /327,680 /40,960 bytes. Default GQA payload grows linearly with occupied L; optional log-y view explicitly labels GiB and actual tick values. The saved seven-length count table is formula-derived, not measured serving evidence. Do not draw an H100 capacity line suggesting the entire model fits if the cache lies under it.

**Controls:** halve L→half payload; keep all stored dimensions fixed while changing Hq to another compatible value→cache unchanged but score arithmetic/parameter terms can change; set Hkv=Hq→MHA; set Hkv=1→MQA; change dk without dv and verify independent contribution. Practice altered shape B=2, N=12, Hkv=3, L=1024, dk=64, dv=32, s=2 yields 14,155,776 bytes/13.5 MiB.

**Related panels:** a small projection-weight count schematic uses the complete general expression D(Hq+Hkv)(dk+dv), with biases excluded by visible label. A separate Amdahl panel takes assumed affected-time fraction f ∈[0,1] and speed factor s ∈[1,32], computes 1/((1-f)+f/s), and is permanently labelled “assumed timing model.” Default f=0.6, s=8 gives 2.105263. Never relabel its result as GPU latency or make a measured-looking quality/speed scatter from head counts.

**Distributed inset:** eight device boxes with Hq=32, Hkv=2, four local readers each and two logical KV groups each copied to four devices. Count 8 physical head copies versus 2 logical. The scenario is an explicitly defined replicated layout; toggling a hypothetical communicating layout shows “requires a separate cost model,” not free automatic division by 8. This is a static explanatory extension, not a full distributed simulator.

**Access/performance:** responsive equation/factor table; unit selector is display-only and does not invalidate a byte prediction, while dimension changes do. No texture proportional to multi-GiB storage, no long-array generation, at most 32 plot points. Phase two checks unit conversion, ratios, integer overflow, input validation and the separation of modeled/measured quantities.

## F4 / I3: compact cache assembly and offset-mask repair

**Place:** §4/§5 before the API comparison. Show separate rails for logical query/key positions, physical cache slots, native KV groups and the legal mask. A query of length 1 can sit at logical position 2 while its K/V tensor includes positions 0,1,2. Do not number the query at 0 merely because its local tensor has one row.

Use I1's editable projected Q/K/V arrays as the baseline data, with separate new-entry versus prefix views. The cache stores K/V at Hkv=2. The learner can edit positions, reassign cache slot order while moving K/V/IDs together, edit one K/V value or choose a different new-query vector. Allow 3…8 keys and one or two query rows. For the two-query mask example, provide fully specified bounded custom Q rows copied from learner inputs rather than fabricated “actual model” outputs.

**Task:** fill an initially blank/wrong logical query ID or edit the legal mask; predict whether compact cached attention equals the full reference, differs or has no legal result. Do not prepopulate the repair answer. After reveal show actual legal cells, selected query scores/weights, mixed outputs and maximum output difference. A metadata-only slot move with records intact must be a null. A changed key/value or logical relation may change outputs.

**Mask controls:** queries 3,4 against keys 0…4 produce rows 11110/11111. Upper-left non-square rule produces 10000/11000, clearly a different relation. One query at 2 against three keys gives correct I1 mixtures; upper-left mask permits only key 0, giving group 0 value `[2,0]` for readers 0/1 and group 1 value `[1,3]` for readers 2/3. Saving this incorrect output as a contrast must not present it as valid inference.

The API comparison explicitly names the inspected PyTorch 2.14 upper-left convention and current FlashAttention bottom-right convention; the UI's own reference uses explicit IDs. No browser SDPA/GPU dependency is necessary for this tiny calculation. A static RoPE inset shows native-head rotation then compact write, explains equivalent rotation of correctly labelled copies, and links prior positional encoding for full rotation geometry. Do not add a duplicate rotary lab merely to match another page.

**Cache lifetime lesson:** if the learner edits an observed input rather than a projected cache entry, cached representations must be recomputed for the corresponding prefix/function. The real study below provides that computation; this projected-array lab must not pretend that changing a hidden-state source leaves old K/V valid. Physical slot rearrangement alone is different.

**Access/runtime:** use a focusable mask grid plus a native table representation and clear legal/illegal text. Mobile shows one selected query rail; all keys remain reachable via local scrolling. Restrict vectors to −4…4 and IDs 0…256; reject empty legal sets. Reference float64 tolerance 1e−10. No unbounded append loop; repeated teaching concatenation is explicitly not production allocation advice.

## F5 / I4: what averaging preserves and what it does not

**Place:** §6's conversion explanation. Small two-head panel separates original parameters/keys/values, their shared mean, and the resulting weights/outputs. Arrows show that Q0/Q1 and output heads remain distinct while K/V are merged. An optional original-weight block diagram shows PyTorch `[Hq*dk,D] → [Hkv,R,dk,D] → mean over R`; label bias averaging too.

**Default editable entity:** two scalar queries 1,2; original keys head 0 `[2,0]`, head 1 `[0,2]`; values head 0 `[1,3]`, head 1 `[5,-1]`. All entries editable −6…6. Scores use dk=1. No pretrained attribution. Original outputs 1.238405844/−.892082740; converted keys [1,1], values [3,1], outputs 2/2. Show both original distributions and the two new uniform distributions; do not imply the two original outputs are averaged.

**Prediction:** mean conversion leaves both outputs unchanged / changes one or both / custom tied case, with optional numeric guesses. Reveal actual per-head differences, and separately show the squared parameter/key/value distance minimized by a mean. A learner-selected candidate shared value/key can be compared with the arithmetic mean in coordinate distance, but lower coordinate distance must not be labelled lower forecast loss without a defined target and computation.

**Controls:** set original K/V heads equal→exact preservation despite distinct queries; restore different heads→actual nonzero differences. Swap the two original heads together with their Q/output interpretation→function-preserving relabeling before grouping, while arbitrary regrouping of larger fixtures is a separate operation. Values changing independently from keys must recompute exactly. No invented stochastic convergence animation or “90% recovery” progress bar.

**Training bridge:** a static/read-only plot uses the real `author-results.json` validation-MSE histories for the three declared continuation branches, including their zero-update states. X-axis extra updates 0…45; y-axis transformed-coordinate validation MSE, explicitly not test score or large-model benchmark. Selected checkpoints 41/45/42 labelled using the recorded criterion. Curves may have different scales; offer log axis with actual labelled positive values or separated panels. Do not fabricate an interpolation through unrecorded additional updates or claim GQA converged at 45.

**Gradient inset:** in deeper §7, two readers' contributions 0.2×3 and 0.7×(−1) sum to −0.1 at one shared scalar value. Learner may edit these four numbers if a small computation is useful; the primary explanatory goal is sum-of-uses rather than an automatic group average. The complete native derivative check is retained in fixtures. No backward-rendered whole network is required.

**Access/runtime:** scalar input tables and labelled old/new distributions are sufficient; clearly distinguish “copy”, “mean”, and “continue training.” No actual browser optimization. Loading the compact 45-step history on opening this panel is fine; never import all historical model files. Recompute ≤4 small heads×8 keys for custom algebra. Phase two checks exact nonlinearity counterexample, tied null, grouping axis and visible separation of parameter distance, prediction error and validation selection.

## F6 / I5: actual causal forecast with a shared cache

**Place:** §6 after the data/protocol/outcome tables. The observed prefix and unobserved future are separated by a visible time boundary on a real trajectory and a linked x/y-by-frame plot. The default is source row 77, first 32 observed points, all retained without model-input downsampling. The true next point [.5938100219,.25] remains hidden until evaluation reveal, and never feeds a generated rollout.

### Actual model and editable input

Select among the three saved **post-selection** models in `forecast-models.json`: Hkv=4/2/1, Hq=4, width 24, headwidth 6, one pre-norm block, biased maps, FFN 48 with GELU, final LayerNorm epsilon 1e−5, no dropout and per-row two-coordinate output. Run the complete CausalForecaster with fixed `2*x-1` transform and full adjacent-pair Q/K RoPE base 10000. Keys/values are cached compactly; Q projection has 24 outputs, K/V have 6×Hkv outputs. These are separate named linear maps, not the packed QKV convention of preceding packets. PyTorch weights are output-by-input.

Genuine edits: every observed point's x/y in 0…1, prefix boundary within 2…40 source points, and generated rollout length 1…5. Edit by dragging a selected real point or equivalent numeric controls; maintain source-versus-modified labels. Default signed x-reflection at frame 23 is a control, not the only available edit. Display logical positions separately from storage slot labels. An optional common-position-shift control 0…128 checks the fixed RoPE symmetry; do not reset one angle independently and call it a global shift.

For each selected query head, show its actual KV group, actual attention weights over legal observed positions and actual head output. A time strip can highlight which historical points that head weights; it is not a causal attribution of the full forecast. New forecasts remain two coordinates, never recast as class probabilities. An output outside 0…1 must be shown honestly with an expanded/indicated view, not clipped into a supposedly valid prediction.

### Prediction/reveal

At baseline show observed input, source identity, branch/checkpoint and task units. For a pending point edit or cache processing change, prediction begins unset: will the same causal computation be preserved, or will forecasts change? Optional next-coordinate numeric guess is genuinely empty. Include which outputs can change: before the edited time or at/after it. Commit, then reveal vector forecast, actual difference, linked head computation and the selected cache shape/payload.

Changing model, prefix boundary, any coordinate, position rule, mask or rollout length invalidates the previous prediction and any cache derived from that input. A display-only head selector can inspect already-revealed computation without earning a new practice score. Reset returns the exact source 32-point prefix, chosen initial model, no generated state and no committed answer. Do not compare a fresh model with stale cached vectors from the old model.

### Checked outputs/contrasts

- Default next-coordinate forecasts: MHA [.6009761095,.2584558427], GQA [.5974121094,.2590175867], MQA [.6071050167,.2532032132]. True next source point [.5938100219,.25]. Baseline original-prefix outputs, all learned weights and attention traces are retained.
- Compact K and V each shape `[1,Hkv,32,6]`; actual float32 payload 6144/3072/1536 bytes, excluding metadata. Show occupied versus reserved counts if the browser preallocates; the calculation's logical payload remains labelled.
- Full pass versus one-point-at-a-time compact cache: maximum transformed-coordinate difference 2.68221e−7/2.08616e−7/2.38419e−7 in MHA/GQA/MQA. This checks the entire small network, not only concatenating precomputed K/V. Browser parity target ≤2e−5 original-coordinate under declared arithmetic; inspect larger differences rather than inflating tolerance.
- Uniform position shift +100: max transformed-coordinate difference 2.38419e−7/2.38419e−7/2.08616e−7. All relevant IDs move, with unchanged frequencies and legal order.
- Reflect frame 23 x→1−x, recomputing the edited prefix: final forecasts MHA [.6003332734,.2567420900], GQA [.5974684358,.2589251399], MQA [.6095705628,.2538693547]. Earlier-than-frame 23 outputs are exactly unchanged in the independent author check for all three models. Max changes over the whole prefix are transformed-coordinate .8338091373/.7526575327/.6527418494; do not substitute those large maxima for the much smaller final-query changes.
- Five-point GQA generated path after original prefix: [.5974121094,.2590175867], [.5900123119,.2612144351], [.5834799409,.2659775615], [.5776410699,.2725118995], [.5717208385,.2794670463]. Later predictions feed the preceding generated point. Other two actual rollouts are retained. Future ground truth may be displayed only after reveal for reference; it cannot enter generation.
- Unchanged input/repeated reveal deterministic. A learner edit producing a small/null final change is accepted and explained rather than exaggerated. Changed or generated samples are never added to the held-out accuracy/RMSE table.

### Observation table and data limits

Keep the manuscript's real before/after table with MHA/GQA/MQA and both simple baselines. Do not hide that MHA wins this declared run or that GQA's selected update 45 is the budget boundary. The one-step aggregate RMSE uses all 44 slots on each of 60 held-out trajectories; the five-point rollout is a separate example. No universal language-model quality ranking or timing axis is appropriate.

Data/protocol limitations belong beside the table: exact duplicate handling, classwise row split without reliable performer/session IDs, fixed coordinate transform, one parent seed and matching additional updates with fresh optimizer. Brief local labels are sufficient elsewhere. The model selector loads only saved selected branches; a before-conversion model is not presently exported as an interactive checkpoint. Its actual zero-step aggregate metrics are read-only. Do not fabricate its weights or label a selected branch as the untrained conversion.

### Access, bounded browser work and later verification

Offer a frame table, labelled x/y coordinate fields and next/previous selection with keyboard nudges; a drag must have equivalent controls. On mobile stack trajectory, time plot and head details. Mark observation boundary with text and line style, observed points with solid lines and generated points with a different pattern. Live announcements summarize the committed forecast, not every pointer motion. Reduced motion gives discrete updates, preserving temporal order arrows.

Maximum runtime input 45 points plus ≤5 generated steps, width 24, Hq=4, Hkv≤4, one block; compute on explicit reveal/commit, throttle continuous drag preview if any. Use compact per-model weights loaded lazily, not the full author JSON/training histories eagerly. Release or replace stale model/cache/trace arrays on selection changes. A worker is optional if profiling identifies blocking; there is no reason to download a large ML runtime for these small operations. Do not score all 330 trajectories, train models or run historical fit campaigns in the browser.

Phase two implements native model parity, full/cached equivalence, causal no-future effects, cache invalidation after past edits/model switches, rollout input provenance, units and unbounded-output display. Then verify keyboard, screen-reader summaries, narrow viewport, overflow, reduced motion and lazy-load failure recovery. Source calculations and this author contract do not constitute rendered acceptance.

## F7: comparisons and application connections

The §7 published T5 table uses exact source Table 1 rows 47.2/46.6/47.1 and 1.51/.24/.28 with its measured TPUv4 per-sample units and task-average label. If graphed, use three categorical points only; do not add an invented head-count quality curve, error bar or “eight is optimal” boundary. Its source protocol is visibly separate from our real forecast table and formula-derived cache plot.

The methods-comparison table distinguishes head sharing, local legality, tiled exact computation, paged allocation and precision. An encoder-decoder schematic labels a fixed encoder K/V cache and a growing decoder K/V cache with different lengths. A multimodal prefix sketch labels patch/audio positions without claiming all architectures use one mechanism. An MoE block sketch places experts in the FFN and cache in attention for the specifically described architecture. These brief diagrams solve real boundary confusions; no extra mandatory lab is needed.

## Phase-two continuation and retention

Implement the concept-specific figures and investigations with these inputs/controls, preserving meaningful edits and readable first-pass progression. Any simplification requires preserving the learning hurdle and an adequate route to its evidence. Independent review must cover head mapping, shapes/masks/rotary convention, compact physical/logical accounting, parameter averaging, actual model parity and practice correctness, followed by formal browser/accessibility/integration work. No such implementation was performed in this phase.

Retain all eleven packet files, including original data/metadata, programs, model and numeric evidence; they are pending inputs rather than scratch. Offer full reproduction files as optional downloads and derive compact semantic runtime assets later. If implementation finds a material content issue, fix the affected packet and rebind its content checkpoint through the root-owned ledger. Do not rerun unchanged training campaigns solely for editorial formatting.
