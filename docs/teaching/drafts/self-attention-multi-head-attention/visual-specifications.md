# Self-attention — visual and investigation contracts

## Live exploration contract — 21 September 2026

Open each investigation with its current inputs, intermediate mechanism and complete current output visible. Apply valid edits to meaningful entities immediately and update diagrams, tables, units and causal explanation together. No prediction entry, predicted-answer choices, commitment, prediction grading or answer-unlock feature is part of this packet, even optionally. Model predictions and mathematical masks/gates remain subject matter.

Use the topic-specific controls and checked fixtures below. Pair sliders or direct manipulation with labeled keyboard/numeric controls; keep presets as starting points, not the only editable values. A pinned baseline preserves its inputs, seed, units and outputs while the current case changes. Explain both a meaningful contrast and an unchanged/null result, then connect the observed effect to a practical design decision. Reset restores the stated fixture and current result. Invalid text has a local explanation and a clearly identified last valid result; never silently clamp or pair new inputs with old output.

Step/Back and bounded Run controls advance a real computation or reveal its chronological stages, not permission to view an answer. Show the current state and its result throughout. Keep exact small calculations live. For costly frozen inference, debounce or run bounded work with pending/current state labels and stale-result cancellation; inspect saved measurements without implying fresh training. Respect reduced motion, keep focus stable and avoid announcing every animation frame. Independent written practice and its hints/solutions stay separate.

Phase two must test default results without any action, meaningful edits, quick consecutive edits, valid extremes, null/invalid cases, reset, linked-view agreement, keyboard operation and readable phone layouts. The mathematical/reference checks already specified below remain; these live browser checks have not been performed in this content-only revision.

### Topic-specific live route
**Build an attention read.** Drag/edit vectors, manipulate legal communication edges, change head projections/temperature and supported trajectory points.
**See the consequence.** Show scores, weights, mixture point, per-head outputs, mask legality and sensitivity immediately. A direct vector edit reaches every linked output and accessible table.
**Decision connection.** Distinguish compatibility, value content and legal access; choose head/mask/scale settings by those separate effects.


These specifications are complete content-first inputs. Implement them in phase two using topic-owned components and the existing lazy lesson/lab conventions. No visualization has been rendered or browser-tested in this phase. The manuscript's figure markers give placements; keep each visual beside the reasoning it supports. Do not move all demonstrations into a final generic lab panel.

## Shared interaction, evidence and accessibility rules

Maintain a clear distinction among **constructed arithmetic**, **derived cost models**, **observed fitted-model results** and **hypothetical edits**. Labels appear at the point of interpretation. A fit's heatmap/probabilities must come from `attention-model.json`, and an edited input must run the retained small model again. Do not interpolate saved outputs, invent an embedding-to-word interpretation or imply this model recognizes complete sign language.

Use one consistent role legend within this lesson: query/receiver, key/matching, value/contribution. Head identity has an additional label or line pattern; it must not reuse role colors ambiguously. Exact numbers, labels and focus states convey the same information as color. Every probability plot uses a declared fixed scale rather than rescaling a nearly unchanged row to look dramatic.

Every investigation opens with useful current results. For cheap arithmetic, valid input edits immediately update the figure, tables and explanation from one state. A costly frozen-model run may retain a clearly labeled previous-input result while a bounded calculation is pending; cancel stale work. A baseline remains immutable. No prediction field, answer disclosure or correctness comparison is retained.

Controls have associated labels, keyboard operation, visible focus, useful bounds and semantic buttons. Numeric input is always an alternative to drag. No autoplay or time-limited challenge; optional transition motion is short and respects reduced-motion preferences. A screen-reader summary states receiver, allowed donor count, computed weights/output and the change that was measured in the current live view. It must not expose an visible answer prematurely. Tables/CSV-style text views provide exact accessible alternatives to matrix heatmaps.

At narrow widths, stack input → mechanism → result in that order. Keep table headers and row/column direction visible; permit internal horizontal scrolling for a small matrix if required, never page-wide overflow. Large 45×45 maps show one selected row with an optional overview, not tiny illegible cell labels. Touch and keyboard users can select a frame without hitting a 2-pixel point. Explain units: fixture features are abstract; trajectory coordinates are normalized and dimensionless; indices are sampled point numbers, not seconds; entropy is nats; memory is bytes/MiB/GiB with explicit conversion.

Computation is small and bounded. Recompute elementary fixtures synchronously on valid edits; coalesce pointer events to animation frames when necessary. For the real model, cap the browser workspace at 60 points, two heads, width 24 and 15 classes; the original has 45 points. Recompute after valid input edits using bounded debouncing for the frozen model, or on an explicitly requested expensive diagnostic; never recompute on scroll. Lazily import its small weight asset/component when the investigation is near or opened. Full training, the 360-row source file, torch, sklearn, and native author programs do not belong in the initial reader bundle. Keep only the selected row/head details in active rendering state. Reuse shared controls/typography without forcing every diagram into the same visual composition.

## SA1 — who asks and who contributes

**Placement:** §1, after the role table. Static diagram with a receiver-selection control only if it improves reading; it is an explanation, not a scored lab.

Draw three source-position cards A/B/C across a baseline, each with separate q/k/v outputs. A larger receiver A sits below. Its query goes to three small key-comparison nodes. Their score wires go to a single row-normalization node. Separate value wires carry A/B/C values to the weighted-sum node and finally to A's new representation. Use arrowheads and explicit verbs “compare,” “normalize,” “mix.” Never draw query/key vectors flowing into the final sum as if they were values.

Initially show only the A receiver, with B/C receivers implied by repeated faint lanes labeled “same operation for each receiver.” An optional B selector reveals the analogous B lane without changing model parameters. Label Q/K/V as independently learned maps from the same X; the equality in SA2 is identified as a constructed example, not the self-attention definition.

Text alternative: “A's query is compared with keys A/B/C. The normalized three scores weight values A/B/C. Their sum becomes A's output.” On mobile the query matching lane sits above the value-contribution lane, with donor labels repeated so arrows do not cross an unreadable page.

## SA2 — match vectors, then move the mixed point

**Placement:** §2 immediately after the complete fixture and convex-hull explanation. Combine a static worked trace with a separate changed-input investigation.

The worked trace uses `author-results.json.fixtures`: Q=K rows `[1,0]`, `[0,1]`, `[1,1]`; V rows `[2,0]`, `[0,2]`, `[1,1]`. Show the exact score grid, selected row's exponential/denominator calculation and three labeled value points. All three V lie on x+y=2: draw their **line-segment hull**, not a nonexistent triangle. Plot the output at its actual coordinates, with contribution components available in the exact-number table. In another explicitly labeled value edit, moving V_C to `[1,2]` opens a genuine triangle; weights do not change solely from that V edit.

**Changed workspace inputs:** start with the visible query `[.5,1.5]`, the same three keys and values, d_k=2, temperature 1 and no mask. The learner can edit either query coordinate, each key coordinate and each value coordinate in [-3,3], or drag q/k in their comparison plane and v in a separate value plane. Distinct planes make it impossible to confuse “closer in key space” with “where the output lies in value space.” Draw q/k dot-product alignment, not Euclidean distance masquerading as a dot product. Expose raw and scaled score numbers.

**Live causal comparison:** edit a key and show which donors gain/lose weight; edit a value and show whether weights, mixture output or both change. Use the edited entity's name and actual contribution in the explanation. Keep both effects visible without asking the learner to choose an answer.

**Mechanism:** stable row softmax `(q·k_j / sqrt(2) + common_offset) / temperature`, then weighted sum of V. Temperature positive, bounded [.25,4] when visible; its deeper derivative explanation belongs SA7. A value-only edit does not touch scores. A key-only edit changes the matched donor's score but normalizes all donor weights. Query edit can change several scores. No learned semantics or automatic optimization is implied.

**Checked examples:** `additional-fixtures.json.message_lab` records the initial weights approximately `[.169022,.342796,.488182]` and output `[.826226,1.173774]`. Edit V_C.y from 1 to 2: weights stay exactly unchanged and output becomes `[.826226,1.661956]`. Reset; edit K_A.y from 0 to 1: weights become `[.370070,.259859,.370070]` and output `[1.110211,.889789]`. Add 1000 to every finite score: same distribution up to floating-point rounding; stable normalization must remain finite. These are a meaningful contrast and null, not only screenshots of a slider at two values.

**Feedback/perceptibility:** in the current live view, show before/after weight bars on a common 0–1 scale, signed numeric deltas, and the actual output point. The value edit moves the point vertically by about .488; the key edit transfers about .201 weight to A. Explain the dependency that changed. A numerical estimate can be checked within .005; comparisons use underlying values before rounding. Mobile stacks key plane, donor table, value plane; every point has keyboard-selectable coordinate fields.

**Phase-two verification:** verify row sum/nonnegativity, exact labels, all three checked cases, stable translation, and point/hull agreement for collinear/noncollinear values. Test keyboard edits, live result updates, reset and text equivalence. No browser fitting.

## SA3 — build a legal communication graph

**Placement:** §3, synchronized with causal attention and shifted-target discussion. Include a static three-position trace and an editable four-position investigation, followed by the rectangular decode view.

The static trace reproduces `fixtures.causal_weights` and outputs. Display query rows vertically and donor columns horizontally. A blocked cell is hatched/marked with a lock glyph and text; do not encode blocked as merely “low weight.” Show that remaining entries renormalize. The input/target strip above the matrix aligns each query with the next-token target and makes the diagonal's legality visible.

**Workspace:** four explicit input positions A/B/C/D, query vectors and keys `[[1,0],[0,1],[1,1],[.5,-1]]`, values `[[1,0],[0,2],[3,1],[-2,3]]`. Let the learner select receiver index, edit allowed edges in that row or construct a full mask, and edit a donor value. Initial task selects zero-based receiver 1 and asks the learner to mark legal donors for a next-token causal model. Start with blank/visible mask choices in the challenge; offer the already-explained causal pattern only as an optional closed hint. Keep every legal-set validation local. Do not normalize an empty set; show “choose at least one legal donor” with no probability result.

**Live observation:** after setting a mask, show whether changing future donor D.x from -2 to 4 can alter receiver B's output. Show the current computed result and its contributing terms immediately. The learner can use arbitrary edits rather than a preset-only replay.

**Checked contrasts/nulls:** under `[True,True,False,False]`, B output is `[.330238,1.339523]` both before and after the D edit. With all donors allowed it changes from `[1.098915,1.362974]` to `[1.632030,1.362974]`. These are retained in `additional-fixtures.json.mask_lab`. Show the .533115 x-change numerically and geometrically. If the learner chooses another mask, compute actual feedback from that input rather than reusing these results.

**Rectangular extension:** an editable query absolute position and editable key positions are the primary entities, not an opaque “causal” toggle. Default changed challenge has query 2, keys0/1/2 and the original three-token numeric fixture. Let the learner edit the mask cells while the legal dependency routes and resulting weights remain visible. Correct all-allowed output is `[1,1]`; the mistaken top-left row gives `[2,0]`, checked in `author-results.json.fixtures`. Add a document-ID field as a deeper optional task: legal iff same document and key position<=query position. Use practice 2's exact sets for that check. Do not silently equate array index with absolute position.

**Accessibility/layout:** the graph arrows are redundant with a keyboard-operated grid of named checkboxes, e.g. “Receiver B may read donor D.” Input and target are distinct labeled rows. On mobile, show one receiver and its legal-donor checklist at a time, with a collapsed full overview. A masking failure is a reasoned connection error, never a red/green-only image.

**Phase-two verification:** reproduce computed null/contrast, verify renormalization and empty-row handling, check document boundary cases and rectangular alignment; confirm that valid input changes recompute every dependent result and explanation; retained baselines keep their original inputs. Native checks already support arithmetic; browser behavior remains unverified.

## SA4 — split, mix, merge

**Placement:** §4 after the exact one-head/two-head example and before the general shape table. Primarily a diagram plus a small optional construction workspace, not another generic slider plot.

Draw features as labeled channels attached to positions. Q/K/V maps split channels into head lanes; each lane has its own receiver–donor grid and value-mixture result. Concatenate lanes at the same position, then apply W_O. A reshape must visibly preserve position identity; do not animate contiguous memory as if that were the conceptual operation.

Worked two-position fixture: input rows `[1,0]`,`[0,1]`, identity maps, one head of width 2 gives receiver 0 output `[.66976155,.33023845]`. Two heads of width 1 give `[.73105858,.5]`. Numbers are in the main and additional fixtures. Label which softmax normalized which scores, and state that this is a fixed-parameter arithmetic comparison, not a trained model ranking.

If implemented as a lab, expose the actual two input vectors and all four 2×2 maps with a one/two-head partition selector. Input edits must recompute Q/K/V, split and merge; do not simply cut up the final attention map. The current computed result is visible.: “Will the outputs remain equal after changing the head partition?” The fixed identity contrast is nonzero. For a null, set W_V=0: both outputs are zero for any valid partition when there are no biases; this follows directly and should be checked numerically. Do not claim all same-parameter models are equivalent because this special null holds. W_O edits may move outputs outside the original V hull, which should be shown as the appropriate projection-space change.

The shape view lets the learner enter B,L,d and a valid divisor H (small diagram bounds B<=4,L<=8,d<=24), with clear validation for nondivisibility. Show formula counts and a count of actual displayed channels; use labeled brackets rather than hundreds of boxes. The computed counts update immediately when dimensions change. Practice 3 checks the 576 weights and 900-byte tensor. Treat this view as derived arithmetic, not a hardware profiler.

Phase two should verify reshape order with indexed channels, output/parameter invariance under merely permuting head labels with corresponding output blocks, the two-head contrast, zero-value null and small-screen readability. A static version is acceptable if the editable version adds no new understanding beyond SA2; preserve the exact worked comparison either way.

## SA5 — what the real classifier reads

**Placement:** §6 between architecture and measured-result discussion. Static data-to-decision diagram and small actual-result plot/table.

Draw source row 77 as a normalized x/y path with 45 sampled points and small arrows indicating source order. The y-axis follows numeric coordinate values; do not label screen-clockwise direction from an undocumented camera convention. The source class label is “anticlockwise arc” from metadata. Distinguish repeated coordinates with a count/point-index selector rather than jitter that changes the measurements. Three visible pipelines share the pointwise stem: mean baseline; attention 1 + residual+mean; attention 2 + residual+mean. An adjacent ordered-linear pipeline retains all 90 coordinates. Explicitly label the common stem width 24 and class width 15 widths and show where order is lost.

Plot the twelve **actual** test outcomes as dots grouped by model, using a 0–60 correct axis with seed labels/accessible table. No error bar derived from a guessed uncertainty distribution. Use all three seeds; do not display only attention's favorable run or imply that seed variance is data-sampling uncertainty. The visible table in the manuscript is sufficient if a plot would be redundant.

Optional learning-curve disclosure reads the selected model's exact `history`: epoch on x, training/validation cross entropy in nats per example on y, or F1 clearly labeled on a separate panel. Do not join training loss and F1 on a single unlabeled axis. Mark selected epoch 110 for seed 101/two-head and explain that it maximized validation macro F1 with a cross-entropy tie-break, not that it necessarily minimized validation loss. Never draw a test trajectory across epochs; none was used for checkpoint selection.

Text alternative identifies data shape, parameter counts, 330 unique trajectory boundary, 220/50/60 split and the observed ranking. Put the performer/session limitation beside the result rather than repeating warnings in every panel.

## SA6 — edit a trajectory and inspect its learned attention

**Placement:** §6 “A wrong prediction that teaches something.” This is the main real-input investigation and must have its own trajectory/attention/class composition.

**Actual asset and computation:** load `attention-model.json` only when needed. Implement the exact saved model: input coordinate transform 2*x-1; tanh stem; bias-free Q/K/V width 24; two width 12 heads; stable softmax; weighted values; learned output map; residual addition; masked mean; linear 15-class logits; final stable softmax. PyTorch state_dict arrays are out×in. No FFN/norm/position term may be inserted during porting. The original 45 points, learned weights and source label are retained. Biases exist only in stem/classifier as in the program.

**Visible arrangement:** left/top is editable normalized trajectory with indexed current receiver; right/next is selected head's donor-weight strip with exact values and optional 45×45 overview; final panel is the 15-class probability bars. Baseline original is available beside the current hypothetical edit. Receiver selection changes the attention row being inspected, not the classification itself. Head selection displays heads 1/2 with explicit identifiers; no “direction detector” captions. Show original true label class 4 and predicted class 5 after the visible worked baseline.

**Meaningful inputs:** select and move any point via x/y fields or drag bounded [0,1]; reorder selected points using keyboard move-up/down or reverse the whole sequence; insert up to 15 padding points with configurable coordinates and a declared validity flag. The cap of 60 points bounds work and keeps the selected-row plot readable. Prevent an empty valid-point set: explain the missing legal input and do not return class probabilities or divide by zero. An optional single-point duplication preserves source provenance of the original but marks the workspace hypothetical. Reset restores the stated inputs and immediately displays their computed result.

**Live fitted comparison:** show the measured original as a baseline. Edit/reverse/pad the supported trajectory and immediately recompute the frozen model, displaying probability/logit differences. The exact point values, order, validity and mask/pooling policy define the input. Explain unchanged versus changed results without a prediction prompt or correctness score.

**Checked baseline and contrasts:** source row 77/class 4: p(class 4)=.09948105, p(class 5)=.81108749. Reverse all points: max logit error 2.15e-6, attention map permutation error 4.17e-7. Correctly masked five (.75,.75) padding points: max logit error 1.91e-6. Unmasked padding: max logit change 2.23520. Change original point 23.x to 1-x≈.35783: max logit change 15.60714; class 5 probability .49740759 and class 4 .18151061. The unchanged top class is not “no effect.” These data are stored, and implementation should reproduce within a declared float tolerance rather than replacing exact recomputation with lookups.

**Null explanation:** reversal permutes both key and query rows and cancels under the mean; correctly excluded padding does not alter the legal distribution or average. **Contrast explanation:** coordinate editing changes the numerical features; unmasked padding changes the donor set and pooling distribution. Do not label the edited trajectory with a new ground-truth class or declare reversal an actual independently labeled clockwise recording. The model's observed source mistake remains visible.

**Perceptibility and text:** common 0–1 probability scale, signed changes in percentage points, largest absolute logit change and a selected receiver's donor-rank/value readout. A tiny reversal residual is formatted scientifically and described as numerical roundoff, not amplified into a large bar. Large low-probability-logit changes and moderate visible probability changes can coexist; explain both if shown. Screen readers get the top classes, original label, selected point edit and exact change summary. Point overlap has a list selector so every repeated coordinate is reachable. Mobile puts the path and coordinate editor first, donor strip second, probabilities last; the overview is optional.

**Phase-two verification:** compare native saved baseline/probes with ported computation; independently check transposed weight storage, heads merge, stable softmax, masks and pooling. Use numeric tolerance around 1e-4 for logits/probabilities unless measured JS behavior justifies a tighter bound; show actual residuals. Verify genuine new arbitrary coordinate edits recompute, current results recompute appropriately, duplicate positions remain selectable, maximum workload stays responsive, and the weight/data chunk is absent from unrelated lesson loads. No new training is required for these checks.

## SA7 — score sharpness and local sensitivity

**Placement:** §7 alongside scaling/softmax derivatives. Use a two-donor score-gap chart, avoiding the visual form of the trajectory lab.

Working inputs are two raw dot-product scores in [-8,8], per-head dimension selectable 1/2/4/16/64 and positive temperature [.25,4]. Compute p1=sigmoid((r1-r2)/(sqrt(d_k)*temperature)), p2=1-p1. Plot p1 against the scaled score gap on a fixed [-8,8] horizontal window; show the current exact point even if a chosen raw setting produces an off-window gap via a labeled boundary indicator. A second aligned graph shows p1(1-p1), the derivative with respect to its scaled score gap. Do not present this two-logit sensitivity as the complete norm of an arbitrary many-key gradient.

Show the weight distribution and its change under current score/dimension/temperature edits. A common offset to both raw scores is a null; doubling their difference at fixed dimension/temperature is a contrast. Retain enough exact precision to distinguish a small saturated change from a mathematically unchanged result.

The initialization variance inset is static: d_k independent zero-mean unit-variance products sum to variance d_k, scaling by 1/sqrt(d_k) gives variance 1. Put the assumptions on the figure; do not fabricate noisy histograms without a recorded generator. If a sampled histogram is added in phase two, label it simulated and preserve its deterministic generator separately.

Feedback must connect saturation to small derivatives and distinguish changing temperature from changing the model's learned vectors. Keyboard inputs and numeric table make the curves usable without pointer hover. Avoid automatic resizing that makes two nearly identical states look different. Verify formulas, shared-offset invariance and displayed derivative axis/units; no fitting/performance campaign.

## SA8 — distinguish matrix storage from cache storage

**Placement:** §9 immediately after the two memory formulas. Static formula-to-size comparison with an optional bounded editable calculator.

Use two separate storage diagrams: B×H square receiver/donor matrices for materialized weights; N_layers lists of per-token key/value vectors for the cache. Mark each axis and the element bytes. The worked defaults B=1, H=16, L=2048, d=1024, N_layers=12, b=2 yield 128 MiB for one attention-weight tensor and 96 MiB for full K+V cache. Derived matrix-product MAC counts are 4BLd² and 2BL²d. If plotting length, use L values 256/512/1024/2048/4096 and exact formulas; label “derived counts, not measured runtime.”

Do not draw a line for milliseconds, GPU throughput or implementation ranking. One attention matrix is not full training memory. Values use binary MiB=2²⁰bytes. Add a small legend explaining MAC versus two-FLOP convention. Count softmax/masking separately in prose rather than silently treating the formula as total operation exactness.

If editable, let the learner enter the actual dimensions and bytes, and inspect the computed scaling effect in the current live view. Doubling L at fixed other inputs quadruples stored attention-matrix size and doubles cache size. Changing H while keeping standard total width d fixed leaves K+V cache size unchanged but scales the number of materialized weight matrices. These contrast/null relations are algebraically checked. Bound inputs and use safe integer/BigInt arithmetic where needed; no arrays of size L² are allocated just to calculate their size. Display a formula substitution and exact integer byte count to make the result verifiable.

For the tiled-computation inset, use practice 7 scores 0/log(2)/log(4) and values 1/3/6. Draw two blocks carrying (m,l,u), then a normalization merge producing 31/7; the equal average 25/6 is marked as a different operation with its missing mass weights. The inset is a static worked proof, not a pretend GPU benchmark. Ring Attention owns the later device-communication extension.

## SA9 — build two heatmaps with the same mixture

**Placement:** §8 after the interpretability explanation. An investigation of non-identifiability using editable distributions and donor values, visually distinct from the learned trajectory map.

Start with two editable normalized distributions over three donors, e.g. `[.6,.3,.1]` and `[.2,.3,.5]`, and editable scalar or 2D value vectors. Each distribution can be changed by nonnegative weights with explicit normalization; display the normalized weights rather than quietly altering hidden values. Enable direct coordinate edits for each donor value. Initial values `[0,2,4]` make outputs 1 and 2.6 different. Ask the learner to construct values that make outputs agree despite different distributions, with the The current computed result is visible. Setting V1=V3=4 and V2=2 makes both outputs 3.4 while weights remain different. This is a genuine unsolved entity-editing task, not a single preset switch to an already-equal answer.

A separate worked two-donor control uses distributions [.9,.1] and [.1,.9] with both values [3,-2], producing exactly [3,-2]. Their entropies match because entries are permuted. Show output difference and weight difference as different quantities. To contrast, change only the second donor value to [3,2]; outputs become [3,-1.6] and [3,1.6]. A shared downstream deterministic output map cannot recover a distinction already absent from its input; a many-to-one output map can remove additional distinctions.

 Compute the comparison from the complete current inputs. Each edit calculates both mixtures and entropies; feedback identifies weighted cancellation/equal donor values rather than declaring one distribution the true explanation. Use tolerance 1e-8 for scalar exact-arithmetic choices in JS double and a visible numerical difference; a broad rounding tolerance must not count clearly different vectors as identical. Reset clears all states.

Display two parallel weighted-balance diagrams with contributions and exact sums, not another large heatmap. The table alternative lists donor,value,weight1,weight2,contribution1,contribution2. State that these are constructed mathematical distributions, not alternative weights proven reachable by a given trained model. This preserves the interpretation nuance in the paired research references.

Phase two verifies the three-donor contrast/equality construction, the two- Compute the comparison from the complete current inputs. No claims about model explanations are scored beyond the specified mathematical question.

## Practice and handoff checks

All eight practice questions retain separate initially closed optional hints and solutions. Their question text contains the inputs and asks for an explanation; it must remain usable without opening a solution or running a lab. Do not automatically open every details panel during rendering. The independent changed inputs and exact expected results live in the manuscript, `author-results.json` and `additional-fixtures.json`.

During implementation, resolve figure IDs to their actual adjacent placements; provide download routes for the source/program/provenance packet; preserve the existing module sequence and published-status distinction. Verify the current lesson can be read with all interactive investigations collapsed. Complete browser/accessibility/layout/runtime checks in phase two and report evidence; no such checks have been claimed here.

## Precise fixture and scope details retained for implementation

**Two-score sensitivity.** At gap 0, p1=.5 and derivative .25. At gap 2, p1≈.880797 and derivative≈.104994. A common score offset preserves p1; doubling the gap changes sharpness at fixed dimension/temperature. These are exact scalar identities evaluated numerically, not sampled trained Q/K statistics. Display the value and derivative together on each edit.

## Written implementation route and placement — 22 September 2026

Extend the existing Q/K/V layout explanation with the packed-library row correspondence and a nonuniform output-probe arrow. The runnable program exposes gradients and one update. Do not add a generic second lab: the existing vector/mask editor already lets a learner change the relevant mechanism; its exact outputs stay visible.

Use topic-owned responsive diagrams and local scrolling for code/matrices. Long filenames and links wrap within the reader at 320px. Show source/setup/download dependencies at the relevant explanation; deferred Python programs load only on request. Keep labels outside geometric marks where possible, fixed scale comparisons truthful, and current results visible during edits. No learner prediction field, submit button, answer lock or optional prediction gate is specified. Existing numerical/interaction checks still apply, and optional package/checkpoint routes carry their actual unexecuted status until phase two supplies evidence.
