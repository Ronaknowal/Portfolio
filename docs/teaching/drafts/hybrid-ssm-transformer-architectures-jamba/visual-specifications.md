# Hybrid SSM–Transformer lesson: visual and investigation specifications

This is a content-first specification. No website components, SVGs, interactive labs or browser validation are implemented here. The complete learner manuscript is lesson.md. Place each figure and investigation where that manuscript introduces its mechanism. Internal IDs below need not appear in learner-facing titles.

## Shared teaching and rendering contract

Use different representations for different mechanisms: addressable record strips, weighted contributions, converging state histories, residual junctions, a depth ladder, additive memory bands, expert routing, pen trajectories and request timelines. Match the site's typography and controls while preserving these differences. Do not replace them with repeated generic sliders and prose boxes.

Sequence position and network depth are separate axes. A recurrent matrix, short-convolution history, attention K/V bank and model-weight shelf are distinct objects. Shared weights do not imply shared K/V. Caption quantities as exact calculations, actual small-model measurements or architectural schematics. Never invent a quality leaderboard, latency ranking or neural activation.

Every investigation starts with an **unset prediction** and fresh unsolved inputs. Require an intentional answer before revealing its computed outcome. Bind the answer, result and feedback to a signature of all active inputs, model/fixture versions and operation modes. An edit invalidates them. Historical results may remain only with their original inputs and an explicit stale label.

Provide Run/Compare, restore original inputs, and Reset. Reset restores the fresh fixture, cancels pending work, clears the prediction/result/feedback and collapses hints. Worked examples remain readable without a lab. Do not preselect the correct answer or infer a prediction from an untouched default. Reflection should ask which information path changed.

Use semantic labels, accessible tables and textual equivalents for colour. All dragging has numerical or keyboard alternatives. Keep focus on the initiating control; announce only completed result summaries. Motion is optional, respects reduced-motion preferences and has a static/step equivalent. On narrow screens put controls above figures and results below; use a local labelled scroller or vertical table instead of shrinking text. Maintain readable equations, explicit units and the site's established target sizes.

Reject nonfinite values, out-of-range integers and inconsistent dimensions inline. Preserve invalid text for correction and disable Run; never silently replace it with a convenient valid value. A constrained setting must explain why it changed. Calculate with full precision and round only for display.

Load an investigation's computation and selected small parameter arrays only when opened. There is no browser training, Python runtime, internet request or foundation-model download. Cancel stale calculations and stop hidden animations. Full data, all training logs and all model fits are optional downloads, not initial page payloads. A phase-two author may reuse a component only after checking that it supports the required learning representation.

## Inline figures

| ID and placement | Required representation | Accuracy and accessibility |
|---|---|---|
| J01, §1 records | A:2, B:7, C:4 enter a running-summary register and a separately retained labelled-value strip. | Name the two questions; arbitrary measurement units; all records in a text table. |
| J02, §1 attention | Weights 9/11, 1/11, 1/11; weighted contributions 18/11, 7/11, 4/11; output 29/11 beside summary 32/7. | Exact softmax mixture, not exact retrieval of value 2. |
| J03, §1 collision | Histories [2,7,4] and [6,5,4] converge to state (8,1.75), while attention reads differ. | State the particular recurrence; no universal impossibility claim about SSMs. |
| J04, §2 residuals | Two addition junctions. First x+Mixer; second u+FFN, with the u bypass visible. | A four-position strip distinguishes sequence mixing from positionwise FFNs. |
| J05, §2 stack | First eight mixers M,M,M,M,A,M,M,M; FFNs D,E,D,E,D,E,D,E. Repeat across 32 indexed layers. | Original attention indices 4,12,20,28; these attention layers have dense FFNs. Show depth and token axes separately. |
| J06, §3 recurrence | State indexed by channel/state-coordinate; retain, write and read terms; separate short-convolution buffer. | Match the manuscript equation; a conditional affine state update does not imply a linear full network. |
| J07, §3 heads | Four query heads share a K/V bank; original model has 32 query heads and eight banks. | Causal mask j≤t; query count still matters for arithmetic. |
| J08, §4 memory | Additive K/V, recurrent and convolution bands, total, and hypothetical 32-attention comparator. | Exact points in mechanism-results.json; label bytes/GiB and all dtype/batch assumptions. Include a short-context inset; handle zero on linear axes. |
| J09, §4 arithmetic | Causal triangle and new-query row, with projections outside the pair region. | Ideal pair count T(T+1)/2 versus T+1; multiply+add=2; no wall-clock units. |
| J10, §5 router | Probabilities .5,.25,.125,.125; selected E0/E1; outputs [2,−1],[0,3]; weighted result [1,.25]. | Show selected mass .75 and the separately named renormalized comparison. |
| J11, §5 parameters | Stored FFN pool 47,915,728,896; active 8,455,716,864; hypothetical dense 5,637,144,576. | Three SwiGLU matrices; exclusions named; inactive weights remain in model storage. |
| J12, §6 releases | Distinguish original 32×4096 from Large 72×8192, with source links. | Dated 13 September 2026; reported approximate parameter totals are not hardware measurements. |
| J13, §7 trajectory | Source2452 path with eight numbered points, start/end and direction arrows. | Coordinates and table below; 0–100 normalized units, no invented timestamps, speed or pen lifts. |
| J14, §7 candidates | MMM, AAA, MAM, AMM, each with the actual width16/FFN32/head10 structure. | Parameter counts 9314,8474,9034,9034; all FFNs dense, all attention one head. |
| J15, §7 training | Actual 80-epoch validation cross-entropy histories and selected checkpoints. | Preserve all six outcomes and the assessment reversal; no invented smoothing or efficiency ranking. |
| J16, §7 continuation | MAM request split after point3, with recurrent matrices, short buffers and growing K/V. | Use actual saved shapes/norms and logits. Never fabricate a heatmap from a scalar norm. |
| J17, §8 requests | Two request lanes share weights but retain separate state bundles; show scheduler and prefix fork. | Explicitly schematic; a continuation cannot overwrite another branch's starting snapshot. |
| J18, §9 alternatives | Serial Jamba, weight-shared Zamba, windowed Samba, parallel Hymba. | Distinct arrows for parameter ties, activation flow and explicit cache sharing; no quality ranks. |

Worked trajectory J13: [(0,45),(17,4),(55,0),(89,24),(100,70),(72,100),(36,83),(18,39)], original development row2452, class0. Plot x horizontally and y vertically using the numerical coordinates; do not swap features based on a visual guess.

J15 selected epochs: linear37=80, MMM37=79, AAA37=64, MAM37=56, AMM37=68, MAM73=78. Source files contain all histories, not just selected points. J05/J12 facts come from the primary configuration and papers recorded in design.md.

## Investigation JA — what information does a memory retain?

Place at the end of §1. The learner should discover which information a particular state ignores, how a query changes a read, and why finite softmax attention returns a mixture.

Inputs: one to eight editable records; each has a categorical key A/B/C and value in −10…10. Allow adding, removing and moving rows with accessible buttons. Query choices are A/B/C/Absent. Decay λ ranges from zero to one; score gap β ranges from zero to log(100), with useful named log4/log9 presets. An empty record list is invalid because the displayed normalized read has no observations.

The fresh fixture is A:3, B:8, A:1, C:5, query A, λ=.75 and β=log4. The first prediction asks which read changes if record2 is relabelled B→A: summary, attention read, both or neither. Require an answer and optionally reasoning. For subsequent free edits, ask a predicted direction or value for each output. Bind the prediction to row values/order/keys, query, λ, β and requested intervention.

Use hybrid_mechanisms.memory_read exactly. Start s=n=0 and update s=λs+v, n=λn+1, output s/n. Attention assigns score β to matching keys and zero otherwise, then applies softmax. An absent query produces equal scores. This is a deliberately simple categorical-score attention example, not a trained Mamba model.

Show the recurrence register trace and the addressable weighted contributions together. Fresh states are (3,1,3), (10.25,1.75,5.857142857), (8.6875,2.3125,3.756756757), and (11.515625,2.734375,4.211428571), with columns s,n,s/n. Fresh attention weights are [.4,.1,.4,.1], giving 2.9. Relabelling record2 as A gives [4/13,4/13,4/13,1/13] and 53/13≈4.076923077; the summary is unchanged.

Checked contrasting and null cases:

- Worked collision: [2,7,4] versus [6,5,4], keys A/B/C, λ=.5 and β=log9. Final state is identical (8,1.75); label-A reads are 29/11 and63/11.
- Constant values [4,4,4] give both outputs4 despite changed query weights.
- Query Absent on [2,7,4] gives13/3, not “no answer.”
- At β=0 all attention weights are uniform; λ=0 reads the final value; λ=1 gives the ordinary average. These boundary identities were executed in packet_checks.py and must also be verified in the phase-two port.

Feedback explains that this summary's update ignores keys. Repeated matches share probability; changed weights can leave constant-value outputs unchanged. The collision explanation must name this recurrence, never assert that all SSMs cannot copy. Add one initially closed hint expanding recurrence weights and one closed explanation after the prediction.

The accessible table contains key, value, recurrence weight, attention weight and weighted contribution. On mobile, put row edits above two stacked views. Reset restores the fresh case, clears the answer and hides the result. Verify all fixtures, nulls, row order, prediction invalidation, keyboard editing and narrow/reduced-motion views in phase two.

## Investigation JC — budget persistent request memory

Place in §4 after the cache table. The purpose is to separate length-dependent K/V from fixed recurrent/convolution tensors and from model weights.

Fresh configuration: batch3, depth12, attention layers3, model width512, expansion2, state width8, convolution-buffer width4, KV heads2, head width64, K/V bytes2, recurrent bytes4, convolution bytes2, context2048. First ask which components double when context becomes4096; a second optional field asks for total MiB. Both predictions start empty.

Editable bounds: batch1–32; depth1–96; attention count0…depth; context0…262144; width32…8192 in multiples of32; expansion1–4; state width1–256; convolution buffer1–8; KV heads1–64; head width8…256 in multiples of8. Dtypes are bytes/scalar: K/V1,2,4; recurrent2,4; convolution1,2,4. Optional windowed mode uses W1…262144. These values keep integer byte arithmetic within JavaScript's exact integer range.

Use cache_bytes as written. The number of recurrent layers is depth minus attention count. This accounting reserves C convolution values per channel; it is explicitly independent of the teaching model's minimal previous-two buffer. Full attention retains T tokens; windowed attention retains min(T,W). Sum bytes before converting to MiB/GiB. Do not add made-up allocator overhead or model weights. If future UI adds a user-entered weight/reserve budget, label those inputs hypothetical and keep them separate.

The dimensions here are inputs to a storage formula, not automatically a valid complete attention architecture. If query-head arithmetic is added, separately validate its relation to model width. Do not silently derive query count from the KV count.

Exact fixtures:

| Case | K/V bytes | Recurrent bytes | Convolution bytes | Total bytes |
|---|---:|---:|---:|---:|
| Fresh T2048 | 9,437,184 | 884,736 | 221,184 | 10,543,104 |
| Fresh T4096 | 18,874,368 | 884,736 | 221,184 | 19,980,288 |
| Worked T262144 | 4,294,967,296 | 14,680,064 | 1,835,008 | 4,311,482,368 |
| Worked window W4096, T≥4096 | 67,108,864 | 14,680,064 | 1,835,008 | 83,623,936 |

Fresh totals are10.0546875MiB and19.0546875MiB. The worked dimensions are those in §4. Full worked table, zero context and all-attention/all-recurrent cases are retained in mechanism-results.json.

Nulls: T0 gives zero K/V but preallocated recurrent buffers remain. Attention count zero gives zero K/V at any length; all-attention gives zero recurrent/convolution tensors. Increasing T beyond a fixed window does not increase its K/V. Doubling batch scales every request component.

Display additive bands with formula substitution and an exact table, not actual allocated arrays. Plot at most128 calculated sample points. When attention count changes, highlight both the changed K/V slope and changed recurrent-layer count. Windowed mode must draw the direct-read range being removed, because it changes the operator. Linear axes include T0; an optional logarithmic view must explain exclusion of zero.

Bind predictions to every setting and comparison. Reset restores the fresh case and clears results. Invalid inputs keep stale results visibly marked. Phase two verifies byte arithmetic, units, batch scaling, extreme counts, window saturation and accessible narrow layouts. No quality scores or latency estimates belong in this investigation.

## Investigation JD — selected experts and retained probability mass

Place in §5. The learner should distinguish router scores, selected indices, selected weights, expert outputs and stored parameters.

Four expert entities E0–E3 each have a router logit in −8…8 and an editable two-dimensional output in −10…10 per component. k ranges1…4. The default uses probabilities retained from the full softmax; a comparison switch explicitly names selected-weight renormalization.

Fresh logits are [0,log3,log6,log2] and expert outputs [[1,2],[3,0],[−1,4],[2,−2]]. The prediction asks whether increasing E0's unnormalized weight from1 to2 can change the output while E0 remains unselected. Record yes/no and reasoning before showing the answer. Later questions can predict selection or a numeric vector. Bind every logit, expert output, k, mode and intervention.

Compute p=softmax(logits), select top-k, and use w=p[selected]. The comparison divides those selected weights by their sum. Declare deterministic ties by lower expert ID first and visibly label a selection-boundary tie. Do not claim this tie order is guaranteed by every GPU top-k implementation. Any tie exercise must explain the announced policy rather than invent a unique selection without it.

Fresh probabilities are[1/12,3/12,6/12,2/12], selected indices[2,1], mass.75 and output[.25,2]. Raising E0's unnormalized weight to2 gives probabilities[2/13,3/13,6/13,2/13], the same selected indices, mass9/13 and output[3/13,24/13]. In renormalized mode, both inputs give[1/3,8/3].

Worked fixture: logits[log4,log2,0,0], outputs[[2,−1],[0,3],[−2,0],[1,1]]. Selected[0,1], mass.75, output[1,.25]; renormalized[4/3,1/3]. Raising E2's weight to8 selects[2,0] and gives[−8/15,−4/15]. Four zero logits create a boundary tie: under the declared policy choose[0,1], mass.5 and output[.5,.5].

Nulls: changing only an unselected expert's output cannot change the result with fixed scores; all-zero expert outputs give[0,0]; k4 makes the two normalization conventions coincide; adding the same constant to every logit leaves the result unchanged. Worked zero/tie fixtures are saved in mechanism-results.json; the additional identities were executed in packet_checks.py. Verify them again for the phase-two port.

Use four router bars, selected brackets, a visible selected-mass label and vector contribution arrows, with a numeric-table alternative. Keep the parameter shelf visible to avoid suggesting that unselected experts require no storage. Editable expert vectors are teaching inputs, not claimed outputs of a trained MoE. Feedback explains the full-softmax denominator before selection and distinguishes score edits from value edits.

Reset restores the fresh inputs, k2, retained mode, empty prediction and hidden result. Mode changes invalidate predictions. Compute is bounded to four experts and two outputs. Full training, balancing, capacity and dispatch remain with the dedicated MoE topic.

## Investigation JB — draw a stroke and continue its actual state

Place in §7 after worked source2452. This investigation must perform real small-model inference on editable coordinates, not switch between scripted transcripts.

Fresh input: development source2970, original class1, coordinates[(38,100),(100,92),(88,77),(75,62),(50,46),(25,30),(12,15),(0,0)]. Model MAM-37, boundary after three points, intervention K/V reset. Initially show the trajectory, editable table and prediction question, but no model probabilities or feedback.

All eight points are editable integers0–100 using fields or keyboard-accessible point controls. The count stays eight because that is the trained representation. Any coordinate edit marks the input “edited; original source class1, new ground truth unassigned.” An optional all-(50,50) degenerate probe has no true digit. Do not retain the original label as a correctness judgment on a modified trace.

The model selector may offer MAM-37, AMM-37, MMM-37, AAA-37 and MAM-73, loading their actual saved weights and showing counts. The flattened linear model has no streaming path; show it only in the full-trace/static comparison. Boundary is0…8. Modes: carry; recurrent reset, which clears both recurrent state and short history; K/V reset; convolution-only reset; position-offset reset.

The initial two questions are independently unset: “Will K/V reset reproduce uninterrupted logits?” and “Must a changed computation change the final class?” Optionally request a predicted class/probability. Bind the answer to all coordinates, model key/weight version, boundary, fault mode and position policy. Model/input changes must never reuse another run's request state.

Port the exact selected StrokeModel. Inputs are coordinates/50−1 plus t/7,(t/7)^2 on the original grid, embedding4→16, three residual layers, learned RMS weights with epsilon1e−6, exact SiLU/softplus, selective state width4, three-tap depthwise convolution, dense SwiGLU32, final RMS and ten-logit head. Preserve biases and learned parameter arrays. Attention is one head with scale1/sqrt16 and mask j≤t.

Cache shapes per request: recurrent H[1,16,4] and short history[1,16,2]; attention K,V[1,prefix,16]. At boundary3 MAM contains two pairs of recurrent/history tensors and one K/V pair[1,3,16]. Prefix diagnostics keep global position indices; they must not renormalize to the prefix length.

Every fault branch must receive a clone of the same correctly computed prefix state. No branch mutates the prefix snapshot used by another branch. Author fixtures retain shapes, norms, full/prefix/branch logits and probabilities; optional individual state cells must be actually computed rather than painted from a norm.

| Fresh2970 mode | Final digit | Probability of original label1 | Maximum logit difference from full |
|---|---:|---:|---:|
| Carry | 1 | .996476829052 | 5.7220459e−6 |
| Recurrent reset | 1 | .855561196804 | 5.892972946 |
| K/V reset | 2 | .171031087637 | 5.051756859 |
| Convolution reset | 1 | .937770783901 | 5.178080559 |
| Position reset | 1 | .928655028343 | 4.049905300 |

Worked2452 remains class0 under every fault. Its probabilities of zero are .997078061104 carry, .997088611126 recurrent reset, .996498703957 K/V reset, .997109115124 convolution reset and .957256793976 position reset. Corresponding non-carry maximum logit differences are9.208744049,3.938186169,9.030287743 and9.952294350. Preserve this example: an unchanged argmax can hide a large computation error.

Input-edit contrasts: fresh point4(75,62)→(50,50) gives uninterrupted class1 with P1=.995079159737. Worked point4(89,24)→(50,50) gives class0 with P0=.991652309895. All points(50,50) give class7, P7=.719944179058 and P8=.213528305292; neither uniform output nor a true label is expected.

Nulls: reset at boundary0 clears empty state; reset after boundary8 has no suffix to affect. Compare the fault branch with the matching carry branch to verify these identities. Full matrix-attention versus tokenwise floating-point reductions can still differ slightly, so a bitwise comparison against full inference is the wrong null test.

Show the path, boundary marker, per-layer cache bundle, full/branch probability bars and per-position logit differences. Include the numeric maximum difference, not only a class badge. Explain how K/V reset removes earlier addressable representations, convolution reset changes transformed inputs and state, and offset reset changes supplied position tags. Earlier logits are internal diagnostics; fitting supervised only the final position.

Bound computation to one eight-point input, three width16 layers and a small number of branches per Run. Load the chosen roughly9K-parameter model only when needed. Do not bundle all300 validation-prefix logs or all six model states on initial page load. Use a bounded worker if needed; coordinate edits mark stale results and wait for Run rather than recomputing on every keystroke.

Phase-two verification must reproduce saved worked/fresh/edited/degenerate fixtures and full/stream/chunk paths from actual arrays. Initial logit tolerance is1e−4 for a faithful port; native float32 full/stream differences are below1.26e−5. A precision-driven tolerance revision needs explanation and cannot hide a wrong operator. Verify future edits leave earlier outputs unchanged, branches/requests are isolated, boundary nulls hold, invalid coordinates are blocked, all changes invalidate predictions and mobile/keyboard operation is usable.

## Practice, downloads and phase-two completion

Keep all ten practice questions and both hint/solution disclosures initially closed. Preserve changed examples and explanations. Optional branches must not obstruct the core route. Use the actual previous Neural ODE and next Titans links without reordering the curriculum.

Make complete programs and data attribution available as deliberate downloads. The separate large-model deployment example is labelled unexecuted and never runs on page load. Author filenames and evidence belong in maintainer/download contexts, not as new curriculum topics.

The phase-two finisher implements these topic-specific representations, ports and verifies the numerical functions, checks content integration, accessibility, responsive layout and browser performance, then updates implementation status separately. None of those delivery checks is claimed complete by this specification.

