# MLA — figures and investigations for phase two

Research/write only, 13 September 2026. Implement these contracts later; no rendered UI, browser acceptance or runtime asset integration is claimed here. Use representation flow, algebraic movement, rotation geometry, tensor accounting and real latent interventions for their distinct learning purposes.

## Evidence and interaction rules

`mechanism-calculations.py` and `mechanism-fixtures.json` own constructed arrays, exact matrix/rank/rotation contrasts and calculated work/payload. `author-calculations.py`, `author-results.json`, `forecast-model.json` and the original Libras data/metadata own actual trained forecasts and the declared rank intervention. A selected model's weights are not interchangeable with a hand vector; a payload count is not a hardware measurement.

The manuscript's worked examples are **ungraded explanatory walkthroughs**. They may reveal their already-published answers without awarding practice. Each investigation below has a different, explicitly named **fresh gated baseline** supplied by `practice-calculations.py` and `practice-fixtures.json`. That latter file is author/verification evidence: expected outputs, hidden future points and control answers must not appear in the learner interface before committing the input-bound prediction. Merely requiring a prediction about the manuscript's already-solved default is not independent practice. Compute results from editable inputs; do not make a fixture lookup the only supported interaction.

Every investigation starts with an unanswered prediction linked to the exact current and proposed inputs. No selected answer, populated numerical guess or already-shown repair counts as learner practice. A consequential edit closes feedback and invalidates the old attempt; preserve it only as explicitly historical evidence tied to its original inputs. Commit then reveal. An optional ungraded exploration action may reveal without a prediction but earns no practice completion. Reset restores the stated baseline, clears predictions and closes feedback. A purely presentational selection of an already-revealed head or unit need not invalidate unchanged arithmetic.

Accept nulls, small changes and ties according to actual results. Use numeric tolerance appropriate to the representation, and explain rather than force a desired winner. Free-text explanations use a visible rubric, not fabricated semantic grading. Invalid/NaN/infinite input gives local guidance and keeps the last valid state; never silently clip an invalid prediction into a valid one. All-masked attention has a clear no-legal-key state.

Show input units, position IDs, head IDs and exact shape meanings. Probabilities use 0–1; latent vectors and scores are signed quantities. Color has redundant text/line-style/symbol meaning. Every drag has native numeric or keyboard controls; outputs have equivalent tables and text summaries. Keep initially closed practice/hint/solution disclosures, visible focus, reduced-motion support and concise reveal announcements. No autoplay, timer pressure or simulated training animation. On mobile stack related views and inspect one head/coordinate pair at a time; bounded local table scrolling must not widen the entire article.

## F1: what one memory position stores

Place in §1 after the common-description explanation. A current input projects to a content latent; different head-specific key/value maps point to visibly distinct outputs. A second branch projects and rotates the shared positional key. Draw a persistent-storage bracket around **latent plus rotary key**. Queries and head outputs remain outside it. Label the layer; no shared cross-layer state is implied.

Use a neighbouring GQA sketch to show one directly shared K/V pair versus a shared MLA latent producing distinct pairs. The diagram should not imply that a latent is a probability vector, that all reconstructed keys agree, or that the rotary branch is a content-free integer position. Accessible text describes all paths in calculation order. This small inline explanatory figure needs no mandatory lab.

## F2 / I1: two exact paths through a latent

### Placement and baseline

Place the score-contribution figure in §3 and the full manipulation workspace in §4 after the hand calculation. Align two panes: expanded-head computation on one side, latent-first computation on the other. They share visible input entities; moving a linear map changes the order of calculation, not its values. Connect the two signed content/rotary score terms to their sum, then the original scale, legal mask, one softmax and weighted value/latent paths.

The ungraded walkthrough uses `manual_inputs` from the fixture JSON: two query heads, content/key/value/latent/rotary widths all 2; three memory positions 0,1,2, one current query at 2. Latents `[[1,0],[0,1],[1,1]]`; content queries `[[1,0],[1,1]]`; key maps `[[[1,0],[0,2]],[[1,1],[1,-1]]]`; value maps identity and `[[2,0],[1,-1]]`. All raw rotary keys `[1,0]`, raw rotary queries `[1,0]` and `[0,1]`. This hand fixture rotates by π/2 per position, explicitly different from the ordinary base-10000 real model. Original score scale is 1/2. Output-map rows `[1,0,.5,0]` and `[0,1,0,.5]` complete the attention contribution.

**Fresh gated baseline I1:** content queries `[[.5,1],[-1,2]]`; latents `[[2,-1],[.5,1],[-.5,2]]`; key maps `[[[1,.5],[0,2]],[[1,-1],[.5,1]]]`; value maps `[[[1,0],[.5,1]],[[1,1],[-1,2]]]`; raw rotary queries `[[1,.5],[-.5,1]]`, raw rotary keys `[[1,0],[.5,1],[1,-1]]`; memory positions `[0,1,3]`, query position 3, frequency π/3 and original scale .5. Keep the same optional output map. Propose adding .75 to record 1's first latent coordinate. Predict which score/output paths can change and whether expanded/absorbed execution still agrees, then reveal. The unchanged fresh head outputs are `[-.1917888763,1.5849537615]` and `[1.4882424885,3.7067992196]`; full changed results are saved author-only. A second fresh control changes only head 0's value-map `[0,0]` entry by +.5: its weights and head 1 output are exactly unchanged. Reset of gated practice returns to these fresh inputs, not the solved walkthrough.

### Genuine edits and prediction

Edit every latent coordinate, content query, raw rotary vector and key/value/output-map entry in −6…6. Edit logical memory/query positions within 0…16 and a common rotation frequency within 0…π, retaining explicit causal legality. Add/remove up to eight memory records with fully editable values. Keep the two-head hand operator small; an independent schematic may compare larger counts without generating unprovided trained activations.

The pending question is whether expanded and absorbed results agree and which quantities change under the proposed edit. No default answer. An optional numeric estimate can target one head output or weight. Reveal actual reconstructed K/V, effective queries, the two score terms, normalized weights, latent mixtures, expanded output and final attention contribution. A path switch alone is a representation change; editing inputs is a change to the calculation.

### Checked results and controls

- Baseline head-0 logits `[0,0,1]`, weights approximately `[.211942,.211942,.576117]`, latent/output `[.788058,.788058]`. Head-1 logits `[1,-.5,1]`, weights `[.449816,.100368,.449816]`, latent `[.899632,.550184]`, output `[1.799265,.349449]`. Final output `[1.687691,.962783]`. Full precision is retained; formatted decimals must not drive computation.
- Expanded and absorbed paths agree to float64 tolerance 1e−12 for the fixture and actual valid edits. A wrong operation may differ; do not apply a permanent equality label independent of calculation.
- Change first latent by `[0,1]`; both its key and value reconstructions can change, affecting weights and mixtures. Actual contrasting results are saved. This differs from changing only a value map.
- Change head 0 value-up entry `[0,0]` by +1: all attention weights remain unchanged; head 1 output remains unchanged. Head 0 output follows its changed linear read. This is a controlled path isolation.
- Apply invertible latent basis S=`diag(2,.5)`, transform every c by S and right-multiply each up-map by `inverse(S)`: exactly unchanged outputs. A learner may edit a bounded invertible 2×2 S; reject near-singular cases with an explanatory condition indicator instead of silently amplifying numerical noise. Changing c without compensating maps is a contrasting operation.
- Shift every position by +5, keeping frequency and vector conventions: unchanged output to tolerance. Shift queries only: saved non-null contrast. Do not reset cached key angles behind the scenes.
- The nonlinear inset uses latents −1,1 with weights .5,.5: mixing ReLU values gives .5, ReLU of the mixed latent gives 0. Teach why this operation cannot move through the sum. No nonlinear path may be represented as another exact MLA branch.

For wrong-scale teaching, the baseline widths coincide and cannot expose the error. Use the full real model in I5, or a fully specified constructed extension with `dc != dk`, while keeping the intended scale defined by `dk+dr`. Explicitly label the equal-width null; do not manufacture an error in this hand example.

### Access, feedback and bounds

Signed vector drawings and contribution tables show weight×coordinate and the final sum. A matrix's axes identify input/output coordinates; token positions and head connections are not interchangeable. Mobile selects one head's detailed path, keeping the shared memory context visible. Limit eight positions, two heads and small matrices; compute on commit, not an unbounded animation loop. Phase two checks arbitrary valid edits, controls, output-map orientation, stable softmax, legal masks, prediction binding and accessible parity with the full NumPy reference.

## F3 / I2: does a rotation commute with this projection?

Place in §3 directly after the obstruction example. Show the same two-dimensional input vector through two paths: `U then R` and `R then U`. Each has an intermediate vector and final vector on common axes. The learning goal is operation order, not a decorative rotating arrow.

The ungraded walkthrough uses U=`diag(2,1)`, R a quarter turn, input `[1,0]`. First path ends `[0,2]`; second ends `[0,1]`. The matrix commutator `RU−UR=[[0,1],[1,0]]` is saved. The effective-query inset for q=`[1,0]` shows required latent queries `[2,0]` for key rotation zero and `[0,−1]` for quarter turn. It explains why one fixed absorbed query cannot handle all positions in this general construction.

**Fresh gated baseline I2:** U=`[[1,1],[0,2]]`, input `[1,-1]`, angle π/3. Ask for equality and the difference's direction before reveal; project-then-rotate is `[1.7320508076,-1]`, rotate-then-project `[1.7320508076,.7320508076]`. The fresh isotropic control U=`2I` makes both paths `[2.7320508076,.7320508076]`. These author-only values must not be displayed in the initial question or a preselected answer. Gated reset restores the sheared U and unsolved vector, while ungraded walkthrough reset remains separate.

Edit all four U entries in −4…4, vector coordinates in −4…4 and angle in −π…π. The learner predicts equality/difference and optionally its direction before revealing. Controls: identity/isotropic U commutes with every rotation; angle zero is a null; unequal stretch with a generic nonzero rotation differs. Some edited vectors can lie in the commutator's null space even if the full matrices do not commute—feedback distinguishes “same output for this vector” from “matrices commute for every vector.” Compute both matrix and selected-vector differences.

Use angle in radians with a degree readout; keep axis units abstract vector coordinates. Native numeric controls and an ordered table accompany dragging. No time-based simulation is necessary. Update on commit; bounded 2×2 arithmetic. Phase two verifies both multiplication orders, sign conventions, zero/π/special commuting cases, input-bound prediction and truthful distinction between a selected-vector null and general identity.

## F4 / I3: latent storage versus arithmetic

Place in §5 next to the exact payload and work counts. One record has a content field `dc` and a rotary field `dr`; one stack represents positions, then layers and requests. A separate transient-work panel counts query heads and dot-product widths. Do not draw one block per real token or allocate the large tensor being estimated.

Worked ungraded inputs are B=1, N=60, L=32768, H=128, dk=dv=128, dr=64, dc=512, two-byte unquantized values. Editable bounds B 1…64, N 1…256, L 1…262144, H 1…256, dk/dv/dc 1…2048, positive even dr 2…256; enforce bounds and safe integer/rational arithmetic or BigInt. Payload precision 1/2/4 bytes, with any sub-byte option explicitly labelled idealized and excluding scales/packing. A real source-reported quantized format is a separate fixed record illustration, not an inference from that scalar precision control.

**Fresh gated baseline I3:** B=3, N=24, L=8192, H=24, dk=dv=64, dc=192, dr=32, s=2 and T=1. Propose doubling H to 48. Ask for the new payload and which arithmetic terms change, with no populated guess. Author-only payload is 264,241,152 bytes (252 MiB) before and after; original expanded/absorbed cores are 188,743,680 / 490,733,568 operations and both double. These dimensions are not the worked model or the separate manuscript exercise. Gated reset restores this fresh configuration and closes results.

Prediction starts empty for the new payload, ratio or which costs change. Reveal `B*N*L*(dc+dr)*s`, exact bytes and selected binary/decimal units. Keep all comparison assumptions visible:

- Plain MHA with qk/v widths 128/128: 32,768 numbers per token/layer; GQA-8 2,048; MQA 256. These are distinct parameterizations.
- The same MLA function's literal expanded K 192 / V 128 representation: 40,960; expanded content with one shared rotary key: 32,832; compact MLA 576.
- At declared 60-layer/32,768-position/two-byte boundary, compact bytes 2,264,924,160=2.109375 GiB; plain MHA 120 GiB; GQA-8 7.5 GiB. Use real derived values from fixtures, not rounded estimates in further calculations.
- The 93.3% source headline compares different complete models; do not annotate it as the rounded counterpart of either same-shape ratio.

The arithmetic panel computes `2*B*H*T*L*(dk+dr+dv)` for expanded core, `2*B*H*T*L*(2*dc+dr)` for absorbed core, and separately the cost of reconstructing all prefix K/V from latents. Fix/allow T 1…L, label multiply-add=2 and omitted projection/softmax overheads. At B=T=1,L 32768 the core counts are 2,684,354,560 versus 9,126,805,504; ratio 3.4. All-prefix reconstruction 1,099,511,627,776 is another comparison, not part of an already-expanded-cache core. Query absorption and per-query value up-projection each 16,777,216 operations for those widths.

Controls: double H with fixed cache widths→compact payload unchanged but core work doubles; double L→payload and single-query core double; double all-query T with L fixed→core doubles but stored history unchanged; count c once despite its dual key/value role. Changed practice payload is 3,145,728 bytes = 3 MiB. No latency axis, GPU-capacity promise or quality curve. A static metadata inset shows the documented V3.2 sparse FP8 record 512 + 16 + 128 = 656 bytes with its exact field types; later V4 formats are explicitly different, not silently converted to this layout.

Unit changes are presentation-only. Shape/model assumptions invalidate the old prediction. Readable factor table, accessible statement of ratios and maximum 32 chart samples suffice. Phase two verifies overflow, units, identical-comparison dimensions, omitted-state labels and the absence of measured-looking invented plots.

## F5 / I4: a direction can be small in a matrix and large in an input

Place in §7 after the rank-truncation explanation. The ungraded walkthrough uses M=`diag(10,1)`, x=`[0,10]`, retaining one singular direction. Show input x, retained/discarded directions, Mx and the rank-one approximation's output in a linked pair of planes. Squared matrix error is 1 while output changes from `[0,10]` to `[0,0]`. Use independent scales or clear units so matrix and output errors are not drawn as the same quantity.

**Fresh gated baseline I4:** M=`[[3,1],[0,2]]`, x=`[2,-3]`. Ask whether retaining its leading singular direction preserves this input's output, then reveal the actual geometric error. Original output is `[3,-6]`; reduced output `[1.0839748528,.3282011774]`, with squared parameter error 3.3944487245. The independent calculation retains its full direction/basis. An input along that retained direction with the same norm, `[3.1789229741,1.7013079452]`, is a checked equality control. Sign-flipped singular vectors define the same projection; do not grade their signs as an error. Gated reset restores the unsolved non-diagonal matrix and original x.

Allow all entries of a 2×2 M and the input x to be edited in −10…10. Learner predicts whether the output changes and whether small matrix error ensures small output error. Keep no implicit target label. Compute the top right singular direction from A=MᵀM and form `M_reduced=M*v*vᵀ`; apply both to the entered x. The two eigenvalues of A are `(trace(A) ± sqrt((A00−A11)^2+4*A01^2))/2`. Normalize a valid eigenvector; handle zero and repeated-eigenvalue cases explicitly. A deterministic tie direction is a display convention, not a unique optimum. Expose the chosen direction and label ties rather than claiming every singular vector has unique meaning.

Controls: default input along retained axis `[10,0]` gives output agreement; default input along discarded axis gives a large discrepancy; full two-direction reconstruction is a null; zero matrix gives zero outputs and no unique direction. Any matrix edit recomputes the decomposition and invalidates the previous prediction. Do not label the parameter-optimal result as a trained classifier or forecast-optimal compression.

The neighbouring rank inset displays the retained fixture with logits `outer([0,1,2],[0,1,2])` of rank 1 and its actual row-softmax matrix of rank 3, determinant≈.024430727. It is a static counterexample showing that matrix rank does not survive softmax; no unstable live rank-estimation interface is required.

Accessible input/output tables and direction coordinates accompany the geometry. Bounded 2×2 arithmetic, optional discrete transitions, no large linear-algebra package needed. Phase two checks arbitrary matrices, tie/zero conventions, reconstruction error and actual input-specific output effects against independent NumPy results.

## F6 / I5: actual normalized latent cache and rank intervention

### Place and complete model

Place in §6 after the declared outcomes. The real observed trajectory appears with a time boundary; a linked cache strip has one normalized content row and one rotary row per observed position. Select a query head to inspect its content score, rotary score, final scaled score, weights, mixed latent and head output. These are actual recorded/evaluated activations, not reconstructed from decorative token labels.

Use the complete `LatentForecaster` defined in `author-calculations.py` and exported in `forecast-model.json`. One pre-norm block, model width 24, four heads, dk 4 / dv 4 / dr 2, content latent 8 / query latent 12, latent RMSNorm epsilon 1e−6, residual/final LayerNorm epsilon 1e−5, GELU FFN 48 and two-coordinate forecast. Attention projection weights output-by-input and bias-free; stem/FFN/forecast include biases. Ordinary adjacent-pair RoPE base 10000, no dropout, fixed `2*x−1` transform. Score scale 1/sqrt(6) remains unchanged after any basis projection. Implement every residual and norm path before claiming model parity.

The original eight-coordinate c is computed and normalized before projection. Optional basis P has shape 8×r; cache `c@P`, key/value up maps become `U@P`. The saved full basis is an orthogonal coordinate change; its first four columns define the predeclared truncation. Never replace the original RMSNorm with an r-coordinate norm or describe the intervention as a separately trained smaller model.

### Actual baseline and controls

The ungraded worked example uses source row 77, first 32 observed points, full selected model. Its already-published true next point is `[.5938100219,.25]`. Actual full prediction `[.5997370481,.2636269927]`; rank-4 `[.6137580872,.2609272301]`. Actual compact payloads 1,280 bytes and 768 bytes; cache shapes `[1,32,8]+[1,32,2]` versus `[1,32,4]+[1,32,2]`. These bytes exclude metadata and other model state.

**Fresh gated baseline I5:** use the same source row's **first 27 points**, with its next point and forecasts hidden. Propose reflecting frame 19's **y** coordinate with `y→1−y`. Ask which earlier outputs must remain unchanged, whether the old cache remains valid, and optionally for a fresh next-point estimate. The saved model produces full prediction `[.6383265853,.3148220479]` and edited `[.6382712722,.3151544929]`; rank-4 gives `[.6740129590,.3336106837]` and edited `[.6720473766,.3352644145]`. Both preserve every output before 19 exactly, and expanded/absorbed agree within 1.79e−7 transformed units. Compact payloads are 1,080 and 648 bytes. All results and the true next point in `practice-fixtures.json` are author-only until reveal. These are changed prefix/question/coordinate inputs, not another benchmark or a new fit. Gated reset restores this 27-point fresh prefix and closes answers.

- Expanded versus absorbed same full model: max transformed output difference 1.78814e−7 for this prefix. Full versus incremental: 3.57628e−7. Full rank-8 basis on all validation inputs: 3.57628e−7. Browser full-network parity target ≤ 2e−5 in original coordinates under its documented arithmetic; inspect materially larger differences rather than inflating tolerance.
- Full versus rank-4 is a genuine changed function. Keep the actual test RMSE .0249463655 versus .0919342231 and simple baseline table. Rank4 retained matrix error 2.862722 and singular values are actual saved data. The selected update 200 is the budget endpoint, not convergence evidence. Do not imply the dataset scores are recomputed live for each edit.
- Reflect observed frame 23 x→1−x: full next forecast `[.6005192399,.2634072304]`, rank-4 `[.6144915819,.2612116635]`. Outputs strictly before frame 23 remain exactly unchanged in the checked model; later effects may be small or cancel for other inputs.
- Uniform position shift +100: max transformed error 1.78814e−7 for both saved cases. Shift all relevant IDs while preserving frequency and cache meaning. A query-only shift or stale cache must be treated as a different calculation.
- Wrong scale on full model changes next output to `[.5997453928,.2636736035]`, max prefix transformed error 9.32515e−5. This is small, not zero; show the honest magnitude. Rank4 happens to have dc=dk, making the accidental default scale numerically correct—an explicit null illustrating inadequate test coverage.
- Repeated reveal of identical input/model/basis produces the same output. A changed point with negligible final effect is accepted rather than exaggerated. No invented class probabilities: the outputs are unconstrained coordinates, possibly outside 0…1.

### Genuine edits, prediction and reset

Edit every observed point's x/y in 0…1, with pointer or native numeric controls. Move the observed-prefix boundary within 2…40 points. Choose expanded/absorbed execution, original/full-rank/rank-4 representation, optional exploratory rank 1…8 using the saved fixed singular basis, and a common position shift 0…128. Other exploratory ranks get newly computed selected-input outputs, not fabricated aggregate benchmark rows or claims they were trained. Input source versus edited input is explicit.

Prediction begins unset: will the chosen intervention preserve the computation, change it, or make the current cache invalid? Optional next-coordinate estimate is empty. Commit then reveal original/changed outputs, numerical difference, per-head mechanism and record payload. A head-selector after reveal is presentational; changing head as the subject of a new graded prediction creates a new attempt. A basis/model/input/position/mask change invalidates old prediction and dependent cache. Reset restores exact source input and full model, clears cache/generated state and closes feedback. Do not compare a new basis with old-basis cached coordinates.

The true future must never enter an observed prefix before its boundary is extended by a deliberate new experiment. This packet evaluates one-step teacher-forced forecasts; it does not export generated rollout traces. If phase two adds a rollout, it must compute each actual next input from the preceding model prediction, explicitly distinguishing that new computation from recorded evidence. A rollout is not required to meet this contract because the preceding GQA topic already teaches it fully.

### Access, performance and phase-two acceptance

Provide a native frame/coordinate table, previous/next selection and keyboard nudges alongside trajectory dragging. Label positions zero-based, source rows one-based, axes original normalized x/y. Show any forecast outside source bounds with an expanded range or visible overflow indicator; do not silently clip. Cache values use signed scales and exact tooltips, not probability colors. Mobile stacks trajectory, chosen cache row and one head's scores; preserve all data through selection and accessible tables.

Bound inference to at most 45 points, width 24, one block and four heads. Recompute on explicit commit; throttle any optional preview. Load only this selected small model plus necessary example/basis data lazily; leave full training histories and all 330 unique source trajectories as optional downloads, not eager runtime imports. No browser optimizer, GPU dependency, historical fit rerun or whole-corpus evaluation. Release stale arrays when input/basis changes; show a recoverable asset-load failure state. A worker is optional only if profiling warrants it.

Phase two verifies complete network parity, changed-coordinate results, full/absorbed/incremental/basis controls, input-bound predictions, exact source/edited distinctions, mask/position/basis cache validity, units and no hidden future input. Then perform independent content/code review, keyboard/screen-reader/narrow-viewport/reduced-motion checks and browser integration. Native author calculations do not constitute rendered acceptance.

## Retention and continuation

Retain all thirteen files: four Markdown documents, original data/metadata, full model study/results/weights, independent mechanism program/results and independent fresh-practice program/results. They are pending implementation inputs, not disposable scratch. Derive compact semantic runtime assets later and preserve optional full reproductions. Any consequential phase-two content correction requires updating the affected packet and root-owned content checkpoint; unchanged passing fits do not need to rerun for prose or spacing edits.
