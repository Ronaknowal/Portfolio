# Modern Hopfield Networks — visual and investigation specifications

## Live exploration contract — 21 September 2026

Open each investigation with its current inputs, intermediate mechanism and complete current output visible. Apply valid edits to meaningful entities immediately and update diagrams, tables, units and causal explanation together. No prediction entry, predicted-answer choices, commitment, prediction grading or answer-unlock feature is part of this packet, even optionally. Model predictions and mathematical masks/gates remain subject matter.

Use the topic-specific controls and checked fixtures below. Pair sliders or direct manipulation with labeled keyboard/numeric controls; keep presets as starting points, not the only editable values. A pinned baseline preserves its inputs, seed, units and outputs while the current case changes. Explain both a meaningful contrast and an unchanged/null result, then connect the observed effect to a practical design decision. Reset restores the stated fixture and current result. Invalid text has a local explanation and a clearly identified last valid result; never silently clamp or pair new inputs with old output.

Step/Back and bounded Run controls advance a real computation or reveal its chronological stages, not permission to view an answer. Show the current state and its result throughout. Keep exact small calculations live. For costly frozen inference, debounce or run bounded work with pending/current state labels and stale-result cancellation; inspect saved measurements without implying fresh training. Respect reduced motion, keep focus stable and avoid announcing every animation frame. Independent written practice and its hints/solutions stay separate.

Phase two must test default results without any action, meaningful edits, quick consecutive edits, valid extremes, null/invalid cases, reset, linked-view agreement, keyboard operation and readable phone layouts. The mathematical/reference checks already specified below remain; these live browser checks have not been performed in this content-only revision.

### Topic-specific live route
**Shape an associative memory.** Flip cue bits and visit order; edit continuous memory vectors, temperature, keys/queries/values and real handwriting pixels.
**See the consequence.** Show current energy, attractor steps, retrieval weights, payload and classifier/reconstruction outputs. Step iteration without hiding its current state.
**Decision connection.** See how ambiguity, scale and address/content choices change retrieval and when classification and reconstruction objectives diverge.


Content-first packet, 13 September 2026. This file specifies teaching visuals; no website visual or lab is implemented. Stable ID: modern-hopfield-networks. Consume lesson.md and design.md with this contract.

## Shared representation and evidence rules

Use the concept's own representations: signed binary links, coordinate-update traces, memory geometry and energy contours, separate key/value lanes, actual 8×8 handwriting and weighted-memory thumbnails. Do not replace them with a generic prose-only simulator. Reuse existing accessible controls when appropriate; neither visual novelty nor repeated card structure is a goal.

All displayed mathematical quantities are dimensionless unless marked as bytes, counts, coordinate indices or normalized pixel intensities. β is inverse temperature for the stated score scale. Matrix X stores memories in rows throughout the manuscript. Score softmax is over the memory dimension. Keep query/key/value dimensions and coordinates explicit.

**Sources:** associative_memory.py and mechanism-results.json contain exact constructed mechanisms and the measured eight-bank binary scan. digit_memory.py, digit-results.json and digit-memory-fits.npz contain executed CPU fits, source IDs, weights, class distributions and metrics. author_calculations.py and investigation-checks.json contain additional exact/float fixtures. Original UCI files and data-provenance.md are required retained inputs. Geometry plots calculate from those equations; do not label hypothetical geometry as measured handwriting, or present CPU math as GPU timings.

**Access:** each nontext representation has a nearby descriptive caption, concise alt summary and an expandable labeled numeric table. Sign is indicated by +/− and line style, not color alone. Query, stored memory and output have different shapes as well as color. Use real text for labels. Keyboard users can edit all coordinates/pixels without dragging. Honor reduced motion. No automatic animation. Formula and text equivalents remain readable without activating a lab.

**Narrow layout:** at 360 CSS px use sequential labeled panels sharing the same entity IDs, rather than compressing a desktop diagram. Keep minimum readable axes and values; wrap tables only within their own labeled scroll region when necessary. The full lesson must not overflow horizontally. Large matrices are a compact selected-row view with a full numeric disclosure. Pixel grids remain square, with a separate focused-cell editor large enough for touch.

**Engineering boundary:** phase two derives topic-owned lazy assets and controls from this packet. Load only the current investigation's required weights/data. No model training in the browser. Use small pure numerical functions, memoized from active inputs; do not re-read every corpus row on keystrokes. Expensive view recalculation waits for explicit Run. Keep source data/complete downloadable programs available without bundling the full research record into the lesson entry.

## Inline figures

Each row defines placement, instructional question, exact content, narrow/access equivalent and required phase-two validation. Figures introducing essential ideas are visible inline; they are not hidden behind a later lab tab.

| Figure / placement | Purpose and exact content | Presentation, access and checks |
| --- | --- | --- |
| 1 / §1 | Address lookup, cue lookup, then separate recall/reconstruction/classification checks | Schematic only. Two lookup arrows and three named evaluation branches; stack on phones. The unseen query is not labeled an exact stored pattern. |
| 2 / §1 | Norm trap: q=(1,0), memories (1,0)/(3,1), scores 1/3 and distances 0/√5 | Equal geometric scales, x range [−0.25,3.5], y [−0.25,1.5]. Label vector lengths, scores and distances; provide a numeric table. |
| 3 / §2 | Four-feature pattern [1,1,−1,−1], six signed connections, W entries ±0.25 and zero diagonal | Sign labels and line styles distinguish edges. Highlight construction of one matrix row through coordinate products. Verify symmetry and all edge signs. |
| 4 / §2 | Worked cue [1,−1,−1,−1], forward visits: field 1 is .25; field 2 is .75 and repairs the cue | Before/field/after trace from binary.worked. Highlight the current coordinate. Use newly updated state at subsequent visits. |
| 5 / §2 | Energy after initial state and four coordinate updates: [0,0,−1.5,−1.5,−1.5] | Staircase x axis is individual updates, not sweeps. Explain dimensionless objective. Keep the flat steps and a numeric table. |
| 6 / §2 | Finite scan: P=[2,6,10,16,24], trials=[16,48,80,128,192], fixed=[16,48,61,49,12], recalled=[16,47,56,29,2] | Paired proportions on [0,1] with counts. Caption d=64, eight banks, exactly six flips and sequential recall. No theoretical capacity line or invented uncertainty band. |
| 7 / §3 | Memories ±(1,0), q=(.2,.4), β=2; scores ±.2, logits ±.4, weights [.6899744811,.3100255189], read (.3799489623,0) | Geometric convex hull beside score and weight bars. Show all intermediate values; keep memory/query/read identities distinct. |
| 8 / §3 | Cobweb plots of y=tanh(βx) versus y=x for β=.5 and β=2 | Equal axes x,y=[−1.1,1.1], actual worked trace. Mark zero and stable points near ±.9575 for β=2; do not call ±1 exact fixed points. Stack panels on phones. |
| 9 / §3 | Exact E=.5(x²+y²)−log(exp(βx)+exp(−βx))/β, β=2; worked energy trace | Stable log-sum-exp, plane x=[−1.3,1.3], y=[−.8,.8], at most 61×41 samples, 6–10 contour levels. Correctly mark zero as a saddle. Matching energy strip uses [−.285550333,−.406683911,−.472651043,−.505745739]. |
| 10 / §4 | K=I₂, q=(.6,−.2), β=1, weights [.6899744811,.3100255189]; V=[10,−2] returns 6.2796937735 | Key and value lanes share memory IDs. Clearly label the scalar payload versus two-dimensional key read. No energy iteration claim for that payload. |
| 11 / §4 | Input-query/input-bank association; learned-query/input-bank pooling; input-query/learned-bank lookup | Three diagrams label input-dependent arrays and parameters separately. Explain paired permutation condition. All captions remain visible on phones. |
| 12 / §4 | Gradient fixture q=(.2,−.1), target 2, β=1, learning rate .1; loss .8543552445→.7899801106 | Before/after weights, gradient [.5744425168,−.5744425168], updated q [.1425557483,−.0425557483]. Separate state-refinement arrow from parameter training. |
| 13 / §5 | Official 3,823 training images from 30 writers split into memory200, fit queries800, validation300, unused2523; test1797 from other13 writers | Test container stays outside learning arrows. Counts sum. Do not invent writer IDs for the internal validation split. |
| 14 / §5 | Pixels64 → shared Wₑ of shape 16×64 → unit vectors → weights200 → classes10, with a second branch to pixel values64 | Label 1,024 trainable parameters, 200 fixed images, β=16. Backprop goes through both projection uses; pixel reconstruction is not the training objective. |
| 15 / §5 | Test clean/occluded errors: nearest156/663, fixed136/667, seed17 100/564, seed41 113/740, denominator1797 | Same count axis beginning at zero, range to800. Validation errors25/18/13/16 in separate labeled column. State both seeds and the frozen selection protocol. |
| 16 / §5 | Validation rows2946 and3052: actual clean/damaged images, top three references and weighted image | Original 8×8 arrays, row-major, intensities divided16, columns4/5 zeroed. Same grayscale [0,1]. Labels, source IDs, weights, class mass and MSE come from investigation-checks.json. Original reference is evaluation-only. |
| 17 / §7 | P=100, gap3, β=2, competitor factor99exp(−6), target lower bound .8029571528 and distance bound .3940856945 at M=1 | Denominator mass diagram and algebra, explicitly an assumed-gap bound. Do not manufacture 100 unit vectors purporting to realize the assumed scores. |
| 18 / §7 | Stored bytes=256P for d64 float32; one million patterns need256,000,000 bytes≈244.14MiB; pair count BP and arithmetic proportional BPd | Separate storage and arithmetic quantities. Exact formulas only, no seconds or named implementation speed ranking. |
| 19 / §7 | Clamped parity inputs and candidate outputs: degree2 energies both−12; degree3 E=24abz | Four input pairs and two candidate-output columns; lower energy highlighted with text. Verify all eight assignments using polynomial_parity results. |
| 20 / §8 | Variable-sized bags → shared encoder → learned-query pooling → one bag label | Schematic bags with explicitly illustrative counts. Supervision arrow ends at the bag. No invented biological diagnosis or per-instance truth label. |
| 21 / §8 | Hopular sample-to-sample and feature-to-feature memory reads, alternating refinement and masked query target | Separate axes and labeled memory sources. Schematic with no invented output numbers. Follow primary §3/Figure2; preserve both axes on phones. |

## Shared investigation behavior

Display the current memory state, retrieved output and relevant energy immediately. Valid edits update the same model in every representation; an earlier result is retained only as a labeled baseline. No answer choices or prediction state is stored.

Input edits recompute current outputs, while a pinned baseline preserves its original cue, memories, parameters, visit order and result. Step/back selects actual process states; Reset restores the declared fixture and its current output.

Feedback states the actual changed quantity, numerical difference and mechanism. Use explicit tolerances for comparisons, exact numeric tables and accessible category labels where helpful. No numeric guess, category prediction or correctness score is requested.

Reset restores the fresh default and recomputes current outputs, answers, edits and history. A separate Restore cue preserves model settings but recompute current outputs. Back moves through an already computed trace without implying a new experiment. Do not autoplay.

All numbers below are checked fixtures, **not hardcoded answers for arbitrary inputs**. The implementation computes from the active model. Author results are verification oracles. Keyboard order follows inputs → Run → concise result → details. Announce one concise result through an aria-live region after computation, with no forced focus jump.

### Investigation A — repair a binary pattern

**Question:** Which bit will this memory repair, and can visit order select another attractor when a cue is ambiguous?

**Fresh inputs:** one stored pattern [1, 1, −1, −1], cue [−1, 1, −1, −1], order [1, 2, 3, 4], maximum eight sweeps. The outcome stays hidden .

Every stored-pattern cell and cue cell is an editable toggle with a spoken sign and coordinate. Permit one to three patterns. Four dimensions are the core mode; an optional eight-dimensional mode must expose all coordinates. Derive W = XᵀX/d with zero diagonal. Let users reorder visits with numbered controls rather than dragging alone. Editing a pattern must actually recompute W.

**Live observation:** show the first coordinate changed by the selected update order, current/final vector and its category: stored pattern, inverse pattern or neither. Stepping the update exposes its energy change; editing the cue or visit order recomputes the trajectory. Do not force recall to succeed.

**Display:** signed votes into the active coordinate, local field, before/after state, energy and energy difference, plus the current recall comparison. Explain zero-field ties. A step-limit result says “limit reached,” not “converged.”

**Checked contrasts and nulls:**

- Fresh cue: coordinate 1 changes with field +0.75; energy 0 → −1.5; final state [1, 1, −1, −1].
- Worked cue [1, −1, −1, −1]: coordinate 2 changes instead.
- Ambiguous cue [−1, −1, −1, −1]: forward visits reach the stored pattern; reverse visits reach its inverse. Both finish at −1.5 from initial energy +0.5.
- Stored cue: every update leaves the state unchanged at energy −1.5.
- Optional separate synchronous demonstration: W = [[0, 1], [1, 0]] and [1, −1] alternate in a two-cycle. Clearly label this fixed-weight comparison as a different update regime.

**Transfer:** construct a cue equally close to two opposite attractors and inspect which first visit breaks symmetry. With multiple stored patterns, permit a final “neither” result; do not force successful recall.

**Bounds/access:** at most eight dimensions, three patterns, eight sweeps and 64 coordinate updates. Compute on a valid edit. Announce changed coordinate, sign and field. Provide a labeled numeric matrix and a text trace.

**Phase-two checks:** individual flip energy differences, use of the most recently changed state, visit-order behavior, zero-field ties, the synchronous counterexample, fresh a live computed readout and reset. All primary numeric fixtures are in mechanism-results.json.

### Investigation B — shape a continuous memory landscape

**Question:** Will repeated retrieval move this cue toward an individual memory or toward an average?

**Fresh inputs:** X = [(1, 0), (−1, 0)], q = (−0.35, 0.6), β = 2, twelve requested updates. Memory and query coordinates are genuinely editable by drag or numeric fields, range −2 to 2, step 0.05. Permit one to six memories. β has slider and numeric controls from 0.1 to 8. Stable memory IDs remain attached to edited rows.

**Live observation:** display the resulting region or memory ID, or “mixture,” together with the first and final computed reads and their difference. An optional live comparison asks for the first horizontal coordinate. For arbitrary banks, use the nearest-memory ID and distance of the computed endpoint rather than claiming a proven limit. A display threshold of 0.05 means “close to this memory”; last-step norm below 10⁻⁶ means “little change at the displayed precision.” Neither is a convergence theorem.

**Display:** query and memory geometry, convex hull, selected update point, score/weight distribution, actual energy and step norm. A cobweb view is available for the special opposite-horizontal pair. General banks use the same energy formula for contours. Label the endpoint “12 computed updates,” with its last-step norm.

Changing β immediately recomputes the current retrieval and energy trace from the unchanged initial cue. Compare energy decline within each run; absolute energies for different β values are not a better/worse model ranking.

**Checked contrasts and nulls:**

- Fresh β = 2: horizontal coordinates −0.6043677771, −0.8362997950, −0.9318945491 after updates 1–3. Energies start −0.218958705 and then −0.464434879, −0.503924135, −0.509563995.
- Fresh β = 0.5: horizontal coordinates −0.1732351578, −0.0864016079, −0.0431739486 head toward the middle.
- Symmetric q = (0, 0.6), β = 2: first read (0, 0), all later reads unchanged with weights (0.5, 0.5). The exact zero is fixed even though horizontal perturbations grow.
- q = (−0.2, 0.4), β = 8: first horizontal coordinate −0.9216685544. Sharpening selects the negative side even if the learner intended the positive memory.
- Duplicate the positive memory: [(1, 0), (1, 0), (−1, 0)] at q = 0 gives weights one-third each and read (one-third, 0), instead of zero.
- One-memory null: every query reads the same memory in one update.

**Feedback:** connect the score gap and changed weight allocation to the movement. Explain why β cannot create evidence in an exact tie or reverse a wrong score ordering.

**Phase-two checks:** gradient finite differences; convex-hull property; normalized positive weights; the energy decrease inequality; all contrasts and nulls; contour aspect ratio and correct stationary-point types. Use stable log-sum-exp. Bound computation to six memories, two dimensions, twelve steps and a 61 × 41 contour grid recomputed only on Run.

### Investigation C — route a query to a payload

**Question:** Does editing a key, query or value change the same part of retrieval?

**Fresh inputs:** keys [(1, 0), (0, 1), (−1, 0)], values [(1, 0), (0, 1), (0, 1)], query (−0.3, 0.4), β = 2. Use a key-space plot paired with a value editor. Every key/query coordinate is editable in [−2, 2]; every value coordinate in [−5, 5]. Permit two to six memory rows. Shared row IDs and connector lines preserve the key/value pairing.

Value mode distinguishes one-hot class labels from arbitrary vectors. A generic vector read is not displayed as a class probability.

Display the returned vector and its signed payload contributions beside the query/key weights. Editing key, query or value shows the separate change to addressing and content. No optional predicted-vector input is included.

**Checked contrasts and nulls:**

- Fresh scores [−0.3, 0.4, 0.3], weights [0.1193984673, 0.4841846607, 0.3964168719]. Key read [−0.2770184046, 0.4841846607]; value read [0.1193984673, 0.8806015327].
- Query changed to (0.8, −0.4): weights [0.8837980884, 0.0801763537, 0.0360255580], value read [0.8837980884, 0.1162019116].
- Restore the fresh query and change only value 3 to (1, 0): weights remain unchanged; value read becomes [0.5158153393, 0.4841846607], changing the larger component.
- Set every value to (4, −2): output remains (4, −2) for every query or β, within rounding.
- Reorder paired key/value rows: output remains unchanged.

**Feedback:** identify whether the edit changed matching or payload. In one-hot mode sum class mass across memories. Display a class tie when masses differ by less than 10⁻⁶. Do not attach the fixed-key Hopfield energy-decrease label to an arbitrary value read.

An optional revealed training example can display the query-gradient step from §4, with a changed target for independent transfer. It is not required to expand this investigation into another full training simulator.

**Phase-two checks:** editable arrays against the matrix formula, paired-permutation invariance, equal-value invariance, weights summing to one, output dimensions and live recomputation. At most six rows and two dimensions make this a small CPU calculation.

### Investigation D — retrieve information from real handwriting

**Question:** Which memories support this shape, and does editing a stroke improve classification, reconstruction, both or neither?

**Fresh inputs:** validation-source row 3748 (zero-based validation index 31), original clean pixels, seed-17 learned projection, β = 16 and the frozen 200-image memory bank. Show its computed result as soon as the inputs are valid. Let the learner select any of the 300 validation rows and choose seed 17, seed 41 or fixed pixel geometry at β = 64. Keep test images out of interactive setting selection.

**Meaningful edits:** each 8 × 8 cell can be selected, then edited through an integer intensity field 0–16 and plus/minus controls. Arrow keys move cell focus; Enter selects a cell. Optional brush input has a full numeric equivalent. Buttons for zeroing columns 4 and 5, blanking the grid and restoring the source are conveniences. Every model reads the actual edited pixels.

The label remains “original source label”: an arbitrary edited or drawn image does not acquire independently verified ground truth.

**Live observation:** display the computed class 0–9 or tie and whether the weighted image improves, worsens or preserves pixel MSE relative to the current cue and the original reference. Optional prose describes a stroke change but is not automatically computed. Compute the comparison from the complete current inputs. Advanced exploration may edit fixed-model β on validation inputs, explicitly separated from the already recorded test comparison.

**Output:** show the edited cue, top three weighted memory thumbnails with labels and source IDs, weighted image, ten class-mass bars on [0, 1], and the original reference marked “evaluation only.” Include both input-to-reference and read-to-reference MSE. The class masses and weighted image use all 200 memories. Display how much total mass the three thumbnails account for, with a full weight table available.

**Checked contrasts and nulls, seed 17 at β = 16:**

- Fresh row 3748, original class 1: clean class-1 mass 0.8659867644, predicted class 1, top memory row 1532 with weight 0.4352892935; read MSE 0.050425753.
- Zero columns 4 and 5: predicted class 0, class-1 mass 0.000134387461; top rows 27/3525/1786 with weights 0.249811724/0.143593594/0.133669704. Read MSE 0.113757864 versus input MSE 0.200744629.
- Fresh single-cell edit at row 4, column 5, complement intensity from 1 to 0 in normalized units: predicted class remains 1, class-1 mass 0.7930867076, read MSE 0.0604081824, input MSE 0.015625. Feedback must not invent a class flip.
- Worked row 2946: clean class-1 mass 0.9893615246; occlusion changes prediction to 0 with class-1 mass 0.000141083394. Read MSE 0.110814080 versus input 0.203674316.
- Successful row 3052, class 0: occluded prediction remains 0 with class mass 0.838620484. Read MSE 0.022475775 versus input 0.074035645.
- Restore null: original weights return exactly in the recorded CPU run, maximum error zero.
- Blank null: the bias-free projection and normalization return zero query; every score is zero, memory weights are 1/200, and every class mass is 0.1. The code's argmax returns 0 by tie order, but the UI must say **all classes tied**, not “recognizes 0.”
- Positive common input scaling before unit normalization leaves geometry unchanged up to rounding; a retained author check confirms it. This is a verification case, not a required UI control outside allowed pixel bounds.

**Model contract:** selected P has shape 16 × 64, float32; normalize with epsilon 10⁻¹²; score βqKᵀ; stable log-softmax; sum class mass in log space; aggregate original pixel values with the same weights. For the fixed baseline use normalized 64-dimensional images and β = 64. Preserve the NPZ memory order and class order 0–9. Cache frozen memory keys.

One query needs a small projection, 200 similarity scores and a 64-dimensional weighted image. Compute only on Run. Load only selected model weights and required images; never ship a full Python runtime or train in the browser. Derive compact assets only during phase two, keeping provenance and the full downloadable reproduction separately.

**Phase-two checks:** compare all 300 validation rows' clean and occluded distributions with the stored NPZ at absolute/relative tolerance 10⁻⁵ unless a wider justified tolerance is necessary. Check row-major image orientation, mask columns, intensity scale, memory/class mass sums, no reference leakage into inference, blank ties, edits, invalidation and restore. Inspect the informative failed-retrieval state and successful state at desktop and 360 px, including focused-cell editing and keyboard editing and result inspection.

## Phase-two acceptance and deferred work

After an authorized finish request passes the finish preflight, implement the 21 inline figures and four distinct investigations, preserving the full manuscript and exact data/learning boundary. Use semantic topic-owned components and on-demand assets.

Then independently review correctness and pedagogy; verify current-input behavior, contrast and null cases; execute any changed displayed programs; inspect informative desktop and phone states; check keyboard, focus and text equivalents; measure relevant loading costs; and run applicable integration checks. These activities remain deferred. Record final evidence and remove only disposable implementation artifacts.

The author performed a full manuscript/specification reread and bounded research/data/calculation checks, recorded in design.md. Root owns the content checkpoint. This packet does not claim rendered verification, independent review or user acceptance.


## Scratch/tool bridge presentation — 22 September 2026

Place the source-owner strip beside §5: binary state update, softmax bank read, and learned addressing have different state and cost. In the existing key/value view expose score scale β explicitly; default SDPA scale is 1/sqrt(d). With distinct K/V show equal returned vectors between routes while keeping the fixed-bank energy caption inapplicable. Direct edits update values and explanation immediately.

The complete source and teaching explanation are already written in the manuscript and companion programs. Phase two implements the presentation and verifies actual behavior; it does not invent an omitted algorithm. Show code only when requested, load large code assets on demand, preserve exact source equality, and keep immediately visible numerical explanations usable without running Python in the browser. No learner-prediction entry or grading state is permitted.
