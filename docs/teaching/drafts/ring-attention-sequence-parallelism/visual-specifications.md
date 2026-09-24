# Ring Attention: visual and investigation contracts

## Live exploration contract — 21 September 2026

Open each investigation with its current inputs, intermediate mechanism and complete current output visible. Apply valid edits to meaningful entities immediately and update diagrams, tables, units and causal explanation together. No prediction entry, predicted-answer choices, commitment, prediction grading or answer-unlock feature is part of this packet, even optionally. Model predictions and mathematical masks/gates remain subject matter.

Use the topic-specific controls and checked fixtures below. Pair sliders or direct manipulation with labeled keyboard/numeric controls; keep presets as starting points, not the only editable values. A pinned baseline preserves its inputs, seed, units and outputs while the current case changes. Explain both a meaningful contrast and an unchanged/null result, then connect the observed effect to a practical design decision. Reset restores the stated fixture and current result. Invalid text has a local explanation and a clearly identified last valid result; never silently clamp or pair new inputs with old output.

Step/Back and bounded Run controls advance a real computation or reveal its chronological stages, not permission to view an answer. Show the current state and its result throughout. Keep exact small calculations live. For costly frozen inference, debounce or run bounded work with pending/current state labels and stale-result cancellation; inspect saved measurements without implying fresh training. Respect reduced motion, keep focus stable and avoid announcing every animation frame. Independent written practice and its hints/solutions stay separate.

Phase two must test default results without any action, meaningful edits, quick consecutive edits, valid extremes, null/invalid cases, reset, linked-view agreement, keyboard operation and readable phone layouts. The mathematical/reference checks already specified below remain; these live browser checks have not been performed in this content-only revision.

### Topic-specific live route
**Move blocks without changing the attention result.** Edit tiny Q/K/V and block ownership, merger order, causal positions, communication budgets and supported trajectory points.
**See the consequence.** Show stable summary accumulation, legal work grid, circulating ownership, payload timeline and current output/error against the dense reference.
**Decision connection.** Separate mathematical equivalence from communication cost and identify invalid identity, masking or state-carry changes.


Stable ID: `ring-attention-sequence-parallelism`. Content-only revision 1. These are specifications, not implemented browser labs or GPU benchmarks. Consume the complete manuscript, reference programs, model/data and JSON evidence together. The eventual implementation still requires independent review, rendering, runtime and accessibility verification.

## Shared behavior

Choose a representation for each question: token strips and owned buffers for placement; contribution and normalizer bars for algebra; causal grids for semantics; device lanes for scheduling; token/head cells for resharding; movement paths and actual tensors for the real example. Consistent labels, units and controls should support these different forms.

Every investigation opens with its current inputs and output visible. Valid edits recompute both dense-reference and ring results and their difference where applicable. Step advances block communication or summary merging; Reset restores the fixture. No prediction or reveal state is used.

Use explicit Run, Step, Previous and Reset controls. Reset restores the stated inputs and immediately displays their computed result. Keep original and edited states distinct. Async jobs carry the exact revision; discard older responses. Debounce dragging or require Run to avoid queued calculations. Do not train models or fetch external datasets in the browser.

Provide keyboard fields for every drag action, visible units, errors and focus order. Pair color with rank/head/document labels or line patterns. Touch targets should be at least 44px. Reduced motion removes travel animation without hiding states. No hover-only numbers. On narrow screens, stack lanes/planes and offer row-oriented tables. Caption every figure as constructed arithmetic, frozen real-model evidence or a hypothetical cost model, as applicable.

## 1. Position strip, memory inventory and circulating KV

**Placement:** sections 1–2. A stepped explanation; it need not be a computed lab.

Start with L = 8, P = 4 and two positions per owner. Q cards stay in rank lanes. K and V travel as one packet labeled with original owner, global positions and document ID. Highlight visited owners and the current output-summary card. At round r, the visiting KV owner is (rank − r) mod P; reverse circulation uses (rank + r) mod P. There are P computations and P − 1 necessary transfers in this forward schedule. Do not draw a final unused transfer without accounting for it.

Memory bars separate Q/K/V/O, model state and a hypothetical dense score array. Dividing positions does not divide model state unless a separate model-state sharding operation explicitly does so. The million-position MHA example totals 65,536,000,000 bytes for Q/K/V/O at B = 1, Hq·d = 8192 and two-byte elements: about 61.03515625 GiB. Distinguish decimal GB from binary GiB. This is an inventory, not measured peak memory or a universal 80GB capacity limit.

The score grid should represent computation, with only the current stored tile filled. Never allocate a million-square grid. The CPU reference holds full arrays; label its distinct memory behavior.

An optional ownership editor supports L from 2–32 and P from 1–min(8,L). Each position has exactly one owner and every owner is nonempty. A rank-name change preserves numerical identity. Validate missing, duplicate and noninteger IDs. Check the uneven L = 7, P = 3 case, P = 1 without transfer arrows, both directions, each KV owner visited exactly once and logical output reassembly. Mobile alternatives list each owner's positions and the round schedule.

## 2. Stable summary merger

**Placement:** section 3. Use score/value records, signed numerator contributions, denominator bars and a visible maximum-reference line.

The worked records have scores [0, ln2, ln4, 0], values [2,8,1,−2], and blocks [0,1] then [2,3]. These are dimensionless constructed values. Allow editing four finite scores in [−20,20], four values in [−10,10], the block partition and arrival order. A separate common-offset control in [−1000,1000] acts on every score before stable computation. Do not compute exp(1000) simply to animate an overflow.

For visited records, compute m = max(scores), ℓ = Σexp(s−m), u = Σexp(s−m)v. When m changes, rescale both old ℓ and old u before adding the new block. Initially m = −∞, ℓ = 0 and u = 0. A completely masked block contributes nothing. An entirely empty query has undefined attention probabilities; show the chosen zero-output convention and invalid-row flag. Use explicit infinity flags rather than JSON NaN. Ordinary live comparison tasks require at least one valid record.

The saved scalar trace gives:

- First block: m = ln2, ℓ = 1.5, u = 9, output 6.
- Rescaled old summary: m = ln4, ℓ = .75, u = 4.5.
- Final summary: m = ln4, ℓ = 2, u = 5, output 2.5.

Named bug comparisons are equal block averaging (3.2) and omitted rescaling (9.5/2.75). Show these only as explained failures. Reversed arrival and a common offset of 1000 are computed null controls.

Since the first answer appears in the prose, computed mode starts with the fresh scores [ln3,0,ln2,0] and values [4,−1,7,2], or a learner-created edit. Display 27/7 immediately for the matching fixture, with its weighted contributions. Calculate and explain a numeric output with absolute/relative tolerance 1e−6, against the current computation. A separate categorical task asks whether an edit changes the output. in the current live view, explain the signed contributions and actual denominator.

The optional decoding view merges independently generated (m,ℓ,u) summaries at their common maximum. Generate summaries from records by default; do not imply arbitrary inconsistent triples are valid probability summaries. Check zero/negative values, block masking, offset/reversal nulls, edit invalidation, reset, keyboard input and stable arithmetic.

## 3. Global identity and causal-work grids

**Placement:** sections 4–5. Synchronize a logical-position grid with the same cells arranged by storage owner.

Axes label global query and key positions. Local indices are additional metadata. Distinguish allowed cells, future masking, cross-document masking and padding. A selected cell shows the exact logical predicate: same document, key position ≤ query position and valid records.

Start with L = 16 and P = 4. Use the exact owners and counts in systems-results.json:

- Contiguous totals: [10,26,42,58]. The owner-pair matrix is [[10,0,0,0],[16,10,0,0],[16,16,10,0],[16,16,16,10]].
- Striped totals: [28,32,36,40]. Diagonal/lower entries are 10; upper entries are 6.
- Zigzag totals: [34,34,34,34]. Diagonal entries are 10; off-diagonal entries are 8.

A tile is charged for all its cells whenever at least one cell is valid. Sum the maximum rank work in each synchronized round. For tile sizes 1,2,4, critical counts are contiguous 58/60/64, striped 40/48/64 and zigzag 34/36/64. This is the stated toy scheduler, not a claim about a production kernel's handling of diagonal triangles.

The editor swaps actual labeled position records between owners, preserving equal counts. Recompute from individual positions. Support L = 4–32 and P = 1–8, with P ≤ L. Equal-layout formulas require L divisible by P; this zigzag preset requires L divisible by 2P. Disable an unsupported preset with a precise explanation. Uneven ownership remains available in the identity explanation, but must not use the equal-c formulas.

Start an assessment with L = 12, P = 3 or an unsolved user swap. show whether arrangement has lower summed round-critical work, allowing a tie. Calculate and explain computed counts, not an assumption that striping always wins. Show both useful-pair total and executed-cell count. At coarse 4×4 granularity in the L16/P4 example, all three arrangements reach 64: retain this null.

A second mode uses the seven-position asymmetric fixture, with document IDs [0,0,0,1,1,1,1]. Removing document membership changes the recorded output by a maximum 1.7043724677. Advanced tables expose the actual Q/K/V inputs. Editing a document ID changes the mask and recomputes attention; the quoted maximum applies only to its saved fixture. Renaming owners or reversing circulation preserves the function.

Use separate explanatory insets for:

- Rotary positions: unrotated vectors [1,0], frequency 1, global q = 5 and k = 1 give unscaled dot cos4; resetting q to local 0 gives cos1. If displaying an attention score, explicitly apply the additional √2 scaling.
- Target boundaries: input [10,…,15], correct targets [11,12,13,14,15,ignore]; shifting after a split loses target 13.
- Loss weighting: valid-token counts 3/1 and mean losses 2/6 give global mean 3, not 4. The changed practice uses counts 2/6 and means 1/3, giving 2.5.

Check arbitrary swaps, ownership conservation, tile boundaries, global masks, packed and empty rows, target alignment, weighted loss, computed result checks, reset and accessible tables.

## 4. Communication lanes and capacity calculator

**Placement:** sections 6–7. Two lanes show compute and transfers with dependency arrows.

The initial local block is available at time 0. At rounds 0 through P−2, post a transfer while processing the current block. The next compute requires both the preceding compute and its receive to finish. The final round has no unused next transfer. Provide a text table of start/end times and exposed waits.

All timing is explicitly hypothetical. Defaults: P = 4, Hq = 8, Hkv = 2, d = 64, B = 1, two bytes per stored element, effective compute 100 TFLOP/s, effective one-direction bandwidth 50 GB/s and latency 2µs. Validate positive finite throughput/bandwidth, nonnegative latency and integer dimensions with Hq divisible by Hkv.

Use payload = 2BcHkvds and two-GEMM work = 4Bc²Hqd. Compute C = work/F and D = latency + payload/R. Serial total is PC + (P−1)D; ideal overlap is C + (P−1)max(C,D). Softmax, projections and other operations are excluded. These lanes must not look like a captured profiler trace or a hardware benchmark.

Before Run, show whether compute or transfer limits a round, allowing equality within relative tolerance 1e−6. A fresh abstract timeline task uses C = 4µs, D = 7µs and P = 4, producing serial 37µs and ideal overlap 25µs. Keep direct C/D mode visibly separate from dimensional mode. P = 1 removes transfers; increasing bandwidth retains the latency floor.

Scaling controls explicitly distinguish fixed global L from fixed local c. The basic equal-chunk calculator requires L divisible by P. With global L = 4096, P = 4 gives 85.89934592µs, while P = 16 gives 70.66377728µs. With fixed c = 1024, display the growing global sequence and step cost. The x axis is P, y axis microseconds; label every line as calculated. Do not allocate tensors to draw analytic large-L results.

The memory inventory lists Q, current KV, next KV, float32 numerator, two float32 row statistics, distinct O and one at-most-128×128 float32 score tile per head. At default c = 1024, total listed bytes are 5,832,704. Show bytes and MiB; explain aliases and excluded training/model storage. Do not silently multiply this into a whole-training estimate.

Bound dimensions, for example L ≤ 2^24, heads ≤ 256, d ≤ 1024 and B ≤ 64. Detect unsafe integer arithmetic or use a precise representation; reject nonfinite inputs. Limit drawn ranks and plotted samples to 16. Check dimensional units, C/D, timelines, P1, aliases, safe ranges, mobile composition and the persistent hypothetical label.

## 5. Ulysses tensor ownership puzzle

**Placement:** section 8. Track persistent token/head cell identities through resharding.

Use L = 8, H = 4 and P = 2. Label each cell token×4+head. Sequence-owned boards are 4×4; head-owned boards are 8×2. Each rank has 16 elements per channel before and after, sending eight nonlocal cells for this tensor. Invert the operation and verify all 32 cell identities, not just shape. Keep Q/K/V/O tensor names separate.

An optional task deliberately swaps two transmitted records and asks which restored positions differ. Compute the mismatches. Correct forward/inverse and rank renaming are nulls. Missing/duplicate records are errors before attention is attempted.

An advanced inset has Hq = 8, Hkv = 2, P = 4. It shows why four nonempty disjoint KV-head partitions do not exist. Replication, a hybrid partition or smaller P are conceptual alternatives, not implemented toggles. Do not claim MQA is incompatible with every conceivable implementation.

A separate DP2×TP2×PP2×CP2 mesh has 16 labeled coordinate tuples. Highlight the group obtained by varying one coordinate while keeping the others fixed. Ordinary replicated DP and FSDP model-state storage are different annotations. Check exact reshards for the default and fresh L12/H6/P3 case, network-hop versus application-payload terminology, keyboard interactions and the GQA constraint explanation.

## 6. Gradient return map

**Placement:** section 9. A signed-contribution diagram with an optional derivative investigation.

Select a key. Highlight every query owner with an allowed contribution to its dK/dV, then show the partial sums arriving at the key's owner. dQ stays with its query owner. Derivatives can be negative: attention weights alone are not the backward contributions. Keep forward and backward transport counts distinct.

For the scalar worked input and upstream gradient 1, dV = [1,2,4,1]/8 and dS = [−.0625,1.375,−.75,−.5625]. Edit an actual value coordinate or upstream gradient in [−3,3], inspect a selected derivative's sign, then compute it. Use a fresh input for result checks once the worked answer is visible. Zero upstream gives zero gradients; common score shifts preserve derivatives. Editing a value can change other score derivatives through the row average.

The advanced fixture uses the saved rng91 Q/K/V and upstream arrays, two heads, seven positions, three channels and three uneven owners. Phase two can extract those actual arrays from the deterministic program. Compare complete blockwise and dense derivatives against the named bug that drops remote contributions. The recorded maximum dK discrepancy is .8648028024; do not assign that maximum to an arbitrary selected coordinate.

Changed Q/K/V requires a new forward pass and log-normalizer before backward. Empty rows follow the stated zero-output/zero-gradient convention. The reference uses a central CPU reduction; the diagram does not implement network transport. Check selected coordinates with finite differences and include cross-owner valid pairs. If this advanced investigation is too costly, preserve a stepped exact figure and offline practice rather than substitute fake updates.

## 7. Real movement-attention investigation

**Placement:** section 10. Link an editable 45-point trajectory to actual queries, owner buffers, attention contributions and output comparisons.

Start with source 77 (actual class 4, predicted 5). The fresh source is 20 (actual 1, predicted 2). Preserve both classification errors in the current live view. Classification accuracy is not a Calculate and explain of execution equivalence.

Inputs are normalized hand-centroid coordinates in [0,1]. Edit x/y by dragging or fields; display frame indices 1–45 while arrays use 0–44. Controls choose head 1–2, query 1–45, owner count 1–8, direction, ownership, point edit and Run. Show the original trajectory as a ghost path. Selecting a query/head changes only the view; editing a point recomputes the model. Do not replace a learner's edited path with a precomputed scenario.

Exact model contract:

1. Features = tanh((2·points−1)Wstemᵀ+b), width 24.
2. Bias-free Q/K/V projections, reshaped to 2×45×12.
3. Bidirectional scores QKᵀ/√12 and row softmax.
4. Weighted V, merged heads, bias-free output projection.
5. Residual addition, mean over 45 positions, linear 24→15 classifier and softmax.

All 2,751 frozen parameters are in movement-attention-model.json. Do not add LayerNorm, dropout, position encoding, a causal flag or class-label inputs. Such changes would define a different model. Stored values originate from float32 parameters; phase two must document its compute dtype.

Display two separate live comparisons: whether the edited model output differs from the original, and whether dense/ring equivalence holds for the same current input. A model output can change while both implementations still agree. Explain those quantities separately, with exact numerical discrepancies.

Worked edits:

- Source77, frame23 x + .10: maximum probability change .009604586289.
- Source20, frame10 x − .15: maximum probability change .003941857836.

These numbers are fixtures, not answers for arbitrary edits. Require a new unsolved edit after the demonstration. Reversing circulation, changing ownership or renaming ranks preserves the mathematical function. A point edit may have a very small effect; show that result honestly.

The per-query ring view uses current Q/K/V. For each received block, show actual key IDs, contributions and m/ℓ/u. The saved trace records owner IDs, not all intermediate summaries: compute the latter from retained arrays using the reference. A partial normalization must say “among visited keys,” not pretend to be final global attention. Final global weights require all blocks.

partition-results.json is author evidence, not an eager browser import. Extract only complete frozen weights, the two 45×2 inputs and compact fixtures. Compute at most two 45×45 score grids for this model. Load the model on demand; use a worker if measurement justifies it, with revision-bound results. The full 360-row dataset is retained offline but need not ship to the client. On asset failure provide retry; never invent a fallback model.

Recorded NumPy float64 dense/ring errors are <2.23e−15. The inherited float32 probability comparison has expected rounding differences around 1e−7. Phase two checks complete outputs/probabilities, controlled point edits, both directions, uneven P3/P4, owner permutations, actual controls, labels, reset and stale async results. A canvas needs point and probability tables. This bidirectional model has no position encoding; separate fixtures test causal and rotary semantics.

## Finish boundary

Seven representation families do not require seven full widgets. Keep a figure static or stepped when that teaches the mechanism; add a live investigation only when a live comparison and input change improve understanding. Preserve all written practice with closed hints and solutions.

Independent content/correctness/learning review, browser implementation, lazy model packaging, parity, mobile/keyboard/accessibility, performance and failure recovery remain deferred. GPU integration and benchmarks require their own authorized execution and evidence. The implementer may improve representations with a recorded reason while preserving the teaching purpose and numerical contract.


## Scratch/tool bridge presentation — 22 September 2026

Add a distinct real-process code panel to §11, not an extra generic lab. Reuse the ring ownership visual with incoming/outgoing packet lifetimes and valid length versus padding. Forward takes P−1 rotations; backward takes P and exposes returned dK/dV. Browser animation is a model of this protocol, never represented as running actual processes. Full SDPA oracle storage belongs to validation only.

The complete source and teaching explanation are already written in the manuscript and companion programs. Phase two implements the presentation and verifies actual behavior; it does not invent an omitted algorithm. Show code only when requested, load large code assets on demand, preserve exact source equality, and keep immediately visible numerical explanations usable without running Python in the browser. No learner-prediction entry or grading state is permitted.
