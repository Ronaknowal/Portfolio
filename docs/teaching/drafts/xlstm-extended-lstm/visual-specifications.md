# xLSTM visual and investigation specifications

Prepared 13 September 2026. Content phase only: these are implementation contracts, not implemented components. Read the complete manuscript and design record before building. Stable topic ID `xlstm-extended-lstm`; keep all future assets and models topic-owned. Use the site's existing visual language for framing, but let scalar mass, matrix address geometry, chunk computation and image scanning determine the representations.

## Shared mathematical and interaction contract

Use C with key rows and value columns throughout: shape d_k×d_v; write k vᵀ; read Cᵀq. Hand-fixture queries are already scaled; learned-model queries are divided by √d_k exactly once in MatrixMemory. A signed matrix coefficient is not an attention probability. State, read, output normalization and final class logits are distinct quantities. Dimensionless example values and digit block counts must not acquire invented physical units.

Scalar raw recurrence: c=f c_old+i z; n=f n_old+i; h=o c/n, empty c=n=0. Exponential write i=exp(a); core retention f in(0,1], represented by log f. Stabilized m=max(log f+m_old,a), i'=exp(a−m), f'=exp(log f+m_old−m). Store c',n',m and show the unchanged ratio. In a learned sLSTM, h is also required for the next recurrent gates. No epsilon/prior may be inserted silently.

Exponential matrix recurrence: C=f C_old+i k vᵀ; n=f n_old+i k; raw r=Cᵀq/max(|nᵀq|,1). Stabilized uses C',n',m and floor exp(−m). Output gating and RMS normalization are additional layers in the learned model; show their location. Pure operator fixtures omit those layers deliberately. The later mLSTM_sig variant is a separate formula and must not be selected by relabeling the same checkpoint.

For every investigation, initially unset prediction has no default radio choice or prefilled text. The learner edits genuine inputs, records a prediction in their own words or selects a justified direction plus a reason, and then requests the outcome. Associate it with a canonical serialization of all causally active inputs, selected model/seed, operation mode and experiment condition. Any such edit invalidates the prior prediction/outcome comparison, marks it stale and requires a fresh recorded prediction for new feedback. Do not erase the user's text silently; retain it as a previous attempt, clearly detached from the new run. Focus/step changes that merely inspect the same run need not invalidate it.

Worked presets are clearly marked and already solved. Fresh defaults below are different problems. Reset to fresh clears results, recorded prediction, cached state and history for that investigation. Restore a worked preset may show its worked output but must not count it as fresh practice. Reset execution state alone is a deliberate experiment and distinct from reset the whole interface.

Before execution show inputs, field labels and the question, not the completed result diagram or outcome-revealing class label. After execution show exact selected values, predicted versus observed change, a mechanism-based explanation, and an invitation to revise one input. Do not use confetti, completion claims or correctness feedback based only on choosing a preset.

Keyboard operations must cover all drag/edit functions. Inputs have visible labels, units/ranges and error text. Text/table equivalents expose all quantities conveyed by color or position. Use signed diverging matrix colors with printed values; nonnegative weight bars use a separate sequential scale. Maintain readable contrast and focus. SVGs have titles/descriptions and non-hover access to cells. Motion is step-triggered; reduced-motion mode changes state instantly. No autoplay.

At narrow widths, stack the input and output panes while retaining one spatial matrix/grid at a time. Tables can scroll inside a labeled region when necessary; the whole page must not overflow. Do not shrink 8×8 pixels or matrix text below usability to retain a desktop composition. Announce one concise status on Run/Step, not every cell change. All collapsed explanations, hints and solutions initially closed. Never place an interactive button inside another clickable control.

Bounds and costs are part of teaching: finite inputs only; actual mathematical zeros accepted where valid. Out-of-bound values show a local reason, preserve the previous valid data, and do not silently clamp the story. Use float64 JavaScript arithmetic for small operators, conservative dimensions, and on-demand inference assets. No browser training. Do not load raw source datasets, all research metadata or every model at initial route load. A phase-two implementation may derive compact typed arrays from retained NPZ with hash/provenance and parity checks; it must preserve the selected model exactly.

## Inline figures: required where the mechanism first appears

These twenty figures are anchored in the manuscript; several can share rendering primitives, but not a generic box layout.

| ID / location | Representation and exact content | Caption / accessible alternative |
| --- | --- | --- |
| X01 opening | Eight image strips enter a fixed state and final classifier; adjacent explicit bank has query lines | Fixed state summarizes processed rows; the bank retains separate patterns. Text lists inputs available at each step. |
| X02 §1 | Signed bars +.4 and −.1 join at +.3, then tanh and output gate | Separate cell content from hidden output. Specify output gate before displaying a hidden value. |
| X03 §2 | Three-step evidence ledger: candidates .2,−.6,.8; writes 1,3,9; retention .5; output .75 | Final weights .25,1.5,9; content 6.35, mass 10.75, output .443023. Negative content has positive mass. Include the complete table. |
| X04 §2 | Recurrent within-head mixing matrix feeds per-channel gates; coordinatewise update below | Schematic weights only. Text explains the cross-channel dependency. |
| X05 §3 | Raw/scaled ledgers connected by exp(−m_t), with coincident outputs | Use scalar_worked_raw/stable. At steps two/three, f'=1/6; the scale changes at each time. |
| X06 §3 | Analytic loss curve for θ from −.4 to .4; checked points 0 and .0259259259 | Define the exact own y/L formulas. Prediction .466667→.472403; loss .027222→.025900. This is not a measured fit curve. |
| X07 §4 | Key [1,0] and value [2,−1] feed a product grid | Shape 2×2, entries [[2,−1],[0,0]]. Explain each row/column product. |
| X08 §4 | Three signed matrix heatmaps with matching n and queries | Final C=[[8.5,1.75],[0,1.5]], n=[2.25,.5], q=[1,0], r=[3.777778,.777778]. Full traces in mechanism-results. |
| X09 §4 | Opposing key arrows and signed value contributions | Keys ±e1, values 2/−1, coefficients +1/−1, signed mass zero, numerator three, floor one, read three. Mark nonconvex result. |
| X10 §4 | Raw/scaled numerator and floor rails; incorrect-floor branch | Correct raw/stable 3.694528049465325 versus incorrect .5. Both numerator and floor change scale. |
| X11 §5 | Lower-triangular time matrix and selected influence path | New write does not receive its own forget factor. Future cells are excluded. Include indexed equation and row table. |
| X12 §5 | Old-state rail plus local tile enter one denominator | Actual fixture, full incoming state, one combined normalization. No independently normalized parts. |
| X13 §6 | Full dimensioned block and two residuals; only final row enters loss | 8→16 projection, pre-RMSNorm, cell, residual, post-RMSNorm, 16→32→16 gated branch, residual, ten logits. |
| X14 §7 | Real source 3451 image, row-major 8×8 grid, counts/16 and scan | Counts are 0–16; credit UCI and collectors. Spatial scanning is the model's choice. |
| X15 §7 | Source and derived data roles with decision arrows | 3,823 training rows → 1,000 fit / 300 validation / 2,523 unused; 1,797 separate-writer test rows. Internal writer IDs are absent. |
| X16 §7 | Error-count dots by model/seed, clean/reverse; actual selected loss curve | Denominator 1,797. Use exact JSON counts, selected epoch and correctly timed fit/validation curves. No invented smooth fit or latency curve. |
| X17 §7 | Source 3451 clean/edited scan paths and state summary | First five prefixes identical; scalar classes [1,2,2,1,1,1,1,1] versus [1,2,2,1,1,7,9,9]. Scores are not calibrated confidence. |
| X18 §8 | Original scalar, original matrix and 7B blocks | Show expansion after/before/after the sequence operation respectively; 7B uses RMSNorm/SwiGLU and all matrix blocks. Link dated versions. |
| X19 §8 | Exact persistent-state allocation and schematic length dependence | Matrix 128 MiB; C/n/m 128.2509765625 MiB for stated configuration. A cache comparison needs all entered dimensions/dtype; no measured-memory claim. |
| X20 §9 | Image directions, forecast presence masks and control reward timeline | Show decision-time availability and loss axis. Current action cannot consume its future reward; a forecast cannot consume future observations. |

Figures X03/X05/X08 remain visible inline as worked support; the investigations do not replace them. X11/X12/X18/X19 are in the optional deeper branch. A narrow view may show a single state with previous/next controls plus a small textual trace; preserve values and location context.

## XA — Scalar evidence ledger

### Fresh problem and controls

The learner should distinguish candidate content, write strength, retention, normalization and output exposure. Fresh inputs are saved in `investigation-results.json` under `scalar_independent_inputs`: candidates `[.1,−.8,.5,.3]`, writes `[1,4,2,6]`, retention `[.9,.7,.4,.8]`, output gate `.6`.

Prompt before execution: “If the last write is weakened from 6 to .75 while the earlier observations stay fixed, does the final exposed estimate rise, fall or stay unchanged? Explain which surviving evidence gains influence.” No choice is selected. This problem differs from both solved scalar examples and the practice exercises.

Allow 2–12 editable observations, candidates in [−1,1], write log-weights in [−12,12], retention in [.01,1], and output gate in [0,1]. Raw write weight and log-weight can be synchronized equivalent input controls; they represent one active field. Allow adding, removing and reordering actual observations. Local errors preserve the previous valid data. A requested zero retention is a meaningful limiting case, but is outside this finite-log control; explain that boundary rather than silently substituting .01.

### Representation and prediction state

Use a time-column ledger showing each observation's original candidate and its surviving weight at the selected step. Separate signed content contributions from positive mass. Totals feed the ratio and then the output valve. A signed output trace aligns with the time columns. A raw/stabilized display toggle changes only intermediate representation. Keyboard focus exposes the same numbers as optional hover. Run computes the experiment; Step inspects the already computed trace.

Bind prediction to candidates, write logs, retention, output gate, order, initial-state choice, comparison edit and common log shift. Changing any of these invalidates its feedback. Changing the selected time or raw/stabilized display does not. Reset to fresh restores the four-row default and clears prediction/results/state. Worked resets are labeled solved examples and do not count as independent practice.

### Checked outcomes, nulls and feedback

Fresh final stabilized state: c=.2664, n=1.5173333333333334, m=ln 6, h=.10534270650263619. With last write .75: c=.014624999999999888, n=2.40875, m=.4700036292457356, h=.0036429683445770348. Compare computed numbers to 1e−10. Feedback explains that older negative evidence gains relative influence when the newest positive candidate is written less strongly.

Worked reset one uses §2 candidates .2/−.6/.8, writes 1/3/9, retention .5 and output .75. Outputs are .15, −.36428571428571427, .4430232558139535. Worked reset two uses `mechanism-results.json`'s `scalar_fresh` and `scalar_fresh_weaker_last_write`; despite those historical field names, these are solved manuscript examples.

Null one sets all candidates to .6 and output to .8. All outputs equal .48 for any allowed positive weights/retention. Explain agreement among the evidence, not unused gates. Null two applies a common +1,000 to all write logs of worked reset one from empty state. Stable outputs differ by at most the recorded 1.504352198367087e−14. Do not form exp(1000) in the raw display or plot Infinity. Show symbolic scale information and a local explanation that raw totals cannot be represented here. This advanced control has fixed offsets 0/1000; it is not an unrestricted log-weight input. Null three sets output gate zero: exposed outputs are zero while memory updates continue.

Phase two must verify raw/stable parity, each fresh/worked/null fixture, positive mass, the convex range before output gating, the special large-log mode, add/remove/reorder and reset. Check prediction invalidation, keyboard decimal entry, readable narrow ledger scrolling inside its own region, and table/graphic agreement. No timer or continual render loop is required.

## XB — An address grid, not a probability chart

### Fresh problem and editable entities

Use three writes, each with a two-dimensional key and value, and an already-scaled two-dimensional query:

| Step | Key | Value | Query | Write | Retention |
| --- | --- | --- | --- | ---: | ---: |
| 1 | [1,0] | [1,2] | [1,0] | 1 | .8 |
| 2 | [.5,1] | [−2,1] | [0,1] | 2 | .5 |
| 3 | [−.5,1] | [3,−1] | [.5,1] | 1 | .7 |

Prompt: “Reverse only the last key to [.5,−1]. Will the first output coordinate at step three get closer to zero or farther from zero, and why might its denominator change too?” No answer or outcome appears before the learner records a prediction.

Allow 2–8 editable key/value/query rows, coordinates in [−4,4], write logs in [−4,4] and retention in [.05,1]. Keep key/value dimensions at two for this address plane. A separately labeled one-dimensional worked floor preset uses its own one-dimensional display. Key and query arrows can be dragged within labeled axes, with equivalent numeric and keyboard controls. Value vectors use editable coordinate bars. Do not clip an out-of-range vector silently.

### Representation and state

Link the address plane, outer-product grid, stored C and n, signed contribution bars and read denominator. Show both output coordinates. Output values may exceed stored value-coordinate bounds; adapt the output axis with explicit labels rather than clamp. A selected matrix cell identifies its key row and value column. Raw and stabilized views represent the same operator. The deliberately incorrect floor branch is marked as an incorrect operator, not a legitimate checkpoint variant.

Prediction is bound to all keys, values, queries, gates, order, initial state and chosen comparison edit. Selecting a timestep, matrix cell or raw/scaled view only inspects the attempt. Editing values or invoking “zero query” invalidates it. Reset returns the fresh unsolved table and clears results/prediction. Worked presets remain labeled solved.

### Exact contrasts and nulls

The fresh final raw state is C=[[-2.55,1.9],[.2,.4]], n=[.55,2.4], numerator=[−1.075,1.35], denominator=2.675, read=[−.4018691588785046,.5046728971962616]. Changing the last key yields C=[[.45,.9],[−5.8,2.4]], n=[1.55,.4], numerator=[−5.575,2.85], denominator=1.175, read=[−4.74468085106383,2.425531914893617]. Full traces are in `investigation-results.json`.

Feedback must connect the changed signed association and denominator to the output. These coefficients are not a probability distribution. Accept a reasoned prediction without attempting to grade arbitrary free-text wording by exact string matching.

All values zero gives zero C/read at every step while n remains as in the nonzero-value case. All queries zero gives zero reads and leaves C/n unchanged. An orthogonal key/query gives zero contribution for that write, while other writes may remain. Restoring exact inputs recovers the same trace within 1e−10.

Worked fixtures are the manuscript's three-write matrix, signed cancellation and scaled-floor counterexample in `mechanism-results.json`. The floor example must show raw/stable 3.694528049465325 versus incorrect .5. An active-floor test is mandatory; ordinary large-alignment parity alone is insufficient.

Phase two checks orientation, query scaling once, raw/stable parity at 1e−10, signed cancellation, zero query/value, full reset, honest output axes and keyboard plane editing. Matrix colors have signed labels and a text table; screen-reader cell names include key and value coordinates. No essential operation depends on panning or hover.

## XC — Causal chunks and full state carry

### Purpose and input contract

Distinguish a different execution schedule from a different history. Use the exact seven-token fixture generated in `memory_mechanisms.py`: NumPy default_rng(229), normal q/k arrays of shape 7×3, normal v of shape 7×2, then write logs uniform [−.7,.9] and forget logs uniform [−1.2,−.05], in that draw order. Phase two must save the actual generated arrays as explicit fixture data; JavaScript Math.random does not reproduce NumPy's stream.

Expose the actual q/k/v coordinates and gate values in an editable table. Allow 2–12 tokens, key width three, value width two, manual coordinates [−4,4], write logs [−2,2], retention [.05,1]. An optional initial-state table accepts C and n entries in [−4,4]. Default is empty state, chunk size three, state carry enabled. No precomputed parity badge appears before prediction.

Prompt: “Will changing chunk length from three to two change the outputs if the full incoming state is preserved? Describe which evidence crosses the boundary.” An explicit reset-at-boundary experiment is a separate changed-history contrast.

### Representation

A timeline groups writes into chunks. Boundary capsules show C/n (and m for a stabilized implementation). At a selected output, reveal incoming and local numerator contributions and their combined signed mass. The local triangular tile visibly excludes future entries. Only after old and new terms join does the denominator apply. A difference plot compares selected chunk outputs to the recurrent reference; label absolute numerical error, not classification accuracy.

Step, previous and selection inspect the same run. An optional small dense view shows the full causal coefficient matrix, never probability colors or labels. The timeline can wrap at narrow widths only if chronological order and boundary relationships remain explicit; otherwise use a labeled local scroll region.

### Checked comparisons and nulls

Recorded dense/recurrent maximum error is 2.220446049250313e−15. Chunk sizes 1,2,3,4,7,9 have errors at most 1.7763568394002505e−15. A nonzero initial C/n fixture is drawn immediately after the base arrays using the same RNG stream; chunk sizes 2,3,7,9 have errors at most 2.6645352591003757e−15. Preserve these explicit nonzero arrays rather than checking only empty state.

Changing chunk boundaries while retaining full state is a null: outputs agree within 1e−10. Adding seven to value vectors at indices 4 onward leaves the first four outputs unchanged exactly in the author's float64 calculation. This named future-edit fixture can exceed manual coordinate bounds; show its actual values in a clearly labeled fixture mode with bounds [−10,10], or use a manually edited future value within normal bounds. Never silently clip the recorded fixture.

Resetting state every three tokens preserves the first chunk and changes later outputs. The checked maximum difference is 5.04953803825383. Final carried output is [−1.2892444014209963,.39571805420333844]; final reset output is [−.600058764465178,.2235330173440005]. Exact input arrays, incoming state, full carried/reset/future-edit outputs are saved in `investigation-results.json`. With empty initial C, all values zero gives zero outputs even if normalizer/gates are nonzero. With nonzero incoming C that statement no longer applies; feedback must identify the remaining old information.

The worked normalization trap is old numerator/mass 2/2 plus local 3/−1: correct combined output five versus separately normalized output four. It explains the operation but is not the fresh prediction question.

The authored chunk reference intentionally uses moderate unscaled inputs. At at most 12 steps, stated finite bounds and retention at most one, its float64 arithmetic is bounded. A production stable chunk algorithm requires correct scale alignment and active-floor checks. Agreement between two equally incorrect stable ports is not enough.

### Prediction, reset and verification

Bind prediction to every input vector/gate, initial state, chunk size, reset mode and comparison. Changing a boundary invalidates it because that boundary change is the intervention being predicted, even when outputs should remain equal. Selecting an output or matrix cell does not. Full Reset restores the seven fresh writes, empty state and unset prediction. “Reset state at boundary” changes the experiment and never silently replaces the data.

Phase two checks all chunk sizes including one, nondivisors and sizes at least T; nonzero incoming state; causal masks; exclusion of a write's own forget gate; short final chunks without padded writes; combined denominator; carry/reset and future-prefix nulls. Verify numeric tables, difference plots, keyboard controls, narrow layout and input validation. No timing benchmark loop is needed for a twelve-token investigation.

## XD — A digit read as eight decisions

### Fresh problem, data and model assets

Use the six actual selected fits in `row-sequence-fits.npz`; no browser training. Default is scalar model seed 19 and validation index 142 (zero-based), training-file source ID 187, true label four. Worked validation index 35/source ID 3451/label one belongs to the solved inline example. Both input grids, full logits and selected state traces are in `investigation-results.json`.

Show “validation example, source 187” before Run, not the true label. Prompt: “If rows six through eight are replaced by zeros, will the final class change? Which earlier prefix scores must stay the same?” The learner may edit actual pixels, row order, model/seed and boundary mode before recording a prediction. Labels are metadata for subsequent interpretation and are never model features.

The 64 pixel cells are integers 0–16. Support keyboard row/column navigation and a numeric input for the focused cell; pointer drawing is optional. Natural/reversed row order is sufficient. If explicit permutation editing is added, require each index one through eight exactly once. Named blank/last-three-zero actions edit the real cells. Restore original restores all pixels and natural order and invalidates prediction. Full Reset additionally restores the default model and clears all state/results/prediction.

### Exact model and representation

Port the complete `DigitReader`, including both RMSNorm layers with epsilon 1e−6, both residuals, SiLU projection multiplied by the second projection, classifier and the selected cell. For mLSTM retain query scaling once, gate preactivation softcap 15 and the exp(−m) denominator floor. For sLSTM retain hidden-to-gate recurrence and complete h/c/n/m carry. For ordinary PyTorch LSTM retain its i/f/g/o gate ordering and both bias arrays; do not assume another library uses the same layout.

Show the image with scanned rows, ten class logits or softmax probabilities, a per-row class trajectory, and a selected state view. Label probabilities as model scores rather than calibrated confidence. Only final-row loss trained the model; prefix classifiers are inspection views. Scalar state can show a selected channel's h,c',n',m trace with recurrent mixing context. Matrix state shows its 8×16 C', n and m, with selectable cells and a numeric before/after-normalization read summary. Do not assign invented “stroke detector” meanings to learned coordinates.

A carry mode processes rows one through three and continues rows four through eight using the full state. A separate explicit reset processes the latter part from empty state. Do not describe the cropped latter part as an independently labeled digit. Logits may change without changing argmax, so show both.

### Checked contrasts and nulls

For fresh source 187, seed 19:

| Model | Clean prefix classes | Last three rows zero |
| --- | --- | --- |
| LSTM | [4,4,1,2,2,8,8,4] | [4,4,1,2,2,7,7,7] |
| sLSTM | [4,1,1,2,2,2,8,4] | [4,1,1,2,2,9,9,9] |
| mLSTM | [4,4,2,2,3,3,9,4] | [4,4,2,2,3,4,4,4] |

The matrix model's final class remains four in this fixture while its logits change. Preserve that difference; all models need not tell the same story. Worked source 3451 has scalar final one→nine, ordinary LSTM one→seven, matrix one→four when the last three rows are zeroed. Full reverse cases, logits, probabilities and states are retained.

Editing only rows six through eight leaves the first five prefix logits unchanged. Splitting after row three with complete carry matches full-sequence logits within 1e−5 in author checks. Explicit reset produces the saved `reset_final_logits`; do not invent a universal class change. Restoring pixels/order/model restores outputs. Blank is not a uniform-score null: seed-19 final classes are LSTM seven, sLSTM nine, mLSTM four. Explain learned biases and state transitions without treating blank classification as recognition success.

Reversing rows is not a null because it changes the causal history. A model switch invalidates prediction and cached state. An unchanged input gives deterministic results; Run is not a random resampling button.

### Feedback, performance and phase-two verification

Bind prediction to all pixels, row order, model/seed, boundary/carry mode and the planned edit. Pure inspection of another prefix or state cell does not invalidate it. After Run, compare the predicted change with the observed result, highlight the first changed row, and show causal prefix equality. Reveal the true label then. Where argmax is unchanged despite changed logits, explain that an unchanged decision does not mean identical computation.

Load only the selected small model when the investigation opens; use a bounded cache if switching among models. Derive compact typed arrays from the NPZ during phase two with hashes and attribution. Keep raw source, all curves and author metadata out of initial page load. Eight steps and these small widths permit bounded CPU inference on Run/Step; do not recompute all models on every keystroke. Provide loading/error/retry states without losing edited input. Dispose of listeners and state on unmount.

Verify every layer and all eight prefixes against Python for both seeds and all three cells, including original/edited/reversed/blank/carry/reset cases. Target float32 logit maximum error at most 1e−4 with matching argmax; investigate near ties rather than silently loosening thresholds. Check softmax and state arrays separately. Verify pixel 0/16 round trips, invalid entries, no label leakage, full Reset, keyboard grid editing, readable state tables, narrow class chart, concise screen-reader announcements, reduced motion and lazy imports. The downloadable program bundle must include complete inputs and attribution.

## Deferred publication work

Implement only under an authorized phase-two continuation. These author calculations are truth fixtures, not formal independent review or browser evidence. Phase two extracts compact assets, builds figures/investigations, checks affected displayed programs and numerical ports, performs independent correctness/learning review and browser/accessibility/loading/performance checks, integrates downloads and actual local links, and updates source-bound delivery status. Do not expose author-only expected answers in a fresh investigation's pre-run state. Preserve this pending packet until the later implementation and retention decision are complete.
