# State Space Models: S4 and the Mamba Family — implementation

26 September 2026. Topic `state-space-models-s4-mamba-mamba-2`, DL position 18.
This finishes the complete revision-3 preparation; root owns the shared ledger,
blueprint registration, generated catalogue, final build and integration status.
Author implementation and native checks are complete. Independent and browser
closure are in progress; this record does not yet claim full readiness.

## Content and teaching conservation

The complete 736-line manuscript and full visual specifications were consumed.
`scripts/generate-state-space-lesson.mjs` uses the existing authoring-only renderer
to preserve the explanations, all changed practice and solutions, references,
local sequence/prerequisite links, and complete scratch/library source routes.
No Markdown parser or draft directory is shipped in the browser. The revised
display title names the already-prepared Mamba-3 extension while preserving ID,
progress and the Long-Context → State Space → RWKV module sequence.

The prepared packet's residual “predict before running” instructions conflict
with its later live-exploration contract. The publication generator replaces them
with immediate exploration, and removes the proposed green equality indicator.
The original manuscript checkpoint is preserved. Actual model predictions and
independent written practice remain valid subject matter.

All specified concepts have representations: write/read/feedthrough paths;
continuous sampling and the singular integrator; impulse ledger, kernel, state
and FFT padding; timescales, oscillator and conjugate read; polynomial memory and
normal-plus-rank-one structure; fixed delay, selective contributions, full Mamba
branches and affine composition; signed SSD coefficients and chunk carry; real
ordered paths, training pipeline, measured learning/confusion evidence; retained
array counts; and endpoint/rotation/rank mechanisms for Mamba-3. Related visual
entries are combined where one adjacent representation explains their connection.

Four independent investigations open on demand, display current results
immediately and unmount when closed. They edit actual bounded entities:

- Tiny system: sample values/count, continuous rates, B/C/D, Δ and initial state;
  singular, direct-only and zero fixtures; independent radix-2 FFT comparison.
- Selective memory: event values/count, markers, per-event gates, constant gate,
  initial state and separate retention. Signed retained/incoming terms explain
  errors and the difference between no writing and no forgetting.
- SSD: every a/b/c/v entry, length, chunk size and initial 2×2 state; content,
  decay and influence matrices, local/incoming/total chunk outputs.
- Fitted trajectory: validation record, retained diagonal/selective model,
  every coordinate by numeric/range/drag controls, reflection and reversal,
  selected response channel and all 15 class probabilities.

There is no entry/reveal gate. The full program disclosures load source only
when opened. The three programs remain offline downloadable artifacts; the
Mamba CUDA bridge is clearly labeled as requiring appropriate hardware.

## Numerical evidence

`scripts/prepare-state-space-assets.py` exports original programs/data and two
retained seed-17 models without retraining. Public `asset-provenance.json` records
SHA256 identities. The raw Libras sources retain Dias/Peres/Bíscaro attribution,
CC BY 4.0 and the exact row-level split/provenance record.

`scripts/verify-state-space-native.py` freshly executes both CPU classifiers on
all 50 validation records and four coordinate/order/extreme interventions, plus
singular and nonzero-initial-state FFT cases. The 108 inference cases and source
hashes are retained in `evidence/state-space-native.json`. Existing four training
runs are reused as measurements; no new training or GPU experiment is claimed.

`scripts/verify-state-space-models.mjs` verifies hand-derived linear, selective
and SSD fixtures; 64 linear/FFT configurations including near-zero/zero rates;
all 64 supported length/chunk combinations with zero decays and nonzero carry;
and all 108 independent PyTorch inference cases. Browser-model versus PyTorch
maximum error is 7.14e−6 logits and 6.96e−7 probabilities, inside the specified
1e−4 / 1e−5 tolerances. The JS model uses float64 arithmetic and a bounded-error
erf approximation for exact-form GELU; parity is measured rather than assumed.
`evidence/state-space-models.json` binds the final checked source and precise
errors. Its local Node timings are not browser or GPU throughput benchmarks.

The browser fitted lab deliberately uses recurrent evaluation. The complete
native program retains an independent full-model FFT route and its measured
parity. No duplicate serial evaluation is labeled as FFT. The tiny-system lab
does implement a real FFT and reconciles all three evaluations interactively.

## Learning-experience author review

The first-pass path runs from a hand-movement problem through state mechanisms,
selection and chunking to real fitted data. Dense HiPPO/S4/S5/Mamba-3 material is
explicitly optional depth. A dedicated Transformer lesson is not assumed:
dimensions, state updates, outer products and the unnormalized attention analogy
are explained locally. Fitted-data results preserve the stronger logistic
baseline and protocol limits rather than manufacturing an architectural win.
Written practice changes inputs and includes independent implementation changes,
data-split decisions and resource accounting. Complete local programs implement
the owned mechanism, with explicit earlier ODE/linear-algebra reuse and a normal
Mamba package route. No unmeasured optimality or speed claims were added.

The current official Mamba repository and Mamba-3 arXiv abstract were checked on
26 September. The prepared section-level source research remains recorded in the
original design. The optional fused CUDA value/gradient/full-block bridge remains
source-checked, not executed on unavailable GPU hardware.

## Independent review and browser evidence

The Long Context implementation agent independently reviewed the complete
explicit manuscript transformation list, source model, figures, labs, styles and
verifiers against the prepared mechanisms. The grouped SSD matrix/chunk workshop
covers the planned block decomposition rather than omitting it. Supplemental
independent probes compared FFT/direct convolution for lengths 1–11 (maximum
difference 5.55e−16), split-stream continuation with zero rates/nonzero initial
state (difference 0), and closed writes retaining state 4. No unresolved
scientific or manuscript-conservation finding remained in that bounded review.

The review caught a genuine interaction problem: dragging anywhere on the whole
trajectory could teleport the selected point. Dragging now begins only on a
44-viewBox-unit selected-point ring, uses the inverse screen transform, retains
the grab offset and cleans up pointer cancellation/capture loss. On the actual
1280×720 production preview, an off-centre grab followed by a (+27, −22) pixel
gesture moved the painted handle by exactly (+27, −22). The selected coordinates
changed from (.62476, .46065) to (.724837, .542194); class probabilities changed
immediately. Numeric controls supply an equivalent keyboard route.

Desktop browser checks on 26 September used the production preview on port 4196
after the development server failed independently of the lesson. All 15 top-level
visual groups were inspected, including state paths, sampling, contribution
matrix, timescales, oscillator, polynomial basis, DPLR, shift register, Mamba
branches, SSD, actual trajectories, training pipeline/curves, cache counts and
Mamba-3. Axis labels, signed values, literal units and neutral/amber controls were
readable. Review also improved the Mamba diagram to explicitly join its feature
and gate branches before projection/residual, replaced the generic matrix-axis
caption and formatted an exposed floating-point tail.

All four labs opened with computed results. Typed and keyboard slider edits
changed system outputs; the singular-integrator preset reconciled [4, 3.5] with
initial response 3. Closing writes retained 4 in the coupled rule, while independent
retention .5 yielded [2, 1, .5, .25]. SSD chunk size 8 preserved the four-step
answer, and zero decay removed incoming memory while preserving the current
write. Both fitted models loaded; reversing the selective-model trajectory moved
its top class from 10 to 14 and source-label probability from .1897047 to .1446197.
The endpoint control at λ=1 gave total 6.5; the measured confusion view switched
between counts and recall. The complete mechanisms program loaded on disclosure.

At 320×844 the four labs were also exercised: system input edits, separated
retention, SSD chunk-size changes and a fitted-path drag. An off-centre mobile
drag moved its handle by exactly (+20, −20) pixels. All 15 top-level figure groups
fit their article bounds and all painted SVG text stayed inside its own SVG;
document scroll width equalled the 305px client width. Mobile screenshots covered
the live charts, local matrix scrollers, Mamba branches, real paths, oscillator and
measured curves. The invalid λ=2 state exposed an accessible validation message
and retained the previous valid output; returning to .5 restored normal editing.
All three complete program readers fetched their expected full sources. A console
inspection found no warnings or errors during these checks.

The 760px check revealed an important difference between viewport and article
width: the retained sidebar leaves a 389px article. Topic-owned container queries
now stack figures and controls when the actual article is below 600px, so a
three-path comparison cannot shrink into three tiny panels. Numerical matrix
minimum widths are also sized for numbers rather than generic prose columns;
larger matrices retain keyboard-accessible local scroll. The final production
rebuild was checked at both widths: at 760px all three real-data plots render
360px wide in one column, controls stack, and the 4×4 influence matrix fits the
342px lab interior. At 320px its 326px numeric table scrolls inside a 239px region;
two ArrowRight presses moved that region by 55.33px without moving the page.
All top-level figures fit, with document/client widths both 305px. The temporary
viewport override was reset. No remaining topic-specific findings are open.

The native and JS checks passed after final source changes. The JS receipt also
checks all four 100-epoch learning curves and assessment confusion matrices
against the retained original results, and binds the measurement export and
authoring generator. Final shared ledger/navigation integration belongs to the
parent task; this record does not claim a GPU run or user acceptance.
