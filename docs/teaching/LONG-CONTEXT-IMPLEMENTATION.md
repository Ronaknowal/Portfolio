# Long-context sequence models — implementation record

Date: 26 September 2026. Stable ID: `long-context-sequence-models-transformer-xl-griffin-perceiver`. Continues content-first revision 3, DL module position 17, between Bahdanau/Luong Attention and State Space Models. The prepared title and reading order remain unchanged. Root owns publication, shared registry, delivery ledger and final integration.

## Current state

Author implementation, independent correctness/learning-experience review and rendered browser checks are complete, with no unresolved scoped findings. Root owns final shared integration and phase-ledger completion. The exact prepared manuscript, design, specs, programs and results remain unchanged. Its CRLF-only checkpoint drift was reconciled to the recorded LF bytes by the integration owner before finishing began.

## Source and teaching coverage

- Full prepared manuscript rendered to `src/learn/data/topics/long-context-sequence-models-transformer-xl-griffin-perceiver.jsx`, using the author-only `scripts/render-long-context-lesson.mjs`. The renderer preserves all explanations, formulas, worked examples, measured results, eight changed exercises, hints/solutions, annotated sources and the implementation-depth branch.
- `LongContextFigures.jsx` supplies 16 inline figures. The 18 prepared visual contracts are covered: the cache-tail comparison is the opening live cache strip, and the latent mixture/collision figure is the initial live latent read. Every other contract has an inline component, including query/output shapes, IO queries, the two causal masks, value-versus-gradient paths and associative affine composition.
- `LongContextLabs.jsx` owns the retained-record cache, signed retention/injection and paired-array latent investigations. Learners edit entities, sizes, gates and masks directly; presets supplement those controls. Results are visible immediately and no learner prediction entry/reveal gate exists.
- `LongContextTrajectoryLab.jsx` owns the real Libras path, coordinate editors and drag, masks, record/position changes, second-read weights, original/current outputs and mean-coordinate baseline. The workbench offers all 50 predeclared validation rows and all four original frozen models. It deliberately omits a test-set browser: the optional assessment-exposure state is unnecessary because no test trajectories are exposed interactively. Published aggregate test measurements remain intact.
- `long-context-models.js` owns stable masked softmax, exact scalar cache/recurrence/latent operators and the bounded two-round width-24 frozen classifier. LayerNorm epsilon, projection biases, residual ordering and erf-form GELU follow the retained Torch program. A bounded numerical erfc approximation implements erf; its total forward error is checked against actual saved/native logits, not relabeled as bit-exact Torch arithmetic.
- `long-context-labs.css` scopes black/charcoal/amber surfaces and controls, wrapping stages and labels, paired data views and independently scrollable exact tables. Theme overrides do not target KaTeX geometry.
- Complete canonical Python programs and data are available under `public/learn-code/<stable-id>/`. Programs load into keyboard-scrollable source panels only when opened. `trajectory-models.json` (820,823 bytes before transport compression) loads only on opening the learned workbench. No Python/Torch runtime or browser training is shipped.

The long-context comparison teaches concrete mechanisms rather than full pretrained replicas: segment cache versus complete Transformer-XL; RG-LRU versus complete Griffin; the small Perceiver-style classifier versus published large systems. Scratch and normal PyTorch paths use the same operators/masks/state and expose meaningful modifications. Later Transformer lessons are linked as deeper owners, with Q/K/V, causal legality and shapes explained locally because this topic precedes them.

## Scoped content repairs

The frozen drafts remain a conserved input. The production renderer records these changes explicitly:

1. Replaces figure/lab instructions with the actual representations, thereby removing stale learner-guess wording in two investigation paragraphs.
2. Repairs the fragment “outputs ting the retained baseline” to “outputs against the retained baseline.”
3. Changes a relative link into another prepared manuscript to the stable Self-Attention curriculum route, explaining that it is a follow-on owner rather than a prerequisite assumed already read.
4. Records that the ordinary Torch operator comparisons are now executed. The optional `recurrentgemma --griffin` path remains explicitly unexecuted; no large checkpoint/GPU result is invented.
5. Parses asterisk lists as real lists and separates inline details/summary markup so practice hints and solutions render correctly.
6. Replaces only the obsolete phase-two to-do at the end of the public provenance copy with the implemented frozen-validation-workbench scope. Scientific provenance and the original research record remain intact.

No title, curriculum membership, other manuscript, training setting or measured result was changed. Source research remains the inspected primary-paper/API/data locators in the prepared design. This finish does not claim a new watch of the linked conference recording.

## Actual verification

`node scripts/verify-long-context-models.mjs`:

- Hand-derived cache endpoints 0, 10/3, 5; every length 1–16, segment 1–8, memories 0/1/4/16 and query position; strict legal sets, future-edit causality, equal-value nulls and empty-mask rejection.
- Ordinary/near-hold/hold-limit RG-LRU fixtures, input-only gate null, signed inputs/initial states, all retained contributions summing to the current state.
- Latent exact fractions, paired-permutation null, uniform collision and affine associativity within floating-point tolerance.
- Four original frozen models × all 50 validation trajectories: maximum saved-native logit difference **5.562e−6**, below 5e−5 absolute tolerance; paired permutation and five masked padding rows differ by at most **1.03e−14** in the browser-model arithmetic.
- Row 7 N4/seed29 class 1, first-point-only class 10, small reflection still class 1 with changed probability; all-invalid and nonfinite input rejection; canonical source/data byte conservation.
- One local Node run measured a 2.56 ms median and 13.34 ms maximum. This is author-machine Node timing, not browser latency or a model-family speed comparison; the original review run measured 5.24/24.62 ms, illustrating host/runtime variation. The latest run's exact timing lives in the evidence receipt.

`scratch/lesson-tools/Scripts/python.exe scripts/verify-long-context-native.py`, NumPy 2.3.5, PyTorch 2.14.0+cpu:

- Explicit segmented attention versus SDPA, including the last short segment.
- Latent projected outputs plus source/query/all-parameter gradients versus SDPA.
- Scratch masks, short lengths and causal legality.
- All four saved states on all 360 source trajectories: **1,440** cases, exactly matching stored logits in this environment. Reversed-tagged-record/padded null maximum **6.676e−6**.
- First-point class-change fixture; learned-gate RG-LRU reset and zero upstream gradient across a fresh-document boundary.

Original training/validation-selected fits are reused, with no post-test tuning or unnecessary retraining. The supplied full training program remains exposed. Native optional RecurrentGemma package parity is not executed. GPU training, huge language-model checkpoints and clinical/sign-language translation claims are outside this lesson.

Evidence: `docs/teaching/evidence/long-context/model-checks.json`, `native-checks.json`, and the saved native validation-logit oracle. `scripts/export-long-context-assets.py` losslessly exports fixed learned arrays and validation roles; it never retrains or selects new outcomes. Model and native evidence bind their actual inputs with SHA256.

## Author learning-experience review

The first-pass route connects the entrance-detail question to three concrete memory objects, then real hand geometry. Shapes and causal boundaries are refreshed before use. Inline support accompanies each representation change; deeper gradients, budgets and scans come later. The labs have manipulable records/gates/coordinates, a checked contrast and a null, immediate outputs and a deterministic reset. A scalar recurrence is not mislabeled a full Griffin model. Real-data role/label caveats have explicit homes; the stronger ordered baseline and unfavorable neural fits are retained. Complete source is present on demand rather than forcing hundreds of lines into the main route. Changed practice requires transfer, not remembering a preset. This is an author heuristic review, not an observed beginner study.

## Complementary independent review

The State Space author independently read the models, lab components, figures, bridge/export/render code, verifiers and prepared visual/lab specifications, and reviewed the manuscript's mechanisms and learning progression. The review found no unresolved scientific or mechanism defects. It requested visible path direction/ordinal cues and correction of “bars sum to one” to “weights sum to one.” These are closed by a painted start ring, end square/arrow, selected-point ordinal label and the corrected weight caption. The selected point also has a larger dashed drag handle with inverse-screen-CTM coordinates, preserved grab offset and pointer-cancel/lost-capture cleanup.

The independent reproducible verifier is `scripts/verify-long-context-independent.mjs`, with source-bound results in `docs/teaching/evidence/long-context/independent-checks.json`: 24 fresh PyTorch coordinate-edited, partial-mask and reassigned-position cases agree with the browser model within **4.04e−6**; six separately evaluated NumPy/SciPy mean baselines agree with the JSON-export JavaScript baseline within **1.39e−17**. This complements rather than repeats the author's original-sample oracle checks. Both author verifiers now invalidate a prior receipt before starting, preventing a failed rerun from retaining an apparently current success.

## Rendered review (production preview)

At the initial 1280px desktop viewport on the production build, all 16 inline figure families were visually inspected, alongside the signed recurrence chart, latent weight bars and learner context. Figure widths stayed within their containers; the document did not overflow horizontally. There were zero KaTeX errors, all 11 lesson jump links resolved, and the review tab had no console errors.

Actual controls verified cache memory 2→4 (10/3→5), keyboard ArrowRight 2→3 (10/3→2.75), future-value exclusion, recurrence near-hold (0.594668384), input-only-gate null (0.196608), zero inputs (0), paired-tag permutation, value reassignment, uniform latent collision and reset. The learned workbench loaded on request: row 7 N4/seed29 class 1 and P=0.780652991; first-point-only class 10/P=0.00007098385698; attempted final-point removal rejected; the small reflection retained class 1 while changing P to 0.780797242; whole-tagged reversal differed by only 4.441e−16. Blank numeric input preserved the last finite model result. All three full source panels fetched successfully and included their complete ending entry points.

The final rebuilt page was checked at 320×844 and 760×900. Every figure/lab container remained within the reading column; exact wide tables retained their own visible horizontal scrollbar. A 760px inspection exposed a genuine weakness missed by viewport-only CSS: the persistent sidebar left too little width for a two-column path. Topic-scoped container queries now stack the path, model selectors, workspaces and paired figures according to actual article width. The path grew from roughly 156px to 342px at that breakpoint. At 320px, flow arrows point down between full-width stages and trajectory labels use larger logical type. Both fixes were visually confirmed after rebuilding. Mobile slider and latent-collision controls were exercised successfully.

Actual off-center pointer drags preserved their grab offset and updated the model live: desktop movement (+24,−19) CSS pixels moved the handle by exactly that amount and changed P(class 1) from 0.780652991 to 0.788298407; a fresh mobile (+14,−12) gesture moved the handle accordingly and changed P to 0.783051722. The selected-point ring, start/end markers and ordinal label remained visible. The final tab had zero console errors; the temporary viewport was reset. `browser-review.json` binds this bounded author review to the delivered source. No automated broad browser suite or observed learner study is claimed.

No scoped findings remain open. A simulated network failure was not forced; fetch abort/retry paths were source-reviewed and successful local resources were exercised. Root retains final shared build/sequence/ledger ownership.
