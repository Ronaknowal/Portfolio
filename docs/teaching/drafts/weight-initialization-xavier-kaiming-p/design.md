# Initialization: content design and continuation

Stable ID: weight-initialization-xavier-kaiming-p. Actual module: deep-learning-fundamentals, topic 6; position 27 of the authorized next-30 content-only range. Owner: /root/deep_foundations_content. Research and writing are prepared for root checkpointing; implementation is not started.

## Preflight, scope and sequence

Ran the inventory with --topic weight-initialization-xavier-kaiming-p --work content. Read current handoff, teaching standard, design brief, ML/deep-learning guidance, coordination/retention rules, scoped record and destination note. Read the complete 1,146-line published source, recovering truncated ranges 925–961 and 960–end. Do not infer absence from a filename: this source is the manifest-mapped topic file.

Display title proposed: “Weight Initialization: Xavier, Kaiming, Orthogonal Methods & μP.” Stable identity unchanged. Own initialization scale, second moments, fan conventions, direction-versus-average preservation, symmetry, actual initialization comparisons, restricted μP/Adam width scaling and numerical/API pitfalls. The extra title term makes an existing substantial mechanism visible, without inventing a new curriculum topic.

Prior transfer learning motivates fresh heads and zero output factors. Here μTransfer explicitly means hyperparameter transfer, distinct from reuse of pretrained weights. Normalization precedes this topic, so refresh axes/state only at their relevance. Residual paths occur next; give a local x+F(x) explanation for Fixup and defer full path mechanics there. Convolution follows later; core models use flat 64-input MLPs and explain all needed dimensions locally. Optional LSUV/Fixup/precision branches and μP mastery do not silently become prerequisites for the next lesson.

Consumed the random-matrix-theory destination note. Its expectation/actual-draw/worst-direction distinction now has an exact diagonal map, fixed Gaussian draw, QR identity check and nonlinear Jacobian counterexample. This is an adapted local explanation with a source bridge, not a repeated random-matrix chapter.

## Source conservation and coverage

Baseline commit 8c5da59f18516be77c29d5aeeafca3decca4f738. Original src/learn/data/topics/weight-initialization-xavier-kaiming-p.jsx; SHA-256 245f576b74ecb27f628d08f5b285e18469b29d67662ceca8f57daeb860de67a5. Source and runtime are unchanged.

| Original scope | Decision/current home |
| --- | --- |
| Motivation and depth amplification | Opening actual twenty-layer forward/backward probe |
| Forward/backward variance derivations, fan-in/out | Rebuilt using explicit second moment and assumptions; unequal dimensions example |
| LeCun, Xavier, Kaiming, uniform bounds, activation gains | Core recipe table and complete API example; local leaky-ReLU derivation |
| Orthogonal initialization and dynamical isometry | Exact directional geometry and ReLU Jacobian distinguish linear guarantees from nonlinear claims |
| Symmetry and zero initialization | Two actual gradient fixtures plus zero-head exception |
| Twenty-layer program and visual trajectories | Complete seeded float64 probe; no invented flat or underflow curves |
| μP recipes, coordinate check and learning-rate sweep | Complete restricted bias-free Adam model, 54 actual fits, fixed inputs and base-width null |
| LSUV and Fixup | Sourced optional algorithm/placement explanations; later residual owner handles full paths |
| Architecture defaults and precision | Inspect concrete APIs; omit unverified architecture catalogue and hardware anecdotes |
| Truncation, biases, normalization and failure modes | Absolute-bound example, actual conversions, local mechanism-based cautions |
| Old exercises | Six changed problems with hints/solutions, core versus optional scope explicit |

Important repairs: ReLU halves symmetric second moment, not variance; input need not be centered for the conditional independent-weight calculation. Pooled tensor variance, per-coordinate sample variance, and initialization expectations are different. Xavier cannot keep both unequal-fan factors near one. Orthogonal plus ReLU does not guarantee exact isometry. A scalar variance condition does not certify full Jacobian conditioning. Deterministic asymmetry can work; zero heads are not all-zero hidden networks. Large logits do not automatically overflow stable CE. The old μP Adam-style rates paired with SGD, missing readout machinery, inconsistent optima and universal transfer claims are replaced. Truncated-normal bounds are absolute and actual variance falls; BF16 does not zero every small number. LSTM defaults are not asserted orthogonal. Fixup's exact depth exponent and zero-last placement replace a general “scale the second layer” claim. No fabricated speed, GPU memory, training-corpus or universal architecture claims remain.

## Learner hurdles and representation decisions

| Hurdle | Response |
| --- | --- |
| Small weights feel automatically safe | Measured depth traces with orders-of-magnitude changes |
| Variance and squared magnitude sound identical | Four persistent values, moved mean, separate squared-distance bars |
| Fan notation feels arbitrary | Count contributions forward/backward and work 100→25 example |
| Average stability feels like every-direction stability | Editable circle/ellipse and locally gated Jacobian |
| Randomness seems magical | Equal and deliberately distinct hidden units with first gradients |
| Good initialization seems to promise best accuracy | Actual matched digit fits with three seeds and validation CE/correct distinction |
| μP appears to be one weight formula | Matrix shape, raw readout, forward divisor and optimizer rates shown together |
| A coordinate check must be perfectly flat | Preserve measured initial output decay and official transient explanation |

Seven visual homes have distinct purposes; direct manipulation and live comparisons are used where they clarify a causal question. There is no repeated text-console layout or fixed quota. The first-pass route follows the introduction. All investigations specify current visible results, meaningful entity edits, synchronized readouts, deterministic reset, real contrasts/nulls, bounded work and accessible narrow-screen alternatives.

## Canonical-reference section audit

Canonical reference: Glorot and Bengio (2010), “Understanding the difficulty of training deep feedforward neural networks.” Read section structure and full main body through conclusions (PDF lines 0–592), including assumptions and experiment interpretation; bibliography need not be duplicated. The canonical reference informs coverage rather than imposing a 2010 article outline.

| Reference section | Decision |
| --- | --- |
| 1 Deep Neural Networks | Local weighted-sum and depth intuition |
| 2 Experimental Setting and Datasets: Shapeset, finite datasets, settings | Replace with explicit real offline digit protocol and separate Gaussian diagnostic; no copied historical results |
| 3.1 Sigmoid / 3.2 Tanh / 3.3 Softsign activation and saturation | Retain saturation/centering lesson; detailed activation catalogue belongs to preceding perceptrons lesson |
| 4.1 Cost Function and Gradients | Local stable-logit/loss diagnosis; loss-function lesson owns complete objective taxonomy |
| 4.2.1 Initialization theoretical considerations | Re-derive with second moments and exact fan compromise; declare assumptions |
| 4.2.2 Gradient propagation study | Actual forward and backward tensors; distinguish parameter gradients from activation gradients |
| 4.3 Back-propagated gradients during learning | Explain independence breakdown and why initialization diagnostics do not determine trained outcomes |
| 5 Error curves and conclusions | Actual matched fits, development-only interpretation; no universal winning initializer |

Do not repeat the source's problematic sentence identifying mean singular value with volume change. Volume relates to a product of singular values in the square case; this lesson uses norm gains and squared singular values correctly.

## Research register and actual review extent

Checked 2026-09-12. Direct primary links are embedded in manuscript; wording, exact examples and experiments are independently written.

- Glorot/Bengio PMLR v9 PDF: full main-body and section-list review as above.
- He et al., arXiv 1502.01852 PDF: §2.2 forward/backward rectifier derivations and assumptions read; selected depth comparison inspected, no full experimental reproduction or complete appendix review claimed.
- Saxe et al., arXiv 1312.6120 PDF: abstract and dynamical-isometry body near lines 799–861 plus selected nonlinear discussion; no complete proof/appendix audit.
- LSUV, arXiv 1511.06422 v7 PDF: abstract, §3 and Algorithm 1 including iteration/tolerance and sequential calibration read. No full dataset benchmark verification.
- Fixup, arXiv 1901.09321 v2 PDF: §2 variance reasoning and §3 boxed recipe including exponent/zero layers/scalars read, selected experiment context. No full appendix proof audit.
- μTransfer, arXiv 2203.03466 v2: abstract read. Technical executable convention verified from Microsoft mup README, layer.py and optim.py rather than claiming full paper derivation review.
- Microsoft/mup current main: README basic usage/base shapes/reinitialization/limitations, coordinate-check section including initial-output decay, and basic math explanation read. Full 79-line layer.py and MuAdam portion through line83 of optim.py read. MuSGD was not audited or executed. Package absent; no environment install. Review version is current main on the stated date, not a pinned release.
- PyTorch 2.14 nn.init: gain table/SELU caveat, no-grad/return contract, Xavier/Kaiming including orientation, trunc_normal absolute bounds and orthogonal shape behavior read. LSTM 2.14 initialization note read. The relevant example uses exactly this declared orientation.
- Microsoft Research μTransfer blog, March8 2022: full article body read, annotated as a historical alternative explanation.
- Stanford CS231n official 2017 syllabus and Lecture6 destination: scope and video title/link verified; video was not watched and no timestamp/content claim is invented.

## Author checks and deferred work

Executed the complete program: 18 digit fits, 54 width fits, 18 twenty-layer probes, exact/autograd/truncation/precision fixtures. Recomputed added changed/null fixtures only; did not rerun all fits without a reason. Base-width mode traces match exactly for all seeds/rates. Geometry constraints preserve mean squared gain; changed moment fixture gives variance1.5. Spectrum/QR/Jacobian values are actual calculations. JSON-derived manuscript figures retain tiny nonzero float64 values. Validation is not relabeled test.

Read the full final manuscript and visual/provenance/design packet as author review, checking first-pass progression, terminology, equation assumptions, source support, interpreted units and numbers, source conservation, links, practiced transfer, optional depth, actionable investigation state, and sequence bridge. Root may still raise scoped reconciliation findings; this is not phase-two browser/formal review.

Deferred: runtime wiring/rendering, actual visual construction, browser interactions, keyboard/mobile/reduced-motion checks, translated-code parity, code-download UI, lazy-loading profiling, site build and formal implementation review. Do not mark implementation complete because this packet contains executable teaching programs. Preserve all packet files as pending production inputs; no disposable own scratch assets were created.


## Scoped practice disclosure repair — 12 September 2026

Root reconciliation requested a presentation-only repair after the original author checkpoint. Every practice hint and worked solution is now in its own initially closed details block; missing hints were added as non-answer reasoning prompts. Existing questions, solution text, numerical calculations, source claims, programs, data and measured outcomes are unchanged. Verified 6 paired hint/solution disclosures, matched closing tags, no open attribute, exact preservation of all solution bodies, and unchanged lesson text outside the practice section. Root refreshes the affected content hashes; no repeated fitting or full implementation review is implied.

## Live exploration revision — 21 September 2026

The user replaced prediction-and-reveal lab assessment with immediately visible, playable exploration, including removal of optional prediction controls. The manuscript and specifications now use that contract. This changes the teaching interaction, not the topic's model predictions or its mathematical masks/gates.

**Topic disposition:** Explore scale and directional transmission. Inspect saved initialization/seed traces; edit four activation values, singular directions, depth and width/rate scaling. Show forward/backward second moments, means/variance, directional gain and shape/update formulas together. Continuous tiny models are distinct from selectors over measured training records. Choose an initialization/parameterization by the signal and update behavior it preserves, without treating average scale as every-direction stability.

Retain all source data, formulas, measured results, code programs, references and independent practice. The existing author calculations remain evidence for those unchanged quantities, not evidence that a browser implementation already satisfies this new interaction contract. Phase two must implement and verify live updates, linked-view agreement, bounded work, reset, invalid/null cases, keyboard access and responsive diagrams. Content remains prepared; implementation remains not started.


## Implementation-depth writing revision — 22 September 2026

Delivery mode: **content first**. The mechanism and ordinary-tool teaching below is written now; it is not an instruction for the finishing agent to invent missing content. Existing measured experiments and their historical evidence remain unchanged unless explicitly stated. The current manuscript section “Construct an orthogonal draw, then match a width-aware optimizer” gives the learner route.

| Computational outcome | Scratch owner and abstraction | Ordinary tool and matched comparison | Control / practice and boundary |
| --- | --- | --- | --- |
| Xavier, Kaiming, signal statistics and symmetry | `initialization-experiments.py:fill_weight, summary, exact_fixtures`; arithmetic scales and random primitives | `nn.init.xavier_normal_ / kaiming_normal_`; orientation and gain mapping already in manuscript | Changed fan, gain and depth exercises; measured fits retained |
| Orthogonal initialization | `initialization_library_bridge.py:orthogonal_matrix`; reduced QR owner is Matrix Decompositions | `nn.init.orthogonal_`; rectangular Gram contract, not fabricated equal random draws | 3×7 gain0.5 practice; dense QR cost explicit |
| Restricted μP width scaling | `initialization-experiments.py:WidthMLP`; same three explicit rates and forward divisor | `initialization_library_bridge.py:check_mup`; MuReadout/base shapes/MuAdam, two same-state updates | Width160 extension with solution; full arbitrary-architecture μP is outside this restricted contract |

All local filenames in the map are retained draft sources beside this design. A linked prepared prerequisite is not yet the improved published page: finish in module order or carry its declared source with the lesson. Already implemented autograd/loss/normalization/tensor lessons may be reused as stated; no new differentiation engine, BLAS or convolution backend is implied. Optional historical families remain explanations of a distinction unless a local implementation is explicitly named.

The content packet is ready for phase-two construction after central source checkpointing. Finishing must execute the supplied comparisons on declared compatible versions, resolve any observed numerical/convention differences, expose the exact code/downloads, and verify rendering, live controls, accessibility and production loading. Unexecuted optional package/GPU/checkpoint examples remain explicitly unexecuted; do not print invented outputs or copy previous measurements onto new code.

## Prepared-content implementation — 22 September 2026

Current state: **author implementation ready for independent review and parent-owned production integration**. This section supersedes the earlier implementation-not-started statements for author work only; the central ledger remains parent-owned. The conserved revision-3 manuscript, specifications, data and native program remain unchanged. Full teaching content is rendered to the stable topic source by `scripts/generate-weight-initialization-lesson.mjs`; only visual placeholders and the now-executed library status are replaced.

### Implemented teaching route

- All eleven manuscript sections, six changed-case practices and the additional orthogonal/width extension retain their full explanations and initially closed hints/solutions. No shortened substitute article was written.
- Six distinct live investigations: recorded forward/backward traces plus an explicitly separate recurrence; four editable observations and second-moment/variance views; equal-axis directional geometry plus a separate gated local Jacobian; simultaneous hidden/head gradient steps; actual digit-training checkpoints; width/shape/forward/Adam recipe plus separately identified recorded coordinate evidence.
- Supporting figures show an actual computed singular-value histogram, original CSV digit specimens, a labeled schematic truncated density and exact native precision conversions. Exact zero traces use a separate zero rail instead of log epsilon. Observation lines connect only recorded checkpoints and explicitly state the unobserved intervals.
- Long complete Python sources load on disclosure and remain downloadable beside the data/provenance. Canonical Python is not eagerly copied into the topic bundle. The topic-only compact measurement JSON is 110,710 bytes before compression; it contains values derived directly from the executed record. JSON serialisation normalizes signed negative zero to mathematical zero, without changing any measured quantity.
- Scoped CSS overrides inherited green surfaces with neutral charcoal and amber. Figures use explicit neutral/amber marks. No shared historical stylesheet, registry, catalogue, generated inventory or phase ledger was edited by this author.

### Actual execution and author verification

The complete `public/learn-assets/weight-initialization-xavier-kaiming-p/initialization-experiments.py` was run with the existing CPU runtime. All 18 digit fits, 54 width fits, 18 propagation probes and fixtures reproduce the conserved `calculated-inputs.json` exactly. Program/data bytes are unchanged.

`scripts/verify-weight-initialization-native.py --mup-path scratch/weight-initialization-deps` passes six substantive groups: preserved complete output; actual MuReadout/MuAdam outputs, all gradients and two updates at widths 32/96; changed 3×7/gain0.5 Gram/rank; width160 rate/divisor extension; both inline Python blocks; and all nine base-width identical traces. Recorded environment: Python3.12.14, PyTorch2.14.0+cpu, NumPy2.3.5, scikit-learn1.9.1, mup1.0.0. Evidence: `docs/teaching/evidence/weight-initialization-native.json` and the identical downloadable `native-verification.json`. `--replay-full` is available for a necessary changed-source replay; the default verifier reuses the completed sweep's output.

The mup package was installed with `--no-deps --target scratch/weight-initialization-deps`, leaving the shared runtime unchanged. Its installer-created directory needs the host permission context to execute. Retain it until independent review has finished, then remove that exact temporary directory after resolving its absolute path inside this workspace. No GPU or pretrained checkpoint was needed.

`node scripts/verify-weight-initialization-models.mjs` passes seven substantive groups including all compact empirical values, exact/changed/null moments, directional and gated identities, native symmetry parity, independent central finite differences on changed inputs/targets, width arithmetic and split isolation. JSX parses; all eleven section bodies and fourteen independent practice disclosures are retained. Source-bound record: `docs/teaching/evidence/weight-initialization-author.json`.

The author reviewed the full rendered flow, complete scripts, separate measured/theoretical quantities, update state, evidence roles, original scope conservation and next-topic link. This is not independent review or completed browser evidence. The scoped `scripts/verify-weight-initialization-browser.cjs` is ready for the parent's final production build and covers default/change/reset/null/invalid controls, true pointer plus keyboard, deferred source, theme, math glyphs and desktop/390/320 layouts with retained screenshots. Inspect those screenshots rather than treating geometry guards as visual approval.

### Current source check and limits

Microsoft's official mup repository and the versioned PyTorch initialization documentation were revisited on 22 September. The repository now states it was archived on 21 September2026; the lesson records this and pins the actually executed package instead of implying ongoing maintenance. The controlled comparison is valid for this bias-free Adam MLP only. No generic architecture μP certificate, MuSGD comparison, full LSUV/Fixup implementation, unseen-writer/test estimate, GPU timing or universal optimum claim is made. The public bridge docstring/provenance now reflects implementation execution; the conserved preparation files retain their original historical wording.

Remaining closure: independent reviewer and parent production/browser integration; the author must not mark those complete on their behalf.

### Loaded-font integration repair — 22 September 2026

The production integration found a fractional exponent tick outside its SVG with the site's JetBrains Mono font. `Plot` now reserves an 80-unit left margin and uses integer-decade logarithmic tick positions; measured points, axes, zero rail and readable font sizes are unchanged. On the current development route, all 54 seed/scheme/viewport combinations (three seeds, six schemes, widths1366/390/320) keep every signal-plot label inside its own SVG after the site fonts load. The scoped browser verifier now asserts this containment explicitly. The 320px rendered signal figure was inspected. Seven author source/model groups pass with refreshed hashes; unchanged native training evidence is reused. Parent still owns final production confirmation.
