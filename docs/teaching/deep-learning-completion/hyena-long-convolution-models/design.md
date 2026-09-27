# Hyena author implementation and review handoff

Scope: complete prepared lesson, all20 specified figures, four named investigations,
the additional efficient blocked-FFT workspace, every practice/solution, and every
scratch/library program. Existing production path and publication identity are
preserved. Root owns registration, final site build and painted browser review.

## Concept progression and actual representations

| Learning transition | Intuition / concrete worked route | Implemented representation |
|---|---|---|
| Different long-memory tasks | Fixed delay versus content-addressed association; copying is not universal memory | `DelayAddressFigure`: delayed value edges and key-selected request |
| Causal convolution | A pulse leaves a delayed echo; y2=3+1+.25 | `EchoFigure`: aligned contribution ribbons, complete output stems |
| Distance to matrix | Repeated lag becomes a diagonal, future positions are unavailable | `ToeplitzFigure`: numeric lower-triangular matrix, highlighted lag2 and forbidden future cells |
| FFT correctness before speed | Tail terms need independent slots; unpadded wrap reads a future input | `PaddingFigure`: all7 linear positions, padded eighth zero and wrapped tail paths; `ConvolutionLab` computes real FFT arrays/products |
| Input-dependent sending/receiving | Suppress an event before mixing or a receiver afterward | `GateRailsFigure`: three projected/short-causal streams, multiplication/filter/multiplication; `CoefficientFigure` factors and signed conditional matrix |
| Hierarchy | The earlier filter introduces routes through intermediate gates | `HierarchyFigure` and `PathGeometry`: every legal intermediate path, factor and sum; `GateLab` editable optional phi |
| Implicit filters | A shared position network emits a strip instead of independent taps | `LearnedFilterFigure` / `FilterReadView`: actual saved two-layer sine filter, fixed9 features, decay envelope, all60 coefficients and864 parameters |
| Stable position ruler | Rescaling coordinates changes a prefix even without new evidence | `CoordinateFigure`: both exact illustrative curves, current support and dashed unmeasured extension |
| Learning | Prediction error multiplies the inputs that contributed to each coefficient | `GradientFigure`: exact backward/update path from loss. Full NumPy/PyTorch programs and fresh native Adam steps |
| Complete block | Mixing is one branch of a residual model; pointwise transformations do not mix positions | `SequenceBlockFigure`: actual pre-LN mixer, skip, output projection, two residual junctions, exact-GELU FFN and final-position head; separate shifted teacher-forcing/generation geometry |
| Biological input semantics | Both sides of an observed boundary are available; ambiguous symbols are not extra bases | `DnaWindowFigure` and `DnaStrip`: all60 bases, biological labels with no zero, observed label separate from synthetic edit, EI/IE direction and ambiguity key |
| Data roles | Connected observed duplicates/source prefixes travel together before splitting | `DataRolesFigure`: union component, exclusion tray, actual role counts. Full raw grouping replayed natively |
| Model selection | Validation selects a checkpoint, assessment answers a separate question | `LearningCurvesFigure`: all320 actual validation CE points on a common axis, four selected epochs; independent assessment table in full manuscript |
| Counterfactual inference | An edited string asks what this fixed model does, not what biology proves | `CounterfactualFigure`: source3 original/GT→AA strings, probabilities and logits; `DnaStudy` starts from distinct source4 and executes all valid edits |
| Overlap-add | Earlier tails remain contributions after a new block begins | `OverlapFigure`: globally aligned complete block tails; `BlockedFftLab` actual cached filter FFT, each padded block and accumulated global result |
| Exact recurrence | Exponential modes summarize the whole past algebraically | `ModesFigure` / `ModalRegisters`: actual positive/alternating coefficient curves, every register update/read and boundary state |
| Deliberate approximation | Omitting coefficients changes outputs; bound has a finite support and input scale | `ApproximationFigure`: omitted-tap stems, output errors and bound; analytic6×6 Hankel plus two rank-one factors and retained measured singular values |
| Neighboring models | The stored object and update mechanism distinguish families | `MechanismMapFigure`, full prepared family prose: lag kernel, gated filter, selective state and normalized attention |
| Current applications | Short, medium, long and attention layers own distinct decode objects | `StripedFigure`: conceptual SE/MR/LI/attention strip with buffer/modal/KV objects; genomic pretraining/adaptation, learned prompt vectors and causal/bidirectional distinction |
| Honest efficiency comparison | An operation count is a reason to specify a benchmark, not a timing | `BenchmarkFigure`: editable workload, exact BigInt tensor counts, mixing/projection work scales and unmeasured timing; complete protocol practice |

The complete manuscript provides the concrete motivation, equations, assumptions,
worked steps, code explanations, consequences and failure cases at each transition.
The figures replace author directives at their exact anchors; they do not replace
the manuscript with summaries. First pass §§1–7/practice1–6 and deeper §§8–9/
practice7–10 remain explicit. All ten hints and solutions plus blocked practice
remain independently available and initially closed. No answer/prediction gates.

## Scientific implementation and evidence

`hyena-convolution-models.js` is independently importable and contains direct,
radix-two complex FFT, circular, blocked overlap-add, one/two-filter gated and
modal recurrences. Time-major arrays and output-by-input dense matrices are
documented. Exact erf-based GELU is reproduced by a convergent gamma algorithm,
not replaced by a tanh or low-accuracy approximation. Stable softplus is used.

The full DNA forward pass includes every saved embedding, normalization affine,
short depthwise cross-correlation with the correct lag orientation, fitted filter,
per-channel skip, gate, output projection, residual, exact-GELU feed-forward layer
and final-position classifier in both blocks. The one-hot linear baseline and
ungated model are their actual complete saved computations.

Native42 named checks execute all four embedded runnable Python blocks, complete
canonical mechanism/calculation programs, the SciPy overlap-add bridge, eight new
literal convolution cases, raw deduplication/grouping/roles and all four selected
fits on fit/validation/assessment data. Both reported post-fit interventions are
replayed. Each fit performs a real128-example gradient/Adam step. Thirty fresh
float64 complete traces and parameter gradients are retained as private evidence.
The linear model's fixed one-hot float32 cast is explicitly converted at its head
input only for double-precision comparisons; ordinary native replay is unchanged.

The model checker passes1,036,712 scalar comparisons:1840 saved validation logits,
exact classifications, all fresh traces and gradients,1201 SciPy erf probes,
144 signal/filter length pairs, blocked tails, signed path expansions, every split
and truncation across bounded modal cases. Maximum saved float32 logit difference
is5.896e-6; fresh float64 difference5.33e-15; finite-difference parameter gradient
difference5.81e-9. Evidence files record explicit tolerances and limitations.

No original80-epoch campaign was repeated. All four complete curves and selected
checkpoints are reused and validated. No pretrained model, new empirical accuracy,
infinite-tail bound or hardware speed claim is invented.

The historical `check_author_packet.py` predates the complete blocked-FFT bridge:
it expects three displayed programs/twenty disclosures. Its original source and
receipt are preserved; current native verification explicitly executes all four
displayed blocks and current rendering retains21 practice disclosures. Historical
checks are not silently refreshed to claim they ran against a different packet.

## Loading, source and accessibility contracts

- Only the small lossless display extract is imported by the article. The NPZ,
  raw data and full model JSON are reader downloads, not eager imports.
- The fitted-filter figure and DNA workspace fetch the selected model near the
  viewport. The latter fetches460 validation records. Model changes abort stale
  requests. Friendly errors and retry preserve learner state.
- DNA raw draft, last valid sequence, original/source identity, selected position,
  span, model choice, all intervention/view settings and pin live above asynchronous
  model/data rendering. Model switch/retry does not reset them. Pins contain a
  sequence and computation settings and are explicitly recomputed with the selected
  model for controlled comparison; the creation model is labeled.
- Invalid DNA strings retain the prior valid calculation. All60 supported symbols
  are editable by text, exact position or accessible buttons/selectors. N ambiguity
  and N class are explicitly distinct. Full source change and Reset are intentional.
- Every numeric domain is bounded; shared opt-in `NeuralNumberControl` preserves
  last valid values and invalid-message footprint. Zero/identity cases are legal.
- All scientific SVGs have a caption, accessible description and nominal-width
  named keyboard-scroll region. The local plot uses460px with105px left gutter and
  span-aware bounded tick strings (including tiny signed residues), so scientific numbers do not depend on shrinking text.
  Exact arrays/tables accompany dense geometry. Parent grids use minimum zero.
- Probability bars always share fixed[0,1], including all-N and pinned comparisons.
  Signed diagrams use explicit numbers/zeros plus semantic colors. Neutral/amber
  surfaces are topic-scoped; shared prior lessons remain unchanged.
- Public allowlist is exactly14 files: four full programs, NPZ, results, raw data,
  raw names, provenance, four selected model JSON and validation sequences. No
  author manuscript, design/specification, private fixtures or test caches deploy.
- Four complete program readers load their actual source only when opened, include
  download links and expose a friendly retry path. The original code snippets and
  explanations remain inline, including the complete efficient library bridge.

## Author learning-experience assessment

Problem and intuition precede terminology throughout; equations and advanced
branches each retain a mechanism/worked example and a relevant visible consequence.
The distinct investigations begin with meaningful editable fresh fixtures and show
current outputs immediately. Negative, zero, identity, reordered/chunked and changed
model cases are explicitly handled. Controlled comparisons preserve their inputs.
Practice includes changed calculations, causality counterexamples, interpretation,
approximation and benchmark design, with reasoning available without learner entry.
All scratch and ordinary library routes are runnable and locally explained; the
full data/model experiment is real and includes provenance, roles and limitations.
All references and application scopes from the complete manuscript are retained.
Primary original Hyena/official HyenaDNA operator/StripedHyena2/SciPy alignment
contracts were additionally re-read during implementation; video metadata is not
claimed as watched material. Render/source checks pass; painted layout, focus and
live-browser assessment remain root-owned, not implied by this source assessment.

## Browser inventory and meaningful checks

1. `hyena-convolution`: default u=[2,−1,3,0,1],h=[.5,1,−.25] gives causal
   [1,1.5,0,3.25,−.25], circular[2,1.25,0,3.25,−.25]. Pin; change final1→5:
   causal first four unchanged, circular first two[6,.25]. Switch padded FFT;
   inspect selected terms/spectra. Identity/zero/impulse, length2/12 and K>L
   circular-unavailable explanation. Invalid numeric→single Reset and keyboard.
2. `hyena-gates`: fresh output[1,.625,1.5,−1.5]. Open sender2→[1,.625,7.5,−2.25].
   Optional phi→initial hierarchy[1,1.125,2,−.5]. Choose a future matrix cell
   (zero/no routes), signed values, receiver zero and all-gates1. Pin full settings;
   resize2/8 and inspect every intermediate path/contribution.
3. `hyena-dna`: default source4/gated29 predicts EI. Original30–31→AA gives IE
   probability≈.760186. Worked source3 gives N≈.632591 for the same edit.
   All-N gated29 gives N≈.954785, not uniform. Edit near/far spans, D/R/S,
   exactly60 validation input and invalid/blank text. Restore exact source result.
   Test lag5/full60, gates unit, block1/channel15 and selected base59. Switch all
   four models on the same edited input; linear controls explicitly have no effect,
   ungated gates-off is a no-op. Pin persists and is recomputed through each model.
   Break one model asset, preserve sequence/raw/pin/settings through error and Retry.
   Change a later base only: earlier hidden-vector difference stays zero; no-earlier
   positions after editing index0 is explicitly labeled, not a vacuous proof.
4. `hyena-blocked-fft`: default7 samples/four taps B3 gives
   [1,1.5,0,3.5,−.375,2.375,1.5]. B1/2/3/8 preserve; inspect final short block and
   complete tail. Ten-sample/five-tap/B3 preset selects final block. Show reset and
   centered-same mistakes, zero kernel and K>N. Full per-bin tables are optional.
5. `hyena-streaming`: fresh full[2,.4,−.65,2.9375,1.503125,.19296875]. Boundary3
   carries[−.5,−.875]; reset suffix[3,1.6,.225]. Move boundary unchanged with carry;
   retain all taps/null zeros and deliberately truncate against actual bound. Add
   up to4 modes, signed residues, poles±.95, input2/12; pin remains complete.
6. H08 fitted filter: all60 raw/envelope/product values; change block/channel and
   check lazy-load/retry. H13 shows all four80-epoch curves and selected dots.
   H20 integer workload controls and exact byte counts; timing remains unmeasured.
7. Open all ten solutions plus blocked practice and four source readers:
   convolution_mechanisms.py, splice_models.py, author_calculations.py,
   blocked_convolution.py. Verify program loading/error/retry and download URLs.
8. Desktop and320/390px painted checks: scroll actual nominal diagrams horizontally,
   no page overflow, readable plot ticks and byte strings, all text/arrow endpoints
   inside bounds; neutral controls and visible keyboard focus. Six-base phone grid.

Commands already passed: `node scripts/generate-hyena-lesson.mjs`, native checker,
`node scripts/check-hyena-models.mjs`, `node scripts/check-hyena-render.mjs`.
No whole-site build/browser was run by the author, in accordance with ownership.
