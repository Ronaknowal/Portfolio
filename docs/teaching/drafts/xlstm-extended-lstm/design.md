# xLSTM: content design, source decisions and handoff

Prepared 13 September 2026. Stable ID `xlstm-extended-lstm`, module position 34, Deep Learning Fundamentals & Architectures. This packet is research and writing only; visual/lab implementation, independent review, browser checks and integration remain not started. Root owns the shared phase ledger and final checkpoint. No live lesson, catalogue, manifest or navigation was edited.

## Scope and original source

The actual content preflight was run with `node scripts/build-curriculum-inventory.mjs --topic xlstm-extended-lstm --work content`. It returned the existing advanced topic in Alternative Architectures & Historical Models, content in progress and implementation not started. No destination note existed for this ID. The applicable shared notes contained no unresolved item for this topic. Individual design is required; a historical publication is not evidence that its content is accurate or that this revision is implemented.

The complete 1,318-line original `src/learn/data/topics/xlstm-extended-lstm.jsx` was read, including its prose, mathematics, programs, interactive components, benchmark tables, failures and exercises. Source SHA256, verified again during this packet: `d9594e806e11b12e8eedbd763c141da9b36bf9067a4ef82e986e80ee80f9b995`. Baseline commit `8c5da59f18516be77c29d5aeeafca3decca4f738` retains the original. The original hash remains unchanged.

Keep the existing display title **xLSTM (Extended LSTM)** and stable ID. The manuscript heading is an explanatory lesson heading, not a proposed catalogue rename. The scope naturally owns scalar and matrix cells, normalization/stabilization, parallel/chunk schedules, full blocks, model training, relevant version distinctions and proportionate current applications. TFLA's sigmoid-input variant and later 7B/scaling work are important same-owner updates because they prevent a false definition of xLSTM as one immutable exponential-gate architecture.

Sequence is **Modern Hopfield Networks → xLSTM → Hyena** with actual local links and `module=deep-learning-fundamentals`. Earlier RWKV and SSM links support comparison but do not skip the next topic or reorder the module. Root owns the later Neural ODE packet; no ODE work was started here. The attention author was told the exact signed matrix operator and scaled-floor distinction so that their sparse/linear lesson does not misclassify xLSTM as a softmax approximation.

## Learner outcomes and local prerequisite support

A learner should be able to:

1. Trace a normalized scalar write history and identify surviving contribution mass.
2. Explain how learned recurrent gates differ from hand-selected arithmetic examples.
3. Transform raw scalar/matrix states into a stabilized representation without changing the intended read.
4. Calculate key/value outer products, query reads, signed interference and the active denominator floor.
5. Distinguish the same operator under recurrent/dense/chunk schedules from a changed gate or normalization architecture.
6. Read and run the complete small digit-model program, identify fit/validation/test roles, and interpret a negative result honestly.
7. Diagnose causal-prefix, state-carry, shape and numerical mistakes with a focused counterexample.
8. Calculate persistent state bytes and interpret a current architecture or performance claim with its configuration and evidence.

The introduction explicitly defines the core route and optional deeper branches. Local prerequisites include recurrence, candidate/gate meaning, sigmoid and tanh roles, weighted averages, dot/outer product shapes, residual paths, RMS normalization, logits/softmax/cross-entropy and one chain-rule gradient step. These refreshers make the lesson readable even if a learner cannot yet recall every earlier mathematical or attention topic.

## Hurdles and chosen teaching forms

| Hurdle | Mechanism/explanation | Representation and independent work |
| --- | --- | --- |
| Forget gate as a deleted token | It scales a mixed state before a new write | Signed contribution bars and chronological weight ledger |
| Exponential write means unbounded output | Positive mass normalization bounds scalar averages while raw totals can grow | Raw totals, convex interval and output valve |
| Stable write one means raw write one | Each time uses a shared but changing scale | Raw/scaled twin ledger and +1,000 log-shift null |
| Gates are magical hyperparameters | A task loss changes gate-generating weights | One exact quotient gradient step and six actual trained models |
| Matrix means a list of stored tokens | Outer products mix associations in key/value coordinates | Address plane, product grid and surviving-write expansion |
| Matrix weights are probabilities | Dot products and normalizer sums can be signed | Cancellation example with read outside value range |
| Stabilization only rescales matrix entries | The fixed denominator floor changes scale too | Checked raw/scaled/wrong-floor counterexample |
| Chunking changes the model | Carry old state and local writes into one read | Causal tiles, boundary state and difference trace |
| A memory cell is the complete model | Projections, normalizations, residuals, FFN and loss also matter | Dimensioned full block and complete offline source |
| Fixed state implies endless accurate recall | Finite memory, learned representations and task stress matter | Actual clean/reverse errors, state bytes and controlled-retrieval exercise |
| One impressive sample proves a model wins | Preserve all predeclared models/seeds and roles | Six-run table, loss curves, failure examples and changed practice |

There are twenty inline figure contracts and four focused investigations. Their number follows the separate mechanisms, not a shared quota. The scalar ledger, matrix address plane, chunk timeline and pixel/state reader use different interaction forms. Worked inline support remains at the point of introduction. The fresh scalar four-row question, fresh matrix inputs, editable seven-token chunk input and fresh digit source 187 differ from preceding solved examples. Outputs are visible for the initial valid state and remain bound to active inputs; changed input recomputes the linked views. All ten practice problems have initially closed hints and solutions.

## Conservation map and necessary corrections

| Original useful coverage | Destination / decision |
| --- | --- |
| LSTM history and motivation | §1 refreshes actual classical update without claiming sigmoid gates cannot strongly overwrite |
| Exponential writes, scalar normalization | §§2–3 retain and expand weighted history, empty initialization and learned gates |
| Stabilizer state and numerical robustness | §3 derives common rescaling; §4 adds correct matrix floor and an independent oracle |
| Scalar memory mixing and heads | §2 explains learned within-head mixing before pointwise state update; §5 explains dependency limits |
| Matrix/covariance-style memory | §4 teaches explicit shapes, outer products, read geometry, interference and nonconvex signed behavior |
| Parallel, recurrent and chunk forms | §5 derives all three, includes complete small operators, nonzero incoming state and causal checks |
| Cell/block code | §6 and full downloadable programs replace incomplete/fragile examples with complete learnable cells and a working network |
| Toy character-training example | Replaced by licensed real digits with explicit roles, six actual fits and preserved failures; complete end-to-end training capability retained |
| Original gate stepper and matrix display | Twenty specific inline figures plus four editable investigations, with live computed results and checked nulls |
| Large-model configuration and integration | §8 corrects actual 7B architecture, state accounting and current official loading references |
| Performance/scaling comparisons | Exact operation/storage analysis and actual small results replace invented timing/quality hierarchy |
| Application suggestions | §9 explains distinct image, missing-patch forecast, variate-mixing and offline-control uses with primary references |
| Failure modes, exercises and resources | Consolidated diagnostic table, ten changed exercises with explained solutions, annotated primary/alternate resources |

Consequential original problems resolved:

- A sigmoid can produce strong write/forget ratios and near-overwrite; no structural impossibility claim is retained.
- sLSTM input is exponential in the core, but the original family permits sigmoid or exponential forgetting. The 7B release uses sigmoid forgetting. Positive raw forget preactivation is not itself an exponential retention factor near .95.
- Stabilized gate ratios from different timesteps are expressed in different scales. Their values do not directly equal the original cross-time write ratio. The stabilizer need not increase monotonically.
- Empty normalization starts at zero. Starting n at one supplies a zero-valued prior, not an obligatory numerical safeguard.
- Scalar recurrence carries h/c/n/m; h influences subsequent gate preactivations. Lack of the same direct affine scan does not imply absence of batch/head/channel parallelism or optimized kernels.
- C orientation is consistent: key rows, value columns, k vᵀ update and Cᵀq read. Query scaling occurs once. A covariance-style outer product is not automatically a centered empirical covariance.
- The raw matrix floor one becomes exp(−m) after common scaling. Matching two incorrect ports does not validate a wrong floor; the counterexample deliberately activates it.
- Signed matrix coefficients do not define positive normalized attention probabilities. Scalar convex-average bounds do not transfer to matrix reads. Additive write is not exact dictionary replacement or a delta-rule correction.
- A scalar head forget gate scales all old associations in that head. Read addressability does not grant selective old-key deletion.
- Original versus 7B blocks, parameterization and training data are distinguished. The 7B release is all matrix blocks with actual 32/8/256/512 state dimensions; mandatory float32 C/n/m state is about 128.251 MiB per sequence, not a tiny unqualified megabyte claim.
- Chunking retains O(T d_k d_v) state read/write work in addition to local comparisons. It does not divide all work by chunk length. Cached single-step attention and full-sequence attention are different workload counts.
- Native current loading guidance is sourced from the model card/repository; no guessed `from_pretrained` API or claim that Transformers integration is categorically absent.
- xLSTM-Mixer and TiRex use scalar mechanisms in their cited adaptations. Modern successful mLSTM language models do not make scalar recurrence obsolete in all tasks.

No useful original depth is intentionally discarded merely to shorten reading. The large scale benchmark rows and hardware ratios were not retained as universal claims because their original grounding/configurations were unreliable. Their learning purpose is met by exact complexity, real local measurements, primary-source methodology and a scoped advanced reading path.

## Primary research and claim locators

All resources below were retrieved/read on 13 September 2026. “Read” means the specified primary text, not an unobserved video or executed external project. Research improved the explanation and investigated misleading claims; manuscript examples, diagrams, programs and exercises are original to this packet.

| Resource / actual material read | Claim locator / role | Scope retained |
| --- | --- | --- |
| https://arxiv.org/html/2405.04517v2 ; also extracted 55-page PDF text from the same version | Full main §§1–3, §4 structure/formal and associative-recall setup, §§5–6; full appendix section list; Appendix A.1–A.4 including vector cells, stabilization, recurrent/parallel forward and backward and block Figures 10–11; Appendix B opening retrieval/formal setup | Cell family and mechanisms. Own derivations and checked programs explain them. No assertion that every appendix benchmark table was independently reproduced. |
| https://arxiv.org/html/2503.13427v1 ; 21-page PDF text | Main intro, §§2–3.2, gate/normalization/read equations, block architecture, training data/configuration; §5.1 long-context protocol and §5.2 prefill/generation discussion | Dated 7B changes, correct scaled read, practical long-context limitations. No GPU measurements copied into our experiment. |
| https://arxiv.org/html/2503.14376v2 | Full section/appendix list; §§2.1–2.2 recurrent and chunk operators, §§3.1–3.2 GPU/tiling, §§4.1–4.2 sigmoid variant and normalization, §5.1 setup/selected comparison context | Separate sigmoid-input variant; incoming/local decomposition; hardware tiling context. Empirical normalization similarities are not an exact all-input operator identity. |
| https://arxiv.org/html/2510.02228v2 | Intro, §§2.1–2.2 scaling definitions/fitting, §§3.1–3.4 setup/equal-compute and token/parameter reasoning | Scoped advanced methodology, dense Llama-2 baseline and measured range; not a proof of universal architecture superiority. |
| https://github.com/NX-AI/xlstm | README requirements, large/native examples and architecture configurations | Verified practical repository and current backend context; no package installation or official model execution. |
| https://huggingface.co/NX-AI/xLSTM-7b and /blob/main/config.json | Model card loading/license section and complete configuration fields | Transformers route, checkpoint license, 32 blocks/eight heads, key/value ratio and state dtype. An old config's Transformers version field is not treated as the latest package version. |
| https://github.com/NX-AI/mlstm_kernels | Repository identity and linked TFLA purpose; algorithm details read from the paper | Useful implementation destination, not a claim that its code/tests were executed or all source was inspected. |
| https://openreview.net/pdf?id=HEQBFj9Lnb ; https://nx-ai.github.io/vision-lstm/ | VisionLSTM introduction/method and full short project page | Image patch order and task readout adaptation. Our row reader is not ViL or ViL2 replication. |
| https://arxiv.org/html/2410.16928v3 | Section list and main introduction/§3 method through experiment introduction | xLSTM-Mixer's initial linear forecast and scalar variate mixing. Avoid over-specific axis claims in ambiguously worded paired-view prose. |
| https://arxiv.org/html/2505.23719v2 | Problem setup and §2 architecture, input/output/quantile loss, multi-patch forecasts and §2.1 contiguous masking | Scalar memory plus presence masks, future missing patches and trained quantile outputs; no benchmark replication. |
| https://arxiv.org/html/2410.22391v2 | §§3.1–3.2 encoders, token sequence, action prediction and offline objective; experiment context | Observation/desired-return/previous-reward availability, no current future reward, episode state/reset. Do not turn simulated/offline benchmarks into deployment claims. |
| https://maxbeck.ai/talks/ ; https://maxbeck.ai/resources/talks/2026-03-PhD_Defense_Beck_share_selected.pdf | Author talks index and text of the full 33 selected-slide PDF, including later limitations/future work | Alternate visual route and research discovery. Slide simplifications are checked against primary equations rather than copied as universal claims. No image-layout review was performed. |
| https://www.youtube.com/watch?v=KjvCtslDJv0 | Author-index recording link/identity verified; no viewing or transcript | Annotated optional recording, explicitly not watched. No fabricated timestamp or claim of reviewed demonstration. |
| UCI dataset 80 and original metadata | Current page/license from same authorized session, full original names reread, byte hashes and feature uniqueness checked here | Real input roles, attribution and limitations; see data-provenance.md. |

The original canonical paper's main progression was explicitly mapped before freezing: motivation/history → scalar cell and stabilization → matrix cell → block architecture → state-tracking/retrieval/language experiments → limitations. Appendix progression: vector/gradient details → block variants → experiment/task definitions and broader comparisons. The manuscript includes the core operations and training bridge, while full hand-optimized GPU backward kernels, large-language benchmark replication and every architecture hyperparameter table remain optional source branches. These omissions do not remove the ability to understand, calculate or train the instructional model.

A particularly useful source discrepancy was the denominator floor: both the original parallel appendix and later 7B/TFLA equations use the transformed exp(−m) floor, whereas the original website implementation did not. The packet independently derives and tests this fact. The later sigmoid-input formulation is explicitly different, so an exact floor requirement for the exponential operator does not wrongly prohibit newer purposeful designs.

## Programs and bounded author evidence

`memory_mechanisms.py` executes scalar raw/stable scans, matrix raw/stable scans, a dense causal form, a moderate-input chunk form, exact signed/floor counterexamples, nonzero-state carry, future-prefix nulls and a scalar autograd comparison. Float64 raw/stable scalar output error is zero in the worked example; the +1,000 log shift differs by 1.50e−14; scalar gradient error is 2.43e−17. Dense/recurrent and chunk errors are below 3e−15. These are author mathematical probes, not a formal phase-two model suite.

`row_sequence_models.py` ran six CPU fits once: LSTM/sLSTM/mLSTM, seeds 19/43, using the declared data protocol. It records all curves/selection/metrics and asserts finite gradients and valid carry/prefix behavior. The weaker mLSTM outcomes and seed variation remain in the manuscript. No hyperparameter rescue or GPU test followed those observations.

`author_calculations.py` loads saved fits without retraining, records all fresh scalar/matrix and real-image investigation fixtures, a one-step learned gate update and changed scalar exercise. It checks carry and causal-prefix behavior. The author's initial gradient probe accidentally constructed one temporary tensor at default float32; the comparison was corrected to explicit float64, rerun, and passed. This was an oracle precision correction, not a suppressed model failure.

The complete small teaching programs have setup, inputs, commands, output files and interpretation in the manuscript. The inline scalar program is separately checked during closure. External official models, GPU kernels, released scaling notebooks, videos and pretrained application systems were not executed. They are annotated resources and optional advanced branches, not hidden prerequisites.

## Author closure and remaining implementation

Author closure checks are recorded in `author-checks.json` after full manuscript/specification reread. Required checks: inline scalar expected output; actual source/data hashes; exact role separation and unique features; six saved models loaded without training and compared to saved prefix logits; confusion totals/errors; all local route IDs/module membership; fresh/worked distinction and 20 initially closed exercise details; figure/investigation anchors; exact numerical fixtures and finite states. No application build or browser campaign is relevant to this content-only scope.

Learning-experience reread must confirm: simple opening before terminology; local prerequisites; scalar → matrix progression; visible examples before abstraction; learner edits real entities; current outputs remain visible and update with the controls; exercise inputs differ from worked inputs; causality and numerical nulls explain something useful; poor results are interpreted honestly; resources have specific roles; core/deeper route and next Hyena transition are clear. No question quota or uniform-lab template substitutes for that assessment.

Phase two must consume the complete packet, extract small lazy assets, implement twenty figure contracts and four investigations, preserve exact operators and full recurrent state, check ports and displayed programs, independently review mathematical/learning correctness, render-check keyboard/narrow/accessibility/loading/failure/performance, integrate the download bundle and actual references, and update source-bound status. Root's checkpoint is content completion only. User acceptance and implementation review remain distinct.

Temporary primary PDF text extracts and this packet's Python import cache are disposable only after their claim locators/results are preserved. Remove only those exact resolved owned paths; retain source data, fit arrays, executable programs, fixtures, manuscript, specifications and design. Do not scan or clean another author's scratch or shared runtime.

### Completed author closure, 13 September 2026

The author reread the full 671-line learner manuscript (including all worked examples, practice hints/solutions and references), the complete visual specification and design/provenance. Where a tool output truncated a middle range, that range was read separately. The reread moved the scalar investigation to a genuinely new four-observation input, removed the fresh digit label from preceding prose, clarified field labels versus answer labels, and rewrote compressed specification language for an implementer to read directly. Chunk fixture coordinates required a manual bound of four rather than three; the spec now includes every original input and a separately bounded future-edit fixture.

The explicit chunk-reset contrast was calculated and retained: maximum difference 5.04953803825383, first chunk unchanged, with full input/carry/reset/future arrays in investigation-results.json. A temporary author-script variable collision between a chunk reset and a tensor reset was corrected with a descriptive reset_chunks variable; the complete affected author calculation then ran successfully. No fit was rerun or changed.

`check_author_packet.py` passed: inline printed output matched exactly; all six saved selected models reproduced every clean validation prefix logit with maximum error zero; parameters and all confusion totals/errors reconciled; source/data hashes matched; 5,620 unique feature rows, 1,000/300 disjoint fit/validation rows and 1,797 test rows; four actual local routes resolved to this module; twenty unique figure anchors, four investigation anchors and twenty initially closed practice details; active floor, scalar gradient, matrix zero-value/query and chunk contrasts checked. Results are in author-checks.json.

Learning-experience checklist is complete: the opening poses a concrete task before terminology; scalar and matrix mechanisms have visible numerical traces; mathematical refreshers and a single gradient update connect operations to learning; all four investigations edit meaningful entities and start with unset input-bound results; fresh exercises/fixtures do not merely replay solved inputs; nulls have a stated invariant; the six real fits retain unfavorable results and finite-data limitations; current resources are annotated with what was actually read or not watched; deeper material is optional and the immediate next route is Hyena. No outstanding content-author finding remains.

No browser, production build, official large-model run, GPU-kernel benchmark or formal independent implementation review was performed. Those remain phase two. Exact owned temporary primary text extracts and import cache were removed after preserving this source record; all necessary pending data, models, programs and authored content remain.

## Live exploration revision — 21 September 2026

The user replaced prediction-and-reveal lab assessment with immediately visible, playable exploration, including removal of optional prediction controls. The manuscript and specifications now use that contract. This changes the teaching interaction, not the topic's model predictions or its mathematical masks/gates.

**Topic disposition:** Inspect scalar and matrix recurrent memory. Edit evidence gates/values, address vectors, chunk boundaries/state carry and supported digit pixels. Update stabilized scalar numerator/denominator, matrix-address contributions, causal chunk states and exact model outputs together. Distinguish a probability normalization from signed matrix addressing and identify what state must cross a chunk boundary.

Retain all source data, formulas, measured results, code programs, references and independent practice. The existing author calculations remain evidence for those unchanged quantities, not evidence that a browser implementation already satisfies this new interaction contract. Phase two must implement and verify live updates, linked-view agreement, bounded work, reset, invalid/null cases, keyboard access and responsive diagrams. Content remains prepared; implementation remains not started.

## Implementation ownership and content-depth revision — 22 September 2026

Delivery remains **content-first**. The complete computational teaching route is part of this prepared packet now; phase two receives written code, explanations, mapped state/settings and closed practice, rather than an instruction to invent the missing mechanism. Earlier authoring records remain dated evidence; this section supersedes their incomplete depth handoffs. The title and stable ID are retained because the new material fulfills the existing scope.

| Advertised computational outcome | Scratch/source owner | Ordinary tool route | Matching bridge | Independent practice | Scope boundary |
| --- | --- | --- | --- | --- | --- |
| Scalar exponential memory with stabilization and learned gates | memory_mechanisms.py::scalar_scan | row_sequence_models.py::ScalarMemory | raw/stabilized arithmetic and gradients; h/c/n/m correspondence | Practice 1,2; changed carry split | One-head full learned scalar cell; block-diagonal multihead/fused variants are family comparisons |
| Matrix read, floor, recurrent/parallel/chunk execution | memory_mechanisms.py::matrix_scan/parallel_read/chunk_read | row_sequence_models.py::MatrixMemory | C orientation, query scale, denominator floor, extra gate/RMS mapping | Practice 3–5,7; state-reset diagnosis | Unscaled chunk oracle moderate logs; extreme route uses stabilized scan |
| Full trainable network, data roles and customization | row_sequence_models.py::DigitReader/main | nn.Module, RMSNorm, LSTM baseline, Adam | state_dict, forward/state carry and causal prefixes | Practice 6,8–10 | 7B/TFLA/vision/forecast/control models introduced as named comparisons, not recreated end-to-end |

All local source owners above were inspected at their actual function/class definitions. Full model fitting, data/provenance and existing worked results are retained. Reused actual prerequisite code is named explicitly in the manuscript; prepared owners are not described as already published updated instruction. Whole-family releases mentioned for context do not expand the promised executable outcome into every checkpoint or every GPU kernel.

The teaching sequence is construct → explain the state/update → normal tool use → compare the same contract → changed-constraint practice, inserted where the relevant mechanism is explained. Original mechanism programs remain canonical; new programs depend on them only where the import is explicit. No browser program, published lesson, manifest or curriculum sequence is changed by this revision.

Author checks for this revision: source/API-contract reading, Python syntax parsing, matching embedded/downloadable source and local links, and scoped arithmetic probes where recorded in the specialist-writing report. These are content-authoring checks. Earlier fit outputs remain their original evidence; new multi-process/GPU/specialist-package execution, formal independent implementation review, rendered diagrams/labs and browser/accessibility/integration checks are **deferred**, with exact targets in the current visual specifications and specialist report.

Next action: after the root records the new content checkpoint, consume the full current packet for an authorized finish request, execute the relevant new programs and capture honest outputs, build the specified topic-owned views, independently check the translated models and integrate them. No core scratch/library manuscript writing is left as a finish-only TODO.
