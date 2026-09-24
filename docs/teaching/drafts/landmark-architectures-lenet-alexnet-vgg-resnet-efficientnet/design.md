# Landmark architectures — research/write design and handoff

Stable topic: landmark-architectures-lenet-alexnet-vgg-resnet-efficientnet.
Module: Deep Learning Fundamentals & Architectures, position10; position1 of the authorized2026-09-13 next30 increment.
Phase: content prepared for root reconciliation; implementation not started. Root alone owns central ledger/checkpoint status.

## Scope, title and learner outcome

Keep the topic identity and the five named landmarks. Manuscript title: **Landmark Architectures: LeNet, AlexNet, VGG, ResNet & EfficientNet**. Inception is an explicit bridge because parallel branches and1×1 reduction explain important choices between homogeneous VGG and later efficient blocks. DenseNet, RegNet, NFNet and perceptual loss remain optional branches preserving useful original breadth. No catalogue/runtime/title mutation is authorized in this phase.

After the first-pass route, a beginner should explain a classifier as data/shape/operation flow; derive a parameter and convolution/linear-MAC budget; distinguish concat/add/gate/expansion; understand compound scaling as an approximation; follow actual training/development results; reconstruct a class score from spatial maps; and propose a defensible resource-constrained comparison. It is not a guarantee of recognizing every architecture or reproducing historical ImageNet training.

Predecessor **Convolution, Pooling & Receptive Fields** is content-complete. Its actual definitions were inspected in the previous work and referenced here: NCHW cross-correlation, output shape, spatial/channel distinction, receptive-field support versus learned gradient, global averaging and its conditional invariance. This lesson refreshes each required notion locally. **Residual Connections**, **Normalization**, **Transfer Learning**, **Dropout**, **Loss Functions**, **Initialization**, and **Backpropagation** packets supply supporting depth; none is used as an excuse to omit the local explanation.

Next actual topic is **Depthwise Separable & Dilated Convolutions**, not the next published page. Current lesson introduces factorization/inverted blocks; next owns full per-channel filter restrictions/rank, dilation/sampling support, implementation details and method-specific diagnosis. ConvNeXt owns modern CNN modernization and its ablation protocol. Full transfer policy stays in the already frozen Transfer packet; current prose corrects the old name-substring freeze heuristic locally without reopening that packet.

## Preflight and original-source conservation

Executed actual preflight:

```text
node scripts/build-curriculum-inventory.mjs --topic landmark-architectures-lenet-alexnet-vgg-resnet-efficientnet --work content
```

Result: revision1 content-first/in-progress; implementation not started; record is this design. No destination note addressed to this topic. Unrelated resolved historical bit-manipulation entry was not treated as a work item.

Original full source read, including all11 sections, code, plots, recommendation tables, source descriptions and exercises:
src/learn/data/topics/landmark-architectures-lenet-alexnet-vgg-resnet-efficientnet.jsx.
Baseline commit **8c5da59f18516be77c29d5aeeafca3decca4f738**.
SHA-256 **e6a5fe7df260309646198710a61c2b36f4bc23fa1b72baf1967732aaf93dbe0e**.
Original source remains unchanged; recover it from the baseline rather than creating an archive copy. Large read outputs were followed by specific missing-range reads before recording full coverage.

| Original useful coverage or problem | Decision and resulting home |
| --- | --- |
| Historical lineage and reasons behind designs | Preserve as design questions in§§2–5; separate original paper, teaching geometry and current weight package |
| LeNet input/stages and local-feature learning | Preserve; correct156 first-layer parameters and distinguish learned subsampling/sparseC3/scaledtanh/RBF from modern average-pool/linear adaptations |
| AlexNet/ReLU/GPU/augmentation/dropout | Preserve explanatory combination; remove “first/all/universal” claims and distinguish single/five/seven-network result protocols |
| VGG repeated small kernels and dense head | Preserve full3×3 receptive-field reasoning, changed-width counterexample, exact trunk/head count and storage arithmetic |
| Inception multiscale branches and reductions | Preserve actual concat/shape and14.42fold weight arithmetic; do not claim1×1 was first invented there |
| Basic/bottleneck residual blocks | Preserve add/projection/normalization placements and actual bottleneck weights; distinguish training degradation from overfit and avoid guaranteed gradient protection |
| MobileNet/SE/EfficientNet scaling | Preserve mechanism/depth; correct channel-mixing explanation, SE bias counts, DWS count/speed conflation, B-index/φ assumption, and B0 accuracy+FLOPs search rather than hardware-latency claim |
| DenseNet,RegNet,NFNet | Preserve useful distinct mechanisms as optional branches; correct chronology/quantization/AGC-versus-forward-scale; remove unsupported dominance claims |
| PyTorch snippets, feature hooks and pretrained use | Replace dubious “executed” outputs with one complete runnable small program, measured fixtures and actual hook-based costs; retain weight/preprocess contract and explicit adaptation diagnostic; no unsupported pretrained-download experiment |
| Historical curves / “Pareto family” plot / depth-tier heatmap | Retire invented post2017 winners, mismatched conditions and repeated fictional depth tiers. Use exact budgets and real local outcomes with conditions |
| Subjective recommendation-strength heatmap, fixed modern defaults | Replace arbitrary scores/defaults with task/device/evidence questions; no unsupported current frontier, mobile timing or mandatory architecture claim |
| Claimed universal transfer/data thresholds and medical exercise | Remove numerical promises and unsafe oversimplified medical prescription; replace resource-constrained educational task and explicit unknowns |
| VGG perceptual use and practical diagnostics | Preserve feature-loss mechanism and gradient-through-frozen distinction; no claim every generative model uses VGG |
| Old five exercises with exposed answers | Replace with seven changed tasks plus inline retrieval, closed independent hints/solutions, exact outputs and reasoned open-ended criteria |

## Canonical reference section-list audit

Actual canonical agenda inspected: [Stanford CS231n2017 Lecture9 slides](https://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture9.pdf), slide7/p6 and summary100/p99. Actual agenda: AlexNet,VGG,GoogLeNet,ResNet; also NiN,WideResNet,ResNeXt,StochasticDepth,DenseNet,FractalNet,SqueezeNet. LeNet review occurs slide8. Read the opening architecture/shape material and retrieved substantive selected later summaries; this is not a claim to have reviewed all101 slides or watched the full video.

| Headline | Coverage decision |
| --- | --- |
| LeNet review | Local refreshed hierarchy + historical fidelity§2 |
| AlexNet | Local architecture/training context and exact shape§3 |
| VGG | Local kernel composition/head accounting§3 |
| GoogLeNet | Local fanout/reduction/concat§4 |
| ResNet | Local identity refinement/projection/bottleneck§4 and prior residual depth |
| Network in Network | Local learned1×1 mixing and spatial-average head; historical precursor noted without a separate architecture chapter |
| Wide ResNet / ResNeXt | Brief distinction§7; width costs here, grouped-channel machinery in next topic; detailed architecture search/variants not core readiness |
| Stochastic depth | Existing complete Dropout/DropPath/StochasticDepth packet linked; do not duplicate its full training-mode lesson |
| DenseNet | Local growth/concat/transition explanation§7 |
| FractalNet / SqueezeNet | Intentionally outside this five-landmark-plus-bridges scope. The canonical lecture's optional model catalogue is broader than the current title; these are not required to understand a retained original mechanism. No misleading claim that all optional architectures are taught or that following one lecture agenda defines all CNN knowledge |
| EfficientNet/SE/MobileNet/RegNet/NFNet absent from2017 agenda or later revisions | Supplement actual primary papers, retaining relevant original useful depth |

The manuscript is organized around the learner's questions rather than copying the agenda or source wording. Optional reference material is never required to complete core explanation or practice.

## Hurdle map and learning sequence

| Learner hurdle | Explanation and worked bridge | Representation / learner action |
| --- | --- | --- |
| Width means pixels or channels? | NCHW local vocabulary; actual8×8 route | Map grids and explicit channel stack/shape strips |
| Parameters equal work or speed? | Shared weights reused across output positions; bias/MAC conventions | Exact counts; separate measured-result table |
| Small stacked kernels equal a larger filter? | Support derivation and intermediate nonlinear representation | Intermediate paths; changed-width counterexample |
| Several arrows mean the same merge? | Inception concat versus residual addition | Parallel channel lanes versus aligned-plus inset; changed merge practice |
| Efficient block is “ordinary conv but free”? | Spatial/channel factorization restriction and expansion/narrow skip | Explicit operator routes; next-topic bridge |
| An SE gate “looks at one pixel”? | Mean→hidden unit→sigmoid→broadcast chain | Editable map cells with nonlocal channel consequence and mean-preserving null |
| B4 equals exactφ4 scaling? | Idealized dw²r² budget versus rounded/mixed-operator architecture | Allocation contours, parameter/MAC distinction |
| Newer architecture guarantees better? | Same-data shallow skip counterexample and full seed table | Saved paired traces, actual tie, budget eligibility |
| Heatmap proves image explanation? | Algebraic CAM/logit identity, signed contributions, raw2×2 limits | Two calculation routes, changed class/feature and actual failure |

First-pass route follows introduction immediately. Detailed history and§7 are labeled optional. Cautions live with the relevant operation/evidence claim: floating-point under CAM, protocol under data/results, speed under budgets, interpretation under maps. The lesson does not repeatedly print cautions from programs.

## Research and claim ledger

Research date2026-09-13. These are primary papers/documentation; the LeNetPDF is an original-paper mirror because author-host requests failed. Record actual reading extent, not an implied full-paper review.

| Source / location | Reviewed extent and claim use |
| --- | --- |
| [LeCun1998](https://gwern.net/doc/ai/nn/cnn/1998-lecun.pdf),§II.B,pp2284–2285; introduction | Read extracted architecture details including C1/S2/C3/S4/C5/F6 and RBF, original-paper identity/introduction. Verify32 input,156 C1 parameters, learned subsampling/sparseC3/scaledtanh and end-to-end document context. No citation-count/deployment-volume claim needed |
| [AlexNet2012](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf),§§3–6/Table2 | Read substantive extracted architecture/result sections and table. Single18.2val versus five16.4 versus seven15.3test/extra pretraining. Historical geometry is explicitly Stanford teaching variant |
| [VGG](https://arxiv.org/pdf/1409.1556),§2/Table1 | Read configuration/layout and contextual small-filter discussion. Exact count additionally derived independently from layout and current library specification |
| [GoogLeNet](https://arxiv.org/pdf/1409.4842),§§4–5 | Read selected architecture details around bottlenecks/branches and original figure context; preserve input/branch/concat mechanics, no unverified current ranking |
| [ResNet](https://arxiv.org/pdf/1512.03385),intro/§3/Fig2 | Read selected formulation/projection/degradation sections; matched plain/residual meaning. Current small experiment is original teaching construction |
| [MobileNetV1](https://arxiv.org/pdf/1704.04861),§3; [V2](https://arxiv.org/pdf/1801.04381),§3 | Read core factorization/linear-bottleneck sections and selected application/ablation context. Full method-specific analysis is next topic |
| [SE](https://arxiv.org/pdf/1709.01507),§3 | Read intro and selected squeeze/excitation definition; current arXivPDF is v4,2019 expanded version of the work, not a claimed2017-only text |
| [EfficientNet](https://proceedings.mlr.press/v97/tan19a/tan19a.pdf),§§3–4,Tables1–4 | Read actual equations, coefficients, B0 search target and staged architecture/results protocol. PMLR PDF table differs from values repeated in older sources; manuscript avoids a mixed-version historical accuracy curve |
| [DenseNet](https://arxiv.org/pdf/1608.06993),§3 | Read selected growth/concatenation/transition definitions |
| [RegNet](https://arxiv.org/pdf/2003.13678),§3,Eq2–4 | Read quantized linear width construction and design-space discussion; no universalPareto claim |
| [NFNet](https://arxiv.org/pdf/2102.06171),§§3–4 | Read selected forward-scale/weight-standardization/AGC definitions and ablation context; keep these distinct |
| [CAM](https://arxiv.org/pdf/1512.04150),§2 | Read pooling/linear-classifier derivation; paper notation uses a spatial sum in displayed equation while manuscript explicitly keeps average and bias. Own exact and real calculations support local formulas |
| [Perceptual losses](https://arxiv.org/pdf/1603.08155),§3.2/Fig3 | Read substantive feature-loss/reconstruction description, not the full18-page paper. Optional application explains frozen feature gradient flow |
| [Torchvision models](https://docs.pytorch.org/vision/stable/models.html), [ResNet50](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html), [VGG16](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html) | Official current served docs identified0.29. Read weight/transforms/specification details. Earlier search-index snippets may show olderdoclabels; source directly opened for manuscript claims. Pin weight enum rather than treating DEFAULT as timeless |
| [UCI](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits), [sklearnload_digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) | Dataset/schema/licensing source and current1.9.1 API description checked; actual400-row uniqueness audited |
| [Stanford lecturevideo](https://www.youtube.com/watch?v=DAOcjicFr1Y) and official slides | Creator description/agenda plus substantive companion slides inspected. No fullvideo watching/timestamp claim. Suitable alternate intuition with explicit historical/API caveat |

No copyrighted figure is copied into a runtime asset; figures are specified from mathematical mechanisms and own examples. Numerical/history claims in the learner manuscript have near-claim links, while detailed derivations/calculations are original author work.

## Program, data and visual handoff

Required packet inputs:

- lesson.md — complete learner manuscript with setup, locally defined concepts, prose/captions, referenced complete program, worked CAM script and practice.
- visual-specifications.md — topic-specific inline structures and five investigations: head budget, channel context, scaling budget, recorded comparison, signed score map. Every investigation specifies input state, meaningful edits, immediate results, contrasts, nulls, accessibility and the model boundary.
- architecture-experiments.py — complete original CPU program; imports/classes/inputs/fitting/evaluation/trace/cost/actual visual-output saving. All12 fits actually executed.
- digits-400.csv and data-provenance.md — actual offline licensed inputs, selection/schema/split/uniqueness/evidence limits.
- calculated-inputs.json — actual12 fits, intermediate trace points, exact budgets, real features/maps/head/logits/probabilities.
- author-checks.py and author-check-results.json — bounded independent calculation checks; no second fitting campaign.

No separate fake “benchmark dataset,” no downloaded ImageNet model, no browser inference kernel, no run scripts in scratch, and no large source archive. The saved backbone is intentionally incomplete: recorded image outputs and full head support declared activities; arbitrary image inference requires the complete program to be rerun and outputs retained.

## Author review, checks and concrete revisions

The author reread the full final manuscript and visual specification as separate teaching/correctness tasks. This is an **author heuristic assessment**, not observed beginner testing or formal phase-two review.

1. **Route:** explicit immediately after introduction; optional history/additional-family branch labeled. Readiness does not require ImageNet reproduction, all historical dates or optional NFNet/RegNet.
2. **Cautions:** consolidate budget/latency, dataset/split and CAM interpretation in their homes. Displayed code prints only calculated outputs; program/check separation retains a readable mechanism.
3. **Real question:** actual handwriting recognition returns in the small-data experiment and its correctly/wrongly classified source records. Fresh400/400 vector/ID uniqueness check avoids a silent duplicate split.
4. **Investigations:** visible default results, meaningful feature/head/budget edits, exact categorical/numeric reference checks and synchronization of results with the active input are described. Actual greater/equal/less plain/residual cases across three seeds and gate/CAM nulls are recorded. Scope a below-bound scaling solution as infeasible, rather than clamping its value.
5. **Figures:** manuscript/spec reading covered each first-use structure and intermediate transformation; raw2×2 maps and exact budget values remain interpretable without decorative plots. Rendered desktop/mobile visibility remains explicitly deferred.
6. **Connections:** frozen convolution→architecture→depthwise sequence; add/concat/GAP/feature-loss and two CAM calculation routes explained. Canonical agenda decisions and intentional optional exclusions recorded above.
7. **Code:** complete program provides all imports/classes/data/fitting; small excerpts are clearly labeled as excerpts, while independent CAM code is complete. Hook accounting includes all invoked Conv/Linear layers. No backend benchmark or training-quality promise is printed.
8. **Practice:** seven changed tasks with closed hint/solution disclosures; independent exact43,911/903 head,36,864/25,600 kernel,1,152/73,728 projection, CAM0→−1, and budget membership. An additional inline channel-doubling answer was moved into closed disclosures during reread.
9. **Screenshots:** not captured; no implementation or browser exists at this phase. Required contrast/null/error/real-mistake/desktop/mobile states are recorded for phase two.

Correctness checks actually run:12 real fits once; cost hooks; exact VGG/branch/DWS/SE/scaling computations; fresh data uniqueness; independent meta-tensor head counts; direct-loop CAM reconstruction against80 saved class logits; changed signed-map and context-gate fixtures; eligibility endpoint checks. Float32 two-route CAM maximum5.722e−6; independent double-loop against saved logits maximum7.100e−6. Full results/conditions in provenance.

Revisions from author reread: corrected ambiguous early channel-doubling task to explicitly bias-free weights; closed its reveal; removed implementation-direction prose from learner captions; repaired prose/inline-code spacing without altering executable programs; labeled SiLU locally; clarified MAC exclusions and learned-feature versus edited-image boundary; corrected invalid scaling allocation behavior in spec. No fitted weights/results changed, so no fit rerun was necessary. **Historical interaction record:** the earlier prediction/reveal behavior described here is superseded by the 21 September live-exploration contract; it is not a phase-two implementation requirement. Preserve the recorded mathematical checks and fixtures.

## Current status and next action

Content research and writing: prepared, awaiting root source-bound reconciliation/checkpoint.
Implementation: not started.
Computational verification: bounded author calculations/program outputs as recorded; no formal native implementation campaign.
Browser/visual review: deferred/not started.
User review: not yet performed on this new packet.
Open content concern: none identified by author; root may report scoped findings.

Root reconciliation, 13 September: added the existing module predecessor as a contextual link and replaced the next-topic draft-file link with the real full-curriculum route, both retaining module=deep-learning-fundamentals. This bounded navigation repair changes no calculations, data, visual behavior or evidence; author checked stable IDs against the assigned sequence.
Next action: root reconcile/checkpoint; implementation owner later translates exact visual contracts and complete manuscript into existing topic/lab structure, runs required functional/numerical/browser/build checks and records phase two separately. Author proceeds to the next assigned depthwise/dilated packet after root handoff.

## Live exploration revision — 21 September 2026

The user replaced prediction-and-reveal lab assessment with immediately visible, playable exploration, including removal of optional prediction controls. The manuscript and specifications now use that contract. This changes the teaching interaction, not the topic's model predictions or its mathematical masks/gates.

**Topic disposition:** Compare architecture operations and budgets. Edit head dimensions, channel-context cells, scaling allocations, deployment budgets and signed score-map weights. Show exact parameter/MAC counts, gate contributions, candidate eligibility and current CAM/logit arithmetic live. Recorded model/seed selectors display existing evidence immediately. Identify which operation consumes the budget, what information a head discards and why a smaller model is not automatically better.

Retain all source data, formulas, measured results, code programs, references and independent practice. The existing author calculations remain evidence for those unchanged quantities, not evidence that a browser implementation already satisfies this new interaction contract. Phase two must implement and verify live updates, linked-view agreement, bounded work, reset, invalid/null cases, keyboard access and responsive diagrams. Content remains prepared; implementation remains not started.


## Implementation-depth writing revision — 22 September 2026

Delivery mode: **content first**. The mechanism and ordinary-tool teaching below is written now; it is not an instruction for the finishing agent to invent missing content. Existing measured experiments and their historical evidence remain unchanged unless explicitly stated. The current manuscript section “Turn the architecture diagram into a complete model” gives the learner route.

| Computational outcome | Scratch owner and abstraction | Ordinary tool and matched comparison | Control / practice and boundary |
| --- | --- | --- | --- |
| Construct title families | `landmark_builders.py:lenet, alexnet, vgg16, ResidualBlock/resnet18, MobileBlock/efficientnet_b0`; explicit complete composition | Torchvision get_model and explicit weight enums; component state mapping, metadata/default/meta-mode inspection | GAP seven-class VGG modification with parameters/shape solution; original-paper vs maintained-model differences declared |
| Parallel branches, channel gating, training and CAM | `architecture-experiments.py:ParallelBranches, InvertedGated, SmallClassifier, exact_examples, main` | Ordinary PyTorch training; transfer policy reuses actual published transfer-experiments.py section3 | Existing branch/gate/CAM exercises retained; historical ancillary families remain mechanism context |

All local filenames in the map are retained draft sources beside this design. A linked prepared prerequisite is not yet the improved published page: finish in module order or carry its declared source with the lesson. Already implemented autograd/loss/normalization/tensor lessons may be reused as stated; no new differentiation engine, BLAS or convolution backend is implied. Optional historical families remain explanations of a distinction unless a local implementation is explicitly named.

The content packet is ready for phase-two construction after central source checkpointing. Finishing must execute the supplied comparisons on declared compatible versions, resolve any observed numerical/convention differences, expose the exact code/downloads, and verify rendering, live controls, accessibility and production loading. Unexecuted optional package/GPU/checkpoint examples remain explicitly unexecuted; do not print invented outputs or copy previous measurements onto new code.

## Prepared-content implementation — 22 September 2026

Current state: **author implementation complete; independent review in progress; parent production integration pending**. The historical content-first statements above remain a record of the prepared phase, not the current runtime state. The author preserves the entire revision-3 manuscript/specification and its original measured inputs. Central phase/registry/build ownership remains with the parent.

### Implemented route and ownership

The stable topic source now contains all thirteen main sections, eighteen closed practice hints/solutions, complete named-family construction guidance, ordinary-library/pretrained distinctions and the actual small-network investigation. `scripts/generate-landmark-architecture-lesson.mjs` statically renders the conserved manuscript, translates relative draft-topic links to real routes, removes placement comments, inserts the specified visuals and replaces only the now-executed comparison status. The full manuscript was not compressed into an outline.

`LandmarkArchitectureLabs.jsx` and `landmark-architecture-models.js` implement five live investigations plus a separately opened sixth recorded-map view. They expose actual head dimensions and exact biases/MACs; editable channel cells and broadcast gates; aligned parameter/compute bars and an honest infeasible allocation; fixed-cost eligibility and observed training traces; and signed CAM contributions with both arithmetic orders. Source identity, changed variables, null results and evidence limits remain visible. No learner prediction-entry or reveal grading exists.

Inline figures show channel/spatial shape routes, historical LeNet fidelity, two distinct sampled grids, selectable intermediate kernel paths, concat/add branch lanes, and a narrow skip around expansion/depthwise/gate/projection. The kernel interaction maps one of nine intermediate positions to its actual 3×3 patch in a 5×5 input union. Scoping keeps all new surfaces neutral charcoal with amber, rather than inheriting old green styling. Numerical tables, visible signs and separate input/output labels remain available without color interpretation.

Browser measurements are compact (20,673 bytes before compression), topic-local and exactly derived from the native packet. The heavier saved image/features/head observations are a separate JSON fetched only when their disclosure opens. Both complete Python programs also fetch only on disclosure. The original 12-fit training loop, large model constructors and pretrained downloads never execute in the browser. No shared CSS, core runtime registry, generated catalogue or phase ledger was edited by this author.

### Actual native execution and retained evidence

The full `architecture-experiments.py` was executed once in the public asset directory. All twelve fits, checkpoint values, model costs, feature/head outputs and exact fixtures match the conserved JSON exactly; producer and CSV bytes are unchanged. The fixed split still supplies development evidence only.

Torchvision 0.29.0 was installed with `--no-deps --target scratch/landmark-architecture-deps`; PyTorch/shared packages were not modified. Each following CLI comparison was executed individually, with the package target added to `sys.path`, one CPU thread, seed9, a randomly generated input, `weights=None`, copied Conv/Linear/BatchNorm state, both models in evaluation mode, and no download. The original command constructs the model specified by `--family`:

```text
python landmark_builders.py --family resnet18 --compare
python landmark_builders.py --family efficientnet_b0 --compare
python landmark_builders.py --family vgg16 --compare
python landmark_builders.py --family alexnet --compare
```

Actual printed maximum differences were respectively **0.0, 0.0, 0.0, 0.0**. The VGG and AlexNet processes were individually launched and terminated before the next family; this was not parallel allocation of all models. This checks matched computation, not historical training or pretrained quality.

The complementary author verifier `scripts/verify-landmark-architecture-native.py --library-path scratch/landmark-architecture-deps` passes six substantive groups and records exact source hashes at `docs/teaching/evidence/landmark-architecture-native.json`, also downloadable as `native-verification.json`. It checks all five full meta constructors and copied component state shapes against Torchvision; changed seven-class GAP VGG at 64/96 pixels; branch-drop probability boundaries and gradients; all80 saved CAM scores through independent scalar loops; and the standalone hand-sized CAM code. Actual parameter counts: LeNet variant61,706; AlexNet61,100,840; VGG16 138,357,544; ResNet18 11,689,512; EfficientNetB0 5,288,548. Float64 re-accumulation of stored float32 values differs from saved logits by at most7.099608438920768e−6; this is separate from the original float32 reorder fixture's5.7220458984375e−6.

`node scripts/verify-landmark-architecture-models.mjs` passes seven substantive groups: complete source/JSX and practice preservation, every compact/deferred empirical field, three head modes and exact nulls, actual context cells and permutations, coefficient/scaling/infeasible arithmetic, budget boundary/seed invariance, and all80 class-score reconstructions plus changed signed fixtures. Report: `docs/teaching/evidence/landmark-architecture-author.json`. Source hashes bind the final model/lab/generated body. Original author checks need not be rerun over unchanged historical evidence.

### Review changes already addressed

Independent reviewer `/root/decision_depth_prose` identified two mechanism figures that initially remained too textual. The author added the sampled-window/selected-intermediate route and aligned shared-zero budget bars with a labeled target-MAC marker. The reviewer also demonstrated that customizing `MobileBlock` to drop_probability1 produced nonfinite values. The canonical public source now accepts [0,1], rejects out-of-range/NaN values and uses `inputs + correction * 0` at training probability1, preserving a finite identity route and zero branch gradients without division. Default B0 uses rates below1 and is unchanged. This is an explicit canonical implementation correction; the frozen draft's source remains intact. Minor copy/aria spacing repairs distinguish retaining an input from allocating an extra physical copy.

The author reviewed the full generated article, program flow, remaining optional branches, point-of-use relationships, source conservation and all data/constructed-model boundaries. Independent review is separate and still being finalized. The official versioned Torchvision0.29 EfficientNet page was checked for the compatible model/weight contract. No pretrained photograph/weight example, full historical ImageNet fit, hardware latency or arbitrary-image browser inference is claimed.

### Parent integration and cleanup handoff

`scripts/verify-landmark-architecture-browser.cjs` is parsed and ready for the single parent-owned production build; it installs the shared font fixture, checks genuine pointer/keyboard changes, each default/change/null/reset/invalid state, actual saved mistake and competing class, program/map demand loading, theme/math geometry and desktop/390/320 captures. It has not been executed by this author. Parent must inspect the retained images, close independent findings, bind final browser evidence and update the phase ledger.

Keep the isolated `scratch/landmark-architecture-deps` until the independent reviewer finishes native access, then resolve and remove only that exact temporary directory inside this workspace. Its installer-owned files need the host permission context. The full public programs, measured JSON, dataset/provenance and referenced author/review evidence are durable outputs, not cleanup candidates. No separate obsolete scratch manuscripts or screenshot sets were created by this author.

### Phone integration repair — 22 September 2026

The parent production browser exposed five pixels of page overflow at320px: the fixed150px intermediate grid sat inside a135px flex stage. The topic-scoped route stage now has a170px basis and the small-grid rule caps width at its container, allowing stages to wrap naturally. This does not clip the grid or change any convolution support. A targeted dev4195 check passes document and intermediate-grid containment at1366/390/320px with the actual-map disclosure and both complete program disclosures open. The 320px composition figure was visually inspected. The scoped author browser script now asserts containment of the grid inside its stage, supplementing page-width checks. All seven source/model groups pass with refreshed hashes; native evidence is unchanged. Independent scoped rebinding and final production browser confirmation remain parent/reviewer work.
