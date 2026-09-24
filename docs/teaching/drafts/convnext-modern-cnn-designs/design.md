# ConvNeXt & Modern CNN Designs — content design and continuation

## Phase, identity and baseline

Research and writing only, revision1,13 September2026. Author deep_foundations_content owns this stable-ID packet. Module deep-learning-fundamentals, position12; third in the current next30 scope. Root owns shared ledger, inventory, scope reconciliation and final checkpoint hashes. Implementation is not started. No production JSX, catalogue, navigation, browser code, SVG or publication was changed.

Actual preflight completed before writing: node scripts/build-curriculum-inventory.mjs --topic convnext-modern-cnn-designs --work content. Returned content-first in progress, implementation not started, no topic-specific destination note; unrelated resolved historical note does not create a queue.

Complete original read: src/learn/data/topics/convnext-modern-cnn-designs.jsx, ordered segments0–210,210–430,430–660,660–910,910–1160,1160–end covering prose, programs, labs, charts, exercises and references. Original SHA-256996fbe6ba48f0ca90b434785fbc7e10b908c2dad37e830b1cb08f71365ab7b0a. Baseline commit8c5da59f18516be77c29d5aeeafca3decca4f738. Original remains untouched; existing publication was not a correctness endorsement.

Current AGENTS/handoff/teaching standard/domain/design/code ownership, coordination and retention instructions read during this authoring run. Previous Convolution, Landmark and Depthwise packets are known actual prerequisite coverage; current lesson refreshes tensors, convolution types, normalization and residual masks locally. Actual route links connect previous Depthwise/Dilated, relevant Dropout/Transfer and next Capsule, all within module deep-learning-fundamentals.

## Learning contract, scope and title decision

Retain stable ID and catalogue title. “Modern CNN Designs” legitimately includes ConvNeXtV1/V2, the masked-learning co-design question, structural branch fusion and limited hybrid mechanism comparisons. Renaming to list every adjacent architecture would hide the principal learning thread.

Learner can trace a complete ConvNeXt block and hierarchy, calculate its exact small/large shape budgets, identify normalization reduction axes, reason about GRN's coupling and initialization, protect masked-input information, run a real small reconstruction-plus-probe experiment and separate supported observations from untested explanations. Advanced routes cover checkpoint/layout contracts, exact inference-time branch folding and purposeful hybrid locality.

First pass appears immediately after introduction: §§1–6, actual example and practice1–5. Optional§7 and practice6–8 do not silently become the core readiness gate. A beginner need not remember ImageNet rankings, implement a native sparse runtime, train a large network or know the later self-attention lesson to follow the main thread.

| Scope question / inspected ownership | Decision and location |
| --- | --- |
| Spatial/channel factorization already developed by previous Depthwise lesson | Refresh enough to read the block; new ownership is ordering, axes, topology and architecture/recipe reasoning |
| V1/V2 normalization and masked co-design absent from prior primitives | Own complete local equations, gradient example, visibility contract and actual applied program, §§2,4–6 |
| Transfer learning already owns full supervised adaptation | Link actual topic; own feature-output/checkpoint interface and frozen-probe question here |
| Dropout packet owns Bernoulli masks and execution semantics | Link actual topic; retain local mask shape/expectation/no automatic compute savings |
| Kernel reparameterization is meaningful modern-CNN extension in old source | Preserve with exact fold and nonlinear counterexample, optional§7/figure6 |
| CoAtNet/MaxViT need attention before their dedicated later lessons | Explain input-dependent weighted aggregation locally only as a comparison; full attention remains at actual later owners |
| D2L design-space/NAS implications | Annotated alternate route; do not reopen dedicated NAS/AutoML notes already owned by root |
| Dense/sparse normalization parity | Record source discrepancy below for any later native reproduction; current local contract is dense per-image GRN and no parity claim |

No new topic, rename or cross-owner destination-note request is necessary for this packet. If future implementation adds a native sparse reproduction, it must investigate its normalization axes and masking parity rather than inherit an assumed equivalence.

## Conservation and corrected claims

Preserved useful old depth: historical experiment logic; all V1 block operations and V2 change; stage counts/widths; bias-aware parameter/MAC derivation; full model code and feature outputs; LayerNorm axis/layout implementation; LayerScale/DropPath initialization and mode; GRN/feature-collapse question; FCMAE masking/decoder/target; comparative evaluation; deployment contracts; large kernels, branch fusion and hybrid designs; independent changed practice and annotated resources. Replaced generic fabricated traces and large unexecuted training skeletons with full small executed programs, saved real inputs and model states.

Important repairs:

- Original modernization interpretation attributed fixed bonuses to changes and treated depthwise replacement as neutral. Primary small-regime table shows a drop followed by widening recovery; intermediate cost budgets vary. Final roadmap81.97±.06 and reported final-model82.1 are different records. No “read any vision paper in15minutes” or universal2026 architecture default.
- Avoid confusing different Liu authors in Swin/ConvNeXt or claiming every ingredient originated with Transformers. The mechanism lineage includes prior CNN designs.
- A depthwise spatial mixer is not the whole attention layer: attention includes channel projections and input-dependent coefficients; Swin's full residual/window topology is not isomorphic to one ConvNeXt block.
- V1 coefficient58C, not57C. V2 difference+7C; atC96 extra672 comes from768added minus96removed. Pointwise weights about93%, not99%. Larger V1 variants3,3,27,3 have36blocks, not54/60.
- MAC counts explicitly exclude non-convolution/linear work. Stage per-block cost, not an undocumented FLOP estimate, explains stage allocation. DropPath masks computed branch contributions and does not automatically save MACs.
- Correct order: stemConv→LN; transitionLN→Conv; final spatialaverage→LN→head. Reference full-model initialization is truncated normalstd.02 from official code; the PDF Table5 extraction says.2, so code-level convention governs this supplied implementation. Full program initializes; meta counts skip allocating/initializing large tensors.
- Dense GRN norms reduce over spatial axes per image; channelLN reduces over channels per location. GroupNorm(1,C) is not a drop-in replacement. InitialGRNidentity does not imply zero parameter gradients or tinywholeV2block.
- Learned GRNgamma can be negative. Nonzero channel variance does not establish nonredundancy; cosine distance and actual task performance remain separate measurements.
- FCMAE paper's60%masking applies32×32inputpatches at final-grid granularity, not4×4stempatches. Hiddentargets never enterencoderpreprocessing. Sparse coordinate computing differs from dense numerical masking and nohardware speedupisassumed.
- PaperV2Table3 ablation values differ from laterTable14 fullresults/trainingdurations. A comparison across pretraining, labels, resolution and recipes does not isolatearchitecture.
- Logicalpermuteview versus channels_lastmemoryformat is explicit; downstreammaterialization possible. No blanket batchsize cutoff forBNfailure or universalchannelslast speedup.
- RepLK's main5×5 parallel branch is distinguished from its exploratory3×3 examples. Exact inference folding requires compatible linear branches and fixed BN statistics. CoAtNet's early convolutions operate at higher resolution and later attention at lower resolution. No universal larger-kernel or categorical SE-useless claim remains.

## Canonical-reference coverage audit

Audited actual section lists of both original ConvNeXt papers, not an invented generic outline. The current topic owns the following decisions. “Read” means the retrieved section text/table content; reference bibliographies are locators, not a claim to have read every cited work.

### A ConvNet for the 2020s, arXiv2201.03545v2,2 March2022

| Actual canonical section | Review extent / decision | Manuscript home |
| --- | --- | --- |
|1 Introduction|Read motivation and architecture/recipe confounding; historical framing contextualized rather than copied leaderboard narrative|§1|
|2 Modernizing a ConvNet: a Roadmap|Read whole framework andFigure2; preserve conditional trajectory and non-monotonic steps|§1,experiment genealogy|
|2.1 Training Techniques|Read complete; enhanced recipe distinction and three-seed averaging retained; fullImageNet recipe not required for local experiment|§1/design|
|2.2 Macro Design|Read stage ratios and patchify; explain local shapes and budget rather than values to memorize|§3|
|2.3 ResNeXt-ify|Read depthwise/widening sequence; previous lesson owns factorization derivation|§§1–2|
|2.4 Inverted Bottleneck|Read and checked earlier claimed improvement against actualsource; tensor meaning/parameter count retained|§2|
|2.5 Large Kernel Sizes|Read entire move/filter-size discussion; conditional saturation and prior CNN history|§§1–2,7|
|2.6 Micro Design|Read complete activation/norm/downsample sequence; exact final orderingchecked against officialsource|§§2–3|
|3 ImageNet Classification,3.1Setup,3.2Results|Read configs/training settings and table conditions; preserve comparison hygiene, not reproduce completeleaderboard|§§1,3,7|
|3.3 Isotropic ConvNeXt vsViT|Read actual18/18/36depths and unchanged-grid design; briefblockversushierarchy contrast sufficient; no extrasmalltrainingcampaign|§3|
|4 Downstream Task Evaluation|InspectedCOCO/ADE20Ktables and accompanyingcontext/featurehierarchy; fulldetector/segmentertrainingbelongselsewhere|§3featureoutputs,§7application|
|5 Related Work;6Conclusions|Read; architecture choices/taskfitness and CNNantecedents retained; noSOTA timelessclaim|§§1,7|
|AppendixA Experimental Settings(A.1/A.2/A.3)|Read training/finetune/downstream recipe tables and EMA exception; publiccodeinitialization difference noted; no fullImageNetrun|design/provenance|
|B Robustness Evaluation|Read metrics, table and conditions; do not infer digit robustness or plot unmeasured corruptions|§7 evaluation contract|
|C Modernizing ResNets detailedresults|Read bothTables10/11; genealogyusesTable10actualsmallregime, conditionalstepdifferentregimesexplicit|visual§1|
|D DetailedArchitectures|ReadTable9/context; completeownparameter/shapeprogram|§3/program|
|E Benchmarking onA100GPUs|Read hardware/version/TF32/channelslast context; donottransplantthroughputnumbers|§7storage/measurement|
|F Limitations;G SocietalImpact|Readtask-choiceconstraints and resourcecontext; boundedCPUroutechosen; fullresource/ethicalcurriculumunnecessaryhere|§§6–7|

### ConvNeXtV2: Co-designing and Scaling ConvNets with Masked Autoencoders, arXiv2301.00808 / CVPR2023

| Actual canonical section | Review extent / decision | Manuscript home |
| --- | --- | --- |
|1Introduction;2RelatedWork|Readresearchquestionandmaskedmodelingcontext; no current“allvisiondefaults”claim|§§4–5|
|3FullyConvolutionalMaskedAutoencoder|Readentiremasking,encoder,decoder,target,andTable1ablation; exactinformationboundaryretained;explicitlocaldeviations|§§5–6|
|4GlobalResponseNormalization|Readfeature-collapseanalysis,Eq1–4/Algorithm1 andTables2a–f; denseofficialsourcecrosscheck; addactualgradient/coupling/null|§4|
|5ImageNetExperiments|Readco-designTable3,pretraindurationTable4,modelscaling/22Kintermediatefinetuning; no conflation with frozenprobe|§5/design|
|6TransferLearningExperiments|ReadCOCO/ADE20Kconditions andTables6/7; knownstagefeatureheadpurpose, fullsystemsdeferred|§§3,7|
|7Conclusion|Read; task-evidence framing rather thanarchitectureuniversalism|§9|
|AppendixAImplementationDetails|Readmodelconfiglist;pretrain/fine-tuning/nativesparse-denseoverview and relevantsettings; no nativeMinkowskiinstallation|provenance/nativeboundary|
|BComparisonsonImageNet|ReadTables14/15; note distinctionsfromTable3/1600pretrainconditions; noneportedtoourresults|design|
|C Further Analyses|Read sparse-efficiency conditions and class-selectivity explanation; distinguish channel activity from diversity and avoid borrowed timings|§§4,6–7|
|DAdditionalExperiments|ReadGRNcomponentTable16,maskratioFigure8,MoCoV3comparison; optimalratioisempirical, local62.5%declaredratherthancallingpaper60%|§§5–6|

Canonical coverage does not require repeating every experiment or its prose. The section audit locates deliberate scope decisions, mathematical ownership and omitted unrelated training campaigns.

## Source locators and actual alternate-resource review

All links verified/retrieved13 September2026. Learner links use direct pages, not search result URLs. Explanations and constructed examples are original; sources support mechanism/version/claim checking rather than supply a copied article structure.

- [V1paper](https://arxiv.org/pdf/2201.03545),15pages: section audit above; consequentialroadmapFigure2/AppendixCTables10–11,architectureTable9,finalTable1,settingsTables5–6,hardwareTable12.
- [Official V1 code](https://raw.githubusercontent.com/facebookresearch/ConvNeXt/main/models/convnext.py): read Block, full model, LayerNorm, variant definitions, initialization and forward/GAP order. The packet provides a pedagogical implementation with independent names and interfaces, not official packaging.
- [V2paper](https://arxiv.org/pdf/2301.00808),15pages: aboveaudit,mask32patchdefinition§3;GRNAlgorithm1;Tables2/3/14comparability;AppendixCnativeefficiency;AppendixDmaskratio.
- [Official V2 model](https://raw.githubusercontent.com/facebookresearch/ConvNeXt-V2/main/models/convnextv2.py), [dense/sparse utilities](https://raw.githubusercontent.com/facebookresearch/ConvNeXt-V2/main/models/utils.py): read full utilities, block, full architecture and configurations. Dense GRN uses a spatial norm per image in NHWC. The current MinkowskiGRN source takes norm(x.F,dim0) over all active feature rows, crossing batch specimens unless handled by additional grouping elsewhere. The packet's dense contract does not claim native parity. A native reproduction needs actual batch-coordinate and statistic inspection.
- [RepLKNet](https://arxiv.org/pdf/2203.06717): readreparamguidelineandlargekernelblocks§§3–4,5×5parallelmainbranchversus3×3exploratorycase; optionalmechanismlink, noentire16pageclaim/no copieddevicebenchmark.
- [MobileOne](https://arxiv.org/pdf/2206.04040): read the abstract and §3.3 overparameterization/reparameterization mechanism. Device claims were not generalized; the complete training discussion was not read.
- [CoAtNet](https://arxiv.org/pdf/2106.04803): read§2stacking/design/arrangementcomparison tofixreversedresolutionclaim; optionalfutureattentionbridge.
- [MaxViT](https://arxiv.org/pdf/2204.01697): read§3block/gridmechanism,architecturediagramdescription,orderingablationtables andappendixpseudocode; notall31pages. Optionalconceptualextension, no maxvitprogramclaimed.
- [TorchvisionTiny documentation](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.convnext_tiny.html): readfullAPI/weightmetadata/transforms; DEFAULT→IMAGENET1K_V1,82.52modifiedrecipe,28589128params,236resize→224crop andImageNetnormalization. Onlycontractinstructions, no installednativeexampleclaim.
- [PyTorchchannelslast tutorial](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html): readshape/strideconversion andsupported-op/layoutprovisions; usefulhands-onalternateroute, historicalhardwaremeasurements notours.
- [D2L1.0.3networkdesign chapter](https://en.d2l.ai/chapter_convolutional-modern/cnn-design.html): reviewedintro,AnyNet/RegNetsectionstructure/design-spaceargumentanddiscussion; annotationwarnsitsrankingsaretime-specific. Notalllinkedbookchapters/codeframeworkvariantswereexecuted.
- [Authors' CVPR2023 slides](https://cvpr.thecvf.com/media/cvpr-2023/Slides/22892_lw8881R.pdf): read all32 slides' extracted text, including the incremental mask construction, GRN and co-design. Screenshot for page17 failed with a cache miss;24/25 returned references without model-visible images. No complete visual-inspection claim is made. This is a verified author slide alternative; no video was claimed watched. Search did not establish a reviewed official video, and no format quota requires one.

No clinical advice or unsupported neural-modelmedicalefficacyappears. Thefactoryexample is a proposed evaluation scenario, notreporteddeploymentorrealcollecteddata. Realdata provenance appliesonlytoopticaldigits.

## Hurdle-to-teaching map

| Learner hurdle | Explanation / representation | Independent transfer and boundary |
| --- | --- | --- |
| Newernameequalsbetterarchitecture | Architecture/recipe/checkpointdefinitions,experimentgenealogy | Interpretconditionedcomparison; practice5 |
| Channelsandlocationsmixedup | Blockshape ribbon, explicit localvectorformula, editableaxisreductionlattice | Shape-correctLNbug, practice1;three-channeledit/shift-null |
| “Depthwisecheap”meanswholeblockcheap | Bias-awareexactcounterandstagegrid | Changedkernel/expansion/resolution,practice2/7 |
| Normsallstandardizethesamething | LNperlocationversusGRNspatialnorm/sharedchanneldenominator | Cross-channelentityedit,zeroγnull,practice3 |
| Identityinitializationmeansnolearning | Exactfirstgradientandparameter/inputderivativedistinction | Closedoptionalgradientbranch |
| Maskedinputleaksitsanswer | Two-pathsource/targetdiagram,actualhiddeneditnull/visibleeditcontrast | Full-imagemeanleakdiagnosis,practice4 |
| Reconstruction/diversityprovesrecognition | Actualpairedsixfits,rawprobe,twoseparatescoreplots | Preserveunfavorablebaselinecomparison;pretrainingattributiongap,practice5/8 |
| Traininggraphmustremaininferencegraph | Alignedkernelstencil/BNfoldalgebraandnonlinearcounterexample | Editablefusion,practice6 |
| A modern CNN must be a universal choice | Hierarchy/isotropic/hybrid connections tied to the task unit | Grouped factory scenario and optional advanced comparison |

## Author checks, learning-experience pass and stopping boundary

Actual calculations and environment are detailed in data-provenance.md and the three programs/results. Six bounded fits completed without rerunning to improve an unfavorable outcome. Checked independent NumPy forwards, GRN central-difference gradients, normalization edits/nulls, masked-pixel and mask-swap contrasts, and inference-folding algebra. Meta full models count parameters without claiming large executed inference. A serialization error was corrected and the final JSON saved. No disposable scratch artifacts were created.

The author reread the full manuscript in ordered sections0–165,165–325,325–end. Repairs included numeric spacing, an unused vector name, the distinction between79,968 total and672 additional parameters, and brief isotropic coverage. Practice includes16 closed hint/solution disclosures and changed problems. The complete visual specification was reread for controls, fixtures, state, feedback and information boundaries. This revealed a weak two-channel contrast; a checked three-channel edit/null and actual patch-swap calculations were added without repeating fits. A final readability pass separated compressed handoff prose into ordinary sentences.

Learning-experience checklist:

- Intuition and motivation precede terminology; spatial/channel roles are explicit.
- First-pass and optional advanced routes appear near the start and readiness check.
- Tensor, statistic and probe concepts are taught locally; prerequisite links supplement explanation.
- Consequential operations and counts have calculations or primary locators; original false claims are corrected.
- The actual problem, data, features, targets, split, baseline, program, results and interpretation form a coherent example.
- Representations follow the need: genealogy, shape/axis ribbon, spatial grid, shared denominator, masked information paths and kernel fusion.
- Genuine entity edits, immediately computed input-bound results, result checks, contrasts/nulls, feedback, reset, invalidation, mobile layout and text accessibility are specified.
- Practice changes constraints and conceals hints/answers; advanced readiness is separate.
- Alternate resources record review extent and version limits; no invented video viewing or benchmark claims.
- The complete pending packet uses semantic filenames and preserves curriculum identity, order and previous work.

Scoped checks passed: all JSONs parse, all three Python files parse, every local manuscript file link resolves,16 disclosure pairs are closed, and the original source hash is unchanged. Root reconciliation binds final checkpoint hashes. This author pass is not formal independent phase-two review. Later finish requires its topic preflight, the complete packet, visual/browser-model implementation, affected native-code checks, independent technical/learning review, accessibility/mobile checks, loading/error recovery and production integration. Content completion is not publication or user acceptance.

## Live exploration revision — 21 September 2026

The user replaced prediction-and-reveal lab assessment with immediately visible, playable exploration, including removal of optional prediction controls. The manuscript and specifications now use that contract. This changes the teaching interaction, not the topic's model predictions or its mathematical masks/gates.

**Topic disposition:** Inspect modern convolution blocks. Change stage dimensions, normalization groups, GRN feature cells, valid visible-patch selections and branch-folding coefficients. Show parameter counts, shared GRN denominator, changed feature maps, reconstruction consequences and folded-kernel equality immediately. Preserve image masking as the learning objective, not UI answer hiding. Separate architecture from recipe, global channel context from local normalization, and valid reparameterization from a changed function.

Retain all source data, formulas, measured results, code programs, references and independent practice. The existing author calculations remain evidence for those unchanged quantities, not evidence that a browser implementation already satisfies this new interaction contract. Phase two must implement and verify live updates, linked-view agreement, bounded work, reset, invalid/null cases, keyboard access and responsive diagrams. Content remains prepared; implementation remains not started.


## Implementation-depth writing revision — 22 September 2026

Delivery mode: **content first**. The mechanism and ordinary-tool teaching below is written now; it is not an instruction for the finishing agent to invent missing content. Existing measured experiments and their historical evidence remain unchanged unless explicitly stated. The current manuscript section “Match the block you built to the maintained implementation” gives the learner route.

| Computational outcome | Scratch owner and abstraction | Ordinary tool and matched comparison | Control / practice and boundary |
| --- | --- | --- | --- |
| ConvNeXt V1/V2 block, hierarchy, GRN and masks | `convnext-blocks.py` complete mechanism and model; `masked-reconstruction.py` complete trained task | `convnext_library_bridge.py:compare_block`; exact V1 CNBlock state/layout/gradient mapping | Expansion2 changed-control exercise; V2 is local GRN, not false CNBlock equivalence |
| Ordinary checkpoint application | `classify_image` with fixed enum/transform/output categories; no claimed scratch pretrained training | Torchvision convnext_tiny, Pillow, inference_mode | Actual implemented Transfer Learning owner for adaptation; optional download/application unexecuted |

All local filenames in the map are retained draft sources beside this design. A linked prepared prerequisite is not yet the improved published page: finish in module order or carry its declared source with the lesson. Already implemented autograd/loss/normalization/tensor lessons may be reused as stated; no new differentiation engine, BLAS or convolution backend is implied. Optional historical families remain explanations of a distinction unless a local implementation is explicitly named.

The content packet is ready for phase-two construction after central source checkpointing. Finishing must execute the supplied comparisons on declared compatible versions, resolve any observed numerical/convention differences, expose the exact code/downloads, and verify rendering, live controls, accessibility and production loading. Unexecuted optional package/GPU/checkpoint examples remain explicitly unexecuted; do not print invented outputs or copy previous measurements onto new code.
