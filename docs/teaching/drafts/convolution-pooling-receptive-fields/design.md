# Convolution: content design and handoff

Stable ID convolution-pooling-receptive-fields. Module deep-learning-fundamentals, position9; authorized batch position30. Owner /root/deep_foundations_content. Research/write packet prepared for root reconciliation; implementation not started.

## Scope, prerequisites and original conservation

Ran topic inventory with --work content and consumed returned notes. No destination note exists; unrelated resolved bit-manipulation routing history is not an obligation. Current handoff, teaching standard, design brief, model domain, ownership/coordination and retention policies were read for this continuing range. No catalogue-wide audit or rewrite is inferred.

Read the entire original JSX in contiguous ranges 0–220,220–470,470–720,720–990,990–1280,1280–end, including the worked programs, plots, references, tables and solutions. Baseline commit 8c5da59f18516be77c29d5aeeafca3decca4f738; original src/learn/data/topics/convolution-pooling-receptive-fields.jsx SHA-256 3c4cdb5c82d53eb30845151150469bdffd15d568dd3acee9173002eae5479f3f. Runtime unchanged.

Title and stable ID remain appropriate. Earlier perceptrons/backprop/loss lessons own full neural/gradient/objective foundations; this page refreshes those locally through a shared-filter update before the image experiment. Normalization/residual/dropout are earlier, so fixed BN folding, branch alignment and dropout-before-pool consequences can connect without making them first-pass prerequisites. Basic multiplication/indexing suffice to enter the lesson; deeper adjoint/ERF/engineering branches are explicitly optional.

Actual next topic is landmark-architectures-lenet-alexnet-vgg-resnet-efficientnet, then depthwise-separable-dilated-convolutions. Verified the exact landmark ID with scoped inventory and read current track order. The original ended with incorrect next-topic suggestions. Do not skip the architecture owner. No renamed topic or new catalogue entry is needed.

| Original coverage | Decision and preserved teaching home |
| --- | --- |
| Long visual-cortex/history chronology and named architecture survey | Short local motivation; detailed evolution belongs to immediate Landmark Architectures. Drop unsupported dates/neuroscience analogies rather than duplicating that owner's survey |
| Locality, sharing, input/output channels, bias | Core patch/channel construction, complete multi-channel equation, explicit batch versus parameter axes |
| Cross-correlation versus flipped mathematical convolution | Precise convention before code, fixed-filter interpretation retained |
| Stride, padding, dilation and same/full arithmetic | General asymmetric formula, exact centers and boundary meaning; “full” preserved through canonical comparison and optional transpose rather than an isolated second formula catalogue |
| Groups, depthwise and pointwise | Correct local connection map/multiplier and operation budget; full design tradeoffs remain the later specialized owner |
| Theoretical RF and effective RF | Separate exact ancestor set/bounding region/jump/center, holes and clipping; exact linear coefficients versus actual input-dependent gradient maps |
| Direct NumPy convolution, im2col, backward | Complete direct grouped program, five parity cases, actual Unfold/Fold data and a fully explained shared-gradient/adjoint example |
| Max, average, adaptive, global pooling | Local winner/distributed gradients, overlapping bins, padding denominators, nonunique reconstruction |
| Production CNN block and untrained synthetic snippets | Replace broken-channel example with complete real-data CPU train/evaluate/inspect program; every model input, class label, split, metric and output supplied |
| VGG parameter/FLOP table | Preserve transferable parameter/MAC calculation on actual fitted architecture; detailed VGG variants/FC head budgets belong to immediate architecture owner |
| Transposed convolution/checkerboards | Optional matrix/scatter/output-size/coverage route; preserve actual ambiguity and qualified decoder alternatives |
| Hardware, cuDNN, TF32, layout, compilation/fusion | Preserve implicit GEMM/transform concepts, shape/layout/budget and exact inference BN folding; device-specific optimization/performance belongs GPU engineering, no fabricated or outdated speed hierarchy |
| Architecture decision matrix | Replace slogans with measured baseline/control/shift-development decisions and task-specific information loss |
| Existing exercises | Eight changed core/optional tasks with hints and full solutions, including meaningful shape/holes/protocol transfer |
| Applications beyond images | Fixed derivative and discrete diffusion stencil with units/assumptions, not a vague list of fields |

Specific correctness repairs: dense networks can learn feature correlations; convolution's inductive bias is its locality/sharing. Weight gradients sum uses, with any averaging supplied by objective reduction. Overlapping patches are not independent new samples. Pooling is not global shift invariance; circular versus zero-padded shift conditions and sampled phase are separate. Same padding is not floor(k/2) for arbitrary even/dilated/strided cases. Even kernels are valid with intentional alignment. Pooling contributes to RF: corrected32-wide example r3/4/8/10/18/46. Padding does not wrap unless circular. Bounding widths can include holes/padding; dead gradients are not absent architecture. Nonlinear ERFs are not universally Gaussian and no extrapolated depth bound is claimed.

Depthwise output may be a channel multiplier; corrected separable weight ratio9*Cout/(9+Cout). Fold sums overlap rather than automatically inverting Unfold. Transpose is not an inverse; output_padding resolves size, not appended zero values. Checkerboard mitigation is qualified. Source pointwise layer expected256 channels but received64; removed in favor of matching actual model. ResNet adaptive pooling is not universally7×7. Full VGG FC MAC/FLOP confusion is not retained. Channel-last storage does not change logical axes. BN folding needs fixed evaluation statistics and does not turn ReLU linear. No universal compilation/fusion/timing promise, “FP8 on Ampere” claim, or general benchmarking/determinism equivalence remains. Small-batch BN involves spatial counts too; predecessor owns full mode/statistics details.

## Learning design and completeness

| Hurdle | Response |
| --- | --- |
| Filter looks like a picture pasted onto another picture | One coordinate/product/sum, with edit→specific affected outputs |
| Every channel is treated independently | Multiple channel partial sums feed one output, then groups as an explicit restriction |
| Shared weights appear to need different incompatible updates | Sum location gradients and measure one actual update; overshoot contrast when target changes |
| Formula memorization hides off-by-one geometry | Window-start counting, even-kernel asymmetric center repair |
| Pooling is assumed reversible or invariant | Two different windows with same summaries, overlap gradients, boundary motion |
| “Sees an18-pixel region” hides coordinates and holes | r/j/a plus exact offset set and real-versus-padding display |
| A CNN diagram never becomes a model | Complete real data→forward→loss→update→evaluation program, maps and baseline |
| An attractive heatmap is treated as explanation proof | Exact linear coefficients versus signed local logit sensitivity, zero guard |
| Parameter reduction is equated with speed | Actual dense/CNN affine MAC discrepancy plus explicit measured-latency deferral |
| Next steps jump across module order | Landmark bridge preserved before specialized convolution designs |

First-pass route immediately after introduction. Core §§1–7 teach a complete basic CNN workflow; §§8–10 are optional advanced branches with core readiness separated. Eight practice problems mix manual computation, editable construction, new geometry, diagnosed evidence and optional adjoint/cost transfer. Programs do not require following external links to obtain missing training code. Direct resource annotations state useful sections/level and actual review extent.

Eleven visual homes use patch sliders, channel summation, shared gradient rails, geometry rulers, pool routing, coordinate ancestry, recorded feature maps/evidence, influence profiles, phase-sampling repair, transposed scatter and patch-matrix/budget views. Some are static or compact insets; no fixed interactive quota. All scored activities specify unset input-bound answers and actual live updates, meaningful unsolved edits where construction is appropriate, contrast/nulls, resets/invalidation and accessible alternatives. Experimental views select genuine records rather than simulating browser training.

## Canonical reference section audit

Canonical reference: Dumoulin & Visin, A Guide to Convolution Arithmetic for Deep Learning, arXiv1603.07285 v2 (11 January2018; PDF dated12 January),31 pages. Full table of contents audited before scope freeze. Text/caption arithmetic review, not a claim to have inspected every rendered numeric figure cell.

| Canonical section | Coverage decision |
| --- | --- |
| 1 Introduction / 1.1 Discrete convolutions | Core moving dot product, parameter sharing and multi-channel equation; modern logical tensor contract verified separately |
| 1.2 Pooling | Core pooling summaries and geometry with current padding/adaptive extensions |
| 2 Convolution arithmetic | Core general formula plus selected actual shape fixtures |
| 2.1 No zero padding, unit strides | Opening3×3/2×2 example |
| 2.2 Zero padding, unit strides | Core side-specific padding and boundary semantics |
| 2.2.1 Half/same | Core odd/even/dilated and asymmetric alignment conditions |
| 2.2.2 Full | General formula subsumes p=k−1 at d1,s1; transpose branch contextualizes full versus valid, no mandatory separate drill |
| 2.3 No padding, non-unit stride | n7,k3,s2 fixture and unused-start reasoning |
| 2.4 Padding, non-unit stride | Core combined count formula and shape planner |
| 3 Pooling arithmetic | Core dimensions plus max/average/adaptive distinctions and gradient routing |
| 4 Transposed arithmetic / 4.1 Matrix representation | Optional explicit sparse C, shared kernel and transpose-scatter |
| 4.2 Transposed convolution | Adjoint identity plus exact noninverse counterexample |
| 4.3 No padding, unit strides | General transposed formula includes n+k−1; local scatter |
| 4.4 Padding, unit strides | General formula includes n+k−1−2p |
| 4.4.1 Half/same / 4.4.2 Full | Optional specialization through formula rather than duplicate figure parade |
| 4.5 No padding, non-unit strides | Actual stride2 overlap example, zero-insertion as conceptual interpretation |
| 4.6 Padding, non-unit strides | Output ambiguity and output_padding with current API contract |
| 5 Miscellaneous / 5.1 Dilated convolutions | Core effective kernel and exact holes; later specialized owner expands design |

## Research and actual review extent

Research verified 12 September2026. Canonical main text §§2–3, §§4.1–4.6 and §5.1 read with captions; introductory definition and whole contents read. Numerical figure matrices interleaved through the PDF were not individually image-reviewed. No benchmark or result adopted from them. Specific locators: output relationship7 around pp18–19, matrix/transpose pp19–21, relationship14/output ambiguity pp25–26, dilated relationship15 at final chapter. The section audit is a coverage decision, not a claim of every cell/appendix review.

| Source | Actual reviewed extent and role |
| --- | --- |
| [Dumoulin & Visin](https://arxiv.org/pdf/1603.07285) | Extent above; arithmetic/backward structure, canonical breadth |
| [Distill receptive fields](https://distill.pub/2019/computing-receptive-fields/) | Main recurrence, coordinate intervals/centers and multi-path alignment through graph algorithm; historical model table/discussion observed but not adopted as causal evidence. Locators “Computing receptive field region in input image”, “Alignment criteria” |
| [Luo et al. ERF](https://arxiv.org/pdf/1701.04128) | Intro/definition, §§2.1–2.4 selected complete derivation passages, §2.5 and experiment opening. Explicit assumptions/no-nonlinearity/random-weight variance/CLT qualifications inspected. Not full experiment appendix; exact profiles here derived independently. The PDF's Bernoulli derivative second-moment simplification is not copied |
| [PyTorch2.14 Conv2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Conv2d.html) | Main cross-correlation equation, channels/groups/depthwise multiplier, same/stride limitation and shape contract |
| [MaxPool2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MaxPool2d.html), [AvgPool2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.AvgPool2d.html) | Main operators, negative-infinity/zero padding, ceil behavior, count_include_pad and shape contracts; exact1D analogues executed |
| [AdaptiveAvgPool2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.AdaptiveAvgPool2d.html), [v2.14 AdaptivePooling.h](https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/native/AdaptivePooling.h) | Output-size API plus actual start_index/end_index definitions; overlapping five→three bins executed |
| [ConvTranspose2d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.ConvTranspose2d.html), [Fold](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Fold.html) | Main operator descriptions, output-size ambiguity, no-inverse warning, overlap/divisor relationship and shape |
| [Distill checkerboard article](https://distill.pub/2016/deconv-checkerboard/) | Full main article text through conclusions/acknowledgments, including uneven overlap, even-overlap limitations, resize alternatives and backward artifacts; bounded qualitative use with independent coverage arithmetic |
| [Zhang2019](https://proceedings.mlr.press/v97/zhang19a.html) | Full abstract/metadata only; motivates anti-aliasing direction, no exact paper scores or detailed method claims |
| [Stanford2017 syllabus](https://cs231n.stanford.edu/2017/syllabus), [official lecture video](https://www.youtube.com/watch?v=bNb2fEVKeEo) | Syllabus entry and official video title/description checked, video not watched. Link annotated as alternate spoken route, no invented timestamps |
| [Channels-last tutorial](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html) | Definition and API example preserving dimensions while changing strides, early conversion section; not its performance benchmark. Page updated9July2025 |
| [NVIDIA convolution guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html) | §§3–4.2 implicit GEMM/transform/layout/virtual dimensions read; device-specific benchmark context visible, not adopted as current timing |
| [UCI dataset](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits) | Attribution/license/schema previously verified for identical retained range data; local exact source IDs/hash and partition provenance preserved |

Source disagreements handled by distinguishing definitions and conditions: theoretical span versus actual connected set versus local gradient, historical “convolution” convention versus cross-correlation API, mathematical pooling summary versus library padding/ceil behavior, and transpose shape recovery versus value inversion. No exact quote or externally copied diagram asset is required.

## Author checks and deferred work

Complete program executed twelve actual fitted models and bounded exact operator fixtures; additions reran fixtures only. Data, split, outputs and limits are documented in data-provenance.md. All manuscript numbers are tied to stored results or explicit arithmetic. Full author reread completed for the whole 5,377-word manuscript, full specifications, program, provenance and this design. Checked the first-pass local foundation route, explained notation, complete inputs/training/inference, honest baseline/control outcomes, source annotations, actual next-topic bridge, optional-depth/readiness separation, and eight changed exercises with closed hint/solution disclosure.

Bounded author checks passed: program AST, 124 balanced inline math pairs before the final full-padding addition (final count remains balanced), 13 balanced display pairs, closed details, original JSX and real-data hashes, all retained JSON numbers finite, twelve fits with declared six-step traces, disjoint280/120 source-ID split, actual map dimensions/probability sums, local download links, and absence of own bytecode cache. A late exact dtype check confirmed integer-valued input/kernel with fractional bias yields1.5 rather than truncating the bias; fitted float32 logic is unchanged. Existing calculations were not rerun merely for text/disclosure revisions.

Learning-experience review corrected a misleading channel-practice heading, provided direct environment interpreter setup, required zero-input/bias checking in the sum/difference construction, changed the influence activity from a practically fixed yes/no question to an input-bound numeric prediction, and labeled transposed preset comparisons as exploration rather than construction. Saved maps are explicitly post-ReLU; absent raw logits, biases and arbitrary edited-input maps cannot be fabricated. No further author finding is pending. Root reconciliation remains a separate step before binding the content checkpoint. **Historical interaction record:** the earlier prediction/reveal behavior described here is superseded by the 21 September live-exploration contract; it is not a phase-two implementation requirement. Preserve the recorded mathematical checks and fixtures.

Implementation remains not started. Phase two must build topic-specific figures/labs, adapt the large local author dataset into measured on-demand runtime inputs without losing needed examples, integrate complete downloadable code/data, translate/verify any browser formulas, preserve route/context links, and perform required author/independent/browser/accessibility/performance/curriculum/build checks. A phase-one program run does not establish browser correctness.

Root content reconciliation read the complete manuscript, design and visual contracts. Corrected two placement captions to match the actual contracts: sum/difference channel construction with changed/zero inputs, and saved post-ReLU maps/class probabilities rather than unavailable raw logits. Made the dense-to-separable ratio direction explicit. No formulas, retained fits or data changed; this bounded manuscript correction requires no new training. Formal independent phase-two review remains pending.

Retain all pending packet files; no package installations, unrelated cleanup, other-author edits or runtime files. Root owns the final source-bound content hash/status and shared ledger. Only own manuscript/specs/program/data/design were authored here.

## Live exploration revision — 21 September 2026

The user replaced prediction-and-reveal lab assessment with immediately visible, playable exploration, including removal of optional prediction controls. The manuscript and specifications now use that contract. This changes the teaching interaction, not the topic's model predictions or its mathematical masks/gates.

**Topic disposition:** Move a window and see every dependency. Edit image/kernel cells, stride/dilation/padding, pooling inputs, shared-weight targets/rate and receptive-field threshold. Synchronize the selected patch, products, output map, transpose contributions, gradient accumulation and ancestry paths. Geometry edits visibly change output size, alignment and holes rather than only a summary label. Choose window geometry and pooling from reach, alignment, information loss and update behavior.

Retain all source data, formulas, measured results, code programs, references and independent practice. The existing author calculations remain evidence for those unchanged quantities, not evidence that a browser implementation already satisfies this new interaction contract. Phase two must implement and verify live updates, linked-view agreement, bounded work, reset, invalid/null cases, keyboard access and responsive diagrams. Content remains prepared; implementation remains not started.


## Implementation-depth writing revision — 22 September 2026

Delivery mode: **content first**. The mechanism and ordinary-tool teaching below is written now; it is not an instruction for the finishing agent to invent missing content. Existing measured experiments and their historical evidence remain unchanged unless explicitly stated. The current manuscript section “Implement the pullback and batch the arithmetic” gives the learner route.

| Computational outcome | Scratch owner and abstraction | Ordinary tool and matched comparison | Control / practice and boundary |
| --- | --- | --- | --- |
| General convolution indexing and receptive geometry | `convolution-experiments.py:direct_conv2d, receptive_trace`; all groups/stride/dilation fixtures | F.conv2d/unfold/fold and nn.Conv2d in actual model | General address oracle retained, no opaque replacement |
| Efficient batched dense convolution and explicit pullback | `convolution_pullbacks.py:convolution_forward, convolution_backward`; tap-loop tensor contractions | Same input/parameter/upstream F.conv2d gradient comparison | Stride2 pullback extension with exact slice solution; declared dense-valid scope |
| Pooling and overlap derivatives | `convolution_pullbacks.py:pool1d, pool1d_backward`; explicit saved winners/average routing | F.max_pool1d / avg_pool1d matched values and derivatives | Changed upstream vector with exact answers; fixed valid 1-D core extends along each axis |
| Adaptive average pooling | `convolution_pullbacks.py:adaptive_average1d, adaptive_average1d_backward`; prefix sums and range-add pullback | F.adaptive_avg_pool1d matched values and derivatives for reduction, expansion and global bins | Floor/ceiling overlap exercise with exact gradient; O(n+m) work, floating-point cancellation caveat |

All local filenames in the map are retained draft sources beside this design. A linked prepared prerequisite is not yet the improved published page: finish in module order or carry its declared source with the lesson. Already implemented autograd/loss/normalization/tensor lessons may be reused as stated; no new differentiation engine, BLAS or convolution backend is implied. Optional historical families remain explanations of a distinction unless a local implementation is explicitly named.

The content packet is ready for phase-two construction after central source checkpointing. Finishing must execute the supplied comparisons on declared compatible versions, resolve any observed numerical/convention differences, expose the exact code/downloads, and verify rendering, live controls, accessibility and production loading. Unexecuted optional package/GPU/checkpoint examples remain explicitly unexecuted; do not print invented outputs or copy previous measurements onto new code.
