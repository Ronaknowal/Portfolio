# Depthwise Separable & Dilated Convolutions: content design and handoff

## Phase, identity and source baseline

Research and writing only, revision1,13 September2026. Author deep_foundations_content owns this stable-ID packet. Module deep-learning-fundamentals, position11; second in the authorized next30 scope. Root owns ledger, inventory, shared scope and checkpoint hashes. No runtime, publication, navigation, JSX, SVG or browser implementation was changed.

Actual preflight completed: node scripts/build-curriculum-inventory.mjs --topic depthwise-separable-dilated-convolutions --work content. It returned content-first in progress, implementation not started, no destination note for this topic. The unrelated resolved bit-manipulation note is historical context, not a queue.

Complete original published source read: src/learn/data/topics/depthwise-separable-dilated-convolutions.jsx, in ordered segments covering its entire body, including old code, figures, labs, benchmarks, exercises and references. Original SHA-2569d2d40d607268706bd3b0536e184876c1b15809954a830494aac32e5af3b065b; recover from baseline commit8c5da59f18516be77c29d5aeeafca3decca4f738. Original remains untouched. Existing source is an input, not a correctness endorsement.

Current standards/domain/design/code-ownership/retention instructions were read for this authoring run. Previous completed Convolution and newly completed Landmark packets supply known actual prerequisite coverage. Their IDs and content remain frozen except root-coordinated scoped link repair to Landmark. Sequence is Landmark→Depthwise/Dilated→ConvNeXt; actual full-curriculum module routes are used in the manuscript.

## Learning contract, scope and title

Retain the exact catalogue title and ID. Spatial/channel factorization and sampling geometry form two related but independently selectable convolution choices. A broader title listing compression, MobileNet and ASPP would overpromise whole architectures and obscure the mechanisms.

Learner can: trace filtered channels into mixed outputs; differentiate a small pair; count parameters/MACs with actual shapes; explain rank restrictions and multiplier; distinguish dilation/stride/padding; calculate receptive-field jump and exact support; diagnose gridding/boundaries; inspect an actual compressed-model failure; and plan adaptation/evaluation without converting operation savings into a latency promise.

First-pass route is stated after introduction: §§1–4 operator foundations,6–7 experiment/diagnosis,practice1–5. §5 applications and the gradient/rank/coverage proof details deepen the route; the closed interval-construction branch and practice7 are optional. No memorized historical years, full segmentation training or hardware benchmark is required for beginner readiness.

| Idea / discovery | Inspected coverage and actual owner | Decision and depth | Durable location |
| --- | --- | --- | --- |
| Full channel-spatial rank restriction | Original formula/counts weakly explain representational loss; Landmark intentionally defers exact restriction | Own here: two-probe impossibility, multiplier, exact construction and real SVD compression | lesson§§1–2,4,6; rank visual |
| Exact sampled set versus RF box | Original lab shows grids but prose incorrectly treats gcd1 as sufficient | Own geometry here; refresh prior convolution r/j definitions locally | §3, exact support fixtures/lattice |
| Finite map and global context | Original DeepLab versions and rate/stride mapping blur | Own local mechanism and version-correct ASPP/V3+ bridge; full segmentation systems elsewhere | §5, parallel spec, context-blocks.py |
| Modern mobile blocks | Original MBConv complexity falsely cancels t; hard-swish called piecewise linear | Own formulas/placements needed for factorization; preserve complete compact block code | §5, context-blocks.py |
| Deployment timing/quantization | Original invented microseconds and categorical hardware advice | Replace invented timings with exact counts and a concrete profiling contract; no runtime experiment now | §§2,7/provenance |
| ConvNeXt-specific block/GRN | Next exact topic belongs to same author; full body will be read at its actual preflight | Link actual next route; do not duplicate its full lesson | §9/next packet |
| Causal temporal context | Useful extension of actual sampling availability, not full sequence treatment | Local centered-versus-causal explanation; recurrent/attention packets own state/masks in depth | §5; future assigned sequence packets |
| Factorization algorithm | SVD/linear algebra existed in earlier curriculum; naming prerequisite insufficient | Local rank-pattern explanation, full implemented conversion and empirical error meaning | §§2,6 |

No new topic or rename is required. No other destination-topic note is needed for these discoveries: useful mechanism corrections are owned here or within this author's next assigned packet. Root may reassess a later cross-range discovery.

## Original depth conservation and corrections

Preserved/deepened standard/depthwise/pointwise/grouped definitions; MAC/parameter formulas; width/resolution multipliers; MobileNetV1/V2/V3/Xception context; expansion/linear projection/skip shape; dilation/output-size/RF; gridding; ASPP/image branch/decoder; complete operator and building-block programs; interpretable real-data experiment; numerical checks; hardware considerations; independent exercises and references. These are reorganized around mechanisms, not the original uniform article sections.

Repairs before carrying old claims forward:

- Dilation changes sampling, not a factorization of dense spatial weights. Distinguish spatial separability from channel-spatial separability.
- Finite ratio1/Cout+1/k² never magically reaches its limit; Cout1 can increase cost. MAC arithmetic is not latency.
- Expansion t remains in both pointwise terms; downsampling occurs after expansion and affects only later areas. RegNet grouped convolutions and ConvNeXt need not share MobileNet topology.
- Hard-sigmoid is piecewise linear; hard-swish middle is quadratic. Correct gradient/bias/activation placement rather than implying arbitrary linear-pair folding.
- Gcd1 is not sufficient for dense sampled coverage; [1,4] is a concrete counterexample. Exponential/interval constructions distinguish theoretical support, finite boundaries, learned zeros and active gradients.
- Original exact RF percentages and “largest RF with arbitrary rates” exercise were unsound; new exact sets and3^L structural bound state assumptions.
- ASPP V3 is parallel; its rate list is not a serial schedule. Match original rates6/12/18 to OS16, doubled rates to OS8; retain incoming field r rather than multiplying OS by a stencil span.
- V3 image-pooling branch and V3+ decoder are separate version-specific mechanisms. Large rates can degenerate to center-only taps.
- Group96→128,g32 is valid;96→130 invalid. Hardware rounding is not arbitrary group divisibility. Pointwise mixing can combine RGB channels.
- Correct normalization folding bias includes original b, and eval versus gradient tracking remain distinct. Singleton pooled training BatchNorm is a concrete issue, not a universal minibatch minimum.
- Remove unsupported dominance/hardware thresholds, device counts, old invented microsecond tables, unverified historical percentages and exact speed rankings. New integer cost and actual6-fit/24-factorization data replace them.
- Original “scratch versus library” wrappers sometimes both called convolution kernels. New direct-loop NumPy reference actually traverses indices independently; dtype and grouped shape behavior explicit.

## Canonical reference agenda audit

This paired topic has two canonical origins: MobileNetV1's operational factorization discussion and Yu/Koltun's dilation/context paper. Read their actual section lists and relevant text, not only abstracts; use the following decisions rather than copying either organization.

| Canonical source actual section | Decision for this lesson |
| --- | --- |
| MobileNets1 Introduction;2 Prior Work | Brief mobile-resource purpose; omit unsupported historical first/dominance claims. Related model family context in§5. |
| MobileNets3.1 Depthwise Separable Convolution | Core§§1–2, exact own worked values/rank restriction; don't infer universal accuracy from source experiments. |
| MobileNets3.2 Network Structure and Training | Explain actual two-stage activation placement; compact block/experiment route differs and is explicitly declared. Full historical28-layer reproduction not necessary. |
| MobileNets3.3 Width Multiplier;3.4 Resolution Multiplier | Preserve exact linear/quadratic terms and spatial-area effect in§5, including rounding caveat. |
| MobileNets4 Experiments (compression, architecture comparisons, object recognition/fine-grained/localization/detection/attributes/face embedding);5 Conclusion | These establish historical use cases under their protocols. We do not copy their benchmark table or promise those tasks from a small classifier; real six-fit compression experiment plus segmentation context supply meaningful local applications. |
| Yu/Koltun1 Introduction | Context/dense-prediction question locally introduced; dilation not newly invented out of nothing. |
| Yu/Koltun2 Dilated Convolutions | Core indexed sampling/shape/field in§3, explicitly distinguish span and support. |
| Yu/Koltun3 Multi-Scale Context Aggregation | Preserve serial context rationale and growing scales; add exact support construction. Do not replace its actual rate sequence with a claimed identical generic list. |
| Yu/Koltun4 Front-End Prediction Module | Explain output-resolution/backbone role; full VGG-derived training implementation belongs to segmentation rather than this operator lesson. |
| Yu/Koltun5 Experiments;6 Conclusion | Read inspected evaluation portions to keep front-end/context contribution and training data distinct; no copied quantitative generalization claims. Our actual model comparison is separately scoped. |

Supplementary coverage decisions: WangHDC§3.2 gap recurrence versus common-factor warning directly informs corrected geometry; Xception§4.7 informs conditional activation advice; MobileV2§§3.2–3.4/block table informs expansion/projection; MobileV3§5.2 supplies hard-swish; DeepLabV3§3 context and boundary discussion plus V3+decoder/Xception subsection supply version-specific applications. NAS algorithm internals and all historical benchmark reproduction are excluded with a clear local owner boundary, not silently missing foundational teaching.

## Hurdle map and representations

| Hurdle/outcome | Local prior bridge and mechanism | Complete example | Representation and learner evidence |
| --- | --- | --- | --- |
| Channel axis is not spatial axis | Refresh NCHW, map versus scalar, shared weighted sum | Two channels/two taps→two outputs; one gradient update | Distinct channel lanes and mixing matrix; inspect touched/untouched output |
| Savings impose a restriction | Rank as independent patterns, no assumed SVD mastery | Identity two-probe failure; multiplier2 repair; exact k3 budgets | Editable spatial/mixing coefficients and simultaneous probes, actual residual result checks |
| Dilation differs from stride | Explain sampled coordinates, output centers and zero padding | Nine-value signal, equal-ramp/null and asymmetric edits | Movable disconnected stencil with numeric products |
| RF outline is not coverage | Refresh incoming width/jump | [1,4] gap/gcd counterexample and finite8×8 boundary | Layer lattice/offset sets, unsolved gap-repair task and true reordering null |
| Architecture diagrams must correspond to shapes | Expansion before stride, branch source identity, concatenation axis | MBConv32t6 arithmetic; parallel mean/local fixture; complete block program | Shape diagram, parallel dependency plot and linked branchwise output |
| Weight approximation differs from task quality | Local SVD rank terms and labels/logits/CE | Six fits,24 fixed factorizations, full costs/actual failure | Real digit pixel editor, signed logits and filter residuals; bounded actual inference |
| Evaluation and deployment need separate evidence | Refresh development consumption and operation count convention | Plan adaptation without invented recovery/timing | Changed open practice with criteria and worked valid answer |

## Research ledger and actual reading extent

All URLs directly opened on13 September2026. Summaries below describe actual read extent, not full-paper/video claims. No videos were watched or transcripts claimed; D2L and Distill supply substantive alternate visual/code routes rather than adding an irrelevant video to meet a quota.

- https://arxiv.org/pdf/1704.04861 — paper section list, §3.1–3.4 body/Table1–3 and selected experiment context. Used factorization, architecture activation placement and width/resolution terms; no source timing ranking carried forward.
- https://arxiv.org/pdf/1511.07122 — actual section list; intro,§2,§3 context/table and inspected§4/§5 evaluation text. Used dilation operator/context rationale and honest history. Did not reproduce full experiments.
- https://arxiv.org/pdf/1702.08502 — intro/§3.1context and full§3.2 HDC including Eq2 and examples[1,2,5]/[1,2,9]; start§4. The manuscript's set proof/counterexamples are independently computed, not a copied heuristic theorem.
- https://arxiv.org/pdf/1610.02357 — inspected§4.5.2 cost setting,§4.6 residual caveats,§4.7 intermediate activation and closing discussion. Contextual evidence only.
- https://arxiv.org/pdf/1801.04381 — abstract,§3.2–3.4 and block table/selected§6.4 from earlier Landmark research. Actual expansion expression and shape/activation boundary checked.
- https://arxiv.org/pdf/1905.02244 — inspected§4.2 search,§5.1–5.4 network changes with hard-swish equation/figure explanation. No platform speed generalized.
- https://arxiv.org/pdf/1706.05587 — inspected ASPP/large-rate boundary/image-pooling discussion and training/evaluation context around extractedlines290–594. Version and OS conditions retained.
- https://arxiv.org/pdf/1802.02611 — inspected decoder and modified-Xception sections around extractedlines208–227/378–395. Not a full segmentation replication.
- https://docs.pytorch.org/docs/2.9/generated/torch.nn.Conv2d.html — complete API operative group/dilation/padding/shape definitions; stable2.9 page used because generic stable URL redirected without useful extraction. Actual local program version2.14.0+cpu recorded separately.
- https://docs.pytorch.org/vision/main/_modules/torchvision/models/segmentation/deeplabv3.html — inspected DeepLabHead,ASPPConv,ASPPPooling,ASPP definitions and relevant backbone/weight context. Main-branch view is date-bound; source has dynamic branch count and explicit pooling BN, not a universally frozen implementation.
- https://d2l.ai/chapter_convolutional-neural-networks/channels.html — actual subsection list7.4.1–7.4.5; read channel definitions, numeric multi-input example,1×1 matrix-multiply explanation/code, discussion/exercise agenda. Annotated alternate for channel mixing. No claim all notebook variants executed.
- https://distill.pub/2019/computing-receptive-fields/ — inspected variable definitions, receptive-field recurrence, region/location and multi-path article structure. Alternate geometry article; interactive controls not operated in browser.
- https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits — dataset schema/context/CCBY4.0 checked. Unchanged existing attributed subset reused; fresh duplicate audit recorded.

The manuscript also derives elementary linear/rank/count/index identities independently. No quoted passage, figure, media asset or table wording was copied. Source links stay near corresponding historical/API claims; own numerical evidence is clearly labelled.

## Programs, author checks and pending review

Required inputs: lesson.md,visual-specifications.md,design.md,data-provenance.md,digits-400.csv,convolution-factorization.py,calculated-inputs.json,author-checks.py,author-check-results.json,context-blocks.py,block-check-results.json. Provenance records exact split, fitted status, all result rows, saved weights, direct-loop and pixel-edit checks. Root binds final file hashes in the shared ledger.

Actual author execution: six small400-stepCPU fits,24 fixed SVD conversions/scores; all chosen model weights saved; nine direct-loop operator checks and16 saved-factorized predictions; four full-model independent reconstructions and pixel edits; one complete manuscript equality program; compact mobile/context block forward/backward probe. No app build, browser test, runtime integration or formal independent review.

During author checking a JSON-writing error revealed a local variable reused for dense costs and dense logits; renamed it and reran only the affected small calculation program. Training/results were unchanged. The manuscript's complete equality example ran independently and returned stated shapes/True. No fabricated outputs were substituted for failed checks.

Author full manuscript and visual-specification reread completed in ordered full-file segments on13 September2026. Repaired two author-directed visual captions into learner-facing explanations, made the rank-probe target update explicit, and extended the optional gradient step to include a checked loss-increasing contrast. Learning-experience checklist completed below. Root reconciliation may return scoped content findings; formal phase-two implementation review remains a separate task.

### Learning-experience checklist and completion

- Foundations introduced locally: spatial/channel distinction, NCHW, weighted sums, logits versus correct count, rank/SVD intuition, sampling/jump units.
- First-pass route precedes technical detail; advanced interval construction is closed and optional. Deep applications do not silently replace core readiness.
- Mechanisms connect prose, symbols, tables, complete programs and interpreted numbers. Every major fixture has a meaningful changed input and null case.
- Topic-specific forms: channel lanes, independent-probe rank repair, sampled-site lattice, parallel context branches and actual digit/filter/logit comparison. No fixed lab quota or generic text panel substituted.
- Seven changed practice tasks have separate closed hints and explained solutions; open experimental-design question supplies criteria. The two-probe task begins unsolved and computed comparisons outputs.
- Sources are annotated with real read extent and version caveats. Historical accuracy/speed claims are not manufactured; all new reported numerical results are retained.
- Data leakage/protocol limits, finite-map assumptions, normalization mode and inference cost caveats appear at useful homes; runtime instructions do not print disclaimers into product flows.
- Phase-two rendering, accessibility, live input/output synchronization, responsive layouts and complete implementation checks explicitly deferred. Content completion does not imply publication or user acceptance.

Current state: complete written lesson/specifications and bounded author calculations, author reread/checklist complete, pending root checkpoint. Implementation not started. Next author action after freeze: actual ConvNeXt preflight/full source research; next implementation action only under future finish authorization: consume this entire packet and build/verify its specified mechanisms.

## Live exploration revision — 21 September 2026

The user replaced prediction-and-reveal lab assessment with immediately visible, playable exploration, including removal of optional prediction controls. The manuscript and specifications now use that contract. This changes the teaching interaction, not the topic's model predictions or its mathematical masks/gates.

**Topic disposition:** Construct channel routes and sampling coverage. Edit depthwise/pointwise filter entries, channel cells, stencil dilation/offsets, serial rates and parallel branch choices; manipulate retained digit inputs where weights are available. Update output contributions, rank restrictions, visited lattice sites, branch union and exact frozen-model outputs. Keep coverage geometry separate from learned influence. Choose separability, dilation or parallel context from expressiveness, blind spots and the measured budget.

Retain all source data, formulas, measured results, code programs, references and independent practice. The existing author calculations remain evidence for those unchanged quantities, not evidence that a browser implementation already satisfies this new interaction contract. Phase two must implement and verify live updates, linked-view agreement, bounded work, reset, invalid/null cases, keyboard access and responsive diagrams. Content remains prepared; implementation remains not started.


## Implementation-depth writing revision — 22 September 2026

Delivery mode: **content first**. The mechanism and ordinary-tool teaching below is written now; it is not an instruction for the finishing agent to invent missing content. Existing measured experiments and their historical evidence remain unchanged unless explicitly stated. The current manuscript section “Reuse the spatial operator and own the factorization” gives the learner route.

| Computational outcome | Scratch owner and abstraction | Ordinary tool and matched comparison | Control / practice and boundary |
| --- | --- | --- | --- |
| Separable rank and SVD conversion | `convolution-factorization.py:factor_spatial, reconstructed_weight`; published matrix decompositions own SVD | Native grouped and pointwise Conv2d with explicit channel/bias mapping | Adaptive energy budget exercise with solved singular-value case; common-rank layout limit |
| Dilated address geometry, inverted residual and context branches | Actual prepared convolution direct_conv2d reused; local author-checks support and context-blocks.py composition | F.conv2d/nn.Conv2d and F.interpolate/BatchNorm as explicit model primitives | Coverage/stride/context practice retained; full segmentation training is not claimed |

All local filenames in the map are retained draft sources beside this design. A linked prepared prerequisite is not yet the improved published page: finish in module order or carry its declared source with the lesson. Already implemented autograd/loss/normalization/tensor lessons may be reused as stated; no new differentiation engine, BLAS or convolution backend is implied. Optional historical families remain explanations of a distinction unless a local implementation is explicitly named.

The content packet is ready for phase-two construction after central source checkpointing. Finishing must execute the supplied comparisons on declared compatible versions, resolve any observed numerical/convention differences, expose the exact code/downloads, and verify rendering, live controls, accessibility and production loading. Unexecuted optional package/GPU/checkpoint examples remain explicitly unexecuted; do not print invented outputs or copy previous measurements onto new code.
