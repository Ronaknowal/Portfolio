# Landmark Architectures: independent implementation review

Reviewer: `/root/decision_depth_prose`; implementation author: `/root/initialization_implementation`. Reviewed 22 September 2026. Topic: `landmark-architectures-lenet-alexnet-vgg-resnet-efficientnet`. **The complete source/content review and complementary native checks passed after the corrections below. Final production browser and visual review remain with the parent integrator.** No user-acceptance or trained-ImageNet-quality claim is made.

## Coverage and evidence

Read the complete revision-three manuscript, visual specifications, implementation ownership map, canonical small experiment and all five complete architecture builders. Reviewed the generated reader, authoring conversion, live components, pure models, measurement/deferred-map split, owned styles and author records. The main text retains historical fidelity distinctions, named-family mechanisms, the explicit scratch-to-library bridge, all recorded outcomes, code modification, seven changed-context exercises, annotated video/documentation and source references. Relative draft links are converted into actual curriculum routes.

- [Source-bound review and findings](evidence/landmark-architectures-independent-review.json).
- [Complementary native evidence](evidence/landmark-architectures-independent-native.json), reproduced with `scratch/lesson-tools/Scripts/python.exe -X utf8 -B scripts/verify-landmark-independent.py`.
- [Author model/source evidence](evidence/landmark-architecture-author.json) and [author native evidence](evidence/landmark-architecture-native.json), reused for the complete twelve-fit replay and matched-state ordinary-library execution. The reviewer did not repeat the training campaign or the memory-heavy paired VGG/AlexNet forwards.

## Findings and confirmed corrections

| Finding | Correction and confirmation |
| --- | --- |
| `MobileBlock` accepted training drop probability 1 and divided by zero. A native call with shape `[2,4,4,4]` produced nonfinite values; default B0's rates did not reach this endpoint. | The public reusable builder now validates `[0,1]` and at 1 returns the direct input plus a zero correction after computing the branch. Independently checked finite identity, p0, evaluation identity of the masking operation, per-example shared masks and invalid-rate rejection. The prepared historical source is retained; the current download and learner explanation carry the repaired contract. |
| The initial stride/pool and composed-kernel figure gave prose and a nested grid without showing the two sampled grids or intermediate dependencies required by its teaching job. | It now shows neighboring image windows separately from neighboring feature-pool windows, plus selectable intermediate positions whose actual 3×3 input neighborhoods form a 5×5 union. The source-to-intermediate-to-output route and intervening activation are explicit. Reviewed the positions, window shifts, bounds and all nine selectable supports. Final rendered legibility remains a browser check. |
| The scaling lab initially offered controls and numbers without its specified visual budget comparison. | Added aligned parameter/MAC factor bars with one common zero-origin scale, numeric values and a target marker only on the MAC bar. The scale and units are declared; no accuracy or exact named-EfficientNet count is implied. |
| Minor clarity and current-state concerns. | The skip route now retains the input without implying an obligatory physical copy. Constructed accessibility labels have word spacing. Public provenance distinguishes the historical content-only packet from executed current implementation evidence. |

No additional material source/content finding remains after these corrections. The reviewer changed only independent test/review artifacts, not the author's lesson or model sources.

## Technical and implementation-depth assessment

The lesson teaches architecture as composition of previously opened primitives. This is the correct scratch boundary here: rebuilding a Conv2d backend again would obscure stage widths, repetitions, downsampling, branch connectivity, SE and the classifier. Complete local constructors exist for the declared modern LeNet variant, Torchvision-style AlexNet, VGG-16, ResNet-18 and EfficientNet-B0. Their original-paper differences are explicit. The ordinary route uses actual Torchvision model factories, an exact component-state mapping and matched input/evaluation state rather than comparing unrelated random networks. The separate pretrained photograph example is complete and clearly labeled unexecuted; no photograph or pretrained weight was downloaded for this work.

The native review independently verified:

1. All five full meta constructions have the promised output shape and parameter totals: 61,706; 61,100,840; 138,357,544; 11,689,512; 5,288,548. Meta execution establishes shapes/counts, not numerical training.
2. The learner's VGG modification actually yields a 3,591-parameter seven-class global-average head and `[3,7]` output at both 224 and 256 pixel inputs.
3. The complete `ChannelGate` respects a changed, nontrivial spatial permutation: global summary/gates are preserved while output positions move correspondingly.
4. The repaired public mobile block handles drop-probability endpoints and invalid values, and its sampled row mask really shares one decision across channel/spatial entries of an example.
5. B0 contains the complete sixteen mobile blocks, stated stochastic-depth schedule, and SE squeeze widths derived from each block's incoming width.
6. Scalar accumulation reconstructs every saved image/class CAM score; joint spatial permutation preserves logits, and an independent signed single-cell intervention changes a score by the expected weight times delta divided by four. The maximum double-precision reconstruction error against saved float32 logits is about `7.10e-6`. This is a different calculation from the packet's float32 reordering error of about `5.73e-6`; they are not contradictory.

Budget equations, parameter/MAC exclusions, dense-head versus GAP contracts, bottleneck counts, compound-scaling approximation, cross-channel context, bias-inclusive CAM identity and changed exercises were reviewed directly. The twelve small-body fits are not described as ImageNet reproductions. Validation/development selection remains distinct from an untouched test; hypothetical budget edits change eligibility, not model performance. Raw 2×2 maps remain visible and are not presented as pixel-precise causal explanations.

Primary-source spot checks verified [Torchvision 0.29 ResNet-50 weight metrics and transforms](https://docs.pytorch.org/vision/0.29/models/generated/torchvision.models.resnet50.html) and [its VGG-16 specification](https://docs.pytorch.org/vision/0.29/models/generated/torchvision.models.vgg16.html). The prior packet documents historical paper/video review extent; this phase does not claim a new exhaustive reading of every cited paper or a full video viewing.

## Independent learning-experience assessment

This is a source-based reviewer assessment, not an observed beginner study.

| Checklist area | Concrete result |
| --- | --- |
| First-pass route | The handwriting question precedes terminology; stem/stage/backbone/head and tensor axes are introduced before named architectures. Core sections and optional history/family branches are clearly separated. |
| Cautions and reading load | The distinctions about variants, budgets, training recipes, validation and CAM meaning appear at the claims they qualify. They protect substantive conclusions; no generic warning stack or cautionary program output replaces teaching. |
| Real question/data | The digit-recognition question returns in actual small-network fits and a real mistaken example. Attribution, split, scale, writer-ID limitation and measured conditions are retained. |
| Live exploration | Head dimensions, actual map cells, signed weights, scalar bias and compute allocations are editable. Current results and a pinned comparison are visible; no learner-prediction entry is present. Exact limits, infeasible allocations and unchanged comparisons remain meaningful. |
| Inline visual flow | Shape grids distinguish channel count from spatial size; historical route labels preserve variant fidelity; repaired sampled windows and intermediate supports explain stride/pooling/composition; branch lanes distinguish append versus add; expanded/narrow routes, global summary/broadcast, budget bars and signed tiles serve different teaching jobs. |
| Quantitative honesty | Bars share declared zero origins; exact counts are separate from measured fits. Plot markers are real checkpoints with exact tables. Recorded class maps and probabilities retain separate meanings and only saved examples can be selected. Desktop/phone perception still needs rendered inspection. |
| Mechanism/library connection | Named complete builders, two explicit composition classes, runnable comparisons, the weight enum/transform route and reusable transfer-learning owner make both implementation paths findable. The interpretation says exactly what was and was not executed. |
| Independent practice | Tasks change head dimensions, intermediate width, merge shape, signed class weights, comparison protocol, freeze policy and deployment constraints. The additional seven-class construction has a concrete code change, parameter count and shape checks. |
| Browser/screenshots | Not performed by this reviewer. Parent must confirm pointer/keyboard edits, all revised visual states, request timing, error/reset behavior and informative desktop/phone screenshots before implementation completion. |

Unchanged numerical evidence can be reused after presentation-only browser fixes. Any material model/content change must be reported for a scoped follow-up and source-hash refresh.

## Scoped phone-layout follow-up — 22 September 2026

The parent production check found horizontal overflow at 320 pixels with recorded maps and both complete programs open. Reviewed the author's two declaration changes: the topic's wrapping route stages use a 170-pixel flex basis instead of 135 pixels, and small grids have `max-width: 100%`. The 150-pixel intermediate grid can therefore fit its stage or shrink within available space; no content is clipped and its nine selectable convolution supports are unchanged. Both rules are scoped to Landmark classes and do not alter shared diagrams, numerical code or native programs.

Reconstructing the previous CSS from exactly those two declarations reproduced its original reviewed hash. Removing only the appended design repair note likewise reproduced the original design hash. All other sixteen bound sources remained identical. Reviewed the added browser assertion that checks the intermediate grid against its own stage bounds, alongside the existing whole-page overflow check. The author's [targeted development-route evidence](evidence/landmark-architecture-layout-repair.json) reports containment at 1366, 390 and 320 pixels with maps and both sources open, plus visual inspection of the 320-pixel composition.

The independent record now binds the corrected CSS and appended design note. This follow-up is a source/layout review that reuses the unchanged numerical evidence; it does not claim a new native execution or independent browser observation. The parent still owns the final production rebuild and browser confirmation.
