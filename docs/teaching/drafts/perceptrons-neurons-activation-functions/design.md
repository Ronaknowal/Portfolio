# Content handoff: Perceptrons, Neurons & Activation Functions

Stable ID `perceptrons-neurons-activation-functions`. Deep Learning module position1; authorized batch position22. Owner deep_foundations_content; research/write only, revision1. Deliverables: lesson.md, visual-specifications.md, digits-400.csv, data-provenance.md, author-calculations.py, calculated-inputs.json. Root owns ledger and final file hashes; no implementation/publication claimed.

## Scope, title and source conservation

Preflight `node scripts/build-curriculum-inventory.mjs --topic perceptrons-neurons-activation-functions --work content` was run. Published lesson needs individual design; no destination notes required consumption. Original manifest-owned body `src/learn/data/topics/perceptrons-neurons-activation-functions.jsx` was read in full, including the portions omitted by the first truncated output. Starting commit `8c5da59f18516be77c29d5aeeafca3decca4f738`; original body SHA256 `8f14a5c65e37d55725a662687db15d98e6bc088301c38548e05f2ff2ffb8f521`. The full body has substantial mathematics, NumPy/PyTorch code, XOR, six-function comparison, smooth activations, gating, geometry and performance claims; this packet is a conservation-and-correction rewrite, not an introductory replacement.

Retain the title and stable ID: perceptron learning, neuron representation and activation choices remain the promised scope. No catalogue/order/blueprint changes. Original named prerequisites are not assumed sufficient: weighted sums, vector norm, bias, shape convention, derivatives, training/validation meanings and loss roles receive local bridges. The preceding capstone motivates controlled experiments, not neural superiority. Next Backprop receives the reverse-pass derivation and manual network training.

| Original idea or discovered gap | Decision and reason | Current home |
| --- | --- | --- |
| Weighted neuron and geometry | Preserve; divide score by norm for distance, handle zero weights | §1 and investigationA |
| Perceptron AND/XOR learning | Preserve with exact integer update fixture and explicit tie rules; updates during pass differ from final errors | §2 program, proof and practice2 |
| Hidden XOR, affine collapse | Preserve exact ramp construction; remove false BooleanOR interpretation and overbroad “remove activations makes GPT linear” claim | §3 and investigationB |
| Activation values/derivatives | Preserve sigmoid,tanh,ReLU,leaky,ELU,GELU exact/approx,SiLU/Swish and Mish; staged core/deeper | §§4/7 and investigationC |
| Softmax and stable outputs | Local refresher; full loss derivations belong immediately following Backprop in Loss Functions | §4 |
| Scalar gradient products and failure modes | Preserve with weights included, state-specific zero gradients and qualified symmetry | §§4/6; deeper matrix singular values in initialization |
| Manual backprop XOR and gradient check | Full algorithm belongs next Backprop lesson, including correcting original factor2 for mean squared loss | Destination within same author range; no useful mechanism silently lost |
| Six-activation real experiment | Replace unbound or unsupported ranking with retained actual UCI input and matched18runs | §5 |
| PyTorch implementation and SwiGLU | Complete local classifier and small standalone gate module; no unexplained Transformer normalization/residual prerequisites | §§5/7 |
| Universal approximation | Add explicit ramp/spline construction and conditions; “any nonlinearity” is false | §7 |
| Cost, memory, padding/alignment | Exact counts; single bf16 tensor is1GiB, not old4.3GB; rounded gated width not exactly same budget | §7 |
| Long vivid historical timeline | Keep a brief contextual mention; remove unverified demo details, single-cause AI-winter narrative and inaccurate Neocognitron dating | §2 |
| Unsupported modern architecture/default/runtime claims | Omit claims of universal GELU/SiLU use, fixed hardware speed rankings, automatic irrecoverable act-swap damage | §§5–7 evidence boundaries |

Full XOR reverse-pass teaching is an internal range dependency: before freezing Backprop, read this packet and preserve the old forward/backward program's useful learning goal there with correct derivatives. Parent was notified of this ownership. Classical regularization sibling was informed that later deep dropout will own unit/channel/path masks, expectation limits and normalization interactions. This lesson's GELU expected-mask interpretation is a narrow mathematical bridge, not a replacement for that material.

## Canonical-reference section-list audit

Canonical reference is Goodfellow/Bengio/Courville, [Deep Learning chapter6](https://www.deeplearningbook.org/contents/mlp.html). The full chapter URL exceeded the retrieval limit; [TOC](https://www.deeplearningbook.org/contents/TOC.html) was successfully read. Record honestly: section-list audit, not full-chapter review. Primary mechanism evidence was obtained independently below.

| Canonical section | Decision |
| --- | --- |
| 6.1LearningXOR | Core full constructive example, impossibility proof and changed repair |
| 6.2GradientBasedLearning | Local loop/gradient/loss explanation and full small train program; derivatives next |
| 6.3HiddenUnits | Core three shapes; broader families retained in optional branch |
| 6.4ArchitectureDesign | Affine composition, shapes, capacity, gating and qualified approximation; convolution/residual/normalization have later dedicated homes |
| 6.5BackpropagationAndOtherDifferentiation | Explicit next-lesson destination with retained manual-training requirement |
| 6.6HistoricalNotes | Brief context only; original unsupported storytelling removed |

The audit adds no new topic. Advanced theorem proof, optimizer moments and model-specific architecture surveys are deliberately outside the core; they do not become implicit readiness criteria.

## Hurdle map and learning flow

| Hurdle/outcome | Local foundation | Worked/independent evidence | Representation |
| --- | --- | --- | --- |
| Distinguish input/weight/bias/activation | Product and signed sum |3+2−1; changed boundary problem | Contribution strip + geometry |
| Explain score versus distance | Norm and perpendicular direction | Common rescale contrast/null | InvestigationA |
| Explain update versus expressivity | Signed margin and AND/XOR truth table | Exact executed loop; impossibility proof | Boundary/table, no unnecessary simulator |
| Make a new nonlinear feature | ReLU as ramp | Exact XOR and failed shifted repair | InvestigationB |
| Match diagram to tensor code | Rows as examples, output×input weights |280×64→280×32→280×10; changed83parameter problem | Pixel flattening correspondence |
| Distinguish value, slope and full sensitivity | Small-change derivative | Sigmoid×4, ReLU×.5, SiLU negative slope | InvestigationC |
| Compare activation fairly | Fixed split, matched initialization |18actual CPUruns; independent width change | Measured loss/counts, paired ID retention |
| Understand gate and expressive capacity | Coordinatewise products, slope changes | Tiny SwiGLU, triangular pulse | Two-lane gate and summed ramps |
| Avoid fabricated compute claims | Elements×bytes, matrix sizes |1GiB and exact/rounded budget | Equations/table sufficient |

First-pass route is immediately after the introduction. Optional sections and practices are identified before the reader reaches them. Interest comes from XOR's impossibility/repair, a negative derivative in a smooth gate, real digit errors and an activation comparison without a universal winner. No fixed count of facts or labs imposed.

## Primary research and review extent

Reviewed12September2026. Links are also annotated in the learner manuscript. No video was claimed watched; relevant creator-hosted text or lecture notes were read.

- [Cornell CS4780 lecture6](https://www.cs.cornell.edu/courses/cs4780/2022sp/notes/LectureNotes06.html): complete short lecture notes, including bias augmentation, update rule and convergence argument. Supports perceptron mechanism; bounded-input proof in lesson is an explicit derivation.
- [3Blue1Brown neural-networks lesson](https://www.3blue1brown.com/lessons/neural-networks/): substantive text companion including weighted pixels/matrices and sigmoid/ReLU discussion. Companion video is an alternate resource, not independently viewed.
- [GELU paperv5](https://arxiv.org/html/1606.08415v5): abstract, introduction, §2formulation/equations and experimental section list; paper's particular experiment results were not used as universal rankings. Exact weighting, tanh approximation and SiLU relation verified.
- [GLU variantsv1](https://arxiv.org/html/2002.05202v1): §2definitions and §3.1width matching, with surrounding experimental context. Supports three-versus-two projection accounting; no copied benchmark.
- [Swish abstract](https://arxiv.org/abs/1710.05941): generalized formula/context only. No broad superiority claim.
- [Leshno/Lin/Pinkus/Schocken NYU working-paper record](https://archive.nyu.edu/handle/2451/14329): title, abstract and threshold condition reviewed. Published author-hosted PDF timed out, so no claim to have read its full proof. The manuscript gives a restricted one-dimensional constructive argument separately and cautious general theorem summary.
- [PyTorch2.14 Linear](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Linear.html), [GELU](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GELU.html), [SiLU](https://docs.pytorch.org/docs/2.14/generated/torch.nn.SiLU.html): relevant API definitions/formulas/shapes read. Stable URLs only returned redirects, so explicit2.14URLs are retained. Existing installedtorch2.14.0+cpu used.
- [UCI source](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+): dataset collection/preprocessing, classes, citation and license inspected; [load_digits1.9.1API](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html) relevant full API and local DESCR inspected. Historical test partition and writer-split limitation are explicit.
- Glorot/Bordes/Bengio2011 [paper page](https://proceedings.mlr.press/v15/glorot11a) abstract and paper located for contextual rectifier history; no numerical claims adopted.

## Author calculations and checks

Python3.12.14,NumPy2.3.5,scikit-learn1.9.1,torch2.14.0+cpu; existing shared runtime inspected and used read-only with one CPU thread. No installation, external model or data download at run time.

Executed author-calculations.py performs the exact perceptron traces, XOR construction, nine activation value/derivative checkpoints,18full200-step matched digit fits, memory counts and contrasting/null investigation fixtures. All numeric inputs are retained in calculated-inputs.json. Displayed training results are actual author calculations, not invented or measured browser output. Author calculation source is purpose-named and scoped; no batch files or runtime implementation.

Checks include: AND zero-update epoch9 after valid boundary, XOR four updates/two final errors; old hiddenOR correction; denominator120 and train/validation distinction; exact sigmoid/ReLU/gate figures; independent memory arithmetic; shape/count/zero-input SwiGLU by equations; checked no-response fixtures; finite-input/zero-norm/corner conventions. Full rendered interaction checks remain deferred. Full displayed program replay and clean installation are not represented as complete; their core calculation was executed, and deterministic expected shape/count outputs were reasoned explicitly.

Author reread of manuscript, source conservation map and specs completed before freeze. Learning-experience review checked: immediate beginner path; every new symbol/shape before use; prose↔formula↔code correspondence; no unsupported activation ranking; worked→repair→independent practice; finite assessment with hints and solutions; no repeatedly printed caveats; different visual forms for geometry, feature coordinates, derivative curves, actual pixels and paired training observations. Physical graph legibility and assistive technology operation cannot be claimed without phase2 rendering.

## Deferred implementation and completion boundary

Implement figuresA–G and three investigation contracts with actual live updates, input invalidation and fixtures; carry CSV attribution; choose semantic per-topic components/models and on-demand assets under the code standard. Integrate current module navigation without jumping to next published content. Verify code excerpts and generated outputs, accessibility/mobile interactions, measured graph readability, data join IDs and numeric corner behavior; then perform independent content/runtime review and relevant app checks.

No runtime, source lesson, manifest, blueprint, navigation, shared ledger or instructional policy file was modified by this packet. Parent records content completion only after review; implementation remains not started.

## Live exploration revision — 21 September 2026

The user replaced prediction-and-reveal lab assessment with immediately visible, playable exploration, including removal of optional prediction controls. The manuscript and specifications now use that contract. This changes the teaching interaction, not the topic's model predictions or its mathematical masks/gates.

**Topic disposition:** Move a point and its weighted evidence. Edit input coordinates, weights, bias and common scale; move the activation operating point and incoming weight; edit XOR hidden bias and output coefficient. Synchronize contribution bars, boundary distance, hard/smooth outputs, activation value/slope and all four XOR rows. Compare a shared coefficient rescaling with a moved input, and a repaired corner with the remaining corners. compare whether the decision boundary, smooth confidence or local sensitivity needs to change; a one-row repair need not solve the whole task.

Retain all source data, formulas, measured results, code programs, references and independent practice. The existing author calculations remain evidence for those unchanged quantities, not evidence that a browser implementation already satisfies this new interaction contract. Phase two must implement and verify live updates, linked-view agreement, bounded work, reset, invalid/null cases, keyboard access and responsive diagrams. Content remains prepared; implementation remains not started.

## Prepared-packet implementation — 21 September 2026

Author: `implement_dl_perceptrons`, within the explicitly authorized first-five Deep Learning finish request. The saved `scratch/deep-learning-core-implementation/perceptrons-neurons-activation-functions-preflight.json` reports the complete current packet and `canFinish: true`. The title, stable ID, publication path, prerequisites and module order remain unchanged. No destination note exists for this topic; the packet's reverse-pass ownership remains with the simultaneously implemented Backpropagation lesson.

### Implemented scope and conservation

The full prepared manuscript is incorporated into `src/learn/data/topics/perceptrons-neurons-activation-functions.jsx`, retaining all nine sections, the perceptron update and convergence argument, XOR contradiction and repair, shapes, output interpretation, activation derivatives, real training protocol, diagnostics, deeper smooth/gated/approximation/memory material, all eight changed practice tasks with their sixteen hints/solutions, and annotated references. The original packet remains intact. Scoped prose changes repair “by hand train” to “by hand and train”, remove the stale prediction-before-running instruction in practice 8, and replace deferred program replay statements with actual execution status. These do not narrow the content.

| Prepared representation | Implemented source and teaching behavior |
| --- | --- |
| A: weighted evidence and geometry | `PerceptronLabs.jsx`: `WeightedEvidenceFigure` has signed products, sum and separate sigmoid curve; `GeometryLab` has exact live coordinates/coefficients/positive scale, clipped equal-scale boundary, closest point and distance, original-versus-scaled quantities, pinned inputs and a constant zero-weight plane. |
| B: XOR representation and repair | `XorWorkedFigure` exposes every corner and a four-stage highlight with outputs always visible; `XorLab` edits hidden bias/output coefficient, supports exact fraction −4/3, and displays every hidden value, signed contribution, model output, target and residual. Inactive hidden features produce the declared null. |
| C: value, slope and local sensitivity | `ActivationLab` samples the actual analytic definitions, separates value/slope axes, marks derivative jumps with open endpoints and the framework convention, and displays the incoming-weight product. The optional deeper function selector adds no prediction feature. |
| D: image/tensor correspondence | `PerceptronFigures.jsx`: ten real thumbnail selectors with stable IDs, exact 8×8 grayscale/count table, highlighted row-0 mapping to the flattened vector and 280×64→280×32→280×10 flow. |
| E: measured comparison | All eighteen retained runs, actual update coordinates 0/1/10/50/100/200, logarithmic training loss, full count axis with explicitly truncated optional zoom, exact tables, all seeds and optional paired-error union joined by source ID. This inspects recordings; it does not retrain or invent weights/probabilities. |
| F: SwiGLU correspondence | Two separately labelled learned projection branches, SiLU only on the gate, exact paired-coordinate products, and named output-projection shapes. |
| G: triangle from ramps | Four shared-axis calculated small multiples, exact corner/midpoint table and slope-by-interval explanation; no smoothing spline. |

Topic-owned supporting sources are `src/learn/data/perceptron-models.js`, `perceptron-data.js`, `perceptron-examples.js`; `src/learn/components/lesson-labs/PerceptronShared.jsx`, `PerceptronLabs.jsx`, `PerceptronFigures.jsx`, `perceptron-labs.css`; and the stable-ID blueprint under `src/learn/data/curriculum/blueprints/`. The increment owner registers the blueprint and updates shared metadata/ledger. Offline data and attribution are copied unchanged to `public/learn-assets/perceptrons/`. `scripts/build-perceptron-data.mjs` regenerates only these topic assets from the conserved prepared data.

### Author evidence and reproducibility

- All three complete displayed programs were executed with the existing `scratch/lesson-tools/Scripts/python.exe` environment, without package installation or runtime modification. `docs/teaching/evidence/perceptron-native.json` retains their actual output. The eighteen displayed digit losses/counts agree with the saved experiment after six-decimal printing; the perceptron trace and SwiGLU shape/count/zero output match the manuscript. Fresh-environment installation remains untested.
- `scripts/verify-perceptron-examples.py` replays the exact exported displayed programs and creates an independent dense PyTorch oracle; `--dense-only` reuses recorded program execution when code is unchanged. `docs/teaching/evidence/perceptron-activation-oracle.json` contains 481 operating points for each of nine functions, with value and float64 autograd slope.
- `node scripts/verify-perceptron-models.mjs` passed ten semantic groups, including 8,658 value/slope comparisons with maximum absolute error 3.13×10⁻¹⁵, score/distance/tie/zero-norm cases, all XOR fixtures, CDF symmetry/stability, numeric invalid states, triangle corners, all validation-label joins and complete printed digit-output correspondence. Evidence: `docs/teaching/evidence/perceptron-models.json`.
- The topic JSX and all three component files were parsed early with the installed Babel parser. All 113 inline and seventeen block formulas were passed to KaTeX with `throwOnError: true`; no parse error occurred. Static supported metadata and the existing `RunnableExample.expected` contract are used. Styles target the topic's chart class rather than all descendant SVGs, preserving KaTeX assets.
- `scripts/verify-perceptron-browser.cjs` contains reusable `runPerceptronChecks(page, {capture})` plus a standalone 1366/390/320 px run. It tests immediate results, every native range endpoint/interior, numeric twins, actual pointer dragging, pin identity, invalid text, reset, real geometry crossings/null, exact-rational XOR failure/null, sensitivity signs/corners, recorded seeds/digit IDs and every representation's bounds. Its captures include contrasting and zero states. Browser execution/visual dispositions are recorded below once actually run; script existence alone is not passing evidence.

### Author learning-experience assessment

This is the author's heuristic review, not independent review or a novice study. The first-pass/deeper route remains explicit. The lesson returns from its opening digit question to observed pixels, a complete CPU fit and honest validation comparison. Worked equations connect to mechanism figures before changed practice; the browser labs contain no learner-answer, commitment, grading or reveal states. Geometry, XOR features and local sensitivities use distinct representations rather than a repeated generic plot. Exact tables accompany spatial/colour encodings. The causal code remains compact; validation lives in verifier scripts. A caution pass retained distinct scientific boundaries (geometry/probability, capacity/training, observed validation/generalization and raw tensor/total memory) in their existing homes; it added no repeated disclaimer blocks. The full inline reading pass revealed two implementation gaps and repaired them before review: the weighted-sum figure needed its separate activation curve, and the digit selectors needed actual image thumbnails rather than text-only labels.

### Author browser and visual closure

`scripts/verify-perceptron-browser.cjs` passed seven semantic groups at 1366, 390 and 320 px on the shared development server at `http://127.0.0.1:4197`, using bundled Playwright with Microsoft Edge and the retained Space Grotesk/JetBrains Mono font fixture. Evidence: `docs/teaching/evidence/perceptron-browser.json`, with exact source hashes and 39 retained panel captures in its adjacent `perceptron-browser/` folder. All native ranges received actual keyboard Home/End/interior events and matching numeric-twin assertions; a genuine mouse drag and phone-context touch taps changed the linked model. Geometry scale/tie/crossing/zero-vector, stale invalid text, pin identity, reset, XOR −4/3/all-row conflict/inactive-feature null, sigmoid compensation, SiLU negative slope, ReLU corner, recorded digits/seeds/count zoom and all figure bounds passed.

The author visually inspected all distinct A–G representations at desktop and 320 px, plus the meaningful XOR repair, negative SiLU slope and zero-weight/pinned states. Labels, curves, signs, boundary geometry, image identity and ties remained interpretable. Narrow table captions were made to wrap independently of the horizontally scrollable exact-data table. The first pointer failure was a verifier scroll-settling issue: an isolated actual pointer probe changed the slider correctly; the verifier now centres the control with instant scrolling and confirms the hit target. Contrast captures initially included the fixed site header over their top edge; capture-only centring repaired those images. Neither was hidden as a successful check before correction. Mathematical/native evidence was reused because no affected numerical source changed.

Current state: complete author implementation, computational verification and development-browser/visual closure. Independent review and final production integration remain separately tracked by the increment owner. Nothing is deployed or committed. User acceptance is separate.

### Targeted author corrections after independent review

The independent reviewer identified ordinary prose/number spacing collisions and an ambiguous parameter-count axis. The learner body now spaces prose/table values while preserving code, mathematical strings and the prepared packet. The count explicitly uses feature width d = 4096 and hidden width h = 16384; d and the earlier sequence length are separate axes. Counts are unchanged.

A further author visual check found that auto-rescaling live contribution bars could hide changes in the largest product. Geometry now fixes its unscaled contribution domain at −16 to 16 score units; live XOR uses −8 to 8. Static worked figures keep their fixed-case scales. The clipped boundary caption now also covers a boundary touching a corner without crossing the window interior. Models/native examples did not change.

The source-bound browser verifier now asserts that doubling each inspected contribution doubles its rendered bar width while its domain stays fixed. The full affected-topic rerender passed seven groups at each of 1366, 390 and 320 px with 39 refreshed captures. The author visually inspected the corrected geometry and XOR panels at desktop and 320 px: labels, readouts and domains are legible and contained. All 130 mathematical expressions also passed strict KaTeX parsing after prose cleanup. Updated source hashes are in `docs/teaching/evidence/perceptron-browser.json`. This author check is separate from the independent review and final production integration.

## Final production integration — 21 September 2026

The author handoff above is closed by independent review and the final production browser pass. The [first-five completion record](../../DEEP-LEARNING-CORE-IMPLEMENTATION.md) links the reviewed lesson, native/model evidence, actual production checks and preserved scope baseline. Next/previous order, selected-body loading, section anchors, themed links and applicable rendered math geometry pass in the integrated build. Both delivery phases are complete; user acceptance is separate. Earlier pending integration sentences describe the historical author checkpoint, not current work.
