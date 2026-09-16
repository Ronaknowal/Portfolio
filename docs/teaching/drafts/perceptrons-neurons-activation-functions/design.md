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

Implement figuresA–G and three investigation contracts with actual grading, input invalidation and fixtures; carry CSV attribution; choose semantic per-topic components/models and on-demand assets under the code standard. Integrate current module navigation without jumping to next published content. Verify code excerpts and generated outputs, accessibility/mobile interactions, measured graph readability, data join IDs and numeric corner behavior; then perform independent content/runtime review and relevant app checks.

No runtime, source lesson, manifest, blueprint, navigation, shared ledger or instructional policy file was modified by this packet. Parent records content completion only after review; implementation remains not started.
