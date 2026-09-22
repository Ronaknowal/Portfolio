# Content handoff: Backpropagation & Automatic Differentiation

Stable ID `backpropagation-automatic-differentiation`; Deep Learning position2, batch position23. Content-first revision1, owner deep_foundations_content. Full manuscript, visual-specifications, complete teaching-autodiff.py, author-calculations.py, calculated-inputs.json and licensed digits input/provenance are retained. Root owns shared ledger and frozen hashes; implementation remains not started.

## Source and scope decisions

The topic inventory was run with --work content and its destination note consumed. The original manifest maps this canonical ID to **src/learn/data/topics/backprop.jsx**, not a same-ID filename. Entire original body read, including all programs, charts, advanced sections and exercises. Commit `8c5da59f18516be77c29d5aeeafca3decca4f738`; body SHA256 `22d7afa2afdd3bb95d2ffe405dbb278f022a7c29aa56042208fb633e0498de3c`.

Retain title/ID/order. Differentiation, graph bookkeeping, practical APIs and numerical evidence are a coherent topic. Prior Perceptrons teaches forward values, slopes and shapes; this manuscript refreshes them locally and opens the backward step. The next Loss Functions lesson owns choosing/deriving the objective families in depth. Tensor convention changes are explicit: scratch NumPy weights input×output; nn.Linear stores their transpose.

| Original coverage/finding | Current decision |
| --- | --- |
| Long history, exact giant-model speed claims | Brief contextual mathematical origin through primary survey; remove unsupported historical superlatives and fixed numerical speed ratios |
| Forward/reverse,JVP/VJP,Jacobian | Preserve full local explanation, actual products, shape comparison and matched inner-product identity |
| Reverse chain rule and multilayer CE gradients | Preserve and refresh with scalar example, repeated paths, broadcast map and full matrix rules |
| Complete NumPy tensor engine | Preserve via complete purpose-named downloadable teaching program, with mapped excerpts; meaningful requires_grad, scalar seed rule, explicit2Dmatmul, copied read-only values, fresh-buffer reset |
| Softmax CE forward/log epsilon inconsistency | Correct to shifted-logsumexp; forward/backward describe same loss even at extreme logits |
| Gradient checks with undefined init names/stale parameter state | Replace with explicit same-state inputs and complete author script comparing every parameter group to torch plus central differences |
| “Engine is correct” after one epsilon | Replace with scoped evidence, perturbation sweep, dtype/scale/zero/corner/state qualifications |
| Original blobs-only training and invented causal gradient-norm diagnoses | Actual XOR construction-to-learning continuity plus licensed real digit data; measured traces only, no automatic “dead/fully trained” interpretation |
| Perceptrons manual-XOR backward coverage | Carried into complete engine XOR MSE loop with correct factor2 and actual trace; representation-versus-training distinction preserved |
| PyTorch accumulate/retain/create/detach/no_grad/freeze | Preserve complete examples and explicit semantic distinctions, including frozen-weight input gradients |
| Custom HardSigmoid and surrogate gradients | Preserve piecewise derivative and boundary conditions; exact-gradient check does not validate intentional straight-through estimator |
| Higher-order and JAX | Correct false Hessian-diagonal claim; actual non-diagonal HVP and JVP/VJP programs, official JAX alternative mapping |
| Checkpointing and memory | Preserve dependency/recompute mechanism and complete deterministic program; idealK+L/Kmodel stated with assumptions, no fixed25%slowdown or universal activation dominance |
| Microbatch accumulation | Preserve with unequal2+3counterexample, actual−44versus−76⅔ and stated batch/state conditions |
| FSDP,ZeRO,pipeline/compression detail | Concise derivative-dependency bridge; actual specialist blueprint homes inspected below; no unsupported current implementation benchmark or fixed accuracy tradeoff |
| Practice | Changed numerical problems, diagnosis and independently specified primitive extension with hints/solutions and evaluable criteria |

## Destination note and neighboring ownership

[Finite-difference destination note](../../topic-notes/backpropagation-automatic-differentiation.md) is **adapted in content**: originalε1e−5claim investigated; $10^{12}+x$ cancellation reproduced; full smooth-network primitive/parameter checks retained; absolute/relative and nondifferentiability interpretation added. Its implementation/rendered evidence remains pending until phase2.

Scoped ownership reads: cross-domain-expansion.js plan “Mini-Batches, Training Loops & Gradient Accumulation” explicitly owns sample/batch/epoch/optimizer-state trace, unequal means and stochastic assumptions. gpu-expansion.js plans “Gradient Checkpointing (Activation Recomputation),” “FSDP (Fully Sharded Data Parallelism),” “Pipeline Parallelism (PP)” and “ZeRO Optimization Stages (1,2,3)” own actual scheduling, transient storage, API-generation and measured compute checks. This lesson retains the necessary local mathematical bridges, routing implementation depth there. Any compression-specific absence remains a scoped question rather than a claim that the entire catalogue lacks it.

No current source/blueprint/runtime changed. No next published-topic shortcut. Parent receives cross-owner findings rather than concurrent edits to their lessons.

## Canonical-reference section-list audit

Canonical free reference: Baydin/Pearlmutter/Radul/Siskind [JMLR survey](https://jmlr.org/papers/volume18/17-468/17-468.pdf),43pages. Section roadmap and relevant portions of §§2–3 were inspected; not a full43-page review. Section roadmap:1Introduction;2WhatADIsNot;3AutomaticDifferentiation;4ADinMachineLearning;5Implementation;6FutureDirections.

| Reference area | Coverage decision |
| --- | --- |
| Numerical versus symbolic versus AD | Core §2; concrete finite-difference error in §4 |
| Forward/reverse accumulation | Core reversegraph and deeper complete directional-products route |
| Machine-learning applications | Training core; input sensitivity/inverse problems and higher-derivative context without claiming causality |
| Implementation approaches | Core tensor/graph system; operator overloading in full teaching engine; source transformation and compiler internals routed to systems/compiler teaching |
| Future research/history | Optional annotated survey route, not necessary readiness or copied historical narrative |

The audit preserves broad depth without making the full engine or higher-order branch mandatory for the first-pass route.

## Hurdles and learning-experience design

| Hurdle | Local bridge and example | Evidence/representation |
| --- | --- | --- |
| Gradient is local, update is separate | Scalar neuron8loss/8weightgrad; MSEtwo-row large/small/no-op steps | GraphA and live investigationA |
| Branches and repeated operands accumulate | x*x,u+c*u | GraphB with both slot edges, cancellation/null |
| Bias gradient shape differs from output | Three uses per bias entry | Broadcast figureC and changed practice |
| Matrix derivative orientation | Indexed product derivation, row-batch table | Forward/reverse shape lanesD |
| Training success versus derivative correctness | Full engine XOR/digits plus same-state independent comparisons | Actual curvesE, author evidence |
| Numerical check can fail honestly | Sine step sweep, large-offset cancellation, ReLUcorner | InvestigationF and diagnostic practice |
| VJP/JVP/HVP differ | Actual2→3function and non-diagonal quadratic | FigureG, complete API programs |
| Storage is a dependency problem | Equal-layer segmented ideal and deterministic recomputation | TimelineH, explicit phase2 check |

First-pass route immediately after introduction: §§1–5 and practices1–4. Engine internals/JVP/HVP/custom/checkpoint and remaining practices are marked deeper. The code download is complete and offline, not a reference replacing the concept explanation. Interesting application arises from input sensitivities, graph cancellation, function-invariant numerical cancellation, and exact mean-loss accumulation; no trivia quota.

## Research record

Reviewed12September2026. No video playback claimed.

- Baydin survey: section roadmap and relevant numerical/symbolic distinctions, elementary traces and mode discussion; statements about exact speed/number of calls were independently qualified rather than copied.
- [Stanford CS231n notes](https://cs231n.github.io/optimization-2/): local addition/multiply, compound expression and chain-rule discussion. The informal max tie expression is not adopted as a unique derivative.
- [3Blue1Brown backprop lesson](https://www.3blue1brown.com/lessons/backpropagation/): introductory text companion on output desires/hidden contributions reviewed; hosted video offered as alternate route, not claimed watched.
- [PyTorch2.14 autograd mechanics](https://docs.pytorch.org/docs/2.14/notes/autograd.html): saved tensors, nonsmooth rules, disabling gradients, eval distinction and inplace checks read. Use the broad mathematical issue without assuming all implementation details permanent.
- [PyTorch2.14 gradcheck](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.gradcheck.gradcheck.html): actual signature, double-oriented defaults, nondifferentiable points, overlapping storage and nondeterminism.
- [JAX Autodiff Cookbook](https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html): Jacobian/Hessian composition and JVP mathematical/API sections; advanced-autodiff URL failed, so the successfully read cookbook is linked. No JAX program executed.
- [PyTorch JVP](https://docs.pytorch.org/docs/2.14/generated/torch.func.jvp.html) and [VJP](https://docs.pytorch.org/docs/2.14/generated/torch.func.vjp.html): official APIs located, actual behavior checked through installed2.14execution.
- [Checkpoint2.14](https://docs.pytorch.org/docs/2.14/checkpoint.html): current signature/use_reentrantFalse, recomputation and RNG/global-state caveats read; no benchmark adopted.
- [FNC5.5](https://fncbook.com/python/fd-converge/): central-difference convergence and stability/cancellation paragraphs; local functions and numbers are author calculations.
- UCI/scikit-learn source/loader license and historical partition were reviewed during the preceding packet; byte-identical retained input and attribution reused rather than redownloaded.

## Author work and verification actually performed

Existing shared Python3.12.14,NumPy2.3.5,sklearn1.9.1,torch2.14.0+cpu used read-only with1CPUthread. No installs or app/browser/formal phase2 check. teaching-autodiff.py was executed completely: XOR2000updates and digits500updates. author-calculations.py executed varied primitives, repeated backward reset, exact branch fixtures, allfourparameter groups against torch and central differences, extreme-logit CE consistency, actual finite-difference sweep, JVP/VJPduality,HVP, microbatch weight comparisons and both training runs.

Network gradient differences: torch≤5.56e−17,centralε1e−5≤1.16e−11 for the retained smooth initial fixture. Primitive inputs include repeated use,broadcast singleton,matrix,tanh,ReLUnegative/positive/zero,positive log-exp. These are bounded author calculations, not universal certification. A torch internal deprecation warning during functional transforms did not affect numeric output; do not rewrite APIs or install dependencies just to suppress it.

Other displayed standalone excerpts (PyTorch repeated-backward,customHardSigmoid,checkpoint) have explicit derived expected outputs; independent exact-excerpt replay and fresh install are deferred. The lesson does not label them measured benchmarks. Complete source is provided for each.

Full author reread of manuscript and specifications performed. Learning-experience checklist: all key symbols/shapes/reductions introduced; first-pass route honest; forward/backward/update distinctions clear; code scopes complete; same-state checks; actual real-data provenance and use limitations; independent changed practice; answer disclosure and null fixtures; appropriate static versus interactive forms; no fabricated curves or semantic gradients-as-causal-explanation. Screen geometry/assistive-technology behavior remain unverified until phase2.

## Finishing handoff

Render diagramsA–H and implement only the three specified investigations with shared input-bound output consistency rules. Carry downloadable teaching source, CSV and attribution; avoid executing general Python or training networks in the browser. Preserve on-demand topic ownership/loading and current module reading order. Verify per-topic numeric model/fixtures, exact programs and publication downloads, invalidation/reset, responsive/accessibility behaviors, log/zero plots, then independent review and relevant integration checks.

Only packet artifacts and own destination-note disposition changed. Temporary own __pycache__ was removed after absolute-path containment verification; author script now avoids bytecode generation. Prior packets and shared runtime are retained.

## Live exploration revision — 21 September 2026

The user replaced prediction-and-reveal lab assessment with immediately visible, playable exploration, including removal of optional prediction controls. The manuscript and specifications now use that contract. This changes the teaching interaction, not the topic's model predictions or its mathematical masks/gates.

**Topic disposition:** Trace credit and the effect of a step. Edit the two-example fit, learning rate, repeated-path coefficient and finite-difference step/offset. Show fitted line, residuals, derivative contributions and before/after loss immediately; step the backward accumulation without hiding the current total. The finite-difference panel displays both errors and the analytic derivative. Distinguish a correct gradient from a useful step size, and truncation/cancellation from a faulty derivative.

Retain all source data, formulas, measured results, code programs, references and independent practice. The existing author calculations remain evidence for those unchanged quantities, not evidence that a browser implementation already satisfies this new interaction contract. Phase two must implement and verify live updates, linked-view agreement, bounded work, reset, invalid/null cases, keyboard access and responsive diagrams. Content remains prepared; implementation remains not started.

## Prepared-content implementation — 21 September 2026 (author handoff)

The authorized finish preflight passed (`scratch/deep-learning-core-implementation/backpropagation-author-preflight.json`). The complete retained packet was consumed. Stable ID, title, existing manifest destination and module position remain unchanged. This section supersedes the packet's earlier “implementation not started” statements for author work only; independent review and final production integration are separate below.

### Sources and coverage

| Prepared obligation | Implemented ownership |
| --- | --- |
| Complete §§1–8 prose, mathematics, seven standalone programs, engine excerpts, eight practice tasks with separate hints/solutions, annotated references | `src/learn/data/topics/backprop.jsx`; author-only `scripts/build-backprop-lesson.mjs` preserves the full prepared manuscript in direct JSX and maps every visual marker to its component. Runtime never imports the Markdown packet. |
| A: scalar trace and two-example simultaneous update | `BackpropScalarFigure` in `BackpropFigures.jsx`; `BackpropFitLab` in `BackpropLabs.jsx`. Both current/updated fits, residuals, per-row gradient contributions and both losses are always visible. |
| B: repeated operands and two shared paths | `BackpropSharedLab`: one x node with two slot edges, one u node with two consumers; complete gradient stays visible while reverse stages highlight contributions. Cancellation at c = −1 and zero-input invariance are explicit. |
| C/D: broadcast reversal and two-layer pullback shapes | `BackpropBroadcastFigure` and `BackpropShapeFigure`: feature-column correspondence, exact 9/12 sums, input×output convention, every object/gradient shape and contracted-index explanations. Phone layouts stack correspondence blocks. |
| E: actual training observations | `src/learn/data/backprop-training.js` copies only the retained XOR/digit observations; `BackpropTrainingFigure` preserves initial values, actual update spacing, log-scale XOR loss, full 0–120 digit count axis and exact tables. No browser training or invented per-image predictions. |
| F: finite-difference evidence | `BackpropDifferenceLab` displays two editable checks, actual evaluated numbers, local sampled function/secant, analytic derivative and absolute/relative error; sine, large-offset linear, zero-reference square and separate ReLU corner interpretation. |
| G/H: directional products and storage dependencies | `BackpropProductsFigure` has signed product bars, Jacobian arithmetic and duality values. `BackpropCheckpointFigure` has unique saved/regenerated slots and a full dependency-valid eight-operation schedule, with no byte or speed claims. |
| Complete offline engine, real data, attribution, same-state checks and all displayed programs | `public/learn-assets/backpropagation/`: engine, licensed CSV, combined provenance, author calculations, saved observations and seven purpose-named complete Python files. |

The pure model is `src/learn/data/backprop-models.js`. UI ownership is limited to `BackpropShared.jsx`, `BackpropLabs.jsx`, `BackpropFigures.jsx` and `backprop-labs.css` under `src/learn/components/lesson-labs/`. Native range controls preserve arbitrary valid exact values (`step="any"` with bounded arrow-key increments); explicit unique labels bind sliders and numeric editors separately. Invalid numeric buffers retain the stated last valid result. Changing model inputs preserves the shared-path inspection stage. Reset clears buffers and restores dependent state.

### Author evidence actually executed

- `scripts/verify-backprop-models.mjs` → `docs/teaching/evidence/backprop-models.json`: **174 assertions**, including independent finite differences of varied two-row objectives, shared-path/operand identities, both null families, broadcast sums, robust finite-difference contrasts, JVP/VJP duality, checkpoint input availability and exact recorded-data identity.
- `scripts/verify-backprop-native.py` → `docs/teaching/evidence/backprop-native.json`: **nine complete programs and 51 assertions**, running the full engine, the full author calculation script in an isolated temporary copy, and all seven standalone examples. Python 3.12.14, NumPy 2.3.5, PyTorch 2.14.0+cpu, scikit-learn 1.9.1; one CPU thread. The previously deferred custom-operation gradcheck and checkpoint comparison both execute successfully. All six XOR and six digit observation rows reproduce; source CSV SHA256 is retained. Prepared source/data files are not mutated.
- `scripts/verify-backprop-browser.cjs` → `docs/teaching/evidence/backprop-browser-author.json`: author **development-preview** checks at 1366/390/320 px, including immediate outputs, real pointer input, every visible slider's native endpoints/interior keyboard edits, non-preset exact input, invalid/reset, fast edits, preserved inspection stage, explicit cancellation/zero-reference cases, download responses, KaTeX, control paint and containment. Current source hashes and screenshots are in the receipt. The final production run must use a separate `backprop-browser.json` receipt.
- Early Babel parsing succeeded for the body and all three component modules before browser review. Author checks intentionally do not run the shared application build while sibling authors edit.

### Author learning-experience pass and corrections

The route keeps the first-pass §§1–5 independent of the deeper engine/JVP/HVP/custom/checkpoint material. The complete manuscript and all eight changed-constraint practices are retained; no learner prediction/commit/grading feature was added. Each numerical caution has its own mechanism: local gradient versus step length, path accumulation, broadcast reduction, same-state evidence, nonsmooth convention, finite precision, and checkpoint dependency preservation. No new uncertain scientific claim or API was introduced beyond the researched packet; current runtime execution substantiates the displayed API examples.

The rendered pass found and repaired a real 390 px page overflow from long display equations by containing their own scroll, without any descendant SVG sizing rule. Image review then found undersized phone SVG annotations and squeezed table headings; phone font sizes were enlarged, exact tables receive deliberate horizontal scrolling, and their visible captions reflow outside the scroll region. Screenshot capture hides only global fixed navigation/sidebar during element capture so it cannot paint over a tall diagram; ordinary control/containment checks use the unchanged page. Retained font assets are fulfilled locally and hash-recorded because the sandbox cannot reliably fetch the Google Fonts endpoints.

Author evidence is not independent certification. Independent content/learning review, any resulting repairs and final production/build/loading integration remain with the designated reviewers/integration owner. The destination note's numerical obligations are implemented and checked; final closure should link the independent and production receipts. No shared manifest, registry, ledger, global stylesheet or unrelated lesson was changed by this author.

### Independent-review repairs and final author freeze

The designated reviewer requested readable ordinary-prose spacing, an integrated section route, meaningful table captions, reverse-pointing highlighted arrows during the shared-path backward stages, and persistent starting-case summaries in all three investigations. These are implemented. The generator protects code and mathematics while normalizing ordinary text, sets `hasIntegratedGuide: true`, and emits eight unique semantic section links. The six prose reference tables have specific captions. Active reverse arrows now point toward the graph inputs while the forward-use edges remain explicitly described. Each live lab preserves a compact baseline alongside the edited result.

The affected author browser suite was rerun after these repairs: **94 assertions and 30 captures** at 1366/390/320 px, including the route, fixed baselines and active reverse-arrow markers. The source-bound result is `docs/teaching/evidence/backprop-browser-author.json`. Numerical model/native programs are unchanged, so their passing evidence is reused. The standalone stable-ID blueprint is `src/learn/data/curriculum/blueprints/backpropagation-automatic-differentiation.js`; its shared registration belongs to the integration owner. Runtime sources are frozen for final independent closure and shared production integration. The separate independent reviewer owns its report and certification.

## Final production integration — 21 September 2026

The author handoff above is closed by independent review and the final production browser pass. The [first-five completion record](../../DEEP-LEARNING-CORE-IMPLEMENTATION.md) links the reviewed lesson, native/model evidence, actual production checks and preserved scope baseline. Next/previous order, selected-body loading, section anchors, themed links and applicable rendered math geometry pass in the integrated build. Both delivery phases are complete; user acceptance is separate. Earlier pending integration sentences describe the historical author checkpoint, not current work.
