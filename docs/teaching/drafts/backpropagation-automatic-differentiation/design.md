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
| Gradient is local, update is separate | Scalar neuron8loss/8weightgrad; MSEtwo-row large/small/no-op steps | GraphA and graded investigationA |
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

Render diagramsA–H and implement only the three specified investigations with shared input-bound grading rules. Carry downloadable teaching source, CSV and attribution; avoid executing general Python or training networks in the browser. Preserve on-demand topic ownership/loading and current module reading order. Verify per-topic numeric model/fixtures, exact programs and publication downloads, invalidation/reset, responsive/accessibility behaviors, log/zero plots, then independent review and relevant integration checks.

Only packet artifacts and own destination-note disposition changed. Temporary own __pycache__ was removed after absolute-path containment verification; author script now avoids bytecode generation. Prior packets and shared runtime are retained.
