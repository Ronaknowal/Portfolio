# Advanced optimizers: content design and handoff

Stable ID: `advanced-optimizers-lion-sophia-prodigy-schedule-free`. Module: Deep Learning Fundamentals & Architectures, position37. Delivery: **research and writing only**,13 September2026. Content checkpoint comprises the complete [manuscript](lesson.md), [visual/investigation specifications](visual-specifications.md), [provenance](data-provenance.md), full programs, actual data and saved calculations/model states. Implementation remains **not started**. Run the topic's `--work finish` preflight only under a later authorized implementation request.

## Scope, preflight and conservation

Actual topic preflight ran with `--work content`; the published body and catalogue scope were read, not inferred from a title. No destination note existed for this owner. The complete original JSX was read in contiguous ranges, including the tail recovered after truncated output. Original source: `src/learn/data/topics/advanced-optimizers-lion-sophia-prodigy-schedule-free.jsx`; SHA256 `179254cc0ec5b8c9232b6a3ad7862cca77b618383bac7020ea47dbf81c20539c`, baseline commit `8c5da59f18516be77c29d5aeeafca3decca4f738`. The original publication remains untouched.

| Original area | Decision in the new manuscript |
| --- | --- |
| Motivation, optimizer taxonomy, historical context | Retain practical resource/tuning/end-horizon questions; teach gradient→parameter→prediction first. Separate optimizer trajectory from preceding Ring Attention's computation-preserving distribution. |
| AdamW intuition, formal update and native parity | Retain complete baseline and runnable parity. Correct second moment≠variance, direction not universally≤1, actual dtype/precision and bounded parity claims. |
| Lion sign intuition, rule, discovery and tuning | Retain both betas, program search, one-state motivation, actual worked history. Repair fixed nonzero sign assumption, magnitude irrelevance, automatic100×decay and universal precision claims. |
| Sophia-H/G formal mechanisms and implementation | Retain both estimators and clipped update. Restore parameter Jacobian, batch normalization, independently sampled labels and coherent forward states; distinguish GGN from full Hessian and paper/package epsilon/scaling conventions. |
| Prodigy adaptation and code | Replace capped Adam-like sketch with full paper Algorithm4, all histories/initialization, scaled moments and old-d/next-d order. Preserve theory motivation with its assumptions, not universal nonconvex tuning guarantees. |
| Schedule-Free interpolation, averaging and curves | Restore y=βx+(1−β)z, post-update indexing, actual averaging coefficients, x evaluation and library modes. Explain conditional theorem/horizon and BatchNorm. Preserve observations where y happens to score better. |
| Tiny language-model comparison and output | Replace unsupported synthetic benchmark curves with declared24 actual fits on licensed real handwriting, full weights/history and all candidate results. Do not retain invented losses/timings or imply the linear classifier is an LLM. |
| Visual walkthrough and labs | Preserve mechanism exploration while specifying13 inline figures and five distinct investigations with fresh inputs, genuine contrasts/nulls and feedback. One mechanism may need multiple representations; no uniform lab quota. |
| Scaling, memory heatmaps, ZeRO and production | Preserve resource reasoning; count named arrays/dtypes, distinguish payload from peak and moment storage from gradient/parameter collectives. Replace universal method rankings/adoption rumors with testable decision criteria. |
| Adafactor and neighboring optimizers | Retain shape-dependent factors and optional state; brief current Muon singular-geometry bridge adds useful coverage. Its finite iterations are not ideal polar output and it is not Hessian estimation. |
| Failures, references and five exercises | Retain diagnostics and expand to nine changed problems, each with separate closed hint/solution, annotated primary/article/video routes and explicit readiness/transfer tasks. |

Keep the stable title/ID: the four named methods remain the center, with AdamW prerequisite, Adafactor storage comparison and a short Muon extension. Do not turn this lesson into an optimizer catalogue. A deeper matrix-update/curvature comparison is routed to the existing [second-order methods owner](../../topic-notes/second-order-methods-l-bfgs-k-fac-shampoo-natural-gradient.md); its note remains open until implemented. The next actual route lesson is Neural ODE, not a later published optimizer page.

## Learner hurdles, representations and applications

| Hurdle and observable outcome | Teaching mechanism | Independent use |
| --- | --- | --- |
| Explain why a gradient alone does not select a safe step | Local derivative, noise/scale and actual update loop | Diagnose gradient, parameter and update norms separately |
| Calculate AdamW/Lion and identify state order | Two moment lanes and signed history balance | Changed gradient/history/decay; explain reversal despite unchanged gradient sign |
| Separate gradient magnitude from curvature | Rotated quadratic plus two estimator instruments | Predict sampled-label expectation; find negative probe in a PSD matrix |
| Follow adaptation without an optimum oracle | Prodigy displacement accounting and full state trace | New target/start/scale; detect overshoot and wrong-code sketch |
| Distinguish gradient/evaluation parameter locations | Spatial x/y/z board and averaging-weight strip | New perturbations, endpoint beta contrasts, correct evaluation mode |
| Connect formulas to a real model | Edited digit→ten scores→residual gradient→state update | Change pixels and label, preserve label-independent pre-update inference |
| Compare training/resource claims honestly | Actual declaration/selection/curves and allocation ledger | Design a12-run experiment, count factored/sharded bytes, distinguish time/work/steps |

Applications are developed where they explain a mechanism: automated optimizer-program discovery shows an inspectable result from search; repeated model-scale experiments motivate adaptation; uncertain run horizons motivate an evaluation iterate; handwritten pixels expose concrete parameter contributions; singular geometry explains why matrix transformation differs from an entrywise sign. They are not decorative trivia or current deployment claims. Ring Attention links communication accounting; the actual Neural ODE continuation distinguishes model-state evolution from training-parameter evolution.

## Canonical research and coverage decisions

Sources were opened during this authoring work,13 September2026. Read extent is stated below; bibliographies, appendices or full proofs were not silently credited as read. Algorithms/formulas are independently explained and implemented here. Original wording is not reproduced as a new article. Current repository source is version-sensitive; later implementations should recheck the specific options they actually use.

### Lion: actual reference agenda

[Symbolic Discovery of Optimization Algorithms](https://arxiv.org/html/2302.06675v4) exposes Introduction; Symbolic Discovery(search space, efficient search, generalization/selection/simplification); Derivation/Analysis; Evaluation(image classification, vision–language contrastive, diffusion, LM/finetuning, other optimizers, ablations); Hyperparameter Tuning; Limitations; Related Work; Conclusion; appendices algorithms, vision/NLP setups, other programs, architecture details, proxies, landscape, functions and abstract execution. The actual hierarchy was inspected. Read introduction/programs, search-space and selection/simplification passages, full derivation/analysis, relevant evaluation/tuning/ablation passages and limitations opening; not every appendix or full experimental table.

Disposition: teach the discovered compact rule, separate beta roles, reasoned tuning range and proxy-to-transfer discovery process. Explain evaluated application families without reprinting benchmark tables or converting them into universal rankings. Search-engine implementation and all proxy recipes remain optional paper depth, outside the optimizer-use outcome. Google's [complete Lion source](https://raw.githubusercontent.com/google/automl/master/lion/lion_pytorch.py) was read through its update, including Apache header; no third-party source was copied into our program and no claim of executing that package is made.

### Sophia: actual reference agenda

[Sophia](https://arxiv.org/html/2305.14342v4) contains Introduction; Method(motivation, Sophia, estimators); Experiments(setup/tuning, evaluation methodology/technical details, results, analysis, ablation); Theory; Related Work; Conclusion; appendices additional results, detailed setups/tuning/downstream, limitations and proofs/lower bound. Read the actual hierarchy, full core method and estimator sections including Algorithms1–3 and batch/Jacobian argument, and experimental setup opening. Full theory proofs and all downstream tables were not read.

Disposition: core teaches both Hessian-vector and GNB instruments, coordinate clipping, actual normalization/state timing and costs. Proofs are not reproduced; performance is source-scoped and never a universal2× claim. The [complete official Sophia implementation](https://raw.githubusercontent.com/Liuhong99/Sophia/main/sophia.py) was read: unscaled `update_hessian`, `bs`-scaled denominator and additive epsilon differ from the paper-scaled/max-epsilon teaching rule. This is a documented variant boundary, not an untested parity claim.

### Prodigy: actual reference agenda

[Prodigy](https://arxiv.org/html/2306.06101v3) contains Introduction; Prodigy approach; D-adaptation with resetting; lower complexity of exponentially bounded methods; Related Work; deriving Adam-like step sizes; Experiments(logistic, CIFAR10, nanoGPT, large-scale LSTM/RoBERTa/GPT/DLRM/VarNet/ViT); Conclusion; proof appendices. Read actual hierarchy, introduction, resetting statement/rule, main lower-complexity discussion, related work, full Algorithm4/derivation and logistic/CIFAR/nanoGPT/large-scale experiment passages. The complete §2 algorithms and appendix proofs were not read and no proof-completeness claim is made.

Disposition: retain convex distance-scale motivation and full executable Adam-style rule; distinguish theoretical variants from nonconvex use. Resetting/lower-bound proof machinery is optional source depth, not needed to execute or diagnose the present algorithm. The [complete current package source](https://raw.githubusercontent.com/konstmish/prodigy/main/prodigyopt/prodigy.py) and [README options](https://github.com/konstmish/prodigy) were read. Record rescaled adaptation state, optional bias correction/slicing, epsilon/update detail, warmup safeguard and global reductions. They are not silently substituted for the paper rule. No package execution or quality guarantee is asserted.

### Schedule-Free: actual reference agenda

[The Road Less Scheduled](https://arxiv.org/html/2405.15682v2) contains Introduction(summary/notation); Method(general and larger learning rates); Related Work; Experiments(deep learning, AlgoPerf, convex, implementation); Conclusion/contributions; appendices proofs, online-to-batch/linear weights, Bregman, acceleration, strongly convex, large-step and detailed experimental families. Read actual hierarchy, introduction, core rules and main theorem statements, large-rate discussion, related-work connections, convex/implementation sections including Algorithm1/BatchNorm/warmup and conclusion. Selected competition failure passages were read; full appendix proofs and every figure were not.

Disposition: teach x/y/z mechanism, actual coefficients, mode/normalization handling and a bounded theorem caveat. Optional theorem proof and all competition recipes stay with the primary source. Read [official repository guidance](https://github.com/facebookresearch/schedule_free) and complete current [reference](https://raw.githubusercontent.com/facebookresearch/schedule_free/main/schedulefree/adamw_schedulefree_reference.py) and [optimized implementation](https://raw.githubusercontent.com/facebookresearch/schedule_free/main/schedulefree/adamw_schedulefree.py). Current inner momentum and decay options affect storage/behavior; readable and optimized array counts are separate. Source reading does not establish that a current package was executed. [NeurIPS author-video page](https://slideslive.com/39024867/the-road-less-scheduled) metadata/title/attribution were checked; recording not watched or used to substantiate calculations.

### Supporting depth and current information

[AdamW paper](https://arxiv.org/abs/1711.05101) metadata/abstract and [PyTorch2.14 algorithm/options](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html) were read; native fixed-sequence parity executed. [Dive into Deep Learning Adam](https://en.d2l.ai/chapter_optimization/adam.html) algebra, implementations and Yogi opening were read as an alternate prerequisite route; its remote notebooks were not run and loose variance terminology was corrected locally.

[Adafactor](https://proceedings.mlr.press/v80/shazeer18a.html) metadata/abstract and the primary PDF's factorization/Algorithm2/marginal argument were read. This supports the actual shape-dependent memory example and optional-momentum boundary; the complete later algorithm is not claimed reconstructed here. [PyTorch2.14 Muon](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Muon.html) current rule, finite Newton–Schulz coefficients, shape requirements and example were read. Its exact current numerical recipe is referenced, while only an ideal polar analogy is calculated locally. UCI source attribution/license metadata and inherited extract provenance were checked as recorded in the data document.

## Evidence, author reread and handoff

The comparison declaration was set before fitting:650-parameter real-data classifier,240/80/80 split, two seeds, two method-specific scales,400 updates, zero decay and validation-only final selection. All24 fits executed and all12 selected models/states are retained. No timing benchmark, pretrained model, late hyperparameter retuning or erased failed comparison supports this packet. See provenance for exact outputs, software, roles, model counts and limitations.

Bounded author checks executed: native AdamW parity across four nonidentical gradients including zero and nonzero decay; independent sampled-label outcome enumeration versus analytic curvature; all Hutchinson probes versus diagonal; Prodigy EMA expansion versus recurrence; stationary and altered-scale traces; Schedule-Free beta/weighting/stationary calculations; real edited-pixel and changed-label gradient controls; frozen-cosine update null; ideal polar SVD; changed practice loss/resource arithmetic. A final stored-state check recalculates all selected metrics and selection from retained candidates and checks data-role separation. It reuses the24 completed fits rather than retraining unchanged inputs.

The author reread the full current manuscript, all specifications, complete programs, provenance and this design. Findings addressed before checkpoint: corrected a floating-point-sensitive Lion null to exact-zero state; gave each investigation a fresh default separate from worked exposition; computed Sophia fresh batch, Prodigy scale contrast and retained small real-pixel changes with a delta view; preserved Schedule-Free y-better counterexample and cosine zero-rate behavior. The final manuscript has nine changed practice questions with eighteen closed hint/solution disclosures plus two optional theory branches.

Learning-experience checklist, author assessment:

1. First-pass route is visible; ordinary weighted averages and derivative meaning precede specialist terms.
2. Each hard mechanism has a matching visual representation and a complete worked calculation.
3. Five investigations edit real mathematical/data entities, require an unset prediction and recompute feedback from current inputs.
4. Fresh/changed/null fixtures demonstrate the intended contrast; small or absent effects are retained honestly.
5. Real data, full model state and complete runnable programs connect formulas to predictions.
6. Nine exercises change constraints or values, with reasoning and closed hints/solutions; no fixed interview/mastery guarantee.
7. Measured fits, exact arithmetic and hypothetical storage are labeled at their point of use.
8. Warnings explain specific mechanisms; core intuition remains readable and optional proof/library branches do not interrupt it.
9. Canonical scope, annotated alternatives, route continuity, title decision and destination ownership are recorded; no identical-lab template was imposed.

This is author evidence and content reconciliation, **not independent phase-two review**, rendered accessibility verification, implementation completion or user acceptance. There is no known material writing gap. Next action: on authorized finish, consume this full packet, implement topic-specific figures/labs and complete required independent/content/model/browser/accessibility/build/integration checks. Keep current publication and unrelated proposals unchanged until that authorized work.
