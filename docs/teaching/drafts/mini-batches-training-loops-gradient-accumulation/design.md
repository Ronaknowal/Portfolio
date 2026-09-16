# Mini-Batches, Training Loops & Gradient Accumulation — content/design record

Stable ID: `mini-batches-training-loops-gradient-accumulation`. Deep Learning Fundamentals & Architectures, position 41 of 42, Training Mechanics & Diagnosis. Author: `training_mechanics_content`, coordinated by root. Revision 1, content-first, authorized 13 September 2026 within the [remaining-module scope](../../DEEP-LEARNING-MODULE-CONTENT-COMPLETION.md). Root owns the central delivery ledger, scope/handoff and inventory; this record does not replace phase status there.

## Intended outcome and source conservation

The original entry was planned, with no published lesson body. The actual `--work content` preflight was run on the exact stable ID, returned content in-progress / implementation not-started, and supplied the complete blueprint and authoring notes. No destination note existed for this topic. The unassigned inbox's resolved bit-manipulation history is unrelated and creates no obligation here. The full stored blueprint was read both through the preflight and at `src/learn/data/curriculum/cross-domain-expansion.js`, the Mini-Batches plan. No live catalogue, blueprint, manifest, route, runtime source or existing draft was changed.

Original blueprint conservation:

| Original promise | Where preserved and developed |
| --- | --- |
| Trace one update through examples, loss, gradients and optimizer state | §§2–4 scalar derivation, parameter/gradient/momentum state stores, full runnable trace and investigation 1 |
| Distinguish samples, batches, epochs and optimizer steps | §1 nested clocks and tail, §8 successful versus skipped AMP attempts, practice 1 |
| Scale accumulated microbatch loss correctly | §§3–5 linearity and numerator/denominator derivation, uneven chunks, target weights/masks, investigation 2 and practice 3 |
| Tiny batch with explicit shapes, forward/backward and separate optimizer state | §2 row table and linear-classifier shapes, §6 network shape4→8→3, full CPU programs |
| Compare training and evaluation modes | §6 eval/no_grad distinction and §7 output versus running-buffer behavior, investigation 3 |
| Step microbatches with separate weights/gradients/counters | Figure B and investigation 1 retain this exact teaching question; entity editing, prediction capture and fault/null cases make it an investigation |
| Match a large-batch update under reduction/stochastic assumptions | §3 stated identity; §6 multi-update state comparison; §7 concrete batch-normalization and dropout counterexamples |
| Misconceptions: backward updates weights; epoch equals step | Directly resolved in state trace, clock lanes and changed practice |
| Primary source: PyTorch Basics | Reviewed actual optimization chapter, preserving its basic learning order while correcting its batch-mean reporting shortcut for this lesson's uneven batches |

Retain the current title. Its three terms accurately describe the finished scope; target weighting, mode semantics and execution boundaries are necessary parts of implementing these mechanics. No stable identity, progress, title or sequence change is proposed. Actual predecessor Titans retains the preceding position; diagnostics follows and closes the module. Root teaches Titans' inner-memory updates separately; this lesson concerns ordinary optimizer updates across data batches.

## Learning contract and scope decisions

Beginner finish line: from a small dataset and a declared mean objective, compute one update, identify which state changes at each call, and write an accumulation loop that handles an uneven final group. Intermediate finish line: apply the same contract to masked/weighted supervision and offline measured observations; compare fixed groups across different physical partitions. Deeper finish line: explain failures of physical-batch equivalence and place clipping, scheduler, AMP and DDP operations correctly without mistaking this packet for a complete distributed-training tutorial.

Prerequisites checked in actual material: backpropagation's pending complete manuscript teaches chain-rule paths, broadcasting sums and linear-layer shapes in §§1–3; the existing NumPy lesson teaches axes/reductions, broadcasting and matrix multiplication in §§5–8. Links use the stable `/learn/topic/<id>` route, confirmed from current lesson source. The local scalar derivative and shape refresher make the core self-contained even if a reader has not completed those routes. Their publication/review states are distinct from this packet's preparation.

| Hurdle | Representation and explanation | Assessed evidence | Route |
| --- | --- | --- | --- |
| Different clocks and incomplete work | Ten-row nested lanes, ceil counts with explicit sampling/tail policy | 23-row changed count, drop-last versus accumulation-tail explanation | Core |
| Backward deposits derivatives; optimizer changes state | Three-row derivative table, three storage lanes, executed scalar program | Fresh row edits, before-backward/after-step prediction, policy comparison | Core |
| Unequal chunk means reweight rows | Exact coefficient bars and linearity proof | Different partition/targets and equal-mass null | Core |
| Supervision mass differs from tensor size | Weighted/masked numerator and denominator rails, token/sequence example, loss-convention table | Editable inclusion/weight table, undefined-mass case and variable-length practice | Core |
| Loop contract holds over multiple updates | Real Iris data, complete4→8→3 program, identical initial/order/optimizer pairing | Parameter and momentum comparisons plus changed microbatch size7 | Core |
| Physical forwards can change the objective before reduction | Shared/local normalization number lines, explicit downstream scale, fixed dropout masks | Fresh normalization fixture and matching-output/different-buffer null | Deeper |
| Nonlinear/stateful operations live at update boundaries | Clipping counterexample, schedule clocks, AMP branch strip and DDP derivation | Ordering/diagnosis practice; executable arithmetic, no hardware claim | Deeper |

First-pass route is explicitly placed immediately after the introduction. §§1–6 with practices1–4 reach a complete useful loop; §§7–8 and practice5 supply specialized assumptions and transfer. Estimated reading/practice times were assigned after the actual manuscript existed; no word/lab quota drove the design.

Chosen applications add distinct understanding: classification on measured Iris observations establishes a real usable loop and a paired-state comparison; variable-length supervised targets transfer denominator reasoning beyond example count; normalization makes partition dependence visible before backpropagation; DDP exposes the same denominator error across ranks. A training speed comparison is excluded because no timing/peak-memory measurement was needed to resolve the authored claim. Hyperparameter sweeps, broad debugging, multi-seed inference and checkpoint implementation are diagnostics or later owners' work. The diagnostics author received the exact scalar fixture and explicitly linked the correct-loop boundary; its independent Wine experiments are not imported here.

Saved discovery: [Data Parallelism (DDP): unequal target mass](../../topic-notes/data-parallelism-ddp.md). The existing DDP brief already plans uneven inputs; the new open note supplies a concrete rank-weighting fixture, source locator and implementation boundary. Root authorized that destination-note edit. No destination runtime was changed. Mixed precision and GPU performance remain brief links to their existing owners, not new rewrite requests.

## Canonical-reference section-list audit

Canonical treatment selected: Zhang, Lipton, Li and Smola, *Dive into Deep Learning*1.0.3, [§12.5 Minibatch Stochastic Gradient Descent](https://d2l.ai/chapter_optimization/minibatch-sgd.html). Actual page section list inspected 13 September 2026 (not inferred from memory):

| Actual section | Disposition in this packet |
| --- | --- |
| 12.5.1 Vectorization and Caches | §1 explains shared matrix work and framework/data-reuse overhead; architecture-specific cache numbers and timing benchmarks deliberately remain in that reference and the GPU performance owners. No source benchmark is relabeled as ours. |
| 12.5.2 Minibatches | Mean gradient, independent-draw variability derivation and finite-data example in §§1,3,7. Added distinction between changing an effective group and only repartitioning it. |
| 12.5.3 Reading the Dataset | §1 loader/sampler/collation bridge and complete supplied real-data pipeline in §6. Use Iris rather than the book's Airfoil because this lesson asks about classification-loop equivalence; fit preprocessing only on the training split. |
| 12.5.4 Implementation from Scratch | Manual scalar derivative/update and explicit momentum state, then full trace program. A second general autograd engine is already owned by backpropagation and unnecessary here. |
| 12.5.5 Concise Implementation | Full PyTorch program with recorded loss/counters/state comparison. Preserve the distinction between the book's half-squared convention and PyTorch MSE. |
| 12.5.6 Summary | Recap/readiness connects statistical batching and computational partitioning; no universal speed or optimal-batch assertion. |
| 12.5.7 Exercises | Independently authored changed-data/tail/target-mass/state practice with hints and closed solutions, plus exact size7 transfer. Book learning-rate/timing/sampling exercises are useful alternate practice, not copied wholesale. |

Actual canonical read extent: complete section agenda, opening batch/vectorization rationale, §12.5.2 gradient/variance discussion, relevant PyTorch dataset and implementation snippets in §§12.5.3–.5, and summary/exercise statements. Other-framework implementations and full hardware benchmark execution were not reviewed/run. Headline facts found missing from the first blueprint were statistical versus computational batching tradeoff, sampling policy and loss convention; these were included. The reference's cache implementation depth remains deliberately out of scope, not silently omitted.

Complementary canonical API agenda: PyTorch *Optimizing Model Parameters* has Prerequisite Code, Hyperparameters, Optimization Loop, Loss Function, Optimizer and Full Implementation. Those actual headings and the core code were inspected. Our packet preserves setup/forward/loss/backward/optimizer/evaluation, adds the essential uneven-chunk and final-partial-group contracts, and explicitly adapts the tutorial's equal average of evaluation batch means to numerator/denominator aggregation. Validation and test roles are named separately here.

## Claim/source ledger and actual review extents

All sources below were accessed 13 September 2026. PyTorch API references use the inspected2.14 snapshot; local execution versions are listed separately. Links supplement the complete local reasoning.

| Claim or convention | Source and exact area | What was actually checked / limits |
| --- | --- | --- |
| Basic forward/loss/backward/step sequence and mode calls | [PyTorch optimization tutorial](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html), Hyperparameters through Full Implementation | Read these substantive sections and both loops. Identified batch-mean evaluation averaging as a shortcut inappropriate for unequal group mass. Did not execute FashionMNIST downloads/training. |
| Computational versus statistical batching | [D2L12.5](https://d2l.ai/chapter_optimization/minibatch-sgd.html), extent above | Section audit and relevant explanations read; original finite-gradient variance example derived and exactly calculated locally. No CPU/GPU performance extrapolation. |
| DataLoader, sampler/collation and drop_last | [torch.utils.data](https://docs.pytorch.org/docs/2.14/data.html), Loading Batched and Non-Batched Data, automatic batching, BatchSampler/DistributedSampler notes | Inspected finite map-style batch semantics and iterable/worker differences. Manuscript's count formula explicitly scopes itself to the ordinary finite case. No multiworker runtime experiment. |
| Missing gradient versus zero gradient | [Optimizer.zero_grad](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Optimizer.zero_grad.html), parameter notes | Read skip-versus-zero behavior and None slots. Scalar trace verifies clearing preserves weight/momentum; API note covers branch-unused parameters. |
| Momentum equation under stated configuration | [SGD](https://docs.pytorch.org/docs/2.14/generated/torch.optim.SGD.html), recurrence and PyTorch implementation note | Reviewed convention; example sets dampening0, no Nesterov/decay. Local first-step trace and Fraction calculation establish displayed values. |
| Weighted class-index versus probability-target CE | [CrossEntropyLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html), both unreduced/reduction formulas, ignore_index and shapes | Read formulas and target/shape contracts. Ran weighted integer-target full/split gradients, total eligible weight7; separately verified probability-target denominator4, and all-ignored sum0. No label smoothing code is supplied. |
| MSE reduction counts elements | [MSELoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MSELoss.html), mean-reduction definition | Inspected API definition; manuscript declares the scalar half-squared convention explicitly, instead of equating it with default MSE. |
| BatchNorm depends on current physical batch; running buffers | [BatchNorm1d](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm1d.html), normalization and running estimates notes | Read population/unbiased variance distinction, momentum update and track_running_stats caveat. NumPy current-output/running-mean calculations compared with actual Torch for contrast/fresh/null/permuted cases. Model does not claim to implement preceding-layer gradients or running variance. |
| Dropout realization and eval behavior | [Dropout](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Dropout.html), mask/scale/evaluation definition | Read semantics. Original fixed-mask gradients8/20 and disabled6.5 calculated; no claim that a particular hardware RNG yields these exact masks. |
| eval and gradient recording are distinct | [no_grad](https://docs.pytorch.org/docs/2.14/generated/torch.no_grad.html), definition and example; BatchNorm reference above | Read reverse-mode recording behavior; Iris evaluates under both eval and no_grad. Factory-function/forward-mode exceptions are outside the ordinary forward inference context described. |
| Clip complete unscaled gradient; scale fixed per group | [AMP examples](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html), Gradient clipping and Gradient accumulation | Read ordering, one unscale per step, fixed scale across group, inf/NaN skip and scaler-update boundary. Arithmetic clipping counterexample independently checked. CUDA/AMP not executed. |
| Scheduler clock and optimizer-first order | [torch.optim](https://docs.pytorch.org/docs/2.14/optim.html#how-to-adjust-learning-rate), scheduling section | Read update ordering and scheduler classes; packet distinguishes update/epoch/metric clocks. No claim that every scheduler has one identical API or skip-detection implementation. |
| Default rank averaging and no_sync forward coverage | [DDP](https://docs.pytorch.org/docs/2.14/generated/torch.nn.parallel.DistributedDataParallel.html), reduction notes and no_sync | Read relevant semantics. Global-denominator equation derived and CPU rank arithmetic checked; no distributed runtime executed. Saved destination note carries the concrete implementation question. |
| Real-data attribution and variant | [UCI Iris](https://archive.ics.uci.edu/dataset/53/iris), feature/target/license/citation metadata; [load_iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html), version0.20 corrections note | Both pages checked. CSV exported from actual installed1.9.1 bundle; added header/row IDs and retained all150 observations. [Provenance](data-provenance.md) supplies the adaptation/license/split record. |

Learner alternatives curated in the manuscript, not just here: the beginner optimization tutorial, D2L chapter and [Training with PyTorch video/companion](https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html). For the video alternative, actual inspected extent is companion Introduction, Dataset/DataLoader abstractions, Optimizer, Training Loop and Per-Epoch Activity including validation/checkpoint code. The embedded video was not watched, no timestamps are claimed, and TensorBoard/checkpoint runs were not executed. The accessible companion grounds the fit recommendation; metadata alone was not used to endorse technical content. Older/default code choices should be adapted to this lesson's explicit objectives and current API snapshot.

## Retained author calculations and data

Required files: [lesson.md](lesson.md), [visual-specifications.md](visual-specifications.md), this design, [data-provenance.md](data-provenance.md), [iris.csv](iris.csv), complete [trace_update.py](trace_update.py) and [train_iris.py](train_iris.py), [author_checks.py](author_checks.py), [trace-output.txt](trace-output.txt), [iris-output.txt](iris-output.txt), [author-results.json](author-results.json). All are necessary content/evidence handoff inputs. The author created no scratch working directory, package installation, image files, browser sessions or disposable download helpers; nothing unrelated was cleaned.

Author command from repository root:

```text
scratch/lesson-tools/Scripts/python.exe -B docs/teaching/drafts/mini-batches-training-loops-gradient-accumulation/author_checks.py
```

Environment actually used: Python3.12.14, Torch2.14.0+cpu, NumPy2.3.5, Windows11, one Torch CPU thread, float64. Source input hashes are embedded in `author-results.json`; the central ledger will bind the full final packet. The source hash proves correspondence, not review approval.

The bounded calculations establish:

- Scalar mean gradient−5/3, accumulated contributions−2/3 and−1, update1/6, post-update loss67/108, and the separate momentum continuation13/60. Integer/fraction arithmetic supplies an independent check on displayed roundoff values.
- Fresh state lab correct/clearing/early-step alternatives, plain SGD versus momentum, one-chunk and zero-rate nulls. The early-step branch recomputes derivatives at updated weights.
- Weighted-denominator fresh contrast and equal-mass null, zero-mass undefined case, weighted integer-target CE full/split gradient equality, probability-target denominator distinction.
- Normalization current values and buffers against real Torch, including fresh fixture, prose contrast, equal-statistics/current-output null with different running mean, constant input, one physical group and noncontiguous groups. Frozen statistics retain row/target identity.
- Exact fixed dropout masks, clipping nonlinearity, independent two-rank arithmetic, and finite-gradient sampling moments.
- Real fixed Iris20-epoch paired full/accumulated run: max final parameter difference2.914e−16, momentum8.327e−17, 80 updates,80 versus220 calls. A prescribed change to microbatch7 yields380 calls and differences4.441e−16/5.551e−17. Full epoch data and all reported outputs are retained, including validation accuracy dips. No seed/hyperparameter sweep or negative-outcome filtering occurred.

Authoring corrections from calculation/reading: corrected the first draft's post-update scalar loss to67/108 using exact fractions; replaced provisional transfer values with the actual microbatch7 execution; detached parameter differences to avoid a Torch scalar-conversion warning; kept frozen normalization outputs in original row order for noncontiguous grouping; distinguished identical current normalization outputs from repeated running-buffer updates. These are author corrections, not independent review. Reuse the final evidence for unchanged programs and calculations.

## Author learning-experience assessment and handoff

Author heuristic assessment only; no recruited beginner or independent reviewer participated. On 13 September 2026 the author reread the complete final manuscript, visual specifications, design and provenance, inspected the complete teaching programs and recorded outputs, and checked the diagnostics manuscript's opening transition. The source-bound author calculations passed after the final calculation changes. The final checklist disposition is:

1. **Route:** introduction names a concrete memory/grouping problem; first pass and deeper branches are explicit. Core does not depend on the AMP/DDP details.
2. **Cautions and voice:** core equivalence assumptions live in §3, physical-batch details in §7, experiment scope in §6, hardware limits in §8. Code prints values, not cautions. The reread confirmed these homes and retained only qualifications that explain a distinct limit or learner decision.
3. **Real question/data:** the opening32/12 grouping is carried into real offline Iris groups32/24. Predictive results are visible, and state equivalence is separately reported. Underlying data license and corrected variant are explicit.
4. **Investigations:** all three use fresh initial problems and unset predictions, entity edits, contrasts and null cases. The reread found that a mandatory numeric prediction would prevent answering the no-target-mass case; the specification now explicitly offers an initially unset `undefined: no target mass` answer. It also clarifies exact row tuples and the step-and-clear fault policy. Numerical fixtures have author calculations; prediction controls, invalidation/reset and grading are detailed specifications, not implemented evidence.
5. **Figures:** seven placements each have a different needed teaching job. Coefficient bars share a scale; normalization exposes a sign change; Iris curve includes epoch0 and a uniform baseline; tiny state gaps are direct numbers. Rendered perceptibility remains deferred.
6. **Connections:** the same−5/3 is explicitly recovered by full mean, microbatch sum and DDP arithmetic; the same optimizer state explains multi-update equivalence. Canonical agenda is audited above; diagnostics transition is coordinated.
7. **Code:** complete scalar and Iris programs supplied; displayed excerpt emphasizes the group denominator and single step. Validation/assertion machinery lives in author_checks, and CPU baseline is complete without external downloads.
8. **Practice:** changed23-row counts, a different scalar dataset, variable sequence lengths and new microbatch limit avoid reciting the worked answer. Hints precede closed reasoning. The changed25-group exercise has a precise success contract without an invented score.
9. **Screenshots:** none in phase one. Specs name required informative changed/null/prediction states, desktop/phone/zoom rendering and keyboard checks for phase two. No screenshot or accessibility pass claimed.

Inline-visual reading pass completed over §§1–8 without operating labs. Counts, state stores, coefficient redistribution, eligible-target mass, recorded learning and the normalization sign change are all introduced with an immediate diagram/table specification. The AMP strip is a control-flow explanation rather than a performance drawing. No missing introductory relationship was found in that written pass. Phase two must inspect actual values/labels at both screen widths rather than substituting successful buttons for a reading pass.

Final packet integrity check passed: 11 retained files, 22 relative links in the packet/destination note, all six learner route IDs present in the current inventory, all four calculation-input hashes current, complete displayed scalar source and both recorded outputs matching their supplied files, and no trailing whitespace in the scoped Markdown. These were read-only content checks, not an application build or ledger completion. Root receives the unchanged numerical evidence with the complete packet.

Current delivery boundary: written content/specifications and author calculations; implementation not started. Browser/visual review not performed. Formal independent content/learning-experience review not performed. Website publication not changed. User review/acceptance unknown. No application build, integration campaign, deployment or commit performed.

The content packet is **ready and frozen for root's content checkpoint**; no material author finding remains open. Next action: root checkpoints it as content complete, implementation not-started. A later authorized finish request starts with the exact topic `--work finish` preflight, consumes every required file, implements the topic-owned visuals/labs/program presentation, and completes independent review, relevant runtime/native/browser/accessibility checks and shared integration. Do not regenerate the manuscript or repeat unchanged historical evidence merely because implementation occurs in another session.
