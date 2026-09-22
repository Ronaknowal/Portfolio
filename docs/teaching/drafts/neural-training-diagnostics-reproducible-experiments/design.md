# Neural Training Diagnostics & Reproducible Experiments — content design and continuation

Stable ID `neural-training-diagnostics-reproducible-experiments`; Deep Learning Fundamentals & Architectures, section Training Mechanics & Diagnosis, module position42 of42. Author `training_diagnostics_content`; root owns shared phase ledger, scope and handoff. Revision1, content-first work authorized by the current three-topic [module completion scope](../../DEEP-LEARNING-MODULE-CONTENT-COMPLETION.md). Prepared13 September2026. Publication remains planned and implementation not started.

## Entry point, starting source and conservation

Read repository AGENTS, the active authoring handoff, exact current module-completion scope, teaching standard, domain playbook, topic-design brief, learning code standard and working-artifact retention policy before authoring. Ran the actual read-only preflight `node scripts/build-curriculum-inventory.mjs --topic neural-training-diagnostics-reproducible-experiments --work content`; read the complete returned blueprint, delivery state and notes. The destination note does not exist. The only unassigned inbox item is resolved bit-manipulation history unrelated to this topic. No destination note was rewritten or newly proposed.

The original is a **planned catalogue/blueprint entry**, not a published lesson. Actual source is `src/learn/data/curriculum/cross-domain-expansion.js` at the diagnostics `plan(...)` following training mechanics. Its full returned agenda was read: data/target alignment → deterministic tiny overfit → loss/gradient/activation scales → training/validation comparison → controlled repetition/variance. Original practice asks for three broken-run diagnoses and a minimal experiment with evidence; original misconceptions cover falling training loss and seed limitations. Original sources are PyTorch Basics and reproducibility notes. No original narrative existed to conserve; all blueprint obligations are taught below, and the live catalogue/planned source remains unchanged.

Title retained: diagnostics plus reproducible experiments accurately names both the local fault-isolation process and the complete experimental record. No rename, prerequisite insertion, navigation change or curriculum reordering is proposed. This lesson **closes the module**; readiness/review/extension choices replace an invented next in-module lesson.

## Learning contract and ownership

Intended learner: an intermediate reader who can follow a forward pass, mean objective, gradient and SGD update but has not yet learned to separate a measurement from its explanation. Prerequisites are the exact planned Mini-Batches, Training Loops & Gradient Accumulation topic and the prepared Bias–Variance Tradeoff & Learning Curves lesson. Scoped reading of the latter's §§3–4 confirms conditional seed variability versus data variability and the distinction among training trajectories, sample-size curves and complexity curves. Local refreshers explain all notation actually used.

Early coordination with the mechanics author agreed the common scalar fixture `x=[1,2,3]`, `y=[2,0,1]`, `w=0`, per-item half-squared gradients`[-2,0,-3]`, mean`−5/3`, SGD at.1→`1/6`. Mechanics owns loop/microbatch accumulation, denominators and clocks. Diagnostics uses that known computation as a reference and owns observation/intervention/inference, training failure evidence, mode diagnosis, checkpoint state and reproducibility. It supplies its own full-batch and checkpoint programs only where needed to make diagnostic measurements concrete.

Observable outcomes:

- Establish data/target, baseline and split contracts before explaining a training curve.
- Reproduce a scalar update and interpret a feasible tiny-data fit/failure without overclaiming its scope.
- Distinguish absent gradients, zero derivatives and actual parameter movement; use local scale and finite-difference evidence.
- Explain and test independent module-mode and graph-mode behavior.
- Compare evaluation curves and actual row predictions on stated measurement terms.
- Design a minimal experiment with competing predictions, retain all planned outcomes and interpret conditional variation.
- Reconstruct checkpoint continuation state and locate the first divergence from an uninterrupted run.
- Deliver a reproducible diagnosis and identify what remains untested.

### Scope decisions reconsidered during writing

| Candidate | Evidence/owner | Decision |
| --- | --- | --- |
| Correct batch/update and unequal-denominator machinery | Predecessor author owns and supplied exact shared fixture | Refresh the invariant locally; preserve predecessor's detailed ownership |
| Specific symptom→single-cause recipes | Original blueprint visual phrased symptoms as “points to” | Replace inference-by-shape with competing hypotheses and discriminating/null fixtures; preserve all intended diagnosis outcomes |
| Real memorization versus useful generalization | Actual Wine clean/shuffled runs fit supplied targets equally in final accuracy | Include measured curves, original/supplied target distinction and actual row predictions; stronger than a hypothetical overfit cartoon |
| Early stopping/tuning algorithms and complexity/sample-size learning curves | Existing cross-validation and bias–variance packets already own these mechanisms; scoped bias–variance §§3–4 read | Bridge checkpoint/selection protocol here; retain deeper statistical/tuning development at existing owners rather than reopen them |
| Optimizer derivations | Existing optimizer lessons and immediate mechanics topic own ordinary update design | Teach only scalar momentum state needed for restart, with exact recurrence |
| Evaluation-mode buffer changes | Official PyTorch behavior and four-mode actual BatchNorm probe | Include local worked mean/variance update; normalization theory remains earlier module owner |
| Distributed/data-worker mid-epoch restoration | Would require independent sharding/prefetch/streaming prerequisites | State exact boundary and missing state categories; no untested general resume implementation or unsolicited destination note |
| Broader project error analysis | Prepared end-to-end supervised lesson's question/splits inspected; it also uses Wine with a different protocol | Link as a deeper review choice, explicitly use this packet's different split and model; no borrowing of that lesson's test results |
| Video/course alternatives | Official CS231n2017 syllabus and official Lecture7 description inspected | Provide annotated course route with honest unwatched status; use read written notes for technical verification |

First-pass route is explicit immediately after the learner's concrete problem. The finite-difference and full checkpoint implementation sections are labeled deeper branches; core intuition, full scalar calculations, measured evidence, mode distinction, restart state and independent cases remain readable without executing code.

## Canonical reference agenda and coverage check

Canonical reference: Goodfellow, Bengio and Courville, [*Deep Learning*, chapter11 Practical Methodology](https://www.deeplearningbook.org/contents/guidelines.html), legitimately free HTML chapter. Inspected the **actual section agenda**, not an invented topical summary, and the debugging text/finite-difference discussion. Headings and disposition:

| Actual section | Disposition in this packet |
| --- | --- |
| 11.1 Performance Metrics | §2 defines the actual class task, accuracy reference and cross-entropy units; §§6–7 require comparable measurement and metric direction. Detailed PR/F-score decisions belong to the existing evaluation-metrics lesson. |
| 11.2 Default Baseline Models | §§1–3 establish a correct scalar reference, constant/uniform classifier references and feasible tiny neural baseline. Historical model-choice defaults are not treated as current universal recommendations. |
| 11.3 Determining Whether to Gather More Data | §6 separates poor fitting from held-out failure and §7 names unmeasured sampling variation. The original chapter's sample-size decision work is linked through the bias–variance prerequisite rather than recreated as another full data-size experiment. |
| 11.4 Selecting Hyperparameters | §7 fixes the comparison protocol/budget before results and preserves negative outcomes; specific tuning algorithms belong to cross-validation/tuning. |
| 11.4.1 Manual Hyperparameter Tuning | Include measured update/activation scales and hypothesis-driven changes; avoid numeric alarm rules or unsupported curve→rate inference. |
| 11.4.2 Automatic Hyperparameter Optimization Algorithms | Reasoned deferral to the existing tuning/AutoML owners; no automatic sweep is required to diagnose the local computation. |
| 11.4.3 Grid Search | Same owner/depth decision; broad search is outside this diagnostic lesson's goal. |
| 11.4.4 Random Search | Same decision; reading source does not turn a diagnostic intervention into a sweep. |
| 11.4.5 Model-Based Hyperparameter Optimization | Deeper tuning/AutoML ownership; excluded from the core route without changing dependencies. |
| 11.5 Debugging Strategies | Core §§1–8: inspect original/supplied targets and predictions, tiny fit, gradient/activation/update probes, independently computed finite differences, mode/buffer behavior and controlled replay. |
| 11.6 Example: Multi-Digit Number Recognition | Do not duplicate a separate image-recognition case study; the role of example/error inspection is fulfilled by the complete real Wine case with visible row IDs and wrong predictions. No claim that the Wine study reproduces the historical deployed system. |

Companion canonical practitioner notes: [CS231n Learning](https://cs231n.github.io/neural-networks-3/) actually lists Gradient checks; Sanity checks; Babysitting the learning process (loss, train/validation accuracy, weights/updates, activation/gradient distributions, visualization); Parameter updates (SGD/momentum/Nesterov, annealing, second-order, adaptive rates); Hyperparameter Optimization; Evaluation/Model Ensembles; Summary; Additional References. Read gradient/sanity/monitoring sections and the listed agenda. Monitoring and finite checks are included; detailed optimizers, ensembles and search stay with existing owners. Do **not** import that historical page's approximate1e−3 update ratio or its curve-based prescriptions as universal thresholds. This is an explicit accuracy/teaching adaptation of the source.

## Hurdles, examples, representations and assessments

| Hurdle | Mechanism/example | Representation and independent evidence |
| --- | --- | --- |
| A loss curve underdetermines a cause | Scalar derivative unchanged by omitted update; tiny neural run with nonzero gradient/zero movement | F1 and editable LabA; changed0.425 calculation and stationary null; caseA |
| Valid shapes can hide invalid labels or leakage | 178 real rows, split72/36/70, target excluded; fixed training-only scaling | F2 information-flow diagram; original/supplied/predicted row5; caseB and synthesis |
| Layer statistics have local meanings | Same seed/layer, input scale1 versus100, tanh output quantiles and derivative relation | F3 quantile strips (not invented histograms), finite-difference branch and exact scalar check |
| “Evaluation” has two controls | Fresh BatchNorm `[1,3]`, running mean0→.2 in train/no-grad; eval leaves buffers | LabB four combinations, editable input/state, linear/matching-buffer nulls; caseC |
| Memorization can optimize the wrong task | Clean and shuffled labels both72/72 training accuracy; validation36/36 versus11/15/14 out of36 | F4 measured curves, original-target switch and final-row inspection; caseB |
| Repeats support only chosen variation | Same split/permutation, three initializations; paired differences and SD | LabC selected recorded checkpoints and genuine learner score table, same-run null, changed-score closed exercise |
| Weights alone do not encode next update | Scalar momentum1.48 versus1.38; actual restore/omission traces | F5 and LabD editable recurrence, actual replay table, fresh3.25 versus2.875 and zero-momentum null; caseD |
| Reproducible result versus credible conclusion | Generator start versus current state, exact CPU replay versus seed variation | Saved-state category table, experiment-record fields, practical synthesis with explicit untested alternative |

All specific contracts, editable ranges, state invalidation, result checks, accessible/mobile layouts, numerical labels, data provenance and lazy/bounded execution are in [visual-specifications.md](visual-specifications.md). Four investigations were selected for distinct hurdles; there is no lab-count target. F1/F5 initial lab views also perform the inline explanatory job. Static split/activation/curve views remain readable without operating a lab.

## Research and claim ledger

All sources below retrieved/reviewed13 September2026. Moving stable PyTorch notes redirected without useful content through the browsing tool, so the reproducibility/autograd/BatchNorm/optimizer-loading references use explicit2.9 documentation. Relevant behavior was also verified by actual local2.14CPU probes. The tutorial pages themselves identify2.14.0+cu130; local execution uses2.14.0+cpu. These are distinct tested/read snapshots, not a version-equivalence guarantee for unrelated APIs.

| Claim/convention | Primary/canonical source and locator | What was actually checked and boundary |
| --- | --- | --- |
| Tiny isolated checks, component inspection and direct example inspection | [Deep Learning ch11](https://www.deeplearningbook.org/contents/guidelines.html), §11.5 and agenda §§11.1–11.6 | Read actual headings and substantive debugging/examples/finite-difference discussion; local examples are original exact/actual calculations, not copied figures |
| Centered finite differences, stochastic/kink/precision conditions | [CS231n Learning](https://cs231n.github.io/neural-networks-3/), Gradient Checks and Sanity Checks | Read the section and nearby monitoring text; scalar result independently derived and checked with Torch; reject blanket heuristic thresholds |
| Module eval versus graph control | [PyTorch Autograd mechanics2.9](https://docs.pytorch.org/docs/2.9/notes/autograd.html), Locally disabling gradients / Evaluation Mode | Read relevant mode table and prose; actual train/eval×grad/no-grad outputs/buffers checked in local2.14CPU |
| BatchNorm variance/buffer conventions | [BatchNorm1d2.9](https://docs.pytorch.org/docs/2.9/generated/torch.nn.BatchNorm1d.html), equation and running-statistic notes | Read population-forward versus unbiased-running variance contract; independently calculate0.2/1.1, check all modes/fresh/null inputs |
| Checkpoint needs optimizer state and correct module mode | [Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html), General Checkpoint and state-dict discussion | Read official section; actual full restore matches uninterrupted trace and parameters; omission probes preserve other named state |
| Scheduler construction before optimizer restoration | [Optimizer.load_state_dict2.9](https://docs.pytorch.org/docs/2.9/generated/torch.optim.Optimizer.load_state_dict.html), warning/call order | API page inspected; program constructs scheduler before loading optimizer and checks actual learning-rate sequence |
| Seeds, separate generators, worker reseeding and determinism limits | [Reproducibility2.9](https://docs.pytorch.org/docs/2.9/notes/randomness.html), randomness, deterministic algorithms and DataLoader sections | Read all relevant sections; no multiworker/GPU determinism claim; save actual state after constructors in replay |
| Cross-entropy logits and update sequence | [Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) and [Optimization tutorial](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html), loss/optimizer/full-loop sections | Introduction/API sections read; FashionMNIST tutorial not run. Its example held-out batch-mean aggregation is not copied for unequal batches; this packet uses full-set scores and keeps validation/test roles distinct. |
| Multiple benchmark variation sources | [Bouthillier et al.2021](https://proceedings.mlsys.org/paper_files/paper/2021/file/0184b0cd3cfb185989f858a1d9f5c1eb-Paper.pdf), introduction, §2 model, selected estimator discussion and §5 recommendations | Read these portions; do not claim a full-paper/appendix reproduction, algorithm ranking, budget reduction or statistical guarantee from three Wine seeds |
| Actual Wine question, schema, attribution/license | [UCI Wine](https://archive.ics.uci.edu/dataset/109/wine), dataset info/variables/CC BY4.0/citation | Page checked; local sklearn1.9.1 copy exported with documented transformations; whole CSV checked against bundled source |
| Course/video alternative | [Stanford CS231n2017 syllabus](https://cs231n.stanford.edu/2017/syllabus), Training Neural Networks I/II, and [official Lecture7](https://www.youtube.com/watch?v=_JB0AO7QxSA) description | Syllabus and official description inspected; no video/transcript watched, no timestamps asserted, no technical claim rests on metadata |

Learner-facing resource annotations explain fit, level, source extent and version/watch limits. Core teaching is self-contained. No downloaded paper/video is redistributed. Data licensing/transformations are retained separately in [data-provenance.md](data-provenance.md).

## Actual author calculations and substantive revisions

Before first training execution, wrote [experiment-protocol.md](experiment-protocol.md) with fixed split, architecture, optimizer, seed list and update budgets. Ran all eight Wine treatments as declared; all outcomes retained. Clean runs did not show late validation deterioration within the budget, so the lesson names that null/negative contrast rather than refitting for a prettier overfit curve. Shuffled-label runs all memorized supplied targets; reported validation counts are11,15,14 of36. The tiny omitted-update case retained nonzero gradients and zero parameter movement.

Ran [checkpoint_replay.py](checkpoint_replay.py) once for full restore and four individually omitted state categories. Full trace and final parameters are exactly equal to uninterrupted execution; each omission differs. The optimizer intervention removes momentum buffers but restores learning-rate/groups, isolating learned history. No binary checkpoint file retained; in-memory serialization exercises the tensor/plain-container contract.

Ran [calculations.py](calculations.py) for independent rational scalar fixtures, centered finite differences and actual BatchNorm mode combinations. Extended only the affected calculation checks when specifications gained fresh-input/linear/matching-statistics nulls and paired arithmetic. Their result is [calculation-results.json](calculation-results.json).

During the author reread, identified that aggregate losses and saved target labels did not let a learner inspect actual misclassified rows. Extended the Wine output contract to retain final per-row class probabilities/targets/row IDs and reran the **same** fixed experiments. [prediction-retention-check.json](prediction-retention-check.json) records the prior output hash and exact equality of every prior result field; only final-prediction fields were added. No budget/seed/architecture/treatment was selected after looking at results. The manuscript now traces training row5 and validation row120 using actual predictions, while retaining every row so those examples do not define the conclusion alone.

Also repaired an early specification's mistaken reserved-test class-count labels to23/35/12 and replaced a proposed activation “distribution” graphic with honest quantile strips because only quantiles, not histogram bins, were retained. These are author corrections before the content checkpoint, not independent-review findings.

Full manuscript/specification reread and the final learning-experience checklist are recorded below after the bounded artifact checks. Shared checkpoint hashes are owned by root; no shared registry/ledger mutation is performed by this author.

## Readiness, checklist and remaining phase

**Content complete and frozen for root's checkpoint, 13 September 2026.** Read the complete final manuscript and specifications in ordinary order, including all closed solutions and every figure/lab contract. Read the predecessor's current §§1–3 and its author confirmed the shared transition/fixture agrees. Bounded author checks passed in [author-verification.json](author-verification.json): offline source/partition equivalence, independent probability-to-loss/accuracy reconstruction, every planned run and key null, mode formulas versus actual Torch, checkpoint first-divergence evidence, exact extracted displayed-program execution, source parsing and local links. The final changed CaseA arithmetic was checked with exact fractions. Scoped `git diff --check` passed; source-level checks are separate from browser evidence.

The reading pass also corrected “unit conversion” to “scale mismatch after standardization,” since a consistent raw-unit conversion with a correspondingly fitted scaler need not create that intervention. It changed independent CaseA to new rows/rate with answer0.725 rather than repeating the demonstrated0.425 example, tightened the checkpoint-selection wording to match the specified control, and removed author-facing implementation remarks from the learner's closing paragraphs. Checks affected by new arithmetic/text hashes were rerun; unchanged neural measurements and checkpoint evidence were reused.

Author learning-experience checklist, performed separately from numerical checks:

| Item | Author finding and disposition |
| --- | --- |
| First-pass route | Explicit after concrete problem; finite-difference/full-replay detail marked deeper; reader reaches module readiness without hidden program execution. |
| Once-stated cautions | Full hedging pass kept source/data scope in§2, symptom scale conditions in§1/4, comparison scope in§7 and reproducibility limits in§9. Removed repeated author-facing overfit/next-topic caveats. Code prints only values/results. |
| Real question and result | Actual Wine cultivar task, label provenance, final per-row predictions and all predeclared results make the result inspectable. No synthetic curve passed off as real data. |
| Genuine investigations | LabA edits rows/weights; B edits inputs/stored statistics and mode switches; C selects actual pairs/checkpoints or enters fresh scores; D edits recurrence/save state. Each displays current results and recomputes them from the active inputs. Default, changed and null fixtures are supported by exact or actual-run evidence; no browser execution is claimed. |
| Figure perceptibility | Static contracts specify meaningful axes, common log-loss range, baseline points, endpoint-safe quantile labels and phone stacking. Actual desktop/phone rendering remains explicitly deferred to phase two. |
| Connections/canonical facts | Scalar gradient and update agree with the predecessor; finite-difference/autograd routes are connected; canonical chapter agenda is enumerated with deliberate owner/depth dispositions; final module has review choices. |
| Displayed code | Complete finite-difference program is directly executed from Markdown; full explained Wine/replay sources are supplied with all inputs. Excerpt shows central update without instrumentation obscuring it; verification code is separate. |
| Changed practice | Independent CaseA uses new signed gradient/rate and exact0.725 result; CaseD changes target/momentum/rate and reproduces3.25/2.875; label/mode diagnoses and open measured synthesis test transfer. Hints/solutions begin closed. |
| Screenshots | Not applicable to the authorized content phase: no figure was rendered, no browser screenshot or accessibility pass is claimed. Informative states to capture are specified for phase two. |

No material content finding remains open. Implementation, formal independent review, browser/visual verification, publication and user acceptance remain unperformed.

The retained packet has complete explanations, worked calculations, changed practice with initially closed hints/solutions, annotated alternate resources, fixed protocol, offline data and full programs/results. Pending content is authoritative handoff material and must be retained. No scratch/history scan, shared environment installation, runtime/app build, visual implementation or publication occurred; bytecode was disabled for local Python work.

**Next action:** after root records a complete current content checkpoint, a later authorized finish request runs this topic's `--work finish` preflight, reads the entire manuscript/specifications/design and consumes the packet. Implement its semantic topic-owned models/labs/figures and publication, execute displayed programs as necessary for the implemented bytes, conduct formal independent correctness/learning-experience and actual browser/accessibility/figure checks, resolve findings and complete relevant shared integration. Do not start a new module from this final topic's completion.

## Live exploration revision — 21 September 2026

The user replaced prediction-and-reveal lab assessment with immediately visible, playable exploration, including removal of optional prediction controls. The manuscript and specifications now use that contract. This changes the teaching interaction, not the topic's model predictions or its mathematical masks/gates.

**Topic disposition:** Design a useful diagnostic and reproduce an operation. Edit tiny training rows/step settings, train/eval and graph modes, experimental evidence choices and restored checkpoint fields. Show gradient versus parameter movement, statistic buffers, comparable measured outcomes and the exact next operation under restored/missing state. Choose a check that distinguishes a real failure from a null example and preserve the state required for a meaningful replay.

Retain all source data, formulas, measured results, code programs, references and independent practice. The existing author calculations remain evidence for those unchanged quantities, not evidence that a browser implementation already satisfies this new interaction contract. Phase two must implement and verify live updates, linked-view agreement, bounded work, reset, invalid/null cases, keyboard access and responsive diagrams. Content remains prepared; implementation remains not started.

## Implementation ownership and content-depth revision — 22 September 2026

Delivery remains **content-first**. The complete computational teaching route is part of this prepared packet now; phase two receives written code, explanations, mapped state/settings and closed practice, rather than an instruction to invent the missing mechanism. Earlier authoring records remain dated evidence; this section supersedes their incomplete depth handoffs. The title and stable ID are retained because the new material fulfills the existing scope.

| Advertised computational outcome | Scratch/source owner | Ordinary tool route | Matching bridge | Independent practice | Scope boundary |
| --- | --- | --- | --- | --- | --- |
| Fault localization, exact update and finite-difference diagnosis | calculations.py::scalar_step/finite_difference; wine_diagnostics.py::fit/activation_probe | nn.Module/SGD and direct tensor measurements | same objective/rows and separate loss/gradient/parameter-change signals | Cases A–C; practical diagnosis synthesis | Derivative/loss/optimizer engines reused from actual implemented owners |
| Checkpoint process reconstruction and controlled omissions | checkpoint_replay.py::snapshot/restore/train_until | state_dict, torch.save/load(weights_only=True), generator states | uninterrupted versus restored trace and parameter comparisons | Case D; complete mid-group checkpoint solution | Current runnable checkpoint boundary is completed update; distributed/prefetch not certified |
| Experimental protocol and evidence interpretation | wine_diagnostics.py::load_inputs/main; experiment-protocol.md | NumPy/PyTorch ordinary local experiment workflow | declared roles/seeds/treatments and retained row predictions | Practical synthesis; uncertainty discussion | No invented package for scientific inference; different sampling sources remain distinct |

All local source owners above were inspected at their actual function/class definitions. Full model fitting, data/provenance and existing worked results are retained. Reused actual prerequisite code is named explicitly in the manuscript; prepared owners are not described as already published updated instruction. Whole-family releases mentioned for context do not expand the promised executable outcome into every checkpoint or every GPU kernel.

The teaching sequence is construct → explain the state/update → normal tool use → compare the same contract → changed-constraint practice, inserted where the relevant mechanism is explained. Original mechanism programs remain canonical; new programs depend on them only where the import is explicit. No browser program, published lesson, manifest or curriculum sequence is changed by this revision.

Author checks for this revision: source/API-contract reading, Python syntax parsing, matching embedded/downloadable source and local links, and scoped arithmetic probes where recorded in the specialist-writing report. These are content-authoring checks. Earlier fit outputs remain their original evidence; new multi-process/GPU/specialist-package execution, formal independent implementation review, rendered diagrams/labs and browser/accessibility/integration checks are **deferred**, with exact targets in the current visual specifications and specialist report.

Next action: after the root records the new content checkpoint, consume the full current packet for an authorized finish request, execute the relevant new programs and capture honest outputs, build the specified topic-owned views, independently check the translated models and integrate them. No core scratch/library manuscript writing is left as a finish-only TODO.
