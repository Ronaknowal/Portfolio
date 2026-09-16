# Educational authoring: current handoff

Updated 16 September 2026. This is the active entry point. The user's current request defines scope and delivery mode; historical reports and scratch files do not define a new queue.

The [sequential implementation audit through Bayesian Networks](docs/teaching/CLASSICAL-ML-IMPLEMENTATION-AUDIT.md) is complete. All twelve topics at Classical ML positions 17–28 were closed individually; the earlier PCA–GMM review was reused. The final integration record confirms current source/checkpoints, unchanged prepared packets and publication membership, and changes confined to the twelve scoped central-ledger rows. It includes the Feature Selection gradient repair and further correctness, completeness, interaction and diagram fixes. Use its per-topic evidence when resuming related work; it is not an unfinished queue or authorization to implement MCMC.

**Quality comes first.** Follow the standard's [quality-first principle](LESSON-TEACHING-STANDARD.md#quality-takes-priority-over-efficiency). Token/time optimizations are flexible defaults: agents should do additional research, explanation, visual work, revision or verification whenever their judgment identifies a worthwhile quality improvement within the authorized scope. No efficiency target overrides correctness, completeness or the learning experience.

## Start here

1. Read [the teaching standard](LESSON-TEACHING-STANDARD.md), starting with [delivery modes](LESSON-TEACHING-STANDARD.md#delivery-modes-and-stopping-boundaries), the [six-stage workflow](LESSON-TEACHING-STANDARD.md#six-stage-authoring-workflow) and **Keep verification bounded and reusable**. It owns phase boundaries, pedagogy, quality and completion requirements.
2. Use [the domain playbook](docs/teaching/DOMAIN-PLAYBOOK.md) and [topic design workflow](docs/teaching/TOPIC-DESIGN-BRIEF.md) for the current subject. Choose representations for its mechanisms; do not impose identical labs or article sections.
3. Read [the learning code standard](docs/engineering/LEARNING-CODE-STANDARD.md), including temporary-work retention. It owns semantic files, lazy loading and performance contracts.
4. Read the [two-phase ledger](docs/teaching/lesson-delivery-progress.json) through `node scripts/build-curriculum-inventory.mjs --topic "Exact title or stable ID"`. Check `topic.delivery`, its linked record and destination notes. Read the complete current manuscript/specifications or implemented lesson appropriate to the requested phase. [Ledger instructions](docs/teaching/LESSON-DELIVERY-LEDGER.md) define checkpoint/version handling.
5. Use [the curriculum plan](LEARNING-CURRICULUM-PLAN.md) and relevant specialist plans only when a scoped coverage question needs them. For DSA, also use [the practice standard](docs/teaching/DSA-PRACTICE-STANDARD.md).

Do not reread all historical batches or enumerate scratch to rediscover the task. Preserve passed evidence for unchanged source and continue from the recorded next action. Within an ongoing session, reuse current policies already available in context; retrieve missing or changed instructions when necessary. The standard's [source-bound review handoff](LESSON-TEACHING-STANDARD.md#source-bound-evidence-and-review-handoff) and code standard's [context/coordination rules](docs/engineering/LEARNING-CODE-STANDARD.md#efficient-context-tools-and-coordination) explain what to pass between authors, reviewers and the integration owner without repeating the whole history.

## Request either delivery mode

- **Full:** “Implement the next topic end to end.” Complete content, visual/lab implementation, all applicable review/fixes/checks and integration without an intermediate approval pause. This is the default for ordinary implementation requests.
- **Content first:** “Research and write the next topic only, including detailed visual/lab specifications.” Deliver the complete manuscript, examples, practice/solutions and annotated resources plus actionable specifications. Implement the visuals/labs and website code in phase two, not during this request.
- **Later continuation:** “Finish implementation and verification for the prepared topic.” Run the topic command with `--work finish`; it requires a complete, current content checkpoint. Consume that work and perform phase two, including necessary content corrections. A different agent can continue using the saved files.

The user's selected mode persists through the requested batch and clear continuations. A content-only completion must say that implementation remains pending. Publication, historical implementation review and user acceptance are separate from the current revision's two statuses. Both phases of the 107 previously completed revisions were migrated as complete; source changes can make their current effective status stale. No lessons were rewritten by this workflow migration, and it authorizes no next topic.

## Latest completed scope — hidden Markov models and Bayesian networks, 16 September 2026

The request was to continue the sequence. Both are implemented, independently reviewed and integrated: **Hidden Markov Models (27)** and **Bayesian Networks & Causal Graphical Models (28)**, the first two topics of the Probabilistic & Graphical Models section.

| Position | Topic | Record | Independent review |
| --- | --- | --- | --- |
| 27 | Hidden Markov Models (HMM) | [design](docs/teaching/drafts/hidden-markov-models-hmm/design.md) | [review](docs/teaching/HMM-INDEPENDENT-REVIEW.md) |
| 28 | Bayesian Networks & Causal Graphical Models | [design](docs/teaching/drafts/bayesian-networks-causal-graphical-models/design.md) | [review](docs/teaching/BAYESIAN-NETWORKS-INDEPENDENT-REVIEW.md) |

**Six things this scope established.**

**Both optional-library programs were executed for real, in isolated environments, and the isolation was not theoretical.** `hmmlearn` resolved NumPy 2.5.3 and `pgmpy` resolved NumPy 2.5.3 with pandas 3.0.5 — genuinely off the shared runtime's 2.3.5/3.0.1. Installing either into `scratch/lesson-tools` would have moved the floor under every completed lesson's recorded output. The pattern to reuse: an isolated venv under `scratch/`, the resolved versions recorded, and — as the Bayesian Networks lesson did — the generated module recording the environment each program actually ran in, so the prose names it from a record a verifier can check rather than from memory. This supersedes the AutoML precedent of declining to execute; decline only if isolation genuinely fails.

**Check by a different theorem, not a different call.** Bayesian Networks verifies d-separation by ancestral moralisation, which never enumerates a path; its reviewer then decided 357,634 cases three ways — path enumeration, Bayes-ball reachability and moralisation — with zero disagreements. HMM's reviewer rebuilt inference in exact rationals with neither scaling nor logarithms, independent of both the lesson's retained scaling and the packet's log space. A second call to the same idea is not a second route.

**The DOM can be right while the paint is wrong, and every assertion still passes.** Two builders hit this independently: `.hmm-edge { stroke-width: 1 }` outranking a presentation attribute so twelve trellis edges painted at 1 px while carrying correct widths, and a `fill="none"` attribute losing to a CSS `fill` so a figure's bands were entirely invisible. Compare attributes against **painted** values — and treat "no attribute" as a case to examine, not one to skip, because that exclusion hid a zero-mass edge painted wider than a near-zero one.

**A guard's domain is where defects hide.** Every inert guard this scope found was correct within a domain that excluded the defect: a leak check matching values surrounded by spaces when the DOM renders `alcohol13.17training`; a clearance check sampling `route.trace` while the browser paints `route.path`, up to 9 px apart, so a live edge crossed a badge unnoticed; ~750 assertions inside advertised grids comparing a function with itself. **Falsification is the only proof.** One builder's breakage *survived* its browser run because no edge carries zero mass as first painted — which is precisely why the original defect was never caught — so it drove the lab into a reachable zero-mass state and asserted there.

**A record must not describe a check the tree does not contain.** One lesson reported an escape auditor as a passing row in two phases; it existed only in the builder's scratchpad, so it described a check nobody could run, and it misled the integration owner who repeated the claim. Resolution: commit the tool, and correct the superseded rows in place with an explicit note rather than deleting them.

**Grading leaks are a property, not a list of sites.** For the third time, a builder fixed an instance of "the page shows or grades the answer before commitment" and its reviewer found the same class elsewhere — three of four investigations in one lesson, one printing its verdict on first paint above a box promising the answer appears only after a prediction. The check that works asserts the graded **value** is absent from the investigation's rendered text before commitment, for every investigation, and pins values numerically as well as by marker class; asserting the absence of a verdict *banner* is what let it pass.

Two further notes. HMM found, in exact rational arithmetic, that three of the forty development sentences have two complete paths of exactly equal probability, making the packet's recorded counts one member of a band (269/270/272 and 267/268/270) flippable by 1 ulp. Its reviewer confirmed all three parts and strengthened one: the tie is **structural** — the paths share identical multisets of factors — so no parameter perturbation breaks it, and the conclusion survives every consistent tie rule. The packet is correct, the margin is fragile, the conclusion is robust, and the lesson teaches all three. Separately, the open destination note on Bayesian Networks is **closed**: its backdoor finding was verified independently by path enumeration rather than accepted, and the closure records a qualification against the note's own second concern.

Nothing is committed. User acceptance is separate.

## Previous scope — imbalanced learning and AutoML, 16 September 2026

The request was to continue the sequence. Both are implemented, independently reviewed and integrated: **Imbalanced Learning (25)** and **AutoML & Neural Architecture Search (26)**.

| Position | Topic | Record | Independent review |
| --- | --- | --- | --- |
| 25 | Imbalanced Learning (SMOTE, Cost-Sensitive Learning) | [design](docs/teaching/drafts/imbalanced-learning-smote-cost-sensitive-learning/design.md) | [review](docs/teaching/IMBALANCED-LEARNING-INDEPENDENT-REVIEW.md) |
| 26 | AutoML & Neural Architecture Search (NAS) | [design](docs/teaching/drafts/automl-neural-architecture-search-nas/design.md) | [review](docs/teaching/AUTOML-NAS-INDEPENDENT-REVIEW.md) |

Position 25 carries the packet's expanded display title, *Imbalanced Learning: SMOTE, Cost-Sensitive Learning & Rare-Event Decisions*, from its lesson module. The catalogue title in `track-definitions.js` is unchanged, because the stable ID is slugified from it and `authoredBlueprints` is keyed on it. That is the compatible route for a display rename: change the module's title, never the catalogue's. Position 26 asked about the same rename and was told not to, because its manuscript H1 is close enough to the catalogue form that the move buys nothing.

**Five things this scope established.**

**The dominant defect class is now named: the page grades a correct answer wrong, or shows the answer first.** Six of the seven blocking findings across these two lessons were that. A preset labelled "Exact null: double both costs" graded the correct null answer wrong, because it graded the total cost rather than the selected gate the null exists to demonstrate. A step size of zero — typeable, because the control's own minimum is zero — was marked wrong while the verdict printed identical before-and-after values. One investigation shipped candidate A over an identical E under a tiebreak rule the page never stated. Ask of every graded comparison: at its degenerate inputs — zero step, exact ties, identical rows — is the answer the page's own displayed evidence supports the answer it marks correct?

**Fix the class, not the case.** Both lessons fixed a grading defect in phase C and had the *same* defect found one stage over by their reviewer. What finally worked was asserting the property rather than the example: an 80-case sweep asserting the verdict says "stay" **iff** the printed output did not move, whatever the reason; a guard that asks whether the answer string is on screen rather than enumerating the containers that might hold it.

**A guard can be correct and still never fire.** This scope found guards that were inert for four distinct reasons: a regex whose `` had been turned into a literal backspace byte by a heredoc, inside the verifier written to prevent the very defect that then survived; a cap-boundary check using `<=` so it could not fail for its own regression; a size floor measuring SVG user units rather than rendered pixels, reporting an impossible identical figure at two widths and passing on an empty set; and a guard that ran after a figure toggle, so its subject was off-screen when it looked. **Writing a guard is not evidence. Breaking the code and watching it go red is.** One builder applied sixteen deliberate breakages one at a time, restoring each byte-for-byte; all sixteen fired.

**Count what your checks cover, and say it honestly.** A hand-maintained "82 trust-root values" counter turned out to be 78 of 149 real leaves; another lesson's coverage was 47.9% of 18,239, with 9,000 of the uncovered leaves in a block nothing consumes. Both are now asserted leaf-path coverage rather than counters — one at 100%, the other stated plainly as 107 re-derived plus 42 pinned against the manuscript, which is the honest description of what re-derivation can and cannot reach.

**Screenshot evidence should record digests, not just paths.** One lesson's evidence records `{file, digest, bytes}` per capture, which let a stale orphan from a pre-fix run be found and removed; the other records paths only. Prefer the former.

Nothing is committed. User acceptance is separate.

## Previous scope — feature selection and bias-variance, 15 September 2026

The request was to implement the next two unimplemented topics in module order. Both are implemented, independently reviewed and integrated: **Feature Selection & Importance (23)** and **Bias-Variance Tradeoff & Learning Curves (24)**.

| Position | Topic | Record | Independent review |
| --- | --- | --- | --- |
| 23 | Feature Selection & Importance (SHAP, Permutation, Mutual Info) | [design](docs/teaching/drafts/feature-selection-importance-shap-permutation-mutual-info/design.md) | [review](docs/teaching/FEATURE-SELECTION-INDEPENDENT-REVIEW.md) |
| 24 | Bias-Variance Tradeoff & Learning Curves | [design](docs/teaching/drafts/bias-variance-tradeoff-learning-curves/design.md) | [review](docs/teaching/BIAS-VARIANCE-INDEPENDENT-REVIEW.md) |

Unlike the previous scope, **both topics were already published with historical bodies**, so each packet replaced a live lesson rather than filling an empty slot. The builders preserved the originals and checked them against `HEAD` before overwriting. Registration was also sequenced differently: the builders stopped after their three offline verifiers, the integration owner registered the blueprints, and only then did browser evidence get captured — which removes the sequencing artefact the ICA review raised last scope, where registration postdated the evidence.

**Four things this scope established.**

The pattern held again, and harder: **two reviews recomputing 22,098 and 3,194 values from first principles found no numerical disagreement at all**, while between them raising four blocking findings — every one in a figure, a caption, a record or a verifier. Fourteen scopes of evidence now say the same thing: the mathematics survives, the presentation does not.

A trust root that nothing re-derives is not verified. `calculated-inputs.json` fed all 19,764 comparisons in the bias-variance model verifier but was itself checked by nothing in the repo. Its reviewer re-derived all 21,993 values independently, and the data verifier now does so on every run. Ask of any new lesson: what checks the thing the checks are checked against?

**A defect class can look closed while the repair only moved it.** Feature selection caught a drawn-rule/applied-rule mismatch in its own screenshots and fixed it by displaying two decimals; the review then found the same defect still reachable at another node, where a learner typing the obvious round number 1.59 reasons correctly from the drawn rule and is graded wrong. The real fix was to make the drawn threshold the largest value on the control's own grid that the tree still sends left — exploiting the monotonicity of `Math.fround` so the two rules cannot disagree at any enterable value — and to assert that over all 21,192 of them rather than at samples.

A verifier that asserts nothing passes. Feature selection's five-width geometry sweep never inspected a single investigation diagram, because every block reset first and nothing asserted the subject set was non-empty; five further assertions could not fail under any input. Both are now guarded, and a sweep that collapses fails loudly.

**Recommendation for the next scope, deliberately not taken here.** The bias-variance builder wrote a curve-geometry check that samples every path, polyline and polygon along its own geometry against every label box, closing a blind spot in `scripts/lib/lesson-visual-layout.cjs`, which inspects only straight lines. It is written to be lifted as-is (`sampleCurvesThroughLabels` in `verify-bias-variance-browser.cjs`). Promoting it into the shared inspector would apply a stricter geometric standard to every closed lesson and would likely reopen completed work, so it was left in place rather than promoted mid-scope. That is a scope decision for the user, not a technical obstacle.

Nothing is committed. User acceptance is separate.

## Previous scope — the next five prepared packets, 14 September 2026

The request was to implement the next five unimplemented topics in module order, using subagents. All five are implemented, independently reviewed and integrated: **Independent Component Analysis (18)**, **Non-Negative Matrix Factorization (19)**, **Feature Scaling, Encoding & Imputation (20)**, **Cross-Validation & Hyperparameter Tuning (21)** and **Regularization (22)**. Each topic's own design record carries its phase-two section and the disposition of its review; the central ledger carries the final source hashes.

| Position | Topic | Record | Independent review |
| --- | --- | --- | --- |
| 18 | Independent Component Analysis (ICA) | [design](docs/teaching/ICA-LESSON-DESIGN.md) | [review](docs/teaching/ICA-INDEPENDENT-REVIEW.md) |
| 19 | Non-Negative Matrix Factorization (NMF) | [design](docs/teaching/drafts/non-negative-matrix-factorization-nmf/design.md) | [review](docs/teaching/NMF-INDEPENDENT-REVIEW.md) |
| 20 | Feature Scaling, Encoding & Imputation | [design](docs/teaching/drafts/feature-scaling-encoding-imputation/design.md) | [review](docs/teaching/FEATURE-SCALING-INDEPENDENT-REVIEW.md) |
| 21 | Cross-Validation & Hyperparameter Tuning | [design](docs/teaching/drafts/cross-validation-hyperparameter-tuning/design.md) | [review](docs/teaching/CROSS-VALIDATION-INDEPENDENT-REVIEW.md) |
| 22 | Regularization (L1, L2, Elastic Net, Dropout) | [design](docs/teaching/drafts/regularization-l1-l2-elastic-net-dropout/design.md) | [review](docs/teaching/REGULARIZATION-INDEPENDENT-REVIEW.md) |

Each lesson owns its models, generated data, executed programs, labs, figures, stylesheet, blueprint, four verifiers and evidence, and serves its own copy of its dataset under `public/learn-assets/`. Two topics use the same penguin file and two use the same digit file; each serves its own bytes rather than sharing, so no lesson can break another by editing an asset.

**Five things this scope established, worth carrying to the next one.**

Every reviewer found numbers that hold and presentation that does not. Across the five, independent recomputation from first principles — refitting nested experiments by hand, rebuilding preparations without the library, checking 602 transformed values and 4,000 envelope values at zero difference — produced **no numerical disagreement at all**. Every blocking finding was in a figure, a caption, an investigation's behaviour or a record.

Screenshots are not optional and assertions are not enough. Fifty defects across the five lessons survived a fully green verifier run and surfaced only by opening the images: captions clipped mid-word, a CSS `font` shorthand silently overriding every `fontSize` attribute in every figure of one lesson, a flow diagram that drew nothing, a magnified panel clamping out-of-range points onto its axis, a figure drawing "red" as a blue dot.

Two defect classes recurred often enough to be checked by default. First, a record asserting an implementation state that contradicts the tree; four of the five reviews raised it. Second, an investigation showing or grading something before the learner commits — and the lesson that avoided it entirely, cross-validation, did so because its labs open with nothing computed at all.

A pointer into a lab is a claim that must be true. Two lessons shipped a practice task telling the learner to enter values the lab's own range checks refuse.

Resuming the implementing agent to fix its reviewer's findings works better than fixing them centrally. Each builder still had the context to find root causes rather than symptoms: a no-wrap rule behind two clipped cells, a font shorthand behind an inert style, an unreachable branch behind a missing comparison. Two of them also declined findings with sound reasons, including a refusal to edit a frozen manuscript for a defect the implementation had introduced.

Nothing is committed. User acceptance is separate.

## Previous scope — finish prepared manifold-learning content

The 14 September request to implement the next complete research/write packet is complete for **t-SNE, UMAP & Manifold Learning**, Classical ML position 17 of 39. Both phases of revision 1 are complete. Use [the implementation/design record](docs/teaching/MANIFOLD-LEARNING-LESSON-DESIGN.md#prepared-content-implementation--14-september-2026), its source-bound evidence and the central phase ledger. The implementation preserves all 13 written sections and adds nine inline figures, four investigations, four verified Python programs, ten actual saved digit maps and seven practice tasks. Independent technical/learning reviews, actual visual inspection, responsive checks and production integration are recorded there. Do not rerun completed checks for unchanged source.

**Next eligible prepared topic is Independent Component Analysis (ICA), position 18.** Its research/write phase is complete and its implementation remains not started. Continue it only when requested, using `--work finish` and its current packet; do not infer permission to implement another topic from this completion. All other pending packets and the separate K-Means proposal are preserved.

## Previous completed scope — diagram distortion and layout follow-up

The user's GMM allocation screenshot exposed a collision missed by the earlier selected-image review. The [visual-layout follow-up](docs/teaching/LESSON-VISUAL-LAYOUT-REVIEW.md) records the stage-layout repair, analogous published-lesson fixes, shared-plot behavior, actual browser evidence and the limits of the broad geometry triage. Its final reconciliation is the continuation point for this repair; it does not reopen prepared content or authorize another topic. For future implementation, use the standard's [separate visual-layout completion check](LESSON-TEACHING-STANDARD.md#visual-layout-is-a-separate-completion-check) and the code standard's diagram-layout contract. A passing overflow assertion is not evidence that labels inside a figure are legible.

## Previous scope — PCA through GMM implementation verified

On 14 September 2026 the user requested verification that prepared content had been fully and correctly implemented for Classical ML positions 12–16: PCA, Clustering Evaluation, DBSCAN, Anomaly Detection and GMM/EM. The [five-topic audit](docs/teaching/PCA-THROUGH-GMM-IMPLEMENTATION-AUDIT.md) records full content/specification comparisons, independent mathematical and learning-experience reviews, concrete repairs and 64 passing browser cases. The prepared manuscripts/specifications are retained unchanged; current implementation checkpoints include the reviewed repairs. Historical completion dates and user acceptance remain distinct.

The scoped verification is complete. Do not rerun every historical suite or start another topic on this record alone. After GMM the actual module sequence continues with t-SNE, UMAP & Manifold Learning (17), then ICA (18); both have prepared content awaiting a separately authorized finish request. The other pending packets and separate K-Means proposal are preserved. Current global counts come from the generated inventory; dated counts in earlier completion sections below remain historical snapshots.

## Previous scope — remaining Deep Learning module content

The remaining research-and-writing request is **complete**. The [completion record](docs/teaching/DEEP-LEARNING-MODULE-CONTENT-COMPLETION.md) covers positions **40–42**: Titans (Multi-Memory Architecture); Mini-Batches, Training Loops & Gradient Accumulation; and Neural Training Diagnostics & Reproducible Experiments. All **42 module topics now have complete prepared content**, with implementation **not started** for these revisions. This completes the module's content phase, without authorizing another module or website implementation.

Three authors worked in disjoint topic directories; root reconciled the sequence, source conservation and handoff contracts. Each packet contains a full manuscript, worked examples and changed practice, primary/canonical research with annotated alternatives, necessary offline author evidence and precise topic-specific visual/lab specifications. Author rereads and bounded calculation checks are recorded in each design file. The central ledger binds all **41 new packet files** to content-complete checkpoints; all three finish preflights pass. The preceding 174 ledger entries and 646 files in 64 earlier prepared packets are unchanged. Ledger validation passed all eight behavior groups; regenerated inventory records **176 current content-complete checkpoints and 109 implementation-complete checkpoints**, with all catalogue/publication counts preserved.

For an authorized finish request, run the selected topic's `--work finish` preflight and consume its full saved packet. Implement visuals/labs and complete formal independent review, relevant fixes/checks and integration then. This content completion does not claim rendered verification, publication or user acceptance. A new OPEN [DDP destination note](docs/teaching/topic-notes/data-parallelism-ddp.md) records how unequal eligible loss mass affects distributed gradient averaging; the destination lesson was not changed. Preserve all earlier packets and the existing separate K-Means proposal. The completion record retains exact evidence and continuation.

## Previous scope — 30 deep-learning architecture content packets complete

The user requested “Now similarly do next 30,” continuing research-and-writing-only mode with the existing parallel workflow. The exact [deep-learning architecture scope and source baseline](docs/teaching/DEEP-LEARNING-ARCHITECTURES-CONTENT.md) covers module positions **10–39**, from **Landmark Architectures** through **Hybrid SSM–Transformer Architectures (Jamba)**. All 30 belong to Deep Learning Fundamentals & Architectures. Titans and the two subsequent training-mechanics topics were outside that request and are now complete under the latest scope above.

All **30 content packets are complete**, and all **30 implementations remain not started**. Four authors, including root, worked in disjoint topic directories; root reconciled sequence, scope, original-source conservation and consequential content/visual/data contracts. Every packet contains a full manuscript, worked examples and changed practice with closed solutions, primary/canonical research and annotated alternatives, necessary offline data/calculations, a design/continuation record and precise topic-specific visual/lab specifications. Actual author rereads, focused corrections, calculations and unexecuted optional-code limitations are recorded per topic. This is phase-one completion, not formal independent review, rendered/browser verification, publication or user acceptance.

The [central ledger](docs/teaching/lesson-delivery-progress.json) binds the complete packets to their file hashes and owns both phase states. Under a later authorized finish request, run the selected topic's `--work finish` preflight and consume its full saved packet; implement the lesson/visuals/labs and complete the independent review, relevant fixes/checks and integration then. Preserve the previous thirty packets and earlier pending work; reuse unchanged source-bound evidence instead of restarting them.

At the close of that thirty-topic request, the next unprepared entry was **Titans (Multi-Memory Architecture)**, stable ID `titans-multi-memory-architecture`, position 40. It and the two following training-mechanics topics have since been prepared under the latest scope above; this historical boundary is not an active queue. New discoveries were saved as OPEN notes for [shared cross-attention memory](docs/teaching/topic-notes/interleaved-cross-attention-architectures.md), [Muon versus curvature-based optimization](docs/teaching/topic-notes/second-order-methods-l-bfgs-k-fac-shampoo-natural-gradient.md), and [DINOv3's fuller self-distillation treatment](docs/teaching/topic-notes/self-distillation-byol-dino-dinov2.md). Future destination authors must reason on their evidence and ownership; these notes do not authorize an extra rollout. The scope record retains final validation and exact topic links.

**Final checkpoint:** all thirty finish preflights pass, with 367 packet files hash-bound. The original 29 publications and one planned entry remain unchanged, as do all 241 files in the previous thirty-packet scope. Ledger validation passed; inventory now has 173 current content-complete checkpoints and 109 implementation-complete checkpoints, with all 1,218 topics, 28 modules, 228 publications and seven paths preserved. This is documentation/content work; no application build, browser campaign or publication was performed.

## Previous scope — 30 content packets complete; implementation deferred

The user authorized **research and writing for the next 30 topics**, continuing content-only mode. The exact [scope and source baseline](docs/teaching/CLASSICAL-ML-DEEP-LEARNING-CONTENT.md) covers Classical ML positions 19–39, from NMF through End-to-End Supervised Learning & Error Analysis, then Deep Learning Fundamentals & Architectures positions 1–9, from Perceptrons through Convolution, Pooling & Receptive Fields. Follow actual module order; this request crosses the module boundary but does not authorize changing it.

All **30 content packets are complete**, with implementation **not started**. Four authors, including root, worked on disjoint topics; root reconciled sequence, scope and consequential content/visual/data contracts. Each stable-ID directory contains a complete manuscript, visual/lab specifications, design/source record and required offline data/calculation inputs. Author rereads and bounded calculation checks are recorded per topic. This is content completion, not formal independent review, rendered verification, publication or user acceptance.

For a later authorized implementation, run `node scripts/build-curriculum-inventory.mjs --topic <stable-id> --work finish` and consume the full current packet returned by the ledger. Implement the specified figures and investigations, execute or correct the explicitly unexecuted optional programs, and complete the required independent, browser/accessibility and integration work. Reuse source-bound evidence where it remains valid. Do not regenerate these manuscripts from scratch or reopen unrelated completed batches.

At the end of that **previous** request, the next unprepared topic was Landmark Architectures (LeNet → AlexNet → VGG → ResNet → EfficientNet), stable ID `landmark-architectures-lenet-alexnet-vgg-resnet-efficientnet`. It has since been prepared in the latest scope above; this historical boundary is not a current queue. That earlier request routed discoveries to the actual [dedicated NAS](docs/teaching/topic-notes/neural-architecture-search-nas.md), [AutoML as Meta-Learning](docs/teaching/topic-notes/automl-as-meta-learning.md) and [classical time-series](docs/teaching/topic-notes/arima-garch-classical-time-series.md) owners; those notes do not authorize their implementation. Preserve all earlier completed and pending work below. Its scope record retains final checkpoint evidence and exact topic links; the central ledger remains the only current phase queue.

## Previous scope — GMM, manifold learning and ICA content complete

The user continued the research-and-write-only mode for the next three topics, with parallel authors. Follow the actual Classical ML module order:

| Position | Topic | Content record |
| --- | --- | --- |
| 16 | Gaussian Mixture Models (GMM) & EM Algorithm | [Design and continuation](docs/teaching/GMM-LESSON-DESIGN.md) |
| 17 | t-SNE, UMAP & Manifold Learning | [Design and continuation](docs/teaching/MANIFOLD-LEARNING-LESSON-DESIGN.md) |
| 18 | Independent Component Analysis (ICA) | [Design and continuation](docs/teaching/ICA-LESSON-DESIGN.md) |

All three content packets are complete; implementation is **not started**. Three authors worked in parallel on topic-owned manuscripts, visual/lab specifications, research and necessary offline inputs. The root author read all three complete manuscripts, specifications, provenance and design records, reconciled the sequence, and returned focused findings for correction. This content reconciliation is distinct from the formal independent review required during phase two. Preserve existing published sources and the completed/pending work below. That three-topic request did not authorize implementation; NMF research/writing is now authorized by the subsequent scope above.

The sequence connects **Anomaly Detection → probability mixtures and EM → neighborhood/manifold representations → independent source separation → nonnegative factorization**. Explain different goals and assumptions at these boundaries, with local prerequisite refreshers. Use the saved destination notes and the existing lessons' actual coverage. Original sources are recoverable at commit `8c5da59f18516be77c29d5aeeafca3decca4f738`; per-topic designs record their source hashes and retained coverage.

Each packet includes a complete learner manuscript, changed practice with hints and explained solutions, annotated primary and alternate resources, offline licensed data, bounded author calculations and actionable visual/investigation contracts. The representations follow each mechanism: mixture probabilities and covariance, graph/neighborhood relationships and image identity, and signal mixing and source contributions. Inputs, predictions, expected contrasts and null cases are specified; no browser visuals or labs were implemented. UMAP's complete teaching programs remain explicitly unexecuted because the dependency was absent; actual UMAP coordinates and comparison results must be generated and assessed during phase two, never fabricated.

The final content reconciliation corrected first-pass/deeper readiness, copied practice inputs, math notation, setup instructions and actual lesson routes. It preserved informative unfavorable outcomes and local exceptions instead of tuning examples to guarantee a preferred method wins. Scope/title decisions and canonical-reference coverage are recorded separately for each topic. Discoveries owned elsewhere are saved against the actual [NMF](docs/teaching/topic-notes/non-negative-matrix-factorization-nmf.md), [neural preprocessing](docs/teaching/topic-notes/neural-preprocessing-artifact-rejection-and-leakage-safe-pipelines.md) and [audio source separation](docs/teaching/topic-notes/source-separation-audio-denoising-demucs-band-split-rnn.md) topics; those lessons were not rewritten.

For an authorized finish request, use `node scripts/build-curriculum-inventory.mjs --topic <stable-id> --work finish` for the selected topic and consume its full saved packet. Complete implementation, displayed-program execution, model checks, independent correctness/learning-experience review, browser/accessibility review and integration then. Content completion is neither publication nor user acceptance; do not reopen unrelated completed topics or start the next topic without scope from the user.

**Checkpoint validation, 12 September 2026:** all 20 required packet/design files are hash-bound in the central ledger. `verify-lesson-delivery.mjs` passed its eight behavior groups with 114 tracked topics; all three read-only finish preflights returned content `complete`, implementation `not-started`, and `canFinish: true`. Scoped relative-link, actual route and module-membership checks passed after the ICA route correction. Inventory regeneration retained 1,218 topics, 28 modules, 228 publications and seven paths; its only topic-state changes are these three content checkpoints. All three original JSX hashes still match the starting source, and the working changes are documentation/data handoffs only. `git diff --check` passed. Application build, browser review and formal phase-two review were not performed. Disposable ICA source/download helpers were removed after preserving the offline extract and provenance; pending packets and the shared runtime were retained.

## Previous scope — Clustering Evaluation and DBSCAN finished; Anomaly Detection has since been finished too

On 12 September 2026 the user authorized research and writing for the next three Classical ML topics in module order, then asked for phase two on the first two of them. State at the end of that day:

| Position | Topic | Content | Implementation | Record |
| --- | --- | --- | --- | --- |
| 13 | Clustering Evaluation & Validation (Silhouette, ARI, NMI) | complete | **complete**, independently reviewed, integrated | [design and phase-two record](docs/teaching/CLUSTERING-EVALUATION-LESSON-DESIGN.md), [independent review](docs/teaching/CLUSTERING-EVALUATION-INDEPENDENT-REVIEW.md) |
| 14 | DBSCAN & Density-Based Clustering | complete | **complete**, independently reviewed, integrated | [design and phase-two record](docs/teaching/DBSCAN-LESSON-DESIGN.md), [independent review](docs/teaching/DBSCAN-INDEPENDENT-REVIEW.md) |
| 15 | Anomaly & Outlier Detection (Isolation Forest, One-Class SVM, LOF) | complete | **complete**, independently reviewed, integrated | [design and phase-two record](docs/teaching/ANOMALY-DETECTION-LESSON-DESIGN.md), [independent review](docs/teaching/ANOMALY-DETECTION-INDEPENDENT-REVIEW.md) |

The two finished lessons follow the PCA pattern: topic-owned models, embedded real data with provenance and a downloadable CSV under `public/learn-assets/`, executed displayed programs, four or five investigations with recorded predictions, model/native/browser verifiers with evidence under `docs/teaching/evidence/`, blueprints registered, and both ledger phases closed with final hashes. Every disposition of the independent reviews is recorded in the respective design record. User acceptance is separate. Nothing is committed.

Anomaly & Outlier Detection was finished on 13 September 2026; the section below records it. The new content-only scope above proceeds independently.

## Gaussian Mixture Models & the EM Algorithm — both phases complete, 13 September 2026

**Gaussian Mixture Models (GMM) & EM Algorithm**, Classical ML position 16 of 39, was implemented from its content-first packet immediately after Anomaly Detection, on the user's request to finish the next unimplemented topic in sequence. The [design record](docs/teaching/GMM-LESSON-DESIGN.md) carries the phase-two section: what was built, the draft-and-commit interaction contract the specification required, the sixteen formulas re-set for a narrow column, the checks actually run and the disposition of the [independent review](docs/teaching/GMM-INDEPENDENT-REVIEW.md).

The published lesson is the rewritten `src/learn/data/topics/gaussian-mixture-models-gmm-em-algorithm.jsx`: twelve sections, six inline figures, three investigations, three executed Python programs and practices 1 to 8. It owns `gmm-models.js`, the generated `gmm-iris-data.js`, `gmm-examples.js`, three lab components and a stylesheet, with the Iris CSV served from `public/learn-assets/gmm/iris.csv`. Its blueprint is registered by title.

Evidence: [model checks](docs/teaching/evidence/gmm-models.json) (80 grouped checks), [executed programs](docs/teaching/evidence/gmm-native.json), [the refitted candidates](docs/teaching/evidence/gmm-iris-data.json) and [the browser pass with screenshots](docs/teaching/evidence/gmm-browser.json) (10 cases at 1366, 390 and 320 px). The [phase ledger](docs/teaching/lesson-delivery-progress.json) records implementation complete with the final source hashes.

Two process notes carry forward. A figure can be mathematically wrong while every assertion passes: the covariance gallery drew two of its four families as straight lines because a closed-form eigenvector helper returned one direction twice, which only the screenshots revealed. And a prediction must be graded against the inputs committed with it, not against the state that happened to be rendered; computing the answer from the committed draft is the fix.

User acceptance is separate from this completion. Nothing is committed.

## Anomaly & Outlier Detection — both phases complete, 13 September 2026

**Anomaly & Outlier Detection (Isolation Forest, One-Class SVM, LOF)**, Classical ML position 15 of 39, was implemented from its content-first packet on the user's request to finish the first unimplemented topic in module order. The [design record](docs/teaching/ANOMALY-DETECTION-LESSON-DESIGN.md) carries the phase-two section: what was built, why the real series ships as precomputed exact outcomes, the six formulas re-set for a 320 px screen, the checks actually run and the disposition of the [independent review](docs/teaching/ANOMALY-DETECTION-INDEPENDENT-REVIEW.md).

The published lesson is the rewritten `src/learn/data/topics/anomaly-outlier-detection-isolation-forest-one-class-svm-lof.jsx`: thirteen sections, four static figures, six investigations, five executed Python programs and practices A to J. It owns `anomaly-detection-models.js` (exact BigInt rational arithmetic), `anomaly-temperature-data.js` (the generated Numenta Anomaly Benchmark series), `anomaly-detection-examples.js`, four lab components and a stylesheet, with the pinned CSV, annotation JSON and MIT notice served from `public/learn-assets/anomaly-detection/`. Its blueprint is registered by title in `src/learn/data/curriculum/blueprints/index.js`.

Evidence: [model checks](docs/teaching/evidence/anomaly-detection-models.json) (63 grouped checks), [executed programs](docs/teaching/evidence/anomaly-native.json), [the regenerated series](docs/teaching/evidence/anomaly-temperature-data.json) and [the browser pass with screenshots](docs/teaching/evidence/anomaly-detection-browser.json) (13 cases at 1366, 390 and 320 px). The [phase ledger](docs/teaching/lesson-delivery-progress.json) records implementation complete with the final source hashes; the content checkpoint re-binds only the design record, which phase two appends to by design.

One process note worth carrying forward: the first browser pass satisfied every assertion and still looked wrong in five places that only screenshots revealed, including an SVG path filled black because its shared class set a stroke but no fill, and a plot panel painting over the paragraph beneath it. Inspect the screenshots; a green verifier is not a rendered page.

User acceptance is separate from this completion. Nothing is committed.

## PCA & Dimensionality Reduction — both phases complete, 12 September 2026

**PCA & Dimensionality Reduction**, Classical ML position 12 of 39, was written content-first and then finished on the user's request the same day. The [design, research and phase-two record](docs/teaching/PCA-LESSON-DESIGN.md) holds the implementation inventory, the checks actually run, the author's learning-experience checklist and the disposition of the [independent review](docs/teaching/PCA-INDEPENDENT-REVIEW.md). Evidence: [model checks](docs/teaching/evidence/pca-models.json), [executed programs](docs/teaching/evidence/pca-native.json), [browser pass and screenshots](docs/teaching/evidence/pca-browser.json). The [phase ledger](docs/teaching/lesson-delivery-progress.json) records implementation complete with the final source hashes. The published lesson is the new `src/learn/data/topics/pca-dimensionality-reduction.jsx` with topic-owned models, Wine data, labs, figures and a downloadable CSV at `public/learn-assets/pca/wine.csv`. The phase-one drafts remain as retained inputs.

User acceptance is separate from this completion. Nothing is committed. The three content packets above continue independently; this completion does not authorize implementing them.

## Previous single-topic implementation — historical completion and pending revision

**K-Means & Hierarchical Clustering**, Classical ML position 11 of 39, is implementation-reviewed and production-integrated as of 11 September 2026. Its [completed design and integration record](docs/teaching/K-MEANS-HIERARCHICAL-LESSON-DESIGN.md), [exact-source ledger](docs/teaching/classical-ml-unsupervised-progress.json), [independent content review](docs/teaching/K-MEANS-HIERARCHICAL-INDEPENDENT-REVIEW.md), [final visual review](docs/teaching/K-MEANS-HIERARCHICAL-VISUAL-MODEL-REVIEW.md) and [production evidence](docs/teaching/evidence/k-means-hierarchical-browser.json) close the single-topic request. All concrete findings are resolved. The final label-layout amendment preserves numerical geometry and reuses the unchanged complete production checks.

**12 September 2026:** a proposed quality revision of this lesson is implemented in the working tree, uncommitted and unreviewed; see the design record's final section. The user is deciding whether to keep it. Until then the ledger's source hashes describe the previous reviewed state, not the working tree.

The K-Means proposal remains a separate user decision; the subsequent authorized PCA content scope is recorded above. User acceptance is separate from implementation review. Preserve IDs and module order, and do not reopen K-Means or the previous ten-topic increment merely on resume.

## Previous authorized increment — complete

The first **ten Classical Machine Learning topics** have been rewritten, independently reviewed and production-integrated in their existing module order. Their source-bound completion is recorded below. All ten previously had older published bodies; those earlier publications were not current-standard review. Existing depth and every stable identity were preserved while inaccurate claims and incomplete mechanisms were repaired.

- [Completed implementation record and checkpoint](CLASSICAL-ML-SUPERVISED-IMPLEMENTATION.md)
- [Ten-topic progress ledger](docs/teaching/classical-ml-supervised-progress.json)
- [Immutable starting catalogue, publication and original-source baseline](docs/teaching/evidence/classical-ml-supervised-baseline.json)
- [Final production integration](docs/teaching/CLASSICAL-ML-SUPERVISED-INTEGRATION.md)

There is no unfinished implementation in this increment. User acceptance remains separate. K-Means & Hierarchical Clustering was subsequently completed under the separate single-topic scope above. The earlier DSA/mathematics rollout remains complete. Reuse passing evidence for unchanged source instead of reopening completed checks.

## Teaching decisions to preserve

Linux Basics is the user's explicitly approved quality reference. Preserve its mechanism-first explanation, concrete examples, focused investigations and independent practice. Its four labs are an example, not a quota. The [Linux implementation record](PROGRAMMING-REWRITE-LINUX.md) and [subsequent visual review](VISUAL-TEACHING-REVIEW.md) provide context when a specific design decision needs it.

A completed, correctness-reviewed lesson was found on 12 September 2026 to teach less well than its evidence suggested: cautions repeated after every result, labs that were guided traces rather than investigations, a fixture that could not show the contrast placed beside it, a chart whose axis hid its own elbow, no real data, and displayed code bloated by reviewer-driven guards. The standard now carries the rules that would have caught each of these: the once-stated caution rule, the first-pass route, the three investigation requirements and fixture check, figure perceptibility, real data for data methods, displayed-code clarity, the canonical-reference coverage check, and a [learning-experience checklist](LESSON-TEACHING-STANDARD.md#learning-experience-checklist) that both author and independent reviewer run separately from correctness review. An agent doing end-to-end delivery runs that checklist before reporting readiness and records its findings; passing verifiers is not a substitute.

Start with the learner's problem and familiar intuition, then introduce terminology, mechanism, formal detail, worked examples and changed practice. Keep essential reasoning visible. Use inline diagrams at the point of explanation, plus interactive investigations when changing inputs reveals something useful. Multiple difficult mechanisms may need different diagrams or labs. Consistency means reliable teaching and usable controls, not identical boxes.

During writing, reconsider scope/title and valuable applications. Include material here when this is its best teaching home; otherwise persist a reasoned [destination-topic note](docs/teaching/topic-notes/README.md) for the actual owner. Preserve IDs, progress, memberships and old links if a title changes. Do not expand into a whole-catalogue audit.

Curate annotated articles, documentation and useful videos/playlists. Record what was actually read, executed or watched. Metadata alone does not establish a video's technical accuracy. Keep the lesson self-contained. Graphs must distinguish exact calculation, measured data and hypothetical illustration; attractive visuals do not verify nearby claims. The tokenization timing concern remains recorded in [its destination note](docs/teaching/topic-notes/byte-pair-encoding-bpe-wordpiece-sentencepiece-unigram.md).

## Sequence and engineering contracts

The live catalogue's module/section/topic order is the reading order. Sidebar, Previous/Next and compact topic/completion counts share the same resolved module scope. Planned entries remain in sequence. Do not skip ahead to a later published lesson or globally reorder by difficulty/prerequisites. Shared lessons retain their selected module context and one progress identity; provide explicit prerequisite review links.

Use topic-owned lesson, example, model, lab and blueprint files. Register publication in `lesson-manifest.json`; generated navigation and on-demand lesson/outline imports remain separate. Do not restore the old eager registry or batch-named production bundles. Keep interactive work bounded, preserve readable code and use the required checks for the structure actually changed.

## Completed work: consult only when relevant

All 17 Programming & Scientific Computing topics, all 22 DSA topics and all 57 Mathematical & Statistical Foundations topics have recorded implementation review. User acceptance is a separate state; Linux remains the explicitly approved reference.

- [Completed DSA/mathematics rollout and bounded extensions](DSA-MATH-FOUNDATIONS-IMPLEMENTATION.md), [74-topic ledger](docs/teaching/dsa-math-foundations-progress.json), [final integration](docs/teaching/DSA-MATH-FOUNDATIONS-INTEGRATION.md).
- [Programming completion](PROGRAMMING-MODULE-COMPLETION.md), [concept-specific visual improvements](VISUAL-TEACHING-REVIEW.md).
- [Source/loading migration](docs/engineering/LEARNING-LOADING-REVIEW.md), [runtime source ownership](docs/engineering/RUNTIME-SOURCE-ORGANIZATION.md).

Those records retain detailed history. Their old queues, counts and commands are not current instructions. Use the generated inventory for current catalogue counts and the active ledger for current work. Historical plans under `docs/superpowers/` and `docs/archive/` remain implementation records.

## Finish and hand off

Honor the requested phase. For content-first work, close the manuscript/specifications and research handoff, update only content completion and explicitly defer phase two. For full/finish work, close actual outstanding findings, record source-versioned author/independent evidence and complete the required final integration. A build, a registered page or a lab count does not establish teaching quality. Unknown checks stay unknown; passed unchanged checks are reused.

Update the central delivery ledger and the existing topic record's next action, remove disposable working material, and report the two phases separately with actual verification and limits. Preserve pending content-first drafts, referenced evidence and shared tools. Do not deploy unless requested.

Use [the working-artifact retention workflow](docs/engineering/WORKING-ARTIFACT-RETENTION.md) to remove obsolete images, drafts, patch scripts and library outputs when their job is finished. Keep selected final evidence; do not retain every intermediate capture or treat retained historical images as a reason to repeat old reviews.
