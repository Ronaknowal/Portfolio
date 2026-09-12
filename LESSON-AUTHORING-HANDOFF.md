# Educational authoring: current handoff

Updated 12 September 2026. This is the active entry point. The user's current request defines scope and delivery mode; historical reports and scratch files do not define a new queue.

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

## Latest authorized scope — Clustering Evaluation and DBSCAN finished; Anomaly Detection content pending

On 12 September 2026 the user authorized research and writing for the next three Classical ML topics in module order, then asked for phase two on the first two of them. State at the end of that day:

| Position | Topic | Content | Implementation | Record |
| --- | --- | --- | --- | --- |
| 13 | Clustering Evaluation & Validation (Silhouette, ARI, NMI) | complete | **complete**, independently reviewed, integrated | [design and phase-two record](docs/teaching/CLUSTERING-EVALUATION-LESSON-DESIGN.md), [independent review](docs/teaching/CLUSTERING-EVALUATION-INDEPENDENT-REVIEW.md) |
| 14 | DBSCAN & Density-Based Clustering | complete | **complete**, independently reviewed, integrated | [design and phase-two record](docs/teaching/DBSCAN-LESSON-DESIGN.md), [independent review](docs/teaching/DBSCAN-INDEPENDENT-REVIEW.md) |
| 15 | Anomaly & Outlier Detection (Isolation Forest, One-Class SVM, LOF) | complete | **not started** | [design and continuation](docs/teaching/ANOMALY-DETECTION-LESSON-DESIGN.md) |

The two finished lessons follow the PCA pattern: topic-owned models, embedded real data with provenance and a downloadable CSV under `public/learn-assets/`, executed displayed programs, four or five investigations with recorded predictions, model/native/browser verifiers with evidence under `docs/teaching/evidence/`, blueprints registered, and both ledger phases closed with final hashes. Every disposition of the independent reviews is recorded in the respective design record. User acceptance is separate. Nothing is committed.

Current next action: wait for the user. A request to finish Anomaly & Outlier Detection starts with `node scripts/build-curriculum-inventory.mjs --topic anomaly-outlier-detection-isolation-forest-one-class-svm-lof --work finish` and consumes its complete packet; that request has not been made. Gaussian Mixture Models & EM (position 16) has no content packet.

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
