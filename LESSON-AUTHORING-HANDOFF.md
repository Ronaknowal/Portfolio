# Educational authoring: current handoff

Updated 28 September 2026. Read this entry point and the selected topic's record,
not the whole history of completed batches. The user's current request defines
scope and delivery mode. A completed historical increment is not a new work queue.

## Current state

The [Copy-button feedback follow-up](docs/teaching/lesson-code-access/copy-feedback.md)
adds an immediate, temporary amber “✓ Copied” state at each shared code button.
Its later presentation receipt owns the two changed shared runtime files; earlier
code-access and scientific receipts remain preserved snapshots.

The [visible teaching and code-access follow-up](docs/teaching/lesson-code-access/README.md)
keeps explanations, derivations, diagrams and instructional code in the reading
flow without dropdown clicks. Authored practice, hints and solutions retain
their disclosures. Code blocks share Copy/Download controls, canonical program
names and a common file index; practice and optional files keep their own
disclosure boundary. Large instructional source can load automatically near
the viewport. Future authoring must preserve these visible sections and use the
shared code/file components, including the renderer's explicit teaching metadata.
This later presentation receipt supersedes the lesson-ending receipt for current
source identity. Earlier ending, opening and scientific receipts remain unchanged
evidence for their recorded scopes; this follow-up is not a new scientific review
or an authoring queue.

The [recurring lesson-ending follow-up](docs/teaching/lesson-endings/README.md)
standardizes practice boundaries, native hint/solution controls, annotated
resource lists and continuation styling across published lessons. Further
learning and technical references remain separate in the shared `Sources`
component; authored resource groupings and DSA practice stages retain their
meaning. Use explicit ending metadata and the shared styles for future work.
This is a presentation review, not a new scientific review or authoring queue.
Its recorded presentation evidence remains preserved; use the later code-access
receipt for current source identity.

The [shared lesson-opening follow-up](docs/teaching/lesson-navigation/README.md)
is complete. Every published lesson now uses the same Learning compass immediately
below its title: topic-owned guidance and a full list of main-section jump links.
All 42 DL openings and 15 older formats were checked in the browser. The central
ledger retains the 81 current ML/DL checkpoints through a scoped presentation
review, preserving the original scientific receipts and 96 historical rows.
Use the latest presentation review for the shared-source identity. Register future
guidance with `LessonIntro` or `Prose opening`; do not add separate custom TOCs.

The implementation of **all 24 prepared DL topics, positions 19–42, is complete**.
All **42 Deep Learning Fundamentals & Architectures lessons** now have completed
implementation checkpoints. The first 18 were preserved; the remaining 24 have
current author, independent, browser and production-build evidence. See the
[completed implementation record](docs/teaching/deep-learning-completion/README.md)
and [final integration checks](docs/teaching/deep-learning-completion/integration-checks.json).
There is no remaining implementation queue for this module. A new user request
defines the next scope; user acceptance remains separate from implementation review.

The 26 September concept-by-concept intuition revision is **complete for all 39 implemented Classical ML lessons and the first 18 implemented Deep Learning lessons**. All 57 revised topics have current content and implementation checkpoints, independent concept/correctness reviews, rendered observations and final build evidence. See [the completed review record](docs/teaching/concept-intuition/README.md). Review every learning transition within each future lesson, not just its opening; keep depth, code and practice. Prepared but unimplemented topics retain their phase boundary. The central delivery ledger owns phase status; this completed review is evidence, not a new work queue.

The [central delivery ledger](docs/teaching/lesson-delivery-progress.json) records
**177 content-complete topics, 177 implementation-complete topics, and zero prepared
implementations remaining**. These are the 27 September recorded checkpoint counts, not
hard-coded future targets. Publication, current authoring revision and user
acceptance are separate states. Use the topic preflight for effective status and
source identity instead of interpreting an old count as current.

All Programming & Scientific Computing, DSA, Mathematical & Statistical Foundations,
and Classical ML improved implementations have recorded reviews. All **42**
Deep Learning Fundamentals & Architectures topics have current implementations.
For future content-first requests, prepared still does not mean implemented.
Retain packets and their code, references, visual/lab specifications and
source-bound checkpoints when a phase-two implementation consumes them.

The earlier [57-topic concept-by-concept revision](docs/teaching/concept-intuition/README.md) includes local teaching improvements to Attention, Long Context and State Space and remains current for its unchanged scope. Their earlier [revision-4 reading-progression work](docs/teaching/ATTENTION-MEMORY-INTUITION-REVISION.md) and [original implementation record](docs/teaching/DEEP-LEARNING-ATTENTION-MEMORY-IMPLEMENTATION.md) remain historical evidence for unchanged explanations, numerical engines, programs and measured experiments. The central ledger points to each topic's current manuscript and review; do not substitute an older packet for that source.
The [prepared-writing revision](docs/teaching/implementation-depth/PREPARED-WRITING-REVISION.md)
added explained scratch/library code and specifications. All 24 of those DL packets
have now completed phase two, including native execution, independent implementation
review, website integration and browser checks. The completed [implementation-depth remediation](docs/teaching/implementation-depth/REMEDIATION.md)
and [lesson usability review](docs/teaching/LESSON-USABILITY-REVIEW.md) are evidence
for their recorded scope, not a reason to reopen every lesson or certify unrelated code.

The current Windows checkout's strict source-identity inventory reports **81 current
content and implementation checkpoints**, distinct from the recorded totals above.
These 39 Classical ML and 42 DL topics match their exact hashes. The other **96
historical ledger rows** remain preserved, not retrospectively recertified.
The DL completion preserved all **232 original publication mappings** and added
Mini-batches and Training Diagnostics, for **234 current mappings**.
Earlier work identified CRLF checkout differences in some historical prepared files.
If a future selected topic fails preflight, distinguish a line-ending change from a
substantive edit. Restore line endings only when the resulting bytes equal its
existing hash; otherwise investigate the real change. Never refresh all historical
hashes merely to make counts agree. See [the final integration evidence](docs/teaching/deep-learning-completion/integration-checks.json)
and the topic's browser receipt for actual coverage and limits.

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

The user's selected mode persists through the requested batch and clear continuations. A content-only completion must say that implementation remains pending. Publication, historical implementation review and user acceptance are separate from the current revision's two statuses. The active ledger and source hashes determine current effective status; historical migrations do not authorize a new queue.

<a id="follow-up-feedback-inline-diagrams-and-concept-specific-labs"></a>

## Current teaching requirements

Follow the standard's [quality-first rule](LESSON-TEACHING-STANDARD.md#quality-takes-priority-over-efficiency).
Efficiency defaults never justify omitting necessary explanation, research or
verification. Preserve from-scratch **and** normal library/tool routes with locally
explained code, exact prerequisite reuse, stable/efficient algorithms, stated
abstraction boundaries and customization practice. “Optimal” requires a workload
and cost model; do not claim unmeasured universal performance. Content-first work
must already include complete explained code and precise visual/lab specifications,
not a task for the finish agent to research missing core teaching.

Use topic-specific illustrations and playable investigations where they help.
Apply the [teaching loop at every conceptual transition](LESSON-TEACHING-STANDARD.md#apply-the-teaching-loop-at-every-conceptual-transition), including variants, implementation choices and advanced branches. The topic's design map must locate the actual explanation, worked intermediate steps and useful representation for each new hurdle; an introductory analogy alone is insufficient.
Changing a control should show its effect immediately; there is **no learner-
prediction entry, even optional**, and no answer-unlock gate. Scientific model
predictions and separate practice remain valid. Use multiple investigations when
distinct mechanisms need them, not a fixed quota or identical plain boxes.

For substantive verifier/model or visual work, consult the relevant sections of
[verification pitfalls](docs/engineering/LESSON-VERIFICATION-PITFALLS.md): independent
trust roots, painted versus nominal geometry, nonempty subject sets, meaningful
failure checks, honest coverage, crash-safe mutation tools and preserved evidence.
This consolidates reusable lessons from completed batches; it is not a new audit queue.

## Guided projects and curriculum planning

For an educational build, read [PROJECT-AUTHORING-STANDARD.md](PROJECT-AUTHORING-STANDARD.md)
and [LEARNING-WORKSPACE.md](docs/engineering/LEARNING-WORKSPACE.md). The approved
Typed Decision Model direction is captured in [its depth integration](docs/teaching/projects/TYPED-DECISION-DEPTH-INTEGRATION.md)
and separate [project ledger](docs/teaching/projects/project-delivery-progress.json).
Project milestones and authoring phases must not increment lesson completion.
Do not describe unexecuted pretrained/RL/service extensions as completed experiments.

The live catalogue and topic preflight own module order, prerequisites, named
subtopics and destination notes. Domain plans for neural engineering, GPU engineering,
professional trading, system design and quantum computing define planned scope;
they do not prove a topic is written or implemented. Read the relevant plan only
for the task's subject. Site architecture and Articles have separate instructions
in the root [AGENTS.md](AGENTS.md).

## Teaching decisions to preserve

Linux Basics is the user's explicitly approved quality reference. Preserve its mechanism-first explanation, concrete examples, focused investigations and independent practice. Its four labs are an example, not a quota. The [Linux implementation record](PROGRAMMING-REWRITE-LINUX.md) and [subsequent visual review](VISUAL-TEACHING-REVIEW.md) provide context when a specific design decision needs it.

A completed, correctness-reviewed lesson was found on 12 September 2026 to teach less well than its evidence suggested: cautions repeated after every result, labs that were guided traces rather than investigations, a fixture that could not show the contrast placed beside it, a chart whose axis hid its own elbow, no real data, and displayed code bloated by reviewer-driven guards. The standard now carries the rules that would have caught each of these: the once-stated caution rule, the first-pass route, the three investigation requirements and fixture check, figure perceptibility, real data for data methods, displayed-code clarity, the canonical-reference coverage check, and a [learning-experience checklist](LESSON-TEACHING-STANDARD.md#learning-experience-checklist) that both author and independent reviewer run separately from correctness review. An agent doing end-to-end delivery runs that checklist before reporting readiness and records its findings; passing verifiers is not a substitute.

Start with the learner's problem and familiar intuition, then introduce terminology, mechanism, formal detail, worked examples and changed practice. Keep essential reasoning visible. Use inline diagrams at the point of explanation, plus interactive investigations when changing inputs reveals something useful. Multiple difficult mechanisms may need different diagrams or labs. Consistency means reliable teaching and usable controls, not identical boxes.

The user's 26 September review of Attention, Long Context and State Space requested
stronger intuition even after their correctness checks passed. Revision 4 rebuilds
those lessons around a concrete need, a visible limitation and one complete repair,
then connects diagrams, worked values, notation and implementation. Follow the
standard's **Build the reason before naming the machinery** guidance: investigate
how strong resources teach the idea, preserve technical depth, and review the full
reading progression rather than adding a generic introduction. Revised content and
reviews live under `docs/teaching/revisions/<topic-id>/4/`; revision-3 scientific
evidence remains historical evidence for its unchanged engines and assets.

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
