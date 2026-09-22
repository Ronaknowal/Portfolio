# Research project authoring standard

Updated 22 September 2026. The user selected the finished **Typed Decision Model, depth revision 2**, as the reference for future projects. Preserve its explanatory depth, inspectable implementations, contextual concept links, varied live investigations and honest experimental reasoning. Adapt the system, stage sequence and representations to each project; do not copy its eight stages or four labs as a quota.

This is the dedicated project-authoring extension of the [teaching standard](LESSON-TEACHING-STANDARD.md). That standard owns shared teaching, accuracy, implementation-depth, live-exploration and quality-first requirements. This document owns how to apply them to end-to-end projects. [Learning workspace architecture](docs/engineering/LEARNING-WORKSPACE.md) owns routes, metadata, loading and learner progress; the [learning code standard](docs/engineering/LEARNING-CODE-STANDARD.md) owns naming, performance, source organization and temporary-file retention. Historical implementation reports describe what happened; they do not replace these current instructions.

## 1. Start with the current project, not the whole history

Read the user's scope and delivery mode, this standard, the relevant shared standards, the selected project's author record and current entry in the [project delivery ledger](docs/teaching/projects/project-delivery-progress.json). Then read its actual metadata, complete current manuscript/reader stages, canonical programs, specifications and relevant review findings. Use the ledger's current revision; consult superseded evidence only for a specific dependency or open question. Preserve unrelated uncommitted and untracked work.

The reference project is documented in its [current author record](docs/teaching/projects/typed-decision-model.md), [independent depth review](docs/teaching/projects/typed-decision-model-depth-review.md) and [final integration](docs/teaching/projects/TYPED-DECISION-DEPTH-INTEGRATION.md). Inspect the relevant stage sources and canonical programs when using an example from it. The integration describes the final reading experience; revision-1 gaps and its former single-lab/outline-only adapter are superseded. User preference for this teaching approach is not evidence that an unexecuted experiment has succeeded.

Implement only the requested project or extension. Adding a project does not implement its companion topic, authorize a curriculum rewrite or start the next project. Within the authorized scope, proceed without additional permission gates. Record the next concrete action so a different agent can resume without the original conversation.

## 2. What a project must help the learner do

A project connects concepts into a working artifact and teaches the decisions between them. The learner should be able to explain the problem, reconstruct the owned mechanisms, run and modify the system, diagnose failures, compare alternatives, and justify what the evidence supports. A collection of commands, a repository download, or a tour of APIs alone does not meet that goal.

Promise a bounded outcome precisely: what is built, its inputs and outputs, the supported setting, required prior knowledge, and what counts as a completed artifact. Explain why the system matters before naming its components. Give a simple map of the whole build, then deepen each part without requiring unexplained future knowledge. Distinguish research understanding, a reproducible local prototype, and operational deployment; claim only the level actually taught and implemented.

Keep these three layers connected:

| Layer | Its job | Link and ownership rule |
| --- | --- | --- |
| Concept lesson | Explain a mechanism deeply across settings. | Reuse an actual taught implementation through its exact owner; inspect that owner first. |
| Project stage | Apply and connect mechanisms to make a working system and reason about decisions. | Explain the local contract and every project-specific transformation; contextual links extend that explanation. |
| Program and experiment | Make the claims inspectable and reproducible. | Display and download the same canonical source, with real inputs, outputs and execution status. |

Do not force learners to leave the project repeatedly just to understand its dataflow. Links provide prerequisite refreshers and deeper alternatives; they must not conceal missing project explanation.

## 3. Design the build before writing its stages

Create or update `docs/teaching/projects/<stable-project-id>.md` as the durable author/design record. Keep it concise enough to navigate while specifying the following decisions in usable detail:

- **Question and artifact:** intended learner, motivation, concrete output, supported inputs, success criteria, resource assumptions and exclusions.
- **Coverage and ownership:** each promised mechanism, whether taught locally or reused, exact prerequisite/companion IDs and sections, and the newly owned integration work. A planned topic cannot count as an already-taught prerequisite.
- **Stage graph:** stable stage IDs, prerequisite order, what enters each stage, what is built, and the artifact/evidence carried to the next stage. The displayed route must follow that order, not skip to the next published stage.
- **Running example:** one understandable input or scenario that can be traced across stages, plus changed cases that expose generalization and failure.
- **Implementation routes:** the scratch abstraction boundary, complete ordinary-library/tool route, their correspondence, meaningful customization, and the comparisons needed to check them.
- **Teaching support:** conceptual hurdles, appropriate inline diagrams, live investigations, worked cases, practice and feedback. Choose forms for the mechanism rather than a fixed component template.
- **Research and evaluation:** sources for material claims, dataset provenance and splitting where relevant, baselines, measurements, diagnostic questions, unresolved uncertainty and explicit execution boundaries.
- **Delivery:** requested mode, prepared/source locations, current revision, affected interfaces, required verification and precise next action.

Revisit scope and title while writing. If a valuable missing mechanism belongs here and is needed to meet the outcome, include it and rename the display title if necessary while preserving stable identity. If another topic/project owns it better, add a durable destination note with the discovery, reason, exact source owner and suggested teaching treatment; link it from the current record and the destination's author record or existing topic-note workflow. Do not silently expand into that other implementation. Avoid promising “everything” without a stated scope.

## 4. Let the artifact determine the stage sequence

Stages follow dependencies and meaningful deliverables. Common needs include framing a problem, constructing inputs, establishing a baseline, building mechanisms, running the system, evaluating it and packaging its results. Combine or split these when it improves learning. A short stage is justified by a real decision or handoff, not a desire for more navigation items.

The reference project demonstrates the following sequence; these are examples, not mandatory headings:

| Reference stage | What it teaches locally | Evidence carried forward |
| --- | --- | --- |
| Define the decision | Separate a typed answer, a probability and a consequential action. | Task contract and explicit costs. |
| Build the evidence | Trace a request into vocabulary IDs, positions, candidate indices, batches and distinct masks. | Data/split contract and valid encoded examples. |
| Establish a baseline | Set up the program, calculate a simple competitor and interpret its outputs. | Reproducible baseline report. |
| Build the mechanism | Follow attention, residual/FFN transformations, candidate gathering and shared scoring. | Inspectable model, shapes and mechanism checks. |
| Train with a purpose | Follow error through gradients, clipping, optimizer state and parameter updates. | Trained checkpoint and interpreted learning trace. |
| Calibrate and decide | Separate score order, probability quality and cost-sensitive policy. | Saved calibration and explicit policy assumptions. |
| Try to break it | Examine errors, shifts, ordering effects and controlled comparisons. | Diagnostics, limitations and release reasoning. |
| Package and extend | Restore artifact meaning, run inference and map to an ordinary library adapter. | Reproducible bundle and concrete next experiments. |

Other domains need different builds:

| Project kind | Typical emphasis and appropriate evidence |
| --- | --- |
| Systems or GPU engineering | Contracts, data/memory flow, scheduling, concurrency, faults, recovery, resource budgets and measured end-to-end latency/throughput. Use timelines, queues, memory maps or kernel traces. |
| Mathematics or algorithms | Derivation, assumptions, proof/invariant, numerical realization, complexity and counterexamples. Use geometric constructions or state traces; do not invent a library route for an outcome that has no computational counterpart. |
| ML or paper reproduction | Data protocol, simple baselines, architecture/objective, controlled comparisons, seed uncertainty, ablations, error analysis and restoration. Distinguish a faithful reproduction from an adaptation. |
| Robotics or neurotechnology | Units, coordinate/signal conventions, sampling, sensing, feedback, simulation/hardware boundaries, noise, latency and relevant safety constraints. Simulated evidence cannot establish real-device or clinical performance. |
| Trading or financial systems | Data timing, leakage, costs, execution assumptions, risk, replay and regime changes. Backtests and constructed markets do not establish live returns. |

These are adaptation prompts, not completeness checklists for every domain. Research the actual project and current authoritative sources; do not transplant assumptions from the decision-model reference.

## 5. Write the explanation around the mechanism

Within each stage, establish the next problem in plain language, connect it to the previous deliverable, trace a concrete case, explain the implementation choices, let the learner inspect or change the mechanism, interpret the result, and carry the artifact forward. Distribute this naturally; do not stamp identical headings onto every stage.

For every important transformation, make the learner able to answer:

1. What enters and leaves it? Name the entities, meanings, units, shapes or state.
2. Why is it necessary here? What breaks if it is omitted or changed?
3. How does it work? Derive or trace the essential steps before compressing them into terminology.
4. Where is that step in the code? Explain important operations beside the relevant excerpt.
5. What can be controlled? Identify a useful alternative and its consequence.
6. How can correctness or usefulness be assessed? Separate invariants from empirical hypotheses.

Use progressive disclosure for full source, detailed derivations, optional diagnostics and solutions. Keep the core reasoning visible. A long collapsed program followed by a short description is insufficient. Explain symbols before using them and preserve naming across text, diagrams, code and saved artifacts. If examples differ, state the changed input instead of letting two incompatible counts or shapes look like the same trace.

The reference's padding explanation illustrates the required depth: it distinguishes application IDs, candidate indices, vocabulary IDs and sequence positions; constructs a concrete batch; explains why token and candidate masks solve different problems; traces the filler position; and provides a changed-order exercise whose failure can pass a shape test. Aim for that causal clarity, not the same number of paragraphs.

## 6. From scratch, ordinary tools, and researcher control

Follow the shared [implementation-depth contract](LESSON-TEACHING-STANDARD.md#build-the-mechanism-then-control-the-library). For every core computational outcome, provide the actual explained scratch implementation or inspect and link its already-taught owner. State which primitives are delegated: using arrays, tensor operations, autograd or an existing optimizer is appropriate when those are outside the owned mechanism. Wrapping a library encoder is not a from-scratch implementation of the encoder.

Teach the construction, then the ordinary tool route and a meaningful modification. Map variables, layouts, state, masks, reductions, initialization, randomness, train/eval behavior, persistence and defaults when they affect semantics. An executable adapter and its explanation fulfill a library promise; naming a package or linking documentation does not. Match inputs and state when comparing outputs, gradients or transitions; disclose intentional differences and numerical tolerances.

The final scratch implementation must be carefully engineered: an appropriate algorithm, stable numerics, clear contracts, sensible time/memory costs, correct lifetime/state handling and useful extension points. A naive introductory version may help intuition, but follow it with the efficient owned implementation or an exact taught owner where the naive approach is materially inadequate. Do not claim universal hardware optimality from asymptotic reasoning or one benchmark. Keep validation and infrastructure organized so they do not bury the algorithm being taught.

Give files semantic names reflecting their responsibility. Explain filenames when downloads, imports or commands depend on them. In the reference, `research_tools.py` and `pretrained_decision.py` import the canonical `typed_decision.py`; that relationship is taught explicitly. Do not invent batch-oriented names such as `next-three` or keep disconnected copies of the same implementation. Reuse the canonical core, and explain a deliberate fork or adaptation when genuinely needed.

Write this code and its explanation during content preparation. “The implementer will add the scratch version later” is a missing content outcome. Full packaging and execution can be deferred by the requested delivery mode; the promised teaching cannot.

## 7. Choose visuals and live labs for each hurdle

Match the representation to what the learner must see: tokens and marker positions for encoding; aligned tensor axes for reshaping; linked weights/outputs for attention; a parameter-update trace for optimization; a queue timeline for contention; a spatial scene for geometry. Use diagrams inline where they reduce inference work. Use multiple investigations when different mechanisms need different views; reuse an existing form when it genuinely fits. There is no required count or one generic lab for a whole project.

Every specified figure or lab records:

| Specification | Required teaching information |
| --- | --- |
| Placement and purpose | Stage/paragraph, question it answers, and what the learner should notice. |
| Entities and model | Formulas or process, units, shapes, identities, data provenance, assumptions, ties/tolerances and supported bounds. |
| Controls and feedback | Editable entities, starting state, valid changes, linked readouts, decision consequence and deterministic reset. |
| Representation | Labels, legend, meaningful intermediate states, text equivalent and small-screen composition. |
| Verification | Independent arithmetic/native comparison where relevant, boundary cases, cross-view agreement and informative browser states. |

**Live exploration has no learner-prediction entry, grading or reveal gate, even optional.** The current result is visible immediately. Valid changes update every dependent view together. A Step/Run button may advance a real process or launch bounded expensive work; it must not unlock an answer. Independent exercises and their solutions can remain separate from the lab.

Affordances must be honest. A handle that looks draggable needs real pointer/touch behavior and keyboard or numeric access. A static ruler should look and read like a static diagram. Show exact values, clear control labels, reset and a concise guide to what changing the control reveals. Do not use generic grey browser buttons, unstyled blue links, decorative green/olive panels or broad gradient overlays that violate the project's neutral/amber theme. Distinguish data series with labels/shape as well as color.

Bound browser work. Cheap derived models can update immediately; expensive work needs an appropriate explicit run, cancellation and current/pending-state handling. Never retrain an unbounded model on every keystroke. Label a constructed attention or frozen-feature gradient lab as such; it is not execution of the whole trained model. Source a factual quantitative graph from executed code/data or a derivation. An illustrative graph must identify its assumed relationship and cannot establish measured timings or rankings.

## 8. Examples, practice, and interesting connections

Keep a running example to establish continuity, then vary the conditions so the learner must transfer the idea. Add ordinary and unusual applications when they teach a useful connection: explain the problem, map the entities, work through the mechanism and interpret its limits. Avoid trivia lists or a quota of surprising facts.

Practice should include appropriate combinations of tracing, implementing a missing transformation, debugging a failure, changing a design, comparing an alternative and interpreting evidence. Provide concrete inputs, success criteria, hints and a reasoned solution or diagnostic guide. An open research exercise may have several valid outcomes; give a protocol and interpretation criteria instead of fabricating one correct result. Do not make every exercise a copied run of the preceding command.

A stage milestone names a verifiable deliverable and explains how the learner checks it. Browser checkboxes record local learner progress only; they are not proof of experiment success or mastery. The final project should let the learner recreate the artifact, make a meaningful change, and explain its consequences using the linked concepts.

## 9. Research, resources, and honest results

Research uncertain, niche, current or version-sensitive claims with primary papers, official documentation and inspectable source. Record what a source supports and its version/date when relevant. Read a model's disclosed architecture rather than inferring it from a marketing name or similar interface. Distinguish known mechanisms, interpretations, adaptations and undisclosed details. Never reproduce promotional certainty as a scientific conclusion.

Place exact curriculum links beside the relevant explanation with a sentence saying what the learner gains there. Verify stable topic IDs and section targets. Clearly label planned outlines, keep companion publication independent, and supply the local teaching needed to complete the project despite an unwritten companion.

Provide annotated references and another way to learn: useful papers, docs, articles, talks, videos or playlists where they genuinely improve understanding. Inspect relevance before linking, explain which part helps and any prerequisite or version mismatch, and use reliable source attribution for technical claims. Do not require an external video to supply a missing project mechanism or pad the list to meet a resource count.

Make experiment reports reproducible and interpretable: source revision, data provenance, preprocessing and split roles, configuration, seeds where relevant, environment/hardware, protocol, metric definitions/denominators, measured outputs, uncertainty and limitations. Separate setup/compilation/warmup from timed work when benchmarking, and measure the scope claimed. Keep real data, constructed fixtures, analytic examples and simulations distinct.

Preserve negative findings. In the reference, the simple baseline wins on the constructed test set, calibration worsens loss, and candidate ordering matters. Explaining these outcomes and useful follow-up experiments is better teaching than presenting an unsupported success story. A tiny offline library model can establish API mechanics; it cannot establish meaningful pretrained quality. Printed commands do not establish that training, deployment, hardware or clinical experiments were executed.

Each proposed extension states the question, controlled change, expected evidence, comparison and interpretation, plus resource/dependency limits. Keep extensions clearly separate from delivered outcomes. A release decision may legitimately be “do not use this artifact in that setting.”

## 10. Support full delivery and content-first work

Use the same two-phase philosophy as lessons, but keep project files and status in the project system. An ordinary request to implement a project means full delivery. A content-only request stops at the content boundary; quality-first guidance does not authorize crossing it.

| Request | Deliverable and stopping point |
| --- | --- |
| Full implementation | Complete research, writing, code, topic-specific UI/labs, native checks, independent review, corrections, browser integration and handoff for the scoped outcome. |
| Research and write only | Complete learner-facing prose for every scoped stage, worked examples, actual explained scratch/library instructional code, practice/solutions, contextual links, references and actionable visual/lab specifications. Do not implement the reader/labs or publish the prepared revision. |
| Finish the prepared project | Verify a complete current content packet, read it in full, implement and verify it, and make necessary content corrections discovered during finishing. Reuse valid research instead of restarting it by default. |

For content-only work, use `docs/teaching/projects/<stable-project-id>/drafts/guide.md` and `visual-specifications.md` in that same directory, with semantic instructional source files beside them when needed. Keep the author record at `docs/teaching/projects/<stable-project-id>.md` as the entry point. The manuscript must contain the teaching, not instructions for the next author to write it. Small calculations or probes needed to substantiate written claims belong in writing; the full execution/review/integration campaign remains deferred.

Before finishing, check the packet's revision and hashes, completeness, scope and latest requirements. If an advertised core is absent, record and repair the content gap within the authorized finish scope; do not call an outline complete or implement around missing teaching. The lesson inventory command with `--work finish` is a topic preflight, not a project preflight. Use the project ledger and linked packet for projects; do not invent a project command that does not exist.

Full-mode authors can use semantic production stage sources as the content checkpoint without duplicating the entire guide into Markdown. Both modes record the content source identity before claiming its implementation is complete.

## 11. Workflow, review, and definition of done

These work stages implement the shared six-stage workflow. They are not six full rewrites or approval gates, and do not prescribe how many corrections a project may need.

| Work stage | Required outcome |
| --- | --- |
| Assess and research | Scoped design, ownership map, source review, dependency sequence and plan for each conceptual hurdle. |
| Write and specify | Complete connected manuscript, instructional code and comparisons, practice/feedback, resources and implementable visual/lab specifications. |
| Implement and author-check | Canonical runnable artifacts, rendered stages, actual native comparisons, live models and a full reading-flow assessment. |
| Independent review | A reviewer other than the author reads the complete scoped teaching and relevant source, investigates complementary correctness/learning risks and records findings against exact source. An author's reread is not independent review. |
| Correct and close browser behavior | Resolve findings, verify affected behavior, check actual interactions and inspect informative desktop/phone states. |
| Integrate and hand off | Preserve identities and navigation, verify loading and progress, record separate phase/evidence status, clean disposable work and state the next action. |

If no independent reviewer is available, record that limit honestly; do not mark that phase complete. Coordinate delegation only when authorized by the user or applicable instructions. Keep the complete review outcome available to the integration owner.

Review teaching and implementation separately. Useful checks include reconstruction of a small case, an independent implementation comparison, state restoration, a supported boundary or failure, a changed-input exercise, and coherence across code/diagram/readout. Choose checks for the actual claim. Two programs calling one shared helper establish agreement rather than independent correctness. A figure count, long manuscript, source-excerpt count or passing build cannot establish depth.

Final browser verification covers every scoped stage, source disclosures, precise contextual links, controls with actual pointer and keyboard input, reset, valid/invalid changes, loading/error recovery, stage order, local progress and readable geometry. Inspect informative changed states at desktop and narrow widths, including a 320-pixel viewport when supported by the current reader. Check label collisions and clipping, not only page overflow. Preserve useful table meaning on phones through reflow, labeled stacked values or appropriate deliberate scrolling.

Use the current [workspace checks](docs/engineering/LEARNING-WORKSPACE.md#verification-and-continuation) for registration and integration. `node scripts/verify-learning-projects.mjs` checks the project registry, ordered stages and metadata boundaries; it does not certify teaching or numerical correctness. Run a production build and relevant browser checks after source changes, with project-specific checks for new mechanisms. Documentation-only changes need link/instruction consistency checks, not retraining or a new application build.

Ready for review means the promised outcome is taught and runnable within its stated limits, scratch/tool correspondence is explained, practice has usable feedback, material claims have support, interactions work, and no material review/integration finding remains. User acceptance stays separate. Neither acceptance nor a clean review proves perfection or validates an unexecuted extension.

## 12. Source organization, checkpoints, and cleanup

Follow the [workspace source ownership contract](docs/engineering/LEARNING-WORKSPACE.md#source-ownership-and-loading). Keep discovery metadata compact and project bodies lazy. Register metadata and the explicit stage map together; preserve stable project/stage IDs, ordered navigation, retry behavior and isolated learner progress. Use semantic project-local helpers and styles. Fetch large canonical code only when opened, and never put model weights, full manuscripts or training data in hub imports. Make practical performance improvements without changing the teaching model or hiding meaningful detail.

The [project delivery ledger](docs/teaching/projects/project-delivery-progress.json) is the current project phase index; the topic ledger is not. Preserve revision history. Track content, implementation, independent review, browser integration, publication and user review distinctly using the existing fields. A revision can have prepared content while its old version remains published. A new revision awaiting implementation is not ready merely because the old revision passed.

Link the current author record and retain there: requested mode, revision, source/packet hashes, design and source decisions, checks actually run with environment/commands, unresolved findings, deferred work, evidence links and exact next action. In content-only mode, content can be complete while implementation/review/browser phases remain not started. Never fabricate an independent review or native/browser result. Include canonical source and relevant data identities in execution evidence; explain which unchanged checks are reused after a correction.

Keep one current author record, a separate independent review and a scoped integration record as needed. Avoid a new report for every small edit. Existing measured reports, reproducible verifiers, required prepared packets and selected screenshots are durable evidence; transient downloads, patch scripts, redundant images and disposable run directories are not. Follow the [retention policy](docs/engineering/LEARNING-CODE-STANDARD.md#temporary-work-and-evidence-retention), verify resolved paths before deletion, and preserve user work, active tools and imported artifacts. Do not enumerate all scratch work or reopen old campaigns just to rediscover the next step.

Quality takes priority over efficiency. Reuse source-bound evidence, batch independent reads and target corrections, but expand research, explanation or verification whenever it meaningfully protects quality. No token-saving instruction limits depth or justifies leaving a known in-scope gap. Stop redundant work when it adds no evidence; do not stop substantive work because a preferred pass count was reached.

## 13. Reusable handoff and future request

The author record should make these answers easy to find, without copying the whole guide:

```text
Project ID / revision / requested delivery mode:
Promised artifact, learner outcomes, resource assumptions and exclusions:
Current stage IDs, order and deliverables:
Coverage owners and exact companion/prerequisite links:
Manuscript / instructional source / visual specifications / hashes:
Canonical program and runtime destinations:
Research decisions, provenance and unresolved questions:
Checks actually run, environment and retained evidence:
Independent findings, corrections and final reviewed source:
Deferred execution, publication state and extensions:
Project ledger entry and exact next action:
```

A future-session request can be:

> Read PROJECT-AUTHORING-STANDARD.md and the current project ledger. Use the completed Typed Decision Model depth revision as the teaching reference, adapt the stages and investigations to this project, and [implement it end to end / research and write only / finish its prepared content]. Preserve scratch and ordinary-tool depth, explain the code locally, link exact concept owners, and report evidence and remaining limits honestly.

The prompt selects a delivery mode and scope; the linked files contain the durable requirements. Do not depend on remembering the originating conversation.
