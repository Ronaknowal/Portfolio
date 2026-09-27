---
name: lesson-authoring
description: Research, write, implement, improve, or review in-depth technical learning topics with concept-by-concept intuition, topic-specific visuals and live labs, explained from-scratch and library code, practice, and source-bound two-phase handoffs. Assess significant new research and models for curriculum updates or missing topics; maintain handoffs and clean completed authoring work safely. Use for learning-site lessons and prepared-content workflows, not general news summaries, ordinary blog writing, site design, or standalone software projects.
---

# Lesson authoring

Continue the established beginner-to-researcher teaching approach. Help a learner explain the mechanism, implement and modify it, use ordinary tools deliberately, diagnose failures and transfer the idea. Preserve useful existing depth. Neither a short definition nor an unexplained program meets that outcome.

## Start with the requested operation

**Help first:** If the user asks for `help`, “show options,” or what this skill can do, answer from the operation table below before any authoring preflight. Show a short plain-language menu with one example request per option, grouped as creating/continuing, checking/improving, and maintenance. Explain that mode names and exact syntax are optional: infer the operation from natural language. Do not inspect lesson inventories, browse research, edit files or begin lesson work merely to display help. For a bare skill invocation with no task, show this menu and let the user choose. Explain the content-only versus full boundary clearly; introduce less-used options without requiring the user to memorize them. Derive the menu from this table so future modes stay discoverable.

1. Read the selected checkout's `AGENTS.md` and current educational handoff. Preserve uncommitted work. For the ronak site, use [the repository adapter](references/portfolio-adapter.md); it locates current state and real commands. Do not use the personal skill's installation directory as the application root.
2. Resolve the topic IDs, module order, requested count and delivery mode from the user's request and existing continuation. Announce the mode and scope briefly. For “next,” inspect current effective ledger state in syllabus order, not publication order or an old batch list. If scope is ambiguous, progress on safe independent work and ask only for missing information that matters.
3. Load [workflow](references/workflow.md) when authoring or changing phases; for a scoped research-update, review or cleanup, begin with its reference and the applicable repository contract. Reuse current guidance already read; inspect changed or missing instructions. Read the complete selected manuscript/specifications or lesson when revising or certifying it. For research triage, inspect the actual affected sections before declaring a gap; a title or summary is insufficient evidence.

| Operation | Work and boundary | Guidance to load |
| --- | --- | --- |
| **full** — “implement” / “end to end” | Complete content, then implementation, review, corrections, browser checks and integration. This is the default for ordinary implementation requests. | Writing set below, then review and the repository's engineering contract. |
| **write** — “research and write only” / “content first” | Complete the actual manuscript, explained code, solved practice, annotated sources and actionable visual/lab specifications. Record content complete only when complete; implementation remains not started. Do not build runtime visuals/labs, change publication or start the full verification campaign. | Writing set; workflow's content-only handoff. |
| **finish** — “implement prepared content” | Require a complete current content checkpoint. Read and consume its entire packet, then implement and verify. Correct content when findings warrant it. A missing/stale prerequisite needs assessment, not a silent new writing project or blind rehash. | Topic packet; review; visuals and code; repository engineering contract. Read other teaching guidance for the decisions being changed. |
| **plan** — “design the lesson” / scoped coverage planning | Produce or improve the design, ownership map and research questions; route justified omissions. An outline is not content complete. Planning does not authorize lesson implementation. | Topic design, teaching, relevant domain; resources as needed. |
| **review** — “verify/check” | Assess the requested content or implementation and record findings against its actual source. Respect read-only scope unless fixes are authorized. Reading source alone cannot certify browser behavior or executed outputs. | Review and the standards governing the reviewed surface. |
| **improve** — “fix/revise these topics” | Repair the authorized gaps; determine which phase and evidence each change affects. Preserve working depth and interfaces. Do not reopen unrelated modules. | Affected teaching/visual/code/resource guidance, review and relevant engineering contract. |
| **resume** — “continue” | Read the current topic checkpoint, actual source identity, unresolved findings and next action. Continue its existing mode, or honor a newly specified mode. | Only missing/changed guidance and affected sources; no automatic replay of passed checks. |
| **cleanup** — “clean completed authoring work” | Retire confirmed disposable artifacts and stale active instructions within the named scope; preserve pending content, cross-topic discoveries and required evidence. No lesson rewrite or completion promotion. | Handoffs/cleanup, selected topic records and repository retention contract. |
| **research-update** — “check significant latest research/models against the curriculum” | Search current primary evidence, assess credibility and learning value, compare actual coverage, and save prioritized update/new-topic recommendations. By default, do not rewrite lessons, add catalogue entries or change completion. If the user also requests those actions, continue in the corresponding authorized mode. | Research updates, relevant domain, existing topic coverage/notes and repository adapter. |

These are invocation choices, not new ledger states. There are still two delivery modes (`full`, `content-first`) and two independently tracked phases (`content`, `implementation`). A finish request continues the prepared revision. Explicit mode persists through its batch and clear continuations. Quality improvements do not authorize crossing the content-only boundary.

## Select the guidance that changes this task

The **writing set** is teaching, topic design, the relevant domain section, visuals, examples/code, and practice/resources. Read these while planning and writing, not only at final review. For a small presentation repair, use the affected guidance and the site contract instead of rereading the entire writing set.

| Reference | Use it for |
| --- | --- |
| [Workflow](references/workflow.md) | Six stages, two-phase checkpoints, scope, sequence, stopping conditions and handoff. |
| [Teaching](references/teaching.md) | Beginner continuity, every conceptual transition, coverage/title decisions, destination notes, depth and reading load. |
| [Topic design](references/topic-design.md) | Outcome/coverage map, implementation owners, hurdle-by-hurdle teaching, complete written and visual contracts. |
| [Domain playbook](references/domain-playbook.md) | Subject-appropriate representations and practice. Read the relevant subject, not every domain by default. |
| [Visuals and labs](references/visuals-and-labs.md) | Inline explanation, distinct live investigations, actual model semantics, quantitative provenance, responsive geometry and meaningful controls. |
| [Examples and code](references/examples-and-code.md) | Real examples, high-quality scratch implementations, exact reuse, ordinary library routes, controlled comparisons and useful applications. |
| [Practice and resources](references/practice-and-resources.md) | Independent transfer, hints/solutions, consistent endings, authoritative research and annotated alternate learning resources. |
| [Review](references/review.md) | Separate correctness and learning-experience review, rendered checks, evidence identity, focused fixes and readiness. |
| [DSA practice](references/dsa-practice.md) | DSA-only curation and verification of official LeetCode practice, staged transfer and local teaching gaps. |
| [Repository adapter](references/portfolio-adapter.md) | This site's commands, ownership, shared components, ledgers and retained project-specific contracts. |
| [Handoffs and cleanup](references/handoffs-and-cleanup.md) | Cross-topic continuity, note disposition, routine completion cleanup, explicit cleanup requests and minimal resumable context. |
| [Research updates](references/research-updates.md) | Current research/model screening, evidence and importance criteria, adoption without popularity bias, curriculum placement and dated recommendations. |

## Preserve these decisions throughout

- **Quality takes priority.** Efficiency is a default, never a cap on necessary research, depth, examples, code quality, review or verification. Investigate a worthwhile teaching improvement or unresolved risk within the authorized scope. Do not measure readiness by word count, lab count or a fixed number of revisions.
- **Teach each new idea.** Establish the question, concrete entities and need; expose intermediate operations; connect the picture, notation and code; interpret the result and a changed case. Revisit this at variants and deeper sections, not just the introduction. Make prerequisites and reuse explicit.
- **Carry the explanation forward.** Connect each section to the problem or limitation that motivates it, and connect each lesson to exact earlier teaching and useful next questions. Preserve useful notation, examples and code contracts across those bridges; explain changes instead of silently switching conventions. Save discoveries for the best teaching owner and close their disposition when used.
- **Choose representations for the mechanism.** Use as many distinct diagrams/labs as materially help. Reuse a format when it remains clear; change it when it conceals the topic. Labels, geometry, data and adjacent claims must agree. Never invent benchmark points or quantitative evidence.
- **Make labs playable immediately.** Show current output and meaningful live controls, linked intermediate states, comparison/reset and decision guidance. No learner prediction entry, commitment, grading or reveal gate, even optional. Model predictions are legitimate subject matter; independent practice stays separate. Static marks must not masquerade as sliders.
- **Teach both implementation routes.** For promised computational outcomes, write the explained high-quality scratch code and ordinary library/tool use during content preparation. Inspect and link exact prerequisite implementations when reusing them. State abstraction, costs and limits; universal optimality is not a meaningful promise. A library link, ownership table or “add code later” is not a completed manuscript.
- **Keep teaching visible and the reader consistent.** Explanations, derivations, diagrams and instructional code are not dropdown content, including deeper teaching. Practice/hints/solutions may retain disclosures. Use the site's shared opening/section index, endings, file index and Copy/Download controls with visible copied/error feedback. Keep domain-specific pedagogy flexible within those common reader affordances.
- **Use research to teach and verify.** Check the canonical treatment for missing important ideas, use authoritative sources for correctness, and examine good teaching resources for better intuition. Offer annotated articles/books/videos where useful. Never imply viewing, execution, peer review or universal competence that the evidence does not establish.
- **Keep evidence honest.** Implementation requires current completed content. Track source, checks, unresolved findings and next action per topic; preserve earlier evidence. Author self-review is not independent review. Documentation changes do not recertify lesson content.

## Coordinate and close the selected increment

For a batch, checkpoint each topic separately so interrupted work can resume. Parallel work is useful only with authorized agent delegation and separable topic/file ownership. Give each worker the exact topic, mode, design, current source and phase boundary; assign one owner for shared manifests/ledgers/integration. Reconcile prerequisite explanations, reuse and sequence at the end. Do not introduce another queue or duplicate manuscripts merely for coordination.

Use the six stages as responsibilities, not six mandatory rewrites or approval pauses. Combine related fixes and reuse unchanged source-bound evidence. Independent review should target complementary teaching/correctness risks; record honestly if unavailable. Broaden checks when a changed dependency, finding or quality concern justifies them.

At each topic/phase handoff, perform the relevant [handoff and cleanup steps](references/handoffs-and-cleanup.md) as part of the authorized work, without a separate cleanup request. Save the complete scoped artifact, note dispositions and precise continuation first. Remove confirmed disposable work and retire completed instructions from the active next-action summary; preserve required manuscripts, provenance and evidence. A standalone cleanup request uses the same rules for its named scope. This is agent-performed housekeeping during a task, not a background deletion service or a way to erase conversation history. Completing one increment does not authorize the next.

## Maintain one instruction source

The versioned package is `skills/lesson-authoring/` in the site repository; the personal skill folder links to it. Edit this package, not copies at former policy paths. Link directly to its relevant references. The repository adapter locates current state and any retained compatibility indexes. Use the selected checkout's package when working in another branch/worktree; do not accidentally edit the installation target in a different checkout.

Repository state and engineering contracts remain in the repository. For another learning project, preserve this teaching/phase method but resolve that project's actual sources, commands, components, theme and ledger contract; the Portfolio paths embedded in the references are adapter examples, not permission to create them everywhere. A guided-project request also needs that repository's project-authoring standard and its separate project checkpoints. General articles or site-shell work do not acquire a lesson workflow merely by using this repository.

Examples: `$lesson-authoring research and write the next three topics only`; `$lesson-authoring finish the prepared topic <title>`; `$lesson-authoring implement <title> end to end`; `$lesson-authoring review <title> without edits`; `$lesson-authoring resume the current topic`; `$lesson-authoring clean completed working material for <topic or batch>`; `$lesson-authoring check significant research updates for <module or model> and recommend curriculum changes`.
