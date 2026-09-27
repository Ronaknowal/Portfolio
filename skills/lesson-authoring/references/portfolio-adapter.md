# Repository adapter: ronak learning site

Use this reference for the ronak site. These are repository contracts, not a second teaching standard. Resolve every repository path against the **selected application checkout** containing `package.json` and `src/learn`. The outer workspace currently contains that checkout in `Portfolio/`; do not assume a personal skill folder or another worktree is the target repository.

## Read the selected state

1. `AGENTS.md` establishes source ownership and scope. `LESSON-AUTHORING-HANDOFF.md` owns current completion, selected evidence and the approved teaching reference. Read its current-state section and the requested topic record; do not reconstruct a queue from historical batches.
2. The user's approved Linux Basics teaching direction remains a reference for depth, approachable explanations and useful visual investigations. Its exact source/review pointers live in the handoff. It is not a universal section template. Later consistency requirements and concept-level intuition apply across subjects.
3. Inspect `docs/teaching/lesson-delivery-progress.json` through the topic preflight. `docs/teaching/LESSON-DELIVERY-LEDGER.md` owns its exact schema, hashes, revision rules and update procedure. Read it before changing phase records. Never introduce competing current ledgers.

From the application root:

```sh
node scripts/build-curriculum-inventory.mjs --topic "Exact title or stable ID"
node scripts/build-curriculum-inventory.mjs --topic "Exact title or stable ID" --work content
node scripts/build-curriculum-inventory.mjs --topic "Exact title or stable ID" --work full
node scripts/build-curriculum-inventory.mjs --topic "Exact title or stable ID" --work finish
```

Select **one** applicable command, not four repeated preflights. These are read-only checks, not status updates. For scoped “next” requests inspect the selected module's catalogue order and effective phase states; apply the finish preflight to each selected prepared topic. Don't select by difficulty, published availability, stale summary counts or the currently open browser page alone. If all requested-module topics are complete, report that instead of silently moving to another module.

Distinguish never-started work from previously completed historical rows whose source bindings are stale. Staleness needs scoped assessment; it does not automatically make those topics a fresh “next topics” queue. Use the user's continuation context and report or clarify the actual scope when the module has no unstarted work. Do not reopen completed topics solely because older evidence uses another source revision.

Use `topic.delivery` (including effective vs recorded status and `canFinish`), linked source record, `authoringNotes`, `implementationDepth`, individual blueprint and domain guidance. Read `docs/teaching/topic-notes/README.md` before adding/routing a discovery. Existing notes are inputs to consider with reasons, not blindly accepted instructions. Preserve stable IDs/URLs/progress; propose scoped title/order changes with a rationale.

## Put outputs with their real owners

| Artifact | Owner / handling |
| --- | --- |
| Full content-first manuscript | `docs/teaching/drafts/<stable-topic-id>/lesson.md` |
| Its actionable visual/lab specs | `docs/teaching/drafts/<stable-topic-id>/visual-specifications.md` |
| Design, sources, learning map, actual checks and continuation | Existing topic/increment record, linked from the central ledger; don't duplicate into a parallel report per edit. |
| Destination discoveries | `docs/teaching/topic-notes/<stable-topic-id>.md`, or the established routing inbox when no owner is resolved. |
| Topic runtime / examples / models | Follow actual ownership in `docs/engineering/LEARNING-CODE-STANDARD.md` and existing semantic topic paths. |
| Published topic registration | `src/learn/data/lesson-manifest.json` and its generated metadata; only in authorized implementation. |
| Phase status | Central delivery ledger, preserving prior revisions and exact reviewed files. |

Full-mode work may use complete semantic production source as its content checkpoint; it need not duplicate the manuscript. Content-first work must leave published bodies, navigation and runtime untouched. A file explaining what a future agent should write is not a complete content checkpoint. Preserve prepared code, data, specs and research until consumed and retained appropriately.

## Load engineering guidance when implementing or reviewing runtime

Read `docs/engineering/LEARNING-CODE-STANDARD.md` for semantic filenames/variables, early import/component compatibility, explicit publication, lazy loading, real production dependency checks, browser verification and cleanup. Its sections also own:

- **Shared lesson opening and section navigation:** one Learning compass directly below the title, actual section jump links and shared numbering; no competing custom TOCs.
- **Shared lesson endings:** separate practice, optional feedback, further learning, technical references and next-step roles with consistent shared presentation.
- **Visible teaching and code access:** shared `Code`, `RemoteCodeBlock` and `LessonCodeDownloads`; explicit teaching metadata; displayed/downloaded source agreement; real filenames, excerpt labeling and answer-only asset boundaries. No code-opening gate. Large source may load near the viewport.
- **Copy feedback:** immediate temporary amber copied state/checkmark, stable control width, screen-reader status, honest local error feedback, repeated-click timer restart and unmount cleanup. Use the existing component; don't reimplement feedback for each topic.
- **Diagram layout and SVG legibility:** inspect actual painted labels/marks, viewport/text-zoom behavior, intermediate diagram states and truthful model semantics. The whole SVG's bounds alone do not prove legibility.

Use `docs/engineering/LESSON-VERIFICATION-PITFALLS.md` when planning numerical/visual verification or investigating a relevant regression. Use `docs/engineering/LEARNING-WORKSPACE.md` for learning routes/projects/discovery; `docs/engineering/REPOSITORY-STRUCTURE.md` and `SITE-ARCHITECTURE.md` in the same directory for wider ownership. `.impeccable.md` holds design context. Keep the neutral black/charcoal and amber theme; avoid unrequested green/olive decoration, browser-default grey controls and default blue links.

Read only relevant evidence. Historical code-access/opening/ending/scientific receipts remain records of the source they examined; later presentation receipts may own newer shared files. Follow the latest selected record from the handoff/ledger. Do not rerun historical integration tools or refresh old hashes to make an instruction-only migration pass.

## Load specialty context only when needed

- Curriculum scope/order questions: `LEARNING-CURRICULUM-PLAN.md` and the applicable plan under `docs/curriculum/`. Preserve listed topics; no finite map promises every future fact or job/interview outcome.
- DSA authoring: skill `dsa-practice.md` plus `docs/teaching/DSA-PRACTICE-STANDARD.md`, which retains the evolving site coverage map and dated evidence. Topic-specific links/data remain lazy.
- Quantum: `docs/curriculum/QUANTUM-COMPUTING-PLAN.md` and its teaching contract. Other specialist modules similarly use their named plan from preflight.
- Guided build projects: `PROJECT-AUTHORING-STANDARD.md`, the actual project sources and separate project ledger. Preserve the approved Typed Decision Model depth reference; do not turn project stages into lesson phase rows.

## Scope checks and handoff honestly

Follow the chosen operation and six-stage workflow. Content-first includes claim/reasoning checks but defers rendered/native integration certification. For implementation, complete the topic's meaningful native/model checks, independent review, actual browser/keyboard/narrow rendering and required final integration. Use the repository's existing scoped commands rather than a generic test list. A file containing a verifier is not proof it ran.

Use `npm run check:repository` after repository-structure or ignore changes. A documentation/skill-only edit needs instruction/link/contract checks, not a new production build or browser campaign. Keep learner-facing code quality intact when fixing a verifier. Reuse passing results only for applicable unchanged source, assumptions and environment.

Before stopping, update the selected topic record/phase only if work actually advanced it. An instruction migration does not change topic status. Record the current revision, actual checks, outstanding findings, source identity and one precise next action. Keep those facts out of the reusable skill. Remove only your own disposable scratch after preserving needed inputs and final evidence.

Follow [handoffs and cleanup](handoffs-and-cleanup.md) at each substantive delivery boundary and for the explicit cleanup mode. `docs/teaching/topic-notes/README.md` owns note statuses and prepared-versus-implemented dispositions; `docs/engineering/WORKING-ARTIFACT-RETENTION.md` owns exact repository retention and safe-removal checks. Record closure/removals in the existing scoped record, keep resolved notes out of the active task list, and preserve every pending packet and still-required historical input. Cleanup does not promote either authoring phase or authorize a content rewrite.

## Research-update records

The `research-update` mode uses [the research-update criteria](research-updates.md). Reuse an existing scoped design/research record if it already owns the assessment. For a new scoped assessment, save one current review at `docs/teaching/research-updates/<scope-id>/review.md`, with a semantic module/topic/mechanism scope name, checked date/window, coverage, evidence, decisions and continuation. Create it only for an actual assessment, not when merely installing the skill. Retain material prior decisions compactly in the same record; source-versioned evidence stays at its recorded location. This is research provenance, not a competing completion ledger.

Persist actionable changes in `docs/teaching/topic-notes/<stable-topic-id>.md` and link the assessment. Proposals without an existing owner use `docs/teaching/topic-notes/UNASSIGNED.md` with suggested module/prerequisites and ownership rationale. Use the existing statuses: a recommendation is open, a watch item is deferred with a specific trigger, and implemented/adapted requires the actual content and checks. Don't create an empty topic file or fabricate a catalogue ID for a proposed lesson.

An assessment alone leaves `lesson-delivery-progress.json`, published content, pending manuscript hashes and catalogue membership unchanged. Apply authorized additions under curriculum planning and authorized writing/implementation under their normal phase contract. If prepared and published versions differ, record which needs the change and why. Preserve stable routes and the module's learning order.

## Installation and maintenance

The canonical versioned package is `skills/lesson-authoring/`. Personal discovery links `C:/Users/ronak/.agents/skills/lesson-authoring` to that directory. Do not create another separately maintained copy or a second repo-discovery entry with the same name. This keeps changes reviewable in Git without duplicate skill selectors. The personal link depends on this checkout remaining at its current path; recreate it deliberately when moving the repository.

Future sessions use `$lesson-authoring` or ordinary matching requests. If discovery has not refreshed, start a new session/restart the app, or read the repository's `skills/lesson-authoring/SKILL.md` directly. The old standard, design brief and playbook paths are forwarding indexes for saved links. Maintain detailed teaching rules in this skill; maintain site engineering contracts and live evidence in their existing repo owners. Read `docs/teaching/skill-migration.json` only to investigate the relocation, not for ordinary authoring.
