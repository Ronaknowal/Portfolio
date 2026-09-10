# Educational authoring: current handoff

Updated 10 September 2026 for the active goal to implement all remaining DSA topics and all Mathematical & Statistical Foundations topics. This is the entry point for future sessions. The user's current instructions control scope; historical batch numbers do not define the reading order or the next authoring task.

## Read these in order

1. [Teaching standard](LESSON-TEACHING-STANDARD.md): authoritative teaching policy, depth, source research, visuals, practice, validation and publishing rules.
2. [Domain playbook](docs/teaching/DOMAIN-PLAYBOOK.md): subject-specific strategies for programming, maths, DSA, AI, systems, robotics and neurotechnology. These are adaptable, not identical article templates.
3. [Topic design brief](docs/teaching/TOPIC-DESIGN-BRIEF.md): prepare outcomes, conceptual hurdles, examples, representation contracts, practice and evidence before authoring.
4. [Curriculum plan](LEARNING-CURRICULUM-PLAN.md) and relevant specialist research plan under docs/curriculum. The generated inventory distinguishes a completed lesson, an individual starting brief and a topic still needing design.
5. Actual lesson source, models, labs, runnable fixtures and checks. Run `node scripts/build-curriculum-inventory.mjs --topic "Exact topic title or stable ID"` and read its returned authoring notes, including the destination Markdown and relevant unresolved inbox entries.
6. [Learning code standard](docs/engineering/LEARNING-CODE-STANDARD.md): topic/purpose-based files and identifiers, source ownership, manifest registration, generated compact metadata, lazy loading and performance/recovery validation.
7. For DSA, [practice standard](docs/teaching/DSA-PRACTICE-STANDARD.md): verified official LeetCode statements, staged hints/transfer, readiness and evolving pattern coverage. The first sets are examples, not a fixed-count template.

This file owns queue/status; the standard owns policy; domain plans own coverage rationale; code owns implemented behavior. Archived pilot/batch records are evidence of their dated increments, not competing instructions.

## Current scope and status

**Active goal:** implement all **17 remaining DSA topics and all57 Mathematical & Statistical Foundations topics**, including review of the42 previously published maths lessons and completion of its15 planned topics. Read [the implementation record](DSA-MATH-FOUNDATIONS-IMPLEMENTATION.md) and [exact74-topic ledger](docs/teaching/dsa-math-foundations-progress.json). **49 scoped topics are implementation-reviewed: all DSA positions6–22 and mathematics positions1–32.** Together with the preserved first five DSA lessons, the full22-topic DSA module has reviewed implementations and guided practice. The latest completed four are f-Divergences/IPMs, Graph Fundamentals, Spectral Graph Theory and Combinatorial Optimization. Twenty-five mathematics topics still require final review. Stochastic Processes, Random Matrix Theory, Queueing Theory and Dynamical Systems Theory & Chaos are entering scoped design and implementation. Publication, author review, integrated review and user acceptance remain separate. The full goal stays active until every scoped topic and required integration check is complete.

The subsequent engineering pass preserved these lesson improvements, including both DSA topics. Authored plans now have individual stable-ID files; larger cross-topic runtime bundles were split by responsibility. Publication is registered in `src/learn/data/lesson-manifest.json`; navigation uses generated metadata and lesson bodies/outlines load on demand. The old eager `topics/index.js` registry and batch-named production bundles are removed. Read the learning code standard and its linked migration evidence before adding another lesson; this source refactor does not authorize more lesson rewrites or change topic order.

[Integrated engineering verification](docs/engineering/LEARNING-LOADING-REVIEW.md) passed: exact curriculum/publication conservation, source/model checks, production desktop/mobile teaching and route checks, and eight loading/error cases. Measured compressed JavaScript decreased 95.0% for the hub and 93.8% for Python. These are local payload measurements, not universal load-time promises. The new standard documents semantic ownership, registration/generation and the browser's retry/reload limits end to end.

The preceding visual pass reviewed **22 previously improved/reference topics**: all 17 programming lessons, Arrays and Linked Lists, and the three original pilots. **21 received targeted visual or interaction improvements; Linux's effective existing representations were retained.** Read [the complete visual review](VISUAL-TEACHING-REVIEW.md) and its four linked domain records for per-topic decisions, current representations, verification and limits. These changes are implemented, not merely proposals in the teaching standard. User acceptance remains pending.

The preceding completed increment implemented **Trees & Binary Search Trees → Heaps, Priority Queues & Tries → Graphs: Representations, BFS & DFS**, in that existing module order. Each has an individual design, concept-specific visual investigations, complete Python programs, local independent practice and annotated written/video resources. Guided LeetCode practice was also added to Arrays and Linked Lists while retaining their explanations and visuals. [The DSA implementation record](DSA-CORE-STRUCTURES-IMPLEMENTATION.md) owns actual per-topic/native/model/browser/production evidence and limits. No other lesson or curriculum order was rewritten by that increment.

The programming module remains complete: **all 17 Programming & Scientific Computing topics are published with the current approach and individual briefs**. The earlier module-completion increment reimplemented Iterators, Decorators, Testing, Notebooks, Code Documentation and Bash, and completed the previously planned Threads lesson. Those seven had **21 focused investigations and 60 complete executable programs**, preserved useful earlier coverage, independent exercises/solutions and annotated written/video resources. Counts describe that increment, not future quotas or the current visual inventory.

Implementation, native/model verification and desktop/mobile/keyboard review passed. **User acceptance remains pending for these rewrites. Linux is still the only explicitly user-approved current quality reference.** A registered lesson, passing build, source ledger or author review does not imply user approval or an observed beginner study.

| Work | Evidence and review status |
| --- | --- |
| Linux Basics, Filesystems & Processes | Approved reference from 9 September. Four focused labs and complete native practice: [Linux record](PROGRAMMING-REWRITE-LINUX.md). Core teaching retained; its current closing bridge is Bash, then OS. |
| Original hypothesis-testing, Bayesian and spectral-graph pilots | Retained their effective distribution/spectrum labs; added interval-decision, evidence-update and local Laplacian-row views in [the current visual review](docs/teaching/SYSTEMS-PILOTS-VISUAL-REVIEW.md). Historical comparison is in docs/archive/lesson-rollout; they are not the final quality ceiling. |
| Python, NumPy, File Formats, SQL, OOP | Implemented and author-verified in [the first-five record](FIRST-FIVE-REIMPLEMENTATION.md). Its batch order is historical. User acceptance pending. |
| Pandas, Matplotlib, Git | Implemented and author-verified in [the next-three record](NEXT-THREE-REIMPLEMENTATION.md). User acceptance pending. |
| OS, Arrays, Linked Lists | Implemented and author-verified in [the systems/structures record](SYSTEMS-STRUCTURES-IMPLEMENTATION.md). Retain DSA work. User acceptance pending. |
| Iterators, Decorators, Testing, Notebooks, Code Documentation, Bash, Threads | Completed and verified in [the module-completion record](PROGRAMMING-MODULE-COMPLETION.md), with individual design/source/native/browser records. User acceptance pending. |
| Trees, Heaps/Tries, Graphs; practice in Arrays and Linked Lists | Implemented in [the DSA core-structures record](DSA-CORE-STRUCTURES-IMPLEMENTATION.md), with separate design, model and native/browser evidence. Fifty official problem links currently span the five lessons; the count is not a quota or mastery guarantee. User acceptance pending. |
| Remaining DSA and all Mathematical & Statistical Foundations | Active authorized 74-topic goal in [the implementation record](DSA-MATH-FOUNDATIONS-IMPLEMENTATION.md). Existing publication is not final teaching review. |
| Other website modules | Outside the active rollout. Preserve their existing work and record useful destination discoveries. |

There are **1,218 unique topics,28 modules,seven paths,213 published lessons,331 individual briefs and361 recorded prerequisite reviews** in the latest integrated snapshot. **887 topics still need individual design.** All identities and memberships are conserved. The latest integrated record covers49 reviewed scoped topics and62 production loading/recovery cases. Subsequent in-progress designs can change the live brief count; the inventory and source-versioned ledger own exact current state.

## The reading order is the module order

The old algorithm globally sorted difficulty and inserted individual prerequisites, while the sidebar regrouped topics by module. That produced jumps. It has been replaced. The catalogue's actual section/topic order now owns progression; a path visits its selected modules in declared order and keeps each module's selected topics in syllabus order.

Programming & Scientific Computing now has four coherent sections:

| Section | Topic order |
| --- | --- |
| Python Foundations | Python Basics → OOP → Iterators → Decorators → Testing |
| Scientific Python & Data Workflows | NumPy → Scientific File Formats → SQL → Pandas → Matplotlib → Reproducible Notebooks → Code Documentation |
| Developer Workflow | Git → Linux → Bash |
| Operating-System & Concurrent Programming Foundations | OS Processes → Threads |

File Formats and SQL were moved into the scientific workflow after NumPy; no topics were deleted. Other modules retain their catalogue topic order. This is an explicit repair to navigation and the programming syllabus, not a certification that every older module has been pedagogically audited.

Recorded dependencies still add supporting topics and appear as review links. Additional prerequisite modules follow selected modules in catalogue order. Do not promise all cross-module prerequisites precede dependents: contiguous module reading and the dependency graph are different constraints. Explain/link readiness requirements rather than silently rearranging topics. Difficulty and publication status never reshuffle reading order.

The sidebar and Previous/Next use the same module contents, including planned entries. Shared lessons retain every relevant membership and one completion record; the URL's module context keeps navigation in the selected module after reload. Headings show position within that module's selected topics. Topic/completed counts share scope and remain together in the module header; focused selections link to the full module. Hub counts use the resolved route, including supporting modules. Entry URLs resolve to a stable lesson, so marking it complete does not silently advance it. Closing lesson bridges were corrected where the new order changed them.

**Active authoring locations:** DSA's22-topic module is implemented and reviewed. Mathematics continues with **Stochastic Processes** (33), **Random Matrix Theory** (34), **Queueing Theory** (35), then **Dynamical Systems Theory & Chaos** (36). Preserve each module's actual order while independent work proceeds in parallel. Published older mathematics bodies still require actual scoped review. Read each topic's notes and the DSA practice map; archived queues do not define authorization.

## What the approved Linux reference establishes

- Start from a concrete problem and introduce each entity before using its terminology or syntax.
- Make hidden mechanisms visible: state, relationships, transformations, ownership, failure gates and resulting consequences.
- Ask for a prediction, let the learner change meaningful inputs, explain the result and test a changed situation.
- Keep essential reasoning in the first-pass route; use progressive disclosure for additional depth, not missing core explanations.
- Use as many distinct diagrams/labs as the actual conceptual hurdles need. A static diagram is sufficient when interaction adds no learning value.
- Supply complete native examples with inputs, setup, expected outputs and interpretations. Browser models have explicit boundaries and independent runtime/oracle checks.
- Give independent practice, hints and explained solutions with acceptance cases. A worked example copied unchanged is not independent practice.

More prose, decorative cards or a generic slider alone do not meet the reference. StructuredLesson's legacy headings/single-example props are not the new default. LessonGuide supplies reading advice, not absent teaching content. The current seven-topic records demonstrate iterator frames, wrapper lifetimes, debugger state, notebook/kernel separation, API ownership, shell argument/status/publication and thread coordination. Their arrangements are examples, not a compulsory template for other subjects.

### Follow-up feedback: inline diagrams and concept-specific labs

On 10 September the user highlighted the tokenization lesson's compact same-input comparison and asked why that kind of diagram is less evident in programming. Static figures were already allowed and some exist (Python's module map and NumPy's labelled data grid), but the inspected programming lessons rely heavily on lab panels, code and tables. Completed labs do not establish that inline visual teaching is sufficient. This is a presentation/teaching gap to assess, not a technical limitation of the component system.

The standard's “Inline diagrams are part of the explanation” guidance, design brief and programming playbook explicitly require considering figures at the point of need and reviewing ordinary reading flow separately from lab interaction. Reuse the user's example as a presentation pattern only, not as an accuracy endorsement of the entire tokenization lesson. Keep subject-appropriate representations, no diagram/lab quota, and no redundant figure when an existing visible representation suffices. The initial documentation-only update did not retrofit lessons; the subsequent authorized 22-topic pass now applies this guidance as recorded in the current visual review. Prior evidence remains valid within its scope, with fresh checks recorded for this increment.

The user further highlighted the BPE merge walkthrough, editable trainer and plotted comparison. Their clarification is that illustration style, terminology and interaction should fit the topic and the particular learning need, with variation where useful and reuse where the same form works well. The standard covers this choice and graph provenance; the brief requires a learning rationale and model/data evidence, and the domain playbook gives adaptable examples. Consistency means dependable teaching and usable controls, not identical lab layouts. The latest implementation includes concrete reference maps, byte/record/argument boundaries, coordinate inspections, worker lanes, repository boundaries and mathematical views chosen for distinct questions. Preserve their learning jobs while remaining free to improve the form for a new need.

The highlighted BPE timing chart is labelled illustrative and contains literal point arrays; this review did not establish benchmark provenance. Nearby claims of stable implementation rankings therefore remain unverified. A [destination note for tokenization](docs/teaching/topic-notes/byte-pair-encoding-bpe-wordpiece-sentencepiece-unigram.md) records the exact concern and required evidence or reframing for its next scoped revision. Do not reuse that chart as a verified benchmark or interpret the user's presentation preference as technical approval.

## Continuous scope, title and resource review

Before writing, during discoveries and before delivery, reconsider coverage and title. Include valuable related material when this topic is its best teaching home. Rename only when the new scope warrants it, preserving stable identity across catalogue consumers, registry, prerequisites, links, progress and notes. Do not append every application to a title.

For an idea better taught elsewhere, search plausible existing owners and save a reasoned [destination-topic note](docs/teaching/topic-notes/README.md). A future author receives it through the topic-plan command, reassesses its evidence and records inclusion, adaptation, rerouting, reasoned deferral or rejection. This is scoped discovery, not a requirement to audit the whole catalogue before every lesson.

Useful familiar and less familiar applications have no quota. Include them when they explain a mechanism or enable transfer, with enough detail to follow the connection. Sources should improve original teaching rather than dictate copied wording or structure. Curate worthwhile articles, documentation, videos and playlists in “References & another way to learn it,” with direct links and concept/level/version annotations. Record what was actually reviewed; a verified video title or description is not full-video viewing.

Open destination notes include ML data availability/leakage, Hashing resize thresholds, Caching validity/LRU, [neural firmware buffer policies](docs/teaching/topic-notes/embedded-processing-fpga-pipelines-and-real-time-neural-firmware.md), [the tokenization visual/timing evidence concern](docs/teaching/topic-notes/byte-pair-encoding-bpe-wordpiece-sentencepiece-unigram.md), and [weighted graph frontier/finalization and directed-state reasoning](docs/teaching/topic-notes/shortest-paths-spanning-trees-topological-ordering.md). The firmware note refines an already planned owner with blocking/dropping/coalescing policies and timing boundaries. These notes create no new rollout authorization.

The [routing inbox](docs/teaching/topic-notes/UNASSIGNED.md) now records an unresolved DSA teaching owner for bit manipulation, discovered during practice coverage planning. The scoped catalogue search did not establish a suitable owner; no topic was added and this is not a claim that every existing lesson was searched. Resolve it in a relevant future scoped task, with language-specific evidence and prerequisites.

## Resume and verification workflow

Inspect scope/worktree → retrieve topic plan and saved notes → review coverage/title/ownership → map outcomes and hurdles → research nuanced/current claims → design concept-specific representations and examples → implement complete teaching and independent practice → revisit discoveries → verify native/model behavior and rendered experience → update notes, brief, evidence and handoff → deliver the completed authorized increment.

For curriculum work run `node scripts/verify-curriculum.mjs`, regenerate the inventory and build. Current route checks are `scripts/review-module-order.cjs`; the historical reader-sequence entry point delegates there. The latest DSA increment uses `scripts/verify-dsa-publication.mjs` to preserve its 193 prior publication mappings and add exactly three; the earlier programming conservation script is dated evidence of its own increment. Individual lesson/native/browser commands and actual runtime versions are in the linked evidence records. Production loading checks are `scripts/verify-learning-load-boundaries.cjs`. Do not describe old batch position assertions as current policy. Do not deploy unless requested.

Make routine implementation decisions autonomously within the user's authorized scope. Preserve unrelated work. Do not mark a lesson ready on the strength of a build, a lab count or a generated plan. Record actual evidence and material limitations, and keep implementation, native verification, browser/visual review and user acceptance separate.
