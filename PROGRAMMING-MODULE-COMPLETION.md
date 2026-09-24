# Programming module completion and sequence repair

Completed 10 September 2026 for the user's request: make reading follow topic order inside each module, then implement all remaining Programming & Scientific Computing lessons. This is an implementation/evidence record. [The handoff](LESSON-AUTHORING-HANDOFF.md) owns current scope and review status; [the standard](LESSON-TEACHING-STANDARD.md) owns policy.

## Result

All **17 programming topics** now have complete lessons under the current teaching approach. This increment reimplemented six older lessons and completed Threads, previously planned. The seven contain **21 focused investigations and 60 complete executable programs**, with self-contained explanations, independent practice, hints, solutions, useful applications and annotated written/video alternatives. Existing useful content was preserved, not replaced with generic outlines. All seven titles remain appropriate and unchanged.

| Topic | What changed and why | Design and evidence |
| --- | --- | --- |
| Iterators, Iterables & Generators | Replaced a generic trace with cursor ownership, suspended-frame and pull-pipeline investigations; retained iterator/generator depth and added tee-buffering/first-crossing applications. Fourteen programs. | [Iteration/decorator record](docs/teaching/iteration-decorators-design.md) |
| Decorators & Context Managers | Made wrapper call order, normal/error resource lifetime and ExitStack unwinding inspectable; retained complete examples and added independent nested-setting restoration. Eleven programs. | [Iteration/decorator record](docs/teaching/iteration-decorators-design.md) |
| Testing, Debugging & Dependency Management | Added controlled execution, test discrimination and dependency-constraint investigations; deliberately broken implementations must fail the independent tests. Six programs. | [Reliability record](docs/teaching/reliability-authoring-design.md) |
| Reproducible Notebooks & Experiment Structure | Separated document history from live kernel state, randomness consumption and provenance; actual fresh kernels verify reruns and reject hidden/stale assumptions. Seven programs/projects. | [Reliability record](docs/teaching/reliability-authoring-design.md) |
| Code Documentation, Type Hints & API Design | Made validation, alias ownership and caller compatibility visible; preserved complete API/loader examples and added checked duration conversion contracts. Eight programs. | [Reliability record](docs/teaching/reliability-authoring-design.md) |
| Bash Scripting & Command-Line Automation | Replaced the short strict-mode claim and missing train.py with exact argument/status explanations, three investigations and a complete staged report/collector. Eight executable programs plus data fixtures. | [Bash evidence](docs/teaching/bash-completion-verification.md), [initial design](docs/teaching/bash-and-concurrency-design.md) |
| Threads, Concurrency, Locks & Deadlocks | Completed shared-state mechanisms, whole-operation locking, predicate waiting, wait-for cycles, result/error collection and bounded shutdown. Three investigations and six programs, including an independent sensor-validator queue. | [Design/source ledger](docs/teaching/bash-and-concurrency-design.md) and verification below |

Three labs per lesson happened to fit these distinct hurdles. It is not a site template or quota. Static maps, tables and stepwise explanations accompany them. Videos are linked/annotated alternatives; the records distinguish transcript/notes review, metadata/description review and unperformed full-video viewing.

## Sequence repair

The old route globally sorted difficulty and recursively inserted individual prerequisites. The sidebar then regrouped those lessons by module, so Next could leave one module even while its contents list showed more topics. Publication had already been removed as a sequencing rule; the remaining problem was the mismatch between those two orders.

The catalogue now explicitly orders programming as:

1. Python → OOP → Iterators → Decorators → Testing.
2. NumPy → File Formats → SQL → Pandas → Matplotlib → Notebooks → Code Documentation.
3. Git → Linux → Bash.
4. OS Processes → Threads.

These are four sections within one 17-topic module. File Formats and SQL retain their IDs and are placed after NumPy, before Pandas. Other modules keep their catalogue topic order. Paths visit selected modules in declared order and selected topics in that syllabus order. Additional prerequisite modules follow in catalogue order; their lessons remain available through readiness links. This does not claim all cross-module dependency edges occur earlier. No automatic difficulty or prerequisite detour rewrites the reader's next step.

`getLearningRoute` supplies both module navigation and reading steps. Shared lessons retain one unique progress ID and all relevant module memberships. A `module` query parameter preserves the chosen occurrence, so Next/Previous works locally even after reloading a shared topic. Invalid/missing module context falls back to a valid occurrence. Unique path totals do not double-count shared lessons; the header shows local module position. Entry URLs remain stable when completion is toggled. Planned topics remain ordinary sequence entries and can be the first unfinished lesson.

Updated closing bridges in Python, OOP, SQL, Matplotlib, Linux, OS and Linked Lists agree with the new module order. Linked Lists now points to planned Trees, not Algebra. The curriculum plan, standard, design brief, AGENTS and handoff replace the superseded ordering rules. Historical increment reports retain their dated evidence. The old reader-sequence test command delegates to the new maintained module-order review, and earlier lesson reviewers no longer assert obsolete global batch positions.

## Conservation and current inventory

The pre-change snapshot is [pre-programming-module-completion.json](docs/curriculum/pre-programming-module-completion.json). `node scripts/verify-programming-module-conservation.mjs` passed:

- All **1,218 stable topic IDs**, all **28 module memberships** and all previously selected path topics remain.
- Only the programming module's own syllabus order changed; global routes now follow module order deliberately.
- LLM Engineer adds Iterators/Decorators, and GPU Engineering adds OOP/Iterators/Decorators through explicit reviewed Threads prerequisites. No prior topic is removed from either path.
- All 17 programming entries are published, individually planned and have an explicit review status.

The regenerated inventory records **193 published lessons, 289 individual briefs, 339 recorded prerequisite reviews, 28 modules and seven paths**. **929 other topics still need individual design.** Published does not mean approved; planned neural/GPU coverage is not implemented instruction.

## Verification actually completed

| Check | Evidence |
| --- | --- |
| Curriculum | `node scripts/verify-curriculum.mjs` passes IDs, exact prerequisites, acyclic graph, supporting membership, canonical module order, shared memberships and all 1,002 pre-expansion IDs. |
| Conservation | `node scripts/verify-programming-module-conservation.mjs` passes the complete 1,218-topic snapshot and the deliberate path additions. |
| Inventory | `node scripts/build-curriculum-inventory.mjs` regenerated Markdown/JSON from live definitions, with module rows in actual syllabus order and separate review/publication status. |
| Route browser | `node scripts/review-module-order.cjs` passes at 1440/390px: all 17 actual Next steps, matching sidebar order, all seven path counts/entries, cross-module transition, shared-topic reload/back/next/progress, planned resume, completion stability, prerequisite links and no page errors/overflow. Evidence: scratch/module-order-review/results.json. |
| Iterators/Decorators native | Python 3.12.14 runs all 25 displayed programs; 27 model configurations, 341 changed consumption cases, 78 batch cases, 16 restoration cases and wrapper error/metadata contracts pass. |
| Reliability native | Python 3.12.14 runs 21 displayed programs and actual fresh notebook kernels; 340 independent selections, rejected mutants, ownership and compatibility checks, unittest discovery, doctests and strict mypy acceptance/rejection pass. Versions and latest run are in the reliability record. |
| Bash native | Ubuntu WSL as ordinary nobody/UID65534, Bash 5.2.21 and Python 3.12.3: five exact script outputs, 16 model configurations, schemas, invalid/empty/missing inputs, repeated runs, argument boundaries, status/errexit probes, staged/partial failure preservation, rename visibility and cleanup of owned fixtures/process group after TERM pass. |
| Threads native/models | `node scripts/verify-thread-completion.mjs` passes six exact native program outputs on Python 3.12.14, 18 changed queue cases with 1/2/3 workers and no unfinished items/leaked workers, and 73 reachable abstract race/lock states. Lost update and deadlock are witnessed; protected/ordered terminal states preserve their invariants. All subprocesses have verification timeouts; no unbounded native deadlock is launched. |
| Lesson browser/visual | The four lesson review scripts cover all 21 investigations at desktop/390px, all relevant controls/steps, reset/back, keyboard/focus, disclosures, anchors, rendered executable content, references and overflow. All mobile labs and representative desktop views were visually inspected. Independent Threads review enlarged graph labels and touch targets. Records/screenshots reside in the respective scratch/*-review directories. |
| Build | `node node_modules/vite/bin/vite.js build` passes. Log: scratch/programming-module-build.log. Existing Bayesian-network JSX greater-than warnings and the large content-bundle warning remain; neither originated in these lesson implementations. |

Updated earlier browser/conservation scripts were syntax-checked after removing obsolete sequence assumptions. Their historical full lesson suites were not all rerun: their unchanged content/model evidence remains dated, while the new route review exercised the changed navigation and closing-order integration. Do not describe those historical runs as fresh full-suite verification.

## Discoveries and boundaries

The Threads instrument example distinguishes blocking/backpressure, every-sample recording and latest-sample display policies. Its detailed hardware/timing treatment belongs to the already planned neural firmware topic. [The destination note](docs/teaching/topic-notes/embedded-processing-fpga-pipelines-and-real-time-neural-firmware.md) records the reason, proposed comparison and source checks the future author should perform. It adds no new topic or authorization. Existing ML leakage, Hashing and Caching notes remain available.

The complete module teaches its stated core and transfer tasks, not every detail of software engineering. Thread labs are abstract schedules, not bytecode simulators or performance benchmarks. Ordinary CPython was executed; free-threading differences were researched, not benchmarked. Bash staging tests concern trusted local single-publisher fixtures, not arbitrary network filesystems or power-loss durability. GNU manual pages timed out in the web reader; indexed identification and installed Bash help/probes are recorded honestly. No full-video watching, external learner study, deployment or new user approval is claimed.

**Status:** implementation, native/model verification, browser/visual review and build passed; **user acceptance pending**. Current authorization is fulfilled. The next unimplemented DSA topic is Trees & Binary Search Trees if the user requests continuation. Do not restart the completed seven or resume an archived Algebra queue.
