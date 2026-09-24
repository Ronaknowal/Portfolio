# Learning curriculum plan

**18 September 2026 quantum expansion:** the [quantum computing plan](docs/curriculum/QUANTUM-COMPUTING-PLAN.md) and [ordered syllabus](docs/curriculum/QUANTUM-COMPUTING-SYLLABUS.md) broaden the existing quantum module to 106 topics in 12 sections and add a dedicated guided path. All 48 earlier quantum topics and all other catalogue topics remain. Current totals are **29 modules, 1,460 unique topics and 10 guided paths**. Named coverage includes information, algorithms, physical engineering, fault tolerance, networks, cryptography and sensing. This is catalogue planning only; earlier numerical paragraphs below are dated snapshots.

**17 September 2026 expansion and coverage follow-up:** [professional trading and system design](docs/curriculum/PROFESSIONAL-TRADING-SYSTEM-DESIGN-PLAN.md) is the current specialist coverage supplement. The complete catalogue now has **29 modules, 1,402 unique topics and 9 guided paths**. Quantitative Trading expands the existing finance module to 117 topics; System Design adds one coherent 109-topic module with shared foundations. The [named-concept review](docs/curriculum/PROFESSIONAL-COVERAGE-REVIEW.md) adds dedicated algorithm depth and makes concept ownership searchable. The detailed ordered syllabi, scope research, role routes and checks are linked there. Older numeric status statements below describe their dated increments; use the generated inventory for current counts. No lesson or delivery checkpoint was completed by this expansion.

Updated 10 September 2026. This is the current curriculum architecture and coverage plan. [The teaching standard](LESSON-TEACHING-STANDARD.md) owns policy; [the handoff](LESSON-AUTHORING-HANDOFF.md) owns scope and acceptance. Earlier blanket claims that areas were already strong, that there were only five paths, or that all prerequisites were known are superseded.

## Outcome and limits of coverage

Build a connected route from foundations to independent application, professional engineering and deliberate specialist/research branches. Preserve the approved Linux lesson's quality while changing the teaching approach for each subject. A topic title is not a teachable lesson.

“End to end” means mapped foundations, mechanisms, tools, workflows, failures, evidence and practical projects for a stated field. No finite catalogue guarantees every fact or every professional skill. Supervised laboratory/clinical experience, hardware access and continuing research remain necessary where applicable. Newly discovered omissions should become sequenced topics or explicit scoped extensions.

The curriculum scope review checks primary references and the catalogue; it does not fact-check every sentence in every lesson. There are now 196 registered published lessons after completing Programming & Scientific Computing and the next three DSA topics, with separate topic-level evidence in [the module-completion record](PROGRAMMING-MODULE-COMPLETION.md) and [DSA record](DSA-CORE-STRUCTURES-IMPLEMENTATION.md). Other older lessons still need individual teaching and accuracy review.

## Implemented architecture

Dedicated **Neural Engineering** and **GPU Engineering** paths reuse shared topics. The existing neuroscience and hardware modules are expanded; stable module IDs remain `computational-neuroscience` and `hardware-systems`. Other modules gain missing foundations and specialist bridges, especially maths, DSA, data/evaluation practice and robotics.

Every added topic and every original neural/GPU topic has an individual starting brief: purpose, outcomes, exact prerequisites, concrete sequence, proposed investigation, independent practice and success criteria, misconceptions, source URLs, depth and review focus. The accepted Linux design is recorded, including its additional representations. All 17 programming topics and the first five DSA lessons have briefs tied to actual teaching and evidence. There are 292 briefs in total; 926 topics still need individual design. A brief alone does not mean a published lesson.

Detailed briefs appear on planned lesson pages. Their explanations, activities and solutions are explicitly not yet available; planned topics cannot be marked as completed lessons. Reading order follows the module's actual section/topic order. Paths visit selected modules in their declared order; additional prerequisite modules follow in catalogue order. Recorded prerequisites remain included and linked for review, without injecting individual detours or globally sorting difficulty. Previous/Next follows adjacent entries in those module contents, including planned lessons. Default entry resumes the first unfinished unique topic and resolves to a stable URL, so completion does not silently advance. The full syllabus has 28 real module groups. Topic/completion counts remain together with matching selected-topic scope, and partial modules link to the full module. A shared lesson retains every membership and one progress record; a module query parameter preserves its chosen local reading position after navigation or reload.

The hub and reader use `getLearningRoute` for the same resolved module/topic membership. Displayed module totals include modules contributing prerequisites: currently ML Foundations 6, LLM Engineer 13, Embodied Intelligence 10, Neural Engineering 7, GPU Engineering 8, Research & Revision 12, and Complete Curriculum 28. Derive these values from data at runtime; the initial `trackIds` selection alone is not a route's final module count. Publication and completion counts are separate, and unique topics are counted once across a path.

The generated [coverage inventory](docs/curriculum/CURRICULUM-INVENTORY.md) gives current counts, topic-level design/publication status and module guidance. [Its JSON companion](docs/curriculum/curriculum-inventory.json) contains the full stored briefs and route orders. Every module has domain guidance. Older topics without individual briefs are explicitly marked as needing design; inherited guidance is not presented as bespoke research.

## Research and coverage

| Report | Coverage and rationale | Boundaries |
| --- | --- | --- |
| [GPU engineering plan](docs/curriculum/GPU-ENGINEERING-PLAN.md) | Architecture; C/C++/build foundations; correctness; memory/synchronization; profiling; kernels; compilers; distributed operation; reliability; portability; scientific/graphics applications; capstones. | A CPU or browser exercise does not verify GPU performance. Vendor-specific features need device/version checks. Hardware/RTL research is a bounded specialist branch. |
| [Neural engineering plan](docs/curriculum/NEURAL-ENGINEERING-PLAN.md) | Biology/biophysics; signals; instrumentation/interfaces; modalities; data/decoding; closed-loop BCI, stimulation and prostheses; clinical evidence; quality/ethics; reproducibility. | Reading is not clinical qualification, invasive procedural competence or device authorization. Core practice uses suitable open data, simulation or bench phantoms. |
| [Cross-domain review](docs/curriculum/CROSS-DOMAIN-COVERAGE.md) | Each of the other 26 modules, existing coverage, material gaps, additions, primary sources and specialist boundaries. | Broad coverage is not a full factual audit of all older lessons or proof that no niche subfield is missing. |

These reports distinguish source-supported facts from curriculum design judgments. Research should verify mechanisms and improve original explanations. Recheck specific claims at authoring time; moving APIs/device support, clinical guidance and benchmark availability need dated primary sources. A source link is not a substitute for explaining the concept.

## Seven guided paths

| Route | Intended progression |
| --- | --- |
| ML Foundations | Programming, problem solving, mathematics, classical ML and neural foundations; core outcomes first, advanced branches later. |
| LLM Engineer | Programming, mathematics and classical/deep learning into LLM data/training, post-training, optimization, GPU systems, serving, agents and evaluation. The dedicated GPU path offers deeper systems study. |
| Embodied Intelligence | Programming, mathematics and learning foundations into sensing/frames, dynamics/control, RL, neural inspiration, robotics and fly embodiment. Classical/deep foundations now precede the RL module. |
| Neural Engineering | Quantitative/programming foundations, biological mechanisms, signals, measurement, interfaces, computational analysis, closed-loop neurotechnology and translation, matching the user's confirmed broad scope. |
| GPU Engineering | Programming, algorithms and numerical foundations into C/C++, architecture, correct kernels, measurement, optimization, compiler/framework work, multi-GPU operation, portability and engineering capstones. |
| Research & Revision | Mathematical/model foundations followed by generative, self-supervised, meta-learning, frontier and safety work. |
| Complete Curriculum | Every unique catalogue topic for navigation and systematic revision. Not every specialist branch is required for every role. |

Paths select relevant topics and their recorded prerequisite closure, then present that selection in module order. GPU and Neural Engineering select their full domain material with specifically required shared skills. LLM and Embodied Intelligence retain broader foundations and selected supporting GPU/neural core material. Difficulty is a label, not a global sorting rule. Prerequisite edges drive inclusion and readiness links; the module syllabus drives Next/Previous. Contiguous modules do not guarantee that every cross-module prerequisite occurs earlier, so keep readiness requirements explicit. Older unreviewed dependencies remain visible in the inventory; ordering software cannot discover pedagogy by itself. Specialist/frontier depth is labeled, and old links outside a focused selection open the canonical topic page.

Shared topics retain one ID and one implementation. Each relevant module can show the same lesson at its own position; unique path counts and completion records do not double-count it. Module shelves organize discovery and do not certify that every section requires no prior knowledge.

## Content design and rollout

Use the [domain playbook](docs/teaching/DOMAIN-PLAYBOOK.md) and [topic-design brief](docs/teaching/TOPIC-DESIGN-BRIEF.md). All neural/GPU entries are individually planned. For other older entries marked `individual-design-required`, complete their concept-level plan before authoring. Do not invent generic filler to suggest this work is finished.

The stored sequence and scope are revisable before and during writing: an author may add valuable missing material whose best home is the current topic, revise its title when warranted, change examples, split an overloaded lesson within coherent existing modules, choose another representation or add multiple focused labs. Preserve useful coverage, stable IDs/progress, prerequisite continuity, accuracy, complete examples, independent practice and evidence. Title-derived identity must be handled compatibly across consumers; changing a label alone must not remove or orphan a topic. One proposed visual in the schema is not a one-lab rule.

Search plausible existing owners when a discovery appears; distinguish missing coverage from a mention, an unimplemented plan or an actual explanation. Route ideas better taught elsewhere through [persistent destination-topic notes](docs/teaching/topic-notes/README.md), including the ownership reason and proposed learning treatment. Future authors receive these through the topic-plan command and make a reasoned disposition. Record unresolved ownership in the inbox. Scoped authorship does not require a catalogue-wide audit or speculative topic suggestions.

Develop interesting facts and familiar or less familiar applications when they expose a mechanism, useful consequence or new transfer setting. Use enough examples and detail for that purpose, with no quota and no forced trivia section. Keep the common core example coherent; place advanced connections where their prerequisites are available and link their actual teaching owners. The standard defines the source/accuracy and adequate-explanation requirements.

Keep a first-pass core route, intermediate application and optional specialist depth. Teach or link the specific prerequisite skill at the point of use; being in the same module does not require finishing all advanced mathematics before a practical first step. Keep required reasoning visible, and move long reference catalogues and secondary derivations to deeper routes.

Before publishing, prepare a claim ledger, complete examples/fixtures and solutions, individual visual contracts, model/runtime checks and desktop/mobile/keyboard review. A novice study is desirable when available; label author-only assessment honestly. The Linux record demonstrates the distinction between simulation, actual-runtime verification and user acceptance.

The user authorized fixing module sequencing and completing all seven remaining programming lessons; all 17 Programming & Scientific Computing topics now use the current approach. The subsequent DSA increment implemented Trees, Heaps/Tries and Graphs after Arrays and Linked Lists, and added guided LeetCode practice to all five. [The handoff](LESSON-AUTHORING-HANDOFF.md) owns review status and future scope. In DSA, Disjoint Sets & Union-Find is the next unimplemented topic in module order. These increments do not authorize rewriting the remaining website. Prior ordering claims belong to historical records and no longer define navigation.

For an authorized domain batch, select a small foundation-to-application slice with usable prerequisites, implement and validate it, and respect the requested review boundary. Do not bulk-generate full lessons from titles. Record improvements to the method in the standard and demonstrate the learning decisions.

## Maintenance and verification

Review foundational routes, missing professional workflows, source changes, specialist branches and capstones when substantive updates or discovered gaps warrant it. Recheck fast-moving interfaces and clinical/regulatory material at authoring. No recurring automation is created by this document.

- Live catalogue/integration: `src/learn/data/track-definitions.js`.
- Domain expansion plans: `src/learn/data/curriculum/*-expansion.js`. Authored lesson plans: [one source per stable topic ID](src/learn/data/curriculum/blueprints/index.js); the accepted reference is [the Linux blueprint](src/learn/data/curriculum/blueprints/linux-basics-filesystems-processes.js). See [blueprint ownership and migration](docs/engineering/BLUEPRINT-ORGANIZATION.md).
- Pending discoveries and ownership decisions: `docs/teaching/topic-notes/<stable-topic-id>.md`; unresolved routing: `docs/teaching/topic-notes/UNASSIGNED.md`.
- Dependency metadata/traversal: `topic-catalogue.js`; module strategies: `domain-guidance.js`.
- Guided paths: `src/learn/data/curriculum.js`; publication registry: `src/learn/data/topics/index.js`.
- `node scripts/verify-curriculum.mjs` checks IDs, brief fields, recorded prerequisites/cycles and inclusion, module/path membership, all pre-expansion IDs, one group per real module, complete route coverage and shared memberships. Reading order must match the module syllabus. `scripts/review-module-order.cjs` verifies actual reader navigation, counts, shared context, progress and planned entries at desktop/mobile sizes. Neither certifies unknown edges or source claims.
- `node scripts/build-curriculum-inventory.mjs` regenerates the status and plans.
- `node scripts/build-curriculum-inventory.mjs --topic "Exact topic title or stable ID"` prints one topic's plan, inherited domain guidance and persisted destination/routing notes. Read and assess them before authoring; they are not automatically approved content.
- The application build and curriculum browser checks validate integration. Future lesson numerical/runtime checks must match the claims and models actually implemented.

Record actual tested behavior and limitations in the increment's verification record. A script's existence does not prove it passed, and publication is not user approval.
