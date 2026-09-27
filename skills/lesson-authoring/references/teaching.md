# Teaching

Part of [$lesson-authoring](../SKILL.md). Repository paths below are relative to the selected application checkout; see [the repository adapter](portfolio-adapter.md). Load only the sections needed for the selected mode.

For continuity across sections and lessons, see [Carry a coherent explanation across sections and topics](#carry-a-coherent-explanation-across-sections-and-topics).

- [2. Intended learning experience](#2-intended-learning-experience)
- [Say each caution once, where it belongs](#say-each-caution-once-where-it-belongs)
- [3. Plan the learning before writing](#3-plan-the-learning-before-writing)
- [Revisit coverage and the title throughout authoring](#revisit-coverage-and-the-title-throughout-authoring)
- [Carry discoveries to their future authors](#carry-discoveries-to-their-future-authors)
- [4. Flexible lesson flow and depth](#4-flexible-lesson-flow-and-depth)
- [Build the reason before naming the machinery](#build-the-reason-before-naming-the-machinery)
- [Apply the teaching loop at every conceptual transition](#apply-the-teaching-loop-at-every-conceptual-transition)

## 2. Intended learning experience

Build a coherent learning resource where a newcomer can understand, explain, visualize, use, question, and practise a topic, then connect it to subsequent topics. A returning practitioner should also be able to revise or find a precise detail efficiently.

The standard is **simple explanation plus complete understanding within an explicit scope**. Plain language must preserve technical truth. More words, APIs, headings, diagrams, or controls do not by themselves establish depth. A reader clicking “complete” does not demonstrate mastery.

### Say each caution once, where it belongs

Rigor is expressed by stating conditions precisely, not by qualifying every result. Give each important caution one clear home: an early callout, a dedicated paragraph or a table row that the rest of the lesson can refer back to. After a result, say what it shows; do not append a sentence about what it does not prove unless that limitation is new information at that point. Code must never print disclaimers; explanatory sentences belong in prose.

Before delivery, reread the manuscript for this pattern specifically. A rough count of hedging phrases (“does not prove”, “not a certificate”, “not automatically”, “alone does not”) divided by the number of paragraphs is a useful signal: more than one such phrase per four or five paragraphs means the cautions are crowding out the teaching and should be consolidated. Correctness review tends to add qualifications; this pass removes the redundant ones without weakening any condition that matters.

Preserve the project's encouraging, technically serious voice and existing calm dark/gold visual direction described in .impeccable.md (repository path: `.impeccable.md`). This is a teaching improvement program, not authorization for a visual rebrand.

Each lesson should make these answers available at the point they are needed:

| Learner question | What the material must provide |
| --- | --- |
| What is this? | A concrete situation and plain explanation before specialised terminology. |
| Why should I care? | A realistic task, question, or decision the concept helps with. |
| What must I know first? | Specific prerequisite skills, a quick refresher or linked route, and no hidden prerequisite leaps. |
| How does it work? | A sequence of causal steps with the intermediate states or calculations exposed. |
| What does it look like? | A representation that makes the relevant relationship visible. |
| Can I see a complete example? | Inputs through process to result, interpretation, and practical conclusion. |
| How does it connect? | Explicit links to prior ideas and correspondences between words, diagrams, notation, and code. |
| When is it useful or unsuitable? | Applications, assumptions, alternatives, limits, and failure examples. |
| Can I do it myself? | Guided and independent practice with explanatory feedback. |
| What next? | A justified next lesson or deeper branch and a readiness check. |

## 3. Plan the learning before writing

For the authorized topics, make a short coverage map before implementation. Do this work autonomously unless a missing decision actually affects scope.

1. Read the existing material, examples, labs, reference sections, prerequisite/next-topic metadata and saved destination-topic notes (repository path: `docs/teaching/topic-notes/README.md`). Identify what to keep, repair, extend, or move to an optional branch.
2. State observable outcomes: for example, “predict which names observe a list mutation and explain why,” rather than “understand Python.”
3. Break the topic into concepts in dependency order. Record likely misconceptions and the difficult transitions between concepts.
4. Choose a continuing example for each coherent module. Keep the data, variable names, units, and entities consistent across explanation, visuals, code, and exercises. Explain any change of model or dataset.
5. Assign each difficult concept appropriate support and practice. Do not start by deciding how many labs the page should have.
6. Mark core coverage, optional depth, and linked follow-on topics. Check that the core route still reaches its promised outcome independently.

Useful planning table; adapt its rows to the actual topic:

| Concept/outcome | Prerequisite or refresher | Likely confusion | Explanation/example | Visual or interaction and its question | Practice and evidence of understanding | Core/deeper |
| --- | --- | --- | --- | --- | --- | --- |

Completeness means the scoped outcomes are covered and assessed. A large field should become sequenced modules when needed, not an unbounded page. “Advanced” is not a reason to omit a necessary explanation, and “beginner” is not a reason to remove important conditions.

### Revisit coverage and the title throughout authoring

The existing title and brief are starting hypotheses about the lesson's scope. Before writing, whenever research or implementation reveals a valuable connection, and again before delivery, check whether that scope still serves the learner. This is a focused review of the authorized topic and plausible related owners, not a requirement to audit the whole curriculum before each lesson.

Search the live catalogue, briefs and relevant lesson sources using the idea's name, synonyms and underlying mechanism. Distinguish **absent**, **mentioned only**, **planned**, **taught**, and **verified in the relevant implementation**. A title, source link or passing mention does not establish that the learner is taught the idea. Compare the proposed scope with appropriate primary references when checking substantive omissions; state what was checked rather than claiming universal completeness.

For each meaningful gap or discovery, choose and record its best teaching home:

| Finding | Authoring action |
| --- | --- |
| Necessary to the current outcome, or most coherently taught with this topic's mechanism | Include it here with explanation, examples and appropriate practice. Update the brief and finish line as needed. Do not reject it merely because it was discovered after writing began. |
| Valuable here but requiring additional depth or prerequisites | Supply the necessary core bridge, then a well-explained optional branch or sequenced follow-on. Keep essential reasoning in the core. |
| Better taught in another existing topic | Explain the local connection where useful and save a destination-specific note with rationale and proposed treatment. Do not force a full unrelated lesson into the current page or silently rewrite an unrequested destination. |
| Already adequately taught elsewhere | Link the exact relevant explanation and teach the local connection; duplicate only the refresher needed to follow the current reasoning. |
| No suitable owner exists | Record a proposed topic, module, prerequisite relationship and reason in the discovery inbox; create a sequenced entry when catalogue work is within scope. Do not discard the idea or manufacture a full lesson now. |

Revise the learner-facing title if the actual content now promises a materially different or broader useful outcome. Keep it concise and accurate; adding an example or interesting application does not automatically need a longer title. A rename may be justified during writing, not only in the initial plan. Record old/new title, why it changed, retained coverage and additions. Never remove useful existing coverage to make the new title convenient.

Preserve stable topic identity, saved progress, existing links and shared memberships. This repository currently derives IDs from titles in several places, and some briefs/prerequisites are keyed by exact title. A bare title replacement or an unused `id` property is therefore not a safe rename. When a rename is warranted, implement compatible title/identity handling in the actual catalogue, registry, route and prerequisite consumers, preserve old URLs/progress, and run conservation/navigation checks. Make that routine compatibility work part of the authorized topic change; do not freeze an inaccurate title because of the current plumbing.

### Carry discoveries to their future authors

Persist a useful idea as soon as its destination is clear, including discoveries during research, coding or verification. Use `docs/teaching/topic-notes/<stable-topic-id>.md` following the note workflow (repository path: `docs/teaching/topic-notes/README.md`). The normal topic-plan command surfaces that file and the unresolved routing inbox. Do not leave the only copy in a conversation, the originating lesson's report, an archive or a vague module-level TODO.

Record the idea, originating topic, destination and why it is a better fit, present coverage/gap, learning benefit, proposed placement and depth, prerequisites, possible example/visual/practice, sources and uncertainty. A future author must read and reason about these notes: include, adapt, reroute, defer with a concrete reason, or reject with evidence. They are proposals to assess, not unverified facts or automatic instructions to paste content. Mark a note implemented only after linking the actual explanation and relevant checks. Keep unresolved notes visible across handoffs; merge duplicates without losing their rationale.

## 4. Flexible lesson flow and depth

Use this as a learning progression, not a mandatory sequence of identical headings:

**Problem and reason to care → simple intuition → concrete representation → mechanism → formal detail → complete example and application → practice and diagnosis → deeper connections and next steps.**

Repeat a smaller loop where a new conceptual hurdle appears:

**Explain simply → show the mechanism → change an input or work one step → observe the result → explain the cause → try a variation and make a decision.**

An early lab may prepare the intuition; another can follow a derivation to investigate its implications. Exercises belong near the relevant concepts as well as at the end.

### Build the reason before naming the machinery

#### Apply the teaching loop at every conceptual transition

Intuition is a continuing responsibility throughout a lesson, not a section completed by its opening analogy. Review every newly introduced idea the reader must reason with: a mechanism, quantity, representation, objective, assumption, algorithmic step, architecture variant, implementation choice or evaluation result. A single topic may contain many such transitions, including inside advanced branches. A familiar prerequisite can be recalled and linked; do not silently treat a new idea as familiar because it belongs under an existing heading.

For each transition, establish what the reader already knows, the question or limitation that motivates the new idea, and a plain explanation of what changes. Follow the relevant entities through an inspectable example with intermediate steps and an interpreted result. Connect the prose, visual, symbols and implementation where those forms occur. Explain why the operation has that effect and what decision or further question follows it. Compare a nearby alternative or counterexample when that exposes the key distinction. Choose the amount of support the actual hurdle needs; a short local bridge may suffice for a small transition.

Map these transitions before and during writing in the topic design. Record the exact section or component that supplies each bridge, the existing support worth retaining, any gap, and the chosen explanation/representation. Independently review the full reading path against that map. “An intuition section exists,” a figure count, or a general introductory diagram does not close gaps later in the lesson. Name remaining gaps explicitly instead of certifying the page from its strongest example.

Use as many useful diagrams, worked traces and live investigations as the mechanisms require, at their point of need. A picture must expose a relationship or transformation the learner otherwise has to reconstruct mentally. Do not replace that job with summary cards, boxes containing prose or decorative graphs. Do not force a diagram onto a transition already clear through a compact calculation, table, familiar earlier figure or direct explanation. Reuse and link an established representation when it still works; show the precise new part when an idea extends it.

Research teaching approaches at the weak concept, not just at the topic title. Compare original explainers and primary references for the particular difficulty, check the limits of analogies, then write an original self-contained explanation. Preserve advanced coverage and complete scratch/library implementations. Layer detail and improve transitions rather than deleting difficult ideas or adding repetitive introductions everywhere.

The 26 September attention/memory revision exposed a gap that numerical correctness
checks did not catch: dense openings, immediate notation, and diagrams that named
components without establishing why a learner would need them. For each major new
mechanism, teach a concrete problem, make the simpler approach's relevant limitation
visible, and follow one complete repair before presenting the taxonomy or advanced
counterexamples. Do not imply the simpler method always fails; state the actual
information or computation constraint.

Keep the example's objects recognizable through prose, diagram, worked numbers,
formula and code. Explain when the example changes and why. Before a formula,
identify what its inputs and result do; afterward, connect its intermediate
operations to the visible example. A compact executable core can bridge arithmetic
to a full program. Keep the complete scratch/library routes and deep derivations,
but route optional branches so they do not interrupt the first successful operation.
Do not prepend a generic analogy and leave an otherwise inaccessible lesson unchanged.

Research strong original explainers, textbooks, creator tutorials and current
primary documentation where the intuition is weak. Compare how they introduce the
problem, sequence the reasoning and use visual correspondences. Adopt useful
teaching decisions with an original example and wording; popularity or attractive
animation does not establish accuracy. Record what was read or watched, distinguish
recent resources from useful classics, and verify nuanced technical claims against
primary sources. Retain a self-contained explanation and annotated alternate routes.

An early lab should reveal only quantities whose roles have been introduced.
Introduce an output loss or auxiliary classifier before displaying it as a central
result. Add more controls when their learning question arrives. A technically correct
lab is not a substitute for the explanatory buildup that makes its controls meaningful.

| Depth | Expected experience |
| --- | --- |
| Beginner core | Explain the purpose in ordinary words; identify the entities; follow one small example; read its visual; predict a simple change. |
| Intermediate application | Carry out the method or workflow; interpret results; compare choices; handle common edge cases; solve a changed problem. |
| Advanced branch | Explain why the method works more formally; examine assumptions and counterexamples; compare alternatives, tradeoffs, generalizations, and relevant implementation limits. |

Do not require a beginner to already know all notation or advanced software used later. Define a term immediately before relying on it. Give every symbol a meaning, units or shape where applicable, and its role in the current example. Translate a formula into a sentence and substitute small values before skipping to a library call. State where an analogy stops being faithful.

Use explicit core/deeper labels, meaningful section navigation, short summaries, and visible, clearly labeled derivations or secondary implementation details. Do not hide essential steps or safety-relevant conditions in optional material. Reference catalogues and long API inventories belong on a revision route, with links from the tasks that need them.

Any lesson longer than about one sitting states a **first-pass route** in the shared opening directly below the reader title: which sections to read now, which labs and programs to run on the way, and which sections are deeper branches to return to. Long sections that are deeper branches say so in their first line. The opening's complete section index supplies jump links; the first-pass guidance explains which of those sections to choose.

Every published lesson uses that same opening position and neutral/amber navigation treatment. Supply its topic-specific summary, first-pass guidance, prerequisites and instructions for using examples/labs through the shared opening components. Keep this orientation together instead of scattering separate route cards, partial tables of contents and exploration instructions between introductory paragraphs. The reader derives the full section list from the lesson's actual H2 headings; do not maintain a competing hand-written list. Preserve existing fragment links and authored section numbers, including when the lesson contains unnumbered introductory or reference sections. See the opening component and prepared-renderer contract (repository path: `docs/engineering/LEARNING-CODE-STANDARD.md#shared-lesson-opening-and-section-navigation`).

This common navigation structure does not prescribe identical content sections, diagrams, labs or exercises. The lesson body still builds its concrete problem and intuition before introducing new mechanisms, and topic-specific representations and styles remain available where they teach the subject.

When the same quantity, object or partition appears by two different routes in one lesson, say so explicitly at the second appearance and explain why the routes agree or when they would not. Readers do not reliably notice that a number has recurred, and the connection is often the most valuable idea on the page.

Retain useful detail, but remove repetitions that add no new understanding. Break a long explanation when its learning question changes, not at an arbitrary word count. Estimate reading separately from hands-on practice.

### Carry a coherent explanation across sections and topics

Use the anchor problem as a thread: what we can do now, the next limitation or question, the mechanism that addresses it, and what the result enables. Return to the original problem with an interpreted result. Keep intermediate state, names, units and notation consistent; when a different example or formalism teaches better, explain the correspondence before relying on it. This is explanatory continuity, not invented storytelling or extra narrative padding. Different subjects may need different flows, proofs, comparisons or experiments.

Across lessons, link the exact earlier explanation or runnable implementation, briefly recover the contract needed here, and make the new contribution explicit. A prerequisite link alone must not conceal a missing conceptual bridge. Use related applications and later questions where they deepen understanding, while keeping the official next lesson in syllabus order. Save unresolved downstream connections through the [handoff and note lifecycle](handoffs-and-cleanup.md#carry-learning-decisions-between-topics), with enough reasoning for the receiving author to assess them. Reuse taught mechanisms without needless repetition; repeat or rederive when changed assumptions or notation require it.
