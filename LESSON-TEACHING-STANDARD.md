# Lesson teaching standard

Updated 12 September 2026 with the learning-experience checklist, the once-stated caution rule, the first-pass route, the investigation requirements for labs, real-data and displayed-code rules, figure perceptibility and the canonical-reference coverage check, after a quality review of a completed lesson found that correctness review alone had let these through. Earlier the same day: the user's full versus content-first delivery modes and separate phase tracking. Coverage, titles and useful applications must be reconsidered during authoring as well as before it. This is the authoritative teaching policy. [The handoff](LESSON-AUTHORING-HANDOFF.md) owns current scope and status; historical reports do not override this standard.

## 1. Start here in a future session

Read [the current handoff](LESSON-AUTHORING-HANDOFF.md), this standard, [the domain playbook](docs/teaching/DOMAIN-PLAYBOOK.md), [the topic-design brief](docs/teaching/TOPIC-DESIGN-BRIEF.md), and the actual topic/component sources before editing. Read historical evidence only when needed for the topic. Preserve useful existing work.

Follow the user's current scope and earlier authorizations. A request for one topic authorizes that topic, not the whole track. Do not repeatedly ask for permission within an already authorized batch. At an agreed review boundary, deliver the concrete changes and evidence before expanding the rollout. Curriculum planning may add detailed syllabus entries without authorizing full lesson rewrites. Do not infer the active scope from a dated batch report.

The codebase is the nested `Portfolio` project containing `package.json` and `src/learn`. Existing work may be uncommitted or untracked. Do not reset, replace, or discard it to establish a clean starting point.

Follow [the learning code standard](docs/engineering/LEARNING-CODE-STANDARD.md) for source ownership, file/variable names, publication registration, generated metadata and browser performance. Topic-specific depth and visual variety do not justify loading unrelated lessons or cross-topic example bundles. Register completed lessons in `lesson-manifest.json`; the former eager `topics/index.js` registry is removed. Preserve the loading/error/navigation contract and verify the actual production dependency graph when changing it.

Suggested future-session prompt:

> Read LESSON-AUTHORING-HANDOFF.md and follow its reading order. Continue the existing educational improvement work for the topics I name. Preserve the approved Linux lesson's teaching strengths, adapt the flow to the domain, plan support for each conceptual hurdle, and report what changed, why, validation, and remaining gaps.

### Quality takes priority over efficiency

Content quality, correctness, completeness within the intended scope, clarity, useful visual teaching, meaningful practice and reliable behavior take priority over token or time savings. Efficiency guidance in these documents is a set of working defaults, not a limit on depth, research, context reading, implementation effort, review rounds, verification or evidence. Do not apply an optimization when it could weaken those qualities; if its effect is uncertain, do the work needed to assess and protect quality.

The author or reviewer may depart from an efficiency default whenever their judgment identifies a worthwhile improvement or a quality concern. They may read more source material, investigate further, revise the teaching structure, add examples/diagrams/labs, broaden relevant checks, seek additional authorized review or revisit an earlier decision. A reproduced defect is not required: a confusing transition, weak explanation, potentially misleading assumption or opportunity for substantially better understanding is sufficient reason. Continue autonomously within the user's authorized scope; this does not create an extra permission gate. Briefly record the reason in the existing work record when it helps the handoff, without creating another reporting requirement.

Preserve valid evidence and avoid repetition that adds no value, while using professional judgment about what the topic needs. No suggested pass count, concise-output preference, small-team preference or completed checklist justifies reducing quality or leaving an identified useful improvement unfinished within the agreed scope.

### Six-stage authoring workflow

Use these stages for each authorized lesson, within the delivery mode below. They divide into **content (stages 1–2)** and **implementation, review and integration (stages 3–6)**. These are work stages, not six full rewrites or six user-approval gates. In a full request, proceed through both phases without pausing for another approval; stages and checks can inform one another. Adapt the teaching sequence and representations to the domain.

| Stage | Responsible role and work | Completion condition |
| --- | --- | --- |
| **1. Assess and research** | The author reads the current lesson and destination notes, defines outcomes and prerequisites, assesses scope/title, preserves useful depth, researches material claims, and maps examples, visuals and practice to the conceptual hurdles. | A usable topic design explains what to keep, repair and add, what belongs elsewhere, and how learners will demonstrate the scoped outcomes. Research can continue when writing reveals a new question. |
| **2. Write the complete lesson and visual specifications** | The author writes the actual explanations, worked examples, applications, changed practice, hints/solutions and annotated alternatives. Specify each diagram/lab at its place in that explanation, including its quantities, states, learner action and intended feedback. Check claims and example reasoning as part of writing, and reread the complete manuscript for continuity. | A complete written lesson and actionable visual/lab specifications cover the promised outcomes. Record the source checkpoint, research and any deferred runtime verification. An outline or a collection of TODOs is insufficient. This completes the content phase, not implementation review. |
| **3. Implement, author-review and verify** | Use the completed content to build the reader body, diagrams, labs, models and example/output presentation. Apply early component/metadata compatibility checks. Review the whole reading flow, verify complete native outputs, model/visual agreement and supported edge cases, and begin actual browser/keyboard review. | A complete runnable lesson exposes the intended teaching through the real reader. Its source, actual checks, scope and remaining findings are recorded honestly. The reviewer is not given an implied clean result. |
| **4. Independent review** | A reviewer other than the author assesses coverage, beginner continuity, technical accuracy, representation fit and changed practice, and separately runs the [learning-experience checklist](#learning-experience-checklist). Inspect the existing evidence and investigate complementary risks instead of duplicating the author's full test run. | A bounded set of actionable findings or a no-findings assessment is attached to the reviewed source, with correctness and learning-experience findings distinguished. An author's own second reading is not independent review; record that distinction if no independent reviewer is available. |
| **5. Focused corrections and browser closure** | The author resolves findings and checks affected behavior; the reviewer confirms the relevant corrections. Complete the required desktop/narrow-screen, actual reading, keyboard, interaction and selected-image checks, capturing the informative lab and figure states rather than defaults only. | No material finding remains; affected rechecks pass, unchanged evidence is reused, and the final reviewed source is identified. Check geometry and teaching clarity as well as successful clicks. A correction must not degrade the teaching surface it repairs: prefer a prose sentence, a verifier case or a separate helper over complicating displayed teaching code or adding another qualification to the prose. |
| **6. Website integration and handoff** | The increment owner validates generated metadata, catalogue/identity conservation, sequence/counts/progress, the production build and relevant loading/recovery behavior. Update the existing ledger, brief, note dispositions and handoff; remove disposable working material. | The authorized increment has passing relevant integration evidence, a concise completion/next-action record and stated limits. User acceptance remains separate; completing this stage does not authorize the next topic. |

Let the topic's complexity and quality needs determine the number of revisions and review rounds. Work from a complete draft and combine related corrections when useful, but do not treat one author revision or one independent round as a target that constrains better work. A stage may require no edits or several revisions. Further work is justified by a quality concern, a worthwhile teaching improvement, relevant source/environment change or learner feedback. Changing a sentence, checking three viewport widths or exercising many numerical fixtures does not mean rewriting the lesson that many times.

Use the existing topic/increment record to state the current stage, owner, reviewed source, findings and next action. Record substantive correction rounds with their reason and affected checks when they occur; do not create a new report for every edit or retroactively invent an exact iteration count. Keep implementation, numerical verification, browser review, independent review, integration and user acceptance distinguishable.

**Readiness requires that the scoped outcomes are taught and assessed, material claims/examples are verified, the reading and interaction experience works, and no material review or integration finding remains.** Meeting this minimum does not require stopping if an identified in-scope improvement would materially improve understanding or correctness. Neither a fixed number of passes nor a claim of perfection establishes readiness. Apply section 11's bounded-verification guidance under the quality-first principle above.

### Delivery modes and stopping boundaries

There are two delivery modes. A later **finish** request continues the second mode; it is not a third teaching standard.

| User request | Authorized work and stopping point |
| --- | --- |
| **Full implementation** / ordinary “implement the next topic” | Complete both phases and all six stages, including topic-specific implementation, independent review, fixes, browser checks and integration. This remains the default for an implementation request. No additional approval is needed between phases. |
| **Content first** / “research and write only” | Complete stages 1–2: the full manuscript and detailed visual/lab specifications. Stop with content complete and implementation not started. Keep implementation, formal review, browser work and integration deferred. |
| **Finish the prepared topic** / “implement and verify the existing content” | First verify the current revision has a complete content checkpoint. Then perform stages 3–6 using that material, including necessary content corrections, additional research and improvements discovered during implementation/review. Finish the scoped topic without restarting a completed research/writing phase by default. |

A user's explicit delivery mode persists for their named batch and clear continuations until they change it. Announce the selected mode briefly. Do not turn a content-first request into a full implementation because the quality-first principle allows extra work; the phase boundary is part of authorized scope. Within that boundary, do any additional research, reasoning or writing needed for quality. Full delivery keeps the existing depth and completion standards.

**Content first means writing plus specifications, not building the visuals/labs.** Save a complete learner-facing manuscript at `docs/teaching/drafts/<stable-topic-id>/lesson.md` and its specifications at `visual-specifications.md` in the same directory. Link the existing design/research record instead of duplicating it. Include instructional code in the manuscript when it is part of teaching the concept, with complete inputs, intended results and explanations; distinguish derived/predicted results from outputs actually executed. Production code organization, runnable-output verification, React/SVG/lab implementation and browser integration belong to phase two.

The manuscript contains the real explanations and practice solutions at the required depth, not instructions telling the next agent to write them. Specifications identify the exact placement, entities, data/formulas and provenance, labels/units, states/transitions, controls, prediction/feedback, text equivalent, mobile composition, edge cases and what must be verified. Supply enough detail to build each representation without inventing its teaching purpose; allow the implementer to improve it with a recorded reason. A specified graph is not a verified rendered graph.

Research and elementary correctness are never optional: resolve material conceptual contradictions and check the reasoning/answers actually stated. Small calculations or code probes needed to substantiate a claim can be part of writing. Defer the full execution, independent review, interaction, browser, performance and integration campaigns; never invent evidence for them. If a material content gap remains, leave content in progress and state the next action. A content-complete checkpoint means ready to implement, not guaranteed error-free or ready to publish.

Content-first work leaves the currently published lesson, runtime components, publication manifest and navigation unchanged. A suggested title/prerequisite/sequence change goes in the handoff for compatible implementation later. New topics remain planned. Full-mode work may use the semantic production source as its written-content checkpoint and need not create a duplicate Markdown manuscript; the scoped design still holds the visual specifications. It must establish complete content before marking the implementation phase underway.

### Two-phase handoff and prerequisite

[The delivery ledger](docs/teaching/lesson-delivery-progress.json) owns current phase status for all topics. Its [schema and update procedure](docs/teaching/LESSON-DELIVERY-LEDGER.md) preserve earlier evidence, source identity and revision history. Per-increment ledgers continue to hold detailed historical verification; do not create competing current phase queues.

At a content-first boundary, record the stable topic ID/revision, requested mode, exact manuscript/specification file hashes, design/research links, completed author checks, deferred checks/uncertainties, intended production destinations and precise next action. Retain these drafts until phase two consumes them. Another agent must read this packet and the full written content/specifications, not rely on the originating chat. It may correct the content when evidence or implementation reveals a better explanation, and must refresh the checkpoint for changed material.

Before a finish-only request, run `node scripts/build-curriculum-inventory.mjs --topic "<ID>" --work finish`. A missing/incomplete/stale content checkpoint fails the preflight. A topic title, blueprint, older published body or unfinished draft alone does not satisfy the prerequisite. Report the specific missing prerequisite and ask for content work or identification of an existing complete draft; do not silently author a whole new lesson under finish-only scope. If a complete manuscript already exists, assess and register its checkpoint without rewriting it merely to satisfy the ledger.

Keep `content` and `implementation` independently recorded as `not-started`, `in-progress` or `complete`; implementation cannot start until content is complete, and completion belongs to one source revision. Later source changes make prior completion stale rather than silently approving new bytes. Preserve completed historical revisions and their evidence when beginning a new revision. A full request ends with both complete; content first ends with only content complete; a successful finish request completes implementation for that content. User acceptance and publication remain separate. Do not ask for extra approval simply because a different agent will perform phase two.

## 2. Intended learning experience

Build a coherent learning resource where a newcomer can understand, explain, visualize, use, question, and practise a topic, then connect it to subsequent topics. A returning practitioner should also be able to revise or find a precise detail efficiently.

The standard is **simple explanation plus complete understanding within an explicit scope**. Plain language must preserve technical truth. More words, APIs, headings, diagrams, or controls do not by themselves establish depth. A reader clicking “complete” does not demonstrate mastery.

### Say each caution once, where it belongs

Rigor is expressed by stating conditions precisely, not by qualifying every result. Give each important caution one clear home: an early callout, a dedicated paragraph or a table row that the rest of the lesson can refer back to. After a result, say what it shows; do not append a sentence about what it does not prove unless that limitation is new information at that point. Code must never print disclaimers; explanatory sentences belong in prose.

Before delivery, reread the manuscript for this pattern specifically. A rough count of hedging phrases (“does not prove”, “not a certificate”, “not automatically”, “alone does not”) divided by the number of paragraphs is a useful signal: more than one such phrase per four or five paragraphs means the cautions are crowding out the teaching and should be consolidated. Correctness review tends to add qualifications; this pass removes the redundant ones without weakening any condition that matters.

Preserve the project's encouraging, technically serious voice and existing calm dark/gold visual direction described in [.impeccable.md](.impeccable.md). This is a teaching improvement program, not authorization for a visual rebrand.

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

1. Read the existing material, examples, labs, reference sections, prerequisite/next-topic metadata and [saved destination-topic notes](docs/teaching/topic-notes/README.md). Identify what to keep, repair, extend, or move to an optional branch.
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

Persist a useful idea as soon as its destination is clear, including discoveries during research, coding or verification. Use `docs/teaching/topic-notes/<stable-topic-id>.md` following the [note workflow](docs/teaching/topic-notes/README.md). The normal topic-plan command surfaces that file and the unresolved routing inbox. Do not leave the only copy in a conversation, the originating lesson's report, an archive or a vague module-level TODO.

Record the idea, originating topic, destination and why it is a better fit, present coverage/gap, learning benefit, proposed placement and depth, prerequisites, possible example/visual/practice, sources and uncertainty. A future author must read and reason about these notes: include, adapt, reroute, defer with a concrete reason, or reject with evidence. They are proposals to assess, not unverified facts or automatic instructions to paste content. Mark a note implemented only after linking the actual explanation and relevant checks. Keep unresolved notes visible across handoffs; merge duplicates without losing their rationale.

## 4. Flexible lesson flow and depth

Use this as a learning progression, not a mandatory sequence of identical headings:

**Problem and reason to care → simple intuition → concrete representation → mechanism → formal detail → complete example and application → practice and diagnosis → deeper connections and next steps.**

Repeat a smaller loop where a new conceptual hurdle appears:

**Explain simply → show the mechanism → work one step → ask for a prediction or action → explain the result → try a variation.**

An early lab may prepare the intuition; another can follow a derivation to investigate its implications. Exercises belong near the relevant concepts as well as at the end.

| Depth | Expected experience |
| --- | --- |
| Beginner core | Explain the purpose in ordinary words; identify the entities; follow one small example; read its visual; predict a simple change. |
| Intermediate application | Carry out the method or workflow; interpret results; compare choices; handle common edge cases; solve a changed problem. |
| Advanced branch | Explain why the method works more formally; examine assumptions and counterexamples; compare alternatives, tradeoffs, generalizations, and relevant implementation limits. |

Do not require a beginner to already know all notation or advanced software used later. Define a term immediately before relying on it. Give every symbol a meaning, units or shape where applicable, and its role in the current example. Translate a formula into a sentence and substitute small values before skipping to a library call. State where an analogy stops being faithful.

Use explicit core/deeper labels, meaningful section navigation, short summaries, and expandable derivations or secondary implementation details. Do not hide essential steps or safety-relevant conditions in optional material. Reference catalogues and long API inventories belong on a revision route, with links from the tasks that need them.

Any lesson longer than about one sitting states a **first-pass route** immediately after its introduction: which sections to read now, which labs and programs to run on the way, and which sections are deeper branches to return to. Long sections that are deeper branches say so in their first line. A section list in the intro is navigation; it is not a route.

When the same quantity, object or partition appears by two different routes in one lesson, say so explicitly at the second appearance and explain why the routes agree or when they would not. Readers do not reliably notice that a number has recurred, and the connection is often the most valuable idea on the page.

Retain useful detail, but remove repetitions that add no new understanding. Break a long explanation when its learning question changes, not at an arbitrary word count. Estimate reading separately from hands-on practice.

## 5. Visuals and labs: no fixed count

**There is no requirement for exactly one lab, at least one lab, or a maximum number of labs per lesson.** Support every major conceptual hurdle using the representation that teaches it best. Several small labs and diagrams are appropriate when a topic contains several distinct mechanisms. A clear static diagram may fully explain a relationship without interaction.

### Let the concept determine the visual form

Keep the learning philosophy, accessibility, navigation and control behavior dependable across the site. Choose the illustration's geometry, visual vocabulary, terminology and interactions for the specific concept and learner question, including different choices within one lesson. Consistency does not require every topic to use the same lab arrangement, text panel, slider or stepper. Variation has no quota either: reuse a familiar representation when it still exposes the mechanism well.

Before choosing a component, identify the actual objects, relationships and operations the learner needs to understand, and what is hard to see in prose or code. Choose the form that makes those features inspectable. Use the subject's useful conventional representations and terms, introducing their plain meaning and visual legend first. For example, token boundaries and pair merges suggest symbol strips; reference sharing suggests names, objects and arrows; concurrent execution suggests separate lanes and synchronization points. Changing a panel's title or colors does not adapt it to a different mechanism. An existing table or text trace remains appropriate when the task is best understood that way.

For an interactive lab, let the learner act on meaningful entities or parameters and see the corresponding transformation. A tokenization trainer can connect an editable corpus, the selected pair, its frequency, changed word segmentations and merge history. A guided trace and an open investigation can serve different purposes in the same lesson. This is a reference for designing around a topic's operations, not a required layout or an endorsement of every detail in the existing BPE implementation.

Keep connected views consistent with the same model state. Make it clear when edited input is applied; labels, counts, highlights and outputs must describe the active state. Use controls named for the action when helpful, such as merging a pair or advancing one instruction. Do not require unrestricted editing, animation or a full simulator when a smaller investigation teaches the question. Share useful controls and rendering primitives, but adapt or build a representation when the available generic panel obscures the concept. Record the learning reason for that choice in the brief; novelty alone is not a reason.

### Inline diagrams are part of the explanation

Plan visual support throughout the reading flow, including compact, immediately visible diagrams between paragraphs. Do not treat completing the interactive labs as completing visual teaching. An inline diagram helps the learner see a structure, correspondence, contrast or transformation while reading; an interactive investigation helps them explore how it changes. Use either or both when their teaching jobs warrant it.

The user's tokenization comparison is a useful presentation reference: the same input is shown as character units, an unknown-word result and subword units in three labelled rows. Boundaries and contrasting representations are visible without operating a control. Reference this visual pattern, not the surrounding page as a blanket accuracy endorsement or a compulsory token-chip style for every subject.

For a new concept, ask whether the reader currently has to assemble its relationships mentally from prose or code. When a picture would make those relationships materially clearer, place it at that point in the explanation. Useful forms include aligned before/after states, labelled values or cells, reference arrows, nested ownership, branching control flow, data transformations, miniature timelines and same-input comparisons. Introduce the objects and labels before asking the learner to manipulate them. Connect the picture to the exact example and explain what to notice in a short caption or nearby sentence.

A static diagram's design contract is its learner question, concrete example/state, meaningful visual encoding, placement, interpretation and limits. Verify its labels, values, arrows and correspondences against the example. Controls, step/reset behavior and simulated state are additional requirements only when the diagram is interactive. Prefer the representation the concept needs; there is no requirement for SVG, animation, a full lab panel or a particular renderer. The existing React/HTML/CSS/SVG approach supports these figures.

Review the lesson in ordinary reading order without operating every lab. Check that important structures and contrasts are visible where introduced, and that their meaning does not depend on reconstructing several code blocks or discovering a later lab state. A useful, already visible lab illustration can satisfy this need; do not duplicate it automatically. Record any substantive visual gap and its proposed placement. Counts of labs, code examples, tables or colored boxes are not evidence that this reading pass succeeds. Keep tables when they explain a comparison well, and omit diagrams that merely decorate or restate a clear sentence.

Use consistent labels and restrained color to track identity or correspondence, with text/position/shape cues as well as color. Preserve readable labels, correct grouping and logical order on narrow screens. A row of boxes containing paragraph summaries is not a mechanism diagram unless its arrangement actually reveals the relationship being taught.

### Choose the representation for the hurdle

| What the learner needs to understand | Useful representation |
| --- | --- |
| Identity, containment, shared references | Labeled objects and connecting arrows; distinguish a name from the object it refers to. |
| A changing process or hidden state | Stepped execution with the current instruction, before/after state, and changed items highlighted. |
| A parameter's effect | A focused experiment with a labeled control, comparable outputs, and an explanation of the change. |
| Distributions and uncertainty | Meaningful axes, reference lines, correctly shaded probability areas, and repeated samples where relevant. |
| Shapes, broadcasting, reductions | Aligned dimensions, labeled row/column/cell correspondence, and an explicit resulting shape. |
| Joins and grouping | Trace source rows into result rows; show unmatched records, multiplicity, and aggregation. |
| Data or control movement | Directed flows with labeled inputs, outputs, order, and failure paths. |
| Competing methods | Same-input comparisons with assumptions, outcome differences, and when to choose each. |
| Exact values or accessible inspection | Tables alongside the conceptual representation. Tables are sometimes the primary representation, especially for tabular data. |

For every lab, specify a short teaching contract:

- **Question:** what particular uncertainty should this resolve for the learner?
- **Starting point:** what entities and initial state are shown, and what must the learner already know?
- **Prediction:** what should they predict before manipulating anything?
- **Control:** what changes and what is deliberately held fixed?
- **Visible consequence:** which states, correspondences, or quantities change and why?
- **Check:** what observation or follow-up task would show understanding?
- **Boundary:** what is simulated, simplified, fixed, unsupported, or outside the model?
- **Fixture check:** which alternative settings were run to confirm that the chosen data actually exhibit the contrast the surrounding prose promises, and which null case shows the contrast is absent when it should be?

Implement the interaction as **predict → manipulate → observe → explain → transfer**. Start with a useful preset and a suggested investigation. Supply meaningful reset/back controls, readable feedback, and a way to recover from invalid states. If a simulation has randomness, support reproducible comparisons and clearly distinguish changing a parameter from drawing fresh data.

Three requirements turn a guided trace into an investigation and are checked at review:

1. **The prediction is recorded, not merely suggested.** Wherever the outcome is determinate, the lab offers a control to record a prediction before acting and compares it with the model's actual result afterwards. A sentence beginning “Predict:” with no way to answer it is a caption, not a prediction step. The approved Linux path lab is the reference for this pattern.
2. **The learner acts on the topic's entities.** At least one control changes something the prose has not already resolved: choosing which items start a process, placing or selecting a threshold, changing an input rather than picking among two or three labeled presets whose outcomes are described beside them. Presets remain useful as suggested starting points.
3. **Every fixture demonstrates what it is placed beside.** Before accepting a dataset or example for a lab, run the alternatives the lab offers and confirm they differ in the way the adjacent explanation claims. A table that promises four rules behave differently is contradicted by a fixture on which all four agree. When the natural running example cannot show the contrast, add a second fixture built to show it, and check that its critical values are exactly representable so a promised tie is a real tie.

Connect representations directly: selecting an edge may highlight its matrix entries; stepping a line may move a reference arrow; choosing a result cell may reveal its input operands. Use stable labels and color meaning. Explain whether layout position, line thickness, area, or color encodes a quantity or only organizes the drawing.

Do not treat a changing table, a “Next” button, decorative animation, or a screenshot of code as proof that the mechanism is intuitive. Ask whether a first-time learner can explain what changed without reconstructing everything mentally. Retain exact tables as inspection/accessible companions when adding spatial explanations.

Give each additional lab a distinct job. Put it beside the concept it teaches. Split crowded controls or unrelated questions into focused activities; offer an optional integrated investigation after the parts are understood. Share UI components where useful, without forcing every concept into the same visualization.

### Graphs and simulations need traceable evidence

Code-rendered charts can teach relationships clearly; rendering code does not establish the truth of their values or interpretation. Identify what each quantitative figure represents: an exact calculation, an analytic model, a simulation, measured data, externally sourced data or an illustrative sketch. Put the relevant assumptions and status next to the figure, and retain its equation, generator, measurement record or source locator in the authoring evidence.

- Check values independently where appropriate, along with units, scales, log transforms, ranges, normalization, rounding and correspondence to the text. Distinguish probability from density, observed samples from interpolation, and supported ranges from extrapolation. Show variability or uncertainty when the claim depends on it.
- Real performance comparisons need applicable sourced measurements or a reproducible benchmark: record the task, corpus/input sizes, algorithm settings, software versions, hardware, parallelism, timing boundaries and variability. The caption and prose must not generalize beyond those conditions.
- An illustrative sketch can explain a hypothetical relationship, but invented points must not imply measured seconds, real-product rankings or established scaling laws. Use explicit model assumptions and generic series for a hypothetical example, or replace it with verified data. The word “illustrative” does not justify nearby unsupported empirical claims.
- **The claimed feature must be perceptible in the rendered figure.** If the prose says a curve bends, flattens, peaks or separates, a reader must be able to see that at the size the figure renders, on desktop and on a phone. Check the range of the plotted values against the axis: when one value dwarfs the rest, a linear axis turns the interesting region into a flat line, and a logarithmic axis, a second panel or a relative measure is needed. Include the natural baseline point (the one-component, zero-change or untreated case) when it anchors the comparison. Show repeated runs as individual marks when their disagreement is part of the message. A caption admitting that the numbers are easier to read in the table is a sign the figure has failed.
- Full-width SVG figures scale their text with their width. Inspect each figure at desktop width as well as phone width, and constrain the figure's maximum width or text size so labels stay in proportion at both.

Validate guided traces, trainer outputs and their prose against the stated algorithm/conventions as well. A convincing visual is a design reference until its model and claims have been checked; an attractive lab does not establish the accuracy of its entire lesson.

## 6. Examples and practical work

Every core workflow needs a complete worked example: initial data or state, the question, necessary setup, intermediate steps, result, interpretation, and conclusion. Prefer a small, inspectable example before realistic scale. Include a changed-context example to establish transfer beyond the original story.

For code, provide all imports, fixtures/data, required files, environment/run instructions, and visible results needed to reproduce it. Do not reference missing files such as a training script that the lesson never supplies. Separate commands from their output and explain both. Label version-dependent behavior and record tested versions as tested snapshots, not automatically the newest recommendation.

Verify displayed output with real execution where appropriate. Label illustrative output, normalized paths, fake clocks, rounding, seeds, timing variability, and hardware-sensitive values honestly. A browser teaching model is not an arbitrary-code runtime; describe that distinction once where it helps a learner choose how to practise.

### Use real data where the topic is about data

Constructed fixtures are right for hand calculation and for exposing a specific mechanism. They cannot supply the moment where a learner asks “does this method find anything in the world?” For any topic whose subject is a method applied to data, include at least one small, openly licensed, real dataset: introduce it with the question its collectors were asking, use it for the library-level example and diagnostics, and record its provenance, version and retrieval date in the lesson and the claim ledger. Embed the data in the runnable program or supply it as a file the lesson provides, so the program runs offline and the displayed output is reproducible. Keep the constructed fixture for the hand-traced steps and say why the two are different. When no suitable real dataset exists or licensing is unclear, say so in the design record rather than silently using only synthetic data.

### Displayed teaching code shows the mechanism

A runnable teaching program has two audiences: the reader following the algorithm and the verifier checking its contract. Serve the first on the page. Keep the algorithm's core compact and visible; put input validation in a small clearly named helper or a single finiteness check, and describe the gap between teaching code and a production library in prose. Never print explanatory or cautionary sentences from code. When a reviewer finds that a pathological input misbehaves, the default remedy is a sentence in prose or a check inside the verifier, not a guard woven through the displayed algorithm; add a displayed guard only when the failure is one a learner could plausibly hit. A displayed program whose mechanism occupies a quarter of its lines has stopped teaching.

For mathematics, show meaningful intermediate steps and explain why the operation is valid. Distinguish exact identities, estimates, approximations, and simulations. Preserve the assumptions under which the statement holds. If code numerically estimates a probability or solves an equation, make the relevant approximation and tolerance clear without overwhelming the primary lesson.

For operational topics, use disposable invented data and safe local fixtures. Explain effects on files, processes, or history before the learner runs commands that change them. Teach diagnosis and intentional recovery, not command copying as a substitute for understanding.

### Useful surprises and applications

Consider interesting logical/technical facts, counterintuitive consequences and applications during planning and while writing. Include them when they reveal how or why something works, expose an assumption, explain a design choice, connect fields or show a new situation in which the learner can use the idea. Familiar practical uses can establish relevance; less familiar uses can broaden transfer. Neither novelty nor popularity alone makes an example educational.

There is no required section, fixed number, minimum or maximum. Some introductory topics need only a simple concrete use. Later topics may benefit from several applications with distinct mechanisms, constraints or decisions. Add as many as materially improve understanding, with sufficient explanation for each; remove repetitive examples that merely change the story. Do not insert decorative trivia or repeated “fun fact” boxes to satisfy a checklist.

For an included application, establish the problem and why it matters, map the topic's entities/operations to that setting, walk through a small example or causal chain, interpret the result, and explain the relevant assumptions or limits. Add a prediction, changed condition, diagram or exercise when it helps the learner test that connection. A sentence listing industries or a link to an impressive project is not adequate treatment. For a surprising fact, explain why it is true and the misconception or consequence it clarifies; a full extra lab is not required.

Place the application next to the mechanism it illuminates, use it as a second worked example or independent task, or offer a clearly signposted deeper branch. A link to another topic should state the actual connection and prerequisite skills; label planned destinations honestly. If another topic offers the stronger explanation, route the idea there with a saved note. Preserve the continuing beginner example and reading flow rather than repeatedly switching context without explanation.

Verify factual, historical, empirical and real-world-use claims with sources appropriate to those claims. Distinguish a documented use from an original teaching scenario, a demonstrated result from a possible application, and a measured benefit from a hypothesis. Do not advertise invented scenarios as deployed systems or exaggerate a clinical, scientific or performance result for interest. Record sources and limitations in the same claim ledger used for the rest of the lesson.

## 7. Practice, feedback, and progression

DSA lessons additionally follow [the DSA practice standard](docs/teaching/DSA-PRACTICE-STANDARD.md). Include curated official LeetCode links for the actual mechanisms and transfer patterns taught, with independently attempted local exercises, optional hints, prerequisites for extensions, edge cases and readiness criteria. Verify the linked statements and metadata; balance familiarity with unseen/mixed transfer practice. There is no fixed count and no finite-list guarantee for all possible interviews. Future authors maintain the standard's evolving cross-topic pattern coverage map within their requested scope, saving unresolved ownership decisions for the relevant topic.

Distribute practice across the lesson and gradually reduce support:

1. **Prediction or retrieval:** anticipate a result, identify a quantity, or explain a diagram before the reveal.
2. **Worked and partially worked practice:** show one complete solution, then ask the learner to complete omitted reasoning or steps in a variation.
3. **Independent application:** solve a changed-input or changed-context task without following an identical recipe.
4. **Diagnosis:** identify why a plausible explanation, output, or implementation is wrong and repair it.
5. **Synthesis where appropriate:** combine the concepts in a small practical task with explicit success criteria.

This is a progression of support, not five mandatory exercises per subsection. Match volume and difficulty to the scoped outcomes.

Offer an optional hint before the complete answer where a task has substantial reasoning. Explain why the answer works and address tempting wrong approaches. For open-ended scenarios, give evaluation criteria and an example response; acknowledge multiple valid solutions. A final number alone is insufficient feedback.

A learner should have a chance to try before seeing the answer; avoid placing the answer directly in a prediction prompt. Plain answer reveals remain useful, but are not automated grading. If grading is added, evaluate meaningful behavior or reasoning under suitable criteria rather than exact wording alone.

Finish with a concise conceptual recap, retrieval prompts, and a readiness check: can the learner explain the mechanism, predict a change, complete a practical task, and identify a relevant limitation? Provide a next-topic link with the reason it follows. Include occasional review of prerequisite ideas in later lessons rather than assuming one exposure is sufficient.

## 8. Subject-specific adaptations

| Subject | Teaching flow and particular obligations |
| --- | --- |
| Mathematics/statistics | Question → intuition → notation → derivation → worked values → interpretation → assumptions/counterexample → transfer. Separate data variability, parameter uncertainty, and prediction where relevant. |
| DSA/discrete methods | Small problem → naive method → state trace → invariant → algorithm → correctness/termination → time/space analysis → edge cases and transfer. A passing example is not a proof. |
| Programming | Task → inputs → code → execution/state → output → explanation → edge case → practice. Show references, control flow, scope, and state transitions when hidden behavior is the difficulty. |
| Scientific computing | Start with the meaning and shape of the data. Trace transformations, dtype/index behavior, numerical limits, output checks, and the final scientific interpretation. |
| ML/model architecture | Trace data → representations/tensor shapes → computation → objective → training/evaluation → failure case. Use small examples, explicit assumptions, and comparisons or ablations with a clear question. |
| Systems/workflows | Show components, boundaries, state, data/control flow, evidence for diagnosis, and recovery. Separate model behavior from platform/version-specific implementation. |
| GPU engineering | CPU reference → work/data mapping → kernel → numerical and concurrency correctness → measurement → diagnosed bottleneck → measured change. Explain architecture restrictions and avoid unmeasured speedup claims. |
| Robotics/embodied intelligence | Task/environment → frames and units → observations/state estimates → decisions/actions → physical response → feedback/delay → failure/recovery. Distinguish simulation success from physical validation. |
| Neural engineering | Biological question → measurable signal → sensor/interface → acquisition and noise → analysis/decoding → closed-loop behavior → evidence, uncertainty and translation. Distinguish neural correlates, causal intervention and clinical benefit. |

These are adaptations within one educational system. They do not require identical lesson layouts or identical visual tools. The [domain playbook](docs/teaching/DOMAIN-PLAYBOOK.md) supplies deeper recipes, domain-specific failure modes, practice and assessment. A stored topic brief is an informed starting design, not a prohibition on a better explanation.

## 9. Research and accuracy

Research the particular weak explanation, uncertain claim, current API, or difficult visual model. Prefer official documentation, original papers, authoritative textbooks, standards, and maintained primary references. Use high-quality educational sites and videos for alternate explanations and representation ideas, then create original material appropriate to this curriculum.

Check that each source actually supports the associated claim. Record its URL, the concept it helped verify, and date/version where relevant. Do not claim to have watched a video or executed a reference tool unless that happened. Popularity alone is not evidence of accuracy or instructional effectiveness.

Research is not only verification of claims already written; it is also a coverage check against the field's canonical treatment. During stage 1, identify the standard reference for the topic (the textbook chapter, survey or specification a practitioner would name first, preferring legitimately free editions) and read its section list for the topic. Note ideas it treats that the draft omits, especially: the result that explains why the method is hard or why it is fast in practice, the historical origin when it clarifies a name or convention, incompatibilities between common implementations of the same idea, and the canonical counterexample. Decide for each whether it belongs in the core, a deeper branch, a saved destination note or nowhere, and record the decision. A lesson that is correct in every sentence can still miss the one fact a reader would meet on the first page of the standard text. Add the canonical reference itself to the learner-facing alternatives when it is accessible.

Reuse an inspected source for the same claim, scope and applicable version when it remains valid. Research a new claim or changed/version-sensitive behavior explicitly. Stop searching once the actual uncertainty is resolved and the needed alternate learning resources are assessed; accumulating more links is not a quality measure. Keep source locators and unresolved questions in the existing claim ledger so another author can retrieve the relevant passage without repeating the whole search.

Investigate conflicting definitions, assumptions, conventions, and nuanced cases. State the convention used. Keep consequential qualifications beside the claim: examples include statistical assumptions, nonunique solutions, indexing behavior, and shell/platform differences. Avoid words such as “always” or “guaranteed” without their conditions.

Sources should be optional for following the core walkthrough; a reference link must not replace a missing explanation. Separate evidence for technical correctness from inspiration for pedagogy.

### References & another way to learn it

Curate useful learner-facing alternatives as well as the technical claim sources. During every rewrite, look for good explanatory articles, worked tutorials, books/chapters, interactive exercises, and videos or YouTube playlists where they offer a helpful alternate explanation. Include well-matched resources in the lesson's existing “References & another way to learn it” section; do not keep them only in the author's research record. No format or link count is a quota, and a full playlist is not automatically better than one focused lesson.

Annotate each selected resource with creator/title, format, the particular concept or activity it helps, intended level or suggested point in this lesson, and important prerequisites/version/access caveats. Prefer a direct lesson, relevant chapter or creator's playlist over a channel/search homepage. Supply useful timestamps or playlist item names only when verified. Keep references navigable by separating alternate explanations/practice from precise API or claim references when that aids scanning.

Verify the destination and fit using the resource itself or its substantive transcript, companion notes/notebooks and chapter listing. Record exactly what was reviewed; never claim to have watched an entire video when only its notes or listing were inspected. Metadata-only discovery is not enough to endorse its technical details. Use primary documentation/research to check current semantics; older videos can remain useful for intuition with a clear warning about changed APIs/defaults. Avoid unsupported rankings, popularity-as-proof and link dumps. Label paid/sign-in requirements when known and favor accessible alternatives. A video must never be the only way to follow a core explanation or practice task.

Research informing this standard, reviewed 9 September 2026:

- [IES practice guide](https://ies.ed.gov/ncee/wwc/PracticeGuide/1): supports alternating examples and practice, combining verbal/graphical explanations, connecting concrete and abstract representations, retrieval, and explanatory questions. Evidence ratings differ by recommendation; this broad 2007 synthesis does not prove a particular website template effective.
- [PhET original simulation design research](https://phet.colorado.edu/publications/archive/Phet%20Interview%20Paper.htm): informs clear initial states, understandable controls, meaningful responses, and novice walkthroughs. Findings arise primarily from science simulations and student interviews, not universal tests of all technical subjects.
- [Seeing Theory creator's account](https://blog.cs.brown.edu/2018/01/22/seeing-theory-teaching-statistics-through-interactive-web-based-visualizations/) and [frequentist chapter](https://seeing-theory.brown.edu/frequentist-inference/index.html): illustrate sequenced visual intuition and distinct explorations for connected concepts. This is a design reference, not proof of mastery.
- [Python Tutor](https://pythontutor.com/index.html): demonstrates visible execution, objects, references, and stack frames. Borrow the principle of exposing hidden state, not its interface or promotional effectiveness claims.

The framework here is a project-specific synthesis of the user's requirements, local lessons, and these sources.

## 10. Accessibility and visual quality

Preserve keyboard access, visible focus, readable labels, meaningful heading order, and clear table semantics. Do not communicate a distinction by color alone: add text, values, shapes, or line patterns. Give diagrams an explanatory text equivalent and accessible exact data where needed.

Validate mobile layouts and zoom: the controls, relevant input, visual consequence, and explanation should stay understandable together. Recompose or sequence a complex visual instead of merely shrinking its text. Avoid hover-only instructions. Use reduced-motion support and manual stepping when motion carries teaching information.

Visual polish serves comprehension: highlight the current relationship, minimize competing emphasis, keep explanations near the referenced element, and avoid a repeated wall of equally weighted boxes. Do not mistake a consistent component style for a consistent teaching experience.

## 11. Validation and definition of ready for review

This section defines complete implemented-lesson readiness in phase two. At a content-only boundary, apply the research, reasoning and manuscript-completeness requirements above and explicitly defer the implementation checks; do not claim this section has passed.

Assess each revised lesson on separate axes. Do not infer one from another:

| Review | Evidence to collect |
| --- | --- |
| Coverage/pedagogy | Updated scope/title and coverage map; saved discoveries considered; prerequisite continuity; plain explanation; complete example; meaningful visual support; interpreted result; independent practice; useful feedback; justified applications/connections where useful; next-step route. |
| Accuracy | Claim/source checks; assumptions/conventions; numerical and code outputs; relevant counterexamples and boundary cases. |
| Interaction | Controls change the intended model; diagrams and numbers agree; reset/back and invalid states work; the investigation teaches its stated question. |
| Browser/accessibility | Desktop and narrow-screen inspection; keyboard/focus; text alternatives; readable labels; valid anchors; no blocking overflow or rendering errors. |
| Learner experience | An actual beginner walkthrough when available: ask for a prediction, explanation, diagnosis, and transfer. Otherwise label the review as an author's heuristic assessment, and run the learning-experience checklist below in full. |
| Voice and reading load | Hedging density, a stated first-pass route, connections made explicit, displayed code that shows its mechanism and prints no disclaimers. |

### Learning-experience checklist

Correctness review and learning-experience review are different activities and are recorded separately. A lesson can pass every numerical oracle and still teach poorly. The author runs this checklist before handing off and the independent reviewer runs it again; both record concrete findings, not a pass mark.

1. **Route.** Is there a first-pass route after the introduction? Are deeper branches labeled where they begin?
2. **Cautions.** Is each important caution stated once in a clear home and referred back to, rather than repeated after every result? Does any code print a cautionary sentence?
3. **Real question.** Does the lesson open with a concrete situation a reader can care about, and return to it with a result the reader can judge? For data methods, is there real data?
4. **Labs as investigations.** For each lab: is a prediction recorded and checked? Does at least one control change something the prose has not already resolved? Was the fixture run under every offered alternative to confirm the promised contrast, including the case where no contrast should appear?
5. **Figures.** For each quantitative figure: is the claimed feature visible at rendered size on desktop and phone? Is the baseline present? Does the caption ever apologize for the figure?
6. **Connections.** Where the same result appears by two routes, is the link stated? Are the canonical reference's headline facts present or deliberately routed?
7. **Code.** In each displayed program, does the mechanism occupy most of the lines? Is validation separated and minimal?
8. **Practice.** Do the exercises change the numbers and the context, and does at least one give the learner exact values to reproduce after an independent variation?
9. **Screenshots.** Were the informative states captured and looked at, not only the default state: the fixture that shows the contrast, the lab after a prediction is checked, the figure at full desktop width?

Include an explicit inline-visual reading pass in coverage/pedagogy review: inspect introduction of structures, alternative cases and intermediate transformations, separately from checking that lab controls work. Record concrete omissions and improvements rather than reporting only a lab count. The policy above governs when an inline figure is needed; it does not create a diagram-per-section quota.

Check the fit of each representation as well: can a beginner identify the topic's actual entities, perform or follow its operation, and explain the visible consequence? Revise a generic layout when it hides that mechanism, and retain a repeated format when it remains the clearest fit. Confirm that quantitative figures and their adjacent conclusions match the recorded evidence category and scope.

Run checks appropriate to the actual change. For implemented code/numerical models, test meaningful behavior and compare results against an independent reference or real runtime, not merely the same fixture on both sides of an assertion. Run the relevant project build and browser checks after lesson/component changes. Documentation-only updates do not need a new application build.

### Keep verification bounded and reusable

Plan the checks needed for the topic's actual claims, models and interactions, run them, resolve material findings, and record the reviewed source version. A passing recorded check remains evidence for that unchanged version; resuming a session does not make it stale. Read its result and unresolved findings instead of automatically running it again. Do not reopen completed modules or launch historical integration scripts merely because their artifacts still exist.

After a fix, rerun the affected behavior and any dependencies it can change. A CSS label repair needs focused visual/keyboard checks; it does not require regenerating unchanged Python examples. A numerical model repair needs relevant numerical and displayed-result checks; it does not reopen unrelated topics. Broaden testing when a concrete failure, shared change, source-version mismatch, environment change or unresolved concern justifies it. Perform the required final build and integration once the authorized increment is ready, rather than after every documentation or ledger edit.

An independent reviewer should assess complementary correctness and teaching risks, report a bounded set of actionable findings, and close them with targeted evidence. Do not duplicate the author's entire run by default, repeatedly manufacture new review stages, or pursue inputs outside a declared supported model without a specific reason. Retain sufficient depth and accuracy; avoid turning evidence production into a separate expanding project.

Choose complementary checks for specific failure risks: an independently derived small case, a counterexample to an assumption, a change of units, a permutation/relabeling invariant, a limiting case or a comparison with an independent implementation. State why the expected relationship holds. Two views or programs calling the same helper establish agreement, not independent correctness. These examples are options, not a checklist to apply to every lesson. Scope edge cases to the lesson's claims and supported inputs; extra exotic cases need an identified risk.

Keep current phase status in the delivery ledger and a concise next action in its linked record. Detailed passed results belong in the linked evidence record. Content-first drafts awaiting implementation are required handoff inputs, not disposable scratch. Other temporary drafts, patch scripts and superseded captures are not instructions; follow the [scratch retention policy](docs/engineering/LEARNING-CODE-STANDARD.md#temporary-work-and-evidence-retention) and remove disposable working material when its job is done.

Check agreement between every linked representation at intermediate steps, not only the final answer: if fault handling updates a mapping, its table must update with the diagram; if copying retains old storage, show which representation still holds the authoritative sequence. Inspect actual screenshots as well as bounds checks—labels can remain inside an SVG yet spill outside their node or intersect an arrow. Fix those teaching ambiguities and repeat the affected checks.

Historical reports identify older verification scripts and their dated scope. Consult those only when changing the covered behavior; they are not an automatic test queue for new lessons. Current checks must reflect the topic's actual investigations rather than assume one lab per page.

Do not mark a lesson ready if the core example is incomplete, a material claim is unverified, a required mechanism remains unexplained, or learners have no way to assess their practice. Record remaining concerns explicitly. Passing a build, meeting a word count, registering a route, and clicking completion are not teaching-quality certificates.

### Source-bound evidence and review handoff

Preserve the original lesson baseline and its useful coverage once, using the existing increment record. Attach each completed review to the exact relevant source version, using commit/file hashes as appropriate. A hash proves identity, not correctness. Reuse a passing result only while its relevant code, data, assumptions and environment remain applicable; a byte-identical model cannot validate a newly added claim or changed surrounding interpretation.

Give the reviewer a compact summary in the existing topic record: scope/design link, current source paths and versions, checks actually passed with relevant environment/commands, open findings and retained evidence links. Report a finding with its location, learner or correctness consequence, supporting example and closure check. The reviewer still reads the complete lesson and relevant implementation; the summary replaces rediscovery of history, not substantive review.

Finalize source changes and selected-evidence retention before handing the version to final integration. Keep recorded executions faithful to the version actually run. For a later correction, record the changed files, reason, affected dependencies and checks rerun or reused; do not rewrite an old result to imply a new execution. Preserve an exact prior source or a reproducible difference when needed to establish conservation, without copying the whole lesson and every attachment after each edit. One increment owner coordinates shared integration after its topic versions are ready.

Efficiency must preserve the scoped explanations, independent practice, accuracy and actual visual review. Use the code standard's [context and coordination rules](docs/engineering/LEARNING-CODE-STANDARD.md#efficient-context-tools-and-coordination) to reduce repeated reading/output and unnecessary work. If reporting efficiency, use available actual usage or coarse process counts already recorded (for example, full native reruns and late corrections); mark unavailable token usage as unknown. Do not invent a savings percentage or build a separate measurement campaign unless requested.

## 12. Delivery and handoff

For each authorized increment, report what was weak, what changed and why, what useful existing content was preserved, coverage and intentional boundaries, any title/scope decision, worthwhile applications and their learning purpose, discoveries saved for other topics, visuals and what to try, practice and solutions, research used, validation actually completed, and remaining limitations. If no title change or extra application is justified, say so briefly in the design record; do not manufacture changes. Provide direct lesson/source links.

Maintain separate status fields: **implementation**, **computational verification**, **browser/visual review**, **user review**, and **next action**. Record unknown states as unknown. A test script's existence is not a passing result; a source comment claiming checks were run is not a fresh verification in the current session.

Update [the current handoff](LESSON-AUTHORING-HANDOFF.md), the relevant topic brief and an increment-specific evidence record when work advances. Do not append new instructions to archived pilot/batch reports. Replace superseded active instructions rather than accumulating competing versions. Expand the rollout only within the user's authorized scope.

## 13. Curriculum and authoring plans

Every catalogue entry must belong to a coherent module. Guided paths reuse topic IDs and include recorded prerequisites; reference glossaries and frontier surveys are not substitutes for foundational teaching. Distinguish a missing topic from a listed topic needing design, an unpublished detailed brief, and a published lesson needing review.

The module's section/topic order owns learner progression. A path visits its selected modules in declared order and keeps each module's selected topics in that same syllabus order. Do not globally sort lessons by difficulty or inject individual prerequisite lessons between module topics. Recorded dependencies add supporting topics to the route and provide explicit review links; they do not silently rewrite reading order. Additional prerequisite modules follow the selected modules in catalogue order. This is not a guarantee that all prerequisite edges precede their dependents: inspect and teach/link the specific readiness requirements. Any pedagogical change to a module's topic order must be explicit in the catalogue, with preserved identities and a reviewed rationale.

Publication does not alter teaching order. Previous/Next name adjacent entries in those module contents, including planned lessons; default entry resumes the first unfinished unique topic and resolves to a stable URL. Shared lessons may occur in several module views with one ID/progress record; navigation retains the chosen module context so selecting a shared lesson does not jump to its first occurrence elsewhere. Keep the sidebar unnumbered and show position within the current module's selected topics in the header. Keep compact topic/completed counts with matching scope, and derive hub counts from the same resolved route. Prerequisite links and deliberate browsing remain available. Never substitute a later published lesson for the next entry.

A topic-specific brief records scope, observable outcomes, exact prerequisite topic titles, teaching sequence, a visual question and interaction, practice and success criteria, misconceptions, sources, core/specialist/frontier depth, and current-information review focus. One `visual` field proposes an initial representation; authors must add separate visual contracts wherever another mechanism needs them. The [design brief](docs/teaching/TOPIC-DESIGN-BRIEF.md) expands this into a complete lesson plan before implementation.

For older entries without an individual brief, the generated inventory identifies the relevant domain strategy and explicitly marks individual design as required. Do not represent inherited domain guidance as completed bespoke research. Design that topic before writing it. Keep unverified prerequisite coverage visible rather than implying that an algorithmic ordering proves every learner dependency is known.

Coverage is a maintained map of established foundations, professional applications and specialist/frontier branches as of a recorded date. It is not a claim to contain every fact, guarantee professional competence, or replace supervised hardware, clinical or research practice. Check current primary sources again at authoring time, especially hardware support, APIs, datasets/benchmarks, safety standards and regulations. Add a newly discovered omission as a sequenced topic or an explicit scoped extension.
