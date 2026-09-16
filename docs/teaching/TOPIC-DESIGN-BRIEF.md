# From a syllabus entry to a teachable lesson

Use with the [standard](../../LESSON-TEACHING-STANDARD.md) and [domain playbook](DOMAIN-PLAYBOOK.md). Complete the design for the requested topic before implementation; this is ordinary authorized authoring work, not a separate approval gate. Improve an existing brief instead of starting over. Keep the completed design with the topic or its increment record.

The standard's [delivery modes](../../LESSON-TEACHING-STANDARD.md#delivery-modes-and-stopping-boundaries) and [six-stage workflow](../../LESSON-TEACHING-STANDARD.md#six-stage-authoring-workflow) own the process. Record the requested mode and current revision first. Sections 1–5 support the complete written lesson and specifications; section 6 belongs to implementation/review; section 7 has a content-only or full-delivery handoff as authorized. In the existing record, identify the phase, owner, intended checks, deferred work and next action. Roles may overlap except that an author's own review must not be described as independent.

## 1. Establish the learning contract

Record topic title and stable ID, domain/module, intended learner, prior skills, scope and excluded follow-on topics. State observable outcomes: explain a mechanism, interpret a representation, predict a change, carry out a task and recognize a limit. Do not use “understand X” as the only success criterion.

Run `node scripts/build-curriculum-inventory.mjs --topic "Exact topic title or stable ID"` and read its `authoringNotes`, including the destination file and relevant unresolved routing notes. Follow the [topic-note workflow](topic-notes/README.md). An existing brief or completed earlier design does not override a new, well-supported discovery.

Review scope before writing, during research/implementation when new ideas emerge, and before delivery. Search plausible related owners by title, synonyms and mechanism, then inspect their actual briefs/content; do not audit every topic. Record substantive findings in this compact map:

| Idea / subtopic | Current coverage and evidence | Best owner and why | Decision / depth / prerequisite effect | Durable destination |
| --- | --- | --- | --- | --- |
| A meaningful gap, connection or application | Absent / mentioned / planned / taught / verified, with exact source or section | Current lesson, another existing topic, or proposed new topic | Include, deepen, bridge/link, route or omit with reason | Current section/brief, destination-topic note, or unresolved inbox |

If an idea best belongs here, include it with the support it needs even when discovered halfway through writing. If another topic offers more benefit, save its instruction and reasoning there immediately instead of relying on the current conversation. Do not silently rewrite that unrequested lesson. The receiving author must reassess the evidence and record inclusion, adaptation, rerouting, deferral or rejection.

Record a title decision: retain the current title with a reason, or give the revised title and the materially changed scope it promises. Rename when warranted at any stage; do not append every new example to the title. Preserve useful coverage and stable IDs/progress. Follow the standard's compatibility guidance: IDs are currently title-derived in multiple consumers, so editing only the title or adding an unused `id` field can break routes, registry lookup, exact-title prerequisites and notes. A real rename must update the relevant consumers compatibly and verify old links, shared progress, prerequisite membership/links, module reading order and conservation.

Check prerequisite skills against actual lessons. A title in the catalogue is not evidence that its lesson teaches the required skill. Add a short refresher or an honest link to planned prerequisite material when necessary. If a major foundation is absent, add a scoped prerequisite topic rather than silently requiring outside knowledge.

Identify a practical anchor problem and small consistent dataset/state. Record why the result matters and what decision follows it. Specify the beginner finish line, intermediate applications and advanced branches. Plan reading and practice time separately after the material exists; no arbitrary word or lab count.

For a topic whose subject is a method applied to data, also record a **real dataset candidate**: a small, openly licensed dataset with the question its collectors asked, its provenance and how the program will carry it offline. Constructed fixtures remain for hand calculation. If no suitable dataset exists, record why. Then record the **canonical reference** for the topic (the chapter a practitioner would name first, preferring a legitimately free edition), read its section list, and list the ideas it treats that the current plan omits, with a decision for each. Plan the lesson's **first-pass route** here as well: which sections a first reading covers and which are deeper branches.

## 2. Map every conceptual hurdle

| Outcome / hurdle | Required prior idea | Plain explanation and exact mechanism | Complete example and interpretation | Representation and learner question | Prediction / independent practice / feedback | Core or deeper |
| --- | --- | --- | --- | --- | --- | --- |
| Complete this for each actual hurdle | Name the skill and its lesson or local bridge | Explain causality, not only terminology | Include inputs, intermediate states and result | Choose static, stepped or interactive support | Specify evidence that distinguishes understanding from copying | Keep required reasoning in core |

Write the core route in concept order. Explain the correspondence between prose, symbols, visual entities, code and output. Record vocabulary, symbols, units, shapes and conventions at first use. A new module may repeat the teaching loop; it need not wait until the final “practice” section to ask a question.

Linux example: “Why can I read a known filename but not list the directory?” needs directory-name listing versus path traversal, then file read permission. A sequence of gates makes the first failed check visible. Predict the outcome with directory read disabled but search/file read enabled; observe that file access can succeed, then diagnose a changed case. An octal table alone does not expose the mechanism.

## 3. Design each visual separately

First map where visuals belong in the prose, not only which labs will be built. For each important structure, contrast or transformation, record whether the existing explanation is sufficient or whether it needs an immediately visible figure, a stepped example, an interactive investigation, or a combination. Give each chosen representation a location and distinct learning purpose. A relevant initial lab view can suffice; a later state hidden behind controls may not introduce the relationship clearly.

Choose from the concept's objects, relationships and operations before selecting a component. Briefly explain why the chosen spatial arrangement, terminology and interaction expose this hurdle. Adapt subject conventions with plain-language introductions and legends. Reuse a familiar format when it works; vary formats within or between lessons when that improves understanding. Neither identical lab layouts nor novelty are requirements. Follow the standard's “Let the concept determine the visual form” guidance.

For every visual, including a static inline diagram, record:

- The precise question and misconception it resolves.
- Initial state, entity labels, units/scales, what appearance encodes and what is merely layout.
- Placement beside the relevant explanation, correspondence to its concrete example, what the reader should notice and any simplification.
- Accessible description and narrow-screen composition that preserves relationships and readable labels.
- Annotation/layout contract: which text reflows outside the geometry, room between successive stages, equal-scale comparisons, and the desktop/phone/intermediate-width states to inspect. Plan separation of labels from foreground marks; do not rely only on the outer SVG bounds or a fixed row-height guess.
- Independent numerical/runtime/reference checks for the values and relationships it asserts, as appropriate.
- For a quantitative graph: whether it is calculated, modeled, simulated, measured, sourced or illustrative; its equation/generator/data/source, assumptions and verification. Record benchmark conditions and uncertainty for empirical comparisons. Do not give invented curves real implementation names or imply verified rankings.

For an interactive investigation, additionally record learner prediction, controls, held-fixed quantities, visible consequences, step/reset/back behavior, invalid states, causal feedback, transfer task and simulation boundaries. Specify the topic's operation the learner performs, when edited input takes effect, and how connected views stay consistent with the active model. Static figures do not need invented controls or simulator code to satisfy the design brief.

Record three things the standard checks at review for every investigation: how the prediction is **recorded and compared** with the model's result (a control, not a caption); which control lets the learner **act on the topic's entities** beyond choosing among described presets; and the **fixture check**, listing the alternative settings that were actually run to confirm the fixture shows the promised contrast and the null case in which it correctly shows none. If the natural running example cannot show a contrast the prose promises, specify a second fixture built for it with exactly representable critical values. For quantitative figures, record how the claimed feature was confirmed to be visible at rendered desktop and phone size, including the axis transform and baseline point chosen.

The catalogue's `blueprint.visual` is the first proposed investigation, not an instruction to limit the lesson to one lab. Add more contracts for other hurdles. Prefer static diagrams for relationships that do not benefit from manipulation. Keep exact-value inspection available alongside intuitive geometry or flow. Randomized simulations need seeded/repeatable comparisons.

## 4. Design complete examples and practice

For each example specify fixtures/data, environment, full inputs/code or mathematical setup, intermediate reasoning, expected output, interpretation and relevant failure case. Verify all displayed numbers and runnable results. Label approximations, rounding, simulated/illustrative values and hardware/version dependence.

Sequence support from a worked example to a partially guided variation and independent transfer. Include diagnosis or a counterexample where misconceptions matter. Give a hint before a full solution; the solution explains the reasoning and tempting wrong route. Open tasks need evaluation criteria and an example acceptable answer. A capstone combines already-taught skills and has a reproducible success check, not unexplained prerequisites.

Map each promised outcome to at least one meaningful evidence opportunity. Avoid exercises that only repeat the exact demonstrated input. Completion buttons are reading bookkeeping, not automatic mastery assessment.

For DSA, also map outcomes and interview patterns to curated official LeetCode statements using [DSA-PRACTICE-STANDARD.md](DSA-PRACTICE-STANDARD.md). Plan the learner's attempt, optional hint, transfer variation and edge cases for each selected task; distinguish core readiness from extensions with later prerequisites. Record verified title/number/difficulty/access and ownership of overlapping patterns. Update the coverage map as topics are authored without turning each rewrite into a full-catalogue audit or imposing a question quota.

### Plan useful facts, applications and connections

Consider both familiar and less familiar uses, surprising consequences and illuminating technical facts. Ask what each candidate helps a learner explain, predict, decide or transfer. Include none, one or several according to the topic's needs; this is an authoring decision, not a fixed learner-facing section. An early foundations lesson can use a very simple application; later material can support richer and more varied settings.

For each selected candidate, record its learning purpose, required prior ideas, problem/setup, exact mapping to this topic's mechanism, worked reasoning/result, assumptions or failure limit, source/verification status, and suitable placement. Specify a useful visual, prediction or independent variation when appropriate. Develop it far enough to explain the connection; do not substitute an industry list, trivia, namedropping or an unexplained external link. Several examples should add distinct understanding rather than repeat a method with renamed variables.

Keep essential applications in the core; use optional depth when new domain knowledge would otherwise interrupt the first pass. A verified technical fact may need only a concise explanation and consequence. A substantial application may deserve multiple steps, a diagram or practice. Route candidates whose learning purpose fits another topic to its saved notes. Before delivery, remove weak additions, explain deliberate omissions briefly in the design record, and preserve useful unresolved candidates for their correct owners.

## 5. Research claims and maintain currency

Create a claim ledger with claim/convention, source URL and locator, what was verified, retrieval date, relevant version/device/jurisdiction and unresolved uncertainty. Use primary documentation, original research and authoritative teaching sources. Educational videos can inspire a representation; record whether a video, transcript or accompanying notes were actually reviewed.

Resolve competing definitions and contradictory results. Keep qualifications next to the associated claim. For empirical comparisons, record data, split, baseline, metric, uncertainty and scope. A paper's abstract or marketing page is not sufficient evidence for all its implementation details. Recheck moving `/latest` documentation before writing, and do not infer release dates from a current page alone.

The expansion's source URLs are research starting points with a coverage rationale, not blanket verification of every sentence a future author writes. A future lesson must verify its specific claims and implemented behavior.

Curate the learner-facing “References & another way to learn it” section alongside the claim ledger. Seek appropriate articles/tutorials, interactive practice, books and videos/YouTube playlists; include the formats that actually help this topic, with no fixed number. For each selected resource record a direct URL, creator/title, format, concept and learner level, what was actually reviewed (page, transcript, notebook, video segment), verification date, relevant version/access caveats and any verified chapter/item/timestamp guidance. Publish a concise usefulness annotation beside the link. Distinguish current technical references from older alternate explanations, and keep the core lesson self-contained. Follow the standard's resource-selection rules.

## 6. Implement and verify

Start this section only after complete content exists and full implementation or continuation is authorized. For **content first**, write the complete manuscript and detailed specifications using sections 1–5, then use the content checkpoint in section 7 and stop. Diagrams/labs are specified at that boundary, not implemented in the website. Do not mistake completing this brief for writing the full manuscript. New rendered assets, lab/model code and comprehensive execution/browser checks belong to this implementation phase.

Use the established content/lab components where appropriate. Separate a nontrivial interaction model from rendering so its behavior can be independently checked. Preserve stable IDs, progress and existing useful content. Follow [the learning code standard](../engineering/LEARNING-CODE-STANDARD.md) for semantic names, ownership and on-demand loading. Show planned entries as planned until complete content is registered in `src/learn/data/lesson-manifest.json`; regenerate and verify the browser artifacts. The old eager `topics/index.js` registry is removed.

Apply the code standard's [early compatibility checks](../engineering/LEARNING-CODE-STANDARD.md#early-compatibility-checks) at the first runnable draft. Before independent review, record what the author actually checked and any open findings. Give the reviewer the design, current source and existing evidence; ask for complementary teaching/correctness review. Resolve the resulting findings with targeted checks, then identify the final reviewed version. Reuse unchanged native and browser evidence; final shared integration belongs to the increment owner. Do not add repeated full rewrites or test campaigns solely to complete these stages.

Use the standard's [source-bound evidence handoff](../../LESSON-TEACHING-STANDARD.md#source-bound-evidence-and-review-handoff) in this existing record. Select verification cases for the lesson's actual failure risks and name what each establishes; do not equate more fixtures with more confidence automatically. Record changes that invalidate earlier evidence and finalize the source and selected attachments before integration. Keep the design, evidence and next action linked rather than duplicating them into a second workflow document.

Verify the final scope/title and disposition of relevant saved notes; scoped outcomes and prerequisite continuity; factual/application claims and numeric/code behavior against appropriate sources or an independent reference/runtime; controls/feedback/reset/edge cases; desktop and narrow-screen rendering; keyboard/focus/text equivalents; project build. Test what can fail meaningfully, not only the implementation's own duplicated fixtures. A source comment, a saved script or a build pass is not a fresh accuracy check.

Run a novice walkthrough when possible: can the learner explain the picture, predict a change, perform the task and diagnose an error without hidden help? Label author-only evaluation honestly. Fix blockers before declaring the lesson ready for review.

Review representation fit, not just interaction correctness: do the chosen layout, terms and controls reveal this topic's mechanism? Keep reuse that aids understanding; adapt a generic panel that conceals it. Check that each graph's caption and surrounding conclusions stay within its verified data or stated model assumptions.

Also read the page without operating the labs: are the entities, comparisons and transformations understandable as they are introduced? Inspect compact inline illustrations separately from interactive correctness. If a useful diagram is missing, record the exact concept, example and placement and implement it within the requested lesson scope. Do not infer sufficient visual teaching from a fixed number of completed labs, or add a redundant picture only to meet a count.

Run the standard's [learning-experience checklist](../../LESSON-TEACHING-STANDARD.md#learning-experience-checklist) as a distinct pass from correctness verification, and record its findings separately. Two items are commonly skipped and must not be: a hedging pass over the whole manuscript, and screenshots of the informative states (the fixture that shows the contrast, a lab after its prediction is checked, each full-width figure at desktop width) rather than default states only. When a reviewer's finding concerns a pathological input, resolve it in prose or in the verifier before touching displayed teaching code.

## 7. Deliver and hand off

For content-first delivery, save `docs/teaching/drafts/<stable-topic-id>/lesson.md` and `visual-specifications.md` plus any necessary topic-owned draft inputs. Link the actual design/research record. Include full prose, worked examples, practice/hints/solutions, references and intended visual placement. Give each visual/lab an actionable state/data/interaction contract, rationale, accessible/narrow layout and verification plan. Mark output as derived/predicted or executed accurately. List deferred checks and unresolved implementation questions; do not leave material gaps in the written core.

Update only the content phase to complete with its exact file hashes in [the delivery ledger](lesson-delivery-progress.json), following [the schema](LESSON-DELIVERY-LEDGER.md). Implementation remains not started. A different agent receives this complete packet through the topic command and runs `--work finish` before proceeding. Preserve pending drafts during cleanup. For full delivery, continue into implementation and complete both phases without another approval gate; avoid duplicating the full manuscript if production source already serves as its content checkpoint.

Record implementation, computational checks, browser/visual review, user review and next action separately. Report the original weakness, changed learning experience, important depth retained, coverage/title decisions, applications and their purpose, what to try, practice/solutions, sources, checks actually run and remaining limits. Update the current handoff, topic brief and relevant note dispositions; link ideas routed to future topics. Open notes remain open until reasoned resolution. Keep historical records non-authoritative.

## Stored catalogue schema

New expansion entries use `title`, `level` and `blueprint`:

```js
{
  summary: "Scoped problem and why this lesson matters",
  outcomes: ["Observable skill", "Observable skill"],
  prerequisites: ["Exact topic title already in the catalogue"],
  sequence: ["Concrete teaching stage", "Mechanism", "Application", "Diagnosis"],
  visual: { type: "Representation", question: "Learner's question", interaction: "What changes and what is observed" },
  practice: { task: "Independent task", success: "Observable checks and reasoning" },
  misconceptions: ["Specific tempting mistake to address"],
  sources: ["Primary source URL"],
  depth: "core", // or specialist / frontier; distinct from difficulty level
  reviewFocus: "Claims, conventions or versions to recheck before authoring"
}
```

This is a compact topic-specific plan. The completed coverage/ownership map, evolving title decision, outcome table, application designs, additional visual contracts, claim ledger, full examples and solutions belong in its linked design record. Destination notes live in `docs/teaching/topic-notes/<stable-topic-id>.md` and are surfaced by the topic-plan command even when the topic has no individual brief. They do not require speculative fields in the learner-facing blueprint. Older entries that only inherit a domain strategy are explicitly marked `individual-design-required` in the generated inventory. Do not claim that all older topics already have bespoke plans or verified prerequisite edges.
