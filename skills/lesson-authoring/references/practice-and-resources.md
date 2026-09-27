# Practice And Resources

Part of [$lesson-authoring](../SKILL.md). Repository paths below are relative to the selected application checkout; see [the repository adapter](portfolio-adapter.md). Load only the sections needed for the selected mode.

- [7. Practice, feedback, and progression](#7-practice-feedback-and-progression)
- [Keep recurring endings recognizable](#keep-recurring-endings-recognizable)
- [9. Research and accuracy](#9-research-and-accuracy)
- [Further learning and technical references](#further-learning-and-technical-references)

## 7. Practice, feedback, and progression

DSA lessons additionally follow [the DSA practice standard](dsa-practice.md). Include curated official LeetCode links for the actual mechanisms and transfer patterns taught, with independently attempted local exercises, optional hints, prerequisites for extensions, edge cases and readiness criteria. Verify the linked statements and metadata; balance familiarity with unseen/mixed transfer practice. There is no fixed count and no finite-list guarantee for all possible interviews. Future authors maintain the standard's evolving cross-topic pattern coverage map within their requested scope, saving unresolved ownership decisions for the relevant topic.

Distribute practice across the lesson and gradually reduce support:

1. **Retrieval and explanation:** identify a quantity, explain a diagram, or reason through a changed case in an independent exercise.
2. **Worked and partially worked practice:** show one complete solution, then ask the learner to complete omitted reasoning or steps in a variation.
3. **Independent application:** solve a changed-input or changed-context task without following an identical recipe.
4. **Diagnosis:** identify why a plausible explanation, output, or implementation is wrong and repair it.
5. **Synthesis where appropriate:** combine the concepts in a small practical task with explicit success criteria.

This is a progression of support, not five mandatory exercises per subsection. Match volume and difficulty to the scoped outcomes.

Offer an optional hint before the complete answer where a task has substantial reasoning. Explain why the answer works and address tempting wrong approaches. For open-ended scenarios, give evaluation criteria and an example response; acknowledge multiple valid solutions. A final number alone is insufficient feedback.

A learner should have a chance to attempt an independent practice task before opening its solution. Optional hints and solution disclosures belong to that practice; they must not hide a lab's live output. Do not add a predicted-answer widget or correctness comparison to the lab under the label of optional practice. Separate practice assessment should evaluate meaningful behavior or reasoning, not exact wording alone.

Check that all lab representations, including form-control values, selected options, diagrams and accessible descriptions, describe the same current inputs. Counterfactual editors show their actual computed consequences immediately. When conventions or units change, update the table, plot and causal explanation together. A retained comparison keeps its original inputs and units; establish a valid baseline when the earlier quantity is undefined rather than inventing an “unchanged” result. Check that historical prediction gates have been removed from the controls, state model, surrounding prose and test assumptions.

Finish with a concise conceptual recap, retrieval prompts, and a readiness check: can the learner explain the mechanism, predict a change, complete a practical task, and identify a relevant limitation? Provide a next-topic link with the reason it follows. Include occasional review of prerequisite ideas in later lessons rather than assuming one exposure is sufficient.

### Keep recurring endings recognizable

Use the shared lesson-ending presentation for practice, further learning, technical references and readiness/next steps, with explicit boundaries that reflect the authored content. Keep these purposes distinct: practice invites an attempt, feedback explains it, further learning offers another explanation or activity, references support precise claims or API behavior, and readiness connects demonstrated skills to the next topic. A mixed legacy heading does not make every paragraph or link serve the same purpose. Preserve useful annotations and their relationship to the resource they qualify.

Keep each exercise's prompt visible. Where authored separately, present optional hints and complete solutions as separate native disclosures with clear labels; preserve combined feedback or visible worked answers when changing presentation alone. Keep all code, diagrams, evaluation criteria and explanation within their original task. Do not hide live lab output or invent missing hints, solutions, resource metadata or readiness claims to fill a template.

Share section and exercise styling without requiring identical titles, counts, order or scientific content. Topic-specific investigations and deeper teaching branches remain part of the lesson body. Preserve existing heading IDs, authored numbering, links and progress identity. Plan explicit ending roles during authoring and retain them through regeneration; the implementation contract is in the learning code standard (repository path: `docs/engineering/LEARNING-CODE-STANDARD.md#shared-lesson-endings`).

## 9. Research and accuracy

For a requested check of current research/models against existing curriculum coverage, use [research updates](research-updates.md). It defines evidence, importance, adoption and placement decisions before authoring changes. Ordinary lesson research remains scoped to the topic; this reference does not require a full curriculum scan on every rewrite.

Research the particular weak explanation, uncertain claim, current API, or difficult visual model. Prefer official documentation, original papers, authoritative textbooks, standards, and maintained primary references. Use high-quality educational sites and videos for alternate explanations and representation ideas, then create original material appropriate to this curriculum.

Check that each source actually supports the associated claim. Record its URL, the concept it helped verify, and date/version where relevant. Do not claim to have watched a video or executed a reference tool unless that happened. Popularity alone is not evidence of accuracy or instructional effectiveness.

Research is not only verification of claims already written; it is also a coverage check against the field's canonical treatment. During stage 1, identify the standard reference for the topic (the textbook chapter, survey or specification a practitioner would name first, preferring legitimately free editions) and read its section list for the topic. Note ideas it treats that the draft omits, especially: the result that explains why the method is hard or why it is fast in practice, the historical origin when it clarifies a name or convention, incompatibilities between common implementations of the same idea, and the canonical counterexample. Decide for each whether it belongs in the core, a deeper branch, a saved destination note or nowhere, and record the decision. A lesson that is correct in every sentence can still miss the one fact a reader would meet on the first page of the standard text. Add the canonical reference itself to the learner-facing alternatives when it is accessible.

Reuse an inspected source for the same claim, scope and applicable version when it remains valid. Research a new claim or changed/version-sensitive behavior explicitly. Stop searching once the actual uncertainty is resolved and the needed alternate learning resources are assessed; accumulating more links is not a quality measure. Keep source locators and unresolved questions in the existing claim ledger so another author can retrieve the relevant passage without repeating the whole search.

Investigate conflicting definitions, assumptions, conventions, and nuanced cases. State the convention used. Keep consequential qualifications beside the claim: examples include statistical assumptions, nonunique solutions, indexing behavior, and shell/platform differences. Avoid words such as “always” or “guaranteed” without their conditions.

Sources should be optional for following the core walkthrough; a reference link must not replace a missing explanation. Separate evidence for technical correctness from inspiration for pedagogy.

### Further learning and technical references

Curate useful learner-facing alternatives as well as the technical claim sources. During every rewrite, look for good explanatory articles, worked tutorials, books/chapters, interactive exercises, and videos or YouTube playlists where they offer a helpful alternate explanation. Publish well-matched alternatives under further learning and precise claim/API sources under technical references; do not keep them only in the author's research record. Choose the role from the resource's actual use in this lesson, not its URL or format. No format or link count is a quota, and a full playlist is not automatically better than one focused lesson.

Annotate each selected resource with creator/title, format, the particular concept or activity it helps, intended level or suggested point in this lesson, and important prerequisites/version/access caveats. Prefer a direct lesson, relevant chapter or creator's playlist over a channel/search homepage. Supply useful timestamps or playlist item names only when verified. Keep references navigable by separating alternate explanations/practice from precise API or claim references when that aids scanning.

Verify the destination and fit using the resource itself or its substantive transcript, companion notes/notebooks and chapter listing. Record exactly what was reviewed; never claim to have watched an entire video when only its notes or listing were inspected. Metadata-only discovery is not enough to endorse its technical details. Use primary documentation/research to check current semantics; older videos can remain useful for intuition with a clear warning about changed APIs/defaults. Avoid unsupported rankings, popularity-as-proof and link dumps. Label paid/sign-in requirements when known and favor accessible alternatives. A video must never be the only way to follow a core explanation or practice task.

Research informing this standard, reviewed 9 September 2026:

- [IES practice guide](https://ies.ed.gov/ncee/wwc/PracticeGuide/1): supports alternating examples and practice, combining verbal/graphical explanations, connecting concrete and abstract representations, retrieval, and explanatory questions. Evidence ratings differ by recommendation; this broad 2007 synthesis does not prove a particular website template effective.
- [PhET original simulation design research](https://phet.colorado.edu/publications/archive/Phet%20Interview%20Paper.htm): informs clear initial states, understandable controls, meaningful responses, and novice walkthroughs. Findings arise primarily from science simulations and student interviews, not universal tests of all technical subjects.
- [Seeing Theory creator's account](https://blog.cs.brown.edu/2018/01/22/seeing-theory-teaching-statistics-through-interactive-web-based-visualizations/) and [frequentist chapter](https://seeing-theory.brown.edu/frequentist-inference/index.html): illustrate sequenced visual intuition and distinct explorations for connected concepts. This is a design reference, not proof of mastery.
- [Python Tutor](https://pythontutor.com/index.html): demonstrates visible execution, objects, references, and stack frames. Borrow the principle of exposing hidden state, not its interface or promotional effectiveness claims.

The framework here is a project-specific synthesis of the user's requirements, local lessons, and these sources.
