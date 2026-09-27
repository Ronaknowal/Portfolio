# Dsa Practice

Part of [$lesson-authoring](../SKILL.md). Repository paths below are relative to the selected application checkout; see [the repository adapter](portfolio-adapter.md). Load only the sections needed for the selected mode.

- [Purpose and scope](#purpose-and-scope)
- [Curate for this topic and the actual reading sequence](#curate-for-this-topic-and-the-actual-reading-sequence)
- [Verify and annotate resources](#verify-and-annotate-resources)
- [Practice process and readiness](#practice-process-and-readiness)
- [Implementation and review contract](#implementation-and-review-contract)

# DSA practice: from a lesson to unfamiliar problems

Current policy, added 10 September 2026 for the user's request to connect DSA lessons with useful LeetCode practice. Read alongside [the teaching standard](../SKILL.md), [the domain playbook](domain-playbook.md) and the learning code standard (repository path: `docs/engineering/LEARNING-CODE-STANDARD.md`). The current user request controls implementation scope; this document is not authorization to rewrite every lesson.

## Purpose and scope

Give a learner enough varied practice to recognize the underlying objects, choose a representation, explain an invariant, justify an algorithm and adapt it to a changed requirement. Work toward broad interview and practical problem-solving ability without promising that a finite list covers every future problem. A completed checklist, platform difficulty label or accepted submission is not evidence of universal mastery.

External problems supplement the self-contained lesson. Keep its complete examples, live interactive investigations, independent exercises, hints, explained solutions and success criteria. A link must never stand in for teaching a required mechanism. Representation-specific ideas such as Unicode normalization, object identity, hash collisions and intermediate pointer changes may need local tasks even when they are poorly represented in a platform's judge inputs.

## Curate for this topic and the actual reading sequence

1. Run the topic-plan command, read the lesson and destination notes, then list the mechanisms and decisions its title promises. Cover the meaningful breadth: for example, Heaps, Priority Queues & Tries needs both priority selection and prefix lookup; Linked Lists, Stacks & Queues needs both reference edits and LIFO/FIFO behavior.
2. Choose problems whose differences require a useful new deduction, contract adaptation, representation choice or failure diagnosis. Do not add repeated variants just to increase the count, or one token problem to each topic solely for visual consistency. There is no fixed count or difficulty quota.
3. Arrange a foundation stage for direct reconstruction and a core stage for transfer using concepts already taught. Names and groupings can change with the topic. The syllabus's topic order remains the teaching sequence; problem difficulty or publication status does not reorder it.
4. Put problems requiring an unintroduced technique in an optional extension with concrete prerequisites. Name the missing skill, and link its existing topic where a verified destination usefully helps. A generic “advanced” badge is not an adequate prerequisite explanation. Optional problems are not a gate for continuing the module.
5. Select a legitimate approach matched to the lesson even if the platform has other tags or a more sophisticated alternative. Distinguish the basic task from stronger follow-up bounds. Do not imply a bounded heap always satisfies a strictly sub-n-log-n requirement for unrestricted k, or that ordinary BFS solves unequal-cost shortest paths.
6. Reassess the list when the lesson changes. Keep the problem's contract distinct from the teaching model: strict BST ordering versus a duplicate policy, all matching nodes versus the first, cell-count versus edge-count distance, or sentinel returns versus exceptions. Explain such differences next to the task.

## Verify and annotate resources

Use the official problem statement as the primary source. Verify the direct URL, displayed ID, title, current difficulty, constraints relevant to the annotation and known access requirements. Check the resource itself rather than trusting a community sheet, search snippet or remembered difficulty. If a destination cannot be inspected, resolve it or label the limitation; do not silently invent verified status.

Provide original short annotations for each selected problem:

- **Learning focus:** the lesson mechanism or transferable decision being exercised, without giving away a complete solution.
- **Before attempting:** additional prerequisites or important contract differences when needed.
- **Optional hint:** a small reasoning prompt, initially closed. Reveal support gradually; do not print the solution or pattern recipe beside every unopened question.
- **After solving:** a changed constraint, adversarial case, comparison, correctness explanation or application that checks transfer. Keep expected reasoning and costs accurate; do not invent platform outputs or grade the learner by exact wording.

Use the actual platform difficulty as metadata, while labeling the site's stages as pedagogical choices. Say when access was checked. Favor inspectable statements and accessible practice; label premium/sign-in restrictions when known. A public statement does not establish that submitting, every editorial, a video solution or other account feature is freely accessible. Do not claim that a solution was executed or a video watched unless it was.

Do not reproduce full problem statements, examples, editorials or solutions. Link to the original, then contribute original instructional guidance. A curated problem set belongs near the lesson's practice/readiness material, with a route anchor, before its references. Continue to curate articles, videos and other alternate explanations in the references under the main teaching standard; LeetCode is not a replacement for them.

## Practice process and readiness

Ask the learner to restate inputs/outputs and constraints, draw a small state trace, propose a correct baseline, and identify what an optimization must preserve. Their explanation should include initialization, invariant preservation, termination and why the result follows, with resource costs under a stated model. Test relevant boundaries and adversarial cases rather than only the sample.

Offer hints before a full answer. After assistance, close it, reconstruct the approach in a later session, and vary an input or contract. Interleave earlier topics once their basics are secure so that choosing the data structure is itself practice. Do not impose an unsupported universal timebox, streak or reattempt interval. A learner who can explain and adapt one approach has stronger evidence than one who can copy several solutions.

Readiness prompts should be topic-specific: preserve node identities, enforce ancestor bounds, account for heap ties, distinguish prefixes from complete words, explain discovery state, or identify when a graph traversal needs richer state. State limitations honestly and use the next module topic as the continuation; extensions do not force prerequisite detours into the reading order.

## Implementation and review contract

The shared presentation is `src/learn/components/lesson-labs/DsaPractice.jsx` with `dsa-practice.css`. It imports no problem data, compiler, judge, network client or global DSA aggregate. Each lesson imports only `src/learn/data/practice/<stable-topic-id>.js` and passes that default-exported dataset as `practice`:

```jsx
import { DsaPractice } from '../../components/lesson-labs/DsaPractice.jsx';
import treePractice from '../practice/trees-binary-search-trees.js';

// Include ["guided-dsa-practice", "Guided LeetCode practice"] in the route.
<DsaPractice practice={treePractice}/>
```

The dataset has `topicId`, `verifiedOn`, `introduction`, `groups`, `readiness` and `localBridge`. Each group has `id`, `title`, `introduction`, `problems` and optionally `optional: true`. Each problem has the verified `number`, `title`, `slug`, `difficulty`, plus original `focus`, `hint`, `transfer` and optional `prerequisite`. Optional groups and hints start closed. The component is a curated reading/practice guide, not an in-page execution engine, submission client or persisted completion tracker.

Use normal links, native disclosures, visible keyboard focus, meaningful accessible labels, responsive text and readable contrast. Problem links open a new tab with `noopener noreferrer`, a visible explanation of that behavior and an accessible per-link announcement, so learners can retain their lesson. Keep essential prerequisites outside hidden hints. Review narrow screens and keyboard opening/closing, anchors, link destinations, absence of page overflow/errors, and isolation of selected-topic data. If adding execution/grading/progress later, separately design and verify its real behavior and identity/privacy contracts; do not imply it exists now.
