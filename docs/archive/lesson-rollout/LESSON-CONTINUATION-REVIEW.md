> HISTORICAL RECORD — archived 9 September 2026. All queue, authorization, approval and teaching instructions below describe an earlier increment. They are not current policy. Start at [the current handoff](../../../LESSON-AUTHORING-HANDOFF.md).

# Educational content continuation review

**Latest continuation, 9 September 2026:** the user subsequently requested one topic. Linux has now been improved with four focused labs, supporting diagrams, progressive disclosure and a complete investigation. Model checks, native Linux verification, desktop/mobile browser tests and the build passed; user review is pending. See [PROGRAMMING-REWRITE-LINUX.md](../../../PROGRAMMING-REWRITE-LINUX.md). Bash is the next untouched rewrite; no further topic is authorized yet. The review below is retained as the historical pre-implementation assessment.

9 September 2026. Scope: recover the established direction, compare reference lessons with programming work, identify the continuation point, and save reusable authoring instructions. No lesson or runtime component was rewritten in this review.

Use [LESSON-TEACHING-STANDARD.md](../../../LESSON-TEACHING-STANDARD.md) as the future-session authoring brief. It incorporates the user's request for clear explanations with substantial depth and purposeful visual teaching, including multiple labs where appropriate.

## Goal and baseline

The goal is an integrated learning resource: a newcomer can follow concepts from their foundations to useful application, visualize the mechanism, understand relationships and assumptions, practise with feedback, and progress toward advanced material. The existing improved lessons are the baseline to preserve and extend. This is a continuation of that work.

The review used the original pilot and programming batch notes, the three pilot lesson sources and labs, shared teaching components, programming examples/labs and topic order, and saved screenshots. The saved screenshots inspected cover all three pilots at desktop width plus Python basics, NumPy broadcasting, Pandas joins, and a narrow notebook lab. They are historical captures, not fresh browser tests. Numerical scripts and the application build were not rerun. This is a targeted source and visual review, not a full-site factual or accessibility audit and not an actual learner study.

## Patterns in the three reference lessons

| Pattern | How it supports learning |
| --- | --- |
| Concrete opening question | Establishes a reason to care before technical manipulation. |
| Prerequisites and reading route | Gives newcomers an ordered path and returning readers a revision route. |
| Continuing example | Reuses entities and numbers across the explanation, calculation, and decision. |
| Intuition and purposeful exploration | Lets the learner inspect what changes and make a prediction. |
| Formal mechanism and complete computation | Connects the intuition to notation, steps, code, and visible output. |
| Interpretation and conditions | Explains what the result means and when it can mislead. |
| Answer reveals | Supports independent attempts with an explanation available afterwards. |
| Optional deeper sources | Gives another learning route without outsourcing the core explanation. |

The progression is strongest through intermediate application. Advanced material is often introduced in later sections, but generally remains expanded and sometimes compresses another substantial conceptual jump. Consistent headings alone do not solve this.

### Hypothesis Testing & Confidence Intervals

[Lesson source](../../../src/learn/data/topics/hypothesis-testing-confidence-intervals.jsx), [coverage lab](../../../src/learn/components/lesson-labs/ConfidenceLab.jsx).

Preserve the paired latency example, visible interval coverage, distinction between statistical evidence and useful effect size, t-based worked calculation, and assumptions about independent differences. The graph gives intervals a common fixed reference and combines line styles with color. The exercises include changes to data and a dependence misconception.

Further opportunities:

- Explain the transition between the lab's known-standard-deviation z interval around a 100 ms mean and the worked example's estimated-variability t interval for paired differences. This is disclosed and mathematically intentional, but requires a beginner to switch models.
- Add a separate null-distribution/tail-area visual for the p-value. Power and practical effect size may need another focused investigation or deeper module. The coverage lab does not explain all three mechanisms.
- Label optional depth and link a concrete next lesson. Keep the warning that the worked example's fixed t critical value applies to its fixed sample size; a generalized code path must compute it from the new degrees of freedom.

### Bayesian Inference & Conjugate Priors

[Lesson source](../../../src/learn/data/topics/bayesian-inference-conjugate-priors.jsx), [posterior lab](../../../src/learn/components/lesson-labs/BayesLab.jsx).

Preserve the 8 successes/2 failures example, prior sensitivity, the count-update derivation, posterior-versus-prediction distinction, explained results, and sequential-update practice. The same plotted density axis makes changes to prior and posterior comparable.

Further opportunities:

- Introduce candidate rates and prior × likelihood → normalization visually before the integral formula. The current early notation can exceed the stated beginner entry point.
- Shade the posterior probability area under the curve for the credible interval, rather than relying only on a rectangular interval band. Explain density versus probability and what its area represents.
- Use another visual for future outcome counts versus uncertainty about the underlying rate. Let deeper prior sensitivity distinguish prior location from strength.
- Move the custom numerical CDF/quantile implementation into optional depth if it obscures the main inference workflow. Preserve its limits and verified numerical behavior when restructuring.

### Spectral Graph Theory

[Lesson source](../../../src/learn/data/topics/spectral-graph-theory.jsx), [graph lab](../../../src/learn/components/lesson-labs/SpectralLab.jsx).

Preserve the six-node graph, bridge control, values attached directly to nodes, graph-to-Laplacian calculation, energy explanation, interpreted NumPy result, and eigenvector sign/eigenspace caveats. The lesson distinguishes the initial unnormalized sign split from the later normalized clustering workflow.

Further opportunities:

- Selecting a node could highlight its incident edges and the corresponding Laplacian row, then calculate the terms of the local operation.
- A separate edge-disagreement visual could explain energy and the role of normalization.
- A diffusion/smoothing experiment could show how signals evolve and why disconnected components cannot exchange information. This is a different question from finding a weak bridge.

All three currently contain one early lab. Neither the pilot's stated philosophy nor the user's goal requires that count.

## Why some programming visuals feel weaker

The programming rewrites made substantial improvements: complete examples, visible output, interpreted failures, topic-specific sequences, and solved practice. Preserve these. The original generic `StructuredLesson` wrapper has already been replaced for most of the track; do not assume every programming page still uses it.

The current track already varies its visual support: Matplotlib includes seven rendered plot figures, while documentation/type hints/API design has no specialized lab. Topic components accept arbitrary JSX, so there is no one-lab technical constraint. The question is whether the selected representations cover the difficult concepts effectively.

The gap is often the representation of the mechanism. [PythonTrace](../../../src/learn/components/lesson-labs/PythonTrace.jsx) says names point to objects but displays a table of names and object labels. The learner reconstructs the shared-object relationship mentally. [BroadcastLab](../../../src/learn/components/lesson-labs/BroadcastLab.jsx) exposes shapes and per-cell arithmetic, but dimension alignment and conceptual reuse are still largely textual. [GitIndexLab](../../../src/learn/components/lesson-labs/GitIndexLab.jsx) shows three stored versions in a table; movement between working tree, index, and commit is not drawn.

Tables are useful and should remain where they teach the data or provide precise inspection. The improvement is to connect them to visible identity, movement, correspondence, or execution when that is the conceptual difficulty.

| Topic | Focused visual improvements to consider in a future authorized pass |
| --- | --- |
| Python basics / OOP | Name-to-object arrows; shared versus copied objects; separate function-call/local-scope view; highlight the changed instance. |
| Iterators / generators | Caller and generator shown together; execution cursor, suspended position, retained locals, and yielded value. |
| Decorators / context managers | Nested call/return flow separately from resource acquisition, exception, and cleanup paths. |
| NumPy | Aligned broadcasting dimensions; a separate view-versus-copy memory model; reduction axes and resulting shapes. |
| Pandas | Trace selected source rows into join output, including duplicate multiplication; separately visualize label alignment and split/group/aggregate. |
| Notebooks | Visible cells, execution order, kernel variables, and saved outputs; highlight stale derived values and show what rerunning repairs. |
| Git | Snapshot transfer across working tree/index/HEAD; a separate commit graph for branching, merging, and fetch; a conflict-resolution walkthrough. |
| Linux | Path/tree navigation; file versus directory permission actions; process relationships; standard stream connections. |

These are candidates with distinct learning questions, not a mandate to implement every listed lab or expand this review into rewrites.

The main persona risks are concrete: a first-time learner must imagine reference arrows while reading table labels; a mobile learner may need to remember source tables while scrolling to results; a returning practitioner needs a clearer way to skip derivation while finding the relevant method and conditions. Existing text labels, table semantics, and some live feedback are strengths. Complete keyboard/screen-reader behavior remains to be checked in a fresh browser review.

## Highest priorities for the next implementation

1. **P1: make hidden mechanisms visible.** Place focused diagrams/labs beside the concepts they explain, preserving exact tables and useful existing examples.
2. **P1: remove prerequisite jumps.** Introduce concrete cases before dense notation and explicitly bridge changes between simplified labs and real examples.
3. **P2: strengthen progressive disclosure.** Separate the beginner route from optional derivations, broader APIs, and advanced extensions without hiding essential understanding.
4. **P2: strengthen independent practice.** Add hints, partially worked tasks, misconception diagnosis, and changed-context application where current full-answer reveals provide too large a jump.
5. **P2: improve continuity and status.** Link prerequisites/next steps and distinguish source implementation, testing, visual review, and user acceptance.

These are heuristic teaching priorities, not measured learner outcomes. Visual quantity is not the target; the learner's ability to explain, predict, diagnose, and apply is.

## Where to resume

The actual [track order](../../../src/learn/data/track-definitions.js) ends with **Git → Linux → Bash**. The original pilot's September 9 follow-up and batch-03 ending still say all three remain, but source has advanced further.

| Position / work | Implementation evidence | Verification and review evidence available in this review | Next action |
| --- | --- | --- | --- |
| Programming 1–3: Python basics, OOP, iterators/generators | Topic-specific rewrites; [batch 01](PROGRAMMING-REWRITE-BATCH-01.md). | Historical batch report records checks; not rerun here. The note records subsequent authorization to continue. | Preserve; targeted visual improvements can be scoped separately. |
| Programming 4–6: decorators/context managers, testing/debugging/dependencies, NumPy | Topic-specific rewrites; [batch 02](PROGRAMMING-REWRITE-BATCH-02.md). | Historical checks and subsequent narrower continuation recorded; not rerun here. | Preserve; prioritize difficult representations in any later refinement. |
| Programming 7: Pandas | Topic-specific rewrite; [Pandas increment](PROGRAMMING-REWRITE-PANDAS.md). | Historical check report and subsequent continuation recorded; not rerun here. | Preserve; refine row correspondence and other mechanisms if requested. |
| Programming 8–10: Matplotlib, notebooks, documentation/type hints/API design | Topic-specific rewrites; [batch 03](PROGRAMMING-REWRITE-BATCH-03.md). | Historical checks reported; note says awaiting review. Later source changes are not proof of user acceptance. | Do not redo blindly; separate review status from implementation. |
| Programming 11: Git/GitHub | [Detailed lesson](../../../src/learn/data/topics/git-github-collaborative-version-control.jsx), staging lab, examples, checkpoints, references. | User reports work reached Git. Source and verification scripts exist; no completed batch-04 report or matching browser screenshot directory found. No fresh pass claimed. | Preserve the current implementation; capture missing verification/review evidence during a scoped continuation. |
| Programming 12: Linux | [Detailed lesson](../../../src/learn/data/topics/linux-basics-filesystems-processes.jsx), permission lab, eight terminal examples, checkpoints, references. | Source describes prior Ubuntu/WSL execution. Scripts and exported cases exist, but a finished batch review record was not found. Verification/user review remain unconfirmed here. | **Next review/completion point after Git.** Inspect, validate, and assess this existing implementation rather than start over. |
| Programming 13: Bash | [Short existing lesson](../../../src/learn/data/topics/bash-scripting-command-line-automation.jsx) still uses `StructuredLesson`. | Does not have the topic-specific rewrite pattern or a dedicated lab. Its example refers to an unsupplied `train.py` and practice lacks a worked solution. | **Next topic needing the full rewrite**, after resolving the Linux continuation and receiving the user's selected scope. |

Relevant unfinished-batch evidence: [browser review script](../../../scripts/review-programming-batch-four.cjs), [native Linux example verifier](../../../scripts/verify-system-lessons.py), [example exporter](../../../scripts/export-system-lesson-cases.mjs), [Git/Linux examples](../../../src/learn/data/programming-batch-four-examples.js), and [teaching models](../../../src/learn/data/system-lesson-models.js). The browser script assumes one lab per topic and would need adaptation for multiple labs. Existence of these files does not establish that the latest source passed.

Bash's future rewrite should teach argument/quoting behavior, streams, exit statuses, branching/loops, functions, and a complete small automation task with failure handling. Check the current “fail early” explanation carefully: `set -e` has contextual exceptions; it is not a universal stop-on-any-failure guarantee. The [GNU Bash manual](https://www.gnu.org/s/bash/manual/bash.html#The-Set-Builtin) documents those conditions. Focused argument and stream diagrams plus a failure-path exercise would answer different learning questions.

The recommended continuation is therefore **review and finish the existing Linux work, then rewrite Bash**. Improving the earlier programming visuals can be a separate user-selected sample to calibrate the stronger standard. Do not assume authorization for either implementation step from this review request.

## Teaching framework and supporting research

Use an outcome/concept map, a continuing example, a flexible basic-to-advanced route, local explanation/visual/practice loops, and progressively independent application. Keep accurate conditions, output interpretation, practical context, and next-topic connections throughout. [The teaching standard](../../../LESSON-TEACHING-STANDARD.md) gives the detailed instructions and acceptance criteria.

The [IES learning practice guide](https://ies.ed.gov/ncee/wwc/PracticeGuide/1) supports connecting representations, alternating worked examples with practice, retrieval, and explanatory questions. [PhET's original design research](https://phet.colorado.edu/publications/archive/Phet%20Interview%20Paper.htm) informs simple initial states and meaningful interaction feedback. [Seeing Theory](https://seeing-theory.brown.edu/frequentist-inference/index.html) illustrates multiple focused explorations within a chapter; [Python Tutor](https://pythontutor.com/index.html) demonstrates visible execution state. These informed the recommendations; their popularity or use does not establish effectiveness for this website, and their content was not copied.

No website-wide rewrite, deployment, or lesson modification was performed. This review adds the reusable standard, this continuity record, and a pointer in the historical pilot note.
