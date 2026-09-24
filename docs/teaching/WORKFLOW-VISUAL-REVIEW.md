# Workflow lesson visual review

Reviewed and implemented 10 September 2026 for the user's request to revisit previously improved lessons using concept-specific illustrations and interactions. This record covers the five topics below. It supplements their original design/native records; it does not replace the teaching standard or claim user acceptance.

Read `AGENTS.md`, the current handoff/standard, domain playbook, topic-design brief, `.impeccable.md` and frontend-design skill. Retrieved each stable ID through `node scripts/build-curriculum-inventory.mjs --topic ...`; none of these five has a destination-topic note, and the unresolved routing inbox was empty. Existing uncommitted content was preserved.

## Dispositions and representation contracts

### Reproducible Notebooks & Experiment Structure

- **Stable ID:** `reproducible-notebooks-experiment-structure`. Retain title/scope: source, live execution, saved evidence and replay were already in scope.
- **Reading gap:** the interactive notebook has useful actual cells and separate memory, but starts before any calculation. A first-pass reader must operate it to see a concrete disagreement between edited source, live objects and old output. Saving that disagreement also needs an explicit boundary.
- **Added:** `NotebookStatePicture`, immediately before the existing notebook/kernel investigation in section 2. The document has cell-shaped source regions and an output region inside its save boundary; the live process is outside that boundary. Offset 2 is visibly edited to 5, while memory remains at offset 2 / mean 18.0 and historical output remains 18.0. This is a fixed explanatory state, with no unnecessary controls.
- **What to notice / check:** saving source plus retained output does not rerun the calculation or capture all Python objects. Reexecuting the new input alone still leaves the old mean; rerunning input, calculation and display gives 15.0. The adjacent existing lab lets the reader perform those operations.
- **Retained:** editable notebook cells, execution counts, restart/run-all controls, random-consumer strips with separate/shared streams, provenance lookup, full native notebook project, manifests, practice and resources. Random tapes already reveal consumption directly; replacing them with another generic diagram would reduce their usefulness.
- **Accuracy / boundaries:** native Python independently computes 18.0 then 15.0 and nbformat writes/reads a file containing edited source and retained old output. Saving outputs is an explicit condition, not a claim that every frontend always retains every output. The fresh-kernel project remains covered by its original verification record; this increment's added serialization check does not claim a fresh execution of the entire notebook project.
- **Access / mobile:** actual text cells, values and a caption are readable without color or interaction. At 390px document and kernel stack, with their labels and distinct boundaries retained.

### Code Documentation, Type Hints & API Design

- **Stable ID:** `code-documentation-type-hints-api-design`. Retain title/scope: the addition clarifies an existing core distinction rather than expanding the topic.
- **Reading gap:** prose and executable code explain type annotations, but the relationship between static checking and runtime execution is easy to misread as a mandatory pair of sequential gates.
- **Added:** `ApiCheckingPicture`, after the annotation introduction in section 2, before the complete runnable example. One short source file branches into two uses: a checker compares the string argument to the float annotation; ordinary Python evaluates string repetition and returns `"haha"`. The same input on both routes isolates what differs.
- **What to notice / check:** checker rejection does not automatically insert runtime validation. Predict why the annotated function can still print `haha`; then identify a rule such as units or finiteness that needs an explicit contract/check. The caption distinguishes Python behavior from a project's optional rule that type checks must pass before execution.
- **Retained:** ownership arrows and shared/copy cases, validation gates, compatibility investigation, complete loader, optional-value examples, doctests/static checks, independent adapter exercise and annotated resources. Their existing object identity and gate representations already fit those different questions.
- **Accuracy / boundaries:** the actual existing `double` example was executed on Python 3.12.14 and separately rejected by the installed mypy checker. This is a fixed source/behavior comparison, not a browser Python interpreter or universal statement about frameworks that deliberately perform runtime validation.
- **Access / mobile:** both branches contain readable code/value correspondences and their own labels. They stack at narrow widths; the caption explicitly preserves the separate-route meaning.

### Git, GitHub & Collaborative Version Control

- **Stable ID:** `git-github-collaborative-version-control`. Retain title/scope and all four investigations.
- **Lab gap:** the remote investigation showed three parallel snapshot panels. It named `origin/main` correctly, but the panels did not spatially distinguish two local references from a branch in another repository. A beginner could read “remote-tracking” as a live object residing on the server.
- **Changed:** `GitRemotePicture` replaces only the generic snapshot arrangement in the section 9 remote investigation. The shared repository sits outside a labelled local repository containing `origin/main`, `main` and the working file. Fetch crosses the boundary; a separate integration arrow sits between local references. Active operations highlight the relevant path, while all values come from the existing `remoteTrace` state.
- **Prediction / operation:** predict which references move after a colleague pushes; advance the existing native-named commands. Compare ordinary local history to a local commit C, then fetch and attempt fast-forward integration. Observe `origin/main → B` while `main` and the working file remain unchanged after fetch; a divergent merge is explicitly refused.
- **Reset / transfer / boundary:** scenario changes and reset restore step 0; Back restores earlier visible references. Transfer is to explain why a successful fetch need not update an open file. The model still assumes ordinary origin fetch mappings, a clean working tree and two small histories. It does not operate on the learner's repository.
- **Retained:** commit-parent graph, branch movement, three stored staging versions, common-base conflict comparison, complete disposable commands and independent practice. Those representations already show actual ancestry/content and do not need novelty for its own sake.
- **Accuracy:** fresh local Git repositories reproduce all five reference/content states for each scenario, including fast-forward success and divergent refusal. The official fetch documentation was reviewed for object/ref fetching and configurable remote-tracking mappings.
- **Access / mobile:** labelled repository containment and operation arrows remain meaningful without color. On narrow screens the local reference/integration sequence stacks with a downward arrow; fetch and integration remain distinct.

### Bash Scripting & Command-Line Automation

- **Stable ID:** `bash-scripting-command-line-automation`. Retain title/scope; complete programs, publication investigation and exercises are unchanged.
- **Lab gaps:** argument results were line-oriented text inside a generic state panel, despite boundaries being the concept under study. The pipeline lab combined output and exit information into one flow, even though the lesson asks the learner to distinguish those channels.
- **Changed argument representation:** `BashArgumentPicture` in section 2 shows a single stored string transformed into individually bounded received arguments, with numbered slots and explicit empty/no-argument cases. The existing quoted/unquoted and spaces/wildcard/empty controls still drive `argumentTrace`; words appear after expansion and are labelled received only at launch. A quoted empty string occupies one slot; the unquoted empty value produces none.
- **Argument contract:** predict the number and contents of strings, select a fixture/quote mode, advance the shell phases, then explain what changed. Counts, outlined strings and phase feedback describe the same active model. Existing Back/Reset and invalid-case boundaries are preserved. JSON-style quotes denote the string representation, not literal quote characters sent to the command. Default IFS and the two-file glob fixture remain explicit limits.
- **Changed pipeline representation:** `BashPipelinePicture` in section 4 separates a stream path (producer output → consumer input) from the two exit statuses and the parent's selected pipeline status. Selecting statuses and toggling pipefail changes the lower decision, without erasing the fixed emitted rows above. This is a two-command fixture, not a full process/pipe emulator or timing measurement.
- **Pipeline contract:** predict the parent-visible status for producer 4 / consumer 0; enable pipefail and observe 4 while emitted bytes remain. With two failures, observe the rightmost failure. Explain why status handling cannot roll back bytes. The native oracle checks both selected status and retained emitted bytes for all eight combinations.
- **Retained:** publication state remains a compact direct comparison between staged/public contents across a known producer outcome, where an exact state view is appropriate. Quoting, option parsing, child environment, path assumptions, `set -e` caveats, native worker/wrapper programs and independent collection exercise remain intact.
- **Accuracy / access:** all six argument combinations and eight status combinations were checked against GNU Bash 5.2.37 through Git Bash. This is evidence for these Bash semantics, not a new Linux filesystem/native publication verification. Full GNU manual retrieval failed during this increment; those semantics were independently executed rather than described as freshly webpage-verified. Narrow layouts use explicit labels, legible string slots and a vertical pipe flow; keyboard Space toggles pipefail.

### Threads, Concurrency, Locks & Deadlocks

- **Stable ID:** `threads-concurrency-locks-deadlocks`. Retain title/scope; no new interpreter or scheduling guarantee is introduced.
- **Lab gap:** the lost-update model was correct but the shared counter and two workers appeared as generic parallel state cards, with advance buttons away from the individual workers. That required the learner to reconstruct program order and copy ownership from prose/log entries.
- **Changed:** `ThreadRacePicture` replaces that arrangement in section 2 with a single shared counter above two independently controlled execution lanes. Each worker has its own copied value and ordered read/compute/write operations, marked pending/next/done with text and symbols. Advance A/B controls now sit with the corresponding lane. Shared lock ownership and blocked entry are visible without consulting the log.
- **Prediction / operation:** schedule A read, B read, A compute, B compute, A write, B write. Watch both private copies become 1 while both writes target the same shared location, ending at 1. Switch to the whole-operation lock and attempt B while A owns it: B's private copy remains unread. Finish A, then B, to obtain 2.
- **Reset / transfer / boundary:** protocol changes reset the model, Reset restores it, completed workers disable their control, and blocked attempts remain inspectable. Explain why locking only each write would still permit two stale copies. The original log is retained as an exact text trace. Lanes encode each worker's program order, not wall-clock time, real CPU assignment or CPython bytecodes.
- **Retained:** the wait-for graph already draws real directed dependencies and the two-lock cycle, so it is unchanged. The condition investigation keeps its queue predicate, ownership and chosen legal schedule together; the process-containment introduction, native examples and bounded worker pipeline remain.
- **Accuracy / access:** existing exhaustive model/native verification was rerun: 73 reachable states across race and lock models, all six displayed thread programs, 18 changed queue cases and no leaked workers. The new renderer uses the same state directly; browser checks verify lost/protected totals, blocked unread state, keyboard operation and reset. At 390px the lanes remain side by side with readable steps and per-worker controls; no time axis or performance claim is invented.

## Evidence, provenance and limits

Commands run successfully for this increment:

```text
node scripts/verify-workflow-visuals.mjs
node scripts/verify-thread-completion.mjs
node scripts/review-workflow-visuals.cjs
```

- Focused native artifacts/results: `scratch/workflow-visual-review/native-results.json`, `native-fixtures.json` and the uniquely named native evidence directory recorded in that JSON. Python 3.12.14, GNU Bash 5.2.37 (MSYS), Git 2.48.1.windows.1; mypy rejects the intentional string call independently of successful Python execution.
- Browser results: `scratch/workflow-visual-review/browser-results.json` and 12 component screenshots at 1440px/390px. Verified all six added/adapted representations, no page/figure horizontal overflow or page errors, notebook stale/recomputed output, six quoted/unquoted combinations, eight pipefail combinations, both race protocols, both remote histories, Back/Reset and keyboard activation. Representative desktop/mobile screenshots were visually inspected for readable labels and preserved relationships.
- Additional thread evidence remains in `scratch/thread-completion-review/`; the native suite is an independent check in addition to browser renderer assertions.
- Source review on 10 September 2026: [Python typing runtime note](https://docs.python.org/3/library/typing.html), [nbformat code cells, source and outputs](https://nbformat.readthedocs.io/en/latest/format_description.html#code-cells), [Git fetch description and remote-tracking configuration](https://git-scm.com/docs/git-fetch). These support only the specific distinctions above; existing lesson sources continue to cover their broader claims.
- The new pictures are concrete fixed examples or views of existing bounded models. None is measured benchmark data or a quantitative performance graph. No real-product rankings or invented timing samples were added.
- Implementation/native/browser/author visual review passed within this recorded scope. No novice study or user acceptance is claimed. The coordinating task runs the integrated build and broader checks; this subtask did not run a competing full build or alter catalogue IDs, navigation, publication/progress, shared lab foundations or global CSS.

## Files

The figures now live in `NotebookFigures.jsx`, `ApiDesignFigures.jsx`, `BashFigures.jsx`, `ThreadFigures.jsx` and `GitFigures.jsx` under `src/learn/components/lesson-labs/`, with shared `workflow-figures.css`. `BashWorkflowLabs.jsx` and `ThreadCoordinationLabs.jsx` integrate the separate investigations; `GitFoundationsLabs.jsx` integrates the repository boundary. The notebook and API topic files place their static figures beside their introductions. This later [runtime organization](../engineering/RUNTIME-SOURCE-ORGANIZATION.md) preserves visual behavior while keeping each topic's data/lab dependencies separate. Reusable controls, existing models, complete programs and sources were retained; no claim of sufficient visual teaching is based on a lab or figure count.
