# Runtime source organization

Implemented 10 September 2026. This record covers the semantic source split and preserved lesson behavior. The coordinating change owns the publication manifest, compact catalogue, lazy lesson registry and measured loading boundaries. Current policy lives in the repository's learning/code standards; this is implementation evidence, not a new curriculum or rewrite queue.

## Topic ownership replaces rollout groupings

The previous runtime files grouped examples and labs by the order in which lessons were implemented. Selecting one topic could therefore request modules containing several unrelated topics' examples, models or figures. Runtime ownership now follows the subject that uses the content.

| Previous runtime grouping | Current ownership |
| --- | --- |
| `programming-batch-one-examples.js` | `python-core-examples.js`, `oop-core-examples.js`, `iterator-core-examples.js` |
| `programming-batch-two-examples.js` | `decorator-core-examples.js`, `testing-examples.js`, `numpy-reference-examples.js` |
| `programming-batch-three-examples.js` | `plotting-examples.js`, `notebook-examples.js`, `api-design-examples.js` |
| `programming-batch-four-examples.js` | `git-examples.js`, `linux-command-examples.js` |
| `next-three-examples.js` | `pandas-practice-examples.js`, `plotting-practice-examples.js`, `git-practice-examples.js`; exports use `PracticeExamples`, not `NewExamples` |
| `programming-batch-two-traces.js` | `decorator-introduction-trace.js`, `debugging-introduction-trace.js` |
| `iteration-decorator-examples.js` / `iteration-decorator-models.js` | Iterator and decorator examples/models each have their own module. Iterator/decorator example extensions import only their corresponding core fixtures. |
| `reliability-examples.js` / `reliability-models.js` | Testing, notebooks and API design each own examples/practice/models. |
| `data-foundations-examples.js` / `data-foundations-models.js` | Scientific file examples/models and SQL examples/models are separate. Shared source fragments such as the measurement validator stay with their subject. |
| `workflow-completion-models.js` | `bash-workflow-models.js`, `thread-coordination-models.js` |
| `bash-completion-examples.js` / `thread-completion-examples.js` | `bash-workflow-examples.js`, `thread-coordination-examples.js` |
| `scientific-visual-models.js` | `csv-parsing-model.js`, `pandas-pivot-model.js` |

These files remain under `src/learn/data/`. All original exported example values, expected outputs, source fragments, model functions and retained reference fixtures were conserved. An example that is still useful as a regression fixture does not need to become an eager import in a different lesson.

## Labs, figures and shared UI

The corresponding component graph is also separated. Merely renaming a large data file would have left the same cross-topic loading dependency through its labs.

- `BashWorkflowLabs.jsx` and `ThreadCoordinationLabs.jsx` replace the combined completion lab module.
- `TestingLabs.jsx`, `NotebookLabs.jsx` and `ApiDesignLabs.jsx` replace the combined reliability lab module.
- `IteratorLabs.jsx` and `DecoratorContextLabs.jsx` own their respective investigations.
- `ScientificFileLabs.jsx` and `SqlLabs.jsx` own file/schema/publication versus relational/transaction investigations.
- Notebook, API design, Bash, Threads and Git figures each have their own file.
- NumPy figures, CSV figures, SQL figures, the Pandas pivot lab and plot ownership figure no longer share a module importing all their models.
- Python references, bound methods, iterator ownership, decorators/context routes and testing boundaries each have a subject-specific figure file. The OOP/iterator/decorator figures no longer import Python reference-model data solely because they used to share a source file.

All component files are in `src/learn/components/lesson-labs/`. Small presentational controls remain reusable:

| Shared file | Purpose |
| --- | --- |
| `LessonInvestigation.jsx` | Investigation container, prediction disclosure, step/reset controls and learning-resource list; replaces `NextThreeElements.jsx`. |
| `RunnableExample.jsx` | Full runnable-program presentation; replaces `WorkflowCompletionElements.jsx`. |
| `LabControls.jsx` | Labelled selection, live feedback and value formatting. It imports no lesson data. |
| `MechanismLab.jsx` | Mechanism lab container, state panel and execution log; imports no Bash or thread model. |
| `DataLabControls.jsx` | Data-oriented step controls and prediction disclosure; imports no file or SQL model. |

CSS filenames now describe their purpose: `lesson-investigations.css`, `workflow-labs.css`, `plot-output.css` and `workflow-figures.css`. Existing CSS class names, DOM investigation IDs, controls and visible teaching content were preserved. Small shared CSS and pure rendering helpers are intentionally reusable; source organization does not require a separate file for every trivial function.

The generic reader prerequisite rules are owned by the coordinating reader/loading change. Topic-specific components import their required styles rather than relying on a global import of a rollout lab stylesheet.

## Names inside source

Twenty-seven lesson import aliases such as `ex`, `added`, `practice`, `reference` and `foundations` now use the actual subject collection name: for example, `notebookExamples`, `apiPracticeExamples`, `plottingPracticeExamples` and `pythonFoundationsExamples`. Full example strings, mathematical notation, sample variables and output fixtures were not blindly renamed.

Touched interactive components use clearer state names where a short alias concealed meaning: `notebookState`, `executionState`, `boundaryState`, `ownershipState`, `cursorState`, `generatorState`, `pipelineState`, `contextState`, `cleanupState`, `conditionState` and `pipefailEnabled`. JSX references and local bindings were updated together through parsed binding references. Shared helpers similarly expose `LabChoices`, `LabFeedback`, `formatLabValue`, `MechanismLab`, `StatePanel`, `ExecutionLog`, `DataStepControls` and `DataPrediction`.

## Assets and references

Eleven assets formerly grouped under `public/learn-assets/programming-three/` now live under:

- `public/learn-assets/plots/` for the nine generated SVG charts.
- `public/learn-assets/notebooks/` for the source and executed notebook.

`PlotOutput` and the notebook download links use the corresponding `/learn-assets/plots/` and `/learn-assets/notebooks/` URLs. Verification generators publish future artifacts to these same destinations. Resolved absolute source and destination paths were checked to stay in this workspace before moving any file. Existing unrelated files were preserved.

Current teaching design/visual records point to the new owners. Historical rollout-named verification scripts remain identifiable as historical or regression checks; their imports and affected collection discovery were updated. Their existence does not define current teaching order or authorize a lesson rewrite.

## Verification follows imports, not an alias convention

`scripts/lib/lesson-examples.mjs` parses JSX executable-example props and resolves their imported data collections. A verifier therefore continues to find `example={notebookExamples.state}` without requiring every author to call the collection `ex`. It distinguishes executable examples from a `PlotOutput` displaying the same example's artifact. The Bash lesson's ID-based example helper retains its dedicated exporter and native suite.

Affected verification scripts use this collector, or import their actual subject fixtures directly. Current publication is determined by the publication manifest and authoring tooling, not by an eager `topics/index.js` import. The coordinating change owns those interfaces and route/loading checks.

## Evidence and limits

The split used Babel parsing and top-level binding dependency closures. Function bodies and complete example values were copied from their existing source; shared dependencies were retained with the owning subject. Before/after conservation checks passed for **66 original data exports**, including full code/output strings and model functions. **62 replacement runtime modules** were parsed, and **29 retired runtime files** were confirmed absent. Subsequent shared helper files and renamed lesson consumers were exercised through browser/native checks.

Evidence is under `scratch/runtime-source-organization/`: `file-map.json`, `export-snapshots.json`, `conservation-results.json`, `lesson-alias-map.json`, `verified-workspace-paths.json`, `asset-moves.json` and original source copies. The migration conservation check uses this captured before-state; it is dated migration evidence, not a rule forbidding future authorized lesson improvements.

Checks completed after the source split:

```text
node scripts/verify-runtime-source-organization.mjs
node scripts/verify-iteration-decorators.mjs
scratch/lesson-tools/Scripts/python.exe scripts/verify-iteration-decorators.py
node scripts/verify-reliability-lessons.mjs
node scripts/verify-thread-completion.mjs
node scripts/verify-workflow-visuals.mjs
node scripts/verify-scientific-visuals.mjs
node scripts/verify-data-foundations.mjs
node scripts/review-workflow-visuals.cjs
```

The data-foundations command used `LESSON_PYTHON=scratch/lesson-tools/Scripts/python.exe`. Verified behavior includes 25 iterator/decorator programs and 27 model configurations; 21 testing/notebook/API programs, real fresh notebook kernels and mypy acceptance/rejection; six thread programs and 73 reachable model states; scientific-file/SQL native examples and variants; CSV/NumPy/Pandas/SQLite/Matplotlib visual oracles; Bash argument/status and Git reference states. The final reliability rerun after verifier cleanup passed with evidence in `scratch/reliability-review/run-ije2aE`. Sixteen affected verification scripts also passed `node --check`. Browser review passed the workflow figures at 1440px and 390px, including keyboard controls, reset, no page errors and no page/figure overflow.

The coordinating task runs the integrated production build and loading/performance checks. This subtask did not deploy, alter stable topic IDs or rewrite educational explanations. Source splitting preserves behavior; observed learner effectiveness and user approval remain separate questions.
