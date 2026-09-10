# Learning code: ownership and browser loading

Updated 10 September 2026. This is the current engineering policy for educational code. Read it with the teaching standard: efficient delivery must preserve complete explanations, topic-specific visuals, accurate models and practice. The user's current request controls scope.

## File and identifier names

- Name a file for its topic or responsibility, never the implementation batch, number of topics, current queue or agent. Names such as `next-three-blueprints`, `programming-batch-two`, `NextThreeElements` and `WorkflowCompletionLabs` are retired production names.
- Authored plans live at `src/learn/data/curriculum/blueprints/<stable-topic-id>.js`. Each default export belongs to one topic; `blueprints/index.js` registers the authored plans. See [blueprint ownership](BLUEPRINT-ORGANIZATION.md).
- A published lesson lives in `src/learn/data/topics/<descriptive-topic-name>.jsx`. Prefer the existing stable topic ID for new files. Descriptive legacy filenames may remain when their publication mapping is explicit; changing a filename must not change progress IDs or URLs.
- Supporting examples and pure models use descriptive kebab-case names under `src/learn/data/`, such as `pandas-examples.js`, `sql-models.js` and `thread-coordination-models.js`. Keep nontrivial model logic separate from rendering so native examples and independent checks can verify it.
- React components use descriptive PascalCase names, such as `PandasPivotLab` or `IteratorFigures`; new component filenames should match. CSS filenames describe the component family or purpose. Retain useful established descriptive paths instead of making unrelated cosmetic migrations.
- Use descriptive camelCase identifiers for state, helpers and sample collections: `selectedAddress`, `currentWordSegments`, `pythonCoreExamples`. Prefer named example imports over ambiguous `ex`, `added`, `nextThree` or rollout flags. Boolean names should state the condition. Use conventional mathematical symbols and short local indices when those are genuinely clearer; do not rename lesson code's disciplinary terminology mechanically.
- Keep authored models, control flow and render helpers conventionally formatted and readable. Do not hand-minify whole functions or pack unrelated state changes into one line to save source length; production bundling owns minification. Format only files within the task's scope and preserve JSX text, comments and behavior. Algorithmic clarity and explicit invariants are part of maintainability.
- DSA practice data lives at `src/learn/data/practice/<stable-topic-id>.js`; the lesson imports only its own dataset and passes it to the presentation component. Follow [DSA-PRACTICE-STANDARD.md](../teaching/DSA-PRACTICE-STANDARD.md). Do not introduce an eager all-DSA question bundle or an in-page judge merely to show external practice links.
- Share a small component by its function, such as `LessonInvestigation`, `RunnableExample` or `LabControls`. A shared wrapper provides accessible behavior and styling; it must not force identical investigations. Do not create a generic component with dozens of unrelated topic modes or one file per trivial helper.
- Split large cross-topic bundles along real ownership/dependency boundaries. A selected topic should not import another topic's complete example collection or model merely to reuse one small control. Shared mathematical utilities and semantic styles can remain shared.
- Dated implementation reports and migration evidence can mention former batch names. They are historical records, not file-naming precedents or current authoring instructions. Avoid generating new production files from old batch scripts.

The [runtime source map](RUNTIME-SOURCE-ORGANIZATION.md) records the actual migration, scope and conservation checks.

The [integrated loading review](LEARNING-LOADING-REVIEW.md) records production measurements and recovery/navigation evidence. Hashed build-asset filenames are compiler-generated cache identities; their automatic names do not define source ownership or become authoring filenames.

## One source for publication; separate data for navigation

| Responsibility | Source |
| --- | --- |
| Module/section/topic reading order and membership | `src/learn/data/track-definitions.js` and its semantic curriculum sources |
| Full authoring plans, dependencies and notes | Curriculum blueprint sources and `docs/teaching/topic-notes/` |
| Published stable ID → lesson source | `src/learn/data/lesson-manifest.json` |
| Build/dev artifact generator | `scripts/generate-learning-artifacts.mjs` |
| Compact browser catalogue | `src/learn/data/catalogue.js` and generated navigation data |
| Current-lesson imports and request cache | `src/learn/data/lesson-loader.js` |
| Loading, route changes and recovery | `useTopicResource`, `TopicContent`, `LessonBoundary` |
| Shared route rules for both environments | `src/learn/data/curriculum/navigation.js` |
| Full-source route adapter for authoring tools | `scripts/lib/authoring-curriculum.mjs` |

The former `topics/index.js` eager registry is removed. Do not recreate a barrel that imports every lesson and then exports their content functions. Do not import `track-definitions.js`, the full authoring `topic-catalogue.js`, domain plans or authored blueprint aggregates from browser navigation/search components.

`generated/navigation.js` contains only fields used by navigation, search, prerequisites, status and the reader header. It has no React components, complete examples, source ledgers or full blueprints. `generated/lesson-imports.js` contains explicit dynamic imports for registered lessons only. Unregistered drafts remain outside the production build. `generated/outlines/<stable-topic-id>.json` contains only the learner-facing outline for an unpublished topic with a plan. Published lessons do not fetch their authoring plans.

Generated files are deterministic outputs; never edit them manually. The generator parses static `title`, `readTime` and optional boolean `hasIntegratedGuide` from the lesson's default-exported object, without evaluating its component or imports. Keep these fields static; introduce a deliberate schema change if richer metadata is necessary. It validates publication paths, catalogue membership, distinct ownership and required content. New lesson registration requires a manifest entry and real complete content; a blueprint alone remains planned.

Vite regenerates artifacts at build/dev startup and when relevant source metadata changes during development. It writes only changed files. Run `node scripts/generate-learning-artifacts.mjs` after source changes outside Vite, and `node scripts/generate-learning-artifacts.mjs --check` to detect stale outputs. Authoring CLI tools read full sources through the authoring adapter, so topic notes and curriculum verification do not silently depend on stale browser snapshots.

## Runtime behavior to preserve

1. The hub loads the compact catalogue and UI without downloading lesson bodies, outlines or lab engines.
2. Opening a published topic imports that topic and its actual static dependencies. Opening a planned page imports only its own outline when one exists. Do not prefetch every route or eagerly evaluate all examples.
3. Keep course metadata, current position, prerequisites and Previous/Next available while loading. Publication status comes from the manifest, not the success or failure of a network request. A broken published lesson is never disguised as a planned one.
4. Loading has an accessible status. Import and render failures have a visible recovery message. Retry evicts rejected application promises; an explicit page reload remains available because browsers may retain failed module imports. Do not claim retry always bypasses the browser's module cache.
5. Ignore asynchronous results from an old route. Reset mounted lesson state when the topic changes. Late imports cannot replace the new topic's body. Keep direct links, hash targets, module context, browser Back and progress identities working.
6. Completion stays disabled until a published lesson loads and renders successfully. Errors and planned pages cannot become completed through the footer.
7. Reuse successfully imported modules on revisit. ES modules remain cached by the browser; this is not a claim that visited JavaScript can be unloaded. Unmount inactive components and clean up listeners, timers, observers, animation frames and workers.

## Computation and payload decisions

Measure a real problem before adding complexity. The 2026 migration addressed an observed all-lessons download; it did not remove examples or simplify teaching. Do not blanket-memoize cheap operations, add workers for tiny calculations, virtualize explanatory prose or split every small component into an extra network request.

Keep interactive work bounded: validate user input, make reset deterministic, and avoid unconstrained loops or rendering one DOM node per unbounded data item. For costly investigations, use a measured need to choose cached pure results, explicit run controls, incremental work, interaction/visibility-triggered loading or a worker. Explain any simulation/data-size limits honestly and preserve the actual mathematical model. A static code example must not automatically execute as part of rendering.

Load large specialist dependencies with their consuming lesson/investigation. Mathematical rendering and its styles belong to mathematical content, not the global hub/reader shell. Share dependencies through normal imports, then inspect the production graph: filenames alone do not prove that a bundle is isolated. Preserve mobile, keyboard, reduced-motion and text/data alternatives while optimizing.

## Verification when changing this structure

- Run `node scripts/verify-learning-artifacts.mjs`, `node scripts/verify-curriculum.mjs`, the inventory generator and the application build. The artifact check compares every browser route against fresh authoring sources, including planned topics, shared memberships and prerequisite inclusion.
- Run the relevant model/native-example checks after source moves. Preserve exact code/output fixtures and exported behavior; a passing JSX parser is insufficient.
- Run `node scripts/verify-content-import-boundary.mjs` when changing shared content imports. Import `Math`/`MathBlock` directly from `components/content/Math.jsx`; the generic content barrel must remain free of the math engine and its styles.
- Against a production preview, run `node scripts/verify-learning-load-boundaries.cjs` with Playwright configured. It checks actual requests against Vite's manifest, lesson/outline separation, revisit reuse, slow and failed loads, render errors, retry/reload and stale navigation.
- Measure before and after using `scripts/measure-learning-performance.cjs <label>` and `scripts/compare-learning-performance.cjs`. Use fresh contexts and matching routes, report compressed and decoded bytes separately, and retain the methodology. Do not substitute a build's total disk size for a page's downloaded payload.
- Open representative affected lessons at desktop and narrow widths and exercise their controls after moving CSS/components. Retain teaching-specific checks rather than replacing them with performance-only tests.

Do not raise a warning threshold just to hide a large chunk. The catalogue remains shared because search and path navigation need it; if its measured cost becomes material, evaluate compact encoding or route/search partitioning with the same behavior and accessibility checks. Do not promise a universally optimal bundle or universal page-speed improvement from local measurements.

Implementation references: [Vite glob and lazy imports](https://vite.dev/guide/features.html#glob-import), [React render error boundaries](https://react.dev/reference/react/Component#catching-rendering-errors-with-an-error-boundary), and [MDN lazy loading](https://developer.mozilla.org/en-US/docs/Web/Performance/Guides/Lazy_loading). These explain mechanisms; repository builds and browser checks establish behavior with the installed versions.
