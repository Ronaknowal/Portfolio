# Learning code: ownership and browser loading

For hub/navigation or end-to-end project work, also read [the learning workspace architecture](LEARNING-WORKSPACE.md). Projects use separate compact metadata, stage content loaded on demand, canonical downloadable programs and their own learner/authoring progress. Preserve existing lesson routes, ordered curriculum and source-bound reviews when extending discovery.

For learner-facing algorithm code, also follow the teaching standard's [scratch/library implementation contract](../../LESSON-TEACHING-STANDARD.md#build-the-mechanism-then-control-the-library). Use semantic owners for reusable primitives and complete downloadable programs. Generated displayed copies must come from the canonical program and be checked for agreement; independent hand-edited copies are a drift risk. Show instructional program views in the reading flow without an opening click. Large source may load automatically as its view approaches the viewport; keep it keyboard-scrollable without page overflow. Inspectable teaching code still needs appropriate algorithms, stable arithmetic, efficient memory use, explicit contracts and meaningful tests. A readable scalar trace can introduce a vectorized or asymptotically better final route. Do not claim optimal performance without a stated workload/cost model and, for empirical claims, measurements.

Updated 11 September 2026. This is the current engineering policy for educational code. Read it with the teaching standard: efficient delivery must preserve complete explanations, topic-specific visuals, accurate models and practice. The user's current request controls scope.

## File and identifier names

- Name a file for its topic or responsibility, never the implementation batch, number of topics, current queue or agent. Names such as `next-three-blueprints`, `programming-batch-two`, `NextThreeElements` and `WorkflowCompletionLabs` are retired production names.
- Authored plans live at `src/learn/data/curriculum/blueprints/<stable-topic-id>.js`. Each default export belongs to one topic; `blueprints/index.js` registers the authored plans. See [blueprint ownership](BLUEPRINT-ORGANIZATION.md).
- A published lesson lives in `src/learn/data/topics/<descriptive-topic-name>.jsx`. Prefer the existing stable topic ID for new files. Descriptive legacy filenames may remain when their publication mapping is explicit; changing a filename must not change progress IDs or URLs.
- Content-first manuscripts and visual/lab specifications live in `docs/teaching/drafts/<stable-topic-id>/lesson.md` and `visual-specifications.md`. They are complete teaching inputs awaiting implementation, outside browser imports. Keep additional required draft inputs in the same topic directory; no temporal batch names. Follow [the delivery modes](../../LESSON-TEACHING-STANDARD.md#delivery-modes-and-stopping-boundaries) and [phase-ledger contract](../teaching/LESSON-DELIVERY-LEDGER.md).
- Supporting examples and pure models use descriptive kebab-case names under `src/learn/data/`, such as `pandas-examples.js`, `sql-models.js` and `thread-coordination-models.js`. Keep nontrivial model logic separate from rendering so native examples and independent checks can verify it.
- React components use descriptive PascalCase names, such as `PandasPivotLab` or `IteratorFigures`; new component filenames should match. CSS filenames describe the component family or purpose. Retain useful established descriptive paths instead of making unrelated cosmetic migrations.
- Use descriptive camelCase identifiers for state, helpers and sample collections: `selectedAddress`, `currentWordSegments`, `pythonCoreExamples`. Prefer named example imports over ambiguous `ex`, `added`, `nextThree` or rollout flags. Boolean names should state the condition. Use conventional mathematical symbols and short local indices when those are genuinely clearer; do not rename lesson code's disciplinary terminology mechanically.
- Keep authored models, control flow and render helpers conventionally formatted and readable. Do not hand-minify whole functions or pack unrelated state changes into one line to save source length; production bundling owns minification. Format only files within the task's scope and preserve JSX text, comments and behavior. Algorithmic clarity and explicit invariants are part of maintainability.
- Displayed teaching programs (the code a learner reads on the page) keep the mechanism compact and visible: validation lives in one small helper or a single finiteness check, guards for pathological inputs belong in the verifier or in prose unless a learner could plausibly hit them, and code never prints cautionary sentences. Browser teaching models may carry stricter range guards because they are not displayed, but their fixtures must be checked to exhibit the behavior the lab promises, with critical tie values chosen to be exactly representable.
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

Compact `subtopics` labels may appear in generated navigation for discovery and planned-page scope. Their semantic curriculum coverage files own them; never maintain a separate hand-edited search-keyword list. They are teaching obligations, not publication or review status. Search indexes titles and these labels once, without importing full authoring plans or loading unvisited lessons. Future topic authors must assess each label returned by preflight, teach it at the appropriate depth or explicitly move its ownership with a reason. A name in the index alone is not a completed explanation.

Generated files are deterministic outputs; never edit them manually. The generator parses static `title`, `readTime` and optional boolean `hasIntegratedGuide` from the lesson's default-exported object, without evaluating its component or imports. Keep these fields static; introduce a deliberate schema change if richer metadata is necessary. It validates publication paths, catalogue membership, distinct ownership and required content. New lesson registration requires a manifest entry and real complete content; a blueprint alone remains planned.

`hasIntegratedGuide` is retained as legacy metadata; it no longer controls whether the reader shows its shared opening. Do not set or clear it to create an alternate lesson layout.

The reader renders `readTime` as supplied. Use an explicit learner-facing estimate with units, such as `~60 min read + 90 min practice`, rather than a bare number. Check the generated header as well as the lesson body. Before producing data for a shared component, read its actual field contract: for example, `RunnableExample` renders its recorded output from `example.expected`. If a topic-owned execution record calls that field `output`, map it explicitly at the boundary and verify the actual displayed code/output text. An existing component on screen does not prove that all its data was rendered.

Vite regenerates artifacts at build/dev startup and when relevant source metadata changes during development. It writes only changed files. Run `node scripts/generate-learning-artifacts.mjs` after source changes outside Vite, and `node scripts/generate-learning-artifacts.mjs --check` to detect stale outputs. Authoring CLI tools read full sources through the authoring adapter, so topic notes and curriculum verification do not silently depend on stale browser snapshots.

### Early compatibility checks

At the first runnable implementation, check the actual consumers before producing more code around an incompatible shape. This supports stage 3 of the [authoring workflow](../../LESSON-TEACHING-STANDARD.md#six-stage-authoring-workflow). Content-first work may inspect these contracts to make feasible specifications, but does not build React/lab components, change live publication/navigation or run an implementation campaign. Pending manuscripts are not production lessons.

- Use the metadata extractor's supported lesson form: a default-exported object literal, or a plain top-level `const lessonDefinition = { ... }` followed by `export default lessonDefinition`. The current extractor does not resolve a variable declaration wrapped in a named export. Keep header fields static strings, with explicit reading/practice units.
- Match the current blueprint schema: URL strings in `sources`, an allowed `depth`, a specific `reviewFocus`, exact prerequisite titles and the required outcome/sequence/visual/practice fields. The detailed source annotations belong in the design record and lesson references.
- Verify that the real reader displays the intended example code and output, and that prerequisite links use actual module IDs. Check a representative narrow layout early when the lesson introduces a new visual form.
- For changed lesson metadata/publication or blueprint structure, run the relevant artifact-generation/check and curriculum checks at this point. A build/dev generator failure must be resolved before claiming a working draft. Reuse passing results for unchanged contracts; do not run the production build after every edit. The increment owner performs the final shared build/integration when the increment is ready and reruns it only after changes or failures that affect it.

## Runtime behavior to preserve

1. The hub loads the compact catalogue and UI without downloading lesson bodies, outlines or lab engines.
2. Opening a published topic imports that topic and its actual static dependencies. Opening a planned page imports only its own outline when one exists. Do not prefetch every route or eagerly evaluate all examples.
3. Keep course metadata, current position, prerequisites and Previous/Next available while loading. Publication status comes from the manifest, not the success or failure of a network request. A broken published lesson is never disguised as a planned one.
4. Loading has an accessible status. Import and render failures have a visible recovery message. Retry evicts rejected application promises; an explicit page reload remains available because browsers may retain failed module imports. Do not claim retry always bypasses the browser's module cache.
5. Ignore asynchronous results from an old route. Reset mounted lesson state when the topic changes. Late imports cannot replace the new topic's body. Keep direct links, hash targets, module context, browser Back and progress identities working.
6. Completion stays disabled until a published lesson loads and renders successfully. Errors and planned pages cannot become completed through the footer.
7. Reuse successfully imported modules on revisit. ES modules remain cached by the browser; this is not a claim that visited JavaScript can be unloaded. Unmount inactive components and clean up listeners, timers, observers, animation frames and workers.

## Shared lesson opening and section navigation

`TopicContent` places one `LessonGuide` directly below the lesson title/meta. `LessonOpeningContext` lets lesson-owned guidance render in that opening through React portals. Keep the common neutral/amber presentation and responsive, keyboard-accessible links; retain topic-specific layouts and scientific encodings in the body.

- Use `LessonIntro` from `components/lesson-labs/LessonElements.jsx` for an existing lesson summary and `prerequisites`, or `LessonOrientation` from `components/LessonOpening.jsx` for new direct use. Their contents belong to the opening. The old `LessonIntro.sections` prop no longer defines navigation; do not add new partial route arrays.
- Mark guidance paragraphs explicitly with `<Prose opening="route">`, `<Prose opening="prerequisites">`, or `<Prose opening="exploration">`. `opening="summary"` is available for a summary without `LessonIntro`. Preserve the authored inner JSX, including links and locally explained terms. For a structured guidance fragment, use `<LessonOpeningNote kind="route">` (or another supported kind) directly.
- Keep the explanation of the learner's actual problem in the body. Register guidance deliberately; do not move arbitrary introductory prose by keyword, hide it with CSS, or add another bespoke compass/TOC. The shared note components retain their contents inline when rendered outside the reader.
- Use meaningful actual H2s for navigable lesson sections. `lesson-navigation.js` collects them after the selected lesson mounts, excluding local lab, aside/navigation and disclosure headings. The collection includes wrapped/template-generated section headings. Keep sections that need a main TOC entry outside local disclosures. H3s and lab controls remain local content.
- Preserve stable IDs and existing wrapper anchors. Retain authored section numbers because prose and practice may refer to them; unnumbered sections must not shift those numbers. Display each number once. Heading display normalization must not rename existing fragments.

The authoring-only `scripts/lib/prepared-lesson-renderer.mjs` supports explicit Markdown paragraph annotations:

```js
renderPreparedLesson(manuscript, {
  assetBase: `/learn-assets/${id}/`,
  preserveOpeningFrom: `src/learn/data/topics/${id}.jsx`,
  opening: [
    ["**First pass:**", "route"],
    ["**Explore as you read.**", "exploration"],
  ],
});
```

Each `opening` entry is an exact Markdown prefix and one of `summary`, `route`, `prerequisites`, or `exploration`. It must select exactly one paragraph; missing, overlapping and repeated matches fail. This is an explicit author choice, not a general prose classifier. Use the paragraph's actual prefix, including Markdown formatting.

When regenerating an existing lesson, `preserveOpeningFrom` reads its explicit `<Prose opening="…">` annotations before the destination is written. It restores them only when each previous paragraph has one exact static-text match in the generated JSX, with formatting whitespace normalized. It preserves generated inner JSX and links. A missing/changed/ambiguous paragraph, unsupported dynamic annotation or conflicting kind fails visibly so the author can reconcile the source and generator instead of silently scattering guidance again. A new destination has no previous annotations; supply `opening` entries for its guidance. Bespoke generators that do not use this renderer must preserve the same attributes themselves. Do not regenerate a completed lesson merely to apply a navigation convention or change its delivery ledger.

After navigation changes, check one opening, actual section coverage, unique fragments, native hash/Back behavior, focus, narrow-width bounds and unchanged topic/progress identity. Reuse numerical evidence for unchanged mechanisms. `scripts/check-lesson-opening-authoring.mjs` verifies annotation preservation and its failure cases without running topic generators or scientific programs.

## Shared lesson endings

Give recurring closing material explicit authored boundaries. Use `section.lesson-ending` with `data-lesson-ending` for its purpose and the shared `lesson-ending--practice`, `lesson-ending--next` or `lesson-ending--resources` presentation. Keep practice, readiness/next, further learning and technical references distinguishable; preserve topic-specific titles, authored section numbers, stable heading IDs and every existing content node. A later substantive teaching branch is not a closing section merely because its title includes “next”, “deeper” or “connections”. Do not classify or move ending content by runtime text matching or DOM scanning.

Within final practice, `lesson-exercise` marks the existing task boundary. Keep the prompt visible, use native `details`/`summary` for authored hints and solutions, and preserve their distinct labels and contents. An in-body `Checkpoint`, a lab's live output and an independent final exercise have different roles. Do not wrap all checkpoints or all disclosures globally, manufacture missing feedback, or rewrite a topic's practice helper merely to impose identical teaching structure.

`Sources` in `lesson-labs/LessonElements.jsx` uses its existing `alternatives` prop for a normal H2 “Further learning” section (`data-lesson-ending="further-learning"`) and its children for a separate H2 “Technical references” section (`data-lesson-ending="references"`). Both use `lesson-ending--resources` and a `lesson-resource-list` body. Keep every resource annotation and optional-reference note. Topic-authored mixed lists need an explicit content-aware boundary decision; URL patterns do not establish learning purpose. Use normal semantic sections so actual main headings enter the shared opening's full TOC.

Prepared authoring supports explicit `endings` entries shaped as `{ level: "H2" | "H3", title: exactAuthoredTitle, kind: "practice" | "next" | "resources" | "further-learning" | "references" }`. The renderer's `preserveOpeningFrom` destination also retains existing ending annotations through regeneration. Keep matching exact and fail visibly when a recorded boundary cannot be preserved. A prior custom range without a direct static H2/H3, such as a group of task components or a topic-local `Section`, makes generic regeneration fail with an explicit preservation error; it must never silently drop the wrapper. Preserve that range through the bespoke generator or an explicit supported adapter before regenerating. Bespoke generators must retain the same markers. Do not regenerate completed scientific content or refresh ledger evidence solely to apply presentation conventions.

For an explicitly reviewed mixed ending whose lists contain resources, an entry may set `resourceList: true`; the generated `data-lesson-resource-list` marker preserves that list presentation through regeneration without changing the section's readiness/next purpose. Do not infer this flag from link destinations or apply it to a mixed section containing ordinary readiness lists. Keep a practice section's identifying H3 outside its individual exercise wrappers.

For ending-only changes, verify preservation of text, links, IDs and authored node order; distinct purpose boundaries; actual TOC links; native disclosures, keyboard focus and narrow-screen bounds. Reuse unchanged scientific evidence. Shared spacing, surfaces and typography should support each topic's content rather than force all endings into a fixed sequence or quota.

## Visible teaching and consistent code access

Explanations, derivations, diagrams, lab output, worked teaching examples and instructional code belong in the visible reading flow. Optional depth can remain clearly labeled without a disclosure gate. Do not require a dropdown, expand button or “show code” click to read teaching material, and do not simulate visibility with `details open`, which still permits collapsing it. Keep authored practice, in-body retrieval prompts, hints and solutions collapsible when they already serve an independent attempt. A disclosure's role depends on its actual content and context; words such as “solution” can also describe a mathematical method, and an explanatory branch can occur inside a practice section.

Topic-owned visible branches use an ordinary `section` with `className="lesson-teaching-section"` and `data-lesson-teaching=""`; preserve any existing classes, styles and IDs. Its former summary becomes a semantic heading with `lesson-teaching-section__title`. Use H3 for a local teaching branch, H2 for a genuine main section, and an appropriate subordinate heading within a lab. Preserve every child, diagram, program and its order. Never move a nested practice answer out of its own disclosure when revealing its surrounding explanation.

Prepared authoring accepts explicit `teaching: [{ summary: exactAuthoredSummary, headingLevel: "h2" | "h3" | "h4" }]` entries (`h3` by default). `preserveOpeningFrom` also preserves the destination's visible-teaching annotations. `scripts/lib/lesson-teaching-disclosures.mjs` matches the exact static summary once and refuses missing, ambiguous, conflicting or stateful conversions; dynamic teaching helpers require an explicit adapter instead of silently returning to a dropdown. Topic-specific generators must preserve the same visible structure. This is authoring-time source transformation, never a browser text classifier. `scripts/check-lesson-teaching-visibility.mjs` verifies the scoped migration's complete source-tree conservation and byte-identical retained practice disclosures.

Use the shared code display and controls for every instructional code block, including short snippets and full programs. Provide consistent, keyboard-accessible Copy and Download actions for the exact displayed source. Retain canonical filenames and file types for real programs; distinguish a downloadable snippet from a complete executable file. Use the common lesson file index for actual downloadable assets, keeping useful labels, descriptions and existing context links. Do not add custom toolbar/download-list variants, duplicate hand-maintained source, invent files or metadata, or reveal answer-only material outside its practice context. Loading code automatically near the viewport must retain visible loading/error feedback and must not eagerly download every lesson's assets.

Implementation entry points are `components/content/Code.jsx` (`CodeBlock` for
inline source) and `components/content/RemoteCodeBlock.jsx` (canonical local
programs loaded automatically near the viewport). Supply `filename` whenever
the prose tells the learner to save/import/run a named file; pass `kind="output"`
or `language="output"` for recorded results. A `downloadUrl` must identify the
same complete source that is displayed. For an excerpt, omit it so Download
saves that exact excerpt. Keep snippets and outputs inside their authored
practice context when appropriate.

`TopicContent` supplies the lesson-scoped registry and `LessonCodeDownloads`
index automatically. Do not append another custom file index. Local asset
namespaces include `/learn-code/`, `/learn-assets/` and `/learn/examples/`;
use real, deployed files and preserve original filenames. Register future
asynchronously inserted source with the shared components so its file appears
without scanning or fetching unrelated lessons. The viewer preserves code
whitespace, uses local scrolling and wrapping toolbars, and reports unavailable
copy/load operations. Check filename agreement, exact copied/downloaded payload,
viewport loading, practice separation, keyboard access and narrow layout when
changing it. `scripts/check-lesson-code-controls.mjs` exercises the actual
shared handlers; native browser integration needs its own observed evidence.

Copy success must be visible at the clicked button: briefly show a checkmark and
“Copied” in the amber theme, then restore “Copy”. Reserve the button width to avoid
shifting adjacent controls. Confirm only after the Clipboard API succeeds, keep
an accessible live announcement, show failures beside the controls, restart the
feedback interval on repeated clicks, and clear pending timers on unmount.

## Diagram layout and SVG legibility

Use topic-specific representations while keeping a reliable layout contract. Quantitative coordinates belong to the model; room for prose belongs to the layout. Give annotations their own measured space instead of choosing a repeated row stride smaller than the row's actual contents. Reflow headings, legends and before/after values in semantic HTML; use CSS grid/flex wrapping and separate compact SVGs when that preserves the concept better than one densely positioned drawing.

Supply an intentional viewBox and preserve aspect ratio. Match physical axis scales when interpreting distances, angles or circles. Constrain desktop enlargement and keep phone text readable; stacking is usually better than scaling a crowded multi-panel drawing into a thumbnail. Plan padding for full glyph bounds, including minus signs, decimals and end ticks. Long numerical labels may need a separate readout or leader, not a displaced data point. Do not fix collisions with overflow hiding, text truncation, nonuniform scaling or unexplained clipping of model coordinates. Set chart domains from every encoded value, including individual repeats/folds and computed optima, rather than just their averages. A stated proportional bar must not acquire an unexplained minimum data length.

During affected browser verification, check text/text and foreground-line/text intersections as well as SVG/page bounds. The read-only `scripts/audit-lesson-visual-layout.cjs --topics <stable-id,...>` scans rendered SVGs at 1366, 390 and 320 px by default; use `--widths` for intermediate/breakpoint cases and `--output` for a semantic evidence file. Its reusable inspector is `scripts/lib/lesson-visual-layout.cjs`. Supply `PLAYWRIGHT_PACKAGE` and `LEARNING_BASE_URL` as for other browser checks. `--all-published` is for an explicitly scoped cross-site check, not a mandatory full-site run after every edit. The tool is author-only and must never enter a lesson's browser bundle.

Treat results as triage: it inspects visible SVG text and straight foreground lines in the rendered states, omits hidden branches and canvas/HTML visuals, and can flag intended overlaps. Use actual screenshots and source context to disposition candidates. For a confirmed defect, add a focused regression using final font geometry and an informative fixture. A zero-candidate report cannot replace the teaching standard's figure-by-figure visual pass.

Verify the actual painted encoding as well as geometry. CSS can override SVG presentation attributes, so a correct `stroke` attribute alone does not prove that an arrow matches its legend. Check computed styles in the affected state. Table overflow cues must remain distinct from data encodings; decorative fades must not obscure values or look like a heatmap. Preserve scrolling and keyboard access when replacing a cue.

If a label needs an opaque background stroke or backplate, keep that background opaque in every state. Dimming the entire parent group can make the line behind the label show through again; dim the relevant glyph/mark separately. For coincident or nearby data positions, prefer a separate wrapping key or intentionally separated annotation rows rather than moving the quantitative marks apart. Check the built-in coincident/boundary presets and relevant updated state as well as the opening view.

## Computation and payload decisions

Lesson text links inherit the shared gold theme from `components/topic-content.css`, including visited links, with the global visible keyboard-focus outline. Keep this baseline in the reader rather than requiring a particular topic wrapper or repeating link colors in every lesson. Topic-specific styles may override it deliberately; check actual link colors and hover/focus states during the affected visual review.

Measure a real problem before adding complexity. The 2026 migration addressed an observed all-lessons download; it did not remove examples or simplify teaching. Do not blanket-memoize cheap operations, add workers for tiny calculations, virtualize explanatory prose or split every small component into an extra network request.

Keep interactive work bounded: validate user input, make reset deterministic, and avoid unconstrained loops or rendering one DOM node per unbounded data item. For costly investigations, use a measured need to choose cached pure results, explicit run controls, incremental work, interaction/visibility-triggered loading or a worker. Explain any simulation/data-size limits honestly and preserve the actual mathematical model. A static code example must not automatically execute as part of rendering.

Implement labs as live exploration with a single coherent current input/model state. Do not retain prediction fields, answer choices, commitment state, correctness comparisons, answer masking or prediction reveal buttons, even as optional features. Valid slider, numeric, drag or toggle edits update all dependent views and causal readouts together. Opening a lab already displays its current result. A process step/run advances the modeled process; it never unlocks an answer after a quiz. Keep separate independent practice and its solutions outside this interaction contract.

Use a pure derived model for cheap live computations, bounded input ranges, and deterministic reset. Do not launch expensive training on every keystroke. If profiling justifies debounce, incremental work, a worker or an explicit bounded run, show current versus pending inputs, preserve a readable last valid result, cancel obsolete work and reject stale asynchronous completion. Numeric text may have a temporary invalid edit buffer with a local explanation; it must not silently clamp to another value. Pair draggable controls with keyboard/numeric equivalents and preserve focus during updates. Avoid flooding screen readers on each animation frame; make current exact values available and announce meaningful settled changes.

Presets obey the same bounds as manual edits. Mathematical helpers accept legitimate derived values outside a slider interval; never clamp a computed inverse to fit a control. Show enough precision to interpret equality, small differences and boundary cases, with declared tolerances where appropriate. Explain null results from the actual operation, including identity inputs, zero contributions and tolerance-only equality. Reset restores dependent process state and view settings as well as visible inputs. A pinned comparison or saved run retains its original parameters, units, seed and output identity; later edits never relabel it. Test live changes, fast consecutive changes, null/invalid cases, reset and agreement between linked views with focused regressions.

Check the actual browser range serialization against the pure model at interior ticks and endpoints. Irrational maxima may round just below a structural threshold; use a normalized control coordinate with an exact endpoint mapping when needed, rather than weakening mathematical comparisons. Keep range steps consistent with exact numeric twins. A label containing an `<output>` can bind to that output instead of its input: use explicit unique `htmlFor`/`id` pairs and test accessible names. For browser pointer tests, settle scrolling/layout before measuring coordinates; a test that drags at stale screen coordinates is not evidence of a broken slider. Long inline code and token labels must reflow without changing copied text or token identity, while actual code blocks retain deliberate scrolling.

Editable numeric text must round-trip through its parser: do not populate a probability-row editor with rounded thirds or display-only scientific notation. Readouts support every computed outcome, including counts beyond the initial range and undefined quantities. Conditional probabilities require a possible conditioning event; a formula's cancelled ratio is not a value at a zero denominator. Compute baseline and edited results under the same tie rule and distinguish unchanged categories from unchanged numbers. Verify labels for meaningful small and extreme values, not only default fixtures; compact diagram labels may defer precision to an adjacent accessible table.

Load large specialist dependencies with their consuming lesson/investigation. Mathematical rendering and its styles belong to mathematical content, not the global hub/reader shell. Share dependencies through normal imports, then inspect the production graph: filenames alone do not prove that a bundle is isolated. Preserve mobile, keyboard, reduced-motion and text/data alternatives while optimizing.

## Verification when changing this structure

Apply the teaching standard's bounded verification policy: reuse passed evidence for unchanged source, rerun affected checks after a relevant change, and keep completed historical rollouts closed. The checks below apply when the associated structure changes; they are not a mandatory full-suite loop for each prose edit or resumed session.

For changes limited to delivery policy, the phase ledger and authoring-only CLI logic, run the phase-state/continuation tests and inventory generation. A new website build or browser review is required only when runtime, catalogue, publication or loading behavior actually changes. The delivery ledger is authoring data; keep it out of browser imports.

- Run `node scripts/verify-learning-artifacts.mjs`, `node scripts/verify-curriculum.mjs`, the inventory generator and the application build. The artifact check compares every browser route against fresh authoring sources, including planned topics, shared memberships and prerequisite inclusion.
- Run the relevant model/native-example checks after source moves. Preserve exact code/output fixtures and exported behavior; a passing JSX parser is insufficient.
- Run `node scripts/verify-content-import-boundary.mjs` when changing shared content imports. Import `Math`/`MathBlock` directly from `components/content/Math.jsx`; the generic content barrel must remain free of the math engine and its styles.
- Against a production preview, run `node scripts/verify-learning-load-boundaries.cjs` with Playwright configured. It checks actual requests against Vite's manifest, lesson/outline separation, revisit reuse, slow and failed loads, render errors, retry/reload and stale navigation.
- Measure before and after using `scripts/measure-learning-performance.cjs <label>` and `scripts/compare-learning-performance.cjs`. Use fresh contexts and matching routes, report compressed and decoded bytes separately, and retain the methodology. Do not substitute a build's total disk size for a page's downloaded payload.
- Open representative affected lessons at desktop and narrow widths and exercise their controls after moving CSS/components. Retain teaching-specific checks rather than replacing them with performance-only tests.
- Inline figure controls share the lesson theme contract with full labs. `topic-content.css` supplies low-specificity dark/amber defaults; topic styles may override these for meaningful encodings and geometry. Test computed styles and selected/disabled/focus states outside named lab wrappers too. A shared fallback must not distort compact graph/matrix controls. For draggable SVG marks, invert the actual screen transform, preserve the grab offset, capture/release the pointer, keep the domain stable during the gesture, and provide keyboard-equivalent edits with accessible values. Verify mouse/touch gestures in the rendered layout, not only synthetic state changes. Static marks must not impersonate controls.

Do not raise a warning threshold just to hide a large chunk. The catalogue remains shared because search and path navigation need it; if its measured cost becomes material, evaluate compact encoding or route/search partitioning with the same behavior and accessibility checks. Do not promise a universally optimal bundle or universal page-speed improvement from local measurements.

Implementation references: [Vite glob and lazy imports](https://vite.dev/guide/features.html#glob-import), [React render error boundaries](https://react.dev/reference/react/Component#catching-rendering-errors-with-an-error-boundary), and [MDN lazy loading](https://developer.mozilla.org/en-US/docs/Web/Performance/Guides/Lazy_loading). These explain mechanisms; repository builds and browser checks establish behavior with the installed versions.

## Efficient context, tools and coordination

Reduce coordination and diagnostic overhead while preserving full lesson quality. Follow the standard's [source-bound review handoff](../../LESSON-TEACHING-STANDARD.md#source-bound-evidence-and-review-handoff); this section adds engineering practices, not more review stages.

The [quality-first principle](../../LESSON-TEACHING-STANDARD.md#quality-takes-priority-over-efficiency) governs every efficiency suggestion below. These are defaults that the agent may override for better output quality within the authorized scope. Read more context, make a broader coherent edit, run further relevant checks or use additional authorized review when that better protects correctness, completeness or comprehension. Do not force a smaller patch, shorter response, reused result or smaller team when it would make the result weaker.

- **Retrieve only what answers the current question.** Load the required current policies once while they remain available in context, then use the scoped design, source and checkpoint. Read a complete lesson in sensible sections when reviewing its teaching; do not repeatedly reload unchanged policies or whole historical reports. After a context reset, recover the current instructions and missing evidence rather than guessing them from memory.
- **Keep tool output useful.** Search exact directories/symbols and request bounded ranges. For large JSON or generated one-line navigation files, parse and print the selected fields; do not dump the entire catalogue. Return a check's exit status, scope, counts and failures, with a path to necessary detailed evidence. Inspect truncated or failing results through focused follow-ups; a short success summary cannot hide unread errors. Batch independent reads/checks, but serialize dependent edits, generation and builds.
- **Edit the affected source.** Patch the smallest coherent section and preserve formatting/line endings and useful content. Avoid regenerating a complete long lesson or every native example for a metadata/label fix. Reuse existing semantic generators, checks and recorded runtimes where they fit. Diagnose a harness, selector, environment or capture problem before treating it as a lesson defect; rerun the affected check after the actual cause is repaired.
- **Coordinate parallel work deliberately.** When delegation is authorized and useful, assign a bounded outcome, necessary context paths, exact file ownership, existing evidence to reuse and completion criteria. Use the smallest useful team; extra agents are not automatically a token saving. Give different writers disjoint files, or serialize work that shares them. One named owner changes shared registries, generated metadata and the increment ledger. Return findings and evidence references instead of copying full lessons or raw logs into coordination messages. Preserve enough context for the task's reasoning, and avoid repeated unchanged status polling.
- **Keep one current next action.** Update the central delivery ledger's two phases at a substantive transition, with the existing topic record holding owner, findings and evidence. Historical increment ledgers retain prior proofs, not competing phase queues. A content-first handoff ends before implementation; shared integration begins only after phase-two authors finish the versions being integrated. Changes require an affected-check decision, not an automatic return to stage 1.

These practices reflect this repository's observed late metadata fixes, source-handoff changes and oversized diagnostic output. Official OpenAI guidance also recommends focused task context and concise durable instructions, and describes the benefits and coordination costs of bounded subagent work: [best practices](https://learn.chatgpt.com/guides/best-practices), [subagents](https://learn.chatgpt.com/docs/agent-configuration/subagents) (relevant context/coordination sections read 11 September 2026). They do not establish a measured token reduction for this project. Keep current model/reasoning settings unless a change is requested or otherwise explicitly authorized.

## Temporary work and evidence retention

Follow [the working-artifact retention workflow](WORKING-ARTIFACT-RETENTION.md) at topic completion. It covers screenshots, datasets, scripts and temporary outputs inside and outside `scratch/`, including reference checks and removal of superseded captures.

- Before implementation, the ledger-bound content-first manuscript/specifications are authoritative and must be retained. Once incorporated and reviewed, production source becomes authoritative; update the content checkpoint to retained equivalent source/specifications before retiring a redundant manuscript. Never remove the only pending handoff. Remove superseded scratch drafts and one-off installers that could overwrite newer work. Put reusable generators and checks in semantic files under `scripts/`, with their input/output contracts documented.
- Use a topic-named scratch directory for active working files. Do not create a new folder or script for every tiny correction. Remove disposable patch scripts, failed downloads, unused captures and generated caches after the relevant result is recorded.
- Preserve original-source archives, source-versioned evidence, referenced final screenshots and native outputs, required script inputs, active datasets and shared tools. A path is not disposable merely because it is ignored by Git or contains `scratch` in its name. Check references and active ownership before removal; keep uncertainty visible rather than deleting unknown work.
- Keep durable evidence summaries and the current progress ledger under `docs/teaching/`. Existing historical evidence may retain scratch attachments at their recorded paths; do not relocate or delete those attachments without preserving access and updating the relevant consumers. Retention does not mean rerunning the historical checks.
- `scratch/` is excluded from normal Git/ripgrep searches. Search current source, the scoped design and the current ledger first. Open an exact retained artifact only to answer a concrete outstanding question; do not recursively inventory the shared runtime, datasets or old review directories on every session.
- Cleanup and documentation changes need reference/path/source-preservation checks, not a fresh application build or every lesson test. Record the cleanup once and return to the authorized teaching task.
