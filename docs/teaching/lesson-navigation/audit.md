# Lesson opening and navigation audit

27 September 2026. Pre-change inventory for the shared navigation work. Scope is the 234 current publication mappings, with all 42 Deep Learning lessons inspected for opening/navigation structure. This is a Babel JSX AST source audit, not a new teaching, scientific or completion review. Existing source, labs, publication mappings and delivery evidence remain authoritative. No runtime or ledger files were changed by this audit.

The compact [source inventory](opening-inventory.json) records each stable topic ID, source path, shared-intro use, first element, heading count and identified opening guidance. Dynamic JSX is explicitly marked; source heading counts are not asserted to equal rendered heading counts.

## Baseline

| Opening pattern | Published topics | Deep Learning |
| --- | ---: | ---: |
| Shared `LessonIntro` | 163 | 29 |
| Custom in-lesson section navigation | 3 | 3 |
| Integrated guide flag, but no opening section list: Gaussian Processes | 1 | 0 |
| No shared intro or custom section list | 67 | 10 |

`TopicContent.jsx` currently inserts `LessonGuide` only when `hasIntegratedGuide` is false. Consequently 70 topics receive that generic compass, including three topics that also render `LessonIntro`: Dynamical Systems, Topology/TDA, and Category Theory. The 67 others are 57 older published lessons and ten DL lessons. A universal opening should be owned by the reader rather than this metadata flag.

`LessonIntro` lives in `src/learn/components/lesson-labs/LessonElements.jsx:3`; `lessons.css:9` lays its `<ol>` out as a wrapping flex row. Authored route labels already contain numbering in 29 literal route arrays (25 DL, four Classical ML), plus Recommender Systems' derived `headings.map` list. Those 30 lists acquire a second visible number from the ordered list. Backpropagation and Transfer Learning also put numbered labels inside custom ordered lists. Residual Connections' custom labels are unnumbered. The source H2s inspected use a single `N. ` prefix, not a second heading-number generator.

## Minimal manual migrations

All paths below are relative to the repository root. Line numbers identify the baseline before migration.

| Topic/source | Baseline variation | Smallest change |
| --- | --- | --- |
| `src/learn/data/topics/backprop.jsx:18–19` | First-pass paragraph follows two content paragraphs; `.backprop-section-route` owns eight links. | Preserve and register the first-pass paragraph as opening guidance. Remove the custom TOC once the reader renders the complete list. |
| `src/learn/data/topics/transfer-learning-fine-tuning-strategies.jsx:16` | First child is `.transfer-downloads` containing “Learning route”, prerequisites and a custom list; another first-pass paragraph appears at line 24. | Move the opening summary/prerequisite guidance into the shared opening and replace only this route block. Preserve later download panels and their contents. Existing `transfer-section-*` anchors must remain valid. |
| `src/learn/data/topics/residual-connections-skip-connections.jsx:11,21,23` | Exploration instructions precede the introduction; first-pass prose follows; `.res-route` combines prerequisites and a custom list. | Register those three guidance fragments in the shared opening; remove only the custom TOC. Preserve section anchors and every mechanism/lab. |
| `src/learn/data/topics/grouped-query-attention-gqa-multi-query-attention-mqa.jsx:9,17` | No `LessonIntro`; exploration and first-pass prose. | Register the two opening guidance paragraphs. Derive all section links from rendered headings. |
| `src/learn/data/topics/multi-head-latent-attention-mla.jsx:8,16` | Same opening form as GQA. | Same shared-guidance migration. |
| `src/learn/data/topics/boltzmann-machines-restricted-boltzmann-machines-rbm.jsx:7,15` | No `LessonIntro`; exploration and first-pass prose. | Same shared-guidance migration. |
| `src/learn/data/topics/ring-attention-sequence-parallelism.jsx:12,20` | First paragraph describes how to explore without the standard prefix; line 20 combines prerequisites with “On a first pass”. | Preserve both complete paragraphs when registering guidance. Do not rely only on a `First pass:` string match. |
| `src/learn/data/topics/advanced-optimizers-lion-sophia-prodigy-schedule-free.jsx:9,17` | No `LessonIntro`; exploration and first-pass prose. | Same shared-guidance migration. |
| `src/learn/data/topics/neural-ode-continuous-depth-models.jsx:9,19` | No `LessonIntro`; uses “First pass.”. | Same shared-guidance migration, accepting the period. |
| `src/learn/data/topics/titans-multi-memory-architecture.jsx:9,15,17` | Exploration paragraph, “First-pass route.” and a separate prerequisite paragraph. | Register all relevant guidance; hyphenated route label requires explicit handling. |
| `src/learn/data/topics/hybrid-ssm-transformer-architectures-jamba.jsx:10,20` | Exploration paragraph and “Core route:”. | Register these two fragments rather than treating the core route as scientific prose. |
| `src/learn/data/topics/mini-batches-training-loops-gradient-accumulation.jsx:9,15` | No `LessonIntro`; exploration and first-pass prose. | Same shared-guidance migration. |
| `src/learn/data/topics/neural-training-diagnostics-reproducible-experiments.jsx:9,15` | Exploration paragraph and “On a first pass”. | Same shared-guidance migration. |
| `src/learn/data/topics/gaussian-processes-gp.jsx:23,25` | `hasIntegratedGuide: true`, no opening TOC; first-pass and prerequisite paragraphs. | Register guidance and use the universal reader TOC. Its `.lesson-intro` aside around line 345 is offline-download guidance: retain it in place. |

The other 29 DL lessons already use `LessonIntro`, but many have separate first-pass/exploration paragraphs outside it. The inventory records the candidate paragraphs without rewriting their inner JSX. Explicit registration is preferable to hiding or moving arbitrary DOM paragraphs by keyword.

## Nested/helper and heading cases

- `src/learn/data/topics/active-learning.jsx:14` defines a local `ActiveLearningRoute()` that renders `LessonIntro`; it mounts after five prose paragraphs at line 32. A shared intro portal will find it, whereas a “first child only” migration will not. Its core-route prose at line 31 also belongs with opening guidance.
- `src/learn/data/topics/end-to-end-supervised-learning-error-analysis.jsx` wraps the body in nested divs. Its `LessonIntro` includes dynamic example values and prerequisite links; preserve that JSX intact. Do not infer absence of an opening or H2s from the outermost child.
- `src/learn/data/topics/numerical-pdes-grids-finite-elements-stability.jsx:14` defines `Section({ id, title, children })`, rendering a section wrapper plus `<H2>{title}</H2>`. Static source contains one H2 template for multiple rendered sections. Preserve the explicit wrapper IDs.
- Twenty-seven `LessonIntro` route arrays are derived from `headings.map(...)`. Most remove the numeric prefix; Recommender Systems retains it. Rendering a single normalized list from actual headings removes this inconsistent author responsibility.
- `src/learn/components/content/Headings.jsx` maps older labels such as “1. Why it exists” to a learner-facing label before deriving the current fragment. Any new display normalization should preserve that existing fragment and any wrapper anchors. Do not renumber headings from their position: some lessons intentionally mix unnumbered introductions/next steps with numbered sections.
- No duplicate nonempty H2 fragment was found by static evaluation of the topic sources. All H2 children were statically resolvable strings except the Numerical PDE `Section` template. This does not replace a rendered duplicate-ID check of imported components.
- No additional lesson-section navigation helper was found under `src/learn/components/lesson-labs`. `MinibatchLoopStudy.jsx:13`, `TitansMemoryStudy.jsx:16`, and `TrainingDiagnosticsStudy.jsx:20` contain `<nav>` elements for downloadable programs/data. Keep those local. Many mechanism diagrams have `route` in their names/classes; they are teaching content, not lesson navigation.

## Shared implementation and bounded verification

Use one reader-owned opening immediately below the lesson title/meta, with explicit slots for existing summary, prerequisites and reading/exploration guidance. Build its full contents list from actual article H2s after the selected lesson mounts. Exclude headings inside labs and closed disclosures; keep unnumbered top-level sections and semantic wrapper headings. `LessonIntro` can register/portal its existing summary and prerequisites while dropping its separate partial list. All text and links remain the authored JSX.

Use the neutral/amber reader tokens for the common opening, give jump links clear focus and sufficient touch spacing, and remove native list markers when the displayed heading already owns its number. Custom baseline route containers currently have separate styling in `backprop-labs.css`, `transfer-learning.css` and `residual-connections.css`; replacing only those opening blocks avoids repainting scientific figures or other uses of the same CSS classes.

Verify every published lesson has one opening and a complete, unique TOC, with no duplicate markers. Use targeted desktop/320px checks for a shared-intro DL lesson, Backpropagation, Transfer, Residual, GQA/MLA, Ring/Titans/Jamba, Gaussian Processes, Active Learning, End-to-End Supervised Learning, Numerical PDE, and one older fallback-compass lesson. Check direct fragments, browser Back, a narrow long title, keyboard focus, topic changes and slow imports. Preserve existing scientific checks and completion status; this audit has not rerun them.
