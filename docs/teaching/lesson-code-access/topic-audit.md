# Visible teaching: authored topic audit

28 September 2026. Scope: all 234 manifest-owned topic JSX sources, based on the integration owner's captured baseline. This audit changes presentation and reuses existing scientific evidence. It does not change the manifest, ending boundaries, progress data, lesson calculations, programs or delivery ledger.

## Disclosure decisions

[The source-bound inventory](topic-disclosures.json) classifies all 1,524 native `details` nodes, recording exact source hashes, ordinal, line, summary and role. These are source constructs, not rendered-instance counts.

- **257 teaching disclosures in 88 topics** became ordinary visible sections. Each retains the original class, ID, child JSX and order; its former summary becomes a heading. Examples include deeper derivations, diagrams, executable examples, environment guidance and optional theory branches.
- **1,267 practice disclosures** remain byte-for-byte unchanged, including body checkpoints, local practice helpers, optional hints, worked solutions, and already-collapsible task prompts.
- K-means and Survival Analysis use local `Deeper` helper definitions with dynamic titles. Converting those definitions reveals all their teaching branches without rewriting their call sites.
- Scientific File Formats and SQL had genuine numbered main-section summaries with existing IDs. These become H2s; other topic branches use H3s. Summary IDs remain on their headings.

Classification was reviewed using source context rather than keywords alone. Git's “Use history as an experiment: find the first failing commit” teaches bisect inside a practice section and is now visible. Numerical PDE's “a nonlinear face needs a wave solution” is a teaching method, not an exercise solution. Conversely, Residual Connections' “Worked extension”, Capsules' “Worked changed-vote calculation”, Training Diagnostics' “Explanation after trying”, Landmark Architectures' “Worked reasoning” and RBM's “Reveal the reasoning” answer explicit changed-case/body practice prompts and remain collapsible. Revealing an outer explanation preserves nested checkpoints and their feedback.

The implementation uses `section.lesson-teaching-section[data-lesson-teaching]` and `lesson-teaching-section__title`. It does not set `open`, add a click handler, or remove content. Shared runtime behavior and styles belong to the integration owner and component reviewer.

## Authoring and conservation

`scripts/lib/lesson-teaching-disclosures.mjs` performs an authoring-time transformation from explicit reviewed disclosure ordinals or exact summary entries. It is not shipped as a runtime content classifier. `prepared-lesson-renderer.mjs` supports `teaching: [{ summary, headingLevel }]` and preserves the destination's visible-teaching annotations through the existing `preserveOpeningFrom` path. Missing/ambiguous summaries, conflicting heading levels and stateful disclosures fail clearly; a dynamic local helper requires an explicit adapter.

`node scripts/check-lesson-teaching-visibility.mjs --record` passes for all 234 topics. [The conservation receipt](topic-conservation.json) proves their complete JSX trees unchanged after reversing only the declared section/title/class/marker substitution. Every retained native practice disclosure also matches its exact baseline bytes, and all publication mappings match. Nine focused assertions cover visible conversion, IDs, nested feedback, idempotence, missing/ambiguous matching, stateful/dynamic fail-safes and prepared-renderer integration. Browser/build evidence is owned by the integration receipt; no numerical execution was repeated here.

The teaching standard, learning code standard and topic design brief now require visible explanations/diagrams/code, consistent Copy/Download controls, canonical source agreement and a shared file index. Large instructional source may load automatically near the viewport rather than waiting for an opening click. Opening preparation guidance and project-specific workflow instructions remain outside this topic-teaching change.

## Existing download-link patterns

[The download-pattern inventory](topic-download-patterns.json) records 338 candidate native anchor source nodes across 68 topics; 56 have an explicit `download` attribute. This is a structural candidate inventory, not a count of resolved files: components and mapped data can produce additional links.

Patterns include literal `/learn-assets/`, `/learn-code/` and `/learn/examples/` paths; template/concatenated asset paths; provenance-object fields; lists mixed into references; and inline contextual links in prose. The inventory records file/line, href expression, surrounding structure and a bounded source excerpt. No authored download list or contextual link was changed in this task. The shared file index can give them a consistent access point while retaining their explanatory context and verified labels.

Practice-only assets need their original boundary. For example, `t-sne-umap-manifold-learning.jsx:552` links `calculated-inputs.json` inside a solution. The integration owner's file index keeps practice files/solutions in a separate collapsible group. Standalone snippet downloads must not be represented as complete executable programs; recorded output downloads are separate from program source.
