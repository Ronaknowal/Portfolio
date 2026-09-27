# Shared lesson opening and section navigation

28 September 2026. This is a completed presentation follow-up, not a new lesson
writing or scientific review queue. All 42 Deep Learning Fundamentals &
Architectures topics retain their completed implementation checkpoints.

The subsequent [lesson-ending review](../lesson-endings/README.md) now owns the
current shared-source identity. The receipts and snapshot comparison below
remain evidence for this earlier opening change; do not run the historical
snapshot check to refresh today's hashes. Use `node scripts/verify-lesson-endings.mjs`
and `node scripts/check-lesson-opening-authoring.mjs` for the current presentation
follow-up, together with the central delivery preflight.

The reader now owns one Learning compass directly below the title and metadata.
It gathers existing topic-specific summaries, reading routes, prerequisites and
exploration instructions. The complete main-section list comes from the active
lesson's rendered H2s. There is one displayed number, consistent neutral/amber
styling, two columns on wider screens and one on narrow screens. Unnumbered
sections remain unnumbered; authored numbers and old fragment aliases survive.
Native links update history, focus their heading and clear the fixed header.

The shared change applies to all 234 published lessons. Explicit placement
annotations move 110 existing paragraphs in 73 topics. Three custom partial
contents lists are replaced by the shared full list. Scientific prose, formulas,
programs, labs and practice remain unchanged. Local download navigation and
disclosure headings remain with the material they support. Main sections must
mount with the lesson; the reader does not rescan on every lab interaction.

## Evidence and limits

- [Initial audit](audit.md) and [source inventory](opening-inventory.json) identify
  the previous inconsistent layouts.
- [Conservation checks](conservation-checks.json) compare all 234 lesson ASTs,
  allowing only explicit opening placement and the three identified wrappers.
  `node scripts/verify-lesson-navigation.mjs` also checks section labels,
  existing IDs, collisions, lab exclusions and repeat collection.
- [Independent source review](implementation-review.md) covers the common reader
  and authoring contract. It does not claim browser or scientific execution.
- [Browser/build checks](browser-checks.json) record all 42 DL openings plus 15
  older lesson formats, 639 section destinations, representative 320px layouts,
  keyboard focus, direct fragments, history and in-app topic changes. These are
  navigation checks, not fresh execution of every lesson experiment.
- [Presentation review](presentation-review.json) binds the final source and
  preserves exact before/after checkpoint changes. The central ledger retains
  all 177 recorded completions, with the 81 previously current ML/DL checkpoints
  updated for this reviewed delta and the 96 historical rows left unchanged.

The earlier DL and intuition review receipts remain evidence for their original
snapshots and unchanged scientific work. Their batch conservation scripts should
not be used to overwrite this later presentation checkpoint. Use the current
ledger, this review and the original evidence together. User acceptance remains
separate from implementation review.

## Future authoring

Follow the current teaching and code standards. `LessonIntro` registers existing
guidance with the shared opening; `Prose opening="summary|route|prerequisites|exploration"`
places additional guidance explicitly. Do not create custom partial contents
lists or rely on `hasIntegratedGuide` to control placement. Write real H2 titles
that describe the lesson's own learning sequence; the body remains topic-specific.

The prepared renderer supports explicit `opening` mappings and
`preserveOpeningFrom`. All 38 callers preserve existing annotations by exact
paragraph matching; Active Learning's bespoke generator has an explicit route
mapping. Missing, ambiguous or conflicting mappings fail instead of silently
dropping guidance. `node scripts/check-lesson-opening-authoring.mjs` passes 12
fixtures and restores all 110 current annotations in memory. Historical lesson
generators were not run, and their older manuscripts were not substituted for
the current reviewed content.

The initial 16 MB temporary source snapshot was reduced to original hashes,
semantic fingerprints and compact checkpoint deltas after conservation passed.
No authored material or pre-existing workspace files were removed.
