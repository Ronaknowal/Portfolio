# Visible teaching and consistent code access

28 September 2026. Completed presentation follow-up across the 234 published
lesson sources and their shared teaching components. This is not a new content
queue or a scientific recertification.

## Learner-facing behavior

- Explanations, derivations, diagrams and implementation programs appear in the
  reading flow. The topic audit revealed 257 teaching disclosures in 88 topics;
  the component audit revealed 301 further teaching/program sections.
- Practice, hints, solutions, optional bulk controls and saved comparisons keep
  their appropriate disclosure boundaries. The 1,267 topic-source practice
  disclosures remain byte-for-byte unchanged; 104 component disclosures remain.
- Shared code blocks provide Copy and Download. Real files retain authored
  filenames; anonymous snippets receive topic-specific filenames. Copy and
  snippet download retain whitespace, Unicode and line endings. An excerpt
  downloads the excerpt, not a different complete program.
- Large programs load automatically near the viewport. Investigations that
  previously needed an opening click also load as the learner approaches them.
  Only the active lesson mounts; code is displayed, never executed by this viewer.
- A shared **Code & supporting files** index follows the article and precedes
  completion/navigation. It includes the current lesson's registered snippets
  and canonical local assets. Practice-only files stay in a separate disclosure;
  recorded output is downloadable beside its code but excluded from the file
  index. Useful contextual source/provenance links remain in the article.

The shared Learning compass and guided-project workflows retain their existing
purpose-specific controls. Three topic `<pre>` blocks are worked mathematical
diagrams, not implementation code; these and inline diagram labels intentionally
do not receive a misleading source-program toolbar.

## Evidence and limits

- [Topic audit](topic-audit.md), [decisions](topic-disclosures.json) and
  [conservation](topic-conservation.json): all 234 authored JSX trees conserved
  modulo the explicitly reviewed teaching-wrapper substitutions.
- [Component review](lab-review.md), [source evidence](lab-source-review.json),
  [program evidence](lab-program-review.json), and
  [independent review](lab-independent-source-review.json): preserved payloads,
  model/data assets, diagrams, program routes and optional controls.
- [Shared runtime review](shared-review.md): three findings resolved, covering
  canonical asset namespaces, practice context and authored filenames.
- [Code control tests](code-control-checks.json): actual component event handlers
  preserve exact text and canonical routes; clipboard errors give useful feedback.
- [Browser observations](browser-checks.json): representative programming, DSA,
  math, Classical ML and DL pages, visible teaching, lazy program loading, file
  index, theme and narrow layout. The native browser download-event adapter timed
  out and its clipboard read returned empty despite the page's successful Copy
  status; those native integration checks are not claimed as passed. Exact
  handler/Blob payloads were verified separately. The rendered remote program
  matched its canonical file byte-for-byte.
- [Integration evidence](integration-checks.json) and the source-bound
  [presentation receipt](presentation-review.json) preserve the original scientific
  evidence, 81 current ML/DL checkpoints and all 96 historical ledger rows.

A narrow-screen check found that the newly visible Gaussian Process table could
force its grid child wider than the screen. The grid child now has `min-width: 0`;
the table retains its local scrolling and all values. No figure coordinates or
scientific outputs changed.

Run `node scripts/check-lesson-code-controls.mjs`,
`node scripts/check-lesson-teaching-visibility.mjs`, and
`node scripts/verify-lesson-code-access.mjs` for this reviewed scope. The earlier
opening/ending receipts remain immutable snapshots; their source-identity checks
predate this follow-up. Do not regenerate them to make changed files pass.

Future authors should follow the updated teaching standard, topic design brief
and learning code standard: visible core teaching, shared controls, exact source
identity, automatic bounded loading and preserved practice boundaries. Changes
to lesson content still need their own appropriate scientific review.
