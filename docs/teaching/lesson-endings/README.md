# Recurring lesson endings

28 September 2026. Completed presentation follow-up to the shared lesson opening.
This is evidence, not a new content-writing or scientific-review queue.

Practice now has a consistent section rhythm, clearly separated exercises and
keyboard-operated native hint/solution disclosures. Existing always-visible
answers remain visible. DSA's local exercises, staged LeetCode work, optional
extensions and readiness criteria remain distinct. In-body checkpoints and
playable investigations retain their existing behavior.

The shared `Sources` component gives further learning and technical references
separate headings and section links. Existing resource annotations, paired
links, versions, provenance and viewing limits are retained. Authored mixed
resource lists keep their context, with the same readable spacing and amber
links. Readiness and continuation remain separate from practice where the
source already identifies that boundary. No exercises or resources were merged,
removed, invented, or recertified as newly researched.

## Current implementation

- [Audit and independent review](audit.md) records the original formats and
  reviewed exceptions. [Layout plan](layout-plan.json) records exact section
  purposes rather than runtime guesses based on titles or URLs.
- [Conservation evidence](conservation-checks.json) compares all 234 published
  lesson ASTs after removing only explicit transparent layout wrappers. All
  original prose, formulas, programs, expressions, links and ordering match.
  Publication mappings remain identical.
- [Browser/build evidence](browser-checks.json) records representative rendering,
  section links, responsive behavior and native disclosure checks. This does
  not claim new scientific or numerical execution across the curriculum.
- [Presentation receipt](presentation-review.json) binds the final files and
  evidence. It preserves the 81 previously current ML/DL checkpoints, their
  original scientific receipts and the 96 other historical ledger rows.

Run `node scripts/verify-lesson-endings.mjs`,
`node scripts/check-lesson-opening-authoring.mjs` and the usual applicable
delivery/build checks. Earlier snapshot verifiers remain historical evidence;
do not overwrite their receipts to make old source hashes match this layout.

## Future authoring

Use the current teaching, code and topic-design standards. Keep practice,
readiness/next study, alternative explanations, technical references and
provenance recognizable; retain topic-specific content and learning order.
Use `.lesson-ending` with `data-lesson-ending` and the appropriate purpose class.
Use `.lesson-exercise` for an independent prompt plus its associated answer
controls, without adding an extra number or changing the authored numbering.

The prepared renderer accepts exact-title `endings` entries with `level`,
`title` and `kind`. Existing `preserveOpeningFrom` also preserves this ending
metadata and groups practice exercises. `resourceList: true` explicitly marks
a known resource list inside a combined continuation section (Ring Attention);
URLs do not determine purpose. Missing/ambiguous headings and unsupported
custom ranges fail instead of silently dropping layout. Preserve custom ranges
explicitly in bespoke generators. Do not regenerate an older manuscript over
later reviewed teaching merely to change presentation.

No runtime source parser, full-curriculum scan, new dependency, or eager lesson
import is introduced. Styles are shared by the reader; lesson data stays lazy.
Temporary source copies used to prove conservation are removed after retaining
compact hashes, exact plans and review evidence. Pre-existing workspace work
and all authored content are preserved.
