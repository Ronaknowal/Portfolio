# Lesson lab visibility and code access review

28 September 2026. This presentation change makes teaching explanations, worked
values, diagrams and code visible in lesson labs. Optional practice, hints,
solutions, bulk editing controls and saved comparison records keep their native
disclosures. It does not revise numerical engines, datasets or lesson content.

The owned scope was `src/learn/components/lesson-labs/`, excluding
`LessonElements.jsx`, `RunnableExample.jsx` and `MechanismProgram.jsx`. Shared
code controls, the reader, topic sources, ending conventions and delivery ledgers
belong to the integration work.

## Changes and preserved behavior

- Classified all **405** original lab disclosures: **301** teaching disclosures
  became visible sections or shared program views; **104** optional disclosures
  remained. Every retained disclosure's complete source payload matches its
  original source.
- Standardized **36** program viewers. **33** use `RemoteCodeBlock`; the two
  excerpt viewers retain their original slice rules, and `NeuralProgram` retains
  its four dynamic source loaders.
- Routed all **43** original raw code/output blocks through the shared code
  presentation. The local-code check covers **20** blocks; the program review
  covers the other **23**. Nineteen local blocks retain the exact JSX payload
  expression. The remaining `version {state.versions[i]}` uses the equivalent
  template interpolation with the same rendered text.
- Changed **17** investigations from disclosure-triggered loading to persistent
  viewport-triggered loading. Their headings are visible immediately, and their
  content loads as the reader approaches. Existing input state and scientific
  behavior remain intact.
- Passed authored filenames through `PythonExample`, `TerminalExample` and the
  other known program wrappers. Output blocks remain identified as output.
  Existing save/run instructions and code/output expressions are preserved.

## Source checks

The final parser check passed for **388** owned JSX/helper files. Final hashes for
**154** changed or added files still matched when this evidence was retained.
The source audit found no remaining raw `<pre>` code blocks and confirmed that
all **278** SVG subtrees in the changed files were byte-identical to the baseline.

The program review verified **98** program call sites using **95** distinct
sources, including **seven** excerpt uses with unchanged displayed-payload
hashes. All 98 source routes are conserved. Sixty uses have a matching retained
asset/module baseline hash. The other 38 uses have no asset record in that
baseline: the evidence records their unchanged canonical source route and their
current asset hash, without claiming a historical asset-byte comparison.

A separate agent reviewed **169** relevant lab files and **266** converted
non-program sections. It found no actionable issues, confirmed preservation of
**6,911** prose/geometry nodes after the four recorded visibility-copy changes,
and checked **489** model/data hashes without finding a change. That independent
review preceded the final filename-attribute additions. Those additions preserve
the existing code/output expressions, are recorded separately, and are included
in the final parse and source hashes.

## Retained evidence and limits

- [Lab source review](lab-source-review.json): final source identities,
  disclosure decisions, SVG checks, investigation-loading changes, local code
  conservation, the four copy changes and final filename metadata changes.
- [Program review](lab-program-review.json): original and final program-view
  functions, final function hashes, canonical routes, asset hashes and excerpt
  checks.
- [Independent source review](lab-independent-source-review.json): per-file
  content/geometry findings, converted-section checks and model/data conservation.
- [Shared runtime review](shared-review.md): the separate review of shared code
  controls, download registration and the resolution of filename propagation.

These records establish source and presentation conservation. They do not claim
a new execution of the scientific programs or comprehensive browser coverage.
The integration owner reports a passing production build after the final
filename changes; its integration receipt owns the build, browser, copy,
download-byte, retry and navigation results. No runtime files were changed while
retaining this evidence, and no additional build was needed for this
documentation-only step.

The three JSON records above consolidate the useful final lab checks and their
necessary original function/payload evidence. Their internal references are
relative to this directory. They do not depend on the temporary lab inventories,
editing scripts or intermediate migration records; those working copies can be
retired when the integration owner finishes the shared working-directory cleanup.
