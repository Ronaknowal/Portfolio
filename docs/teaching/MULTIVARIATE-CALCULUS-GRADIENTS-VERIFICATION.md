# Multivariate Calculus & Gradients — verification

Author review completed 10 September 2026. Lesson/model/lab source freeze: **09:53:34 UTC**; this includes the final standalone-program setup paragraph and AST-preserving formatting. The final small keyboard/setup review ran after that freeze. Root integration/build review remains a separate responsibility; this document does not equate publication with user acceptance.

## What changed and what was preserved

The former page introduced the gradient and one quadratic update in six short sections, with one runnable program, pseudocode for autograd and limited practice. The new lesson builds partial slices → local prediction → differentiability → directional geometry → chain rule/level constraints → curvature → finite updates → applications/computation → independent practice. Each step defines the required objects before using them and connects symbolic formulas to a complete numerical case.

Retained and deepened all useful original coverage: scalar losses with many inputs, signs and local meaning of gradient entries, the bowl x²+2y², the original gradient (6,−8) and update (2.4,−1.2), partial/directional/Hessian concepts, automatic differentiation, zero-gradient limitations, poor scaling, vanishing/exploding sensitivities and the two requested learning-rate variations 0.6 and 0.01. The pseudocode was replaced with a complete small forward-mode implementation and a link to the existing complete reverse/matrix-gradient teaching.

The added depth includes an exact first-order remainder, a counterexample with existing partials and directional derivatives but no differentiability, Euclidean/weighted notions of a unit move, input gradient versus graph normal, constrained circle extrema, strict versus semidefinite Hessian tests, boundaries, the exact quadratic iteration interval, physical unit conversions, and a scoped path-integral bridge. The beginner route does not require the deeper metric or integration branch. The title and stable identity are unchanged; **Convex Optimization** remains next. Matrix Calculus is a review connection rather than a required prerequisite, preserving the acyclic dependency graph.

Five investigations have distinct teaching jobs: linked contour/tangent slice; line versus curved approach; motion constrained to a circle; exact versus quadratic curvature slices; and a stepped descent trajectory. Two immediately visible figures compare coordinate slices and distinguish the two-coordinate gradient from the three-coordinate graph normal. There is no lab-count requirement behind this selection.

## Source ownership

- `src/learn/data/topics/multivariate-calculus-gradients.jsx`
- `src/learn/data/multivariate-calculus-models.js`
- `src/learn/data/multivariate-calculus-examples.js`
- `src/learn/components/lesson-labs/MultivariateCalculusLabs.jsx`
- `src/learn/components/lesson-labs/multivariate-calculus-labs.css`
- `src/learn/data/curriculum/blueprints/multivariate-calculus-gradients.js` — registered by the parent
- [Design and claim/resource review](MULTIVARIATE-CALCULUS-GRADIENTS-DESIGN.md)
- [Measure Theory destination note](topic-notes/measure-theory-probability-spaces.md) — open proposal for concrete product-space integration; not implemented by this task

Existing publication mapping, title-derived progress identity and curriculum order were preserved. No shared lesson manifest, blueprint index, ledger or generated navigation was edited by this author. The original source is retained at `scratch/multivariate-authoring/original-lesson.jsx` as a coverage comparison.

## Native numerical checks

Run from the repository root:

```powershell
node scripts/verify-multivariate-calculus.mjs
```

This exports actual browser-model states and current standalone programs, then runs `scripts/verify-multivariate-calculus-native.py` with the existing lesson Python. All checks passed with **Python 3.12.14 / NumPy 2.3.5**. The final model/formatting suite completed at 09:51:28 UTC; an expanded direct native pass at **09:53:51 UTC** additionally checked the actual practice program against generated polynomial derivatives. Results: `scratch/multivariate-verification/results.json`; fixtures/programs are in the same directory.

| Check | Actual scope and independent reference |
| --- | --- |
| 10 standalone programs | Execute every complete block and compare all stdout. Imports and inputs are self-contained. |
| 3,750 local-change states | Five values per input coordinate, 25 angles, six signed steps; independent NumPy quadratic form, gradient, Cauchy bound, exact remainder and selected slice values. |
| 24 approach configurations | Axis, line and parabola with eight coefficients; seven decreasing scales per path; exact rational arithmetic for coordinates, values and path limits. |
| 123 circle states | Full angular sweep plus exact extreme candidates; direct scalar parameter-function central derivatives, unit/tangent orthogonality, tangent projection and Cauchy bounds. |
| 150 curvature states | Six functions and 25 directions; derivatives generated from monomial coefficient maps, independent NumPy symmetric eigenvalues, exact polynomial slice values and quadratic forms. |
| 988 descent states | 76 rates including 0, 0.5 and unstable settings, and 13 iteration counts; NumPy matrix powers and spectral-radius condition, rather than replaying the model's coordinate loop as the oracle. |
| 100 unseen AD inputs | Extract the actual lesson Dual class through Python AST; test against an independent analytic gradient, including scalar lifting and left/right arithmetic. |
| Six practice groups | New polynomial gradient/finite change/direction, rational path contradiction, semidefinite saddle, changed quadratic rate interval, radius-two constrained extrema and physical units/cancellation. |
| 18 invalid model groups | Nonfinite/out-of-range coordinates and angles, unknown paths/presets, invalid rates and noninteger/out-of-range steps. |
| Strict small-value regression | Exact zero and negative zero format as zero; 10^-12 and −10^-15 remain nonzero scientific notation. |

The proof conditions were also read independently of the computations. Finite samples are not presented as proof that all directions are positive, that a limit exists, or that every optimization problem converges. Exact algebra establishes the counterexample, definiteness implications and quadratic recurrence.

## Independent review and resolved issues

Another agent read the actual lesson, examples, models and geometry. Its independent script is `scratch/review-multivariate-mathematics.mjs`; results are `scratch/multivariate-cross-review/results.json`, recorded at 09:44:22 UTC. The additional cases include a non-diagonal symmetric positive-definite metric, three alternate curved paths for the line-integral identity, graph-normal/tangent orthogonality, circle extrema and quartic sign tests. It found no further material mathematical blocker after the corrections below.

**Tiny-coordinate display defect:** the initial generic number formatter rounded every magnitude below 10^-11 to zero. In the parabola table that displayed x=10^-6, **y=0**, g=0.5 although the actual y was 10^-12. This contradicted g(x,0)=0. The final formatter reserves zero for an actual numerical zero, preserving genuinely nonzero inputs in scientific notation. Trigonometric cancellation residues remain visible and are explained as computed floating-point values. Both native and browser regressions now require the last parabola row to display **y=1.000e-12 and g=0.5**. The independent review JSON intentionally preserves the initial failure; the final author results establish its resolution.

The review also prompted explicit **symmetric** positive definiteness for the square-root metric argument and a **nonempty** closed bounded set for the extreme-value statement. Mobile review found several wide display equations; they were split into justified derivation lines, preserving their formulas and smoothness conditions. The 320px quotient line was shortened by stating its nonzero-parameter condition in adjacent prose.

Initial browser harness issues were selectors for nested select labels and an assumption that the shared code renderer used `pre` tags. These were corrected to native label-prefix selectors and complete rendered code/output text comparison; no lesson behavior was changed to make a failing harness expectation pass. A Windows-default text-decoding failure in the native harness was corrected to explicit UTF-8 before numerical evaluation.

## Browser and actual visual review

```powershell
node scripts/review-multivariate-calculus.cjs
node scripts/review-multivariate-calculus-reading.cjs
node scripts/review-multivariate-calculus-keyboard.cjs
```

Used the existing Playwright package and headless Microsoft Edge against the shared development server on port 5173. No browser library was added to production.

- **Full behavior, 09:51:45 UTC, 1440px and 390px:** per width, 72 local-change states, 15 path states, 12 circle states, 42 curvature states, 65 descent states, eight invalid-input groups retaining active state, all reset/back/end behavior, all nine anchors, six practice blocks, all ten complete code/output blocks, and eleven actual reference links. The strict tiny-coordinate regression passed. Forty-two eligible controls/table regions/practice summaries accepted focus; range arrows and disclosure Enter/Space operated correctly. No page errors, math errors, display-equation overflow or page overflow. Record: `scratch/multivariate-browser/results.json`.
- **Ordinary reading, 09:51:44 UTC, 1440px, 390px and 320px:** captured all nine section starts without operating labs; separately captured both inline figures and sources; all ten display equations fit, every SVG text box stayed inside its figure, and no page/math errors or page overflow. Record: `scratch/multivariate-browser/reading-results.json`.
- **Final keyboard/setup, 09:54:56 UTC, 1440px and 390px:** after the final prose-only setup addition, Enter activates the next update, Space activates the previous update, native select arrow keys change the path, and a coordinate form submits by keyboard. Program setup and its NumPy link are present and readable. Record: `scratch/multivariate-browser/keyboard-results.json`.

The final source addition after the full behavior/reading runs is only the explicit optional-program execution/setup paragraph. The final keyboard/setup run verifies it in the browser. No model, interaction, equation or geometry changed after the full behavior/reading runs. The final Babel formatting comparison preserves normalized ASTs, JSX text and template literal values: `scratch/multivariate-verification/formatting-results.json`, **09:53:34 UTC**.

Screenshots were actually opened and inspected, not merely saved. Opened evidence includes:

- `inline-1-320.png`: both coordinate slices, distinguishable exact/tangent curves, independently labeled vertical scales.
- `inline-2-390.png`: input-plane gradient, oblique graph-normal/tangent-plane patch, readable labels and explicit projection limitation.
- `local-tangent-390.png`: zero first-order tangent rate versus positive exact remainder; arrows and linked slice correspond to the same state.
- `parabolic-path-390.png`: the final nonzero 10^-12 coordinate and constant 0.5 output are readable together.
- `circle-maximum-390.png`: tangent direction, fixed gradient, actual objective curve and tiny computed residual clearly separated.
- `flat-saddle-1440.png`: identical zero Hessian versus nonzero quartic slice; quadratic and exact curves remain distinct.
- `diverging-descent-390.png`: adaptive equal-axis trajectory and readable step history, including the small x and growing y coordinates.
- `reading-3-390.png`, `reading-4-320.png`, `reading-5-390.png`, `reading-6-1440.png`, `reading-8-390.png`, `reading-9-1440.png`: differentiability, directional definition, chain rule, Taylor explanation, physical units and independent practice in ordinary reading flow.
- `program-setup-390.png` and `sources-390.png`: complete optional execution instructions and annotated alternate learning routes.

All are under `scratch/multivariate-browser/`. The fixed site header is hidden only for isolated tall figure/lab/source captures, then restored; ordinary-reading captures retain it. Code and wide data tables use local horizontal scrolling instead of forcing the entire page wider.

## Research, readiness and limits

The [design record](MULTIVARIATE-CALCULUS-GRADIENTS-DESIGN.md) records primary textbook/documentation locators and actual video/transcript review bounds. The selected MIT recordings provide worked calculations and a longer geometry route; full playback was not claimed. The source selection supplements original derivations, exact examples and practice rather than copying an external chapter structure.

This is author verification plus bounded independent mathematical review, not observed novice testing or user acceptance. Browser plots are bounded calculated teaching examples, not symbolic algebra or empirical benchmarks. The AD class is a complete illustration for its stated operations, not a general tensor framework. The weighted-metric and integration connections are scoped bridges. The open Measure Theory note preserves a genuine coverage discovery for later assessment; no claim is made that a finite lesson covers all multivariable calculus.

Source is ready for the parent's curriculum/build/loading integration and final source-hash review. No deployment, external publication, package installation or unrelated lesson rewrite was performed by this author.
