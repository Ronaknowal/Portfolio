# Non-Convex Optimization Landscape — implementation and verification

Author review completed 10 September 2026. Stable ID/title retained; the existing publication mapping was preserved. This is scoped author evidence, not user approval or a claim that the entire active DSA/maths goal is complete. Root owns independent review and integrated publication/build checks.

## What changed and why

The original short lesson introduced minima/saddles, SGD noise, symmetries and diagnostics. Those useful ideas remain, with missing assumptions and mechanisms supplied. The new self-contained route distinguishes parameters from model inputs, local from global comparisons, singular Hessians from positive-definite certificates, exact stable-subspace recurrences from generic escape claims, coordinate-dependent curvature from predictions, selected paths/slices from complete geometry, and training fit from declared held-out behavior. It retains the preceding Second-Order bridge and the actual next Constrained & Multi-Objective topic.

Nine sections have four concept-specific investigations, three inline calculated/correspondence figures, ten complete executable NumPy programs, seven independent changed-input practice groups with hints/explained acceptance checks, and five annotated resource links including the direct MIT recording and its written transcript. Counts describe this lesson, not an authoring quota. Two deeper branches supply exact sign-distribution moments and stationary coordinate-transform rules without hiding the core reasoning.

The title still matches the scope; no topic was removed, renamed or reordered. [Design and claim ledger](NONCONVEX-LANDSCAPE-DESIGN.md) records preserved coverage, representations and review boundaries. [Regularization destination note](topic-notes/regularization-l1-l2-elastic-net-dropout.md) carries the exact data-loss symmetry versus whole penalized objective distinction to its correct future owner; that lesson was not modified.

## Executed numerical checks

Run `node scripts/verify-nonconvex-landscape.mjs` (optionally set `LESSON_PYTHON` to an existing Python/NumPy/SciPy runtime). It executes every displayed code string and compares full stdout, then invokes `scripts/verify-nonconvex-landscape-native.py` on exported visual states. Actual environment: Python 3.12.14, NumPy 2.3.5; SciPy installed in the isolated lesson runtime.

Final [results](../../scratch/nonconvex-landscape-verification/results.json) at **12:51:07 UTC** passed:

- All ten complete displayed programs and outputs. Code imports/setup are present; no notebook state or earlier example must be run first.
- 1,001 well parameter configurations / 61,061 frames. JavaScript bracketed roots versus NumPy polynomial roots, and two separate SciPy bounded minimum searches per tilt for the global-value comparison. Actual update/value checks use independently evaluated polynomial objects.
- 900 stationary geometry/direction/radius configurations against independent polynomial coefficient evaluation and NumPy eigenvalues. Explicit neutral 45°,135°,225°,315° regression for both quadratic and quartic saddles.
- 400 finite saddle-noise runs / 8,711 frames against the weighted closed-form convolution, independent of the gradient recurrence. Stopped runs check the proposed outside-window point and ensure no clipping.
- 350 rescaling/perturbation states against NumPy Hessians, direct perturbed objective evaluation and the fourth-order tangent formula; 160 changed arbitrary factor points checked by central finite differences of actual displayed functions.
- 101 mode-path fractions, 42 input/predictor evaluations, 80 arbitrary ReLU networks (1/2/5/9 hidden units with positive unequal scales and permutations), 2,304 changed exact noise-sign sequences, 18 inactive/active diagnostic runs and seven changed practice checks, including an independent scalar optimization of the alternate path barrier.
- 22 invalid model input groups reject unknown/prototype names, nonfinite or out-of-range values and invalid step counts. Tiny genuinely nonzero values are preserved in scientific notation.

Evidence: [exported cases](../../scratch/nonconvex-landscape-verification/model-cases.json), [native code inputs](../../scratch/nonconvex-landscape-verification/examples.json), [full stdout](../../scratch/nonconvex-landscape-verification/stdout.json). These establish the displayed bounded models and selected mathematical identities, not arbitrary-range floating-point stability or global results for deep networks.

## Actual browser and visual review

`scripts/review-nonconvex-landscape.cjs` used actual headless Edge at **1440 and 390 pixels** against the shared Vite route. Final [browser results](../../scratch/nonconvex-landscape-browser/results.json), **12:51:31 UTC**, passed at both widths:

- 120 well states, 90 stationary states, 108 noise states and 40 symmetry states per width; actual controls update the expected readouts and plotted model state.
- Restart/previous/next/final transitions; reset-on-input behavior; disabled boundaries; zero rate, zero amplitude, exact stable axis, signed perturbations, both noise directions and explicit local-window termination.
- All nine in-lesson anchors resolve and scroll to their intended sections, all ten full code/stdout strings render (including the disclosed deeper example), all seven practice groups are present and expand, and all five resource links are visible.
- Zero runtime errors, KaTeX errors, page overflow or out-of-viewport SVG text in the reviewed states. Slider/buttons and practice are actually keyboard-operated.

`scripts/review-nonconvex-landscape-reading.cjs` independently captured the ordinary reading flow at **1440, 390 and 320 pixels**. Final [reading/keyboard results](../../scratch/nonconvex-landscape-browser/reading-results.json), **12:50:47 UTC**, passed: all nine section openings, three inline figures, seven expanded display-math blocks; no math/SVG clipping; seventeen initially enabled lab controls visited in DOM order with visible focus at each width; keyboard slider/selection/practice actions; one narrow table scrolled by keyboard at 320px. Other disabled controls are exercised by the interaction suite. This is keyboard and render evidence, not a screen-reader user study.

Actual opened screenshots inspected by the author include all nine 390px ordinary section openings; the 320px stationary/Taylor opening; desktop path-reading flow; all four labs at 390px; the desktop quartic geometry and high-scale tangent views; all three inline figures including the 320px interpolation; and the actual resource section. Representative evidence:

- [Competing wells at 390px](../../scratch/nonconvex-landscape-browser/wells-local-390.png)
- [Quartic saddle map and exact slice](../../scratch/nonconvex-landscape-browser/stationary-quartic-390.png)
- [Actual noise trace with declared window exit](../../scratch/nonconvex-landscape-browser/noise-window-exit-390.png)
- [Equivalent predictor, different curvature](../../scratch/nonconvex-landscape-browser/symmetry-normal-390.png)
- [Straight and curved path in ordinary desktop reading](../../scratch/nonconvex-landscape-browser/reading-6-1440.png)
- [Inline same-input network correspondence](../../scratch/nonconvex-landscape-browser/inline-1-390.png)
- [Input/output interpolation at 320px](../../scratch/nonconvex-landscape-browser/inline-3-320.png)
- [Actual references](../../scratch/nonconvex-landscape-browser/sources-390.png)

Review fixed two long equations at 320px and adaptive noise-axis labels that could clip. The equations now wrap by mathematical steps; the coordinate plot prints its adaptive range inside the panel. Familiar diagonal directions use exactly shared components to prevent trigonometric roundoff from falsely tilting a mathematically neutral slice. General rounded readout provenance is explicit. Earlier browser harness failures involving JSON's −0 serialization and an implicit output status role were test-selector/serialization issues, not learner defects; the corrected final suites pass.

## Source quality, research and remaining boundaries

Model/render/body formatting preserved the normalized AST, JSX text and template values: [formatting evidence](../../scratch/nonconvex-landscape-verification/formatting-results.json), **12:50:10 UTC**, followed by final native/browser runs. Example storage was subsequently made readable as multiline `String.raw` programs; [exact content-equivalence evidence](../../scratch/nonconvex-landscape-verification/example-formatting-results.json), **12:51:35 UTC**, confirms every title, code string and expected output is identical. This formatting-only change does not invalidate the preceding execution/render checks.

[Final six source fingerprints](../../scratch/nonconvex-landscape-verification/final-source-hashes.json) record the source freeze. The parent should use these for the integrated build snapshot. No shared registry, ledger, global CSS or navigation file was edited by this scoped author.

Primary Dinh/Jin paper sections and MIT substantive transcript passages were actually inspected; precise URLs and locations are in the design record. The direct YouTube ID was confirmed in the official MIT page. No full recording playback or empirical deep-network training experiment was performed. All plots are disclosed toy calculations, sampled functions or explicit recurrences, not invented measured loss surfaces or generalization rankings. A paused SGD example does not certify a global minimum; a sharpness number does not certify generalization. User acceptance and an observed beginner study remain pending.

## Paragraph semantics repair — 10 September 2026

The parent identified that `Prose` renders a paragraph, so wrapping literal `<p>` elements inside it produced invalid nesting and bypassed the intended paragraph typography. Replaced 49 such groups with fragments containing 65 individually styled `Prose` paragraphs. No explanations, inline semantics, equations, models or examples changed. The source transformation asserts full normalized-AST equality against exactly that tag-only change; [source evidence](../../scratch/optimization-paragraph-repair/source-results.json). Body SHA-256 is now `a5808ff066db96746c4976775a377d9bdfde1f976cd8ec65d7d7dd385a3f75ed`; the six-file freeze record was refreshed.

`node scripts/review-optimization-paragraph-repair.cjs` passed actual browser checks at1440/390/320:119 paragraphs,81 with the intended Prose styling, no nested block elements, text overflow, math errors, page exceptions or console errors/warnings. All nine ordinary section openings were recaptured and all disclosures opened for DOM checks. The author actually opened the first section and expanded practice screenshots at all three widths; the paragraphs now have consistent readable size, line height and separation. [Browser results](../../scratch/optimization-paragraph-repair/browser-results.json); [mobile opening](../../scratch/optimization-paragraph-repair/nonconvex-reading-1-390.png); [mobile expanded practice](../../scratch/optimization-paragraph-repair/nonconvex-practice-open-390.png). Initial harness runs encountered an intentionally blocked Vite socket and a transient network-resource failure; the final run allowed the development socket and passed without console errors. No mathematical or broad interaction rerun was needed for this tag-only correction.
