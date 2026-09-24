# Numerical Methods — implementation and author verification

Stable ID `numerical-methods-finite-differences-quadrature-root-finding`, Mathematics Foundations position 38. Full author implementation is complete. The exact freeze and source fingerprints are in [durable author evidence](evidence/numerical-methods-author-review.json). Independent reviewer sign-off, integrated production/loading/build checks and user acceptance remain separate; none is inferred from publication or the checks below.

## What changed and what was preserved

The previous lesson introduced useful concepts in seven short sections but offered no topic-specific illustrations or investigations, only one complete program and two unworked practice suggestions. The rewrite develops the mechanisms and their boundaries: bracket invariants and midpoint error, tangent iteration and safeguards, derivative stencils and competing errors, interpolant-based quadrature, adaptive error budgets and blind spots, Gaussian moment matching, and a complete integral-to-target workflow that preserves unresolved signs.

The original complete Python program and its exact output are preserved. The [original archive](evidence/numerical-methods-original-content.json) also retains the entire pre-rewrite body and its source hash. Its initial execution and final conservation checks are distinct evidence. No title, stable ID, route order or other lesson body was changed.

The final lesson has thirteen complete executed Python programs: the preserved original; bracket/status handling; Newton cycle/safeguard; actual finite-difference sweep; exact stencil/directional checks; sampled/callable quadrature; adaptive Simpson; exact blind-polynomial integration and singularity bridge; Gaussian moments; candidate calibration; complete nested calibration; linear/ODE/library transfer; and the independently changed calibration solution. Each question is visibly rendered before its own code, with actual stdout and interpretation. New Python blocks were conventionally formatted with Black, preserving their normalized Python AST; the original was excluded to conserve it exactly.

Ten independent practice groups include separate optional hints and complete explained solutions. They change brackets, root multiplicity, stencils, error constants, sample spacing, blind-spot scale, Gaussian degree, volume/time units, a runnable calibration input and physical time steps. These are teaching and transfer tasks, not a claim that a finite collection covers every numerical problem.

## Representation review

| Representation | Learning purpose and actual evidence |
| --- | --- |
| Initial rate/volume figure | Directly connects area under a synthetic rate to accumulated litres and a threshold time. Both panels are calculated from the same primitive; phone layout retains both. The shaded three-minute area is not measurement data. |
| Bisection investigation | Function signs, retained interval strips, halving count and midpoint radius stay synchronized. Includes touching-root failure, reverse stepping, reset and bounded-trace status. Root membership/radius checked independently against SciPy and exact width arithmetic. |
| Newton investigation | Tangent geometry, intercept, proposal decision and visited values expose the exact 0↔1 cycle, ordinary convergence, multiple-root behavior and a bracket safeguard. The final SVG clips a straight line at its boundary rather than flattening its values; a geometric assertion checks its slope. |
| Derivative investigation | Moves actual stencil points, displays their rounded values and compares an analytic local line/quadratic with the computed one. Its error plot contains actual JavaScript values on a declared log scale, fitted to the entire selected sweep; changing only h keeps that scale fixed. Computed zero and unresolvable coordinates have distinct representations. |
| Quadrature investigation | Shows the actual curve, the line/parabolic interpolant, its shaded area, sample weights and analytic reference. Polynomial exactness checked with Fraction arithmetic; changed sine input defeats an overgeneralized exactness claim. |
| Adaptive partition | Reveals actual samples, terminal interval budgets and work-limit states. Five zero observations of a positive-integral polynomial expose an estimator blind spot. Extra drawing points never feed the algorithm. Final tiny vertical values are explicitly scaled by 10⁵ while area/error values retain the original scale. |
| Calibration investigation | Maps an integral discrepancy and conditional discretization bound into time error using a stated positive lower slope. The reader can produce resolved and unresolved sign states. The separate complete search program implements refinement and explicit failure instead of leaving the composition as prose only. |

The manuscript explains symbols and units before use and keeps the essential derivations visible. Eight display equations were inspected in ordinary reading. Five originally exceeded the 320-pixel content width; they now use shorter equivalent lines or locally defined quantities. No global math or CSS change was required. Long code and precise sample tables retain horizontal scrolling, rather than discard digits or rewrite the preserved program.

## Computational evidence actually run

Commands, from the repository root:

```text
scratch/lesson-tools/Scripts/python.exe -X utf8 -I scripts/generate-numerical-methods-examples.py
node scripts/format-numerical-methods-source.cjs
scratch/lesson-tools/Scripts/python.exe -X utf8 -I scripts/verify-numerical-methods-native.py
```

Final native run: 10 September 2026, 21:54:05 UTC. Python 3.12.14, NumPy 2.3.5 and SciPy 1.18.1. The native script invokes the actual JavaScript model suite and then checks its exported fixtures against complementary references. It executes every displayed Python block, rather than merely evaluating a counterpart implementation.

- 18 bracket cases, including changed square-root targets/tolerances, plus endpoint, missing-sign, nonfinite, large-sign and representability guards.
- 30 Newton configurations. Fraction-based polynomial values/slopes and independently formed tangent intercepts complement the model tests; exact cycling and safeguard position are checked.
- 180 composite quadrature configurations checked against exact Fraction sums and polynomial moments. Maximum arithmetic discrepancy was approximately 2.84×10⁻¹⁴.
- 48 adaptive configurations with independent SciPy integrals, exact terminal-budget/interval accounting and actual Python-helper comparison. The blind polynomial intentionally has zero estimated error and positive actual error. Fixture-specific accepted-error checks are explicitly not an estimator theorem.
- 64 derivative configurations checked with independent trigonometric identities and a stated amplification allowance for evaluation rounding; nine irregular-sample linear integrals checked against analytic primitives.
- 96 candidate calibration configurations checked against independent quadrature and root solves; 48 additional complete nested-search cases check retained brackets, conditional time bounds and unresolved-budget results.
- All thirteen actual stdout records, the original code/output bytes, changed hand-practice fractions and six exact oscillator identity states pass.

The first development run exposed an error in the independent one-sided-stencil oracle, not the implementation: its trigonometric rearrangement was missing a contribution. Re-deriving it as `[4 sin(h/2) cos(x+h/2) − sin(h) cos(x+h)]/h` resolved the mismatch. No assertion was loosened to hide that issue. The final suite passes with the corrected independent expression. The formatter checks normalized AST conservation for four owned JavaScript/JSX files and CSS; semantic values and executable example output are separately verified.

Raw replayable results are `scratch/numerical-methods-verification/native-results.json`, `model-fixtures.json` and `format-conservation.json`. The durable author JSON embeds the final result summaries and hashes the scripts. The native checks are independent mathematical/runtime oracles performed by this author, not a substitute for a separate author's source review.

## Actual browser and ordinary-reading evidence

The existing localhost 5173 route was exercised in headless Microsoft Edge with the actual Space Grotesk, JetBrains Mono and KaTeX fonts, at 1440, 390 and 320 pixels. The browser runs used the authorized public-font network access. These are local development checks; no production benchmark is claimed.

```text
node scripts/review-numerical-methods.cjs
node scripts/review-numerical-methods-reading.cjs
node scripts/review-numerical-methods-programs.cjs
```

The full interaction run at 21:51:52 UTC passed 264 states per viewport, including changed methods/data/work limits, actual slider-key operation, focused Enter activation, back/reset, all ten section anchor arrivals, ten practice disclosures, no page/runtime errors and no page-wide overflow. The then-present twelve programs matched their exact stored questions/code/output. The final reading run at 21:52:22 UTC verified all eight equations fit each width, all plot labels fit, the straight tangent clipping and scaled five-zero blind-spot geometry, references and ordinary screenshots.

The only subsequent production change added the complete nested-search program and its surrounding explanation. The final targeted program run checks **all thirteen** programs, their exact prompts/code/stdout, the disclosed changed solution, eight fitting equations and page geometry at all three widths. Its timestamp and captures are in `scratch/numerical-methods-browser/final-program-results.json` and the durable evidence. The earlier 264-state interaction behavior and models were unchanged by this addition; they are not misreported as a new full interaction run after it.

Generated screenshots were actually opened and inspected, including the initial two-panel pump at 320, bracket and Newton states, derivative local geometry/error plot, shaded quadrature at 1440, scaled blind polynomial at 320, calibration uncertainty at 390, the repaired narrow equations, original program/output, changed practice and references. Final opened-file hashes are recorded separately in the author JSON. This distinguishes image inspection from screenshot generation or bounding-box checks. During development, one targeted script incorrectly called `innerText()` on an SVG; switching that read-only assertion to `textContent()` fixed the harness. No production behavior was changed to satisfy that API mistake.

## Sources, scope and handoff

The [design's research ledger](NUMERICAL-METHODS-LESSON-DESIGN.md) records the exact primary pages and sections inspected: Driscoll/Braun's numerical-computation text, NIST DLMF differentiation/quadrature, relevant SciPy API contracts and the official MIT Newton video page with its first three pages of notes. The learner-facing source list supplies usefulness annotations. Full video watching and a timestamp-specific endorsement are not claimed. Documentation displayed SciPy 1.18.0 during research; actual executable evidence records 1.18.1 separately.

The [incoming Dynamics note](topic-notes/numerical-methods-finite-differences-quadrature-root-finding.md) is adapted and resolved with the section 8 bridge and exact/native evidence. The [Backpropagation destination note](topic-notes/backpropagation-automatic-differentiation.md) preserves the discovered fixed-step/universal-correctness overclaim and a tested counterexample for its future author. No other topic was rewritten during this scoped work.

Mathematical discretization bounds assume the stated smoothness and function-evaluation model. These ordinary floating-point programs are not certified interval-arithmetic implementations. A library's success flag, a finite set of passing tests, a sampled plot or a zero adaptive discrepancy is not a universal accuracy guarantee. Advanced ODE/PDE solvers, general conditioning, AD engines and rigorous rounding remain their appropriate deeper owners. Functional Analysis & RKHS remains the actual next module topic.
