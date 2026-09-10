# Independent review: Second-Order Methods

Reviewed 10 September 2026 by a separate lesson author. This is a bounded mathematical and pedagogical cross-review while the implementing author completes browser review. It does not certify frozen source, integration or user acceptance.

## Scope actually read

Read the complete draft, then the installed semantic lesson `src/learn/data/topics/second-order-methods-l-bfgs-k-fac-shampoo-natural-gradient.jsx`, its eight initial complete programs, six model functions, six investigations, two inline figures, stylesheet and `SECOND-ORDER-METHODS-DESIGN.md`. Read all seven independent practice prompts and their explanations. The initial snapshot had no worked changed-data experiment for the seventh prompt. Its subsequent complete program was independently executed and checked below; the final set now has nine programs.

The progression is coherent: exact quadratic geometry establishes why scalar scaling can struggle; a nonlinear double-well separates curvature, direction and step acceptance; secant observations motivate limited-memory inverse action; an explicitly defined Bernoulli model introduces Fisher/KL before the neural-layer factorization; Shampoo is clearly distinguished from Fisher and Hessian estimation. Matrix and gradient prerequisites are stated, while the later probability prerequisite is bridged locally.

The original `100x²+y²`, `(1,1)` Newton example remains, with its solve and warnings about local validity, memory and comparisons. Neither the tiny Fisher fixture nor the fixed Shampoo replay is represented as production training or a measured benchmark. The source describes displayed scales, contour clipping, mathematical expectations and approximation status explicitly. Browser legibility and actual keyboard operation remain the author's separate evidence.

## Findings and disposition

**One concrete display correction is required before final acceptance.** At an allowed Natural Gradient lab setting `p=.05`, target `.8`, fraction `1`, the correct logit endpoint is `0.9999973608069337`. The generic five-decimal formatter labels it `p=1`, while the same view labels failure `2.639e-6` and reports finite KL `12.004270365631227`. That misrepresents an interior distribution as a boundary point. Use a probability-specific formatter that preserves this distinction, explain rounded readouts, and check the actual UI state. The implementing author was notified before any source edit. Preserve the [pre-repair reproduction](evidence/second-order-probability-display-before.json) separately from subsequent evidence. **Resolved:** the probability-specific formatter now displays `1 − 2.639e-6` in the visible label, paired mass text and accessible label. A genuine 390×844 Edge check changed both sliders with keyboard Home/End, verified those labels and the rounding explanation, and captured the actual lab. The screenshot was opened and read. [Post-repair evidence](evidence/second-order-probability-display-after.json) passed at `2026-09-10T12:41:32.031Z`; [opened image](../../scratch/second-order-independent/probability-repair-390.png). This targeted regression does not replace the author’s broader browser review.

No mathematical defect was found in the reviewed mechanisms or six closed-form practice solutions. The seventh prompt originally supplied experiment acceptance criteria rather than a complete worked comparison. The author added the complete `changedFitComparison` program inside its answer reveal. The reviewer executed that actual program, matched all displayed stdout, and instrumented its method loop to compare final full-precision losses and gradients against a separately derived NumPy Jacobian. Both passed: L-BFGS used 15 evaluations including the final check, MSE `4.710055231541857e-5`, gradient infinity norm `6.008173197582405e-8`; GD used 1000, MSE `4.7800117944029675e-5`, norm `.0001421243705369792`. The different stopping outcomes and fixed-data/rate limitation are explained. No empirical winner is generalized. [Independent comparison evidence](evidence/second-order-changed-comparison-independent.json) passed at `2026-09-10T12:41:27.548717+00:00`. No review blocker remains in these scoped changes.

## Independent calculations

Run from the repository root:

```powershell
scratch/lesson-tools/Scripts/python.exe -X utf8 -I scripts/review-second-order-mathematics.py
```

The [durable JSON result](evidence/second-order-independent-mathematics.json) passed at `2026-09-10T12:35:28.394807+00:00`. The script imports the actual JavaScript model state into a separate Python/NumPy calculation. It captures the model hash at review time only; this is not a final-freeze hash.

| Contract | Complementary oracle and actual result |
| --- | --- |
| Inverse secant and two-loop order | Sixteen combinations of history budget, initial scaling and reversed newest pair. Apply the **Hessian-form** BFGS update, then invert and solve, rather than repeating the JavaScript inverse formula. Newest inverse-secant residual ≤`2.23e-16`; inverse matrix discrepancy ≤`1.12e-16`. Positive definiteness checked after each accepted pair. |
| True Fisher and column-major indexing | Six strength/damping fixtures. Differentiate the complete joint Bernoulli log-probability in flattened weight coordinates by central differences, then enumerate all eight input/outcome cases. This independently checks score signs, outcome weights and coordinate order. Fisher maximum error `4.74e-11`; direction error `1.31e-10` within declared finite-difference tolerance. Supervised fixed-label gradient is checked separately. |
| K-FAC approximation and damping | Constant covariance makes this fixture exact; nonzero layer strengths produce nonzero error. Expanded factor damping matches the extra cross terms, and two factor solves match a column-major Kronecker solve. These checks do not equate the factor damping policy with a full diagonal shift. |
| Spectral matrix roots | Nine replay/epsilon/rotation fixtures, including epsilon `.0001`. NumPy's symmetric eigensolver supplies independent inverse-quarter roots; fourth powers times accumulators return identity with maximum residual `1.45e-15`. Original-basis and rotated updates agree up to `4.45e-16`. |
| Natural coordinates | Six changed/boundary/no-motion fixtures verify model-score expectation, tangent mapping, finite Euler endpoints and actual Bernoulli KL. Distinct finite endpoints are explicitly checked when the target differs. Equal tangent action is not promoted to finite-coordinate invariance. |
| Newton/GD and plots | Four GD trajectories compared against a matrix power, four damped Newton states checked against their systems and Armijo acceptance. The singular fixture returns no direction/plot. All actual objective samples follow the stated fixed direction. |
| Changed practice | Six closed-form exercise groups recomputed. Fractions verify the indefinite Newton direction and scalar expectation/product counterexample exactly; NumPy solves and spectral powers check coupled Newton, the inverse secant and changed diagonal Shampoo update. |

Additional source reasoning checked the SPD BFGS proof, weak Wolfe implication and Armijo distinction, local Newton assumptions, finite-trial safeguards, local KL constrained direction, conditional output independence, spectral versus entrywise powers, and the HVP example's SPD Hessian structure. These are source-review findings, not extra executed-case counts. The HVP objective has positive diagonal curvature, nonnegative quartic curvature and a positive-semidefinite neighbor-difference term.

## Primary sources inspected independently

- [Martens and Grosse, K-FAC](https://proceedings.mlr.press/v37/martens15.pdf): section 1.2's input/model-outcome expectation; section 2, equation 1 and column-stacked vec; section 5/6's distinction between factored damping and a complete practical algorithm. The supplement's full damping derivation was not inspected in this cross-review. The lesson's explicit damping expansion was independently checked algebraically.
- [Gupta, Koren and Singer, Shampoo](https://proceedings.mlr.press/v80/gupta18a/gupta18a.pdf): matrix Algorithm 1 and section 1.1, tensor contraction definition and section 4.2/Algorithm 2. The original quarter/inverse-`2k` exponents and convex analysis scope agree with the lesson. No historical benchmark ranking was reused.
- [Yudong Chen, L-BFGS lecture notes](https://pages.cs.wisc.edu/~yudongchen/cs726_sp25/Lecture_23_L-BFGS.pdf): two-loop Algorithm 1, initial inverse scale and Wolfe-dependent update. The note's transpose convention for `V` differs from the lesson's symbol definition; both formulas correctly express the same inverse update.

No videos were watched as part of this independent review. The lesson's written/alternate-video resource claims remain tied to the implementing author's own inspected scope. This review does not duplicate the author's native program execution, exhaustive control grids, production loading checks or browser screenshot record.

## Scoped follow-up commands

```powershell
scratch/lesson-tools/Scripts/python.exe -X utf8 -I scripts/review-second-order-followup.py
node scripts/review-second-order-probability-fix.cjs
```

The initial mathematical JSON deliberately retains its dated seventh-exercise status; the separate follow-up records establish its subsequent resolution without rewriting historical evidence. Root owns final author freeze, full browser checks and production integration.
