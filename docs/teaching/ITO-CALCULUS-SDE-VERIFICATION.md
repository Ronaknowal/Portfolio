# Itô Calculus & Stochastic Differential Equations — author verification

Mathematics position 37; stable ID `it-calculus-stochastic-differential-equations`. The existing legacy source filename continues to begin `ito-`. Title, ordering, memberships and progress identity are unchanged. The next module topic remains Numerical Methods, regardless of publication status elsewhere.

The final author status, exact six source fingerprints, actual verification timestamps and opened image identities are recorded in [the author evidence](evidence/ito-sde-author-review.json). This record does not substitute author checks for independent review, shared production integration or user acceptance. Root owns those separate integration steps.

## Preserved material and completed learning route

The complete original source and original program/output are retained in [the preservation record](evidence/ito-calculus-original-content.json). Original body SHA256: `eadd01f669092ce887284f14ee69c537f505c01d1f853a36f199048f2b453954`. The original four-step Python simulation still prints the five values ending at 0.777 and the separate model expectation 1.492. Native verification compares its exact code and output strings with that archive, including comments.

The old drift/diffusion and Brownian statements now begin with a physical state, elapsed time, conditional information and units. Exact finite left/right/symmetric sums precede the stochastic limit. The isometry is proved for finite adapted coefficients and extended under explicit measurability and moment conditions. The former formula-only chain rule now includes curvature geometry, a complete time-dependent centered-cubic example, vector covariance and the nonsmooth boundary. The GBM example retains exact positivity and the original simulation while adding moments, quantiles, almost-sure growth and a rare-outcome explanation. OU adds restoring flow, stationary versus fixed initial laws, an integrating-factor derivation and correct weighted-noise coupling.

The original convention and modeling warnings become worked model comparisons. Strong/weak solution terminology is distinguished from numerical error, with sufficient existence conditions, an outline of the Picard/isometry/Gronwall argument and a deterministic explosion example. EM/Milstein updates connect to the integral already derived. The exact error reference, sampled error budget and stability conditions have separate meanings. Generator, forward density, backward terminal question and a finite-horizon change of probability supply transparent deeper connections.

The original 40-step/1,000-path task is retained with analytical mean, median, quantile, variance and sampling references. There are thirteen complete displayed Python programs and eleven independent tasks with hints and explained solutions, plus an early information-timing checkpoint. Each program has its actual question before the code, exact output and interpretation. These counts follow the completed mechanisms; they were not quotas. Subthreshold voltage and a linear forward-corruption SDE are meaningful applications with explicit model and ownership limits.

## Representations and their contracts

| Representation | What it teaches and computes | Reading and evidence limits |
| --- | --- | --- |
| Noise scaling | Drift magnitude `|a|h` versus diffusion SD `|b|sqrt(h)` for a full and quarter interval | Exact ratios; bars are scales, not equal realized shocks. Four independent quarter-step variances sum correctly. Mobile stacks the comparisons. |
| Adapted integral investigation | Coefficient before versus after a selected increment; exact linked sums, endpoint and Q; a complete interval table | Hand fixture is explicitly deterministic. Seeded 256-point paths refine by grouping the same increments. A new seed changes the example. Q is not asserted to converge monotonically on a single finite path. |
| Curvature diagram | Quadratic curve, tangent at one, and two finite positive corrections from opposite perturbations | Both selected points fit at 320px; labels remain at least 14px. Green curve and dashed blue tangent are named. Finite algebra motivates, but does not prove, the stochastic limit. |
| Growth investigation | Exact sampled GBM in log coordinates, analytic log-density, mean/median markers and ordinary-state summaries | `log(E[X])` is explicitly distinguished from `E[log X]`. The same fine Brownian driver survives coefficient/time changes. Ordinary-state values remain visible without adding a duplicate path plot whose large dynamic range obscures the high-noise example. Point masses replace singular density formulas at zero time/noise. |
| OU restoring-flow investigation | Analytic Gaussian or atom, restoring directions, common-scale injection/removal budget and initial-law choice | No simulated OU path is claimed by this view. The exact OU joint-noise construction lives in the complete native example and tested companion model. Tiny budget residuals are disclosed as binary64 rounding. A non-Gaussian initial law does not inherit a Gaussian finite-time claim. |
| Convention comparison | Three equations, their log drifts and means; two equivalent laws after conversion | Static exact model laws, not three fitted samples. Scalar conversion and the separate vector correction are qualified in the surrounding explanation. |
| Coupled solver investigation | Fine-increment groups feed exact GBM, EM and Milstein; linked paths and the selected update's drift/noise/curvature terms | Drift/time/start stay fixed here to isolate refinement and noise; the separate error experiment varies drift. Shared grouping, not seed reuse alone, defines the comparison. The controlled negative-EM fixture is not a failure-frequency sample. No value is silently clipped. |
| Error budget investigation | Analytic finite-grid RMS curves, named first/second-moment biases, then a separately executed paired sample | No empirically fitted convergence slope is presented as a theorem. MSE and its standard error retain squared-state units; RMS is their separately identified square root. Zero errors are omitted from the logarithmic plot but retained exactly in text/table. |
| Probability duality | Forward law versus backward question, showing where derivatives act | A structural diagram; it does not claim a solved numerical PDE. The nearby OU moments and terminal-square example provide quantitative checks. |

Five investigations and four inline figures use different representations according to the learning hurdle. Default reading supplies the model and prediction before the controls. Simple analytic distributions, error curves and the one-step solver fixture fit the phone; denser path plots and interval tables have clearly announced local horizontal scrolling and keyboard access. Controls are at least 44px high; focus, invalid drafts and reset are checked. Color, hover and screenshots are not the only source of numeric meaning.

## Mathematical and numerical boundaries

- Brownian motion is relative to the declared filtration. The finite simple-integrand proof conditions on the later interval's past; it does not mistake adaptedness alone for a complete integrability theorem. Progressively measurable square-integrable or predictable square-integrable integrands are standard sufficient contracts. Local martingale is not silently replaced by mean-zero true martingale.
- Deterministic-partition Q has variance `2 sum h_i²`. The mesh bound proves L² convergence; the separate dyadic summability argument gives almost-sure convergence along that sequence. No arbitrary path-dependent partition or ordinary iid triangular-array argument is claimed.
- Scalar `C^{1,2}` and vector Hessian/covariance terms are explicit. The centered cubic `W³−3tW` uses the time derivative and has finite-horizon second moment `6T³`. The absolute-value kink is flagged as requiring local time.
- GBM positivity is established by verifying the exponential solution before applying a logarithm to the state. Pointwise quantiles are not simultaneous path bands. The long-time decay/mean-growth example cites the actual almost-sure Brownian law and does not exchange limits and expectations.
- OU exact sampled transition laws do not, by themselves, establish an exact shared-Brownian pathwise reference. The weighted integral's covariance and residual conditional variance are derived and independently checked. Stationary initial laws are independent of future driver increments; non-Gaussian starts preserve moment formulas but not an automatic Gaussian marginal.
- Itô/Stratonovich is a modeling convention requiring a converted drift. Sufficient strong-solution and numerical-order conditions are stated, along with their limits. General nonlinear or noncommuting multidimensional problems do not inherit the scalar Milstein update.
- Mean-square decay, almost-sure decay, finite-time error and positivity have separate criteria. A weak claim names its test function; exact mean under drift zero does not validate the full law.
- Forward equations require appropriate smoothness or a weak formulation and boundary terms. Coefficients remain inside forward derivatives. Backward expectations retain terminal conditions, smoothness/integrability and the OU polynomial example. The constant Gaussian change of measure is finite-horizon and normalized by a genuine martingale; no generic unstopped stochastic-exponential claim is made.

### Accepted arithmetic and stable formulas

The browser's seeded experiments use strictly interior uniforms and a bounded synthetic normal generator, not an empirical dataset. Fine grids contain at most 512 increments; displayed grids use at most 256. Seed input accepts only integer strings from 1 through 2,147,483,647. Invalid drafts preserve the active model. Growth parameters support drift −0.5…1, positive start 0.1…3, noise zero or 0.05…1.2, and time zero or 0.0625…4. Exact zero states have separate branches; paths require positive time. Underflow of an exact positive path or a nonfinite path state is explicitly rejected.

OU supports reversion zero or 0.05…3, noise zero or 0.05…1.2, initial/target within ±2, initial variance 0…4 and time zero or 1/8192…4. The weighted-noise helper uses positive steps 1/8192…4. Error references restrict T to 0.25…2 and n to 1…512; paired sampling allows 16…4,096 paths and at most 1,048,576 path-steps. Display controls are a smaller subset. The complete Python helpers state similarly bounded teaching ranges; they do not promise arbitrary floating-point numerical robustness.

The OU residual avoids subtracting near-equal binary64 quantities. With `z=theta*h`, it is

`h exp(-z) sum_{k>=1} 2k z^(2k)/(2k+2)!`.

Every term is nonnegative. The independent oracle instead evaluates the original covariance subtraction with 90 decimal digits; a complementary native oracle integrates the centered exponential kernel. Zero reversion gives exactly zero residual.

For GBM errors let `d=mu*h`, `u=1+d`, `s=sigma²h`, `r=s` for EM or `s+s²/2` for Milstein, and `F=u²+r=E[A²]`. The local normalized cross moment is `rho=(u+r)/sqrt(F exp(s))`. In the supported range u≥0. Its nonnegative defect is

`1-rho² = exp(-s) [tail(exp(s), k) + r*d²/F]`, with tail beginning at power k=2 for EM or k=3 for Milstein.

For n independent steps, `1-rho^n` is evaluated by `-expm1(n/2 * log1p(-defect))`. The final error is the sum of a squared norm difference and a nonnegative correlation-loss term:

`MSE = (||X||₂ - ||Y||₂)² + 2 ||X||₂ ||Y||₂ (1-rho^n)`.

The norm ratio and weak biases use `log1p`/`expm1` and small-argument series. This avoids an unexplained negative-MSE clamp. Zero noise uses the squared deterministic bias. The independent Python oracle uses the original full moments at 100-digit precision; the native checks also integrate one-step Gaussian products with independently selected quadrature.

## Actual verification

Commands, run from the application repository:

- `node scripts/verify-ito-sde-models.mjs` exports actual model fixtures and invokes the independent Python oracle. It passes 11,757 numerical comparisons: 900 finite-grid error cases, 25 weighted OU noise cases, 225 OU laws, 64 growth laws, 60 exact finite increment lists, six grid groupings, 18 GBM paths, four unrolled OU paths and two reproducible sampled smoke checks. Three additional adapted-isometry quadratures and 22 explicit input rejections pass. Model results are preserved in the author evidence.
- `scratch/lesson-tools/Scripts/python.exe scripts/verify-ito-sde-native.py` executes all thirteen exported programs unchanged and matches stdout, then tests changed actual helpers. It passes 1,960 numerical comparisons plus 363 exact rational list cases; coverage includes 36 isometry, 54 lognormal, 88 OU, 16 rectangular covariance, 81 shared-increment solver, 384 error, 72 generator, twelve constant-tilt and 36 time-transform cases. Seven rejected-input contracts and exact original preservation pass. The constant-tilt changed cases are a complementary mathematical check, not a claim to call a reusable helper the program does not expose.
- `node scripts/review-ito-sde-lesson.cjs` uses Edge with actual public fonts at 1440, 390 and 320px. It exercises every selector value, selected endpoint/interior time cursors and coupled resolutions across the five investigations, validates linked values, checks original and new rendered question/code/output strings, opens practice disclosures, tests invalid seed preservation, reset/keyboard activation, range-key operation and mobile plot scrolling, and saves reading/figure/control-state screenshots.
- `node scripts/review-ito-sde-reading.cjs` independently captures every displayed equation and the compact figure in ordinary page context at all three widths, checks actual fonts, local geometry, browser/network errors and native-readable compact labels. Its final results and screenshot identities are saved with the author packet.

During authoring, tests exposed cancellation in a preliminary error formula, a JSX exponentiation syntax mistake, reset not clearing an already-active invalid seed draft, a too-strict equality assertion for stationary OU rounding, missing display of the example interpretation, and narrow equation/plot labels. These were corrected before freeze. The final record preserves completed checks rather than counting failed setup runs as evidence. Larger relative residuals near a vanishing OU rate are governed by an explicit absolute tolerance; for references above 1e−9, the model oracle's maximum relative discrepancy was about 1.56e−12. Statistical smoke checks use a declared wide tolerance and do not guarantee interval coverage on every seed.

The ordinary reading review chose a compact, fully visible two-point curvature chart; denser paths retain local scrolling with exact readouts. It also added visible mean/median positions on the log-density, clarified nonnegative scale magnitudes, split equations without reducing type size, and separated the time-dependent transformation from a formula-only mention. The original educational coverage is preserved while the reading flow is substantially expanded.

## References, scope and handoff

The [approved design](ITO-CALCULUS-SDE-LESSON-DESIGN.md) records exact inspected source sections and the limits of the source review. The body includes annotated primary written references and an official MIT lecture alternative with companion notes. Only the video identity/course page and companion notes were checked; no full watch or timestamp claim is made. The neuronal source's changing noise normalization and informal smoothness wording are not copied. The Song paper is explicitly a checked identity/abstract contextual pointer, not authority for an unreviewed reverse-time derivation or performance claim.

The incoming [Stochastic Processes note](topic-notes/it-calculus-stochastic-differential-equations.md) is addressed by the actual QV/information/solver-coupling treatment and bounded crossing reminder. A nonlinear SDE bridge correction remains outside the exact Brownian result. New discoveries were routed to the existing diffusion-sampler and Hodgkin–Huxley/LIF owners; those topics are not claimed complete. No shared catalogue, manifest, blueprint index, lesson registry, global stylesheet or rollout ledger was changed by this author.

Independent review and root's shared production checks should use the final six source hashes in the author packet. Any resulting correction must preserve the before/after evidence and refresh the affected checks.

## Independent-review amendment: active stress controls

After the first author freeze, the reviewer identified a learner-visible state mismatch: the one-step positivity fixture correctly used h=1 and σ=1, but two disabled selectors could still show the previous seeded resolution and noise. The lab now displays the forced active values, one step and σ=1, while retaining the seeded choices for switching back. Its algorithms, body, examples, blueprint and stylesheet did not change.

The focused `node scripts/review-ito-sde-stress-correction.cjs` check loads the final source with actual fonts at 1440, 390 and 320px. For each width it switches from four different seeded noise settings at 128 steps into the stress fixture, checks the displayed forced values and actual negative-EM output, switches back and checks restoration, then verifies keyboard reset. Fresh state/ordinary-reading screenshots were opened. The author packet preserves the first freeze and full 318/319-state browser runs, identifies this narrow amendment separately, and updates the exact lab fingerprint. It does not claim that the entire interaction matrix was rerun after these two rendered-value expressions changed.
