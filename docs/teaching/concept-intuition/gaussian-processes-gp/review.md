# Gaussian Processes — concept intuition author review

26 September 2026. Entire lesson read through all core and deeper subsections, worked code explanations, five core practice solutions, transfer/connection checks and references. Existing finite GP data, native programs and all live labs remain unchanged. No topic-body generator found in current scripts; direct JSX is authoritative.

## Whole-lesson concept map

| Location / conceptual transition | Assessment and action |
| --- | --- |
| Opening and §1 finite vectors → functions → GP definition | Existing linked vector/function illustration and random-line construction explain the new random objects. Retained. Renamed the old “Quick prediction” label to a direct explanatory label. |
| §1 covariance, units, PSD and impossible matrix | Existing negative-variance counterexample and weighted-combination argument give local intuition. Retained. |
| §2 latent value versus noisy measurement; Gaussian update | Two-number calculation, geometric slice, noise distinction and live controls already connect the mechanism. Retained. |
| §2 values move mean, geometry moves variance; pointwise bands | Existing explicit invariance/counterexample and two interval calculations sufficiently distinguish these claims. Retained. |
| §3 block covariance → conditioning → many targets | Shapes, correction/removal interpretation and matrix illustration retained. |
| §3 midpoint cancellation, extrapolation and noisy/correlated observations | Existing numerical comparisons and live matrix lab sufficient. Retained. |
| §3 Cholesky implementation, jitter, alpha/WhiteKernel, units | Full scratch program and library equivalence retained, including covariance rather than mean-only comparison. |
| §4 RBF length, kernel smoothness and alternatives | Existing matched-draw/matrix geometry and precise differentiability assumptions retained. |
| §4 addition, multiplication, local periodicity | Added computed periodic-versus-decaying-periodic covariance curves, same-hour values and explanation of recurrence versus persistence. Existing sum-of-independent-components and product-not-product-of-samples cautions retained. |
| §4 ARD, units, periodic counterexample | Existing observed-range/causality caveats and changed-kernel counterexample retained. |
| §5 marginal likelihood, quadratic fit and determinant | Added covariance ellipse/eigendirection comparison for the actual opposite-sign observations. Shows why the long scale narrows the direction containing the data; separates this fit-cost mechanism from the full objective. |
| §5 hyperparameter optimization versus integration, CV | Existing plug-in distinction, local optimum and held-out forecasting limits retained. |
| §6 CO₂ task, kernel composition, training-only centering, model selection | Full data provenance, complete library program, forecast illustration, live forecast lab, baseline and actual poor coverage retained. |
| §7 changed practice | All five tasks and solutions preserved, including singular constant kernel and unchanged covariance tests. |
| §8 posterior covariance → measurement choice | Existing exact candidate comparison and live target-reduction lab already sufficient. Retained. |
| §8 expected improvement | Added positive-part gain intuition before the closed form; retained noiseless-incumbent conditions and changed numerical practice. |
| §9 classification likelihood and nonlinear averaging | Added explicit two-point arithmetic showing averaging a sigmoid differs from sigmoid of the mean. Clearly distinguished this pedagogical distribution from a Gaussian posterior approximation. |
| §9 dense costs, inducing variables and trace correction | Added exact one-inducing-variable residual-variance diagram and trace penalty. It explains what is lost by the representation before showing other approximation families. |
| §9 SVGP, SKI, random features | Existing distinct mechanisms, costs and shared-map requirements retained; no blanket linear-time claim. |
| §9 KRR equivalence and RKHS sample paths | Existing nλ convention, probabilistic boundary and Brownian/finite-rank contrast retained. |
| §9 derivative and multiple-output observations | Existing covariance derivatives, smoothness requirement and units retained. |
| Finish / references / offline route | All readiness checks, downloads and sequence preserved. |

## Added visual contracts

- `LocallyPeriodicKernelFigure`: period one, periodic length parameter one, RBF decay length two. Dashed covariance exp(−2 sin² πr); amber multiplies by exp(−r²/8). Shared domain r∈[0,6] days and covariance∈[0,1]; 241 deterministic points. HTML same-hour values at days 2/4/6. These are kernels, not sampled functions.
- `GaussianEvidenceDirectionsFigure`: covariance [[1.25,c],[c,1.25]] with c=exp(−2/ℓ²), ℓ=.3 and 3. Eigenvariances 1.25±c; ellipses use their square roots and a consistent y₂-up coordinate transformation. Actual y=[1,−1] lies in the contrast direction. One-standard-deviation contours explicitly are not 95% regions. HTML numbers supplement geometry; panels stack.
- `InducingResidualFigure`: unit RBF length one, X=[−1,0,1], inducing Z=[0]. Q diagonal [e⁻¹,1,e⁻¹], residual [1−e⁻¹,0,1−e⁻¹], trace 1.264241; noise .25 gives correction 2.528482. Shared-height columns preserve unit variance. Clearly distinguishes conditional-prior residual from final noisy-data posterior uncertainty.

## Research actually read

Read [Distill's authored visual GP article](https://distill.pub/2019/visual-exploration-gaussian-processes/), covariance geometry, conditioning and kernel combination passages. Used it to assess whether a picture supports each operation, while preserving this lesson's distinctions between covariance/correlation, variance/SD and noisy interpolation; no interaction or full-video viewing claim. Read [GPML chapter 5](https://gaussianprocess.org/gpml/chapters/RW5.pdf), marginal-likelihood decomposition around printed pages 113–114, and [Titsias 2009](https://proceedings.mlr.press/v5/titsias09a/titsias09a.pdf), equation 9 and the conditional-prior trace interpretation. New diagrams are independently computed examples, not reproductions. These sources are already annotated in the lesson.

## Author verification

Changed JSX parses; kernel curve bounds and exact recurrence peaks; eigenvalue/determinant/quadratic-cost equivalence plus ellipse-domain bounds; inducing diagonal/trace and nonlinear average checked. Original numerical outputs and programs were not rerun because unchanged. Browser/mobile and independent review remain pending root, recorded in the source-hash receipt.
