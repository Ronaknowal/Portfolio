# PCA: conceptual-transition review

26 September 2026. Full canonical JSX read, including every deeper branch, eight exercises and the independent mini-project. The core lesson already has excellent point-of-need geometry, worked intermediate calculations, live controls and data-based decisions. Preserved those rather than adding a generic introduction. This record concerns author improvements; independent and browser review remain pending.

| Exact location and conceptual hurdles | Disposition |
| --- | --- |
| §1 measurement compression, observation/feature, mean baseline, task-dependent loss | Retained: same four physical measurements and real Wine overview give a concrete need before naming PCA. |
| §2 center, direction, score, projection, reconstruction, sign reversal | Retained: movable ruler and explicit point calculation connect the fitted mean to the recoverable observation. |
| §3 projected spread, perpendicular loss, orthogonality, conservation, percentage denominator | Retained: numerical horizontal/diagonal comparisons and conservation drawing teach equivalence before formal optimization. |
| §4 fitted state, array shapes, SVD orientation, transform vs refit, inverse, rank, implementation checks | Retained: complete first-principles program, shape diagram and explained numeric output. |
| §5 units, standardization, covariance vs correlation geometry, matching metric, train-only scales | Retained: live metric controls and concrete changed-unit results make the objective change visible. |
| §6 component choice, error budget, validation, mean baseline, monotonically decreasing input error, future data | Retained: live budget, actual Wine example, held-out selection and purpose-dependent baseline. |
| §7 scores vs directions, loadings conventions, biplot scaling, signs, interpretation | Retained: coordinate-level definitions and biplot reading figure with explicit conventions. |
| §8 clustering distance contraction, low-variance signal, predictive target, nonlinear dependence | Retained: existing label-information lab demonstrates discarded information despite high variance retention. |
| §9 practice and independent project | Retained all eight exercises, hints, explained solutions and changed-setting task. |
| §10.1 covariance/eigenvector, weighted eigenvalue average, stationary vs maximum, SVD, reconstruction identity | Gap fixed: weighted-average argument now returns to the earlier horizontal-ruler calculation (10/3), connecting formal proof to visible geometry. Existing complete proof, shapes and example retained. |
| §10.2 rank, tied basis/subspace, constant data, whitening, sample/population denominators | Gap fixed: same four score points drawn before/after whitening on equal numeric scales; explicit pair-distance changes distinguish rotation from metric change. Existing zero-variance/tie boundaries retained. |
| §10.3 sample spectrum, stability, denoising, noisy-input vs clean-target error | Gap fixed: perpendicular and parallel noise shown side by side with actual measured points, fixed projection line and clean target. Existing seeded Gaussian spectrum retained. |
| §10.4 supervised pipeline and model comparison | Retained: complete fold-boundary implementation and explicit no-PCA baseline. |
| §10.5 decoder storage, image weight patterns, residual alarms, neural time interpretation | Retained: scalar accounting and existing sensor residual figure connect each application to its actual mechanism. |
| §10.6 full/covariance/sketched/incremental computation, conditioning, memory, sparse centering | Gap fixed: randomized SVD intermediate shape table and candidate-span explanation expose what the approximation discards and where extra oversampling directions help. Existing cost bounds and input-memory calculation retained. |
| §10.7 alternatives and objective boundaries, regression vs PCA | Retained: scoped owner links and slope .8 versus 1 calculation. These are connections to separate lessons, not claims to teach those algorithms fully here. |
| §11 readiness and next lesson | Retained: exact module sequence and capability-based readiness. |

## Research and actual source reading

Read the PCA portion of the authors' [ISLP unsupervised-learning notebook](https://islp.readthedocs.io/en/stable/labs/Ch12-unsup-lab.html): scaling, fit/transform, score and component shapes, biplot and variance explanations. Read the geometric and algebraic SVD explanation in [Gregory Gundersen's original SVD article](https://gregorygundersen.com/blog/2018/12/10/svd/), including rotation/scaling, Av=σu and outer-product expansion. Used the geometric progression to assess the jump into later algebra; no wording copied. Existing canonical survey, MIT slides/video alternatives and official API sources retained. No claim that the linked full video was watched.

## New representations

- `WhiteningGeometryFigure`, §10.2: two SVG score planes, same pixels per numeric unit, raw wide rectangle versus unit-variance square. Exact four-point lesson fixture, not empirical output from a fresh fit. Caption explains magnification and changed distances. Paired panels stack at narrow widths.
- `DenoisingDirectionFigure`, §10.3: equal-axis sensor plots. White circle denotes clean signal, amber diamond measured input, projection segment amber. A fixed line is stated explicitly so the picture cannot imply fitting PCA from one point.
- §10.6 table: A, Y, Q, B dimensions and the retained meaning at each operation. Full-rank sketch assumption stated. This teaches the approximation through shapes instead of another generic flow diagram.

All original models, examples, lab controls, results and practice retained byte-for-byte in their source modules. Scoped parse/arithmetic/hash checks are in `author-checks.json`; original numerical programs were not retrained. Browser and independent review are not claimed.
