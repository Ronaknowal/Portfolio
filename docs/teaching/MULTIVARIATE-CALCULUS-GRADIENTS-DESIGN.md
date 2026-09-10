# Multivariate Calculus & Gradients — teaching design

Stable ID: `multivariate-calculus-gradients`. Mathematics position 7, next **Convex Optimization**. Source ownership is semantic and lesson-local; the parent agent owns registration, generated files and integration.

## Coverage and title review

The original source was read in full. Retain the many-parameter motivation, gradient signs, local limitation, bowl x²+2y², original (3,−2)→(2.4,−1.2) update, partial/directional/Hessian concepts, automatic differentiation, poor scaling, exploding/vanishing derivatives, and learning-rate practice at 0.6 and 0.01. Replace pseudocode with a complete executable derivative example and diagnose the original update practice mathematically. No useful original concept is removed.

Retain the existing title and stable identity. This lesson's scope is multivariable **local change and gradients**; it is not a claim to teach every theorem of vector calculus. Single-variable slopes and vector products are refreshed in place. The existing recorded prerequisites remain; Matrix Calculus depends on this topic, so it is a review link and not a required dependency. Earlier Matrix Calculus already develops vector-output Jacobians, JVP/VJP, matrix-shaped gradients, finite-difference pitfalls, solves and softmax. Here we deepen scalar geometry, smoothness, level curves, curvature and optimization readiness rather than repeating that entire workflow. Existing Eigenvalues provides curvature-axis interpretation; the local 2D cases are derived without requiring that detour.

| Idea | Evidence / best owner | Decision |
| --- | --- | --- |
| All-input local approximation | Original only says local; earlier Matrix Calculus introduces maps | Derive exact bowl remainder, then a differentiability contract here |
| Partial derivatives versus differentiability | Not developed in original | Include axis/line/curved-path counterexample with explicit values |
| Steepest direction and units | Original unqualified negative-gradient statement | Prove Euclidean unit-vector result, distinguish speed and coordinate scaling |
| Level curves / allowed directions | Absent in original | Derive chain-rule tangency and a complete circle-constrained example; later KKT owns general constraints |
| Hessian / stationary point conditions | Original names bowls/saddles | Explain Taylor quadratic form, strict tests, zero-Hessian failures and boundary caveat |
| Gradient descent and scaling | Original one step and rate prompts | Derive exact coordinate recurrence, convergence interval and invariant/unstable boundaries; optimizer variants remain later |
| Sensitivity with physical units | Valuable distinct application | Work resistor-power changes and first-order cancellation; no clinical/financial use |
| Multidimensional accumulation | Narrow catalogue search found measure/probability owners but no introductory multiple-integral lesson | Save a scoped destination note for measure-theory/probability, rather than imply this differential lesson covers Fubini/change-of-variables/vector integral theorems |

## Learning route and representations

Learner finish line: compute and interpret a derivative, predict a small move with error, reject an invalid smoothness/optimality argument, and analyze a simple gradient iteration. Core reasoning stays visible; deeper counterexamples/coordinate metrics and AD mechanics use disclosure only after a complete local explanation.

| Hurdle | Representation / model contract | Learner action / independent evidence |
| --- | --- | --- |
| A surface has many slopes | Inline two coordinate-slice plots at (1,1), f=x²+2y²; axes and tangent slopes labeled | Read which input is fixed; derive both partials from difference quotients |
| Gradient lives in input space | Contour plane of the exact bowl, unit direction, normalized gradient-direction arrow plus exact magnitude | Rotate a direction; compare dot product with the slope of its linked function slice |
| Local prediction has a remainder | Linked exact slice and tangent line, actual and predicted finite changes, exact quadratic residual | Reduce step size; explain fourfold residual reduction; zero-rate tangent does not imply zero finite change |
| Partials inspect too few approaches | Calculated approach paths for g=x²y/(x⁴+y²), g(0,0)=0; exact small-input table | Compare axes/straight lines with y=cx²; one bad path disproves continuity |
| Motion and constraints | Unit-circle input plane, actual tangent velocity, fixed gradient (2,1), output versus angle | Reach constrained maximum/minimum where along-circle rate vanishes while ambient gradient stays nonzero |
| Curvature is directional | Exact origin slices for quadratic bowl/max/saddle and quartic minimum/saddle, quadratic approximation and signed curvature | Rotate direction; distinguish definite, indefinite and inconclusive zero Hessian |
| A finite step may fail | Discrete bowl descent trajectory, per-coordinate factors, step/reset/back, exact loss history | Compare η=.01,.1,.49,.5,.6; explain zero/negative/unit/unstable recurrence factors |

All plotted values are calculated from stated functions; no empirical speed or accuracy ranking. Plots retain readable 330px native geometry with narrow adaptations and explicit numeric alternatives. State has bounded inputs and deterministic reset. Direction arrows that are normalized for display say so; curve tangents and gradients have separate spatial meanings. Logarithmic loss displays, if used, must label scale and avoid zero logs. Avoid relying on color alone: curve styles, point markers, legends and table values carry the same distinctions.

## Planned complete examples and independent practice

Use NumPy only for optional numeric programs; a small forward-mode dual class with NumPy derivative arrays supplies a complete AD mechanism without importing a framework or promising a universal autodiff engine. Capture stdout from the actual programs. Examples cover local expansion, normalized direction, path limits, chain-rule rates, circle constraints, Taylor/Hessian counterexamples, original quadratic descent including unstable rates, units/sensitivity, and derivative verification. Independent practice changes functions/points and includes proofs/counterexamples, not just copied inputs.

Native verification will compare pure browser states with independent analytic/NumPy oracles; exercise quadratic iterates against closed forms, path values against rational arithmetic, circle extrema against Cauchy–Schwarz and direct parameter differentiation, curvature versus analytic Hessians and central second differences. Actual browser controls, keyboard, invalid states, anchors, code, math and ordinary-reading screenshots at 1440/390 (plus relevant 320 geometry) must pass before source freeze. Parent production integration is distinct.

## Research ledger, 10 September 2026

- OpenStax *Calculus Volume 3* [4.4](https://openstax.org/books/calculus-volume-3/pages/4-4-tangent-planes-and-linear-approximations): read tangent plane, differentiability and differential definitions; compare exact remainder to the first-order model. Original examples here are independently constructed.
- OpenStax [4.6](https://openstax.org/books/calculus-volume-3/pages/4-6-directional-derivatives-and-the-gradient): read unit-direction definition, differentiability theorem, steepest direction and level-set normal claims. Keep smoothness and nonzero-gradient qualifications next to the claims.
- OpenStax [4.7](https://openstax.org/books/calculus-volume-3/pages/4-7-maxima-minima-problems): second-derivative test, critical points and closed bounded domains. Strict definiteness establishes strict extrema; a zero determinant is inconclusive.
- OpenStax [4.8](https://openstax.org/books/calculus-volume-3/pages/4-8-lagrange-multipliers): equality-constrained stationarity; regularity and checking candidates rather than guaranteeing every solution is optimal.
- MIT [Session 38](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/resources/gradient-and-directional-derivative/), Joel Lewis: direct video `https://www.youtube.com/watch?v=XZ1QwS1IKgw`; official title/index checked and [six-page recitation transcript](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/38801e7f462cb1980d49fbcd7bc96ed7_MIT18_02SCF10Rec_27_300k.pdf) inspected for the worked gradient-direction route. No full video playback claimed.
- MIT Denis Auroux [Lecture 12](https://www.youtube.com/watch?v=2XraaWefBd8): official title verified, [transcript](https://ocw.mit.edu/courses/18-02-multivariable-calculus-fall-2007/5628908f9b3f6cfe621f9ee38a65d199_18_022007L12.pdf) is available for the gradient, direction and tangent-plane route; record exact portions read in final evidence. No entire-course or video-watch claim.

## Final decisions and additional source checks

Implemented the five concept-specific investigations and two inline figures described above, including an original oblique projection of the tangent-plane patch and its graph normal. Every plotted relationship is calculated from its equation; the normal caption distinguishes 3D geometry from apparent projected angles. Both axes expand together for diverging descent and all raw coordinates remain available.

The complete source has nine core sections, ten runnable programs and six independent practice tasks with separate hints and explained solutions. Program execution/setup is stated before the first code block. The title and prerequisite decisions remain unchanged. Curvature versus stationary/constraint conditions, mathematical derivatives versus numerical results, and per-distance versus per-parameter rates remain explicit.

- [OpenStax University Physics 2, section 9.5](https://openstax.org/books/university-physics-volume-2/pages/9-5-electrical-energy-and-power): read the ideal-resistor P=V²/R and P=I²R equations used in the independently calculated sensitivity tasks. No hardware experiment or thermal-constancy claim.
- [OpenStax Calculus 3, section 6.3](https://openstax.org/books/calculus-volume-3/pages/6-3-conservative-vector-fields): checked Theorem 6.7 and its chain-rule proof for the optional path-integral connection. This does not claim every vector field is a gradient or develop all of vector calculus.
- [NumPy 2.3 gradient](https://numpy.org/doc/2.3/reference/generated/numpy.gradient.html): inspected parameters, sample-spacing and finite-difference behavior, distinguishing it from differentiating a Python program. Actual programs ran with NumPy 2.3.5.
- [PyTorch autograd mechanics](https://docs.pytorch.org/docs/2.14/notes/autograd.html): the stable URL resolved to 2.14 on 10 September 2026; inspected the nondifferentiable-function rules. PyTorch was not installed or executed for this lesson.
- MIT resource review bounds: Lewis recitation transcript portions covering computed gradients and normalization in two to four dimensions, particularly PDF pages 3–5; Auroux lecture transcript portions covering level surfaces, normal vectors and tangent planes, particularly PDF pages 2–5. Official video identities were verified. Full video playback and entire-course review were not performed.

The product-integration discovery is now saved in [the Measure Theory destination note](topic-notes/measure-theory-probability-spaces.md). That proposed teaching remains open; no extra topic or destination lesson was rewritten. No relevant incoming note existed for this lesson; the unrelated bit-manipulation inbox entry was assessed as outside scope.

This is an author design, not observed beginner testing or user acceptance. Actual numerical, independent review, browser and screenshot evidence is in [the final verification record](MULTIVARIATE-CALCULUS-GRADIENTS-VERIFICATION.md).
