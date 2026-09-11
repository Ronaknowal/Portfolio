# Numerical Methods: lesson design

11 September 2026 local date. Mathematics Foundations, position 38, stable ID `numerical-methods-finite-differences-quadrature-root-finding`. This design was reviewed and its individual brief registered before implementation. The original plan below is preserved; the final implemented disposition and actual evidence are recorded in [Numerical Methods verification](NUMERICAL-METHODS-VERIFICATION.md). Planned checks in the design are not themselves evidence that they ran.

## Entry, preservation and learning contract

The exact-topic inventory command and all returned authoring notes were read. The current entry is foundation level, in Advanced Mathematical Structures, published with individual design/review required and no recorded prerequisites. The route places it after Itô Calculus & Stochastic Differential Equations and before Functional Analysis & RKHS. Preserve this order and stable ID. Do not substitute the next published page for the next curriculum topic.

Retain **Numerical Methods (Finite Differences, Quadrature, Root Finding)**. The three named mechanisms remain the core. Connections to linear algebra, automatic differentiation and differential equations explain the common error discipline without promising complete treatment of those fields.

Read the entire original seven-section body, its complete Python program and its final two practice suggestions. Its useful ideas are the distinction between error sources, bracketed versus tangent-based roots, forward/central differences, area approximation, appropriate linear solvers, derivative checking and a numerical verification workflow. The weakness is that most relationships are asserted briefly: no diagrams, no investigation, limited derivations, little failure evidence, incomplete adaptive-quadrature teaching and no worked answers to the final tasks.

Preserve the exact original program and its three-line output as an integrated first reproducible checkpoint, with an explanation of each result. [The original archive](evidence/numerical-methods-original-content.json) stores the complete source, code/output blocks and execution. Source SHA-256: `01546bccc0729783f7184fb1fa3a9d9ba8e49c701b62e9a01a427e1a3a9604f3`. Python 3.12.14 executed it on 10 September 2026 at 20:56:59 UTC; stdout matched exactly:

```text
1.414214
0.539402 0.540302
1.896119 2.0
```

The opening question is **“How can a computer give a useful answer when it cannot calculate the mathematical answer exactly?”** A synthetic pump supplies a concrete common setting: a rate curve, accumulated volume and the time at which a target volume is reached. The reader should finish able to choose a method, carry out its small steps, explain the approximation, report a defensible accuracy statement and recognize a failed assumption or unsuccessful computation.

Proposed exact-title prerequisite: **Single-Variable Calculus: Limits, Derivatives & Integrals**. Its actual curriculum plan covers slope, area and the fundamental theorem, but the lesson is currently planned and appears later in module order. Record this prerequisite honestly; teach short slope/area/Taylor refreshers locally, rather than silently requiring an unavailable page or rearranging the route. Python arithmetic/functions/loops are needed only to run the programs; explain setup before the first program. No new hard ODE, AD, probability or numerical-PDE prerequisite is needed. A named order such as “second order” must be defined by how the error changes as the step changes.

## Scope and neighboring owners

| Idea / evidence inspected | Decision, depth and teaching owner | Durable destination |
| --- | --- | --- |
| Scalar roots, derivative stencils and quadrature: all mentioned in the old body, but adaptive control and method assumptions are thin. | Develop these mechanisms here, including usable failure/status contracts and complete changed-input practice. | This design, sections 2–7 below. |
| Local interpolation: old trapezoid/Simpson discussion gives no reason for the weights. | Teach line/quadratic interpolants and small polynomial moment matching only as needed for stencils and quadrature. A full spline/approximation-theory course would dilute the named scope. | This design, sections 4–7. |
| Floating-point representation and general conditioning: actual expansion plans for Floating-Point Representation & Numerical Error and Conditioning, Stability & Numerical Analysis own those broader skills. | Refresh evaluation error, cancellation, representability and residual sensitivity here. Retain a small linear-system counterexample and link the existing Matrix Decompositions lesson. No new factorization/CG chapter. | Existing owners, with this design's local bridge; no new gap asserted. |
| Gradient checking: actual Backpropagation section 4f has useful code but overgeneralizes fixed perturbations and finite-test conclusions. | This lesson teaches error-aware scalar/directional checks; engine-specific correctness remains with Backpropagation. Qualify the old Numerical Methods claim that AD is automatically accurate to machine precision. | [Saved Backpropagation note](topic-notes/backpropagation-automatic-differentiation.md), open for reassessment. |
| ODE error: finalized Dynamical Systems cooling/oscillator sections and the incoming destination note distinguish dynamics from numerical evolution. | Adapt as a short equal-horizon integration bridge. Teach local versus accumulated error; do not duplicate the full oscillator investigation or pretend solver tolerances bound every trajectory. | [Incoming note](topic-notes/numerical-methods-finite-differences-quadrature-root-finding.md). Assessed during design; implementation resolution remains pending. |
| Full ODE methods and numerical PDEs: actual expansion plans own ODE linear systems, mesh/boundary/stability/finite-element questions. | Keep them as follow-on owners. Explain why their error questions connect; detailed Runge–Kutta tableaux, stiff solvers, symplectic theory, PDE stencils and finite elements are outside this core. | Existing planned owners; no new topic requested. |
| Gaussian quadrature: absent from original, but node selection is a useful contrast to equally spaced samples. | Derive the two-node rule as optional depth after Simpson. Do not imply a new-node method can recover unmeasured observations. | This lesson, section 7. |
| Interval arithmetic and verified computation: useful when strict floating-point guarantees matter, not otherwise developed here. | Distinguish mathematical error bounds from an implementation that rigorously encloses every rounding error. Give the boundary explicitly; no unsupported certification language. | Deeper numerical-analysis follow-on, not a promised complete subcourse. |

## Teaching flow and conceptual hurdles

The sequence is a learning dependency plan, not mandatory identical headings. Expand or combine sections if the actual reading makes a stronger explanation. Introduce each symbol, interval convention and unit beside its first example. Keep essential reasoning visible; use optional detail for longer derivations and the Gaussian rule.

### 1. A number, a target and an accuracy claim

Use a pump's rate as a synthetic mathematical model, not measured device data. A slope estimates how the rate is changing, an integral estimates delivered volume, and a root locates a desired accumulated amount. Show these three objects immediately on aligned, labelled curves. Start with a tiny hand calculation before general symbols.

Define approximate answer, true error, computable bound, estimated error and tolerance separately. Absolute tolerance carries the answer's units; a relative tolerance needs a meaningful scale and does not replace absolute control near zero. Separate uncertain measurements/model assumptions from discretization and arithmetic. Ask whether “the residual is small” identifies an accurate location on a nearly flat curve; establish the slope dependence before a solver success flag appears.

### 2. Keep a root inside an interval

Build bisection from continuity and opposite endpoint signs; handle an exact endpoint root first. A sign change need not imply a root across a discontinuity, and a continuous function may have a touching root without a sign change. Distinguish existence inside a bracket from uniqueness.

Trace actual evaluations and the retained interval. After `k` halvings of the initial interval, its width is `(b0-a0)/2^k`; the midpoint of that current interval has distance at most half that width from at least one bracketed root, in exact arithmetic. State the iteration convention so a formula cannot silently be off by one. Check signs without multiplying huge function values. Teach maximum work, nonfinite evaluation and a midpoint equal to an endpoint as explicit outcomes. A computed sign requires an adequately accurate function evaluation; return uncertainty when an inexact oracle cannot determine it.

### 3. Use slope without losing the root

Draw Newton's tangent and derive its horizontal intercept. Work the familiar square-root iteration, then the exact cycle `x=0 → 1 → 0` for `f(x)=x^3-2x+2`. A zero derivative, an excursion outside the domain, a small step without a small residual and a multiple root each need interpretation. Explain the local smooth/simple-root assumptions for quadratic convergence; `(x-1)^2` instead gives an error factor of one half under ordinary Newton.

Build a modest safeguarded Newton method: retain a valid bracket and accept a finite Newton candidate only in its central portion, otherwise bisect. A chosen 10% margin gives at least 10% width reduction per successful exact-arithmetic update; the implementation must still detect floating-point stagnation. Name this actual teaching algorithm, not Brent's algorithm. Introduce a secant as a slope from two points, then annotate the real `brentq` library hybrid and its status interface. Derivative-free does not mean assumption-free.

### 4. Estimate derivatives from nearby values

Connect a secant to a tangent, then derive forward and central stencils by Taylor expansion. Introduce a three-point second derivative and a one-sided stencil when a boundary prevents a symmetric sample. A small polynomial moment table explains cancellation of unwanted terms; avoid an unexplained list of high-order coefficients.

Under the stated smoothness and exact sampling assumptions, expose forward first-order and central second-order truncation behavior. Then use a separate additive evaluation-error model: if each sampled value is within `η`, forward error is at most `M2*h/2 + 2η/h`, while central error is at most `M3*h^2/6 + η/h`. These expressions bound the sample-error model in exact subsequent arithmetic; actual floating-point subtraction/division adds further error. Constants, units and vanishing leading derivatives matter. Square-root/cube-root step scales are conditional balances, not a universal prescribed `h`.

Retain and fully solve the old `sin(1)` step sweep. Plot actual computed errors, including irregular last-bit behavior, separately from a labelled theoretical envelope. Never invent a smooth empirical U-curve or turn tiny positive errors into exact zeros. Detect `x+h == x`; distinguish exact mathematical zero from a rounded display. Give one directional gradient-check transfer with a small locally defined two-variable function. AD avoids the finite-difference truncation tradeoff by propagating derivative rules through the implemented computation, but still has arithmetic, primitive and branch conventions; agreement in finite tests is evidence with a scope.

### 5. Turn local shapes into total area

Treat callable functions and fixed sampled data as different inputs. Derive the trapezoid rule by integrating each line segment, including unequal sample spacing. Derive Simpson's panel from a quadratic interpolant and its `1,4,1` weights; explain why classic composite Simpson needs an even number of subintervals. Work a polynomial case exactly, then revisit the original sine-area approximation.

Present the usual composite error bounds with bounded second/fourth derivatives and equally spaced grids where required. “Exact for cubics” is not “exact for every smooth curve,” and the equally spaced guarantee must not be copied onto arbitrary sample positions. Compare geometric areas, evaluations and units, not only returned decimals. Explain when a denser measurement record is needed rather than calling an algorithm that asks for unavailable new values.

### 6. Refine where the function needs attention

Derive a Richardson comparison from a leading error term, then derive the Simpson difference-over-15 estimate. A complete adaptive Simpson program reuses existing values, divides an absolute local error budget between children, combines returned values/errors and reports depth/evaluation/stagnation limits. A request for tolerance is not a certificate that an estimator is reliable.

Use a narrow but resolved peak to show uneven subdivision, followed by the exact blind-spot fixture `g(x)=[x(x-1/4)(x-1/2)(x-3/4)(x-1)]^2` on `[0,1]`. The first five coarse/refined Simpson sample values are all zero, while the exact integral is `5/1419264`. This is a smooth, nonnegative counterexample, not a discontinuity trick. It explains what finite samples cannot establish without additional information. A visual curve may use extra points for drawing; those points must not secretly feed the integration algorithm.

Show a useful repair mechanism, not only a warning: transform `∫_0^1 x^(-1/2) dx` by `x=u^2`, giving the constant integrand 2 with the removable endpoint defined by its limit. Truncating the original lower endpoint at `δ` discards `2√δ`, which can dominate the requested tolerance. Known singularities, breakpoints and oscillations need an appropriate method or domain treatment. Read actual `quad` diagnostics and distinguish its QUADPACK algorithms from the teaching Simpson program.

### 7. Choose nodes, then compose numerical operations

Optional Gaussian depth asks a new question: if evaluation locations are free, can two points reproduce more polynomial moments? Symmetry on `[-1,1]` gives weights 1 and nodes `±1/√3`, exact through degree three; map to a general interval and test a fourth-degree failure. This adds an explanation of node choice without a catalogue of every quadrature family.

Return to the synthetic pump. Let `t` denote the numerical time in minutes, with rate `r(t)=t*exp(-t)` litres/minute; equivalently the exponential uses `t/(one minute)` and the scale factors carry units. Its accumulated volume is `V(T)=1-(1+T)exp(-T)` litres. Locate 0.8 litres on `[2,4]`: the analytic reference root is approximately `2.99430834700233` minutes. The analytic primitive is a validation oracle for the teaching experiment; the numerical workflow still evaluates an integral.

On `[2,4]`, the derivative of volume is at least `m=4exp(-4)` litres/minute. For an approximation at a candidate `T`, a known integral error bound `εI` gives the conditional location bound `(|Ihat(T)-0.8|+εI)/m`, provided the candidate and true root are in that interval. For the composite trapezoid on `[0,T]`, `|r''|≤2`, giving a mathematical discretization bound `T^3/(6n^2)`. State evaluation/arithmetic error separately; do not call this a rigorously rounded enclosure by default. If an integral interval straddles the target, refine it or report unresolved sign rather than discarding half of a root bracket. This application makes nested error budgets affect an actual decision.

### 8. Transfer the error discipline

Keep the original linear-solver bridge: solve via a suitable factorization rather than explicitly forming an inverse; CG has symmetry/positive-definiteness assumptions and preconditioning alters practical convergence. Use a tiny diagonal example to show a small residual with a large error in a weakly scaled direction. The existing Decompositions lesson owns the full mechanism.

Adapt the incoming Dynamics note as a short same-physical-horizon cooling comparison. Distinguish one-step defect from accumulated error, and numerical growth from true decay. Link the already worked oscillator where geometric fidelity is the issue: forward Euler multiplies its energy by `1+h^2`, while kick-drift symplectic Euler preserves a different positive quadratic only for `|h|<2`. Do not relabel it exact physical-energy conservation. `solve_ivp` local tolerances and event detection are limited contracts; `t_eval` chooses reported times, not necessarily integration steps. Full ODE/PDE instruction remains with its owners.

Close with a task-based method choice and evidence report: target/units, assumptions, approximation/error source, independent comparison, diagnostics and remaining uncertainty. Bridge honestly to the next actual topic, Functional Analysis & RKHS: derivatives and integrals can be viewed as operators on functions, with the space and norm determining what continuity/error means. This is motivation, not a hidden functional-analysis prerequisite.

## Representation contracts

Every graph below is calculated from the named fixture or a labelled error model. No timing benchmark or measured pump performance is proposed. Color supplements labels; it does not carry correctness alone. Provide concise textual state descriptions, readable units/ticks, keyboard controls and a narrow layout preserving the mathematical relation. Inspect each initial view during ordinary reading, not only after manipulating it.

| Placement and question | Initial objects and encoding | Learner operation, held fixed and feedback | Validation and mobile contract |
| --- | --- | --- | --- |
| Opening: how are a rate, a total and a threshold related? | Aligned rate/volume axes for the same pump, with a labelled area and vertical threshold time. Actual coordinate values; no pictorial area used as a numerical bound. | Static first view; a short caption maps slope, area and crossing before terminology. | Check primitive, units and alignment independently. Stack axes vertically on narrow screens with a shared time marker. |
| Bisection: what remains known after one evaluation? | Function curve with signed endpoints and an interval-nesting strip. Initial square-root bracket and a visible width/midpoint bound. | Predict retained half, then evaluate; back/next reconstruct the same trace, reset restores input. Choosing discontinuity/touching-root cases visibly changes what can be concluded. | Independently check every sign, retained bracket and iteration count. Strips fit the viewport; text identifies endpoints without color. |
| Newton: why can a good local line give a bad global move? | Actual tangent, intersection and retained bracket; square-root success and the exact cubic cycle. | Change start/method, hold function fixed; step displays slope, proposed intercept and accept/reject reason. Stop states and history stay visible. | Verify tangent algebra, cycle and safeguard width. Limit plot extents transparently if a proposed step leaves view; never clip failure into apparent convergence. |
| Derivatives: why does moving the points closer stop helping? | Nearby function samples, secant/tangent and a separate log error plot from the active computed sweep. | Change `h`, stencil and a clearly labelled evaluation-error scenario. Hold target location fixed; show value subtraction and division, plus representability failures. | High-precision/analytic comparisons distinguish truncation from runtime error. Zero error gets an explicit special marker, not log(0). Geometry and error panel stack on phones. |
| Quadrature: which shape is actually being integrated? | Function, sample nodes and trapezoid/parabolic interpolants on the same interval; chosen polynomial first. | Change panel count/method or fixed-data mode; show weights, contributions, evaluation count and independently known reference. | Exact moment/area tests, parity/spacing guards. Fit small panels directly; no oversized canvas that hides endpoints. |
| Adaptivity: where did the evaluation budget go? | Active interval partition and sampled nodes beside local estimated-error/budget rows; visible initial five samples. | Refine an eligible interval or run to budget; compare a peak with the blind polynomial. Reset and input edits clear stale partitions. | Check conservation of child budgets/value sums and actual sample cache. Extra drawing samples remain outside algorithm state. On mobile partition and table retain common interval labels. |
| Nested calibration: when is an approximate integral enough to decide a time? | Volume crossing and a vertical integral uncertainty range mapped through a stated slope bound into time uncertainty. | Change target, quadrature resolution and requested time accuracy. Hold rate model fixed; label resolved/uncertain sign and unmet accuracy separately from solver completion. | Check analytic primitive and a posteriori bounds on changed inputs. Describe every interval in text; do not imply a probabilistic confidence interval. |
| Optional node matching and ODE bridge | Small two-node moment diagram; same-horizon cooling comparison if existing linked representation alone is insufficient. | Static calculated explanations are sufficient unless a real learner question calls for controls. | Check exact moments and exact cooling solutions. No additional lab merely to meet a count. |

## Complete programs and independent practice

Plan semantic `numerical-methods-examples.js`, `numerical-methods-models.js`, `NumericalMethodsLabs.jsx` and `numerical-methods-labs.css`, imported only by this lesson; direct `Math.jsx` imports when needed. Existing global components supply article primitives, not numerical algorithms. Use explicit variables such as `bracketWidth`, `estimatedIntegralError` and `evaluationCount`; short mathematical indices are fine where local meanings are clear.

The program set should cover the preserved combined original; bracket/status handling; Newton versus safeguarded failure; derivative stencils/error sweeps; callable and sampled quadrature; adaptive budget/limits; the exact blind spot; Gaussian moment matching; nested calibration; and a small transfer example where it adds independent understanding. Combine only when that improves the flow. These are mechanism obligations, not a predetermined program count. Each standalone program must include imports/input/setup, visible question before the code, actual stdout and interpreted results, plus a changed-input task. Store no setup or teaching prompt solely in a property that the shared renderer ignores. Show pip/environment instructions only where SciPy is first needed; core programs should work with Python's standard library when practical.

Final practice should mix hand reasoning, diagnosis and implementation, with a separate optional hint before each substantial worked solution. Intended independent cases include:

- Repair an off-by-one bisection accuracy claim; distinguish a discontinuous sign change from a missing touching-root bracket.
- Predict the cubic Newton cycle and show why the safeguarded step changes it; contrast a multiple root's convergence.
- Derive a one-sided stencil from polynomial conditions and choose useful perturbations for changed scales, including a near-zero true derivative.
- Integrate changed unequal sample data, diagnose a Simpson parity assumption and compare an actual error with a conditional bound.
- Prove the blind polynomial is invisible at the initial nodes, calculate its positive integral and explain why a zero estimator is insufficient.
- Resolve an endpoint singularity by substitution and quantify the area lost by arbitrary truncation.
- Calculate the two-node Gaussian rule's fourth-degree error, rather than only repeat an exact cubic case.
- Choose inner and outer accuracy for a changed pump target; report an uncertain sign honestly and explain the slope's effect on time accuracy.
- Compare numerical evolution at equal physical times and explain a small linear residual with a large solution error.

Solutions must show intermediate quantities, hypotheses, units and an accepted changed program/output where the task asks for code. Reattempt advice asks the learner to reproduce the mechanism with changed inputs and explain why a method could fail, rather than memorize an API name. External resources complement these self-contained tasks.

## Research and alternate learning resources

Inspected 10 September 2026 UTC (11 September local). URLs and scope below are evidence for specific claims, not an endorsement of every page sentence. Library documentation reported 1.18.0 while the actual design runtime has SciPy 1.18.1; record the implementation runtime separately and recheck affected API behavior. Do not copy textbook prose or its complete example structure.

| Source and inspected scope | Use and learner-facing annotation |
| --- | --- |
| Driscoll & Braun, [Newton's method](https://fncbook.com/python/newton/), section 4.3 linearization/convergence and stopping discussion. | Written alternate after tangent geometry: local convergence and implementation choices. The lesson supplies its own failure fixture and safeguarded algorithm. |
| Driscoll & Braun, [Convergence of finite differences](https://fncbook.com/python/fd-converge/), sections 5.5.1–5.5.2; [NIST DLMF 3.4](https://dlmf.nist.gov/3.4), interpolation-based differentiation and errors. | Taylor/cancellation reference and optional stencil depth. Balance constants and assumptions locally rather than prescribe a universal machine-epsilon step. |
| [NIST DLMF 3.5](https://dlmf.nist.gov/3.5), composite trapezoid/Simpson formulas and hypotheses; Gauss/Legendre nodes and weights. | Precise formula reference after the geometric derivation. Higher-order and Gaussian branches are optional, not prerequisites for the core. |
| Driscoll & Braun, [Adaptive integration](https://fncbook.com/python/adaptive/), section 5.7 derivation, algorithm and limits. | Another worked explanation of local refinement. Our program explicitly divides child budgets; this does not turn its error estimator into a universal bound. |
| [SciPy brentq](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html) and [newton](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html), requirements, tolerances, return/status and stopping notes. | Practical library references after implementing the simple methods. Inspect residual and assumptions as well as convergence status. |
| [SciPy quad](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.quad.html), error estimate, tolerances, breakpoints, diagnostics and QUADPACK dispatch; [simpson](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.simpson.html), sampled-data spacing/exactness. | Choose between a callable and measurements, and learn what the returned error means. Do not conflate `quad` with the displayed Simpson recursion. |
| [SciPy solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html), local error control, `t_eval`, `max_step` and events. | Optional ODE workflow bridge linked to the existing Dynamics lesson. Local tolerance is not a universal global-trajectory certificate. |
| MIT OCW, Srini Devadas, [6.006 Lecture 12: Square Roots, Newton's Method](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/resources/lecture-12-square-roots-newtons-method/), official video page and [accompanying notes](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/d16678659cb7c98d611f12e7c1f7298d_MIT6_006F11_lec12.pdf), pages 1–3. | A spoken alternate plus notes for square-root refinement, useful after section 3. Page identity and notes were inspected; full video viewing or a verified timestamp is not claimed. High-precision arithmetic complexity is optional follow-on depth. |

The three pump views are an original synthetic worked application; the exact blind-spot polynomial is an original finite-sampling counterexample, not a claim about a measured SciPy failure. During design, Fraction coefficient integration verified its area `5/1419264`, and SciPy `brentq` checked the pump target near `2.9943083470023306`. These are feasibility checks, not substitutes for the implementation's independent native/model suite.

## Verification and completion plan

Independent checks should exercise contracts, not repeat the same implementation with renamed variables. Use exact Fraction polynomial moments for stencils/quadrature, analytic primitives, high-precision or independent root references, literal Newton-cycle arithmetic and changed brackets/targets. Check derivative error against a high-precision or analytic reference; actual Python and JavaScript last-bit sweeps need not have an identical best step. Compare quadrature against exact integrals and a separate SciPy method, including the blind spot, nonsmooth/singular cases and explicit nonconvergence. Test the sampled-data distinction, parity, tiny intervals, large offsets, nonfinite evaluations and exhausted budgets.

For nested calibration, independently verify the derivative bound and analytic primitive before claiming a location bound; include uncertainty that prevents a sign decision. The pump's second derivative bound is **2**, giving `T^3/(6n^2)`, not `T^3/(12n^2)`. Test literal displayed Python helpers on changed inputs, not only a JavaScript analogue. Verify the original complete program/output remains byte-conserved. New code's expected stdout must come from actual execution; retain environments and hashes.

Actual browser review must run with the website's fonts at 1440, 390 and 320 pixels: ordinary reading, each initial inline figure, math and code wrapping, exact question-before-program placement, route/section anchors, mouse and keyboard, reset/back/input changes, invalid states, accessible descriptions and reference links. Open the generated screenshots and inspect geometry, labels and text flow; bounding-box counts alone cannot establish good teaching. Use the shared Vite server rather than launch a duplicate. Root owns integrated publication/loading/build checks; no source is ready merely because an individual brief has been registered.

Before freeze, reassess the title, every outcome, incoming-note disposition and any new discovery. Resolve the Dynamics note with links to actual implementation/evidence; keep the Backpropagation note open until that owner handles it. Save a topic verification record with exact production hashes and separate author, independent, integrated and user review status. User acceptance and universal numerical reliability must not be inferred from finite automated checks.

## Implemented disposition

The approved title, identity and module sequence are retained. The full rewrite now develops every core mechanism and the optional two-node Gaussian derivation, including a complete outer search that refines the inner integral or returns unresolved sign. There are six distinct investigations, an immediately visible rate/volume figure, thirteen complete executed programs (including the unchanged original and one changed-task solution), and ten substantial practice groups with separate hints and explained solutions. Counts describe this implementation; they are not quotas for other lessons.

The representations were adapted after actual reading: Newton uses genuine line clipping instead of flattening a tangent onto a plot boundary; the derivative investigation shows the computed local slope/curvature alongside the analytic local model, with a range fitted to the full fixed sweep; quadrature shows the interpolant's shaded area; the blind polynomial uses explicitly scaled vertical coordinates while retaining unscaled area readouts. A static Gaussian node illustration and a duplicate oscillator lab were unnecessary because the short moment derivation and existing Dynamics visual already support those bridges. These are reasoned representation choices, not omitted core mechanisms.

The original suggested linear/AD/ODE bridges remain in section 8. The incoming Dynamics note is adapted and resolved against actual code/algebra evidence. The discovered Backpropagation overclaim is saved in its canonical destination note and remains open for that author, with no unrequested body edit. No full curriculum audit, universal numerical certification, production integration pass or user acceptance is claimed by this author record.
