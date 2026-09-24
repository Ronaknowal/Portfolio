# Non-Convex Optimization Landscape: teaching design

Stable ID: `non-convex-optimization-landscape`. Mathematics position 13; title retained because the expanded material explains landscape interpretation rather than introducing a different subject. Authoring started 10 September 2026. Exact inventory command was run; no incoming destination note existed. Publication already exists and does not imply review.

## Contract and preserved coverage

The original 55-line lesson introduced nonlinear networks, minimum/maximum/saddle classification, the example x²−y², SGD noise, permutation/scaling symmetries and training diagnostics. All are retained and developed. Its isolated NumPy fragment lacked an import; replacements are standalone. The first-time learner can read a gradient and Hessian from Multivariate Calculus, eigenvalue signs from Eigenvalues, and gradient updates from Gradient Descent Variants. Local refreshers supply Taylor remainders, independent signs and simple ReLU networks. Second-Order Methods precedes this page; Constrained & Multi-Objective Optimization follows it.

Finish line: distinguish local/global and strict/non-strict extrema; apply the C² interior-point second-order tests without misclassifying zero eigenvalues; predict stable/unstable saddle directions and noise support; exhibit exact parameter symmetries and distinguish parameter curvature from behavior; interpret paths/slices without universal claims; diagnose a stalled toy training run and separate training fit from held-out performance.

## Scope and ownership

| Idea | Evidence / owner | Decision |
| --- | --- | --- |
| Multiple wells and stationary classifications | Original sparse discussion; current topic | Develop an exact tilted quartic and six origin geometries. Keep actual recurrence separate from landscape. |
| Local indefinite Newton, damping, HVP/CG | Parent authors preceding Second-Order Methods | Bridge, do not repeat its optimizer catalogue. |
| Anisotropic sample noise, exact stable subspace | Original unqualified SGD paragraph | Derive a two-sample loss and repeatable batch realization; optional second-moment calculation. |
| Scaling, permutation, reparameterized curvature | Original brief symmetry mention | Exact two-factor and biased ReLU examples; demonstrate prediction invariance and limits. |
| Mode paths, disconnected minima, high-dimensional slice limits | Valuable closely related gap | Original small counterexamples with proofs; no universal neural-network connectivity theorem. |
| Training fit versus generalization | Original practical observation | Fully specified interpolation family; no sharpness/generalization guarantee. Broader statistical bounds belong to the existing Rademacher/generalization topic. |
| Regularization and symmetry | Existing Regularization lesson is best full owner | Explain that adding parameter penalties changes the objective; route the exact scaling example if useful. |

No full Morse theory, neural loss-surface census, optimizer catalogue, SAM recipe or generalization-bound proof is promised. Mathematical loss surfaces use dimensionless coordinates and cost; physical coordinates would require declared scales/metric. Scope was reconsidered during writing; no rename was warranted. The exact factor-penalty example is saved in [the Regularization destination note](topic-notes/regularization-l1-l2-elastic-net-dropout.md); its body was not changed.

## Hurdles, examples and representations

| Hurdle | Mechanism and complete example | Representation / learner evidence |
| --- | --- | --- |
| Local is not global | Wδ=(x²−1)²/4+δx; classify all three roots and compare values | WellBasinLab: actual calculated curve, critical points and fixed-algorithm GD trajectory; change tilt/start/rate and predict endpoint. |
| Zero gradient and singular Hessian | Bowl, cap, quadratic saddle, quartic minimum, quartic saddle, flat valley | StationaryGeometryLab: sampled signed-value map and exact directional slice beside Hessian/Taylor prediction. Zero Hessian yields incompatible classifications. |
| Noise must reach an unstable direction | F=x²−y²; losses F±a v·θ, explicit finite sign realization | SaddleNoiseLab: parameter trajectory plus coordinate histories; stable-axis versus unstable-axis noise; no clipping or fake random draw. |
| Equivalent model, different coordinate curvature | F(a,b)=½(ab−1)², (a,b)=(s,1/s), eigenvalues 0 and s²+s⁻² | ReparameterizationLab: unit normal/tangent perturbation profile and invariant prediction coefficient; distinguish finite tangent from curved symmetry. |
| Symmetry/path/readout is not a metaphor | Two biased ReLU units, exact permutation; hyperbola path versus chord; h_a(x)=x+a(x²−1) | Inline same-input network correspondence, calculated mode-path geometry and loss, calculated input/output interpolation plot. Each adjacent to its derivation. |
| A flat trace has competing explanations | Inactive ReLU: zero gradient at positive loss; active initialization progresses | Complete logged example; changed diagnostic scenarios and evidence needed before action. |

Interactive changes apply immediately and reset trace when inputs change. Each lab has a prediction, units, generator, exact readout, accessible text, keyboard controls and changed-input transfer. All curves are analytic samples or explicit finite recurrences, never empirical deep-learning benchmarks. Plots use labelled domains/ticks; two-dimensional slices do not certify the hidden directions. Signed maps state grid sampling rather than pretending to be exact contours. Narrow layout stacks paired views while preserving readable axes; inspect ordinary reading separately from controls.

## Practice and implementation evidence plan

Complete standalone NumPy programs cover roots, Hessian classifications, saddle recurrence, exact sign-distribution enumeration, factor curvature/perturbations, ReLU symmetry, mode paths, held-out interpolation, inactive-unit diagnosis and a hidden saddle direction. Python is a verification aid, not a prerequisite for the proofs. Changed-input tasks ask for independent classifications, recurrences, symmetry repairs, path barriers, and falsification of a generalization claim. Each has a hint and explained acceptance conditions.

Use independent SciPy root/optimization oracles, NumPy eigenvalues, finite differences away from kinks, arbitrary changed-input identities, exhaustive small sign sequences and direct sampled model-function evaluation. Browser checks: all controls, resets/edges, keyboard, anchors, full runnable code/stdout, expanded solutions, math rendering, overflow and actual screenshots at 1440/390 with narrower reading checks where needed. Do not infer author or user review from counts.

## Research ledger (in progress)

Retrieved 10 September 2026. Primary Dinh et al. ICML 2017 paper: definitions of parameter flatness and sections 3–4 positive-homogeneity transformations, including theorem 4's Hessian claim, inspected; the lesson uses independently derived two-scalar calculations, not copied figures. URL https://proceedings.mlr.press/v70/dinh17b/dinh17b.pdf, pages 2–5 (PDF numbering 1–4). Resource annotation explains advanced level and scope.

Primary Jin et al. ICML 2017 paper https://proceedings.mlr.press/v70/jin17a/jin17a.pdf: assumption A1, theorem 3, section 3.1's A2/corollary 4 and the stated algorithm qualification inspected on PDF pages 4–6. The lesson states Lipschitz gradient/Hessian, finite objective gap, specified perturbed algorithm and approximate second-order/high-probability result; it does not extend this to arbitrary SGD or the unbounded local saddle toy. No current best-rate or universal deep-network guarantee is claimed.

MIT OCW 18.02 Lecture 10 official recording page https://ocw.mit.edu/courses/18-02-multivariable-calculus-fall-2007/resources/lecture-10-second-derivative-test/ embeds https://www.youtube.com/watch?v=3_goGnJm5sA. Its linked transcript https://ocw.mit.edu/courses/18-02-multivariable-calculus-fall-2007/9c2738f5bbc2c2914fd90471f7792116_3_goGnJm5sA.pdf was inspected, particularly page 1 critical/global candidate comparison and pages 1–2 completion of squares/degenerate higher-order caveat. This verifies the video’s teaching fit using substantive transcript content. Full recording playback was not performed. The separate MIT CBMM tutorial page timed out twice and was not recommended. Native numeric APIs were actually executed in NumPy 2.3.5; no changing optimizer API claim is introduced.
