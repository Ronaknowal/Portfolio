# Differential Geometry & Riemannian Manifolds — author verification

Mathematics Foundations position42, stable ID `differential-geometry-riemannian-manifolds`. The complete scoped rewrite is implemented. Its final source freeze and actual evidence are in [the author packet](evidence/differential-geometry-author-review.json). Independent review, integrated production checks and user acceptance remain separate; parent owns those shared stages.

## Learning outcome and preservation

The old lesson provided a useful sphere update but largely named geometric tools without the coordinate, differential, connection and curvature mechanisms needed to connect them. The new route starts with circle charts, constructs allowed velocities and local measurement, derives a metric gradient, accumulates length and area, compares sphere maps, differentiates a moving basis, transports an arrow, computes curvature and finishes a checked sphere optimization.

The exact original seven-section source remains in [its archive](evidence/differential-geometry-original-content.json), baseline SHA256 `4f9a22535e04bf93314583373a2dcde737579ea8a54e5af2426b4bc3ed07989c`. Its complete code and displayed output are preserved byte-for-byte as template-cooked strings. Actual rerun output: `(0.0, 2.0)`, `(0.981, -0.196)`, `1.571`. The last value is explained as the separate orthogonal-endpoint calculation, not the optimizer step's angle.

Useful original applications are retained and developed: normalized retrieval, intrinsic/extrinsic direction means, covariance geometry, natural-gradient/Fisher geometry, hyperbolic distance, and modeling/conditioning/downstream-validation limits. Rotation states and geometric deep learning remain scoped connections. Momentum now has a concrete transport rationale. Title, stable identity and actual module order are unchanged; the next entry remains Algebra, Functions, Exponentials & Logarithms.

## Concept-specific visual decisions

Ten forms serve different jobs: seven investigations and three inline figures. Counts describe implementation, not a template.

| Actual representation | Learning job and model contract |
| --- | --- |
| Circle atlas | One actual point linked to two open coordinate strips. Exact seams0/180/360 make a chart unavailable without removing the point; overlap changes are explicit. |
| Tangent sphere patch | Computed orthographic point, coordinate tangent arrows and radial reference. The vectors and unequal lengths are printed. Projection is disclosed. |
| Metric/differential lab | Metric-unit ellipses and selected/steepest directions in coordinate and physical views. Shear leaves the physical gradient unchanged; physical cost changes it. Pairings and metric matrices accompany the views. A redundant family of level-set lines was unnecessary; its equation is stated while unit directions carry the visual job. |
| Sphere paths | Faithful great-circle cross-section with short/long arcs and off-sphere chord, linked progress points, lengths and explicit coincident/antipodal states. Figure rescaling is disclosed. |
| Sphere bands | Exact-area strips for equal polar-angle widths, calculated by integrating sinθ. A second wireframe would duplicate the patch; area strips expose the missing measurement. |
| Tangent update | Actual tangent candidate, raw ambient step, normalized endpoint and Exp endpoint. Adaptive drawing scale includes long steps; exact readouts distinguish near-overlapping small-step endpoints. |
| Polar connection | Moving basis arrows on a straight Cartesian path, with coordinate second derivatives and cancelling connection contributions. |
| Parallel transport | State-derived sphere loop and transported arrow, plus an undistorted tangent-plane endpoint compass. Wedge, arrow, orientation, progress and radius have separate meanings. Mid-route and final views are distinguished. |
| Curvature comparison | Calculated geodesic circumference/area at the same radius in flat, cylindrical, spherical and hyperbolic geometry. Shared-scale bars make the comparison directly; disk/cut restrictions are stated. |
| SPD midpoint | Static contours for the same input pair and two metrics. Axes encode square roots of variances. Unit Mahalanobis contours are not mislabeled confidence regions. |

The browser model returns recursively frozen snapshots and rejects invalid bounds. UI controls are finite bounded ranges with deterministic resets, no timers or hidden service. Unit/tangent tolerances cover binary64 roundoff. Shortest Log and transport reject documented unresolved near-antipodal regions rather than invent directions.

## Mathematical and numerical checks

Commands from the repository root:

- `scratch/lesson-tools/Scripts/python.exe -X utf8 -I scripts/check-differential-geometry-original.py`
- `scratch/lesson-tools/Scripts/python.exe -X utf8 -I scripts/build-differential-geometry-examples.py`
- `scratch/lesson-tools/Scripts/python.exe -X utf8 -I scripts/verify-differential-geometry-models.py`
- `scratch/lesson-tools/Scripts/python.exe -X utf8 -I scripts/verify-differential-geometry-native.py`

The builder executes each whole program and saves actual stdout. The independent native verifier executes all14 displayed programs again, checks exact stdout and preservation, then calls their actual helpers with changed inputs.

Pure-model evidence: `scratch/differential-geometry-verification/model-results.json`, embedded in the durable packet. Passed44,059 scalar comparisons:73 chart positions,100 metric states,108 sphere patches,40 map cases,35 steps,28 polar states,144 transport states,20 curvature states and15 covariance paths. The21 listed invalid-input cases, separate strict tangent-boundary checks and recursive immutability passed. Maximum absolute discrepancy was7.12×10⁻¹¹ within explicitly chosen independent-ODE tolerances, not a universal accuracy guarantee.

The references differ meaningfully from the implementation:

- NumPy transformations/linear solves recover metric, differential and gradient and check unit ellipses/pairings.
- A skew-matrix exponential independently produces the sphere map;80-digit mpmath checks cancellation-sensitive small angles.
- SymPy differentiates the polar path and constructs all connection coefficients before forming curvature. The declared convention gives plane/cylinder0, sphere+1/R² and hyperbolic−1/R².
- SciPy integrates the ambient geodesic and transport ODE together for144 loop states. The expected arrow is not obtained by replaying the endpoint formula.
- An independent Jacobi ODE and quadrature give circumference/area; spectral matrix functions check covariance paths.

Native evidence: `scratch/differential-geometry-verification/native-results.json`, embedded in the packet. Passed3,461 comparisons,96 changed sphere-map cases,20 additional transport ODE cases,72 noncommuting general-SPD paths,30 rotated optimization cases,90 directional finite-difference steps, four new symbolic curvature functions, two changed Laplacian eigenfunctions, all11 actual changed questions and12 rejection cases. General SPD paths use SciPy matrix logarithm/exponential references and congruence invariance. Polar second derivatives use Richardson-extrapolated differences; the maximum7.94×10⁻⁸ discrepancy is within their explicit2×10⁻⁷ finite-difference tolerance.

Two verifier issues were corrected without changing the model: a nominal norm8 from floating-point normalization could round outside the strict contract, so rotated fixtures use7.5 and the exact axis-aligned8 boundary is checked separately; a basic polar difference stencil's truncation exceeded its tolerance, so the reference now uses Richardson extrapolation. The actual checks were rerun.

The final capstone prints objective/reference, status, gradient norm, feasibility residual and monotonicity. The first fixture meets10⁻⁸ tolerance with gradient norm about8.641×10⁻⁹. The changed rotated fixture reaches objective2 at printed precision but has gradient norm3.050×10⁻⁸ and reports a line-search limit. This finite-precision limitation stays explicit. Changed practice uses a justified10⁻⁶ tolerance and contrasts a stationary maximum. No global optimizer guarantee is inferred from a small residual or successful fixture.

## Actual browser and reading review

Commands:

- `node scripts/review-differential-geometry-lesson.cjs`
- `node scripts/review-differential-geometry-lesson.cjs --reading-only`
- `node scripts/review-differential-geometry-final.cjs`

Playwright used installed Edge against root-owned Vite at127.0.0.1:5173, at1440/390/320 widths. Public font loading was authorized and Space Grotesk loaded. Pages blocked only the intentional Vite HMR WebSocket; actual failed requests and runtime errors were empty. No content/source request was mocked.

The full behavior run passed343 states per width: changed/extreme ranges, chart seams, metric invariance, coincident/antipodal paths, zero/large updates, polar cancellation,180 combined transport states and reversal, curvature ordering, all11 hint/solution pairs and real keyboard activation/range adjustment. All14 program questions/code/stdout matched the executed fixtures. All11 reading anchors resolved. Controls were at least44 CSS pixels tall and there was no document horizontal overflow.

Opened screenshots found two crowded labels: atlas seam labels moved away from axis values, and the candidate annotation became `x+v` below the marker. The full run then found four equations overflowing only at320px. They were split into shorter lines, retaining every term/sign. A targeted reading run confirms all equations fit all three widths. Ordinary-reading screenshots and all ten default forms were captured separately from changed states.

A final output amendment prints capstone residuals, then the final three-width check revalidates all14 programs/questions and their interpretation, equations, fonts, keyboard controls and review links. A cross-product/area bridge is explicit before the first program using it. Exact timestamps, hashes and actually opened final images are in the packet. Earlier behavior and later targeted text/output amendments are distinguished, not described as one unchanged-source test.

## Research, scope and limits

The [approved design](DIFFERENTIAL-GEOMETRY-LESSON-DESIGN.md) records inspected primary sections and resource metadata: Boumal3.7–3.8,4.5,7.2,10.2–10.3,11.7; Tong2.1–2.3,3.2–3.3. Full Christoffel/Riemann signs were reproduced symbolically. The official MIT transport video and Boumal course page are annotated alternatives; only actually inspected descriptions/links are claimed, not unperformed full-video viewing. MIT18.950 is correctly a notes/problem-set course. Research informed original explanations and changed examples.

The [Tensor Algebra note](topic-notes/differential-geometry-riemannian-manifolds.md) is resolved with exact conventions and source/evidence links. Existing specialist owners retain full rotation/group algorithms, geometric architectures, manifold-learning validation, general relativity, exterior calculus and geometric PDE proofs. This end-to-end introductory workflow does not claim to replace those courses or contain every theorem. No unrelated lesson, module membership, stable identity or sequence was changed.

Shared curriculum/build/loading checks and independent review are parent-owned. Author verification covers the finite computational/browser scopes above; learner studies and user approval have not been performed.
