# Convex Optimization — implementation verification

10 September2026. Complete rewrite of mathematics position8, stable ID `convex-optimization`. The original publication mapping and module order are unchanged. [Design and primary-source ledger](CONVEX-OPTIMIZATION-DESIGN.md); [independent mathematical review](CONVEX-OPTIMIZATION-INDEPENDENT-REVIEW.md). This is author/integration evidence, not user approval or an observed learner study.

## Complete teaching and ownership

The lesson develops decisions and feasible sets, function chords and tangent bounds, the local-to-global proof, existence/strict/strong convexity, constrained first-order certificates, composition and epigraph modeling, ridge curvature, nonsmooth subgradients and proximal steps. It closes with actual solver checks and six independent practice groups with hints and explained certificates. The retained original ridge input/output is directly verified; the title remains appropriate. Gradient Descent Variants is the next actual module topic. General duality and constrained splitting discoveries are saved in their destination notes rather than treated as assumed knowledge.

Body, blueprint, examples, pure models, `ConvexOptimizationLabs.jsx` and its CSS have semantic individual ownership. Browser models perform bounded formula calculations only. NumPy/CVXPY and the isolated solver runtime are verification dependencies, not browser imports. All numerical figures derive from explicit formulas/fixtures; none is an empirical timing chart.

## Native and independent mathematical evidence

Commands from the repository:

```text
node scripts/verify-convex-optimization-models.mjs
scratch/lesson-tools/Scripts/python.exe scripts/verify-convex-optimization-native.py
node scripts/verify-convex-optimization-examples.mjs
scratch/lesson-tools/Scripts/python.exe scripts/verify-convex-optimization-practice.py
```

- Nine complete programs ran separately and every expected stdout matched: analytic allocation, original ridge, full/duplicate curvature steps, signed thresholding, proximal lasso, CVXPY allocation, DCP/status cases, robust epigraph and total variation. Runtime: Python3.12.14, NumPy2.3.5, CVXPY1.9.2, SciPy1.18.1, Clarabel0.11.1. Solver versions/tolerances can change final digits; the programs check status and original quantities as well as printing results.
- Model coverage:140 chord cases,378 allocations,70 ridge configurations with25 frames and97 points per contour,315 threshold cases,12 rejected contracts and8 additional precision regressions. Native independent geometry compares all three allocation edges, NumPy eigenvalues/least squares and matrix powers, direct contour quadratic forms and scalar half-line minima. Final maximum contour residual was about1.07e-14.
- Independent practice checks use weighted allocation and its supporting bound,75 matrix-power comparisons,45 weighted threshold cases, two changed robust-model solutions and two exact rational total-variation certificates, with separate CVXPY comparisons. Exact total-variation costs are137/150 and15/8.
- Cross-review separately verifies the ridge coefficients38/41 and24/41 and objective35/41, the exact TV subgradients, and6,390 rational feasible weighted-allocation points. Finite checks support implementations; the lesson's proofs establish general statements under their written hypotheses.
- The independent review found and rechecked tiny-positive-penalty uniqueness/cancellation, tolerance-based feasibility and exact subgradient-membership issues. Preserved before/after JSON and the durable reproducer are linked in that review. Rejected nonfinite geometry cannot silently reach SVG paths.

Result files are under `scratch/convex-optimization-review/`: `example-results.json`, `model-fixtures.json`, `practice-results.json` and independently executed programs. The cross-review has durable evidence under `docs/teaching/evidence/`.

## Actual browser, reading and opened-image review

```text
node scripts/review-convex-optimization-lesson.cjs
node scripts/review-convex-optimization-reading.cjs
```

Final full interaction pass: **10:38:07UTC**, after the model corrections. Reading pass: **10:32:21UTC**; the subsequent model corrections do not change lesson text/equations, and the full final pass checks the resulting rendered lab states. Headless Microsoft Edge at1440×1000,390×1000 and320×1000, actual local route on5173, with HMR closed during review.

- Four investigations work through meaningful control changes: convex/nonconvex chord cases, feasible/infeasible/optimal allocation over budgets including zero, full/duplicate ridge at stable and unstable step sizes with Prev/Next/Run24/Reset, and signed threshold/kink/zero-penalty cases. Keyboard buttons, sliders and disclosures are exercised; reported states are checked against the actual model.
- Three inline figures and nine SVGs have accessible title/description and finite geometry. All13 displayed equations fit320px with optional details open. Every one of the nine rendered full programs and its output matches the executed fixture.
- Eight intro anchors resolve. The separate reading pass activates first/last links using Enter and checks actual destination-heading arrival below the fixed header, beyond checking the hash. Seven ordinary reading sections per width were captured without hiding the navigation; two independent disclosures were toggled with keyboard.
- Eleven reference/alternate links render with direct HTTPS targets, meaningful annotations and new-tab attributes. The first and last independently solved exercise disclosures open. No uncaught page errors or document overflow at any tested width.

Actual opened/read screenshots include the desktop chord counterexample, allocation optimum, ridge flat direction and threshold lab;390px inline feasible-mixing/epigraph/TV figures and threshold;320px chord counterexample, infeasible allocation and ridge flat-direction controls/readouts; ordinary sections1/4/6/7 at320px and the final setup section8 at390px. Individual320px ridge-Hessian, proximal-definition and proximal-gradient equations were opened and are readable. Sources, labels, bounds, clipping descriptions and the original numerical relationships were assessed, not merely counted.

Issues found and resolved: five long math displays were rewritten using equivalent named quantities/short lines; raw JSX comparison text was escaped correctly; the formatting helper preserves JSX/code strings and the PowerShell `.venv\Scripts\Activate.ps1` command; browser anchor review exposed a global smooth-scroll preference gap, repaired by honoring `prefers-reduced-motion` in `src/index.css`. Failed initial harness runs are not represented as passes. Production integration owns the shared-CSS regression and final loading checks.

## Limits and sources

Primary research read selected Stanford/Boyd convexity and proximal sections, Bubeck's relevant smooth/strong-convexity sections, and current CVXPY/Clarabel modeling/status/options documentation; exact scope is in the design ledger. Three Stanford video parts are linked as official alternate learning resources. The index was inspected, but video playback fetches failed; no viewing or full-book reading is claimed.

Small original examples demonstrate mechanisms and certificates. They do not certify generalization, universal solver success, a deployment or a hardware benchmark. Optional general splitting, duality qualifications and stochastic optimizer internals retain their appropriate later owners. The shared production integration and user acceptance remain separate from this completed author review.
