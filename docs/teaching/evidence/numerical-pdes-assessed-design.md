# Numerical PDEs: Grids, Finite Elements & Stability — scoped design

Prepared 11 September 2026. This is a design, not a completed lesson. Full implementation, actual native/browser verification, independent review and production integration remain separate. The [original planned entry and complete inventory](evidence/numerical-pdes-original-plan.json) are preserved before authoring. No old published body or complete program existed for this stable identity.

## Identity, ownership and prerequisites

Keep `numerical-pdes-grids-finite-elements-stability`, the existing title, advanced level and mathematics position57. Retain the recorded prerequisites PDE56, Conditioning52 and Matrix Decompositions. PDE56 reintroduces its required ODE/Fourier calculations; Real Analysis54 is a useful explicit review link for restriction/reconstruction and different convergence quantities, not a new prerequisite cycle. This lesson completes the mathematics module; it must not invent a next mathematics topic or skip to a conveniently published page. The route's actual next/previous controls remain authoritative.

The exact inventory command and all three destination notes were read. The original commitments—finite differences for heat/Poisson, boundary conditions, method-specific stability, weak form and element bases, refinement, and residual versus discretization error—are retained and substantially developed. Numerical PDEs legitimately includes finite-volume conservation, sparse two-dimensional structure and elementary error certification; the title already fits. Do not move GPU kernel design or production parallel solvers here. Locate existing GPU stencil and scientific-computing owners before any outgoing note. General nonlinear PDE existence, full Sobolev theory, every finite-element family, domain decomposition, fluid turbulence, all-purpose adaptive meshing and validated physical models remain deeper branches, not claims of local mastery.

Incoming notes are planned adaptations, not yet resolved:

- PDE56: preserve the continuous balance, source/flux compatibility, weak point-load functional and exact reference problem. Its implementation is underway; inspect the final body and independent record before relying on its publication. Re-derive the local weak identity needed here rather than omit the assembly step behind a link.
- Conditioning52: derive the exact defect/residual identity with one declared equation normalization and an actual inverse bound; changing units cannot create a better certificate.
- Real54: distinguish grid values, reconstructed functions, integrals and elementwise slopes; add interpolation/approximation error where needed. A sampled curve is never silently a supremum theorem. The original author freeze has since passed browser review; independent review repairs only endpoint-display lab wording, not these mathematical claims.

## Finish line and teaching progression

The learner can formulate a small spatial problem, choose and derive a discretization, assemble and solve it, inspect conservation and boundary treatment, report errors in a named quantity, and defend a changed refinement/solver decision. Each mechanism gets a worked hand-sized instance before complete code. Core proofs and acceptance reasoning stay in the main flow; optional depth provides wider methods and sharper qualifications without hiding the basis of the solver.

The following stages are a connected teaching design, not a compulsory heading or lab count. Consolidate where the learner's reasoning remains explicit.

### 1. Decide what the numbers represent

Begin with a rod on0≤x≤1, fixed endpoint temperatures and distributed heating, using the already defined PDE balance. State a normalized steady equation `−u''=f`, boundary values, units and requested output. A solver cannot decide whether the task wants temperature at nodes, the hottest point, total heat, or a boundary flux. Define N intervals, h=1/N, nodes x_j=jh and the N−1 unknown interior values. Distinguish a known boundary from an unknown row. Define restriction R_hu=(u(x_j)) and the piecewise-linear reconstruction I_hU; later cell averages and finite-element coefficients get their own definitions. Use three aligned views—continuous target, stored numbers, reconstructed field—before matrix notation.

### 2. Derive a stencil instead of memorizing it

Use Taylor expansions with a stated C4 bound to derive `(u(x−h)−2u(x)+u(x+h))/h²` and error at most h²||u''''||∞/12. Preserve the sign: the operator for `−u''` has diagonal2/h² and neighbors−1/h². Explain cancellation of odd terms, the h² denominator and the distinction between a derivative approximation at known samples and an equation for unknown samples. A quadratic is exact under this stencil; exactness on that one test does not establish general order. Use a quartic to expose the defect. A local three-node stencil figure should expand into one actual matrix row with boundary values moved to the right-hand side, including a changed nonzero endpoint.

### 3. Assemble, solve and check an actual boundary-value problem

For N4 and f2 with endpoints0, the scaled system T U=h²f has diagonal2/neighbors−1 and solution(3/16,1/4,3/16). Show all three equations, one elimination step and back substitution. Derive a bounded Thomas elimination for strictly diagonally dominant or this SPD tridiagonal family; do not claim the no-pivot algorithm is safe for arbitrary tridiagonal input. Complete program accepts source and both endpoint values, assembles the right side and reports the actual residual under both A=T/h² and T normalizations.

Explain why A is SPD by summation `h UᵀAU=Σ(U_{j+1}−U_j)²/h` with zero endpoints. It implies uniqueness/invertibility; it does not by itself establish a good condition number. Exact sine eigenvectors give smallest eigenvalue approachingπ² and largest of orderh⁻². Reconcile the bounded physical-norm inverse with growing relative condition number. This is a direct application of52 rather than contradictory advice that a finer mesh always makes a solution meaningless.

### 4. Prove what a residual can certify

Define `τ=f_h−A R_hu`, `r=f_h−A Uhat`. Then `A(Uhat−R_hu)=τ−r`. Prove the discrete maximum principle by propagation from a negative interior minimum, then use the barrier g_j=x_j(1−x_j)/2 with Ag=1 to obtain `||A⁻¹||∞≤1/8` on the unit interval. Therefore the nodal error is at most `(||τ||∞+||r||∞)/8`. If equations are scaled toT=h²A, both defect/residual scale byh² and the inverse bound becomes1/(8h²); the final certificate is identical. Explicitly display these two normalizations side by side.

Use `u=x−2x³+x⁴`, `f=12x(1−x)`, homogeneous endpoints. Here τ=2h² exactly and the exact discrete solution exceeds the exact nodal target by `h²x(1−x)`. For even N the nodal max error is h²/4. A changed numerical vector formed by stopping Jacobi early has a separately measured r; a smaller raw scaled residual is not a smaller physical error. The lab should show the reconstructed curves, the actual discrete residual and separate error-budget contributions, not an arbitrary convergence line.

### 5. Say what happens between nodes

Prove `||I_hR_hu−u||∞≤h²||u''||∞/8` on each interval by the interpolation remainder or a barrier argument, stating C2. Since linear interpolation is a convex combination of endpoint errors, `||I_hUhat−u||∞` is bounded by the nodal certificate plus this interpolation term. Derivative/flux convergence needs a separate argument; differentiating a piecewise-linear reconstruction gives elementwise constants with jumps. It does not produce a classical derivative at mesh nodes.

For the quartic fixture ||u''||∞=3, combine h²/4+3h²/8=5h²/8 before algebraic error. The bound is sufficient and conservative. Explain pointwise/grid max, weighted discrete L2 `sqrt(hΣe_j²)`, continuous L2 and energy/gradient errors. Teach an exact local integration for linear or polynomial pieces; if a graph reports a sampled diagnostic, label its finite grid. Connect directly to54's missed triangle rather than assume a dense plot closes the error budget.

### 6. Refine the right things and produce a reproducible report

Use N8/16/32/64 under the same boundary/source/quantity. Compute observed order `log(E_h/E_{h/2})/log2` only for positive comparable errors with a known solution. If no reference is available, a three-grid difference ratio is evidence under an assumed asymptotic error expansion, not proof. Exact-on-grid quadratic data can hide a broken order test. Solver error, roundoff, boundary approximations and quadrature can create a plateau; report them independently. Manufactured solutions mean choosing u first and deriving all forcing/data, not choosing a pretty expected graph afterward.

A capstone starts here and returns at the end: choose N and a normalized residual threshold to certify a reconstructed-field tolerance for the quartic. With target1/1000, N32 gives discretization/reconstruction bound5/(8·32²)=.0006103515625. The remaining budget permits `||r||∞≤8(1/1000−5/(8·32²))=.0031171875` for A, or h² times that forT. Check the arithmetic exactly. A changed tolerance, domain length or source scale must change the certificate; a displayed pass must come from these numbers, not a preset success badge.

### 7. Turn diffusion into coupled ODEs and control time error

For `u_t=αu_xx` with homogeneous Dirichlet endpoints, discretize space first to `U'=−αAU`; this is the method of lines. Explicit Euler gives `U_j^{n+1}=rU_{j−1}^n+(1−2r)U_j^n+rU_{j+1}^n`, r=αΔt/h². When0≤r≤1/2 the weights are nonnegative and their sum is at most1 after zero boundaries, giving maximum-norm contraction and positivity. Derive the sine-mode factor `1−4r sin²(kπ/(2N))`. For a fixed Dirichlet grid the exact spectral stability upper bound is `1/[2cos²(π/(2N))]`; r≤1/2 is the uniform sufficient/monotonicity bound, not falsely the exact finite-grid threshold.

Run actual bounded stencil steps on low and near-grid-scale initial sine modes and compare with their continuous heat solutions at the **same physical time**. Show sign alternation and amplification directly. A stable run can still be inaccurate. A fixed smooth initial field is different from changing its frequency with the mesh. Derive a complete finite-horizon error estimate: a local one-step defect bounded by Δt²||u_tt||∞/2+αΔt h²||u_xxxx||∞/12, summed through contraction, gives O(Δt+h²) with explicitly uniform smoothness bounds and compatible data. No general Lax equivalence theorem is claimed from this one calculation.

### 8. Implicit stability does not erase resolution requirements

Derive backward Euler `(I+αΔtA)Unew=Uold` and Crank–Nicolson `(I+αΔtA/2)Unew=(I−αΔtA/2)Uold`. Reuse a factored tridiagonal operator when h and Δt stay fixed; each step has linear work/storage in this1D family. Show scalar mode factors1/(1+μ) and(1−μ/2)/(1+μ/2), μ=αΔtλ≥0. Backward Euler strongly damps largeμ; Crank–Nicolson tends to−1 and can preserve alternating high-frequency noise despite energy stability. Derive the discrete energy identity by pairing the update with the average. A-stability is not a bound on truncation error or a guarantee of positivity for every time step. Compare methods at a declared final time and separate spatial from temporal convergence. Use actual solves, not just scalar formulas in a component advertised as a solver.

### 9. Preserve a flux before averaging a coefficient

Introduce finite volumes from the integral balance over a cell, distinguish cell average from nodal value, and show cancellation of shared face fluxes when summing cells. For steady `−(ku')'=f`, derive a two-material face flux from two resistances in series: the conductance is1/(d_L/k_L+d_R/k_R). Equal half-widths give a harmonic coefficient, not an arithmetic mean. With left halfk1, right halfk10, boundary temperatures1/0 and f0, total resistance11/20, flux20/11 and interface temperature1/11 are exact chosen teaching values. A mesh aligned with the material interface is different from a coarse element that crosses it; do not claim every P1 coefficient average automatically captures the same exact interface flux.

For Neumann data use the outward convention from56. A discrete conservative operator has a constant nullspace; summed loads must match net flux. Derive the compatibility test and impose a weighted mean only after compatibility holds. Show incompatible input as a valid no-solution result. Robin exchange contributes the appropriate boundary term and bath load; introduce its signs from the continuous balance. A pinned value must not silently “fix” an incompatible physical problem by injecting unreported flux.

### 10. Match the direction of information in transport

For periodic advection with v>0, derive the cell-average upwind flux vU_left and update `(1−c)U_j+cU_{j−1}`, c=vΔt/h. Under0≤c≤1 it preserves mass and is an L1 contraction; prove by triangle inequality and index shift. At c1 it shifts the entire cell-average array one cell exactly. For negative velocity the upstream side reverses. A cell strip with face arrows and the exact update should make this understandable before code.

Contrast centered-space forward Euler with amplification1−ic sinθ, magnitude greater than1 at nontrivial modes; satisfying an information-cone CFL requirement alone is insufficient. Explain both the necessary domain-of-dependence principle and this particular scheme's sufficient monotonicity condition. A bounded smooth-data modified-equation calculation gives leading numerical diffusion v h(1−c)/2 for positive v; label it an asymptotic explanation, not an exact replacement PDE for discontinuities. A square pulse tests mass, positivity and smearing without claiming a smooth-solution order theorem.

An optional nonlinear connection may derive the Burgers Godunov flux from56's entropy Riemann solution: minimum of u²/2 on[uL,uR] for rarefaction orderuL≤uR, maximum of endpoint fluxes for shock order. Check sign-changing rarefaction versus expansion-shock failure. This can be a focused exact face-flux program and diagram rather than another full generic solver. Full high-resolution limiting and all-system entropy convergence remain further study with an authoritative source, not a named-method list pretending to teach them.

### 11. Build finite elements from actual shape functions

Re-derive `a(u,v)=∫ku'v'=∫fv` for zero Dirichlet tests, including how known fluxes contribute at natural boundaries. On one element[a,b], define N0=(b−x)/h_e,N1=(x−a)/h_e and their derivatives. Compute `K_e=(k/h_e)[[1,−1],[−1,1]]` for constant element k; for variable k the integral ofk multiplies1/h_e². With constant source f, local load is f h_e/2(1,1). Show a nonuniform three-element assembly with global index mapping and overlapping contributions. Dirichlet elimination must modify the right side before removing rows/columns; merely replacing rows can destroy symmetry used by a solver.

Explain U_h=ΣU_iφ_i as an actual function; nodal coefficients are values becauseφ_i(x_j)=δ_ij. Derive SPD from energy on the zero-boundary subspace. A point source at a supplies loadφ_i(a), not a finite-height sample of delta. For a1/3 on mesh0,1/4,1/2,3/4,1, the two adjacent nonzero source weights are2/3 and1/3. The exact continuous Green function has a kink there; a mesh node at the source represents it exactly with P1, whereas a crossing element cannot. The precise between-node error makes the benefit of aligning/refining a mesh visible.

### 12. Explain approximation quality in its own norm

Derive Galerkin orthogonality `a(u−U_h,v_h)=0`. For a symmetric coercive form the energy norm gives best approximation: expanding `u−v_h=(u−U_h)+(U_h−v_h)` yields a Pythagorean identity, hence an energy-error minimum. State exact integrals/conforming spaces/exact algebraic solution. Approximate load quadrature and incomplete solves add terms; no blanket claim applies to arbitrary elements or singular/noncoercive problems.

For constantk1 in1D with exact loads, prove U_h equals the nodal interpolant ofu: on each element a test derivative is constant, so integrating the derivative ofu−I_hu gives its zero endpoint difference. This is a useful special fact, not a multidimensional theorem. For u=x(1−x), uniformh: the nodal error is zero, supremum interpolation errorh²/4, continuous L2 errorh²/√30, and derivative-energy errorh/√3. Derive the integrals on one cell. This gives a concrete counterexample to “zero nodal error means exact field or slope.” For a smooth general function, supply a local Cauchy–Schwarz interpolation-gradient bound and the resulting energy estimate; do not import a full general Céa/L2 regularity theorem without explaining its assumptions.

A short adaptive branch compares a point-load kink with a smooth solution. Derive actual piecewise error or local flux jumps and discuss what an estimator would need to certify. Marking the largest indicator is a choice; an unexplained residual color does not guarantee convergence. General reliability/efficiency and shape-regularity theory belongs in annotated deeper resources.

### 13. Go to two dimensions without hiding the storage or geometry

Derive the five-point stencil for `−Δu=f` on a rectangular grid, including hx/hy separately and nonzero boundary contributions. Define a single row-major flattening and show actual nearby coordinates versus their vector indices; do not wrap a right-edge neighbor into the next row. State the sparse nonzero count/order and matrix-free stencil alternative. A complete native sparse solve uses SciPy with explicit setup/version, reports shape/nonzeros/residual and compares to a manufactured smooth target. A sine-product target offers an exact discrete eigenvector and separable error oracle; changing a frequency or unequal spacings is meaningful transfer.

Provide a concrete triangle P1 bridge: reference vertices(0,0),(1,0),(0,1), shape functions1−x−y,x,y, constant gradients and local stiffness `area·Bᵀ k B`. Show the actual3×3 matrix for k1 and explain mapping gradients with an affine Jacobian and |detJ|. A changed scaled/rotated triangle checks geometry. This is a complete single-element calculation plus assembly principle, not a claim to provide an arbitrary-domain mesh generator. Shape quality, quadrature and boundary geometry can alter errors. More elements do not automatically repair a wrong domain or bad aspect ratios.

### 14. Choose a solver and know the limits of the computation

Connect SPD to conjugate-gradient eligibility, symmetric boundary elimination and a residual stopping rule; contrast a direct sparse solve with iterative work and matrix-free products. Derive why unweighted Jacobi smooths different sine modes at different rates, then show a small two-grid error correction only if its restriction/prolongation/coarse operator are all explicitly defined and independently checked. This is optional depth; a labelled damping curve and a worked low-mode coarse correction may teach more than an unfinished “multigrid lab.” Avoid broad runtime rankings, dense matrices for large browser meshes, or O(N) claims for an arbitrary2D sparse factorization with fill.

The main solver program is a hand-understandable Thomas method for the1D SPD problem. The2D program uses a mature sparse routine. The browser uses bounded pure numerical models and controls, no SciPy/Python auto-runtime, external compute or unbounded while loops. A requested large/invalid problem gets a clear supported-range message. Keep numerical state local and compute only the current lesson's data.

### 15. Finish the changed solver report

Return to the quartic capstone: report the physical problem, unknowns, h, stencil, boundary treatment, equation scaling, reconstructed output, residual bound, interpolation bound, observed refinement and exact/reference discrepancy. Use an explicit finite iteration budget; if not certified, show the actual unmet tolerance and stop. Changed practice should require at least one of a nonzero endpoint, source scaling, a tighter tolerance or a different output such as flux. A second thermal capstone may adapt56's forced rod only if it adds independently separated spatial/time/solver evidence; don't duplicate its closed-form settling example as “numerical mastery.”

End with readiness criteria: derive a row and an element, explain finite-volume cancellation, choose a scheme-specific stability condition, identify a changed exact-on-grid trap, and defend an error certificate for a reconstructed quantity. State that this completes the current mathematics module and link appropriate further owners without overriding route order. No finite lesson guarantees knowledge of all PDE algorithms.

## Representation and practice contracts

Use the concept-native form at the point of need. Proposed forms are a restriction/reconstruction inline comparison; a stencil-to-row inspector; a residual/defect/normalization budget with actual curves; a calculated refinement chart; a spatial-mode/time-step investigation; a BE/CN damping comparison; a material-interface resistance/flux diagram; a cell-average flux strip with mass; a hat-function/element assembly explorer; a source-between-nodes zoom; a2D stencil/index map and triangle-gradient diagram. Combine controls when their shared state helps; split when the mechanism is genuinely different. This is not a quota or a request for identical cards.

Every chart states its equation/data provenance, axes, units or normalization, finite horizon and exact-versus-rounded status. Log plots never invent finite positions for zero or negative errors. A theoretical rate guide is labelled theoretical and is not a measured timing benchmark. Measured errors must come from actual exported solver results. Plot ranges must contain valid overshoots/unstable states and expose their changed scale. Inspect phone labels and ordinary reading before interacting; diagrams must help without hover or autoplay.

Practice includes a nonzero boundary row; an exact quadratic that fails to expose truncation; quartic defect/sign derivation; an equivalent scaled residual; a between-node versus nodal error; fixed-h time refinement; a stable but oscillating CN mode; two-material face averaging; incompatible Neumann source; negative-velocity upwind; a moved point load; nonuniform element assembly; energy versus L2 rate; a2D index/anisotropic-spacing repair; and the final changed certificate. Each task has a meaningful hint and full explained solution. Programs are complete imports/input/functions/output/workflow units, not hidden context fragments. Change actual helper inputs in native verification.

## Research actually reviewed and remaining checks

Research on11 September2026 supports original local derivations rather than copied prose. The author must revisit nuanced hypotheses and complete any currently pending source inspection before claiming final verification.

| Source | Actual scope and intended use |
| --- | --- |
| [FNC finite-difference convergence](https://fncbook.com/python/fd-converge/) | Definitions, Taylor truncation/order and actual refinement/roundoff example read. Our full BVP inverse/barrier and reconstruction bound are separately derived; this page alone is not a PDE convergence theorem. |
| [FNC absolute stability for diffusion](https://fncbook.com/python/absstab-diffusion/) | Scalar amplification derivations and periodic heat eigenvalues/step scaling read. The lesson uses its own explicitly Dirichlet spectrum and distinguishes the zero periodic eigenvalue. No universal implicit-method performance claim is imported. |
| [FNC upwinding and stability](https://fncbook.com/upwind/) | Domain-of-dependence definitions, necessary CFL reasoning and its insufficiency caution read; later full upwind derivation remains a final-author source-reading target. Our convex-combination/conservation arguments are given locally. |
| [FNC Laplace/Poisson](https://fncbook.com/laplace/) | Introductory equation, Sylvester/Kronecker mapping and sparse-structure passages read. The reference uses+Δ, while this design uses−Δ; the signs and flattening are derived explicitly. Full chapter not claimed read. |
| [MIT18.085 Lecture17](https://ocw.mit.edu/courses/18-085-computational-science-and-engineering-i-fall-2008/resources/lecture-17-finite-elements-in-1d-part-1/) and [transcript](https://ocw.mit.edu/courses/18-085-computational-science-and-engineering-i-fall-2008/e7883a6431c768890965fb50f7fab2dc_18-085F08-L17.pdf) | Official video/instructor identity checked; transcript pp7–9 weak form, test-boundary restriction and Galerkin coefficients read. The embedded YouTube target failed to open through the tool; retain the public OCW page, no playback claim. |
| [MIT18.085 Lecture18 transcript](https://ocw.mit.edu/courses/18-085-computational-science-and-engineering-i-fall-2008/676ba0a00eef45a6cb3823c4da61ee11_18-085F08-L18.pdf) | Selected pp1–3 hat/coefficient explanation, pp6–7 overlapping slopes and pp8–9 moving point-load weights read. Useful alternate learning route with a different boundary convention. No full-video viewing claimed. |
| [SciPy spsolve](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.spsolve.html) | Actual1.18 documentation: square system, CSC/CSR conversion, vector shapes, ordering and sparse-result qualification read. Final complete program must run against the installed1.18.1 runtime. |

Two guessed UCI PDF paths and the old Clawpack/author-hosted Burgers HTML were inaccessible through browsing; do not cite them as read. The UCI lecture catalogue and current filename listing were located, but their theorem PDFs were not yet retrieved. Prefer accessible primary notes/official repository content if final optional FEM/adaptive/Godunov branches require a further source. This limitation does not replace local derivation or justify omitting core planned teaching.

## Implementation and evidence plan

Semantic owners: stable-ID topic body and blueprint; `numerical-pde-models.js`, `numerical-pde-examples.js`, `NumericalPdeLabs.jsx`, an optional separately owned figure file and `numerical-pde-labs.css`. Split by responsibility only when justified, not by chronological batch. Parent owns manifest/index/generated metadata/ledger; register only after the entire body is complete and parseable.

Before implementation assessment, calculate proposed fixtures independently with exact Fraction/SymPy matrix and piecewise integrals, sine eigenvalue identities and different linear solvers. These are design calculations, not evidence for as-yet-unwritten JS. Final native review must import actual models, execute actual saved programs/stdout, vary actual helpers, and verify all supported boundary states. Check nonuniform meshes, h/Δt separately, compatibility and zero mass/nullspace, exact versus rounded reports, source/load quadrature, and final-time step handling. No tests that merely restate a hard-coded displayed result.

Then inspect actual intended fonts at1440/390/320, all equations, ordinary diagrams, control changes, keyboard/reset, programs/expected output, practice and references. Bound loops/mesh sizes; memoize expensive field evolution separately from a cursor; no all-lessons imports. Freeze sources with separately attributed author and independent packets, preserve every original planned/author record, and close incoming/outgoing notes only after their implementation is evidenced. Build/loading/route integration and user acceptance remain separate.
