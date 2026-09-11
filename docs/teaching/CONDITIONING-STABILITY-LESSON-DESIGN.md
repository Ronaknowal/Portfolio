# Conditioning, Stability & Numerical Analysis — lesson design

Status: assessed design implemented and author-verified on 11 September 2026. The exact source freeze and actual native/browser checks are recorded in [CONDITIONING-STABILITY-VERIFICATION.md](CONDITIONING-STABILITY-VERIFICATION.md) and [the author packet](evidence/conditioning-stability-author-review.json). The original proposal and its design-only fixtures below are retained as design history; they are not substituted for the final production checks. Independent review and shared production integration remain separate parent-owned steps.

## Identity, baseline and prerequisites

- Stable ID: `conditioning-stability-numerical-analysis`; retain the existing title.
- Mathematical & Statistical Foundations, position 52, in Bridges: Mathematical Language, Proof & Measurement. Keep the actual previous Complex Numbers, Fourier & Laplace Transforms and next Decision Theory, Risk & Cost-Sensitive Decisions. Review links do not reorder the module.
- Exact inventory command: `node scripts/build-curriculum-inventory.mjs --topic "Conditioning, Stability & Numerical Analysis"`, repeated with the stable ID when archiving. The original returned plan is preserved in [conditioning-stability-original-plan.json](evidence/conditioning-stability-original-plan.json), SHA256 `6574c21ecb0e51794eb18d4a4d6669d6eecadc2c5fa8f0caddcbeabf92ae9e5c`.
- Baseline is a planned cross-domain expansion entry. There is no legacy topic body, runnable program, output or publication mapping to preserve. Preserve its useful sensitivity-versus-algorithm distinction, forward/backward/residual outcomes, equivalent-expression experiment, across-scale repair task, tolerance/reference requirement and Goldberg resource.
- Returned destination note was absent. Read the unresolved inbox; its DSA bit-operation issue does not apply. Scoped searches of plausible numerical owners and note synonyms found no omitted incoming Conditioning note. This is not a catalogue-wide audit.
- Baseline misconceptions are phrased as true warnings. The individual brief will instead name the tempting false beliefs that feedback must correct.

**Proposed required prerequisites, for root to register:**

1. **Numerical Methods (Finite Differences, Quadrature, Root Finding)** — `numerical-methods-finite-differences-quadrature-root-finding`. Learner can read a computed approximation, a derivative and a fixed-interval refinement experiment; its recorded Single-Variable Calculus prerequisite supplies the necessary calculus. This lesson still refreshes the meaning of a derivative, the geometric-series calculation and a Taylor substitution before using them.
2. **Vectors, Matrices & Tensor Operations** — `vectors-matrices-tensor-operations`. Learner can multiply a small matrix by a vector. Infinity norms, induced row sums, uncertainty sets, two-by-two inverse action and all local sensitivity calculations are taught here.

Remove the inherited hard dependency on **Floating-Point Representation & Numerical Error** (`floating-point-representation-numerical-error`). Its actual plan is an unpublished GPU Engineering Readiness topic requiring C & C++ Foundations for GPU Programming. That branch is not needed to explain this mathematics lesson. Teach the essential representable-number/rounding model locally; retain the specialist as further study for dtypes, hardware instructions, compiler reassociation and parallel reductions. Root assessed and registered this prerequisite replacement. This author changed no shared prerequisite entry.

Matrix Decompositions is an optional explicit review for the deeper LU/QR/SVD connection. A local factor/solve explanation and a complete small example must remain readable without remembering those algorithms. Ordinary Differential Equations50 is a useful review link, not an additional required edge: the constant trajectory, a rate equation and the particular recurrence used here are introduced locally. Do not require later Numerical PDE57 or Real Analysis54 to read the core.

## Ownership and scope decisions

The central question is: **When a computer returns a plausible number, what would justify trusting it?** Start with two nearly redundant measurements of two unknown amounts. Use that concrete inverse problem again for perturbations, residuals, scaling, mixed precision and the final report. Equivalent expressions and reductions add distinct mechanisms rather than a disconnected list of failure anecdotes.

| Existing owner inspected | Useful content already taught or planned | What this lesson adds and what stays there |
| --- | --- | --- |
| Numerical Methods38, complete actual body, brief, design and verification | Safeguarded roots; derivative/stencil derivations and actual step-error sweeps; quadrature/interpolation; adaptive blind spot; error budgeting for a pump integral inside a root; a diagonal residual warning; short linear/AD/ODE bridges | Full conditioning and perturbation models; exact normwise/componentwise backward-error meaning; arithmetic versus data error; reduction analysis; a uniform propagation/convergence argument. Link its pump workflow and derivative trade-off without reproducing every method. |
| Matrix Decompositions2, actual lesson and brief | LU/pivoting, QR, Cholesky, SVD and least squares; normal-equation conditioning penalty; rank and singular directions | Interpret the numerical consequences of an actual rounded Gram matrix, reused low-precision factors and exact stored-input references. Do not reteach four factorization algorithms. |
| ODE50, current design/brief, being authored separately | IVP foundations, exact linear systems, Euler/midpoint/RK, local/global error, stiffness and event handling | Distinguish finite algorithm backward stability from uniform stability of an approximation family. Derive one deliberately bad constant-trajectory recurrence and its exact failure. No second ODE-solver catalogue. |
| Numerical PDE57, exact planned brief | Grid/boundary models, heat/Poisson, explicit time-step limits, finite elements and mesh refinement | Supply the norm/defect/amplification language and a finite linear error decomposition. PDE57 must establish its own mesh norms, boundary conditions, consistency, stability and solver tolerances. No universal CFL or nonlinear Lax-equivalence assertion here. |
| Floating-Point Representation & Numerical Error, exact planned GPU brief | Bits/dtypes, FMA, compiler and CPU/GPU differences, reductions | Own local binary64 assumptions and small deterministic arithmetic traces; leave hardware performance and parallel/compiler contracts there. |

Keep the title: it covers the resulting coherent learning outcomes. Do not add every application name. Theoretical deeper branches include singular directions, nonnormal transient growth, refinement, thresholded rank and the distinction between solving the original problem and regularizing it. Full interval-arithmetic software, verified LAPACK drivers, pseudospectra, arbitrary nonlinear perturbation theory, mixed-precision performance claims, Krylov/preconditioner catalogues and finite-element proofs are beyond this lesson.

Persisted discoveries: [GPU arithmetic continuation](topic-notes/floating-point-representation-numerical-error.md) and [Numerical PDE error normalization](topic-notes/numerical-pdes-grids-finite-elements-stability.md). Both remain open for their destination authors; their origin status now links the completed authored lesson and final numerical/browser evidence. Neither destination is claimed implemented.

## Outcomes and first-pass route

The learner should be able to state the exact problem and data, label a perturbation model and units, compare absolute/relative errors, derive a condition estimate, explain an algorithm's rounding loss, calculate a backward error, turn a residual into a conditional bound, choose a repair, and explain why refinement converges or fails. They should finish with a reproducible numerical report rather than only recognize terminology.

### 1. Separate the problem, its stored input and the computed answer

Open with two standardized sensor channels measuring the same two unknown amounts. Both unknowns use the same amount unit; channel outputs use the same calibrated signal unit. Define each row equation before writing `Ax=b`. Two almost identical mixtures poorly distinguish the difference between amounts. A more accurate solver cannot create a missing independent measurement.

Use a small inline flow: intended input → stored input → exact mathematical answer for stored input → computed approximation. Put **data/representation error**, **arithmetic/algorithm error**, and **model/discretization error** at their actual links. Forward error compares answers for the same stated problem; backward error asks for the smallest allowed data change explaining a computed result. A residual is what happens when that result is substituted into the equations. These are distinct quantities, not synonyms.

Define absolute error with units and relative error only for a nonzero reference. A target of zero needs an absolute scale. Introduce the infinity vector norm as the largest absolute component; for compatible scales the induced matrix infinity norm is the largest absolute row sum. Show why the row-sum bound follows from the triangle inequality, with a substituted two-row example. Norms on mixed physical units require scaling or a stated weighting, not an unqualified norm of metres and seconds.

### 2. See which numbers the arithmetic can store

A small binary lattice with four significant bits makes rounding cells visible. In `[1,2)` the gap is `1/8`; in `[2,4)` it is `1/4`. Halfway values `17/16` and `19/16` round to `1` and `5/4` under ties-to-even. The even bit belongs to the stored significand, not the parity of the decimal input.

Then connect to the checked runtime: binary64 has 53 significant bits, gap above one `2^-52`, and unit roundoff `u=2^-53` under nearest-even. Define the convention because texts use “machine epsilon” differently. Distinguish precision from exponent range, normal numbers from subnormals, and infinity/NaN from a completed finite answer. `2^-1074` is the smallest positive binary64 subnormal, while `2^-1022` is the smallest positive normal.

Under the declared finite, normal-result, round-to-nearest arithmetic assumptions, use `fl(a op b)=(a op b)(1+δ)`, `|δ|≤u`, for the relevant basic operations. Exact zero is handled directly. Do not silently apply that relative model to arbitrary underflow, overflow, all transcendental libraries, compiler reassociation or another device. The examples will inspect actual runtime behavior and guard nonfinite intermediates.

Work the input `10000000000000001`: converting it to binary64 stores `10000000000000000`. Higher precision applied after that conversion retains the rounded input. Compare an intended decimal string with the exact rational value of a stored float; neither is automatically the right reference for the other question.

### 3. A well-conditioned function can have a bad evaluation path

Use `f(x)=sqrt(1+x)-1=x/(sqrt(1+x)+1)` for `x>-1`. Explain rationalization algebraically before calling it a numerical repair. Show every intermediate operation for the same small dyadic input and identify where a perturbation is amplified by subtraction. Subtracting close *exactly represented* numbers is not itself always inaccurate; it can expose error already present in its operands.

Derive `f'(x)=1/(2 sqrt(1+x))` and the local relative input sensitivity `κ_f(x)=|(x f'(x))/f(x)|=(sqrt(1+x)+1)/(2 sqrt(1+x))` for nonzero `x>-1`. It tends to one near zero, so the spectacular loss is not explained by the function's conditioning. At `x=0` relative output error is undefined, though the formula has a limiting condition value; use absolute error there. At `x=-1` the function is defined but the derivative argument does not apply.

At `x=2^-54`, the actual direct result is zero, while the rationalized result is `2.7755575615628914e-17`. The exact dyadic square-root enclosure gives a reference near `2.77555756156289131254e-17`. Present actual sampled error points, including zeros and representability plateaus, not a fabricated smooth trend. Also evaluate negative dyadic inputs and domain boundaries.

The exact reference construction is an optional mathematical branch: for `x=±2^-k`, calculate an integer square root at 160 fractional bits, retain rational lower/upper bounds and subtract one exactly. Explain what is certified by these bounds and what remains rounded in a display. A standard-library complete program also compares against Decimal at two precisions; arbitrary-precision output alone is not a proof. A short transfer mentions `expm1`, `log1p` and scaled norms only with their native contracts and a worked input, not as an unexplained API list.

### 4. Measure sensitivity before blaming the solver

For `Aε=[[1,1],[1,1+ε]]`, `b=(2,2+ε)`, exact `x=(1,1)`. Changing only the second reading by `δ` gives `xδ=(1-δ/ε,1+δ/ε)`, from subtracting the two equations. This makes the mechanism accessible before matrix notation.

For `ε>0`, derive `||Aε||∞=2+ε`, `||Aε^-1||∞=(2+ε)/ε`, and `κ∞(Aε)=(2+ε)^2/ε`. With `ε=1/16`, `δ=1/256`, the answer moves to `(15/16,17/16)`. The relative output change is `1/16`; the standard finite right-hand-side bound is `κ∞ ||δb||∞/||b||∞=33/256`. It is an upper bound over perturbation directions, not a prediction that every perturbation attains it.

Derive `δx=A^-1 δb`, then the absolute and relative bounds. For simultaneous normwise changes `ε_A=||ΔA||/||A||` and `ε_b=||Δb||/||b||`, derive the denominator in `||δx||/||x|| ≤ κ(ε_A+ε_b)/(1-κ ε_A)` when `κ ε_A<1`. Nonzero reference and nonsingular matrices stay beside the formula. Local scalar condition estimates are first-order statements; this linear bound is finite under its hypotheses.

An `ε=0` state should visibly distinguish coincident/nonunique from incompatible/no-solution equations. Do not display an invented finite inverse. Singular directions explain the two-norm/SVD interpretation in a deeper branch, with an explicit readiness link. Warn that “κ=10^k loses exactly k digits” is only a rough worst-case scale heuristic. Condition numbers depend on the problem map, input/output scaling, norm and allowed perturbations.

### 5. Read a residual as evidence with a stated perturbation model

Continue the already introduced Numerical Methods example `A=diag(1,10^-6)`, `b=(1,10^-6)`, `xhat=(1,0)`, exact `x=(1,1)`. Residual `r=b-Axhat=(0,10^-6)` is small in an unscaled norm while forward infinity error is one. Derive `xhat-x=-A^-1 r`, so an exact residual plus an appropriate inverse bound can control forward error. In a general problem the computed residual is itself rounded.

Teach two exact unstructured backward-error measures with all denominators visible:

- Allow `||ΔA||∞≤η||A||∞`, `||Δb||∞≤η||b||∞`: minimum `η=||r||∞/(||A||∞||xhat||∞+||b||∞)`. For this fixture it is `5e-7`. Prove the lower bound by substitution/triangle inequality and give an attaining perturbation; a deeper general construction can place row perturbations in a largest-magnitude component of `xhat`.
- Allow each entry `|ΔAij|≤η|Aij|`, `|Δbi|≤η|bi|`: minimum `η_c=max_i |r_i|/(|A||xhat|+|b|)_i`. It is one here. A zero denominator with zero residual contributes zero; invalid/inconsistent arithmetic cannot silently become zero. The allowed changes retain exact zero entries; they do not necessarily retain symmetry, positive definiteness or other structure.

The first model can explain the answer by adding `5e-7` to the originally zero `A21` and subtracting `5e-7` from `b2`. The second cannot change that zero entry. This explicit witness is more meaningful than merely plotting two different error numbers.

Scale the second row by `10^6`. The exact solution is unchanged, componentwise backward error remains one, but normwise backward error becomes `1/2` and matrix condition becomes one. Rescale measurement uncertainty too: a physical second-channel error of `1e-8` becomes `0.01` in the scaled row and still gives the same amount uncertainty. Scaling changes the numerical representation and normwise perturbation model; it does not manufacture information.

For a joint normwise backward error η, the conditional finite bound becomes `2κη/(1-κη)` when `κη<1`. Do not call `κη` an unconditional exact certificate. Estimated condition numbers and residuals computed at working precision produce estimates unless their own error bounds are included. Explain why least-squares residuals need not be small at an optimum; a nonsingular square-system certificate cannot be pasted onto an inconsistent least-squares problem.

### 6. Summation reveals what algorithmic stability does and does not promise

Show the operation tree for a left fold, a balanced tree and an explicit compensation channel. The running sum of `[1e16,1,-1e16]` loses the middle one; the alternative order `[1e16,-1e16,1]` does not. Balancing is not automatically better for every ordering. `[2^53,1,1,1,1]` loses four under the naive left fold even though its componentwise sum condition is one.

Derive `κ_sum=(sum |xi|)/|sum xi|` for independent componentwise relative input perturbations and a nonzero exact sum. Work backward through the rounded additions to derive the bound `|ŝ-s|≤γ_(n-1) sum |xi|`, where `γ_m=mu/(1-mu)` and `mu<1`. State the arithmetic assumptions and dimension-dependent constant. A small backward error can accompany a large relative forward error on a cancellation-sensitive sum. Balanced tree depth changes a worst-case bound, not a universal measured ranking.

Explain the Neumaier correction by following the discarded low part on one step; include complete implementations of the naive loop, balanced recursion, classic Kahan and Neumaier, plus `math.fsum`. The measured first fixture is also a useful counterexample: classic Kahan gives zero while Neumaier/fsum give one. No method here is promised exact for every finite sequence. Python 3.12's built-in `sum` changed, so it must not be mislabeled as the naive loop.

Define **backward stable** explicitly rather than silently making it the definition of every use of “stable.” A short counterexample `f(x)=1+x`, tiny nonzero x, has a very accurate rounded answer one but relative backward error one if x alone may change. Stability terminology depends on the chosen forward/mixed/backward criterion and perturbation model. Later propagation stability is a different, explicitly introduced property.

### 7. Repair the algorithm, and recognize when the problem remains sensitive

Core remedies: algebraic reformulation, appropriate accumulation, scaling with consistent units, trustworthy factor/solve routines, meaningful tolerance and better source data. More printed digits is not a remedy. Compare these with deliberately changing the problem by regularization; that may reduce sensitivity while introducing bias relative to the original target.

Deeper complete mixed-precision experiment: factor the same two-channel matrix in float32 once, compute a float32 solve, store the iterate in float64, compute the residual using the original float64 matrix and RHS, and reuse the low-precision factors for each correction. Show actual dtypes, rejection of failed factors, finite checks and stopping limits. Separate the intended `(1/3,2/3)` input construction from the exact solution of the stored matrix/RHS. The latter is the forward-error reference.

At `ε=2^-20`, the initial solve is `(0.375,0.625)`. Two observed correction steps exactly recover this fixture's stored-input solution `(715827883/2147483648,1431655765/2147483648)`. This exact coincidence is local, not a general promise. At `ε=2^-24`, the rounded float32 matrix becomes singular; it must produce a visible failure rather than NaNs presented as estimates. The same experiment is repeated at `2^-12`.

Derive the exact-arithmetic refinement error recurrence `e_next=(I-M^-1 A)e` for a correction solve using nonsingular M. A norm less than one is a sufficient contraction condition; real rounding and residual errors add forcing terms. Do not reduce the mixed-precision convergence theorem to an unconditional `κu<1` claim, nor infer hardware speed from a small demonstration.

One complete optional QR comparison uses the three rows `(1,1),(1,1+2^-27),(1,1-2^-27)` and RHS `(0,-2^-27,2^-27)`. Its exact solution is `(1,-1)` and exact Gram lower-right entry is `3+2^-53`; actual binary64 Gram formation collapses it to three and produces a singular normal system. Actual QR/lstsq retain the direction. Explain `κ₂(AᵀA)=κ₂(A)^2` only for full column rank, and keep a numerical rank threshold distinct from exact rank. Matrix Decompositions owns the factorization derivations.

### 8. Understand how errors accumulate across a family of approximations

Begin with the transparent scalar recurrence `e_(k+1)=q e_k+δ_k`, `|δ_k|≤ρ`. Repeated substitution gives `|e_n|≤|q|^n|e_0|+ρ sum_(j=0)^(n-1)|q|^j`. Let the learner compare one pulse, repeated same-sign and alternating disturbances. Distinguish an actual signed trajectory from a worst-case envelope. A negative multiplier reverses signs; it is its magnitude that enters this simple bound.

For a fixed final time T, one-step defect at most `C h^(p+1)`, step amplification at most `1+Lh` with `L≥0`, and `N=T/h`, derive a global estimate bounded by `exp(LT)|e_0|+C h^p (exp(LT)-1)/L`, with limiting expression `C T h^p` at `L=0`. Constants and smoothness must be uniform over the stated interval. A fixed arithmetic perturbation per step has a different scaling and can grow with the number of steps. This is an explained sufficient result, not a theorem about every nonlinear solver.

Use a deliberately designed, fully checked counterexample to “consistent therefore convergent.” Taylor expansion shows `[3u(t+h)-2u(t)-u(t+2h)]/h=u'(t)-(h/2)u''(t)+O(h²)`. The resulting first-order consistent method `y_(n+2)=3y_(n+1)-2y_n-h f(t_n,y_n)` applied to `u'=0`, `u(0)=1`, has `y_0=1`, `y_1=1+h²`. At fixed T=1 its error is `(2^N-1)/N²`, where `h=1/N`. The shrinking start error is amplified by the parasitic root two. Compute the root recurrence locally; do not assume a multistep course. Exact N=4,8,16,32 results make the failure undeniable without simulation noise.

Deeper matrix caution: `B=[[1/2,10],[0,1/2]]` has both eigenvalues `1/2`, but starting with `(0,1)` gives `B^n e_0=(10n(1/2)^(n-1),(1/2)^n)`. Large early growth coexists with eventual decay. Eigenvalues alone do not establish a small finite-horizon perturbation bound. A full pseudospectral theory is not required.

Close the PDE bridge with a declared finite linear discretization: let `u_h` be the exact continuous solution sampled/restricted to the discrete space, `τ_h=b_h-A_h u_h` its consistency defect, and `r_h=b_h-A_h xhat_h` the algebraic residual. Then `A_h(xhat_h-u_h)=τ_h-r_h` and the error is bounded by `||A_h^-1|| (||τ_h||+||r_h||)`. Norms, scaling and invertibility matter. Uniform stability plus vanishing properly normalized defects/residuals gives this sufficient convergence conclusion; rescaling the equations also rescales the defects. Numerical PDE57 will prove the relevant bounds for its actual schemes and boundaries.

### 9. Make an accuracy decision the evidence supports

Write an explicit protocol before the closing tasks: state target quantities/units; intended versus stored data; dtype and evaluation order; observed/derived reference method; norm and perturbation model; residual/backward error; sensitivity conversion and its hypotheses; acceptance tolerance; rejected states and unresolved data/model uncertainty.

Use near-zero comparisons to show why a relative tolerance alone can be inappropriate and why library defaults are not a scientific requirement. Python `math.isclose` uses a symmetric maximum rule; NumPy `isclose` uses `atol+rtol*abs(b)` with b as reference. Actual `np.isclose(1e-9,2e-9)` is true under its defaults; a chosen absolute budget of `1e-12` rejects that comparison. Spell out units and how thresholds transform under unit conversion.

The changed capstone uses `ε=2^-16`, with stored central RHS `(1,1.0000101725260415)` and uncertainty only in the second reading, `|δb2|≤2^-24`. This is a declared bound around the stored central observation, not a claim that the initial `(1/3,2/3)` data-construction target is the exact solution. The initial float32 solve gives `(0.3359375,0.6640625)`; two observed corrections recover the stored-input center `(11453246123/34359738368,22906492245/34359738368)`. The uncertainty radius in each inferred amount is `2^-24/2^-16=1/256`. An absolute amount budget of `1e-4` therefore cannot be certified across the allowed readings merely by repairing arithmetic. This does not assert that the unknown actual error necessarily equals its worst-case radius. If the second-reading bound improves to `2^-30`, the radius becomes `1/16384≈6.10352e-5`, which fits that budget after accounting for the calculation's own error.

Compare the initial calculation, repaired calculation and exact stored-input oracle, then decide whether to improve arithmetic, collect a more informative measurement, report a wider interval or stop. Supply a complete accepted example report and an executable acceptance check; do not leave the answer as generic reporting advice. Connect to Decision Theory53: numerical bounds constrain the evidence available for a decision; selecting actions additionally needs consequences and costs, which that next lesson owns.

## Representation contracts

These are distinct learning jobs, not a mandatory lab count. Essential diagrams appear within the explanation even if the learner never changes a control. Do not put all reasoning in collapsed experiments.

| Placement and form | What changes / exact mapping | What the learner predicts or diagnoses | Data/geometry and boundary checks |
| --- | --- | --- | --- |
| Input-to-answer inline flow | Intended decimal, stored rational, exact answer and computed answer occupy separate nodes | Which error a higher-precision rerun can repair | Labels and arrows are definitions; no invented quantitative lengths |
| Rounding-cell number line | Exact toy significand grid and midpoint choice; exponent-bin toggle | Which neighbor wins and why spacing changes | Fraction oracle for all cells/midpoints; do not stretch unequal numeric gaps into equal lengths |
| Same-input arithmetic graph with actual error samples | Direct/rationalized branches retain visible intermediate values; dyadic exponent and sign | Locate loss rather than merely observe a red error number | Exact integer-square-root enclosure plus Decimal; zero/reference/domain states explicit; no log of zero or invented epsilon |
| Measurement lines and perturbed-intersection geometry | Near-parallel rows, a bounded second-reading change and exact resulting amounts | Sensitivity to direction and separation | Coordinates from the actual row equations; fixed units/scales, readable intersection; ε=0 separate ill-posed state |
| Backward-error perturbation ledger | Highlight changed matrix/RHS entries for normwise witness versus componentwise restrictions; row-scaling toggle | A small reported backward error depends on what may change | Exact rational witnesses, denominator conventions, rescaled uncertainty; zeros do not silently become editable in componentwise model |
| Summation trees and compensation channel | Actual ordered additions, grouping, stored subtotal and correction | Explain why a ranking changes with input/order | Independently evaluated binary64 or explicitly labeled exact toy arithmetic, not stylized nodes pretending to be measurements |
| Error propagation/constant-trajectory refinement | Pulse/repeated/alternating disturbances; signed states beside bound; fixed-T refinement table/plot | Why a smaller local discrepancy can still be amplified | Exact Fraction recurrences and geometric sum, stable horizon labels, limits and log-scale zero handling; no universal simulation guarantee |

Refinement can be an annotated step table rather than another generic slider panel. Nonnormal growth may be a compact two-coordinate trace with exact sample points. Choose a different form if actual implementation reveals clearer geometry, while preserving the specified causal question and evidence.

All visual entities have prose introductions, captions, units and text equivalents. Use bounded controls with honest rejection, a visible reset and keyboard operation. Phone layouts at 390 and 320 pixels keep number/axis labels readable; local horizontal scrolling is reserved for genuinely wide arithmetic traces with a visible cue, not used to hide a clipped endpoint. Check every actual control selection agrees with the active model. Every plot distinguishes symbolic/derived bounds, exact finite fixtures and measured machine arithmetic. No fabricated benchmark or universal speed comparison is planned.

## Complete runnable examples and independent practice

Planned standalone programs have imports, local inputs, setup before first use, a prediction question actually rendered by the example component, executable code, captured output and interpretation. Python 3.12 standard library suffices for rounding, exact references, sums, rational certificates and recurrence; NumPy/SciPy are introduced explicitly only for the factorization comparisons. Use the existing isolated runtime; no package install is currently needed. The final authoring pass decides the number by learning need, not this table's row count.

| Program family | Required runnable work and evidence |
| --- | --- |
| Representation and error | `as_integer_ratio`, `ulp`, adjacent ties, intended decimal versus stored float and absolute/relative comparison |
| Equivalent expressions | Direct/rationalized values over signed dyadic inputs; exact sqrt enclosure plus independent Decimal check; deliberate zero/domain states |
| Sensitivity | Solve the two equations exactly and numerically, perturb a changed direction, compare actual error with a derived bound |
| Residual/backward | Compute normwise/componentwise diagnostics and an explicit attaining perturbation; repeat after scaling, carrying the uncertainty units |
| Summation | Explicit naive, balanced, Kahan and Neumaier algorithms alongside `fsum`, including a changed ordering and zero exact sum |
| Mixed precision | One float32 LU factorization reused for corrections, float64 residual/iterate, exact stored-input oracle, precision-collapse failure |
| Optional least squares | Show the exact versus rounded Gram entry and the actual QR/lstsq outputs; full-column-rank/rank-threshold conditions |
| Error propagation | Exact signed recurrences and bounds; fixed-time consistent-but-unstable example, with no floating simulation needed to prove it |
| Numerical report | A complete changed measurement/solve protocol, tolerance choice, uncertainty conversion and defensible conclusion |

Practice is independent of those exact demonstrations, with staged hints, worked substitutions, explanations of wrong answers and runnable acceptance cases where appropriate. Interleave a small early check, then use descriptive task titles in the final set; avoid numbering that appears to skip earlier tasks.

- **New rounding cells:** use `17/8` and `19/8`; choose `2` and `5/2` and explain the changed bin width. Check actual `1+3u` versus `1+u`.
- **New input/reference contract:** distinguish exact intended decimal data from `Fraction(float_value)`; identify which digits have already been lost and what rerunning from the original string can recover.
- **New sensitivity:** take `ε=1/128`, `δ=-1/256`, so the amounts are `(3/2,1/2)`. Compute the forward infinity error `1/2` and bound `257/256`; explain why the bound is larger. Include a δ=0 case that does not acquire error merely because κ is large.
- **New residual model:** use `diag(1,10^-3)` with the same kind of omitted component. Normwise backward error is `1/2000`, componentwise error one, and forward error one. Construct the perturbation and repeat in scaled units.
- **New reduction order:** `[1e16,3,-1e16]` has exact stored-input sum three. Actual naive/balanced/Kahan return four, Neumaier/fsum return three on the checked runtime. Also explain why relative error is unavailable for `[1,-1]` even when the computed result is exactly right.
- **New refinement diagnosis:** compare `ε=2^-12` and the `2^-24` singularity. The answer must separate arithmetic repair from the original input's uncertainty and avoid continuing a failed factorization.
- **New propagation:** for q=`1/2`, zero initial error and constant disturbance `1/100`, derive the exact error after n steps as `(1/50)(1-2^-n)`. Change the disturbance signs and explain the bound's non-attainment.
- **New consistency counterexample:** derive the exact N=8 error `255/64` from starting error `1/64`; account for the second recurrence mode. A refinement plot is evidence of this fixture, not a general proof for another method.
- **New tolerance/units:** compare the same small discrepancy expressed in volts and millivolts; convert the absolute tolerance too, rather than accepting one representation and rejecting the other for an accidental reason.
- **Changed complete report (final authoring refinement):** section 9 supplies the full executed `ε=2^-16` example. Independent closing practice changes it again to `ε=2^-18`, preserving the larger reading radius `2^-24` while changing the smaller one to `2^-34`. Actual unchanged report code with these two literal edits gives central fractions `2863311531/8589934592` and `5726623061/8589934592`, initial float32 values `(0.34375,0.65625)`, zero final central error for this fixture and amount bounds `1/64` (fail) and `1/65536` (pass). The separate changed-report verifier checks exact stored-input Cramer values and both interval conversions. The original assessed `ε=2^-16` design fixture remains historical evidence, not the independent task answer.

## Research and resource ledger

Retrieved 11 September 2026. These sources support scoped claims, not copied wording or a borrowed whole-page structure. Mathematical derivations and constructed fixtures above are written for this lesson. No complete video was watched. The source review deliberately distinguishes a historical explanatory resource from current library behavior.

| Source / inspected locator | Verified use and learner annotation | Bounds on review |
| --- | --- | --- |
| [Driscoll & Braun: Problems and conditioning](https://fncbook.com/conditioning/), main derivative/condition discussion | An approachable written alternative for absolute/local relative sensitivity and perturbing inputs | Read main definitions and examples; our finite linear proofs and changed fixtures are independently derived |
| [Driscoll & Braun: Stability](https://fncbook.com/stability/), evaluation chain and §1.4.2 | Explains algorithm versus problem and backward error; useful after the first repair | Inspected definitions, examples and the stable-but-not-relatively-backward-stable `1+x` counterexample; did not run/copy its Python programs or watch embedded videos |
| [Driscoll & Braun: Floating-point numbers](https://fncbook.com/floating-point/), §1.1.3 and range/precision discussion | Optional written rounding refresher | Source simplifies subnormals and terminology. This lesson explicitly distinguishes unit roundoff, gap, smallest normal and smallest subnormal |
| [Goldberg: What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html), introduction/format/rounding sections | Retained authoritative historical deeper resource for representation and arithmetic assumptions | 1991 article; inspected relevant opening sections, not all architecture/compiler claims. Not evidence about current GPU performance |
| [Higham: What Is Backward Error?](https://nhigham.com/2020/03/25/what-is-backward-error/), main definition and forward-condition relation | Concise expert written alternative after the residual section | Its first-order relation is not substituted for our finite denominator bound |
| [MIT18.335: Week2](https://ocw.mit.edu/courses/18-335j-introduction-to-numerical-methods-spring-2019/pages/week-2/), Lectures3–5 summaries | Deeper written explanation of recursive summation, uniform backward stability and norms | Read summaries and scoped condition discussion; linked handout PDFs were not claimed fully read. Constants may depend on dimension; terminology is made explicit locally |
| [LAPACK DLA_LIN_BERR](https://www.netlib.org/lapack/explore-html/d0/df7/dla__lin__berr_8f_source.html), purpose and safeguards | Implementation reference for componentwise residual denominator | Inspected formula/argument docs including safe-minimum guard. Our exact educational quotient is not a claim to reproduce every protected LAPACK driver behavior |
| [LAPACK: Improved Error Bounds](https://www.netlib.org/lapack/lug/node79.html) | Explains why componentwise perturbations retain small entries and zeros | Read main componentwise discussion. Date 1999; use for mathematical distinction, not promises that all current routines meet one guarantee |
| [Carson & Higham: Iterative refinement in three precisions](https://nhigham.com/2017/07/), complete article | Worthwhile advanced application: keep expensive factors in lower precision and inspect the residual more accurately | Read algorithms/precision roles and scope; do not generalize its reported speedups or shorthand conditioning thresholds to our two-by-two experiment |
| [SciPy lu_factor](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lu_factor.html) and its linked lu_solve | Current API reference for factor reuse and pivot conventions | Read factor/return/check-finite contract. Fetched manual labels1.18.0; executed SciPy1.18.1 locally. Actual dtype and warning behavior verified in fixtures |
| [Python3.12 math](https://docs.python.org/3.12/library/math.html#math.fsum) and [built-in sum](https://docs.python.org/3.12/library/functions.html#sum) | Accurate accumulation and explicit comparison/reference APIs | Inspected fsum caveats, isclose formula and sum3.12 change. Runtime 3.12.14; no universal exactness claim for fsum |
| [NumPy isclose](https://numpy.org/doc/stable/reference/generated/numpy.isclose.html), Notes | Practical reference for asymmetric reference scaling and unsuitable near-zero defaults | Fetched stable manual currently labels2.5; actual behavior checked with installed NumPy2.3.5. Record/recheck actual final runtime rather than infer equality of all versions |
| [Driscoll & Braun: Zero-stability](https://fncbook.com/zerostability/), instability example, root condition | Deeper written continuation from our exact two-mode counterexample | Inspected counterexample and root-condition statement/partial derivation. We do not copy its example or claim its displayed program executed locally |
| [MIT/Gilbert Strang: Difference Methods for Ordinary Differential Equations](https://ocw.mit.edu/courses/18-086-mathematical-methods-for-engineers-ii-spring-2006/resources/lecture-1-difference-methods-for-ordinary-differential-equations/) ([direct YouTube](https://www.youtube.com/watch?v=gv-AB35V2k8)) | Optional graduate-level visual explanation of repeated multipliers and step stability, especially after section8 | Official page/embedded video identity verified; [official transcript](https://ocw.mit.edu/courses/18-086-mathematical-methods-for-engineers-ii-spring-2006/806b14f6a7a07a2209735c5162b567aa_gv-AB35V2k8.pdf) inspected through definitions and scalar Euler/stiffness discussion, particularly pp8–9. No playback claim. Lecture 2006; historical MATLAB defaults and performance remarks are not adopted as current API guidance |

The learner-facing sources section will use concise usefulness/level annotations, not this whole author ledger. Resource links supplement the full explanation and include a genuine alternate video format without a media quota.

## Evaluated design fixtures and verification plan

Actual script: [verify-conditioning-design-fixtures.py](../../scripts/verify-conditioning-design-fixtures.py). Results: [fixtures.json](../../scratch/conditioning-design/fixtures.json), completed **2026-09-11T05:16:59.233402+00:00**, **547 explicit checks**. Python3.12.14, NumPy2.3.5, SciPy1.18.1 on Windows. The JSON records the exact script fingerprint, all actual output states and clear design-only scope.

Evidence includes 120 signed dyadic cancellation inputs with exact rational sqrt enclosures and independent Decimal150 comparisons; exact two-by-two perturbations and backward witnesses; five actual reduction orders including counterexamples; reused float32 factors and exact stored-input solution references; exact-versus-rounded Gram formation; exact signed geometric bounds, fixed-horizon recurrence and nonnormal transient values. These counts describe a bounded fixture test, not a production verifier, novice study, successful browser or a general proof of numerical safety. No random benchmark is used.

Before author freeze, extend independent checks against actual exported production models/native programs. Test all bounded UI inputs plus direct-call contracts: dense nonempty arrays; exact zeros; nonfinite/underflow/overflow; singular and nearly singular matrices; domain endpoints; invalid enumeration keys; relative denominators; representability limits; changed order, signs and scales. Reject arithmetic outside supported bounds rather than report a fictitious infinity as an exact mathematical divergence. Preserve exact versus rounded and estimated versus certified labels.

For model geometry, independently substitute intersection points into row equations, validate midpoint cell locations and tie states, compare every plotted arithmetic node with the executed operation, verify row-perturbation witnesses, and inspect all sampled error coordinates and bound envelopes. Independent exact rational construction must not simply repeat the production algorithm's own formula.

Actual browser review at 1440/390/320 must cover all meaningful presets, controls, changed drafts, errors/reset, keyboard/focus, captions, math rendering, reference links, question/answer components, native code/output fidelity and page overflow. Open final screenshots from ordinary reading as well as operated labs. Use actual fonts; inspect narrow formula and label sizes. Do not infer valid paragraph structure from pageerror-only tests; collect React console errors too. Freeze six semantic production sources with native/browser evidence only after these checks. Root owns publication registration, shared prerequisites, production build/loading integration and the global ledger.

## Implemented ownership and independent-review handoff

Root assessed and registered the brief and the complete body. Implemented body `src/learn/data/topics/conditioning-stability-numerical-analysis.jsx`; supporting semantic files `conditioning-stability-models.js`, `conditioning-stability-examples.js`, `ConditioningStabilityLabs.jsx`, `conditioning-stability-labs.css`; individual blueprint `src/learn/data/curriculum/blueprints/conditioning-stability-numerical-analysis.js`. Model/runtime/example separation follows the learning code standard. Use existing LessonElements and runnable components after checking their actual prop contracts; avoid expanding a generic shared lab solely for this topic.

The author changed only these scoped semantic files, individual evidence/scripts and outgoing notes. Root owns the manifest/index, generated catalogue, shared prerequisites and integration ledger. The final author packet is ready for a separate independent source review and production integration; no broader rollout completion or user acceptance is claimed.
