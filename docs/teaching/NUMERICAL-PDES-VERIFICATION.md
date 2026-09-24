# Numerical PDEs: Grids, Finite Elements & Stability — author verification

Author implementation and verification are complete. The [author packet](evidence/numerical-pdes-author-review.json) records the final freeze time and all six production hashes, matched to native and actual-font browser evidence. Publication enables real route review; it is not independent review, integrated production verification or user acceptance.

## Scope and preserved baseline

The stable identity is `numerical-pdes-grids-finite-elements-stability`, mathematics position 57. There was no original published body or complete program. The [original full planned entry](evidence/numerical-pdes-original-plan.json), [assessed design snapshot](evidence/numerical-pdes-assessed-design.md), [parent design calculations](../../../scratch/numerical-pde-design/results.json) and [independent design assessment](NUMERICAL-PDES-DESIGN-INDEPENDENT-REVIEW.md) remain distinct from actual production verification.

The authored route has 15 connected sections, 16 complete executed Python programs and 15 substantial changed practice tasks with separately hidden hints and explained answers. Eleven investigations and one inline restriction/reconstruction figure were chosen for distinct mechanisms; those counts are descriptive, not a template quota. Pure models and data are topic-owned; no cross-topic dataset or lab aggregate is imported.

The three required prerequisites remain PDE56, Conditioning52 and Matrix Decompositions. The lesson locally refreshes its derivatives, boundary signs, weak form, energy and linear solves. Real54's different convergence quantities are adapted without introducing a prerequisite cycle. This is the last mathematics entry; route controls remain authoritative about what follows.

## Mathematical and numerical contracts

The rod uses positive −u″ and prescribed endpoint values; scaling A to T=h²A rescales residual, defect and inverse bound together. The barrier proof supplies L²/8, and the reconstructed-field bound explicitly includes interpolation. The polynomial certificate computes the residual of the actual stored binary64 vector using exact rational arithmetic, then rounds bound conversions upward. The displayed sampled field discrepancy is not the certificate. The certificate concerns the declared manufactured model and a maximum temperature error, not physical-model calibration or flux accuracy.

Heat fields are actual finite updates and factored implicit solves at the same time. The exact spatial-ODE reference separates time error from spatial error. Finite-grid Euclidean spectral stability is distinguished from the r≤1/2 positivity condition. Crank–Nicolson's large-mode sign alternation is shown at successive times. Optional two-grid correction decreases energy, not necessarily every norm.

Finite volumes preserve one shared face flux. Two-material series resistance, outward Neumann compatibility before the mean gauge, and Robin weak-form signs are explicit. Upwind direction changes with velocity; periodic mass does not imply bounded amplitudes. Godunov's optional scalar face follows the entropy shock/rarefaction solution, including a stationary shock with unique flux but no claimed unique trace.

P1 assembly handles nonuniform lengths, natural loads and symmetric Dirichlet elimination. A point source is a shape-function evaluation. The constant-k 1D nodal-interpolation identity is not claimed for general higher-dimensional/variable-coefficient elements. Field/L2/energy errors are integrated for the actual target families. Galerkin best approximation assumes a conforming symmetric coercive problem, exact integrals and exact algebra. No adaptive reliability or general multigrid complexity theorem is asserted.

## Reproducible author checks

- `node scripts/verify-numerical-pde-models.mjs` imports actual production models and examples, parses the authored JSX, validates all 16 KaTeX expressions, exercises malformed-input boundaries and exports changed cases with six source hashes.
- `scratch/lesson-tools/Scripts/python.exe -X utf8 -I scripts/verify-numerical-pde-native.py` independently checks actual cases using NumPy dense solves/matrix powers, SciPy matrix exponentials/piecewise quadrature, exact fractions, Kronecker construction and an independent affine interpolation solve. It executes every displayed program and compares stdout, then changes the actual Poisson, heat, finite-element, sparse rectangle, Neumann and finite-budget certificate helpers.
- `scripts/generate-numerical-pde-examples.py` produces expected output by executing each standalone program. It does not guess outputs. Actual runtime: Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1.
- `node scripts/review-numerical-pde-lesson.cjs` checks the live route in Edge with Space Grotesk and JetBrains Mono loaded. Final runs require all three widths 1440/390/320, actual anchor arrivals, meaningful changed states, keyboard/reset, initially hidden hints/solutions, all code/output identities and question adjacency, equation/diagram geometry and ordinary-reading captures.
- `scripts/format-numerical-pde-source.cjs` conserves normalized JS and CSS ASTs. Source formatting changes are included in the final fingerprint, not attributed to an earlier hash.

The final source-matched native suite passed at **09:26:20 UTC**, recorded in `scratch/numerical-pde-native/native-results.json`. It includes 108 independent Poisson solves and exact rational certificates, 2,034 stable heat matrix-power comparisons, 720 exact spatial-ODE comparisons, independent conservative transport operators, 72 nonuniform FEM cases with piecewise field/gradient quadrature, 130 Kronecker stencil rows, affine triangle gradients/energy, and coarse energy projection. All 16 actual programs reproduce their stored stdout; changed calls exercise the displayed solvers, point loads, rectangular eigenmodes and finite-budget certificate. Counts describe cases and do not establish untested general theorems.

For unstable explicit heat states, an independently powered matrix can magnify tiny implementation/initial-data rounding differences; those states are checked by an independent one-step operator on the actual previous vector with a norm-scaled arithmetic allowance. Stable powers and the exact spatial-ODE exponential retain independent global comparisons. This change corrects an unsuitable oracle comparison, not a production recurrence.

The final actual-font browser run passed at **09:23:11 UTC** on the live route at **1440/390/320**. Each width verifies 21 named changed states, 41 keyboard/reset operations, all 15 actual anchor arrivals, all 16 program/output pairs and visible preceding questions, 15 initially hidden hint/solution pairs and all 16 mathematical displays. This includes the large explicit instability scale, conserved-but-oscillating transport, incompatible outward flux rejection, source alignment and actual changed triangle aspect ratio. There were no page/console errors, clipped diagram labels, equation overflow or document horizontal overflow. Local code/matrix panes keep their intended horizontal scrolling.

## Findings closed during authoring

The first source read corrected a Robin constant before body readiness: left insulated, right β=3/bath5/source2 requires u=20/3−x². Native exact substitution checks its outward flux. The translated triangle fixture required coordinates up to 5; its bounded model now accepts magnitude up to 8 and tests the actual fixture. The first browser attempt found ambiguous select labels that included option text; explicit `aria-labelledby` now names the controls. Phone review found overlong equations and inline equality chains; equations were broken at mathematical operations and prose chains were separated with words. No hidden overflow clipping substitutes for readable notation.

Full ordinary reading also moved the multi-method heat investigation after the implicit formulas, and the point-source investigation after its load weights. The final triangle image uses its actual coordinates with equal axis scale, rather than a fixed illustrative triangle, and its phone choices fit without truncation. Earlier failed captures and provisional outputs remain scratch evidence; the final packet identifies only final inspected source/image hashes.

## Source research and boundaries

The [design](NUMERICAL-PDES-LESSON-DESIGN.md) preserves the parent's original inspected passages and design-only calculations. The receiving author additionally inspected FNC finite-difference definitions/Taylor/order examples, diffusion scalar factors and periodic-versus-Dirichlet spectrum, the upwind domain-of-dependence/inflow/CFL discussion, and Laplace/Poisson sign and Kronecker passages. No complete-book read is claimed.

SciPy's official `spsolve` API page was checked for square CSR/CSC systems, vector shapes and ordering. The accessible official Clawpack `Burgers.ipynb` source was read for shock speed, rarefaction and entropy selection; its GitHub notebook page was verified as the learner link. The previously inaccessible HTML was not treated as read. Schöberl's finite-element notes were read around affine derivative transformations, shape regularity, H2 interpolation and the regularity-dependent L2 argument; the lesson retains only its locally proved 1D guarantee, with those notes an annotated advanced route.

The MIT 18.085 Lecture 17 official video/instructor/transcript page is verified and linked, with the parent's specifically inspected transcript pages retained in the design. Lecture 18's hat and point-load transcript is an alternative. Neither full video playback nor complete lecture viewing is claimed. All sources improve the original exposition; the lesson does not copy another course's wording or exercise list.

## Final freeze status

Thirty final screenshots were actually opened after the last browser run. They cover the first-pass introduction, ordinary stencil/energy/finite-element reading, all eleven investigation mechanisms, 320px mathematical displays, changed hint/solution reading, complete program, accepted output and annotated alternate resources. The exact image paths and hashes are in the packet; screenshot generation alone is not counted as visual inspection.

`scripts/finalize-numerical-pde-author.py` checks that all six current production hashes agree with the fixture/native/browser records, that the programs still agree with executed generation output, and that each opened image belongs to the final run. It refuses to overwrite a previous author packet; any later correction must preserve this freeze and record an explicit amendment.

All three incoming notes have implemented dispositions. The original planned entry and exact assessed design remain preserved. No unrelated lesson, shared registry, route order or global style was changed by this author task. **Independent source review, integrated build/loading checks and user acceptance remain separate and are not claimed complete here.**

## Endpoint amendment after independent review

The independent reviewer found an accepted-input defect: for three intervals and rod length 0.7, `j * length / intervals` produced the final mesh coordinate `0.6999999999999998`. Evaluating the curve at the prescribed endpoint 0.7 then correctly failed the interpolator's strict domain check. The repair pins both generated JS mesh/curve endpoints to their prescribed inputs and does the same in the actual Python Poisson helper. It does not loosen the interpolator contract or change the interior equations, solve, residual or certificate.

The exact original author packet, all six production files, generator and original program-execution record are preserved under [the before-repair archive](archive/numerical-pdes-before-endpoint-repair/), indexed with hashes by [the preservation manifest](evidence/numerical-pdes-endpoint-before.json). The [amendment record](evidence/numerical-pdes-endpoint-amendment.json) supplies the new frozen six-source identity; only the model and examples files changed. The author packet now points to that amendment while retaining its original full native/browser fields with their original source hashes. Those earlier checks are not relabelled as post-repair runs.

Targeted current-source checks passed:

- `node scripts/verify-numerical-pde-endpoint-amendment.mjs`: 150 changed mesh/length/profile/solver states, exact prescribed endpoints, unchanged interior arithmetic and certificates in the 144 previously accepted cases, six previous failures repaired, and strict out-of-domain rejection retained. The archived source independently reproduces the reported error.
- `scratch/lesson-tools/Scripts/python.exe -X utf8 -I scripts/verify-numerical-pde-endpoint-native.py`: all 16 actual programs executed with identical stdout, plus 105 changed calls to the actual displayed Poisson helper. Twelve old Python endpoint coordinates differed from the prescribed length; the amended helper pins them exactly while preserving every interior coordinate, solution and residual. The other 15 entire example records are unchanged.
- `node scripts/review-numerical-pde-endpoint-amendment.cjs`: actual-font 1440/390/320 checks verify the served module's 0.7 endpoint case, existing unit-rod/nonzero-boundary plot, keyboard Jacobi and Reset behavior, exact amended program and visible question, conserved output and absence of page overflow/errors. The unit-rod UI itself does not expose a variable-length control; the changed-length browser assertion is explicitly a served-model API check.

Five final captures were actually opened, covering the amended complete code, output and unchanged field/certificate at all three widths. Code/output keep their established intentional local horizontal scrolling on phones. The amendment records a corrected test-harness assumption about which example embeds the helper; that correction did not require another production change. Independent review, production rebuild/loading and user acceptance retain their separate evidence and ownership.
