# Matrix Decompositions — implementation and verification

Author review completed 10 September 2026. Read with [the design and source ledger](MATRIX-DECOMPOSITIONS-LESSON-DESIGN.md). Stable title, ID, publication mapping and second position in the maths module are preserved. This records lesson-level evidence; the active rollout owns application-wide integration and user acceptance remains pending.

## Teaching and preservation

The rewritten lesson retains all four original practical fixtures: the square solve, noisy line fit, correlated-noise construction and ratings SVD. Native execution confirmed that the old rounded QR/SVD answers were mathematically consistent. The repairs are complete executable setup, explicit formatting, actual mechanisms, shape/assumption contracts and independent practice, not invented numerical defects in the old material.

Core teaching derives triangular substitution, pivoted LU, QR projection and least-squares optimality, covariance construction, singular directions and rank truncation. Four different investigations expose those mechanisms. An immediately visible triangular dependency figure and directional outer-product matrices support the ordinary reading flow. Optional branches develop Householder reflections, factor shapes, whitening/log determinants, pseudoinverses, conditioning and the spectral approximation bound. Four independent exercises have concealed hints and explained solutions, including a complete new calibration program.

An independent author read the full lesson, ten programs, models, labs and native checks. They recomputed the calibration, covariance, rank-budget and hand-factorization examples. Their one substantive finding was repaired: the paired SVD identity is explicitly restricted to min(m,n); extra right-basis directions of a wide matrix map to zero, whereas extra left-basis directions of a tall matrix are unreachable outputs and lie in the null space of Aᵀ. A concrete 1×2 example now makes the distinction visible in prose. No other blocker was identified. This was an author cross-review, not an observed beginner study.

The lesson distinguishes low rank from actual storage savings (the retained tiny example requires 14 scalars versus 9 originally), observed zeros from missing ratings, approximation from guaranteed denoising, small residual from accurate parameters, and a successful Cholesky call from independent symmetry validation. Geometric outlines are expressly not empirical measurements or probability contours.

## Independent computation — passed

Command: `node scripts/verify-matrix-decomposition-examples.mjs`.

Runtime: Python 3.12.14 and NumPy 2.3.5 in `scratch/lesson-tools`. No SciPy execution is claimed. The source ledger separately identifies the versions of online documentation reviewed.

| Check | Actual coverage and oracle |
| --- | --- |
| Complete displayed Python | All 10 standalone programs executed in separate processes; every displayed stdout compared exactly |
| Partial-pivot LU | 400 seeded systems, dimensions 1–8; NumPy solve reference, PA=LU, triangular structure, multiple targets and previous-multiplier row swaps |
| LU boundaries | Empty, nonsquare, nonfinite, complex and singular inputs rejected under the stated real-square contract |
| Elimination investigation | All four preset traces; every augmented state obeys PA=L·current-left and Pb=L·current-target, with independent solve/consistency checks |
| QR geometry | 81 bounded column states, including zero and dependent columns; independent projection, orthogonality and reconstruction checks |
| Covariance geometry | 41 correlation values including ±1; LLᵀ, determinant, positive-semidefinite eigenvalues, image coordinates and SPD-only native Cholesky |
| SVD geometry | 1,500 angle/scale/rank configurations; independent NumPy singular values, reconstruction, transformed points, numerical rank and spectral/Frobenius errors, including zero and tied singular values |
| Invalid model inputs | Unknown presets, malformed/out-of-range/nonfinite controls rejected |

Evidence is in `scratch/matrix-decomposition-verification/native-results.json`, the emitted model fixtures and the ten exact `.py` programs. Initial expected-output fixtures were corrected to actual seeded covariance and signed-zero formatting before the suite passed; no hand-authored output is presented as an execution. The checks are finite numerical evidence, not substitutes for the lesson's derivations or general asymptotic proofs.

## Browser, keyboard and visual review — passed

Command: `node scripts/review-matrix-decomposition-lesson.cjs` on the registered Vite route, Microsoft Edge, 1440×1000 and 390×1000.

At each width the suite checked all 10 rendered programs/outputs and eight valid anchors; 18 elimination states across four presets; seven QR column cases; five covariance states including singular endpoints; and ten SVD scale/rank/angle configurations. Values agree with the separately verified models. Reset, dependent/singular messaging, range-keyboard interaction, Enter/Space solution disclosure, visible keyboard focus, practice-anchor arrival and the next-topic link passed. KaTeX rendered without errors. No page errors or page/figure overflow occurred.

Opened and inspected actual desktop/mobile screenshots of elimination, QR, covariance, SVD geometry, the factor-coordinate chain and triangular dependency. Axis scales and geometric values agree, text legends carry numerical alternatives, zero vectors remain represented, and the mobile paired plots stack without losing the transformation comparison. The normal reading pass introduces each mechanism and the needed terms before its investigation. The next lesson remains Eigenvalues & Eigenvectors, following the existing module sequence.

Evidence: `scratch/matrix-decomposition-lesson-review/results.json` and corresponding PNGs. The initial focus assertion used programmatic focus after mouse input; the harness now reaches the summary through actual Shift+Tab/Tab navigation and verifies focus visibility in keyboard modality. No application focus styling was bypassed. Tests close only the development HMR WebSocket to isolate concurrent authoring, and temporarily hide fixed navigation during element screenshots. These are test-harness behaviors, not production changes.

After review, four owned production sources were conventionally formatted. Normalized Babel AST equality, including JSX text and comments, was asserted before writing; this did not change rendered content, model behavior or displayed programs. Topic data and models remain locally owned, bounded and loaded with this lesson, without added dependencies, automatic simulations or timers.

## Limits and next integration

Primary references, inspected locators, annotated alternate resources and viewing limits are recorded in the design. MIT's video session is linked with its fully read companion summary; full-video viewing is not claimed. No deployment, user acceptance or measured learning outcome is implied. Production build, route conservation and loading checks are recorded separately by the active rollout.
