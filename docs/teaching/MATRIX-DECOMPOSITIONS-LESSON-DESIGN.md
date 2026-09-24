# Matrix decompositions: lesson design and source ledger

10 September 2026. Scope: improve the existing published Matrix Decompositions (SVD, QR, Cholesky, LU) lesson within the active DSA/maths goal. Title and stable ID remain appropriate. Implementation and lesson-level author review are complete; [the verification record](MATRIX-DECOMPOSITIONS-VERIFICATION.md) owns actual evidence and remaining integration limits. User acceptance is pending.

## Existing material and repair

The original four practical questions remain: solve repeated square systems, fit inconsistent observations, transform covariance, and expose/compress low-rank structure. Preserve the original linear solve, noisy four-point fit, correlated-noise construction and ratings matrix as runnable examples or developed branches. Existing weaknesses: library calls substitute for mechanisms; several snippets rely on earlier imports; outputs mix manually rounded values and approximate descriptions with unformatted code prints; no diagrams or interactive investigations; no justified independent exercises; no sources or precision/shape contracts. Native execution confirmed the old rounded QR/SVD values are mathematically consistent; the repair is explicit formatting and reproducible outputs, not a claim those numerical answers were wrong. A decomposition is not automatically the only solver for a task. Explain actual factorization conventions and validity conditions.

The module currently places this page before Eigenvalues & Eigenvectors. Keep that sequence and give a local definition of orthogonality, span, rank, positive definiteness and singular directions before relying on them. The eigenvalue derivation is optional and explicitly points to the next topic. No title/order change is necessary. The unassigned bit-manipulation note does not apply to this maths topic.

## Outcome and teaching map

| Outcome | Mechanism, fixture and representation | Practice / assessment |
| --- | --- | --- |
| Explain factoring versus solving | Same A acting through simpler matrices; column-vector order right to left; triangular dependency diagram | Recover a small triangular solution and diagnose reversing factors |
| Carry out pivoted elimination and reuse factors | A=[[2,1],[4,3]], b=[5,11]; row swap, multiplier and upper system; linked augmented matrix and stored L/P | Changed right-hand side; zero first pivot on invertible matrix; reconstruction and residual |
| Derive an orthogonal basis and least-squares solution | Two independent columns in 2D for Gram-Schmidt geometry; tall four-point fit for residual orthogonality | New observation; dependent-column rejection; QR versus direct least-squares reference |
| Construct and validate a covariance factor | C=[[4,2],[2,2]], L=[[2,0],[1,1]]; shared independent components and transformed unit directions | Derive variance/covariance by hand; rho=±1 PSD boundary; symmetry validation |
| Explain SVD, rank and approximation | Analytic A=R(alpha) diag(s1,s2) R(beta)^T; explicit input direction → stretch → output; rank truncation on retained ratings fixture | Remove a singular term; squared/Frobenius error and nonunique sign conventions; missing-entry caveat |
| Choose and assess a numerical method | Shape, assumptions, reconstruction/orthogonality/residual/conditioning as distinct checks; rank-deficient minimum-norm solution | Method selection with counterexamples; tiny residual with inaccurate coefficients |

Core includes the mechanisms and one worked calculation for all four methods, complete reproducible code and independent feedback. Deeper branches: Householder reflections, full/reduced shapes, truncated pseudoinverse and tolerance, log determinant/whitening, conditioning versus backward error, work/storage assumptions. Do not describe rotations when reflections or rectangular isometries are also possible. Compression and denoising are conditional objectives, not automatic benefits of discarding small directions.

## Visual contracts

- Elimination investigation: bounded 2×2 presets, including row-swap and singular cases. Show equations, augmented values and multipliers; the current factorization obeys PA=LU. The recorded transform of b must match the row operations. Distinguish zero pivot from proof that the original matrix is singular before pivot search.
- QR geometry: editable bounded second column, fixed first column [1,1]; show projection and remaining perpendicular component at equal aspect ratio, with exact numeric data. Parallel columns must produce a meaningful dependent state, not NaN arrows. The geometric construction is an explanation, not a production recommendation for classical Gram-Schmidt.
- Covariance investigation: parameter rho in [-1,1], C=[[4,2rho],[2rho,1]], L=[[2,0],[rho,sqrt(1-rho²)]]. Distinguish algebraic square root at PSD endpoints from the strictly positive-definite input required by ordinary Cholesky. Deterministic geometric outlines are not Monte Carlo samples or probability contours.
- SVD direction investigation: construct factors analytically, never pretend to numerically solve arbitrary SVDs. Linked input point, rotated/reflected basis, singular scaling, output and rank-one reconstruction share state. Zero and equal singular values are supported; sign/nonuniqueness and approximation errors are explained.
- Inline diagrams: triangular dependency, projection residual, factor dimension chain and singular outer-product sum. Use figures at the mechanism's introduction, not only in the final lab region. Exact-value tables accompany geometric views. Each plot has axes, equal scale, provenance and accessible explanation.

## Sources checked and boundaries

Research date: 10 September 2026. These sources verify particular claims; original examples, prose, numerical derivations and SVG geometry are authored for this lesson. Current online NumPy pages identify v2.5; the local execution snapshot is NumPy 2.3.5 on Python 3.12.14. Do not claim local use of the newest release.

| Source / locator | What was read and used |
| --- | --- |
| [NumPy QR](https://numpy.org/doc/stable/reference/generated/numpy.linalg.qr.html), parameters/returns/notes | Reduced/complete shapes, orthonormal columns, LAPACK interface. QR factors need not have positive diagonal; validate reconstruction, not an exact signed factor printout. |
| [NumPy SVD](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html), parameters/returns/notes | Reduced factor shapes, sorted singular values, conjugate transpose for complex matrices, stacked convention. Basic lesson uses real matrices. |
| [NumPy Cholesky](https://numpy.org/doc/stable/reference/generated/numpy.linalg.cholesky.html), contract/notes | SPD/Hermitian requirements; routine does not check symmetry and only uses the selected triangle. Validate the model input separately. |
| [NumPy solve](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html), notes | Square full-rank system, LAPACK gesv, 2.0 vector-shape convention. No claim of automatic SPD detection. |
| [NumPy lstsq](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html), returns/rcond | Minimum residual and minimum norm among ties; rank cutoff; empty residual-array caveat. Compute b-Ax explicitly for diagnostics. |
| [SciPy LU](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lu.html), identity/returns | SciPy returns A=P L U; classroom PA=LU uses the transposed permutation. This is a convention difference, not contradictory mathematics. SciPy is not installed in the local lesson environment, so no SciPy execution is claimed. |
| [MIT OCW SVD session](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/pages/positive-definite-matrices-and-applications/singular-value-decomposition/) and [five-page summary](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/d273f75ee2552a5c3c35ccab37e5edce_MIT18_06SCF11_Ses3.5sum.pdf) | Session resources plus full substantive summary read: paired input/output singular directions, null directions, eigenvalue relationship, sign correction in worked source example. Summary's isolated reversed row/column-space sentence is inconsistent with its final list; use dimensional identities to verify conventions. Video is a useful linked alternate with companion summary; no full-video viewing claimed. |
| [Cornell CS4220 least squares](https://www.cs.cornell.edu/courses/cs4220/2026sp/lec/2026-02-20.html), normal equations/economy QR/sensitivity | Projection, residual orthogonality, economy/full factor distinction and sensitivity to matrix and target. Use the geometric argument, not the source's incidental notation slips or Julia snippets as native evidence. |
| [Cornell CS6241 SVD and low-rank decompositions](https://www.cs.cornell.edu/courses/cs6241/2025sp/lec/2025-02-13.html), singular value decomposition section | Full/economy distinction, norm identities, low-rank optimality and centered-data interpretation. Entire relevant section read. Avoid its minimum/maximum wording slip: the minimization of a Rayleigh quotient selects the smallest eigenvalue. Our spectral-error proof and numeric fixtures verify the claim independently. |

## Verification plan

Execute every displayed standalone Python program and compare printed results. Check tiny hand calculations independently, factor reconstruction, triangular structure, orthogonality, solve residuals, covariance identities, rank/truncation and pseudoinverse conditions. Compare bounded browser models with independent NumPy results and invalid cases; do not treat reconstruction alone as accuracy of inferred coefficients. Test controls/reset, all figure quantities and limits, axes/labels and full reading order at desktop/mobile, keyboard, anchors and KaTeX rendering. Keep numerical tolerances tied to bounded fixtures. The final record will distinguish verified computation, browser inspection and user acceptance.
