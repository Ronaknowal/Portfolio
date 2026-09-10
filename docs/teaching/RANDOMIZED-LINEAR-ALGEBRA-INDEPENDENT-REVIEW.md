# Randomized Linear Algebra — independent mathematical review

Reviewed 2026-09-10 by the agent that did not author this lesson. This is a bounded correctness review, supplementing the author's visual/model/native evidence. The author owns integrated browser and production checks.

## Scope and outcome

Read the complete lesson draft and the corrected published source at `src/learn/data/topics/randomized-linear-algebra.jsx`, all eleven standalone NumPy programs, the browser models, and the relevant lab readouts. No remaining material mathematical defect was found in this review. The review does not claim a proof of every numerical implementation or a probability guarantee from sampled cases.

The distinction between a sampled range of width at most k+p and a final approximation of rank at most k is explicit and correct. The Frobenius error decomposition uses orthogonal components. Flat spectra illustrate the irreducible rank constraint; power iterations improve separation when one exists. The standard 2q+2 pass budget is labeled as the standard algorithm's count, with the zero-matrix feedback explicitly permitting a short circuit. The blocked program is accurately called two-pass processing.

The residual embedding argument requires preservation throughout span([A,b]), applies the same sketch to features and responses, and derives the norm factor from squared-length inequalities. The lesson does not confuse an expected squared length with a uniform subspace guarantee. Leverage scores use an orthonormal column-space basis and the row example's full-rank/nonconstant-feature conditions; sampling weights produce an unbiased Gram estimate, with no unwarranted claim about an unbiased coefficient estimate. The preconditioning branch continues to minimize the original residual.

The Rademacher trace estimator's expectation and symmetric-matrix variance are correct. Its discussion distinguishes variance across runs from monotonically improving realized error, and handles signed/zero traces without applying a relative-error conclusion. Fresh residual probes are independent of construction probes; the page properly distinguishes an unbiased squared-norm estimate from the square root and from a confidence certificate.

## Concrete finding and resolution

The reviewed draft's last practice question asked whether a **fourth** sample could worsen a preceding estimate of 8, while its solution used the sequence [4,12,12]. For the stated matrix, a single probe is either 4 or 12; a three-sample mean cannot equal 8. The author corrected the question to a separate run and “another sample,” preserving the valid two-to-three-sample counterexample. The actual lesson and draft were reread after that correction. The native review explicitly enumerates all three-probe possibilities to verify this distinction.

No source edits were made by this reviewer. The theorem notation, masking, or layout of unrelated lessons was not changed.

## Independent native evidence

Run from the repository root:

```powershell
scratch/lesson-tools/Scripts/python.exe scripts/review-randomized-linear-algebra-native.py
```

Actual result: **passed**, 2026-09-10 09:09:19 UTC, NumPy 2.3.5. Machine-readable results: `scratch/randomized-independent-review/results.json`.

The review script imports the current JavaScript examples, extracts the Python function definitions with Python's AST, and executes those actual definitions. It does not copy the helper into the verifier. All four standalone helper copies have identical normalized ASTs.

- **1,296 helper cases:** seven square/tall/wide shapes, including one-row and one-column inputs; zero, rank-one, intermediate-rank, and full-rank matrices; three scales from 10^-80 to 10^80; rank/oversampling settings; zero, one, and three power iterations.
- **891 rank-deficient or zero cases:** output dimensions and observed rank are correct, without fabricating arbitrary completion directions. A separate deliberately zero-valued probe generator checks empty observed-range behavior for a nonzero matrix; it is an edge contract, not a claim about a Gaussian draw.
- **1,161 full-range cases:** the captured projection reproduces the known matrix range and the final error reaches the prescribed singular-spectrum rank-k floor within tolerance.
- Across all helper cases, checked orthonormality, factor shapes, B=QᵀA, best-rank error lower bound, error relative to the zero approximation, and the spectrum of the projected full rectangular matrix. The largest absolute discrepancy in the normalized squared-error decomposition was **5.33×10^-14**.
- **13 rejected input cases:** empty/one-dimensional/nonfinite/complex matrices, out-of-range ranks, negative iteration/oversampling values, and fractional parameters.
- **24 independent embedding cases:** arbitrary tall feature matrices, including dependent columns, with explicit controlled distortions of the complete residual space; independently solved both least-squares problems and checked each inequality in the displayed proof.
- **48 trace matrices:** exhaustive enumeration of all sign vectors for dimensions 2, 3, and 5 verifies expectation for arbitrary real matrices and the stated variance for symmetric matrices without Monte Carlo uncertainty.

Oracle independence comes from known prescribed singular values, full-rectangle SVD of the projection, original/sketched least-squares solves under a deliberately constructed embedding, and exhaustive finite sign sums. These checks are additional to, and do not repeat, the author's fixed browser-fixture comparisons.

## Source verification and boundaries

Checked Theorem 10.5 and its nearby assumptions directly in [Halko, Martinsson and Tropp, *Finding Structure with Randomness*](https://arxiv.org/pdf/0909.4061), printed pages 56–57 / PDF pages 56–57: real input, standard Gaussian sketch, k≥2, p≥2, and k+p≤min(m,n). The lesson's displayed expectation bound and its restriction to the range projector match the theorem. No further long passages are reproduced here.

This independent review did not watch the linked videos or rerun the author's browser suite. It checked finite, controlled numerical fixtures and mathematical claims; it does not establish performance across all finite inputs, numerical-rank choices, random streams, or hardware. Near-threshold numerical rank and the distinction between browser and NumPy random streams are disclosed in the lesson.
