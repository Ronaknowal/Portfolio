export default {
  summary: 'Use simpler matrix factors to solve equations, fit inconsistent observations, construct covariance and control low-rank approximation error.',
  outcomes: ['Trace elimination and solve PA=LU through two triangular systems', 'Construct perpendicular directions and justify a QR least-squares solution', 'Derive and validate a small Cholesky factor and its covariance interpretation', 'Follow SVD input/output directions and calculate rank-truncation error', 'Distinguish factor reconstruction, residual, numerical rank and conditioning'],
  prerequisites: ['Vectors, Matrices & Tensor Operations', 'Eigenvalues & Eigenvectors'],
  sequence: ['Read triangular, orthogonal and diagonal factors as operations', 'Eliminate equations, pivot and reuse LU factors', 'Remove shared column directions and project a noisy target', 'Construct correlated variation and diagnose covariance boundaries', 'Decompose independent stretches, approximate rank and inspect error', 'Diagnose numerical sensitivity and rank-deficient solutions', 'Solve independent calibration, covariance and rank-budget tasks'],
  visual: {
    type: 'linked factor operations with domain-specific geometric investigations',
    question: 'What does each factor change, and which property makes the resulting task easier?',
    interaction: 'Step pivoted equations; remove a projected column component; vary correlation; follow a vector through analytically constructed singular factors and truncate a direction.'
  },
  practice: {
    task: 'Fit a changed calibration dataset, repair row operations, construct a covariance factor and choose a rank from an error budget.',
    success: 'Hand reasoning agrees with independent numeric checks; the learner states shape, assumptions, residual/error meaning and a changed-condition consequence.'
  },
  misconceptions: ['An invertible matrix can have a zero first pivot before row reordering', 'A nonzero least-squares residual is not automatically an incorrect solve', 'A valid covariance can be semidefinite and fail ordinary Cholesky', 'A successful Cholesky call does not itself check both triangles for symmetry', 'SVD factors can change signs or tied bases without changing the matrix', 'Low rank does not automatically mean storage savings or removal of noise', 'Small residuals do not prove insensitive or accurately estimated parameters'],
  sources: ['https://numpy.org/doc/stable/reference/generated/numpy.linalg.qr.html', 'https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html', 'https://numpy.org/doc/stable/reference/generated/numpy.linalg.cholesky.html', 'https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html', 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.lu.html', 'https://www.cs.cornell.edu/courses/cs4220/2026sp/lec/2026-02-20.html', 'https://www.cs.cornell.edu/courses/cs6241/2025sp/lec/2025-02-13.html', 'https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/pages/positive-definite-matrices-and-applications/singular-value-decomposition/'],
  depth: 'core',
  reviewFocus: 'Full individual design and claim ledger: docs/teaching/MATRIX-DECOMPOSITIONS-LESSON-DESIGN.md. Keep the local eigenvalue bridge because the recorded prerequisite is later in module order. Review shape/permutation conventions, numerical versus exact rank, original example outputs, meaningful visuals, and production isolation.'
};
