export default {
  summary: 'Use random matrix products to discover a low-rank space, compress a least-squares problem or estimate an operator trace, while preserving the correct error and probability contracts.',
  outcomes: ['Trace a weighted-column probe and explain cancellation or redundant observed directions', 'Derive a compressed randomized SVD with correct shapes and rank handling', 'Separate exact rank limits, range error and truncation error', 'Choose oversampling and stabilized subspace iteration while counting data passes and storage', 'Evaluate a sketched fit on original data and derive a residual guarantee from an embedding', 'Explain row leverage and the rescaling required for nonuniform sampling', 'Derive a random-sign trace expectation and distinguish variance from a per-run error bound', 'Design fresh residual validation and assess application-specific consequences'],
  prerequisites: ['Matrix Decompositions (SVD, QR, Cholesky, LU)', 'Vectors, Matrices & Tensor Operations'],
  sequence: ['Probe a column space', 'Compress and lift a factorization', 'Account for approximation error', 'Stabilize iteration and count passes', 'Preserve a least-squares residual space', 'Estimate a trace through signs', 'Validate the original task', 'Practise on changed inputs and counterexamples'],
  visual: {
    type: 'Weighted-vector geometry, singular-value/error budgets, influential observation selection and actual running trace estimates',
    question: 'What did the sketch preserve, what did it miss, and does the smaller answer satisfy the original objective?',
    interaction: 'Choose three column weights; vary spectrum/rank/oversampling/iterations/seed on bounded six-dimensional matrices; select observations and compare fitted lines; advance independent-sign trace probes.'
  },
  practice: {
    task: 'Compare a new noisy three-factor signal with the exact rank floor; repair training-probe validation; construct unbounded original residual despite a zero sketch objective; reason about finite sign averages.',
    success: 'Correct shapes, original-space residuals, explicit scope of probability statements, independent error checks and meaningful reasons for parameter changes.'
  },
  misconceptions: ['A zero probe proves the matrix is zero', 'Probe width equals final rank', 'Expected error bounds hold for every seed', 'More iteration can beat the exact rank-k floor', 'One final normalization always repairs unstable powers', 'A small sketch residual certifies a small original residual', 'Unbiased Gram matrices imply unbiased fitted coefficients', 'Every new trace sample reduces error', 'Reusing training probes provides independent validation'],
  sources: ['https://arxiv.org/pdf/0909.4061', 'https://arxiv.org/pdf/2002.01387', 'https://scikit-learn.org/stable/modules/generated/sklearn.utils.extmath.randomized_svd.html', 'https://databookuw.com/page-2/page-4/'],
  depth: 'core',
  designRecord: 'docs/teaching/RANDOMIZED-LINEAR-ALGEBRA-DESIGN.md',
  reviewFocus: 'Distinguish Gaussian range projection from rank-k truncation, actual calculated errors from benchmarks, and original residual guarantees from coefficient inference. Check numerical zero/rank loss, finite-input controls, all source links, narrow equations and original next-topic sequence. One-pass, Krylov, CUR/Nyström and preconditioned solver branches are orientation rather than complete implementations.'
};
