export default {
  summary: 'Derive and inspect Newton, limited-memory secant, Fisher and tensor-statistic updates, with explicit approximation, stability and computational contracts.',
  outcomes: [
    'Solve a Newton system and justify when it gives a descent direction',
    'Diagnose indefinite curvature and distinguish damping from step acceptance',
    'Trace BFGS secant information and the two-loop L-BFGS update',
    'Run a complete repeatable PyTorch L-BFGS closure-based fit',
    'Derive a Bernoulli natural direction and explain its local coordinate invariance',
    'Compute exact and Kronecker-factored Fisher blocks with explicit vectorization',
    'Calculate original Shampoo row/column accumulators and spectral inverse roots',
    'Compare storage, matrix-vector work and experiment quality without unsupported rankings',
  ],
  prerequisites: ['Matrix Decompositions (SVD, QR, Cholesky, LU)', 'Eigenvalues & Eigenvectors', 'Multivariate Calculus & Gradients', 'Gradient Descent Variants (SGD, Adam, AdaGrad, RMSProp, LAMB, LARS)'],
  sequence: ['Unequal curvature', 'Newton and safeguards', 'Secant history and L-BFGS', 'Probability geometry', 'K-FAC factorization', 'Shampoo tensor statistics', 'Practical scaling', 'Independent practice'],
  visual: {
    type: 'Calculated contour geometry, directional models, history transformations, probability bars and matrix-factor correspondence',
    question: 'Which information changes the direction, which approximation was made, and what can the resulting step actually guarantee?',
    interaction: 'Inspect bounded geometric trajectories, step through retained secant pairs, compare coordinate updates and calculate factor-based transforms.',
  },
  practice: {
    task: 'Solve changed quadratic and secant fixtures, diagnose an unsafe step, compare finite natural updates, expose a factorization error and verify a tensor-statistic transform.',
    success: 'Correct intermediate calculations and residuals, explicit assumptions and a reproducible implementation with fair comparison criteria.',
  },
  misconceptions: ['Every second-order method stores a Hessian', 'A Newton direction always descends', 'Positive definite damping guarantees a full step decreases loss', 'L-BFGS stores past parameter vectors without gradient differences', 'A true Fisher equals the observed-label gradient outer product', 'Natural-gradient finite updates are identical in every nonlinear parameterization', 'K-FAC factorization is an exact independence identity', 'Shampoo matrix powers act entrywise', 'Fewer updates proves better wall-clock performance'],
  sources: ['https://proceedings.mlr.press/v80/gupta18a/gupta18a.pdf', 'https://proceedings.mlr.press/v37/martens15.pdf', 'https://arxiv.org/pdf/1412.1193', 'https://pages.cs.wisc.edu/~yudongchen/cs726_sp25/Lecture_23_L-BFGS.pdf', 'https://docs.pytorch.org/docs/2.14/generated/torch.optim.LBFGS.html', 'https://see.stanford.edu/Course/EE364A/79'],
  depth: 'core',
  designRecord: 'docs/teaching/SECOND-ORDER-METHODS-DESIGN.md',
  reviewFocus: 'Check signs, positive-definiteness conditions, finite-step/local distinctions, secant ordering, expectation and vectorization conventions, spectral powers, native outputs, independent practice and readable narrow-screen geometry.',
};
