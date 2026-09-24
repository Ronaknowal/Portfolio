export default {
  summary: 'Find directions preserved by a square map, use a valid eigenbasis to separate changes, and interpret spectra in repeated updates, data variation and physical modes.',
  outcomes: ['Derive and verify a small eigenpair through a shifted null space', 'Distinguish signed scales, zero output, repeated eigenspaces and a defective basis', 'Use diagonalization with its correct inverse and identify symmetric structure', 'Separate eventual decay, finite transient growth and initial-mode participation', 'Derive a variance-maximizing direction and its reconstruction error', 'Choose an appropriate numerical routine and interpret residuals, gaps and failed iteration starts'],
  prerequisites: ['Vectors, Matrices & Tensor Operations'],
  sequence: ['Preserve a line under a transformation', 'Derive a characteristic equation and its null spaces', 'Change to an eigenbasis and examine exceptions', 'Trace repeated updates and stability boundaries', 'Measure and retain variance along a direction', 'Interpret mixing, graph smoothing and coupled vibration modes', 'Compute, verify and diagnose an eigenpair', 'Solve independent changed-matrix and interpretation tasks'],
  visual: {
    type: 'Equal-scale direction geometry, raw recurrence trajectories and projected point clouds',
    question: 'Which part of a change stays along a direction, and what does its signed scale mean in this model?',
    interaction: 'Turn a unit input across six maps; compare six repeated-update rules and three initial vectors; project three small datasets while calculating variance and squared error.'
  },
  practice: {
    task: 'Find a changed spectrum and rank-one approximation; repair a stability claim; construct a new mixing rule; interpret a tied covariance variance budget.',
    success: 'Correct original-coordinate checks, explicit assumptions and units, consistent variance/error denominators, and explanations of what numerical and application results do not establish.'
  },
  misconceptions: ['Zero is an eigenvector because it satisfies the equation', 'Negative scales or zero outputs cannot be eigenvalues', 'Every square matrix has a real eigenbasis', 'Repeated eigenvalues always prevent diagonalization', 'The inverse of any eigenvector matrix is its transpose', 'Eigenvalues inside the unit disk guarantee decreasing norm at every step', 'A small residual certifies the dominant eigenpair', 'Leading covariance directions are necessarily meaningful signal', 'Discrete and continuous dynamics share the same decay test'],
  sources: ['https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html', 'https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html', 'https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/pages/least-squares-determinants-and-eigenvalues/eigenvalues-and-eigenvectors/', 'https://www.cs.cornell.edu/courses/cs4220/2026sp/lec/2026-03-04.html', 'https://ee263.stanford.edu/archive/eig.pdf'],
  depth: 'core',
  designRecord: 'docs/teaching/EIGENVALUES-EIGENVECTORS-DESIGN.md',
  reviewFocus: 'Keep the nonzero definition, eigenvalue/vector pairing, field and multiplicity distinctions, quantitative variance fixture, stability exceptions and numerical-contract boundaries. Derived visuals must match the independently checked model and use equal geometric scales. Preserve module position and actual next Jacobian topic.'
};
