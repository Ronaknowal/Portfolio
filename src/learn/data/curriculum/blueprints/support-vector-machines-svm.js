export default {
  summary: 'Construct a margin classifier, verify its optimization certificate, choose a kernel and validation pipeline deliberately, and adapt the same ideas to a regression tolerance.',
  outcomes: [
    'Calculate signed scores, perpendicular distances and canonical margins without confusing their scale',
    'Derive hard and soft margin objectives, finite dual constraints and KKT support-vector cases',
    'Explain conditional support-point stability and distinguish unique weights from nonunique coefficients or bias',
    'Construct valid feature kernels, diagnose a Gram counterexample and calculate an XOR decision from support contributions',
    'Perform a legal two-coefficient update including zero curvature and assess a bounded solver with primal-dual evidence',
    'Execute fold-local scaling, model selection, calibration, multiclass and regression workflows with held-out baselines',
    'Explain target-unit changes, sparse/linear versus kernel costs, and the scope of explicit kernel approximations'
  ],
  prerequisites: ['Linear & Logistic Regression', 'Vectors, Matrices & Tensor Operations'],
  sequence: ['See a separating score and its true distance', 'Find the observations that constrain the margin', 'Pay for shortfalls and derive a dual certificate', 'Change similarity with a valid kernel', 'Optimize a feasible pair and inspect convergence', 'Select and calibrate a complete pipeline', 'Fit a regression tube and transfer to structured features', 'Diagnose changed data and complete independent practice'],
  visual: {
    type: 'Margin corridor and perpendicular projection, support-motion certificates, hinge/KKT states, XOR feature and kernel contributions, feasible pair geometry, measured validation selection and regression tubes',
    question: 'Which geometric constraint, loss or similarity makes this prediction, and what evidence certifies the fit or supports using it on a new observation?',
    interaction: 'Change bounded boundary, support-point, penalty, kernel, pair-coordinate and tube states; inspect linked exact scores, constraints, objectives and actual fitted validation evidence.'
  },
  practice: {
    task: 'Repair changed margin, kernel, optimization and preprocessing cases, then document an independently selected and held-out classification or regression experiment.',
    success: 'Calculations and witnesses agree; scale, support conditions, kernel validity, convergence status, target units and data-role boundaries are explicit.'
  },
  misconceptions: ['An SVM raw score is always a distance or probability', 'Any zero-coefficient observation can move anywhere without changing the solution', 'Every tight point has a positive coefficient', 'All support vectors implies underfitting', 'A PSD kernel guarantees unique coefficients and intercept', 'An indefinite quadratic on a finite box is automatically unbounded', 'A random inactivity counter certifies convergence', 'Zero pair curvature permits silently skipping a necessary update', 'SVC and LinearSVC use identical losses and intercept regularization', 'Scaling before cross-validation is harmless', 'Calibration guarantees good probabilities after arbitrary model selection', 'C and epsilon remain equivalent when regression target units change', 'Kernel approximations have no feature-construction cost'],
  sources: ['https://www.microsoft.com/en-us/research/wp-content/uploads/1998/04/sequential-minimal-optimization.pdf', 'https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf', 'https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf', 'https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html', 'https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html', 'https://cseweb.ucsd.edu/~eeskin/papers/spectrum-psb02.pdf'],
  depth: 'specialist',
  reviewFocus: 'Canonical versus geometric margins, finite C and duplicate degeneracy, KKT boundary cases and bias interval, PSD versus strict curvature/uniqueness, atomic feasible pair updates and zero curvature, actual 1.9.1 calibration migration and fold-local preprocessing, sum-loss normalization, regression target scaling, independently executed examples and purposeful topic-native visuals.'
};
