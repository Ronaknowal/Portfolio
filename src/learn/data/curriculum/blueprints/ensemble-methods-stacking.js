export default {
  summary: 'Combine fitted predictions with a clear data boundary, explain error cancellation and sequential reweighting, and build an honestly evaluated stacking workflow.',
  outcomes: [
    'Distinguish numerical forecasts, probability columns, margins and ballots before combining models',
    'Derive weighted-error and covariance identities with their assumptions and limits',
    'Trace bootstrap row multiplicity into fitted rules and row-specific out-of-bag eligibility',
    'Fit a weighted stump, derive a signed AdaBoost update and distinguish training bounds from generalization',
    'Construct each out-of-fold meta-feature using legal training rows and interpret full-data refitting',
    'Implement complete classifier stacking and forward-only or group-safe alternatives; explain a holdout blend’s data tradeoff',
    'Diagnose copied errors, uncalibrated averages, leakage and context-dependent weighting limits',
    'Compare a frozen ensemble with individual and simple baselines using actual held-out results and explicit serving constraints'
  ],
  prerequisites: ['Linear & Logistic Regression', 'Decision Trees & Random Forests'],
  sequence: [
    'Name the prediction unit and reserve evaluation data',
    'Calculate weighted forecasts, ballots and probability averages',
    'Explain and fit error cancellation without a best-model guarantee',
    'Trace bootstrap draws, fitted rules and OOB eligibility',
    'Work through weighted stumps and derive AdaBoost updates and bounds',
    'Build an out-of-fold matrix and apply a fully refitted stack',
    'Handle actual API shapes, preprocessing, blending, groups and time',
    'Inspect measured prediction disagreements and a calibration counterexample',
    'Evaluate a controlled complete workflow and context-dependent combinations',
    'Practise changed cases and continue to Recommender Systems'
  ],
  visual: {
    type: 'Signed residual rows, bootstrap row/fit correspondence, weighted-stump geometry, OOF ownership matrix and actual prediction regions',
    question: 'Which errors can this combination correct, and which training rows were allowed to create each prediction?',
    interaction: 'Change a blend weight, inspect bootstrap membership, step an actual weighted stump, construct held-out rows and compare fixed fitted predictions while linked values show the mechanism.'
  },
  practice: {
    task: 'Calculate changed weights and training bounds, diagnose data ownership and calibration mistakes, and produce a reproducible changed held-out ensemble report.',
    success: 'Each calculation states its probability/error meaning and assumptions; every meta-feature has valid provenance; the report includes baselines and does not convert a finite gain into a universal guarantee.'
  },
  misconceptions: [
    'Bagging never changes bias and stacking always beats its best member',
    'Prediction correlation alone determines classification accuracy or justifies a fixed pruning threshold',
    'Independent-majority guarantees apply to repeated copies of one model',
    'Averaging calibrated probabilities necessarily preserves calibration',
    'Weighted training examples and model vote weights are the same quantity',
    'Every weak error below one half establishes a uniform exponential rate',
    'Out-of-fold predictions make preprocessing and outer evaluation unnecessary',
    'StackingClassifier.transform on training data returns the OOF training matrix',
    'A plain linear combiner automatically changes model weights by input region',
    'TimeSeriesSplit can always be passed directly to a partition-based stacking API'
  ],
  sources: [
    'https://www.stat.berkeley.edu/~breiman/bagging.pdf',
    'https://www.schapire.net/papers/explaining-adaboost.pdf',
    'https://cs229.stanford.edu/extra-notes/boosting.pdf',
    'https://doi.org/10.1016/S0893-6080(05)80023-1',
    'https://arxiv.org/pdf/0911.0460',
    'https://scikit-learn.org/stable/modules/ensemble.html',
    'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html'
  ],
  depth: 'core',
  designRecord: 'docs/teaching/ENSEMBLE-METHODS-LESSON-DESIGN.md',
  reviewFocus: 'Data ownership, exact loss/covariance assumptions, calibration versus decisions, binary versus SAMME conventions, actual OOF/full-fit shapes, source-backed quantitative figures and complete controlled evaluation.'
};
