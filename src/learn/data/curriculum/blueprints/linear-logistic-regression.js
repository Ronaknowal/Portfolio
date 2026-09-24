export default {
  summary: 'Learn a numerical prediction or a binary probability from visible observations, then evaluate a frozen pipeline against a baseline without leaking future information.',
  outcomes: [
    'Define an observation, prediction-time features, target and train/validation/test roles',
    'Calculate residuals and losses, derive scalar least squares and explain matrix rank and projection',
    'Trace gradient updates and derive the safe fixed-step condition for a stated quadratic',
    'Convert scores, odds and probabilities and derive stable logistic loss and gradients',
    'Separate probabilistic fitting, weighting, threshold selection, calibration and decision cost',
    'Diagnose nonunique coefficients, separation, scaling, leverage and uncertain extrapolation',
    'Explain ridge/lasso differences, nonlinear feature maps and matched solver conventions',
    'Run independent complete examples and report held-out probability and decision performance',
  ],
  prerequisites: ['Vectors, Matrices & Tensor Operations', "Probability Distributions & Bayes' Theorem"],
  sequence: ['Define shipment information and a split protocol', 'Inspect each residual against a mean baseline', 'Derive scalar and matrix least squares', 'Follow an actual parameter-space update', 'Translate affine score into probability and log loss', 'Choose an action under stated costs', 'Investigate separation and penalties', 'Express nonlinear patterns through features', 'Run a held-out pipeline and diagnose uncertainty', 'Practise changed cases before Decision Trees & Random Forests'],
  visual: {
    type: 'Prediction-time information map, residual geometry, coefficient-space contours, score/sigmoid correspondence, threshold gate and separation objective',
    question: 'Which values change when the data, model parameters, training rule or decision threshold change?',
    interaction: 'Move a fit, step exact gradient updates, inspect one logistic observation, change a decision gate or penalty and connect numeric outputs with the same plotted model.',
  },
  practice: {
    task: 'Calculate changed fits, losses and updates, diagnose rank/separation/leakage, derive a transformed boundary and produce a reproducible baseline-versus-model held-out report.',
    success: 'The score, probability and decision meanings stay distinct; assumptions and units are explicit; all learned transforms use training data; final test labels never select settings.',
  },
  misconceptions: ['Normal noise is necessary for defining squared-loss optimization', 'A pseudoinverse identifies unique true coefficients from rank-deficient data', 'Convexity guarantees a finite optimizer or every learning rate converges', 'A sigmoid makes probabilities calibrated or prevents large-feature gradients', 'Class weighting and threshold changes preserve the same probability interpretation', 'Ridge, lasso and elastic net share an inverse formula', 'A linear model cannot use nonlinear supplied features', 'A narrow model interval certifies extrapolation', 'A fixed solver size cutoff or one test score establishes a universal winner'],
  sources: ['https://cs229.stanford.edu/main_notes.pdf', 'https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html', 'https://scikit-learn.org/stable/modules/linear_model.html', 'https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html', 'https://scikit-learn.org/stable/common_pitfalls.html', 'https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-16-projection-matrices-and-least-squares/'],
  depth: 'core',
  designRecord: 'docs/teaching/LINEAR-LOGISTIC-REGRESSION-DESIGN.md',
  reviewFocus: 'Same-data plots and outputs; rank/conditioning versus numerical solve; finite logistic optimum and intercept exceptions; current penalty normalization and solver semantics; train-only fitting, held-out selection, uncertainty assumptions and accessible spatial explanations.',
};
