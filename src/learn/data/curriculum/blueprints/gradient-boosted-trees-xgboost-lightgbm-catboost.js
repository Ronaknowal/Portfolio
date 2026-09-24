export default {
  summary: 'Build an additive tree predictor from calculated loss corrections, derive regularized split decisions, and choose and validate XGBoost, LightGBM or CatBoost with explicit data and inference contracts.',
  outcomes: [
    'Calculate a correction tree and distinguish its training residuals from the additive prediction made for a new row',
    'Derive half-squared and binary log-loss gradients, quadratic leaf weights and L1/L2/leaf-count split costs with stated assumptions',
    'Explain histogram thresholds, missing routes, growth policies, row sampling and exclusive feature bundling through concrete examples',
    'Separate ordered target statistics from ordered boosting and diagnose whose target can enter a representation or training prediction',
    'Execute current CPU library workflows with a baseline, training/validation/test separation and correct best-round inference',
    'Diagnose extrapolation, leakage, shape, imbalance, importance and resource tradeoffs without unsupported library rankings'
  ],
  prerequisites: ['Decision Trees & Random Forests', 'Linear & Logistic Regression'],
  sequence: ['Predict with a sequence of corrections', 'Fit residual leaves and prove the squared-loss update', 'Change the loss and derive regularized split gain', 'Inspect histogram, missing-data and growth mechanisms', 'Understand GOSS and categorical information boundaries', 'Select, stop and run the three libraries honestly', 'Diagnose failures and complete changed-data practice'],
  visual: {
    type: 'Linked residual stems and additive prediction, gradient/Hessian split balance, raw-to-bin routing, tree growth topology, finite reweighting and ordered-prefix information flow, actual validation curves',
    question: 'What changes in the predictor, which information permits that change, and what evidence supports using it on a new row?',
    interaction: 'Change bounded observations, loss/regularization, split or histogram boundaries, sample/prefix states and training settings; inspect recalculated structure, predictions, objectives and validation evidence.'
  },
  practice: {
    task: 'Calculate and repair changed boosting updates, split choices, sampling/encoding information flow and best-round inference, then report a complete baseline-to-held-out experiment.',
    success: 'Independent calculations agree; training versus validation roles, surrogate versus actual loss, current library contracts, shapes and measured versus unsupported claims remain explicit.'
  },
  misconceptions: ['A correction tree predicts the target directly', 'A learning rate of one fixes every residual or inevitably overfits', 'Binary probabilities can be added like raw scores', 'Gamma is the L1 penalty on leaf weights', 'A positive Newton surrogate gain guarantees the same decrease in actual loss', 'Histogram search preserves every raw threshold', 'GOSS makes nonlinear selected gain unbiased', 'CatBoost eliminates every kind of target leakage or makes every gradient have expectation zero', 'XGBoost and LightGBM require manual categorical encoding', 'A test set may choose the stopping iteration', 'Native and wrapper best-iteration predictions always use the same default', 'A normalized feature importance has the same meaning in all libraries'],
  sources: ['https://arxiv.org/pdf/1603.02754', 'https://papers.nips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf', 'https://papers.nips.cc/paper_files/paper/2018/file/14491b756b3a51daac41c24863285549-Paper.pdf', 'https://xgboost.readthedocs.io/en/stable/prediction.html', 'https://lightgbm.readthedocs.io/en/stable/Parameters.html', 'https://catboost.ai/docs/en/references/training-parameters/common'],
  depth: 'specialist',
  reviewFocus: 'Preserved original programs with evidenced corrections; half-square convention, projection identity, net gamma and soft threshold; logistic Hessian approximation; finite sampling inclusion weights; own-target/prefix assumptions; current categorical/device/stopping/shape contracts; no fabricated benchmark curves; complete native and ordinary-reading visual evidence.'
};
