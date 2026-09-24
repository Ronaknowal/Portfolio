export default {
  summary: 'Update uncertainty with new evidence, carry it into predictions and decisions, and test the assumptions behind useful conjugate probability models.',
  outcomes: [
    'Normalize a posterior and derive the Beta update with explicit support and evidence assumptions',
    'Distinguish credible probability, point summaries, predictive outcomes and repeated-sampling coverage',
    'Derive Beta-Binomial prediction and explain dependence induced by a shared uncertain rate',
    'Update Gamma event rates using exposure and Normal means using measurement precision',
    'Interpret Dirichlet categorical probabilities and their finite-support constraint',
    'Use prior and posterior predictive checks to reveal patterns sufficient totals discard',
    'Retain marginal-likelihood constants and connect posterior uncertainty to a declared loss',
    'Recognize nonconjugate, improper-prior and model-mismatch limits and the need for later computation',
  ],
  prerequisites: ["Probability Distributions & Bayes' Theorem"],
  sequence: ['Model and evidence', 'Beta derivation and credible uncertainty', 'Shared-rate batch prediction', 'Sequential updates and sensitivity', 'Exposure and Gaussian precision', 'Categorical and deeper conjugacy', 'Predictive model checking', 'Evidence and decisions', 'Independent transfer'],
  visual: {
    type: 'Prior/posterior density, evidence lanes, predictive count mass, exposure strips, precision intervals and replicated sequence patterns',
    question: 'Which quantity is uncertain, what observation supplies new information, and what consequences survive averaging over that uncertainty?',
    interaction: 'Change meaningful prior/data/measurement assumptions, compare parameter and outcome distributions, and challenge a common-rate model with the same totals but a different pattern.',
  },
  practice: {
    task: 'Derive changed updates and predictions, diagnose reused data and model mismatch, compare exposure/precision units and make a loss-based choice.',
    success: 'Correct normalized quantities, justified assumptions and intervals, independent numerical checks, and a clear explanation of what the model does not establish.',
  },
  misconceptions: ['Conjugacy validates the model', 'Prior parameters are actual observations', 'Posterior means determine predictive variance', 'Every observation narrows uncertainty', 'Likelihood and posterior are interchangeable', 'Credible and confidence intervals have the same interpretation', 'Event counts suffice without exposure', 'No observed category means impossible future category', 'A posterior predictive check is a classical calibrated p-value', 'An improper prior automatically gives valid evidence'],
  sources: ['https://statproofbook.github.io/P/bin-post.html', 'https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf', 'https://mc-stan.org/docs/stan-users-guide/posterior-predictive-checks.html', 'https://stat110.hsites.harvard.edu/youtube'],
  depth: 'core',
  designRecord: 'docs/teaching/BAYESIAN-INFERENCE-CONJUGATE-DESIGN.md',
  reviewFocus: 'Support, conditional independence, proper normalization, parameter versus prediction, direct tails, observation units, sensitivity and finite/integral evidence.',
};
