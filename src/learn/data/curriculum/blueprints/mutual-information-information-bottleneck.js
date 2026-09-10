export default {
  summary: 'Measure what a joint distribution reveals, trace information through a representation, and build a task-aware bottleneck while distinguishing exact information from optimization surrogates and finite-sample estimates.',
  outcomes: [
    'Calculate finite mutual information from joint, marginal and conditional probabilities with explicit log units and support conventions',
    'Explain conditional information, nonlinear dependence and feature synergy using changed finite examples',
    'Derive the data-processing inequality from its exact Markov and chain-rule assumptions',
    'Compare representations by input information, task relevance and a correctly interpreted bottleneck objective',
    'Execute and diagnose finite stochastic information-bottleneck updates without mistaking stationarity for global optimality',
    'Derive variational rate and predictive bounds, calculate their gaps and translate objective coefficient conventions',
    'Distinguish finite discrete information from continuous, singular and infinite-information cases',
    'Diagnose estimation bias, sparse counts, leakage and limits of causal, privacy or generalization conclusions',
  ],
  prerequisites: ['Entropy, Cross-Entropy & KL Divergence', "Probability Distributions & Bayes' Theorem", 'Variational Inference'],
  sequence: ['Compare joint and independent laws', 'Reveal conditional and joint information', 'Trace a processing pipeline', 'Choose a task-relevant representation', 'Compute finite information-bottleneck updates', 'Inspect variational bounds', 'Handle continuous variables and finite samples', 'Practise changed representations and assumptions'],
  visual: {
    type: 'Joint-versus-product probability cells, conditional label bars, XOR reveal table, information plane, soft-assignment updates, variational bound gaps and sampled count grids',
    question: 'Which information was present, retained or lost, and which number is an exact property rather than a bound or estimate?',
    interaction: 'Change bounded channel probabilities, representation choices, bottleneck weights, finite update steps or a declared sampled count budget; inspect linked distributions and information quantities from one coherent state.',
  },
  practice: {
    task: 'Compute a changed information quantity, diagnose a Markov or estimator claim, select and optimize a small representation, and verify a variational bound with explained gaps.',
    success: 'Joint masses and conditional laws are valid; independent values agree; logarithm units, support, processing assumptions, objective conventions and finite-sample limits are stated accurately.',
  },
  misconceptions: ['Zero correlation proves independence', 'Individually uninformative features cannot help jointly', 'Conditioning always reduces mutual information', 'DPI applies even when the representation receives extra label information', 'The number of latent coordinates is mutual information', 'Large relevance weight requires preserving every nuisance detail', 'An IB stationary encoder is globally optimal', 'A variational KL penalty is exact input mutual information', 'A negative predictive lower bound means negative mutual information', 'Every continuous deterministic encoder has a finite differential-entropy difference', 'An estimated MI value certifies causation, privacy, fairness or generalization'],
  sources: ['https://arxiv.org/html/physics/0004057', 'https://arxiv.org/html/1612.00410v7', 'https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/pages/lecture-notes/'],
  depth: 'specialist',
  designRecord: 'docs/teaching/MUTUAL-INFORMATION-LESSON-DESIGN.md',
  reviewFocus: 'Finite probability and log-ratio stability; conditional MI and exact Markov assumptions; explicit stochastic-encoder information; IB block minimization and local limits; variational bound signs/constants; discrete versus non-atomic information; finite-sample estimation and scope of practical claims.',
};
