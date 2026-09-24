export default {
  summary: 'Choose and compute a distribution comparison by the differences it can detect, connecting probability-ratio penalties, constrained observers, kernel witnesses and finite-sample evidence.',
  outcomes: ['Calculate finite f-divergences with explicit normalization, log units and zero-support limits', 'Explain nonnegativity, shared-channel data processing and the distinction between divergences and metrics', 'Derive total variation as an event witness and an equal-prior classification advantage', 'Construct bounded, linear and Lipschitz observers and explain their different blind spots', 'Compute kernel MMD from feature means and pair similarities, with correctly normalized witnesses', 'Distinguish a nonnegative empirical-law norm from an unbiased squared estimator that may be negative', 'Carry out and interpret an exact finite permutation test under its exchangeability assumptions', 'Derive a variational f-divergence bound and diagnose restricted or fitted-critic limitations', 'Stress-test a metric choice against meaningful changes in support, geometry, representation and sampling'],
  prerequisites: ['Entropy, Cross-Entropy & KL Divergence', 'Optimal Transport (Wasserstein Distance, Sinkhorn)', 'Hypothesis Testing & Confidence Intervals'],
  sequence: ['Compare probability ratios', 'Handle support and processing', 'Derive events and metric conditions', 'Choose the observer class', 'Expose support versus geometry', 'Build kernel distribution witnesses', 'Separate estimates from calibrated tests', 'Connect variational critics and practical decisions', 'Practise changed laws and assumptions'],
  visual: {
    type: 'Aligned mass-and-penalty rows, coarsening flows, constrained critic scores, moving atoms, feature-mean geometry, kernel Gram blocks and witnesses, and exact permutation ranks',
    question: 'What mismatch can this comparison see, what does its numerical value mean, and what additional evidence is needed to make a decision?',
    interaction: 'Apply bounded distribution or sample edits, choose an observer or kernel, move a point mass, inspect a finite relabeling and compare a critic bound with its exact known-law target.'
  },
  practice: {
    task: 'Compute changed finite comparisons, prove or repair their assumptions, diagnose an invisible distribution change and build a reproducible small kernel test with an explained decision rule.',
    success: 'Calculations agree independently; support, constants, geometry, kernel class and estimator assumptions are explicit; statistical significance is separated from effect size and practical harm.'
  },
  misconceptions: ['Every divergence is symmetric or satisfies a triangle inequality', 'Zero numerical discrepancy proves equality regardless of function class', 'Finite samples always have a finite KL comparison', 'A smaller geometric displacement makes every divergence smaller', 'TV has the same factor for all bounded-function conventions', 'All kernels distinguish all probability laws', 'A negative unbiased MMD-squared estimate means a negative population distance', 'Choosing a kernel after seeing labels preserves an unchanged permutation test', 'A trained finite-sample critic automatically gives a population lower bound', 'One score certifies useful generation, no distribution shift or fairness'],
  sources: ['https://people.lids.mit.edu/yp/homepage/data/LN_fdiv.pdf', 'https://jmlr.org/papers/volume13/gretton12a/gretton12a.pdf', 'https://arxiv.org/pdf/1606.00709'],
  depth: 'specialist',
  designRecord: 'docs/teaching/F-DIVERGENCES-IPMS-LESSON-DESIGN.md',
  reviewFocus: 'Extended support terms and metric constants; event and Lipschitz witnesses; kernel separation and witness normalization; biased versus unbiased estimates; exchangeability and selected-statistic calibration; exact population versus fitted variational bounds.'
};
