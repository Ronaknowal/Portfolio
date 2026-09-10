export default {
  summary: 'Build probability models from an experiment, connect mass and density to observable events, and update competing explanations with explicitly conditioned evidence.',
  outcomes: [
    'Define outcomes/events, a normalized distribution and a conditioning event with positive probability',
    'Recover marginal and conditional probabilities from a joint table and derive Bayes normalization',
    'Distinguish conditional independence, repeated new evidence and duplicated evidence',
    'Choose and calculate count or waiting distributions from explicit sampling assumptions',
    'Read PMF, PDF and CDF values, interval mass, point masses and units correctly',
    'Compute and interpret expectation, variance and limitations of moment summaries',
    'Standardize a normal variable and use continuous likelihood densities without treating point probability as positive',
    'Challenge model mismatch and distinguish probability calibration from predictive ranking or parameter estimation',
  ],
  prerequisites: ['Random Variables, Expectation & Covariance'],
  sequence: ['Outcomes and events', 'Conditioning and joint probability', 'Bayes and base rates', 'Independent versus reused evidence', 'Trial and selection distributions', 'Mass, density, CDF and moments', 'Counts and waits', 'Continuous evidence and modeling limits', 'Independent practice and estimation bridge'],
  visual: {
    type: 'Event maps, proportionate joint populations, evidence branches, urn/PMF correspondence, linked density/CDF and event/wait timelines',
    question: 'Which outcomes count toward this event, under which conditioning population, and how does the representation encode their probability?',
    interaction: 'Change base rates and evidence dependence, inspect finite sampling with or without replacement, and compare density area/CDF mass under a unit change.',
  },
  practice: {
    task: 'Calculate changed conditional, count, interval and posterior probabilities; diagnose an invalid independence or density argument; justify a model and its limits.',
    success: 'Explicit experiment/support/units, correct normalization and intermediate reasoning, independently checked numerical results and a defensible changed-case explanation.',
  },
  misconceptions: ['All named outcomes are equally likely', 'P(A|B) equals P(B|A)', 'Disjoint events are independent', 'Repeated readings are independent new evidence', 'A density height is a point probability', 'Every distribution has an ordinary density', 'Counting successes always gives a binomial', 'Rate and scale have the same units', 'Mean and variance determine the whole distribution', 'A mathematically coherent posterior proves the model fits reality'],
  sources: ['https://ocw.mit.edu/courses/6-041sc-probabilistic-systems-analysis-and-applied-probability-fall-2013/pages/unit-i/', 'https://stat110.hsites.harvard.edu/youtube', 'https://www.itl.nist.gov/div898/handbook/eda/section3/eda366.htm', 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.hypergeom.html'],
  depth: 'core',
  designRecord: 'docs/teaching/PROBABILITY-DISTRIBUTIONS-BAYES-DESIGN.md',
  reviewFocus: 'Assumptions, zero-probability conditioning, event versus density units, dependency structure, exact finite oracles, distribution tails and practical model boundaries.',
};
