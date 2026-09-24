export default {
  summary: 'Make probability precise by tracking measurable events, transported distributions, integrals, limits and the information available to a prediction.',
  outcomes: ['Construct a finite probability space and its observable event partitions', 'Trace a random variable through preimages to its induced law', 'Distinguish point mass, reference-dependent density and singular probability', 'Build an expectation from simple functions and identify integrability requirements', 'Justify or reject a limit or iterated-integral interchange using its hypotheses', 'Calculate joint cell probabilities and preserve mass under an invertible coordinate scale', 'Compute conditional expectations and prove their finite event-matching and L2 prediction properties', 'Reweight a measure by a valid density ratio and detect unsupported target events'],
  prerequisites: ['Real Analysis, Sequences & Modes of Convergence', "Probability Distributions & Bayes' Theorem"],
  sequence: ['Begin with observable questions in a finite outcome space', 'Define event closure and countable probability rules', 'Transport probability through measurable maps', 'Separate masses, densities, null sets and singular laws', 'Construct integrals from weighted measurable pieces', 'Check assumptions before exchanging a limit and expectation', 'Build product measures, marginals and coordinate changes', 'Condition on information and minimize justified prediction risk', 'Interpret density ratios and support', 'Practise changed spaces, measures and information; connect to transport'],
  visual: {
    type: 'Finite event partitions, preimage mapping, simple-function bands, shrinking-support curves, transformed probability cells and conditional prediction residuals',
    question: 'Which sets can the information distinguish, where does their probability go, and which operation preserves the integral?',
    interaction: 'Select an event and observation partition, refine simple functions, compare convergent functions with their integrals, transform a selected joint cell and refine a conditional prediction.'
  },
  practice: {
    task: 'Build a changed probability space, preimage law, interval probability and conditional predictor; diagnose invalid limit, product or density-ratio reasoning.',
    success: 'Correct sets, units, integrals and hypotheses, explained counterexamples, independent calculations and a justified information/sampling boundary.'
  },
  misconceptions: ['Every uncountable probability space must exclude some subsets', 'An event with zero probability is impossible', 'Atomless distributions always have a Lebesgue density', 'Measurability is independent of which information is available', 'Pointwise convergence alone allows exchanging expectations and limits', 'Using a product domain proves independence', 'Conditional expectation is a single scalar or uniquely fixed on null cells', 'Squared-error projection holds without a square-integrability assumption', 'Importance weights can create probability on source-null events'],
  sources: ['https://measure.axler.net/MIRA.pdf', 'https://math.mit.edu/~sheffield/2016175/Lecture16.pdf', 'https://thebrightsideofmathematics.com/courses/measure_theory/mt01_info/'],
  depth: 'specialist',
  designRecord: 'docs/teaching/MEASURE-THEORY-PROBABILITY-SPACES-DESIGN.md',
  reviewFocus: 'Countable versus uncountable operations, measurability relative to information, null-set versions, density reference and singular laws, integrability/limit/product hypotheses, conditional L2 projection, coordinate area scaling and exact finite verification.'
};
