export default {
  summary: 'Estimate an effect with a sampling procedure, inspect the null model and practical threshold separately, and report uncertainty without confusing a p-value, confidence, prediction or equivalence claim.',
  outcomes: ['Identify the estimand and independent sampling unit', 'Derive mean standard error and a paired Student t interval from raw differences', 'Interpret repeated-sampling coverage and distinguish it from individual variability', 'Calculate null-tail p-values, select an alternative before observing results and connect compatible tests to intervals', 'Separate effect size, practical superiority, non-rejection and equivalence', 'Explain power, multiplicity, repeated looks and dependence through explicit model assumptions', 'Choose and execute a justified basic interval/test or resampling comparison', 'Report an effect, uncertainty, decision criterion and limitations together'],
  prerequisites: ["Probability Distributions & Bayes' Theorem", 'Sampling, Measurement & Experimental Design'],
  sequence: ['Define the paired workload question and what is sampled', 'Build sampling variability, repeated interval coverage and prediction', 'Derive and calculate the paired t result', 'Inspect null-tail evidence and test inversion', 'Compare practical effects and equivalence', 'Plan power under a specified true effect', 'Invert a score test and enumerate binary interval coverage', 'Protect the sampling and decision procedure from dependent units and extra chances', 'Choose a method by sampling and model assumptions', 'Practise changed data and report what is supported'],
  visual: {
    type: 'Preserved moving-interval coverage mechanism, raw pair-to-difference view, null and alternative sampling distributions, effect/threshold number line, exact finite resampling distribution and sampling-unit comparison',
    question: 'What varies across samples, what remains fixed, and which decision does the observed evidence actually justify?',
    interaction: 'Change sample size/confidence, inspect a paired dataset, move the null reference and alternative, compare practical thresholds and planned test power; expose exact states rather than a single unexplained score.'
  },
  practice: {
    task: 'Recalculate a changed paired result, interpret a coverage panel, invert a test, check an equivalence claim, diagnose pseudoreplication/multiplicity, and report a justified conclusion.',
    success: 'Correct units and sampling assumptions, complete calculations, independent changed-input checks, correct p/coverage/decision language and explicit limits.'
  },
  misconceptions: ['A 95% confidence interval contains 95% of individual outcomes', 'A p-value is the probability the null is true', 'Non-rejection proves zero effect or equivalence', 'Excluding zero establishes a useful effect threshold', 'More correlated requests are equivalent to more independent users', 'Resampling a small dataset creates new independent evidence', 'A fixed-sample threshold stays valid after arbitrary repeated checking', 'A confidence level is a guarantee for every realized batch'],
  sources: ['https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm', 'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html', 'https://www.amstat.org/asa/files/pdfs/p-valuestatement.pdf'],
  depth: 'core',
  designRecord: 'docs/teaching/HYPOTHESIS-TESTING-CONFIDENCE-DESIGN.md',
  reviewFocus: 'Preserve and improve the pilot mechanisms; exact versus asymptotic sampling assumptions; Student t scale/df and tail calculations; test/interval duality conventions; Wilson boundary behavior versus exact conservative binomial coverage; equivalence, power, repeated testing, resampling nulls, dependence and honest numerical/visual evidence.'
};
