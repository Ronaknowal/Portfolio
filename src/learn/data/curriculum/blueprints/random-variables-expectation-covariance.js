export default {
  summary: 'Turn uncertain outcomes into numerical predictions, measure their spread and determine how dependence changes an average, a difference or a prediction made with additional information.',
  outcomes: [
    'Distinguish an outcome, a random variable, an observed value and its probability law',
    'Construct a PMF and CDF by collecting probability from all outcomes with the same numerical value',
    'Compute expectations of transformations directly and use linearity without assuming independence',
    'Derive variance, unit changes and the mean as the best constant squared-error prediction',
    'Build joint and conditional laws, test independence and exhibit dependence with zero covariance',
    'Compute covariance and correlation with their moment, unit and degeneracy conditions',
    'Propagate covariance through sums, differences and linear maps, including shared-noise measurements',
    'Calculate conditional means, total expectation, total variance and within/between covariance',
    'Transform a continuous variable through a non-injective function while conserving probability',
    'Separate population moments from sample statistics and explain when averaging reduces uncertainty',
    'Solve a complete measurement-combination problem, check assumptions and transfer to changed data'
  ],
  prerequisites: [
    'Sets, Logic, Relations & Proof Techniques',
    'Algebra, Functions, Exponentials & Logarithms',
    'Single-Variable Calculus: Limits, Derivatives & Integrals'
  ],
  sequence: [
    'Map concrete outcomes to values and collect their probability',
    'Read a distribution and compute weighted expectations of functions',
    'Measure squared spread and explain the choice of a prediction',
    'Retain paired outcomes in a joint law before describing dependence',
    'Interpret centered products and the limits of correlation',
    'Propagate shared and separate noise through linear combinations',
    'Use information partitions to derive conditional and total moments',
    'Extend sums to continuous transformations and identify nonexistent moments',
    'Distinguish exact population quantities, sample statistics and repeated-sample uncertainty',
    'Practise changed models and continue to Sampling, Measurement & Experimental Design'
  ],
  visual: {
    type: 'Outcome-to-value grouping, PMF/CDF correspondence, balance and squared-loss geometry, joint/marginal grids, signed covariance products, shared-noise lanes, conditional residuals and transformed-interval preimages',
    question: 'Which uncertainty is individual, which is shared, and what changes when outcomes are paired, transformed, averaged or observed through extra information?',
    interaction: 'Change a probability law, compare exact joint pairings, inspect centering, cancel or retain common noise, reveal a group and compare independent versus repeated copies without substituting simulated evidence for the theoretical law.'
  },
  practice: {
    task: 'Compute and explain a changed finite law, two dice-derived variables, covariance counterexamples, conditional decompositions, a continuous transformation and a shared-noise measurement task.',
    success: 'Conserves probability, shows meaningful intermediate calculations, states units and moment assumptions, distinguishes population from sample quantities and produces explicit independence or degeneracy witnesses.'
  },
  misconceptions: [
    'A random variable is the same object as its realized value or its probability distribution',
    'Distinct variable values must be equally likely, or an expected value must be attainable',
    'Expectation commutes with every nonlinear transformation or needs independence to be linear',
    'Matching marginal laws determine how variables are paired',
    'Zero covariance means independence, or correlation is defined for a constant variable',
    'Variance always adds, or averaging many copied readings creates independent evidence',
    'A covariance matrix may contain an arbitrary set of pairwise entries',
    'Conditioning removes all uncertainty or guarantees smaller variance in every selected group',
    'A density height is a probability or a non-injective transformation has only one inverse branch',
    'Every distribution has a finite mean and variance, or n−1 corrects arbitrary dependent samples'
  ],
  sources: [
    'https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_class04-prep-b.pdf',
    'https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_class07-prep-b.pdf',
    'https://ocw.mit.edu/courses/18-05-introduction-to-probability-and-statistics-spring-2022/mit18_05_s22_class06-prep-a.pdf',
    'https://ocw.mit.edu/courses/res-6-012-introduction-to-probability-spring-2018/resources/derivation-of-the-law-of-total-variance/',
    'https://stat110.hsites.harvard.edu/youtube',
    'https://web.mit.edu/18.06/www/Spring21/Lecture%20notes.pdf'
  ],
  depth: 'core',
  reviewFocus: 'Probability-preserving maps; finite-moment and zero-variance conditions; population versus empirical pairing; covariance and conditional decomposition signs; shared-noise assumptions; non-injective transformations; sample-size and numerical-arithmetic boundaries.',
  designRecord: 'docs/teaching/RANDOM-VARIABLES-LESSON-DESIGN.md'
};
