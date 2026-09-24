export default {
  summary: 'Build a density from several Gaussian components, fit it by alternating soft memberships with weighted statistics, and judge the result by held-out density under a rule declared before the test set is opened.',
  outcomes: [
    'Separate a component responsibility from the total density, and show a decisive share sitting on almost no density',
    'Reverse a hidden selector with Bayes rule and read a responsibility row as a soft allocation that sums to one',
    'Carry out one exact EM cycle by hand: effective counts, weighted means and variance around the new mean with denominator N',
    'State the constrained objective a variance floor defines, and derive why the floor is the exact maximizer on the boundary',
    'Evaluate responsibilities in log space and explain what clipping a denominator destroys',
    'Read a covariance ellipse, order two equally distant points by their quadratic forms, and name what each covariance family removes',
    'Choose a complexity rule before fitting, keep training BIC as a diagnostic, and report a reserved test result honestly',
    'Prove the EM nondecrease from the touching bound, and say which step supplies the equality',
    'Derive the k-means limit precisely and distinguish it from the spherical library setting'
  ],
  prerequisites: ['Anomaly & Outlier Detection (Isolation Forest, One-Class SVM, LOF)'],
  sequence: [
    'A hidden selector, two bells and one observed measurement',
    'Bayes rule reverses the selector into responsibilities; density and allocation answer different questions',
    'EM alternates: the E-step recomputes allocations, the M-step refits weighted statistics',
    'A compact constrained program, log-space evaluation and a stopping rule that means something',
    'What improvement guarantees, the symmetric stationary null, and the unbounded collapse',
    'Covariance as shape: Mahalanobis distance, the unit contour and four model families',
    'What a fitted mixture is for, and how to declare a complexity rule',
    'Real flowers offline: one fixed split, sixteen candidates, a reserved test and one ARI diagnostic',
    'From density to an anomaly score, sampling forward, and a diagnostic map',
    'Deeper: the evidence lower bound, the touching equality and generalized EM',
    'Deeper: the k-means limit, conditional prediction and a Bayesian weight prior'
  ],
  visual: {
    type: 'A hidden-selector diagram over weighted curves and their sum; a fractional-allocation diagram where unit masses split into two component lanes; aligned collapse panels with their own windows plus an exact objective table; a four-family covariance contour gallery drawn from each matrix eigenpairs; validation and BIC panels beside a magnified strip showing a fitted width narrower than the recorded resolution; an old/new bound chain with the touching equality and the gap',
    question: "Which component takes this observation, how much density is there at all, will one full cycle raise the objective, and which of two equally distant points does the component find more plausible?",
    interaction: "Edit the measurement, mixing weight and variance and follow densities and responsibility shares immediately; edit observations or initial means to restart the current EM trace, then step E and M phases separately and step back; vary correlation and point positions beside a permanent zero-correlation reference. No learner answer precedes a calculation."
  },
  practice: {
    task: 'Eight changed tasks: unequal-weight responsibilities and the tie boundary, an M-step from a supplied allocation with a floor, three repaired numerical claims, three correlations over the same two points, a restricted real-data selection with its reserved test, the missing line in a bound argument, a shared-variance limit against a library setting, and a conditional prediction that keeps its spread.',
    success: 'Responsibility rows sum to one; variances divide by the effective count and are taken around the new mean; a floor clips rather than adds; density and allocation are never conflated; every real-data claim names the split, the rule declared in advance and what the reserved test actually showed.'
  },
  misconceptions: [
    'A responsibility is a probability that the observation belongs to a real category',
    'A density is a probability, and cannot exceed one',
    'Sampling a mixture averages the component means',
    'EM maximizes the observed likelihood in one step',
    'The E-step changes the observed log-likelihood',
    'A rising lower bound alone proves the likelihood rose',
    'More EM iterations always improve the fit, so a higher training likelihood is better',
    'A singular spike that raises the likelihood is a successful fit',
    'A negative log-determinant means the covariance is invalid',
    'The unit Mahalanobis contour holds 68% of a two-dimensional Gaussian',
    'Diagonal covariance means the features are independent overall',
    'spherical covariance is k-means',
    'BIC identifies the true number of categories'
  ],
  sources: [
    'https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf',
    'https://doi.org/10.1111/j.2517-6161.1977.tb01600.x',
    'https://doi.org/10.1214/aos/1176346060',
    'https://cs229.stanford.edu/notes-spring2019/cs229-notes8.pdf',
    'https://scikit-learn.org/stable/modules/mixture.html',
    'https://doi.org/10.24432/C56C76'
  ],
  depth: 'core',
  designRecord: 'docs/teaching/GMM-LESSON-DESIGN.md',
  reviewFocus: 'Density versus probability versus responsibility throughout; the exact four-row trace (−7.158186977, −6.461856301, −5.724277515) and first M-step values (±1.344824658, 0.691446639); the constrained maximizer argument for the variance floor; log-sum-exp at (−1000, −1001) giving 0.731058579; 8/7 against 8 for the two equally distant points and the 39.346934% unit contour; the parameter totals 17, 11, 14 and 11; the declared validation rule selecting full K=2 while training BIC selects full K=4 with a width at the 1e-4 regularization level; the reserved test showing the baseline ahead by 0.123975 nats; ARI reported as a separate diagnostic on 30 rows; the bound chain −6.461856301 ≥ −6.799547014 ≥ −7.158186977 with entropy 0.910857246.'
};
