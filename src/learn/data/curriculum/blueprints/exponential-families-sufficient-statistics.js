export default {
  summary: 'Decide which summaries preserve parameter information under a model, then use the exponential-family structure to derive likelihoods, moments, fits and Bayesian updates.',
  outcomes: ['Explain sufficiency through conditional datasets and factorization, including model-relative failure', 'Derive natural parameters, base factors, support and normalizers for concrete discrete and Gaussian families', 'Derive mean and covariance identities in valid natural coordinates', 'Solve finite interior moment matching and diagnose boundary or redundant representations', 'Retain the correct summaries for unknown Gaussian spread, fixed features and unequal Poisson exposures', 'Derive conjugate updates while respecting parameter-density coordinates and properness', 'Implement stable complete examples and solve changed inference contracts'],
  prerequisites: ["Probability Distributions & Bayes' Theorem", 'Maximum Likelihood & MAP Estimation'],
  sequence: ['Group datasets by what a model can distinguish', 'Prove factorization and diagnose a richer model', 'Build and normalize exponential-family weights', 'Derive moment and curvature identities', 'Match sufficient moments and inspect boundary geometry', 'Preserve Gaussian spread and merge summaries', 'Use features, exposures and conjugate updates', 'Practise changed assumptions and bridge to probability measures'],
  visual: {
    type: 'Dataset fibers, normalized finite masses, moment triangle and paired coordinate densities',
    question: 'What survives the summary, what changes when the model changes, and why do the same parameters have different density pictures?',
    interaction: 'Edit allocations under two models, tilt finite weights, fit a mean target and compare a prior in probability versus log-odds coordinates.'
  },
  practice: {
    task: 'Prove or refute sufficiency, derive a new family and fit, diagnose boundary/redundancy, and design an exposure/feature-aware summary.',
    success: 'States the model and known context; shows conditional/factorization reasoning; handles natural-domain and finite-MLE limits; verifies changed numerical results and density transformations.'
  },
  misconceptions: ['Sufficiency is lossless for every future question', 'Equal summaries imply equal raw dataset probabilities', 'A log-partition function can be omitted while taking unrestricted derivatives', 'A convex likelihood always has a finite unique optimizer', 'A sum retains unknown Gaussian spread or arbitrary feature allocation', 'Every GLM uses the canonical link', 'A density height is a probability or invariant under reparameterization', 'A conjugate-looking kernel is automatically a proper prior'],
  sources: ['https://www.stat.berkeley.edu/~wfithian/courses/stat210a/exponential-families.html', 'https://slinderman.github.io/stats305b/lectures/04_expfam.html', 'https://ocw.mit.edu/courses/18-655-mathematical-statistics-spring-2016/resources/mit18_655s16_lecnote6/', 'https://www.stat.umn.edu/geyer/f22/5421/notes/prior.html', 'https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/resources/lecture-21-video/'],
  depth: 'specialist',
  reviewFocus: 'Fixed common support; conditional sufficiency versus likelihood equality; natural-coordinate derivatives; finite interior existence and minimality; Gaussian precision signs; feature/exposure scope; prior Jacobians and properness.',
  designRecord: 'docs/teaching/EXPONENTIAL-FAMILIES-LESSON-DESIGN.md'
};
