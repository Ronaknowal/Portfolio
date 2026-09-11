export default {
  summary: 'Build and evaluate recommendations by tracing observed evidence through neighborhood and factor models, while distinguishing missing feedback, ranking goals and exposure assumptions.',
  outcomes: ['Define the request, data unit, observable history, item eligibility and held-out question before fitting a recommender', 'Calculate baseline and neighborhood predictions with explicit masks, centering, overlap support and fallback behavior', 'Explain biased low-rank scores, parameter sharing and nonidentifiability rather than treating missing entries as known zeros', 'Derive simultaneous SGD and block least-squares updates under a precisely counted regularized objective', 'Compute confidence-weighted implicit ALS efficiently and distinguish its scores from probabilities', 'Derive a BPR comparison update and explain the sampling assumptions behind it', 'Evaluate eligible recommendation lists with stated ranking metrics, candidate sets, ties and cohort conventions', 'Run checked explicit and implicit library examples with correct user/item shapes and complete native output', 'Handle cold entities with a declared fallback or content model and explain the retrieval/reranking boundary', 'Distinguish biased logged feedback from identifiable target-policy value under stated support assumptions', 'Produce a reproducible changed-data report with a baseline, validation choices, held-out results and limitations'],
  prerequisites: ['Linear & Logistic Regression', 'Vectors, Matrices & Tensor Operations'],
  sequence: ['Define request, evidence and missingness', 'Freeze splits, availability and baselines', 'Calculate supported neighborhood evidence', 'Build biased low-rank scores and challenge identifiability', 'Train explicit factors with counted objectives', 'Derive block least-squares solves', 'Model implicit confidence and sparse ALS', 'Learn pairwise comparisons with BPR', 'Evaluate eligible lists and candidate recall', 'Use current native libraries', 'Bridge content cold starts and retrieval', 'Separate exposure from causal policy value', 'Complete changed experiments and independent practice'],
  visual: {
    type: 'Evidence matrix, request-time timeline, aligned neighbor contributions, factor-update workbench, implicit quadratic geometry, ranked candidate board and exposure tree',
    question: 'Which information contributes to this score, which assumption creates a target, and which later decision can the result justify?',
    interaction: 'Select an evidence cell, change overlap/shrinkage, step actual factor updates, alter confidence, rerank/remove candidates and vary a known logging policy; connect every change to its actual calculation.'
  },
  practice: {
    task: 'Repair temporal leakage and factor updates, solve changed neighborhood/ALS/BPR cases, evaluate candidate-limited lists, diagnose exposure and cold-start assumptions, and submit an independently changed experiment report.',
    success: 'Calculations match independently checked cases; train-only transformations, eligibility, baseline, metric denominators, sample size and limitations are explicit; each substantial task has a hint and separately explained solution.'
  },
  misconceptions: ['Missing feedback is a zero rating or proof of dislike', 'Low rank alone identifies every missing entry', 'Factor axes must be human-readable genres', 'A small training error or monotone coordinate objective proves generalization or a global optimum', 'All regularization conventions give the same SGD and ALS updates', 'Implicit confidence or a dot-product score is a calibrated probability', 'BPR always wins or sampled negatives are true dislikes', 'NDCG from an easier candidate set measures the same task', 'A popularity baseline winning proves personalization is impossible', 'Observed clicks identify the value of changing recommendations without exposure assumptions', 'A downstream ranker can retrieve a missing candidate'],
  sources: [
    // Hu, Koren and Volinsky: implicit feedback objective and ALS
    'https://yifanhu.net/PUB/cf.pdf',
    // Rendle et al.: Bayesian Personalized Ranking
    'https://arxiv.org/pdf/1205.2618',
    // Schnabel et al.: selection bias and recommendation evaluation
    'https://proceedings.mlr.press/v48/schnabel16.pdf',
    // Google Developers: matrix factorization visual introduction
    'https://developers.google.com/machine-learning/recommendation/collaborative/matrix',
    // Implicit CPU ALS API
    'https://benfred.github.io/implicit/api/models/cpu/als.html',
    // Surprise matrix-factorization API
    'https://surprise.readthedocs.io/en/stable/matrix_factorization.html',
    // Stanford CS246 public lecture archive: Recommender Systems I/II
    'https://snap.stanford.edu/class/cs246-videos-2019/'
  ],
  depth: 'core',
  designRecord: 'docs/teaching/RECOMMENDER-SYSTEMS-LESSON-DESIGN.md',
  reviewFocus: 'Observation masks, train-only information and item eligibility, counted regularization objectives, explicit versus implicit targets, candidate-dependent ranking metrics, exposure assumptions, bounded investigations and held-out transfer.'
};
