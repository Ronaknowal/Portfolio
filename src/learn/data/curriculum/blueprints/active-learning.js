export default {
  summary: 'Choose which trustworthy answers to buy, inspect what a query rule learns, and evaluate the result under an explicit label budget.',
  outcomes: [
    'Operate a pool-based acquisition loop without leaking unacquired or evaluation labels',
    'Calculate threshold elimination, uncertainty and committee-disagreement scores',
    'Construct a diverse batch and distinguish geometric coverage from predictive accuracy',
    'Reproduce and critique a matched-budget real-data comparison with a final refit',
    'Explain sampling bias, annotation costs and when an acquisition objective is suitable',
  ],
  prerequisites: [
    'Linear & Logistic Regression',
    'Semi-Supervised Learning (Label Propagation, Self-Training, Co-Training)',
    'Gaussian Processes (GP)',
  ],
  sequence: [
    'Separate candidate inputs, acquired answers, development and final test labels',
    'Choose an informative threshold question and inspect surviving hypotheses',
    'Contrast confidence, margin, entropy and committee disagreement on exact fixtures',
    'Design a geometric batch, including coincident inputs and stable identity handling',
    'Run the complete offline banknote experiment and read all acquisition checkpoints',
    'Account for annotation workflow, selection bias, costs and stopping',
    'Deeper: gradient objectives, GP variance reduction, BADGE and importance weights',
    'Complete eight changed tasks and continue to evaluation metrics',
  ],
  visual: {
    type: 'Annotation lanes, aligned threshold rulers, probability strips, entropy decomposition, equal-scale geometry and measured acquisition curves',
    question: 'Which question separates plausible explanations, and what benefit does its acquisition criterion actually measure?',
    interaction: 'Edit candidate thresholds and weights, committee probabilities, anchor/candidate geometry and a chosen batch. Record a prediction before revealing calculations; inspect null and contradiction cases. Replay saved experimental query records without browser training.',
  },
  practice: {
    task: 'Predict uneven-family query value, compare tied uncertainty and distinct disagreement, diagnose coincident batches and final-fit errors, then run a 15-query development-only experiment.',
    success: 'Justify the selected action, reproduce the actual quantity with its assumptions, count every acquired label, and separate development selection from a final evaluation.',
  },
  misconceptions: ['High confidence guarantees a small update', 'Committee agreement proves labels cannot help', 'Diverse geometry guarantees classification accuracy', 'Queried training examples are a representative evaluation sample', 'The last acquisition is included without a final refit'],
  sources: ['https://burrsettles.com/pub/settles.activelearning.pdf', 'https://cseweb.ucsd.edu/~dasgupta/papers/twoface.pdf', 'https://mlg.eng.cam.ac.uk/pub/pdf/HouHusGha11a.pdf', 'https://arxiv.org/abs/1708.00489', 'https://arxiv.org/abs/1906.03671', 'https://archive.ics.uci.edu/dataset/267/banknote+authentication'],
  depth: 'core',
  reviewFocus: 'Independent acquisition arithmetic, no answer leakage before commitment, final refit, matched budgets and data splits, exact native outputs, all visual states and source-bound review.',
};
