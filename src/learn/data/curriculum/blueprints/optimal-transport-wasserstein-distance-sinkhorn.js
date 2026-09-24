export default {
  summary: 'Compare probability distributions through the cost of moving mass, certify finite transport plans and derive stable Sinkhorn scaling while separating numerical error, regularization and modelling assumptions.',
  outcomes: ['Construct a nonnegative coupling and verify both marginals', 'Distinguish a ground metric, transport cost and rooted Wasserstein distance', 'Solve and certify a small transport problem with primal and dual quantities', 'Calculate one-dimensional transport using cumulative mass or sorted quantiles', 'Derive entropic matrix scaling and inspect alternating marginal corrections', 'Implement log-domain updates with explicit residual and iteration limits', 'Distinguish the linear transport term, entropic objective and Sinkhorn divergence', 'Explain why barycentric projection, displacement interpolation and mixtures differ', 'Select a cost and transport variant with defensible assumptions and statistical limits'],
  prerequisites: ['Measure Theory & Probability Spaces', 'Convex Optimization'],
  sequence: ['Compare near and far probability shifts', 'Build and check a mass ledger', 'Certify a cheapest plan', 'Exploit one-dimensional order', 'Derive entropic scaling', 'Separate numerical stability and regularization', 'Correct self-comparison and inspect mappings', 'Choose applications, extensions and independent practice'],
  visual: {
    type: 'Mass-flow links with a conservation matrix, dual slack certificate, cumulative-distribution areas, actual alternating matrix corrections and split-mass versus conditional-mean diagrams',
    question: 'Which mass moves, what makes that plan feasible and cheapest, and what changes when we smooth or summarize it?',
    interaction: 'Change bounded source/target weights, feasible coupling, cumulative comparison, row/column step or entropy scale; inspect actual mass, cost, slack and residual values.'
  },
  practice: {
    task: 'Solve changed mass/cost constraints, construct a dual certificate, calculate a weighted 1D distance, diagnose scaling/bias and distinguish a valid plan from a barycentric map.',
    success: 'Both marginals, optimality bounds, metric powers/units, convergence limits, objective conventions and modelling assumptions agree with independent references.'
  },
  misconceptions: ['Every transport cost is a Wasserstein metric', 'A coupling must be a one-to-one assignment', 'A feasible plan is automatically optimal', 'Row normalization preserves the corrected column sums', 'A smaller regularization parameter guarantees a better computed answer', 'A linear Sinkhorn transport term is its complete regularized objective', 'Self-bias correction removes sampling error', 'A barycentric projection always preserves the target law', 'A small geometric distance proves causal or predictive equivalence'],
  sources: ['https://arxiv.org/html/1803.00567v4', 'https://proceedings.mlr.press/v89/feydy19a.html', 'https://pythonot.github.io/quickstart.html', 'https://moore.pims.math.ca/lecture/video/optimal-transport-machine-learning-lecture-1'],
  depth: 'specialist',
  designRecord: 'docs/teaching/OPTIMAL-TRANSPORT-LESSON-DESIGN.md',
  reviewFocus: 'Coupling feasibility and optimality; cost versus powered metric; zero marginals; log-domain residuals; entropy/KL constants; supported divergence properties; projection versus transport; explicit numerical and population limits.'
};
