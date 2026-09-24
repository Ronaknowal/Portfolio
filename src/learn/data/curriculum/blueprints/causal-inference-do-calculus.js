export default {
  summary: 'Turn a causal question into explicit mechanisms and assumptions, determine what an intervention or counterfactual can identify, and calculate a checked finite analysis.',
  outcomes: ['Distinguish conditioning, intervention, counterfactual evidence and causal identification from estimation', 'Trace causal paths, colliders and descendants under conditioning and graph surgery', 'Derive backdoor adjustment and diagnose consistency, exchangeability and overlap assumptions', 'Apply all three do-calculus rules with the correct graph and ancestor conditions', 'Calculate frontdoor identification and expose a failed exclusion assumption', 'Compute counterfactual updates while preserving hidden units and explain nonidentification', 'Verify weighting and augmented-estimator expectations, and distinguish assignment, complier and population effects', 'Design an analysis with explicit estimand, diagnostics, uncertainty and limits'],
  prerequisites: ["Probability Distributions & Bayes' Theorem", 'Hypothesis Testing & Confidence Intervals'],
  sequence: ['Specify an intervention question and build its mechanisms', 'Trace paths and selection in causal graphs', 'Derive adjustment with explicit population weights and overlap', 'Distinguish observational equivalence from identification', 'Apply graph transformations and derive frontdoor adjustment', 'Update paired counterfactual worlds', 'Estimate identified targets and compare experimental designs', 'Solve changed causal contracts and plan an independent analysis'],
  visual: {
    type: 'Mechanism surgery, active paths, population weights, rule-specific graphs and paired counterfactual worlds',
    question: 'Which part of the data-generating mechanism changes, and what information supports the resulting causal query?',
    interaction: 'Condition on graph nodes, modify assignment, inspect do-calculus graph cuts, break a frontdoor assumption and retain the same hidden unit across possible outcomes.'
  },
  practice: {
    task: 'Derive a changed adjustment, diagnose graph/overlap/rule errors, calculate a counterfactual and design a defensible estimation strategy.',
    success: 'Names the target and population; checks graphical and statistical assumptions; shows actual numerical reasoning; distinguishes an identified result from model-dependent extrapolation.'
  },
  misconceptions: ['Conditioning on a treatment replaces its causal mechanism', 'Adjusting for every predictive feature removes bias', 'An active graph path guarantees statistical dependence', 'Any mediator makes frontdoor identification valid', 'Rule three always cuts every proposed action node', 'More observational data identify every intervention or individual counterfactual', 'Doubly robust estimation removes unmeasured confounding', 'Randomized assignment or an instrument always identifies the population treatment effect'],
  sources: ['https://ftp.cs.ucla.edu/pub/stat_ser/r416-reprint.pdf', 'https://www.bradyneal.com/causal-inference-course', 'https://miguelhernan.org/whatifbook', 'https://www.nber.org/papers/t0136', 'https://www.pywhy.org/EconML/spec/estimation/dr.html'],
  depth: 'specialist',
  reviewFocus: 'Causal graph assumptions, positivity, d-separation versus faithfulness, all modified graphs and rule-three ancestor exclusion, frontdoor criteria, same-unit counterfactual coupling, estimand/estimator distinction and local versus population effects.',
  designRecord: 'docs/teaching/CAUSAL-INFERENCE-LESSON-DESIGN.md'
};
