export default {
  summary: 'Choose feasible discrete decisions and justify their quality using exact structure, bounds, approximation proofs and controllable-error schemes.',
  outcomes: ['Write discrete decision variables, objective units and independent feasibility checks', 'Distinguish instance certificates, exact optimality, worst-case approximation and empirical heuristics', 'Use matroid axioms and a justified exchange argument to recognize an exact weighted greedy method', 'Run bound-driven search without confusing an incumbent with a proved optimum', 'Compute minimum-cost assignments with signed residual reversals and explain their optimality contract', 'Derive weighted set-cover charging and weighted vertex-cover rounding or primal-dual guarantees', 'Compare complete cover with cardinality-constrained submodular selection', 'Implement value-scaled knapsack DP and explain its feasibility, approximation and runtime bounds', 'Identify the metric assumptions behind routing approximations and audit a practical optimization result'],
  prerequisites: ['Greedy Algorithms & Exchange Arguments', 'Network Flow, Minimum Cuts & Bipartite Matching', 'Convex Duality & Lagrangian Methods (KKT Conditions)'],
  sequence: ['Model a discrete decision and its certificate', 'Prove when greedy structure is exact', 'Explore exact search with optimistic bounds', 'Solve weighted assignment through residual reversals', 'Derive cover and rounding guarantees', 'Choose under a cardinality or approximation-error budget', 'Use metric structure and solve independent application tasks'],
  visual: {
    type: 'Feasible assignment edges, exchange witnesses, branch bounds, incidence-grid charges, dual loads and value-rounding frontiers',
    question: 'What legal decision did the algorithm produce, and what independent argument bounds how much better any decision could be?',
    interaction: 'Change costs or constraints, inspect an exchange or residual reversal, advance a bounded search, track element charges and vertex loads, and alter epsilon while checking true reconstructed value.'
  },
  practice: {
    task: 'Repair invalid quality claims, solve changed weighted assignments and covers, diagnose pruning/rounding failures and implement a feasible approximation with an explicit bound.',
    success: 'The result satisfies independently checked constraints; bound directions, objectives, numeric conventions, complexity and changed-assumption limits are justified with complete reasoning.'
  },
  misconceptions: ['A feasible candidate proves optimality', 'Every successful greedy proof applies to weighted variants', 'An unweighted cover argument also pays arbitrary vertex costs', 'Reverse residual edges retain the original positive cost', 'Every nonbipartite matching problem is NP-hard', 'Any fractional rounding preserves feasibility', 'Minimum cover and maximum coverage have the same guarantee', 'Pseudopolynomial time is polynomial in encoded input size', 'PTAS means probabilistic time', 'More accurate predicted costs guarantee a better real decision'],
  sources: ['https://assets.cambridge.org/97805218/11514/sample/9780521811514ws.pdf', 'https://theory.stanford.edu/~tim/w16/l/l5.pdf', 'https://theory.stanford.edu/~tim/w16/l/l17.pdf', 'https://ocw.mit.edu/courses/6-854j-advanced-algorithms-fall-2008/a6b27d8a3d0ecda084f106f13b322676_lec16.pdf'],
  depth: 'specialist',
  designRecord: 'docs/teaching/COMBINATORIAL-OPTIMIZATION-LESSON-DESIGN.md',
  reviewFocus: 'Feasibility and objective direction, signed/fixed-cardinality matroid contracts, valid search bounds, residual cost optimality, exact harmonic and dual inequalities, FPTAS filtering/scaling and metric-only routing assumptions.'
};
