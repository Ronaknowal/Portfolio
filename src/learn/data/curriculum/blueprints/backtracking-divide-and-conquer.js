export default {
  summary: 'Design candidate searches with explicit choose/explore/undo state and safe pruning, then design divide-and-conquer return contracts that combine cross-boundary information without losing answers or repeating work.',
  outcomes: ['Distinguish decision trees from subproblem decomposition', 'Implement complete subset, permutation and combination searches under explicit identity/reuse rules', 'Prove safe pruning, restoration and output-sensitive costs', 'Apply column/diagonal and path-local constraints to board searches', 'Count inversions during a justified merge', 'Derive and implement constant-size maximum-subarray summaries', 'Choose sound state keys and recognize memoization/branch-and-bound boundaries'],
  prerequisites: ['Complexity Analysis & Recursion', 'Binary Search, Sorting & Two-Pointer Patterns'],
  sequence: ['Choose the decomposition contract', 'Trace inclusion, exclusion and exact restoration', 'Prune only impossible completion families', 'Distinguish ordering, identity and reuse', 'Enforce geometric and path-local constraints', 'Count pairs across a divide boundary', 'Return total, prefix, suffix and best summaries', 'Solve independent changed-contract and guided practice'],
  visual: {
    type: 'Decision tree with live path/copied outputs, attacked-square chessboard, divide-tree interval summaries and inversion boundary figure',
    question: 'What information does this child explore or return, and what must survive its completion?',
    interaction: 'Change positive subset inputs and pruning, step real queen placements/rejections/undo, and inspect signed-array subregions with computed boundary summaries and witnesses.'
  },
  practice: {
    task: 'Generate valid parenthesis prefixes, return interval witnesses, count ranking disagreements and repair an incomplete memoization key; attempt curated search and divide/combine problems.',
    success: 'Match independent enumeration oracles and explain completeness, soundness, termination, restoration, output size, tie policy and combine cost.'
  },
  misconceptions: ['Every recursive algorithm is the same choose/undo template', 'Saving the working list saves its current contents', 'Sorting makes overshoot pruning valid with negatives', 'Position identity and equal values are interchangeable', 'Global visited state is always correct for path enumeration', 'Both child best answers are enough for a crossing optimum', 'Every divide-and-conquer algorithm costs n log n', 'A compact bitmask or memoization key automatically makes search efficient'],
  sources: ['https://web.stanford.edu/class/archive/cs/cs106b/cs106b.1196/notes/lecture11.html', 'https://see.stanford.edu/Course/CS106B/147', 'https://www.cs.cmu.edu/afs/cs/academic/class/15210-f14/www/lectures/dandc.pdf', 'https://web.stanford.edu/class/archive/cs/cs161/cs161.1168/lecture3.pdf'],
  depth: 'core',
  reviewFocus: 'Position/value duplicate contracts, safe signed versus positive pruning, copy/undo and early-return cleanup, actual queen attacks, expected set costs, strict inversion ties, nonempty interval summaries, arithmetic versus bit costs, true native/model correspondence and narrow-screen readability.'
};
