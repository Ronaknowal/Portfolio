export default {
  topicId: 'backtracking-divide-and-conquer',
  verifiedOn: '10 September 2026',
  introduction: 'Practice both search over candidate choices and divide/combine design. Reconstruct the local programs from their invariants before using the platform; an accepted output alone is not a proof of restoration, completeness or the claimed cost.',
  groups: [{
    id: 'foundation',
    title: 'Foundation · define the choices and their identity',
    introduction: 'Decide whether positions or values identify a choice, whether order matters, and whether an item may be reused.',
    problems: [{
      number: 78,
      title: 'Subsets',
      slug: 'subsets',
      difficulty: 'Medium',
      focus: 'Reconstruct include/exclude enumeration with unique input values and independent saved outputs.',
      hint: 'At position i, divide the answer set into those that contain this item and those that do not. Which mutable object must be copied at a completed decision path?',
      transfer: 'Explain empty-input behavior in your own API and why it differs from no answers. Change input to repeated values and state what the judge’s original uniqueness promise no longer covers. Charge total copied output, not only the number of calls.'
    }, {
      number: 47,
      title: 'Permutations II',
      slug: 'permutations-ii',
      difficulty: 'Medium',
      focus: 'Generate unique value sequences when input occurrences can be equal.',
      hint: 'Can a remaining-frequency table avoid branching separately on indistinguishable copies? If you use positions instead, which same-depth alternatives are duplicates?',
      transfer: 'Test all equal values and all distinct values. Explain why the chosen rule removes duplicate branches without removing a valid sequence; compare with generating and deduplicating all position permutations afterward.'
    }, {
      number: 39,
      title: 'Combination Sum',
      slug: 'combination-sum',
      difficulty: 'Medium',
      focus: 'Allow repeated use while producing each unordered combination once.',
      hint: 'Choose a canonical nondecreasing index order. Which recursive start index permits reuse, and which would allow an occurrence only once?',
      transfer: 'Explain why positive candidates give a decreasing remainder. Consider what zero or mixed signs would do to termination and the number of answers. Those changed inputs lie outside the official positive-candidate contract.'
    }]
  }, {
    id: 'core',
    title: 'Core · maintain constraints and combine boundaries',
    introduction: 'Use actual geometry and interval structure to derive the state, rather than assuming every recursive call has the same role.',
    problems: [{
      number: 79,
      title: 'Word Search',
      slug: 'word-search',
      difficulty: 'Medium',
      focus: 'Find one path under a no-cell-reuse rule and restore temporary state across failed branches.',
      hint: 'Which cells are forbidden only for the current candidate path? Why does a reachability-wide visited set lose potentially valid attempts?',
      transfer: 'Test a tempting route that would reuse its start cell, repeated letters and a failed first starting position. If you mutate the board, prove restoration on both failure and successful early return.'
    }, {
      number: 53,
      title: 'Maximum Subarray',
      slug: 'maximum-subarray',
      difficulty: 'Medium',
      focus: 'Solve the divide-and-conquer follow-up using a return contract that handles an answer crossing the split.',
      hint: 'Best-left and best-right alone omit crossing answers. Which boundary-anchored summaries let the parent account for them without rescanning?',
      transfer: 'Test all-negative values and a best interval spanning both halves. Explain why this four-summary version is linear arithmetic work, whereas recomputing crossing sums by a scan at every split is n log n. Compare Kadane’s algorithm later as a different derivation.'
    }]
  }, {
    id: 'stretch',
    title: 'Optional consolidation · board-wide completeness',
    optional: true,
    introduction: 'The lesson already supplies the required row/column/diagonal reasoning. This larger reconstruction is optional for proceeding; use it to practise translating a coordinate invariant into the platform’s board output.',
    problems: [{
      number: 51,
      title: 'N-Queens',
      slug: 'n-queens',
      difficulty: 'Hard',
      prerequisite: 'The queen investigation, r−c/r+c diagonal keys, choose/undo restoration and generating the required string rows from column positions.',
      focus: 'Enumerate complete nonattacking boards while retaining both mirrored solutions unless the contract says otherwise.',
      hint: 'Assign one row at a time so row uniqueness is automatic. What three sets must be updated together on placement and restored together on return?',
      transfer: 'Check n=1, n=2 and n=4; compare exact solution sets with a brute-force permutation oracle for small boards. Analyze candidate scanning and output board construction separately from the number of accepted placements.'
    }]
  }],
  readiness: ['State soundness, completeness and termination, then identify every temporary mutation and the point that restores it.', 'Distinguish duplicate-valued choices, reusable candidates, all answers, one witness and tied optima before selecting a template.', 'Give a counterexample to an unjustified pruning rule or incomplete memoization key.', 'Explain what each divide-and-conquer child must return and why the combine step accounts for every case exactly once.', 'Use independent enumeration for small inputs, and include copied output, stack depth, candidate scans and representation costs in the analysis.'],
  localBridge: 'Return to the parenthesis-prefix exercise, interval-witness extension and ranking-disagreement task. These require transferring a reasoned invariant, not matching an external problem title to a memorized template.'
};
