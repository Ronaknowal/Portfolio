export default {
  summary: 'Design reusable subproblem states, prove transitions and dependency order, recover witnesses and optimize storage or work without changing the problem.',
  outcomes: [
    'Distinguish repeated calls from complete reusable states',
    'Derive and justify base cases, transitions, order and final result',
    'Implement memoization and tabulation with consistent cache scope',
    'Recover witnesses in weighted schedules, sequence, grid and capacity problems',
    'Explain when compressed update order changes item reuse or counting semantics',
    'Encode finite subsets as masks before deriving subset-plus-endpoint DP',
    'Analyze state and transition costs, pseudo-polynomial bounds and optimization assumptions',
    'Derive fixed-boundary interval splits and recover child-before-parent computation or removal witnesses',
    'Prove why one parent-selected condition makes weighted tree subproblems composable',
    'Count bounded digit completions with explicit tightness, padding, property and endpoint semantics'
  ],
  prerequisites: ['Complexity Analysis & Recursion', 'Arrays, Strings & Hash Maps', 'Backtracking & Divide-and-Conquer', 'Trees & Binary Search Trees'],
  sequence: [
    'Derive a reusable free suffix',
    'Expose hidden history in the state key',
    'Keep compatible-prefix alternatives in finish-ordered weighted scheduling',
    'Grow grid answers and reconstruct routes',
    'Align sequence prefixes',
    'Allocate item capacity and inspect logical rows',
    'Separate optimization, existence and counting',
    'Represent used subsets and endpoints',
    'Optimize from a proven invariant',
    'Plan fixed-shape interval products and reconstruct last-removal decisions',
    'Pass a parent boundary condition through a tree',
    'Count tight and loose digit-prefix continuations without enumerating all integers',
    'Design and test a new command-segmentation problem'
  ],
  visual: {
    type: 'Subproblem graph, spatial grid, alignment table, capacity-generation strip, membership-bit/endpoint map, interval shape/expression planner, conditional tree and digit-prefix branches',
    question: 'Which information makes two histories equivalent, and which already-solved state does this transition use?',
    interaction: 'Step requests or tabulation, toggle obstacles, inspect prefix dependencies, change capacity iteration direction and compare endpoint futures for one subset; inspect interval splits and postorder witnesses, vary tree parent permissions and follow legal digit-prefix completions.'
  },
  practice: {
    task: 'Derive new recurrences, diagnose incomplete states and loop-order bugs, recover witnesses and solve command segmentation plus curated official problems.',
    success: 'Matches independent small exhaustive oracles, explains soundness/completeness/termination and meaningful boundary cases, and adapts safely when constraints change.'
  },
  misconceptions: [
    'A cache repairs an incomplete state definition',
    'Zero means not computed or impossible',
    'Bottom-up means increasing every index',
    'Compression can use either loop direction',
    'Counting sequences and combinations uses the same partition',
    'A mask alone always describes a path state',
    'Every dynamic program is polynomial',
    'A value-only memory bound automatically includes witness reconstruction',
    'A minimum-tail LIS summary is itself an input subsequence',
    'A last-split tree should be executed root-first',
    'One unconstrained subtree optimum works for every parent choice',
    'A loose digit prefix can become tight again',
    'Padding zero and an actual digit zero consume the same property state'
  ],
  sources: [
    'https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/pages/lecture-notes/',
    'https://docs.python.org/3/library/functools.html#functools.cache',
    'https://docs.python.org/3/library/stdtypes.html#bitwise-operations-on-integer-types',
    'https://www.cs.cmu.edu/~15451-s25/slides/lecture11.pdf',
    'https://www.cs.princeton.edu/courses/archive/spring13/cos423/lectures/06DynamicProgrammingI.pdf',
    'https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/resources/lecture-21-dp-iii-parenthesization-edit-distance-knapsack/',
    'https://www.cs.jhu.edu/~jason/425/PDFSlides/16dyna.pdf',
    'https://leetcode.com/problems/burst-balloons/',
    'https://leetcode.com/problems/house-robber-iii/',
    'https://leetcode.com/problems/count-special-integers/'
  ],
  depth: 'core',
  reviewFocus: 'State sufficiency, dependency direction and termination, unreachable states, tie/reconstruction contracts, generation reads under compression, exact count semantics, mask language limits, independent oracles and topic-specific desktop/mobile visuals.'
};
