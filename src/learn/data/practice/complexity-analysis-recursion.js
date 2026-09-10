export default {
  topicId: 'complexity-analysis-recursion',
  verifiedOn: '10 September 2026',
  introduction: 'Use these problems to practise explaining the work and storage, not just producing an accepted result. State the input parameter, counted operation, case and recursion depth before choosing an implementation.',
  groups: [{
    id: 'foundation',
    title: 'Foundation · count, shrink and respect the contract',
    introduction: 'Start after the loop, frame and space sections. A recursive solution can be correct while violating a memory requirement.',
    problems: [{
      number: 896,
      title: 'Monotonic Array',
      slug: 'monotonic-array',
      difficulty: 'Easy',
      focus: 'Recognize when adjacent comparisons can certify a property stated over every pair, and distinguish early rejection from worst-case work.',
      hint: 'Keep track of whether any strict increase and any strict decrease have appeared. Equal adjacent values do not establish either direction.',
      transfer: 'Construct a full-length successful scan and an early rejection. Explain why checking all pairs is unnecessary, and why a passing short example does not prove linear work.'
    }, {
      number: 1342,
      title: 'Number of Steps to Reduce a Number to Zero',
      slug: 'number-of-steps-to-reduce-a-number-to-zero',
      difficulty: 'Easy',
      focus: 'Find a decreasing integer measure and bound the number of arithmetic steps even though not every step halves the input.',
      hint: 'After subtracting from an odd value greater than one, what is the parity? Group subtraction and division rather than charging every step as a separate halving.',
      transfer: 'Include zero and powers of two. Express the bound using the numeric value and then its binary input length; explain which arithmetic-cost model is being used.'
    }, {
      number: 344,
      title: 'Reverse String',
      slug: 'reverse-string',
      difficulty: 'Easy',
      focus: 'Meet the actual in-place, constant-extra-memory requirement while counting swaps and live state.',
      hint: 'Which positions become final after exchanging the two ends? Store only the two moving indices and the temporary values needed for one swap.',
      transfer: 'Compare a recursive two-end swap and a slice-based copy. Both may reverse correctly, but account for frames and copied output before claiming they satisfy the stated memory bound.'
    }]
  }, {
    id: 'core',
    title: 'Core · count calls, retain results and compare shapes',
    introduction: 'Attempt these after the recurrence and repeated-work sections. Maximum Depth is an intentional revisit from Trees: this time justify work and peak storage for contrasting shapes.',
    problems: [{
      number: 509,
      title: 'Fibonacci Number',
      slug: 'fibonacci-number',
      difficulty: 'Easy',
      focus: 'Compare the definition-shaped recursion with memoized and iterative computations of the same value.',
      hint: 'Write down which arguments appear repeatedly in a small call tree. How many distinct indices need answers?',
      transfer: 'Report calls, additions and maximum live depth separately. Explain why constant-count integer variables do not mean constant bit storage for arbitrarily large n.'
    }, {
      number: 50,
      title: 'Pow(x, n)',
      slug: 'powx-n',
      difficulty: 'Medium',
      focus: 'Reuse a computed half-power and extend the lesson’s nonnegative-exponent function to the statement’s signed exponent contract.',
      prerequisite: 'The exponentiation example, reciprocal arithmetic and floating-point rounding. The problem constrains zero-base cases differently from the lesson’s integer example.',
      hint: 'For an even exponent, can one half-power be reused twice? Handle a negative exponent once before entering the shrinking nonnegative recurrence.',
      transfer: 'Test zero exponent, negative base, odd/even exponent and a large negative exponent. Do not call the half-power function twice. Distinguish multiplication counts from bit or floating-point numerical behavior.'
    }, {
      number: 104,
      title: 'Maximum Depth of Binary Tree',
      slug: 'maximum-depth-of-binary-tree',
      difficulty: 'Easy',
      focus: 'Analyze a familiar recursive result on a chain and a balanced shape with the same node count.',
      prerequisite: 'Binary-tree children and the earlier Trees lesson. The statement counts nodes along a longest root-to-leaf route.',
      hint: 'Return a base value for an absent child and combine completed child depths. For a deep tree, what would an explicit stack have to remember?',
      transfer: 'A long allowed chain can exceed Python’s recursion depth. Compare DFS frame space with BFS frontier width, and justify why visiting two children still gives linear work in the number of nodes.'
    }]
  }, {
    id: 'stretch',
    title: 'Optional transfer · derive a new state before reusing a recurrence',
    optional: true,
    introduction: 'This extends the Fibonacci mechanism into counting choices. It is a bridge to the later Dynamic Programming topic, not a substitute for its broader state-design practice.',
    problems: [{
      number: 70,
      title: 'Climbing Stairs',
      slug: 'climbing-stairs',
      difficulty: 'Easy',
      prerequisite: 'Counting disjoint possibilities and the memoization/iteration bridge in this lesson.',
      focus: 'Define what a subproblem counts and explain why the last possible move partitions the answers.',
      hint: 'Separate routes by their final move, then identify the smaller destination for each category. Choose base values that agree with a direct enumeration.',
      transfer: 'Change the allowed move sizes to one or three and derive the bases again. Explain why the old recurrence cannot simply be copied, then compare recursive repeated work with stored states.'
    }]
  }],
  readiness: ['Derive a bound from execution counts rather than indentation alone.', 'Prove progress and distinguish total calls from simultaneous frames.', 'Explain a storage tradeoff and solve a changed contract without the hint.', 'Reattempt assisted tasks later and mix one with a prior graph or tree problem whose structure is not labelled.'],
  localBridge: 'First complete the shrinking-loop, termination, copying and threshold-count exercises above. External statements provide different contracts; their difficulty labels do not measure your learning or guarantee readiness for every interview.'
};
