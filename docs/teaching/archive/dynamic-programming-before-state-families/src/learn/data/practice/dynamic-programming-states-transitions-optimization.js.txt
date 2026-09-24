export default {
  topicId: 'dynamic-programming-states-transitions-optimization',
  verifiedOn: '10 September 2026',
  introduction: 'Before coding, write the state in a sentence, the base cases, legal transitions, dependency order and requested result. Then ask which histories the key merges. These problems vary the state and objective rather than reward memorizing a table shape.',
  groups: [{
    id: 'foundation',
    title: 'Foundation · rebuild a complete recurrence',
    introduction: 'Use the free-suffix and grid sections. Solve without looking at the local implementation, then check boundary cases before compressing memory.',
    problems: [{
      number: 198,
      title: 'House Robber',
      slug: 'house-robber',
      difficulty: 'Medium',
      focus: 'Transfer the nonadjacent-session recurrence to the public statement and distinguish value-only output from a witness.',
      hint: 'If you select this position, which next position is free? Keep the entry contract identical in both recursive branches.',
      transfer: 'Allow negative rewards and an empty choice, then require at least one selection. Explain which base or state changes rather than retaining zero everywhere.'
    }, {
      number: 64,
      title: 'Minimum Path Sum',
      slug: 'minimum-path-sum',
      difficulty: 'Medium',
      focus: 'Define a cell total that includes the destination cell exactly once and derives from legal incoming directions.',
      hint: 'Every path into an interior cell has a final move from above or left. Separate the start from missing predecessors.',
      transfer: 'Add obstacles and a requested route; then allow negative cell costs. Explain why right/down acyclicity matters more than nonnegativity here.'
    }, {
      number: 63,
      title: 'Unique Paths II',
      slug: 'unique-paths-ii',
      difficulty: 'Medium',
      focus: 'Change the objective from minimum cost to the number of paths and make a blocked start meaningful.',
      hint: 'The zero-move route contributes one way only when the start is open. The two final-move cases are disjoint.',
      transfer: 'On paper trace the one-row update with a blocked cell in the first row. What stale value must be erased, and why does the left entry already belong to the new row?'
    }]
  }, {
    id: 'core',
    title: 'Core · change dimensions, reuse and counting rules',
    introduction: 'Attempt after sequence, capacity and counting sections. A similar-looking recurrence may count a different object or permit an illegal repeated item.',
    problems: [{
      number: 1235,
      title: 'Maximum Profit in Job Scheduling',
      slug: 'maximum-profit-in-job-scheduling',
      difficulty: 'Hard',
      focus: 'Carry the finish-ordered weighted-interval recurrence into separate start/end/profit arrays and preserve touching-interval compatibility.',
      hint: 'For each finish-ordered job, compare skipping it with taking it plus the best compatible prefix. Binary-search the boundary among earlier finish times.',
      transfer: 'Use the earlier Greedy counterexample with a value-10 long job versus two value-4 short jobs. Recover original job indices, handle tied finishes, and explain why sorting by value alone has no replacement proof.'
    }, {
      number: 1143,
      title: 'Longest Common Subsequence',
      slug: 'longest-common-subsequence',
      difficulty: 'Medium',
      focus: 'Explain why two prefix lengths form the key, and why subsequence matching permits gaps.',
      hint: 'Partition cases by the last characters of the two prefixes. The empty-prefix row and column are part of the definition.',
      transfer: 'Recover a witness on AB versus BA and state a tie policy. Then ask for a contiguous common substring and explain why merely returning the same table entry fails.'
    }, {
      number: 72,
      title: 'Edit Distance',
      slug: 'edit-distance',
      difficulty: 'Medium',
      focus: 'Assign a precise insertion, deletion or replacement meaning to each grid dependency.',
      hint: 'Transforming an empty prefix requires as many insertions as the other prefix length. A matching character has cost zero.',
      transfer: 'Make replacement cost three while insertion/deletion cost one. Decide whether a replacement or a delete-plus-insert wins; adapt the recurrence without changing the legal operation set.'
    }, {
      number: 416,
      title: 'Partition Equal Subset Sum',
      slug: 'partition-equal-subset-sum',
      difficulty: 'Medium',
      focus: 'Reduce equal partition to exact target reachability, retaining occurrence identity and 0/1 reuse.',
      hint: 'An odd total cannot split evenly. For an even total, how much must one subset sum to? Descending totals keep a value from appearing twice.',
      transfer: 'Explain why [3,5] cannot reach 6 with one-use items while [3,5,3] can. Discuss what signed values change about the table index range.'
    }, {
      number: 322,
      title: 'Coin Change',
      slug: 'coin-change',
      difficulty: 'Medium',
      focus: 'Minimize the number of reusable positive denominations, separating impossible from zero coins.',
      hint: 'Choose a final coin and solve the smaller remaining amount. Positive denominations make that dependency acyclic.',
      transfer: 'Compare [1,3,4] at amount 6 with largest-coin-first. Then constrain each occurrence to one use and explain why the state/order must change.'
    }, {
      number: 518,
      title: 'Coin Change II',
      slug: 'coin-change-ii',
      difficulty: 'Medium',
      focus: 'Count combinations without counting their reorderings as different answers.',
      hint: 'After processing a denomination, let each total count only combinations using denominations processed so far. Reuse the current denomination through ascending amounts.',
      transfer: 'List all results for denominations [1,3], amount 4. Swap the loops and explain exactly which two answers were previously merged.'
    }, {
      number: 300,
      title: 'Longest Increasing Subsequence',
      slug: 'longest-increasing-subsequence',
      difficulty: 'Medium',
      focus: 'Start with best length ending at each index, then justify a minimum-tail dominance summary.',
      hint: 'A previous endpoint can precede this value only if it occurs earlier and is strictly smaller. Equal values must not increase a strictly increasing length.',
      transfer: 'Show that final tails [2,5,6] for input [3,5,6,2] is not an actual subsequence. Add predecessor indices if the output must be a witness; use a different bound for nondecreasing order.'
    }]
  }, {
    id: 'stretch',
    title: 'Optional modeling extensions · make hidden history explicit',
    introduction: 'These are new modeling tasks, not gates for continuing the module. Their prerequisites are named so a difficult problem does not masquerade as a missing core explanation.',
    optional: true,
    problems: [{
      number: 309,
      title: 'Best Time to Buy and Sell Stock with Cooldown',
      slug: 'best-time-to-buy-and-sell-stock-with-cooldown',
      difficulty: 'Medium',
      prerequisite: 'State sufficiency and simultaneous old/new state updates from this lesson; read the public buy/sell/cooldown contract before modeling it.',
      focus: 'Distinguish holding, just sold and free-to-buy histories at the same day; this is a toy optimization model, not investment advice.',
      hint: 'Two histories at one day may have different legal actions. Write the legal transitions before combining their values, and read all previous-day values before overwriting any.',
      transfer: 'Change the cooldown to two days. Explain what additional memory is needed and which histories still share a future.'
    }, {
      number: 980,
      title: 'Unique Paths III',
      slug: 'unique-paths-iii',
      difficulty: 'Hard',
      prerequisite: 'Backtracking path-local used sets, grid adjacency and this lesson’s subset+endpoint state. Adapt optimization to counting and enforce the designated end condition.',
      focus: 'Count visit-every-cell paths; ordinary right/down grid DP and a global visited flag no longer describe the contract.',
      hint: 'Number only traversable cells. The subset and endpoint describe which cells remain and which moves are legal; accept the end only when every required cell has been visited.',
      transfer: 'Compare exponential state storage with direct path enumeration on a sparse corridor and on a branching board. Memoization is not automatically the better choice for every reachable-state graph.'
    }]
  }],
  readiness: ['Derive a recurrence for an unseen task and prove that its choices cover the requested outputs without illegal overlap or omission.', 'Construct a counterexample to an incomplete key or incorrect update order, then repair the invariant.', 'Separate number of states, transition work, output storage, recursion depth and integer bit costs; recognize pseudo-polynomial and exponential bounds.', 'Return and validate a witness, and state whether the tie policy guarantees anything beyond deterministic output.'],
  localBridge: 'After a break, solve the local command-stream task and one unfamiliar mixed problem without pattern labels. Reattempt a failed problem from its state contract, then change a constraint and compare against a small independent brute-force oracle. A finite collection cannot guarantee every interview problem.'
};
