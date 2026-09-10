export default {
  topicId: 'randomized-algorithms-sampling-error-guarantees',
  verifiedOn: '10 September 2026',
  introduction: 'These problems exercise distribution contracts, memory constraints and exact selection. A successful judge run does not establish uniformity: accompany implementations with a probability argument and exhaustive small-state checks. Local verification and error-budget exercises remain essential.',
  groups: [{
    id: 'foundation',
    title: 'Foundation · build the intended distribution',
    introduction: 'Reconstruct the raw outcome space before choosing the transformation. The platform calls every problem here Medium; stages reflect the learning dependencies rather than those labels.',
    problems: [{
      number: 384,
      title: 'Shuffle an Array',
      slug: 'shuffle-an-array',
      difficulty: 'Medium',
      focus: 'Return every permutation with equal probability while preserving a resettable original configuration. The official array entries are distinct.',
      hint: 'Which positions have already received their final items? How many equally eligible occurrences remain?',
      transfer: 'Permit duplicate values while retaining occurrence identity. Enumerate all paths for three labeled items, and explain why uniform single-position counts do not prove a uniform permutation. Mutate the returned list to test whether the original snapshot is truly independent.'
    }, {
      number: 528,
      title: 'Random Pick with Weight',
      slug: 'random-pick-with-weight',
      difficulty: 'Medium',
      focus: 'Choose an index in proportion to its positive integer weight. A random outcome sequence is not a fixed expected-output sequence.',
      hint: 'Can a uniform integer ticket represent more than one ticket for a heavier index? What happens exactly at a cumulative boundary?',
      transfer: 'Add zero weights and reject a zero total. Enumerate every ticket on a tiny example. If weights change frequently, adapt the prefix lookup to the previously taught Fenwick tree and justify the search invariant.'
    }, {
      number: 470,
      title: 'Implement Rand10() Using Rand7()',
      slug: 'implement-rand10-using-rand7',
      difficulty: 'Medium',
      focus: 'Construct a uniform result on 1 through 10 using only the supplied uniform 1-through-7 source; built-in random calls are excluded by this problem.',
      hint: 'Two independent source calls give how many equally likely pairs? Can you retain a multiple of ten outcomes before mapping them?',
      transfer: 'Derive the expected number of rand7 calls and the chance of exceeding a chosen number of attempts. State an unbiased capped API with a distinct failure result. Explore recycling rejected outcomes only after proving the simpler construction.'
    }]
  }, {
    id: 'core',
    title: 'Core · change the eligible population or the resource constraint',
    introduction: 'Use reservoir invariants and exact selection. The scan cost of a linked-list sample and the preprocessing cost of indexed sampling are different tradeoffs.',
    problems: [{
      number: 382,
      title: 'Linked List Random Node',
      slug: 'linked-list-random-node',
      difficulty: 'Medium',
      focus: 'Choose every node with equal probability, then return its value. The unknown-length, constant-extra-space follow-up motivates a one-pass reservoir; repeated values do not merge node identities.',
      hint: 'After seeing one more node, what replacement probability makes both old and new occurrences equally likely?',
      transfer: 'Keep a uniform unordered set of k nodes and prove the joint subset distribution. Explain the cost of one getRandom call versus caching all nodes, and distinguish independent fresh full passes from correlated maintained stream snapshots.'
    }, {
      number: 398,
      title: 'Random Pick Index',
      slug: 'random-pick-index',
      difficulty: 'Medium',
      focus: 'Return an index uniformly among occurrences matching an existing target. The target is guaranteed present in the official statement.',
      hint: 'Should the denominator count all scanned positions or only eligible occurrences?',
      transfer: 'Define an absent-target policy, compare preprocessing each value’s occurrence list with per-query reservoir scans, and test a target appearing only late in the array. State time and memory under repeated queries.'
    }, {
      number: 215,
      title: 'Kth Largest Element in an Array',
      slug: 'kth-largest-element-in-an-array',
      difficulty: 'Medium',
      focus: 'Find the sorted-order kth largest value, counting duplicates rather than distinct values. Randomized selection gives an exact answer with variable work.',
      hint: 'Translate largest-rank to a zero-based ascending rank. Which region contains that rank after a three-way partition?',
      transfer: 'Compare every result on small duplicate-filled arrays with sorting. Exhibit a quadratic pivot sequence, explain the expected bound’s random experiment, and design a capped no-answer API without silently claiming worst-case linear work.'
    }]
  }],
  readiness: ['Can prove an exact probability from equally weighted choice paths and find a counterexample to a plausible biased transformation.', 'Can distinguish uniform occurrences, distinct values, unordered subsets and output ordering; can adapt state when the eligible population changes.', 'Can separate answer correctness, expected work, a tail bound and an explicit exhausted-budget result.', 'Can reconstruct a solution after closing hints, then explain an unseen change of constraints without copying a familiar loop.'],
  localBridge: 'Retain the independent Freivalds, majority, union-budget and estimation exercises: these platform questions do not teach the whole error-guarantee contract. Mix earlier selection/hash/range problems into later reviews, and reconsider the assumptions for new streaming or adversarial applications. These six varied questions are not a universal-interview guarantee.'
};
