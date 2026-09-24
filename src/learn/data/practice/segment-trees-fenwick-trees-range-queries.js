export default {
  topicId: 'segment-trees-fenwick-trees-range-queries',
  verifiedOn: '2026-09-10',
  introduction: 'Choose the query/update contract before choosing a tree. First solve each task by a small direct scan, then state exactly what each cached summary or surviving deque candidate means. Convert endpoint conventions explicitly.',
  groups: [
    {
      id: 'static-to-changing',
      title: 'Change the update contract',
      introduction: 'The two nearly identical interfaces need different treatment once assignments arrive.',
      problems: [
        { number: 303, title: 'Range Sum Query - Immutable', slug: 'range-sum-query-immutable', difficulty: 'Easy', focus: 'Reuse a static prefix cache for repeated sums.', prerequisite: 'Official endpoints are both included and the input is nonempty. Translate [left,right] to the lesson’s [left,right+1).', hint: 'Which two prefixes share exactly the unwanted elements before left?', transfer: 'Allow one assignment between every two queries. Identify every cached prefix it invalidates before deciding whether this method is still appropriate.' },
        { number: 307, title: 'Range Sum Query - Mutable', slug: 'range-sum-query-mutable', difficulty: 'Medium', focus: 'Maintain sums under point replacement using either a segment tree or a Fenwick delta.', prerequisite: 'update(index,val) assigns a new value; it does not add val. Query endpoints are inclusive. Values may be negative.', hint: 'In a Fenwick implementation, what must be subtracted from the new value before propagating it?', transfer: 'Replace sum with minimum while preserving arbitrary point assignments. Explain why ordinary prefix subtraction fails and which segment summary still works.' },
      ],
    },
    {
      id: 'moving-candidates',
      title: 'Prove why a candidate can leave',
      introduction: 'These problems share a deque representation but require different removal proofs. Complete both local investigations before opening their hints.',
      problems: [
        { number: 239, title: 'Sliding Window Maximum', slug: 'sliding-window-maximum', difficulty: 'Hard', focus: 'Separate expiry from newer-candidate domination and account for all deque operations.', prerequisite: 'The official width satisfies 1 ≤ k ≤ n and returns maximum values only. The local example also returns the newest tied original index and handles a width larger than n.', hint: 'If a newer value is at least as large, can the older one ever become the maximum of a later full window that contains it?', transfer: 'Return the oldest tied argmax index instead. Change one comparison, retain equal candidates and test [2,2,2] before claiming the tie contract is unchanged.' },
        { number: 862, title: 'Shortest Subarray with Sum at Least K', slug: 'shortest-subarray-with-sum-at-least-k', difficulty: 'Hard', focus: 'Turn a signed sum requirement into a threshold on earlier prefix boundaries.', prerequisite: 'The subarray must be nonempty, the target is positive, and negative input values are legal. Return -1 when absent; the local program returns None or a length/range witness.', hint: 'For a fixed future end, compare an older prefix with a later prefix that is no larger. Separately explain why a start with a valid ending now need not stay for a longer ending.', transfer: 'Ask for the number of qualifying subarrays instead of the shortest one. Explain why retiring a valid start would discard future counts and why the same deque is no longer sufficient.' },
      ],
    },
    {
      id: 'rank-frequency',
      title: 'Transfer positions into value ranks',
      optional: true,
      introduction: 'Use the coordinate-compression and frequency-tree mechanisms, after the weighted DP example is clear.',
      problems: [
        { number: 315, title: 'Count of Smaller Numbers After Self', slug: 'count-of-smaller-numbers-after-self', difficulty: 'Hard', focus: 'Maintain counts of already processed right-hand occurrences at compressed value ranks.', prerequisite: 'Strictly smaller excludes equal values. Process original positions right to left, query ranks below the current rank, then add this occurrence; do not sort away original position identity.', hint: 'What must the tree contain immediately before answering the current position? Compare a frequency increment with the maximum-score update used in the weighted DP example.', transfer: 'Change smaller to smaller-or-equal, then count pairs with a separate numerical threshold. Decide whether only the rank boundary changes and use binary search when the threshold is not a stored coordinate.' },
      ],
    },
  ],
  readiness: [
    'Specify half-open versus inclusive endpoints, empty intervals, point assignment versus delta, and the identity for the chosen combine.',
    'Derive a segment query’s ordered disjoint cover and a Fenwick cell’s low-bit interval without copying a diagram.',
    'Compose range assignment and addition in the correct order, distinguish a pending tag from stale data, and compare mixed operations with a direct array.',
    'Prove both deque deletion rules and the aggregate linear operation bound; return and validate original witness indices.',
    'Choose rank-frequency or rank-maximum state from its meaning, handle equal values, and reconstruct a weighted strictly increasing witness.',
  ],
  localBridge: 'Local exercises cover richer segment summaries, subtree flattening, lazy composition and algebraic counterexamples. The CSES range-add/range-set task below adds direct lazy practice. Five LeetCode selections assess distinct operations; their difficulty labels and an accepted submission are not a substitute for explaining the invariants or adapting a changed contract.',
};
