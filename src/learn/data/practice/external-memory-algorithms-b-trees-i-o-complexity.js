export default {
  topicId: 'external-memory-algorithms-b-trees-i-o-complexity',
  verifiedOn: '10 September 2026',
  introduction: 'These official problems practise replacement order and bounded merge frontiers. They do not test B-tree page occupancy, physical I/O or crash recovery; keep the local page-count and repair exercises alongside them.',
  groups: [{
    id: 'core',
    title: 'Core transfer · resident order and merge frontiers',
    introduction: 'Reconstruct the relevant earlier map/list or heap mechanism, then explain the new page-level interpretation.',
    problems: [{
      number: 146,
      title: 'LRU Cache',
      slug: 'lru-cache',
      difficulty: 'Medium',
      prerequisite: 'Hash maps and a doubly linked recency list; the earlier linked-list and hashing lessons supply these mechanisms.',
      focus: 'The official contract requires average O(1) get and put. Existing-key updates and successful reads refresh recency.',
      hint: 'How can a key locate its list node without scanning, and which end should represent the next eviction?',
      transfer: 'Interpret entries as pages. Add dirty state, distinguish a miss from a writeback, and count a final flush. Explain why the official item-capacity API alone does not model bytes, disk latency or durability.'
    }, {
      number: 23,
      title: 'Merge k Sorted Lists',
      slug: 'merge-k-sorted-lists',
      difficulty: 'Hard',
      prerequisite: 'Heap-based k-way merging from Heaps, Priority Queues & Tries.',
      focus: 'Maintain one current candidate per nonempty sorted input; preserve duplicate occurrences and handle empty lists.',
      hint: 'Which values can be the next global minimum, and which one input advances after emitting it?',
      transfer: 'Replace each linked list with a paged run. Reserve an output page, derive the allowed fan-in, and count transfers. Then stop after the first z records: explain which reads and final output writes can disappear.'
    }]
  }, {
    id: 'stretch',
    optional: true,
    title: 'Optional variants · memory contracts and partial output',
    introduction: 'These are useful algorithmic comparisons, not additional prerequisites for completing the module.',
    problems: [{
      number: 148,
      title: 'Sort List',
      slug: 'sort-list',
      difficulty: 'Medium',
      prerequisite: 'Linked-list splitting/merging and iterative versus recursive auxiliary-space accounting.',
      focus: 'The follow-up asks for O(n log n) time and O(1) extra space. Recursive call frames count as extra space.',
      hint: 'Can merge widths grow in rounds without storing a recursive call stack?',
      transfer: 'Contrast constant auxiliary RAM with locality: linked nodes may occupy unrelated pages. Explain why an in-memory complexity proof does not establish the external-sort I/O bound.'
    }, {
      number: 378,
      title: 'Kth Smallest Element in a Sorted Matrix',
      slug: 'kth-smallest-element-in-a-sorted-matrix',
      difficulty: 'Medium',
      prerequisite: 'Sorted row streams and a min-heap; binary-search alternatives require a separately justified counting predicate.',
      focus: 'Both rows and columns are sorted. The rank counts occurrences, not distinct values; the statement requests better than O(n²) memory.',
      hint: 'For a row-stream solution, which frontier values are enough to identify the next occurrence?',
      transfer: 'Stop a k-way merge at the requested rank and include repeated values. Predict how page size and input buffering affect a small output prefix. The official stronger follow-ups are optional extensions with their own proofs.'
    }]
  }],
  readiness: ['Can derive page reads and dirty writes from an access trace rather than equating accesses with transfers.', 'Can repair a B-tree deletion and explain why a parent separator moves during borrowing or merging.', 'Can derive a merge budget and count partial pages for changed N, B and M.', 'Can distinguish a copied root, a durable reachable tree and an acknowledged commit.', 'Can reattempt with hints closed, then justify a changed constraint or unfamiliar combination of these mechanisms.'],
  localBridge: 'The local B-tree, B+ range, external-file and crash-stage exercises cover contracts missing from these four platform statements. Revisit the earlier DSA practice sets with mixed unseen cases; no finite list guarantees every interview problem or production storage design.'
};
