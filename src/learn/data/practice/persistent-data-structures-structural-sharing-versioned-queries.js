export default {
  topicId: 'persistent-data-structures-structural-sharing-versioned-queries',
  verifiedOn: '2026-09-10',
  introduction: 'These public problems practise single-timeline histories. Neither requires full branching persistence. Complete the local branching and identity tasks too; a historical get alone does not test structural sharing.',
  groups: [{
    id: 'historical-lookups',
    title: 'Design compact histories before reaching for a tree',
    introduction: 'Identify what changes, what can be queried, and whether any old version can become the source of an update.',
    problems: [
      { number: 1146, title: 'Snapshot Array', slug: 'snapshot-array', difficulty: 'Medium', focus: 'Separate the working snapshot ID from saved snapshots; coalesce repeated writes to one index before saving, then find the last eligible write.', hint: 'A snapshot where an index did not change needs no new record for that index. Search for the first history time greater than the requested saved ID, then step back.', transfer: 'Add sum(version, left, right) and allow updates from any saved version. Explain why a root per branch plus aggregate tree becomes useful and why global timestamp lookup is insufficient.', prerequisite: 'Binary search for an upper bound; section 5 explains its history invariant.' },
      { number: 981, title: 'Time Based Key-Value Store', slug: 'time-based-key-value-store', difficulty: 'Medium', focus: 'Choose a separate ordered history for each key. The official statement guarantees strictly increasing set timestamps; a missing prior value returns the empty string.', hint: 'A missing exact timestamp is not a failure: search for the greatest stored timestamp no later than the query. A key absent from the store has no predecessor.', transfer: 'Now permit out-of-order writes and repeated timestamps. Specify the tie policy and replace the append-only invariant with ordered insertion or a suitable search tree; account for its cost.', prerequisite: 'Dictionary lookup, sorted histories and predecessor search; string values need no interval aggregate.' },
    ],
  }],
  readiness: ['Derive update, snapshot and lookup costs in terms of the actual histories, without copying all entries at snap time.', 'Keep old roots unchanged after branching from a nonlatest version; assert shared child identities and count newly allocated path nodes.', 'Explain why two unrelated branch histograms may produce negative differences and cannot directly support kth selection.', 'State payload, lifetime, empty-array, interval and numeric contracts before calling a structure persistent.'],
  localBridge: 'Implement the independent versioned range-array task below and compare every historical query with a copied-array oracle. Then solve the linked CSES Range Queries and Copies task, translating its inclusive one-based endpoints carefully.',
};
