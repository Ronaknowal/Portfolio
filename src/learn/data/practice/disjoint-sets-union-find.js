export default {
  topicId: 'disjoint-sets-union-find',
  verifiedOn: '10 September 2026',
  introduction: 'Turn a partition into a useful answer. These problems vary the input representation, arrival order, equivalence policy and required output; they do not all ask for a bare find/union class.',
  groups: [{
    id: 'foundation',
    title: 'Foundation · recognize the partition',
    introduction: 'Begin after implementing the class and checking it against graph traversal. Count all vertices, including isolated ones, and state whether links are directed.',
    problems: [{
      number: 547,
      title: 'Number of Provinces',
      slug: 'number-of-provinces',
      difficulty: 'Medium',
      focus: 'Read a symmetric adjacency matrix as a set of merging relationships and compare with a traversal baseline.',
      hint: 'What does every city contribute before examining the matrix? Which matrix entries actually reduce the number of groups?',
      transfer: 'Test an identity matrix and a fully connected matrix. Reading the matrix costs O(n²) even when the DSU operations are almost constant amortized time; do not report only the cost of one union.'
    }, {
      number: 684,
      title: 'Redundant Connection',
      slug: 'redundant-connection',
      difficulty: 'Medium',
      focus: 'Identify a link that cannot reduce the component count under the statement’s tree-plus-one-edge contract.',
      hint: 'Process edges in their given order. What does a failed union establish about a route that already exists? Use the promise of exactly one extra edge when justifying which answer is selected.',
      transfer: 'Explain why the failed edge is the last edge of the unique cycle encountered in input order. Then remove the tree-plus-one-edge promise: there may be several redundant edges, and a new output contract is needed. Account for the statement’s one-based labels.'
    }]
  }, {
    id: 'transfer',
    title: 'Core transfer · constraints, resources and output',
    introduction: 'These use the same invariant but require a different modeling decision. Write down what each DSU element represents before coding.',
    problems: [{
      number: 990,
      title: 'Satisfiability of Equality Equations',
      slug: 'satisfiability-of-equality-equations',
      difficulty: 'Medium',
      focus: 'Separate required equivalence from forbidden equality, including constraints that become contradictory only after later merges.',
      hint: 'Which constraints create groups, and which constraints must be checked against the final groups? Think about an inequality appearing before two equalities that connect its endpoints.',
      transfer: 'Check a variable unequal to itself and a long equality chain. Explain how assigning one distinct integer per component proves acceptance, rather than only showing that your code failed to find a contradiction.'
    }, {
      number: 1319,
      title: 'Number of Operations to Make Network Connected',
      slug: 'number-of-operations-to-make-network-connected',
      difficulty: 'Medium',
      focus: 'Combine component count with a resource argument about reusable connections.',
      hint: 'How many links must join c separate components? Which existing links can be removed while preserving a connecting forest within every component?',
      transfer: 'Distinguish “we need c−1 links” from “we have enough spare cables.” Prove necessity and sufficiency under unrestricted reconnection, and explain which real-world placement restrictions would invalidate that conclusion.'
    }, {
      number: 721,
      title: 'Accounts Merge',
      slug: 'accounts-merge',
      difficulty: 'Medium',
      focus: 'Use shared identifiers to model transitive record membership, then construct the required unique, sorted output.',
      prerequisite: 'Dictionaries and sets from earlier programming/DSA topics; Python sorted for the required email order. The lesson’s record-grouping example returns indices, so this problem adds collecting and ordering identifiers.',
      hint: 'Choose either accounts or emails as elements and keep that choice consistent. What dictionary can avoid comparing every pair? A shared display name does not establish shared identity.',
      transfer: 'Test matching names without shared emails, a transitive chain, and duplicate identifiers within one record. Include hashing/string and output sorting costs; choosing different representative IDs should not change the merged membership.'
    }]
  }],
  readiness: ['Rebuild size-weighted union and iterative compression with correct empty, duplicate and invalid-input behavior.', 'State which facts belong to the partition and which are changeable implementation choices, including root identity and parent shape.', 'Compare a small test suite with an independent BFS component or explicit-set baseline, not only with the same implementation ported to another language.', 'Explain an application’s equivalence policy, metadata, amortized sequence cost and output cost; reject DSU when the query needs information it discarded.'],
  localBridge: 'Return to the disconnected-pair exercise and the grid ring after these problems. Change an input and derive the update before running the code. Rollback and potential DSU are optional extensions, not prerequisites for the next module topic.'
};
