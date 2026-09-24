export default {
  summary: 'Maintain merging equivalence classes with a parent forest. Derive root-only unions, size/rank weighting and path compression, then model incremental connectivity, shared identifiers and active regions with honest query limits.',
  outcomes: ['Distinguish a partition, representative and input graph', 'Implement validated size-weighted union and iterative full compression', 'Explain initialization, root invariants, termination, metadata and amortized costs', 'Model redundant links, equality consistency and shared identifiers', 'Count incremental islands without counting closed cells or duplicate neighbor groups', 'Choose traversal, rollback or another model when the query exceeds ordinary DSU'],
  prerequisites: ['Graphs: Representations, BFS & DFS', 'Trees & Binary Search Trees'],
  sequence: ['Follow connected groups as links arrive', 'Compare eager labels with a parent forest', 'Implement and justify root-only merges', 'Derive size weighting and trace path compression', 'Distinguish rank, height and amortized work', 'Model constraints, resources and meaningful component metadata', 'Activate grid cells and count successful joins', 'Understand deletion limits and optional rollback', 'Solve independent and curated practice'],
  visual: {
    type: 'Equivalent-partition forest figure, parent-pointer traces and active-island grid',
    question: 'What changes in the implementation while the same membership relation is preserved?',
    interaction: 'Apply bounded unions, compare attachment policies, step exact compression rewrites, and activate cells while inspecting component memberships and successful joins.'
  },
  practice: {
    task: 'Repair a non-root merge, count disconnected pairs, preserve metadata under compression and handle a ring with repeated neighbor components; transfer to official connectivity and equivalence problems.',
    success: 'Pass independent partition/flood-fill tests and explain invariants, operation sequence costs, output costs, input boundaries and unsupported directed/deletion queries.'
  },
  misconceptions: ['Immediate parents identify components', 'A representative must be the smallest member', 'Parent pointers are original graph edges', 'Rank is current height after compression', 'Almost constant amortized means every operation is constant time', 'Every neighbor or repeated edge causes a new merge', 'Allocated closed grid cells count as islands', 'Undoing a parent link supports arbitrary graph deletion'],
  sources: ['https://algs4.cs.princeton.edu/15uf/', 'https://algs4.cs.princeton.edu/code/javadoc/edu/princeton/cs/algs4/UF.html', 'https://www.coursera.org/learn/algorithms-part1'],
  depth: 'core',
  reviewFocus: 'Root-only attachment, root metadata versus stale entries, failed-union compression, size doubling, rank/height separation, sequence bound initialization, actual pointer traces, active-grid count, reversible-history semantics, curated prerequisites and desktop/mobile/keyboard correspondence.'
};
