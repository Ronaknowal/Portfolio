export default {
  topicId: 'network-flow-minimum-cuts-bipartite-matching',
  verifiedOn: '2026-09-10',
  introduction: 'These public official statements test whether a bipartite model applies and whether a less obvious problem can be reduced to matching. The local allocation, residual, cut and lower-bound exercises supply the direct flow practice; an external problem count is not a substitute for those proofs. No judge submission or editorial access is claimed.',
  groups: [
    {
      id: 'check-the-model',
      title: 'Check the structural precondition',
      introduction: 'Revisit the graph-search foundation before assuming that a matching/cover equality applies. This stage checks bipartition; it does not require a flow solver.',
      problems: [
        {
          number: 886,
          title: 'Possible Bipartition',
          slug: 'possible-bipartition',
          difficulty: 'Medium',
          focus: 'Decide whether a conflict graph permits two parts, including disconnected components and isolated people.',
          prerequisite: 'BFS or DFS two-coloring; up to 2,000 people and 10,000 distinct conflicts. Every conflict must connect different parts.',
          hint: 'Choose a color for an unvisited component root. What must each neighboring vertex receive, and what constitutes a contradiction?',
          transfer: 'Return the two parts when possible, or reconstruct an odd-cycle obstruction using search parents when not. Then explain why finding two valid parts does not itself produce a maximum matching or minimum cover.',
        },
      ],
    },
    {
      id: 'recover-an-optimization-model',
      title: 'Turn a selection problem into matching',
      introduction: 'The platform supports more than one correct technique. Use this lesson to justify a conflict graph, its bipartition and an optimum selection witness.',
      problems: [
        {
          number: 1349,
          title: 'Maximum Students Taking Exam',
          slug: 'maximum-students-taking-exam',
          difficulty: 'Hard',
          focus: 'Build a conflict graph on usable seats, derive a minimum vertex cover through bipartite matching, and recover a largest conflict-free selection.',
          prerequisite: 'Grid coordinates, bipartite matching/cover and the complement identity: vertices outside a cover form an independent set. The official grid is at most 8×8; conflicts are horizontal or diagonal to the preceding row, not directly vertical.',
          hint: 'Every forbidden pair changes the column by one. Which coordinate parity gives the two parts? Why does row-plus-column checkerboard parity fail on diagonal conflicts?',
          transfer: 'Return the actual chosen seats, not only their count, and verify every forbidden pair. Add a directly vertical conflict: recheck bipartition instead of reusing it blindly. Add unequal seat rewards: cardinality matching no longer solves the weighted objective unchanged.',
        },
      ],
    },
  ],
  readiness: [
    'Audit edge bounds and internal conservation before using a flow value as a lower bound.',
    'Reconstruct a residual augmentation with original-edge IDs and explain each forward addition or reverse cancellation.',
    'Return a feasible flow and valid directed cut of equal value, including zero-capacity, parallel and antiparallel edges.',
    'Recover an integral maximum matching, alternating-reachability minimum cover and an explicit Hall shortage when left saturation is impossible.',
    'Derive node splitting, supply/demand and lower-bound gadgets from their equations and recover witnesses in the original domain.',
    'Explain the solver’s numeric, arithmetic, memory and recursion contracts; distinguish a capacity objective from costs or multiple commodities.',
    'For a binary labeling, prove equality of energy and cut capacity for every assignment, not only one output.',
  ],
  localBridge: 'The nine local tasks test conservation failures, reverse-edge identity, cut direction, shortage recovery, lower-bound signs, binary energies, changed objectives, capacitated allocation and optional random-contraction limits. Rebuild a new case after closing the solutions. The official Maximum Number of Accepted Invitations page was Premium-only during research, so its unavailable statement was not presented as verified public practice.',
};
