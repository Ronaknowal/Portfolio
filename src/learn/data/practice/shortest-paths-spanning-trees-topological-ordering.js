export default {
  topicId: 'shortest-paths-spanning-trees-topological-ordering',
  verifiedOn: '2026-09-10',
  introduction: 'Choose the question and weight assumptions before choosing an algorithm. These official statements were inspected directly; no judge submission or editorial access is claimed. Restate direction, index bases, impossible-result conventions and whether the output is a route, a value, a connecting tree or an order.',
  groups: [
    {
      id: 'graph-objectives', title: 'Reconstruct the three core objectives',
      introduction: 'Start from local invariants and return contracts. Platform difficulty is metadata, not the teaching order.',
      problems: [
        { number: 743, title: 'Network Delay Time', slug: 'network-delay-time', difficulty: 'Medium', focus: 'Turn a vector of source arrival times into the time when the entire directed network has received a signal.', prerequisite: 'Nodes are labeled 1…n; edge delays may be zero. An unreachable node changes the final result to −1.', hint: 'After obtaining reliable per-node arrival times, which one determines when everyone has heard?', transfer: 'Explain why a sum of arrivals or a minimum spanning tree answers a different question. Return one route to the last receiver and test an isolated receiver and zero-cost cycle.' },
        { number: 1584, title: 'Min Cost to Connect All Points', slug: 'min-cost-to-connect-all-points', difficulty: 'Medium', focus: 'Recognize an implicit complete undirected graph with explicitly defined Manhattan costs.', prerequisite: 'Use the local dense Prim branch to avoid storing every pair; the official input has up to 1,000 distinct points.', hint: 'A point needs its best current connection to the growing tree, not its entire route cost from the starting point.', transfer: 'Compare O(n²) on-demand scans and O(n² log n) explicit sorting/heap choices with their storage. Explain why drawing Euclidean straight-line lengths would use the wrong weight function.' },
        { number: 210, title: 'Course Schedule II', slug: 'course-schedule-ii', difficulty: 'Medium', focus: 'Return a valid order of every course, including courses absent from dependency pairs.', prerequisite: 'An input pair [a,b] means b→a, and impossibility returns an empty array. A valid order need not be unique.', hint: 'Count prerequisites using the edge direction you actually intend. What does a zero count certify?', transfer: 'Return a concrete cycle when the order is impossible, then distinguish it from blocked descendants. Adapt the ready structure if the smallest possible lexicographic order becomes required.' },
      ],
    },
    {
      id: 'changed-route-state', title: 'Change the route state or aggregate',
      introduction: 'Each task changes what a distance, matrix or dependency recurrence must mean.',
      problems: [
        { number: 787, title: 'Cheapest Flights Within K Stops', slug: 'cheapest-flights-within-k-stops', difficulty: 'Medium', focus: 'Keep feasibility under a stop budget while comparing route costs.', prerequisite: 'k stops means at most k+1 flight edges. Prices are positive; absence of a feasible route returns −1.', hint: 'Can a cheaper arrival at a city use too much of the remaining edge budget? Keep old and new generations separate.', transfer: 'Construct a case where one scalar cheapest label loses a feasible continuation. If asked for the actual route, store reconstruction by budget generation and verify its edge count.' },
        { number: 2050, title: 'Parallel Courses III', slug: 'parallel-courses-iii', difficulty: 'Hard', focus: 'Combine dependency readiness with completion times and a critical-chain lower bound.', prerequisite: 'The task guarantees a DAG and allows any number of concurrent courses. Vertex labels are 1-based but duration positions are 0-based.', hint: 'A course waits for all prerequisites, including the one that finishes last.', transfer: 'Recover a critical chain and compare its duration with the makespan. Explain why limiting the number of simultaneous courses invalidates the same optimality claim.' },
      ],
    },
    {
      id: 'specialized-routes', title: 'Optional: revisit the specialized route branches', optional: true,
      introduction: 'Attempt after the local all-pairs and 0/1-deque branches, respectively. These are not prerequisites for the next module topic.',
      problems: [
        { number: 1334, title: 'Find the City With the Smallest Number of Neighbors at a Threshold Distance', slug: 'find-the-city-with-the-smallest-number-of-neighbors-at-a-threshold-distance', difficulty: 'Medium', focus: 'Answer a neighborhood statistic from all-pairs minimum route costs, with a specified tie rule.', prerequisite: 'Edges are bidirectional with positive weights. Count other cities; ties favor the greatest city index. Review Floyd–Warshall or repeated Dijkstra first.', hint: 'A direct edge list is not yet the set of destinations reachable within the cost threshold.', transfer: 'Compare cubic Floyd–Warshall against repeated lazy Dijkstra under sparse and dense inputs, including matrix output storage. Test an exact-threshold route and multiple tied city counts.' },
        { number: 1368, title: 'Minimum Cost to Make at Least One Valid Path in a Grid', slug: 'minimum-cost-to-make-at-least-one-valid-path-in-a-grid', difficulty: 'Hard', focus: 'Derive directed 0/1 edge costs from following or changing each departure cell’s sign.', prerequisite: 'Review Graphs’ grid neighbor bounds and the local 0/1-deque branch. The objective counts sign changes, not moves; a sign may point outside the grid.', hint: 'When moving from a cell, compare your chosen direction with that cell’s existing arrow.', transfer: 'Explain why a nonnegative optimal simple route makes the per-cell edit interpretation consistent. Change edit costs to arbitrary nonnegative values and justify replacing the two-level deque rule.' },
      ],
    },
  ],
  readiness: [
    'Separate a tentative distance, a physical stale queue entry and a finalized optimum; explain exactly where nonnegative weights enter the proof.',
    'Classify a source-target answer as unreachable, finite or unbounded below and recover actual edge-identified witnesses only for finite answers.',
    'Derive edge-budget or permitted-intermediate states, including output and reconstruction costs, rather than relying on an algorithm name.',
    'Prove a tied-weight cut exchange preserves some compatible optimum, distinguish Prim from Dijkstra keys, and include isolated components.',
    'Return a complete dependency order or real cycle witness, separate blocked descendants and derive a critical-path schedule under explicit resource assumptions.',
  ],
  localBridge: 'The local independent exercises cover finite route witnesses, target-specific negative-cycle influence, tied cut/cycle proofs, bottleneck certificates, reweighting counterexamples and scheduling assumptions. The seven selected statements exercise meaningful changed contracts; they do not claim exhaustive graph mastery. Close hints and reconstruct the reasoning later with a new input.',
};
