export default {
  summary: 'Model a capacity-constrained allocation, revise earlier decisions through residual cancellation, and return exact flow/cut or matching/cover certificates; extend the same reasoning to mandatory circulation and binary labeling.',
  outcomes: [
    'Audit edge capacities, internal conservation and net source/sink value on original edge identities',
    'Trace positive residual paths, bottlenecks and signed updates, including original antiparallel edges',
    'Derive max-flow/min-cut through residual reachability and independently verify equal lower/upper witnesses',
    'Implement exact integer Edmonds–Karp, explain its capacity-independent graph-operation bound and optional Dinic blocking phases',
    'Recover maximum bipartite matching, minimum vertex cover and a Hall shortage through alternating reachability',
    'Model vertex throughput, supply ceilings, required demands and feasible lower-bound circulation with recovered original flows',
    'Derive a nonnegative binary labeling energy as a minimum cut and account for every unary and pairwise term',
    'Distinguish terminal cuts from global undirected cuts and optionally implement/analyze random contraction with preserved edge multiplicity',
  ],
  prerequisites: ['Graphs: Representations, BFS & DFS', 'Algorithm Correctness, Loop Invariants & Termination'],
  sequence: ['Audit capacity and conservation', 'Make prior choices reversible', 'Follow residual augmentations', 'Prove equal flow/cut certificates', 'Implement and analyze exact solvers', 'Assign pairs and recover cover/shortage witnesses', 'Transform additional capacity constraints', 'Optimize binary labels', 'Distinguish global cuts and randomized guarantees', 'Practise new models and verify original witnesses'],
  visual: {
    type: 'Original flow graph with incidence ledger, separately owned residual pairs, stepped cancellation route and cut inspector, compatibility matrix with alternating arrows/cover rings, vertex gate, and binary-label energy grid',
    question: 'Which feasible change improves this allocation, and which independently checkable obstruction proves that no better answer remains?',
    interaction: 'Edit proposed flow and bounded capacities, preview/apply/back through augmentations, inspect alternate cuts, toggle compatibility and recover shortage certificates, and change binary labels or disagreement penalties.',
  },
  practice: {
    task: 'Repair invalid flows, preserve residual identity, prove directed cut bounds, reconstruct matching/cover/Hall witnesses and derive/implement transformed allocation or labeling models.',
    success: 'Original-domain feasibility, witness recovery and matched optimality bounds pass independent exhaustive tiny oracles; numeric/graph/prerequisite limits and changed objectives are explained.',
  },
  misconceptions: ['A forward-only dead end proves maximum flow', 'A residual reverse edge is extra original capacity', 'A parent vertex identifies a parallel residual edge', 'A cut adds capacities in both directions', 'Every saturated edge belongs to a minimum cut', 'An integer optimum means every feasible flow is integral', 'Maximal matching is maximum matching', 'A bipartite matching-cover equality holds on a triangle', 'Supply ceilings automatically satisfy all demand', 'Subtracting lower bounds preserves conservation without repair', 'Two opposite cut edges double every disagreement cost', 'A global random cut preserves prescribed terminals', 'A sampled valid cut is automatically optimal'],
  sources: ['https://theory.stanford.edu/~tim/w16/l/l2.pdf', 'https://theory.stanford.edu/~tim/w16/l/l4.pdf', 'https://www.cs.cmu.edu/~ckingsf/bioinfo-lectures/flowext.pdf', 'https://web.stanford.edu/class/archive/cs/cs265/cs265.1254/Lectures/Lecture2/l2.pdf', 'https://ocw.mit.edu/courses/6-046j-design-and-analysis-of-algorithms-spring-2015/resources/lecture-13-incremental-improvement-max-flow-min-cut/'],
  depth: 'core',
  reviewFocus: 'Capacity/conservation and signed cancellation invariants; original-edge occurrence identity; net directed-cut inequality and equality proof; exact integer arithmetic versus bit cost and optional recursion depth; cover proof requiring maximum bipartite matching; Hall deficiency and empty partitions; sign conventions and demand saturation; objective-preserving binary cuts; nonempty global cuts, uniform edge multiplicity, conditional survival and independent full trials. Browser illustrations are bounded computed states, not measured benchmarks or external solver guarantees.',
};
