export default {
  summary: 'Translate a network into adjacency, incidence and Laplacian operators; use them to reason about routes, disagreement, connectivity, averaging and anchored interpolation with explicit conventions.',
  outcomes: ['Define vertices, edge direction, weights and label order before constructing a graph matrix', 'Interpret matrix products as neighbor aggregation and weighted walks while distinguishing reachability from walk counts', 'Distinguish undirected, weak and strong connectivity and explain the role of isolates', 'Derive the Laplacian row, edge energy and incidence factorization rather than treating D minus A as a recipe', 'Prove that the nullspace consists of componentwise constants under nonnegative undirected weights', 'Choose degree normalization and isolate/self-loop conventions deliberately', 'Compare conservative exchange, random-walk averaging and lazy averaging with the correct conserved quantity', 'Solve a small anchored harmonic interpolation and diagnose an unanchored component', 'Relate these operators to graph features and GCN aggregation while auditing direction, units and data availability'],
  prerequisites: ['Sets, Logic, Relations & Proof Techniques', 'Vectors, Matrices & Tensor Operations'],
  sequence: ['Model one relationship and synchronize graph/matrix views', 'Read walks, reachability and directed components', 'Derive local disagreement and edge energy', 'Factor through incidence and prove the component nullspace', 'Normalize with isolates and self-loops kept explicit', 'Compare averaging dynamics and interpolate from anchors', 'Connect to learned graph representations and solve changed contracts'],
  visual: {
    type: 'Linked network/matrix cells, edge-current accounting, normalization comparison and node trajectories',
    question: 'Which relationship does each matrix entry or operation represent, and which graph assumptions make its conclusion valid?',
    interaction: 'Change an edge or its direction, inspect a walk contribution, alter node values, compare normalized operators, advance an averaging step and remove an interpolation anchor.'
  },
  practice: {
    task: 'Construct a changed graph matrix, count weighted walks, repair directed/isolate mistakes, derive a kernel and solve an anchored network calculation.',
    success: 'States the graph and row/column conventions, exposes intermediate values, checks assumptions and edge cases, and explains why a numerical result follows from the graph.'
  },
  misconceptions: ['A drawing crossing creates a vertex', 'A matrix power counts simple paths or computes shortest distance', 'A directed graph can be silently symmetrized', 'A Laplacian coordinate is the total disagreement energy', 'Every normalized Laplacian has a one on an isolated diagonal', 'Every averaging operator preserves the ordinary mean', 'A self-loop cancels from every normalized operator', 'A missing value is uniquely determined without an anchor in its component', 'A graph neural network or random edge split is automatically appropriate'],
  sources: ['https://www.cs.yale.edu/homes/spielman/561/lect02-15.pdf', 'https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/resources/lecture-12-graphs-networks-incidence-matrices/', 'https://networkx.org/documentation/stable/reference/generated/networkx.linalg.laplacianmatrix.normalized_laplacian_matrix.html', 'https://arxiv.org/abs/1609.02907'],
  depth: 'core',
  reviewFocus: 'Undirected/nonnegative hypotheses, degree and loop conventions, zero-weight support, adjacency orientation, incidence sign, exact component kernel proof, isolated normalized diagonal, conserved averaging quantities and anchor-based uniqueness.',
  designRecord: 'docs/teaching/GRAPH-FUNDAMENTALS-LESSON-DESIGN.md'
};
