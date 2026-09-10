export default {
  summary: 'Derive reliable planar decisions from signed orientation, handle boundaries explicitly, and build convex hull and polygon algorithms whose exactness and output contracts survive degeneracies.',
  outcomes: ['Derive orientation and signed area from coordinate differences', 'Classify closed segment crossings, contacts, overlaps and point segments', 'Distinguish intended data, represented numbers, robust predicates and exact constructions', 'Implement and justify monotone-chain hulls with corner-only or all-boundary output', 'Locate a point in a simple polygon with boundary-first half-open crossing parity', 'Apply conservative bounds, directional hull support and canonical integer line keys', 'Verify geometric invariants against independent exact small-case oracles'],
  prerequisites: ['Geometry, Trigonometry & Coordinate Reasoning', 'Binary Search, Sorting & Two-Pointer Patterns', 'Floating-Point Representation & Numerical Error'],
  sequence: ['Introduce points, displacements and a directed-side question', 'Derive orientation and doubled area with coordinate conventions', 'Combine side predicates and interval bounds for closed segment classification', 'Expose input/product rounding and distinguish exact constructions', 'Specify convexity and two boundary-output contracts', 'Construct and prove lower/upper monotone chains including all degeneracies', 'Derive signed polygon area and boundary-first crossing parity', 'Transfer to bounds, linear support and rational direction keys', 'Practise changed boundaries, arithmetic failures and integrated geometric reports'],
  visual: {
    type: 'Equal-scale oriented triangles, segment-contact geometry, exact/Number arithmetic paths, hull candidate/pop/push trace, concave-envelope contrast and polygon-ray parity',
    question: 'Which geometric relationship determines the branch, and does the arithmetic preserve that relationship?',
    interaction: 'Move integer points, reverse a baseline, compare exact and rounded products, edit bounded hull inputs and boundary policy, step actual chain removals, and inspect ray crossings.'
  },
  practice: {
    task: 'Implement exact integer geometric queries with declared boundary policies; reconstruct proofs, generate numerical counterexamples and adapt to changed input/output contracts.',
    success: 'Matches independent Fraction, supporting-line and winding-number oracles; handles empty, repeated, collinear, touching and nearly degenerate cases; explains costs and precision limits.'
  },
  misconceptions: ['Equal slopes are required to test collinearity', 'Zero orientation means between the endpoints', 'Bounding-box overlap proves geometric intersection', 'Exact stored inputs guarantee exact floating products', 'An arbitrary epsilon universally repairs topology', 'Exact predicates also guarantee exact constructed coordinates', 'A hull preserves a concave shape or its data density', 'Every point ever pushed belongs to the final hull', 'All hull APIs return the same boundary records', 'Every ray/vertex hit should count once'],
  sources: ['https://www.cs.cmu.edu/~quake/robust.html', 'https://doc.cgal.org/latest/Kernel_23/index.html', 'https://doc.cgal.org/latest/Convex_hull_2/index.html', 'https://doc.cgal.org/latest/Polygon/index.html', 'https://docs.python.org/3/library/fractions.html', 'https://ocw.mit.edu/courses/6-046j-design-and-analysis-of-algorithms-spring-2015/resources/lecture-2-divide-conquer-convex-hull-median-finding/'],
  depth: 'specialist',
  reviewFocus: 'Orientation convention and exact intermediate bounds, boundary/degenerate contracts, hull identity and order, simple-polygon and half-open crossing hypotheses, arithmetic versus measurement uncertainty, and independent model/native/browser verification.',
  designRecord: 'docs/teaching/COMPUTATIONAL-GEOMETRY-LESSON-DESIGN.md'
};
