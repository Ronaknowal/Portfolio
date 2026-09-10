export default {
  summary: 'Design local choices under explicit objectives, disprove tempting rules, and establish exact greedy guarantees through exchange, stays-ahead, structural lower bounds and contraction.',
  outcomes: ['Separate feasibility, optimality and a candidate local rule', 'Prove earliest-finish interval selection and distinguish weighted selection and room assignment', 'Use adjacent exchanges to justify deadline order under its actual scheduling assumptions', 'Derive fractional density allocation and explain why indivisible items break its exchange', 'Build and justify a Huffman merge tree with exact weighted code cost', 'Maintain a reachable-prefix invariant without committing to one path', 'Transfer proofs to point coverage and threshold matching, and recognize when DP or another method is needed'],
  prerequisites: ['Complexity Analysis & Recursion', 'Binary Search, Sorting & Two-Pointer Patterns', 'Backtracking & Divide-and-Conquer', 'Heaps, Priority Queues & Tries'],
  sequence: ['Test a candidate rule against its objective', 'Select intervals and exchange an optimal prefix', 'Match resource usage to an overlap lower bound', 'Swap adjacent deadline inversions', 'Exchange equal weight under divisibility', 'Contract two minimum-frequency leaves', 'Certify a reachable prefix', 'Solve changed-contract proofs and guided practice'],
  visual: {
    type: 'Appointment timelines, aligned schedule exchanges, capacity fractions, merge trees and reachable-prefix figures',
    question: 'What can be exchanged without losing feasibility or worsening the objective, and which changed assumption makes that exchange invalid?',
    interaction: 'Compare interval rules against a small exact oracle, swap adjacent processing jobs and observe signed lateness, then change capacity and divisibility while retaining the same items.',
  },
  practice: {
    task: 'Construct a weighted counterexample, cover closed intervals, assign minimum adequate supplies, analyze a deadline exchange and verify a different Huffman cost.',
    success: 'Provide a complete choice/proof/termination/cost argument, match independent small-instance oracles and adapt endpoint, identity, objective and divisibility contracts.',
  },
  misconceptions: ['Greedy means the chosen rule is automatically optimal', 'A maximal feasible solution must be maximum', 'The same interval ordering solves merging, selection and room allocation', 'A safe exchange must improve every individual job', 'Fractional and whole-item density have the same guarantee', 'The deepest leaves should contain the most frequent symbols', 'Furthest reachable means always jump as far as possible', 'Passing small tests proves a greedy rule for every input'],
  sources: ['https://www.cs.princeton.edu/~wayne/kleinberg-tardos/pdf/04GreedyAlgorithmsI-2x2.pdf', 'https://ocw.mit.edu/courses/6-046j-design-and-analysis-of-algorithms-spring-2015/resources/lecture-1-course-overview-interval-scheduling/', 'https://algs4.cs.princeton.edu/55compression/'],
  depth: 'core',
  reviewFocus: 'Exchange quantifiers, residual problem, half-open versus closed endpoints, max lateness versus tardiness, all-release-zero assumptions, fractional linear benefit, exact rational arithmetic, Huffman scope/tie/empty-symbol conventions and bounded visualization overhead.',
};
