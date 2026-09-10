export default {
  summary: 'Predict resource growth and design terminating recursive computations by connecting exact execution events, bounds, call frames and recurrence trees.',
  outcomes: ['State input size, counted operations, relevant case and auxiliary/output space', 'Derive tight loop bounds and distinguish O, Omega, Theta, expected and amortized claims', 'Trace pending recursive calls, justify a base case and a decreasing progress measure', 'Solve common recurrences by level work and distinguish total calls from live stack depth', 'Diagnose copying, repeated subproblems, integer size and measurement limitations'],
  prerequisites: ['Python Basics: Types, Control Flow, Functions & Modules', 'Algebra, Functions, Exponentials & Logarithms'],
  sequence: ['Count a concrete operation', 'Compare growth and prove bounds', 'Separate input cases and amortized work', 'Follow recursive calls and returned values', 'Account for recurrence levels', 'Include copying, output and arithmetic costs', 'Practise analysis and a changed recursive task'],
  visual: {
    type: 'Iteration lattice, suspended call frames and recurrence level accounting',
    question: 'Which execution events occur, which work is still waiting, and how do they grow?',
    interaction: 'Change loop size/shape, step a bounded suffix-sum call trace, and compare shrinking recurrences with exact work and depth.'
  },
  practice: {
    task: 'Derive a shrinking-loop sum, repair a termination argument, analyze retained slices and implement a balanced count-under-threshold with stated costs.',
    success: 'Explain exact small cases and independent oracle outputs, preserve empty/odd/negative-value cases, separate total work from peak storage and justify tight bounds.'
  },
  misconceptions: ['Big O means worst case or equality', 'Two nested loops must be quadratic', 'Halving guarantees logarithmic total work', 'A base case alone ensures termination', 'Two recursive children mean both stacks are live together', 'One Python operation always has constant cost', 'Amortized means average over random inputs'],
  sources: ['https://algs4.cs.princeton.edu/14analysis/', 'https://introcs.cs.princeton.edu/python/23recursion/', 'https://docs.python.org/3/library/sys.html#sys.getrecursionlimit', 'https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/1869dbf640ded6b31f1bd369d2001ef5_MIT6_006S20_r03.pdf'],
  depth: 'core',
  reviewFocus: 'Stated operation and case, tight versus upper bounds, real progress, native/model frame correspondence, exact recurrence leaf accounting, Python copy/bit/stack limits, finite timing evidence and independent transfer.'
};
