export default {
  summary: 'Develop complete sequential algorithm contracts, expose preserved state relationships, justify progress and diagnose the precise proof obligation behind a bug.',
  outcomes: ['Specify first/any/absent outputs, mutation and permitted input precisely', 'Derive and prove initialization, preservation and every loop exit', 'Diagnose weak or uninitializable invariants with concrete counterexamples', 'Protect original occurrences and unread information during in-place updates', 'Prove termination with natural-number or lexicographic measures', 'Connect local, recursive, algebraic and optimization proof techniques', 'Distinguish bounded tests, output certificates and universal correctness claims'],
  prerequisites: ['Python Basics: Types, Control Flow, Functions & Modules', 'Sets, Logic, Relations & Proof Techniques'],
  sequence: ['Specify the first-occurrence contract', 'Trace a preserved prefix and prove every edge', 'Develop invariants and work backward through assignments', 'Preserve original information during stable compaction', 'Classify a four-region partition without skipping swapped values', 'Establish well-founded progress with Euclid and lexicographic measures', 'Prove recursive powers and conserved iterative work', 'Compose proofs across helpers and algorithm families', 'Test contracts and diagnose their modeling assumptions', 'Derive an independent integer-square-root algorithm and proof'],
  visual: {
    type: 'Proof-boundary prefix inspector, original/working occurrence rows, four-region partition and divisor-preserving Euclid trace',
    question: 'What does this boundary certify, which step could destroy it, and what strictly progresses?',
    interaction: 'Apply tiny inputs, compare candidate assertions, inject a skipped-position or skipped-classification fault, step mutation boundaries and inspect decreasing remainders.'
  },
  practice: {
    task: 'Derive a new invariant, repair a weak maximum contract, distinguish failed exit/progress arguments and complete proof-oriented official problem variants.',
    success: 'Explains all branches and exits, uses an explicit well-founded measure, conserves identities where required, matches independent oracles and states the scope of test evidence.'
  },
  misconceptions: ['An invariant is a variable that never changes', 'The loop invariant must hold after every individual instruction', 'Any true invariant is strong enough to prove the result', 'Stopping implies the desired result is correct', 'Every strictly decreasing bounded quantity proves termination', 'A break makes the loop guard false', 'A sorted-looking output preserves all original occurrences', 'Passing a finite set of tests proves every unbounded input case'],
  sources: ['https://www.cs.cornell.edu/courses/cs2112/2019fa/lectures/loopinv/', 'https://softwarefoundations.cis.upenn.edu/plf-current/Hoare2.html', 'https://dafny.org/latest/OnlineTutorial/Termination', 'https://www.cs.cornell.edu/courses/JavaAndDS/loops/30aloop.html', 'https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement'],
  depth: 'core',
  reviewFocus: 'Defined program points, early exits, weak/strong assumptions, frame/conservation clauses, well-founded measures, exact versus runtime arithmetic, finite-check scope and independent native/browser evidence.',
  designRecord: 'docs/teaching/ALGORITHM-CORRECTNESS-LESSON-DESIGN.md'
};
