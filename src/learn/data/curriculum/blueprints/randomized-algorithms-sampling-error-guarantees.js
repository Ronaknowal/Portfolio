export default {
  summary: 'Design the random experiment before choosing an algorithm: sample without bias, preserve exact answers while randomizing work, and budget explicitly bounded verification or estimation errors.',
  outcomes: ['Specify fixed inputs, eligible outcomes, independence and failure events', 'Construct unbiased integer draws, weighted tickets and uniform permutations', 'Implement and prove uniform k-subset reservoir sampling with occurrence identity', 'Explain randomized three-way selection correctness, expected work and capped tail contracts', 'Derive the exact-integer Freivalds one-sided error bound', 'Distinguish independent amplification, majority error, union budgets and sampling accuracy', 'Replay generator state while recognizing version, call-order, security and adaptive-input limits'],
  prerequisites: ['Complexity Analysis & Recursion', 'Random Variables, Expectation & Covariance'],
  sequence: ['Separate sampling, exact-answer and error-permitting contracts', 'Introduce finite events, conditional probability and expectation locally', 'Repair modulo bias with rejection and derive weighted ticket intervals', 'Fix a suffix to obtain Fisher–Yates uniformity', 'Preserve a whole subset distribution in a bounded stream reservoir', 'Select a rank exactly while analyzing random pivot work', 'Verify an integer product with independent bit-vector probes', 'Set repetition and sampling counts from explicit error budgets', 'Document replay and adversary assumptions; practise changed contracts'],
  visual: {
    type: 'Outcome tree, raw-ticket mapping, shrinking shuffle prefix, stream slots and full subset distribution, partition-survival trace, paired matrix-probe paths, finite error bars and accuracy budget comparison',
    question: 'Which outcomes receive probability mass, and what exactly can vary or fail under the declared experiment?',
    interaction: 'Map or reject raw outcomes; choose individual shuffle/reservoir/pivot branches; expose cancellation with binary probes; compare independent versus reused evidence and change precision budgets.'
  },
  practice: {
    task: 'Construct and test unbiased samplers, prove joint subset/selection invariants, derive error and work budgets, and break biased, correlated or adaptive-input shortcuts.',
    success: 'Agrees with independent combinatorial and exact arithmetic oracles; declares the probability space, failure event, cost model and replay limits; transfers to changed constraints.'
  },
  misconceptions: ['Random output necessarily means an approximate or incorrect answer', 'Modulo always preserves uniformity', 'Uniform marginals establish uniform permutations or subsets', 'A reservoir samples distinct labels uniformly', 'Expected linear work is a fixed deadline guarantee', 'A passing probe certifies exact equality', 'Repeating the same random evidence amplifies confidence', 'A union bound requires independence', 'A false-pass bound is the posterior probability a passing claim is wrong', 'A seed or a frequency histogram proves a theorem'],
  sources: ['https://docs.python.org/3/library/random.html', 'https://www.cs.umd.edu/~samir/498/vitter.pdf', 'https://ocw.mit.edu/courses/6-046j-design-and-analysis-of-algorithms-spring-2015/resources/lecture-6-randomization-matrix-multiply-quicksort/', 'https://www.cs.utexas.edu/~ecprice/courses/randomized/fa21/scribe/lec11.pdf', 'https://cs.uwaterloo.ca/~r5olivei/courses/2025-spring-cs466/lecture-notes/lecture3/'],
  depth: 'specialist',
  reviewFocus: 'Actual joint distributions, exact versus approximate contracts, conditional independence, growing counters and integer costs, duplicate identities, sample order, integer versus modular or numerical verification, and independent native/model/browser evidence.',
  designRecord: 'docs/teaching/RANDOMIZED-ALGORITHMS-LESSON-DESIGN.md'
};
