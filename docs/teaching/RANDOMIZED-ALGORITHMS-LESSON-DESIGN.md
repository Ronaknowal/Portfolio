# Randomized Algorithms, Sampling & Error Guarantees — design

Stable ID `randomized-algorithms-sampling-error-guarantees`; DSA position 18. Designed 10 September 2026. Implementation and verification are separate from user acceptance.

## Scope decision and prior coverage

The exact topic-plan command reports a planned specialist entry; the body does not exist. Retain the title: it already describes the intended finish line. The probability prerequisite `random-variables-expectation-covariance` has no body, so finite events, conditional probability, independence, expectation and indicators are introduced locally. The actual Complexity, Ordered Patterns, Hashing and String Matching bodies were inspected. They teach cost models, sorting/partition ideas, universal chaining and a fixed-input random-base collision argument. This lesson supplies the missing general random experiment and sampling proofs rather than repeating those chapters. Randomized Linear Algebra already owns Gaussian range finding, sketches and trace estimation.

| Hurdle / outcome | Mechanism and representation | Practice / verification |
| --- | --- | --- |
| A random output need not be an error | Three contracts: valid sample, exact selection with random work, error-permitting verification; finite outcome tree | Classify changed contracts; enumerate probabilities independently |
| A random integer transformed carelessly becomes biased | Eight raw cells mapped to three bins, then rejection removes unequal leftovers | Exhaustive raw-state map; geometric tail and expected attempts |
| Uniform marginals do not imply a uniform permutation or subset | Fisher–Yates shrinking active prefix and six terminal permutations; reservoir slots plus exact subset distribution | Enumerate full choice paths, not only frequency histograms |
| A bounded sample survives an unknown-length stream | Incoming record chooses one of all positions seen; a hit on a reservoir slot replaces it | Joint k-subset induction, duplicates as occurrence identities, empty/k>N/filtered inputs |
| Expected work is not a deadline | Quickselect rank interval shrinks under a learner-selected pivot; actual scanned sizes and conditional expected work | All ranks and duplicate inputs checked against sorting; exact rational recurrence oracle |
| A compressed test can miss an error | Integer Freivalds column probes: A(B r) and C r; cancellation and four possible bit vectors | Exact product oracle, exhaustive bit witnesses, independent vs repeated-same amplification |
| A probability target must determine work | Finite failure-pattern columns for one-sided tests and majority; sample-budget curve from stated Hoeffding theorem | Exact binomial enumeration, hand budgets, correlated counterexample |
| Replay does not prove or secure randomness | Input/state/draw-order/output flow; seed and generator-state program | Actual Python replay, adversary fixes input before probes; no security guarantee |

## Core learning flow

1. Define the random experiment and outcome contract, local probability bridge.
2. Derive unbiased integer draws, rejection tails, weighted integer tickets.
3. Prove Fisher–Yates with labeled occurrences and immutable reset snapshots.
4. Build reservoir1, then Algorithm R for k; prove entire subset distribution, not just inclusion.
5. Randomized three-way quickselect, correctness independent of pivot, expected linear work with good-pivot phases and worst-case quadratic behavior; Markov and explicit restart limits.
6. Exact-integer matrix-product verification with a local row/column bridge; one-sided error proof.
7. Independent amplification, correlated repetitions, union budgets, two-sided majority and sampling accuracy.
8. Practical reproducibility, random-source limits and fixed/adaptive input distinction.
9. Independent exercises with explained solutions, guided official practice, written/video alternatives and actual next topic.

Cost contracts distinguish unit-cost draws/arithmetic from random-bit cost, growing integers, Python list allocation and network/I/O. No empirical timing chart. Browser distributions are exhaustive finite calculations; manual choices illustrate one execution, not uniform random generation. The Freivalds UI uses tiny exact integers and cannot certify floating-point products by adding a tolerance to the theorem.

## Sources actually inspected

- Python current `random` documentation: range selection, state restoration, shuffle, sample and reproducibility/security clauses. Programs target the available Python 3.12 runtime; exact seeded output is scoped to that runtime and call sequence, not all versions.
- Vitter 1985, *Random Sampling with a Reservoir*, PDF pages 1–4, especially section 2 Algorithm R and its sample invariant. Paper attributes Algorithm R to Alan Waterman; do not attribute its invention to Vitter. The paper's later skipping algorithms and historical speed results are outside our implementation.
- Cornell ORIE5270 Week4 written selection/streaming exposition inspected. It has a zero-based stream-index versus record-count slip in the informal reservoir section; our one-based `seen` derivation/code uses replacement k/seen. We independently justify the array-allocation and stack costs rather than copying its informal memory wording.
- MIT 6.046 Lecture 6 official video page and opening/algorithm transcript sections: randomization, matrix verification and quicksort. Useful board-based alternative; no full-video viewing claimed.
- UT Austin CS388R 2021 lecture11 PDF matrix verification and polynomial sections inspected. Its first matrix setup uses modulo2; our lesson explicitly verifies integer equality with exact integer arithmetic, using the conditioned nonzero-coefficient proof. It does not reduce arbitrary integer errors modulo2.
- Waterloo CS466 2025 concentration lecture: independent bounded-variable Hoeffding theorem and exponential-moment proof. We introduce the theorem and derive a concrete sample budget; full concentration theory remains its existing maths owner.
- Official LeetCode statements inspected10 September 2026: 384 Shuffle an Array, 470 Implement Rand10() Using Rand7(), 382 Linked List Random Node, 398 Random Pick Index, 528 Random Pick with Weight, 215 Kth Largest Element in an Array. All displayed Medium and publicly readable; no submission/editorial access or solution execution claim.

## Scope and destination decisions

Incoming String Matching note: include fixed-input exact-versus-error bridge, independence/union difference, reproducibility/adaptive-input warning, and true-hit verification accounting. Do not duplicate KMP or the full polynomial derivation.

Detailed concentration proofs belong to `concentration-inequalities-hoeffding-bernstein-chernoff`; local Bernoulli theorem and original finite enumeration suffice for this algorithmic outcome. Correlated MCMC estimates belong to `monte-carlo-methods-mcmc-metropolis-hastings-hmc-nuts`. Graph contraction belongs alongside `network-flow-minimum-cuts-bipartite-matching`, where global versus s–t cut and parallel edges can be taught accurately. Detailed primality, skip lists and streaming sketches require their own coherent prerequisites; this scoped review is not a claim those specialties are fully taught. Relevant concrete destination notes will be saved, not inserted as unexplained optional problem links. The unresolved bit-manipulation inbox does not make this random-algorithm lesson its general owner.

## Evidence plan

Full native programs with exact stdout, exhaustive finite samplers and independent rational/combinatorial/product oracles; model and UI state values independently checked. Actual1440/390 keyboard, reset, edited parameters, end/empty boundaries, ordinary reading and opened screenshots. Verify all route anchor arrivals and rendered code/output, all external problem metadata, import ownership and a frozen-source evidence record. Root performs global registration/build/integration.
