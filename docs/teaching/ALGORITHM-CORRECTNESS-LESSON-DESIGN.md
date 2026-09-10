# Algorithm Correctness, Loop Invariants & Termination — lesson design

10 September 2026; stable ID `algorithm-correctness-loop-invariants-termination`, DSA position 13. New full lesson, replacing no authored body. The existing brief's search/invariant/termination intent is retained and expanded. Root owns registration, shared curriculum and integrated build.

Implemented and locally verified. See [the final verification record](ALGORITHM-CORRECTNESS-VERIFICATION.md) for actual native/model checks, ordinary desktop/mobile reading and interaction evidence, source review, corrected issues, resource-access scope and remaining integration/acceptance boundaries. Owned source frozen at 09:37:14 UTC; final browser run passed at 09:38:51 UTC.

## Scope and continuity

Observable finish: specify permitted input and exact output/mutation; derive a useful invariant from the desired result; prove initialization, preservation, all exits and a well-founded decreasing measure; relate iterative/recursive proofs; find the first failed obligation in a proposed repair; distinguish bounded evidence from a universal proof and a verified algorithm from an incorrect specification.

The named Sets/Logic prerequisite is not presumed completed in this reading order. Introduce conjunction, implication, universal statements, empty ranges and induction in place. Earlier Complexity supplies size/cost and small induction, Ordered Patterns supplies boundary elimination and logical-prefix compaction, Backtracking supplies restoration and exhaustive branches, Greedy supplies exchange/optimality, and DP supplies state sufficiency/dependency order. This lesson makes their proof architecture explicit without rewriting those complete chapters. Next remains Hashing, Collision Resolution & Amortized Analysis.

| Hurdle | Teaching and placement | Representation/action | Independent transfer |
| --- | --- | --- | --- |
| A plausible result has an ambiguous contract | First occurrence versus any occurrence; absent sentinel; original input and mutation | Entry/result contract and exact early-return example | Duplicate targets, empty and absent input; strengthen a specification |
| An invariant is treated as an unchanged variable | Prefix known to contain no match grows as index changes | Editable scan with proved/unknown regions and selected weak/strong claims; exact finite obligation witnesses | Explain why bounds alone cannot justify failure return |
| Assignment order or program point is hidden | Backward substitution, sequence/branch rules, simultaneous swap | Inline assignment timeline and loop proof-flow diagram | Repair a sequential overwrite and a too-strong assertion |
| In-place output overwrites needed information | Stable filtering, original prefix meaning, write≤read and untouched suffix | Original occurrence row linked to actual working array, read/write boundaries, retained source IDs | Change output to include trailing zero fill; reverse the safe write direction for merging |
| A swapped element is skipped before it is classified | 0/1/2 partition, four regions, conservation and exclusive high | Partition investigation with exact regions, boundary labels and optional faulty increment | Small [1,2,0] counterexample; equal/empty inputs; reject invalid categories |
| A decreasing quantity is assumed sufficient | Natural-number measure, no infinite descent; rational counterexample; lexicographic reset | Exact halving and lexicographic inline contrasts; Euclid division/remainder investigation | Repair a loop that stalls, choose a measure when one coordinate resets |
| A recursive call is treated as magic | Strong induction on exponent and complete return contract; iterative conserved product | Worked even/odd decomposition and full recursive/iterative programs | Prove structure as well as values for a tree comparison |
| Testing is mistaken for proof or a postcondition is too weak | Independent oracle, assertions, mutation testing, multiset certificate, environment assumptions | Same sorted-looking outputs with preserved/lost multiplicities | Independent integer-square-root contract, proof and boundary tests |

Four investigations support distinct mechanisms; this is a design choice, not a quota. Static figures introduce proof-flow, assignment dependencies, well-foundedness and specification strength without redundant controls. All results are exact synthetic integer fixtures, not performance measurements. Browser models are bounded teaching interpreters, not formal proof checkers. Failed finite obligations are concrete counterexamples; absence of a counterexample only describes the declared finite domain. General arguments remain in the narrative.

## Implementation contracts

- Scan: up to eight integers, target, correct/skip-a-position transition, selected invariant; fixed-array loop-boundary states examined explicitly. Return exits and normal guard exit are distinct. Inputs apply on submit; previous/next/finish and reset are deterministic.
- Compaction: up to eight integers; remove a selected integer while preserving retained occurrence order. The visible original copy and saved snapshots belong to the explanation, not the algorithm's O(1) auxiliary-state bound. The logical prefix is the result; unused tail is unspecified until an explicit second phase fills it.
- Partition: up to eight 0/1/2 values with original occurrence identity. Four disjoint ranges are [0,low), [low,middle), [middle,high), [high,n). Swap preserves the original multiset; order within a category is not promised. Fault mode intentionally increments after swapping from high, with the violated region displayed.
- Euclid: two nonnegative integers at most 96, not both zero in the drawable divisor-set investigation. Show common divisors, a=q·b+r, strict 0≤r<b and next pair. Native function also handles (0,0) as the stated conventional gcd0. No float arithmetic is used for remainders.

All forms have named controls, visible text alongside color, local scrolling only where needed, invalid-input feedback and exact state/result correspondence. Inspect initial ordinary reading flow, operated states and static figures at 1440/390, then open screenshots. Small diagrams must remain readable rather than merely bounded by the viewport.

## Primary research ledger

Inspected 10 September 2026:

- Cornell CS2112, `https://www.cs.cornell.edu/courses/cs2112/2019fa/lectures/loopinv/`: four obligations, invariant program point, weak/strong invariants and exponentiation. Their binary-search precondition guarantees presence; our scan and square-root contracts differ and are proved independently. Java fixed-width arithmetic is not treated as exact Python integer arithmetic.
- Cornell CS2110 current loop lesson, `https://courses.cis.cornell.edu/courses/cs2110/2026fa/lectures/lec04/`: array regions and deriving a partial-progress description from pre/post diagrams. Original site diagrams/layout not copied.
- Software Foundations, `https://softwarefoundations.cis.upenn.edu/plf-current/Hoare2.html`: decorated programs, backward assignment substitution and strengthening an invariant. Its language uses mathematical natural-number conventions; examples here use explicit Python integer contracts. No Rocq proof checking is claimed.
- Dafny, `https://dafny.org/latest/OnlineTutorial/Termination`: bounded measures, lexicographic tuples and the need for well-founded descent. This lesson uses a sufficient nonnegative-at-boundary convention, not a claim that every tool requires precisely this form. No Dafny execution is claimed.
- Python 3.14.7 reference `https://docs.python.org/3/reference/simple_stmts.html#the-assert-statement`: assertions/optimization and assignment evaluation; `https://docs.python.org/3/library/math.html#math.gcd`: nonnegative gcd and all-zero convention. Native checks use installed Python3.12.14.
- Cornell David Gries JavaHyperText `https://www.cs.cornell.edu/courses/JavaAndDS/loops/30aloop.html`: short video page and written companions about generalizing a postcondition into an invariant. Companion content is reviewed before recommendation; full playback is not claimed.

Verified official public LeetCode statements: 27 Remove Element Easy; 283 Move Zeroes Easy; 75 Sort Colors Medium; 88 Merge Sorted Array Easy; 69 Sqrt(x) Easy; 206 Reverse Linked List Easy; 100 Same Tree Easy; optional50 Pow(x,n) Medium. These are proof-oriented revisits/variants, not eight newly invented patterns. State platform contract differences and deeper prerequisites beside them. Public statement access does not guarantee free editorials or judge submission. No universal interview guarantee.

## Checks and scope limits

Execute complete programs and exact stdout, including meaningful invalid-input behavior. Independent references: list.index/direct filtering, sorted+occurrence identities, math.gcd and divisor sets, built-in integer pow, math.isqrt, complete finite counterexample enumeration independent of the model's claimed proof gates. Validate every intermediate state, not merely the final result. Bounded checks supplement derivations; do not label them machine-checked universal proofs.

This is sequential deterministic algorithm proof development. Concurrent interference, probabilistic termination and full formal verification tooling need separate assumptions and deeper routes. Mention the distinction where it prevents a false conclusion, route useful concrete discoveries to an existing owner, and do not audit the whole catalogue or silently create a new topic. Keep implementation/native/browser/user acceptance separate in the final verification record.
