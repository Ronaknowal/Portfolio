# Backtracking & Divide-and-Conquer — individual design

Stable ID `backtracking-divide-and-conquer`; preserve module position 9, after Complexity Analysis & Recursion and Binary Search, Sorting & Two-Pointer Patterns. Topic-plan CLI reviewed on 10 September 2026: no existing destination note, planned topic without an individual blueprint. Current standards and recent author-verified DSA lessons apply; Linux remains the approved teaching reference.

The learner already understands call frames, base cases, progress measures and merge-sort mechanics. This lesson distinguishes a tree of candidate decisions from a tree of smaller input regions and teaches how to design the state/return contract, prove completeness, prune safely, restore mutation and account for output. It does not repeat generic recursion or basic sorting as new material.

## Hurdles and teaching choices

- Begin with a configuration search: choose items, explore consequences, restore the exact prior state. Show the include/exclude decision tree and a live path/output distinction; an answer must be copied rather than aliased.
- Safe pruning is a proof about *all completions*. Positive subset sums support overshoot and insufficient-remaining bounds; signed values invalidate those particular bounds. Compare pruned/unpruned exploration on the same input and same answers.
- Distinct permutations need positions/used-state; duplicate-valued permutations need a canonical choice or remaining-frequency policy. Positive unbounded combination search keeps a nondecreasing candidate index and proves termination through decreasing remainder.
- N-Queens uses native board geometry, column and diagonal constraints, rejection and restoration; word search uses path-local visited state, not the previous graph lesson's reachability-wide visited policy.
- Divide-and-conquer needs enough returned information to combine. Inversion counting extends merge sort with strict cross-pair counting. A maximum-subarray summary returns total/prefix/suffix/best so each combine is constant arithmetic work, rather than rescanning the boundary.
- Separate decision, counting, enumeration and optimization contracts. Discuss heuristic ordering, feasibility pruning, branch-and-bound and memoization without pretending they are interchangeable. Integer cost model and Python recursion limits remain explicit.

## Visual contracts

1. `SubsetChoicesLab`: at most three editable positive integer occurrences, target 0–30 and optional proven pruning; actual depth-first event frames, current path tokens versus copied answers, binary branch diagram and frame state. Values may repeat but position identities are distinct. Unchosen/restored state is observable. Diagram is finite, geometry derived from binary prefixes, labels retain native size with local scroll.
2. `QueensSearchLab`: four/five-row board, one queen per row, rejected candidate and its attacking queen(s), column/diagonal occupancy and undo state. Step/reverse/next-solution/reset use immutable snapshots, not time-based animation; solution list and coordinate table provide textual equivalents.
3. `SubarraySummaryLab`: editable bounded signed array, selected actual divide-tree node, contiguous left/right regions, four-field child summaries and highlighted left/suffix + right/prefix crossing candidate. Distinguish total, best prefix/suffix, best anywhere, all-negative and crossing winners. Counts derive from the actual divide/combine recursion; this is not a benchmark.
4. Inline inversion boundary figure exposes why choosing a smaller right item contributes every remaining larger left item at once. It complements the earlier sorting lesson rather than duplicating a sorting simulator.

## Complete programs and independent tests

Self-contained Python examples: all subsets, positive subset-sum pruning, frequency-based distinct permutations, positive combination sum, N-Queens, word-search witness with input restoration, inversion count and maximum-subarray summary; optional independent changed-contract solution if useful. Exact stdout verified. Independent oracles: itertools combinations/permutations; brute-force small grid paths; all pairs for inversions; all contiguous intervals for every summary field. Validate soundness/completeness, duplicates, zero/empty/negative boundaries, mutation restoration on early return, all-negative summary and non-mutating inputs.

Actual browser checks at desktop and narrow widths must include changed inputs, error recovery, keyboard, reversible finite trace, next-solution navigation and screenshots opened for visual inspection. Root owns registration/build; source-ready precedes verified completion.

## Routing the bitmask discovery

The unresolved inbox was relevant because subsets and N-Queens are often introduced with bitmask tricks. Do not introduce unexplained operators here. Ordinary lists/sets make the backtracking invariant visible; subset state encoding belongs naturally to later dynamic programming. A destination note is saved for `dynamic-programming-states-transitions-optimization` with a requirement to teach mask semantics before mask-based DP. Full bit-manipulation interview ownership (XOR identities, width/sign, shifts) remains unresolved; this routing does not claim the whole gap is closed or authorize a new topic. See [destination note](topic-notes/dynamic-programming-states-transitions-optimization.md).

## Research

Reviewed Stanford CS106B backtracking notes and Lecture 10 page/transcript for choices, duplicate elimination and search contracts; the lecture page provides an alternate video route (video playback not claimed). Reviewed CMU 15-210 divide-and-conquer notes for the four-value contiguous-summary construction, adapting to Python array indices and an explicit nonempty-subarray contract. Official LeetCode statements verified: 78, 47, 39, 79, 51 and 53. Source annotations distinguish C++/functional-language conventions from these Python examples and do not copy platform solutions.
