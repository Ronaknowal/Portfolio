# Dynamic Programming: States, Transitions & Optimization — design

Stable ID `dynamic-programming-states-transitions-optimization`; DSA position 11. Original implementation begun 10 September 2026 under the active rollout. This design is not completion evidence; see the separate verification record when finished.

## Learning contract and scope

The learner can read Python lists/dictionaries, recursive calls, basic asymptotic costs and the earlier backtracking search contracts. The recorded correctness prerequisite is planned later in the route; this lesson supplies the needed induction, dependency order and termination arguments locally. Do not assume a prerequisite title is a completed explanation.

Start with choosing valuable nonadjacent experiment sessions. Distinguish an already decided prefix from a reusable suffix problem; derive, prove and implement the recurrence before optimizing its storage. Finish by designing new state spaces, reconstructing witnesses, identifying invalid compression and representing finite used subsets explicitly. Retain the existing title: all additions explain state design, transitions or justified optimization rather than a separate subject.

| Hurdle | Mechanism and evidence planned | Representation and learner action |
| --- | --- | --- |
| Repeated calls versus distinct states | Nonadjacent reward suffix; memo and reverse tabulation compute the same values; sufficient entry contract, base and decreasing remaining length | Dependency diagram with actual cache values and active call stack; switch traversal, step reads/writes, compare same result |
| Hidden history changes the future | Same index with previous session selected versus free; repair key or enforce free suffix on entry | Inline two-history timeline; predict whether a reward is still legal |
| An earlier greedy proof no longer fits the objective | Finish-ordered weighted jobs; compatible prefix, binary-searched boundary, skip/take proof and reconstructed original indices | Inline duration lanes preserve the actual 8-versus-10 counterexample; changing one reward reverses the winner |
| Unreachable versus zero and return path | Right/down signed-cost grid, blocked cells, parent pointers, deterministic ties; count variant | Spatial cost grid and computed table combined; toggle an obstacle, fill cells, follow the witness |
| Sequence states are pairs of prefixes | LCS derivation, correspondence to characters, reconstruct one witness; edit-distance transfer | Prefix alignment table with match/skip dependencies and traced character pairs |
| Same array, different logical row | Full 0/1 knapsack to descending compressed updates; ascending changes reuse contract | Capacity strip with source/destination and row generation labels; make the accidental repeated item visible |
| Counting needs disjoint choices | Ordered sequences versus unordered coin combinations, min versus count versus existence, correct neutral/impossible states | Inline complete amount-4 comparison with [1,3], not an invented performance chart |
| A subset needs a representation and sometimes an endpoint | Membership/shift/AND/OR/clear/bounded complement; minimum visit-once route from a fixed start | Linked membership switches, bit row, set, endpoint candidates and weighted next edges; compare same mask at different endpoints |
| Optimization must preserve semantics | Rolling rows, pseudo-polynomial bounds, LIS dominance, state explosion and reconstruction tradeoffs | Exact worked code and small counterexamples; no timing curves |

## Visual contracts

Five proposed investigations have different learning jobs, not a required count. All exact integer computations are derived from pure models, with independent brute-force/native oracles. Browser inputs are deliberately small: rewards up to 7 values; grid fixed 3×4 with independently toggled blocks; strings at most 6 ASCII letters; capacity 0–8, two positive-weight items; mask universe four labeled vertices. Position encodes list order, grid adjacency, prefix length, capacity or bit index. Color is redundant with numbers, arrows, state labels and accessible descriptions. No geometry represents wall-clock performance.

Final inline support also includes the minimum-tail/non-witness contrast and weighted-interval duration lanes. The default six-node dependency graph fits 390px; longer input graphs scroll with the active state visible. Capacity writes remain visible while their source row/version is also explained textually. These choices followed opened screenshot review, not a diagram quota.

Each interaction states when edits apply, exposes current inputs/results, supports previous/next or direct inspection and deterministic reset, and has keyboard and 390px checks. Wide tables retain readable cells in a labeled focusable scrolling region; the page itself must fit. Static timeline/counting figures are immediately visible where the misconception is introduced and need no redundant controls. Review ordinary reading flow as well as fully operated states.

## Scope discovery

| Idea | Prior evidence / owner | Decision |
| --- | --- | --- |
| Mask primer | Backtracking uses explicit sets and routes a detailed destination note here | Include before subset DP, then resolve that note against implemented code/tests. Broader XOR tricks, integer-width interview coverage and the unassigned inbox remain unresolved. |
| State insufficiency | Backtracking's ABABX grid gives two prefixes with opposite feasible futures | Retain that bridge; add local cooldown and subset-endpoint counterexamples rather than copy a full word-search unit. |
| Range-query acceleration / advanced DP optimizations | Segment Trees/Fenwick/Range Queries follows this lesson; broader recurrence optimizations require extra structural assumptions | Teach a justified LIS dominance optimization locally; route potential range-max and advanced optimization comparisons with explicit prerequisites instead of claiming universal DP coverage. |
| Sequence alignment applications | LCS/edit distance make insert/delete/replace costs concrete | Explain diff-style alignment and why application scoring/tokenization changes the contract; do not claim every production diff or biological aligner uses this exact implementation. |
| New Greedy destination note during writing | Actual preceding lesson proves failures at weighted schedule 8 versus 10 and whole-item capacity 160 versus 220 | Add the full weighted-schedule recurrence/program/witness and exact 220 knapsack fixture; retain the existing capacity-generation investigation because it already teaches the reuse flaw. |

The incoming [DP notes](topic-notes/dynamic-programming-states-transitions-optimization.md) record both completed adaptations and the still-open owner assessment for advanced interval/tree/digit DP and specialized optimizations. The [Range Queries destination](topic-notes/segment-trees-fenwick-trees-range-queries.md) records the concrete range-max acceleration proposal. No full-site audit or universal finite-list guarantee is claimed.

## Research and practice plan

Primary sources: official MIT 6.006 2020 dynamic-programming notes and video pages; official Python integer/bitwise and functools documentation; primary university material for subset-route recurrence. Inspect relevant written sections and record actual video metadata/notes scope, not a full viewing claim. Original examples, wording and diagrams; links supplement self-contained explanations.

Practice spans direct state reconstruction, changed objective/constraints and independent mixed tasks. Official LeetCode statements are checked individually for number/title/difficulty/public access; no quota or universal interview guarantee. Core matches taught mechanisms; optional cooldown/visit-every-cell variants name additional modeling work. Local exercises require a complete state contract, correctness argument, adversarial tests and explained answers even if all external links disappear.

## Verification plan

Execute every displayed Python program and compare exact stdout. Exhaustively enumerate small valid choices independently for nonadjacent reward, grid paths, knapsack/subsets, LCS/LIS and visit-once routes; use breadth-first edit search for short strings. Check every table state, parent witness, tie behavior and compressed row-generation read. Repeated/negative/empty/unreachable inputs and invalid browser boundaries are part of the contracts. Inspect every investigation and static figure at 1440/390, operate controls with keyboard, inspect anchors/code/output/practice hints and open screenshots. Root owns registration, shared integration and full build.
