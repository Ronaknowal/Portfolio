# Dynamic Programming — implementation and verification

Stable ID `dynamic-programming-states-transitions-optimization`, existing title and DSA position 11 preserved. Implemented and author-verified 10 September 2026 under the active rollout, including the incoming Greedy examples. Native, independent model, full browser and focused layout checks passed. Parent owns manifest/blueprint registration and integrated build; publication and these checks are not user acceptance.

## What the lesson teaches

[Lesson](../../src/learn/data/topics/dynamic-programming-states-transitions-optimization.jsx), [individual blueprint](../../src/learn/data/curriculum/blueprints/dynamic-programming-states-transitions-optimization.js), [design](DYNAMIC-PROGRAMMING-LESSON-DESIGN.md).

The route derives a free-suffix recurrence from a nonadjacent experiment selection, proves its exhaustive choices/termination, and compares memoization with reverse tabulation. It separates call invocations from complete state keys, returns witnesses and exposes history/immutable-input cache validity. A concrete incomplete-key example compares the same remaining reward under different permissions.

The new Greedy handoff is resolved with finish-ordered weighted scheduling, compatible-prefix boundary search and reconstruction, plus the exact whole-item knapsack counterexample. Later sections develop signed/blocked right-down grid costs, counts and parents; LCS prefix states and traceback; unit-cost edit distance; full 0/1 knapsack and generation-correct compression versus unlimited reuse; exact subset reachability; minimum coins, ordered sequences and unordered combinations; a finite-mask primer and subset-plus-endpoint visit-once routing; pseudo-polynomial/bit/storage costs and a proved LIS dominance optimization.

Independent practice includes command-stream segmentation with complete explained code, exactly-two nonadjacent selection, counting cheapest grid paths and three counterexample repairs. Hidden answers supplement distributed checkpoints. The command example charges actual Python substring/hash work instead of confusing O(n²) candidate pairs with O(n²) character work. No result implies universal interview readiness.

The recorded later correctness prerequisite is supplied locally with the required induction/order arguments. The title remains accurate; no module reorder, prior lesson deletion or shared source edit was needed. [Destination notes](topic-notes/dynamic-programming-states-transitions-optimization.md) record implemented mask/Greedy adaptations and remaining advanced DP-family ownership. [Range Queries](topic-notes/segment-trees-fenwick-trees-range-queries.md) receives a concrete range-max recurrence-acceleration proposal. Broader bit/XOR coverage remains unresolved, and interval/tree/digit DP plus special optimization theorems are orientation/deeper obligations, not claimed complete units.

## Representation contracts and accuracy

[Pure models](../../src/learn/data/dynamic-programming-models.js), [components](../../src/learn/components/lesson-labs/DynamicProgrammingLabs.jsx), scoped `dynamic-programming-labs.css`.

| Representation / placement | Learning job, behavior and boundaries | Evidence |
| --- | --- | --- |
| Reward dependency investigation, section 1 | Exact skip/take graph; actual cache, stack and event counts; switch memo versus table, edit/apply up to seven signed integer rewards, reset/previous/next/finish. Arrows mean question→dependency. Unknown is distinct from cached zero. | Every write's inputs ready; memo/table/native agree; exhaustive free-suffix oracle. Default graph fits 390; long graph centers active state and remains keyboard-scrollable. No timing claim. |
| History timeline, section 2 | Same remaining index, blocked reward 9 versus available 9; answers 0 and 9 differ | Complete two-key native example and exhaustive initially-blocked comparisons. |
| Weighted interval duration lanes, section 2 | A [0,5) value 10 versus touching B [0,2), C [2,5) each value 4; duration geometry and count/value distinction | Exact fixture plus 300 independent all-subset schedule cases; original witnesses and half-open compatibility checked. |
| Spatial grid investigation, section 3 | Fixed 3×4 costs; toggled obstacles restart evaluation; current cell, available predecessors, cheapest prefix, all-path count and final witness share one model. Zero/unknown/unreachable differ. | Exhaustive move-order paths, all 64 obstacle masks on signed 2×3 grid and 250 seeded small grids; every cell total/count and native parent tie policy compared. |
| LCS prefix investigation, section 4 | Select any cell to inspect its two prefixes and match/skip dependencies; trace one witness backward, highlight matched input indices; empty through six ASCII letters, case-sensitive | All 961 short binary-string pairs and every prefix cell checked by independent subsequence enumeration; exact witness correspondence. Wide inputs use local scroll; default fits. |
| Capacity generation investigation, section 5 | Inspect destination/source versions under descending once-only or ascending reusable contracts; capacity 0–8, fixed positive weights; visible occurrence witnesses expose repetition | Full native knapsack and independent count-vector enumeration; every diagnostic witness respects capacity and its reuse contract; 288 model traces. Snapshots/witness copying excluded from value-only bounds. |
| Coin-order comparison, section 6 | Amount 4 with [1,3]: three sequences, two multisets | Explicit enumeration and native exact fixture. No benchmark axes. |
| Membership/endpoint investigation, section 7 | Four bits linked to set/number, chosen endpoint, valid cheapest prefix and exact next-edge costs; complete fixed graph matrix; unreachable invalid states remain explicit | Python bit/set equivalence for 510 masks; independent permutation oracle for every state in 120 directed/missing/signed route matrices. Browser masks stay within JavaScript bitwise range. |
| Tail summary contrast, section 8 | Input [3,5,6,2] versus minimum tails before/after last value; length-3 witness differs from combined summary cells | Both LIS implementations against 1,093 exhaustive arrays; displayed counterexample checked. |

Five investigations and four inline figures are the chosen support, not a quota. All numbers are computed or exact teaching fixtures; no empirical timing curves or unsupported implementation rankings appear. Geometry shows indices, time intervals, grid adjacency, capacity, prefix length or bit membership; scales are not performance measurements.

## Native and independent model results

Command: `node scripts/verify-dynamic-programming.mjs` (uses `scratch/lesson-tools/Scripts/python.exe`, Python 3.12; override with `LESSON_PYTHON`). Complete [examples](../../src/learn/data/dynamic-programming-examples.js), [JS verification](../../scripts/verify-dynamic-programming.mjs), [native verification](../../scripts/verify-dynamic-programming-native.py). Final saved results: `scratch/dynamic-programming-verification/results.json`, 10 September 2026 08:42:51 UTC.

**Passed all 16 exact complete Python programs**, after normalizing Windows newlines. Their stdout is displayed with the code. Independent checks:

- 3,280 nonadjacent reward inputs, including signed/zero/empty and blocked-entry states; every selected witness validated.
- 300 weighted schedules with independently enumerated subsets, ties/touching/negative values and chronological original-index witnesses.
- 250 signed/blocked rectangular grids by enumerating all right/down move orders.
- 961 sequence pairs by all-subsequence sets; 225 short edit pairs by breadth-first legal edit search rather than the same recurrence.
- 450 item/capacity fixtures by all legal multiplicity vectors; 4,368 exact subset targets; 144 coin objectives by complete sequences/multisets.
- 1,093 LIS arrays by exhaustive input subsequences; 510 masks by ordinary set operations/submask enumeration; 508 command-stream cases by every cut set.
- JS/native correspondence: 364 reward tables, 64 full obstacle masks with every grid cell checked, 961 LCS tables, 288 capacity traces and 120 route matrices with every reachable state checked independently.

Tests supplement the prose proofs; they do not prove asymptotic bounds. The first quick smoke runner failed to normalize CRLF and therefore printed mismatches despite identical displayed lines; the durable runner uses normalized exact output and passes. No example output was weakened to hide a semantic failure.

## Browser and visual review

Commands: `node scripts/review-dynamic-programming-lesson.cjs` and `node scripts/review-dynamic-programming-layout.cjs`. Playwright/Edge path defaults are in the scripts; shared Vite `http://127.0.0.1:5173`, 1440×1000 and 390×1000, reduced motion. Test pages close their own Vite WebSocket to prevent unrelated concurrent authoring changes from resetting the active model. Fixed navigation is hidden only while capturing a lab element, not during functional checks.

Initial full functional run passed both sizes before the incoming Greedy addition: per width 83 reward states, 48 grid states, 111 sequence-prefix inspections, 66 capacity states and 24 subset/endpoint combinations; code/output, hints, reset, keyboard controls, anchors, invalid input and no page/lab overflow. An earlier run exposed an ambiguous implicit select-label match; explicit accessible select names fixed it.

Opened screenshot review found two meaningful mobile improvements: the default dependency graph previously required scrolling to see its base nodes, and a capacity update could write beyond the visible strip. The final graph is compact enough to fit while longer graphs follow the active state; the capacity strip keeps its written cell visible without moving focus and retains local keyboard inspection. The focused final layout run at **08:43:10 UTC** passed default fit, long active-state visibility, all 18 capacity writes per size, arrow-key scrolling at 390, ten ordinary reading captures per size and no errors/overflow. Its initial harness assumed End/Home would scroll horizontally; the corrected test uses the actual horizontal ArrowRight/ArrowLeft behavior.

The final full run after weighted scheduling and the refined layout passed at **08:44:45 UTC**, saved in `scratch/dynamic-programming-lesson-review/results.json`. At each width it checked 83 reward states, 48 grid states, 111 sequence-prefix inspections, 66 capacity states, 24 subset/endpoint combinations, all 16 complete code/output pairs, all 12 practice entries and all 10 route anchors. Resets, keyboard controls, closed hints and keyboard opening, empty/negative/unreachable/tied inputs, invalid-input feedback and local/page overflow checks passed. The error list is empty.

Screenshot evidence lives in `scratch/dynamic-programming-lesson-review/`. Opened inspection covers all five investigations at both sizes, empty/unreachable and wide-string states, all four inline figures and ordinary reading samples at the start, sequence transition, optimization transition and independent task. Final weighted-interval duration lanes were inspected at 1440 and 390: shared boundaries at 0, 2 and 5 preserve the intended comparison, labels fit and touching compatibility remains explained. Final compact reward and visible capacity-write captures were opened, as were the mobile practice/hint flow and desktop/mobile coin and tail-summary figures. The route and code/answer flow stay readable; large tables retain labeled local scrolling. No observed beginner study or user acceptance is claimed.

## Sources and external practice

[Practice dataset](../../src/learn/data/practice/dynamic-programming-states-transitions-optimization.js) links **12 verified public official statements**: 198 House Robber (Medium), 64 Minimum Path Sum (Medium), 63 Unique Paths II (Medium), 1235 Maximum Profit in Job Scheduling (Hard), 1143 Longest Common Subsequence (Medium), 72 Edit Distance (Medium), 416 Partition Equal Subset Sum (Medium), 322 Coin Change (Medium), 518 Coin Change II (Medium), 300 Longest Increasing Subsequence (Medium), optional 309 Best Time to Buy and Sell Stock with Cooldown (Medium) and 980 Unique Paths III (Hard). IDs/titles/difficulties and relevant contracts were read on 10 September 2026; public statement availability does not promise free editorials, platform accounts or a judge submission. Hints are closed by default, annotations are original, variants and extension prerequisites are explicit. Weighted scheduling is core despite the platform's Hard label because the necessary method is taught; difficulty does not reorder the syllabus.

Actual inspected primary scope:

- MIT 6.006 spring 2020 lecture 15 notes: reusable subproblems, dependency order, memo/tabulation, reconstruction and bit-aware costs; lecture 16 notes: LCS/LIS state reasoning and parents; lecture 18 notes: numeric states, subset sum and pseudo-polynomial interpretation.
- The corresponding official video pages for parts 1, 2 and 4 were inspected and linked as annotated alternate routes. Written companions were reviewed; full playback is not claimed. A YouTube iframe fetch failed, so the working official course pages are used rather than pretending to have watched it.
- Official Python `functools.cache` and integer bitwise documentation (current page identified as 3.14.7); executable examples use the available 3.12 runtime. Hashability, cache lifetime, shifts, arbitrary precision, complement and language-boundary claims distinguish the two.
- CMU 15-451 spring 2025 lecture 11 slides: subset/endpoint/TSP path framing and range-query LIS bridge. Unannotated slides leave some formulas blank; the lesson derives and verifies its own full path recurrence, and the record does not treat blank slides as a complete proof.
- Princeton COS423 dynamic-programming lecture, weighted-interval pages 5–14: finish order, compatible predecessor, include/exclude proof, runtime and witness recovery. The local code uses explicit prefix lengths and a tested half-open convention.

All primary/alternate links are attached within the lesson. No source's wording, editorial solution or visual layout was copied as a template. Synthetic examples are presented as teaching models rather than claims of deployed systems, clinical validity or measured performance.
