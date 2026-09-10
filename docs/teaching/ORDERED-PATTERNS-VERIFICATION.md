# Binary Search, Sorting & Two-Pointer Patterns: completed author verification

10 September 2026. Implemented and author-verified for the active DSA/maths rollout. This record covers one new lesson at stable ID `binary-search-sorting-two-pointer-patterns`, not the entire rollout. Parent/root registered the lesson and individual brief in the existing module order. Integrated publication/build review is owned by that task; this record does not claim those checks or user acceptance.

## Teaching and source scope

The [design/source ledger](ORDERED-PATTERNS-LESSON-DESIGN.md) records the topic-plan result, ownership/title decisions, representation contracts, primary written/video sources and fourteen verified official LeetCode statements. The lesson teaches lower/upper boundaries and absence; stable insertion/merge sorting; three-way partition and optional heapsort/counting/radix concepts; opposing and read/write pointers; unique value triples; closed interval union; fixed, nonnegative and uniqueness windows; signed prefix frequencies and prefix/suffix summaries; integer feasibility search; and a static time-range report that combines the methods.

Complete code and explanatory practice remain self-contained. The practice dataset includes 704,35,34,912,88,26,56,167,15,643,209,3,560,875 with staged original annotations, initially closed hints/transfer and relevant output/prerequisite distinctions. Statements were inspected on 10 September; no editorials, external submissions or full-video viewing are claimed. The linked MIT video has substantively inspected companion notes. There is no fixed problem/lab quota or universal interview-mastery guarantee.

Primary implementation files:

- `src/learn/data/topics/binary-search-sorting-two-pointer-patterns.jsx`
- `src/learn/data/curriculum/blueprints/binary-search-sorting-two-pointer-patterns.js`
- `src/learn/data/ordered-pattern-examples.js` and `ordered-pattern-models.js`
- `src/learn/components/lesson-labs/OrderedPatternLabs.jsx` and `ordered-pattern-labs.css`
- `src/learn/data/practice/binary-search-sorting-two-pointer-patterns.js`

Only this lesson imports these examples/models/practice. No shared lesson aggregate, math renderer, generated catalogue, global stylesheet or eager loader was added. Native functions/models use semantic names and explicit invariants; canonical index names remain where they aid algorithm reasoning.

## Native and independent model checks — passed

Command: `node scripts/verify-ordered-patterns.mjs`. It uses `scratch/lesson-tools/Scripts/python.exe` unless `LESSON_PYTHON` overrides it, writes reproducible standalone programs/model records, and calls `scripts/verify-ordered-patterns-native.py`.

Observed Python: **3.12.14**. All **17 complete programs** matched their displayed stdout exactly: boundaries, record ordering, stable sorting, partition sort, heapsort, counting sort, pair sum, compaction/tail merge, triples, interval union, fixed windows, nonnegative windows, distinct-character windows, prefix/suffix summaries, signed counts, feasible-rate search and time-range reports. The online Python documentation's newer release is not claimed as the executed runtime.

Independent results saved in `scratch/ordered-pattern-verification/results.json`:

| Check | Actual cases / oracle |
| --- | --- |
| Sorting, stability, duplicates and multiplicities | 3,906 arrays of lengths 0–5 over five signed values; Python sorted plus tagged identity ordering, against insertion/merge/three-way quicksort/heapsort/counting implementations |
| Boundaries, pairs, triples, signed sums, fixed windows and outside products | The same 3,906 arrays; bisect, exhaustive index pairs/triples/ranges and direct products; includes duplicate identities, absence, zeros, negatives and ties |
| Shortest nonnegative windows | 8,744 input/target combinations, compared with all qualifying nonempty intervals and earliest-tie policy |
| Longest unique substrings | 3,280 strings over three characters, compared with exhaustive contiguous substrings and earliest-tie policy |
| Tail merge and interval union | 400 deterministic random fixtures; sorted combined inputs; exact coverage at all integer/half-integer candidate points with disjoint output checks |
| Time-range report | 1,600 queries over 400 generated static datasets; direct filter/count/sum, duplicate timestamps, signed amounts and empty queries |
| Integer feasible rates | 1,440 job/budget cases, compared with exhaustive candidate rates and per-job integer ceilings; includes impossible budgets |
| JS/native correspondence | 4,200 boundary traces, 2,100 pair traces, 968 window traces, 19 feasibility budgets and three valid merge tie paths |

Intermediate-state checks establish before/after boundary partitions and strict progress, preservation of all possible pair solutions, window sums versus actual slices and valid saved best intervals, feasibility monotonicity and ceiling counts. Merge tests distinguish a sorted result from stable original tie order and reject a larger-head move. Meaningful invalid sortedness, NaN, malformed side, negative-window input, nonpositive target/rate, invalid jobs/intervals/query bounds and excess browser input are exercised. No timing benchmark or proof-by-sample claim is made; prose supplies the correctness and cost arguments.

## Browser, accessibility and opened visuals — passed

Command: `node scripts/review-ordered-pattern-lesson.cjs`, shared dev server `http://127.0.0.1:5173`, Playwright Chromium through installed **Microsoft Edge**, at **1440×1000 and 390×1000**, reduced-motion preference. Stateful pages suppress the Vite HMR socket only in the test context so another author's generated metadata changes cannot reset the current trace. Production isolation is a separate root-owned check.

Final full functional record: `scratch/ordered-pattern-lesson-review/results.json`, **2026-09-10T07:52:20.620Z**. Each viewport passed:

- 81 boundary states across empty/singleton/duplicate/signed arrays, both boundary conventions and lower/exact/upper targets. Rendered candidate gaps, proved regions, middle and final boundary match the verified model. Malformed/unsorted/out-of-range/overlong edits leave the active trace unchanged; reset clears errors. Back/restart/terminal disabled state and keyboard activation pass.
- Two merge paths, one stable and one deliberately unstable, with exact record IDs, larger-head rejection, empty-lane disabling and reset.
- 22 pair states covering lower/upper no-answer and matching cases; candidate-region sizes and next-action explanations agree.
- 66 window states covering positive, zero-containing and empty inputs with small/impossible targets; active cells, totals and saved intervals agree.
- Twelve rate configurations, including impossible budgets and small/large speeds; exact slot blocks and totals agree.
- Seventeen full code/output blocks, fourteen practice links, ten unique actual anchor targets, keyboard opening/closing of practice hints and clicking the practice route anchor.
- No page/lab/inline-figure horizontal overflow or runtime page errors. Intended long-array scrolling stays inside its labeled focusable region.

The first browser run stopped on a **test selector mistake**: the practice stages are divs, not nested sections. Fixed the harness selector and completed both viewports. No application failure was hidden by that fix.

The author opened actual screenshots for all five investigation types and all four static figure types at both widths: boundary gap regions; stable/unstable merge lanes; pair-candidate matrix; current/best window; feasible work budget; four partition regions; read/write compaction; closed interval union; prefix cancellation. Also opened the mobile route and guided-practice stage. The separate reading-flow assessment confirms that structures appear beside the prose introducing them, including the easily missed partition swap rule; essential reasoning is not locked inside lab controls.

Visual review repaired the mobile default boundary strip so all six values and seven gaps fit together. Compact i/g labels are explained in the caption; longer arrays retain horizontal keyboard scrolling. Final intrinsic cell widths accommodate four-digit signed values without shrinking their numeric text. A separate final check, `node scripts/review-ordered-boundary-layout.cjs`, passed both sizes on the final CSS: default strip fit, sixteen large signed values, actual keyboard scrolling, empty boundary and page overflow. Saved `scratch/ordered-pattern-lesson-review/boundary-layout-results.json`, **2026-09-10T07:53:43.608Z**. Refreshed/opened `boundary-regions-390.png`, `boundary-long-390.png` and `rates-budget-390.png`; the latter also confirms singular “1 slot” wording. No model/example semantics changed during the layout repair.

The screenshots are reproducible scratch evidence, not committed delivery assets. Figure data are deterministic arrays or analytic integer counts; no invented performance graph is presented as observed data.

## Coverage handoff and limits

The [range-query destination note](topic-notes/segment-trees-fenwick-trees-range-queries.md) preserves the discovered obligation for monotone-deque dominance, sliding extrema and signed shortest-at-least windows, with the `[1,-1,5]` counterexample, prerequisites, proposed visuals/oracles and required owner reassessment. Those specialized methods remain unimplemented here; the current window/prefix scope is stated accurately. Bit-manipulation ownership remains the existing unrelated unresolved inbox item.

This is author testing and a heuristic first-learner walkthrough, not an observed beginner study. Parent/root's independent review and integrated production checks may identify further refinements. Publication, passing tests and this record do not imply user approval. The module continues to Backtracking & Divide-and-Conquer in its existing sequence.
