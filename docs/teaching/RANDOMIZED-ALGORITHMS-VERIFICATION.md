# Randomized Algorithms — implementation and author verification

Root integration amendment: three JSX text nodes containing raw comparison characters were entity-escaped after the original freeze. Complete normalized-AST equality verifies unchanged text and behavior. Final integrated body SHA-256 is `410c36f6376fa5b24b70e46429139863c0b4b09938b723d98f3b29d229a61137`; other owned runtime files are unchanged. [Exact before/after evidence](evidence/lesson-comparison-entities.json) and [production integration](DSA-MATH-FOUNDATIONS-INTEGRATION.md) supplement the dated evidence below.

Completed 10 September 2026 for `randomized-algorithms-sampling-error-guarantees`, DSA position 18. This is author verification, not user acceptance. Root owns integrated publication/build/navigation checks. Source hashes and exact timestamps are recorded in [the durable evidence JSON](evidence/randomized-algorithms-author-review.json).

## What changed and what the learner can do

The original entry contained a short specialist plan and no lesson body. The [individual design](RANDOMIZED-ALGORITHMS-LESSON-DESIGN.md) now leads to a complete route from finite probability experiments through unbiased draws, weighted tickets, permutations, uniform reservoir subsets, exact randomized selection, verification errors, amplification, estimation budgets and reproducibility. It introduces probability locally because the recorded Random Variables prerequisite has no body. Matrix–vector multiplication and the nonzero-coefficient argument are introduced locally as well.

The title and stable identity are retained. Actual prior Complexity/Ordered Patterns/Hashing/String Matching coverage was inspected; expected versus amortized costs and fingerprints are connected without duplicating their full chapters. The incoming String Matching note is resolved in sections 6–8. Scope routed to existing concentration, MCMC and flow/minimum-cut owners is recorded in their destination notes; the latter adds a new optional contraction assessment alongside the existing Reductions note. Detailed primality, skip lists and streaming sketches are not represented as completed by this lesson.

Eleven complete Python programs have actual checked output. Seven original local exercises require changed inputs or contracts, with initially hidden hints and reasoned answers. Six verified official LeetCode problems vary shuffle/reset, weighted tickets, rejection using a restricted source, node/eligible-index reservoirs and duplicate-aware rank selection. Their public statements were inspected10 September 2026:384,528,470,382,398,215, all currently Medium. No full statements, editorials or solutions were reproduced; no submission or universal-interview guarantee is claimed.

## Visual contracts and actual reading review

| Representation / placement | Meaning and limit | Review evidence |
| --- | --- | --- |
| Finite outcome tree, section 1 | Two independent fair bits branch to four equal paths; events select leaf sets | Four labels/masses inspected at 1440 and 390; no sampled probabilities |
| Raw-ticket mapping investigation, section 2 | Eight raw cells become biased modulo bins or six accepted cells with balanced bins; output bars use actual preimage counts | All 12 target/mode combinations per width; reset and exact counts. Mobile Mapping selection widened after actual clipped-label screenshot |
| Weighted ticket strip, section 2 | Six equal tickets partition into intervals of lengths1,3,2; strict cumulative endpoint lookup | Integer endpoints and zero-weight native cases; actual desktop/mobile figure opened |
| Shrinking-prefix shuffle, section 3 | A manual choice fixes the next suffix occurrence; self-swaps permitted | Three complete paths per width,9 swaps, Previous/reset/end-disabled state and fixed-suffix screenshot; full finite permutation oracle separately |
| Equal-marginal counterexample, section 4 | AB or CD each with probability1/2 gives equal inclusion but omits four pairs | Both-width static screenshot read; joint versus marginal distinction explicit |
| Reservoir slots and subset distribution, section 4 | One manually chosen execution is distinct from all uniform branch probabilities; occurrence IDs survive replacement |24 updates per width across k1/2/3 and accept/discard policies; Previous/reset/end/expanded subsets; actual narrow screenshot opened. Rounded percentages labeled; exact combinatorial oracle separately |
| Quickselect survival trace, section 5 | Three groups explain retained values and local rank; scanned-element count is not a timing |30 partitions per width under min/max choices at ranks0/4/8 plus a targeted retained-five-values capture. The initial capture accidentally returned to the first state; harness corrected to show actual partition evidence |
| Integer product probes, section 6 | A(Br) and Cr share one probe; cancellation, all-zero probe, even error and decisive mismatch visible | All 12 fixture/probe states per width, exact residuals, independent/reused modes, keyboard witness button and reset. Long mode selections given full mobile width; final screenshots opened |
| Majority probability bars, section 7 | All four error-count outcomes have exact masses27/64,27/64,9/64,1/64; majority tail is10/64 | Both-width images opened; Fraction and full binary-path oracle |
| Accuracy budget comparison, section 7 | Analytic sufficient sample counts at ε and ε/2; δ is failure probability |16 input pairs per width plus keyboard reset;185/738 visible; no empirical benchmark claim |
| Generator replay flow, section 8 | Input/state and deterministic draw order determine a replay; a seed does not prove independence | Desktop/mobile figure and ordinary-reading screenshot opened; actual Python state restoration and repeated-reseed outputs |

All five inline figures were opened at both 1440 and 390. All six investigations were opened in meaningful desktop/mobile states, including biased raw mapping, a fixed suffix, a nontrivial reservoir, a surviving partition, cancellation and an exposed mismatch. Ordinary-reading screenshots for the probability/stream/verification/error-budget/reproducibility routes and the introduction/sources were inspected. The compact first-pass examples remain visible before deeper derivations; the joint subset proof and exact expectation bound are expandable rather than omitted. Figures do not require operating a later lab to learn their meaning.

## Native/model evidence

Commands run from the application repository:

```powershell
node scripts/format-randomized-algorithms.cjs
node scripts/verify-randomized-algorithms.mjs
node scripts/review-randomized-algorithms.cjs
```

Formatter passes with normalized JavaScript AST/string and CSS-meaning comparisons. Final native/model result is `scratch/randomized-algorithms-verification/results.json`, timestamp 12:20:31.464UTC. Python runs with `-X utf8 -I` under the available 3.12 interpreter. Eleven programs passed exact stdout comparison. Focused evidence:

- 26,478 browser-model trace states; eight invalid model contracts.
- 1,054 independent raw-mapping cases;874 complete shuffle paths with the exact permutation set;10,158 native reservoir paths with uniform whole-subset counts; all 36 model reservoir configurations through n8 compared with combination sets.
- 1,536 weighted tickets across nonzero four-weight configurations;818 filtered-sample paths with exact eligible-index counts.
- 18,045 native selection cases over duplicate-filled arrays, sorted-result oracles and resource bounds;210 exact Fraction expectation states, with independent full pivot-tree work distributions for n≤8.
- 324 probes over all two-by-two error matrices with entries−1/0/1, checked against direct integer dot products and the nonzero-error detection bound; all 12 displayed fixtures checked using independent full matrix products.
- 30 majority distributions independently enumerated over all binary error patterns;20 sufficient budgets computed again with 60-digit Decimal logarithms.
-Seven hand-exercise arithmetic checks;11 invalid native inputs; zero-capacity reservoir proven not to consume its input. Actual empty/short stream, zero weights, missing target, singleton and duplicate examples are retained.

Finite enumerations verify their stated finite cases; they do not by themselves prove the general probability theorems. Those are justified in the lesson under explicit ideal-randomness and arithmetic assumptions. JavaScript percentages/expectations are display approximations of finite model quantities; integer probe arithmetic stays exact for the tiny fixtures. The programs are teaching implementations, not benchmarked production libraries.

## Browser and accessibility evidence

Final browser result `scratch/randomized-algorithms-browser/results.json`, timestamp 12:19:49.412UTC, passes both 1440×1000 and 390×1000 in headless Microsoft Edge. Tests close HMR WebSockets to isolate this lesson from concurrent authoring edits. The result JSON is copied into durable evidence.

Each width verifies actual keyboard activation and arrival at all 10 intro anchors, with section headings near100px below the fixed header and practice near90px. It checks all 11 displayed programs and stdout fixtures by title, six official practice links and initially closed hints, seven local exercise groups with closed solutions, reset/back/end behavior, and no page/lab/figure horizontal overflow or JavaScript page errors. Mapping now has an explicit accessible name. Initial harness failures were resolved: an exact wrapped-select label lookup and a fixture-order assumption (weighted sampling is taught before shuffle). Neither was recorded as a successful pass; the final run passed after these corrections and mobile layout repairs.

Native source, model and CSS files are topic-owned and on-demand. The lesson imports only its selected practice/examples/models; no aggregate DSA bundle or math renderer is introduced. Shared runtime and build integration are root-owned and are not claimed here.

## Sources and remaining limits

The design records exact inspected scope: Python random/secrets docs; Vitter 1985 pages 1–4 and Algorithm R; Cornell selection/stream notes (including an indexing caveat); MIT official video page and transcript through the matrix-verification explanation; UT Austin's matrix/polynomial sections with its field-versus-integer distinction; Waterloo's bounded-variable theorem/exponential-moment discussion. Alternate written/video links are annotated in the lesson. The MIT video was not watched end to end. Native seeded output is scoped to the recorded Python runtime and call sequence, not all versions. No cryptographic, adaptive-adversary or floating-point product guarantee is implied by an exact-integer proof.

No observed novice study or screen-reader speech session was performed. Browser checks establish the tested UI behavior and screenshots, not universal accessibility or learner mastery. Parent integration and user acceptance remain separate. The actual next DSA topic remains Network Flow, Minimum Cuts & Bipartite Matching.
