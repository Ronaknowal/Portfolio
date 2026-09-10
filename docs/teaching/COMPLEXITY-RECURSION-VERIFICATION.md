# Complexity Analysis & Recursion — implementation and verification

10 September 2026. Read with [the individual design](COMPLEXITY-RECURSION-LESSON-DESIGN.md), current teaching/code standards and [DSA practice standard](DSA-PRACTICE-STANDARD.md). Root coordinates registration and application-wide integration. Stable title, ID and module order are preserved.

## What was implemented

The former planned entry now teaches operation/size/case contracts, exact loop counts, O/Ω/Θ and counterexamples, expected versus amortized costs, recursive contracts/progress/induction, actual waiting frames, recurrence level sums, space/copying/output/bit costs, repeated subproblems, and the limits of timing evidence. The next route remains Binary Search, Sorting & Two-Pointer Patterns.

Ten complete Python programs include zero/odd/signed cases and interpret their outputs. An iteration lattice, a suffix-call frame investigation and a recurrence level accountant address different mechanisms. Two immediately visible figures show call/return correspondence and retained slices. Calculated operation-count bars are explicitly not benchmark measurements. Diagnostic snapshot overhead is excluded from uninstrumented algorithm costs, and visit-frame counters distinguish the fixed wrapper.

Local predictions, counterexample repairs and an independent threshold-count implementation have concealed explanations/hints and interpreted solutions. Seven official LeetCode statements were reviewed for title/ID/difficulty/constraints/access: 896,1342,344,509,50,104 and optional70. The tree-depth question is an intentional cost-analysis revisit, and the stairs question is a DP bridge. No external submission/editorial or full-video viewing is claimed; all complete core teaching remains local. The source ledger records MIT PDF/companion-note scope and Python3.14.7 documentation, with examples executed on Python3.12.14.

## Computational checks — passed

Command: `node scripts/verify-complexity-recursion-examples.mjs`. This runs the model check first to create fresh native-comparison fixtures, then executes each exact displayed code string in an isolated Python process and compares all stdout, followed by the independent Python corpus. Runtime defaults to repository `scratch/lesson-tools/Scripts/python.exe`, overridable with `LESSON_PYTHON`. No network or installation is part of verification.

| Evidence | Passing coverage | Independent check |
| --- | ---: | --- |
| Complete displayed programs | 10 | Actual Python execution and exact displayed stdout |
| Loop model | 99 pattern/size configurations | Combinatorial formulas and independent membership of index pairs; includes0…32 |
| Recurrence model | 30 configurations | Closed-form sums and separate depth/call identities for all five patterns, n=1…32 powers of two |
| Frame model | 279 complete traces | Suffix totals, index nesting, base/combine results, counts and snapshot isolation |
| Native sums/reversal | 1,093 arrays | Exhaustive length0…6 over −2,0,3; built-in summation/reversed references; balanced call and peak-depth identities |
| Native threshold task | 6,558 list/threshold combinations | Independent filtering count, including equality and signs |
| Native search | 4,372 cases | Independently enumerated matching positions and exact first/absent costs |
| Native powers | 845 | Built-in pow plus binary length/population identities for call/multiplication counts |
| Native recurrences | 45 | Closed forms through power-of-two size256; chain and halving distinctions |
| Native append sequence | 513 lengths | Separate geometric sum of copied powers; capacity and aggregate bound |
| Native Fibonacci | All n=0…20 | Iteratively built sequence, exact call-count identity, newly computed states |
| JS/native frame correspondence | 279 complete event sequences | Native recursive diagnostic compared to all event/frame/call/result data, not just final sums |
| Invalid boundaries | Passed | Malformed/overlong/out-of-range lists, unknown models, unsupported sizes and selected Python domain/type rejection |

Evidence: `scratch/complexity-recursion-verification/model-results.json`, `native-results.json`, exact `.py` fixtures, `examples.json` and `model-traces.json`. The models are bounded and deterministic. These are finite computational checks, not empirical asymptotic proofs; the lesson supplies its own reasoning. Exact runtime-source formatting preserved normalized Babel AST equality. The first model test encountered signed-zero in its expected triangular formula at n=0; the oracle now normalizes zero without changing the model or counts.

## Browser and visual review — passed

`node scripts/review-complexity-recursion-lesson.cjs` passed on the registered route at `http://127.0.0.1:5173` using Playwright/Microsoft Edge, 1440×1000 and 390×1000, reduced motion. At each width it verified:

- Fifteen loop pattern/size cases, including empty, singleton and maximum visual size; exact event counts and cell membership, row selection, keyboard Home/ArrowRight and reset.
- Sixty-three complete frame states across empty, zero, signed, three-value and eight-value inputs. Live indices, pending additions and returned values agree with the separately native-verified model. Last-event disabling, Back, Restart, four invalid atomic rejections, reset and keyboard advancement pass.
- Fifteen recurrence configurations spanning all five models at sizes1/8/32; every selected level’s node count, exact summed work, base cases and reset agree with the independent formulas.
- All ten complete rendered code/output blocks, including the initially concealed independent solution; seven official practice links, nine unique valid anchors and actual guided-practice anchor arrival.
- Enter/Space hint opening/closing, initially concealed optional material, all lab input/select/button targets at least43px tall, no page/lab overflow and no page errors.

Evidence: `scratch/complexity-recursion-lesson-review/results.json`, dated 10 September 2026 07:13 UTC, and twelve corresponding screenshots. Screenshots are captured with the fixed navigation hidden only during capture, then restored. Test locators were corrected for the selected recurrence control, a hidden solution heading and leading-numeric anchor IDs. One earlier mobile attempt was interrupted by concurrent generated-navigation/TopicContent hot reload, confirmed in the shared Vite log; the final test holds each page's initial source stable by closing its development WebSocket. This is test isolation, not a change to application behavior. Both widths then passed together.

Opened desktop and mobile lattice, pending-frame, recurrence, call/return-ladder and slice-storage screenshots, plus mobile practice. The selected row and pair geometry are clear; suspended parents retain their pending additions; bar lengths agree with exact level work; call direction and return values remain paired; copied reference slots retain their staircase meaning. On mobile, the return ladder and frame rows recompose vertically, the six-column default lattice fits, and exact recurrence tables remain readable. Larger lattices intentionally retain a local scrolling region. The ordinary reading pass confirms that input-size vocabulary precedes count geometry, recursive contracts precede frames, and storage distinctions precede the retained-slice figure. No missing core explanation or outstanding visual blocker was found in this author review. No beginner study or user acceptance has occurred.

## Status

- Implementation: complete and registered by root.
- Computational verification: passed as above.
- Browser/visual: passed at 1440px/390px, with opened screenshots and the inline reading pass above.
- User acceptance / observed beginner learning: not performed.
- Integration/build/next scope: root coordinates the active larger rollout; no unrelated source was edited here.
