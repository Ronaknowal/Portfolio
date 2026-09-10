# Algorithm Correctness, Loop Invariants & Termination — verification

10 September 2026. Stable ID `algorithm-correctness-loop-invariants-termination`; DSA position 13. The preceding route topic is Range Queries; the next is Hashing, Collision Resolution & Amortized Analysis. This completes a previously planned lesson, without removing another lesson, changing an ID or replacing earlier proof chapters. Root owns publication, generated artifacts and integrated build. User acceptance remains separate from author verification.

Read the [lesson design](ALGORITHM-CORRECTNESS-LESSON-DESIGN.md), [DSA practice standard](DSA-PRACTICE-STANDARD.md) and [learning code standard](../engineering/LEARNING-CODE-STANDARD.md) with this record. The exact topic-plan command and returned notes were read before authoring; there was no existing authored body or incoming destination note. The original brief's search/invariant/termination scope was retained and expanded. No new title or out-of-scope catalogue audit was needed.

## What the learner can now do

The lesson develops exact pre/post/mutation contracts, first-versus-any matching, empty ranges and local logic before introducing invariant terminology. It proves initialization, preservation and both early/guard exits separately; explains partial versus total correctness; and derives invariants from a partly completed postcondition. The weak/strong-claim investigation distinguishes arbitrary states admitted by an assertion from merely reachable states.

Backward assignment reasoning leads to an old-value swap timeline. Stable compaction introduces original occurrence identity, the logical result prefix and an untouched unread suffix; zero filling supplies a second compositional proof phase. Four-region partition explains why a value arriving from the unknown end cannot be skipped. Euclid connects preserved divisors to a strictly decreasing remainder; exact real halving and resetting natural-number pairs explain the required well-founded order. Strong induction, iterative exponent conservation and modular reduction connect recursive and algebraic proofs. The final sections connect earlier Greedy, DP, Backtracking and range-query arguments, distinguish soundness/completeness/optimality, test full output contracts and independently derive integer square root.

Core logic and induction are introduced locally because the formal Sets/Logic prerequisite appears later in the reader sequence. The complete earlier algorithm-family chapters are linked conceptually rather than duplicated. The formal-tools discussion is an optional bridge, not a claim that annotated Python or browser indicators are machine-checked proofs.

## Owned source and representation contracts

| Files | Purpose and verified contract |
| --- | --- |
| `src/learn/data/topics/algorithm-correctness-loop-invariants-termination.jsx` | Complete narrative, 10 teaching sections plus guided practice, checkpoints, three independent exercises with hidden explained solutions, annotated references and alternate learning routes |
| `src/learn/data/curriculum/blueprints/algorithm-correctness-loop-invariants-termination.js` | Individual authored outcomes, scope, sequence, sources and verification focus; global registration belongs to root |
| `src/learn/data/algorithm-correctness-examples.js` | 13 complete Python programs with exact expected output, including deliberate faults and explicit edge cases |
| `src/learn/data/algorithm-correctness-models.js` | Pure bounded search/proof-obligation, compaction, partition and Euclid models |
| `src/learn/components/lesson-labs/AlgorithmCorrectnessLabs.jsx` and `algorithm-correctness-labs.css` | Four distinct investigations and four inline figures; semantic topic-owned imports and scoped styles |
| `src/learn/data/practice/algorithm-correctness-loop-invariants-termination.js` | Eight selected official problems with topic-specific proof goals, hidden hints, prerequisites and changed-constraint transfer |

All visual quantities are exact synthetic fixtures. There are no measured-performance charts or invented benchmarks. The Euclid bar divides the current dividend into exact whole-multiple/remainder portions, explicitly rescales each state and handles a zero dividend as two zero-length contributions. The divisor set is recomputed; it is not a hard-coded answer. The original-array row and saved states are teaching allocations, excluded from the native compaction space claim.

Search and compaction accept at most eight integers from −20 to 20, including an empty list. Partition accepts up to eight category codes 0/1/2. Euclid's drawable input is bounded to 0–96 with at least one positive value; the native program separately defines gcd(0,0)=0. Limits are stated beside controls. Invalid edits preserve the last applied state. A faulty partition that crosses boundaries is explicitly labeled as an invalid region, not a negative number of occurrences.

## Native and independent model evidence

Command: `node scripts/verify-algorithm-correctness.mjs`. It invokes `scripts/verify-algorithm-correctness-native.py` using `scratch/lesson-tools/Scripts/python.exe`. Final pass: **2026-09-10 09:37:20 UTC**, Python **3.12.14**. Machine evidence: `scratch/algorithm-correctness-verification/results.json`; extracted exact programs and serialized model fixtures are in that directory and are recreated by the command.

| Check | Actual coverage and independent evidence |
| --- | --- |
| Complete programs | All 13 programs execute and match expected stdout, with CRLF normalized. Includes normal/faulty assertions, assignment, compaction, zero fill, category validation, gcd, resetting counters, recursive/iterative powers, modular powers, output certificates, finite counterexample discovery and integer square root |
| Search and proof obligations | 726 model traces. Python `list.index` supplies the return oracle; every boundary predicate and candidate proof obligation is independently enumerated, including skipped updates, hypothetical admitted states and absent/multiple matches |
| Compaction | 363 traces compared with direct filtering of original occurrence indices and exact native storage; prefix ordering/multiplicity, bounds and untouched suffix checked at every state |
| Partition | 2,186 traces, including correct and faulty updates. Native output compared with `sorted`; every region predicate and original occurrence identity checked. Correct traces consume exactly one unknown occurrence per iteration |
| Euclid | 628 model traces checked with `math.gcd`, independent divisor sets and native `divmod`; native gcd tested on all 9,409 ordered pairs from 0–96, including the all-zero convention |
| Powers | 297 exact recursive/iterative power cases against built-in `pow`; 1,584 modular cases against three-argument `pow`, including negative bases, zero exponent and modulus one |
| Resetting counters | 81 rectangle cases against `itertools.product`, including empty rows/columns and strict lexicographic descent |
| Output certificates | 14,641 input/candidate combinations compared with the independent full sorted-list specification; repeated values distinguish a multiset from a set |
| Additional complete functions | 121 zero-filling arrays; 10,010 integer-square-root cases against `math.isqrt`, including large perfect squares and neighbors beyond the exact binary64 integer range |
| Invalid and debug behavior | 15 invalid model cases and 10 native invalid groups; invalid category validation precedes mutation. A separate optimized Python run confirms `assert` checks are omitted under `-O` |

These are finite executed checks and independent reference comparisons. They supplement the lesson's general arguments; none is represented as a universal formal-verification result. Cost explanations distinguish arithmetic-operation counts, bit costs, original algorithm state and diagnostic/visual storage.

## Browser, reading and accessibility review

Command: `node scripts/review-algorithm-correctness.cjs`, Playwright with installed Microsoft Edge against shared Vite `http://127.0.0.1:5173`. It uses isolated 1440×1000 and 390×1000 contexts with reduced motion and closes only its own development websocket to avoid unrelated authors' HMR during a trace.

The final run passed at **2026-09-10 09:38:51 UTC** on the 09:37:14 frozen source. Per viewport it checked 17 search, 21 partition and 15 Euclid states; 25 desktop and 34 mobile compaction states include the additional local-scroll case. All four Previous/Finish/Reset keyboard sequences, 11 anchors, eight practice links and 13 rendered example/output pairs passed.

- Every displayed action/state in selected search, compaction, partition and Euclid cases matches the already independently checked model. Empty, all-kept/removed/equal, swapped-order gcd, zero-first/second, missing targets, repeated values and both deliberate bugs are exercised.
- Apply/invalid input/reset, candidate selection and fault toggles work without retaining a stale state. Enter/Space operate controls and hidden hints; eight-occurrence mobile strips support focused ArrowRight scrolling. Text labels accompany color and pointer highlights.
- All 11 route anchors have exactly one target. All eight official practice links use new-tab behavior with `noopener noreferrer`; optional content and hints begin closed. All 13 rendered code/output pairs match the native fixtures, including examples inside disclosures.
- Ordinary closed-disclosure reading screenshots preserve site navigation. H2 starts, proof notation, independent exercises and references are inspected in addition to isolated component captures. The four static figures and all four operated investigations are opened at desktop/narrow widths; normal typography and useful legibility matter in addition to overflow assertions.
- No page overflow, no lab/figure overflow and no uncaught page errors in the passing run. Wider occurrence strips scroll locally; full code uses the existing code component's local overflow behavior.

Corrections found during actual review: a topic-owned practice readiness string was changed to the array expected by the shared renderer; mobile input widths were contained within their own labeled columns; cramped prose word/number joins were spaced; an overlong Unicode superscript was replaced by introduced B/E notation with semantic superscripts; a long cell label was shortened to “unknown.” Browser test selectors were corrected to match the shared component's accessible hint text and code container rather than assuming a `pre` element. The final run retains code/output comparisons rather than dropping the check.

Evidence: `scratch/algorithm-correctness-browser/results.json`, plus `reading-*-1440.png`, `reading-*-390.png`, `inline-0` through `inline-3`, search first/skipped/empty, compaction occurrence/finished/empty/long, partition correct/skipped/regions/empty, Euclid remainder/finished/zero-first and practice-hint captures. These screenshots are generated by the checked-in browser command; they are not a substitute for its assertions.

## Source, external resources and independent review

The [design research ledger](ALGORITHM-CORRECTNESS-LESSON-DESIGN.md#primary-research-ledger) records the actual primary-source scope inspected on 10 September 2026: Cornell CS2112 and CS2110, Software Foundations Hoare Logic, Dafny termination, Python assignment/assert/gcd documentation and Cornell David Gries' video page plus written companion. The lesson links the Cornell array-diagram script as an annotated alternative. Full video playback, Rocq/Dafny execution and LeetCode judge submission were not performed or claimed. The inspected constant-replacement companion PDF has a loop-bound typo; that code was not adopted, and the recommended written link is the separate array-generalization script.

Official statement metadata inspected: [27 Remove Element](https://leetcode.com/problems/remove-element/) Easy, [283 Move Zeroes](https://leetcode.com/problems/move-zeroes/) Easy, [75 Sort Colors](https://leetcode.com/problems/sort-colors/) Medium, [88 Merge Sorted Array](https://leetcode.com/problems/merge-sorted-array/) Easy, [69 Sqrt(x)](https://leetcode.com/problems/sqrtx/) Easy, [206 Reverse Linked List](https://leetcode.com/problems/reverse-linked-list/) Easy, [100 Same Tree](https://leetcode.com/problems/same-tree/) Easy, and optional [50 Pow(x,n)](https://leetcode.com/problems/powx-n/) Medium. Public statement access is distinguished from account/editorial limits. Stable order, spare capacity, node identity, structural equality and signed/float exponent differences are called out explicitly. This selection is reasoned proof transfer, not a fixed quota or a guarantee of solving every interview problem.

Root independently reviewed the full lesson, search obligation enumeration, mutation/occurrence model, Euclid and integer examples, including lexicographic/recursive/modular reasoning. Root reported no concrete mathematical or proof-contract defect. This is a source review, not additional numerical-test coverage.

`node scripts/format-algorithm-correctness.cjs` formatted four owned sources and compared normalized Babel ASTs, comments and string/JSX values before/after. Final formatting evidence: `scratch/algorithm-correctness-verification/formatting-results.json`, **09:37:14 UTC**. The Python fixtures were preserved. Topic code imports only its own models/examples/practice plus small shared teaching UI; no aggregate lesson registry or math renderer is introduced. Root performs the integrated build and loading checks separately.

## Final status and boundaries

Owned semantic source last write: **2026-09-10 09:37:14 UTC**. Native verification, root source review and final browser review are complete. Final browser evidence timestamp: **2026-09-10 09:38:51 UTC**. Opened representative ordinary-reading screenshots, all four static figures at both widths and all four operated investigations; no remaining visible issue identified.

No unresolved destination note was discovered requiring a title change or another topic's body rewrite. Sequential deterministic reasoning is the taught scope. Concurrent interference, probabilistic termination and a full proof-assistant course are explicitly distinguished; no claim of those broader guarantees is made. User acceptance and root's combined production integration remain separate stages.
