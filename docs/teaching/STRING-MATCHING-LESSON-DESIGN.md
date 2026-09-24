# String matching: design and research

Author review complete, 10 September 2026; see STRING-MATCHING-VERIFICATION.md for actual final evidence. Stable ID: `string-matching-prefix-functions-rolling-hashes`, DSA position 16. This is a new lesson; no previous body exists. Root owns registration and integrated review. Implementation and user acceptance are separate.

## Scope and continuity

Retain **String Matching, Prefix Functions & Rolling Hashes**. The title already accommodates exact prefix-based search, verified fingerprints and useful string applications. Arrays supplies sequence indexing; Complexity and Correctness supply accounting/invariants; Hashing supplies collision-versus-identity reasoning. Explain borders, modular rolling arithmetic and string units locally. Tries index dictionary prefixes; this lesson searches contiguous occurrences inside a text. No trie prerequisite is hidden in KMP.

Run of the exact topic-plan command and progress ledger confirmed position 16, planned body and existing generic brief. No destination note is currently addressed to this ID. The bit-manipulation inbox is not implicitly assigned here.

| Idea | Existing coverage / owner | Decision and reason |
| --- | --- | --- |
| Exact matching, overlap and empty patterns | Generic brief, no body | Core contract: zero-based code-point offsets, all overlapping starts; empty pattern matches every boundary |
| Prefix functions and KMP | Explicit title/brief | Core: build the table, prove fallback and total comparison work, preserve state across chunks |
| Rolling/prefix fingerprints | Hashing has collision foundation | Core here: derive updates, verify candidates, distinguish fixed toy parameters from randomized guarantees and output verification cost |
| Periods and borders | Natural consequences of prefix table | Include shortest shift versus whole-string tiling, with divisibility counterexample |
| Z matching | Not taught by a nearby lesson; same exact-prefix evidence | Include an optional fully supported branch, complete code and box-reuse illustration; use half-open boundaries, not the source's one-based inclusive convention |
| DNA windows and shortest prepend-palindrome | Specific transfer of rolling encoding/prefix sufficiency | Include exact restricted-alphabet encoding and unique sentinel construction with proofs and native fixtures; do not imply biological significance or grapheme-aware reversal |
| Approximate edits / multi-pattern and text indexes | DP edit-state family / deeper string specialization | Explain changed query contracts; do not pretend KMP implements fuzzy search or all indexing algorithms. Preserve a reasoned advanced owner note if needed during writing |

Core finish line: derive and implement exact all-match KMP, retain a prefix state across chunk boundaries, construct rolling updates with collision verification and state honest costs. Deeper finish line: use prefix structure for borders/periods/palindromes, distinguish Z's starting-position evidence and reason about hash error assumptions. Reading and practice times are estimated after content exists.

## Hurdles, representations and truth contracts

| Hurdle | Representation / placement | Contract, prediction and independent evidence |
| --- | --- | --- |
| What counts as an occurrence? | Indexed text and two offset pattern ribbons in section 1 | Equal-width cells represent code points, not bytes or duration. Highlight overlapping starts of `ababa` / `aba`; show the n+1 empty boundaries in prose. Brute-force substring oracle |
| A suffix can reuse a prefix | Prefix/border builder beside definition | Edit one small pattern, apply explicitly, step comparisons/fallback/table commits. Highlight both equal regions; pi is a length, never a text index. Predict a fallback, inspect known table prefix; exhaustive proper-border oracle |
| A mismatch does not consume the current character | Aligned KMP search tape | State includes consumed text, matched prefix, comparison, reported starts. Back/next/reset and preserved overlap fallback. Input applies atomically; invalid bounds retain last valid trace. Compare all output positions and state evidence with slicing, not another KMP |
| Network chunk boundaries are not semantic boundaries | Chunk strip with retained-state ribbon | Feed whole chunks and inspect prefix length/new global offsets. Fault mode clears state at seams. Empty chunks consume no symbols; reset restores initial state. Independent concatenate-and-slice oracle over many partitions; clearly simulated decoded strings |
| A numeric equality is a candidate | Rolling subtraction/shift/add balance plus candidate window ledger | Same pattern/window length; code-point ord values, small fixed base/modulus labeled teaching choices. Show hash equality separately from exact equality; expose a real collision. Every row recomputed independently from polynomial sums |
| Unicode changes the coordinate system | One displayed string, three indexed unit rows and normalization before/after | Calculated UTF-8 bytes, Unicode code points and UTF-16 code units, not decorative lengths. Preserve combining marks with U+ labels; normalized offsets explicitly refer to transformed text. Native encode/normalize checks |
| Z evidence has a boundary | Two computed box examples in optional Z section | Half-open [L,R), copied prefix evidence capped at R−i, comparison beyond R visibly distinguished. No fabricated running-time curve. Brute-force Z oracle |

Each lab has a visible default meaningful state, instructions before interaction, local readable overflow with keyboard access, descriptive controls and live textual action feedback. Color supplements labels. No timed animation or generic lab-count requirement. Inspect ordinary reading as well as changed states at 1440 and 390 px; open resulting screenshots and verify actual anchor arrivals.

## Complete examples and assessment

Independent Python programs cover brute force; prefix construction; all-match KMP; streaming; borders/periods; verified rolling search; prefix substring fingerprints; Unicode coordinate changes; exact fixed-length DNA encoding; shortest prepend-palindrome; optional Z search. Programs duplicate only necessary definitions so each is executable from an empty file. Explain cost, output and a meaningful failure case beside each.

Local practice must require table construction/fallback diagnosis, overlap and empty cases, streaming partition invariance, collision/verification-cost counterexamples, divisibility and Unicode offset decisions, plus an independently testable detector capstone. Hints precede separately hidden explained answers. Official LeetCode set follows mechanisms, not a problem quota: 28, 459, 1392, 686, 187 and optional 214. Platform Hard on 1392 does not imply a hidden advanced prerequisite; it is direct transfer once borders are taught.

## Research ledger (inspected 10 September 2026)

All explanations, fixtures and figures are original. Source conventions are checked, not copied blindly.

| Source | Actual inspected scope / use / qualification |
| --- | --- |
| [Chvátal, KMP lecture notes](https://users.encs.concordia.ca/~chvatal/notes/kmp.html) | Full written page: fallback as retained suffix, comparison potential, optimized next[] versus simpler variant. Our zero-based pi length convention differs; do not paste the optimized table into our code |
| [Princeton substring search](https://algs4.cs.princeton.edu/53substring/) | Written overview and border/period/all-match exercises. Page says under construction; use as optional broader Java-oriented reference, not sole proof authority |
| [MIT 6.006 lecture 9 typed notes](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/160b3b5f9da2e03815ca1e6ee0dba62a_MIT6_006F11_lec09.pdf) | Parsed pages 3–5: rolling remove/append and verify-on-hash-hit. Its brief pseudocode has an endpoint omission; our baseline explicitly includes n−m. We independently derive all-hit verification costs and do not inherit an unconditional expected-linear claim |
| [MIT lecture 9 video resource](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/resources/lecture-9-table-doubling-karp-rabin/) | Official resource title/description and corresponding notes inspected; video not watched. Annotate first portion revisits resizing; later rolling-hash explanation is relevant. Do not invent timestamps |
| [CMU, String Matching Z](https://www.cs.cmu.edu/~ckingsf/bioinfo-lectures/zalg.pdf) | All 11 parsed slides: prefix-at-start definition, separator construction, two box cases, linear accounting. Original 1-based inclusive coordinates translated explicitly. Earlier syllabus links failed; working direct lecture URL retained |
| [Python Unicode HOWTO](https://docs.python.org/3/howto/unicode.html) | Definitions/encodings/string support/comparing strings: code-point versus byte representations, normalization and casefold expansion. Current page reports Python 3.14.7; our native runtime is 3.12.14. No universal locale/caseless-search policy claimed |
| Official LeetCode [28](https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string/), [459](https://leetcode.com/problems/repeated-substring-pattern/), [1392](https://leetcode.com/problems/longest-happy-prefix/), [686](https://leetcode.com/problems/repeated-string-match/), [187](https://leetcode.com/problems/repeated-dna-sequences/), [214](https://leetcode.com/problems/shortest-palindrome/) | Full accessible statements, numbers/titles/difficulties and constraints read. 28/459/686 lowercase nonempty; 1392 excludes whole string and permits overlap; 187 length-10 A/C/G/T windows; 214 allows empty and only prepends. Editorial/submission access not verified. Original hints and transfer prompts, no copied statements |

## Verification plan

Native: execute every displayed program and exact stdout; independent exhaustive slicing/border/period/palindrome/DNA oracles, Unicode encodings, stream partitions and direct polynomial hashes. Models: exhaustive small alphabets, comparator/state invariants, finite bounds, edge cases and invalid input contracts; verify rolling collisions explicitly. Browser: actual model-linked controls, back/reset/invalid handling, code/stdout, source links, keyboard anchor arrivals, ordinary reading and all static/interactive figures. Record actual results and unresolved limitations in `STRING-MATCHING-VERIFICATION.md`; source-ready registration is not final review.

## Final scope discoveries and added immediate figures

The current title remains accurate. Added immediate paired border evidence and mirrored palindrome construction, plus weighted-prefix cancellation for xab, B=3,Q=101: H[3]=55, shifted H[1] contribution70 and remainder86. These figures sit next to their first derivations, use actual calculated values, and are inspected independently of interactive states. The KMP tape now uses one shared horizontal coordinate axis and an explicit end boundary so text/pattern alignment remains truthful on narrow screens.

Saved follow-up reasoning against [this topic's stable ID](topic-notes/string-matching-prefix-functions-rolling-hashes.md) for many-pattern/suffix-index ownership, and [Randomized Algorithms](topic-notes/randomized-algorithms-sampling-error-guarantees.md) for broader amplification/adversarial assumptions. No curriculum expansion or unrelated rewrite performed.
