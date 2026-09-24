# String Matching: implementation and verification

Author review complete on 10 September 2026. Stable ID `string-matching-prefix-functions-rolling-hashes`, DSA position16. Final lesson/model/example/lab/style/brief/practice source freeze: **2026-09-10 11:27:40.195 UTC**. Root registered the body and individual brief; integrated build/review is root-owned and separate. User acceptance remains pending.

## What changed and why

The previously planned topic now has a complete mechanism-first lesson. It begins with an exact all-occurrence contract and an independently useful naive baseline, then develops borders, prefix-table construction, KMP fallback/overlap, decoded-stream state, period and palindrome reductions, verified rolling/prefix fingerprints, random-base qualifications, Unicode coordinates, exact DNA window codes and an optional complete Z branch. The title and stable ID are preserved. No previous lesson body existed to remove or replace.

Four focused investigations expose different operations: construct a border, move an aligned candidate without consuming the unread symbol, deliver a chunk while retaining state, and roll/verify a fingerprint. Six immediate figures cover overlapping occurrences, paired border evidence, palindrome mirroring, weighted prefix cancellation, Unicode unit representations and Z-box reuse. These counts describe the implemented hurdles, not a teaching quota. All six inline figures present fixed examples; the two Z states use values from the actual Z function, with fixed reviewed inputs.

Eleven complete Python programs have displayed results and nearby interpretation/cost qualifications. Seven local exercises have separate optional hints and explained answers, including a stream-detector capstone and changed-input diagnoses. Six official LeetCode tasks are staged by mechanism, with original hints, transfer, prerequisites and readiness advice. The platform list does not replace the proof, Unicode or collision exercises and makes no universal interview promise.

The [design/research record](STRING-MATCHING-LESSON-DESIGN.md) owns detailed scope, contracts and actual source inspection. Saved notes preserve [deeper text-index ownership reasoning](topic-notes/string-matching-prefix-functions-rolling-hashes.md) and [randomized amplification/contract transfer](topic-notes/randomized-algorithms-sampling-error-guarantees.md). These are clearly scoped future decisions; no unrelated catalogue expansion was performed.

## Source ownership

- Lesson: `src/learn/data/topics/string-matching-prefix-functions-rolling-hashes.jsx`.
- Pure models and complete programs: `src/learn/data/string-matching-models.js`, `string-matching-examples.js`.
- Topic UI/style: `src/learn/components/lesson-labs/StringMatchingLabs.jsx`, `string-matching-labs.css`.
- Individual brief: `src/learn/data/curriculum/blueprints/string-matching-prefix-functions-rolling-hashes.js`.
- Selected-topic practice data: `src/learn/data/practice/string-matching-prefix-functions-rolling-hashes.js`.

The lesson imports only its own model/example/practice graph and semantic shared teaching UI. It adds no math renderer, aggregate topic bundle or global style. `scripts/format-string-matching.cjs` conventionally formatted the seven owned runtime/brief/style files while asserting JavaScript AST/string preservation and CSS semantic preservation. Final formatting evidence: `scratch/string-matching-verification/formatting-results.json`, 11:27:40.195 UTC.

## Independent computational verification

Command: `node scripts/verify-string-matching.mjs`. It executes each complete program with the workspace Python3.12.14 using `-X utf8 -I`, then runs `scripts/verify-string-matching-native.py`. No third-party Python package is needed for this lesson. Final pass: **11:27:42.405 UTC**, `scratch/string-matching-verification/results.json`.

| Check | Actual coverage / independent oracle |
| --- | --- |
| Displayed programs | All11 executed from fresh files; stdout equals the stored displayed result, including emoji and combining-mark fixtures |
| Prefix model | 515 patterns, 10,804 trace states; direct enumeration of every proper prefix/suffix equality checks every committed table entry and highlighted border |
| KMP model | 8,772 search cases, 90,908 states; direct slicing/equality checks the reported set, retained prefix equality, monotone consumed position, unchanged consumption on fallback and total comparison bound |
| Rolling model | 26,316 cases, 100,110 windows; direct BigInt polynomial sums independently check every fingerprint and next-window update. Candidates and exact results are checked separately |
| Stream model | 10,245 partitions with inserted empty chunks, checked against concatenated-input slice results after every feed; reset-at-seam counterexample deliberately fails while correct state reports start0 |
| Z model | 515 patterns checked against naive longest-common-prefix comparisons at every starting position |
| Invalid model inputs | Seven cases: oversized drawings/chunk lists, invalid base/modulus/type and nonstring patterns. Previous valid UI states are preserved on rejected edits |
| Native exact search | 8,514 cases across baseline, KMP, Z and two rolling parameter sets, compared with direct Python slicing |
| Native structure | 1,096 prefix, border-chain, shortest-period and whole-copy decisions, checked by exhaustive candidate lengths/shifts. 1,096 prepend-palindromes checked by enumerating all permitted prepend lengths |
| Native streams | 10,245 partitions; collected per-feed reports match the independent whole-prefix oracle, including the empty-boundary contract |
| Native DNA | 8,190 input/width cases compared with direct substring deduplication, plus five invalid alphabet/width cases |
| Native substring queries | 5,660 valid interval queries compared with direct polynomial powers, plus out-of-range/reversed rejection |
| Probability illustration | 780 unequal fixed pairs, lengths1–3 over digits0–2, prime17 and bases2–16; actual collision-base count never exceeds degree. This corroborates finite cases; the polynomial-root proof supplies the general argument |
| Exact hand fixtures | π(aabaaab)=[0,1,0,1,2,2,3], changed final c→0; cabca→acbacabca; raw Unicode [2,4)↔UTF-8 [5,8); Z values at5 and9; prefix-cancellation H[3]=55, shifted contribution70 and remainder86 |

The initial harness run exposed that Python isolated mode ignores `PYTHONIOENCODING`, so redirected Windows output used cp1252 and could not print the emoji. The verifier now explicitly passes `-X utf8`; the lesson's executable command does too. This was an output-environment repair, not a changed matching algorithm.

## Actual browser and visual review

Shared Vite5173 route reviewed with real Playwright/Edge at **1440×1000 and 390×1000**, with keyboard interactions and reduced motion. HMR sockets were closed in these isolated review pages so another author's edits could not reset traces.

Command: `node scripts/review-string-matching.cjs`. Final pass **11:28:14.034 UTC**, `scratch/string-matching-browser/results.json`.

- At each width: 65 prefix states, 121 KMP states, 21 stream states and12 rolling windows compared with the pure model's actual values; repetitive/fallback/Unicode/empty/longer-pattern/full-text cases included.
- Every KMP retained text cell aligns with its matching pattern cell on a **single shared horizontal axis**. Narrow views follow the current unread symbol; an explicit end-boundary cell remains visible when text is exhausted. The earlier independent scrollers were corrected before final review because they could imply a false alignment.
- Apply/invalid input behavior, resets, previous steps and the deliberate seam-reset fault exercised. Base/modulus controls have explicit accessible names after the first harness run revealed ambiguity in their implicit labels.
- All12 intro links activated with keyboard Enter. The URL hash changes to the intended heading and its actual arrival is visible at approximately100px below the header (practice approximately90px). These are genuine navigation arrivals, not merely an assertion that IDs exist.
- All11 rendered code and stdout bodies compared with the fixtures. Six LeetCode links retain official metadata and open a new tab; hints/extension are closed initially. Local answer reveal and the optional Z branch operate by keyboard.
- Six figures captured at both widths. No page errors, document overflow or offscreen lesson panels. Long position tapes and complete code use deliberate local scrolling rather than shrinking symbols or truncating algorithms.

Additional narrow command: `node scripts/review-string-matching-reading.cjs`. Pass **11:30:23.025 UTC**, `scratch/string-matching-browser/reading-controls-results.json`. Keyboard Finish→Previous→Reset checked in every lab at both widths. Captured and inspected the full intro and references; verified12 route links and six annotated written/video resource links.

**Images actually opened and inspected**, not only saved: prefix-fallback390; KMP-fallback1440/390, overlap390 and long-text390 (including final end-boundary repair); rolling-collision390 and exact1440; stream-correct390 and fault1440; border-evidence390; palindrome390; Unicode390; prefix-cancellation390; Z-reuse390/1440; overlap1440; normal reading sections1/4/6 at390 and3/8/9 at1440; intro390 and sources390. Images are under `scratch/string-matching-browser/`. The ordinary reading review confirmed that definitions and immediate figures precede interaction-dependent details, optional Z depth stays disclosed separately, and source annotations are readable on narrow screens.

## Research, boundaries and remaining review

Official LeetCode statements28/459/1392/686/187/214 were actually read on10 September2026, including difficulty, allowed alphabet, empty policies, overlap and prepend-only constraints. Primary/authoritative written resources inspected: Chvátal's KMP notes, Princeton substring-search material, MIT rolling-hash typed notes, CMU Z slides and Python Unicode HOWTO. The official MIT video resource/description and corresponding notes were reviewed; the video itself was **not watched** and no timestamp claims were invented. The lesson says this beside the resource.

The labs are bounded deterministic teaching models, not networking, normalization services, biological interpretation or performance benchmarks. Their calculated counts and arithmetic do not imply measured implementation rankings. Exact verified hashes can still do quadratic all-hit work; the random-base proof explicitly requires fixed inputs and a specified random family. Python arbitrary-width integer cost, copied outputs, code-point versus byte/grapheme coordinates and normalization-to-source mapping are kept explicit.

There is no known remaining core correctness, model, reading or interaction blocker from this author review. No novice-user study, assistive-technology user study, production benchmark, integrated build or user acceptance is claimed here. Root handles integrated publication/navigation/loading verification; local screenshots can be regenerated by the saved scripts. Durable compact evidence and source hashes: [string-matching-author-review.json](evidence/string-matching-author-review.json).
