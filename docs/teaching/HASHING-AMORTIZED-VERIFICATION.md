# Hashing, Collision Resolution & Amortized Analysis — verification

10 September 2026. Stable ID `hashing-collision-resolution-amortized-analysis`, DSA position 14. Authored, native/model checked and browser reviewed. The last owned runtime/brief source write is **2026-09-10 10:24:12 UTC**; those files are frozen after the final checks below. Root owns integrated production checks and shared completion status. Publication is not user acceptance.

## Teaching and ownership

The previous topic had an authoring brief but no lesson body. Read the exact inventory, current standards, full earlier Arrays/Hash Maps lesson and the incoming resize-threshold note before design. The new lesson preserves the first topic's foundations and develops deeper collision, deletion, rebuild, expectation and mixed-update arguments instead of duplicating its introduction.

- [Lesson design and research](HASHING-AMORTIZED-LESSON-DESIGN.md).
- [Resolved resize-threshold note and optional deeper-scheme routing](topic-notes/hashing-collision-resolution-amortized-analysis.md).
- Body: `src/learn/data/topics/hashing-collision-resolution-amortized-analysis.jsx`.
- Models and ten complete Python fixtures: `src/learn/data/hashing-amortized-models.js`, `src/learn/data/hashing-amortized-examples.js`.
- Topic-specific UI/style: `src/learn/components/lesson-labs/HashingAmortizedLabs.jsx`, `hashing-amortized-labs.css`.
- Selected-topic practice: `src/learn/data/practice/hashing-collision-resolution-amortized-analysis.js`; shared presentation follows [DSA practice policy](DSA-PRACTICE-STANDARD.md).
- Individual blueprint: `src/learn/data/curriculum/blueprints/hashing-collision-resolution-amortized-analysis.js`.

The route moves from a list-map baseline and identity to chains/probes, deletion evidence, rebuilding, expected collision work, amortized growth, mixed-update hysteresis and composite applications. Seven original exercises include hidden reasoning; five official problem links add changed-contract transfer. Complete programs cover list/chained/probe maps, intentional deletion failures, exact family enumeration, resizing, dense sampling, consecutive runs, pair-frequency counting and immutable composite keys. The general pair-counting function accepts a nonzero target as well as unequal/empty groups, matching the stated contract.

No global CSS, catalogue, other lesson bodies or shared runtime bundles were changed by this author. Only the selected lesson imports these helpers, examples and practice; no math-renderer dependency was introduced. The title and stable route are retained. Shortest Paths, Spanning Trees & Topological Ordering remains the next topic.

## Visual contracts and placement

| Representation | What the learner sees/changes | Truth contract and limits |
| --- | --- | --- |
| Collision layout, section 2 | Identical keys 1,9,17 in a chained bucket and successive probe slots | Exact modulo-8 fixture; full stored identities retained. Local slot strip can scroll on a narrow viewport. |
| Probe/deletion investigation, section 3 | Editable bounded script, per-probe states, EMPTY/DELETED/live slots, visited path, reference result and deliberate erase fault | At most 24 operations, capacities 4/8/16, integer values/keys −999…999. Native Python handles broader integers separately. Correct scans are bounded by capacity; updates continue past tombstones to rule out duplicates. A fault indicator is bounded diagnostic evidence, not a formal verifier. |
| Rehash locations, section 4 | Old and new key homes with changing modulus | Home is distinct from final physical destination. Preserve associations by reinsertion, not slot copying. |
| Expected/amortized axes, section 5 | Fixed-input randomness versus sequence-wide accounting | Explicitly separates two different averaging arguments. Does not claim per-operation worst-case constant time. |
| Exact hash-family investigation, section 5 | Hold a distinct key set fixed, change a/b/query and see all 272 outcomes plus candidate-length frequencies | p=17, m=4, a=1…16, b=0…16, keys 0…16. Every cell has equal probability. Candidate counts are not exact successful-search early-stop comparisons. The displayed mean is complete enumeration, with a separate pairwise proof. |
| Resize investigation, sections 6–7 | Compare paired operation-cost plots, then the same mixed stream under half and quarter shrink | Counts successful append/pop and retained-reference copies; allocation excluded from exact totals. Both plots share a vertical scale and preview the known sequence. Threshold inequalities/minimum capacity are explicit; not CPython internals or wall-clock benchmarks. |
| Dense set investigation, section 8 | Inspect array and reverse-index writes during middle/last/absent deletion | Bijection holds at operation boundaries; the temporary duplicate during movement is explained. Uniform selection follows from one slot per distinct value and a uniform index source, not a histogram. |

The finite affine-family collision theorem is applied to chaining, not silently to linear-probing cluster cost. Chained migration directly appends known-unique entries and stays linear under the stated key model; the fixed probe map's rebuild can require quadratic collision work. Append-only potential and mixed-update potential are separate derivations, with minimum-capacity cases and exact allocation exclusions. The mixed-boundary fixture resolves the incoming note using `n≤C/4`, adapting the earlier informal “below a quarter” wording deliberately.

## Native and model evidence

Run from the repository:

```text
node scripts/format-hashing-amortized.cjs
node scripts/verify-hashing-amortized.mjs
```

Final formatting record: `scratch/hashing-amortized-verification/formatting-results.json`, **10:24:12.026 UTC**. Babel parsing before/after confirms normalized AST and strings are preserved for topic, models, labs and individual blueprint; this is formatting evidence, not mathematical validation.

Final native/model result: `scratch/hashing-amortized-verification/results.json`, **10:24:16.161 UTC**. Python runtime **3.12.14**, `scratch/lesson-tools/Scripts/python.exe`; all ten displayed complete programs ran in isolated mode and matched every expected stdout fixture. Independent oracle implementation is durable in `scripts/verify-hashing-amortized-native.py`; generated model cases and executable fixture copies are in that scratch evidence directory.

| Checked behavior | Actual scope | Independent evidence |
| --- | --- | --- |
| Probe maps | 312 scripts, 7,263 committed operations and 37,480 complete/intermediate states | Python dict supplies logical behavior; native probe slots match rendered models; independent scans verify no EMPTY precedes a live key, unique keys and bounded distinct visited positions. Wrapped, negative, full, replacement, absent, tombstone and rejected-rebuild cases included. |
| Native map interfaces | 15,000 additional operations across list, chained and probe maps | Dict oracle on collision-heavy keys, None values, updates/deletes and grow/drain sequences. Every retained mapping is queried after each operation. |
| Rejected native contracts | 29 cases | Invalid capacity, invalid integer-key inputs for chained/probe methods and empty sampling raise the promised exception. |
| Exact hash family | 136 fixed-set/query cases across sizes 1–8 | Python independently enumerates affine residues. Rational means equal `1+(n−1)×7/34`; each other-key collision count is 56/272, selected lengths/distributions and the bound agree. |
| Resize work | 3,099 sequences / 25,168 operation states | Separate actual-storage simulator recomputes copied values, length/capacity and totals. All short binary update streams through length 9, larger append prefixes, growth-by-one and named mixed/drain fixtures. Piecewise nonnegative potential and the appropriate amortized inequalities checked where applicable. |
| Dense collection | 45 model cases + 2,000 native updates | Independent set oracle and array/reverse-index bijection checks; middle, last, absent and empty states. Samples checked for membership; uniformity is justified analytically, not claimed from these samples. |
| Consecutive runs | 5,461 inputs | Sorted-distinct oracle, including duplicate/empty/negative inputs; counted membership operations bounded by three times the number of distinct values. |
| Four-group counting | 1,875 cases | Exhaustive Cartesian quadruples across duplicate, empty and unequal groups, with targets −1/0/1. |
| Bounded UI validation/fault | 21 rejected model inputs and intentional incorrect-delete trace | Bad syntax/ranges/capacity/duplicates/unsupported policies rejected; the deliberate erase fault produces both unreachable entries and a wrong reference result. |

These finite tests substantiate examples and models. The stated invariants and proofs establish the general mathematical reasoning under their assumptions; test counts are not a universal correctness proof or an interview guarantee.

## Browser and opened-image evidence

```text
node scripts/review-hashing-amortized.cjs
node scripts/review-hashing-amortized-anchors.cjs
```

Final record: `scratch/hashing-amortized-browser/results.json`, **10:25:29.506 UTC**, actual Vite route on `127.0.0.1:5173`, headless Microsoft Edge through Playwright. Desktop **1440×1000** and mobile **390×1000** both passed:

- 134 probe states, three finite-family input/parameter cases, 83 resize states and eight dense-set states per viewport. Model readouts and physical entries checked, not just button availability.
- Previous/Next/Finish/Reset operated with keyboard; reset restores default inputs. Fault on/off, empty script, all-live table, tombstone reuse, wraparound and invalid input preserving the previous applied state checked. Mobile active-probe visibility and keyboard scrolling of the local slot strip passed.
- All 11 route anchors resolve; five official problem links open in a new tab. A hint starts closed and opens with keyboard. All ten complete rendered programs and outputs match their native fixtures after disclosure.
- Eleven normal reading-section screenshots plus the introduction preserve ordinary site navigation and closed disclosures. Final document overflow is false and no lab/inline figure overflows its own box on either viewport.
- No uncaught page errors. The test closes HMR websockets so other authors' live changes cannot reset the investigation during checks.

The separate final anchor-arrival check at **10:32:56.970 UTC** goes beyond ID existence: every one of the 11 intro links was focused and activated with Enter at both widths, the URL hash changed, and the complete destination heading arrived below the fixed header (approximately 100px from the viewport top; practice approximately 90px). Evidence: `scratch/hashing-amortized-browser/anchor-arrivals.json`. No runtime source change was needed.

Actual images opened and read include `family-exact-1440/390.png`, `resize-thrashing-390.png`, `resize-geometric-1440.png`, `probe-failure-1440.png`, `probe-wrap-390.png`, `dense-during-move-390.png`, all three `inline-*-390.png`, `potential-mixed-proof-390.png`, and normal `reading-intro-390.png`, `reading-2/4/5-390.png`, `reading-6/7-1440.png`. The final mixed-resize picture clearly shows 159 versus 39 counted units, labels fit, the selected-cost readout no longer overlaps the tallest bars and the proof's cases remain readable at 390 pixels. The family lattice and frequency bars use consistent colors plus numerical counts; correctness does not depend on color alone.

Issues found and resolved before final freeze: implicit label names included control content, so explicit matching accessible names were added; long policy selections caused real narrow-screen overflow, so option wording was shortened and those controls stack on mobile; selected slot scrolling was added for long/wrapped tables; a sampling probability caption now refers to the final representation rather than the temporary duplicate state. An early verifier used incorrect helper names, which was corrected in the harness. The pair-count fixture was generalized to match the prose's target parameter and tested independently. Initial failed runs are not represented as passes.

## Research and completion boundaries

The design records the actual inspected primary written scope: MIT 2020 hashing/dynamic-array notes, Open Data Structures chaining/probing sections, Python data-model/random contracts, historical PEP 456 motivation, MIT 2011 probing/coverage notes and CMU's displacement-scheme sections. Official video resource pages were checked and relevant written companions inspected; full video playback was not claimed. Dated cryptographic/password guidance, hardware timings and current implementation rankings were not adopted.

Official public LeetCode statements verified for ID/title/difficulty and constraints: **705 Design HashSet (Easy), 706 Design HashMap (Easy), 380 Insert Delete GetRandom O(1) (Medium), 128 Longest Consecutive Sequence (Medium), 454 4Sum II (Medium)**. The dataset explains bounded-universe direct addressing versus the sparse-key extension, absence sentinels, exact sampling, duplicate run starts and pair multiplicities. No account submission, paid editorial inspection or guarantee of universal interview ability is claimed.

The completed scope is an exact local map with explicit assumptions and transferable analysis. Full alternative collision schemes, cryptographic design, distributed placement and probabilistic filters are not falsely represented as implemented here. The scoped note preserves a reasoned optional extension and requires a later author to reassess ownership and usefulness. Root's shared production integration and the user's review remain separate from this lesson's completed authoring verification.
