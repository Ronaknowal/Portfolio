# Hashing, Collision Resolution & Amortized Analysis — design

10 September 2026. Stable ID `hashing-collision-resolution-amortized-analysis`, DSA position 14. Root owns registration and shared integration. This topic has a brief but no published/authored body. Read the exact topic inventory, incoming resize-threshold note, current standards and the complete earlier Arrays, Strings & Hash Maps source before designing this continuation. Retain its existing foundation; do not rewrite it.

## Scope and finish line

The learner can build and diagnose a finite map despite collisions; explain the information carried by an empty versus deleted probe slot; preserve mapping identity through capacity changes; distinguish load from distribution and live from used occupancy; prove geometric resize accounting and recognize shrink thrashing; state what expectation ranges over; and combine a map with another structure under explicit invariants and cost assumptions.

Keep the title and identity. Teach exact integer-key structures with explicit equality and absence contracts. The earlier topic introduced direct lookup, chains, hash/equality rules, insertion-order semantics and informal append amortization. This lesson develops the missing mechanisms and proofs. Correctness immediately before this topic supplies invariant/termination habits. Shortest Paths, Spanning Trees & Topological Ordering remains next.

| Hurdle | Representation / action | Narrative and independent transfer |
| --- | --- | --- |
| A hash is mistaken for identity or direct storage | Same keys through home buckets and probe slots; full stored key/value beside the calculated home | Compare direct addressing, a list baseline and collision-resolving maps; missing marker is distinct from a legitimate value |
| A hole seems to prove absence after deletion | Editable operation script, one-probe trace, EMPTY/DELETED/OCCUPIED slots and exact lookup path; optional incorrect erase | Prove no EMPTY occurs before a live key in its probe sequence; bounded full-table search and duplicate update beyond a tombstone |
| Resizing is treated as copying physical slots | Old/new home-location picture for changing capacity; native rebuild discards tombstones and reinserts entries | Preserve the abstract mapping rather than old slot positions; rebuild same size to clean tombstones; do not use a growth threshold that counts only live entries |
| Good-looking bucket balance is called a guarantee | Fixed key set across the complete tiny affine hash family; selected parameters, bucket layout and exact outcome distribution | Explain randomness over a hash choice, collision indicators, expected chain length and the fixed-input/nonadaptive assumption; pairwise collision bounds do not automatically establish linear-probing bounds |
| A rare expensive step contradicts a constant claim | Exact per-operation work/cumulative-cost timeline, geometric growth and visibly stored analysis credit | Aggregate geometric copies and a potential/credit argument; worst individual latency is still linear |
| Insert/delete repeatedly undo one another's resizing | Same mixed operation stream under half-occupancy shrink versus quarter-occupancy shrink | Resolve incoming note with exact thresholds/minimum capacity and copied-element accounting; explain the separation before naming hysteresis |
| O(1) is attached to a whole algorithm without checking its components | Dense array plus index map before/after swap-delete; exact reverse-location relationships | Complete randomized-set mechanism, uniform index argument, expected map plus amortized array costs; stable order is sacrificed |
| A membership loop appears quadratic from indentation | Maximal consecutive runs and unique starting points | Charge each distinct value to one run, preserve duplicate/empty behavior, compare a sort-based independent oracle |

Visual count is driven by these hurdles. Use compact inline pictures for hash-to-location and rehashing; larger investigations only where input changes or stepping exposes a mechanism. Charts are calculated work counts under declared models, not benchmark timings or empirical claims about Python internals. JS models remain tiny and bounded; native examples can support broader integer inputs.

## Teaching route

1. Mapping contract and a correct list/direct-address baseline; full keys, collisions and controlled costs.
2. Chaining versus open addressing; probe sequences and primary clusters.
3. Deletion, stopping conditions, tombstones, duplicate replacement and table exhaustion.
4. Rebuild/rehash, live versus used load, space and migration boundaries; complete implementations.
5. Worst, expected, amortized and expected-amortized as different statements; tiny exact hash-family experiment and indicator proof.
6. Geometric growth, aggregate/potential accounting, fixed-increment counterexample and mixed-update resize hysteresis.
7. Composite operations and applications: dense random-access set, maximal consecutive runs, pair-sum counting and exact keys.
8. Practical hash contracts and deeper alternatives: immutable equality fields, string/byte cost, randomized process hashes, threat versus distribution, probing strategies, latency and alternative structures. Routing prevents pretending that cryptography/consistent hashing/rolling hashes are the same topic.
9. Independent implementation/proof/counterexample tasks, hidden explanations, staged official practice and annotated alternate resources.

## Native implementation and model contracts

- A fixed-capacity linear-probe map uses distinct EMPTY/DELETED sentinels, bounded m-position scans, remembering the first tombstone but continuing to rule out an existing equal key. It supports explicit rebuild to adequate capacity, rejects invalid requests before mutation and distinguishes stored values from absent entries.
- A separate resizing chained map keeps rebuilding linear even for adversarial collisions by migrating known-unique entries directly into new buckets rather than repeatedly searching for duplicates. Its basic deterministic modulo hash is illustrative, not an expected-time guarantee. Equality-scan costs and rebuild costs remain separate.
- The resize simulator counts insertion/removal work and existing-element copies. Allocate/initialize costs are explicitly excluded from exact plotted totals; including linear allocation changes constants, not the geometric conclusion. Define the minimum capacity and every grow/shrink inequality.
- A p=17, m=4 hash-family experiment enumerates all 16×17 parameter pairs for fixed distinct keys in 0–16. It calculates a selected query's chain length and collisions exactly. Changing the key set redefines the fixed problem; it does not pretend the simple theorem protects adaptive adversaries.
- Native randomized-set code can use Python random.choice with a local seeded generator for a reproducible example, but uniform-index correctness is an argument about the sampling contract rather than a histogram proof. Randomized selection is distinct from random hash choice.

## Research inspected and boundaries

Primary sources read 10 September 2026:

- [MIT 6.006 Lecture 4 notes](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/ce9e94705b914598ce78a00a70a1f734_MIT6_006S20_lec4.pdf), pages 2–4: dictionary/direct access, chains, fixed modulo counterexamples, affine universal family and indicator expectation. Use original fixtures/derivations; the historical Python-layout comments are not current implementation claims.
- [Open Data Structures 5.2](https://opendatastructures.org/ods-python/5_2_LinearHashTable_Linear_.html): three slot states, used/live counts, probe stopping and assumptions for cluster analysis. Its exact capacity thresholds and code are not copied. [5.1](https://opendatastructures.org/ods-python/5_1_ChainedHashTable_Hashin.html) supplies the adjacent chaining/analysis route. Section 5.2 explicitly cautions against treating the next probe as an independent uniform sample.
- [MIT Lecture 2 notes](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/79a07dc1cb47d76dae2ffedc701e3d2b_MIT6_006S20_lec2.pdf), page 4: dynamic-array deletion requires separated resize thresholds. This resolves the incoming concrete 8↔16 thrashing proposal through an original exact simulator and native checks.
- [Python data model](https://docs.python.org/3/reference/datamodel.html#object.__hash__), current 3.14.7: equal-key hash contract, mutation and default str/bytes hash randomization; native examples use the installed runtime separately. [PEP 456](https://peps.python.org/pep-0456/) is historical motivation for keyed defenses, not a promise that its old implementation/performance discussion describes today's default.
- [Python random.choice](https://docs.python.org/3/library/random.html#random.choice): sequence selection and nonempty precondition; no cryptographic use claimed.
- MIT official [2020 Lecture 4 video page](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-spring-2020/resources/lecture-4-hashing/) and [2011 Lecture 10 open-addressing video page](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/resources/lecture-10-open-addressing-cryptographic-hashing/) inspected for alternate routes. The former's matching notes and the latter's [pages 1–4 on probing/coverage](https://ocw.mit.edu/courses/6-006-introduction-to-algorithms-fall-2011/a7f609148928e4a10653d3f3be03a2b5_MIT6_006F11_lec10.pdf) were read; full playback is not claimed. The historical cryptographic/password implementation advice is not adopted or described as current guidance.
- [CMU 15-445 Fall 2022 notes](https://15445.courses.cs.cmu.edu/fall2022/notes/07-hashtables.pdf), sections 4.2–4.3: Robin Hood and cuckoo displacement mechanisms support the short deeper-directions paragraph. No empirical ranking, current hash-function recommendation or full alternative-scheme theorem is claimed.

Verified official public problem statements, including current ID/title/difficulty and constraints: 705 Design HashSet Easy; 706 Design HashMap Easy; 380 Insert Delete GetRandom O(1) Medium; 128 Longest Consecutive Sequence Medium; 454 4Sum II Medium. Direct-address solutions are possible under 705/706's bounded key universe, but our transfer prompt requires a sparse/unbounded-key design to exercise collisions. 380's platform “average” wording must be unpacked into expected map work, amortized array updates and a uniform random selection contract. 454 counts index tuples with multiplicity, not distinct value quadruples. These are not a quota or a universal interview guarantee; no judge submissions/editorial access claimed.

## Verification plan

Compare every complete Python stdout fixture with execution. Exercise put/get/remove/rebuild against a native dict oracle on adversarial, wrapped, duplicate, empty, all-deleted and full traces; validate probe-chain invariants after every operation. Independently enumerate the tiny hash family and compare exact rational frequencies. Recompute resize costs with a separately written policy simulator, test growth and alternating boundary sequences, and check potential/aggregate inequalities under the same counted work. Compare consecutive-run answers with sorted unique values and four-array counts with exhaustive Cartesian tuples. Verify dense-array/index-map bijection after each mutation; sampling belongs to a separately stated probability contract.

Inspect browser normal reading and meaningful operated states at 1440/390, with keyboard, reset, invalid input, anchors, hidden hints and code/output checks. Open actual screenshots and read numbers/labels, not merely overflow flags. Source-ready registration is not final verification or user acceptance; root owns the combined production build.
