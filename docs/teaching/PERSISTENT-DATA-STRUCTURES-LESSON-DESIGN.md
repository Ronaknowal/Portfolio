# Persistent Data Structures: lesson design

Stable topic: `persistent-data-structures-structural-sharing-versioned-queries`; DSA position 21. Status: implemented and author-reviewed on 10 September 2026; shared integration and user acceptance are separate. No prior published body was found. The existing title remains accurate. Final evidence: [verification](PERSISTENT-DATA-STRUCTURES-VERIFICATION.md).

The incoming Range Queries note is accepted: teach real immutable path copying, branching from any root, exact allocation and historical interval checks. This is not a replay of mutable segment-tree snapshots. Core prerequisites are tree recursion, half-open intervals, aggregate invariants and binary search; each receives a local bridge. Range Queries is the specific deeper review link.

## Learning decisions

| Hurdle | Treatment and evidence |
| --- | --- |
| A second name is not a historical version | Nested mutable alias example; shared-tail stack picture and complete program; distinguish immutable nodes from mutable payloads. |
| A logical version need not own a physical copy | Real node-identity DAG with multiple root handles. Point assignment copies just the changed path and reuses the other child. Default five-element array, non-power-of-two boundaries. |
| Full persistence is branching, not editing history | Choose any source version before creating a new root; query every retained root; explicit parent-version tree separate from physical node DAG. |
| Prove the old version survives | Induction on the changed interval, frozen node fields, immutable integer payloads, identity checks and exhaustive copied-array oracles. |
| Retaining roots retains nodes | Toggle conceptual external root references, compute reachable-node union and compare unique allocations with logical per-version counts. Explicitly not a browser garbage-collection measurement. |
| Histories can be cheaper when updates never branch | Per-index timestamp histories and predecessor search; before-first-write and same-snapshot writes; sibling-version counterexample to global timestamp lookup. |
| Version differences can answer a different query | Optional coordinate-compressed prefix frequency roots and kth occurrence descent. Histogram counts, left-subtree counts and rank updates; prefix containment is essential. |
| Lazy updates cannot mutate shared children | Optional full functional range-add/range-sum program with a no-push invariant and inherited query carry, independently verified. No assignment-tag mixture implied. |
| Persistence changes cost accounting | Path length and initial 2n−1 allocation proof; no-op identity preservation; retained-root metadata; queue repeated reversal counterexample; no universal constant overhead or thread-safety claims. |

Core finish: implement a fully persistent fixed-size array with point assignment and historical interval sums; explain root sharing, branching, memory ownership and when timestamp histories suffice. Optional finish: implement kth-in-subarray using persistent prefix histograms, and lazy range-add versions. Balanced maps, HAMT/vector engineering, confluent merge policy and persistent queue scheduling are explicit follow-on reading, not unimplemented API promises.

## Visual contracts

- StackSharingFigure: exact immutable tail identities, two new heads, arrows always mean references. Tiny figure fits mobile with legible labels.
- PathCopyFigure: compact ordinary-reading view of the actual default node IDs, copied three-node path and two collapsed identity-shared subtrees. Its exact edge/total contract is checked against the model. It complements, rather than substitutes for, the complete physical DAG investigation.
- PathCopyLab: bounded integer array (1–8 values, −99…99), at most eight versions; draft index/value applied explicitly to selected source; linked old/new roots, physical node DAG, interval/sum labels, allocation counts and query cover. Root reachability toggles describe a conceptual ownership experiment; the demo retains its arena for inspection. Reset restores deterministic IDs. Native supports empty arrays separately.
- HistoryLookupLab: fixed independent histories, editable index/snapshot selection, latest eligible write highlighted; display timestamp order and no-write default. Mechanism is predecessor lookup, not a tree trace.
- PrefixRankLab: editable bounded array, half-open subarray and 1-based rank; compare prefix histograms and step an actual count-guided descent. Changes explicitly reset descent. Duplicates count separately. Geometry displays counts, not benchmark results.

## Sources actually inspected

10 September 2026: original Driscoll/Sarnak/Sleator/Tarjan paper pages 1–4 (definitions and model-dependent bounds); MIT 6.851 Lecture 1 official video page/description (video not watched end to end); Python copy and dataclasses documentation (aliasing, shallow copy and limits of frozen fields); official public LeetCode 1146 and 981 statements, including monotone timestamp constraints; CSES Range Queries and Copies statement; Clojure official collection/persistence reference; Okasaki thesis section 3.2 (printed page 19), abstract and chapter outline for persistent queues. Algorithms, examples and proofs are original implementations, checked against independent finite oracles. The full-branching CSES task fills a practice gap left by the two single-timeline LeetCode interfaces.

Scope discovery: retained the title and covered the incoming Range Queries note completely. The root-publication/durable-page boundary was routed to [the verified next External Memory owner](topic-notes/external-memory-algorithms-b-trees-i-o-complexity.md), with SQLite's atomic-commit source explicitly distinguished from a page-copy protocol. The unrelated bit-manipulation inbox proposal was assessed as outside this lesson's teaching need and remains open. No catalogue-wide audit or unrelated rewrite was performed.

## Practice and review plan

Local exercises diagnose aliasing, derive node allocation, branch from an old root, prove range-fold preservation, trace predecessor lookup, reject sibling subtraction, derive kth descent and lazy carries, and evaluate repeated queue work. Curated LeetCode practice covers Snapshot Array and Time Based Key-Value Store as partial-history designs, not claims that those judges test full persistence. Transfer tasks explicitly add branching/range aggregates/out-of-order timestamps.

Native programs must run in isolation with exact output. Independent tests compare all versions/all small ranges to copied arrays, count changed path nodes and shared identities, branch randomly, test empty/no-op policies, prefix order statistics against sorting, and lazy versions against eager arrays. Browser review covers desktop/390/320, keyboard, invalid inputs, actual screenshots and ordinary reading. Shared registration/build belong to the root agent.
