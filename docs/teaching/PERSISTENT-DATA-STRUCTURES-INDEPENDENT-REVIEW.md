# Persistent Data Structures — independent source review

Root read the complete final lesson, core pure JavaScript ownership/query/history/rank models and the author's verification contract. The body is frozen at SHA256 `c17eadb43bb9d4aff6a363e45100b2d298b4077a6953df072215ecc9a9340ce8`; seven owned hashes are in `scratch/persistent-structures-lesson-review/final-source-hashes.json`. This bounded mathematical/code review is separate from the author's executed native and browser suites.

## Checked reasoning

- Assignment versus shallow/deep copying is distinguished from immutable sharing. Frozen fields are not falsely claimed to recursively freeze payloads. Version ancestry and the physical node DAG have different meanings.
- Path copying rebuilds only the changed branch, preserves untouched child identity, and propagates a no-op by returning identical nodes. The pure browser model freezes its node/handle arrays; its small inspection arena and copied bookkeeping are separate from the Python node-operation cost theorem.
- Half-open query partitions, floor-midpoint termination, empty-array behavior and negative-index rejection agree. The inductive sum invariant proves the new version, while absence of mutation separately proves preservation of older roots.
- A full binary tree with n leaves has 2n−1 nodes; an assignment has at most ceiling(log2 n)+1 path nodes. Range query boundary reasoning justifies logarithmic node work, with arbitrary-precision arithmetic costs and rendering/inspection work explicitly separated.
- Reclamation counts the union of reachable identities from all owners. Releasing one root cannot free shared descendants still reached elsewhere; the browser's retained arena is not presented as an observed garbage collection.
- Snapshot predecessor search correctly separates an unsaved working ID from saved history. Multiple unsaved writes may coalesce, while later saved records cannot be overwritten. Creation IDs do not stand for ancestry in a branching history.
- Prefix frequency differences have the required containment and nonnegative-count invariant for kth descent. Arbitrary version subtraction does not inherit that guarantee. Repeated values occupy separate ranks.
- The optional no-push addition tree includes its own tag in a node total and passes only strict-ancestor contributions when returning a covered sum. Full/partial updates preserve the algebra without touching shared children. Its restriction to commuting additions is explicit; assignment tags require a different ordered composition.
- The queue counterexample validly charges repeated reversals across branches and does not reuse one-timeline amortization credit. Disk durability, publication coordination and confluent merge policy are appropriately separate.

All eight changed local practice groups are consistent with those contracts. The two LeetCode tasks teach timestamp/predecessor interfaces; CSES supplies a more direct branching-array application with an explicit implementation-performance caveat. Root found no remaining actionable defect in this bounded read.

[Author verification](PERSISTENT-DATA-STRUCTURES-VERIFICATION.md) owns the seven executed programs, independent array/sorted/snapshot oracles, allocation checks and desktop/mobile/keyboard/ordinary screenshot evidence. Root does not claim to have rerun all those tests. Production integration and user acceptance remain separate.
