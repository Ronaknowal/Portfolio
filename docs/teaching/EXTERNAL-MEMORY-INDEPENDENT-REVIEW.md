# External-Memory Algorithms — independent source review

10 September2026. Root read the complete lesson's explanations, invariants, bounds, examples' contracts and seven changed practice answers. This is a bounded source/proof review, not an independent rerun of the author's exhaustive model tests. The [author record](EXTERNAL-MEMORY-VERIFICATION.md) owns actual program, model and browser evidence.

The page model consistently distinguishes record countN, records per transferB, record memoryM and page framesP. Cold aligned/unaligned scans, working-set effects, dirty eviction and explicit flush follow those units. The native examples' write allocation and cached writes do not imply physical durability.

The B-tree occupancy sum givesN≥2t^h−1 for the stated root convention. Its bound retains a constant for small trees; degree and byte layout remain different parameters. Split and top-down delete explanations preserve separator intervals, occupancy and equal leaf depth. Temporary primitive states are not presented as completed public operations. CPU shifting/comparison work and page transfers are separated.

The B+ convention places records in leaves and routes separator equality right. Range bounds include initial/boundary pages, proportional occupancy, output size and scattered payload fetches. Static bulk layout is explicitly distinguished from implementing dynamic B+ insertion/deletion.

The external-sort proof uses the minimum unconsumed run head and preserves multiplicity. Buffer fan-in reserves an output page; run formation, partial pages, singleton rewrites, empty input and materialized output are accounted for. The actual file sorter counts logical block calls while excluding input construction/readback, and does not claim thatP×B is a strict Python process-memory limit or that calls equal physical disk requests.

The shadow-page protocol clearly assumes a single writer, retained immutable old pages, completed durable writes and atomic durable root publication. Under those premises, recovery follows either the old complete graph or the new complete graph. Early publication violates reachability. Torn metadata, lost acknowledgement, reclamation and concurrency retain their separate requirements. SQLite's rollback journal is a contrasting actual protocol, not the lab's implementation.

All seven changed tasks agree with these contracts, including42 transfers for the13-record changed sorter, separator borrowing/merging and crash recovery. The official LeetCode choices are labelled as cache/merge transfers; local work carries the actual page-tree/durability skills. No universal storage-performance ranking or finite-interview-coverage guarantee is made.

No actionable mathematical or teaching-boundary defect was found. This review does not imply inspection of every runtime branch, every reference proof, real hardware crash testing or user approval. Final author fingerprints and production checks remain required for integration.
