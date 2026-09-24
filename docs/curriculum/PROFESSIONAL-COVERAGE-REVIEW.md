# Named-concept coverage review

17 September 2026. Follow-up to the professional trading/system-design catalogue expansion, prompted by HyperLogLog being difficult to find. This is a curriculum scope and discovery review, not lesson research/write or implementation completion.

## Finding and correction

HyperLogLog was explicitly in the learning sequence of **Probabilistic Data Structures: Bloom Filters, Sketches & Approximate Counts**. It was not a separately visible lesson, and catalogue search only inspected titles. That made real planned coverage look absent. The original umbrella entry and ID remain; it now introduces the choice of approximation family, followed by dedicated mechanisms.

Eight planned lessons were added inside the existing substantial sections, with prerequisites, teaching sequences, proposed investigations, practice and source anchors:

| Addition | Reason for its own lesson |
| --- | --- |
| HyperLogLog & HLL++: Approximate Distinct Counting | Registers, precision/error, duplicates, union compatibility and window/deletion limits need a distinct-count investigation, not a membership-filter diagram. |
| Bloom, Cuckoo & XOR Filters: Approximate Membership | Separate membership semantics, false positives, saturation, dynamic/static construction and safe deletion. |
| Count-Min Sketch, Count Sketch & Streaming Heavy Hitters | Distinguish point-frequency queries, candidate discovery, collision error and update models. |
| Streaming Quantiles, KLL, t-Digest & Reservoir Sampling | Compare rank/value errors, tails, weighted samples, merging and window expiry; several focused investigations may be needed. |
| Merkle Trees, Content Addressing & Data Integrity | Give hash trees, proofs, change detection, deduplication and trusted-root limits a mechanism-level owner. |
| Design Studio: Experimentation Platforms, Assignment & Metric Integrity | Connect assignment, exposure and telemetry correctness to existing statistical teaching; feature rollout alone did not cover experiment validity. |
| Stochastic Control, HJB Equations & Optimal Stopping in Finance | Bridge discrete dynamic programming and financial SDEs to admissible policies, verification and stopping boundaries. |
| RFQ Markets, Dealer Pricing & Electronic OTC Trading | Cover dealer/client quote workflows and eligibility beyond central order books. |

Current totals: **109 System Design topics, 117 Quantitative Trading topics, 12 sections in each, 1,402 unique topics across 29 modules**. All 1,394 entries from the preceding expansion remain. The original 1,218-topic conservation baseline is retained rather than overwritten. Publication and delivery checkpoints remain unchanged.

## Scope review by area

Every topic in both modules now has an explicit `subtopics` list. The [system-design syllabus](SYSTEM-DESIGN-SYLLABUS.md) and [trading syllabus](QUANTITATIVE-TRADING-SYLLABUS.md) are the generated full topic-to-concept maps. The following records the review decisions, not a competing syllabus.

| System-design area | Explicit ownership checked or strengthened |
| --- | --- |
| Foundations and software design | Requirements, budgets, Little's/Amdahl's laws, DDD, C4/ADRs, SOLID, interfaces, patterns and state machines. |
| OS, networking and transport | Memory/descriptors, event loops, epoll/io_uring, synchronization, CAS/ABA, TCP/QUIC/TLS, DNS/discovery, routing policies, edge cache contracts. |
| APIs and client state | REST/RPC/GraphQL, pagination and schema evolution, deadline/retry budgets, idempotency, sessions, resumable streams, bucket algorithms/GCRA, UUID/ULID/Snowflake, offline/rendering/accessibility boundaries. |
| Storage and approximation | B+ trees, WAL/LSM/compaction, the four sketch families, MVCC/2PL/OCC/SSI, store selection, Reed-Solomon, Merkle proofs, specialized indexes, PITR and restore. |
| Distributed correctness | CAP/PACELC/FLP assumptions, clocks and session guarantees, quorum/read repair/hinted handoff, Raft/Paxos, fencing, consistent/rendezvous/jump hashing, 2PC, CRDT/OT, geo recovery, TrueTime/HLC, membership and failure suspicion. |
| Caches, messaging and workflows | LRU/LFU/ARC/W-TinyLFU comparison, stampede prevention, acknowledgment/replay/dead letters, log offsets, backpressure, outbox/CDC, sagas, CQRS, timer wheels and fair scheduling. |
| Data and retrieval | Columnar/lakehouse structures, joins/shuffles/skew, watermarks, contracts, crawling/robots/canonicalization, BM25/MinHash/SimHash, model serving and HNSW/IVF-PQ architectural choices. |
| Reliability and verification | SLIs/SLOs/burn rates, telemetry/profiling, tail latency/hedging, adaptive admission, benchmark bias, incidents, bounded chaos, TLA+/history checking/deterministic simulation. |
| Security and infrastructure | Identity/PKCE/JWT, authorization/revocation, envelope encryption/keys, isolation, privacy/deletion, supply-chain controls, VM/container/Kubernetes resources, IaC/GitOps, cloud networks/meshes, release/migration/cost. |
| Studios and specialist depth | Payments, reservations, communication, file sync, media, geospatial H3/S2/R-trees, telemetry, object storage, workflows, IoT, experimentation; Byzantine/DHT systems, real-time scheduling and hardware-aware paths. |

| Professional trading area | Explicit ownership checked or strengthened |
| --- | --- |
| Markets, accounts and institutions | Returns/P&L/financing, statements, conventions, macro vintages, fees/mandates/prime brokerage and role boundaries. |
| Instruments and funding | Corporate actions/ETFs, bonds/DV01/OAS, futures/carry, FX/NDFs, commodity location/weather/power, options/exercise, repo/borrow, swaps/credit, digital assets and MBS/prepayment. |
| Quantitative methods | Martingales, covariance/conditioning, econometrics, Bayesian inference, SDEs, GARCH/EWMA, HAC/bootstrap/testing, filtered regimes, online detection, Hawkes/noise and HJB/control. |
| Market structure and execution | Auctions/order types, RFQ/OTC, Kyle/Glosten-Milgrom, TCA, TWAP/VWAP/POV/Almgren-Chriss, routing, Avellaneda-Stoikov, queues/fills and simulation limits. |
| Data and research validity | Vintages/identifiers/bitemporality, reconstruction, storage, entitlements/MNPI, reproducibility, labels, purging, nested evaluation, trial accounting/deflated Sharpe, fill reconciliation and model calibration. |
| Portfolio and risk | Black-Litterman/risk parity, Ledoit-Wolf/factor covariance, turnover/capacity/tax-aware constraints, Kelly, ES/EVT, stress/crowding, XVA/collateral, attribution and governance. |
| Derivatives | Black/Bachelier/Garman-Kohlhagen, measures, higher Greeks, SVI/SSVI, Dupire/Heston/SABR/jumps, trees/PDE, Longstaff-Schwartz/AAD, curve construction, Fourier calibration, exotics/credit and model validation. |
| Strategies | Equity factors, trend, OU/cointegration, carry/value, event-driven, fixed-income and convertible/capital-structure relative value, volatility/dispersion, financial text/graphs/LLMs and digital-asset mechanisms. |
| HFT and production | FIX/ITCH/OUCH/SBE, feed recovery, C++ memory ordering, NUMA/DPDK/AF_XDP, PTP/metrology, FPGA, certification, OMS/EMS, risk controls, shadow/recovery, settlement/treasury, applicable regulation/security. |
| Practice and capstones | Mathematics/coding/trading interviews, reproducible research communication, exchange replay and risk gateways, independently benchmarked pricing/risk libraries. |

This review strengthens current professional breadth and identifies owners for named methods. It cannot establish that no future technique or specialist role requirement will ever be missing. Nor does a term in a plan demonstrate a completed lesson. Preserve continuing scope review during actual authoring.

## Research anchors and limits

The structure is an editorial synthesis. Primary sources inspected for this follow-up include [Redis's HyperLogLog documentation](https://redis.io/docs/latest/develop/data-types/probabilistic/hyperloglogs/) and [Google's HLL++ paper record](https://research.google/pubs/hyperloglog-in-practice-algorithmic-engineering-of-a-state-of-the-art-cardinality-estimation-algorithm/), which anchor the dedicated cardinality lesson. Redis implementation sizes/errors are not universal HLL constants.

[Apache DataSketches' family overview](https://datasketches.apache.org/docs/Architecture/MajorSketchFamilies.html) and [Redis's probabilistic structures](https://redis.io/docs/latest/develop/data-types/probabilistic/) support distinguishing cardinality, membership, frequency and distribution summaries. The [DDSketch paper record](https://arxiv.org/abs/1908.10693), [cuckoo-filter paper](https://www.cs.cmu.edu/~binfan/papers/conext14_cuckoofilter.pdf) and [XOR-filter paper record](https://arxiv.org/abs/1912.08258) were inspected at abstract/search-record level for narrower scope anchors; full algorithm/guarantee verification remains lesson research.

[RFC 9162's Merkle components](https://www.rfc-editor.org/rfc/rfc9162.html#section-2.1) provides a concrete proof/integrity reference, not a claim that every content-addressed store implements Certificate Transparency. [Microsoft's sample-ratio-mismatch research](https://www.microsoft.com/en-us/research/articles/diagnosing-sample-ratio-mismatch-in-a-b-testing/) informs experimentation-system integrity. [NYU's stochastic-control syllabus](https://engineering.nyu.edu/sites/default/files/2025-04/FRE7821_9211.pdf) anchors control/stopping depth. [BIS's electronic fixed-income trading report landing page](https://www.bis.org/publications/electronic-trading-fixed-income-markets) supports the market-structure branch; its older report is structural background, not a claim about current venue rules. Direct PDF retrieval failed; recheck the report and venue-specific rules during authoring.

The preceding expansion's professional sources still anchor the broader review. Adding named scope does not certify every mechanism against a new source in this planning pass. Exact formulas, version behavior, financial conventions and legal applicability require the normal topic-level research phase.

## Authoring and maintenance contract

1. Run the normal topic preflight. `topic.subtopics` and `coverageInstruction` expose the named obligations even when the topic has no individual brief yet. Read them alongside its sequence, domain guidance and destination notes.
2. Map each obligation to a mechanism, example, limitation, suitable visual/investigation and independent practice as appropriate. Several related algorithms may share a comparison; unrelated mechanisms must not be squeezed into one generic lab.
3. Decide depth rather than mentioning terms in passing. Explain the key method fully; distinguish an application, comparison, prerequisite and optional specialist branch. Link deeper existing teaching owners instead of duplicating them.
4. If a concept belongs elsewhere, persist the destination instruction and update both owners with reasons. Do not silently drop labels or rename a stable ID. Preserve existing topic order while placing new prerequisites before their dependents.
5. `system-design-coverage.js` and `quantitative-trading-coverage.js` own the labels. They feed authoring inventory, generated syllabi and compact browser search. Planned pages show them as planned coverage; search matches identify curriculum scope without declaring older published bodies newly reviewed.
6. Rebuild generated artifacts and syllabi after scope changes. Run the professional verifier, search checks and affected browser/build checks. Search must not fetch lessons or outlines; selected outlines continue to load independently.

The teaching standard and two-phase ledger remain authoritative. This update creates no content-complete or implementation-complete checkpoints. Refer to the source-bound verification record linked by the parent plan for actual integration evidence.
