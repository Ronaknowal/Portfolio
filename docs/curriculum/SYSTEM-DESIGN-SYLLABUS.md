# System Design & Distributed Systems Engineering: detailed syllabus

Generated from the live catalogue by `node scripts/verify-professional-curriculum.mjs --write-syllabi`. Do not edit this derived list independently.

See [scope, role routes, research and authoring rules](PROFESSIONAL-TRADING-SYSTEM-DESIGN-PLAN.md). Listed order is module reading order; specialist branches are deliberate optional depth. Briefs are plans, not completed research/write or implementation checkpoints.

## Design Foundations, Requirements & Tradeoffs

### 1. System Design: Requirements, Constraints & Architecture Decisions

- Level: foundation; new planned topic; planned lesson.
- Stable ID: `system-design-requirements-constraints-architecture-decisions`.
- Prerequisites: No topic-level prerequisites; begin here.
- Named concept coverage (planned): Functional and nonfunctional requirements; Invariants; Quality attributes; Tradeoff analysis.
- Scope: Translate a user workflow into a system boundary; Distinguish functional requirements, quality attributes and constraints; State invariants, assumptions, alternatives and architecture decisions; Defend a simple design and identify what evidence would change it.
- Investigation: context and decision map — Which requirement justifies each component? Change a requirement and trace affected decisions.
- Practice: Design a small reservation service before selecting technologies. Success: Specify measurable requirements, invariants and rejected alternatives.

### 2. Capacity Estimation, Workload Models & Performance Budgets

- Level: foundation; new planned topic; planned lesson.
- Stable ID: `capacity-estimation-workload-models-performance-budgets`.
- Prerequisites: System Design: Requirements, Constraints & Architecture Decisions
- Named concept coverage (planned): Little's law; QPS and concurrency; Storage and bandwidth estimation; Skew and burst distributions; Amdahl's law.
- Scope: Describe a workload with requests, payloads and access distributions; Estimate storage, bandwidth, CPU, memory and peak concurrency; Apply latency budgets, queueing and Little's law with units; Validate estimates against measurements and sensitivity ranges.
- Investigation: request budget and saturation curves — Which resource saturates first as demand grows? Change payload size, fanout and peak factors.
- Practice: Estimate resources for an upload-and-read service. Success: Units reconcile and average, peak and uncertainty are separated.

### 3. Architecture Styles: Modular Monoliths, Services & Event-Driven Systems

- Level: foundation; new planned topic; planned lesson.
- Stable ID: `architecture-styles-modular-monoliths-services-event-driven-systems`.
- Prerequisites: System Design: Requirements, Constraints & Architecture Decisions
- Named concept coverage (planned): Modular monolith; Microservices; SOA; Hexagonal architecture; Event-driven architecture.
- Scope: Start with boundaries and a single deployable application; Compare modular monoliths, service-oriented and microservice designs; Distinguish synchronous requests, asynchronous events and batch work; Evaluate coupling, team ownership and failure costs before distribution.
- Investigation: dependency and deployment boundary map — Which boundary needs an independent deployment? Move one component across a network boundary.
- Practice: Compare two architectures for the same requirements. Success: Justify distribution with operational and organizational costs.

### 4. Domain Modeling, Bounded Contexts & Service Ownership

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `domain-modeling-bounded-contexts-service-ownership`.
- Prerequisites: System Design: Requirements, Constraints & Architecture Decisions
- Named concept coverage (planned): Domain-driven design (DDD); Aggregates and value objects; Bounded contexts; Anti-corruption layers.
- Scope: Model business concepts, events and invariants; Distinguish entities, value objects, aggregates and bounded contexts; Assign authoritative ownership and anti-corruption boundaries; Prevent shared-database and dependency coupling across teams.
- Investigation: context map and invariant boundaries — Who may change this fact? Trace an order across billing and fulfillment vocabularies.
- Practice: Partition a commerce domain with explicit contracts. Success: Invariants have owners and cross-boundary coordination is justified.

### 5. Architecture Documentation, ADRs & Technical Design Reviews

- Level: foundation; new planned topic; planned lesson.
- Stable ID: `architecture-documentation-adrs-technical-design-reviews`.
- Prerequisites: System Design: Requirements, Constraints & Architecture Decisions
- Named concept coverage (planned): C4 diagrams; Architecture decision records (ADRs); Sequence diagrams; Threat and failure review.
- Scope: Represent context, containers, components and runtime sequences; Write an architecture decision record with alternatives and consequences; Review security, operations, cost and migration with stakeholders; Update a design when evidence invalidates an assumption.
- Investigation: linked context and sequence diagrams — Do the diagrams agree about where state lives? Follow a request and its failure path.
- Practice: Write and critique a bounded design proposal. Success: Every major choice links to a requirement and testable assumption.

### 6. System Design Interviews: Clarification, Estimation & Deep Dives

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `system-design-interviews-clarification-estimation-deep-dives`.
- Prerequisites: System Design: Requirements, Constraints & Architecture Decisions; Capacity Estimation, Workload Models & Performance Budgets
- Named concept coverage (planned): Scope clarification; Capacity reasoning; Bottleneck deep dives; Changed-constraint design.
- Scope: Clarify the scope before drawing the architecture; Estimate only quantities that affect a decision; Develop APIs, data model, bottleneck and failure reasoning; Adapt a proposal to a changed requirement and communicate limitations.
- Investigation: design decision progression — Which uncertainty deserves the next five minutes? Introduce a scale or consistency change.
- Practice: Complete a timed design and retrospective. Success: Explain reasoning and tradeoffs rather than reciting a memorized diagram.

### 7. Low-Level Design: Interfaces, Composition & Dependency Boundaries

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `low-level-design-interfaces-composition-dependency-boundaries`.
- Prerequisites: Object-Oriented Programming in Python; Domain Modeling, Bounded Contexts & Service Ownership
- Named concept coverage (planned): SOLID; Dependency inversion; Composition over inheritance; Substitutability; Dependency injection.
- Scope: Translate a service responsibility into a small executable model; Design interfaces, composition, dependency inversion and explicit state ownership; Apply SOLID principles as tradeoff questions rather than slogans; Test substitutability, invariants and the cost of changing a requirement.
- Investigation: interface and state-ownership graph — Which module must change when a provider changes? Replace an adapter without changing domain rules.
- Practice: Design a pluggable reservation-pricing component. Success: Tests exercise domain behavior independently of network and persistence.

### 8. Design Patterns, State Machines & Maintainable Service Code

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `design-patterns-state-machines-maintainable-service-code`.
- Prerequisites: Low-Level Design: Interfaces, Composition & Dependency Boundaries
- Named concept coverage (planned): Strategy and factory; Adapter and decorator; Observer and command; State pattern; Finite-state machines.
- Scope: Identify recurring variation, construction and coordination problems; Compare strategy, factory, adapter, observer and command patterns; Replace scattered state flags with explicit transitions and invariants; Choose the smallest pattern that clarifies behavior and testing.
- Investigation: object collaboration and transition diagram — Which transition is illegal even if every field is valid? Trigger events in an unexpected order.
- Practice: Refactor a brittle order workflow. Success: Make invalid states unrepresentable or explicitly rejected.

## Operating Systems, Networking & Request Transport

### 9. Networking Foundations: Packets, Transport, DNS & Sockets

- Level: foundation; retained/shared topic; planned lesson.
- Stable ID: `networking-foundations-packets-transport-dns-sockets`.
- Prerequisites: Linux Basics, Filesystems & Processes
- Named concept coverage (planned): IP addressing and subnets; TCP versus UDP; DNS resolution; Sockets and ports; Packet paths.
- Scope: Draw two hosts and the network between them; Resolve a service name and identify its endpoint; Route packets across a simplified network; Compare datagrams with an ordered reliable byte stream; Frame application messages and diagnose a failed connection.
- Investigation: layered packet and stream timeline — Does one socket read correspond to one application message? Split and combine delivered bytes while preserving a correctly framed message.
- Practice: Specify a small local client/server exchange with framing. Success: Partial reads, connection failure and name/address/port roles are handled distinctly.

### 10. Operating-System Mechanisms for Service Design

- Level: foundation; new planned topic; planned lesson.
- Stable ID: `operating-system-mechanisms-for-service-design`.
- Prerequisites: System Design: Requirements, Constraints & Architecture Decisions
- Named concept coverage (planned): Virtual memory and page faults; Page cache; File descriptors; epoll and event loops; io_uring; Process and thread scheduling.
- Scope: Trace a request through process, memory, file and socket boundaries; Explain virtual memory, page cache, syscalls and scheduling; Compare processes, threads and event loops; Diagnose resource exhaustion with bounded queues and limits.
- Investigation: request and resource-lifetime diagram — Where does a blocked request consume resources? Change I/O delay and worker count.
- Practice: Explain a service failing despite low CPU usage. Success: Account for threads, descriptors, memory and waiting work.

### 11. Concurrency, Memory Models & Synchronization for Services

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `concurrency-memory-models-synchronization-for-services`.
- Prerequisites: Operating-System Mechanisms for Service Design
- Named concept coverage (planned): Happens-before; Mutexes and semaphores; Compare-and-swap (CAS); ABA problem; Deadlock and starvation; Lock-free versus wait-free.
- Scope: Identify shared state and concurrent operations; Compare mutexes, atomics, semaphores and message passing; Explain happens-before, deadlock, starvation and data races; Choose synchronization around an invariant and test adverse schedules.
- Investigation: thread interleaving and state trace — Which interleaving violates this invariant? Step two updates with and without synchronization.
- Practice: Repair a concurrent resource allocator. Success: Demonstrate safety and progress under explicit scheduling assumptions.

### 12. TCP, UDP, QUIC, TLS & Connection Lifecycle Design

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `tcp-udp-quic-tls-connection-lifecycle-design`.
- Prerequisites: Networking Foundations: Packets, Transport, DNS & Sockets
- Named concept coverage (planned): HTTP/2 and HTTP/3 transport; Head-of-line blocking; Congestion and flow control; Connection pooling; TLS handshakes; 0-RTT replay risks.
- Scope: Trace connection establishment and encrypted transport; Compare reliable streams, datagrams and QUIC multiplexing; Explain congestion control, loss, head-of-line blocking and connection reuse; Budget handshakes, timeouts and transport-level failure detection.
- Investigation: packet and connection timeline — Which loss blocks which application message? Drop packets under different transport models.
- Practice: Choose a transport for three stated workloads. Success: Justify reliability, latency, deployment and security requirements.

### 13. DNS, Anycast, Service Discovery & Traffic Steering

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `dns-anycast-service-discovery-traffic-steering`.
- Prerequisites: Networking Foundations: Packets, Transport, DNS & Sockets
- Named concept coverage (planned): DNS TTL and negative caching; Anycast and BGP boundaries; Split-horizon DNS; Client-side and server-side discovery; Health propagation.
- Scope: Resolve a public name through caches and authoritative servers; Compare DNS, registry and mesh-based service discovery; Explain anycast, regional steering and endpoint health; Model TTLs, stale discovery and failover delays.
- Investigation: name-to-endpoint and cache timeline — Why are clients still reaching a failed endpoint? Change TTL and resolver cache state.
- Practice: Design discovery and failover for a regional service. Success: Account for cached answers and health-check limitations.

### 14. Networked Services, HTTP Contracts & Identity Boundaries

- Level: foundation; retained/shared topic; planned lesson.
- Stable ID: `networked-services-http-contracts-identity-boundaries`.
- Prerequisites: Networking Foundations: Packets, Transport, DNS & Sockets
- Named concept coverage (planned): HTTP methods and status codes; Request and response contracts; Authentication versus authorization; Service boundaries.
- Scope: Draw client, server and trust boundaries; Follow a named host to an HTTP request; Define input and output schemas and status codes; Place identity checks at the service boundary; Inspect timeouts and redacted request logs.
- Investigation: request sequence diagram — Where did this request fail before the model ran? Inject lookup, connection, identity and schema failures into a fixed request trace.
- Practice: Specify and test a local prediction service contract. Success: Error cases, identity scope and sensitive-log exclusions are explicit.

### 15. Load Balancers, Reverse Proxies & API Gateways

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `load-balancers-reverse-proxies-api-gateways`.
- Prerequisites: Networked Services, HTTP Contracts & Identity Boundaries; Capacity Estimation, Workload Models & Performance Budgets
- Named concept coverage (planned): L4 versus L7; Round robin and weighted routing; Least connections and least requests; Power of two choices; Consistent-hash routing; Connection draining.
- Scope: Separate layer-four routing from layer-seven request handling; Compare load-balancing policies, connection draining and health probes; Place TLS termination, authentication and gateway policies; Analyze sticky sessions, uneven work and proxy failure.
- Investigation: request distribution and queue view — Why can equal request counts produce unequal load? Change request cost and routing policy.
- Practice: Configure a design for long-lived and short requests. Success: Document health semantics, draining and bottlenecks.

### 16. CDNs, Edge Caching & Geographic Content Delivery

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `cdns-edge-caching-geographic-content-delivery`.
- Prerequisites: Networked Services, HTTP Contracts & Identity Boundaries; DNS, Anycast, Service Discovery & Traffic Steering
- Named concept coverage (planned): Cache-Control and Vary; ETag and conditional requests; Origin shield; Signed URLs; Stale-while-revalidate; Cache poisoning.
- Scope: Trace cache hits, origin fetches and regional delivery; Define cache keys, TTLs, invalidation and signed access; Handle personalized content, stale assets and cache poisoning boundaries; Measure origin protection, latency and regional consistency tradeoffs.
- Investigation: edge-origin cache hierarchy — Can two users safely share this cached response? Change key dimensions and authorization headers.
- Practice: Design static and private-media delivery. Success: Prevent cross-user leakage and specify invalidation behavior.

## APIs, Application State & Service Contracts

### 17. API Design: REST, RPC, GraphQL & Compatibility

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `api-design-rest-rpc-graphql-compatibility`.
- Prerequisites: Networked Services, HTTP Contracts & Identity Boundaries
- Named concept coverage (planned): gRPC and Protocol Buffers; REST and GraphQL; Cursor versus offset pagination; Keyset pagination; Schema compatibility; GraphQL N+1 and query cost.
- Scope: Specify resources, operations and observable outcomes; Compare REST, RPC and GraphQL against client access patterns; Design errors, pagination, filtering, versioning and evolution; Test compatibility and authorization at contract boundaries.
- Investigation: request-response contract comparison — Which API shape reduces this client's work without hiding cost? Change client query and schema version.
- Practice: Design and evolve a collection API. Success: Pagination and errors remain deterministic under concurrent changes.

### 18. Distributed Failure Semantics, Retries & Idempotency

- Level: intermediate; retained/shared topic; planned lesson.
- Stable ID: `distributed-failure-semantics-retries-idempotency`.
- Prerequisites: Networked Services, HTTP Contracts & Identity Boundaries
- Named concept coverage (planned): Partial failure; Timeout ambiguity; Retry safety; Duplicate effects; Idempotency.
- Scope: Trace a request whose reply is lost; Compare at-most-once and at-least-once delivery; Attach a stable operation identifier; Bound retry and backoff behaviour; Recover from a partially completed workflow.
- Investigation: distributed event timeline — Did the operation fail, or did only its reply disappear? Drop messages and compare retries with and without deduplication.
- Practice: Design a restartable batch-inference job. Success: Duplicate requests do not duplicate committed outputs under the declared storage model.

### 19. Timeouts, Deadlines, Retries & Exponential Backoff

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `timeouts-deadlines-retries-exponential-backoff`.
- Prerequisites: Distributed Failure Semantics, Retries & Idempotency; Capacity Estimation, Workload Models & Performance Budgets
- Named concept coverage (planned): Deadline propagation; Full jitter; Retry budgets; Retry amplification; Cancellation.
- Scope: Allocate an end-to-end deadline across nested calls; Classify retryable outcomes and unknown effects; Apply bounded retries, jitter and retry budgets; Analyze retry amplification during overload.
- Investigation: nested deadline and retry timeline — Can retries make recovery less likely? Change dependency delay and retry policies.
- Practice: Repair a cascading retry policy. Success: Bound elapsed time, attempts and duplicate effects.

### 20. Idempotency Keys, Deduplication & Exactly-Once Effects

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `idempotency-keys-deduplication-exactly-once-effects`.
- Prerequisites: Distributed Failure Semantics, Retries & Idempotency; Networked Services, HTTP Contracts & Identity Boundaries
- Named concept coverage (planned): Idempotency-key retention; Request fingerprints; Deduplication atomicity; Unknown commit outcomes; Exactly-once effect boundaries.
- Scope: Separate repeated requests from repeated business effects; Design idempotency-key scope, retention and parameter checks; Coordinate durable deduplication with state changes; Explain crash windows and the boundaries of exactly-once claims.
- Investigation: request and commit state machine — What if the server crashes after committing but before replying? Retry at each persistence boundary.
- Practice: Specify an idempotent payment-creation endpoint. Success: Duplicate requests cannot create extra effects within the stated guarantee.

### 21. Sessions, Stateless Services & Distributed Application State

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `sessions-stateless-services-distributed-application-state`.
- Prerequisites: Networked Services, HTTP Contracts & Identity Boundaries; Load Balancers, Reverse Proxies & API Gateways
- Named concept coverage (planned): Session affinity; Opaque sessions versus JWT; Revocation; Session fixation; Shared state ownership.
- Scope: Locate session, workflow and durable business state; Compare client tokens, server sessions and shared session stores; Handle expiration, revocation, failover and sticky routing; Scale workers without losing or leaking user context.
- Investigation: session ownership and failover map — What state disappears if this worker is replaced? Restart one worker and invalidate a session.
- Practice: Design session behavior across two regions. Success: Define revocation, security and availability tradeoffs.

### 22. WebSockets, Server-Sent Events, Webhooks & Realtime APIs

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `websockets-server-sent-events-webhooks-realtime-apis`.
- Prerequisites: TCP, UDP, QUIC, TLS & Connection Lifecycle Design; API Design: REST, RPC, GraphQL & Compatibility
- Named concept coverage (planned): Long polling; SSE resumption; WebSocket heartbeats; Webhook signatures and replay; Slow consumers.
- Scope: Choose push or polling from freshness and connectivity needs; Compare bidirectional sockets, event streams and callbacks; Design reconnection, resumable cursors, heartbeats and delivery retries; Bound fanout, slow consumers and webhook authorization.
- Investigation: connection and resumption timeline — What events did a reconnecting client miss? Interrupt a stream and resume from a cursor.
- Practice: Design a resumable notification channel. Success: Define gaps, duplicates, retention and client backpressure.

### 23. Rate Limiting, Quotas & Admission Control

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `rate-limiting-quotas-admission-control`.
- Prerequisites: Capacity Estimation, Workload Models & Performance Budgets; Distributed Failure Semantics, Retries & Idempotency
- Named concept coverage (planned): Token bucket; Leaky bucket; Fixed-window counter; Sliding-window log and counter; Hierarchical quotas; GCRA.
- Scope: Separate fairness, protection and commercial quotas; Compare fixed windows, sliding windows, token and leaky buckets; Coordinate per-user and global limits across workers; Handle burst tolerance, clock behavior and degraded dependencies.
- Investigation: bucket and request-admission trace — Can a boundary burst exceed the intended limit? Replay a burst under different algorithms.
- Practice: Design a multi-tenant API quota policy. Success: State accuracy, storage, fairness and fail-open or fail-closed behavior.

### 24. Distributed IDs, Ordering & Uniqueness Guarantees

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `distributed-ids-ordering-uniqueness-guarantees`.
- Prerequisites: Distributed Failure Semantics, Retries & Idempotency
- Named concept coverage (planned): UUIDv4 and UUIDv7; ULID; Snowflake IDs; Clock rollback; Collision probability; Sequence allocation.
- Scope: Distinguish uniqueness, sortability and causal order; Compare random IDs, database sequences and time-worker counters; Handle clock rollback, worker reuse and partitioned generation; Assess information leakage and index-locality consequences.
- Investigation: ID field and clock-rollback trace — Does a larger ID necessarily mean a later event? Roll a clock back and restart a generator.
- Practice: Select an ID scheme for offline-capable clients. Success: State collision, ordering and coordination guarantees.

### 25. Client Architecture: Rendering, State, Offline Data & API Boundaries

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `client-architecture-rendering-state-offline-data-api-boundaries`.
- Prerequisites: API Design: REST, RPC, GraphQL & Compatibility; Sessions, Stateless Services & Distributed Application State
- Named concept coverage (planned): SSR, CSR and hydration; Backend for frontend (BFF); Optimistic updates; Service workers and offline storage; Cancellation and stale responses.
- Scope: Trace initial load, navigation and mutation from browser or mobile client; Compare server rendering, client rendering and backend-for-frontend boundaries; Separate server cache, local UI state and optimistic mutations; Handle offline persistence, cancellation, stale responses and recovery.
- Investigation: client-server state and rendering timeline — Can an older response overwrite a newer user action? Reorder responses and disconnect the client.
- Practice: Design an offline-capable client workflow. Success: Define authoritative state, reconciliation and visible failure behavior.

### 26. User-Perceived Performance, Accessibility & Internationalization

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `user-perceived-performance-accessibility-internationalization`.
- Prerequisites: Client Architecture: Rendering, State, Offline Data & API Boundaries
- Named concept coverage (planned): Core Web Vitals; Critical rendering path; Code splitting; Keyboard and screen-reader access; Localization and bidirectional text.
- Scope: Measure loading, responsiveness and layout stability on realistic devices; Budget network, JavaScript, rendering and background work; Include keyboard, assistive technology, localization and time-zone requirements; Evaluate architectural choices against real user journeys and device constraints.
- Investigation: critical rendering and interaction timeline — Is the backend fast while the interface still feels slow? Change device speed, bundle size and network delay.
- Practice: Review a complete client journey under constrained conditions. Success: Demonstrate usable loading, error, keyboard and localized states.

## Storage Engines, Data Models & Transactions

### 27. Data Modeling, Invariants & Schema Evolution

- Level: foundation; new planned topic; planned lesson.
- Stable ID: `data-modeling-invariants-schema-evolution`.
- Prerequisites: System Design: Requirements, Constraints & Architecture Decisions
- Named concept coverage (planned): Normalization and denormalization; Entity relationships; Unique and foreign-key constraints; Soft deletion; Schema versioning.
- Scope: Derive entities and relationships from read and write workflows; Express uniqueness, referential and business invariants; Compare normalization, denormalization and access-oriented models; Evolve schemas with compatibility, backfill and rollback.
- Investigation: entity and query-access map — Which updates can leave this denormalized value stale? Change a source record and trace dependent copies.
- Practice: Model a booking dataset and migrate one field. Success: Protect invariants during mixed-version reads and writes.

### 28. Database Indexes, B-Trees, Hash Indexes & Query Planning

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `database-indexes-b-trees-hash-indexes-query-planning`.
- Prerequisites: Data Modeling, Invariants & Schema Evolution; Arrays, Strings & Hash Maps
- Named concept coverage (planned): B+ trees; Composite and covering indexes; Selectivity and cardinality estimates; EXPLAIN plans; Sargability; Index-only scans.
- Scope: Trace a query from predicate to access path; Compare ordered, hash, composite, covering and partial indexes; Explain selectivity, statistics, join plans and write amplification; Validate an index choice with representative query plans.
- Investigation: index traversal and row-access trace — Why does this composite index miss the query's filter? Change key order and predicate ranges.
- Practice: Choose indexes for a mixed read-write workload. Success: Explain plan, storage and maintenance tradeoffs.

### 29. Storage Internals: Pages, WAL, LSM Trees & Compaction

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `storage-internals-pages-wal-lsm-trees-compaction`.
- Prerequisites: Operating-System Mechanisms for Service Design; Data Modeling, Invariants & Schema Evolution
- Named concept coverage (planned): Write-ahead log (WAL); Memtables and SSTables; Leveled versus tiered compaction; Write, read and space amplification; fsync and crash recovery.
- Scope: Trace a durable write through buffers, logs and files; Compare page-oriented and log-structured engines; Explain write-ahead logging, checkpoints, SSTables and compaction; Measure read, write and space amplification under a workload.
- Investigation: write path and compaction layers — When is an acknowledged write durable? Crash before and after log synchronization.
- Practice: Design a tiny durable key-value write path. Success: State recovery invariants and amplification costs.

### 30. Probabilistic Data Structures: Bloom Filters, Sketches & Approximate Counts

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `probabilistic-data-structures-bloom-filters-sketches-approximate-counts`.
- Prerequisites: Arrays, Strings & Hash Maps; Database Indexes, B-Trees, Hash Indexes & Query Planning
- Named concept coverage (planned): Membership versus frequency versus cardinality; Bloom filter; HyperLogLog; Count-Min Sketch; Approximation error contracts.
- Scope: Choose approximation only after stating an acceptable error contract; Compare Bloom filters, Count-Min sketches and HyperLogLog; Compute memory, false-positive and estimation tradeoffs; Handle mergeability, deletion and adversarial or skewed inputs.
- Investigation: bit-array membership and error view — Can a positive filter answer prove that an item exists? Insert keys and vary filter size.
- Practice: Choose an approximate structure for a telemetry workload. Success: State error direction, memory bounds and exact fallback behavior.

### 31. Bloom, Cuckoo & XOR Filters: Approximate Membership

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `bloom-cuckoo-xor-filters-approximate-membership`.
- Prerequisites: Probabilistic Data Structures: Bloom Filters, Sketches & Approximate Counts
- Named concept coverage (planned): Counting and scalable Bloom filters; Cuckoo filters; XOR filters; False-positive probability; Deletion safety.
- Scope: Trace hashing into a Bloom filter and test present and absent keys; Derive false-positive and memory tradeoffs under stated hash assumptions; Compare counting Bloom, scalable Bloom, cuckoo and static XOR filters; Reason about deletion, saturation, rebuilds and the exact-store fallback.
- Investigation: hash-to-bit and fingerprint bucket trace — Why does a positive answer still require an exact lookup? Insert colliding keys, fill the filter and attempt a deletion.
- Practice: Design a filter before an expensive storage lookup. Success: State false-positive behavior, update constraints and a safe deletion policy.

### 32. HyperLogLog & HLL++: Approximate Distinct Counting

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `hyperloglog-hll-approximate-distinct-counting`.
- Prerequisites: Probabilistic Data Structures: Bloom Filters, Sketches & Approximate Counts
- Named concept coverage (planned): HLL and HLL++; Register precision and leading zeros; Cardinality estimation; Sparse representation and bias correction; Compatible sketch unions; Distinct-count time windows.
- Scope: Explain why a stream's number of events differs from its number of distinct keys; Trace uniform hashes into register indices and leading-zero ranks; Estimate cardinality and relate precision, relative error, bias correction and sparse representations; Merge compatible sketches for unions and examine limits for deletion, intersections and time windows.
- Investigation: hash-bit register explorer and repeated-trial error distribution — How can a small set of registers estimate millions of distinct users? Replay duplicates, increase precision and merge overlapping streams against an exact set.
- Practice: Design daily and monthly distinct-user telemetry with HLL. Success: Verify unions on overlapping inputs, report measured error and state hash/precision compatibility and privacy limits.

### 33. Count-Min Sketch, Count Sketch & Streaming Heavy Hitters

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `count-min-sketch-count-sketch-streaming-heavy-hitters`.
- Prerequisites: Probabilistic Data Structures: Bloom Filters, Sketches & Approximate Counts
- Named concept coverage (planned): Count-Min Sketch (CMS); Count Sketch; Misra-Gries; Space-Saving; Heavy hitters and top-k; Turnstile update assumptions.
- Scope: Distinguish counting one key from finding the most frequent keys; Trace Count-Min counters and Count Sketch signed updates with their different assumptions; Compare Misra-Gries and Space-Saving candidate tracking with sketch point queries; Evaluate collisions, skew, bounded memory, merges and permitted update models.
- Investigation: counter collision trace and heavy-hitter candidate table — Can an infrequent key look popular because of other keys? Change skew, width and update signs against an exact frequency table.
- Practice: Detect popular keys in a bounded-memory event stream. Success: Separate candidate discovery from count estimation and state the error direction and update assumptions.

### 34. Streaming Quantiles, KLL, t-Digest & Reservoir Sampling

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `streaming-quantiles-kll-t-digest-reservoir-sampling`.
- Prerequisites: Probabilistic Data Structures: Bloom Filters, Sketches & Approximate Counts; Probability Distributions & Bayes' Theorem
- Named concept coverage (planned): KLL sketch; t-digest; DDSketch; Rank versus relative value error; Reservoir and weighted sampling; Mergeable summaries.
- Scope: Explain why averaging shard percentiles does not produce a global percentile; Compare KLL rank error, t-digest tail behavior and DDSketch relative value error; Trace reservoir and weighted sampling as different answers to a streaming-data question; Evaluate mergeability, window expiry, weighting and extreme-tail uncertainty.
- Investigation: distribution, rank and sampled-item views — Which error contract matters for a p99 latency objective? Merge unequal shards and compare estimates and samples with the exact stream.
- Practice: Choose a latency summary and a diagnostic sampling policy. Success: Specify error units, memory, sampling inclusion probabilities and a validation workload.

### 35. Transactions, Isolation Levels & Concurrency Anomalies

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `transactions-isolation-levels-concurrency-anomalies`.
- Prerequisites: Data Modeling, Invariants & Schema Evolution; Concurrency, Memory Models & Synchronization for Services
- Named concept coverage (planned): ACID; MVCC; Two-phase locking (2PL); Optimistic concurrency control (OCC); Snapshot isolation and SSI; Write skew and phantom reads.
- Scope: Specify which multi-record invariant a transaction protects; Trace dirty reads, lost updates, skew and phantoms; Compare isolation guarantees, MVCC, locking and serializable validation; Handle deadlocks, retries and application-visible aborts.
- Investigation: transaction interleaving table — Can both doctors go off call under this isolation level? Step reads and commits under alternative guarantees.
- Practice: Repair a write-skew schedule. Success: Explain which guarantee blocks the bad history and its cost.

### 36. Relational, Document, Key-Value, Wide-Column & Graph Databases

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `relational-document-key-value-wide-column-graph-databases`.
- Prerequisites: Data Modeling, Invariants & Schema Evolution; Transactions, Isolation Levels & Concurrency Anomalies
- Named concept coverage (planned): SQL versus NoSQL tradeoffs; Access-pattern modeling; Joins versus embedding; Graph traversals; Polyglot persistence.
- Scope: Describe query, relationship and update requirements first; Compare database families by access path and consistency needs; Analyze joins, aggregation, traversal and transactional boundaries; Justify one or several stores with operational ownership.
- Investigation: same-workload data-model comparison — Which store makes this access pattern expensive? Change the dominant query and required invariant.
- Practice: Select stores for three concrete workloads. Success: Use measured requirements instead of SQL-versus-NoSQL slogans.

### 37. Object Storage, Filesystems, Block Storage & Erasure Coding

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `object-storage-filesystems-block-storage-erasure-coding`.
- Prerequisites: Storage Internals: Pages, WAL, LSM Trees & Compaction
- Named concept coverage (planned): Reed-Solomon codes; Replication versus erasure coding; Failure domains; Multipart uploads; Checksums and repair; POSIX versus object semantics.
- Scope: Compare object, file and block interfaces; Explain chunking, metadata, replication, checksums and erasure coding; Design multipart upload, integrity checks and lifecycle policies; Evaluate durability, repair traffic and failure-domain assumptions.
- Investigation: object shards and failure-domain placement — How many failures can this placement actually tolerate? Lose disks and entire racks separately.
- Practice: Design storage for large immutable objects. Success: Calculate redundancy costs and document repair and access semantics.

### 38. Merkle Trees, Content Addressing & Data Integrity

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `merkle-trees-content-addressing-data-integrity`.
- Prerequisites: Arrays, Strings & Hash Maps; Object Storage, Filesystems, Block Storage & Erasure Coding
- Named concept coverage (planned): Merkle inclusion proofs; Content-addressable storage (CAS); Content-defined chunking; Hash collision assumptions; Trusted roots; Deduplication and garbage collection.
- Scope: Build a hash tree from canonically encoded data chunks; Verify inclusion proofs and compare roots to locate differing subtrees; Apply content addressing to deduplication, file sync and immutable artifacts; Distinguish integrity from authenticity, freshness, availability and privacy.
- Investigation: chunk tree and proof-path explorer — Which hashes must change when one chunk changes? Alter a leaf and verify a proof against a trusted root.
- Practice: Specify a content-addressed backup with integrity verification. Success: Define chunk boundaries, hash/encoding choices, trusted roots and safe garbage collection.

### 39. Time-Series Databases, Search Indexes & Specialized Retrieval

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `time-series-databases-search-indexes-specialized-retrieval`.
- Prerequisites: Database Indexes, B-Trees, Hash Indexes & Query Planning
- Named concept coverage (planned): Time-series cardinality; Inverted indexes; Downsampling and retention; Index freshness; Approximate nearest neighbors.
- Scope: Compare timestamp ranges, inverted indexes and vector retrieval; Design ingestion, retention, aggregation and index freshness; Explain high cardinality, approximate retrieval and filtered search; Choose specialized stores without duplicating the source of truth.
- Investigation: source-to-index freshness pipeline — Which results are missing after a recent update? Delay indexing and change retention or filters.
- Practice: Design metrics and search access for one dataset. Success: Specify ownership, freshness and approximation guarantees.

### 40. Database Operations: Connections, Vacuum, Backups & Restore

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `database-operations-connections-vacuum-backups-restore`.
- Prerequisites: Transactions, Isolation Levels & Concurrency Anomalies; Storage Internals: Pages, WAL, LSM Trees & Compaction
- Named concept coverage (planned): Connection pooling; Vacuum and MVCC bloat; Point-in-time recovery (PITR); Backup restore drills; Replication lag.
- Scope: Budget connection pools, locks and long-running transactions; Explain maintenance, vacuuming, compaction and storage growth; Plan backups, point-in-time recovery and restore rehearsals; Diagnose replication lag and migration-induced outages.
- Investigation: transaction age and recovery timeline — Can a successful backup still fail the recovery objective? Change log retention and restore throughput.
- Practice: Write and exercise a database recovery plan. Success: Verify recovered data and measured recovery time.

## Distributed Correctness, Replication & Coordination

### 41. Distributed Failure Models, Time & Impossibility Boundaries

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `distributed-failure-models-time-impossibility-boundaries`.
- Prerequisites: Distributed Failure Semantics, Retries & Idempotency
- Named concept coverage (planned): CAP theorem; PACELC; FLP impossibility; Partial synchrony; Crash and Byzantine failures; Network partitions.
- Scope: Separate crash, omission, partition and Byzantine failure models; Explain uncertain failure detection and physical-clock limits; Use logical clocks and happens-before relations; Interpret CAP and FLP under their actual assumptions.
- Investigation: message history with uncertain nodes — Can silence distinguish a slow node from a failed one? Delay messages without revealing the node state.
- Practice: Classify guarantees for several network histories. Success: Name timing and fault assumptions before drawing conclusions.

### 42. Distributed Consistency, Linearizability & Causal Guarantees

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `distributed-consistency-linearizability-causal-guarantees`.
- Prerequisites: Distributed Failure Models, Time & Impossibility Boundaries; Transactions, Isolation Levels & Concurrency Anomalies
- Named concept coverage (planned): Linearizability versus serializability; Sequential consistency; Causal consistency; Read-your-writes; Monotonic reads; Lamport and vector clocks.
- Scope: Describe allowed read-write histories; Compare linearizable, sequential, causal and eventual consistency; Explain read-your-writes, monotonic reads and staleness bounds; Select a guarantee from a business invariant and failure requirement.
- Investigation: operation-history checker — Is there a legal instant when this write took effect? Reorder overlapping operations and inspect witnesses.
- Practice: Classify small histories and repair a violating design. Success: Use precise guarantees rather than strong or weak labels.

### 43. Replication, Quorums & Failover Semantics

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `replication-quorums-failover-semantics`.
- Prerequisites: Distributed Consistency, Linearizability & Causal Guarantees; Storage Internals: Pages, WAL, LSM Trees & Compaction
- Named concept coverage (planned): Leader-follower and leaderless replication; Read and write quorums; Sloppy quorums; Hinted handoff; Read repair; Split brain.
- Scope: Compare leader-follower, multi-leader and leaderless replication; Reason about acknowledgment, read and write quorums; Handle lag, read repair, anti-entropy and failover; Explain lost updates and stale reads under concrete failure histories.
- Investigation: replica acknowledgment and read timeline — Which acknowledged write survives this failover? Crash replicas at different durability points.
- Practice: Evaluate quorum claims with counterexample histories. Success: State membership, versioning and conflict-resolution assumptions.

### 44. Consensus, Raft, Paxos & Replicated State Machines

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `consensus-raft-paxos-replicated-state-machines`.
- Prerequisites: Replication, Quorums & Failover Semantics
- Named concept coverage (planned): Raft; Paxos and Multi-Paxos; Log commitment; Election safety; State-machine replication.
- Scope: Explain agreement, validity and termination for a replicated log; Trace leader election, terms, log matching and commit rules; Compare Raft and Paxos concepts and reconfiguration challenges; Validate safety during leader changes and delayed messages.
- Investigation: term and replicated-log stepper — Which entries are committed after a leader change? Partition, elect and reconnect nodes.
- Practice: Trace and test a small consensus history. Success: Distinguish committed entries, uncommitted tails and liveness assumptions.

### 45. Leader Election, Leases, Distributed Locks & Fencing Tokens

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `leader-election-leases-distributed-locks-fencing-tokens`.
- Prerequisites: Consensus, Raft, Paxos & Replicated State Machines
- Named concept coverage (planned): Lease expiry; Fencing tokens; ZooKeeper and etcd coordination; Lock ownership after pauses; Session failure.
- Scope: Separate leader authority from evidence of liveness; Explain leases, expiration and paused-process hazards; Use monotonically checked fencing tokens at the protected resource; Design ownership transfer across clock and network uncertainty.
- Investigation: lease expiry and stale-writer timeline — Can the old owner still write after losing its lease? Pause a client through lease expiration.
- Practice: Repair a distributed job-owner design. Success: The resource rejects stale owners rather than trusting a lock alone.

### 46. Sharding, Consistent Hashing, Hot Keys & Rebalancing

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `sharding-consistent-hashing-hot-keys-rebalancing`.
- Prerequisites: Replication, Quorums & Failover Semantics; Capacity Estimation, Workload Models & Performance Budgets
- Named concept coverage (planned): Consistent hashing and virtual nodes; Rendezvous hashing; Jump consistent hash; Range versus hash partitioning; Hot-key salting; Online resharding.
- Scope: Choose hash, range or directory partitioning from access patterns; Explain consistent and rendezvous hashing with skew; Move data while preserving routing and write ownership; Detect hot keys, cross-shard queries and rebalance amplification.
- Investigation: partition ownership and movement map — Does adding a shard fix a single hot key? Change key popularity and node membership.
- Practice: Design a live shard split. Success: Show reads and writes during movement and failure recovery.

### 47. Distributed Transactions, Two-Phase Commit & Atomic Commit

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `distributed-transactions-two-phase-commit-atomic-commit`.
- Prerequisites: Transactions, Isolation Levels & Concurrency Anomalies; Consensus, Raft, Paxos & Replicated State Machines; Sharding, Consistent Hashing, Hot Keys & Rebalancing
- Named concept coverage (planned): 2PC prepare and commit; Coordinator recovery; In-doubt transactions; Three-phase commit assumptions; Atomic commit versus consensus.
- Scope: Specify atomicity across separately failing participants; Trace prepare, commit and recovery in two-phase commit; Distinguish consensus replication from transaction commit; Assess blocking, coordinator recovery and alternative boundaries.
- Investigation: participant commit-state timeline — What may a prepared participant do after coordinator failure? Crash the coordinator between protocol steps.
- Practice: Analyze a cross-shard money transfer. Success: Preserve atomicity and state the availability cost.

### 48. CRDTs, Conflict Resolution & Offline-First Synchronization

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `crdts-conflict-resolution-offline-first-synchronization`.
- Prerequisites: Distributed Consistency, Linearizability & Causal Guarantees
- Named concept coverage (planned): State-based and operation-based CRDTs; G-counter, PN-counter and OR-set; Tombstones; Last-write-wins limitations; Operational transformation comparison.
- Scope: Represent concurrent updates without assuming a global order; Compare state-based and operation-based convergence rules; Explain causal metadata, tombstones and conflict-aware user workflows; Identify invariants that require coordination beyond merging.
- Investigation: branch-and-merge replica history — Can convergent replicas still violate a business rule? Edit offline and merge in different orders.
- Practice: Specify an offline collaborative list. Success: State convergence, deletion and uniqueness semantics.

### 49. Multi-Region Architecture, Data Residency & Disaster Recovery

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `multi-region-architecture-data-residency-disaster-recovery`.
- Prerequisites: Replication, Quorums & Failover Semantics; Distributed Transactions, Two-Phase Commit & Atomic Commit
- Named concept coverage (planned): Active-active and active-passive; RPO and RTO; Failover and failback; Geo-partitioning; Residency versus locality.
- Scope: Place reads, writes and authoritative data across regions; Compare active-passive and active-active operation; Budget RPO, RTO, latency, residency and inter-region cost; Practice region loss, failback and split-brain prevention.
- Investigation: regional authority and recovery timeline — Which data can be lost at this failover point? Partition regions and inspect acknowledged writes.
- Practice: Design and rehearse a regional failover. Success: Document authority, allowable data loss and return-to-normal behavior.

### 50. Distributed SQL, Logical Timestamps & Externally Consistent Transactions

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `distributed-sql-logical-timestamps-externally-consistent-transactions`.
- Prerequisites: Distributed Transactions, Two-Phase Commit & Atomic Commit; Multi-Region Architecture, Data Residency & Disaster Recovery
- Named concept coverage (planned): Spanner and TrueTime; Hybrid logical clocks (HLC); Commit wait; External consistency; Cross-shard transactions.
- Scope: Combine partitioned storage, replicated logs and transaction coordination; Compare physical uncertainty, logical and hybrid logical timestamps; Explain timestamp ordering, commit waits and external consistency assumptions; Analyze cross-region transaction latency, locality and failure behavior.
- Investigation: timestamp uncertainty and commit timeline — What must be true before a commit is ordered after an earlier response? Widen clock uncertainty and inspect waiting.
- Practice: Compare two distributed SQL transaction designs. Success: Distinguish timestamp mechanism, replication and isolation guarantees.

### 51. Membership Changes, Snapshots, Gossip & Anti-Entropy

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `membership-changes-snapshots-gossip-anti-entropy`.
- Prerequisites: Consensus, Raft, Paxos & Replicated State Machines; Sharding, Consistent Hashing, Hot Keys & Rebalancing
- Named concept coverage (planned): Joint consensus; Snapshot installation; Gossip and SWIM; Phi-accrual failure detection; Merkle-tree anti-entropy; Failure suspicion versus authority.
- Scope: Separate membership authority from gossip-based discovery; Change replica configurations without incompatible quorums; Install snapshots and compact logs while preserving recovery; Use anti-entropy and failure detection with explicit convergence limits.
- Investigation: membership epochs and snapshot transfer timeline — Can two different configurations both believe they own the data? Change membership during a partition.
- Practice: Design a replica replacement and recovery procedure. Success: Preserve quorum intersection, ownership and snapshot consistency.

## Caching, Messaging & Durable Workflows

### 52. Caching Strategies, Invalidation & Stampede Control

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `caching-strategies-invalidation-stampede-control`.
- Prerequisites: Distributed Consistency, Linearizability & Causal Guarantees; Capacity Estimation, Workload Models & Performance Budgets
- Named concept coverage (planned): Cache-aside and write-through; Write-behind; LRU and LFU; ARC and W-TinyLFU comparisons; Singleflight and request coalescing; TTL jitter and negative caching.
- Scope: Compare cache-aside, read-through, write-through and write-behind; Define keys, eviction, TTL and invalidation contracts; Handle stampedes, penetration, hot keys and stale reads; Prevent cache failure from overwhelming the source of truth.
- Investigation: cache-database race timeline — Can invalidation still leave an old value in the cache? Interleave cache fills with writes.
- Practice: Design a cache for mutable account data. Success: State staleness bounds, failure behavior and consistency risks.

### 53. Message Queues, Publish-Subscribe & Delivery Semantics

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `message-queues-publish-subscribe-delivery-semantics`.
- Prerequisites: Distributed Failure Semantics, Retries & Idempotency; Replication, Quorums & Failover Semantics
- Named concept coverage (planned): Acknowledgments and visibility timeouts; At-least-once and at-most-once; Dead-letter queues (DLQ); Poison messages; Message TTL and replay.
- Scope: Separate work queues from broadcast subscriptions; Trace publish acknowledgment, delivery, acknowledgment and redelivery; Compare at-most-once, at-least-once and scoped exactly-once guarantees; Design retention, dead letters, poison-message handling and replay.
- Investigation: message ownership and acknowledgment trace — What happens if a worker crashes after doing the work? Crash before and after acknowledgment.
- Practice: Specify a durable background-job pipeline. Success: Account for duplicate effects, retries and permanently failing jobs.

### 54. Partitioned Logs, Consumer Groups & Event Ordering

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `partitioned-logs-consumer-groups-event-ordering`.
- Prerequisites: Message Queues, Publish-Subscribe & Delivery Semantics; Sharding, Consistent Hashing, Hot Keys & Rebalancing
- Named concept coverage (planned): Kafka-style partition logs; Consumer offsets; Rebalancing; Idempotent producers; Log compaction; Partition ordering.
- Scope: Map event keys to log partitions and consumer ownership; Explain offsets, retention, replay and consumer-group rebalances; Distinguish partition order from global and causal order; Handle hot partitions, duplicate consumption and changing keys.
- Investigation: partition log and consumer cursor view — Which order survives a consumer rebalance? Move ownership while processing events.
- Practice: Design a keyed event pipeline. Success: Specify offset commits, ordering scope and recovery.

### 55. Backpressure, Bounded Queues & Stream Flow Control

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `backpressure-bounded-queues-stream-flow-control`.
- Prerequisites: Message Queues, Publish-Subscribe & Delivery Semantics; Capacity Estimation, Workload Models & Performance Budgets
- Named concept coverage (planned): Credit-based flow control; Reactive streams demand; Bounded buffers; Load shedding; Batching and latency.
- Scope: Model producer rate, service rate and queued work; Compare pull demand, credit, blocking, dropping and load shedding; Propagate pressure across asynchronous stages; Bound memory and latency under sustained overload.
- Investigation: pipeline queue occupancy plot — Where does unbounded work accumulate? Slow one consumer and compare admission policies.
- Practice: Design a bounded ingestion pipeline. Success: State maximum backlog, drop policy and recovery behavior.

### 56. Transactional Outbox, Inbox & Change Data Capture

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `transactional-outbox-inbox-change-data-capture`.
- Prerequisites: Transactions, Isolation Levels & Concurrency Anomalies; Message Queues, Publish-Subscribe & Delivery Semantics
- Named concept coverage (planned): Dual-write problem; Outbox relay; Inbox deduplication; Log-based CDC; Snapshot-to-stream handoff.
- Scope: Identify the database-plus-message dual-write failure; Persist events with business state through an outbox; Compare polling and change-data-capture publication; Handle duplicates, schema changes, ordering and cleanup.
- Investigation: database commit and publication timeline — Can a state update occur without its event? Crash at each dual-write boundary.
- Practice: Repair an order-created event pipeline. Success: Demonstrate atomic intent and idempotent downstream handling.

### 57. Sagas, Compensation & Durable Workflow Engines

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `sagas-compensation-durable-workflow-engines`.
- Prerequisites: Message Queues, Publish-Subscribe & Delivery Semantics; Idempotency Keys, Deduplication & Exactly-Once Effects
- Named concept coverage (planned): Orchestration versus choreography; Compensating actions; Durable timers; Workflow replay and versioning; Manual reconciliation.
- Scope: Represent a long-running workflow as durable states and actions; Compare orchestration and choreography; Design retries, timers, compensation and manual resolution; Handle non-reversible effects and versioned workflow code.
- Investigation: workflow state and compensation graph — What if refund succeeds but inventory release fails? Inject failure after each external effect.
- Practice: Design a booking-payment-fulfillment saga. Success: Specify recovery and compensations without claiming ACID rollback.

### 58. Event Sourcing, CQRS & Projection Evolution

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `event-sourcing-cqrs-projection-evolution`.
- Prerequisites: Partitioned Logs, Consumer Groups & Event Ordering; Data Modeling, Invariants & Schema Evolution
- Named concept coverage (planned): Event store; Command-query responsibility segregation; Projection rebuilds; Event upcasting; Snapshotting and deletion.
- Scope: Distinguish event history from a mutable current-state store; Separate command validation and read projections; Handle event versioning, snapshots, replay and erasure obligations; Evaluate when auditability benefits justify operational complexity.
- Investigation: event stream and two projections — Can the same events build different read models safely? Replay an evolving projection from a snapshot.
- Practice: Specify an event-sourced account aggregate. Success: Preserve invariants, version compatibility and rebuild semantics.

### 59. Job Scheduling, Timers & Distributed Task Execution

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `job-scheduling-timers-distributed-task-execution`.
- Prerequisites: Message Queues, Publish-Subscribe & Delivery Semantics; Leader Election, Leases, Distributed Locks & Fencing Tokens
- Named concept coverage (planned): DAG scheduling; Timing wheels; Work stealing; Weighted fair queueing and deficit round robin; Priority inversion; Lease-based task claiming.
- Scope: Model scheduled, recurring and dependency-driven work; Design claiming, leases, retries and cancellation; Handle clock changes, missed schedules and duplicate triggers; Provide fairness, resource quotas and observable task history.
- Investigation: schedule-claim-run-retry timeline — Can two workers run the same overdue job? Expire a lease during a long task.
- Practice: Design a durable scheduler with resumable workers. Success: Bound duplicates and protect side effects with fencing or idempotency.

## Data Platforms, Analytics & ML System Interfaces

### 60. Batch Data Platforms, Warehouses, Lakes & Lakehouses

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `batch-data-platforms-warehouses-lakes-lakehouses`.
- Prerequisites: Object Storage, Filesystems, Block Storage & Erasure Coding; Data Modeling, Invariants & Schema Evolution
- Named concept coverage (planned): Parquet and columnar encoding; Iceberg-style table snapshots; Catalogs and lineage; Partition pruning; Small-file compaction.
- Scope: Separate operational data from analytical workloads; Compare warehouse, lake and lakehouse responsibilities; Explain columnar formats, partitioning, catalogs and table snapshots; Plan incremental ingestion, compaction, lineage and access controls.
- Investigation: operational-to-analytical data lifecycle — Which snapshot did this analysis actually read? Introduce a late backfill and inspect versioned tables.
- Practice: Design an auditable analytical dataset. Success: Define ownership, quality, freshness and reproducible snapshots.

### 61. Distributed Query Engines, Joins, Shuffles & Analytical Execution

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `distributed-query-engines-joins-shuffles-analytical-execution`.
- Prerequisites: Batch Data Platforms, Warehouses, Lakes & Lakehouses; Database Indexes, B-Trees, Hash Indexes & Query Planning
- Named concept coverage (planned): MapReduce and DAG execution; Broadcast and shuffle joins; Sort-merge and hash joins; Predicate pushdown; Vectorized execution; Skew and spill.
- Scope: Turn a logical analytical query into partitioned physical operators; Compare broadcast, shuffle, sort-merge and partitioned joins; Handle skew, spill, pushdown, vectorization and partial aggregation; Recover stages and measure data movement rather than only worker count.
- Investigation: query DAG and shuffle-volume map — Why does adding workers fail to speed up this join? Change key skew and broadcast size.
- Practice: Optimize a distributed analytical query plan. Success: Account for network, memory, skew and reproducible results.

### 62. Stream Processing, Event Time, Watermarks & Windowing

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `stream-processing-event-time-watermarks-windowing`.
- Prerequisites: Partitioned Logs, Consumer Groups & Event Ordering; Backpressure, Bounded Queues & Stream Flow Control
- Named concept coverage (planned): Tumbling, sliding and session windows; Watermarks and late data; Retractions; Checkpoints and state backends; Exactly-once sink boundaries.
- Scope: Separate event time, ingest time and processing time; Compare tumbling, sliding and session windows; Explain watermarks, late events, retractions and state expiry; Coordinate checkpoints and output effects across restart.
- Investigation: event-time versus arrival-time timeline — When may this window result be considered final? Delay and reorder events around the watermark.
- Practice: Compute a corrected aggregate after late arrivals. Success: State lateness, retention and output-update guarantees.

### 63. Data Contracts, Schema Registries & Pipeline Quality

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `data-contracts-schema-registries-pipeline-quality`.
- Prerequisites: Data Modeling, Invariants & Schema Evolution; Transactional Outbox, Inbox & Change Data Capture
- Named concept coverage (planned): Avro, Protobuf and JSON schemas; Backward and forward compatibility; Freshness and completeness; Lineage; Quarantine and backfills.
- Scope: Define meaning, types, units and ownership at data boundaries; Check backward and forward schema compatibility; Validate freshness, completeness, uniqueness and lineage; Handle backfills, quarantine, deletion and producer-consumer coordination.
- Investigation: schema-version compatibility matrix — Can every deployed consumer read the new event? Add, remove and reinterpret fields.
- Practice: Plan a safe event-schema migration. Success: Test old and new producers and consumers together.

### 64. Search System Architecture: Crawling, Indexing & Retrieval

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `search-system-architecture-crawling-indexing-retrieval`.
- Prerequisites: Time-Series Databases, Search Indexes & Specialized Retrieval; Partitioned Logs, Consumer Groups & Event Ordering
- Named concept coverage (planned): Crawler frontier and politeness; robots.txt; Canonicalization; Inverted indexes and BM25; MinHash and SimHash near-duplicate detection; Access-filtered ranking.
- Scope: Trace documents from discovery through parsing and indexing; Design inverted indexes, updates, ranking and query fanout; Handle deduplication, freshness, access filtering and abuse; Evaluate relevance, latency and index recovery separately.
- Investigation: document-to-query retrieval pipeline — Why is a known document absent from results? Inspect crawl, index and permission stages.
- Practice: Design an enterprise search system. Success: Define deletion propagation, authorization and measurable relevance.

### 65. ML Serving Architecture, Feature Freshness & Model Rollouts

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `ml-serving-architecture-feature-freshness-model-rollouts`.
- Prerequisites: Data Contracts, Schema Registries & Pipeline Quality; API Design: REST, RPC, GraphQL & Compatibility
- Named concept coverage (planned): Online versus batch inference; Feature-store consistency; Training-serving skew; Batching and fallbacks; Shadow evaluation.
- Scope: Separate offline training from online decision serving; Define feature computation, freshness and model-version contracts; Design batching, caches, fallbacks and staged model rollout; Evaluate latency, quality, drift and rollback together.
- Investigation: feature-and-model version trace — Did training and serving observe the same feature meaning? Change a feature pipeline version under a deployed model.
- Practice: Design a bounded fraud-scoring serving service. Success: Document point-in-time features, fallback rules and model rollback.

### 66. Vector Search, RAG & AI Application System Boundaries

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `vector-search-rag-ai-application-system-boundaries`.
- Prerequisites: Search System Architecture: Crawling, Indexing & Retrieval; ML Serving Architecture, Feature Freshness & Model Rollouts
- Named concept coverage (planned): HNSW and IVF-PQ index tradeoffs; Hybrid search and reranking; Embedding version migration; Permission-filtered retrieval; Prompt injection boundaries.
- Scope: Separate retrieval, context assembly, generation and evaluation; Design chunk and embedding versions, indexes and permission filters; Handle deletion, prompt injection, tool permissions and fallback; Budget latency, cost, freshness and evidence quality.
- Investigation: document-permission-to-answer provenance map — Can a retrieved chunk bypass the user's access scope? Change document permissions and rebuild status.
- Practice: Specify an auditable document-answering system. Success: Trace evidence and enforce access before generation.

## Reliability, Observability & Performance Engineering

### 67. SLIs, SLOs, Error Budgets & Availability Modeling

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `slis-slos-error-budgets-availability-modeling`.
- Prerequisites: System Design: Requirements, Constraints & Architecture Decisions; Capacity Estimation, Workload Models & Performance Budgets
- Named concept coverage (planned): Burn-rate alerts; Error-budget policy; Series and parallel availability; Correlated failures; Dependency budgets.
- Scope: Choose user-visible indicators and measurement populations; Set objectives, windows and error-budget policies; Model dependency availability and correlated failures; Connect reliability targets to release and capacity decisions.
- Investigation: request outcome and error-budget burn chart — Does the service meet its promise for this user group? Change aggregation window and regional failure.
- Practice: Define an SLO for a critical workflow. Success: Specify valid events, objectives, exclusions and actionable burn alerts.

### 68. Observability: Metrics, Logs, Traces & Profiling

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `observability-metrics-logs-traces-profiling`.
- Prerequisites: Networked Services, HTTP Contracts & Identity Boundaries
- Named concept coverage (planned): OpenTelemetry; RED and USE methods; Trace context propagation; Head and tail sampling; High-cardinality labels; Continuous profiling.
- Scope: Instrument a request across service and data boundaries; Compare metrics, structured logs, traces and profiles; Manage sampling, cardinality, correlation and sensitive fields; Use evidence to test a concrete failure hypothesis.
- Investigation: linked trace, metric and log timeline — Which dependency explains the tail-latency spike? Inspect one slow request against aggregate distributions.
- Practice: Diagnose a synthetic incident from telemetry. Success: Support the root-cause hypothesis and bound missing evidence.

### 69. Tail Latency, Queueing, Fanout & Performance Diagnosis

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `tail-latency-queueing-fanout-performance-diagnosis`.
- Prerequisites: Capacity Estimation, Workload Models & Performance Budgets; Observability: Metrics, Logs, Traces & Profiling
- Named concept coverage (planned): p50, p95 and p99; Utilization and saturation; Hedged requests; Fanout amplification; Coordinated omission; Universal scalability law.
- Scope: Measure distributions rather than only means; Explain fanout amplification, queueing and utilization near saturation; Compare batching, caching, parallelism and hedged requests; Identify bottlenecks with load-controlled experiments.
- Investigation: fanout latency and queue occupancy curves — Why does one slow shard dominate the request? Increase fanout and correlate dependency delays.
- Practice: Diagnose a tail-latency regression. Success: Separate service time, waiting, sampling artifacts and retries.

### 70. Circuit Breakers, Bulkheads, Load Shedding & Graceful Degradation

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `circuit-breakers-bulkheads-load-shedding-graceful-degradation`.
- Prerequisites: Timeouts, Deadlines, Retries & Exponential Backoff; Backpressure, Bounded Queues & Stream Flow Control
- Named concept coverage (planned): Half-open circuit recovery; Adaptive concurrency limits; Bulkhead isolation; Priority shedding; Fallback correctness.
- Scope: Bound the blast radius of a failing dependency; Compare circuit breaking, concurrency limits and bulkheads; Prioritize work and define safe degraded responses; Test recovery, oscillation and overload admission.
- Investigation: dependency pressure and breaker state — Does rejecting work early preserve the critical workflow? Overload a low-priority feature and inspect isolation.
- Practice: Design a degradation policy for a checkout flow. Success: Protect invariants and distinguish reduced features from false success.

### 71. Load Testing, Benchmark Validity & Capacity Experiments

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `load-testing-benchmark-validity-capacity-experiments`.
- Prerequisites: Capacity Estimation, Workload Models & Performance Budgets; Observability: Metrics, Logs, Traces & Profiling
- Named concept coverage (planned): Open versus closed workloads; Coordinated omission; Warm versus cold cache; Soak and spike tests; Reproducible percentile measurement.
- Scope: Build workload distributions from stated access patterns; Compare open and closed load generation; Detect coordinated omission, cache bias and unrealistic data sizes; Report saturation, error rates, percentiles and reproducible configurations.
- Investigation: offered versus completed load plot — Did the test stop sending work when the service slowed? Compare fixed-arrival and response-paced load.
- Practice: Design a reproducible service benchmark. Success: Include warmup, data shape, failure rates and load-generator limits.

### 72. Incident Management, On-Call, Runbooks & Blameless Reviews

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `incident-management-on-call-runbooks-blameless-reviews`.
- Prerequisites: Observability: Metrics, Logs, Traces & Profiling; SLIs, SLOs, Error Budgets & Availability Modeling
- Named concept coverage (planned): Incident command; Mitigation versus diagnosis; Escalation and communication; Postmortems; Action verification.
- Scope: Detect and triage a user-impacting incident; Separate mitigation, diagnosis, communication and command roles; Use runbooks, escalation and evidence preservation; Produce a review with causal factors and verifiable follow-up actions.
- Investigation: incident decision and evidence timeline — Which action reduced user impact before root cause was known? Reveal observations in time order.
- Practice: Run a tabletop outage and write its review. Success: Record decisions, impact and actions without hindsight certainty.

### 73. Fault Injection, Chaos Experiments & Resilience Validation

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `fault-injection-chaos-experiments-resilience-validation`.
- Prerequisites: Incident Management, On-Call, Runbooks & Blameless Reviews; Circuit Breakers, Bulkheads, Load Shedding & Graceful Degradation
- Named concept coverage (planned): Fault models; Blast radius; Abort criteria; Dependency and network injection; Recovery validation.
- Scope: State a resilience hypothesis and measurable steady state; Choose bounded faults, blast radius and abort criteria; Inject process, network, dependency and resource failures; Distinguish demonstrated behavior from untested failure combinations.
- Investigation: failure-domain and impact map — Does this supposedly redundant path share the failed dependency? Remove one dependency within a controlled scenario.
- Practice: Design a safe staging resilience experiment. Success: Include observability, stop conditions and recovery evidence.

### 74. Correctness Testing, Model Checking & Deterministic Simulation

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `correctness-testing-model-checking-deterministic-simulation`.
- Prerequisites: Consensus, Raft, Paxos & Replicated State Machines; Transactions, Isolation Levels & Concurrency Anomalies
- Named concept coverage (planned): TLA+ and PlusCal; Linearizability checking; Jepsen-style histories; Deterministic simulation; Property-based testing; Safety versus liveness.
- Scope: Express safety and liveness properties before implementation; Generate small concurrent histories and fault schedules; Use property tests, state exploration and history checkers; Build deterministic replay and minimize counterexamples.
- Investigation: state-space and failing-history explorer — Which smallest schedule violates the invariant? Step a reduced counterexample.
- Practice: Check a bounded replicated service model. Success: Separate tested bounds from a general proof and retain replay seeds.

## Security, Privacy & Multi-Tenant Isolation

### 75. System Threat Modeling, Trust Boundaries & Secure Defaults

- Level: foundation; new planned topic; planned lesson.
- Stable ID: `system-threat-modeling-trust-boundaries-secure-defaults`.
- Prerequisites: Networked Services, HTTP Contracts & Identity Boundaries; Data Modeling, Invariants & Schema Evolution
- Named concept coverage (planned): STRIDE; Least privilege; Injection and SSRF; Trust boundaries; Secure-by-default configuration.
- Scope: Identify assets, actors, trust boundaries and abuse cases; Trace inputs and privileges through the architecture; Apply least privilege, secure defaults and defense in depth; Connect each threat to a control, test and residual risk.
- Investigation: data-flow and trust-boundary diagram — Which untrusted input crosses a privileged boundary? Trace attacker-controlled fields through the design.
- Practice: Threat-model a file-sharing system. Success: Protect assets through concrete controls and testable assumptions.

### 76. Authentication, OAuth, OpenID Connect & Session Security

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `authentication-oauth-openid-connect-session-security`.
- Prerequisites: System Threat Modeling, Trust Boundaries & Secure Defaults; Sessions, Stateless Services & Distributed Application State
- Named concept coverage (planned): OAuth 2.0 and OIDC; PKCE; JWT validation and JWKS rotation; CSRF and XSS boundaries; Passkeys and MFA; Account recovery.
- Scope: Separate identity proof, delegated access and application sessions; Trace authorization-code flow, PKCE, token audiences and lifetimes; Handle refresh, revocation, MFA and account recovery; Protect cookies, redirects and browser request boundaries.
- Investigation: identity-provider and application sequence — Which party is allowed to consume this token? Change issuer, audience and redirect destination.
- Practice: Review a login and delegated-access design. Success: Reject token confusion and document revocation and recovery semantics.

### 77. Authorization, RBAC, ABAC & Relationship-Based Permissions

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `authorization-rbac-abac-relationship-based-permissions`.
- Prerequisites: System Threat Modeling, Trust Boundaries & Secure Defaults; Distributed Consistency, Linearizability & Causal Guarantees
- Named concept coverage (planned): RBAC; ABAC; ReBAC and Zanzibar-style relations; Object-level authorization (BOLA); Revocation consistency.
- Scope: Model subjects, resources, actions and permission relationships; Compare role, attribute and relationship-based decisions; Propagate revocation consistently through caches and search; Enforce object-level access at every relevant operation.
- Investigation: permission graph and revocation timeline — Can stale authorization permit access after removal? Revoke membership during an in-flight request.
- Practice: Design document-sharing permissions. Success: Address inheritance, revocation, audit and cache consistency.

### 78. Encryption, Key Management, Secrets & Certificate Rotation

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `encryption-key-management-secrets-certificate-rotation`.
- Prerequisites: System Threat Modeling, Trust Boundaries & Secure Defaults; TCP, UDP, QUIC, TLS & Connection Lifecycle Design
- Named concept coverage (planned): AEAD; Envelope encryption; KMS and HSM; PKI and mTLS; Password hashing; Key rotation and recovery.
- Scope: Separate confidentiality, integrity and authenticity guarantees; Design envelope encryption and key hierarchy; Manage secrets, certificate issuance, rotation and revocation; Plan backups and recovery without exposing keys.
- Investigation: key hierarchy and rotation timeline — Can old data still be recovered after this key change? Rotate a wrapping key and revoke a credential.
- Practice: Design a service secret and encryption lifecycle. Success: Document access, audit, rotation and recovery failure modes.

### 79. Multi-Tenant Architecture, Noisy Neighbors & Isolation

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `multi-tenant-architecture-noisy-neighbors-isolation`.
- Prerequisites: System Threat Modeling, Trust Boundaries & Secure Defaults; Rate Limiting, Quotas & Admission Control; Sharding, Consistent Hashing, Hot Keys & Rebalancing
- Named concept coverage (planned): Tenant context propagation; Row-level security; Resource quotas; Tenant migration; Isolation tiers.
- Scope: Compare shared tables, schemas, databases and dedicated deployments; Enforce tenant identity in data, caches, jobs and telemetry; Allocate fair resource budgets and limit blast radius; Plan tenant migration, deletion and tier-specific guarantees.
- Investigation: tenant-resource isolation matrix — Can one tenant's workload delay or expose another's data? Increase a tenant load and inspect shared bottlenecks.
- Practice: Design a multi-tenant analytics service. Success: Demonstrate data isolation and resource fairness under adversarial load.

### 80. Privacy, Retention, Deletion & Audit Evidence

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `privacy-retention-deletion-audit-evidence`.
- Prerequisites: System Threat Modeling, Trust Boundaries & Secure Defaults; Transactional Outbox, Inbox & Change Data Capture
- Named concept coverage (planned): Data minimization; Deletion propagation; Retention and legal holds; PII classification; Jurisdiction-specific controls.
- Scope: Classify data and minimize collection and access; Trace retention and deletion through replicas, indexes, events and backups; Separate operational audit evidence from unnecessary personal data; Map jurisdiction-specific obligations to verified controls.
- Investigation: data-copy and deletion propagation graph — Which copy survives a deletion request? Track deletion through async projections and restore.
- Practice: Design verifiable deletion for a user dataset. Success: State scope, delay, backup policy and evidence without universal legal claims.

### 81. Abuse Prevention, DDoS, Fraud & Software Supply-Chain Security

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `abuse-prevention-ddos-fraud-software-supply-chain-security`.
- Prerequisites: System Threat Modeling, Trust Boundaries & Secure Defaults; Rate Limiting, Quotas & Admission Control
- Named concept coverage (planned): WAF and bot controls; DDoS protection; Dependency provenance; SBOM and artifact signing; CI/CD trust boundaries.
- Scope: Distinguish volumetric attacks, expensive valid requests and business abuse; Combine traffic controls, identity friction and risk signals; Protect builds, dependencies, artifacts and production access; Plan incident response with false-positive and availability costs.
- Investigation: attack path and control-layer map — Can a small request trigger disproportionate work? Change request cost while keeping request rate fixed.
- Practice: Review an expensive endpoint and its release pipeline. Success: Bound amplification and trace trusted artifact provenance.

## Cloud Infrastructure, Deployment & Architecture Evolution

### 82. Virtual Machines, Containers, Isolation & Runtime Resources

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `virtual-machines-containers-isolation-runtime-resources`.
- Prerequisites: Operating-System Mechanisms for Service Design; System Threat Modeling, Trust Boundaries & Secure Defaults
- Named concept coverage (planned): Namespaces and cgroups; Container images; OOM kills and CPU throttling; Hypervisors; Sandbox boundaries.
- Scope: Compare VM and container isolation boundaries; Explain images, namespaces, cgroups and runtime limits; Budget memory, CPU, storage and network resources; Diagnose OOM, throttling and shared-host effects.
- Investigation: host and workload isolation map — Why is this container throttled while the host looks idle? Change quotas and shared workload pressure.
- Practice: Specify runtime limits for a service. Success: Explain isolation, scheduling and failure consequences.

### 83. Kubernetes Architecture, Scheduling & Service Operation

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `kubernetes-architecture-scheduling-service-operation`.
- Prerequisites: Virtual Machines, Containers, Isolation & Runtime Resources; DNS, Anycast, Service Discovery & Traffic Steering
- Named concept coverage (planned): Controllers and reconciliation; Probes and disruption budgets; HPA and VPA; Requests and limits; Persistent volumes; Placement and autoscaling.
- Scope: Separate control-plane desired state from running workloads; Explain scheduling, services, probes, controllers and reconciliation; Manage rollouts, disruption budgets, storage and secrets; Diagnose control-plane, node and workload failure separately.
- Investigation: reconciliation loop and pod-placement map — Why does a ready process still fail application requests? Change probes and dependency state.
- Practice: Design a resilient service deployment. Success: Explain readiness, termination, disruption and resource policies.

### 84. Serverless, Managed Services & Build-versus-Buy Decisions

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `serverless-managed-services-build-versus-buy-decisions`.
- Prerequisites: Capacity Estimation, Workload Models & Performance Budgets; Architecture Styles: Modular Monoliths, Services & Event-Driven Systems
- Named concept coverage (planned): Cold starts; Concurrency limits; Vendor lock-in; Shared responsibility; Total cost of ownership.
- Scope: Define which operational responsibility a managed service absorbs; Compare serverless, managed databases and self-operated systems; Model cold starts, quotas, portability and failure visibility; Estimate total lifecycle cost and exit constraints.
- Investigation: responsibility and cost boundary comparison — Which responsibility remains with the application team? Change workload shape and vendor limits.
- Practice: Write a build-versus-buy decision record. Success: Include operational burden, migration costs and failure constraints.

### 85. Infrastructure as Code, Configuration & Environment Reproducibility

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `infrastructure-as-code-configuration-environment-reproducibility`.
- Prerequisites: Virtual Machines, Containers, Isolation & Runtime Resources
- Named concept coverage (planned): Declarative state; Drift detection; Remote state locking; Secrets separation; GitOps.
- Scope: Represent desired infrastructure and configuration as versioned inputs; Plan state management, drift detection and secret separation; Test changes and isolate environments and permissions; Recover from partial application and accidental configuration changes.
- Investigation: desired-versus-actual infrastructure diff — Which unreviewed change caused environment drift? Modify a resource outside the declared configuration.
- Practice: Design a reproducible staging environment. Success: Track dependencies, secrets, approvals and rollback limitations.

### 86. Cloud Networks, Service Meshes & Control-Plane Separation

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `cloud-networks-service-meshes-control-plane-separation`.
- Prerequisites: Kubernetes Architecture, Scheduling & Service Operation; System Threat Modeling, Trust Boundaries & Secure Defaults
- Named concept coverage (planned): VPCs, subnets and NAT; Private endpoints; Service meshes and mTLS; Sidecars and ambient proxies; Control-plane isolation.
- Scope: Design subnets, routing, NAT, private access and network policy; Separate traffic data planes from configuration and identity control planes; Evaluate service meshes, mutual TLS and sidecar or proxy costs; Diagnose policy propagation, shared control dependencies and route failures.
- Investigation: network and control-plane dependency diagram — Does a control-plane outage stop existing traffic or only updates? Freeze configuration and interrupt identity refresh.
- Practice: Design connectivity and policy for a private service estate. Success: Document egress, authorization, propagation delay and failure behavior.

### 87. CI/CD, Canary Releases, Feature Flags & Safe Rollback

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `ci-cd-canary-releases-feature-flags-safe-rollback`.
- Prerequisites: Observability: Metrics, Logs, Traces & Profiling; Data Modeling, Invariants & Schema Evolution
- Named concept coverage (planned): Blue-green deployment; Canary analysis; Rolling upgrades; Feature kill switches; Mixed-version compatibility.
- Scope: Separate build verification, deployment and feature release; Compare rolling, blue-green and canary strategies; Handle mixed-version protocols, schemas and feature flags; Define automated promotion and rollback from user-impact evidence.
- Investigation: version rollout and compatibility timeline — Can the old binary still read data written by the new one? Rollback after a schema-writing release.
- Practice: Plan a safe stateful-service rollout. Success: Include compatibility windows, stop signals and irreversible-change handling.

### 88. Schema Migrations, Backfills & Zero-Downtime Evolution

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `schema-migrations-backfills-zero-downtime-evolution`.
- Prerequisites: Data Modeling, Invariants & Schema Evolution; Transactions, Isolation Levels & Concurrency Anomalies; CI/CD, Canary Releases, Feature Flags & Safe Rollback
- Named concept coverage (planned): Expand-contract migrations; Dual-read validation; Resumable backfills; Throttling; Rollback boundaries.
- Scope: Expand a schema while old and new code coexist; Backfill with bounded load, resumability and correctness checks; Switch reads and writes through a verified compatibility window; Contract obsolete state with a recovery plan.
- Investigation: expand-migrate-contract timeline — Which application versions can safely coexist at this step? Interrupt a backfill and roll back a reader.
- Practice: Migrate an indexed field on a busy service. Success: Bound locks, load, data divergence and rollback scope.

### 89. Monolith Decomposition, Legacy Integration & Strangler Migrations

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `monolith-decomposition-legacy-integration-strangler-migrations`.
- Prerequisites: Domain Modeling, Bounded Contexts & Service Ownership; Schema Migrations, Backfills & Zero-Downtime Evolution
- Named concept coverage (planned): Strangler pattern; Anti-corruption adapters; Shadow traffic; Data ownership extraction; Parity reconciliation.
- Scope: Identify a migration goal and a bounded ownership seam; Route traffic incrementally with anti-corruption adapters; Separate data ownership without unsafe dual writes; Measure parity and retire old paths only after reconciliation.
- Investigation: old-new traffic and data ownership map — Which system is authoritative during the migration? Shift one workflow and compare shadow results.
- Practice: Plan an incremental billing-service extraction. Success: Document authority, compatibility, cutover and rollback.

### 90. FinOps, Capacity Planning, Sustainability & Engineering Economics

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `finops-capacity-planning-sustainability-engineering-economics`.
- Prerequisites: Capacity Estimation, Workload Models & Performance Budgets; Observability: Metrics, Logs, Traces & Profiling
- Named concept coverage (planned): Unit economics; Egress costs; Reserved versus spot capacity; Rightsizing; Energy and utilization tradeoffs.
- Scope: Attribute cost to workloads, tenants and user outcomes; Compare compute, storage, egress and operational labor; Plan reserved, elastic and interruption-tolerant capacity; Evaluate efficiency changes against reliability and quality budgets.
- Investigation: cost per useful operation and capacity curve — Did the optimization merely move cost elsewhere? Change retention, replication and traffic geography.
- Practice: Propose a measured cost-reduction plan. Success: Preserve declared reliability and correctness while stating energy-estimation limits.

## End-to-End Product & Infrastructure Design Studios

### 91. Design Studio: URL Shortener, Redirects & Abuse Controls

- Level: intermediate; new planned topic; planned lesson.
- Stable ID: `design-studio-url-shortener-redirects-abuse-controls`.
- Prerequisites: Distributed IDs, Ordering & Uniqueness Guarantees; Caching Strategies, Invalidation & Stampede Control
- Named concept coverage (planned): Base62 encoding; Alias collisions; Hot redirects; Link expiry; Abuse screening.
- Scope: Specify creation, redirect, expiration and custom-alias workflows; Choose identifiers, storage and cache behavior; Separate redirect availability from analytics ingestion; Handle malicious links, hot destinations and deletion propagation.
- Investigation: redirect critical path and analytics branch — Should analytics failure block the redirect? Fail analytics and expire a cached link.
- Practice: Design and benchmark a bounded redirect service. Success: Protect alias uniqueness and define cache and abuse behavior.

### 92. Design Studio: Chat, Presence & Multi-Device Messaging

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `design-studio-chat-presence-multi-device-messaging`.
- Prerequisites: WebSockets, Server-Sent Events, Webhooks & Realtime APIs; Partitioned Logs, Consumer Groups & Event Ordering
- Named concept coverage (planned): Delivery and read receipts; Presence heartbeats; Offline mailboxes; Multi-device synchronization; Group fanout.
- Scope: Define conversation membership and delivery-read semantics; Design message IDs, ordering, history and offline synchronization; Model presence, typing and fanout with weaker transient guarantees; Handle reconnection, permission changes and multi-device duplicates.
- Investigation: conversation and device cursor timeline — What does delivered mean across several devices? Disconnect one device and revoke a member.
- Practice: Design a multi-device group messaging service. Success: Separate durable messages from transient presence and receipts.

### 93. Design Studio: Social Feeds, Fanout & Notification Delivery

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `design-studio-social-feeds-fanout-notification-delivery`.
- Prerequisites: Message Queues, Publish-Subscribe & Delivery Semantics; Caching Strategies, Invalidation & Stampede Control
- Named concept coverage (planned): Push versus pull fanout; Celebrity skew; Notification deduplication; Privacy changes; Feed ranking boundaries.
- Scope: Specify feed freshness, privacy and ranking inputs; Compare fanout-on-write, fanout-on-read and hybrid strategies; Handle celebrity skew, edits, deletes and blocked relationships; Design notification preferences, deduplication and quiet hours.
- Investigation: fanout and merge cost map — Which users make precomputed feeds expensive? Change follower skew and posting frequency.
- Practice: Design feed delivery with privacy changes. Success: Bound fanout and propagate deletion and permission changes.

### 94. Design Studio: Payments, Ledgers & Financial Reconciliation

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `design-studio-payments-ledgers-financial-reconciliation`.
- Prerequisites: Transactions, Isolation Levels & Concurrency Anomalies; Idempotency Keys, Deduplication & Exactly-Once Effects; Sagas, Compensation & Durable Workflow Engines
- Named concept coverage (planned): Double-entry ledger; Money precision; Idempotent payment intent; Chargebacks and refunds; Reconciliation.
- Scope: Specify monetary units, balances and double-entry invariants; Separate payment intent, provider outcome and ledger posting; Handle retries, refunds, chargebacks and uncertain external effects; Reconcile internal books with independent provider statements.
- Investigation: payment state and balanced ledger trace — Did a timed-out charge succeed at the provider? Retry and reconcile delayed provider records.
- Practice: Build a simulated payment and reconciliation design. Success: Preserve balanced postings and never infer failure from timeout alone.

### 95. Design Studio: Inventory, Ticketing & Reservation Contention

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `design-studio-inventory-ticketing-reservation-contention`.
- Prerequisites: Transactions, Isolation Levels & Concurrency Anomalies; Leader Election, Leases, Distributed Locks & Fencing Tokens
- Named concept coverage (planned): Reservation leases; Oversell prevention; Hot inventory; Waiting rooms; Payment-expiry races.
- Scope: Define inventory conservation and reservation expiry; Compare pessimistic, optimistic and queued allocation; Coordinate payment and expiration without overselling; Handle hot items, bots, fairness and canceled reservations.
- Investigation: inventory and reservation state machine — Who owns the last seat when payment and expiry race? Interleave expiry, payment and confirmation.
- Practice: Design a high-contention booking workflow. Success: Prove inventory invariants across retries and compensation.

### 96. Design Studio: File Sync, Collaboration & Version Conflicts

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `design-studio-file-sync-collaboration-version-conflicts`.
- Prerequisites: Object Storage, Filesystems, Block Storage & Erasure Coding; CRDTs, Conflict Resolution & Offline-First Synchronization
- Named concept coverage (planned): Merkle-tree change detection; Chunk deduplication; Resumable transfer; CRDT versus OT collaboration; Offline conflict resolution.
- Scope: Separate file content, metadata, permissions and versions; Design chunking, resumable transfer and change detection; Handle concurrent edits, deletes and offline conflicts; Propagate sharing revocation and recover previous versions.
- Investigation: device branches and chunk graph — How should simultaneous rename and edit be merged? Reconnect devices after conflicting offline actions.
- Practice: Design a conflict-aware sync service. Success: State data-loss, conflict resolution and permission guarantees.

### 97. Design Studio: Video Streaming, Transcoding & Live Delivery

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `design-studio-video-streaming-transcoding-live-delivery`.
- Prerequisites: CDNs, Edge Caching & Geographic Content Delivery; Job Scheduling, Timers & Distributed Task Execution
- Named concept coverage (planned): HLS and DASH; Adaptive bitrate; Transcoding DAGs; Live latency; DRM and access rights; WebRTC boundaries.
- Scope: Trace upload or live ingest through encoding and packaging; Design renditions, adaptive bitrate, storage and CDN delivery; Budget startup delay, buffering, latency and bandwidth; Handle failed transcodes, regional demand and access rights.
- Investigation: media pipeline and playback-buffer timeline — Why does a fast network still produce playback stalls? Change bitrate, segment availability and bandwidth variability.
- Practice: Design live and on-demand delivery for one service. Success: Separate latency, reliability, cost and quality tradeoffs.

### 98. Design Studio: Maps, Geospatial Search & Ride Dispatch

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `design-studio-maps-geospatial-search-ride-dispatch`.
- Prerequisites: Database Indexes, B-Trees, Hash Indexes & Query Planning; Stream Processing, Event Time, Watermarks & Windowing
- Named concept coverage (planned): Geohash; S2 and H3 cells; R-trees and quadtrees; Nearest-neighbor search; Route versus straight-line distance; Dispatch matching.
- Scope: Represent locations, spatial indexes and changing availability; Compare proximity search with route and travel-time estimation; Coordinate offers, assignment, cancellation and payment workflows; Handle stale positions, regional partitions and fairness constraints.
- Investigation: spatial cells and assignment timeline — Is the nearest available driver the fastest feasible match? Change road connectivity and location freshness.
- Practice: Design dispatch with stale updates and competing requests. Success: Prevent conflicting assignment and state location-error limits.

### 99. Design Studio: Metrics, Logs & High-Cardinality Telemetry

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `design-studio-metrics-logs-high-cardinality-telemetry`.
- Prerequisites: Observability: Metrics, Logs, Traces & Profiling; Time-Series Databases, Search Indexes & Specialized Retrieval; Stream Processing, Event Time, Watermarks & Windowing
- Named concept coverage (planned): HyperLogLog cardinality budgets; Quantile sketches; Retention tiers; Label explosion; Ingest recovery.
- Scope: Specify ingest rate, query patterns and retention tiers; Partition and aggregate metrics, logs and traces separately; Control cardinality, sampling and tenant quotas; Recover ingestion without misleading monitoring during outages.
- Investigation: telemetry pipeline and series-cardinality map — Which label creates millions of new series? Add a user-ID label and inspect resource growth.
- Practice: Design an observable telemetry backend. Success: Bound resource use and distinguish no data from healthy service.

### 100. Design Studio: Cloud Object Store & Metadata Service

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `design-studio-cloud-object-store-metadata-service`.
- Prerequisites: Consensus, Raft, Paxos & Replicated State Machines; Object Storage, Filesystems, Block Storage & Erasure Coding; Sharding, Consistent Hashing, Hot Keys & Rebalancing
- Named concept coverage (planned): Metadata consensus; Object versioning; Erasure-coded repair; Orphan collection; Integrity verification.
- Scope: Specify object naming, versioning and consistency promises; Separate metadata authority from bulk data placement; Design multipart writes, checksums, repair and garbage collection; Handle orphaned chunks, correlated loss and lifecycle deletion.
- Investigation: metadata commit and data-fragment placement — Can metadata expose an incompletely stored object? Crash during a multipart commit.
- Practice: Design an object store with recovery invariants. Success: Explain publication, durability and repair under stated failures.

### 101. Design Studio: Workflow Platform, CI Runners & Build Cache

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `design-studio-workflow-platform-ci-runners-build-cache`.
- Prerequisites: Job Scheduling, Timers & Distributed Task Execution; Multi-Tenant Architecture, Noisy Neighbors & Isolation
- Named concept coverage (planned): DAG scheduling; Hermetic builds; Content-addressed artifacts; Untrusted execution; Cache-key completeness.
- Scope: Model dependency graphs, artifacts and execution environments; Schedule workers with quotas, cancellation and durable progress; Design content-addressed caches and reproducibility boundaries; Isolate untrusted jobs and recover coordinator or worker loss.
- Investigation: DAG execution and artifact provenance graph — Can a cache hit reuse an artifact from the wrong environment? Change dependency or toolchain versions.
- Practice: Design a multi-tenant build platform. Success: Track cache validity, sandboxing and at-least-once execution effects.

### 102. Design Studio: IoT Ingestion, Device Control & Edge Operation

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `design-studio-iot-ingestion-device-control-edge-operation`.
- Prerequisites: Message Queues, Publish-Subscribe & Delivery Semantics; CRDTs, Conflict Resolution & Offline-First Synchronization; System Threat Modeling, Trust Boundaries & Secure Defaults
- Named concept coverage (planned): MQTT and CoAP tradeoffs; Device twins; Store-and-forward; Firmware rollout; Device identity and revocation.
- Scope: Model device identity, telemetry and desired versus reported state; Design intermittent connectivity, buffering and command acknowledgment; Handle firmware rollout, revocation and resource-constrained operation; Separate cloud availability from local safety and control needs.
- Investigation: device-cloud synchronization timeline — Was the command received, executed or merely queued? Disconnect the device and change desired state.
- Practice: Design a fleet ingestion and update service. Success: Bound offline storage and define authorization and rollback.

### 103. Design Studio: Experimentation Platforms, Assignment & Metric Integrity

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `design-studio-experimentation-platforms-assignment-metric-integrity`.
- Prerequisites: Data Contracts, Schema Registries & Pipeline Quality; CI/CD, Canary Releases, Feature Flags & Safe Rollback; Hypothesis Testing & Confidence Intervals
- Named concept coverage (planned): Stable hash bucketing; A/A tests; Exposure logging; Sample-ratio mismatch (SRM); Experiment interference; Metric lineage.
- Scope: Separate feature delivery, treatment assignment, exposure and outcome measurement; Design stable randomization units, namespaces, ramping and reproducible configuration; Trace exposure logs, joins, late events and metric ownership through the platform; Diagnose sample-ratio mismatch, interference and telemetry bias before interpreting effects.
- Investigation: assignment-to-metric lineage and missing-event trace — Can correct assignment still produce a biased experiment result? Drop events from one variant and compare assignment, exposure and analysis populations.
- Practice: Design a multi-team experiment service with an A/A validation. Success: Preserve assignment and exposure history, reconcile populations and link statistical assumptions to the analysis owner.

## Advanced Systems & Production Capstones

### 104. Peer-to-Peer Systems, Byzantine Faults & Trustless Coordination

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `peer-to-peer-systems-byzantine-faults-trustless-coordination`.
- Prerequisites: Consensus, Raft, Paxos & Replicated State Machines; System Threat Modeling, Trust Boundaries & Secure Defaults
- Named concept coverage (planned): PBFT and Byzantine quorum assumptions; Sybil resistance; DHT and Kademlia; Proof of work and proof of stake; Finality.
- Scope: Contrast crash-fault consensus with malicious participants; Explain identities, Sybil resistance and Byzantine quorum assumptions; Compare permissioned BFT, peer discovery and decentralized ledgers; Evaluate finality, partitions and incentive or trust boundaries.
- Investigation: faulty-participant message matrix — Can conflicting messages create different committed decisions? Let one participant equivocate under a bounded model.
- Practice: Compare trusted-cluster and adversarial-network designs. Success: State fault thresholds and identity assumptions precisely.

### 105. Real-Time & Embedded System Design: Deadlines, Safety & Control Boundaries

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `real-time-embedded-system-design-deadlines-safety-control-boundaries`.
- Prerequisites: Operating-System Mechanisms for Service Design; Tail Latency, Queueing, Fanout & Performance Diagnosis
- Named concept coverage (planned): Worst-case execution time (WCET); Rate-monotonic and EDF scheduling; Priority inversion; Watchdogs; Fail-safe local control.
- Scope: Distinguish hard deadlines from average low latency; Budget worst-case execution, scheduling and communication; Separate safety-critical local control from best-effort services; Validate timing assumptions and degraded operation.
- Investigation: deadline and scheduling trace — Can a low average latency still miss a safety deadline? Inject a worst-case blocking interval.
- Practice: Design an edge-control and cloud-monitoring boundary. Success: State deadline evidence and keep unsupported safety claims explicit.

### 106. High-Performance Systems: Zero Copy, RDMA & Hardware-Aware Design

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `high-performance-systems-zero-copy-rdma-hardware-aware-design`.
- Prerequisites: Concurrency, Memory Models & Synchronization for Services; Tail Latency, Queueing, Fanout & Performance Diagnosis
- Named concept coverage (planned): sendfile and memory mapping; RDMA; NUMA locality; Cache coherence; Batching and ownership; eBPF observability.
- Scope: Measure data movement and synchronization before optimizing; Compare batching, zero copy, memory mapping and RDMA; Account for NUMA, cache locality, ownership and device lifetimes; Validate performance without weakening correctness or isolation.
- Investigation: data-copy and ownership path — Which copy is actually on the critical path? Remove a copy and inspect changed buffer lifetimes.
- Practice: Design a bounded high-throughput data path. Success: Measure bottlenecks and specify memory and failure semantics.

### 107. System Design Capstone: Evolve a Service from One Node to Multiple Regions

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `system-design-capstone-evolve-a-service-from-one-node-to-multiple-regions`.
- Prerequisites: Multi-Region Architecture, Data Residency & Disaster Recovery; CI/CD, Canary Releases, Feature Flags & Safe Rollback; FinOps, Capacity Planning, Sustainability & Engineering Economics
- Named concept coverage (planned): Measured architecture evolution; State migration; Regional recovery; Capacity and cost evidence.
- Scope: Start with explicit workflows and a working single-node baseline; Introduce caching, partitioning and replication only for measured needs; Migrate state and deploy regional operation with recovery exercises; Deliver architecture decisions, load evidence, cost and operational ownership.
- Investigation: architecture evolution and evidence timeline — Which measured limit justifies each new component? Advance workload stages and compare alternatives.
- Practice: Produce a staged implementation and migration portfolio. Success: Preserve invariants and verify each scale transition.

### 108. System Design Capstone: Build & Verify a Replicated Key-Value Service

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `system-design-capstone-build-verify-a-replicated-key-value-service`.
- Prerequisites: Consensus, Raft, Paxos & Replicated State Machines; Correctness Testing, Model Checking & Deterministic Simulation; Storage Internals: Pages, WAL, LSM Trees & Compaction
- Named concept coverage (planned): Replicated log; Snapshot recovery; Fault schedules; Independent history checking.
- Scope: Specify read, write, membership and durability guarantees; Implement a bounded replicated log and recovery path; Inject partitions, restarts, delayed messages and snapshot transitions; Check histories independently and publish measured limits.
- Investigation: replicated log, client history and recovery view — Can every acknowledged operation be explained by the promised history? Replay minimized failing schedules.
- Practice: Deliver a reproducible correctness and performance report. Success: Distinguish safety evidence, liveness assumptions and tested scale.

### 109. System Design Capstone: Secure Multi-Tenant Event Platform

- Level: advanced; new planned topic; planned lesson.
- Stable ID: `system-design-capstone-secure-multi-tenant-event-platform`.
- Prerequisites: Multi-Tenant Architecture, Noisy Neighbors & Isolation; Partitioned Logs, Consumer Groups & Event Ordering; Privacy, Retention, Deletion & Audit Evidence
- Named concept coverage (planned): Tenant isolation; Durable event delivery; Quota enforcement; Replay and deletion; Operational readiness.
- Scope: Define tenant isolation, delivery, ordering and retention contracts; Implement ingestion, durable subscriptions and quotas; Exercise tenant abuse, schema evolution, replay and deletion; Deliver SLOs, threat model, runbooks and a cost model.
- Investigation: tenant streams and fault-isolation view — Can one tenant exhaust or read another tenant's stream? Inject noisy-neighbor load and permission changes.
- Practice: Build and review a bounded event-platform prototype. Success: Demonstrate isolation, recovery, retention and delivery guarantees.
