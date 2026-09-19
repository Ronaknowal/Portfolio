# Professional trading and system design curriculum

Updated 17 September 2026. This increment adds catalogue coverage and starting briefs only. It does not write, implement or certify lessons. The [teaching standard](../../LESSON-TEACHING-STANDARD.md), [handoff](../../LESSON-AUTHORING-HANDOFF.md) and two-phase delivery ledger continue to own authoring and completion.

## What is in the website

| Module | Organization | Membership |
| --- | --- | --- |
| Quantitative Trading, Financial Markets & Investment Engineering | 12 substantial sections, 117 topics | Expands the existing `quantitative-finance` module: all 39 original topics retained, 78 new topics. |
| System Design & Distributed Systems Engineering | 12 substantial sections, 109 topics | New `system-design` module: 106 new topics and 3 shared networking/service/failure topics retaining their original IDs and implementations. |

There are two new focused guided paths, **Quantitative Trading** and **System Design**. Shared prerequisites are included through the existing route resolver. The complete catalogue now has 29 modules, 1,402 unique topics and 9 paths. All 1,218 pre-increment topic IDs remain, as do all 176 additions from the initial expansion. Publication remains 228 lessons; no content or implementation checkpoint was marked complete by this planning request.

The [named-concept coverage follow-up](PROFESSIONAL-COVERAGE-REVIEW.md) explains the eight later additions, including dedicated HyperLogLog, and the scope/discovery review across all 226 module memberships. Its 1,150 named concept labels are explicit planned teaching obligations, searchable without fetching lesson bodies. The generated syllabi below expose these labels under their teaching owners.

The complete ordered lists, including per-topic scope, prerequisite links, proposed investigations and practice where individually planned, are:

- [Quantitative trading: detailed syllabus](QUANTITATIVE-TRADING-SYLLABUS.md).
- [System design: detailed syllabus](SYSTEM-DESIGN-SYLLABUS.md).

These are generated from the same catalogue that drives the website. Do not maintain a competing topic list in this document. New topics have compact individual starting briefs; retained finance topics without a prior individual brief still require detailed design before rewriting. A section, title, brief or existing publication is not proof of completed research or teaching quality.

## Coverage philosophy and boundaries

The objective is broad professional competence: foundations, mechanisms, tools, realistic constraints, failure diagnosis, operations, specialist branches and demonstrated work. A finite catalogue cannot honestly guarantee every possible topic, interview question, future discovery, trading opportunity or employment outcome. New omissions should be assigned to an existing owner or added deliberately without breaking identities or creating tiny modules.

The module reading order is explicit, not sorted by difficulty. All currently recorded prerequisites within each of these modules precede their dependent topic. External prerequisites use existing mathematics, programming, algorithms, ML, GPU and infrastructure lessons. The guided paths include their dependency closure; cross-module specialist readiness links remain important, since the curriculum is not a global topological lesson sorter.

Advanced and frontier material is a branch for an appropriate role. Learners need not master every specialist branch before applying the core. Do not present an early specialist label as a beginner's compulsory detour. Keep specialist prerequisites and a useful first pass visible in the eventual lesson.

## Quantitative trading: organizing rationale

The former AI-heavy catalogue had useful forecasting and strategy titles but insufficient market, instrument, execution and operational context. The expanded sequence is:

1. **Markets, institutions and accounting:** trade lifecycle, return sources, statements, cash flows, discounting, P&L, macroeconomics and fund business models.
2. **Instruments and financing:** equities/ETFs, bonds, futures, FX, commodities, options, short selling/repo, swaps, structured credit and digital-asset mechanics.
3. **Quantitative methods:** probability, linear algebra, Python, econometrics, optimization, Bayesian inference, stochastic calculus, time series, decision theory, inference, online estimation and point processes.
4. **Microstructure and execution:** books, matching rules, auctions, adverse selection, transaction costs, execution schedules, routing, market making and queue models. These precede fill modeling and trading backtests.
5. **Data and reproducibility:** point-in-time inputs, identifier history, book reconstruction, storage, data rights, provenance and reproducible computation.
6. **Research validation and forecasting:** hypotheses, labels, overlap, temporal evaluation, trial accounting, backtest overfitting, realistic fills and ML evaluation.
7. **Portfolio and risk:** risk models, costs, capacity, position sizing, tail risk, stress, collateral, counterparty exposure, attribution and governance.
8. **Derivatives and hedging:** pricing measures, Greeks, surfaces, model families, PDE/tree/Monte Carlo methods, curves, numerical calibration, exotics and validation.
9. **Strategies:** equity factors, trend, stat arb, carry, event-driven, fixed-income relative value, volatility, financial NLP/graphs/LLMs and digital assets. Mechanisms and evidence precede strategy claims.
10. **HFT engineering:** inventory/markout risk, order gateways, feed handlers, C++, memory ordering, networking, clock metrology, FPGA boundaries and venue certification.
11. **Production and operations:** OMS/EMS, pre-trade controls, monitoring, incident recovery, clearing, treasury, regulation, model governance and security.
12. **Professional practice:** mathematical and technical interviews, research communication, and independently reviewable research, exchange-replay and derivative-library capstones.

### Role routes within one coherent module

| Role emphasis | Shared core and specialist focus | Demonstrable work |
| --- | --- | --- |
| Quantitative researcher / systematic portfolio manager | Markets and instruments; quantitative methods; data and validation; alpha; portfolio risk; relevant execution and operating constraints | A reproducible research package with negative controls, trial history, realistic costs, capacity and an investment-or-rejection memo. |
| Quantitative trader / execution researcher | Core markets, inference and risk; microstructure, execution, fill/queue models, market-making economics and controls | Event replay with reconciled orders/fills/P&L and execution-quality analysis under latency and liquidity changes. |
| HFT / quant developer | Core market mechanics; programming, algorithms, CPU/C++ and networking prerequisites; gateways, feed recovery, timing, hot paths, hardware and operational controls | Correct and reproducible software plus failure/recovery tests; hardware claims require actual hardware measurements. |
| Derivatives / pricing quant | Cash-flow conventions, stochastic methods, derivatives, numerical methods, curves, calibration, counterparty and model risk | A pricing/risk library with independent analytic/numerical checks, convergence evidence and model limitations. |
| Risk / fund platform / investment operations | Instruments, financing, risk, attribution, governance, controls, reconciliation and continuity | A stress/recovery exercise, reconciled books and a model-use or operational control record. |

These are study emphases, not separate duplicate modules or claims about universal job titles. Overlap is intentional: researchers must understand execution and costs, and developers must understand the market state their code represents.

## System design: organizing rationale

Coverage extends beyond interview diagrams and AI application serving. The sequence starts with requirements, then transport/contracts, data and distributed correctness, then asynchronous systems, data platforms, reliability, security, infrastructure evolution, complete designs and advanced capstones.

| Coverage area | What the learner must eventually be able to do |
| --- | --- |
| Requirements, domain and low-level design | State measurable requirements and invariants; choose boundaries; write ADRs; implement maintainable interfaces and state machines; defend tradeoffs. |
| OS/networking/client/service contracts | Follow a request and its resource lifetime; reason about transport, discovery, proxies, caches, APIs, session state, deadlines, quotas, IDs, client performance and accessibility. |
| Storage and transactions | Choose a data model and access path; explain storage durability, indexing, compaction, isolation anomalies, backups, recovery and specialized stores. |
| Distributed correctness | Analyze histories, consistency, quorums, consensus, fencing, sharding, atomic commit, convergence, membership and multi-region recovery. |
| Messaging and workflows | Specify delivery/ordering boundaries, bounded queues, outboxes, compensation, durable jobs, replay and state evolution. |
| Data/analytics/AI interfaces | Design analytical storage, distributed queries, stream windows, data contracts, retrieval and model-serving boundaries while linking deeper AI teaching owners. |
| Reliability and performance | Define SLOs, diagnose latency, measure honest workloads, shed overload, operate incidents and validate correctness/failure hypotheses. |
| Security and privacy | Threat-model identities, permissions, keys, tenants, abuse, deletion and supply chains, with current primary specifications and jurisdictional applicability. |
| Infrastructure and evolution | Operate deployment, orchestration, private networks, control planes, IaC, migrations, backfills, canaries and cost decisions. |
| Complete design studios | Apply the above to redirects, chat, feeds, payments, ticketing, sync, video, geospatial dispatch, telemetry, object stores, workflow platforms and IoT. |
| Advanced depth and capstones | Understand adversarial coordination, real-time boundaries and hardware-aware paths; build/review evolving services, replicated storage and a secure event platform. |

General language syntax, DSA, mathematics, detailed ML model training and GPU architecture stay in their existing modules. System-design lessons teach the architectural use, contracts and failure consequences. Existing MLOps lessons retain their identities; specialized AI system design is linked rather than treated as newly implemented here.

## How future authors should teach these topics

Run the topic preflight and use the selected delivery mode. A compact catalogue brief is a starting contract, not a substitute for the full manuscript, claim verification, examples, practice/solutions, visual specifications and implementation review required by the standard.

For **finance**, begin with the instrument, contractual obligations, units, information available at the decision time and the business question. Trace cash, positions, financing, orders and risk explicitly. Separate a forecast from a position, a position from an executable fill, and a paper result from evidence of a live strategy. Synthetic data and simplified books must be labeled. Never silently imply realistic queue positions, executable liquidity, independent trials or measured HFT latency. Each pricing example should state measure, discounting, conventions and numerical error; each research example should state universe, vintages, selection history, splits and costs. Explain why a profitable-looking conclusion might fail, then let the learner diagnose a changed case independently.

For **systems**, start with one user workflow and a correct simple implementation. State invariants and workload before adding components. Follow requests, messages and persistent state through both success and failure. Use observable histories to distinguish promises such as atomicity, isolation, durability, ordering, idempotency and delivery. Show the workload or fault that justifies a new component and the complexity it introduces. Use measured performance evidence with environment and workload; label schematic curves as qualitative. Build operational, security, migration and cost reasoning into the design instead of appending a generic checklist.

Choose visuals for the mechanism: cash-flow and collateral timelines, order queues, book replays, volatility surfaces, portfolio exposures, packet and clock traces; or interleaved histories, replica logs, state machines, cache races, shard movement, queue occupancy, control loops and trust-boundary maps. Several mechanisms may need several diagrams or focused labs. Do not impose a generic slider box or a one-lab quota. Preserve accessibility, responsive labels, exact legend semantics, reset behavior and independent checks.

Practice progresses from prediction/hand traces to implementation or investigation, fault diagnosis, changed constraints and a defended decision. Provide independently reasoned solutions and explicit success criteria. Capstones need artifacts another person can reproduce, including failed cases and unsupported claims. Curate annotated official docs, papers, articles and genuinely useful videos/lecture playlists during lesson research; the module source list below is not a completed references section for every lesson.

## Scope research and source limitations

Sources were inspected on 17 September 2026 for coverage and current anchors. The organization, topic boundaries and proposed activities are curriculum design judgments, not a reproduced external syllabus or certification. Recheck exact mechanisms, versions, venue protocols and applicable rules during each lesson's research phase. Generic course indexes anchor scope; they do not verify every named method or subtopic.

| Primary resource | Why it was used |
| --- | --- |
| [CMU MSCF curriculum](https://www.cmu.edu/mscf/academics/curriculum) | Professional breadth across financial mathematics, data science, markets, risk, computing and communication. |
| [QuantLib documentation](https://www.quantlib.org/docs.shtml) | Starting references for contract conventions, curve construction, pricing and numerical implementation. |
| [CME clearing structure](https://www.cmegroup.com/education/courses/clearing/clearing-market-structure) and [pre-trade risk](https://www.cmegroup.com/solutions/market-access/globex/trade-on-globex/pre-trade-risk-management.html) | Clearing, collateral, operational responsibilities and electronic risk controls. |
| [FIX standards](https://fixtrading.org/standards/) and [Nasdaq ITCH specification](https://classic.nasdaqtrader.com/content/technicalsupport/specifications/dataproducts/NQTVITCHSpecification.pdf) | Protocol and trade-lifecycle scope; exact messages require version-specific inspection during authoring. |
| [Linux timestamping](https://www.kernel.org/doc/html/latest/networking/timestamping.html), [DPDK](https://www.dpdk.org/about/) and [NSE institutional connectivity](https://www.nseindia.com/static/invest/first-time-international-investor-connect-to-nse) | Measurement, packet-path and venue-access engineering boundaries. |
| [FINRA algorithmic trading](https://www.finra.org/rules-guidance/key-topics/algorithmic-trading), [SEC market access](https://www.sec.gov/rules-regulations/2011/06/risk-management-controls-brokers-or-dealers-market-access), [CFTC](https://www.cftc.gov/LawRegulation/index.htm), [ESMA Article 17](https://www.esma.europa.eu/publications-and-data/interactive-single-rulebook/mifid-ii/article-17-algorithmic-trading), [SEBI](https://www.sebi.gov.in/legal.html) | Distinguish jurisdictions and primary rule/guidance owners; no universal thresholds asserted here. |
| [Basel Framework](https://www.bis.org/committees/bcbs/basel-framework) | Risk/collateral/counterparty scope; actual regulatory applicability depends on institution and jurisdiction. |
| [MIT distributed systems](https://pdos.csail.mit.edu/6.824/schedule.html), [CMU database course with lectures/videos](https://15445.courses.cs.cmu.edu/fall2025/schedule.html), [Raft](https://raft.github.io/) and [Spanner](https://research.google/pubs/spanner-googles-globally-distributed-database/) | Storage, distributed mechanisms, original-system reasoning and implementation practice. The CMU course is a dated archive, not a claimed latest release. |
| [Google SRE book](https://sre.google/sre-book/table-of-contents/) and [workbook](https://sre.google/workbook/table-of-contents/) | Service operation, reliability, diagnosis and practical production responsibilities. |
| [PostgreSQL concurrency](https://www.postgresql.org/docs/current/mvcc.html), [Kafka design](https://kafka.apache.org/41/design/design/), [Kubernetes](https://kubernetes.io/docs/concepts/overview/) and [OpenTelemetry](https://opentelemetry.io/docs/concepts/) | Concrete product contracts supporting general mechanisms; Kafka link is version-pinned and must not be presented as perpetually latest. |
| [HTTP semantics](https://www.rfc-editor.org/rfc/rfc9110.html), [OAuth security BCP](https://www.rfc-editor.org/rfc/rfc9700.html), [OpenID Connect](https://openid.net/specs/openid-connect-core-1_0.html) and [NIST privacy framework](https://www.nist.gov/privacy-framework) | Standards and frameworks for contracts, identity, security and privacy distinctions. |
| [MDN performance](https://developer.mozilla.org/en-US/docs/Learn_web_development/Extensions/Performance), [W3C accessibility](https://www.w3.org/WAI/standards-guidelines/wcag/) and [distributed-systems patterns](https://martinfowler.com/articles/patterns-of-distributed-systems/) | Client-experience requirements and concrete mechanism patterns. |

One actionable update was found: the existing model-governance title names historical SR 11-7, while [Federal Reserve SR 26-2](https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm) superseded it in April 2026. The stable ID is preserved, its plan now distinguishes the current guidance, and a [destination note](../teaching/topic-notes/explainable-ai-model-governance-in-finance-shap-sr-11-7.md) requires the eventual author to assess scope and a compatible display title. No lesson was rewritten.

OWASP's ASVS page and the initial Nasdaq/SEC guessed URLs were not retrievable through the research tool. They are not claimed as reviewed evidence. The actual Nasdaq specification and SEC rule page were found through official search results; they are scope references, not claims that every document section was read. The AWS Builders Library redirected to an unreadable page and was not used as a substantive source.

## Maintenance, source ownership and verification

- Catalogue: `src/learn/data/curriculum/quantitative-trading.js` and `system-design.js`, integrated by `track-definitions.js`. Quant topics are reorganized with an explicit guard against losing any existing title. Shared system topics resolve to the original object/ID.
- Brief schema mapper and scope sources: `professional-topic-plan.js` and `professional-curriculum-sources.js`. These are authoring data, not eager lesson imports. Future authored topic briefs still use semantic per-topic blueprint files.
- Guided paths and shelves: `learning-paths.js`. Finance now has an appropriate professional-finance shelf instead of being placed among brain/body topics. Counts remain route-derived.
- Conservation snapshot: `professional-modules-baseline.json`, taken before editing; purpose is stable-ID/module/phase-publication preservation, not an alternative live catalogue.
- Run `node scripts/verify-professional-curriculum.mjs --write-syllabi` after scoped syllabus edits. It checks preserved IDs, finance membership, coherent section sizes, recorded within-module prerequisite order and focused/full routes, then refreshes the two detailed lists.
- `--check-planning-boundary` additionally proves this increment left unrelated module order, lesson publication and phase checkpoints unchanged. Do not use that snapshot-specific assertion to block later authorized implementation.
- Also run `node scripts/verify-curriculum.mjs`, `node scripts/build-curriculum-inventory.mjs`, the application build and scoped browser checks. Refer to the completion evidence below for what actually passed.

### Initial expansion verification record (historical snapshot)

Catalogue conservation and within-module prerequisite checks pass. The full curriculum verifier and inventory regeneration pass with 29 modules, 1,394 unique topics, 546 individual briefs, 609 recorded prerequisite reviews and 9 paths. Publication and the two-phase delivery ledger match their pre-edit hashes.

The initial production build and generated-artifact freshness check passed with 64 scoped browser checks at 1440 and 390 pixels: resolved path counts; complete module outlines in syllabus order; one new planned-page sample in every section; prerequisite links; disabled completion for planned lessons; Previous/Next; existing and shared progress; page overflow; and on-demand outline requests. There were no page errors. This was integration and planning review, not content review of the then-218 module memberships. Final captures were taken after navigation and smooth scrolling settled; the earlier transient blank capture was replaced. Desktop trading and both mobile module captures were visually inspected. The live browser evidence file is refreshed by the follow-up review below.

The initial trading and system-design outline chunks are independent dynamic imports (552 and 513 gzip bytes respectively in this build); detailed investigation text remains outside compact navigation. These sample sizes do not claim to measure all future lesson payloads. [Final source and conservation evidence](professional-modules-verification.json) binds the source files, browser record, build manifest and unchanged phase/publication files. Four final captures are retained in the dedicated scratch evidence folder and referenced by digest; no historical scratch audit was reopened. Nothing was deployed.

### Named-concept follow-up verification

The [follow-up review](PROFESSIONAL-COVERAGE-REVIEW.md) supersedes the numeric/source state of the initial snapshot above. Current curriculum verification passes with 1,402 unique topics, 554 individual briefs and 617 recorded prerequisite reviews. All 226 professional-module memberships have named coverage; the verifier rejects orphaned, missing or duplicate scope records. The two-phase ledger and publication manifest remain unchanged.

The production build, artifact freshness check, ten representative title/subtopic search cases and **90 desktop/mobile browser checks** pass. The [current browser record](professional-curriculum-browser-evidence.json) includes every newly added page at both widths, planned concept lists, module-filtered named searches, empty results, path/module counts, sequence/navigation, preserved progress and lazy loading. Search did not request lesson or outline chunks. There were no page errors or horizontal overflows. HyperLogLog captures at 1440 and 390 pixels were visually inspected. The capture check now requires a sustained settled header after navigation rather than accepting a transient zero-scroll frame; no navigation runtime change was needed.

[Current source-bound evidence](professional-coverage-verification.json) records source hashes, the build/browser evidence and payload measurements. Six current captures are referenced by digest in the browser record; the four preceding module captures were refreshed in place. Initial snapshot hashes remain historical, not assertions about these refreshed outputs. No lessons were written or published and nothing was deployed.
