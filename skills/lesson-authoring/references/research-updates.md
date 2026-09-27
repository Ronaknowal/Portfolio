# Research updates: keep the curriculum current without chasing every release

Use for a requested check of new research, models, methods, standards or established professional practice against the learning site. This is a curriculum assessment, not a general news digest or permission to implement every discovered idea. A scoped material update found during ordinary authoring can use the same criteria without triggering a whole-curriculum scan.

- [Define the check](#define-the-check)
- [Discover consequential changes](#discover-consequential-changes)
- [Assess evidence and learning value](#assess-evidence-and-learning-value)
- [Choose a curriculum action](#choose-a-curriculum-action)
- [Compare actual coverage and design the change](#compare-actual-coverage-and-design-the-change)
- [Save a usable decision](#save-a-usable-decision)
- [Continue, revisit and clean up](#continue-revisit-and-clean-up)

## Define the check

Resolve the user's scope: named model/paper, specific topic, module, related domain, or whole curriculum. Use an established task scope when clear; otherwise clarify the meaningful choice while inspecting the catalogue. An ambient browser page alone does not select the scope. Record the actual check date, research window, target domains, source classes inspected and any coverage limits.

For a follow-up, start from the previous scoped record and its unresolved questions. Search new developments since its cutoff with enough overlap to catch late publication, replication or adoption. For a first broad check, choose and state a manageable recent window appropriate to the field; expand when needed to evaluate an older idea's new evidence or a foundational omission. Publication date, release/version date, independent evaluation date and adoption date are different facts. Do not reject an important missing concept just because its original paper is older than the search window.

For a whole-curriculum request, map the actual modules to relevant research/practice channels and track which were examined. Inspect likely affected lessons after discovery instead of loading every full manuscript first. Stop at a supported scoped assessment; if interrupted, save a precise coverage frontier and next action. Never report “everything is current” for unexamined subjects or from a news search alone.

Default output is an evidence-based assessment plus durable notes/recommendations. Do not change published content, prepared checkpoints, catalogue entries or teaching completion merely because the scan found something. An explicit request to add topics, write updates or implement them authorizes the matching plan/write/full workflow without another invented approval gate. Respect a read-only request by returning findings without persisting edits.

## Discover consequential changes

Browse current sources when running this mode; memory, search snippets and the date on a landing page do not establish current evidence. Inspect the actual paper, specification, release notes, implementation documentation or model/system report. When current source access fails, state the limitation and leave unsupported conclusions unresolved.

Choose channels for the domain and the uncertainty:

- Original papers/proceedings, substantive surveys and their corrections or retractions for mechanisms, formal results, limitations and replicated findings.
- Maintained official implementations, framework/compiler/runtime documentation and versioned release notes for practical behavior, defaults, deprecations and engineering use.
- Independent evaluations, reproduction artifacts and documented downstream systems for robustness and real use. Check the connection to the exact method/version being considered.
- Domain standards and relevant authoritative technical guidance where they change what should be taught. A clinical or physical deployment claim needs its own evidence; a benchmark or simulation is not a substitute.

Use research indexes, reputable explainers, videos, community discussion, stars and citation counts to discover candidates; follow them to claim-supporting primary material. Multiple summaries of the same announcement are one evidence chain, not independent confirmation. Include negative results, limitations, replication failures, corrections and competing explanations—not only new benchmark leaders.

Search by the underlying mechanism and synonyms as well as product/model names. Deduplicate a family of renamed models or minor variants into the transferable idea. A model with undisclosed internals does not justify guessing its architecture from outputs, marketing language or another similarly named model. Evaluate the disclosed parts and label unknowns. Avoid executing unfamiliar research repositories just to perform a literature assessment; limited calculations/reproduction probes are useful when necessary to resolve a claim and appropriate to the task.

## Assess evidence and learning value

Make a reasoned judgment along these separate dimensions. Do not collapse them into a popularity score or require a fixed number of citations, papers, users or months of age.

| Dimension | Questions to resolve |
| --- | --- |
| Technical credibility | What is the precise claim and evidence? Are mechanism, assumptions, evaluation and limitations inspectable? For formal work, examine the stated result/proof and assumptions; for empirical work, examine methodology and reproducibility. Peer review or a prestigious author alone is not proof. |
| Independent support | Are there relevant replications, independent analyses or downstream implementations? What do they actually confirm, and what contradicts the claim? Separate authors' reports from independent findings and your own derived reasoning. |
| Distinct learning contribution | Does it introduce a useful mechanism, explanation, failure mode, capability, constraint or tradeoff? Would learning it change what a researcher or engineer can reason about or build? A renamed model or a small leaderboard gain alone usually does not. |
| Practical or conceptual importance | Does it correct an existing claim, change a standard implementation, remove an important bottleneck, establish a meaningful limitation, or enable a substantial class of problems? An important theoretical result can qualify without deployment. |
| Adoption and maturity | Is the mechanism documented in maintained tools, credible downstream projects or multiple real systems? Is use independent and current? Distinguish actual deployment, library availability and experimental integration. Adoption strengthens relevance but is neither necessary nor sufficient for truth. |
| Curriculum fit | Who needs the idea, what prerequisites does it require, and how much genuine new teaching is involved? Is it already adequately taught under another name? Does it belong in core, a specialist branch, a frontier discussion, a project, or nowhere yet? |

**Promote evidence-backed learning value, not novelty alone.** A convincing important result with little adoption may justify inclusion, particularly a clearly scoped advanced branch. Explain its importance and evidence directly; do not claim consensus or practical maturity it lacks. A widely adopted but imperfect technique may deserve teaching because learners will encounter it, with accurate limits and appropriate alternatives. Popularity does not make an unsupported superiority or safety claim acceptable.

Treat empirical performance claims at their actual scope: data/splits, leakage or contamination checks where relevant, baseline tuning, training/inference budget, hardware, precision, memory, latency versus throughput, uncertainty and meaningful task success. Compare like with like. Do not transform one author-reported benchmark into a universal “better” recommendation. Code/weights availability helps inspection but does not establish that a result reproduces or generalizes. Avoid a universal “proven” label; state what is supported, by whom, and under which conditions.

Adoption is not a compulsory gate, and independent replication is not a mechanical prerequisite for every kind of knowledge. A well-supported mathematical insight, compelling new mechanism or important counterexample may be worth teaching before broad use. Conversely, when a claim's importance depends on an unconfirmed empirical gain, frame the bounded research result honestly or defer the recommendation. Do not put every speculative preprint into a frontier section merely to avoid excluding it.

## Choose a curriculum action

| Decision | When it fits |
| --- | --- |
| **Correct/update existing material** | A current claim is false/outdated, an applicable API/default changed, a supported limitation was omitted, or a practical implementation needs revision. Identify the affected statement/example and replacement; prioritize misleading core guidance. |
| **Extend an existing topic** | The established mechanism is still the right owner; add the new variant, comparison, application, derivation, optimization or failure analysis at its point of need. A model name alone rarely warrants a new topic. |
| **Propose a new topic** | A substantial distinct concept has defensible evidence and learning value, cannot be taught adequately as a scoped extension, and has a coherent prerequisite/sequence position and independent outcomes. Explain why existing owners are insufficient. |
| **Add a bounded advanced/frontier branch or project connection** | An important inspectable idea deserves teaching with explicit unresolved questions, or an established combination of ideas is best learned through an end-to-end build. Preserve the distinction between the mechanism and the unconfirmed claim. |
| **Already covered / no content change** | The exact useful concept and relevant limits are taught at adequate depth. Link the checked section; a new brand, citation or example is not automatically a curriculum gap. |
| **Defer and revisit** | Plausible importance, but material evidence or implementation clarity is missing. Save a concrete trigger such as a public method, independent replication, resolved contradiction or stable API—not an indefinite “check again.” |
| **Do not include** | Unsupported claims, little transferable learning, redundant variants, irrelevant scope, or disproportionate complexity without a learning benefit. Record a concise reason when it would prevent repeated rediscovery. |

Prioritize correctness repairs and missing enabling foundations, then consequential mechanisms/practices, then justified specialist/frontier depth. Preserve still-valid foundational methods and their tradeoffs. New work need not replace older material that remains useful. There is no quota of new topics or updates; a well-supported “no change needed in the checked scope” is a valid result.

## Compare actual coverage and design the change

For each candidate worth deeper assessment, inspect the catalogue, synonyms, prerequisites, topic notes and actual published/prepared sections. Classify coverage as absent, mentioned, planned, prepared, taught, or verified in the relevant implementation; do not infer it from a title or past completion label. Name the inspected version/file/section and unresolved inspection limits. Check both a pending manuscript and the published lesson when they differ, so a future finish agent does not restore outdated material.

For software changes, compare the lesson's pinned/tested version, explicit arguments and supported setup with the changed release. An accurately labeled older-version example may remain correct; choose a current-version comparison or migration note when that serves the learner instead of automatically upgrading working code. Correct claims that misleadingly imply current or version-independent behavior. When migration is justified and authorized, verify its semantic effects and update the prepared and implemented routes that actually depend on them.

Choose one primary owner and explicit bridges to related topics. Prefer the smallest coherent teaching change that fully explains the contribution, not the fewest words or smallest file diff. For a new topic, state proposed module, sequence, prerequisites, scope, outcomes and why an extension is inadequate. Use the unresolved topic-note inbox until an actual catalogue identity is created. Do not invent a canonical ID or reorder the curriculum based on publicity or publication availability.

Make each recommendation actionable: exact teaching gap; simple intuition and motivating limitation; mechanism/derivation; example or counterexample; suitable diagram/live investigation; scratch implementation ownership and optimized/library routes where relevant; changed practice and validation needs. Distinguish adding a citation from revising an explanation, a numeric model, runnable code or a whole lesson. Retain full depth through the normal [teaching](teaching.md), [examples/code](examples-and-code.md) and [visual](visuals-and-labs.md) standards when the update is authored. A research assessment is not itself a completed manuscript or a verified implementation.

## Save a usable decision

Use one existing scoped research/design record when available; otherwise use the location in [the repository adapter](portfolio-adapter.md#research-update-records). Keep the actual evidence and decision there, and route actionable discoveries to the normal destination notes. Avoid copying a long paper summary into every affected topic or creating a second implementation queue.

At minimum, preserve:

| Record | Required content |
| --- | --- |
| Scope and coverage | Actual as-of date, window, checked domains/topics/source channels, important exclusions or inaccessible sources, remaining work. Distinguish scan coverage from claim verification. |
| Candidate identity | Stable mechanism/name aliases, exact paper/model/software version, original date and relevant later evidence dates. |
| Evidence and uncertainty | Direct sources with section/table/commit/version locators where useful, claims they support, what was actually read/run, independence and material limitations/contradictions. |
| Curriculum decision | What is new, why it matters, adoption evidence if relevant, inspected existing coverage, action/depth/priority with reasons, exact owner or proposed owner, recommended teaching changes. |
| Handoff | Links to destination notes, current disposition, concrete next action or revisit trigger. Preserve prior consequential decisions when new evidence changes them. |

Keep accepted and unresolved important findings visible. Compress routine exclusions into a short reasoned summary rather than an endless paper database. “Unsupported,” “not inspected,” “not relevant” and “already covered” are different conclusions. Do not describe a scan as exhaustive, a recommendation as implemented, or author-reported evidence as independently verified.

Return a concise prioritized table of update/extension/new-topic recommendations, affected topics, reasons and evidence links, followed by deferred items with triggers and the scan's actual limits. Surface a substantive correction promptly rather than burying it among optional ideas. A no-change result still identifies what was checked.

## Continue, revisit and clean up

Research decisions are not new phase statuses. A proposed update does not invalidate or rewrite historical completion by itself. Use existing destination-note statuses: actionable recommendations stay open, deferred ideas have a revisit condition, and implemented/adapted decisions link actual changed content and verification. If uncertainty cannot be resolved, preserve it rather than approving by default.

When writing/implementation is authorized, use the normal revision and two-phase ledger contract. Update the relevant prepared or production sources and their checkpoints only after the appropriate work and checks. Preserve previous revisions/evidence; never silently mutate a hash-bound manuscript or mark a new topic complete from its proposal. An explicitly combined “check and implement” request proceeds within its stated scope without an additional approval ritual.

On later scans, reuse valid prior findings and check new evidence, changed sources and revisit triggers; do not re-review every rejected variant. Recheck version-sensitive claims when they inform an actual update. Close duplicated reminders once the receiving topic records its disposition. Retain consequential evidence/decisions and pending notes; remove disposable search exports and downloads under [handoff and cleanup](handoffs-and-cleanup.md).

This mode runs when invoked or as a scoped part of authorized authoring. Creating it does not launch a curriculum audit, continual monitoring or a scheduled task. If the user explicitly requests recurring checks, use the available automation workflow with a clear scope and notification rule; do not quietly schedule background work or imply freshness between runs.
