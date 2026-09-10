# Authoring notes: Queueing Theory (M/M/1, M/G/1, Little's Law)

Canonical topic ID: queueing-theory-m-m-1-m-g-1-little-s-law

## 2026-09-10 — Carry process assumptions into actual congestion models

- Status: open.
- Origin: mathematics 33, [Stochastic Processes design](../STOCHASTIC-PROCESSES-LESSON-DESIGN.md). The origin is designed, not yet implemented or verified.
- Destination and ownership rationale: the origin teaches finite-state transition and arrival clocks; Queueing owns service, waiting, occupancy, stability and their infinite-state or finite-buffer consequences.
- Existing coverage inspected: the original Queueing body sections 2–5 give rho, M/M/1, Little's Law and M/G/1 formulas; section 6 mentions batches/retries; section 7 lists measurement checks. It does not derive the birth/death stationary balance, visibly separate independent request routing from load-dependent routing, or account for censored service/exposure observations. This is source inspection, not a current quality verdict on future revisions.
- Learning benefit: connect a process assumption to the quantity the learner is actually predicting. A Poisson-looking marginal count does not establish independent arrivals; a time average is not automatically an average taken only at arrivals or jumps.
- Proposed treatment: derive the M/M/1 birth/death balance and normalizability for lambda<mu; name what changes at/above the boundary without importing finite-chain positive-recurrence guarantees. Include a small arrival/service/queue timeline. Compare independent Bernoulli routing with deterministic or occupancy-dependent dispatch before using independent-arrival formulas. Explain service-time/exposure censoring and an appropriate temporal diagnostic when fitting from traces.
- Example: independent splitting of a rate-lambda Poisson input gives marginal rate-p-lambda streams under independent marks; choosing the currently shortest queue is a different state-dependent mechanism. Two-state clocks in the origin distinguish jump proportions from time occupancy; Queueing can use this bridge for appropriately conditioned observations and any claimed arrival-sees-time-average result.
- Boundaries: do not promise every routing system is Poisson or introduce an unexplained theorem about arrival sampling. PASTA, general renewal residual-life effects and feedback need their own assumptions, sources and examples if included.
- Evidence: [Gallager Chapter2](https://ocw.mit.edu/courses/6-262-discrete-stochastic-processes-spring-2011/3a19ce0e02d0008877351bfa24f3716a_MIT6_262S11_chap02.pdf), sections 2.3–2.5, read 10 September 2026; actual current Queueing source read. The origin will verify its own finite clocks. Queueing-specific theorems and engineering recommendations remain for the destination author to investigate.
- Resolution: not yet reviewed by destination author; no destination body changed.
- Implementation/verification links: none yet.
