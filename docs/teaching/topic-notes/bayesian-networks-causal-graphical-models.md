# Bayesian Networks & Causal Graphical Models — destination notes

## 2026-09-10 — Repair the education/training adjustment exercise

- Status: **resolved 16 September 2026** by the topic's phase-two rewrite. See the resolution at the end of this note. The original entry is kept verbatim below as the record of what was found.
- Origin: scoped Causal Inference & Do-Calculus design, `docs/teaching/CAUSAL-INFERENCE-LESSON-DESIGN.md`. Exact topic-plan CLI confirms this published owner and no existing note. No Bayesian Networks body was changed.
- Actual source: `src/learn/data/topics/bayesian-networks-causal-graphical-models.jsx`, exercise4 around lines991–996. Its stated edges are S→E, S→Y, E→T and T→Y. The prompt asserts that E alone is not a valid backdoor set, while its answer contradicts itself several times and eventually admits validity.
- Correct reasoning for the stated graph: the sole backdoor path T←E←S→Y is blocked by conditioning on E, a non-collider that is not a descendant of T. Both {E} and {S} are valid backdoor sets, subject to required data support. Conditioning on E creates no additional path in this graph. There is no general preference justified merely by calling S more upstream; precision, cost and assumptions need their own evidence.
- Proposed treatment: rewrite the prompt as a comparison or diagnosis, draw the actual path, and provide one coherent answer. If the author intends a counterexample where E is invalid, change and justify the graph explicitly rather than retaining contradictory prose.
- Adjacent scope concern: section9.3's claim that an unobserved common cause means no observed set can block any relevant path is too broad. An observed non-collider elsewhere on a backdoor path may suffice; other graphs permit frontdoor identification. Review against the exact graph and distinguish lack of a valid backdoor set from general nonidentification.
- Sources/validation: Pearl's d-separation/backdoor definitions in `https://ftp.cs.ucla.edu/pub/stat_ser/r416-reprint.pdf`, printed pp2517–2519, plus the original four-edge graph's path enumeration. The current causal lesson will derive the criteria locally; the receiving author should independently execute the changed exercise and check surrounding graphical claims.
- Boundaries: this is one concrete correctness discovery, not a full audit or authorization to rewrite the separate Bayesian Networks topic. Preserve its useful factorization/inference content and do not make it a hidden prerequisite of the causal foundations lesson.

### 12 September 2026 content-phase disposition

Assessed and adapted in the authorized research/write packet [design](../drafts/bayesian-networks-causal-graphical-models/design.md). The complete manuscript§6 explains why both{E}and{S}work for the original graph; practice5 adds E→Y explicitly so S-only becomes invalid. Both original and changed backdoor path calculations were executed in the packet's offline program. Section7 corrects the broad unobserved-confounder claim and supplies an exact frontdoor example. Status remains open for phase-two implementation and independent confirmation; the current published JSX was not edited by the content-only request.

## 2026-09-16 — Resolution

Both findings are closed as implemented. The implementing author verified the note's reasoning independently by enumerating every path in the stated graph rather than accepting it, and the integration owner confirmed that enumeration.

**The backdoor finding was correct.** With edges S→E, S→Y, E→T and T→Y there are exactly two T–Y paths: the directed T→Y, and the single backdoor path T←E←S→Y. On that path E is a chain and S is a fork, so both are non-colliders; the graph's only collider is Y, which is an endpoint of the path and never an interior node of it; and T's only descendant is Y. Therefore **{E}, {S} and {E, S} all satisfy the backdoor criterion, the empty set does not, and conditioning on E opens nothing**. The published exercise's prompt asserted the opposite, reversed itself three times, then conceded validity and invented a preference for S as "more upstream". The replacement body states that both sets work and that being further upstream is not itself a reason, gives the counterexample as a practice item, and asserts the result in the model verifier rather than in prose alone.

**The §9.3 finding was correct, with one qualification the original note did not make.** The sentence claiming no observed set can block the relevant paths holds only when the unobserved variable is a *direct* common parent of treatment and outcome; where the common cause is ancestral and routed through an observed non-collider, that path is blocked. So the claim was indeed too broad. But the note's second concern was aimed slightly wide: that section already mentioned frontdoor identification. The overbroad parts were the heading and that one sentence, not the frontdoor material. The replacement body carries the corrected claim.

Neither correction was inherited on trust: the d-separation verdicts behind them are recomputed in `scripts/verify-bayesnet-models.mjs` by ancestral moralisation — a different theorem that never enumerates a path — across 1,004 preset cases and 124,232 cases on 742 generated DAGs.

This note is closed. It does not authorize further work on the causal-inference topic or any audit beyond the one exercise and the one section it named.
