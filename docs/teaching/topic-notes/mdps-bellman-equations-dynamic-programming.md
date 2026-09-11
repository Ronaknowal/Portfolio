# Authoring notes: MDPs, Bellman Equations & Dynamic Programming

Canonical topic ID: `mdps-bellman-equations-dynamic-programming`.

## 2026-09-11 — Preserve the information order when extending a one-step decision tree

- Status: open.
- Origin: Mathematics53, [Decision Theory design](../DECISION-THEORY-LESSON-DESIGN.md), observation-then-action and contingent allocation branches.
- Current destination: exact inventory says planned, individual design required. No complete source or verified MDP teaching is implied by its title.
- Learning benefit: a learner who can choose an action using a fixed posterior should see precisely what changes when the action changes the next state, information and future choices. Do not treat repeated application of a one-step threshold as a derived optimal sequential policy.
- Proposed treatment: begin with a short horizon whose observation/action/reward timing is explicit; let the learner enumerate feasible history-dependent policies, then compress the decision using a sufficient Markov state and derive the finite Bellman recursion. Explain when beliefs must become the state because the underlying state is hidden. Contrast information purchased before acting with information revealed afterward.
- Concrete bridge: the implemented origin tests one of six independent items, then allocates two quarantine slots. In its stated additive loss model, testing item 4 with prior fault probability .2 has gross information value 10.4; testing item 5 is exactly co-optimal, and the code selects 4 only by a lower-index tie-break. The highest-prior item need not be the most useful test. That is a finite observation/action tree, not an MDP algorithm or a proof that greedy repeated tests optimize a larger budget.
- Prerequisites and limits: expectations, conditional probability, a policy's information set, transition probabilities and a stated horizon/objective. The complete originating lesson and its [author verification](../DECISION-THEORY-VERIFICATION.md) now exist; independent review and integrated acceptance remain separate. Infinite-horizon convergence, exploration and off-policy causal evaluation need their own conditions and owners.
- Evidence: [design-only exact enumeration](../evidence/decision-theory-design-checks.json), checking all64 item worlds for each candidate test; [MIT tree-construction transcript](https://ocw.mit.edu/courses/ids-333-risk-and-decision-analysis-fall-2021/1-2L0LSIpoNnJ9fZabcMEhn2na6aPrj8o_transcript.pdf), full two-page transcript read. These support the local bridge, not a complete Bellman theorem.
- Resolution: future destination author must reassess, adapt or reroute against its actual scope and references. No MDP body or shared curriculum was changed.
- Implementation evidence: the complete displayed capstone program verifies every contingent policy over all 64 worlds; the final [author packet](../evidence/decision-theory-author-review.json) links native/model/browser checks. This adds a verified one-step bridge without marking the MDP destination complete.
