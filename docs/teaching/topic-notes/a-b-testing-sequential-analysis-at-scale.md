# Authoring notes: A/B Testing & Sequential Analysis at Scale

Canonical topic ID: `a-b-testing-sequential-analysis-at-scale`

## 2026-09-10 — Continue from a finite optional-stopping counterexample

- Status: open
- Origin: [Hypothesis Testing & Confidence Intervals design](../HYPOTHESIS-TESTING-CONFIDENCE-DESIGN.md), section8 of its implemented lesson.
- Destination and ownership rationale: the exact inventory command resolves this planned topic in Production Evaluation Systems. It is the appropriate owner for operational sequential experiments; the foundation lesson teaches the failure mechanism and does not need a full monitoring framework. No destination body was changed.
- Idea and learning benefit: distinguish the event “reject at the predeclared final sample” from “ever reject before the horizon,” then teach a procedure whose error guarantee actually covers the stopping rule. Also distinguish multiple independent metrics from correlated successive looks at one growing sample.
- Existing coverage: destination is planned and needs individual design. The origin enumerates all 4,096 fair-coin sequences through 12 tosses: a two-sided exact-binomial test at level .05 rejects at the final look with probability 158/4,096, whereas inspecting every prefix rejects at least once with probability 290/4,096. This is an exact finite teaching example, not an empirical production benchmark.
- Proposed treatment: begin with the shared-path dependence and event union, then choose appropriate depth for alpha-spending/group-sequential boundaries or time-uniform confidence sequences. Explicitly define information available at each look, the monitoring rule, allocation/randomization, sampling unit, null and power target. Extend to metrics and clustered users only after the guarantee's assumptions are explicit.
- Explanation/example: contrast a changed fixed horizon with a predeclared stopping boundary; visualize sample-path crossings and explain why treating looks as independent tests gives a different probability. Make uncertainty after adaptive stopping a hands-on checked exercise, not merely a warning.
- Prerequisites and boundaries: conditional probability, test size, repeated-sampling coverage and independent units. Research current primary sequential-inference sources before selecting an implementation. Do not claim optional stopping makes every statistical method invalid, or that a fixed-sample Bonferroni adjustment for metrics alone licenses unlimited monitoring.
- Evidence: origin `hypothesis-testing-models.js` exact prefix dynamic program independently compared with exhaustive bit-string enumeration by `scripts/verify-hypothesis-testing-native.py`; [origin verification](../HYPOTHESIS-TESTING-CONFIDENCE-VERIFICATION.md). No sequential method was implemented or technically endorsed by this discovery.
- Resolution: not yet assessed by receiving author; preserve stable identity and module order.

