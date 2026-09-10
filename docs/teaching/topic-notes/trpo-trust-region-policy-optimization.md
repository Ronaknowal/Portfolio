# Authoring notes: TRPO (Trust Region Policy Optimization)

Canonical topic ID: trpo-trust-region-policy-optimization

## 2026-09-10 — Connect local natural-gradient geometry to a policy constraint

- Status: open
- Origin: [Second-Order Methods design](../SECOND-ORDER-METHODS-DESIGN.md), sections 4 and 7 of the implemented lesson.
- Destination and rationale: the live catalogue places TRPO in Deep RL and its exact inventory reports a planned body needing individual design. No existing destination note was present. Policy distributions, state visitation and performance bounds belong here; the optimization lesson supplies only local parameter/distribution geometry.
- Learning benefit: explain what changes when a Fisher-scaled direction becomes a practical constrained policy update. Prevent readers from treating equal infinitesimal natural directions as identical finite updates under arbitrary nonlinear coordinates.
- Proposed treatment: locally introduce the policy distribution and relevant expectation; derive the quadratic KL/linear objective subproblem, then show the actual constraint measurement, line search and finite-step limitations. Connect HVP/CG to applying the curvature operator without allocating a dense matrix. Distinguish an average sampled KL constraint from any stronger theoretical condition used in a policy-improvement bound.
- Example: reuse a small two-action Bernoulli policy to compare probability and logit coordinates, predict a local tangent, then evaluate both finite endpoints with the actual KL and surrogate change. Extend to multiple states only after specifying how states are weighted.
- Prerequisites and boundaries: probabilities, expectation, policy objectives, gradients, local Taylor expansion, positive definiteness/damping and iterative solves. The origin's six-variable CG demo is an SPD numerical fixture, not a complete TRPO implementation or proof.
- Evidence: Martens, https://arxiv.org/pdf/1412.1193, local-KL, invariance and empirical-Fisher passages inspected on 10 September 2026. Original TRPO derivations and actual implementation contracts still require scoped research by this topic's author; this note does not assert that they were reviewed here.
- Resolution: not yet reviewed by the destination author. Do not rename or expand the topic automatically; assess fit and sources first.
- Implementation/verification links: [origin verification](../SECOND-ORDER-METHODS-VERIFICATION.md); destination remains planned.
