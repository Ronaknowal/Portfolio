# Authoring notes: Second-Order Methods, L-BFGS, K-FAC, Shampoo and Natural Gradient

Canonical topic ID: `second-order-methods-l-bfgs-k-fac-shampoo-natural-gradient`

## 2026-09-13 — Distinguish curvature from matrix-gradient geometry

- Status: open
- Origin: [Advanced Optimizers content design](../drafts/advanced-optimizers-lion-sophia-prodigy-schedule-free/design.md), manuscript§9.
- Destination and rationale: this existing owner teaches Hessians, Fisher/GGN and matrix preconditioners; it is the best place for a deeper comparison with Muon's matrix-valued momentum transform. The origin supplies a concise bridge while keeping Lion/Sophia/Prodigy/Schedule-Free central.
- Existing coverage checked: actual stable ID/source exists; a scoped search of its source and current catalogue did not find Muon. This is an absence of that named extension, not a full re-audit of the implemented second-order lesson.
- Learning benefit: predict why entrywise sign, inverse-curvature scaling and singular-direction flattening are different operations. Prevent calling every matrix optimizer a Hessian-based method.
- Proposed treatment: optional deeper matrix-geometry branch after the appropriate existing preconditioner explanation. Start with M=[[2,1],[1,2]]: entrywise sign is rank1, ideal polar factorI preserves both singular directions. Compare actions on[1,1] and[1,−1]; then introduce finite Newton–Schulz approximation, momentum/Nesterov, shape/scale choices and parameter grouping. Consider fuller Muon treatment or rerouting only after reading the destination's actual current coverage and broader owner map.
- Prerequisites and boundaries: SVD/polar geometry and the distinction between gradients and curvature; no claim that the ideal SVD operation equals a fixed finite-iteration implementation. Do not infer all2D embeddings should use one hidden-layer recipe or that there is a universal quality/speed advantage.
- Evidence: [PyTorch2.14 Muon documentation](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Muon.html), rule/options/example read13September2026; [origin calculated ideal-polar fixture](../drafts/advanced-optimizers-lion-sophia-prodigy-schedule-free/calculated-inputs.json). The origin did not execute/benchmark Muon. Later author should examine the primary algorithm and current variants appropriate to the chosen scope before deep treatment.
- Resolution: origin content includes the short bridge; destination author has not yet assessed/implemented the extended treatment. Keep open through phase-one drafting and reason on the proposal rather than inserting it blindly. No title/ID change is requested now.
- Implementation/verification links: none for the destination extension; this note does not reopen completed work without a user request.
