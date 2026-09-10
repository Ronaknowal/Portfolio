# Authoring notes: Regularization (L1, L2, Elastic Net, Dropout)

Canonical topic ID: `regularization-l1-l2-elastic-net-dropout`

## 2026-09-10 — A parameter penalty breaks a prediction symmetry

- Status: open
- Origin: Non-Convex Optimization Landscape, section 5; [design](../NONCONVEX-LANDSCAPE-DESIGN.md).
- Destination and rationale: this existing lesson owns penalty choice and changed optimization objectives. The origin only distinguishes identical predictors from different parameter costs; the full regularized optimum is best developed here.
- Existing coverage: the live regularization lesson's L2 explanation discusses weight shrinkage and a parameter penalty. This exact factor-symmetry/whole-objective distinction was not found in the scoped inspection. The exact topic-plan command confirms this stable ID and no prior note.
- Idea: for predictor abx and data cost F=½(ab−1)², every ab=1 represents the same zero-data-loss function. A penalty λ(a²+b²) changes along that curve. Its smallest value on the curve is at a²=b²=1, but this is not generally the optimum of the full regularized objective.
- Proposed treatment: a deeper changed-parameterization example after introducing L2. Let p=ab. Because a²+b²≥2|p| and the bound is attainable, minimize ½(p−1)²+2λ|p| over p. For λ≥0, p*=max(1−2λ,0); for 0≤λ<½ balanced factors ±√(1−2λ) attain it (at λ=0 other ab=1 pairs also minimize), while λ≥½ gives a=b=0. Explicitly distinguish the constrained zero-data-loss curve from the full problem. Reverify the algebra and actual numeric implementation before teaching it.
- Learning benefit: predict how changing a penalty changes both preferred parameter scale and the fitted function; avoid claiming that preserving predictions automatically preserves a regularized objective.
- Prerequisites/boundaries: scalar minimization, absolute-value subgradient or separate p≥0/p≤0 cases, and the elementary square inequality. This is a calculated teaching example, not evidence that balancing layers universally improves generalization.
- Evidence: original two-factor derivation in the origin, plus Dinh et al. sections 3–4 for the distinction between equivalent predictions and parameter geometry: https://proceedings.mlr.press/v70/dinh17b.html (inspected 10 September 2026). The full penalized example is a proposal, not implemented destination content.
- Resolution: not yet reviewed by the destination author; no destination body changed.
- Implementation/verification links: origin's model and native record will appear in [NONCONVEX-LANDSCAPE-VERIFICATION.md](../NONCONVEX-LANDSCAPE-VERIFICATION.md).
