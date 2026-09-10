# Authoring notes: Differential Geometry & Riemannian Manifolds

Canonical topic ID: `differential-geometry-riemannian-manifolds`

## 2026-09-10 — Continue the distinction between a differential and its metric gradient

- Status: open
- Origin: `tensor-algebra-einsum-notation`, [design](../TENSOR-ALGEBRA-EINSUM-DESIGN.md), section8's fixed-space vector/covector/metric bridge.
- Destination and ownership rationale: this existing topic owns varying tangent spaces, Riemannian metrics and gradients. Tensor Algebra now supplies the finite-dimensional coordinate transformation prerequisite; a full tensor-field or covariant-derivative treatment would interrupt its computation outcome.
- Idea and learning benefit: distinguish the derivative as a linear functional from the vector representing it under a metric, and avoid treating an ambient Euclidean projection and a coordinate metric inverse as interchangeable formulas.
- Existing coverage: read `src/learn/data/topics/differential-geometry-riemannian-manifolds.jsx` sections2–4. A varying metric and `grad_M f = G^-1 nabla f` are mentioned, alongside the ambient-sphere projection. Their chart-versus-ambient conventions and differential/covector distinction are not developed there yet.
- Proposed treatment: derive `df(v)=g^T G v` in a specified coordinate basis, hence the Riemannian-gradient coordinate vector is `G^-1` times the differential's coefficient column. Explain separately why an ambient Euclidean gradient is orthogonally projected for an induced-metric embedded sphere. Use a small oblique-basis example before varying G(x); if tensor-field differentiation is included, introduce the need for comparing tangent spaces before connection coefficients. Do not automatically expand into a full relativity course.
- Explanation/example: Tensor Algebra's shear S=[[1,1],[0,1]] gives G'=S^T S=[[1,1],[1,2]]. The functional f=[2,-1] has new coefficients f'=[2,1]. Applying G'^-1 gives the metric-gradient components [3,-1], which reconstruct to old vector [2,-1]. This explains why covector coefficients and gradient vector coordinates differ even though their pairing represents the same change.
- Prerequisites and boundaries: basis/coordinates, linear functionals, derivative-as-linear-map and positive-definite metric. Existing Tensor Algebra handles a fixed real vector space, not manifold tensor fields or Christoffel symbols. Destination author must review broader geometry claims against suitable primary sources.
- Evidence: NumPy independent solve checks and MIT8.962 Lecture3 transcript portions on multilinearity, one-forms, dual pairing and metric identification, reviewed10 September2026: https://ocw.mit.edu/courses/8-962-general-relativity-spring-2020/e76be907216aa182691ac751d957c2c5_H6eR3sG524M.pdf. The transcript's physics context is not a substitute for checking all proposed manifold claims.
- Resolution: not yet assessed by the destination author. Keep or adapt the bridge during its authorized rewrite and explicitly state conventions.
- Implementation/verification links: source lesson's section8 and `../TENSOR-ALGEBRA-EINSUM-VERIFICATION.md`; destination extension not implemented here.
