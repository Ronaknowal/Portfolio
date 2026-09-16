# Authoring notes: Weight Initialization (Xavier, Kaiming, μP)

Canonical topic ID: `weight-initialization-xavier-kaiming-p`

## 2026-09-11 — Average variance preservation does not preserve every direction

- Status: open
- Origin: `random-matrix-theory`; [RMT34 design](../RANDOM-MATRIX-THEORY-DESIGN.md), random-linear-map application.
- Destination and ownership rationale: this later lesson owns activation-specific gains, forward/backward fan conventions, depth effects, orthogonal initialization and μP. RMT's smaller example only establishes the spectral distinction for a random linear map.
- Idea and learning benefit: a learner should distinguish E‖Wx‖² for a fixed input from the worst-direction singular values of an actual draw, and from the product of layer Jacobians. One scalar variance condition is not a certificate that all directions or all training runs remain stable.
- Existing coverage: the present introduction discusses forward and backward variance, fan-in/fan-out and initialization motivation. Its complete later treatment was not audited here; inspect it before adding another explanation.
- Proposed treatment: use the RMT bridge or a comparable exact/seeded singular-value example immediately after the variance derivation if the distinction is absent. Give the product/Jacobian issue its own explanation before any dynamical-isometry claim. Keep μP and nonlinear-network analysis in this owner.
- Explanation/example: for a d×d random matrix with independent mean-zero entries of variance1/d, any fixed deterministic x satisfies E‖Wx‖²=‖x‖² by expanding the squared norm and canceling cross terms. A particular draw need not preserve x, and the maximizing direction depends on that draw. Gaussian singular values can remain broadly spread even though this expectation is exactly correct.
- Prerequisites and boundaries: dot products, expectation/independence and singular values; distinguish squared norm from coordinate variance, and square examples from unequal fan dimensions. A trained matrix or an activation-dependent Jacobian is not automatically iid Gaussian.
- Evidence: exact expectation algebra and [Vershynin's Gaussian singular-value bounds](https://arxiv.org/pdf/1011.3027), `5.3.1. The future author must inspect original initialization/dynamical-isometry sources for any further network-performance claim; none was verified or proposed as an empirical result here.
- Resolution: not yet reviewed. Check the actual destination body and decide whether to include, adapt, link or omit with reasons. This is an explanation proposal, not an instruction to repeat the RMT chapter.
- Implementation/verification links: originating RMT lesson is still in design; no new training benchmark or implementation is claimed.

## 2026-09-12 — Content resolution: adapted; implementation pending

The full destination source was reviewed and the note was adapted in [the prepared manuscript](../drafts/weight-initialization-xavier-kaiming-p/lesson.md), “Average preservation can hide a collapsed direction.” It distinguishes fixed-input expectation over random weights, one actual Gaussian draw, average squared singular value, direction-specific gain, and a product/local gated Jacobian. An exact diagonal matrix, fixed seed-19 spectrum, orthogonal Gram check and ReLU Jacobian support the explanation; see [design and source review](../drafts/weight-initialization-xavier-kaiming-p/design.md) and [visual specifications](../drafts/weight-initialization-xavier-kaiming-p/visual-specifications.md).

This resolves the proposed research/writing treatment, with initialization-specific primary sources inspected. It does not close implementation verification: the prepared visuals and website content have not been implemented. Preserve the note and packet for phase two.
