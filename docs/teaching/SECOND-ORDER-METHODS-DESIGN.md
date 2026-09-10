# Second-order methods: lesson design

Started and completed author review 10 September 2026. Stable ID: `second-order-methods-l-bfgs-k-fac-shampoo-natural-gradient`. Mathematics position 12 in the active 74-topic rollout. This document owns design and inspected sources; the [verification record](SECOND-ORDER-METHODS-VERIFICATION.md) owns actual outcomes and remaining integration boundaries.

## Scope and continuity

The existing published body was read in full. Retain its exact `100x²+y²`, `(1,1)` Newton solve and useful warnings about indefinite/stale estimates, memory and fair comparisons. Its one-paragraph descriptions of L-BFGS, K-FAC, Shampoo and natural gradient do not currently teach those mechanisms or assess them. Replace that gap with a connected, self-contained explanation and complete runnable examples.

Keep the title and stable identity: its named methods are already the intended scope. Explain that the umbrella includes exact Hessian methods, secant approximations, probability geometry and gradient-statistic preconditioning; Shampoo does not become a Hessian estimator merely by appearing here.

Prerequisites: matrix-vector products and linear solves, symmetric eigendecomposition, gradients/Hessians and the preceding gradient-descent lesson. Recap positive definiteness, Taylor's model and the chain rule. Probability/KL occurs later in module order, so introduce the complete two-outcome distribution and its expectation here rather than silently requiring a future lesson. No neural-network architecture is assumed: define one affine layer, its input/bias, output logits and labels before K-FAC.

Inventory command was run for the exact stable ID. No destination note exists. The unrelated unresolved bit-manipulation inbox does not apply. A scoped catalogue search finds Non-Convex Optimization Landscape immediately next and TRPO in reinforcement learning. Retain a local indefinite-curvature counterexample and local KL constraint; route full saddle-landscape analysis and policy-level trust-region guarantees to their owners. Do not audit the catalogue or change its order.

## Outcomes and hurdle map

| Outcome | Mechanism and worked anchor | Representation and independent evidence |
| --- | --- | --- |
| Derive a Newton direction and its conditions | Minimize `gᵀd+½dᵀHd`; solve `Hd=-g`; retain `diag(200,2)` example; rotated SPD variant | Equal-scale contour plane with gradient/Newton paths, selected actual coordinates and objective. Changed SPD quadratic and residual check. |
| Recognize a bad Newton direction and safeguard a step | Double-well `¼(x²−1)²+½y²`; raw indefinite Hessian can aim toward its saddle; SPD damping gives descent but a full step still needs checking | Plot actual one-dimensional loss along the chosen direction against its local quadratic, mark attempted and accepted steps. Armijo decrease, finite backtracking failure and changed point practice. |
| Explain what limited memory stores | `s=θnew−θ`, `y=gnew−g`; inverse-BFGS secant and positive-curvature condition; two-loop products in reverse/forward order | History pair lanes plus live vector transformation, visible dense 2×2 reference only for validation. Compare an independent dense formula, memory truncation and rejected bad pair. |
| Distinguish parameter distance from distribution change | Bernoulli cross-entropy, score expectation, Fisher and local KL; compare probability and logit coordinates | Probability mass bars and tangent/finite-update correspondence. Exact KL vs its local quadratic; finite moves need not agree under nonlinear reparameterization. |
| Identify K-FAC's actual approximation | Two-input/two-output affine Bernoulli layer; exact conditional Fisher `E[(aaᵀ)⊗S(a)]` vs `E[aaᵀ]⊗E[S(a)]`; column-major vec | Factors and corresponding 4×4 matrices with shared signed cell scale, exact difference and damped directions. Enumerate model outcomes independently. Factor damping differs from full diagonal damping. |
| Compute tensor-statistic preconditioning | Original matrix Shampoo accumulates `GGᵀ` and `GᵀG`; inverse quarter powers act on row/column spaces | Gradient matrix, accumulation terms, eigendirections and transformed matrix, stepped fixed replay. NumPy eigensolver and rotation-equivalence checks. Original algorithm distinguished from implementation variants. |
| Choose and test a method fairly | Full-batch scientific parameter fit with actual PyTorch L-BFGS closure; Hessian-vector product; storage/work calculation | Complete native programs with captured output, explanation of closure reevaluation and reproducibility. Changed-constraint exercises and a controlled experiment protocol, no invented throughput ranking. |

## Implemented sequence

1. Why one rate struggles across unequal curvature; original exact example and SPD geometry.
2. Derive the local quadratic, descent proof, damping and line-search checks; explain local convergence assumptions and failure.
3. Learn inverse curvature from secant pairs; derive BFGS properties and two-loop L-BFGS; actual closure-based fitting.
4. Define a distribution and derive natural gradient from its local KL budget; show the finite-coordinate caveat and empirical-Fisher counterexample.
5. Build a small neural layer and inspect K-FAC factors, lost dependence, vectorization and damping.
6. Accumulate row/column gradient statistics and derive the original matrix Shampoo update; link tensor modes and matrix powers.
7. Scale carefully with Hessian-vector products, explicit memory/work and implementation diagnostics.
8. Independent changed-data calculations, debugging, transfer, readiness and the actual next module topic.

These are conceptual groups, not an imposed website template. The final estimate is about 65 minutes reading plus 2–3 hours practice. Essential derivations remain visible; optional derivations deepen rather than supply missing core steps. All nine full programs were executed; the changed fitting experiment includes a complete worked reference inside its answer.

## Visual contracts

All graphs are calculated from the displayed finite model, never benchmark data. Controls operate on bounded small matrices or finite step counts; no timers, workers, native Python or training framework is shipped to the browser. Maintain readable SVG labels, actual numerical text and accessible descriptions; no color-only state.

- Curvature plane: domain and both axes use the same unit scale. Contours are exact level sets, not sampled performance curves. A rotated valley prevents a misleading coordinatewise-only interpretation. Bound paths and show off-plot states honestly rather than clipping them into apparent convergence.
- Newton safeguard: horizontal axis is the step multiplier along one stated direction; vertical axis is actual objective/local quadratic. Explain why descent is a local statement and why an indefinite model can have no minimum. Controls must distinguish changing the direction from shrinking its multiplier.
- L-BFGS history: chronological pairs are retained, then consumed newest-first and oldest-first. Each step labels its vector, scalar and operation. Reset and back are deterministic. Dense inverse comparison is a two-dimensional teaching oracle, not the production L-BFGS algorithm.
- Natural gradient: show both outcomes and probabilities, a common probability axis for mapped directions and exact finite endpoints. Vary starting probability and step fraction within valid domains. Formula, bars and numerical KL share state. The local quadratic is identified as an approximation.
- K-FAC: label flattened weight order `(w11,w21,w12,w22)`. Factors and full block use a common numeric legend where compared. Changing the layer strength changes actual conditional output variances and therefore the approximation error. Damping has an explicit positive minimum for solves.
- Shampoo: step through three fixed gradient matrices and optionally rotate row/column bases. Explain that replay isolates the transform rather than comparing training trajectories. Show `L`, `R`, inverse roots and direction; matrix power is spectral, not entrywise. Epsilon is positive and part of the stated model.

Place compact visible geometry, secant correspondence and row/column contractions beside their introducing prose. An informative initial lab may supply this view; do not duplicate it for a figure quota. Review ordinary reading separately from control operation.

## Sources actually inspected

Retrieved 10 September 2026. Original derivations and new numerical fixtures are authored for this lesson; source annotations distinguish algorithm provenance from locally calculated examples.

| Claim | Primary source and inspected scope |
| --- | --- |
| Matrix and tensor Shampoo, inverse-root exponents, storage and factor-root work | Gupta et al., [2018 PMLR paper](https://proceedings.mlr.press/v80/gupta18a/gupta18a.pdf), Algorithms 1/2 and sections 1.1/2/4. Matrix/tensor equations and spectral-power definition read. Do not reproduce the paper's empirical rankings as contemporary benchmarks. |
| K-FAC expectation factorization and column-wise vec | Martens and Grosse, [2015 paper](https://proceedings.mlr.press/v37/martens15.pdf), section 2/equation 1 and section 5 damping overview read; expectation-of-product is explicitly approximate. The local factored-damping cross terms were derived and independently checked; this is not a claim to reproduce the entire practical optimizer or inspect every supplement. |
| Local KL geometry and empirical Fisher distinction | Martens, [natural-gradient review](https://arxiv.org/pdf/1412.1193), section 6 local metric, finite-step invariance discussion and section 11 empirical-Fisher counterexample read. The lesson derives its separate Bernoulli fixture rather than assuming the later probability module. |
| Two-loop ordering and initial scaling | Yudong Chen, [UW-Madison CS726 Lecture 23](https://pages.cs.wisc.edu/~yudongchen/cs726_sp25/Lecture_23_L-BFGS.pdf), Algorithms 1/2 and positive-curvature/scale discussion read. Original author L-BFGS page checked for provenance. |
| PyTorch L-BFGS closure contract | [Version 2.14 documentation](https://docs.pytorch.org/docs/2.14/generated/torch.optim.LBFGS.html), step closure inspected; installed CPU version is 2.14.0. Two complete fits and the HVP example were actually executed, with final objective/gradient reevaluation. |
| Alternate spoken Newton explanation | [Stanford EE364A Lecture 15](https://see.stanford.edu/Course/EE364A/79), official page, topic outline and relevant transcript passages inspected: unconstrained minimization, Newton step/method and convergence. This is not full-video viewing. |
| Local Newton convergence conditions | Boyd and Vandenberghe, [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf), relevant chapter 9 Newton discussion and local assumptions inspected. The lesson states local positive-definiteness and Hessian regularity rather than promising global quadratic convergence. |

## Verification finish line

Run every complete program in isolated Python/NumPy/CPU PyTorch; capture stdout rather than inventing it. Verify JS transforms against independent NumPy solves/eigendecompositions, exact small probability expectations and dense BFGS. Include invalid/boundary cases and changed independent practice. Confirm all core outcomes have evidence.

Then perform ordinary desktop/390/320 reading with actual screenshots, label/formula inspection, keyboard-only controls and section anchors, reset/back, zero console errors and no page overflow. Build and inspect selected production dependencies; maintain existing manifest mapping and shared route/progress behavior. Record source hashes only after final repairs and distinguish author review, integrated verification and user acceptance.

The author/native/independent/browser portions are now complete as recorded in the linked verification. Root integration remains separate. The Non-Convex author explicitly assessed the local-curvature bridge in its next lesson; policy-level TRPO treatment is recorded as a destination follow-up, without changing that out-of-scope body.
