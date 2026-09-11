# Authoring notes: Diffusion Sampler Error, Schedules & Numerical Solvers

Canonical topic ID: diffusion-sampler-error-schedules-numerical-solvers

## 2026-09-11 — Compare stochastic samplers with an explicit coupling and error target

- Status: open.
- Origin: mathematics37, [Itô/SDE design](../ITO-CALCULUS-SDE-LESSON-DESIGN.md); design-only at the time this note was saved.
- Destination and ownership: the actual planned brief freezes the toy score/model, changes time grids and compares stochastic/deterministic trajectories. This is the right owner for adapting the SDE foundation's numerical contracts to trained generative samplers. The Itô topic supplies basic drift/diffusion, EM/Milstein, common Brownian increments and strong/weak error, not a complete reverse-time diffusion course.
- Discovery and learning benefit: holding only the initial noise fixed does not hold all stochastic forcing fixed when a sampler draws new noise at subsequent steps. A pathwise comparison on nested grids must specify how those increments are coupled. Distributional sample-quality comparisons answer another question and need their own repeat/uncertainty protocol. This prevents a seed or noise-grid change from being misread as an integrator improvement.
- Proposed treatment: use a low-dimensional model with an analytic transition or sufficiently qualified reference. Group fine Brownian increments into coarse increments for a path-error experiment; separately compare a named distributional test statistic, numerical bias and Monte Carlo uncertainty. Explain when a deterministic probability-flow ODE shares marginals with an SDE without implying identical trajectories. Derive the actual chosen reverse-time coefficient/sign convention before code.
- Prerequisites and limits: numerical strong/weak error is not the same terminology as strong/weak SDE solution. Brownian aggregation is appropriate to the declared stochastic driver; solver-specific auxiliary randomness or adaptive conditioning may need a different coupling. A finite finer-step run is not automatically exact. Avoid applying smooth-payoff weak-order guarantees to every discontinuous event or image-quality metric.
- Evidence: [Higham2001](https://webhomes.maths.ed.ac.uk/~dhigham/Publications/P42.pdf), sections4–6 shared-increment and error examples; [Haugh slides](https://www.columbia.edu/~mh2078/MonteCarlo/MCS_SDEs_MasterSlides.pdf), slides5–8, inspected10September2026 UTC. [Song et al. v2](https://arxiv.org/abs/2011.13456) abstract/identity confirms the generative SDE/probability-flow context; the receiving author must inspect and verify its actual relevant derivations, not rely on this abstract for sampler implementation.
- Resolution: not yet assessed by destination author. No generative topic changed or marked complete.
- Implementation/verification links: foundation implementation is pending the approved design; replace this status with actual evidence only after authoring.
