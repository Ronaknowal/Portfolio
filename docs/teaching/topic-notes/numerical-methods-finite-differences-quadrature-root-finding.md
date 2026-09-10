# Authoring notes: Numerical Methods

Canonical topic ID: numerical-methods-finite-differences-quadrature-root-finding

## 2026-09-11 — Distinguish a numerical evolution rule from the underlying differential equation

- Status: open
- Origin: Dynamical Systems Theory & Chaos, [design](../DYNAMICAL-SYSTEMS-LESSON-DESIGN.md).
- Ownership: the existing Numerical Methods introduction and first five sections were inspected. They already distinguish truncation/round-off/conditioning/stability, roots, finite differences and quadrature. The dynamics rewrite develops a concrete ODE numerical-stability example; this destination should connect it to its broader error workflow without duplicating a whole dynamics lesson.
- Proposed treatment: when revising, read the finalized oscillator/cooling examples and decide whether a local bridge or an additional ODE-integration section best supports the actual scope/title. Compare equal physical horizons, step sizes and initial conditions. A low residual, conserved-looking plot or requested solver tolerance does not by itself prove accurate global trajectories.
- Example and learning benefit: forward Euler on q'=p,p'=−q multiplies exact energy by1+h² per step. Kick-then-drift symplectic Euler preserves a modified quadratic q²+p²−hqp for|h|<2, rather than the exact energy at every step. This reveals why geometric fidelity and formal order are distinct questions. The origin will verify these simple algebraic examples independently.
- Boundaries: detailed symplectic integration theory and long-time shadowing require explicit further assumptions. A local solve_ivp tolerance is not a universal global-error bound; event detection can miss multiple zeros inside one accepted step.
- Evidence: [SciPy solve_ivp](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html), inspected tolerance/event interface10September2026, page1.18.0; actual runtime version must be recorded separately. This is a proposed teaching connection, not a measured benchmark.
- Resolution: not yet reviewed by the Numerical Methods author.
- Implementation/verification links: none for this destination yet.
