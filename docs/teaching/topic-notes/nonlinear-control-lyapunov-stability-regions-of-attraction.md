# Authoring notes: Nonlinear Control, Lyapunov Stability & Regions of Attraction

Canonical topic ID: nonlinear-control-lyapunov-stability-regions-of-attraction

## 2026-09-11 — Build from energy analysis to a qualified region certificate

- Status: open
- Origin: Dynamical Systems Theory & Chaos, [design](../DYNAMICAL-SYSTEMS-LESSON-DESIGN.md).
- Ownership and existing coverage: the actual destination blueprint already plans candidate functions, sign/domain assumptions, pendulum analysis and an attraction-region estimate. The origin introduces phase-line potentials and local stability; this destination owns controller construction and region certification.
- Proposed treatment: retain a concrete controlled example with an explicit domain, equilibrium, V(0)=0, positive-definiteness and the sign of its derivative along the closed-loop vector field. Distinguish a sublevel set certified invariant from a collection of simulated starting points. Nonpositive derivative alone needs an additional invariance argument before concluding attraction to one equilibrium.
- Example: a damped oscillator loses energy only through its velocity term; moments with zero velocity need not be equilibrium states. Explain the largest invariant subset of the zero-derivative set, then compare a conservatively verified region with sampled trajectory evidence. Local/global and stable/attracting/exponential claims require separate conditions.
- Evidence: [MIT Underactuated Lyapunov analysis](https://underactuated.mit.edu/lyapunov.html), direct-method/global/LaSalle portions inspected10September2026. Use its mathematical conditions as a source, not a physical safety guarantee or an automatically proven controller.
- Resolution: not yet reviewed. Other robotics bodies remain outside the current rollout.
- Implementation/verification links: none for this destination yet.
