# Authoring notes: Ordinary Differential Equations & Linear Systems

Canonical topic ID: ordinary-differential-equations-linear-systems

## 2026-09-11 — Coordinate the local dynamics bridge with the later ODE methods lesson

- Status: open
- Origin: Dynamical Systems Theory & Chaos, [individual design](../DYNAMICAL-SYSTEMS-LESSON-DESIGN.md).
- Destination and rationale: this planned topic owns systematic initial-value methods, coupled linear systems, forcing and numerical solution. The earlier dynamics lesson needs a self-contained rate/state refresher because module order places this page later.
- Existing coverage: its live stored blueprint teaches cooling, a scalar IVP, coupled first-order systems, eigenvalue behavior and Euler comparison. No published body was found in the manifest at this scoped check. A stored plan is not a finished prerequisite.
- Proposed treatment: assess the final dynamics lesson when writing this one, retain a quick review of rate versus update, then go further into solution construction, existence/uniqueness assumptions, matrix exponentials, forcing and interpretable units. Explain a second-order position equation through position and velocity. Do not assume a learner already knows how to solve an ODE because an earlier page used one.
- Example: cooling x'=−kx gives x(t)=x0 exp(−kt); Euler changes the exact sampled multiplier exp(−kh) to1−kh. Contrast finite-time numerical stability with approximation error, and include x'=x² as a finite-time blow-up counterexample to a blanket global-solvability claim.
- Evidence: [Menon AM219](https://www.dam.brown.edu/people/menon/publications/ds-2020-1.pdf), existence/flow chapters as a research starting point; exact local examples will receive numerical evidence in the dynamics record. No completed destination implementation is implied.
- Resolution: not yet reviewed by the ODE author.
- Implementation/verification links: none for this destination yet.
