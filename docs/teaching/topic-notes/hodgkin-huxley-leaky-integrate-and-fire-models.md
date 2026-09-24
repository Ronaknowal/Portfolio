# Authoring notes: Hodgkin–Huxley & Leaky Integrate-and-Fire Models

Canonical topic ID: hodgkin-huxley-leaky-integrate-and-fire-models

## 2026-09-11 — Connect phase-space excitability to the mechanistic neuron models

- Status: open
- Origin: Dynamical Systems Theory & Chaos, [design](../DYNAMICAL-SYSTEMS-LESSON-DESIGN.md).
- Ownership and coverage: this destination's actual planned blueprint teaches membrane leak, threshold/reset, firing responses and conductance gates. No published mapping was found in this scoped check. The origin's Hopf normal form explains how a mathematical oscillation can emerge; it is not a calibrated neuron.
- Proposed treatment: assess whether a compact phase-plane/nullcline bridge through FitzHugh–Nagumo helps explain a threshold excursion versus a persistent limit cycle before or after the conductance-model comparison. Keep a local vocabulary/variable bridge; do not make an unexplained normal-form name a prerequisite. A LIF reset is a modeling operation, not the same mechanism as a smooth conductance excursion.
- Learning example: contrast a stable resting state with a large transient excursion and a stable repetitive orbit. Explain what a recovery variable adds, how its nullcline changes the flow, and which biological measurements/calibration would still be missing. Include a changed prediction or model-choice exercise only if this adds a distinct learning outcome.
- Evidence: [MIT Rothman bifurcations notes](https://ocw.mit.edu/courses/12-006j-nonlinear-dynamics-chaos-fall-2022/mit12_006jf22_lec10-11.pdf), section 1.5,pp9–11 inspected10September2026; author links original FitzHugh1961 and Izhikevich/FitzHugh2006. Those original biological sources still need direct review if the destination adopts the example.
- Boundaries: avoid a universal bifurcation type for all neurons, uncalibrated biological claims or home stimulation instructions. Core practice should use bounded numerical models with verified units and solver checks.
- Resolution: not yet reviewed. No neural lesson was rewritten by this note.
- Implementation/verification links: none for this destination yet.

## 2026-09-11 — Relate noisy subthreshold voltage to OU without losing units or crossing semantics

- Status: open.
- Origin: mathematics37, [Itô/SDE design](../ITO-CALCULUS-SDE-LESSON-DESIGN.md); design-only when recorded.
- Destination and ownership: the actual planned neural blueprint teaches leak, threshold/reset, firing response and conductance gates. A bounded subthreshold OU bridge can make stochastic input and time constants meaningful here, with threshold/reset and colored-noise mechanisms taught in their full neural context. Itô owns the general stochastic integration and simulation contracts.
- Idea and learning benefit: derive a noisy leak equation with a separate diffusion symbol eta whose units are voltage/sqrt(time). Before introducing a threshold, identify the restoring target, conditional variance, limiting variance eta-squared times tau/2 and the correlation timescale. If the noise coefficient is instead specified as a voltage amplitude scaled by the membrane time constant, explicitly convert that convention. The same letter in two books need not have the same units.
- Proposed treatment: compare deterministic leak, noisy subthreshold voltage and a reset/threshold extension with visible state semantics. An exact OU sampled skeleton still does not observe every crossing between samples, and the unrestricted stationary Gaussian is not the stationary voltage distribution of a resetting spiking process. Treat colored input as a separate state with its own dynamics, not as arbitrary independent per-step noise.
- Source nuance discovered: [Neuronal Dynamics8.1](https://neuronaldynamics.epfl.ch/online/Ch8.S1.html), sections8.1.1–8.1.3, was read10September2026 UTC. Equation8.4's white-noise covariance normalization and equations8.6/8.8 reuse sigma with a different implied scale; derive the conversion directly. The sentence below8.8 calls the limiting trajectory smooth; additive Brownian-forced voltage has continuous but typically nondifferentiable paths. Retain the useful biological context without reproducing either ambiguity.
- Prerequisites and boundaries: introduce current/voltage/time units, stochastic increments and initial versus stationary law. A mathematical toy is not a calibrated neural model. Derive any firing-rate or crossing claim under the stated threshold/reset dynamics; a finite sampled plot does not prove it.
- Resolution: not yet assessed by destination author; the earlier deterministic-excitability note is preserved independently. No neural body rewritten.
- Implementation/verification links: Itô foundation implementation is pending; update after actual evidence exists.
