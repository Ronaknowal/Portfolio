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
