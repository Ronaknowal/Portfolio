# Authoring notes: Sampling, Aliasing & Quantization in Neural Recordings

Canonical topic ID: `sampling-aliasing-and-quantization-in-neural-recordings`

## 2026-09-10 — Distinguish quantizer resolution from a noise guarantee

- Status: open
- Origin: `rate-distortion-theory`, [scoped design](../RATE-DISTORTION-LESSON-DESIGN.md).
- Destination and ownership rationale: this actual inspected brief already teaches clipping, ADC quantization, sampling and why bit depth alone does not establish precision. Instrument-specific assumptions belong here; Rate-Distortion retains the compression/quantization distinction and links the destination.
- Idea and learning benefit: explain when a step-size squared over12 error estimate is reasonable, why overload distortion is separate and why the error need not be independent white noise. A learner should predict what happens when an input concentrates at cell centres/boundaries or clips.
- Existing coverage: planned in the destination's individual brief; source inventory reviewed. No body implementation or verified hardware claim established.
- Proposed treatment: after introducing quantization cells, compare a fine, approximately locally uniform within-cell distribution with a deterministic repeated input; then show saturation. Derive the uniform within-cell second moment from an integral. Distinguish a distributional/high-resolution approximation from device noise, bandwidth, calibration and measured effective resolution. Keep sampling/anti-alias filtering separate from value quantization.
- Explanation/example: a cell of widthΔ with uniform centred error has mean-square errorΔ²/12; an input at the cell centre has zero quantization error, an input near a boundary has nearlyΔ²/4, and a clipped value can have much larger error. These are teaching distributions, not measurements of a recording instrument.
- Prerequisites and boundaries: squared error, expectation, cells and clipping; any real converter specification or dither claim needs primary device/signal-processing evidence. Do not present the approximation as a medical performance guarantee.
- Evidence: MIT6.441 [chapter23 §23.1.1](https://ocw.mit.edu/courses/6-441-information-theory-spring-2016/880c39878ba7dc35a5fbb35758d42a69_MIT6_441S16_chapter_23.pdf), actual text read 10 September2026, explicitly invokes high-rate/local-density conditions and separates overload. Independent examples can be integrated directly; device-specific claims remain unverified.
- Resolution: receiving author must assess and integrate, adapt or reject with reasons.
- Implementation/verification links: none yet; no destination lesson rewritten.
