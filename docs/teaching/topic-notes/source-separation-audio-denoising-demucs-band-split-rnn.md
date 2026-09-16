# Authoring notes: Source Separation & Audio Denoising (Demucs, Band-Split RNN)

Canonical topic ID: `source-separation-audio-denoising-demucs-band-split-rnn`

## 2026-09-12 — Bridge instantaneous ICA to delayed and reverberant audio mixtures

- Status: open
- Origin: [ICA content design](../ICA-LESSON-DESIGN.md), instantaneous-mixture scope and source-separation examples.
- Destination and ownership: this audio topic is the stronger home for delayed/reverberant recordings, waveform/spectrogram representations and modern separation methods. The ICA author confirmed a planned catalogue entry without an individual blueprint; no full audio source or model implementation was reviewed.
- Learning benefit: distinguish successfully undoing an artificial instantaneous remix from separating sources recorded in a room.
- Proposed treatment: contrast `x(t) = A s(t)` with a delayed linear mixture `x(t) = Σℓ Aℓ s(t−ℓ)`. A single instantaneous inverse generally cannot undo all lag terms. For a tiny constructed example, use `x1(t)=s1(t)+s2(t−1)` and `x2(t)=s2(t)`: subtracting one constant multiple of x2 cannot cancel the delayed source for every possible waveform. Compare this with using an explicit delay-aware operation. Introduce the observation unit before changing to a time-frequency representation.
- Practice/visual candidate: two aligned source and microphone lanes with delayed contributions visibly arriving at different times. Have the learner predict which contamination remains after a static matrix operation, then change a delay on an input not already solved in the prose. Extend to a licensed real separated-reference example only when this lesson is authored; keep track/session splits and permutation/scale-aware evaluation explicit.
- Evidence and review bounds: the general instantaneous ICA model was checked against [Hyvärinen & Oja2000](https://www.cs.helsinki.fi/u/ahyvarin/papers/NN00new.pdf). The delayed counterexample above is direct algebra. The ICA author inspected the Hyvärinen/Karhunen/Oja2001 book's chapter19 listing for convolutive mixtures; full chapter proofs and current Demucs/Band-Split RNN research remain unreviewed here. No claim about those systems' measured performance is made.
- Prerequisites/boundaries: linear combinations, samples/time delay and the general ICA lesson's identifiability distinctions. Distinguish simulated remix ground truth from real-room source references, and verify data/model licenses when authoring.
- Resolution: awaiting the audio author; reassess the bridge as part of its individual design.
- Implementation/verification links: none yet.
