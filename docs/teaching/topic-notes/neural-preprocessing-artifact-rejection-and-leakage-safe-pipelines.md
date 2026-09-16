# Authoring notes: Neural Preprocessing, Artifact Rejection and Leakage-Safe Pipelines

Canonical topic ID: `neural-preprocessing-artifact-rejection-and-leakage-safe-pipelines`

## 2026-09-12 — Connect temporal filtering to ICA fitting and the intended deployment

- Status: open
- Origin: [ICA content design](../ICA-LESSON-DESIGN.md), its general source-separation mechanism and fixed ECG demonstration.
- Destination and ownership: the neural preprocessing topic's existing blueprint already includes filtering, ICA and leakage. It is the better home for acquisition/reference choices, rank, artifact removal and participant/session validation; the general ICA lesson supplies a bounded connection.
- Existing coverage: the ICA author inspected this topic's inventory and substantive blueprint; it is planned. This note refines an existing planned outcome rather than claiming the curriculum has no preprocessing topic. No complete neural lesson was audited here.
- Idea and learning benefit: explain why a common linear temporal filter can preserve a constant spatial mixing matrix, while its use across a split boundary may still be inappropriate for a prospective prediction task.
- Proposed treatment: with observation rows, `X = SAᵀ`. Applying the same linear time operator F to every channel gives `FX = (FS)Aᵀ`. Connect this to fitting ICA on an appropriately filtered copy and applying the learned spatial transformation to compatible data. The algebra does not cover different filters per channel or arbitrary nonlinear preprocessing. A filter using future samples must be treated according to the actual offline/online task, with boundary handling and information availability specified before evaluation.
- Practice/visual candidate: compare a shared filter with channel-specific filtering using a known constructed mixture; then inspect whether a predeclared injected event is retained after an artifact-removal decision. A time-support diagram can show exactly which samples contribute to a displayed output near a train/test boundary. Diagnose rank changes from referencing, sensitivity to excluded components and whether a participant/session split matches the intended claim. These are proposals requiring dataset-specific validation.
- Evidence inspected 12 September 2026: [Hyvärinen & Oja2000, §5.3, equations38–39](https://www.cs.helsinki.fi/u/ahyvarin/papers/NN00new.pdf), the common-filter/mixing identity; [MNE ICA artifact tutorial](https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html), author-reviewed fitting/filtering sections and root-checked filter construction, documentation identified as1.13.2. Its example uses a noncausal filter; that is a recorded implementation property, not a blanket prescription for online analysis. No MNE run or neural preservation experiment was performed in this content-only task.
- Boundaries: the general lesson's real ECG result is a limited signal-correlation investigation, not validation of a neural preprocessing pipeline or clinical benefit. Select suitable neural data, annotations, licensing and scientific endpoints when this destination is authored.
- Resolution: awaiting destination-author assessment; incorporate, adapt or defer with a reason during authorized work.
- Implementation/verification links: none yet.
