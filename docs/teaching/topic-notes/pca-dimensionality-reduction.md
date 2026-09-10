# Authoring notes: PCA & Dimensionality Reduction

Canonical topic ID: `pca-dimensionality-reduction`

## 2026-09-11 — Separate retained sample variance from evidence of latent structure

- Status: open
- Origin: `random-matrix-theory`; [RMT34 design](../RANDOM-MATRIX-THEORY-DESIGN.md), covariance/spike/validation branches.
- Destination and ownership rationale: the PCA lesson owns choosing a dimension, fitting transforms on training data, evaluating reconstruction/task value and interpreting retained components. RMT supplies the sampling-distortion explanation and a restrictive Gaussian spike model, rather than replacing that full practical workflow.
- Idea and learning benefit: make clear that a prominent sample component or high explained-variance fraction is not by itself evidence of a meaningful latent variable, useful prediction or a low-dimensional population. A learner should be able to name the statistical and practical evidence needed.
- Existing coverage: targeted read of the current PCA introduction found universal-sounding claims that real data nearly always occupy a low-dimensional manifold and that principal components directly identify topics, signatures or physical modes. This was not an audit of the entire lesson; inspect its remaining validation sections before deciding what is actually missing.
- Proposed treatment: qualify those introductory assertions, retain useful variance/reconstruction algebra, and integrate a small changed-noise or held-out example into the dimension-selection workflow. Explain scale, centering, leakage and the actual purpose of the representation. Keep the title unless the final scope warrants a compatible change.
- Explanation/example: independent population features can produce unequal finite sample eigenvalues; a Gaussian population spike of2 at aspect ratio.25 has limiting leading sample value2.5 and squared alignment.6. Those are three different quantities. A high sample eigenvalue can coexist with an imperfect direction. Use an independently generated or held-out dataset to assess a concrete goal; do not claim a selected null comparison automatically calibrates every decision.
- Prerequisites and boundaries: eigenvalues/SVD, sample versus population covariance and the relevant modeling assumptions. If using the RMT formula, retain its Gaussian, simple-spike and asymptotic regime; it does not establish impossibility for every estimator or structured prior.
- Evidence: [Paul2007](https://www3.stat.sinica.edu.tw/statistica/oldpdf/A17n418.pdf), Theorems1,2,4 inspected during RMT design; existing Eigenvalues section5 supplies local variance algebra. The future PCA author must verify new empirical claims and its actual implementation.
- Resolution: not yet reviewed by the destination author. Reassess current source, pedagogical need and evidence; adapt, reroute or reject with reasons rather than automatically adding the proposed numbers.
- Implementation/verification links: RMT is in design at the time of this note; no completed PCA rewrite or RMT simulation is implied.
