# Authoring notes: Gaussian Processes (GP)

Canonical topic ID: `gaussian-processes-gp`.

## 2026-09-11 — Kernel mean identity does not imply sample-path RKHS membership

- Status: open
- Origin: [Functional Analysis & RKHS design](../FUNCTIONAL-ANALYSIS-RKHS-LESSON-DESIGN.md); scoped read of this destination's definition/posterior and approximation sections.
- Ownership and benefit: RKHS teaches deterministic controlled functions and kernel ridge. This topic owns probability over functions, posterior covariance and what a random sample path is. Explicitly connect the matching formulas without equating their interpretations.
- Proposed treatment: for a zero-mean GP with kernel K and independent homoscedastic noise variance σ², compare its posterior mean with average-loss KRR using σ²=nλ. Explain which model assumptions supply uncertainty; a deterministic ridge norm alone supplies no posterior credible interval. The covariance `min(s,t)` belongs to Brownian motion, while its associated anchored RKHS contains absolutely continuous f with f(0)=0 and square-integrable derivative. Brownian paths almost surely do not belong to this space. Do not generalize that statement to every finite-dimensional GP.
- Further computation bridge: explicit random features require the same frozen sampled feature map at train/test. Feature construction cost and the subsequent regression solve are different; an O(nD) feature array does not imply an exact dense feature-ridge training cost O(nD).
- Evidence: [Kanagawa et al. 2018](https://arxiv.org/pdf/1807.02582), matching regression formulas and section4.3 sample-path discussion, read 10 September UTC; current origin constructs the anchored space directly rather than relying on shorthand derivative notation. Source-level review was limited to the named portions, not the entire GP lesson.
- Resolution: pending destination author assessment. Origin implementation and numerical/browser verification are in progress. No destination body edited.
