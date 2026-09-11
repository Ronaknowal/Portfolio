# Authoring notes: Clustering Evaluation & Validation

Canonical topic ID: `clustering-evaluation-validation-silhouette-ari-nmi`

## 2026-09-11 — Continue from executable clustering diagnostics without turning them into truth certificates

- Status: open
- Origin: [K-Means & Hierarchical Clustering design](../K-MEANS-HIERARCHICAL-LESSON-DESIGN.md), sections 9 and 14 of its new lesson.
- Destination and rationale: this topic explicitly owns silhouette, ARI/NMI and validation; a complete treatment of metric assumptions and selection uncertainty belongs here, beyond the minimum needed to use k-means and a hierarchy.
- Existing coverage: the older destination introduction explicitly distinguishes internal and external evaluation, but calls raw RI nearly useless in general. Only scoped passages were inspected; assess the full body before deciding what is missing or incorrect.
- Learning benefit: distinguish optimizer sensitivity, resampling stability, internal separation, reference-label agreement and actual downstream usefulness. A favorable metric is not a certificate of real categories.
- Proposed treatment: start with changed partitions on the same observation IDs. Derive pair contingency counts and permutation invariance before adjusted or information-based scores. Explain each chance/reference model, degenerate/singleton/noise-label conventions, metric geometry, overlapping versus hard partitions, and why a mean silhouette can conceal one poor group. Use a genuine silhouette plot and pair/contingency representation where they expose the mechanism.
- Carry-forward experiment: the origin's 360-row capstone freezes training/validation/test sets, compares bootstrap fits on the same validation rows and declares size/stability eligibility before testing. Its thresholds (20 rows, median ARI .8) are illustrative task choices, not universal recommendations. Extend with an independently changed perturbation and a contrast where a stable or separated partition is unhelpful. Compare k-means out-of-sample assignments with a hierarchy that lacks an inherent predict rule.
- Boundaries: preserve the current module order and avoid duplicating the entire fitting lesson. Gap-statistic reference distributions, uncertainty and any external labels need their own assumptions and source verification. Reference categories are evidence for a particular question, not infallible ground truth.
- Evidence: [silhouette API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html), [clustering evaluation guide](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation); origin native programs and their evidence are linked from its design. The future author must verify the destination's exact claims and plots.
- Resolution: not yet reviewed by the destination author. Reassess, implement/adapt or reject with reasons; this note does not authorize rewriting that topic now.
