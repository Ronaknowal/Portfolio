# Authoring notes: Anomaly & Outlier Detection

Canonical topic ID: anomaly-outlier-detection-isolation-forest-one-class-svm-lof

## 2026-09-12 — Continue from density clustering to a declared alerting task

- Status: open
- Origin: the concurrent DBSCAN content-first author and the scoped [Anomaly design](../ANOMALY-DETECTION-LESSON-DESIGN.md).
- Destination and ownership rationale: Anomaly 15 owns reference populations, outlier versus novelty fitting, score orientation, alert thresholds and workloads. DBSCAN 14 owns density connectivity/noise; GMM 16 owns probability density and component responsibilities.
- Learning benefit: a learner can explain why noise, local sparsity, model incompatibility and operational fault are different conclusions, and can connect each method's score to a review decision.
- Existing coverage: old Anomaly body had the three algorithms but lacked the full fitting-mode/threshold and reproducible real chronology developed in the current manuscript. The published body remains unchanged; its baseline SHA is recorded in the design.
- Proposed treatment: core §§1–2 distinguish noise from anomaly decisions; §4 compares LOF neighbor-radius/self-exclusion conventions with optional OPTICS/HDBSCAN source/both-radius conventions; §§8/10 turn scores into workload and event-window reports.
- Concrete example: two legitimate 1D groups [0,1,2] and [20,24,28] demonstrate query 4 LOF 35/24 versus query 17 LOF 35/32, despite the latter's larger nearest distance. A licensed real temperature series then gives four event-window hits for all compared methods but very different unmatched row workloads.
- Prerequisites and boundaries: the LOF definition is self-contained; reading prior optional hierarchy material is not required. Do not import DBSCAN min_samples self-inclusion as LOF's k-other-row count.
- Evidence: original LOF definitions and current scikit-learn 1.9.1 fitting/query contracts are linked in the design. The real-data provenance and elementary author calculations are supplied in the draft packet. No production geometry, independent review or browser check has run for this revision.
- Resolution: prepared in [complete content](../drafts/anomaly-outlier-detection-isolation-forest-one-class-svm-lof/lesson.md) and [visual specifications](../drafts/anomaly-outlier-detection-isolation-forest-one-class-svm-lof/visual-specifications.md), pending implementation. Keep status open until the implementation and corresponding evidence exist; content completion alone is not the policy's implemented/adapted status.
- Related destination: [GMM note](gaussian-mixture-models-gmm-em-algorithm.md) owns relative responsibility versus total density and threshold; no GMM source change is authorized by this note.
- Implementation/verification links: none for revision 1; author reasoning inputs are linked from the design.
