# Anomaly detection: intuition through the whole lesson

Author review: 26 September 2026. Full implementation revision; production JSX is the current content checkpoint. Independent teaching review and browser inspection are pending the integration owner's pass.

Read the complete production lesson, deeper branches, programs' surrounding explanations, all ten practices and references. Existing numerical engines, runnable programs, measured temperature data and outcomes are unchanged.

## Complete reading map

| Transition / source location | Assessment and action | Representation and evidence opportunity |
| --- | --- | --- |
| Opening; §1: unusual relative to a task, point/context/sequence | Sufficient: temperature/load example and stuck-sensor distinction motivate representation choice. Retained. | Three-case table; checkpoint and practice J. |
| §2: outlier versus novelty; fit versus score versus threshold | Sufficient: explicit input populations and provenance figure. Retained. | ProvenanceFigure; practice D and evaluation cases. |
| §2: API score direction; threshold ties | Sufficient: exact API table and strict inequality. Retained. | ThresholdRulerFigure and offset investigation. |
| §3: interval cuts → path | Sufficient: 9/12 probability and marked empty gaps. Retained. | FirstCutFigure/IsolationLab; practice A. |
| §3: truncated leaf → correction → normalized score | Gap: formulas moved faster than the unresolved-search intuition. Added that explanation and relative-path table. | Half/equal/double path lengths produce .7071/.5/.25; practice B. |
| §3: exact expectation, finite forest, duplicates | Sufficient exact table, uniform/duplicate contrast. Retained. | Existing lab and complete program. |
| §3: masking, subsampling, axis dependence | Sufficient group shielding and affine/rotation distinctions. Retained. | Concrete rare-mode reasoning; no redundant diagram. |
| §4: spacing → floor → reciprocal → ratio | Floor already well illustrated. Added distinction between the two averages and cancellation of common scale. | ReachFloorFigure, farther-query counterexample, practice C. |
| §4–5: ties, identity, duplicates and query membership | Sufficient exact coordinate example and fixed-reference lab. Retained. | LofModeLab; practice D. |
| §6: RBF → support → nonlinear boundary; gamma and nu | Sufficient two-anchor derivation/live crossing and scale correspondence. Retained. | KernelBoundaryLab; practices E/G. |
| §7: choose mechanism and representation | Sufficient four concrete applications and comparison table. Retained. | Stuck-sensor transfer in J. |
| §8: population → alert workload; calibration → later data | Sufficient actual denominators, population and threshold labs. Retained. | Practices F/H/I. |
| §9: boundary → primal/dual objective | Gap: optimization arrived before plain competing terms. Added projected-height/level/slack bridge. | Existing exact stationarity/dual preserved. |
| §9: cap → support count and violation bound | Added quarter-budget graphic: n=8, nu=.5, cap=.25. | Three capped weights cannot sum to 1; practice G changes values. |
| §10: duplicate policy, causal features, split, baseline, annotations | Sufficient exact timestamp/count reasoning and operational limits. Retained. | Full offline program/TemperatureThresholdLab; H/I. |
| §11: local regularity bound and resource costs | Sufficient interval arithmetic and explicit resource accounting. Retained. | Existing derivation and examples. |
| §11: frozen versus rolling reference | Added 40→50 drift comparison explaining why adaptation can absorb a slow failure. | Conceptual counterexample, not measured data. |
| §12–13: practice, readiness, GMM connection | All ten reasoned exercises and annotated resources retained. | Existing changed-input transfer. |

## Research actually inspected

- [scikit-learn 1.9.1 novelty/outlier guide](https://scikit-learn.org/stable/modules/outlier_detection.html): fitting modes, Isolation Forest and LOF sections. Its informal nu wording was not adopted as a future false-positive guarantee.
- [Schölkopf et al., Estimating the Support of a High-Dimensional Distribution](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-99-87.pdf): original author-hosted report, support optimization and nu-property passages. The quarter-budget graphic is an original elementary consequence of the displayed constraints.

No new benchmark or fault probability was introduced; no video viewing is claimed. Existing annotated video/slides alternatives remain.

## Representation and author checks

The support-weight-budget static SVG has two aligned four-cell rows, one quarter per cell. Amber is contribution; dashed empty outline is missing mass. These are weights, not record counts. HTML supplies the complete calculation. Inline 16-unit text and a bounded maximum width support narrow screens; actual painted verification belongs to integration. It adds no controls, effects or dependencies.

Author-checks.json records JSX parsing, independent normalized-score/cap arithmetic and source hashes. Unchanged native campaigns are reused. This author record does not claim independent or browser closure.
