# Calibration and conformal: conceptual-transition review

26 September 2026. Read the complete canonical JSX (~1,094 lines), including eleven sections, all advanced branches, eight exercises, code provenance and annotated sources. Original core mechanisms are strong and remain intact. This author pass strengthens specific later transitions; independent/browser review remains pending.

| Location and concepts | Disposition |
| --- | --- |
| §1 conditioning, binary vs top confidence, multiclass definitions, resolution, proper score and cost | Retained: distinct conditioning plots and exact cost/resolution example already expose each distinction. |
| §2 bins, count uncertainty, signed gaps, ECE, bin cancellation, in-sample repair, endpoint guards | Retained: ten-card lab and executed example show both mechanism and counterexample. |
| §3 sigmoid slope/offset, stable loss/gradients, smoothing, one-class policy, isotonic pooling/ties/interpolation, temperature | Retained: full weighted pooling trace, competing map plot and temperature controls/exact logits. Every fit's meaning and limitation is explicit. |
| §4 independent probability/conformal roles, wrapper interfaces, preprocessing folds | Retained: four-lane flow and complete leakage checkpoint. |
| §5 score direction, weak comparison, sets, exchangeable rank, n+1, ties/infinity, percentile indexing | Retained: exact rank rotation lab and comparison program. |
| §6 response vs mean interval, local-scale units, quantiles/pinball, CQR, crossed/empty intervals | Gaps fixed: asymmetric pinball loss curves and expected slope explain why the target is a quantile; two endpoint inequalities and candidate 12 versus 11 expose CQR score inversion and negative scores. Existing live intervals retained. |
| §7 two offline experiments, roles, baselines, tie rounding, corpus vs future sampling | Retained all programs, measured counts, uncertainty interpretations, baseline comparisons and data provenance. |
| §8 marginal/conditional/empirical coverage, calibration-set variation, group/class thresholds, singleton selection, label shift | Gap fixed for class-conditional mechanism: same probability vector tested against three separate thresholds yields B/C despite A being top-ranked. Existing group mosaic and population layers remain. |
| §9 APS/RAPS, full conformal, jackknife+/CV+, weighted shift, anomaly p-values, monotone risk, library/cost | Gaps fixed: paired leave-one-out predictions/residual segments expose what differs from ordinary jackknife; weighted-score table retains the future-input infinity atom and gives both a finite and infinite outcome. Other methods remain scoped connections with owner references; no unsupported universal guarantee added. |
| §10 practice and §11 readiness | Retained all eight changed-input exercises, explained solutions, author downloads and next-topic sequence. |

## Actual source reading

Read [CQR](https://proceedings.neurips.cc/paper/2019/file/5103c3584b063c431bd1268e9b5e76fb-Paper.pdf) §2 and the score construction in §4; [jackknife+](https://stat.cmu.edu/~ryantibs/papers/jackknife.pdf) introduction, definitions and §2 through the basic guarantee; [covariate-shift conformal](https://proceedings.neurips.cc/paper/2019/file/8fb21ee7a2207526da55a679f0332de2-Paper.pdf) §2.1 weights, infinity atom and support condition. Used original constructions to explain missing intermediate operations, with independently constructed finite numbers. Existing full tutorial and authors' video routes retained; new full video viewing is not claimed.

`CalibrationIntuitionFigures.jsx` adds two static native plots with scoped CSS: exact pinball costs against a proposed prediction for fixed y=2, and three paired leave-one-out endpoint candidates on a common response axis. Candidate segments are explicitly not guaranteed intervals. The class-conditional and weighted-quantile explanations use tables because keeping candidate identity and denominators visible matters more than geometric decoration. All added examples are declared constructed quantities.

Original fits, models, examples, labs, base figures and measured outputs unchanged. `author-checks.json` records scoped arithmetic, JSX parsing and original-source identities, with independent and actual browser work explicitly pending.
