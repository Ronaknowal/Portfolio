# Evaluation Metrics: concept-level author review

Author pass: 2026-09-26. Read all eleven sections, complete worked tables, eight practice tasks and source figure contracts. Scope search found no current topic-body generator; runtime JSX is the updated content checkpoint. Existing scratch metric calculations, library comparisons and measured evaluation files are unchanged.

## Entire-lesson map

| Location / concept chain | Assessment and action |
| --- | --- |
| Opening / §1 evaluation question, unit, prediction type and positive class | Concrete inspection question and route diagram adequately distinguish decisions, ordering, probabilities and numerical predictions. Retained. |
| §2 individual examples → confusion cells → denominators → F1/Fβ → undefined ratios | Card movement, named populations and numerical counterexamples support each step. Retained all null cases and live threshold lab. |
| §3 cost matrix → expected-cost cutoff → development selection → prevalence workload | Algebra and two 10,000-item populations already connect rates to decisions. Retained. |
| §4 tied score groups → ROC/AP → pair credits → integration conventions → competing rankers | Exact table, plots, pair grid and live edits are already strong. Retained distinctions between area and whole-curve dominance. |
| §5 monotonic score transformation → fixed ranking / changed probability losses → calibration distinction | Squared probabilities and per-item penalties adequately explain the mechanism. Retained existing controls and multiclass normalization warning. |
| §6 residuals → absolute and squared aggregation → units and outliers | Existing movable residual comparisons and area diagram retained. |
| §6 optimal forecast target | Gap: mean, median and quantile were asserted without intermediate mechanism. Added derivative / one-sided-slope reasoning and three objective curves over the existing five durations. |
| §6 R² evaluation-mean reference, negative/undefined cases, MAPE zero handling | Worked baseline trap and live null cases already provide sufficient support. Retained. |
| §7 cutoff metrics → first success → AP/MAP → gain/discount/ideal normalization | Shelf illustration, calculated contributions, gain convention and reorder lab sufficient. Retained incomplete-judgment and tied-order limits. |
| §8 full measured protocol and unfavorable test-cost outcome | All code, counts, threshold, cost and baseline arguments preserved. No rerun or improved-performance claim. |
| §9 per-class metrics → macro/support/micro aggregation | Added visible weight allocations beside the existing calculations. Micro remains count pooling, not another weighted-bar average. |
| §9 balanced accuracy, MCC and multilabel distinctions | Added MCC's binary covariance/std interpretation and original-threshold calculation. Retained multiclass/multilabel conditions and undefined denominator. |
| §9 streaming counts, global ordering, online variance, paired uncertainty | Existing explicit mechanisms and evaluation-unit distinctions adequate. Retained. |
| §10 practice and §11 alternatives / learning-theory bridge | All eight tasks retained; streaming exercise now asks to check equality rather than require a preliminary prediction. |

## Added visual contracts

`ForecastTargetFigure`: exact empirical averages of absolute error, squared error and 90% pinball loss for equally weighted outcomes [1,2,3,4,10]. Their minimizers are 3,4,10. Different units and separate vertical scales are explicit. All 121 plotted values per panel were checked against bounds. No fake model-training trajectory.

`AggregationWeightFigure`: uses the existing confusion table and per-class F1 values. Fixed 0–1 bars encode weights, text shows products and sums. Class 2's failure remains visible even with zero contribution. Native responsive panels, captions and text equivalents use scoped charcoal/amber styling. Four existing live investigations and earlier interactive diagrams remain.

## Research actually read

- Gneiting, [Making and Evaluating Point Forecasts](https://arxiv.org/pdf/0912.0902), introduction and §3.3 quantile scoring definition: used to connect requested functionals to scoring rules. Existing reference retained; durations, slopes and plots are original calculations from this lesson's fixture.
- [scikit-learn model evaluation](https://scikit-learn.org/stable/modules/model_evaluation.html#matthews-correlation-coefficient), binary MCC definition and correlation interpretation. Checked the formula against a direct binary covariance calculation. No broad “balanced means universally best” claim added.
- No newly watched video or newly executed native fit; existing annotated alternatives retained.

## Actual checks / limits

Four author groups passed: JSX parse; plot-coordinate bounds and minimizers; finite-difference derivatives on both sides of relevant breakpoints; MCC covariance/count equivalence and class-weight arithmetic. An initial proposed MSE plot maximum was too small at forecast 12; corrected to 80 and checked every coordinate before finalizing. Browser and independent-review gates remain pending root; source hashes are in author-checks.json.
