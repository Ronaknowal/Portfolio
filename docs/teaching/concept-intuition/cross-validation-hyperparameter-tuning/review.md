# Cross-validation: whole-lesson concept-transition review

26 September 2026. Read all eleven sections, complete program explanations, ten practices and references. Production JSX is the content checkpoint. Independent teaching and browser review remain pending.

| Concept / exact source location | Finding and action | Representation / transfer evidence |
| --- | --- | --- |
| Intro / §1 selection versus assessment; model/recipe/search estimands | Strong explicit decision ownership retained. Corrected overabsolute wording: selected scores are not independent assessments; they are still performance estimates subject to selection bias. | Three distinct questions and selection-score callout. |
| §2 held-out predictions, folds, remainders and pooled score | Sufficient complete seven-row trace and both weighting schemes. Retained. | LoopFigure/FoldBuilderLab, full scratch example; practices 1–2. |
| §3 groups/time/label availability | Sufficient future-use questions and causal availability boundary. Retained. | SplitQuestionFigure; practices 3–4. |
| §3 stratification, repeated splits and OOF | Sufficient changed estimand and coverage assumptions. Retained. | Explicit exactly-once condition. |
| §4 repeated choice and optimism | Sufficient exhaustive fair-coin selection example, no predictive signal. Retained. | SelectionLab; practice 5. |
| §4 nested selection | Sufficient six-step protected outer assessment. Retained. | NestedRoomsFigure/NestedLab; practice 6. |
| §5 grid/random/log search | Sufficient declared distribution and hit-probability calculation. Retained. | CoverageFigure; practice 7. |
| §6 real nested experiment and final refit | Sufficient published outputs, pipeline ownership and allowed-label perturbations. Retained. | RealExperimentExplorer and complete native program. |
| §7 training size and optimism | Sufficient mean predictor exact risks and comparison. Retained. | RiskFigure; practice 8. |
| §7 correlated fold losses | Gap: covariance formula did not first expose what fails to average away. Added shared shock plus independent shock construction and common-scale variance budget. | New cv-shared-variance: independent 0.25 versus shared 0.5 + independent 0.125. |
| §7 inference boundary | Strong overlap-is-not-correlation counterexample and scope of impossibility result retained. | No measured covariance or confidence interval is invented. |
| §7 bootstrap | Added a concrete resample [0,0,2,2] with distinct/OOB identities and repeated-weight interpretation before inference caveats. | Existing expected distinct-share formula and scope retained. |
| §8 surrogate / expected improvement | Sufficient two different predictive beliefs and acquisition outcomes. Retained. | ImprovementFigure; practice 9. |
| §8 TPE | Added reverse-question intuition and numerical density-ratio comparison (16/7 versus 4/7) before optional library extension. | Clearly labelled illustrative densities and proportional scores, not probabilities. |
| §8 Optuna evidence | Preserved actual run and missing better setting. Clarified score is selection evidence from development folds, not independent assessment. | Native output unchanged. |
| §8 halving / Hyperband | Sufficient slow-starter reversal, resumable versus refit cost and distinct brackets. Retained. | HalvingLab; practice 10. |
| §9 cost, ties, failures; §10–11 transfer | Sufficient fit counts, nested parallelism caution, ten solutions and sources. Retained. | Scratch and library routes remain complete. |

## Research actually inspected

- [Scikit-learn, Nested versus non-nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html): explanation and grid-inside-outer-CV code. Its comparison reinforced correct selection-bias wording; no result copied.
- [Bengio and Grandvalet, No Unbiased Estimator of the Variance of K-Fold Cross-Validation](https://www.jmlr.org/papers/volume5/grandvalet04a/grandvalet04a.pdf): abstract and sections 1–2, conditional versus expected performance and dependent errors. The new shared-shock example is an original pedagogic model, not a reconstruction of that paper’s experiment.
- [Bergstra et al., Algorithms for Hyper-Parameter Optimization](https://papers.nips.cc/paper_files/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf), sections 4–4.1: conditional setting densities, quantile split and EI ratio. Existing annotated sources are retained.

## Representation and checks

The variance bars share a 0–1 variance scale; neutral marks retained common variation and amber residual independent variation. They describe an explicit hypothetical repeated-sampling model, not this lesson’s observed folds. All values and interpretation are in HTML. It is a static variance decomposition, not an interval or control.

The scoped verifier checks averaging variance, TPE ratio and bootstrap identities, plus JSX syntax. Existing fold engines, live labs, native programs and run outputs are untouched. Browser and independent teaching review remain pending.
