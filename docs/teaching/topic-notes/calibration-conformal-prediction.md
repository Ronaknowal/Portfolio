# Authoring notes: Calibration & Conformal Prediction

Canonical topic ID: calibration-conformal-prediction

## Current disposition — 12 September 2026

All four proposals below are **accepted and addressed in prepared content**. Their original observations are retained as history; the earlier open statuses do not indicate outstanding authoring work. Production implementation and independent verification remain separate.

- Conditioning, plot direction and finite-bin uncertainty: [prepared lesson](../drafts/calibration-conformal-prediction/lesson.md), sections 1–2 and 8, with an exact confidence-versus-class-probability counterexample.
- Decision information: section 1 preserves the two population groups, both calibrated forecasts, and the exact cost and Brier comparisons.
- Naive Bayes score interface: sections 3–4 distinguish decision scores from probability inputs, sigmoid family changes, class decisions and separate fit roles. Section 7 executes a four-role SVC example; it does not relabel it as a GaussianNB replication.
- SVC migration and complete-procedure validation: section 4 uses the current API and explains frozen estimators, fold-specific ensembles, transformations inside folds, and selection boundaries; the actual scikit-learn 1.9.1 program is retained.

The [design and research record](../drafts/calibration-conformal-prediction/design.md), [visual contracts](../drafts/calibration-conformal-prediction/visual-specifications.md), and [provenance](../drafts/calibration-conformal-prediction/data-provenance.md) identify the source claims, executed author checks and remaining phase-two work. No production topic or navigation was changed.

## 2026-09-10 — Preserve conditioning populations and correct plot direction

- Status: open
- Origin: [Probability Distributions design](../PROBABILITY-DISTRIBUTIONS-BAYES-DESIGN.md), scoped review of its original calibration preview and the destination's introduction/section 2.1.
- Ownership: this existing published lesson owns calibration definitions, reliability diagrams and validation. The probability foundation will teach only the local population/conditional-rate bridge, with no rewrite of this destination.
- Learning benefit: distinguish useful ranking from calibrated numerical probability, class-probability calibration from confidence-of-correctness calibration, and a finite bin estimate from an exact conditional identity. Explain that Bayes' algebra does not guarantee the assumed population/rates still describe deployment.
- Existing coverage: the inspected destination section defines confidence on x and empirical accuracy on y, then says overconfidence bows above the diagonal. Under those stated axes, accuracy below confidence is below the diagonal. Its exact plot and broader claims still need scoped review; this note does not endorse the rest of the page.
- Proposed treatment: use small declared bin counts with uncertainty, a changed base-rate example under explicitly unchanged class-conditional behavior, and a comparison that preserves ranking while changing numerical scores. Resolve the axis-direction inconsistency against actual rendered data. Keep exchangeability/population/selection assumptions adjacent to any conformal coverage claim.
- Prerequisites: conditional probability, joint populations, repeated sampling and the difference between confidence and correctness. No empirical data or universal calibration behavior is inferred from the local toy.
- Evidence: current destination JSX section 2.1 and probability body's previous “only useful when calibrated” sentence were inspected on 10 September 2026. The elementary diagonal counterexample is x=.9, y=.6. Primary calibration/conformal sources and the destination's full plots require fresh research when that topic is authorized.
- Resolution: awaiting the destination author; no production code there changed.
- Implementation/verification links: origin design above; no completed destination verification.


## 2026-09-11 — A calibrated score need not preserve every useful decision distinction

- Status: open; complements the existing conditioning/axis note, without replacing it.
- Origin: [Decision Theory53 design](../DECISION-THEORY-LESSON-DESIGN.md), probability versus action and coarsened-information sections.
- Destination rationale: this published destination owns reliability definitions and probability calibration methods. Decision Theory teaches the complete local decision consequence; full calibration fitting, conditional/group guarantees and conformal scope belong here.
- Actual inspected coverage: destination §§2.1/3.1–2 describes confidence bins and fitted calibration; the existing note already identifies the axis-direction defect. This narrower read does not establish whether every later section addresses decision resolution.
- Concrete example: two equally likely groups have true positive rates .1 and .3, but both receive score .2. The score is calibrated in aggregate. With loss80 for releasing a faulty item and quarantine cost20 for either state, using only the score releases everyone at expected loss16; using the available group information quarantines the .3 group, reducing expected loss to14. Calibration with respect to a score is not identity with the posterior given richer information. For fixed state-dependent costs, a calibrated score can still support the optimum among policies using that same score; varying costs within score bins requires conditioning on the relevant joint information.
- Proposed treatment: incorporate or adapt this after defining the exact conditioning population, then separate calibration, resolution/ranking, proper forecast scores and a particular operational loss. A lower Brier loss alone is not proof of better calibration or lower cost under every threshold. Explain that observed finite-bin rates add estimation uncertainty to these exact model identities.
- Evidence: [scikit-learn calibration guide](https://scikit-learn.org/stable/modules/calibration.html), opening proper-score note and §1.16.1, inspected11September2026, page1.9.1; [design arithmetic](../evidence/decision-theory-design-checks.json) records the proposed exact group fixture. This is not a new benchmark or a universal calibration-method ranking.
- Resolution: reassess this bridge when the destination is authorized. The complete Decision Theory body now teaches and executes the coarse/full-information comparison; see its [author verification](../DECISION-THEORY-VERIFICATION.md) and [exact source packet](../evidence/decision-theory-author-review.json). Independent review/integration are separate. No calibration production source was edited and this destination note remains open.
## Naive Bayes score-interface and fit-boundary discovery — 11 September 2026

Status: open for the destination's scoped rewrite. Origin is now author-reviewed: [Naive Bayes verification](../NAIVE-BAYES-VERIFICATION.md) and [exact source/evidence packet](../evidence/naive-bayes-author-review.json). Independent review/integration remain separate. Retain the earlier conditioning/axis and Decision Theory findings above.

- The new origin supplies a complete separate base-fit/calibration/test experiment with sklearn1.9.1 GaussianNB and FrozenEstimator. GaussianNB has no decision_function; CalibratedClassifierCV therefore fits its sigmoid to predict_proba values, not automatically to generative log odds. A sigmoid of an already saturated probability has a different family from a sigmoid of the original log odds. Explain the actual score interface before suggesting a method. A bounded native duplicate-feature fixture illustrates this, including possible changed argmax and a fitted probability range that affects cost thresholds.
- A lower test Brier/log loss is useful but does not isolate calibration, prove subgroup calibration or imply lower cost for every threshold. The original Naive Bayes page falsely promises both guaranteed repair and unchanged predicted classes; its rewrite repairs these locally.
- When calibrating a text classifier by cross-validation, feature extraction must be inside the estimator cloned in each fold; fitting a vocabulary or supervised feature selector before constructing calibration folds can leak information. A genuinely independent calibration split can instead use a frozen already-fitted complete pipeline.
- Destination should explain raw score versus probability input, independent predictions, ensemble averaging versus one refitted base model, and how class absence/small calibration samples alter interpretation. Use actual library contracts, finite-bin counts, a changed-input experiment and an example acceptable report. Keep conformal coverage distinct.
- Sources inspected: [sklearn calibration guide](https://scikit-learn.org/stable/modules/calibration.html) and actual installed1.9.1 program. The origin's author verification will be linked when frozen. No destination production body was changed.


## SVC calibration migration and whole-procedure data roles — 11 September 2026

Status: open for this destination's scoped rewrite. Origin: [SVM author verification](../SUPPORT-VECTOR-MACHINES-VERIFICATION.md) and [exact source/evidence packet](../evidence/support-vector-machines-author-review.json). Preserve the earlier population/axis, Decision Theory and Naive Bayes notes.

- The current scikit-learn 1.9.1 SVC API deprecates its probability parameter, with removal scheduled for 1.11, and recommends explicit CalibratedClassifierCV with ensemble=False. The SVM origin now executes a complete Pipeline(StandardScaler, SVC) inside that wrapper, with fixed hyperparameters, training-only calibration folds and a separate test set. Future destination author should verify the then-current API rather than perpetuate older probability=True code.
- Explain the entire estimator's score interface. SVC supplies decision_function; Naive Bayes may instead supply predict_proba, as the earlier note records. With ensemble=False, out-of-fold scores fit the calibrator and the base estimator is refitted on all supplied training data. Ensemble=True retains/averages fold-specific calibrated models; those are different inference procedures.
- The scaler and other learned transformations must live inside the estimator cloned by each calibration fold. Out-of-fold base predictions do not automatically isolate earlier hyperparameter selection that used the same labels. For a claim about the whole selected procedure, use nested selection or a separate calibration split; frozen fitted estimators require genuinely separate calibration observations.
- The origin reports held-out Brier/log loss with a prior baseline, checks class order and probability normalization, and qualifies subgroup/shift and argmax claims. Adapt the useful local example if it benefits the destination; do not claim that a better aggregate proper score proves calibration or unchanged decisions.
- Sources read: [SVC API](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html), probability deprecation; [CalibratedClassifierCV API](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html), ensemble and estimator-score contracts. Actual version 1.9.1 program executed and rendered in the SVM lesson. No destination production body was modified.
