# Authoring notes: Imbalanced Learning (SMOTE, Cost-Sensitive Learning)

Canonical topic ID: `imbalanced-learning-smote-cost-sensitive-learning`

## 2026-09-11 — Preserve rows and probability meaning when extending imbalance methods

- Status: open
- Origin: `multi-label-multi-output-learning`; [scoped design](../MULTIOUTPUT-LESSON-DESIGN.md).
- Destination and ownership rationale: this topic owns resampling and cost-sensitive fitting; Multi-Label teaches its local threshold/annotation contract but should not add a general SMOTE chapter. No destination body was edited.
- Existing coverage: the actual current body discusses SMOTE interpolation around lines117–132 and threshold moving around lines73 and142–178. Its threshold prose says raising a threshold increases precision; empirical precision need not be monotone. The multi-label origin's old body also recommended per-label SMOTE without accounting for the other labels on each row.
- Proposed treatment: when this destination is rewritten, distinguish row-level resampling from manufacturing separate feature rows per label and then pretending they are one aligned multi-output dataset. A synthetic feature point requires a justified complete-label/missing-label policy; interpolating features does not establish every other label. Fit any resampler inside training folds and keep related groups together. Add one concrete conflicting-label neighbor example rather than an unsupported warning list.
- Explain class-weighted BCE score meaning: under positive weights `w+`, `w−`, its population optimum is `q=w+ p/(w+ p+w−(1−p))`. At `p=.1,w+=9,w−=1`, q=.5, not original-population probability .5. Threshold selection and probability calibration are different operations; prevalence alone does not invalidate equal-cost threshold .5 for an actual posterior.
- Precision counterexample: descending scores `.9,.8,.7` with truth `[0,1,1]` give precision2/3 when all are selected,1/2 for the first two and0 for only the first. Recall is monotone with the selected set; precision can move either way. Show actual stair steps and ties, not an interpolated guarantee.
- Evidence: current scikit-learn1.9.1 [threshold guide](https://scikit-learn.org/stable/modules/classification_threshold.html), inspected11September2026, distinguishes fitted scores and decision thresholds and train/validation ownership. Weighted-score identity follows by differentiating binary expected weighted log loss; the origin's exact design-fixture record checks the numerical case. The destination must inspect appropriate resampling primary sources before implementing a specific multilabel extension.
- Resolution: not yet assessed by the destination author. Reassess current text and retain, adapt or reroute these scoped bridges; do not treat the note as permission to reopen the topic now.
- Implementation/verification links: origin design only; destination implementation remains pending.
