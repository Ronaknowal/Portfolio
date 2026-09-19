# Evaluation Metrics — independent implementation review

19 September 2026. Reviewer: `semi_supervised_implementation`, separate from the topic's author. Topic ID: `evaluation-metrics-precision-recall-f1-auc-roc-ap-r-mae`. Scope: implementation of the existing prepared-content checkpoint. **Disposition: passed after the corrections below, with no open material findings. Production integration remains the increment owner's separate check.**

## Reviewed material

Read the complete prepared manuscript, design/conservation record, nine figure contracts, four investigation contracts, both complete Python programs, provenance and data-source record. Read the actual replacement narrative, formulas and practice, the pure browser models, figure/lab source and CSS. The finish preflight confirms the existing content checkpoint is complete/current, no topic-specific incoming note exists, and this is authorized phase-two work.

The implementation preserves all eleven teaching sections and eight changed practice tasks, introduces decisions/rankings/probabilities/residuals as different input contracts, and retains the full executable native examples. Retrieval and advanced aggregation remain explicit branches of the first-pass route. Its banknote experiment still reports the development-selected policy's **worse** test cost, 26 rather than 24. It does not optimize a new test threshold or make unverified claims about the source class codes.

Read and reconciled the final [author implementation record](drafts/evaluation-metrics-precision-recall-f1-auc-roc-ap-r-mae/implementation.md) and [author receipt](evidence/evaluation-metrics-author-review.json). Verified every author source hash against the current files. The author's 18 browser groups and 26 inspected captures are separate from this reviewer's checks and inspections; the record accurately distinguishes both roles and the remaining production boundary. The [final independent binding](evidence/evaluation-metrics-independent-final.json) connects this disposition to those exact files and both reviewer receipts.

## Correctness review and complementary numerical checks

Executed the reviewer-owned command:

```text
node scripts/verify-evaluation-metrics-independent.mjs
```

[Source-bound numerical evidence](evidence/evaluation-metrics-independent.json) records **11,829 numerical assertions**, plus structural, validity and exact-byte assertions. These are additional reviewer executions, not a relabeling of the author's checks.

- **240 changed binary datasets.** Used the Mann–Whitney average-rank-sum identity for AUC, independent of the implementation's pair-credit enumeration. AP was rebuilt as the mean, over positives, of the precision at each positive's complete score-block boundary. Tested signed scores, ties, one-class data and reversed input orders. Checked 1,200 threshold/cost/F2 cases using a separate four-bin bit-pattern tally, including infinite endpoint thresholds and zero false-positive cost.
- **Probability boundaries.** Correct certain forecasts retain zero loss; wrong certain forecasts retain infinite mathematical log loss. Binary Brier remains finite and in its stated convention. Scores outside the probability domain are rejected rather than silently clipped into another experiment.
- **60 residual datasets.** Reconstructed SST through the pair-difference identity `sum(i<j, (yi−yj)²)/n`, rather than the implementation's centered-mean sum. Confirmed R², constant perfect targets, RMSE, and linear/quadratic/invariant behavior under unit conversion.
- **720 retrieval cases.** Used all 120 permutations of five stable document IDs, two gain conventions and three cutoffs. Rebuilt discounts with natural-log ratios and AP from relevant ranks. Actual and ideal gains use the same candidate set. All-zero relevance preserves the distinction between zero precision/RR and undefined AP/recall/NDCG.
- **Native evidence integrity.** Exact downloadable program and CSV bytes match the prepared packet and the native receipt's hashes. Embedded code matches the programs, and displayed expected output matches captured native stdout. This review did not perform a second model-fitting run. It independently recomputed every retained development candidate's counts/cost from its raw probabilities and labels, checked the highest-threshold tie rule, and recomputed fixed/selected held-out counts, costs, rank-sum AUC and AP. The selected policy's cost failure remains 26 versus 24.

The author's separate native execution is recorded in [evaluation-metrics-native.json](evidence/evaluation-metrics-native.json). Its recursive comparison covers the retained fixture/prediction arrays, rather than just a few aggregate numbers. The reviewer reuses that actual execution while independently checking its input/output provenance and selected numerical consequences. No claim of freshly refitting the native model, revisiting every external reference, or watching the linked video is made here.

## Findings and closure

### P2 — full-ranking exposure did not retire future graded questions

The original complete-ranking disclosure exposed every future coordinate, but only the current step's fingerprint was marked seen. A learner could inspect those answers and continue to a supposedly unseen graded step. Reported this from the live component source before the author's final campaign.

The author replaced the passive disclosure with an explicit complete-ranking exploration action, retaining a canonical stable-ID-sorted set of fully exposed datasets. Future steps for that dataset stay ungraded after an edit-and-restore cycle or a tied-card display permutation. Ordinary input changes also retire the old response rather than resurrecting it on restoration. **Resolved and independently exercised in the browser.**

### P2 — the numerical tie policy changed when only units changed

Starting from perfect predictions, changing only T1's A prediction to `1.00000000001` produces MAE difference about `2.00e−12` and RMSE difference `4.47e−12` minutes. An absolute `1e−10` tolerance on displayed-unit metrics called these equal in minutes but unequal in seconds (`1.20e−10` and `2.68e−10`). Unit conversion alone must not reverse the comparison rule.

The author now grades preferences in base-minute units and explains that tolerance explicitly. Scientific-notation deltas make tiny differences visible even when six-decimal summary values agree. **Resolved: the same equal-error prediction is accepted in both units in the independent browser test.** This is a documented numerical comparison convention, not a claim that unequal real numbers become exactly equal.

### Accessibility and display fixes

Wrapped selects originally incorporated all option text into their accessible names. The reviewer encountered this in exact-label interaction and inspected the live labels. Explicit concise names now support the prediction, unit and other selection controls. A desktop table also split the token “undefined” across lines; the author changed metric-cell wrapping while retaining local horizontal scrolling. Both fixes preserve the underlying calculations. The final score ruler additionally places tied items at the same quantitative position and uses that mapping for its threshold gate.

## Independent browser evidence

Executed:

```text
node scripts/review-evaluation-metrics-independent.cjs
```

[Browser receipt](evidence/evaluation-metrics-independent-browser.json) binds the actual implementation and verifier source, contains four screenshot digests, and records 23 complementary grouped assertions plus the bounded figure follow-up below. Used headless Edge, reduced motion and a 390 × 900 viewport against the shared development server at port 4184.

The review covers a newly edited C score, the entire full-reveal → next step → edit/restore → display permutation lifecycle, a new threshold-interval null (`0.73 → 0.74`), the tiny residual tie in both units, perfect constant-target R², swapping the two grade-zero document identities, and the no-relevance state. Predictions start unset, are explicitly committed before graded reveals, and these changed/null cases receive the answers their displayed calculations support. No runtime exceptions or page-level horizontal overflow occurred.

Quantitative figure assertions verify the tied B/C x positions and the threshold gate against their declared score scale, as well as the ROC endpoints and the tied-block `(FPR=.25, TPR=.5)` coordinate. The browser script refuses to certify a run if a reviewed source file changes while it is executing.

The author subsequently narrowed the measured-cost chart's SVG coordinate width, enlarged its tick text and explicitly described the excluded no-alert endpoint. Only the affected figure/CSS were reviewed again:

```text
node scripts/review-evaluation-metrics-independent.cjs --measured-followup
```

Sixteen follow-up assertions at 1440, 390 and 320 px verify painted tick size of at least 11.5 px, label containment, the selected threshold's exact new mapping, the stated no-alert cost 195, page containment and absence of runtime errors. Inspected the added `measured-followup-390.png` capture. The receipt retains the earlier source binding in its follow-up history, verifies that all behavioral files stayed unchanged and advances the current source hashes. The passing behavioral scenarios were reused rather than needlessly replayed after this purely visual refinement.

## Separate learning-experience assessment

This is a reviewer heuristic walkthrough, not a novice study. The opening alert scenario supplies a concrete reason to ask what a metric means; denominators are then grounded in named examples before any area or loss summary. The lesson avoids treating F1 as universal, AP as a trapezoid, probability loss as a threshold score, or R² as a guarantee that an honestly trained constant baseline scores zero. The whole-curve versus area comparison and failed development-to-test advantage give useful counterexamples.

The visual mechanisms vary with the concept: a quantitative score gate and named confusion cells; fixed-rate population/alert counts; grouped ROC/PR views; a positive-negative credit grid; probability rulers and separately scaled losses; residual lengths and squared areas; ranked document shelves with discounted gain; and a measured development/test decision comparison. The same neutral inputs connect these views without forcing one generic lab layout.

Inspected informative author captures for grouped curves at 390 px, probability penalties at 390 px and the actual development/test comparison at 1440 px. After the final ruler/token-wrapping change, also inspected `threshold-390.png` and `residuals-1440.png`: tied markers sit together at the gate, and the one large residual square is correctly nine times each moderate square's area. Inspected all three independent captures: an exposed ranking remaining ungraded, a zero-error constant-target result and a no-relevance retrieval result. These verify meaningful changed/null states, not only initial panels. Numeric tables remain the exact, keyboard-scrollable alternative when a narrow viewport cannot show every column. Shape, explicit labels and counts supplement color.

The residual graphics share a square scale across both predictors, distinguish signed residual from nonnegative area, and show the evaluation-mean reference separately from the fitted training constant. Ranked shelves retain stable document identities and use a common contribution scale for actual versus ideal. Empty denominators receive explanations, and the measured example distinguishes observed data from illustrative costs.

The implementation treats already worked or already revealed states as exploration, with editable new cases available for independent predictions. Full-ranking exploration now explicitly ends graded continuation for those exposed values. Practice hints and solutions stay separate and initially closed. Alternate resources explain their purpose, including the known correction attached to the ROC video, without claiming the video was watched.

## Evidence limits and integration boundary

The linked reviewer receipts contain exact SHA-256 bindings for the browser-model/native artifacts, the reader, figure and lab components, CSS, data and reviewer scripts. Reuse the author's broader widths, native outputs and visual checks only where their own source/capture bindings still match. Any later figure/CSS refinement requires affected visual checks and refreshed bindings; unchanged native programs do not need another fitting run.

This review does not claim screen-reader software testing, a novice user study, exhaustive arbitrary-input testing, or final production loading/performance validation. The increment owner owns the final build, routes, shared catalogue, phase ledger and production integration. This file is an independent review record, not a content-checkpoint replacement or user acceptance.
