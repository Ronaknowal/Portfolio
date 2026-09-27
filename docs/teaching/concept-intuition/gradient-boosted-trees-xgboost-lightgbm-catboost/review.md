# Gradient-Boosted Trees: full concept-transition review

26 September 2026. Directly authored production JSX is the current manuscript. Reviewed complete text, all program explanations and practice; preserved all executed examples and original live investigations.

| Location / transition | Existing support and revision |
| --- | --- |
| §1 baseline → residuals → saved additive predictor | Six-row exact example and additive prediction figure already coherent. Retained, including distinction between training residuals and inference. |
| §2 leaf mean → shrinkage → split choice → interaction capacity | Complete derivations, live correction lab, stump and recursive programs suffice. Retained. |
| §3 negative gradient → function-valued update | Added requested-versus-expressible correction bars on the same six rows: one tree cannot satisfy six independent requests. This grounds functional gradient descent before the general loss table. |
| §3 logits, stable derivatives, multiclass caveat, quantile objective | Correct same-score updates, numerical examples and full quantile minimizer reasoning retained. No universal Newton or quantile claim added. |
| §4 gradient/curvature → shared leaf correction | Added direction-versus-changing-direction explanation before aggregation; calculated two-curvature diagram shows why equal G produces different w. |
| §4 L1/L2/leaf cost → optimal weight → parent/child gain | Existing branch derivative, square completion, exact five-row numbers, live Newton/surrogate-vs-actual lab sufficient. Retained. |
| §5 exact prefix → histogram → missing routes | Existing histogram lab and exact lost-boundary/default-route examples sufficient. Retained. |
| §5 growth policy → leaf budget/depth → computation | Existing tree-shape diagram plus explicit upper/worst-depth examples retained. |
| §6 inclusion weighting → expectation → nonlinear gain | Added two-valued estimate showing E[G²]≠E[G]² before discussing alternatives. Existing enumerated GOSS lab and actual program retained. |
| §6 exclusive feature bundling | Original code-slot diagram and exact decoder already explain it. Retained. |
| §7 target statistics → eligible labels | Existing prefix-statistic lab directly exposes own-label exclusion. Retained. |
| §7 Ordered boosting versus ordered statistics | Added side-by-side dependency diagram for the actual miniature's row-4 second-round residual (M3 versus M4), clarifying a second path rather than duplicating category encoding. |
| §8 generalization → early stopping → selected prefix | Existing validation lab, patience example and independent practice suffice. Retained. |
| §9 normal libraries → indices/counts → saved schema | All three complete CPU recipes, save/reload and categorical shape checks retained. Added bridge separating representation, candidate questions and permitted training information before native-categorical comparison. |
| §10 importance, monotonicity, weighting, systems, failure checks | Existing mechanism-specific examples/derivations and practical contract table retained. No invented device or speed claim. |
| §11 changed practice/capstone | All original changed tasks and report criteria retained; new explanation supports the existing Newton-loss and GOSS-fraction tasks. |

## New visual contracts

- `BoostingDirectionFigure`: initial residuals [−3,−3,−2,2,3,3] and best x=3.5 stump corrections [−8/3,−8/3,−8/3,8/3,8/3,8/3]. Signed bars share a scale; exact rounded numbers remain beside each row. Layout makes the two repeated responses visible. Rate 0.5 is stated separately so a displayed full correction is not mistaken for the applied step.
- `BoostingCurvatureFigure`: same gradient G=−2 and λ=1, H∈{1,4}. Curves −2w+(H+1)w²/2 have minima at 1 and .4. Shared axes include the entire displayed range, including value 6 at H=4,w=2. Dots are derived minima, not measured fits. HTML text provides explanation; narrow view stacks the two plots.
- `OrderedBoostingDependencyFigure`: follows the already executed constant-learner program's row 4: eligible M3 predicts 1, full M4 predicts 1.5, observed target 6, residuals 5 versus 4.5. Arrows express information dependency. Explicitly distinguishes this miniature from full CatBoost and from target encoding. No claim that ordinary boosting is an invalid procedure.

## Research actually read

Read [XGBoost's authored model tutorial](https://xgboost.readthedocs.io/en/stable/tutorials/model.html), additive training, Taylor surrogate and structure-score derivations. Used its problem-to-objective relationship as a teaching comparison; new examples are local and preserve exact-versus-surrogate distinctions. Read [CatBoost's categorical transformation documentation](https://catboost.ai/docs/en/concepts/algorithm-main-stages_cat-to-numberic) to keep category statistics distinct from the already cited primary paper's Ordered boosting. Existing creator video annotations remain; no new full-video viewing claim. Library outputs still describe their recorded 3.4.1/4.7.0/1.2.10 environment, not a silent re-execution on newer documentation.

Author checked JSX, correction means and resulting MSE, quadratic minima and full plotted domain, the nonlinear expectation counterexample, and arithmetic of preserved prefix-model states. Source hashes in `author-checks.json`. All native training programs/results unchanged. Root owns browser and independent review closure.
