import fs from 'node:fs';
import assert from 'node:assert/strict';

const bodyPath = 'src/learn/data/topics/gradient-boosted-trees-xgboost-lightgbm-catboost.jsx';
let body = fs.readFileSync(bodyPath, 'utf8');
for (const [before, after] of [
  ['python -m pip install pandas</Code>', 'python -m pip install pandas==3.0.1</Code>'],
  ['If H=λ=0, the quadratic has no unique finite minimizer from this formula; a numerical division by zero is not an alternative definition.', 'If H=λ=0, this division formula is undefined. An L1-only or linear objective must be analyzed separately; a numerical division by zero is not an alternative definition.'],
  ['are optional visual companions to residual fitting; their conceptual scope is separate from the current library API recipes here.', "are optional visual companions to residual fitting. The creator's chapter list places the first tree at 5:50, the second at 10:37, and new-row prediction at 13:50 in the main-ideas video. Their conceptual scope is separate from the current library API recipes here."],
  ['Friedman: Greedy Function Approximation, 2001</a> develops', 'Friedman: Greedy Function Approximation, 2001</a> (with an accessible <a href="https://www.cse.iitb.ac.in/~soumen/readings/papers/Friedman1999GreedyFuncApprox.pdf" target="_blank" rel="noreferrer">1999 manuscript</a>) develops'],
]) {
  assert.equal(body.split(before).length, 2, before);
  body = body.replace(before, after);
}
fs.writeFileSync(bodyPath, body);

const path = 'docs/teaching/GRADIENT-BOOSTED-TREES-LESSON-DESIGN.md';
let design = fs.readFileSync(path, 'utf8');
design = design.replace('The immediate prerequisites are **Decision Trees & Random Forests**, **Linear & Logistic Regression** and **Python Basics: Types, Control Flow, Functions & Modules** (exact display names will be checked against the inventory before registration).', 'The registered conceptual prerequisites are **Decision Trees & Random Forests** and **Linear & Logistic Regression**, verified against the inventory. The optional executable branch also uses basic Python; its setup and operations are explained locally.');
design = design.replace('One actual uneven response fixture yields level-wise, best-first and symmetric-test topology.', 'A specified hypothetical frontier-gain fixture yields distinct level-wise and best-first topology; an adjacent explanation distinguishes the additional symmetric-test constraint.');
design = design.replace('Static computed curve and a same-model, same-holdout permutation example, with a correlated-feature caution.', 'The computed validation predictor also supplies exact horizontal extrapolation pieces. Actual LightGBM output distinguishes split count and summed gain; permutation importance is explained as a different experiment with correlated-feature caveats. An extra unsupported heatmap is unnecessary.');
design += `

## Final implementation and research disposition

The design has been implemented. See [the author verification](GRADIENT-BOOSTED-TREES-VERIFICATION.md) and its exact-source evidence packet for current status; independent review and production integration remain parent-owned. The original plan above is retained as rationale, not a claim that every initially proposed form was necessary.

- The finished lesson contains six distinct investigations and three ordinary-reading figures. The validation view now derives every horizontal prediction segment from the saved split thresholds; vertical joins denote discontinuities, not interpolated predictions. The loss axis begins at zero. The growth comparison explicitly uses hypothetical candidate gains to isolate a policy decision; it does not claim those scores are a fitted library benchmark. Its actual topology, leaf counts and depths are computed from its decisions.
- Extrapolation shares the validation predictor instead of duplicating the same curve. The importance section uses actual split-count/summed-gain output and explains permutation's different estimand; it does not manufacture a second cross-library importance display. Deeper SHAP/causal attribution remains with the established later interpretability owners. No new unassigned topic is needed for this scoped lesson.
- The original complete recursive learner and dataset remain in a repaired executable program. It now returns an unsplittable leaf and clears trees on refit. The old five-row split calculation remains. All three library workflows retain their useful objectives/datasets but use explicit train/validation/test roles and current inference contracts. Old unsupported curves, package rankings and claimed outputs are archived, not relabeled as verified.
- During author review, the changed quarter-rate practice was repaired to predictions 1.5/2.5 with MSE 2.25. An adjacent-float feature check exposed midpoint rounding to the upper observation; both the browser model and actual first Python stump now retain the legal lower-bound threshold in that case. Original displayed stdout remains unchanged by that boundary repair.

### Completed source review and honest alternate-resource limits

Friedman's actual 1999 manuscript was retrieved from the [IIT Bombay-hosted original PDF](https://www.cse.iitb.ac.in/~soumen/readings/papers/Friedman1999GreedyFuncApprox.pdf). Pages 0–7, particularly Algorithm 1 and the squared/LAD leaf-specific updates, were read: finite prediction-vector descent, restricted regression-tree direction, line search, and the loss-specific leaf distinction. The 2001 publication DOI remains linked with the accessible manuscript. The earlier failed author-host/journal requests are not counted as reads.

The current XGBoost categorical/prediction, monotonicity and parameter documentation was read, including native-versus-wrapper iteration behavior, categorical serialization, available grow policies and constrained histogram-candidate loss. LightGBM's Features, Parameters, Advanced Topics and complete early-stopping callback page were read: negative categorical codes/missing values, sparse-zero flags, explicit GOSS/bagging activation, OpenCL versus CUDA, and DART's callback limitation. CatBoost's current relevant common-parameter sections and complete Spark limitations page were read: CPU Plain defaults versus explicitly requested Ordered, growth-policy restrictions, best-model retention, unsupported Spark Ordered/GPU/Windows features, and separation from the local API. These documentation claims are separate from executed CPU fixtures.

All pinned libraries were installed centrally and actually executed: Python 3.12.14, NumPy 2.3.5, scikit-learn 1.9.1, XGBoost 3.4.1, LightGBM 4.7.0, CatBoost 1.2.10 and pandas 3.0.1. The current CatBoost class fixture returns a one-dimensional array. Its deliberately introduced column-vector comparison illustrates a broadcasting hazard; it is not presented as an API defect that occurred in that run.

The XGBoost model tutorial is the substantively read, annotated beginner-friendly written alternative. StatQuest's creator index, exact video identities, expanded creator description, prerequisite list and chapter list were inspected in an actual browser. Its main-ideas chapters are first-tree 5:50, second-tree 10:37, new-row prediction 13:50. The page's transcript drawer did not populate; selected video seeks produced loading/error frames. These attempts are saved in scratch/gradient-boosted-trees-verification/resources. No full or partial substantive video watch is claimed. The links are verified optional companions with creator-declared scope, not relied upon for technical claims. CatBoost's official video index and its ordered/categorical embedded video identity were verified; the corresponding paper was substantively read, but the video itself was not watched. This limit does not weaken the independently derived lesson or its executed results, and future resource review must not turn metadata checks into a fictional watch history.
`;
fs.writeFileSync(path, design);
console.log('Closed owned design dispositions and made the final narrow body clarifications.');
