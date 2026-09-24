> **Current interaction amendment, 21 September 2026:** Read [live-exploration.md](live-exploration.md). Labs now update directly from valid edits and contain no learner prediction feature, including optional predictions. The older prediction/commit/reveal clauses below are historical design records; their numerical, scope, layout and evidence requirements remain applicable where unchanged.

# Imbalanced Learning — visual and investigation contracts

Content-only specification, 12 September 2026. Nothing in this file is a claim of implemented UI, executed browser code or phase-two review. Read `lesson.md`, `design.md`, `data-provenance.md` and the retained `calculated-inputs.json` together. Stable topic identity is `imbalanced-learning-smote-cost-sensitive-learning`; the proposed display-title expansion does not change routes or progress.

## Teaching sequence and representation choices

The first-pass manuscript places a case-flow diagram before metrics, an editable score queue before threshold theory, crossed expected-cost lines before weighting, and a weighted-loss curve before geometric resampling. The real study then combines these ideas. A later population/loss-mass branch supplies more depth without becoming hidden first-pass readiness. Implement figures inline at those explanations; do not move all visual work into one final laboratory section.

These are different mechanisms. A queue should look like identifiable records passing a gate, risk should look like two competing numerical expectations, weights should alter a loss landscape, and SMOTE should look like actual points and an interpolation segment. Use the website's typography and accessible controls consistently while preserving these distinct forms. Existing general-purpose controls may be reused; a generic output textbox is not an adequate substitute for the named representations.

All numerically meaningful illustrations are either exact declared constructions or saved observed-data outputs. The Yeast experiment is a real, fixed five-procedure comparison on an explicit subset. Synthetic geometric points and hypothetical error costs are labeled at their origin. There are no invented timing curves, simulated empirical uncertainty bands or unmeasured probability-calibration claims.

## Shared investigation behavior

Each investigation begins with **no prediction selected and no result revealed**. Inputs, question and labels are visible. A learner edits actual entities or meaningful model/decision quantities, records a prediction, then presses Apply/Reveal. Changes after reveal invalidate the result and its feedback; the historical prediction may remain visible as a clearly previous attempt, but the new input revision requires a new prediction. Capture the complete active input tuple, question, source revision and prediction, not just a chosen preset name. Reset restores declared inputs and clears the prediction/result. Do not auto-answer the prediction when a slider moves.

The prediction can be categorical with a short optional explanation, an entered numerical quantity, or a selected record set as specified below. Expected-answer fixtures support feedback only after reveal; before reveal the interface must not display a derived label that answers its own question. Each result explains the actual computation and interprets the difference from the learner's prediction. A numerically close answer alone is not enough when the task is to distinguish two objectives.

All controls have persistent labels, units where meaningful, focus indicators and keyboard operation. Editable points also have a numeric-coordinate table, editable record labels have ordinary form controls, and charts have corresponding numerical/text representations. Color is redundant with class text, shape, line style and selected-state labels. Keep the main explanation and unsolved task available at narrow widths; wrap short summaries and stack the visual/table rather than shrinking labels. Avoid horizontal page overflow and tiny 1,000-cell grids. A scrollable record table may have its own labeled region. Respect reduced motion and never rely on animation to reveal a counted case.

Report mathematical undefined values as such with their relevant denominator: precision when no cases are selected, recall when there are no positives, and cost-ratio cutoff when both costs are zero. This is an interpretable state, not a fake zero plus a microscopic denominator. Invalid inputs should identify the field and retain other edits. Do not flood the page with repeated caution text; local sections own the interpretation boundary.

## F1 — Population bands and a confusion table

**Location:** §1, after the first case counts. **Question:** how can a model be less accurate but find useful rare events?

Use the exact construction: 1,000 total, 20 actual positives, 980 actual negatives. Actual-model TP14, FN6, FP18, TN962; baseline TP0, FN20, FP0, TN980. Display a population band for each actual class with detected/missed or cleared/false-alarm subdivisions. Band widths within a class represent its proportions; prominently label that the two class bands are separately normalized. Beside them, show the overall class counts and the exact four-cell table. This avoids making 6 missed cases invisible while also avoiding the impression that the classes have equal prevalence.

Below the two panels, show baseline accuracy98%/recall0 and model accuracy97.6%/recall70%. Keep accuracy, recall and counts in separate labeled columns. The number of positive decisions is32, precision14/32=.4375. Do not add an invented monetary conclusion before §3 defines costs.

Text equivalent: “Among20 positives,14 are detected and6 missed. Among980 negatives,18 trigger an alert and962 do not. The always-negative rule detects none but makes only20 errors rather than24.” On mobile, order population totals → class bands → count table → two-model comparison.

**Phase-two oracle:** each actual-class row sums to its population count; prediction-column totals32/968; all four counts sum1,000; percentages derive from counts, not separately stored rounded values. Screen-reader order identifies actual vs predicted class. No lab prediction is necessary for this static setup.

## I1 — Move the gate through an editable scored queue

**Location:** §2 before prevalence and AP. **Mechanism:** thresholding a fixed ranking selects whole equal-score groups; recall is monotone as the gate rises, empirical precision need not be.

Default records in stable ID order: A(score.9,truth0), B(.8,1), C(.7,1). Starting threshold.7, default target threshold.9. The task asks: “Starting with all three selected, will precision rise, fall or stay equal if only scores at least.9 are selected?” Do not show the precision answer in the initial gate preview. The learner may instead edit the target threshold before recording their choice. The current record table shows inputs; the result table/counts are reveal-gated.

Allow individual score edits in[0,1], truth0/1 edits and add/remove of named records up to12 with at least one record retained. These are actual unsolved inputs, not preset buttons alone. Show a score rail with one identifiable card per record and a numerical threshold input plus keyboard slider. Use ID and shape to distinguish positive/negative truth after reveal. At Apply, cards enter selected TP/FP or unselected FN/TN bins with exact IDs; a table always carries the same information.

The result includes selected IDs, TP/FP/FN/TN, precision and recall, plus an exact threshold-step chart over distinct score values and the above-all-scores state. Apply the rule score≥t consistently. Do not connect precision steps with a smooth monotone line. The initial counterexample at t.7,.8,.9 yields precision2/3,1/2,0 and recall1,.5,0. Above.9 gives no selections, undefined precision, recall0. After changing A's truth to1, all three are positive: precision remains1 at nonempty thresholds. That is a real contrast created by editing an input label.

**Tie investigation/null:** let the learner set B and C both to.8 and change their display order. Threshold.8 selects both; counts and AP must not change. For a more diagnostic contrast, use practice records scores[.95,.8,.8,.4],truth[0,1,0,1]. At.8 the selected IDs contain both tied records, precision1/3, recall.5. A separate optional exact-top-k view may expose ID-based tie breaking, but it must not silently reuse threshold behavior or order by truth.

After AP is introduced, an expanded result can show the grouped recall increments and their precision contributions. Distinct scores with descending truth[1,0,1,0] give AP5/6. A constant score gives prevalence when positives exist; no-positive AP is reported as undefined in this teaching view, with an optional explicit library-convention note. Keep this extension after its local definition, not in the first task.

**Recorded prediction tuple:** records(id,score,truth), start and target threshold, rule≥, question, optional ranking mode/tie rule, prediction and input revision. Comparison tolerance for entered ratios1e−6; display exact fractions when small. Bound recomputation to at most12 records. Mobile rail becomes a vertical ordered list with the numerical gate and the same bins/table.

**Phase-two checks:** the default counterexample; all-positive and all-negative inputs; all-selected/no-selected; duplicate scores; identity-preserving display permutation; changing a record after reveal clears correctness feedback; no interpolation between unobserved threshold operating points; genuine keyboard editing and accessible undefined state.

## F2 — The same conditional detector in two populations

**Location:** §2 beside the prevalence equation. **Mechanism:** a fixed false-positive rate acts on a much larger negative population.

Use population10,000, TPR.8, FPR.01. Panel1 prevalence.01: positives100→TP80/FN20; negatives9,900→FP99/TN9,801; precision80/179≈.446927. Panel2 prevalence.001: positives10→TP8/FN2; negatives9,990→expectedFP99.9/expectedTN9,890.1; expected precision8/107.9≈.074143. Explicitly label fractional flow counts as expectations under the declared rates. Do not round99.9 to100 then display precision from the unrounded denominator without explanation.

Draw two class-origin flows into an alert tray. Show the same fixed ROC marker(FPR.01,TPR.8) beneath both and distinct precision values. Population-flow counts and rates are supplied assumptions; this is not a retrained classifier or measured drift experiment. The fixed-conditionals assumption belongs in the caption once. Text equivalent states both sets of counts and explains which quantity stayed fixed.

If implementation adds optional population controls, enforce valid nonnegative population/π/TPR/FPR and show expected rather than fabricated integer cases. The illustration is complete without those extra controls; do not add a lab simply to make every figure interactive. Use separate rate and count axes, with a zoomed labeled ROC inset if the.01 FPR would otherwise be imperceptible.

**Phase-two checks:** exact count conservation, precision equation, both conditional rates unchanged, denominator-zero interpretation, no overloaded axis that visually suggests a changed ROC point.

## I2 — Cross the two expected-cost lines

**Location:** §3 immediately after the two-action rule. **Mechanism:** choose the lower expected cost given a posterior and a stated cost model.

Default FP cost1 unit, FN cost12 units, case posterior.1, correct-action costs0. Controls edit both costs over0–50 in.1 increments and p over0–1; numeric inputs permit more precision. The first task asks the lower-cost action and optionally the two risks. Costs are explicitly hypothetical units per mistake, not measured laboratory prices. No line crossing or selected action is displayed before the prediction; the raw chosen values are enough to solve it.

On Apply, graph posterior p on[0,1] horizontally and expected cost vertically. Draw Rselect=(1−p)cFP and Rskip=pcFN with labeled line styles; mark their crossing and the selected case. Show actual risks.9 and1.2 and cutoff1/13≈.076923 for the default. Selecting wins. At equal risks the decision is a tie; use the manuscript's select-on-equality convention for a concrete action output while stating that either action has equal risk.

**Required contrast:** change p from.1 to.05 with costs fixed: selection risk.95 versus skipping.6, so skip. **Required null:** multiply both costs by3, holding p=.1. Risks become2.7 and3.6 but the cutoff and action remain unchanged. Ask the learner to predict action invariance before reveal; do not leave the previous result visible as current. **Changed exercise:** FP2,FN7,p.2 gives risks1.6/1.4 and cutoff2/9, so skip.

When FP cost0,FN>0, all p>0 favor selection; p0 ties. When FN cost0,FP>0, all p<1 favor skipping; p1 ties. When both are zero, every action ties and there is no unique cost cutoff. Handle these cases directly. These costs are not values to pass as inverse-prevalence weights automatically.

**Recorded tuple:** p,cFP,cFN,correct costs0,tie rule,chosen action/predicted risks and revision. Recompute exact formulas synchronously, no simulation or training. Keep y-axis units and a common scale for the two lines; for large ratios, annotate the small crossing and retain exact numerical risks rather than relying on pixel separation. On mobile, risk tiles precede the graph and their comparison is announced as text.

**Phase-two checks:** default/contrast/common-factor null/changed practice, endpoints and zero costs, equal-cost cutoff.5 independent of an irrelevant prevalence control (do not add such a control merely to test this), stale input resets, graph/table values agree and no action is preselected.

## F3 — One weighted gradient update

**Location:** §4 after gradients, before class-weight conventions. **Mechanism:** weights modify contributions to an additive objective; they do not add observed information.

Use two rows A(x0,y0,a1), B(x2,y1,a3), b=w=0, λ.01, learning rate.4. At the initial parameters the penalty gradient is0. Initial q=.5 on both rows; residuals .5 and−.5. Intercept numerator contributions .5 and−1.5 sum−1; divide byA=4→−.25. Coefficient numerator contributions0 and−3 sum−3; divide by4→−.75. Update b=.1,w=.3. The new scores are sigmoid(.1)=.524979 and sigmoid(.7)=.668188.

An inline row-contribution table flows into two separate sum boxes, then the updated line on a score-versus-feature plot. Label the plotted y-coordinate predicted score, not empirical event frequency. Show the original constant.5 and updated sigmoid on x from0 to2; both endpoints labeled. One update is illustrative, not a converged fitted model. Use a textual arithmetic chain below, preserving sign and normalization.

**Phase-two checks:** saved `constructed.weighted_single_step`; exact parameter update; no penalty on the intercept; common scaling of all weights cancels only because this objective divides by their total; before/after curve from parameters, not hand-drawn points.

## I3 — Weighted loss and posterior meaning

**Location:** §4 after the optimum and inverse derivation. **Mechanism:** population weighting transforms the optimal score even when the underlying event probability is unchanged.

Default p.1,wPositive9,wNegative1. Forward mode asks the learner to enter the optimum q or choose whether it is below/equal/above.5. Controls edit p on[.01,.99] and each weight on[.1,20]; numeric values remain visible. An optional endpoint toggle demonstrates p0 or1 as a boundary solution, using limits rather than log0 evaluation. Inverse mode supplies q with the two weights and asks for p. The initial task is unsolved; the result curve and optimum marker appear after recorded prediction.

Plot unnormalized expected weighted loss−wPositive·p·logq−wNegative·(1−p)·log(1−q) against q. Use a numerical plotting grid strictly inside(0,1) and annotate divergent ends; do not clip q then claim the clipped value is the mathematical optimum. The formula supplies exact q*=wPositive·p/[wPositive·p+wNegative·(1−p)]. The default optimum is.5, with original probability.1. Show two separate nodes labeled “original population p” and “weighted-loss optimum q*,” linked by that transformation, alongside the curve. The inverse node returns p=wNegative·q/[wPositive·(1−q)+wNegative·q].

**Contrast:** at the same p.1, change weights from1:1 to9:1. The optimum moves.1→.5. **Null:** change9:1 to18:2. Optimum remains.5 while unnormalized expected-loss values double. Use the same y-axis scale or directly labeled loss values for that comparison so autoscaling cannot hide the height change. The manuscript distinguishes this unnormalized population loss from the normalized finite-sample objective in F3; label each chart accordingly. **Changed exercise:** p.2,wPositive4,wNegative1 also givesq*.5 but inversep.2.

The feedback derives the result from the two weighted probability masses, not an assertion that a classifier has been calibrated. A separate optional action marker can compare q*.5 with the equivalent p cutoffwNegative/(wPositive+wNegative), after explaining cost roles. The core investigation is score meaning, not a second generic threshold lab.

**Recorded tuple:** forward/inverse mode,p orq,weights,question,prediction,revision. Validate finite positive weights; zero/negative weights are outside the declared strict-minimum task and should receive a field explanation rather than silently switching loss behavior. Work is a few hundred formula evaluations, no native optimizer. Numerical/text optimum is authoritative when curve resolution differs.

**Phase-two checks:** forward/inverse round trip; equal weights; both common factors; endpoints if supported; analytic derivative0 at interior optimum; positive second derivative; plotted declared loss at selected q; inverse mode cannot show the solution in an initial mapping label; narrow-screen readable distinct p/q labels.

## I4 — Draw a synthetic point and inspect the neighborhood

**Location:** §5 immediately after the interpolation formula, before categorical/multi-label limitations. **Mechanism:** ordinary SMOTE interpolates along a chosen within-class neighbor segment; the point's assigned label is a modeling assumption.

Default minority IDs A(0,0), B(2,0), C(0,2.5); majority M(1,0), N(2,2). AnchorA, k1, neighbor rank1, interpolation fraction.5. Fixed feature-scale divisors sx=sy=1 define distance sqrt((dx/sx)^2+(dy/sy)^2); default nearest minority isB, so generatedG=(1,0), exactly on majorityM. Plot both identities visibly at that location using concentric distinct shapes and a leader label, not a hidden overplotted dot.

Controls allow editing both coordinates of every original point, adding/deleting labeled original points within total12/minority≥2, choosing the anchor, k from1 to m−1, choosing among its computed k neighbors, and changing fractionu∈[0,1]. Provide table edits and keyboard controls as full alternatives to dragging. The learner predicts G's coordinates and whether moving a chosen majority point will alter this construction. The initial view may show the raw points and chosen input segment endpoints, but G and the answer annotation remain hidden until reveal. Changing coordinates affects the actual numeric array; no cosmetic dragging.

At Apply, show the complete minority distance row excluding self, selected k-neighbor IDs, chosen endpoint, and arithmeticG=A+u(B−A). Draw the whole segment and marker, plus the majority points for context. Both coordinates use the same scalaru. The fixed default gives[1,0]; u.25 gives[.5,0], and u0/1 gives exact endpoints. A point does not become an observed protein because it lies on this segment.

**Required exact null:** in raw fixed-metric mode, move majorityM from[1,0] to[1,1.5], keeping minority points, scale divisors, anchor, chosen-neighbor policy andu fixed. The generated point remains[1,0]; its visible overlap changes. This directly tests that ordinary SMOTE does not use majority neighbors in its interpolation. Do not secretly refit a scaler using all edited points in this mode, because that would change the promised fixed metric.

**Required geometric contrast:** restore default points, then change the declared y-scale divisor from1 to10. A–C distance becomes.25 versus A–B distance2, so nearest neighbor becomesC and G=[0,1.25] at u.5. If the former chosen neighbor is no longer in the active k set, reset neighbor choice to its first ranked member and require a new prediction. Explain that the real Yeast example estimates its scaler on original fitting rows; this small metric control holds scale values explicitly rather than pretending it is a fresh fitted scaler.

Ties use distance then stable ID order and disclose the tied IDs. Show invalid k locally if a deletion reduces available neighbors; do not create a neighbor from another class. Identical minority coordinates are allowed as distinct observations; their zero-length segment generates that same location, with clear overlapping labels. Do not imply that k neighbors must be geometrically distinct. The input bounds[−5,5] and positive scale[.1,10] keep work/labels manageable. A class switch can remove the anchor; choose a visible valid anchor, mark inputs changed and clear prediction/result.

A static adjacent categorical mini-panel demonstrates why interpolating a one-hot vector can create[.5,.5] instead of a category. A separate label-vector panel uses the manuscript's feature0/2 observations with labels(A1,B0)/(A1,B1); a midpoint has no given B rule. These are explanations of semantic limits, not controls that assign unsupported synthetic labels automatically.

**Recorded tuple:** all IDs/classes/coordinates,scale divisors,anchor,k,neighbor ID and rank,tie rule,u,predicted coordinates/majority-effect judgment and revision. Values in original feature units; metric table separately labels scaled distance. Scatter axes have equal physical scale for default geometry; scale-metric changes affect neighbor calculation and annotations rather than secretly stretching one axis. On mobile the table and formula remain primary and the plot stacks below.

**Phase-two checks:** default collision, endpoint and fraction cases, majority-move null, scaling neighbor-switch contrast, changed practice[1,2]→[5,4]u.25=[2,2.5], stable ties, k invalidation, zero-length segment, simultaneous vector interpolation, text/keyboard parity. No claim that the browser has trained the actual five classifiers.

## F4 — The sampler belongs on the fitting branch

**Location:** §6 before the real program. A role diagram follows raw records into a fitting branch and a separate assessment branch. The fitting branch learns scaler parameters, transforms fitting records, performs resampling and fits the classifier. The assessment branch uses that saved scaler and fitted classifier; no sampler arrow points into it. A separate tuning role chooses an action threshold on scores/labels, followed by a locked inspection role. Reserved records have no arrow into prediction in the saved study.

For a CV inset, draw the same structure inside one fold with arrows carrying fit/validation IDs. A common input-stage box states group/time/identity constraints before partitioning. Do not present `imblearn.Pipeline` as the only mathematically valid way to express this workflow: the full manuscript supplies a correct manual implementation. Nor should the illustration claim that an ordinary sklearn pipeline silently ignores a sampler.

Use original roles600 fitting,200 tuning,200 inspection,462 reserve after exact-ID deduplication. Show preprocessing fitted on original600 rows; resampling affects the training feature/label matrix only. The class-positive counts are21/7/7/16. The optional threefold API demonstration uses the fitting600 only and is a separate protocol, not the source of the table. Accessible text enumerates these roles and arrows. Verify all saved ID sets are disjoint and cover the1,462 retained unique proteins.

## F5 — Observed methods answer different questions

**Location:** §7 adjacent to the outcome table. **Source:** unchanged `yeast.data` plus corrected `calculated-inputs.json`. Source IDs are zero-based row indices in the original1,484-row file, with actual protein identifiers available there. They are not positions in a fresh shuffled array. Display counts from the final corrected experiment only.

First panel uses paired four-cell summaries per procedure for threshold.5 and that procedure's separately selected tuning threshold. Exact values:

| Method | At.5: TP/FP/FN/TN | At selected threshold | Selected cost FP+12FN |
| --- | --- | --- | ---: |
| Original |0/1/7/192|2/12/5/181|72|
| Balanced weights |4/35/3/158|4/15/3/178|51|
| Random oversampling |4/34/3/159|4/17/3/176|53|
| Random undersampling |5/41/2/152|4/19/3/174|55|
| SMOTE |4/29/3/164|4/13/3/180|49|

All inspection rows total200 with7 positives. The always-negative baseline has cost84 and accuracy.965; selected-empty precision is undefined. Keep cost axis in hypothetical error-cost units, not percent. Directly label false negatives because each costs12 units under this declared task. Paired cost bars start at zero; exact counts remain available and differences are perceptible.

Second panel shows AP and top10 results separately, with different units. AP values in the JSON are .173459289,.273316553,.278678657,.193549812,.272614782 in the above order; top10 positive counts2,1,2,2,1. Draw actual top-ten ID cards in saved rank order and reveal their labels, with no label-based reordering of tied scores. Do not put AP and cost on one “best score” axis. Brier values may appear in the accompanying probability-quality table but should not be renamed calibration error.

The narrative identifies SMOTE's lowest realized cost, random oversampling's highest AP, and the three methods sharing the best top-ten count. The visual must preserve this inconvenient disagreement. No error bars or smooth PR curves may be invented from five summaries. If showing PR, derive grouped exact coordinates from the saved200 per-record scores and actual labels.

**Phase-two checks:** reconstruct every confusion table, AP, ROC-AUC and top-ten source-ID list from saved scores/labels; check selected thresholds match tuning records and descending-candidate tie policy; reserve remains unscored; include duplicate-removal provenance. Numerical or rendering bugs justify affected checks, not a new model-selection campaign chosen to improve the results.

## I5 — Change an actual tuning policy and inspect the selected records

**Location:** §7 after F5. **Mechanism:** thresholds act on saved actual predictions; choosing a threshold is part of development, not an opportunity to optimize the final inspection outcome.

Input dataset is exactly the saved200 tuning source IDs, protein IDs, labels and the five methods' tuning scores. Default method original, hypothetical costsFP1/FN12, learner-entered trial threshold.5. Task: predict whether lowering the trial threshold to.15 will increase, decrease or leave unchanged the number of detected positives and total declared cost on these tuning records. The actual checked oracle is threshold.5→TP1/FP0/FN6/TN193,cost72; threshold.15→TP5/FP4/FN2/TN189,cost28. More positives are detected and realized cost falls, while precision falls from1 to5/9. `tuning_investigation_fixture` retains these values. The active record table shows inputs and may hide truth until reveal to support a genuine prediction. A learner can change the trial threshold, method and costs before answering. These are meaningful actions on actual observations, not relabeled canned outputs.

Upon Apply, show the selected protein IDs, exact counts, false-alarm and missed-positive cost contributions and total. A scrollable score queue marks all records, with a compact summary and expanded accessible table. Show score≥threshold; the above-all-scores candidate is a named no-alert policy. Sweep each distinct score group plus no alerts to calculate the tuning-optimal threshold. For cost ties, choose the higher threshold because candidates are traversed descending. Expose tied optimal candidates in an optional table to explain the selected tie rule.

For the declared costs, saved selected thresholds are original.14500574144149464; balanced.7408725512485014; random-over.7061158880883859; random-under.7725687726079359; SMOTE.7592716750213055. Use full numbers internally. Never reconstruct decisions from the rounded display threshold. These are tuned *score* cutoffs, not posterior-derived1/13 unless a separate probability argument establishes that interpretation.

**Real contrast:** allow method original→SMOTE with costs fixed and recalculate selected record IDs, ranking and tuning threshold. Their saved rankings/scores differ because they were fitted differently; do not imply that moving a single threshold changed the ranking. Another contrast changes the learner's hypothetical FN cost and finds the best candidate anew. A selected threshold can remain unchanged over a range of costs because the candidate set is discrete; that informative null must be retained rather than forcing a moving marker.

**Exact null:** at any method/trial threshold, multiply both costs by2. Selected IDs and counts stay unchanged, total cost doubles. When optimizing over all candidates, the same minimizing set and higher-threshold tie result remain. **Ranking null:** move only the threshold; the saved score order and AP stay fixed, while selected records/cost can change. The task should ask about that distinction before reveal at least once.

For a learner wanting another scenario, permit selecting an explicit subset of these tuning records by ID and/or changing an individual hypothetical truth label in a **separate constructed what-if copy**. Mark that copy as edited, show all changes and invalidate predictions. Never overwrite observed labels or apply that hypothetical copy to the observed-study table. This optional mode adds transfer but is not needed to claim the core investigation edits meaningful inputs: threshold, costs and choice among actual fitted score sets already change the substantive decision problem.

Keep the original200 inspection results in a separate locked panel tied to the predeclared cost pair and saved thresholds. If an optional inspection-exploration mode is implemented, activating it changes its status to exploratory and requires a fresh prediction; it must not continue to claim a newly tuned outcome is untouched assessment. Do not compute or display predictions for the462 reserve. No browser controls refit native classifiers or choose new model hyperparameters.

**Recorded tuple:** data revision,method,role=tuning,score source hash,trial threshold,costs,tie policy,optional explicit what-if changes,prediction and revision. The200-row threshold sweep is bounded O(n log n) with cumulative counts after sorting; use cached method order and recompute costs/counts without large training work. Keyboard controls and a record-ID search make the queue usable; search/filtering its display must not alter the analyzed set unless a separate explicit subset action is chosen.

**Phase-two checks:** all five saved threshold oracles; no-alert and all-alert; equal scores; zero costs and all-cost ties; scalar-cost null; fixed ranking/AP under threshold changes; exact original versus SMOTE source-ID contrasts; changed FN cost can cause either a different or unchanged optimum; current versus historical prediction separation; role labels; no reserved prediction. The default.5→.15 tuning question must be computed from saved labels before final UI acceptance and its actual outcome explained without forcing a preferred direction.

## F6 — Many easy losses can outweigh a few difficult losses

**Location:** optional §8 focal-loss branch. Exact construction:10,000 easy examples with pTrue.9, ten difficult with pTrue.2, class factor1, gamma2. Cross-entropy totals1,053.605157/16.094379; focal totals10.536052/10.300403. `constructed.focal_loss_mass` retains unrounded values. Show count × per-example loss = total in a numerical table. Use a panel with CE totals and a separate panel with focal totals, explicitly labeling different vertical ranges if needed; alternatively use a clearly labeled log axis with positive values and exact totals.

A schematic count-times-loss rectangle can explain the product, but if widths/heights cannot simultaneously fit perceptibly, use separate count and per-example-loss rulers rather than fake proportional areas. No gradient axis appears: the derivative has an additional product-rule term. Accompany the chart with that derivative explanation from the manuscript. Gamma0 returns weighted CE; no extra lab is required for this bounded conceptual branch.

## Phase-two continuation

1. Reopen the content checkpoint through the inventory finish command and preserve exact source roles. Implement topic-owned lazy-loaded visuals/models/data and accessible controls; generated publication/navigation changes belong to that authorized phase.
2. Execute the complete displayed core programs and optional API program in a suitable documented environment. The latter was not run during writing. Compare core outputs with the fixed corrected evidence; differing native defaults require explanation, not replacement of the declared protocol.
3. Verify each named calculation and interaction, including the real-record contrasts and exact nulls. Run independent correctness and learning-experience review, then browser/accessibility/mobile/performance checks relevant to the implementation. Check that first-pass learning works without opening advanced branches or solutions.
4. Reconcile any finding with its affected text/spec/model. Do not rerun unaffected model fits or broaden into an unrequested search. Keep reserve unscored, record final source hashes and evidence in the central ledger through its owner, and clear only disposable files after preserving pending inputs.

Content-phase completion means the manuscript and these contracts are ready for implementation; every UI and formal review item above remains pending.
