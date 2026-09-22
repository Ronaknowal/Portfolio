> **Current interaction amendment, 21 September 2026:** Read [live-exploration.md](live-exploration.md). Labs now update directly from valid edits and contain no learner prediction feature, including optional predictions. The older prediction/commit/reveal clauses below are historical design records; their numerical, scope, layout and evidence requirements remain applicable where unchanged.

# Forecasting: visual and investigation specifications

Research/write phase, 12 September 2026. These are implementation contracts, not implemented website components. Read the entire lesson and [design](design.md) before phase two. Preserve different forms for different mechanisms; do not put every activity into one generic score box.

## Common contract

Figures are inline at the named explanatory hurdle. Investigations begin with an unset prediction, editable entities and a deliberate Apply/Issue action. Bind the recorded prediction to the complete applied inputs and selected output. Any edit invalidates it; Reset restores initial entities and clears predictions. Keep earlier results only as labeled trials. Do not reveal a future outcome or correct answer through accessibility text, hover or a hidden default table before the specified reveal. After Apply, feedback explains the actual dependency, with counts and dates, rather than just saying correct.

Use semantic date/index labels, text/shape distinctions in addition to color, visible focus, keyboard alternatives to dragging, labeled numeric inputs and readable mobile stacking. No continuous full fitting, animation loop or route-global dataset import. Use bounded local arithmetic; saved fitted-model evidence remains read-only. Topic-owned modules and assets load only with this lesson. Present original observed data, exact constructed calculations, and counterfactual edits with different explicit labels.

## F1 — Issue time is part of a forecast (§1)

Use two parallel fourteen-day calendar lanes. Both target the same Saturday. One forecast originates seven days earlier; the other one day earlier. Shade the respective known region; the arrow length and visible label give horizon7 or1. The target square remains identical. The mathematical annotation is y[target | origin] with both dates stated. Text equivalent: “Same outcome, different issue dates, different available observations.” No performance numbers are attached.

Add a small event/arrival inset: an observation on Wednesday arriving Friday is unavailable to Thursday's forecast. This connects to the previous lesson without duplicating its full versioned-record lab.

## I1 — Follow a seasonal dependency (§2)

Initial history is six numbered values [10,20,10,20,12,22]; fixed constructed future is [12,22,12,22]. History values are editable finite counts0–100; permit changing period1–6 and forecast horizon count1–8, with at most8 future outcome entries. Defaults horizon count4 and period2. Missing future entries mean unscored horizons, not zero. The baseline helper in [forecast-experiments.py](forecast-experiments.py) provides all four rules; history has at least2 values. Require integer period no longer than observed history. Naive/mean/seasonal cannot become negative; drift clips below zero as the stated count-valued rule does.

Display history and future cells on opposite sides of a fixed origin divider. Before Apply ask for a selected horizon's seasonal source ID and numeric prediction. Selected horizon initially3. Applying highlights exactly index T−m+((h−1) mod m) in the history and draws its connector into the forecast cell. Other rules occupy distinguishable rows with their dependency annotations, not four unlabeled same-style sliders.

Default seasonal future [12,22,12,22], mean15.666667 each, naive22 each, drift[24.4,26.8,29.2,31.6]. MAE is0 for seasonal,5 for mean,5 for naive and11 for drift on the four supplied outcomes. Contrast history value5 (one-based)12→18 gives seasonal[18,22,18,22] and its MAE3. The same edit leaves naive/drift unchanged, while mean becomes16.666667. Show this as an input-dependency difference. A future-only edit must leave every forecast unchanged but can change its scored error. Adding a fixed constant to all history/future values leaves absolute errors unchanged unless clipping activates; do not state this null without the clipping qualification.

Longer-horizon view must repeat the last cycle; do not read future outcomes to extend it. Score only horizons with supplied outcomes and show denominator. All inputs and calculations remain inspectable in a text table.

## F2 / I2 — Train only on labels that have arrived (§3)

F2 uses issue day s8, horizon3, reporting delay2 applied to every count, target day11 and label arrival13. Its seven history features cover observed days0–6, whose arrivals are2–8. At current cutoff12 the row's historical features are legitimate, but the label is not yet available. Show the two decisions independently; do not shade unavailable y8 as a feature of the day8 forecast.

I2 offers origin rows6–12. Allow cutoff10–16, horizon1–4 and uniform reporting delay0–3. For this compact lab the feature window is the three latest count days available at each row's issue time: [s−d−2,s−d−1,s−d]. Show their arrival times and the target y[s+h] arriving s+h+d. The day axis0–20 accommodates every allowed value; expand if necessary rather than clip a late-arriving label. Offered origins remain fixed; it is acceptable for an edit to admit none.

Before Apply the learner selects the complete eligible-origin set (checkboxes, initially none/unsubmitted) and optionally predicts the latest available feature day for an inspected row. Grade the exact set satisfying s+h+d≤cutoff. This also implies s<cutoff for the allowed positive h; explain rather than adding a mystery exclusion. No outcome fitting occurs in this activity.

Verified sets: cutoff12,h3,d2→[6,7];12,1,0→[6,7,8,9,10,11];12,3,0→[6,7,8,9];10,3,2→empty. Contrasting delay2→0 admits8/9 in the default case and moves every row's feature snapshot two days forward. Null: at cutoff12,h3,d2, change an offered row's target value without changing dates; eligibility is unchanged. The target-value edit is allowed through a separate bounded count input to demonstrate that admission is an information-time decision, not a favorable-label decision.

Show an optional splitter indexing inset: with first test origin t, last train origin t−gap−1, enforce gap≥h+d−1. Use h3,d2 so minimum gap4, last train t−5. This is only the stated daily-origin convention; the main lab uses direct date inequalities. Do not copy an arbitrary two-day gap into the real experiment.

## F3 — Recursive versus updated forecasts (§3)

Create three aligned traces with last known count22 and rule next=last+2. The fixed-origin recursive lane computes[24,26,28,30] with prediction-fed arrows. The legitimate updated-one-day lane uses new observations[12,22,12,22] at advancing cutoffs and emits[24,14,24,14]. A third lane deliberately retains the old issue label while drawing future-observation arrows; mark those specific arrows invalid for that claim.

Both first-continuation MAEs equal10. Preserve the equality: a protocol error need not improve the score. The changed future[22,24,26,28] leaves the fixed-origin recursive outputs unchanged, but the updated lane becomes[24,24,26,28], with MAE.5 versus2. These exact constructed fixtures are saved in JSON. This is a static two-example explanation; another lab is unnecessary solely to raise the count.

## F4 — Rehearse the refit schedule (§4)

Training/target lane diagram with three issue dates seven days apart, seven horizons per issue. Expanding and90-eligible-row panels distinguish moving origin from moving training start. The actual experiment has different eligible maximum origin t−h for each horizon, so expanding one horizon's row should show that exact boundary and its own scaler fit. A single simple overview can select h1 orh7 without pretending their training matrices are identical.

An origin-step control is an explanatory trace, not a graded investigation. Save the previous issued forecast when advancing; later observed values may enter new fits but cannot alter its recorded number. The final-assessment boundary locks procedure selection, not all future scheduled learning. Include a separate frozen-model sketch to make the distinction explicit.

## F5 — Measured aggregate and horizon errors (§5)

Use [calculated-inputs.json](calculated-inputs.json).development and .final. All252 development predictions per candidate and112 final predictions per assessed method are actual author calculations. Development candidates6; final methods selected ridge_expanding, naive, seasonal. Do not render final predictions for unassessed candidates or infer them from development results.

Two coordinated views: six-candidate development table/short bars, then three-method final horizon chart plus exact table. Show origin Saturday and weekday labels h1Sunday…h7Saturday. For counts/errors use rentals/day units; MAE axis starts0 and extends beyond maximum plotted value, with actual numeric ticks and no smoothing. Method color/style remain consistent across views. Different final/development periods are labeled, not depicted as repetitions of identical conditions.

Final aggregate MAE ridge1167.622809,naive1310.339286,seasonal1390.910714. Final ridge MAE curve[1134.022878,1118.548389,1530.538620,965.263024,931.640455,926.523098,1566.823201]. Seasonal[1686.5625,1698.625,2377.6875,703.6875,890.5625,1016.8125,1362.4375]; naive[1313.6875,1549.1875,1750.125,931.875,1092.4375,1172.625,1362.4375]. Highlight seasonal wins at h4/h5 and exact naive/seasonal equality at h7 with the common source day. No confidence bands or universal significance badges.

## I3 — Issue a real forecast before revealing its outcomes (§5)

Use the retained731-day count series, restricted to development origins indices364…609 in increments7. Initial origin364 (2011-12-31); season period choices1,7,14; seven horizons fixed. Permit editing an observed count among the displayed last14 days and a future outcome, each integer0–15000. Edited series is immediately labeled counterfactual; Reset restores source values. History chart shows date/count, cutoff and the exact period-length block copied. It is a seven-day request even when period14; copy the first seven positions of that last14-day cycle.

Before Issue, record whether seasonal MAE will be lower/equal/higher than naive and the predicted h3 value. Issue freezes input state and draws forecasts while actual future outcomes stay concealed. Reveal outcomes computes errors and grades the recorded prediction. The learner can inspect historical donor dates and per-day absolute-error contributions. A previous revealed trial does not justify keeping an answer prefilled after an input edit.

Checked actual examples from JSON.real_investigation: origin364 period7 forecast[754,1317,1162,2302,2423,2999,2485], observed[2294,1951,2236,2368,3272,4098,4521], seasonalMAE1042.571429 >naive789.571429. Origin371 (2012-01-07) period7 has MAE978 <naive1466.714286. Origin476 period14 gives1702.857143 versus period7 2449.714286 andnaive2117; these are exploratory local comparisons, not new locked-procedure assessment.

Verified null: origin364, edit future row365 by+1000. Its seasonal predictions remain byte-for-byte equal to the original array. Actual day1 becomes3294, so errors change. Contrast: edit observed row358 (first day of the last7-day block)754→1754 at origin364; the first seasonal prediction changes754→1754, all remaining six remain, naive remains2485 throughout. Seasonal MAE drops by1000/7 to899.714286 on the original future. This is a counterfactual arithmetic result, not an observed data correction.

Optional model comparison is read-only saved ridge evidence for the selected origin. Editing counts must not silently reuse its old predictions as if the model were refitted. Either hide that layer in counterfactual mode or label it frozen-original evidence. No browser ridge training is required for the intended investigation.

## Phase-two completion requirements

Implement the complete manuscript and controls with source-owned arithmetic; retain all changed practices/hints/solutions and the optional deeper route. Execute displayed programs verbatim in the declared setup, compare every used fixture including empty eligibility, delay-adjusted historical feature windows, future-edit invariance, horizon7 equality and real period rank reversal. Formal independent review, responsive/keyboard inspection, accessible text alternatives and route-local loading checks belong to phase two. Author calculations are evidence for intended behavior, not a claim these UI checks have passed.
