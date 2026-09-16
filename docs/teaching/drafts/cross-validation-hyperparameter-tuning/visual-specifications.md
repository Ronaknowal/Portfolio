# Cross-validation and tuning: visual specifications

Content-only packet, 12 September 2026. Implement these figures and investigations after reading the full manuscript. No current browser rendering, runtime integration or formal phase-two validation is claimed.

## Interaction contract

Each investigation starts with an unset prediction, meaningful editable inputs, and an Apply action. Bind the committed prediction to all relevant input values and the inspected output; any edit invalidates it. Keep a previous trial only as a labeled comparison. Outcome feedback appears after Apply and explains the specific mechanism, not merely success/failure. Reset restores the baseline entities and clears the prediction. Explanatory figures can reveal their worked examples before the learner investigates changed inputs.

Use four distinct forms: fold-assignment/prediction strips; candidate-pattern comparison and exact enumeration; nested data-flow with computed tiny neighbor fits; resource-allocation trajectories. Tables are valuable where they expose actual quantities, but are not the sole visual form. Keyboard editing must match every drag interaction. Do not encode training, validation, test, excluded, correct or incorrect solely in color. Show row IDs and labels, signed/directional connectors and a persistent legend. At mobile widths, stack the stages, expose an inspected row's neighborhood in a readable card, and keep complete tables accessible without shrinking text.

All calculations are local and bounded. No on-load full model-training campaign, downloaded dataset or external service. Load this topic's compact CSV/results only when required, keep heavy modules out of other lessons, and compute changed inputs on Apply. Distinguish observed real data, constructed learning environments, and analytic probability values. Never turn descriptive fold spread into a confidence interval or invent an accuracy heatmap.

## F1 — The fitting loop and selection loop (§1)

Draw a setting card and training-row bundle entering a fitting box. Its output is a fitted-model card. Validation inputs enter prediction, then their answers produce a validation score that returns to the setting-selection loop. A protected assessment bundle is outside both loops and only receives the selected/refitted model. Label what is learned at each loop: model parameters versus settings. The target arrow enters training fit where appropriate, but assessment answers do not enter the selection loop. The text equivalent lists the same roles and allowed influences. Avoid drawing a universal fixed 60/20/20 split; proportions are not the principle.

## I1 — Build held-out predictions (§2)

Start with seven editable constructed points x=[0,1,2,3,4,5,6], y=[0,0,0,1,1,1,1]. IDs are fixed0…6. Initial folds are [0,1,2], [3,4], [5,6]. Learner can edit x values, binary labels and fold assignments via named row controls; require every row in exactly one of three nonempty validation folds. Display a role matrix with rows as source IDs and columns as fits. For the currently inspected fold, show training points on a labeled one-dimensional x line and held-out points with a distinct shape. Clicking a held-out row highlights the actual eligible nearest training point.

Algorithm: for each held-out row calculate absolute difference to every eligible training x; nearest distance wins, with smallest source ID resolving ties. Predict the training neighbor's binary label. No state from one fit is carried to another. Record the predicted label for the inspected row before Apply; then reveal its neighbor, distances, label and correctness. Retain a complete held-out-prediction table and two summaries: unweighted fold mean and pooled row accuracy. Display assessed counts so the weighting difference remains interpretable.

Checked baseline: predictions fold1=[1,1,1], fold2=[0,1], fold3=[1,1]; correct0/3,1/2,2/2; mean.5 versus pooled3/7. Initial consecutive order is deliberately constructed for mechanics, not endorsed as a representative split. Contrast: change training row3's label1→0 while inspecting held-out row0 in fold1; its prediction becomes0. Null: change row0's own label0→1; its prediction stays1 although correctness changes. For an x-only null, translate all x values by the same constant with folds/labels unchanged. The contribution table should demonstrate why nearest identities stay unchanged.

Phase-two checks: arbitrary edited assignments use the correct complements, no omitted remainder, tie behavior matches source-ID rule, held-out target never trains its own fit, summaries use exact counts, prediction invalidation works for fold changes, full comparison usable by keyboard. Do not insert rounded score values inconsistent with possible integer correct counts.

## F2 — Group and time are questions about use (§3)

Use a constructed record grid: patient A at days1,2,3 and patient B at days1,2,3. First panel trains on all A rows and assesses all B rows, labeled unseen-person example. Second trains on both patients' days1–2 and assesses their day3, labeled later known-person example. State these are role illustrations, not measured performance. A third inset gives a seven-day target horizon: feature row day9 has an outcome known only at day16, so it is unavailable for a fit made at day10. Show feature date, label-available date and prediction cutoff separately. A gap is a time/availability consequence, not empty decorative spacing.

For OOF contract, show forward splits train0–3/test4–5 and train0–5/test6–7. Rows0–3 have an explicit unpredicted mark in the output table, not zero. Compare with a K-fold partition covering every row exactly once. This figure explains why score evaluation and one-prediction-per-row API contracts differ.

## I2 — Win the familiar cases (§4)

Constructed environment: four validation labels, independent fair bits under the declared generating model. Initial editable labels [1,1,1,0]. Initial candidate rules have four-entry prediction patterns [0,0,0,0] and [1,1,1,1]. Learner can add, remove or edit a pattern up to sixteen candidates; at least one remains. Pattern entries are predictions at four fixed feature positions. This is not a trained real model and no outcome is falsely attributed to one.

Record predicted best validation accuracy before Apply. Compute each candidate's exact correct count/4 and choose first-listed candidate on ties. Display matching positions, winner and best validation result. Alongside, show future expected accuracy.5: every future target is a fresh fair bit independent of the selected prediction. Explain this generative assumption before the result so the number is not a mysterious hardcoded benchmark.

The exact enumeration view lists all sixteen equally likely validation-label patterns and the selected best score for the current candidate set. Compute their mean; no Monte Carlo noise or animated random draw is needed. Initial two constant candidates yield mean.6875 and future.5. All sixteen distinct prediction patterns yield mean1 and future.5. Duplicate-candidate null leaves best score/enumerated mean unchanged. Changed fixture: labels[0,1,0,1], constants alone score.5; adding matching candidate raises score1 while future stays.5. Probability mass per label pattern is1/16, and table counts must sum16.

Mobile: rows of four labeled bits are compact enough to remain visible; candidate list can scroll vertically. Use buttons with accessible names such as “candidate C, case2, prediction1” instead of color-only tiles. Feedback distinguishes improved familiar-case matching from learning signal. No claim that every real search has this amount of optimism or that each candidate is an independent hypothesis test.

## F3 and I3 — Open one nested fold (§§4,6)

F3 first shows an outer protected set and outer training set. Expand only the training set into its inner folds; selected candidate then receives a fresh refit on the entire outer training set, followed by protected prediction. Keep original row IDs throughout. A learner can inspect the real dataset's saved outer splits and all six candidate score rows from calculated-inputs.json.outer. Show observed correct counts and the declared first-candidate tie policy. The real-data results are read-only evidence, not editable score sliders.

I3 uses the separate constructed sixteen-row fixture in calculated-inputs.json.tiny_nested so edits can recompute all small fits. x=[0…15]; y=0 for x<8,1 otherwise. Outer fold0 holds out even IDs and trains odd IDs; outer fold1 reverses these roles. Within each outer training list in increasing ID order, inner validation alternates even/odd list positions; the complement trains. Candidate neighbor counts1 and3 use uniform binary majority. For distance ties, stable sorting by original source ID selects neighbors; count is odd, so majority has no voting tie. Equal mean inner accuracy selects the smaller neighbor count.

Allow editing all sixteen x values and binary labels while retaining source IDs and explicit fold roles. Let learner inspect either outer fold and select the output prediction: chosen neighbor count or a selected protected row's label. Require an unset committed prediction for that output and the applied inputs. Display the two candidates' per-inner-row predictions, fold scores, mean, selected count, outer-training refit and final protected predictions. Each phase is a separate visible step; do not animate held-out labels into fitting. Changed input recomputes actual distances and votes rather than substituting a narrative.

Checked baseline: both outer folds give inner candidate means.875/.875 and select1. Outer fold0 predicts [0,0,0,0,0,1,1,1] on even IDs, correct7/8. Fold1 predicts [0,0,0,0,1,1,1,1] on odd IDs, correct8/8. All indices and states are in JSON. Contrast changing row3 label0→1: outer fold0 inner count1 scores.25/.75→mean.5, count3 scores.5/.75→mean.625; selected count becomes3, protected predictions remain [0,0,0,0,0,1,1,1], correct7/8. Selection can change without improving that particular assessment outcome.

Null: change only row2's label while inspecting outer fold0. Its inner candidate results and protected predictions must stay unchanged, though row2 correctness and therefore outer assessment can change. Another stored contrast changes row7 label0→1: fold0 count1 mean.875, count3 mean.625, selected1; its protected row8 prediction changes0→1 and correct count becomes8. In outer fold1 that same row7 is protected, so the fitting path remains unchanged. The role-relative effect is the learning objective.

Real results above the constructed lab: outer correct114/115,114/115,112/114; pooled340/344, fold mean.9883549453; final all-data selected3/Standard and inner score.9913043478 remains selection evidence. Label familiar-dataset protocol demonstration. No fold-spread CI. Distinguish the real three-by-three/six-candidate experiment from the constructed two-by-two/two-candidate lab.

Implementation can calculate the constructed experiment in plain JavaScript with at most sixteen points and fixed folds; the author-calculations.py tiny_nested function is its numerical reference. Phase-two checks must compare all saved base/changed traces, selected-count null and protected-label invariance, arbitrary edits and source-ID distance ties. The real read-only explorer uses retained numeric evidence and never claims new model training.

## F4 — Grid projection and random coverage (§5)

Show a 3×3 grid with coordinate values .15, .5, .85 beside the nine actual uniform draws stored in calculated-inputs.json.random_coverage_points, generated by NumPy default_rng(5). The first point is approximately (.8050, .8079); use the stored full-precision coordinates. Label these generated search positions, not model measurements. Project onto the horizontal coordinate. The grid has three distinct horizontal values; these random draws have nine. Do not infer a measured score advantage from that picture.

Below, plot the analytic hit function1−(1−p)^T with T as integer trials and disclosed p. Curves forp=.05 and.01 use actual formula values in JSON. Mark60 trials: .9539302 and.4528434. Axis label is probability of entering a region with the stated sampling mass, never “probability within5% of optimal score.” A table provides exact selected points and assumptions of independent draws.

## F5 — Conditional versus expected error (§7)

Show the two expectation diagrams described in the manuscript: fixed fitted model with new test sets versus newly sampled training sets producing different models. Analytic curves use σ²=4 and integer n=2…40: expected training MSE4(1−1/n), expected new MSE4(1+1/n). The irreducible variance line is4. The vertical unit is squared target units. Mark n12 and three-fold training size8 to compare new risks4.3333 and4.5. No fabricated measured dataset or confidence band.

A covariance inset shows the variance sum and off-diagonal covariance contributions. Do not label covariance as a function of training-overlap percentage. For the always-zero classifier with independent fair labels, correctness variance of the average is1/(4n); an explanatory diagram shows no arrow from training rows into predictions. This is the explicit counterexample to the old overlap argument.

## F6 — Expected improvement (§8)

Use incumbent loss.20, A certain loss.18, B equally likely.05/.45. Draw discrete probability masses, not a smooth Gaussian that was never fitted. Shade positive improvement and calculate EI(A)=.02, EI(B)=.075, mean losses.18/.25. The lower-loss direction is labeled. A second calculation in optional practice uses incumbent.30 and outcomes.10/.50; moving only the worse outcome to.90 leaves EI.10 while mean loss changes.30→.50. Text table lists outcomes, probabilities, positive improvement and weighted contributions.

TPE figure can branch the observed candidate history into below-threshold and remaining groups, then connect symbolic density l and g to their ratio. Do not fabricate a fitted density or imply original independent-density choices equal every current Optuna mode. The manuscript names the primary-paper construction; current library details are linked separately.

## I4 — Resource allocation and slow starters (§8)

Constructed candidate trajectories expose allocation risk rather than pretend to benchmark frameworks. Start three candidates at budgets10,30,90: A=[.30,.25,.24], B=[.40,.20,.10], C=[.35,.28,.27]. Lower loss is better. Default factor3 keeps one candidate after the first observed budget. The learner can edit each candidate's three loss values, add up to nine candidates, choose first comparison at10 or30, and choose factor2 or3. Require losses finite and nonnegative; nonmonotone trajectories are allowed. Budget is an explicitly abstract cumulative training resource, not seconds.

Before Apply, record predicted survivor and whether it will match the full-budget best. At a stage, reveal only values actually paid for, rank observed losses, retain max(1,ceil(active_count/factor)) with stable candidate-order tie rule, and advance through remaining budgets. Plot eliminated trajectories only up to their observed point. After the run an optional Hindsight reveal shows their unobserved later values, with clear distinction from information available at the decision. Counter tracks nominal from-scratch work as sum(active_count×stagebudget); optional resumable mode tracks incremental deltas and explicitly states it requires genuine saved-state continuation.

Checked three-candidate baseline starting10 selectsA, whose full loss.24 loses to hiddenB.10. Starting30 selectsB (.20 versusA.25,C.28) and reaches the full-budget winner. Null: alter B's budget90 loss.10→.02 under default starting10; early winnerA remains unchanged although hindsight regret grows. A real decision contrast changes B's first loss.40→.20, making it survive at budget10. A resource-count preset with nine candidates at10/30/90 and factor3 gives stagecounts9/3/1, nominal270 versus full allocation810 and genuine-resume210. These examples remain constructed and editable; no empirical speedup is asserted.

The hidden later values should not be previsible through hover, accessible-label text or a default expanded table before Hindsight. Keyboard users must receive the same information boundary. The plot uses a numeric resource x axis with actual spacing or explicitly categorical stage axes; loss y axis includes all eventually revealed values and zero where practical. Phase two verifies rounding/ceil behavior, late-value null, actual cost accumulation, ties and prediction invalidation.

## Implementation handoff

Implement topic-specific modules with stable descriptive names and route-local imports. Keep all manuscript practices and optional details closed initially. Execute the full displayed tiny/main programs in their declared setup, and separately execute the supplementary Optuna program if phase-two scope includes its native verification; no exact output is promised for that unexecuted extension. Match all displayed measured values and constructed traces to retained evidence. Check mobile/text representations and actual visible contrast magnitudes, complete independent review and only then advance implementation/publication status. Necessary offline data/calculations must remain until their role is safely integrated; no scratch artifacts are required.
