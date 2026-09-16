# Time-Series Validation & Forecasting Baselines

It is Saturday evening. You know how many bicycles were rented each day so far, and must plan for the next seven days. A spreadsheet downloaded months later contains the weather and rental totals for that entire week. Those columns make prediction easy—after the week has happened.

A forecasting experiment must recreate the harder moment when the answer was still unknown. That requires more than putting a date column into a model. We need to reconstruct the information available when a prediction was made, specify how far ahead it reached, and compare it with a useful alternative at the same moment.

**First pass:** follow sections 1–6 and practices 1–6. You need arrays, a train/validation split, and the idea that a fitted scaler learns from training data. We define forecast notation and error summaries locally. Section 7 develops overlapping targets, uncertainty, changing environments and multi-series applications. Its specialist branches are not additional core entry requirements.

## 1. Put a clock on the prediction

The preceding [problem-formulation lesson](/learn/path/full-curriculum/ml-problem-formulation-baselines-data-leakage?module=classical-ml) separated an event's time from the time we learned about it. Forecasting adds a second separation: **when the forecast is issued** versus **when the predicted outcome occurs**.

Use these terms:

| Term | Meaning in the bicycle example |
| --- | --- |
| Observed value \(y_t\) | Rentals recorded on day \(t\) |
| Forecast origin \(t\) | The time we issue the forecast; in this immediate-reporting example, the end of day \(t\) |
| Horizon \(h\) | Number of days from that origin to the target day |
| Forecast \(\hat y_{t+h\mid t}\) | Prediction of day \(t+h\), made using information available at origin \(t\) |
| Forecast error \(e_{t,h}\) | Actual value minus that forecast: \(y_{t+h}-\hat y_{t+h\mid t}\) |
| Refit schedule | When the learning procedure is allowed to update its fitted parameters |

The vertical bar means “given information through this origin.” It does not mean division. At the end of Saturday, Sunday's count is a horizon-one target; next Saturday's count is a horizon-seven target.

Two requests can target the same date with different difficulty. A prediction for next Saturday made today and a prediction for that Saturday made on Friday have different information. Store both origin and target date; grouping only by target date loses the distinction.

**Figure 1 — One target, two information boundaries.** A calendar lane marks a Saturday target. Draw one arrow from the previous Saturday and a shorter arrow from Friday. Shade only information available at each arrow's starting point. The target itself is the same square.

Our decision contract is: predict each of the next seven daily rental totals every Saturday after that day's count is available. Give each forecast equal weight in the initial comparison. Calendar dates are known; future observed weather and rental outcomes are not. A later example assumes counts arrive at the end of their recorded day because the source does not supply actual reporting timestamps. A deployment using delayed counts would need a different availability rule.

Forecasting is one kind of temporal prediction. Estimating today's outcome from measurements already available today is a **nowcasting** or contemporaneous prediction problem, depending on the setting. Filling a missing historical value after observing later dates is a retrospective reconstruction problem. These can be legitimate tasks, but their scores do not answer the Saturday forecasting question.

## 2. Build alternatives that a complicated model must beat

A **baseline** is a complete prediction rule that gives the model a meaningful comparison. “Predict zero” is computationally cheap, but often answers little about whether learning was useful. A recent value or the corresponding day of the previous week is more informative.

Start with a constructed six-day history:

\[
10,\;20,\;10,\;20,\;12,\;22.
\]

Imagine a two-day operating cycle, with quieter and busier days. We need four future predictions from this one origin. Four rules summarize different beliefs:

- **Historical mean:** repeat the average of the available history. This treats the overall level as useful and ignores order.
- **Naive:** repeat the most recent observed value. This treats the current level as the best simple guide.
- **Seasonal naive:** repeat the last observed cycle. “Seasonal” can mean a weekday cycle or a machine's repeating operating schedule; it need not mean summer and winter.
- **Drift:** continue the average change from the first to the last observation. This is a deliberately simple trend rule.

| Rule | Four forecasts from this origin | What it carries forward |
| --- | --- | --- |
| Mean | 15.667, 15.667, 15.667, 15.667 | Average of all six values |
| Naive | 22, 22, 22, 22 | Last level |
| Seasonal, period 2 | 12, 22, 12, 22 | Last two-day pattern |
| Drift | 24.4, 26.8, 29.2, 31.6 | Last level plus 2.4 per day |

The drift slope is \((22-10)/(6-1)=2.4\): six observations span five intervals. Its horizon-\(h\) prediction is \(22+2.4h\), not the mean of the observations.

For a history \(y_1,\ldots,y_T\), write the mean, naive and drift rules as

\[
\hat y^{\text{mean}}_{T+h\mid T}=\frac1T\sum_{i=1}^T y_i,\qquad
\hat y^{\text{naive}}_{T+h\mid T}=y_T,\qquad
\hat y^{\text{drift}}_{T+h\mid T}=y_T+h\frac{y_T-y_1}{T-1}.
\]

For season length \(m\), cycle through the final \(m\) observed values. In zero-based array notation, the selected historical index is \(T-m+((h-1)\bmod m)\), where \(T\) is the number of history entries. This works even when the requested horizon exceeds one cycle; the rule repeats **observed history**, not future answers. The [FPP3 baseline chapter](https://otexts.com/fpp3/simple-methods.html) gives further examples and the equivalent mathematical indexing.

Suppose the four future outcomes turn out to be \(12,22,12,22\). Mean absolute error (MAE) averages the sizes of the misses, ignoring their signs. Mean and naive each have MAE 5, seasonal naive has 0, and drift has 11. Seasonal naive wins this constructed continuation because its assumed cycle matches it. If the continuation changes, the ranking may change.

**Investigation 1 — Which observation feeds this forecast?** Edit the six historical values and choose a period. Before applying, predict the horizon-three seasonal forecast and which source day supplies it. Then inspect the linked history and forecast cells. Change the fifth value from 12 to 18: the period-two forecast becomes \(18,22,18,22\). The naive and drift forecasts remain unchanged because their required endpoint values did not change. Compare different rules on the same revealed future; do not award a rule a universal winner badge.

This small mechanism has a practical application beyond transport. A server with a weekly maintenance cycle can have a better baseline from the same weekday than from yesterday. Conversely, after a lasting level change, a long historical mean may adapt too slowly. A baseline is a hypothesis about what persists.

## 3. Turn history into features without importing the answer

### Name the row by its origin

A lag is an earlier value used as a feature. For a training row whose origin is \(s\), useful features might be \(y_s\), \(y_{s-1}\), \(y_{s-6}\), and the mean of \(y_{s-6},\ldots,y_s\). Its horizon-three label is \(y_{s+3}\).

That row can enter a fit at current origin \(t\) only when its features **and its label** are available by \(t\). If counts arrive immediately at day end, the label condition is \(s+3\le t\). With a two-day reporting delay it becomes \(s+3+2\le t\). The historical feature snapshot must also change: a forecast issued at day \(s\) can then use counts only through day \(s-2\), not \(y_s\). Reconstruct the inputs as they were available at that row's own issue time. A later-completed CSV must not backfill future knowledge into historical feature snapshots.

**Figure 2 — A row has a past window and a future label.** With a two-day delay, show origin \(s=8\), available feature days 0–6, target day 11, and label arrival day 13. With current cutoff 12, this is a complete-looking row in a later CSV but an ineligible training row in the historical fit.

At cutoff 12, horizon 3 and delay 2, eligible origins from the offered list \(6,\ldots,12\) are only 6 and 7. At zero delay, origins 6 through 9 are eligible. At cutoff 10 with the original horizon and delay, none are eligible.

This derives the omitted recent rows from an actual information constraint. A gap is not a magic constant copied from a library example. In an equally spaced, origin-indexed array, a splitter that ends training at \(t-g-1\) before test origin \(t\) needs \(g\ge h+d-1\) to enforce \(s+h+d\le t\). The minus one comes from the splitter's indexing convention. If predictions are issued before the day's count arrives, use that earlier cutoff instead. With irregular dates or different label durations, evaluate the timestamp inequalities directly.

**Investigation 2 — Admit a training row.** Move the cutoff, change the horizon or reporting delay, and predict the complete set of eligible training origins before applying. Reveal each row's feature interval, target date and arrival date. An empty eligible set is a meaningful result; the interface must not secretly train on unavailable rows to produce a number.

### Rolling statistics must face backward from the actual origin

If the row predicts tomorrow from today's end, today's count is a valid feature. If the row is indexed by tomorrow's target date instead, the same feature is a one-day shift. Many off-by-one mistakes come from switching these conventions mid-program.

For target-indexed daily prediction, the historical seven-day mean is conceptually

~~~python
past_week_mean = counts.shift(1).rolling(7).mean()
~~~

For origin-indexed prediction, the seven-day window may include the origin's count. Neither formula is universally correct without naming the prediction time. A centered rolling mean uses observations on both sides and is unsuitable for an online forecast unless those later values were genuinely available under the stated task.

Fit medians, scalers, dimensionality reduction and feature selectors inside the eligible training set. Deterministic calendar features can be computed in advance because next Tuesday's weekday is already known. A weather **forecast issued before the origin** could also be valid, but the weather later measured on Tuesday is a different feature. Use the archived forecast version, not a retrospectively corrected observation.

Missing dates need an explicit policy. In a daily series, “previous row” means “previous day” only when the calendar is complete. Reindex to the intended calendar, distinguish a missing report from zero events, and choose a causal fill or a model that handles missingness. Interpolating across a future observation may be useful for historical visualization while leaking information into a forecast experiment.

### Seven-day forecasts are not seven updated one-day forecasts

Consider the deliberately simple one-step rule “last count plus 2.” From the final historical count 22, a **recursive** four-day forecast is \(24,26,28,30\): each prediction supplies the next input.

If the future outcomes are \(12,22,12,22\), using each newly observed outcome before predicting the next produces \(24,14,24,14\). That is a different procedure. It is legitimate if we issue and update a one-day forecast each evening. It is invalid evidence for four forecasts that supposedly were issued together from the original origin.

Both sequences happen to have MAE 10 on that continuation. Leakage does not have to produce a better score to be a protocol error. For the changed continuation \(22,24,26,28\), the original recursive forecast has MAE 2, while the updated sequence \(24,24,26,28\) has MAE 0.5. The attractive number still belongs to the updated task.

**Figure 3 — Freeze the origin or advance it.** The left chain feeds predictions forward behind a fixed cutoff. The right chain receives a newly observed outcome at each advancing cutoff. A third, explicitly invalid panel copies the right-hand inputs while falsely keeping the original issue date. Trace the information arrow that makes the claim wrong.

## 4. Rehearse the deployment schedule

A single chronological holdout tells us about one later period. **Rolling-origin evaluation** repeats the forecasting task from several historical issue dates. For each origin:

1. Reconstruct eligible history.
2. Fit the entire allowed procedure, if the declared schedule calls for refitting.
3. Issue all requested horizons without seeing their future outcomes.
4. Save origin, target date, prediction and model/procedure identifier.
5. Score after the outcomes become available.

An **expanding window** retains all eligible history. A **sliding window** retains only a recent eligible interval or number of rows. Expanding history supplies more examples and longer cycles; a sliding window can adapt faster after change but discards information and can increase estimation noise. Choose the window using development forecasts, not a story invented after the final errors are revealed.

**Figure 4 — A staircase of genuine rehearsals.** Show three origins, each with its training region and seven future target cells. In expanding mode the left training edge stays fixed; in sliding mode it moves. Fit boundaries, target horizons and newly available observations are distinct marks. Do not draw one large static fit feeding all origins if the code actually refits.

Choosing the model family, window, features or regularization strength is still model selection. A useful arrangement is an earlier set of rolling origins for development, then a later sequence that evaluates the locked procedure. If an extensive search needs its own assessment, reproduce the selection inside earlier information boundaries, just as nested cross-validation protected outer rows in the earlier lesson.

A final **rolling** assessment can allow outcomes from its early weeks to enter scheduled fits for later weeks after those outcomes arrive. This does not retroactively change an earlier forecast. What remains locked is the update/selection policy. By contrast, a frozen-model assessment holds one fitted model fixed. Say which one you ran; “test set” by itself does not settle the distinction.

Randomly shuffling rows usually fails to recreate future-from-past deployment: it can train on later regimes and on targets that were unknown at the historical issue date. It is not a theorem that all timestamped data require one particular splitter. The contract specifies the use. The [scikit-learn lagged-feature example](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html) is an alternate Python walkthrough; its next-hour task should not be silently relabeled as a whole test-block forecast issued once.

### Preserve the horizon in the score

For origins \(\mathcal T\) and horizons \(1,\ldots,H\), calculate

\[
\operatorname{MAE}_h=\frac1{|\mathcal T|}\sum_{t\in\mathcal T}|e_{t,h}|,\qquad
\operatorname{RMSE}_h=\sqrt{\frac1{|\mathcal T|}\sum_{t\in\mathcal T}e_{t,h}^2}.
\]

MAE answers average absolute error in the target's units. RMSE gives large errors more influence and returns to the same units after the square root. Averaging MAE over equal-sized horizons equals pooling all their absolute errors. Averaging per-horizon RMSE generally differs from taking one square root after pooling squared errors.

Longer horizons often lose information, but an empirical error curve need not increase at every step. Here every origin is a Saturday, so horizon and weekday are tied together. A difficult Sunday and an easier Thursday can produce a nonmonotone curve. To separate horizon difficulty from weekday effects, design origins that cover different weekdays or analyze both dimensions with enough data.

If tomorrow matters more than next week, predeclare horizon weights and evaluate that decision. Do not switch weights after discovering where your favorite model performs best.

## 5. Forecast a real week of bicycle rentals

The downloadable [daily CSV](bike-sharing-daily.csv) contains 731 complete calendar days from 2011–2012 in the UCI Bike Sharing dataset. It is the unchanged daily file, renamed for clarity, with attribution and CC BY 4.0 provenance supplied in [the data record](data-provenance.md). The target is recorded system-wide rentals, not unmet demand, the number of bicycles required at a station, or a causal effect of weather.

We use counts and deterministic calendar features. The source also contains casual and registered rentals; their sum is exactly the target, so using the target day's values would reveal the answer. We do not use future observed weather. The source lacks report-arrival timestamps, so this replay explicitly assumes counts are known at each day end.

### Declare the comparison before fitting

- Initial history: 1 January–31 December 2011.
- Issue forecasts every Saturday from 31 December 2011 through 22 December 2012, for each of the next seven days.
- Development: first 36 origins, forecasting 1 January–8 September 2012.
- Final rolling assessment: final 16 origins, forecasting 9 September–29 December 2012.
- Omit 30–31 December from assessment because they do not form another complete seven-day block in this file. They are retained in the source.
- Compare mean, naive, seven-day seasonal naive, drift, direct ridge with expanding history, and direct ridge with the most recent 90 eligible training origins.
- Select by development MAE pooled across origins and horizons. Keep the selected procedure and the predeclared naive/seasonal baselines for final assessment; do not use final results to choose among all six candidates.

For each horizon \(h\), direct ridge fits a separate model to eligible origin rows \(s\le t-h\). Its fourteen inputs are three historical counts \(y_s,y_{s-1},y_{s-6}\), the past-seven-day mean, seven target-weekday indicators, sine and cosine of target day-of-year, and elapsed calendar time in years. The cyclic year features use a disclosed 365.25-day approximation; they are calendar encodings, not learned seasonality guarantees.

Every horizon fit learns its scaler from its own eligible training rows and uses ridge alpha 1 with an intercept. Counts are nonnegative, so clipping negative ridge or drift predictions to zero is part of the predeclared rule. The 90-row option means the last 90 eligible origin rows for each horizon, not the last 90 target days regardless of availability.

### Read the measured results

| Candidate | Development MAE, rentals/day | Development RMSE, rentals/day |
| --- | ---: | ---: |
| Historical mean | 2083.70 | 2383.25 |
| Naive | 1198.30 | 1532.45 |
| Seasonal naive, seven days | 987.32 | 1348.10 |
| Drift | 1203.00 | 1538.28 |
| Direct ridge, expanding | **772.33** | 1045.45 |
| Direct ridge, 90 eligible rows | 883.80 | 1169.20 |

The declared selection chooses expanding ridge. Refit that fixed procedure at each later origin using then-available observations. Its final MAE is 1167.62; naive gives 1310.34 and seasonal naive 1390.91. The selected model remains better on this aggregate, but its error increases markedly compared with the earlier development period.

This is evidence about one two-year system and this update policy. The development-to-final difference can reflect changing conditions, different dates, selection and sampling variation; it does not by itself identify one cause. No repeated independent city sample or universally transferable improvement percentage has been measured.

| Final horizon | Expanding ridge MAE | Naive MAE | Seasonal naive MAE |
| --- | ---: | ---: | ---: |
| 1 | **1134.02** | 1313.69 | 1686.56 |
| 2 | **1118.55** | 1549.19 | 1698.63 |
| 3 | **1530.54** | 1750.13 | 2377.69 |
| 4 | 965.26 | 931.88 | **703.69** |
| 5 | 931.64 | 1092.44 | **890.56** |
| 6 | **926.52** | 1172.63 | 1016.81 |
| 7 | 1566.82 | **1362.44** | **1362.44** |

The aggregate winner does not win every horizon. At horizon seven, naive and seven-day seasonal naive are **identical rules**: both use the latest Saturday count. Their equal errors are required by the mechanism, not a coincidental numerical tie. The table also shows why drawing a smoothly increasing “forecast difficulty” curve would misrepresent these measurements.

**Figure 5 — The measured comparison at two resolutions.** Put the aggregate development/final table beside horizon-specific final curves with all seven points and exact weekday labels. Show the shared naive/seasonal horizon-seven point explicitly. Do not turn fold or horizon spread into an unsupported confidence band.

**Investigation 3 — Replay a forecast request.** Pick a development origin and a seasonal period, predict whether its seven-day MAE will be lower than the naive rule, then issue the forecast before revealing actual outcomes. At 31 December 2011, period-seven MAE is 1042.57 versus naive 789.57; at 7 January 2012 it is 978.00 versus naive 1466.71. A plausible weekly rule wins one week and loses another. Inspect the exact historical days it copied rather than guessing a narrative from the score.

The investigation also allows a clearly marked counterfactual edit to an observed historical count or a future outcome. A future edit can change the subsequently measured error, but must not change the already issued forecast. These edited results are constructed explorations on a real series, not the published measured experiment.

### Run the complete experiment

Save [forecast-experiments.py](forecast-experiments.py) beside the CSV and run it with Python. It contains the full feature construction, six-candidate development comparison, locked final replay, exact toy calculations and figure data; there are no omitted training helpers or network downloads. It writes [calculated-inputs.json](calculated-inputs.json), including every origin's actual values and predictions.

~~~text
python -m pip install numpy==2.3.5 pandas==3.0.1 scikit-learn==1.9.1
python forecast-experiments.py
~~~

The author calculation used Python 3.12.14 and these library versions, with serial deterministic SVD ridge fits. It performed 504 small development fits and 112 final fits. The following complete, independently runnable baseline program exposes the essential forecast loop without hiding it behind model code. Save it beside the same CSV.

~~~python
from pathlib import Path
import numpy as np
import pandas as pd

data = pd.read_csv(Path(__file__).with_name("bike-sharing-daily.csv"))
counts = data["cnt"].to_numpy(dtype=float)
origins = np.arange(364, len(counts) - 7, 7)

for stage, stage_origins in [("development", origins[:36]), ("final", origins[36:])]:
    errors = {"naive": [], "seasonal": []}
    for origin in stage_origins:
        history = counts[:origin + 1]
        actual = counts[origin + 1:origin + 8]
        naive = np.repeat(history[-1], 7)
        seasonal = history[-7:]
        errors["naive"].append(actual - naive)
        errors["seasonal"].append(actual - seasonal)
    for name, rows in errors.items():
        error = np.asarray(rows)
        print(stage, name, "MAE", round(np.abs(error).mean(), 2))
        print("by horizon", np.round(np.abs(error).mean(axis=0), 2))
~~~

The pooled values are development naive 1198.30 and seasonal 987.32, then final naive 1310.34 and seasonal 1390.91. The code uses each origin's preceding week, not one repeated prediction from the beginning of the year. Final-week outcomes can enter a later scheduled forecast only after becoming history.

## 6. Practice on changed requests

Try each question before opening its hint or solution.

### 1. A reporting delay changes the training set

Offered training origins are days 4–10. At cutoff day 10, labels predict three days ahead and arrive one day after their target day. Which rows have known labels? What changes at cutoff 12?

<details><summary>Hint</summary>

Test the label arrival inequality for each row, not just whether its feature date precedes the cutoff.
</details>
<details><summary>Solution</summary>

Require \(s+3+1\le10\), so origins 4, 5 and 6 qualify. At cutoff 12, origins 4–8 qualify. This assumes all features for those rows are also available; label eligibility alone does not certify an arbitrary feature pipeline.
</details>

### 2. Repeat the last cycle beyond one season

History is \(3,9,6,4,10,8\), with period 3. Produce five seasonal-naive forecasts. Which historical value supplies horizon five? Change only that value by adding 2.

<details><summary>Hint</summary>

Repeat the final three values in their existing order.
</details>
<details><summary>Solution</summary>

Forecasts are \(4,10,8,4,10\). Horizon five copies the fifth history value, 10. Changing it to 12 produces \(4,12,8,4,12\). It does not change horizon three.
</details>

### 3. Identify the illegal arrow

A model predicts the next three days from Sunday evening. Its Monday prediction uses Sunday observations, its Tuesday prediction uses Monday's actual count, and its Wednesday prediction uses Tuesday's actual count. The report calls all three “Sunday's three-day forecast.” Repair either the procedure or the claim.

<details><summary>Hint</summary>

The same numerical predictions can be legitimate under a different issue schedule.
</details>
<details><summary>Solution</summary>

For a genuine fixed-origin recursive forecast, feed Monday's prediction into the next step, then Tuesday's prediction, with only Sunday-available additional inputs. Or relabel and evaluate the procedure as an updated one-day forecast issued each evening. A direct three-model approach can also predict each horizon from Sunday-available features without recursion.
</details>

### 4. Equal MAE can hide different large errors

Two procedures have four absolute errors: A \(0,0,4,4\), B \(2,2,2,2\). Calculate MAE and RMSE. Which metric exposes A's concentrated misses?

<details><summary>Hint</summary>

Square before averaging for RMSE, then take a square root.
</details>
<details><summary>Solution</summary>

Both have MAE 2. A has RMSE \(\sqrt8\approx2.828\); B has RMSE 2. RMSE gives A's two larger misses more weight. Whether that is the right preference depends on the cost of large misses; neither arithmetic result proves an application policy.
</details>

### 5. Inspect the real protocol

The final table's seven-day MAE favors seasonal naive over expanding ridge. May we now claim that a hybrid chosen from that same final table has independently established final performance? What should happen next?

<details><summary>Hint</summary>

Choosing a different model per horizon is itself selection.
</details>
<details><summary>Solution</summary>

No. The table can motivate a candidate hybrid, but its reported final performance would be adaptively selected on that period. Specify the new per-horizon policy using development evidence, or treat the observed final results as development for a new experiment and obtain fresh later assessment. Keep the original locked-procedure result visible.
</details>

### 6. A time-series CSV is not necessarily a daily calendar

Rows are Monday, Tuesday, Thursday and Friday. A developer labels the preceding row's count “yesterday” for every target. Identify the failure and propose a repair. Should a missing Wednesday be filled with zero?

<details><summary>Hint</summary>

Separate missing observation from observed absence of events.
</details>
<details><summary>Solution</summary>

For Thursday, the previous row is Tuesday, two days earlier. Build the intended daily calendar and distinguish an absent report from a recorded zero. Preserve missingness or use a declared causal fill based on available history; do not interpolate from Thursday while pretending to forecast Wednesday. If the application truly operates on irregular events, use elapsed time and describe the lag as previous event instead.
</details>

### 7. Overlapping labels and pooled horizons — deeper

A row at origin \(s\) predicts the sum of the next three days, reported immediately after day \(s+3\). At cutoff 20, what is the latest eligible training origin? If forecasts are issued daily, do adjacent target sums form independent error samples automatically?

<details><summary>Solution</summary>

The latest eligible origin is 17. The sums from origins 17 and 18 share days 19 and 20, so their targets overlap. Their forecast errors need not be independent; shared inputs, parameter fits and serial dynamics can add dependence too. A gap that prevents unknown labels from entering training does not establish independent assessment errors.
</details>

### 8. A scaled-error denominator — deeper

For training history \(2,4,3,5\), calculate the nonseasonal naive training-error scale. A later forecast has absolute error 1.5. What is its scaled error? What if all training values were 4?

<details><summary>Solution</summary>

The scale is \((|4-2|+|3-4|+|5-3|)/3=5/3\). The scaled error is \(1.5/(5/3)=0.9\). For constant training history the scale is zero, so this ratio is undefined; silently adding a tiny denominator creates a different metric. Name a suitable alternative and retain the original-unit errors.
</details>

**Core readiness:** given a new request, you can name the issue time and horizon, reconstruct legal features and labels, draw the evaluation/refit schedule, compute an informative baseline, and report errors without losing the horizon or changing the selection boundary.

## 7. Deeper connections that change the design

### Direct, recursive and joint multi-step models

Our real experiment uses **direct** prediction: horizon \(h\) has its own fitted map from origin features to \(y_{t+h}\). This avoids feeding forecast errors into later inputs, but requires several fits and uses fewer eligible recent training origins at longer horizons.

A **recursive** method fits a one-step rule and repeatedly feeds its own predictions back. It can share one model across horizons, but its inputs at long horizons differ from the observed-history inputs seen during ordinary training. Approximation errors can feed forward. This is a mechanism, not a theorem that recursive prediction must lose to direct prediction on every dataset.

A **joint multi-output** model predicts a vector of future values. It can share information across horizons and can support path-level objectives. Labels for a complete length-\(H\) vector are available only when the whole vector has matured, unless training explicitly handles partial labels. This is another reason not to apply one origin cutoff formula blindly to every architecture.

The same distinctions matter in a controller planning battery use for the next hour. A forecast that will be revised every minute serves a different decision than a schedule that must be committed for the full hour. Evaluate the forecast-and-update policy the controller will actually use.

### Overlap is not one problem with one remedy

Three mechanisms can coexist:

1. **Unknown labels entering training:** enforce each label's actual maturity/arrival cutoff.
2. **Shared information between prediction tasks:** decide whether that reuse is legitimate for the intended known-system or unseen-system deployment.
3. **Dependent assessment errors:** account for serial dependence, overlapping target intervals and shared fits when making uncertainty claims.

If every day issues forecasts for horizons 1–7, a particular outcome appears as several distinct forecast tasks. Averaging those errors evaluates an origin–horizon distribution. It does not produce seven new independent outcomes. A joint weekly staffing decision may instead care about total-week error or the probability that capacity is exceeded on any day.

Our weekly nonoverlapping target blocks avoid repeatedly scoring the same target day, but adjacent weeks can still be dependent. A standard error calculated as sample standard deviation divided by the square root of 112 treats those errors as independent; that assumption is not established here. Dependence-aware inference may use a justified block-resampling or time-series variance model, with block length and stationarity conditions examined. More bootstrap replicates cannot repair a wrong independence model.

### From point forecasts to calibrated uncertainty

A point forecast is a single summary. An interval or distribution adds a claim about possible outcomes. A band containing roughly 90% of individual days is not automatically a band containing an entire seven-day path 90% of the time. For seven independent events each covered with probability 0.9, simultaneous coverage would be \(0.9^7\approx0.4783\); real dependence changes this calculation.

Residual checks can reveal patterns the model left behind: persistent positive errors suggest systematic underprediction on those assessed cases; repeating weekday errors suggest a missing cycle. White-looking residuals do not prove future accuracy or correct interval coverage. Residuals from the fitted history and genuine horizon-\(h\) forecast errors answer different questions.

Exchangeability-based conformal guarantees do not become time-series guarantees by changing the x-axis to dates. The earlier Calibration & Conformal Prediction topic explains its assumptions; temporal adaptation needs a stated method and conditions. The [FPP3 distributional accuracy section](https://otexts.com/fpp3/distaccuracy.html) is a deeper route to evaluating probabilistic forecasts, not evidence that the point-only bicycle experiment produced calibrated intervals.

### Scaling, transformations and changing environments

MAE in rental counts cannot directly compare a small station with a whole city on equal footing. Mean absolute scaled error divides forecast absolute errors by a declared training-only naive-error scale. With seasonal period \(m\), that scale is

\[
q_T=\frac1{T-m}\sum_{i=m+1}^{T}|y_i-y_{i-m}|.
\]

Then average \(|e|/q_T\), stating whether each rolling origin uses its own training scale or one fixed reference scale. A score below one compares with that **training** naive-error scale; it does not logically prove a win over the naive forecasts on the later assessment period. Zero scale makes the ratio undefined. MAPE has its own problem at zero or near-zero outcomes. The [forecast-accuracy chapter](https://otexts.com/fpp3/accuracy.html) develops these metric choices and their units.

A log transformation can stabilize large level-dependent variation, but transforming the mean is not the same as averaging transformed predictions back. If \(\log Y\) is modeled as normal with mean \(\mu\) and variance \(\sigma^2\), \(\exp(\mu)\) is the median of that positive lognormal model and its mean is \(\exp(\mu+\sigma^2/2)\). The distributional assumption is essential. A simple exponential back-transform does not universally give an unbiased mean forecast.

Decomposing a series into trend, seasonality and remainder can make patterns easier to reason about. A two-sided smoother fitted using the full series, however, cannot supply a historically available feature at an earlier origin. Fit the allowed decomposition using only that origin's information, or label the full-series decomposition as retrospective analysis. Further ARIMA, GARCH, exponential-smoothing and decomposition modeling belongs to the later time-series specialization; this lesson establishes the evaluation contract those methods must obey.

Drift in the observed error can motivate a shorter window, a new feature or retraining, but monitoring and changing the policy create a new adaptive procedure. Record the trigger and evaluate the whole response policy. A sudden low rental count could reflect weather, service availability, reporting or actual use; this file alone does not identify the cause.

## Where to go next

The next lesson in this module, [End-to-End Supervised Learning & Error Analysis](/learn/path/full-curriculum/end-to-end-supervised-learning-error-analysis?module=classical-ml), brings task formulation, baseline comparison, disciplined selection and error analysis into one complete experiment. Its independent-row example uses a different split because its prediction task is different; the information-boundary principle remains the same.

For another way to learn this material:

- [FPP3: simple forecasting methods](https://otexts.com/fpp3/simple-methods.html) — a free introductory reading with mathematical baseline definitions and contrasting examples. Use it to check which historical observation each rule copies.
- [FPP3: time-series cross-validation](https://otexts.com/fpp3/tscv.html) — rolling-origin diagrams and executable R/fable examples. Translate the issue schedule before borrowing its code for a different horizon.
- [scikit-learn: lagged features](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html) — a current Python/Polars walkthrough of lag construction, forward assessment and quantile prediction. Its hourly next-step task differs from this lesson's weekly direct forecasts.
- [UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) — original data, variable definitions, citation and license. The daily member and complete provider description are retained with this packet.
- [FPP3: prediction intervals](https://otexts.com/fpp3/prediction-intervals.html) and [distributional forecast accuracy](https://otexts.com/fpp3/distaccuracy.html) — optional deeper reading on claims a point forecast cannot make. These are additional learning resources, not a claim that intervals were fitted in the bicycle experiment.

The resource record states which pages and sections were inspected. The complete lesson and offline programs stand on their own; following a video or external library tutorial is not a hidden prerequisite.
