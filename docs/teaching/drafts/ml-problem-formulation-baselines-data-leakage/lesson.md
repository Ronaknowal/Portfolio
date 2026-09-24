# ML Problem Formulation, Baselines & Data Leakage

A team has a model that predicts whether a customer will accept an offer. Its strongest input is how long the conversation lasted. The score looks promising. There is one problem: the team wants to choose whom to call **before the conversation begins**.

The model may have learned a real relationship. It still cannot perform the requested job using that input. Defining a useful prediction requires more than choosing an algorithm and measuring its accuracy.

This lesson connects the methods you have studied to a complete question: **what decision will this prediction support, what information could the system actually know, and what comparison would establish that the prediction helps?** We will use real historical call records, build a small measured comparison, and reconstruct what a system could know at an earlier moment.

First pass: §§1–7 and practices 1–5. You need a table's rows and columns, the idea of fitting on examples, and basic classification metrics. The earlier metrics and cross-validation lessons supply additional detail; their essential meanings are refreshed locally. Section 8 is a deeper branch on selection, causal effects and changing objectives. It is not required to finish the core.

## 1. Begin with a decision, then define the prediction

“Use machine learning on our customer data” leaves nearly everything important unspecified. A more useful starting statement is:

> Before the next day's calls are scheduled, rank eligible contact opportunities by the chance of a recorded subscription outcome, using information available at scheduling time.

This is a proposed operational question, not a claim that the supplied historical dataset can fully validate it. Writing it reveals what evidence we would need.

There are three separate objects:

1. **Outcome:** what we want to improve, such as useful contacts completed under a limited calling capacity.
2. **Prediction:** an estimate, such as a subscription probability for an eligible opportunity.
3. **Decision rule:** how an estimate becomes an action, such as selecting the highest-ranked 50 eligible opportunities.

The model's output is not automatically the decision, and the measured target is not automatically the outcome we ultimately care about. A probability model can support different capacities without retraining, provided its meaning and applicable population remain appropriate.

Imagine the flow as five connected boxes: available records → prediction → capacity/eligibility rule → action → observed outcome. An arrow from the final outcome back into tomorrow's training data is legitimate only after that outcome exists. An arrow taking tomorrow's outcome into today's prediction is the shortcut we must prevent.

### A compact experiment contract

Before fitting, fill in these fields in ordinary language:

| Field | Question to answer | Example for a proposed pre-call system |
| --- | --- | --- |
| Population | Which cases is this meant for? | Eligible contact opportunities under a stated campaign policy |
| Unit | What does one prediction represent? | One customer opportunity at a specific scheduling time |
| Cutoff | When must the prediction be ready? | Before the call list is finalized |
| Target | What observed answer will train/evaluate it? | A precisely defined subscription outcome and observation window |
| Features | Which information is available by the cutoff? | Versioned customer/history records with documented availability |
| Action | What happens to the prediction? | Rank eligible opportunities; select at most 50 |
| Baseline | What would happen without the proposed learner? | Existing policy and a simple probability/ranking baseline |
| Evaluation | What future situation does the split imitate? | Later eligible opportunities, keeping repeated units appropriately separated |
| Success | What improvement and constraints matter? | Useful outcomes at the fixed capacity, with latency and subgroup checks |

This contract prevents a project from silently changing its goal when a convenient metric improves. It also makes a legitimate change visible: a post-call reporting model is a different task from a pre-call scheduling model.

Machine learning may be unnecessary. A reliable explicit rule is attractive when the task is deterministic, the relevant information is already known, or a simple process change solves the problem. For a learned system, we need a learnable relationship, suitable examples, a usable feedback process and enough practical benefit to justify its cost. [Google's problem-framing course](https://developers.google.com/machine-learning/problem-framing) offers another guided route through that decision.

## 2. A row is an observation with a history

### Unit, target and observation window

Suppose a delivery service asks, “Will this parcel arrive late?” One row might represent a parcel when dispatched, that parcel every hour, or a customer order containing several parcels. Those choices create different training examples, dependence and decisions.

Define the label relative to a cutoff. If the question is whether delivery will occur more than one day after dispatch, a row collected six hours after dispatch may not yet have a known answer. Treating “no recorded late delivery yet” as “on time” creates false negatives.

A useful label record includes the event or observation window that defines the answer, when the answer became ascertainable, and the rule applied to missing or censored outcomes. The earlier survival-analysis lesson explains why incomplete follow-up is not automatically a negative event.

For repeated hourly rows from one parcel, a random row split can place nearly identical states from that parcel on both sides. That might assess interpolation among already represented parcels. It does not automatically assess predictions for entirely new parcels. The **evaluation unit** should match the generalization question.

### The name of a column is not its meaning

An integer called customer_id might be a harmless join key, an accidental timestamp, or a shortcut to a customer who appears in both training and validation. A column called previous_outcome might mean the previous campaign's outcome, or it might have been overwritten with the current outcome.

Write feature definitions with their time and provenance, not only their names:

> Count of successfully completed earlier contacts for this customer, using only events and versions available before this opportunity's cutoff.

That is more informative than “previous.” The historical file's documentation and actual data must establish whether its field has that meaning.

**Figure — One parcel, several rows.** Put all hourly observations for each parcel in one horizontal lane. Compare a row-random split with a split that holds out entire parcel identities. Label the question each split can investigate. The figure should show shared identity, not imply that any repeated customer is forbidden in every prediction task.

### Targets can be proxies

A recorded subscription is an observable event. Customer benefit, satisfaction or the incremental effect of calling are different quantities. Similarly, a click is not identical to a useful recommendation, and a short service time is not identical to a well-resolved problem.

A **proxy target** is a measurable substitute for something harder to observe. Explain why it is informative and where it can diverge from the intended outcome. Improving a proxy is evidence about that proxy; check the downstream outcome separately. [Google's framing discussion](https://developers.google.com/machine-learning/problem-framing/ml-framing) distinguishes model outputs, proxy labels and success measures.

## 3. Three clocks determine what was knowable

For a feature record, distinguish:

- **Event time:** when the underlying measurement or event happened.
- **Available-at time:** when the prediction system could use this particular value.
- **Prediction cutoff:** the moment at which we must reconstruct the available information.

A measurement can happen before a prediction yet arrive afterward. A database's insertion timestamp can help only if its contract actually represents availability to the serving system. A delayed sync, publication schedule or feature computation can add another delay.

Use this constructed calibration history. Times are simple abstract units; values are calibration offsets.

| Sensor | Event time | Available at | Version | Value |
| --- | ---: | ---: | ---: | ---: |
| A | 1 | 1 | 1 | 10 |
| A | 4 | 8 | 1 | 20 |
| A | 1 | 6 | 2 | 12 |
| B | 4 | 4 | 1 | 99 |

A prediction for sensor A is required at time 5. Which value can it use?

The event at time 4 looks newest, but value 20 does not arrive until time 8. The correction to event 1 arrives at time 6. At time 5, the appropriate value in this fixture is therefore **10**. Sensor B's available measurement belongs to the wrong entity.

If the event-4 value arrives at time 4 instead, the answer becomes 20. If the cutoff moves to 7 under the original history, event 4 is still unavailable but event 1's correction is known, so the answer becomes 12. At cutoff 9, event 4's value 20 is known and is the newest eligible event.

We have changed knowledge, not historical event order.

### A precise reconstruction rule

For this particular “latest eligible calibration” policy:

1. Match the entity.
2. Keep records whose event time is no later than the cutoff.
3. Keep versions whose available-at time is no later than the cutoff.
4. Enforce the maximum allowed event age.
5. Choose the newest eligible event; among its eligible versions, choose the latest available version.

If none qualifies, represent a missing calibration and follow the application's fallback policy. Do not silently substitute a future value. Equal-time ambiguity requires a documented version ordering; our fixture has an integer version for that purpose.

Here is a complete small implementation. It illustrates the rule; a production historical join should use an appropriately indexed or vectorized implementation.

~~~python
records = [
    dict(entity="A", event=1, available=1, version=1, value=10),
    dict(entity="A", event=4, available=8, version=1, value=20),
    dict(entity="A", event=1, available=6, version=2, value=12),
    dict(entity="B", event=4, available=4, version=1, value=99),
]

def latest_known(rows, entity, cutoff, maximum_age):
    eligible = [
        row for row in rows
        if row["entity"] == entity
        and cutoff - maximum_age <= row["event"] <= cutoff
        and row["available"] <= cutoff
    ]
    return max(
        eligible,
        key=lambda row: (row["event"], row["available"], row["version"]),
        default=None,
    )

for cutoff in [5, 7, 9]:
    result = latest_known(records, "A", cutoff, maximum_age=9)
    print(cutoff, None if result is None else result["value"])
print(latest_known(records, "A", cutoff=5, maximum_age=2))
~~~

The results are 10, 12, 20, followed by None. The last query rejects event 1 as too old and event 4 as not yet available. A maximum-age tolerance constrains freshness; it does not make unavailable information available.

**Investigation — Move the arrival, preserve the event.** The timeline shows an event marker and a separate arrival marker joined by a segment for each record. Edit an arrival, cutoff, age limit or value and inspect the actual eligible set and selected version immediately. A useful null is editing sensor B's value: the selected calibration for A must remain unchanged.

### Why a backward join alone is insufficient

Pandas merge_asof with backward direction finds the last suitable key no greater than a query key, optionally within a tolerance and entity group. If that key is event time, this operation alone does not also enforce the available-at predicate above. Its documented behavior is a key-matching rule, not a guarantee about the meaning of your data. [Pandas API](https://pandas.pydata.org/docs/reference/api/pandas.merge_asof.html).

Feature stores can implement historical retrieval, but inspect the selected version and backend. Feast's July 2026 merged change adds an opt-in created-timestamp cutoff with documented store support limitations. This is separate from event-time TTL matching; do not assume every historical join enables both conditions. It also requires that created time suitably represents availability for the task. [Feast change and support contract](https://github.com/feast-dev/feast/pull/6617).

Keep historical versions if future reconstruction matters. Overwriting event 1's original value with 12 erases the information needed to reproduce the cutoff-5 answer.

## 4. Establish what a simple system already achieves

“Better than random guessing” can be an extremely weak target. In a dataset with 90 negative and 10 positive cases, always predicting negative has 90% accuracy and zero recall of the positive class.

Choose baselines for the question:

| Baseline | What it tests |
| --- | --- |
| Existing operational rule | Does the proposed system improve the decision people currently make? |
| Majority class or training class prior | Is there useful discrimination beyond class prevalence? |
| Training mean or median | Does a regression model improve on a constant matched to the loss? |
| A transparent feature/rule or small model | Does additional complexity earn its cost? |
| Last observed or seasonal value | Does a forecasting model improve on temporal persistence? |

Fit data-dependent baselines on training information. The constant minimizing training squared error is the training mean; the constant minimizing absolute error is a training median. To see the first, expanding the sum around the mean leaves a fixed residual sum plus \(n(c-\bar y)^2\), minimized at \(c=\bar y\). Baselines have assumptions too.

A probability baseline predicts the training positive fraction for every case. Its scores all tie, so it supplies no ranking information. Its validation average precision equals the validation positive fraction; selecting a particular top 50 among tied scores depends on the tie rule and is not a learned advantage.

### Match the metric to the decision

If capacity is 50 calls, inspect the first 50 ranked cases:

\[
\mathrm{precision@50}=\frac{\text{positive outcomes among selected 50}}{50},
\qquad
\mathrm{recall@50}=\frac{\text{positive outcomes among selected 50}}
{\text{all positive outcomes in the evaluated set}}.
\]

These answer different questions. Precision describes concentration in the selected set; recall describes how much of the observed positive population that set captures. Neither alone estimates how many outcomes calling actually causes.

For probability quality, log loss penalizes assigning very low probability to outcomes that occur. Average precision summarizes a precision–recall ranking with its defined threshold convention. The earlier [evaluation-metrics lesson](/learn/path/full-curriculum/evaluation-metrics-precision-recall-f1-auc-roc-ap-r-mae?module=classical-ml) develops those calculations; here we use them to check a stated decision.

For a binary action with fixed false-positive cost \(C_{\rm FP}\), false-negative cost \(C_{\rm FN}\), zero correct-action costs, and a probability \(p\) valid for the current case, acting has expected loss \(C_{\rm FP}(1-p)\), while not acting has loss \(C_{\rm FN}p\). Acting is preferable when

\[
p>\frac{C_{\rm FP}}{C_{\rm FP}+C_{\rm FN}}.
\]

This derivation assumes the action does not change the target's meaning and omits capacity or action-specific effects. With a hard capacity, decisions become coupled: selecting one case may displace another. Document the actual decision rule instead of treating .5 as a universal threshold.

## 5. Real records: a higher score can answer the wrong question

Our offline [bank-additional.csv](bank-additional.csv) is the provider's unchanged 4,119-row random subset of historical Portuguese bank marketing records. It has 20 input columns and a binary recorded subscription target; 451 rows have the positive label. The source is Moro, Rita and Cortez's Bank Marketing dataset under CC BY 4.0. Its own description explicitly identifies final call duration as unavailable before a call. [UCI source and license](https://archive.ics.uci.edu/dataset/222/bank+marketing); [retained variable description](source-description.txt).

The supplied subset lacks the complete entity/availability history needed to establish a deployable pre-call policy or an independent future-customer test. Our measured question is narrower: **how do a simple recorded-feature pipeline and the same pipeline with an unavailable feature compare on matching held-out rows?**

We reserve 824 rows without scoring them. Within the other 3,295, use 2,471 to fit and 824 for the stated diagnostic comparison, with fixed stratified seeds. This is development evidence, not a final operational acceptance result.

Choose a small, explicit candidate feature set: age, previous-contact count, prior-contact availability/days, and seven customer/history categories. Exclude current-call duration in the candidate pipeline. Exclude campaign/scheduling and economic columns here rather than quietly asserting their exact pre-call availability. Their usefulness and historical versions would need a separate contract.

The provider uses 999 in pdays to mean no previous contact. Treating 999 as an actual elapsed time would invent a distance. We create a contacted-before indicator and replace that sentinel with missing in the elapsed-days feature. Medians and scaling are fitted inside the training pipeline; categories retain the provider's explicit unknown values.

### Complete experiment

Place the CSV beside this program. It requires NumPy, pandas and scikit-learn; the recorded calculation used versions 2.3.5, 3.0.1 and 1.9.1 respectively. The model is a regularized logistic regression with fixed settings, not a search for a winning seed.

~~~python
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, log_loss

data = pd.read_csv("bank-additional.csv", sep=";")
y = (data["y"] == "yes").to_numpy(dtype=int)
development, reserved = train_test_split(
    np.arange(len(y)), train_size=3295, stratify=y, random_state=53
)
train, validation = train_test_split(
    development, train_size=2471,
    stratify=y[development], random_state=54
)
x = data.copy()
x["contacted_before"] = (x["pdays"] != 999).astype(int)
x["days_since_previous"] = x["pdays"].replace(999, np.nan)
numeric = ["age", "previous", "contacted_before", "days_since_previous"]
categorical = ["job", "marital", "education", "default",
               "housing", "loan", "poutcome"]

def make_model(include_duration):
    columns = numeric + (["duration"] if include_duration else [])
    preparation = ColumnTransformer([
        ("numeric", make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler()), columns),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return make_pipeline(
        preparation, LogisticRegression(C=1., max_iter=600)
    )

def report(name, probability):
    # Break exact probability ties by the original source row index.
    ranking = np.lexsort((validation, -probability))
    actual = y[validation]
    found = int(actual[ranking[:50]].sum())
    print(name,
          "AP", round(average_precision_score(actual, probability), 6),
          "log loss", round(log_loss(actual, probability), 6),
          "correct", int(((probability >= .5) == actual).sum()),
          "top-50 positives", found)

report("training prior", np.full(len(validation), y[train].mean()))
for name, duration in [
    ("candidate pre-call", False),
    ("unavailable duration", True),
]:
    model = make_model(duration).fit(x.iloc[train], y[train])
    report(name, model.predict_proba(x.iloc[validation])[:, 1])
~~~

The recorded validation set contains 90 positive and 734 negative outcomes:

| Procedure | Average precision | Log loss ↓ | Correct at threshold .5 | Positives in top 50 |
| --- | ---: | ---: | ---: | ---: |
| Training-prior baseline | .109223 | .344889 | 734/824 | 6 |
| Candidate recorded-feature model | .253440 | .336527 | 733/824 | 20 |
| Same feature families plus final duration | .461700 | .264639 | 738/824 | 26 |

The candidate has one fewer correct class decision than the majority baseline at threshold .5, while concentrating more positives in the top 50. This is why the operational question matters: classification accuracy and limited-capacity ranking assess different behavior.

The duration model appears better on several numbers. That does not make final duration usable before a call. Its validation rows can be entirely separate from training and still contain the wrong information for the intended cutoff. A correct train-only scaler cannot repair that feature definition.

The prior baseline's six positives among the first 50 tied cases are a consequence of the fixed source-row tie rule; all baseline probabilities are identical. It did not discover a ranking signal.

**Figure — Scores beside information contracts.** Align the three measured result rows with their actual feature paths. The duration path crosses from after-call information into a proposed before-call prediction and is marked unavailable. Keep the baseline visible. Do not use a green “winner” badge for the highest AP; the figure's question is whether the comparison answers the intended task.

**Investigation — A score becomes a limited action set.** Start with the candidate's ranked opportunities and a capacity of 50. Record whether precision will rise, fall or stay unchanged when you edit the capacity, then reveal the selected cases and their recorded outcomes. You can also move individual cases across the selection boundary to explore a different proposed action set; this edits the decision policy, not the model's measured probabilities. Reordering cases entirely within the selected set is a useful null. The goal is to see which identities enter each metric, not to optimize a policy on these revealed validation answers.

The retained [calculation inputs](calculated-inputs.json) include split identities, each validation target and probability, actual rankings and confusion matrices. The [provenance](data-provenance.md) separates raw source facts, derived features and author calculations.

## 6. Locate the shortcut, not just the suspicious score

Data leakage is an information path that gives a fitting or evaluation procedure access to information disallowed by the intended prediction/evaluation contract. It often improves a score, but leakage need not produce a spectacular metric, and a high score alone does not prove leakage.

| Failure | Concrete shortcut | Appropriate repair |
| --- | --- | --- |
| Target/temporal leakage | Final call duration used before calling | Redefine features at the actual cutoff; recollect historical versions if needed |
| Preprocessing leakage | Select features using all labels before splitting | Fit selection inside each training fold |
| Unit contamination | Same parcel's near-duplicate rows on both sides of a new-parcel test | Split at the relevant independent unit |
| Selection leakage | Repeatedly inspect a final test to choose variants | Use development selection; obtain valid new evaluation for the selected procedure |
| Availability/revision leakage | Latest corrected value used in an earlier snapshot | Reconstruct as-known versions |
| Deployment mismatch | Training on completed historical calls, applying to all eligible future customers | Establish population coverage and evaluate the new question |

The last row need not involve a hidden information channel. It can be a distribution or policy mismatch. Naming the mechanism determines the repair.

A pipeline is useful because each fold can fit its imputer, scaler, selector and model together on that fold's training portion. It does not decide whether an input is from the future, whether two rows represent the same parcel, or whether the target measures the desired outcome. [scikit-learn's worked pitfalls](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) shows train-only transformation and feature-selection examples.

### Follow the information path

For a suspicious result, trace each influential feature backward:

1. Which raw records produced it?
2. Which entity and time does each record describe?
3. When could the system use that particular version?
4. Which rows/labels determined its fitted transformation?
5. What model or reporting choice used the evaluation results?

A lineage diagram makes this more concrete than a blanket instruction to “avoid leakage.” The same outcome can reach a feature through an aggregate, a status code, a filename or a human workflow.

A useful diagnostic removes or delays the suspected feature and repeats a controlled development comparison. A collapse in score may support the suspected dependency; it does not prove all remaining inputs are valid. Conversely, little score change does not legalize an unavailable feature. The admissibility rule comes from the information contract, not its measured importance.

Negative controls can reveal shortcuts. An entity-identity-only model that predicts supposedly unseen entities deserves investigation. A timestamp-only model can expose a temporal regime or a split artifact. Permuting labels can help diagnose some pipeline errors when the permutation respects the relevant structure, but it is not a universal leakage detector and does not replace provenance.

## 7. Turn the study into a reviewable next step

For the real example, a defensible conclusion is:

> On this fixed historical row-level development split, the candidate feature pipeline improves average precision and top-50 concentration relative to a constant prior, while threshold-.5 accuracy does not improve. Adding final duration improves the diagnostic scores but violates a pre-call information contract. These results justify examining a properly timestamped, entity-aware pre-call dataset; they do not establish a deployed policy's benefit.

That conclusion names an actual finding and a next piece of evidence. It does not ask a larger model to fix missing history.

A useful saved experiment packet contains:

- The target, unit, prediction cutoff, population and label-maturity rules.
- Data identity, source/availability definitions and deterministic row/group/time assignments.
- Baseline and candidate pipelines, fitting scope, fixed settings and any selection performed.
- Per-case outputs and the decision rule, alongside aggregate and relevant slice metrics.
- A conclusion, concrete unresolved assumptions and the next discriminating experiment.

Do not freeze a misleading problem forever. If writing the contract reveals that the task should be “predict unresolved requests after 30 seconds,” revise the target, cutoff and available partial-history features together. A final conversation duration from old data cannot stand in for elapsed duration observed at 30 seconds. Reconstruct or collect the correct training examples.

This connects naturally to the next [Time-Series Validation & Forecasting Baselines lesson](/learn/path/full-curriculum/time-series-validation-forecasting-baselines?module=classical-ml): once the deadline and label window are explicit, we can move them forward in time and build a rolling evaluation that respects them.

## 8. Deeper: when a good predictor still chooses the wrong action

### Prediction is not an intervention effect

Suppose two original hypothetical customer groups have these probabilities under two actions:

| Group | Subscription probability if called | If not called | Increment due to calling |
| --- | ---: | ---: | ---: |
| A | .80 | .75 | .05 |
| B | .45 | .10 | .35 |

Ranking by the first column favors A. Ranking by the incremental effect favors B. These probabilities are invented to expose the distinction; the bank file does not identify both potential outcomes for each customer.

Observed outcomes under a historical calling policy estimate associations in that collected population. Estimating the effect of changing the policy needs experimental or justified causal evidence. The preceding causal-graphical-models lesson supplies that framework. No amount of ordinary holdout accuracy supplies an unobserved counterfactual by itself.

### Missing labels can be selected by the old system

If only inspected machine parts receive a precise defect label, the labeled table describes inspected parts. An old inspection rule may preferentially select unusual parts. A model trained and evaluated within that selected table can perform well yet be poorly assessed for uninspected production.

Write the selection process into the contract: who was observed, who was omitted, which outcomes became known, and whether there is support for the intended population. Weighting or extrapolation needs assumptions; it is not a way to create evidence in a region with no relevant observations.

### The loss should preserve information needed by the action

Consider provisioning spare parts for a one-period demand of 10 or 20 with equal probability. Predicting mean demand gives 15. But if each missing part costs three units and each unused part costs one, stocking 10 has expected cost \( .5(3\times10)=15\); stocking 20 costs \( .5(1\times10)=5\). Under this simplified cost model, mean prediction plus an automatic “stock the mean” rule is not the optimal decision.

For underage cost \(c_u\) and overage cost \(c_o\), minimizing expected asymmetric absolute loss selects a quantile at level \(c_u/(c_u+c_o)\). For a continuous demand distribution with CDF \(F\), increasing stock slightly adds overage cost on the fraction \(F(q)\) below the current level and removes underage cost on the fraction \(1-F(q)\) above it. The derivative is therefore \(c_oF(q)-c_u(1-F(q))\), which vanishes at that quantile; at a point mass use the corresponding one-sided condition. In this example the .75 quantile is 20. The earlier quantile-regression lesson owns fitting that conditional quantity; [QuantileRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html) provides a pinball-loss implementation.

This is why “classification for fixed thresholds, regression for changing thresholds” is a heuristic, not a theorem. A conditional distribution, calibrated probability or quantile may preserve more useful information than a hard category. Choose outputs and losses for the actual decisions and information available.

Core readiness does not require a causal estimator or a quantile proof. The deeper skill is recognizing when the target, observed labels and intended action no longer describe the same problem.

## 9. Practice: repair a question before improving a score

### 1. A changed historical snapshot

For sensor A, records are (event, available, version, value): (2,2,1,8), (4,7,1,11), (2,5,2,9). At cutoff 6 with maximum age 5, which value qualifies? What changes if the event-4 value arrives at time 4? What if maximum age becomes 1?

<details><summary>Hint</summary>
Apply entity, event, availability and age conditions before choosing a version.
</details>
<details><summary>Solution</summary>
Originally choose 9: event 2's correction is known by 6, while event 4 is unavailable. Moving its arrival to 4 makes 11 eligible and newest. With maximum age 1, event times must be at least 5, so none qualifies in either arrival case. Keep missingness visible rather than inventing a future calibration.
</details>

### 2. A convincing but invalid delivery model

A model predicts whether a parcel will arrive late using a field set by the final delivery scan. Its test parcels are all different from training parcels. Explain why the split is insufficient and write a valid cutoff/feature rule.

<details><summary>Solution</summary>
Holding out parcel identities prevents one contamination mechanism, but a post-delivery field is still unavailable at dispatch. Specify a dispatch-time prediction and reconstruct only fields/events/versions available then. Define “late” and label maturity independently. If the goal changes to post-delivery reporting, acknowledge that it is a different task and test whether prediction is even needed.
</details>

### 3. Accuracy, service cost and capacity

There are 12 support requests, three truly urgent. Policy A escalates none. Policy B escalates four: three urgent and one ordinary. An unnecessary escalation costs 2 units; missing an urgent request costs 7. Compute accuracy and cost for both. If capacity is only three escalations, is B feasible as written?

<details><summary>Hint</summary>
Count errors by type before using a single total.
</details>
<details><summary>Solution</summary>
A is correct on 9/12 with cost 21. B is correct on 11/12 with cost 2, but exceeds capacity three. A revised rank-and-select rule needs to state which three are selected; B's four-case confusion counts alone do not determine the new result. This is an action constraint, not something an accuracy number encodes.
</details>

### 4. Change the capacity on real data

Using the saved validation probabilities or rerunning the displayed fixed experiment, change capacity from 50 to 25. Calculate the precision/recall changes and explain them from the selected rows. Use the same source-row tie rule and leave model settings unchanged.

<details><summary>Hint</summary>
The top 25 are a subset of the top 50 under the fixed ranking. Recall cannot increase; precision can move either way.
</details>
<details><summary>Assessment and solution method</summary>
Sort by descending probability and then source-row index, count positives among the first 25, and divide by 25 for precision and by 90 for recall. The retained candidate ranking contains 12 positives in its first 25: precision .48 and recall \(12/90=.133333\), compared with .40 and \(20/90=.222222\) at capacity 50. Precision rises here while recall falls; a higher precision is not guaranteed by the word “top.” Include the selected rows and the actual result; do not search for a favorable capacity and report it as untouched evaluation.
</details>

### 5. Write the missing contract

A colleague says, “Our model detects failed equipment with 97% accuracy on randomly split sensor rows.” List at least five missing definitions, then propose an evaluation for predicting failure of a new machine within the next day.

<details><summary>Solution</summary>
Specify the machine/forecast-origin unit, failure event and 24-hour window, feature/label availability, population of machines, repeated-measurement grouping, baseline, meaningful costs/metrics and split time. Hold out the intended new machines and respect prediction deadlines and mature labels within each training snapshot. Describe how many independent machines and events support the evaluation; thousands of rows need not mean thousands of independent units.
</details>

### 6. Derive a changed decision threshold

For the fixed binary-cost setting in §4, let false-positive cost be 3 and false-negative cost 9. Compute the threshold. At \(p=.2\), compare both expected losses. Name a condition that would invalidate this simple decision calculation.

<details><summary>Solution</summary>
The threshold is \(3/12=.25\). Acting costs \(3(.8)=2.4\); not acting costs \(9(.2)=1.8\), so do not act under these assumptions. A hard shared capacity, changed outcome under intervention, different per-case costs or probabilities invalid for the deployment population requires a revised calculation.
</details>

### 7. A delayed label is not a negative label

You predict an event within seven days of signup. At a dataset snapshot on day 20, a person signed up on day 18 and has no event recorded. Explain why assigning a negative label can be wrong. How would the answer differ if the event already occurred and was reliably recorded on day 19?

<details><summary>Solution</summary>
Only two days of the seven-day window have elapsed, so absence so far does not establish a complete-window negative. Wait for maturity or use a method that explicitly represents incomplete follow-up. A reliably observed event on day 19 establishes a positive within the window; delays in recording would need their own rule. The exact training policy should state which known positives and incomplete negatives are eligible.
</details>

### 8. Deeper transfer: rank propensity or impact?

In an invented outreach example, group C has outcome probabilities .6 with action and .55 without; D has .4 with action and .1 without. Which group has greater observed-action propensity, which greater increment, and why can't a standard classifier on acted-upon cases alone settle the second question?

<details><summary>Solution</summary>
C has higher probability under action, .6 versus .4. D has greater increment, .3 versus .05. A classifier on acted-upon cases does not directly observe their no-action counterfactuals; the increment needs an appropriate experimental or causal identification strategy. These are constructed probabilities, not an effect estimate from the bank data.
</details>

You are ready for the next lesson when you can write a target/unit/cutoff contract, reconstruct a changed as-known snapshot, select and interpret a meaningful baseline, identify an actual information shortcut, and state what your evaluation does and does not answer.

## References & another way to learn it

- [Google — Introduction to ML Problem Framing](https://developers.google.com/machine-learning/problem-framing), especially [understanding the problem](https://developers.google.com/machine-learning/problem-framing/problem) and [framing outputs and success](https://developers.google.com/machine-learning/problem-framing/ml-framing). A short interactive reading route for outcomes, outputs and proxy labels. The core text and section structure were inspected; categorical model-choice and feature-correlation rules should be treated as heuristics, with the qualifications in this lesson.
- [scikit-learn — Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html). Executable explanatory articles on inconsistent preparation and feature-selection leakage. The relevant text/code was read; its example scores are separate from our measured study.
- [Pandas — merge_asof](https://pandas.pydata.org/docs/reference/api/pandas.merge_asof.html). Read the backward matching, grouping, sorting and tolerance contract alongside §3. An event-time key alone does not express a second availability condition.
- [Feast — Point-in-time joins](https://docs.feast.dev/getting-started/concepts/point-in-time-joins), with [the merged created-time filtering change](https://github.com/feast-dev/feast/pull/6617). A systems-oriented alternative. Event freshness, known-time filtering and backend support are separate; the change record was inspected, and no Feast installation was executed.
- [UCI — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing). Provider description, citation, license and data download. Moro, S., Rita, P., & Cortez, P. (2014), DOI 10.24432/C5K306. The packet preserves the original variable description and the source paper citation; the full subscription-paper text was not reviewed.
