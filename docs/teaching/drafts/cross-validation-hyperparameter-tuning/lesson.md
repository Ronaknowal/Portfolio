# Cross-Validation & Hyperparameter Tuning

A model can improve its score by learning the subject, or by becoming unusually well adapted to the examples we use to judge it. Those are different kinds of progress. If we keep trying settings until one looks excellent on a familiar evaluation set, the selected score can become more impressive than the model's future behavior.

Cross-validation gives each observation a turn as held-out evidence. Hyperparameter tuning uses such evidence to choose a learning procedure. To combine them well, we must decide **what is being chosen, what is being assessed, and which information each decision is allowed to use**.

**First pass:** read sections 1–6 and try core practice questions 1–6. You need the preceding lesson's fit/transform distinction, Python indexing, and the idea of a nearest-neighbor prediction. We define scores and split roles locally. Sections 7–9 deepen uncertainty, adaptive search and computational budgeting; they do not become hidden prerequisites for the core readiness check.

## 1. What exactly are we trying to estimate?

Imagine building the penguin species classifier from [Feature Scaling, Encoding & Imputation](/learn/path/full-curriculum/feature-scaling-encoding-imputation?module=classical-ml). The program learns medians and scales from training rows, then predicts a species from nearby examples. Three questions can sound like “how good is the model?”:

1. How well does this particular fitted model predict new observations from the intended population?
2. How well does a fixed learning recipe usually work when fitted to a new training sample of a specified size?
3. How well does the whole process of trying settings, selecting one, and refitting usually work?

The first question concerns a fitted object. The second concerns a learning algorithm. The third includes **selection** as part of that algorithm. A score should name which question it answers.

A **parameter** is learned during ordinary fitting, such as a regression coefficient. A **hyperparameter** specifies how that fitting or prediction works, such as a neighbor count, penalty strength, maximum tree depth, or preprocessing choice. The distinction depends on the procedure: a neighbor count supplied by the programmer becomes a choice learned from data when we compare counts using validation scores.

For classification here, **accuracy** is the number of correct predictions divided by the number assessed. For regression, **mean squared error** averages $(y-\hat y)^2$; it has squared target units and lower is better. A scoring rule should reflect the practical question. If rare failures matter much more than common successes, accuracy alone can be unsuitable; the later evaluation and imbalance lessons develop alternatives.

Use these names for data roles:

| Role | What it can influence |
| --- | --- |
| Training | Fit preprocessing and model parameters |
| Validation | Compare candidate settings or stop training |
| Test / outer assessment | Assess a procedure whose choices were made elsewhere |

The names refer to **use**, not permanent properties of rows. A row can be a training example in one fold and a validation example in another. Within a particular assessment, it must not influence the fitted or selected procedure that is being judged on it.

**Figure 1 — Two loops, two kinds of learning.** A training loop turns a setting and training rows into a fitted model. Around it, a selection loop compares settings using validation results. Outside both sits the assessment set. The figure identifies exactly which loop receives each score.

## 2. Build cross-validation one held-out prediction at a time

### A fold is a role assignment

In K-fold cross-validation, partition the available development rows into K nonoverlapping **folds**, as equal in size as possible. For each fold, fit a fresh copy of the entire procedure using the other folds, predict the held-out rows, and save their results. Every row is assessed once in that K-fold run.

“Fresh copy” includes any learned imputer, scaler, feature selector, and model. The previous fold's trained object does not continue learning into the next fold. The folds also need not be exactly equal: seven observations can form folds of sizes 3, 2, and 2 without dropping the remainder.

Consider this constructed one-dimensional dataset:

| Row ID | Feature x | Label y |
| --- | ---: | ---: |
| 0 | 0 | 0 |
| 1 | 1 | 0 |
| 2 | 2 | 0 |
| 3 | 3 | 1 |
| 4 | 4 | 1 |
| 5 | 5 | 1 |
| 6 | 6 | 1 |

Use three consecutive folds and a one-nearest-neighbor classifier. A new x receives the label of the closest training x; equal distances use the smaller source row ID in this example.

| Held-out rows | Available training rows | Held-out predictions | Correct / assessed |
| --- | --- | --- | ---: |
| 0, 1, 2 | 3, 4, 5, 6 | 1, 1, 1 | 0 / 3 |
| 3, 4 | 0, 1, 2, 5, 6 | 0, 1 | 1 / 2 |
| 5, 6 | 0, 1, 2, 3, 4 | 1, 1 | 2 / 2 |

For the first assessment, every training label is 1, so all three predictions are 1. In the second, x=3 is closest to training x=2 and receives the wrong label 0; x=4 is closest to x=5 and receives 1. The third assessment correctly labels both remaining points.

This ordered split is useful for understanding the mechanics. It also reveals a design problem: ordering by class creates very different training problems. For approximately independent classification examples, we will often distribute classes across folds. For a future-in-time question, however, shuffling away the order could destroy the evaluation we actually need. Choose the split from the prediction question, not from whichever arrangement gives the largest number.

**Investigation 1 — Build the held-out predictions.** Predict a selected row's label, edit a feature or label, and assign rows to validation folds. Apply the new assignment to see its available neighbors and the complete prediction table. The selected row's own label affects whether its prediction is correct, but must not train the model making that prediction.

### An average over folds is not always an average over people or rows

The fold accuracies above are 0, 0.5, and 1. Their unweighted mean is 0.5. But only three of the seven row predictions were correct, giving pooled accuracy $3/7\approx0.4286$.

Neither arithmetic operation is mysterious. They assign different weights. The unweighted fold mean gives each fold one-third of the weight; pooled accuracy gives each row one-seventh. For a loss that can be added per row, write

\[
\widehat R_{\text{rows}}
=\frac1n\sum_{k=1}^K\sum_{i\in V_k}L(y_i,\hat f_{-k}(x_i))
=\sum_{k=1}^K\frac{|V_k|}{n}\widehat R_k.
\]

Here $V_k$ is the set of held-out indices in fold k, $\hat f_{-k}$ is the procedure fitted without them, and $\widehat R_k$ is that fold's mean loss. Equal fold sizes make this equal to the unweighted fold mean. Unequal group sizes may motivate another question: should each patient count equally, or should every visit count equally? State the intended unit and weighting.

For a nonlinear summary such as F1 or AUC, pooling predictions can change the quantity even with equal fold sizes. AUC, for example, compares positive–negative score pairs; pooling may compare scores emitted by different fitted models. Do not assume every metric can be averaged or pooled interchangeably.

### A complete splitter that keeps every row

This NumPy-only program recreates the small table. Install NumPy in your Python environment if needed (`python -m pip install numpy`). Save and run it as `cv_tiny.py`.

```python
import numpy as np

def kfold_indices(n, k, shuffle=False, seed=0):
    if not 2 <= k <= n:
        raise ValueError("Require 2 <= k <= n.")
    indices = np.arange(n)
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    folds = np.array_split(indices, k)
    for held in range(k):
        train = np.sort(np.concatenate([folds[j] for j in range(k) if j != held]))
        yield train, folds[held]

x = np.arange(7)
y = np.array([0, 0, 0, 1, 1, 1, 1])
fold_scores = []
total_correct = 0
for train, validation in kfold_indices(len(x), 3):
    distances = np.abs(x[validation, None] - x[train])
    nearest = train[np.argmin(distances, axis=1)]
    prediction = y[nearest]
    correct = int(np.sum(prediction == y[validation]))
    fold_scores.append(correct / len(validation))
    total_correct += correct
    print(validation.tolist(), prediction.tolist(), correct, "/", len(validation))
print("fold mean", np.mean(fold_scores))
print("pooled", total_correct / len(x))
```

The executed arithmetic gives `[0,1,2] → [1,1,1], 0/3`; `[3,4] → [0,1], 1/2`; `[5,6] → [1,1], 2/2`, then fold mean 0.5 and pooled 0.42857142857142855. Shuffling is optional because a splitter cannot know whether rows may be exchanged without changing the problem.

## 3. Choose splits that match the future use

### Independent examples and rare classes

If examples can reasonably be treated as independent draws from a common population, shuffled K-fold is a useful baseline. **Stratified K-fold** approximately preserves class proportions in each fold. This helps avoid folds that omit a rare class and prevents certain model/metric failures.

It cannot create missing examples: with two positive observations and five validation folds, at least three folds contain no positive observation. Nor does stratification make uncertainty disappear. Making folds more homogeneous can hide some variability caused by rare classes. It is a practical allocation choice, not a universal statistical correction. The [cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-with-stratification-based-on-class-labels) describes these engineering and interpretation limits.

Five or ten folds are common practical starting points, not universal optima. More folds use a larger training fraction in each fit and require more fits. Leave-one-out uses n folds, each with one assessed row and n−1 training rows. Its error variance depends on the learning procedure and data; “the training sets overlap, therefore variance must be highest” is not a valid general proof. Section 7 develops this distinction.

Repeated K-fold reruns different partitions of the **same dataset**. It can expose split sensitivity, but it does not collect more independent people. Repeated random holdouts let the training fraction vary independently of the number of repetitions; some rows may be assessed several times and others not at all. Leave-P-out enumerates all choices of P held-out rows, with $\binom nP$ fits, which quickly becomes expensive. These are different resampling plans, not stronger and stronger guarantees.

### New groups versus later records from known groups

Suppose each person contributes several sensor windows. If the intended use is on people absent from training, all windows from a held-out person must stay out of that fold's training set. `GroupKFold` and `LeaveOneGroupOut` express this requirement. `StratifiedGroupKFold` also tries to balance classes, but exact balance may be impossible when groups are indivisible.

If the intended use is to predict later records from already known people, a forward-time evaluation may legitimately include their earlier records. Calling that automatically wrong would change the question to unseen-person generalization. The problem is using information that would not be available at the actual prediction point, or reporting one deployment setting as evidence for another.

For a system serving new people in future calendar periods, both group and time constraints can matter. A standard named splitter may not express both; construct and inspect explicit index pairs. A splitter name is not a substitute for writing the required boundary.

**Figure 2 — Same records, different questions.** Show two patients' time-stamped windows. One panel holds out a whole patient for unseen-person assessment. The other trains on earlier times and assesses later times for a known-person task. Label which records and future use each panel represents, without awarding one panel a universal “correct” badge.

### Future prediction and label availability

For forecasting, train on information available before the prediction time and assess a later horizon. An expanding window keeps accumulating history; a sliding window limits how much old history remains. A deliberate gap may be necessary because labels mature late, features use overlapping windows, or the deployment pipeline has a delay.

For example, if a row formed at day t predicts an outcome measured through day t+7, a training row dated yesterday may not have a known target today. Removing rows solely by their feature timestamp is insufficient. Also ensure any rolling features use only permitted past observations.

`TimeSeriesSplit` is a useful index-based building block with options such as `gap`, `test_size`, and `max_train_size`; a gap counts rows, not elapsed days. Irregular observations and prediction horizons require explicit date logic. A timestamp column alone does not force every retrospective task to use forward validation. The evaluation must match the claimed use. The later [Time-Series Validation & Forecasting Baselines](/learn/path/full-curriculum/time-series-validation-forecasting-baselines?module=classical-ml) develops complete forecasting protocols.

### What an out-of-fold prediction table requires

`cross_val_score` assesses the provided splits. `cross_val_predict` additionally requires that each supplied row appear in a held-out set **exactly once**, so it can return one prediction per row. Ordinary K-fold and GroupKFold can meet that contract. A forward-time plan leaves an initial training prefix with no held-out prediction; repeated holdouts can assess rows multiple times. Those are valid evaluation plans, but they do not satisfy this API's partition requirement.

For a manual forward out-of-fold table, retain unpredicted entries as missing and use only rows with legitimate predictions in a downstream stacking model. Do not fill the prefix with predictions from a model trained on that same prefix. The preceding ensemble topic owns the full stacking construction; here the distinction is between an evaluation plan and a complete once-per-row prediction table. See the [cross_val_predict contract](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html).

Predefined benchmark splits can be represented explicitly too. Respect what the benchmark's held-out partition is intended to measure rather than reshuffling it merely to obtain a convenient score.

## 4. Selection can learn the validation answers

### A tiny example with no uncertain “true performance”

Imagine four validation cases whose labels are independent fair coin flips. Features contain no information about those labels. Two candidate rules predict either `[0,0,0,0]` or `[1,1,1,1]` on those cases. Each rule has expected accuracy 0.5 on independent future fair labels.

For validation labels `[1,1,1,0]`, choose the all-one rule: its validation accuracy is 0.75. That choice still has expected future accuracy 0.5. Across all sixteen equally likely validation-label patterns, the selected validation accuracy averages 0.6875:

| Number of ones | Number of patterns | Best correct count |
| --- | ---: | ---: |
| 0 or 4 | 2 | 4 |
| 1 or 3 | 8 | 3 |
| 2 | 6 | 2 |

Thus the average selected score is $(2\cdot4+8\cdot3+6\cdot2)/(16\cdot4)=0.6875$. Nothing learned a predictive signal. We used validation labels to choose the rule that happened to match them.

If sixteen candidate rules cover every possible four-bit prediction pattern, one scores 1 on every validation set. Its expected accuracy on independent future labels remains 0.5. Adding a duplicate of an existing rule, however, does not increase the best score. The number, dependence and flexibility of candidates matter; there is no universal fixed percentage of optimism per trial.

**Investigation 2 — Win the familiar cases.** Edit the four labels and candidate prediction patterns, predict the best validation score, then reveal the winner and its analytically known 0.5 future accuracy under the stated fair-label model. Enumerate all sixteen label patterns to compare average selected scores. A duplicate candidate is a checked null; a new matching pattern can improve validation without improving future prediction.

Real hyperparameter searches are less artificial, but the same selection mechanism can exploit noise in an estimated score. Its size depends on the task. Reporting a selected validation score as though it were untouched assessment evidence is the mistake; an individual selected score need not exceed every later test result. Cawley and Talbot's [primary model-selection study](https://www.jmlr.org/papers/volume11/cawley10a/cawley10a.pdf) demonstrates how optimizing a finite-data criterion can overfit it.

### Two valid ways to separate choosing and assessing

**Development plus a separate final assessment:** set aside an appropriate test set, perform all search using the development data, refit the selected pipeline on those development data, then assess it on the reserved set. A single split can be useful when it provides enough relevant independent assessment units; there is no universal row-count threshold that makes every test reliable.

**Nested cross-validation:** repeat the entire selection process inside each outer training set. For one outer fold:

1. Protect its outer assessment rows from all fitting and selection.
2. Split only the outer training rows into inner training/validation folds.
3. Evaluate every candidate pipeline in those inner folds.
4. Select a candidate using the declared inner score and tie rule.
5. Refit that candidate on all outer training rows.
6. Predict the protected outer rows and record the result.

Repeat for the other outer folds. The outer score assesses the **selection-and-fitting procedure at the outer training size**, under the split's assumptions. It is not a guarantee about an oracle-best setting or an exactly unbiased estimate of one final all-data model.

After this assessment, run the declared selection procedure on all available development data and fit the final model. Do not vote among outer-fold settings solely because they appeared there most often; each setting was selected using a different training sample. Do not select the outer fold with the highest score as the model to deploy.

**Figure 3 — Nested rooms for information.** Expand one outer fold into its inner folds, showing original row IDs at both levels. When the inner winner is chosen, show a new refit on the whole outer training area before one arrow reaches the protected assessment rows. Close that outer fold and repeat with a fresh object. The changing selected settings remain visible without treating them as competing outer candidates.

Early stopping is also a selection decision. If an outer assessment set picks the epoch, it is no longer untouched. Inside an inner candidate evaluation, choosing an epoch from that same inner validation set makes its score adaptive; a protected outer assessment can still assess that complete rule. For a cleaner inner comparison, use an additional stopping subset within inner training and reserve inner validation for comparison. What matters is keeping the claimed assessment boundary intact, including preprocessing of any stopping set.

## 5. Search the settings you actually mean to compare

### Grid search: a finite, inspectable comparison

Suppose the candidates are neighbor counts `[3,5,11]` and scalers `[standard,robust]`. A grid contains all $3\times2=6$ combinations. With three inner folds, that requires eighteen candidate fits, followed by a refit of the selected candidate if requested.

A small grid is useful when the choices themselves are meaningful and affordable. A grid with ten values for each of five independent choices has $10^5=100,000$ combinations. At five folds and one minute per fit, that is 500,000 fit-minutes before refits and overhead, not the time of one model training. The number of jobs is exact; wall time depends on training sizes, resources, and parallelism.

Some choices are conditional. An RBF kernel has a bandwidth parameter; a linear kernel does not need it. A list of separate parameter dictionaries can express those branches without wasting evaluations on irrelevant combinations. Likewise, do not search impossible layer shapes or a neighbor count larger than an inner training set.

### Random search: specify a distribution, not just a range

Random search draws candidates from a declared distribution. If a positive parameter spans orders of magnitude, a log-uniform distribution gives equal probability to equal multiplicative ranges. On `[0.001,1000]`, each decade has probability one-sixth. Uniform sampling on the original numeric scale instead assigns almost all probability to large values.

Suppose a satisfactory region has probability mass p under the chosen sampling distribution. For T independent draws, the probability of at least one hit is

\[
P(\text{hit})=1-(1-p)^T.
\]

If p=0.05, sixty draws give about 0.9539. If p=0.01, the same sixty give only 0.4528. This is a coverage calculation, **not** a theorem that sixty trials reach within 5% of the best score. A region containing 5% of the sampling probability is different from a score within 5% of an optimum.

A grid can repeatedly test the same few values along an important dimension while varying unimportant ones. Independent continuous random draws explore more distinct values along each coordinate. That is useful when effective importance is concentrated in a few unknown dimensions, but it does not make random search universally dominate a well-chosen small grid. The [Bergstra–Bengio paper](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf) provides the original empirical and geometric argument.

**Figure 4 — Coverage seen from one coordinate.** Show nine points on a three-by-three grid beside nine fixed, disclosed random draws. Project each set onto the important coordinate. This is an illustration of coverage, not a fabricated model-accuracy surface. The rank of search methods on a real task must come from actual evaluations.

Adaptive search and early resource allocation are useful extensions in section 8. They change how candidates receive attention, while leaving the selection/assessment boundary in place.

## 6. A complete nested experiment with real observations

We reuse the preceding lesson's 344-row Palmer Penguins CSV so the model and features stay familiar. Its CC0 data, measurement units and provenance are supplied with this lesson. This is a documented demonstration of a selection protocol on an already familiar dataset, not newly independent validation of a winner from the earlier page. The program protects every outer row from the inner decisions used to make its recorded prediction; generalization claims remain confined to the random-row setting represented by that experiment.

We compare six candidates: neighbor counts 3, 5, and 11, each with standard or robust scaling. Numeric medians, categorical imputation and one-hot vocabulary are learned inside every candidate training fold. Three stratified outer folds use seed 41; each inner three-fold split uses seed 73. Accuracy is the declared selection metric. Exact ties use the first candidate in the declared enumeration, making the procedure reproducible rather than silently choosing a favorable tie afterward.

Save this complete program as `cv_penguins.py` beside `penguins.csv`. The calculation was executed through equivalent operations in the supplied author script with Python 3.12.14, NumPy 2.3.5, pandas 3.0.1 and scikit-learn 1.9.1. For a new environment, install `numpy==2.3.5 pandas==3.0.1 scikit-learn==1.9.1` using `python -m pip install`. It runs serially and requires no network data request.

```python
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

def load_data():
    data = pd.read_csv(Path(__file__).with_name("penguins.csv"))
    columns = [
        "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex",
    ]
    return data[columns], data["species"]

def make_model():
    numeric = [
        "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g",
    ]
    prepare = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), numeric),
        ("category", Pipeline([
            ("impute", SimpleImputer(
                strategy="constant", fill_value="not_recorded",
                keep_empty_features=True,
            )),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), ["sex"]),
    ])
    return Pipeline([
        ("prepare", prepare),
        ("classify", KNeighborsClassifier()),
    ])

def main():
    X, y = load_data()
    grid = {
        "prepare__numeric__scale": [StandardScaler(), RobustScaler()],
        "classify__n_neighbors": [3, 5, 11],
    }
    outer = StratifiedKFold(n_splits=3, shuffle=True, random_state=41)
    held_out_predictions = np.empty(len(y), dtype=object)
    fold_scores = []
    for fold, (train, test) in enumerate(outer.split(X, y), start=1):
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=73)
        search = GridSearchCV(
            make_model(), grid, cv=inner, scoring="accuracy",
            n_jobs=1, error_score="raise",
        )
        search.fit(X.iloc[train], y.iloc[train])
        prediction = search.predict(X.iloc[test])
        held_out_predictions[test] = prediction
        correct = int(np.sum(prediction == y.iloc[test]))
        fold_scores.append(correct / len(test))
        baseline = DummyClassifier(strategy="most_frequent")
        baseline.fit(np.zeros((len(train), 1)), y.iloc[train])
        base_prediction = baseline.predict(np.zeros((len(test), 1)))
        base_correct = int(np.sum(base_prediction == y.iloc[test]))
        chosen_k = search.best_params_["classify__n_neighbors"]
        chosen_scaler = type(search.best_params_["prepare__numeric__scale"]).__name__
        print(fold, chosen_k, chosen_scaler, correct, "/", len(test),
              "baseline", base_correct)

    print("outer fold mean", f"{np.mean(fold_scores):.7f}")
    print("outer pooled", f"{np.mean(held_out_predictions == y):.7f}")
    final = GridSearchCV(
        make_model(), grid,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=73),
        scoring="accuracy", n_jobs=1, error_score="raise",
    )
    final.fit(X, y)
    print("final settings", final.best_params_)
    print("final selection score", f"{final.best_score_:.7f}")

if __name__ == "__main__":
    main()
```

The nested parameter name `prepare__numeric__scale` follows the pipeline into its numeric branch and changes that scaler. The estimator selection object clones and fits the entire pipeline for each candidate/fold. Its final refit uses the selected candidate on all rows supplied to that search, which are the outer training rows inside the loop.

Recorded results:

| Outer fold | Selected neighbor count / scaler | Correct / held-out | Majority baseline correct |
| --- | --- | ---: | ---: |
| 1 | 11 / StandardScaler | 114 / 115 | 51 / 115 |
| 2 | 3 / StandardScaler | 114 / 115 | 51 / 115 |
| 3 | 3 / RobustScaler | 112 / 114 | 50 / 114 |

The unweighted outer-fold mean is 0.9883549; pooled accuracy is $340/344=0.9883721$. The difference is small because fold sizes differ by only one row, but the weighting distinction is still real. The changing selected settings show that near-performing choices can depend on the training sample.

The final search on all 344 rows selects 3 neighbors with standard scaling and has inner selection score 0.9913043. That score is used to choose the final configuration. It does not replace the outer assessment or become a new untouched test result.

**Investigation 3 — Open one nested fold.** First inspect the real experiment's candidate scores and original row IDs. Then use a separate constructed sixteen-row dataset to edit features and labels and predict which neighbor count the inner comparison selects. The small experiment uses two outer folds, two inner folds, and candidate counts 1 and 3, allowing every fit to be traced. Changing a protected outer label can change assessment correctness but must leave that fold's inner choice and predictions unchanged. Changing an outer training label can change the selected count. The real observation results remain a separate, labeled record.

In the small experiment, x values are 0 through 15, with labels 0 below 8 and 1 from 8 upward. The first outer fold assesses even row IDs and trains on odd IDs; inner validation alternates positions within that training list. Both neighbor counts initially average 0.875 in the inner comparison, so the declared smaller-count tie rule chooses 1. Change only row 3's label from 0 to 1: inner means become 0.5 for count 1 and 0.625 for count 3, selecting 3. The change has altered a legitimate selection input. Editing row 2's label instead cannot affect this outer fold's selection because row 2 is protected assessment data.

In ordinary projects, also retain the split indices, preprocessing specification, candidate space, selection metric, tie rule, seed policy, and failed fits. A score without its selection procedure is difficult to reproduce and easy to misinterpret. `cv_results_` contains candidate scores and timing information; `cross_validate` can return additional metrics, fitted estimators and split indices. These are useful records, not evidence that an invalid split became valid.

## 7. Deeper branch: what cross-validation uncertainty does and does not mean

### Training size is part of the target quantity

For a fixed recipe A trained on m independent examples, define its expected new-example loss as

\[
R(m)=\mathbb E_{D_m,Z_{\text{new}}}
[L(A(D_m),Z_{\text{new}})].
\]

The expectation averages both the training sample $D_m$ and an independent new observation. A balanced K-fold estimate under an independent, identically distributed sampling setup targets performance at approximately $m=n(K-1)/K$ training rows. Its folds do not train on all n rows, so it need not target $R(n)$ exactly. Stratified and structured splits introduce additional conditions; do not transfer this simple independent-sampling statement to every design without checking them.

A calculation shows why this matters. Suppose observations $Y_i$ have mean μ and variance σ², and the model predicts their training mean $\bar Y_m$ for every new case. Since the new observation is independent of that mean,

\[
\mathbb E[(Y_{\text{new}}-\bar Y_m)^2]
=\operatorname{Var}(Y_{\text{new}})+\operatorname{Var}(\bar Y_m)
=\sigma^2+\frac{\sigma^2}{m}.
\]

With n=12 and σ²=4, a three-fold fit trains on m=8 and has expected new loss 4.5. A full twelve-row fit has expected new loss $4+4/12\approx4.3333$. The difference is training size, not evidence that the splitter leaked or that an implementation failed.

Now compare the same full-data mean predictor's training loss. The expected average squared residual on its own n training values is $\sigma^2(1-1/n)$. Its expected new loss is $\sigma^2(1+1/n)$, a gap of $2\sigma^2/n$. This is a precise simple example of **training optimism**: the model was fitted using the observations being scored. More complicated models can have different optimism; the mean-predictor formula is not an all-purpose correction.

**Figure 5 — Name the distribution being averaged.** One panel holds a fitted model fixed and draws new assessment observations. A second panel redraws both training and new observations. Beside them, the exact mean-predictor curves σ²(1−1/n) and σ²(1+1/n), for stated σ², separate training error and expected new error. Label them analytic, not measured learning curves.

### Why overlap is not a variance formula

For arbitrary fold losses $E_1,\ldots,E_K$,

\[
\operatorname{Var}\!\left(\frac1K\sum_kE_k\right)
=\frac1{K^2}\left(\sum_k\operatorname{Var}(E_k)
+2\sum_{j<k}\operatorname{Cov}(E_j,E_k)\right).
\]

The covariance terms are about **losses**, not just the fraction of training rows in common. If all variances equal τ² and every pairwise correlation equals ρ, this simplifies to $\tau^2[1+(K-1)\rho]/K$. Those assumptions explain the formula; they do not tell us that ρ equals the training-set overlap.

A useful counterexample is a learning rule that ignores training and always predicts zero. Its leave-one-out losses on independent observations are independent functions of their respective held-out observations. The training sets overlap almost completely, but the overlap has no influence on those predictions. Conversely, an unstable fitted model can make cross-validation losses depend strongly on shared observations. The learning rule and data both matter.

This is also why the standard deviation across a few fold scores, divided by the square root of K, is not automatically a valid standard error for the CV estimate. A neat interval drawn as “mean ± 1.96 fold standard error” can substantially misstate uncertainty when dependence and training variation are ignored. Bengio and Grandvalet's [primary result](https://www.jmlr.org/papers/volume5/grandvalet04a/grandvalet04a.pdf) rules out a universal unbiased variance estimator based on the usual K-fold error measurements across all distributions. It does not say that uncertainty analysis is impossible under additional assumptions.

Report the actual folds, sizes and scores, and call their spread a descriptive spread. Use an uncertainty method matched to the unit, sampling assumptions and estimand when an interval is required. For a fixed model assessed on genuinely independent future cases, uncertainty in its mean loss is a simpler conditional problem than uncertainty in retraining and selecting a new model. Repeating partitions of the same data cannot replace collecting new independent units.

The later [Bias–Variance Tradeoff & Learning Curves](/learn/path/full-curriculum/bias-variance-tradeoff-learning-curves?module=classical-ml) decomposes variation in model predictions over repeated training samples. That is distinct from spread among CV fold scores and from one observed train/validation gap.

### Bootstrap and analytic criteria answer related questions

The bootstrap draws a new sample of n rows **with replacement** from the observed dataset, then recomputes an estimator. One particular row is absent from a bootstrap sample with probability $(1-1/n)^n$, approaching $e^{-1}\approx0.368$. Thus a bootstrap sample contains about 63.2% distinct original rows on average, despite having n sampled positions. This differs from K-fold's sampling without replacement and its exactly-once validation partition.

Bootstrap estimates of an estimator's variability and out-of-bag prediction assessment have different constructions. A question about the variability of a complete selected pipeline may require repeating selection, not merely resampling its final predictions. Grouped data require resampling meaningful units; time dependence requires another design. A naïve bootstrap of rows is not a universal solution to the dependence problem above. The familiar .632 error estimator combines apparent and out-of-bag error with specific weights; it is a particular estimator with limitations, not a consequence that every bootstrap score should be multiplied by .632.

Analytic criteria such as AIC, BIC and complexity penalties are other ways to compare models under specified statistical assumptions. They are not equivalent to one another or guaranteed substitutes for the evaluation question above. The next [Regularization lesson](/learn/path/full-curriculum/regularization-l1-l2-elastic-net-dropout?module=classical-ml) compares predictive AIC, evidence-oriented BIC and minimum description length with their assumptions; this lesson's core contribution is to make data use and selection explicit. The canonical [Elements of Statistical Learning front matter](https://link.springer.com/content/pdf/bfm:978-0-387-84858-7/1) lists these neighboring chapter-7 branches and the conditional-versus-expected-error distinction; it is a section map, not a claim that the linked front matter contains those derivations.

## 8. Deeper branch: adaptive search and spending resources

### Search can learn where to look next

Bayesian or sequential model-based optimization fits a **surrogate**: a cheaper model of the relation between candidate settings and their observed validation outcomes. An **acquisition rule** uses that model to choose the next costly evaluation. After the actual evaluation, the history is updated and the surrogate is refitted.

For minimization, **expected improvement** at candidate λ is

\[
\operatorname{EI}(\lambda)
=\mathbb E_{Y\mid\lambda,\text{history}}
[\max(\ell_{\text{best}}-Y,0)].
\]

The random variable Y here describes uncertainty in the surrogate's belief about a candidate outcome; it is not the target label. If the current best loss is 0.20, a candidate believed certain to achieve 0.18 has EI 0.02. Another candidate with equal believed probabilities of loss 0.05 and 0.45 has mean loss 0.25 but EI $0.5(0.20-0.05)=0.075$. A larger possible improvement can justify an uncertain trial even when its mean prediction is worse.

These are constructed beliefs for understanding the acquisition rule, not fitted results from a real optimizer. The quality of a surrogate and its uncertainty matters. Adaptive search can waste effort or overfit a noisy validation criterion; there is no universal trial count after which it beats random search.

**Figure 6 — Expected loss and expected improvement are different summaries.** Show the incumbent line at 0.20, candidate A's point mass at 0.18, and candidate B's two outcomes at 0.05/0.45 with probability one-half each. Shade only positive improvement below the incumbent, then calculate the two EIs. A text probability table makes the calculation exact without a fabricated smooth posterior curve.

### What TPE models

A Gaussian-process surrogate commonly models loss conditional on settings. The **tree-structured Parzen estimator** instead separates observed settings into a better-loss group and the remaining group, fits densities $l(\lambda)$ and $g(\lambda)$, and seeks settings likely under the better group relative to the other. In its original construction, expected improvement is proportional to

\[
\left[\gamma+(1-\gamma)\frac{g(\lambda)}{l(\lambda)}\right]^{-1},
\]

where γ is the probability mass assigned to the better-loss group. This motivates seeking a high $l/g$ ratio. It is not the same as fitting one Gaussian process, and its tree structure can express conditional choices such as parameters for an optional second layer. The [original TPE paper](https://papers.nips.cc/paper_files/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf) derives the relation and describes the density construction.

For an optional practical extension, save this as `cv_optuna.py` beside the complete `cv_penguins.py` and CSV above. It deliberately imports the already defined loader and full preprocessing pipeline. Install Optuna 5.0.0 in addition to that program's packages. This supplementary program was checked against the current API documentation but **not executed during the content phase**; exact best settings and scores are intentionally not invented.

```python
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
from cv_penguins import load_data, make_model

X, y = load_data()
splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=73)
splits = list(splitter.split(X, y))

def objective(trial):
    model = make_model()
    model.set_params(
        classify__n_neighbors=trial.suggest_int("neighbors", 1, 21, step=2),
        classify__p=trial.suggest_int("distance_power", 1, 2),
        classify__weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
    )
    scores = cross_val_score(
        model, X, y, cv=splits, scoring="accuracy", n_jobs=1,
        error_score="raise",
    )
    return float(scores.mean())

study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=29, n_startup_trials=10),
)
study.optimize(objective, n_trials=20, n_jobs=1)
print("selection score", study.best_value)
print("selected settings", study.best_params)
```

`distance_power=1` uses absolute-coordinate differences in the Minkowski metric; 2 gives Euclidean distance. Uniform neighbor weights count neighbors equally; distance weighting gives nearer ones greater influence according to the estimator's rule. These choices change the search space from the earlier six-candidate grid. Comparing their selected scores alone would therefore not be a controlled claim that one search algorithm is superior.

The program performs development search only. To assess this adaptive recipe, put a new study entirely inside each outer training set, or use a separately reserved final assessment set. A fixed sampler seed improves reproducibility of a sequential run; distributed completion order and implementation versions can still matter. [Optuna's TPE documentation](https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html) specifies its startup and sampling behavior.

### Successive halving: spend more only after an initial comparison

Sometimes we can assess a candidate cheaply with fewer training rows or fewer optimization steps. Successive halving starts many candidates at a small budget, retains a fraction, and gives survivors larger budgets. With nine candidates, budgets 10, 30 and 90, and survival factor 3, a simple schedule is:

| Stage | Candidates assessed | Budget per candidate | Total nominal resource |
| --- | ---: | ---: | ---: |
| 1 | 9 | 10 | 90 |
| 2 | 3 | 30 | 90 |
| 3 | 1 | 90 | 90 |

If each stage retrains from scratch, this costs 270 resource units, compared with 810 to give all nine candidates 90 units. If training genuinely resumes from saved state, incremental cost can instead be $9\cdot10+3\cdot20+1\cdot60=210$. Treating every resource unit as equal wall time is an additional approximation. Three-fold evaluation would repeat corresponding fits; software overhead and refitting the winner still need a budget.

The key risk is **early ranking**, not a requirement that every validation curve be monotone. A candidate that starts slowly may eventually win. In a constructed example, A has losses `[.30,.25,.24]` at budgets `[10,30,90]`, while B has `[.40,.20,.10]`. Eliminating B after budget 10 discards the eventual winner. Both curves improve monotonically; monotonic improvement alone did not make the ranking safe.

**Investigation 4 — Allocate a finite training budget.** Predict the survivor, then edit candidate loss trajectories and choose the first comparison budget. Reveal only outcomes that the chosen schedule paid to observe. A separate after-run explanation can compare the hidden full-budget winner, clearly identified as hindsight in this constructed environment. Edit a late value that was never consulted: the elimination decision stays unchanged, even if hindsight regret changes.

**Hyperband** runs multiple halving schedules, called brackets, with different initial candidate counts and starting budgets. This explores the tradeoff between examining many candidates briefly and fewer candidates more thoroughly. It is not merely a second name for one successive-halving run. The [primary Hyperband paper](https://www.jmlr.org/papers/volume18/16-558/16-558.pdf) explains the distinction and the assumptions behind its analysis.

Scikit-learn's halving search remains experimental. A resource can be sample count or an eligible estimator parameter; it cannot simultaneously be a searched parameter in the same grid. The final stage need not reach the nominal maximum resource because the schedule depends on candidate count, factor and starting budget. Inspect `n_resources_`, `n_candidates_` and `cv_results_` rather than assuming every finalist received all data. The [halving guide](https://scikit-learn.org/stable/modules/grid_search.html#searching-for-optimal-parameters-with-successive-halving) documents these contracts. Native full-state resumption should not be assumed just because the high-level schedule has increasing budgets.

## 9. Run a useful comparison within an honest budget

For G candidates and K folds, a grid makes GK candidate fits, plus one final refit if enabled. A nested plan with O outer folds and I inner folds makes $O(GI+1)$ fits for assessment, plus $GI+1$ if the same inner configuration is then searched and refitted on all development data. This counts fits; inner, outer and full-data fits have different training sizes and costs.

Our six-candidate, three-by-three experiment uses $3(6\cdot3+1)=57$ pipeline fits for outer assessment and 19 for final selection/refit, totaling 76. The three majority baselines are separate cheap fits. This is small enough to inspect serially. The displayed main script does not need all CPU cores for a useful lesson.

Parallelism reduces wall time only within available compute and memory. Parallelizing both folds and an estimator's own native threads can oversubscribe a machine. Large worker pools may duplicate datasets or create too many pending jobs. Begin with a bounded worker count, record actual fit times, and decide whether to parallelize trials or model internals. `pre_dispatch` and pipeline caching can help in appropriate cases; caching is useful only when repeated transformations genuinely share inputs and parameters. Cluster schedulers and database-backed studies extend this idea, but do not remove validation or resource-accounting requirements.

There is no universal threshold such as “Bayesian search is better once one fit takes ten seconds.” Measure the real cost of fitting, scoring, surrogate updates and scheduling for your task. A small grid, a documented random budget, or a representative holdout can be better justified than an elaborate search whose results cannot be assessed reliably.

When two candidates are close, inspect paired results on the same splits and the size of the practical difference. Choosing the highest average is a selection rule, not automatically a collection of independent hypothesis tests. Overlapping fold intervals do not form a valid general significance test, and a multiple-testing correction does not repair dependent or incorrectly constructed evidence. If a formal comparison is required, choose an inferential procedure whose assumptions match the paired data and complete search history.

If a fit fails, keep that failure visible. Invalid settings, insufficient minority examples, numerical problems and unavailable features call for different fixes. Silently removing failures or replacing them with convenient scores changes the selection procedure. During this small lesson `error_score="raise"` exposes failures immediately; a large search may log failed trials and follow a predeclared handling rule.

## 10. Practice: identify the decision before computing the score

Try the first six using only the core route. The remaining questions extend the deeper branches.

### 1. Unequal folds

Three folds assess 4, 3, 3 rows and get 3, 1, 2 correct. Find the unweighted mean fold accuracy and pooled accuracy. Which gives each row equal weight?

<details><summary>Hint</summary>Average the three fractions for one answer; add correct counts before division for the other.</details>
<details><summary>Solution</summary>The fold mean is $(3/4+1/3+2/3)/3=7/12\approx.5833$. Pooled accuracy is 6/10=.6 and gives each row equal weight. Equal weighting of folds is a different declared summary.</details>

### 2. A remainder is still a learner's data

A splitter uses `fold_size = n // k` and slices exactly that many rows for each of k validation folds. What happens at n=11,k=3? How does the provided splitter repair it?

<details><summary>Solution</summary>Only nine rows receive a validation turn; two are omitted. Depending on how training indices are constructed, those omitted rows may be permanently in training or dropped altogether. `np.array_split` creates folds 4, 4, 3 so every row belongs to one validation fold and the remaining folds form its training set.</details>

### 3. New patient or known patient?

A wearable model will predict tomorrow's measurements for people who already provided a week of history. A second product must work on entirely new wearers. Describe an assessment boundary for each. What extra question arises if the second product also launches in a future season?

<details><summary>Solution</summary>The first task can use each person's available earlier history but must respect prediction time and target availability. The second needs held-out people. A future season introduces a time-distribution boundary as well, so a custom group-and-time assessment may be needed. A random visit split does not by itself establish either claimed setting.</details>

### 4. An impossible out-of-fold array

A forward plan trains on rows 0–3 and assesses 4–5, then trains 0–5 and assesses 6–7. Why can it be used for fold scoring but not passed directly to `cross_val_predict` on all eight rows? What should a manual prediction table contain at rows 0–3?

<details><summary>Solution</summary>Rows 0–3 never appear in a held-out set, so the required exactly-once partition is missing. A manual table should retain their predictions as absent, not fit on them and label in-sample outputs out-of-fold. Score the valid held-out rows or train a stacking stage only where legitimate predictions exist.</details>

### 5. A new candidate pattern

Validation labels are `[0,1,0,1]`. Initially the candidates predict all zeros or all ones. Add a candidate predicting `[0,1,0,1]`. Under the independent fair-label model, what changes in best validation accuracy and expected future accuracy? What if you add only another all-zero candidate?

<details><summary>Solution</summary>The best validation score rises from .5 to 1. Every fixed prediction remains independent of future fair labels, so expected future accuracy remains .5. A duplicate all-zero candidate changes neither the original best validation score nor future accuracy. Candidate diversity and how selection uses the labels matter.</details>

### 6. Which rows selected the epoch?

An outer-fold model chooses its number of epochs using outer assessment loss, then reports accuracy on that same outer fold. Identify the violated boundary and give two repairs.

<details><summary>Solution</summary>The assessed rows selected a training setting. Move stopping into the outer training data, either using a stopping subset inside each inner training partition or treating inner-validation-based stopping as part of the complete rule assessed by protected outer rows. Another valid design uses development data for all such choices and a genuinely separate final test set. Renaming the used assessment set does not restore independence.</details>

### 7. Probability mass is not score distance — deeper

A satisfactory parameter region has probability .02 under your sampler. How many independent draws give at least 95% chance of hitting it? Would this guarantee a score within 2% of the global optimum?

<details><summary>Hint</summary>Solve $(1-.02)^T\le.05$ and round upward.</details>
<details><summary>Solution</summary>$T\ge\log(.05)/\log(.98)\approx148.28$, so 149 draws suffice under the assumed independent sampling model. The 2% is sampling mass, not score proximity. The satisfactory region itself must be defined and its assumed mass justified for a practical guarantee.</details>

### 8. An unchanged acquisition value — deeper

The incumbent loss is .30. Candidate C has equal predicted probabilities of losses .10 and .50. Find EI. If only the worse outcome changes from .50 to .90, does EI change? Does expected loss change?

<details><summary>Solution</summary>EI is $.5(.30-.10)=.10$. The worse outcome contributes zero improvement in both cases, so EI remains .10. Expected loss changes from .30 to .50. This does not prove the surrogate is calibrated; it distinguishes the summaries of its stated distribution.</details>

### 9. Account for the whole schedule — deeper

A halving plan starts 27 candidates with budgets 5, 15, 45, 135 and retains one-third after each stage. Find the nominal from-scratch resource cost, and compare it with giving every candidate 135. If each stage instead resumes genuine saved state, what is the incremental resource cost?

<details><summary>Solution</summary>Candidate counts 27, 9, 3, 1 each consume 135 nominal units per stage, totaling 540, versus 3,645 for all candidates at 135. Resuming costs $27\cdot5+9\cdot10+3\cdot30+1\cdot90=405$. Multiply appropriate evaluations by fold count and account separately for scoring/refits. Real time need not be linear in this resource.</details>

### 10. A fixed rule tests an overlap story — deeper

A classifier always predicts class 0 and ignores training. Labels on n observations are independent fair bits. What is the variance of its leave-one-out accuracy? Why does this contradict equating loss correlation with training overlap?

<details><summary>Solution</summary>The correctness indicators are independent Bernoulli(.5), so their average has variance $(.5)(.5)/n=1/(4n)$. The leave-one-out training sets overlap heavily, but those sets do not influence this rule's predictions. Thus overlap alone does not determine loss correlation or imply variance stays near 1/4.</details>

## 11. Readiness and what follows

You are ready to continue when you can build a complete fold assignment, identify what each score was allowed to influence, place learned preprocessing inside that assignment, select a split matching a concrete future use, and explain why a selected inner score differs from protected assessment evidence. You do not need to memorize a preferred number of folds or implement a Bayesian optimizer to demonstrate those core skills.

The next topic is [Regularization: L1, L2, Elastic Net & Dropout](/learn/path/full-curriculum/regularization-l1-l2-elastic-net-dropout?module=classical-ml). It asks what a penalty changes about a fitted model. This lesson supplies the procedure for choosing penalty strength without confusing the score that selected it with an untouched assessment. Later feature selection and AutoML reuse the same boundary around increasingly broad choices.

## References and other ways to learn

- [Scikit-learn cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html): current splitters, multiple metrics, prediction-table contracts and structured-data choices. Use its diagrams to inspect what a splitter actually assigns, then check that assignment against your task.
- [Visualizing cross-validation behavior](https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html): an inspected visual/code alternative showing class, group and training/test stripes together. Compare `KFold`, `GroupKFold` and `TimeSeriesSplit`; explain the different information boundaries before changing a model.
- [Scikit-learn parameter-search guide](https://scikit-learn.org/stable/modules/grid_search.html): grids, random distributions, halving schedules, nested parameter names and selection/assessment separation. Its result fields are useful for making a reproducible search record.
- [Cawley and Talbot, On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://www.jmlr.org/papers/volume11/cawley10a/cawley10a.pdf): primary study of overfitting the criterion used to choose a model. Read its contrast between expected and particular-sample selection curves; do not copy its benchmark magnitudes as universal effects.
- [Bengio and Grandvalet, No Unbiased Estimator of the Variance of K-Fold Cross-Validation](https://www.jmlr.org/papers/volume5/grandvalet04a/grandvalet04a.pdf): deeper primary reading on the distinction between a fitted model's prediction error and error averaged over training samples, and on dependence in CV uncertainty.
- [Bergstra and Bengio, Random Search for Hyper-Parameter Optimization](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf): the coordinate-projection argument and original experiments motivating random search as a strong baseline.
- [Bergstra and colleagues, Algorithms for Hyper-Parameter Optimization](https://papers.nips.cc/paper_files/paper/2011/file/86e8f7ab32cfd12577bc2619bc635690-Paper.pdf): sequential model-based search, expected improvement and the original TPE derivation. Read sections 2–4 after the acquisition example rather than treating their notation as a first exposure.
- [Li and colleagues, Hyperband](https://www.jmlr.org/papers/volume18/16-558/16-558.pdf): sections 3.1–3.2 show why one halving bracket and Hyperband are different, and why early resource allocation can eliminate a slow starter.
- [Palmer Penguins project](https://allisonhorst.github.io/palmerpenguins/): the real input and measurement context used in the worked program; the accompanying provenance preserves its license, byte hash and exact experimental split.
