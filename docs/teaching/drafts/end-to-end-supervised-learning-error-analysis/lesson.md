# End-to-End Supervised Learning & Error Analysis

A classifier is useful only as part of a chain: someone asks a question, measurements arrive, a fitted transformation turns them into inputs, a model makes a prediction, and someone acts on the result. A good score in a notebook does not tell you whether that chain answers the original question.

Here we build one small, complete study. Given chemical measurements of a wine specimen, can we identify its recorded cultivar? We will compare a simple model with two deliberate alternatives, inspect which development examples change, and finish with a held-out report. The interesting result is that an additional measurement helps overall while making one subgroup worse. That is the kind of tradeoff an average can hide.

**First pass:** read §§1–7, run the offline study in §4, and do practice 1–4 in §9. Follow the specimen-flow figure and the development-error investigation as they appear. Return to §8 for deeper uncertainty, selective prediction and iteration design, then try practice 5–6. Allow about 45–60 minutes to read the core and another 60–90 minutes to reproduce and extend the study.

## 1. Write the question before choosing the model

The preceding [Time-Series Validation & Forecasting Baselines](/learn/path/full-curriculum/time-series-validation-forecasting-baselines?module=classical-ml) lesson treated a prediction as an origin and future horizon. Here the unit changes: one row is one wine specimen. The dataset's row order groups cultivars; it is not an observation timestamp. Copying a time split merely because a table has an order would change the experiment for the wrong reason.

Our data are the UCI Wine collection, chemical analyses of 178 wines from three cultivars grown in the same Italian region. There are 13 recorded measurement columns. The lesson supplies all rows in [wine.csv](wine.csv), with an added specimen ID and cultivar codes 0, 1 and 2. These codes are names, not an ordering of wine quality. [Dataset and attribution](https://doi.org/10.24432/C5PC7J).

Write an **experiment contract**: the short statement of what information is available, what prediction is wanted, and what evidence would justify a choice.

| Contract item | This study |
| --- | --- |
| Prediction unit | One measured specimen |
| Target | Its recorded cultivar, among three known classes |
| Initial inputs | Alcohol and color intensity |
| Candidate added input | Flavanoids, measured before classification |
| Primary selection metric | Validation balanced accuracy: mean recall across the three cultivars |
| Supporting diagnostics | Overall accuracy, log loss, confusion matrix and development slices |
| Training rows | 106 specimens, used to fit coefficients and preprocessing |
| Validation rows | 36 specimens, used for the declared comparison and error analysis |
| Test rows | 36 specimens, used after the candidate is frozen |
| Final model protocol | Evaluate the selected already-fitted model; do not refit on validation in this study |

The first two features are a teaching constraint, chosen to make a measurement-versus-model comparison inspectable. We do not have assay-cost data, so we do not claim they are the cheapest measurements. The three-feature candidate asks whether more informative input can help more than making a decision rule more flexible.

**What this dataset can support.** This is a retrospective classification exercise within a small historical collection. The file does not give enough sampling, winery, repeated-specimen or time metadata to establish independent future-vintage performance. A deployment study would need those fields and a split matched to its users and future measurements. The random stratified split below preserves class representation for this limited exercise; it does not create missing population evidence.

## 2. Keep three different kinds of learning separate

Training changes the model's fitted state. Validation changes the researcher's choices. Testing estimates the performance of a choice already made. Information can move through a human as easily as through a function call.

An example makes the distinction concrete. Suppose we calculate the mean alcohol concentration using all 178 rows, scale everything, and then separate training and test sets. The test specimens have already influenced the coordinate system the model sees. The proper sequence is to choose the rows first, calculate the training mean and standard deviation, and apply those same values to the other rows. `Pipeline` keeps those fitted operations attached to the model. [scikit-learn's preprocessing and leakage examples](https://scikit-learn.org/stable/common_pitfalls.html).

For a feature \(x\), training estimates \(\mu_{\mathrm{train}}\) and \(s_{\mathrm{train}}\). Every later specimen is represented as

\[
z=\frac{x-\mu_{\mathrm{train}}}{s_{\mathrm{train}}}.
\]

The transformation of a validation specimen uses its own measured \(x\), but not a newly fitted validation mean. Learning a coordinate system and applying that coordinate system are different operations.

**Figure A — where information is allowed to go.** The three specimen sets occupy separate lanes. Training reaches both the scaler's stored statistics and the model's parameters. Validation receives their predictions and reaches the candidate-selection decision. The locked test lane reaches only the final report. The fitted scaler and classifier travel to inference together; the target label takes a separate path into the evaluator.

This picture also explains why hiding the test target from `fit` is insufficient. If a researcher reads test errors, adds a feature to repair them, and reports the improved score on those same rows, the test has become development data. Continuing to work is reasonable; calling that revised score an untouched final estimate is the mistake. Use a new holdout or a nested outer evaluation for the next confirmatory comparison.

Before training, check the table at the level that affects this contract: row identity, label meaning, feature units, duplicates or repeated entities, missing values, impossible values, and which fields exist when predictions will be made. In our supplied file all 13 measurements are finite and the ID is excluded from features. The ID preserves traceability; it carries no chemical meaning.

## 3. Choose a baseline that answers a specific doubt

A baseline is a comparison with a job. The **majority baseline** asks whether measured inputs help beyond predicting the most frequent training cultivar. **Multinomial logistic regression** asks how well weighted sums of the initial measurements separate the classes. A **small random forest** asks whether more flexible combinations of those same two measurements help. Finally, the **three-feature linear model** asks whether the additional flavanoid measurement helps while keeping the model family fixed. These candidates are declared together before the comparison; the later slice analysis interprets their differences.

Do not change features, data splits and model settings all at once and then attribute a difference to one of them. Our two interpretable comparisons are:

| Comparison | Changed | Held fixed |
| --- | --- | --- |
| Two-feature linear → two-feature forest | Function class and declared fitting settings | Specimens, measurements, primary metric |
| Two-feature linear → three-feature linear | One measured feature | Specimens, classifier family, `C=1`, scaling rule |

For logistic regression, each class has a score \(a_k=w_k^Tz+b_k\). The softmax transformation converts these three scores into positive numbers summing to one:

\[
p_k=\frac{e^{a_k}}{\sum_{j=0}^{2}e^{a_j}}.
\]

The predicted cultivar is the class with greatest \(p_k\). The fitting objective rewards probability assigned to the observed class, with regularization to restrain coefficients. A forest instead pools class probabilities from many trees. Its `max_depth=4` and `min_samples_leaf=3` are declared controls on this comparison, not settings discovered by searching this test set.

**Refresh the measures.** If class 0 has 12 validation specimens and 9 are correctly classified, its recall is \(9/12=0.75\). Balanced accuracy averages the recalls of the three classes, so a class with fewer rows still contributes one third. Overall accuracy gives each specimen equal weight. Log loss averages \(-\log p_{\text{actual}}\), in natural-log units called nats: assigning 0.8 to the true class costs about 0.223 nats; assigning 0.2 costs about 1.609. It distinguishes probabilities even when the winning class remains unchanged. These quantities answer different questions; select one primary measure before comparing candidates.

## 4. Run a complete offline study

Download the supplied CSV beside a file named `wine_study.py`. The CSV already contains the input, so running the example does not download a dataset. A compatible author snapshot used Python 3.12.14, NumPy 2.3.5 and scikit-learn 1.9.1. In your own environment:

```sh
python -m pip install numpy==2.3.5 scikit-learn==1.9.1
python wine_study.py
```

The complete program below exposes the research sequence. `test` is an index list until the final block. The code selects the winning candidate by validation balanced accuracy; ties use the stated candidate order. That deterministic tie rule is part of the protocol.

```python
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, log_loss, confusion_matrix
)

data = np.genfromtxt(Path(__file__).with_name("wine.csv"),
                     delimiter=",", names=True)
two = np.column_stack([data["alcohol"], data["color_intensity"]])
three = np.column_stack([two, data["flavanoids"]])
y = data["cultivar"].astype(int)
ids = np.arange(len(y))
development, test = train_test_split(
    ids, test_size=36, stratify=y, random_state=21
)
train, valid = train_test_split(
    development, test_size=36, stratify=y[development], random_state=22
)

def linear_model():
    return make_pipeline(
        StandardScaler(), LogisticRegression(C=1, max_iter=2000)
    )

candidates = {
    "majority": (DummyClassifier(strategy="prior"), two),
    "linear_two": (linear_model(), two),
    "forest_two": (RandomForestClassifier(
        n_estimators=100, max_depth=4, min_samples_leaf=3,
        random_state=21, n_jobs=1
    ), two),
    "linear_three": (linear_model(), three),
}
validation_predictions = {}
scores = {}
print("split sizes", len(train), len(valid), len(test))
for name, (model, matrix) in candidates.items():
    model.fit(matrix[train], y[train])
    pred = model.predict(matrix[valid])
    prob = model.predict_proba(matrix[valid])
    validation_predictions[name] = pred
    scores[name] = balanced_accuracy_score(y[valid], pred)
    print(name, round(scores[name], 6),
          round(accuracy_score(y[valid], pred), 6),
          round(log_loss(y[valid], prob), 6))

reference = validation_predictions["linear_two"]
for name in ["forest_two", "linear_three"]:
    pred = validation_predictions[name]
    fixed = np.sum((reference != y[valid]) & (pred == y[valid]))
    broken = np.sum((reference == y[valid]) & (pred != y[valid]))
    print("paired", name, int(fixed), int(broken))
for name in ["linear_two", "linear_three"]:
    wrong = validation_predictions[name] != y[valid]
    for lower in [True, False]:
        mask = (two[valid, 1] < 4) if lower else (two[valid, 1] >= 4)
        print("slice", name, "color<4" if lower else "color>=4",
              int(mask.sum()), int(wrong[mask].sum()))

eligible = ["linear_two", "forest_two", "linear_three"]
selected = max(eligible, key=lambda name: scores[name])
model, matrix = candidates[selected]
final_prediction = model.predict(matrix[test])
final_probability = model.predict_proba(matrix[test])
print("selected", selected)
print("test", round(balanced_accuracy_score(y[test], final_prediction), 6),
      round(accuracy_score(y[test], final_prediction), 6),
      round(log_loss(y[test], final_probability), 6))
print(confusion_matrix(y[test], final_prediction))
```

The bounded author calculation executed these same fitted models, inputs and split settings; it saved full probabilities and specimen-level development results in the packet. The expected rounded outputs are:

```text
split sizes 106 36 36
majority 0.333333 0.388889 1.090022
linear_two 0.792857 0.805556 0.522805
forest_two 0.820635 0.833333 0.511293
linear_three 0.888889 0.888889 0.218008
paired forest_two 1 0
paired linear_three 4 1
slice linear_two color<4 17 2
slice linear_two color>=4 19 5
slice linear_three color<4 17 3
slice linear_three color>=4 19 1
selected linear_three
test 0.966667 0.972222 0.128708
[[12  0  0]
 [ 0 14  0]
 [ 0  1  9]]
```

The three numbers after each candidate and after the final test label are balanced accuracy, accuracy and log loss. The full program should be rerun during implementation when converting its output into the website's runnable-example presentation.

Now interpret the result rather than stopping at the printout. Both measured-input models improve greatly over the majority predictor. The forest improves the two-feature linear model by one validation specimen. Adding flavanoids fixes four previously wrong predictions and breaks one previously correct prediction. That is a net gain of three specimens, from 29/36 to 32/36 correct. The improvement is not just a higher aggregate: we can name the changed cases and investigate them.

**Figure D — a net gain contains two directions.** The same validation specimens occupy matched before/after positions. Four move from wrong to correct and one moves from correct to wrong. Keeping their IDs aligned makes the arithmetic visible: four repairs minus one new error gives three additional correct predictions.

## 5. Turn errors into testable hypotheses

An **error slice** is a defined subset of examples: one cultivar, a measurement range, one acquisition device, or a missingness pattern. Its purpose is to find a coherent failure that suggests an action. Always show the denominator. “Three errors” means something different among five cases and among five hundred.

For our two-feature linear model, the validation confusion matrix is:

| Actual class / predicted class | Class 0 | Class 1 | Class 2 | Support |
| --- | ---: | ---: | ---: | ---: |
| Actual 0 | 9 | 3 | 0 | 12 |
| Actual 1 | 0 | 13 | 1 | 14 |
| Actual 2 | 2 | 1 | 7 | 10 |

Its balanced accuracy is \((9/12+13/14+7/10)/3=0.792857\). Notice that the confusion matrix and the score summarize the **same predictions** through different aggregations. You can reconstruct one from the other only if you retain enough counts; a single average loses the pattern.

After adding flavanoids, every validation specimen of classes 1 and 2 is correct, but class 0 has four errors rather than three. The mean of recalls improves to \((8/12+14/14+10/10)/3=8/9\). A model can improve balanced accuracy and still worsen one class. Whether that is acceptable depends on the contract, including the consequences for the affected class.

The color-intensity split tells a related story:

| Validation slice | Support | Two-feature linear errors | Three-feature linear errors |
| --- | ---: | ---: | ---: |
| Color intensity < 4 | 17 | 2 | 3 |
| Color intensity ≥ 4 | 19 | 5 | 1 |

This threshold is an explicit exploratory diagnostic, not a known biological discontinuity. We keep it fixed when comparing the candidates. Searching hundreds of cutoffs until one looks dramatic would turn chance variation into a story.

**Investigation B — follow specimens, not just scores.** The validation specimens appear in alcohol-versus-color coordinates, with actual class encoded by shape and prediction mismatch by an outline. Changing the candidate keeps specimen positions and IDs fixed. Select a color-intensity cutoff of your own, record whether the added measurement will increase, decrease or preserve the error count inside that slice, then compare. A linked table shows IDs, actual/predicted class, flavanoid value and the probability of the actual class. Test rows are never part of this development explorer.

What could explain a persistent error? There are several distinguishable hypotheses:

| Observed pattern | Plausible hypothesis | Next discriminating action |
| --- | --- | --- |
| Both training and validation are poor | Features or function class do not separate the target; optimization or data handling may also be wrong | Inspect a tiny known case, verify transformation and target alignment, then compare a controlled model/feature change |
| Training good, validation much worse | Generalization gap, split mismatch or dependence | Check the unit/split, examine training-size curves and regularization under a fixed validation protocol |
| One acquisition group consistently worse | Measurement or population mismatch | Inspect group metadata and raw records; acquire representative validation data |
| Confident mistakes cluster around suspect labels | Label convention or annotation issue, among other possibilities | Review original records without replacing labels merely to agree with the model |
| A new feature fixes many cases but breaks a coherent subgroup | Useful information with a changed decision surface | Quantify both directions and investigate the subgroup's decision costs |

These are hypotheses, not diagnoses made from one chart. A useful experiment changes a cause and specifies an expected observation. “Try a bigger model” is a choice; “test whether nonlinear boundaries in the same two measurements reduce class-0/class-1 confusion” is a hypothesis.

## 6. Freeze a decision and write a report someone else can reproduce

Under our declared primary metric, choose `linear_three`. Its fitted scaler and coefficients remain exactly those learned on the 106 training rows. We then open the 36-row test once for the report.

The final confusion matrix contains 35 correct classifications, one class-2 specimen predicted as class 1. Accuracy is \(35/36=0.972222\); balanced accuracy is \((1+1+0.9)/3=0.966667\). The test estimate happens to exceed the validation estimate. A test set is not required to score worse: its cases are different and the estimate varies with sampling.

Would refitting on all 142 development rows be wrong? No, if specified before test evaluation. It would produce a different scaler and different coefficients, whose final test predictions need to be evaluated as that different model. This particular study keeps the training-only fitted model so that its development diagnostics and final report refer to the same artifact. Do not silently switch between those protocols.

A compact final report should say:

> We evaluated cultivar classification within the supplied 178-specimen Wine collection. A fixed stratified 106/36/36 train/validation/test protocol compared a majority baseline, two-feature logistic regression, a bounded two-feature forest and three-feature logistic regression. Validation balanced accuracy selected the three-feature model (0.888889). Its added flavanoid input fixed four and broke one validation predictions relative to the two-feature linear model; the low-color slice worsened from 2/17 to 3/17 errors. The selected training-only fitted pipeline achieved test accuracy 35/36 and balanced accuracy 0.966667. The source file, split IDs, software versions and settings accompany the report. Broader winery or future-vintage performance requires representative metadata and evaluation.

This is a worked reporting example, not a suggested claim about wine production. Keep implementation details that affect reproducibility—input schema, units, class encoding, fitted transformations, model settings, split IDs, seeds and versions—beside the artifact. Model cards provide a useful framework for recording intended use, evaluation and limitations; the source paper also emphasizes evaluation across relevant groups. [Mitchell et al., Model Cards](https://arxiv.org/abs/1810.03993).

## 7. Close the loop without erasing what you learned

Maintain an experiment record with the hypothesis, changed factor, evidence already consumed, result, decision and next action. You do not need a new framework for every run; a small table is enough.

| Experiment | Information used for choice | Result | Decision |
| --- | --- | --- | --- |
| Linear → forest, same two features | Training and validation | One validation error fixed | Useful small gain; compare against added measurement |
| Add flavanoids, same linear family | Training and validation | Four fixed, one broken | Select under balanced accuracy; retain subgroup regression in report |
| Final selected pipeline | Test used for reporting | 35/36 correct | Record the frozen result; do not tune on its one error |

Reproducibility is the ability to regenerate a specified study, not proof that its split answers every future use. Likewise, a fixed seed stabilizes one realization; it does not estimate how results vary across samples. Those are separate goals.

For a real application, the next step after this study would include an inference contract: which named fields are required, how invalid measurements are handled, who receives uncertain results, and how inputs and eventual outcomes will be monitored. A dropped column or a unit change can break the chain even if the classifier's coefficients are untouched. Save and apply the whole fitted pipeline, and validate incoming schema before prediction.

The next module starts with [Perceptrons, Neurons & Activation Functions](/learn/path/full-curriculum/perceptrons-neurons-activation-functions?module=deep-learning-fundamentals). It introduces a richer way to compose learned functions. The experimental discipline stays: begin with the task and baseline, examine the function's behavior, and demand evidence for an improvement. A neural model is an option to investigate, not the automatic winner after a classical model.

## 8. Deeper branch: how strong is the evidence?

Return here after the core study. These extensions develop the same decision chain without changing the frozen result.

### A small test is uncertain, even when the score looks impressive

For \(k\) successes among \(n\) independent Bernoulli trials, a Wilson interval for a proportion uses

\[
c=\frac{\hat p+z^2/(2n)}{1+z^2/n},\qquad
r=\frac{z\sqrt{\hat p(1-\hat p)/n+z^2/(4n^2)}}{1+z^2/n},
\]

and reports \(c\pm r\). With \(k=35,n=36,z=1.96\), it is approximately [0.858, 0.995]. This is a useful scale check for the accuracy estimate under an independent common-probability model. Our stratified three-class sampling fixes class counts, so a formal interval targeted to that design or to balanced accuracy should account for the sampling and class weights. Do not attach this binomial interval to balanced accuracy by renaming its axis. [NIST’s Wilson interval formulas](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm).

For model comparisons, pairing matters. On validation, the three-feature model and two-feature model agree on most rows. The informative changed cases are the four repairs and one new error. Treating two accuracy estimates as independent discards that pairing. A paired bootstrap resamples the same row indices for both models, then recomputes their score difference; dependent specimens would instead require group/block resampling. An interval around a validation-selected winner also does not undo the optimism caused by selection.

### More development experiments consume more development evidence

Repeatedly inspecting validation errors can overfit your choices to those rows. A practical response is to reserve independent outer evaluation, limit comparisons to useful hypotheses, and report the selection process. Nested cross-validation repeats selection inside each outer training portion and evaluates the selected procedure on that outer holdout. It estimates the **procedure**, not just one handpicked fit; its computation and interpretation differ from repeatedly retesting the same favorite model. [Cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html).

### A system can choose to defer

Suppose an invented binary inspection service has 10 equally weighted validation cases. At one declared confidence rule it answers eight and gets seven of those correct. Coverage is \(8/10=0.8\); conditional error among answered cases is \(1/8=0.125\). At a stricter rule it answers four and all are correct: coverage 0.4, observed conditional error 0. A useful comparison must include the cost and quality of the six deferred cases' alternative handling. Plotting accuracy only on answered cases would hide half the work.

**Investigation C — allocate the unanswered cases.** In a ten-case constructed inspection queue, move the acceptance threshold and edit the costs of a wrong answer and a deferred case. Record whether your change will lower, preserve or increase total cost, then inspect both automatic outcomes and the remaining queue. Confidence scores stay attached to the same cases; changing a threshold changes the decision, not the model.

The threshold is another decision chosen on development evidence. Confidence rankings, calibration and acceptance rules are distinct; a high score should not be advertised as a verified probability without calibration evidence. This extension connects the model to a service-level decision and to the earlier [Calibration & Conformal Prediction](/learn/path/full-curriculum/calibration-conformal-prediction?module=classical-ml) lesson.

## 9. Practice: make and defend a decision

### 1. Reconstruct a score from changed counts

A new validation study has this confusion matrix, with rows actual and columns predicted:

\[
\begin{pmatrix}8&2&0\\1&5&0\\0&2&2\end{pmatrix}.
\]

Compute accuracy and balanced accuracy. Which class would you inspect first if the three classes have equal importance?

<details><summary>Hint</summary>

Compute each row's recall before averaging. The total number correct is the diagonal sum.

</details>
<details><summary>Solution</summary>

There are 20 cases and 15 correct, so accuracy is 0.75. Recalls are 0.8, \(5/6\) and 0.5, giving balanced accuracy \(32/45\approx0.711111\). Class 2 has the lowest recall. Its four-case support also means one changed outcome would move that recall by 0.25, so inspect records and uncertainty before announcing a stable subgroup pattern.

</details>

### 2. Decide whether a slice is worth prioritizing

Model A makes 12 errors in 100 cases and model B makes 10. In a five-case subgroup, A makes one error and B makes three. What can you conclude, and what additional question determines your decision?

<details><summary>Hint</summary>

Calculate the errors outside the subgroup and distinguish an empirical comparison from a causal explanation.

</details>
<details><summary>Solution</summary>

Outside the subgroup, A makes 11/95 errors and B 7/95. B gains four there and loses two in the subgroup. It improves aggregate accuracy while worsening the small subgroup. The decision depends on the consequences and the subgroup's relevance, along with uncertainty and whether the subgroup was predeclared or discovered during exploration. A testable follow-up is a new representative sample of that subgroup with a fixed comparison; “B is better everywhere” contradicts the given results.

</details>

### 3. Repair the experiment, not just the code

A team opens test predictions, notices that large purchases fail, adds a purchase-size interaction, and reports the new score on the same test rows. No test labels enter `fit`. Explain what happened and propose a valid continuation.

<details><summary>Hint</summary>

Draw the information arrow from the researcher to the feature decision.

</details>
<details><summary>Solution</summary>

Test outcomes influenced feature selection through the team. Those rows now belong to development history. Keep the result as exploratory evidence and freeze the revised procedure before evaluating on fresh appropriate holdout data; alternatively use a genuinely independent outer evaluation of the selection procedure. Deleting the notebook cell does not erase the information consumed.

</details>

### 4. Guided diagnosis: inspect a different error slice

In the supplied validation predictions, compare both linear models on the actual class-1 specimens rather than the displayed color slices. Reproduce support and errors, then explain why the result can coexist with the low-color regression.

<details><summary>Hint</summary>

Use `(y[valid] == 1)` as the mask. Cultivar and color range are different, overlapping partitions.

</details>
<details><summary>Solution</summary>

Class 1 has 14 validation specimens. The two-feature linear model makes one error and the three-feature model makes none. The low-color slice groups by a feature rather than the label and contains specimens from more than one cultivar. Overlapping subsets need not move in the same direction. Record both definitions so that the figures do not imply one partition is the other.

</details>

### 5. Deeper: assess an acceptance policy

An inspection system answers 60 of 80 cases, making 6 errors among them. A stricter rule answers 40 and makes 2 errors. Calculate coverage and conditional error. If every deferred case costs 2 units of human work and every wrong automatic answer costs 10 units, compare the total observed costs, assuming the human path produces correct answers.

<details><summary>Hint</summary>

Account for both wrong automatic answers and all deferred cases.

</details>
<details><summary>Solution</summary>

The first rule has coverage 0.75 and conditional error 0.10; cost \(6(10)+20(2)=100\). The stricter rule has coverage 0.50 and conditional error 0.05; cost \(2(10)+40(2)=100\). Better conditional accuracy need not improve total cost. The calculation changes if human errors, delay or capacity are included. Use development data to choose the policy before its final evaluation.

</details>

### 6. Deliver your own reproducible study

Before opening any new holdout outcomes, declare one additional candidate using the supplied training/development rows—for example a different fixed regularization strength for the three-feature model. State why the change should help, what remains fixed, and how you will choose. Produce a prediction table with IDs, an aggregate comparison, a paired repair/regression count, and one clearly defined development slice. Write a final paragraph separating evidence already consumed from the independent evidence your next conclusion would require.

<details><summary>Hint</summary>

A useful study may reject the new candidate. Do not choose a new split seed just because it improves the result.

</details>
<details><summary>Evaluation criteria and an acceptable outcome</summary>

The packet should reproduce the candidate and its predictions from named inputs, keep preprocessing fitted on the correct training rows, state the selection measure, and report unfavorable results as well as gains. An acceptable conclusion is “The lower-regularization candidate did not improve the declared development score; we retain the original choice. Because this lesson already disclosed its test result, this extension does not create a new independent test of the modified research process.” A production continuation would reserve new appropriately sampled evaluation data.

</details>

**Core readiness:** you can explain which information each split may change, reproduce the fitted pipeline and score, trace an aggregate improvement to actual changed cases, and write an evidence-matched conclusion. The deeper route adds uncertainty, selection-procedure evaluation and decisions that include deferral costs.

## References & another way to learn it

- [Google Machine Learning Crash Course: dividing datasets](https://developers.google.com/machine-learning/crash-course/overfitting/dividing-datasets) — short worked explanations and check-your-understanding exercises about development/test roles and duplicates. Use before §2 if the information boundary is unfamiliar; page and exercises reviewed.
- [scikit-learn: common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) — runnable wrong/right examples for preprocessing, selection leakage and randomness. Useful after §4; the 1.9.1 documentation was inspected.
- [Goodfellow, Bengio and Courville, Practical Methodology](https://www.deeplearningbook.org/contents/guidelines.html) — the canonical methodological chapter behind this topic's coverage review: metrics, baselines, data needs, tuning, debugging and a complete application. Read after the core. It is a 2016 treatment; its model-default recommendations and use of “test” during tuning need to be interpreted in their historical context, using this lesson's explicit validation/test distinction.
- [The scikit-learn cross-validation guide](https://scikit-learn.org/stable/modules/cross_validation.html) — a deeper reference for evaluating procedures and choosing splitters; bring the preceding module's resampling foundations.
- [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) — the authors' reporting proposal, useful for turning a private experiment into an interpretable handoff. The abstract and stated reporting scope were reviewed; this lesson's report is its own worked example, not a reproduction of the paper's full template.
- [UCI Wine](https://doi.org/10.24432/C5PC7J) — original dataset identity, contributors and license. The supplied extraction and changes are documented in [data-provenance.md](data-provenance.md).

