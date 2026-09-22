# Evaluation Metrics: Decisions, Rankings, Probabilities, and Numerical Errors

A model flags twenty items for review. Fifteen need attention and five do not. It also misses ten items that needed attention. Is the model good?

There is no single answer until we say what the review system is supposed to accomplish. A team trying to avoid unnecessary inspections asks a different question from a team trying to miss as few faults as possible. A system that only orders a queue faces a different evaluation problem from one that must report reliable probabilities.

**An evaluation metric is a rule for turning predictions and observed outcomes into a number that answers a particular question.** Its formula defines which successes count, which errors matter, and how examples are weighted. A number can be correctly calculated while answering the wrong question.

In the Active Learning lesson, we compared models by the number correct on held-out data. Here we unpack that choice and build several alternatives from the individual examples upward. You will learn to calculate them, inspect their failures, choose a decision threshold, and write an evaluation report that a reader can interpret.

The core route is sections 1–6, followed by the measured experiment in section 8. Section 7 develops retrieval metrics for ordered results; section 9 extends the ideas to aggregation, computation, and uncertainty. Those branches can be read when the corresponding application becomes useful.

## 1. Write down what one prediction means

Before calculating a score, state the **evaluation unit**: one image, transaction, document, person, experimental run, or query. Define the target, when the prediction is made, and which errors the resulting action can cause. Keep model fitting, development decisions, and final evaluation separate. We will revisit the whole task-design process in ML Problem Formulation; these local choices are enough to start here.

Four common questions require different inputs:

| Question | Prediction supplied to the metric | Examples |
|---|---|---|
| Did the system make the right discrete decisions? | Predicted class or action | Precision, recall, F1, accuracy, cost |
| Did it put useful items ahead of less useful ones? | Unthresholded score or ordered list | ROC-AUC, AP, NDCG |
| Do its numerical probabilities forecast outcomes well? | Probability distribution | Log loss, Brier loss |
| How far is a numerical prediction from the observation? | Predicted value in target units | MAE, RMSE, R² |

**[Inline figure: one prediction, several questions.]** An input produces a score. One branch preserves the score for ranking; another applies a threshold to produce an action. A probability branch is labeled as requiring a probability interpretation, rather than treating any score as a probability. Continuous-value prediction has its own residual branch. Show the corresponding outcome used to evaluate each branch.

For binary classification, choose the class of interest and call it **positive**. This is a naming convention, not a moral judgment or necessarily the more common class. In an inspection example it could mean “requires review.” A score s is larger when the model ranks that class as more likely. In this lesson, the decision rule is **predict positive when s ≥ t**. Equality matters when scores tie.

## 2. Build the confusion matrix by moving individual items

Use these eight constructed inspection items. Their scores are supplied for learning; they are not measured results from a trained inspection model.

| Item | Observed class y | Score s |
|---|---:|---:|
| A | 1 | 0.95 |
| B | 0 | 0.80 |
| C | 1 | 0.80 |
| D | 1 | 0.60 |
| E | 0 | 0.50 |
| F | 1 | 0.30 |
| G | 0 | 0.20 |
| H | 0 | 0.10 |

At threshold t = 0.8, A, B, and C are flagged. A and C are **true positives**: positive decisions on actually positive items. B is a **false positive**. D and F are **false negatives**, because they needed attention and were not flagged. E, G, and H are **true negatives**.

| Actual class ↓ / Predicted class → | Negative | Positive |
|---|---:|---:|
| Negative | TN = 3 | FP = 1 |
| Positive | FN = 2 | TP = 2 |

The orientation is printed deliberately. Libraries and publications can choose different row or column conventions. A confusion matrix without axis labels is easy to misread.

**[Inline figure: cards crossing a threshold.]** Put the eight named cards on a score ruler, with B and C at the same position. A threshold gate divides predicted positives from negatives. Route the cards into the four matrix cells according to their observed classes. The matrix should retain the item IDs as well as counts, so every number can be traced to an example.

Now ask precise questions:

- **Precision:** Of the items we flagged, how many actually needed attention? Here 2/(2 + 1) = 2/3.
- **Recall**, also called sensitivity or true positive rate: Of the items that needed attention, how many did we flag? Here 2/(2 + 2) = 1/2.
- **False positive rate:** Of the actually negative items, how many did we flag unnecessarily? Here 1/(1 + 3) = 1/4.
- **Specificity:** Of the actually negative items, how many did we correctly leave alone? Here 3/4 = 1 − FPR.
- **Accuracy:** Of all eight decisions, how many were correct? Here (2 + 3)/8 = 5/8.

The denominators carry the meaning. Precision conditions on **the model's positive decisions**. Recall conditions on **the observed positive class**. False positive rate conditions on **the observed negative class**. FPR is not the fraction of alerts that are wrong; that quantity is 1 − precision when any alerts exist.

For the opening example—fifteen true alerts, five false alerts, and ten missed positives—precision is 15/20 = 0.75 and recall is 15/25 = 0.6. We cannot determine accuracy or FPR without knowing how many true negatives there were. A report should not invent missing counts.

### Combine precision and recall only when that combination is useful

The F1 score is their harmonic mean:

\[
F_1=\frac{2PR}{P+R}=\frac{2TP}{2TP+FP+FN}.
\]

At t = 0.8, F1 = 4/(4 + 1 + 2) = 4/7 ≈ 0.571429. The arithmetic average of precision and recall would be about 0.583333. F1 gives a lower result when one component is substantially weaker than the other.

The generalized score is

\[
F_\beta=\frac{(1+\beta^2)TP}{(1+\beta^2)TP+FP+\beta^2FN},\quad\beta>0.
\]

Larger β emphasizes recall, with β² appearing in the formula. This is a chosen metric tradeoff. It is not equivalent in general to saying that a false negative has exactly β² times the operational cost of a false positive. If those costs are known, calculate them directly.

F1 ignores true negatives. This can be helpful for a retrieval task with many irrelevant items, but it also means two systems with the same TP, FP, and FN receive the same F1 even if their true-negative populations differ enormously. It is not a universal replacement for accuracy.

### Empty denominators carry information

Move the threshold above 0.95 so nothing is flagged. Precision is 0/0: there are no positive decisions whose correctness can be estimated. Recall is 0/4 = 0, F1 is 0, and accuracy is 4/8 = 0.5.

Show precision as **undefined: no predicted positives**, rather than adding a tiny epsilon and pretending it was measured as zero. An API may offer a chosen zero-division convention for aggregation; report that convention. If the evaluation set contains no actual positives, recall itself is undefined. A perfect-looking specificity then says nothing about detecting positives.

### Investigation: move the decisions, then explain the metric

Lower t from 0.8 to 0.5 and follow each item into its new cell: TP = 3, FP = 2, FN = 1, TN = 2; precision 0.6 and recall 0.75. Then try t = 0.3: TP = 4, FP = 2, FN = 0, TN = 2; precision 2/3 and recall 1.

Both precision and recall increased in that second move, because the newly admitted item F was positive. Lowering a threshold cannot decrease recall on a fixed dataset, but precision can rise or fall. The common tradeoff is a tendency, not a monotonicity theorem for every empirical PR curve.

Change one item's score or observed label and inspect the recalculated result. Also move the threshold within an interval containing no score, such as from 0.70 to 0.75. No decision changes. This null case distinguishes changing a control from changing the evaluated system's behavior.

## 3. A threshold is a decision policy

Suppose an unnecessary inspection costs one unit and a missed positive costs three. Assume correct decisions have zero cost and the costs do not vary by item. The total cost on our eight examples is

\[
\text{cost}=FP+3FN.
\]

At thresholds 0.8, 0.5, and 0.3 the costs are 7, 5, and 2. The comparison explains what the recall gain buys and what the additional false alerts cost. A different cost matrix can favor a different operating point.

If a model supplies an accurate posterior probability p for the deployment population, the expected costs of acting positive and negative are C_FP(1 − p) and C_FN p. Choose positive when

\[
p\ge\frac{C_{FP}}{C_{FP}+C_{FN}}.
\]

This derivation assumes the stated two-action cost structure, meaningful deployment probabilities, and no additional capacity constraints. For equal costs the cutoff is 0.5 regardless of prevalence, because prevalence is already represented in the posterior. With a review queue limited to a fixed number of items, capacity becomes another decision constraint. After reweighting training classes or changing the population, raw model scores may not have the probability interpretation needed by this formula.

A practical alternative is to select a threshold on development data using the intended cost or constraint, then assess that fixed choice on separate test data. An empirical optimum is an estimate; it can fail to remain optimal on new examples. The measured experiment will show exactly that failure. [scikit-learn, decision-threshold tuning and separate data roles](https://scikit-learn.org/stable/modules/classification_threshold.html).

### Rare positives change the workload behind a small rate

Imagine a classifier with TPR = 0.8 and FPR = 0.1 in two populations, holding the class-conditional score behavior fixed. Among 10,000 items at 50% prevalence, it produces 4,000 true positives and 500 false positives: precision 4,000/4,500 ≈ 0.888889.

At 1% prevalence, the same rates give 80 true positives and 990 false positives: precision 80/1,070 ≈ 0.074766. The ROC operating point is unchanged; the fraction of alerts that are useful is very different.

In general, with positive prevalence π,

\[
\text{precision}=\frac{\text{TPR}\,\pi}
{\text{TPR}\,\pi+\text{FPR}(1-\pi)}.
\]

**[Inline figure: the same rates in two populations.]** Use two population-count diagrams and an alert tray for each. Annotate actual TP and FP counts, not just percentages. Changing prevalence alone leaves the two class-conditional detection rates fixed; explicitly label that assumption.

This is why precision-recall views often make rare-event workload easier to understand. It is not a reason to invent a cutoff such as “below 10% prevalence, ROC is invalid.” ROC still measures separation between classes; it does not directly express alert precision or operating cost. [Saito and Rehmsmeier, controlled class-imbalance comparisons](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432).

## 4. Read the whole score ordering: ROC, AUC, and AP

A threshold answers what happens at one operating point. A curve shows the decisions available from a **fixed score vector** as the threshold changes. Ranking scores need not lie between zero and one: a classifier's real-valued decision score can be used directly.

To construct the curve, sort scores from largest to smallest. Begin above the maximum, predicting nothing positive. Lower the threshold through each **distinct** score. Include an entire equal-score group together; no threshold can include B at 0.8 while excluding C at 0.8 under our rule.

For the eight-item example:

| Threshold | TP | FP | Recall / TPR | FPR | Precision |
|---|---:|---:|---:|---:|---:|
| Above 0.95 | 0 | 0 | 0 | 0 | Undefined |
| 0.95 | 1 | 0 | 0.25 | 0 | 1 |
| 0.80 | 2 | 1 | 0.50 | 0.25 | 2/3 |
| 0.60 | 3 | 1 | 0.75 | 0.25 | 0.75 |
| 0.50 | 3 | 2 | 0.75 | 0.50 | 0.60 |
| 0.30 | 4 | 2 | 1 | 0.50 | 2/3 |
| 0.20 | 4 | 3 | 1 | 0.75 | 4/7 |
| 0.10 | 4 | 4 | 1 | 1 | 0.50 |

**[Inline figure: one sorted list, two curve views.]** A highlighted equal-score block in the ranked list updates a linked ROC point (FPR, TPR) and PR point (recall, precision). Show the exact counts in the caption. This makes the curves consequences of the same decisions, rather than unrelated decorative shapes.

### ROC-AUC is a pairwise ranking score

The ROC curve plots TPR against FPR. Its area has an especially useful finite-sample interpretation: compare every positive-negative pair. Award 1 if the positive has the higher score, 0 if it has the lower score, and 1/2 for a tie. Average those contributions:

\[
\mathrm{AUC}=\frac{1}{n_+n_-}
\sum_{i:y_i=1}\sum_{j:y_j=0}
\left(\mathbf1[s_i>s_j]+\tfrac12\mathbf1[s_i=s_j]\right).
\]

Our four positives and four negatives give sixteen pairs, totaling 12.5 credits. AUC = 12.5/16 = 0.78125. The same result comes from trapezoidal area between the grouped ROC points. The straight segment across a tied block reflects the half-credit convention; intermediate points are not different deterministic thresholds that separate equal scores.

**[Inline figure: a positive-negative pair grid.]** Rows are A, C, D, F; columns B, E, G, H. Each cell shows the two scores and its 1, 1/2, or 0 contribution. The single tied pair C/B is explicit. Add the credits before dividing by sixteen.

AUC = 0.5 can arise from constant scores, random ordering in expectation, or other score distributions whose pair contributions happen to balance. It does not prove that the positive and negative score distributions are identical. AUC also cannot tell us a model's performance at one particular alert capacity. [Fawcett, ROC construction, relative scores, and area interpretation](https://ctsilva.github.io/2025-VisML-CSE/refs/Fawcett_2006_Introduction_ROC_Analysis.pdf).

### Average Precision rewards where recall is gained

The PR curve plots precision against recall. **Average Precision**, in the non-interpolated convention used here and by scikit-learn, is

\[
\mathrm{AP}=\sum_k(R_k-R_{k-1})P_k,
\]

with thresholds ordered so recall increases. Each term gives credit for newly recovered positives, weighted by the precision after including that score group. In our example, recall increases four times by 1/4:

\[
\mathrm{AP}=\tfrac14\left(1+\tfrac23+\tfrac34+\tfrac23\right)
=0.770833\ldots.
\]

The zero-recall plotting sentinel sometimes uses precision 1 to anchor a graph; it is not evidence that an empty alert set has measured precision 1.

AP is **not generally equal to trapezoidal area under the PR points**. That area is 0.79375 for this same example. State the integration convention whenever reporting “PR-AUC.” Linear interpolation of precision differs from interpolation of underlying confusion counts; libraries and benchmarks may also use different interpolated AP conventions. [scikit-learn AP definition and interpolation distinction](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).

For a constant score and an evaluation set containing both classes, grouped AP equals the observed positive fraction and ROC-AUC is 0.5. This is an exact useful null. The horizontal prevalence line is also a population reference for an uninformative ranker. Do not claim that every finite random permutation has AP exactly equal to prevalence.

### Better area does not mean a better curve everywhere

Two rankings of the same eight entities, three positive, can produce these descending label sequences:

| Ranker | Labels in score order, 1 = positive | ROC-AUC | AP |
|---|---|---:|---:|
| A | 0, 1, 1, 0, 1, 0, 0, 0 | 0.733333 | 0.588889 |
| B | 1, 1, 0, 0, 0, 0, 0, 1 | 0.666667 | 0.791667 |

A has the better ROC area; B has the better AP and the better first two results. Neither area alone establishes which should operate under a particular review budget. A theorem about one **entire curve dominating another** is a stronger statement than an inequality between their areas. The Davis–Goadrich paper explicitly distinguishes these facts; it does not say that improving ROC-AUC must improve PR area. [Davis and Goadrich, the dominance theorem and area distinction](https://mark.goadrich.com/articles/davisgoadrichcamera2.pdf).

### Investigation: rebuild the curve from the ranked items

Start with the eight cards and inspect the effect of admitting the tied 0.8 block on the next points. Inspect the pair grid and AP contributions. Swap the display order of B and C while keeping their scores unchanged: both metrics must remain unchanged.

Then edit one score so a formerly tied positive moves below the negative, and inspect which pair contributions and threshold blocks change. Finally set all scores equal. The curve should retain only the all-negative and all-positive endpoints, AUC should be 0.5, and AP should be 0.5 for this balanced fixture. If you remove all positives, the interface must explain which metrics are no longer defined instead of drawing a fabricated ROC curve.

## 5. Evaluate probabilities without confusing them with decisions

A score can rank examples well while its numeric values are poor probabilities. To see the difference, keep the eight labels and replace every score p with p². On [0,1], squaring is strictly increasing, so it preserves the ordering and ties. ROC-AUC and AP stay exactly the same. But the numbers offered as event probabilities change.

Two standard probability losses are binary log loss and Brier loss:

\[
L_{\log}=-\frac1n\sum_i[y_i\ln p_i+(1-y_i)\ln(1-p_i)],
\qquad
L_{\mathrm{Brier}}=\frac1n\sum_i(p_i-y_i)^2.
\]

For a positive item, log loss uses −ln p; for a negative item it uses −ln(1 − p). It measures how much probability the forecast assigned to the outcome that happened. A wrong certain forecast has infinite mathematical log loss. Numerical libraries clip or otherwise stabilize boundary values; their finite output should not erase that conceptual penalty.

| Probability vector | ROC-AUC | AP | Log loss, nats | Binary Brier loss |
|---|---:|---:|---:|---:|
| Original p | 0.781250 | 0.770833 | 0.577541 | 0.204063 |
| Squared p² | 0.781250 | 0.770833 | 0.667335 | 0.231326 |
| Constant 0.5 | 0.500000 | 0.500000 | 0.693147 | 0.250000 |

These are computed losses on constructed forecasts, not a fitted calibration experiment. Squaring worsened both losses here; it need not always do so.

**[Inline figure: same order, different probability penalties.]** Connect each original probability to its squared value on a second ruler. For a selected positive and negative item, show its contribution to each loss. Keep the unchanged pair ordering visible beside the changed loss bars.

Changing **only the decision threshold** leaves log loss and Brier loss unchanged, because neither formula uses the threshold. A plot of “log loss versus threshold” is flat when the probabilities and outcomes are fixed. Passing hard 0/1 decisions instead changes the forecasts being evaluated and can create severe log penalties; it is not probability evaluation of the original model.

These losses assess overall probabilistic forecasting quality. They do not isolate calibration from discrimination or resolution. Calibration asks whether observed event frequencies agree with predictions in a specified population; a lower Brier or log loss alone does not prove every group is calibrated. The dedicated Calibration & Conformal Prediction topic will build that distinction carefully.

For C mutually exclusive classes, log loss is −ln of the probability assigned to the observed class. Multiclass Brier loss is often the sum of squared errors across all C probabilities; its range is [0,2]. The conventional binary formula above has range [0,1]. State which normalization you use; scikit-learn's default currently rescales only the binary case. [Official Brier API and normalization](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html).

## 6. Numerical predictions: see the residual before averaging it

Now imagine predicting how long five tasks will take, in minutes. Let y be the observed duration and ŷ the prediction. The **residual** is y − ŷ: positive means the prediction was too low.

| Task | Actual y | Prediction A | Prediction B |
|---|---:|---:|---:|
| 1 | 1 | 1 | −1 |
| 2 | 2 | 2 | 0 |
| 3 | 3 | 3 | 1 |
| 4 | 4 | 4 | 2 |
| 5 | 10 | 4 | 8 |

These are intentionally supplied predictions, including B's physically invalid negative duration for task 1. A metric can calculate a number for that output; the application must separately enforce or examine valid prediction ranges. The example isolates how errors are aggregated.

Model A has residuals [0, 0, 0, 0, 6]. Model B has [2, 2, 2, 2, 2]. Which is preferable depends on whether one large miss is worse than five moderate misses.

\[
\mathrm{MAE}=\frac1n\sum_i|y_i-\hat y_i|,
\quad
\mathrm{MSE}=\frac1n\sum_i(y_i-\hat y_i)^2,
\quad
\mathrm{RMSE}=\sqrt{\mathrm{MSE}}.
\]

| Prediction | MAE, minutes | MSE, minutes² | RMSE, minutes | Median absolute error, minutes |
|---|---:|---:|---:|---:|
| A | 1.2 | 7.2 | 2.683282 | 0 |
| B | 2 | 4 | 2 | 2 |

MAE prefers A; MSE and RMSE prefer B. MAE grows linearly with the size of a residual and MSE quadratically. That makes MAE less sensitive to a large residual, not immune to one: a sufficiently large single error can dominate its mean too. Median absolute error can hide severe errors affecting a minority of cases, so inspect the distribution and relevant tails.

**[Inline figure: residual lengths and squared areas.]** For each task, connect prediction to observation with a labeled segment. Alongside it draw a square whose area is the squared residual, using a clearly stated scale. A's one large square and B's five moderate squares explain the score reversal. A table retains exact lengths and areas.

This choice also changes what a model should estimate. Under expected squared error, the optimal point forecast is the conditional mean; under expected absolute error, a conditional median is optimal. A forecast of a high quantile needs a quantile-consistent loss such as pinball loss. The scoring rule should match the requested quantity. [Gneiting, point forecasts and their scoring functions](https://arxiv.org/abs/0912.0902).

### R² compares squared error with a specific reference

For a nonconstant evaluation target with at least two observations,

\[
R^2=1-\frac{\sum_i(y_i-\hat y_i)^2}
{\sum_i(y_i-\bar y_{\mathrm{evaluation}})^2}.
\]

Our evaluation mean is 4 and the denominator is 9 + 4 + 1 + 0 + 36 = 50 minutes². A's squared-error sum is 36, so R² = 1 − 36/50 = 0.28. B's sum is 20, so R² = 0.6.

The reference in this formula is the **mean of the target values being evaluated**, including the test mean when calculating test R². It is a mathematical reference, not a usable prediction procedure that was allowed to inspect future outcomes.

A real constant baseline can be fitted on the training data. If that training mean were 3, its predictions [3,3,3,3,3] have squared-error sum 55 on our evaluation targets, yielding R² = −0.1. A training-mean baseline therefore need not have test R² = 0. Reporting its actual test error remains useful alongside R².

Negative R² is valid: squared error exceeds that evaluation-mean reference. It is a reason to investigate, not a diagnosis that proves overfitting. Poor model specification, a narrow target range, population change, or an unsuitable constant prediction can also produce it. Nonnegative training R² is guaranteed for an exact ordinary least-squares fit with an intercept on its fitting data, not for every possible trained model.

If all observed targets are equal, the denominator is zero. Our teaching implementation reports R² as undefined and retains the absolute errors. scikit-learn offers a `force_finite` convention that replaces some nonfinite results with 1 or 0; that convenience is distinct from the raw formula. [Official R² definition and constant-target behavior](https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score-the-coefficient-of-determination).

### Investigation: decide which error pattern your score rewards

Compare the predictors using the residual contributions to MAE and RMSE. Edit one target and one prediction to create your own reversal, and explain it using lengths and squared areas. Switch the displayed unit from minutes to seconds: MAE and RMSE multiply by 60, MSE by 3,600, while R² is unchanged.

Then copy the targets exactly into the prediction column. All residuals vanish; MAE and RMSE are zero and R² is 1 for these nonconstant targets. Finally make every target and prediction 4. The errors remain zero, but raw R² becomes undefined. A successful lab should explain that denominator change rather than label the perfect constant prediction a failure.

### Relative errors need a meaningful denominator

MAPE averages |y − ŷ|/|y|, usually multiplied by 100 when expressed as a percentage. It emphasizes errors relative to the target's magnitude. It is undefined at y = 0 and unstable near zero. Silently dropping zero targets changes the evaluation population and can hide the very cases that matter.

The symmetric variant using (|y| + |ŷ|)/2 in the denominator still has a zero-over-zero case when both are zero and has its own weighting behavior. A log transform changes the target scale and decision question; it is not a universal repair. Choose a metric, valid target domain, baseline, and zero handling that match the application before comparing numbers.

## 7. Ordered results: reward the right part of the list

Search and recommendation systems often return a list per query. A user seeking one answer may care mainly about the first useful result; a researcher gathering many relevant documents may care about retrieving the whole relevant set.

Consider five documents in the displayed order with relevance grades [0, 3, 1, 2, 0]. Grades above zero count as relevant for binary metrics; the magnitudes express graded usefulness for NDCG. These grades are constructed judgments, not click measurements.

**Precision@3** is 2/3 because two of the first three documents are relevant. **Recall@3** is 2/3 because the declared candidate set contains three relevant documents overall. In a real retrieval collection, incomplete relevance judgments can make that denominator uncertain.

The first relevant document is at rank 2, so reciprocal rank is 1/2. **Mean Reciprocal Rank** averages that value over queries, with a stated convention for queries with no retrieved relevant result. It concentrates on the first success.

For this complete five-document list with no score ties, AP is the mean of precision at the three relevant ranks:

\[
\mathrm{AP}=\frac{1/2+2/3+3/4}{3}=0.638889\ldots.
\]

**Mean Average Precision** averages each query's AP. Equal weighting of queries differs from pooling all their documents into one global ranking. Nonretrieved relevant documents contribute no retrieval credit; restricting the judged candidate set must not silently redefine “all relevant.” [Stanford's information-retrieval text, ranked evaluation](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html).

### NDCG uses a gain and a position discount

Choose the gain function g(rel) = 2^rel − 1 for this example. The gains are [0, 7, 1, 3, 0]. Discount rank r by log₂(r + 1), and sum through the cutoff K:

\[
\mathrm{DCG@K}=\sum_{r=1}^K\frac{2^{\mathrm{rel}_r}-1}{\log_2(r+1)}.
\]

At K = 3, DCG = 0 + 7/log₂3 + 1/log₂4 = 4.916508. The ideal order of the same judged candidates starts with grades 3, 2, 1, giving ideal DCG 9.392789. Divide actual by ideal: NDCG@3 ≈ 0.523434.

**[Inline figure: a ranked shelf with discounted gains.]** Show five named document cards in order, their grade and gain, the rank discount, and each contribution. Below, show the ideal shelf built from the same candidates. Align the actual and ideal sums before dividing. A move changes the contributions visibly.

Move the grade-3 document to the first position, swapping it with the first grade-0 document. Precision@3 stays 2/3; reciprocal rank becomes 1; NDCG@3 rises to 7.5/9.392789 ≈ 0.798485. The set of top-three relevant documents did not change, but their order did.

### Investigation: change the list without changing the candidates

Swap two document cards and follow the changes to AP, reciprocal rank, precision at the cutoff, and NDCG contributions. Then change a relevance grade and compare the updated contributions. Swapping two equally relevant documents is a null for these relevance-based calculations: identities move but the grade sequence does not.

If every grade is zero, ideal DCG is zero and the ratio is undefined. An evaluation package may assign zero by convention; report the convention and query count. Do not silently drop hard queries from an average.

There is more than one DCG gain convention. scikit-learn uses the supplied relevance values as gains. Passing [0,3,1,2,0] directly yields a linear-gain NDCG@3 of about 0.502491. To reproduce our exponential-gain calculation, pass 2^rel − 1 as its relevance array. Tied predicted scores require a declared policy; its default averages over ties, while our hand example specifies a concrete order. [Official NDCG API and tie handling](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html).

## 8. Run a measured evaluation and preserve an inconvenient result

We will use the same openly licensed Banknote Authentication subset as the preceding lessons. This time all 320 designated training rows have their labels available. The task is to predict the source's class code from the predeclared variance and entropy features. Using two features produces an inspectable, imperfect baseline; it is not a claim that the other recorded features should be discarded for a real system. Class 1 is the positive class by explicit choice, with no unverified genuine/forged mapping. [UCI data description and CC BY 4.0 license](https://archive.ics.uci.edu/dataset/267/banknote+authentication).

Download [the offline CSV](banknote-subset.csv), [the complete evaluation program](banknote-evaluation.py), and [the complete metric-calculation program](metrics-calculations.py) into one directory. The first program fits a model and records actual held-out predictions; the second constructs the instructional examples, computes grouped curves from scratch, checks them against scikit-learn, and handles the null cases.

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install numpy==2.3.5 scikit-learn==1.9.1
.venv\Scripts\python.exe metrics-calculations.py
.venv\Scripts\python.exe banknote-evaluation.py
```

On macOS or Linux:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install numpy==2.3.5 scikit-learn==1.9.1
.venv/bin/python metrics-calculations.py
.venv/bin/python banknote-evaluation.py
```

Author runs used Python 3.12.14. The programs use local data and write JSON results beside themselves. No remote service or model download is needed.

### First fix the comparison procedure

Fit `Pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=500))` using the 320 training rows. The scaler learns only from those rows and transforms development and test data with the same parameters. Independently fitting another scaler on the test inputs would change the feature representation supplied to the trained model.

Use the next 80 rows as development data. Assign a false positive cost of one unit and a false negative cost of five units, with zero cost for correct decisions. These are **illustrative costs for a class-code decision**, not economic estimates for currency authentication.

Sweep the distinct development scores plus the no-alert option. Choose the threshold minimizing FP + 5FN, breaking equal-cost ties by the highest threshold. Freeze the fitted model and selected threshold. Only then use the final 80 test rows to produce the report. The program also reports a fixed 0.5 threshold as a predeclared comparison and a constant training-prior baseline.

The selected development threshold is 0.1201300081. It yields TP = 39, FP = 18, FN = 0, TN = 23 on development data, for a cost of 18. On the held-out test data:

| Decision rule | TP | FP | FN | TN | Precision | Recall | Cost FP + 5FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed threshold 0.5 | 25 | 4 | 4 | 47 | 0.862069 | 0.862069 | 24 |
| Development-selected threshold | 28 | 21 | 1 | 30 | 0.571429 | 0.965517 | 26 |
| Constant training-prior score, threshold 0.5 | 0 | 0 | 29 | 51 | Undefined | 0 | 145 |
| Constant training-prior score, cost-rule threshold 1/6 | 29 | 51 | 0 | 0 | 0.3625 | 1 | 51 |

The selected threshold catches three more positive items but creates seventeen more false alerts. At the stated costs, that changes test cost by 17 − 5×3 = 2 units, from 24 to 26. The development-selected threshold therefore **did not improve the test cost** over the fixed 0.5 comparison. Keep that result. Selecting a development optimum does not make its estimated advantage certain to generalize.

The constant baseline's probability is the class-1 fraction in the training data. Its cost-rule threshold is derived from the fixed cost matrix, not selected from test labels. Reporting it alongside the 0.5 baseline prevents an unfair cost comparison against only a deliberately unsuitable all-negative action.

The two logistic-regression thresholds use identical test probabilities, so both have ROC-AUC 0.956728, AP 0.945489, log loss 0.252139, and binary Brier loss 0.073915. Their different decisions do not alter those probability or ranking scores. The constant baseline has AUC 0.5 and AP 29/80 = 0.3625.

**[Inline figure: a threshold choice crossing the development/test boundary.]** Development cost points identify the frozen threshold. A separate test panel compares the two confusion matrices and the seventeen added false alerts versus three recovered positives. Beside them, show the shared ranking and probability scores once, labeled as unchanged. Use the actual stored scores and counts, not a stylized curve.

A suitable report says what was trained, which features and split were used, how the threshold was chosen, the resulting held-out counts, and the failure of the estimated cost gain to persist. It does not retune on the same test set to manufacture a winning threshold. Further changes belong in a new development cycle with an appropriate final evaluation.

## 9. Extend the calculation without losing its meaning

### Multiclass: which classes receive weight?

For a single-label three-class task, consider this constructed matrix with actual classes as rows and predicted classes as columns:

| Actual ↓ / Predicted → | 0 | 1 | 2 | Support |
|---|---:|---:|---:|---:|
| 0 | 8 | 1 | 1 | 10 |
| 1 | 2 | 2 | 0 | 4 |
| 2 | 1 | 1 | 0 | 2 |

Treat each class in turn as positive against the rest. The per-class F1 values are 16/21 ≈ 0.761905, 0.5, and 0. Class 2 is never correctly predicted, despite the overall accuracy being 10/16 = 0.625.

**Macro F1** gives each class equal weight: (16/21 + 0.5 + 0)/3 ≈ 0.420635. **Support-weighted F1** weights by actual class counts: (10×16/21 + 4×0.5 + 2×0)/16 ≈ 0.601190. **Micro F1** first pools TP, FP, and FN; for an exhaustive single-label multiclass task, every wrong prediction adds one FP and one FN, so micro precision, recall, and F1 all equal accuracy, here 0.625.

Macro F1 is the average of class F1 values, not generally the F1 of macro precision and macro recall. Weighted F1 has no universal guarantee of lying between macro F1 and micro F1. Choose the aggregation to express the intended class weighting, and include per-class counts where an aggregate might hide failure.

**Balanced accuracy** averages class recalls; in binary classification it is (TPR + specificity)/2. **Matthews correlation coefficient** summarizes association between binary predictions and labels using all four cells:

\[
\mathrm{MCC}=\frac{TP\,TN-FP\,FN}
{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}.
\]

It ranges from −1 to 1 when the denominator is nonzero. These are useful additional summaries, not substitutes for a cost or a denominator report. Constant predictions can make the raw MCC denominator zero; document an API's convention if using one.

With multilabel outputs, an example can have several true labels. Exact-set accuracy penalizes any mismatch in its label set; Hamming loss counts individual wrong label decisions. A micro F1 identity from exhaustive single-label classification must not be assumed to hold there. Likewise, multiclass AUC requires specifying one-vs-rest or one-vs-one and the averaging rule. If scores are probability columns, preserve their class-label order.

### Aggregate counts, not arbitrary batch averages

Fixed-threshold confusion counts add across batches. Sum TP, FP, FN, and TN first, then compute precision or F1. Averaging batch F1 values generally gives another number because F1 is a ratio. The complete calculation program includes `StreamingBinaryCounts` and checks that three chunks of our eight-row fixture match the full calculation exactly.

MAE and MSE also admit running sums and counts. R² needs the evaluation target's centered sum of squares; a stable online variance algorithm avoids subtracting two nearly equal large sums. Exact ROC/AP on arbitrary real-valued scores ordinarily needs global ordering or an equivalent ordered representation. A histogram can approximate that ordering, but its binning and error are part of the approximation. No fixed reservoir size guarantees the same accuracy for all prevalences and score distributions.

For predicted single class labels, constructing a sparse confusion tally is O(n) in the number of examples; computing every one-vs-rest prediction matrix is not necessary. Sorting a score vector for grouped ROC/AP takes O(n log n), followed by a linear sweep. Fixed-threshold counts and residual sums are linear. Match the representation and calculation to the metric rather than assume all metrics require sorting or a dense class-by-class matrix.

### A precise score can still be an uncertain estimate

Reporting 0.965517 recall does not mean knowing population recall to six decimal places; here it is only 28/29 positive test items. Preserve numerator and denominator. When comparing two fixed models on the same evaluation items, compare paired errors so the shared examples remain paired. A confidence interval or resampling procedure must reflect the evaluation unit and dependence: several images of one person are not automatically independent trials, and time dependence does not disappear because rows were shuffled.

State whether an interval describes a fixed fitted model on new observations, variation across training runs, or the entire model-selection procedure. These are different sources of uncertainty. Small subgroups and multiple inspected slices require especially careful interpretation. The next topic, PAC Learning & VC Dimension, studies generalization under explicit probabilistic assumptions; the time-series and task-formulation lessons will revisit evaluation design for dependent and operational data.

## 10. Practice: calculate, diagnose, and defend the choice

### 1. A rate is not a count

A system has TP = 12, FP = 8, FN = 3, TN = 77. Calculate precision, recall, FPR, and the fraction of alerts that are false. Explain why the last two differ.

<details><summary>Hint</summary>The false-positive-rate denominator is all actual negatives. The false-alert-fraction denominator is all predicted positives.</details>
<details><summary>Solution</summary>Precision = 12/20 = 0.6; recall = 12/15 = 0.8; FPR = 8/85 ≈ 0.094118; false-alert fraction = 8/20 = 0.4. Both false quantities use the same eight errors, but compare them with different populations.</details>

### 2. A tied ranking

Two positives have scores [0.7, 0.4]; two negatives have [0.7, 0.1]. Calculate ROC-AUC by pairs and AP by grouped score thresholds. Then exchange the displayed order of the two 0.7 items.

<details><summary>Hint</summary>The first threshold group contains one positive and one negative together. Pair ties receive half credit.</details>
<details><summary>Solution</summary>Pair credits are [0.5,1] for positive0.7 and [0,1] for positive0.4, totaling2.5/4 = 0.625 AUC. At threshold0.7 recall rises to0.5 with precision0.5; at0.4 recall rises to1 with precision2/3. AP =0.5×0.5+0.5×2/3 =7/12≈0.583333. Reordering equal-score items must change neither result.</details>

### 3. Perfect ranking, questionable probabilities

The observed labels are [0,1] and reported probabilities are [0.49,0.51]. Another forecaster reports [0.1,0.9]. Compare their AUC, AP, binary Brier loss, and log loss. Does better probability loss here prove population calibration?

<details><summary>Hint</summary>Both positive scores exceed their respective negative scores. For each forecaster, its two correct-class probabilities are equal.</details>
<details><summary>Solution</summary>Both AUC and AP are1. The first Brier loss is0.49²=0.2401 and log loss−ln0.51≈0.673345. The second Brier loss is0.1²=0.01 and log loss−ln0.9≈0.105361. These two observations favor the second forecast under both losses, but do not establish population or subgroup calibration. Moving a threshold while retaining either probability vector leaves its probability losses unchanged.</details>

### 4. The R² baseline trap

Observed evaluation values are [2,4,6]. A training-mean baseline always predicts3. Calculate its R². Is the statement “a training-mean baseline must score zero” correct?

<details><summary>Hint</summary>The evaluation mean is4. Use it only in the denominator reference.</details>
<details><summary>Solution</summary>SSE =(2−3)²+(4−3)²+(6−3)²=11. SST =(2−4)²+0+(6−4)²=8. R²=1−11/8=−0.375. The training mean is an honest fitted baseline, but the formula's zero-score reference is the evaluation mean. They need not coincide.</details>

### 5. Two identical top-three sets

Compare relevance orders [0,2,1] and [2,0,1], treating positive grades as relevant. Which of precision@3, reciprocal rank, and exponential-gain NDCG@3 must change?

<details><summary>Hint</summary>The set of relevant documents is the same, but the first success and rank discounts differ.</details>
<details><summary>Solution</summary>Precision@3 remains2/3. Reciprocal rank rises from1/2 to1. With gains[0,3,1], DCG rises from3/log₂3+1/2 to3+1/2, while ideal DCG remains3+1/log₂3. NDCG therefore rises. The intended user behavior determines whether that ordering difference matters.</details>

### 6. Which model should enter an inspection queue?

Model A has higher ROC-AUC; Model B has higher AP. The team can inspect only fifty items per day. What evidence would you request before choosing?

<details><summary>Hint</summary>An area summarizes thresholds the team may never use.</details>
<details><summary>Solution</summary>Compare the number of useful items among the first fifty, missed-positive rates and costs at the actual capacity, the representative data population and labeling process, uncertainty in the paired difference, operational latency, and behavior in important slices. Use development evidence to select the policy and a separate final assessment. Neither area proves one curve dominates at the relevant operating point, so the given information alone does not determine a deployment choice.</details>

### 7. Verify a streaming result

Use `StreamingBinaryCounts` from the complete calculation program on a changed six-row dataset of your own, divided into two unequal batches. Predict whether the result will equal a single full-batch calculation. Compare pooled F1 with the unweighted mean of the two batch F1 values.

<details><summary>Hint</summary>Keep the threshold fixed. Choose batches with different error counts so the ratio issue is visible.</details>
<details><summary>Solution and criteria</summary>For example, use labels [1,0,1,0,1,0], scores [0.9,0.2,0.7,0.8,0.1,0.05], threshold 0.5, and batches of the first two and last four rows. The first batch has TP = 1, TN = 1, FP = FN = 0, giving F1 = 1. The second has one of each outcome, giving F1 = 0.5. Their unweighted mean is 0.75. Pooled counts are TP = 2, FP = 1, FN = 1, TN = 2, giving F1 = 2/3, exactly matching the full-batch calculation. Counts add; ratios with different denominators generally do not. A complete answer supplies its own six labels/scores, threshold, both matrices, pooled result, batch average and explanation; an undefined batch denominator needs an explicit convention.</details>

### 8. Report the actual experiment honestly

Write a short result paragraph from the banknote experiment. Include the decision objective, the threshold-selection data, the test counts that explain the cost difference, and one limit on the conclusion. Do not choose a new threshold from the test curve.

<details><summary>Hint</summary>The selected threshold improves recall but increases the reported test cost relative to0.5.</details>
<details><summary>Example solution</summary>“We fitted a standardized two-feature logistic regression on 320 training rows and selected a threshold on 80 development rows to minimize FP + 5FN. On 80 held-out test rows, the selected threshold recovered 28 of 29 positives with 21 false alerts; the fixed 0.5 comparison recovered 25 with 4 false alerts. The selected rule's cost was 26 versus 24, so its development advantage did not persist. The result describes this fixed dataset/split and illustrative cost model, not a guaranteed improvement in another population.” A stronger practical study would quantify uncertainty at the appropriate independent unit and examine the deployment population before choosing a final policy.</details>

## 11. References and another route through the material

For a visual alternative after the confusion-matrix section, [StatQuest's ROC and AUC video](https://www.youtube.com/watch?v=4jRBRDbJemM) walks from threshold choices to curve construction. Its published description includes a correction to the confusion matrix shown at12:00: TP3,FP2,FN1,TN2. Use the corrected counts and reconstruct our tied example afterward. The creator's [video index](https://statquest.org/video-index/) links related introductions to confusion matrices and sensitivity/specificity.

For shorter practice on denominators, use [Google's classification-metrics lesson](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall). For a deeper explanation of ROC geometry and relative scores, read [Fawcett's paper](https://ctsilva.github.io/2025-VisML-CSE/refs/Fawcett_2006_Introduction_ROC_Analysis.pdf). Pair it with [Davis and Goadrich](https://mark.goadrich.com/articles/davisgoadrichcamera2.pdf) when reasoning about dominance and interpolation; their distinction prevents the incorrect conclusion that every AUC improvement improves AP.

The [scikit-learn metrics guide](https://scikit-learn.org/stable/modules/model_evaluation.html) is an implementation reference for averaging, undefined cases, prediction types, and regression definitions. Its `scoring` API uses a “larger is better” convention, so names such as `neg_mean_squared_error` negate a loss; a negative scorer output is not a negative mathematical MSE.

For the optional branches, [Stanford's retrieval-evaluation chapter](https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-ranked-retrieval-results-1.html) develops query-level ranking measures, and [Gneiting's point-forecast paper](https://arxiv.org/abs/0912.0902) explains why choosing a mean, median, or quantile must influence its evaluation loss. The [offline-data provenance](data-provenance.md) documents exactly which observations and transformations generated our experiment.

Continue next to **PAC Learning & VC Dimension**. Metrics specify what counts as error; learning theory asks what observed errors can tell us about future errors under explicit assumptions about data, model classes, and sample size.
