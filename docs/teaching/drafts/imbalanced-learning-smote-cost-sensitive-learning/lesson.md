# Imbalanced Learning: SMOTE, Cost-Sensitive Learning & Rare-Event Decisions

A laboratory has time to investigate only a small number of candidate proteins. Most belong to common cellular locations; the location of interest is rare. A classifier that labels every protein “not our target” can be accurate most of the time and still contribute nothing to the investigation. Yet flagging everything is not a solution either: it consumes the entire laboratory budget.

The preceding [Bias–Variance & Learning Curves](/learn/path/full-curriculum/bias-variance-tradeoff-learning-curves?module=classical-ml) lesson asked how a fitted model changes with its training data. Here we add another question: **which observations and mistakes should influence the learning procedure, and which action should follow a score?** Changing training data, changing a loss and changing a decision threshold are three different operations.

**First pass.** Follow sections 1–7, including the small score queue, cost calculation, weighted-probability example and geometric SMOTE investigation. Section 7 joins them in a complete observed-data study with actual outcomes. Try practice 1–7 before opening solutions. Section 8 and practice 8–10 are deeper branches. You need the earlier train/validation distinction, averages and a logistic prediction; confusion counts, costs and the new probability notation are introduced here.

## 1. A rare class is a description, not a diagnosis

In **class imbalance**, the target classes occur at different frequencies in the dataset. We call the event of interest positive and the other class negative. Positive does not mean good, and it need not always be the smaller class. The **prevalence** π is the positive fraction in the population or sample being discussed.

If 20 of 1,000 examples are positive, an always-negative baseline gets 980 correct:98% accuracy, zero detected positives. Now consider a model that detects 14 positives, misses 6, and raises 18 false alarms. It gets 976 correct—slightly less accurate—but may be more useful if detecting the event matters enough.

| Actual class | Predicted negative | Predicted positive |
| --- | ---: | ---: |
| Negative | TN=962 | FP=18 |
| Positive | FN=6 | TP=14 |

**Figure 1 — Follow the cases, not just a percentage.** Show 1,000 cases as two population bands, with the twenty positives visibly separated into detected/missed counts and the 980 negatives into cleared/false-alarm counts. Beside the count matrix, compare the always-negative and actual-model outcomes. Preserve exact totals instead of letting the large TN cell visually erase the other three.

Accuracy still has a precise meaning: the fraction of correct class decisions. It is a valid objective when every error has the same cost and the assessment population matches the intended task. It simply does not answer every rare-event decision problem. A constant predictor is a real baseline, and in a problem with indistinguishable classes and equal error costs it can even be optimal.

Nor does the ratio tell you how much information is available. A 99:1 dataset might contain one positive and 99 negatives, or 1,000 positives and 99,000 negatives. The ratio is the same; the possibilities for fitting, validation and discovering positive subgroups differ greatly. Important questions include class overlap, label quality, rare subtypes, sample dependence, feature availability and the cost of each action. No universal 10%,1% or 0.1% boundary selects the correct algorithm.

Imbalance in training and a change in deployment prevalence are also different problems. Cross-entropy can learn the correct posterior from naturally imbalanced data under suitable model, sampling and optimization conditions. Rebalancing is a modeling choice to investigate; it is not a required repair to a probability law.

## 2. Read the mistakes and the ranked queue

### Counts become decision-specific measurements

From the four cells above:

\[
\text{precision}=\frac{TP}{TP+FP}=\frac{14}{32}=0.4375,
\qquad
\text{recall}=\frac{TP}{TP+FN}=\frac{14}{20}=0.7.
\]

Precision asks how many selected cases were positive. Recall asks how many actual positives were selected. Specificity is TN/(TN+FP); the false-positive rate, FPR, is 1−specificity. **Balanced accuracy** is the average of positive recall and specificity for this binary task. It gives the two actual classes equal aggregate weight, unlike ordinary accuracy's prevalence weighting.

The harmonic summary is

\[
F_1=\frac{2TP}{2TP+FP+FN},\qquad
F_\beta=\frac{(1+\beta^2)TP}{(1+\beta^2)TP+\beta^2 FN+FP}.
\]

For the example, F1=28/52≈0.5385. Increasing β emphasizes missed positives more strongly in this formula. It does not mean “a false negative costs β currency units” or supply a universal monetary-cost conversion. F scores also ignore true negatives. If actual costs or capacity are available, evaluate those directly rather than assuming a particular Fβ encodes them.

When nothing is selected, precision's denominator is zero. That mathematical quantity is undefined; software may report zero by a declared convention. Recall is undefined if the assessment contains no actual positives. Show the counts and the convention rather than silently adding a tiny denominator and pretending the number has its ordinary interpretation.

### A higher threshold does not guarantee higher empirical precision

A score-based classifier selects cases whose score is at least a threshold t. As t rises, the selected set shrinks. The number of true positives cannot increase, so recall cannot increase on the same labeled cases. Precision can move either way because the removed cases might be positive or negative.

Consider descending scores 0.9,0.8,0.7 with actual labels 0,1,1:

| Threshold, with score≥t selected | Selected actual labels | Precision | Recall |
| --- | --- | ---: | ---: |
| 0.7 | 0,1,1 | 2/3 | 1 |
| 0.8 | 0,1 | 1/2 | 1/2 |
| 0.9 | 0 | 0 | 0 |

The highest-ranked case is a false alarm. Increasing the threshold makes precision worse throughout this particular queue. That is not an implementation error; it is what the actual ranking does.

**Investigation 1 — Move the gate through actual scored records.** Edit individual scores and labels, record a prediction, then apply a threshold. Watch the selected row IDs move into TP/FP while the rest move into FN/TN. Show actual stair steps at score values, including tied groups. Begin with the counterexample above, then correct the highest case's label to see a different pattern. Changing the order in which equal-score rows are displayed cannot change a threshold decision that selects the whole tied group.

The [threshold-tuning guide](https://scikit-learn.org/stable/modules/classification_threshold.html) separates fitted scores from actions. Moving a threshold alone leaves the scores, their ranking and their ROC/PR curves unchanged; it selects an operating point on those curves.

### Why prevalence changes what a false-positive rate means operationally

ROC plots recall/TPR against FPR. Precision–recall plots recall against precision. If the same population has prevalence π, then

\[
\text{precision}=\frac{\pi\,TPR}{\pi\,TPR+(1-\pi)FPR}.
\]

The numerator is the fraction of all cases that are true positives; the second denominator term is the fraction that are false positives. At π=0.01, TPR=0.8 and FPR=0.01,10,000 cases contain 100 positives:80 are detected, and 99 of 9,900 negatives raise false alarms. Precision is 80/179≈0.447. A 1% false-positive rate can create more false alarms than true detections.

If prevalence falls to 0.001 **while the within-class score distributions stay fixed**, the same TPR/FPR gives precision≈0.0741. The ROC operating point stays the same under that assumption; the workload composition changes. More negatives do not mechanically inflate ROC-AUC if class-conditional score distributions are unchanged. ROC and PR emphasize different quantities, and a useful report may include both plus counts at the chosen operating point. There is no mathematically privileged 10% prevalence cutoff between them. [Davis and Goadrich](https://mark.goadrich.com/articles/davisgoadrichpr.pdf) explain their relationship and why interpolation in PR space needs care.

**Figure 2 — One detector, two populations.** Keep TPR and FPR fixed and draw the two population-flow diagrams with their expected selected positive/negative counts. Use fractional expected counts when a selected population size does not yield integers. A linked ROC marker stays fixed while the precision marker moves. Label this a declared population calculation, not a retrained model or measured prevalence-shift experiment.

**Average precision**, AP, summarizes a scored ranking by weighting precision at each distinct score threshold by the increment in recall:

\[
AP=\sum_k (R_k-R_{k-1})P_k.
\]

For descending labels 1,0,1,0 with distinct scores, recall increases at ranks 1 and 3, each by 1/2. AP=(1/2)·1+(1/2)·(2/3)=5/6. Ties are handled as a score group, not by inventing an order using labels. Scikit-learn's [AP implementation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html) uses this noninterpolated convention; a trapezoidal area under a drawn PR curve is generally different. Name the computation rather than using “PR-AUC” ambiguously.

A constant-score predictor has AP equal to sample prevalence when positives are present. A random independent ranking has precision equal to prevalence at the population level; an individual finite ranking's AP need not equal prevalence exactly. AP is a ranking summary, not performance at a particular review budget, a calibration measure, or a direct cost objective.

## 3. Choose an action from its expected consequences

Suppose p=P(Y=1|x) is the posterior for the intended deployment population. For a simple binary decision, declare zero cost for correct decisions, cost cFP for selecting a negative and cFN for missing a positive. Then

\[
R(\text{select}\mid x)=(1-p)c_{FP},\qquad
R(\text{skip}\mid x)=p c_{FN}.
\]

Select when the first risk is no greater than the second. With positive costs,

\[
p\geq\frac{c_{FP}}{c_{FP}+c_{FN}}.
\]

At cFP=1,cFN=12, the threshold is 1/13≈0.0769. For a case with p=0.1, selecting costs 0.9 in expectation and skipping costs 1.2, so selecting is preferable under this declared model. The numerical costs are teaching assumptions, not laboratory prices or medical guidance. If costs are equal,0.5 is the correct threshold for the true posterior, however rare the positive class is.

**Investigation 2 — Two action costs cross.** Let the learner edit the two costs and a case's posterior, predict the lower-cost action, then reveal the two risk lines, their crossing and the actual numerical comparison. Multiplying both costs by the same positive factor changes the vertical units but leaves the decision threshold unchanged. At the crossing, either action has the same expected cost; use a declared tie rule.

The full rule is broader: for action a and actual class y, choose the action minimizing Σ_y C(a,y)P(y|x). Correct actions may have nonzero costs; “send to a human reviewer” can be another action; costs can vary by case. Use one consistent accounting baseline. The [Elkan paper](https://cseweb.ucsd.edu/~elkan/rescale.pdf), especially section 1, explains why casually mixing lost opportunities and expenditures can create an incoherent cost matrix.

The optimality calculation assumes the probabilities used in it match the information and population under discussion. A calibration chart checks average observed frequency among similar scores; it does not prove that a score equals the full-feature posterior for every subgroup. With estimated or misspecified scores, a validation-selected decision rule can be useful, but a theorem about the true posterior is not a guarantee for that estimator.

### A review budget is not the same as an error-cost ratio

If exactly k cases can be reviewed and each detected positive has equal value, selecting the k largest **true posterior probabilities** maximizes expected detections, since the expectation is the sum of their probabilities. With estimated scores, evaluate the resulting top-k procedure on appropriate assessment cases. If value, harm or review time differs by case, rank by the relevant expected benefit and solve the actual constrained allocation problem; a plain probability ranking need not be optimal.

Top-k and a fixed threshold differ. A fixed threshold can produce varying daily workload. Top-k fixes capacity but its cutoff moves with the day's scores. Declare what happens when scores tie across the capacity boundary—such as a reproducible label-independent tie breaker or a randomized policy. Do not use unseen true labels to break ties. Precision at k and recall at k describe the resulting workload, while AP aggregates many possible cutoffs.

## 4. Reweighting changes what the fitted score means

### Follow the weighted loss to its gradient

Logistic prediction uses margin z=b+xᵀw and score q=σ(z)=1/(1+e^(−z)). For y∈{0,1}, binary log loss is −y log q−(1−y)log(1−q). Its derivative with respect to z is q−y. Weighting observation i by a positive a_i gives the normalized objective

\[
J(b,w)=\frac{1}{A}\sum_i a_i\,[\log(1+e^{z_i})-y_i z_i]
+\frac{\lambda}{2}\|w\|_2^2,
\quad A=\sum_i a_i.
\]

The intercept is unpenalized. Therefore

\[
\nabla_w J=\frac1A\sum_i a_i(q_i-y_i)x_i+\lambda w,
\qquad
\partial_bJ=\frac1A\sum_i a_i(q_i-y_i).
\]

The same residual still appears; its contribution is scaled. For two observations x=0,y=0 and x=2,y=1, initial b=w=0 gives q=0.5 for both. With weights 1 and 3, the intercept gradient is(0.5−1.5)/4=−0.25 and the coefficient gradient is(0−3)/4=−0.75. A step of size 0.4 gives b=0.1,w=0.3, with new scores about 0.525 and 0.668. These are a single illustrative step, not a converged model or a new accuracy claim.

**Figure 3 — A weighted update has two sums.** Show each row's residual, weight, feature and contribution to the intercept/coefficient gradients. Draw the resulting probability curve before and after that one update. A large weight does not create another measured observation; it changes the optimization objective.

The common balanced-class rule uses a_c=n/(K n_c), where K is the number of observed classes and n_c the count of class c in the **fitting** data. Each class then contributes equal total weight. This is a convention, not an estimate of real-world error costs. Weighting after a resampler has already created equal class counts may make “balanced” weights all one; using old weights afterward instead creates another objective. Check which counts the estimator actually receives.

Our normalization divides by total weight A, so multiplying every a_i by the same factor leaves the full objective unchanged. A library that divides by n or uses a summed loss can change its effective regularization under that scaling unless its penalty is adjusted. The [Regularization lesson](/learn/path/full-curriculum/regularization-l1-l2-elastic-net-dropout?module=classical-ml) explains why these objective conventions matter when comparing C or λ.

### Derive the population score instead of calling it calibrated

At an input with true positive probability p, positive class weight w+ and negative weight w− give expected loss

\[
\ell(q)=-w_+p\log q-w_-(1-p)\log(1-q).
\]

For 0<p<1 and positive weights, set the derivative to zero:

\[
-\frac{w_+p}{q}+\frac{w_-(1-p)}{1-q}=0
\quad\Longrightarrow\quad
q^*=\frac{w_+p}{w_+p+w_-(1-p)}.
\]

The second derivative is positive, so this is the unique interior minimum. Endpoint probabilities give corresponding endpoints. At p=0.1,w+=9,w−=1, q*=0.5. It does **not** mean a 50% event probability in the original population. Solving backward gives

\[
p=\frac{w_-q^*}{w_+(1-q^*)+w_-q^*}.
\]

Thresholding the ideal weighted score at 0.5 is equivalent to thresholding p at w−/(w++w−). If the weights equal the intended FP/FN cost roles—w−=cFP,w+=cFN—this reproduces the simple cost decision under these population assumptions. Inverse-frequency weights generally express another cost preference. Restricted models, regularization and finite optimization can also change the ranking, not merely the intercept; inverse algebra does not automatically calibrate an arbitrary fitted estimator.

**Investigation 3 — A score of one half can mean another probability.** Edit p,w+,w− and predict the weighted optimum before applying. Display the loss curve over q and the explicit mapping between p and q*. At p=0.1, changing weights from 1:1 to 9:1 moves q* from 0.1 to 0.5; multiplying both weights by a common positive factor leaves the minimizer unchanged. An inverse mode asks the learner to recover p from an entered weighted score. Name the population assumptions beside this one central explanation, rather than placing a new calibration warning after every later result.

### Duplication and weights share a limited equivalence

Repeating a row c_i times is algebraically identical to integer weight c_i in an additive full-data loss **with matching normalization and regularization**. Random oversampling gives random repetition counts, so it does not exactly equal uniform class weighting in every run. Mini-batch composition, early stopping and stateful operations can break a practical training equivalence even when full-data objectives agree. Weighting can save storage, but neither weighting nor duplication creates independent new evidence about an unseen positive subtype.

## 5. SMOTE creates a geometric assumption

**Random oversampling** samples existing minority rows with replacement. **Random undersampling** retains a chosen subset of majority rows. The first changes multiplicities; the second can discard valuable coverage. Neither requires generating a physically new input. A target ratio is tunable and need not be 1:1.

**SMOTE**, Synthetic Minority Over-sampling Technique, adds interpolated feature vectors. For a minority anchor x_i, find k other minority neighbors under a declared distance, choose one x_j, and generate

\[
x_{\mathrm{new}}=x_i+u(x_j-x_i),\qquad u\sim\operatorname{Uniform}[0,1].
\]

Use **one scalar u for the entire vector** in this line-segment construction. Independent fractions per coordinate generally generate a different shape. Label the synthetic vector as the targeted minority class, then train the chosen classifier on the augmented training data. Ordinary SMOTE's neighbor search does not consult a fitted classifier or ask whether majority examples occupy the segment. See the [original paper, section 4](https://arxiv.org/pdf/1106.1813) and the [current sample-generation definition](https://imbalanced-learn.org/stable/over_sampling.html#mathematical-formulation).

For x_i=(0,0),x_j=(2,0),u=0.5, the generated point is(1,0). If an observed majority point already sits at(1,0), the interpolation produces a conflicting label at exactly that coordinate. The two positive endpoints do not prove that the segment is positive. This is why synthesis can help one geometry and hurt another; many generated rows are not many independent confirmations.

**Investigation 4 — Move the endpoints and inspect what the rule ignores.** Edit minority and majority coordinates, choose an anchor, k, neighbor and interpolation fraction, then predict the generated location and any visible conflict. Reveal the actual minority distance table and permitted segment. Moving only a majority point leaves the ordinary-SMOTE synthetic coordinate unchanged when the minority points and random choice stay fixed. Moving a minority endpoint changes it. The unchanged output is a useful failure demonstration: the algorithm did not look at the information that made you uneasy.

Distance is part of the model. One large-scale input can dominate Euclidean neighbors; fit a suitable scaler inside the training boundary. Interpolating standardized coordinates and then applying the inverse affine scaler produces a segment in source coordinates, but scaling can change **which neighbor is chosen**. More neighbors may cross separate minority clusters; fewer can make synthesis narrow or repetitive. With m minority rows, k must be at most m−1 when self-neighbors are excluded. A fold with only one minority observation cannot support ordinary distinct-neighbor SMOTE.

Categories, one-hot constraints and biological validity require care. A halfway category code is not a new valid category. SMOTENC uses different categorical handling for mixed data, and SMOTEN targets all-categorical data; neither guarantees that every combined feature pattern is physically realizable. Domain-valid augmentations can be preferable when there is a meaningful mechanism for generating input variations.

### A complete-label row must remain a complete-label row

In a multi-label task, suppose observed feature 0 has labels(A=1,B=0), while feature 2 has(A=1,B=1). SMOTE for A might place a synthetic feature at 1 and assign A=1. What is its B label? The interpolation has supplied no answer. Setting B=0, copying one endpoint, taking the union, or using a missing-label policy are different assumptions. Concatenating independently synthesized per-label arrays and pretending their rows still identify the same entities is invalid.

Randomly duplicating an observed **whole row with its entire label vector** preserves that observation's label alignment, though it changes the frequency of every co-occurring label. A justified multilabel augmentation needs an explicit joint label/missing-label rule and appropriate validation. The earlier [Multi-Label & Multi-Output Learning](/learn/path/full-curriculum/multi-label-multi-output-learning?module=classical-ml) supplies the task semantics; this lesson supplies the resampling boundary.

### Complete teaching functions

Save the following as `imbalance_models.py`. It requires NumPy and SciPy. The optimizer minimizes the exact normalized objective from section 4; `np.logaddexp` avoids unstable direct exponentials, and `expit` supplies the sigmoid. The interpolation routine uses a small, explicit distance matrix so its neighbor identities can be checked. It is appropriate for this lesson's small minority set, not a claim of a scalable million-row implementation.

```python
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

def fit_logistic(features, labels, sample_weight=None, penalty=0.01):
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=float)
    design = np.column_stack([np.ones(len(labels)), features])
    weight = np.ones(len(labels)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    weight = weight / weight.sum()

    def objective(parameters):
        margin = design @ parameters
        loss = np.dot(weight, np.logaddexp(0, margin) - labels * margin)
        loss += penalty * np.dot(parameters[1:], parameters[1:]) / 2
        gradient = design.T @ (weight * (expit(margin) - labels))
        gradient[1:] += penalty * parameters[1:]
        return loss, gradient

    result = minimize(objective, np.zeros(design.shape[1]), jac=True, method="L-BFGS-B",
                      options={"maxiter": 1000, "gtol": 1e-9, "ftol": 1e-13})
    if not result.success:
        raise RuntimeError(result.message)
    return result.x

def smote_points(minority, count, k=3, seed=66):
    minority = np.asarray(minority, dtype=float)
    if minority.ndim != 2 or not np.isfinite(minority).all():
        raise ValueError("Use a finite minority feature matrix.")
    if not 1 <= k < len(minority) or count < 0:
        raise ValueError("Require 1 <= k < minority count and a nonnegative output count.")
    distance = np.sum((minority[:, None, :] - minority[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(distance, np.inf)
    neighbors = np.argsort(distance, axis=1, kind="stable")[:, :k]
    rng = np.random.default_rng(seed)
    anchor = rng.integers(len(minority), size=count)
    neighbor = neighbors[anchor, rng.integers(k, size=count)]
    fraction = rng.random(count)
    return minority[anchor] + fraction[:, None] * (minority[neighbor] - minority[anchor])

def predict_score(parameters, features):
    return expit(parameters[0] + np.asarray(features) @ parameters[1:])

if __name__ == "__main__":
    anchor = np.array([0.0, 0.0])
    neighbor = np.array([2.0, 0.0])
    print("declared midpoint", anchor + 0.5 * (neighbor - anchor))
    print("seeded synthetic points", smote_points([[0, 0], [2, 0], [0, 2]], 3, k=2))
```

Inputs to `fit_logistic` in the complete study below are finite, aligned binary data with positive weights and both classes present. The short teaching function relies on that setup; a reusable public library should enforce its complete input contract. Its printed seeded samples are to be generated when the displayed program is executed, not copied from a different RNG implementation. Equivalent author calculations supporting the study were executed; independent verbatim execution of these assembled teaching files remains an implementation check.

## 6. Fit the sampler only where learning is allowed

A protected assessment should contain appropriate **observed** cases from the population and unit you want to evaluate. Do not balance it just to make metrics look pleasant. A case-control evaluation sample can still be useful with an explicit design and valid reweighting, but its raw precision is not automatically deployment precision.

For each training/validation split, fit learned preprocessing on the training portion, transform it, generate or select training rows, and fit the classifier. Transform validation inputs using the fitted preprocessing and predict those original rows. No synthetic validation rows, validation-neighbor search or validation-driven cleaning enters that fit.

**Figure 4 — A fork in the pipeline, not a transform applied everywhere.** Training follows fit scaler→resample→fit model. Validation follows transform with fitted scaler→predict with fitted model. A separate threshold-tuning set may choose an action rule; an assessment set then evaluates that locked rule. Repeated records and related groups must stay within the appropriate boundary before this diagram begins.

Using an `imblearn.pipeline.Pipeline` is a convenient way to implement training-only resampling in cross-validation. A correct manual fold loop can also do it. Standard sklearn pipelines require compatible transform interfaces and do not magically make a `fit_resample` object work. A pipeline cannot repair duplicated entities split across partitions or a target that was unavailable at prediction time. The [resampling pitfalls example](https://imbalanced-learn.org/stable/common_pitfalls.html) illustrates both information leakage and the changed evaluation population caused by resampling before a split.

Thresholds, resampling ratios, k, class weights, model settings and preprocessing are all choices if selected using scores. Keep them within development. A threshold can be tuned on a separate set or cross-validated out-of-fold predictions. Do not tune it on a set later described as an untouched test. With few positives, one moved case can make a large difference; stratification helps preserve class counts but does not produce independent positive evidence or replace a needed group/time split.

## 7. An observed-data study: rare protein localization

### Define the source and the learning task

The [UCI Yeast collection](https://doi.org/10.24432/C5KG68) contains 1,484 rows describing protein localization, with eight numeric descriptors and a sequence identifier. We define the positive target as **ME2**, membrane protein with an uncleaved signal, versus the other recorded locations. The source has 51 ME2 rows. These are historical engineered sequence descriptors, not modern raw-sequence embeddings or a wet-laboratory trial.

The actual source contains 22 repeated sequence IDs. Each repeated ID has exactly identical descriptors and label. We preserve the whole offline source file, verify this identity, and retain the first occurrence of each ID for the derived analysis. That leaves 1,462 distinct protein IDs and 51 positives. This documented duplicate rule prevents the same observed protein from crossing a split; it does not discard errors after seeing predictions. Protein-family similarity may still create dependence, and the source supplies no family grouping for assessing a new family or species.

We predeclare six score inputs: `mcg`, `gvh`, `alm`, `mit`, `vac`, `nuc`. Respectively, they concern signal-sequence recognition by two methods, membrane-spanning-region prediction, mitochondrial versus nonmitochondrial amino-acid content, vacuolar/extracellular content, and nuclear-localization signals. We leave out binary HDEL indicator `erl` and targeting-signal field `pox` to keep this **continuous-score interpolation experiment** explicit. Their omission is not a claim of predictive uselessness. Interpolating these scores creates a feature-space training example; it does not manufacture a biologically valid protein sequence. The source does not supply physical measurement units for these descriptor scores.

The deterministic split retains source-row IDs:

| Role | Distinct protein records | ME2 positives | What it may influence |
| --- | ---: | ---: | --- |
| Fitting | 600 | 21 | Scaler, synthetic neighbors, weights, fitted coefficients |
| Threshold tuning | 200 | 7 | Each predeclared model's decision threshold |
| Inspection | 200 | 7 | Reported comparison and error analysis |
| Reserved | 462 | 16 | No predictions or scores in this lesson |

The outer development split uses seed 61, fitting split 62 and tuning/inspection split 63, with stratification at each step. We fit five declared logistic procedures: original data, balanced class weights, random oversampling, random undersampling and ordinary SMOTE. The six inputs, λ=0.01, unpenalized intercept, optimizer and preprocessing fit remain fixed. The final comparison contains five fits, with no search for a favorable model family. The scaler fits the original 600 fitting records for all methods; resampling changes the classifier's training input afterward.

There are 579 negative and 21 positive fitting records. Balancing by random duplication or SMOTE adds 558 positives, yielding 1,158 rows. Random undersampling keeps 21 negatives and all 21 positives, yielding 42 rows. Balanced weights leave 600 stored rows and assign positive weight 600/42 and negative weight 600/1158.

For a concrete decision comparison, use **hypothetical** FP cost 1 and FN cost 12, with zero correct-decision cost. For each fitted model, evaluate all distinct tuning-score thresholds plus a no-alert policy, choose minimum tuning cost, and break ties toward the higher threshold. This selected score threshold is not claimed to be the posterior formula 1/13: the scores are imperfect and some were fitted to different objectives.

### Run the complete study

Save as `yeast_imbalance.py` beside `imbalance_models.py` and the provided `yeast.data`. Setup: Python, NumPy, SciPy and scikit-learn. The author calculation ran with Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1 and sklearn 1.9.1. Exact split, convergence, donor choices, predictions and thresholds are retained in the author record. The complete displayed programs still need their independent execution check during implementation.

```python
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from imbalance_models import fit_logistic, smote_points, predict_score

def read_study_data():
    raw = np.loadtxt(Path(__file__).with_name("yeast.data"), dtype=str)
    identifiers, first, counts = np.unique(raw[:, 0], return_index=True, return_counts=True)
    for identifier in identifiers[counts > 1]:
        if len(np.unique(raw[raw[:, 0] == identifier, 1:], axis=0)) != 1:
            raise ValueError("Repeated identifier has conflicting data; resolve before splitting.")
    kept = np.sort(first)
    features = raw[:, 1:9].astype(float)[:, [0, 1, 2, 3, 6, 7]]
    labels = (raw[:, -1] == "ME2").astype(int)
    development, reserve = train_test_split(
        kept, train_size=1000, stratify=labels[kept], random_state=61)
    fitting, remaining = train_test_split(
        development, train_size=600, stratify=labels[development], random_state=62)
    tuning, inspection = train_test_split(
        remaining, train_size=200, stratify=labels[remaining], random_state=63)
    return features, labels, fitting, tuning, inspection, reserve

def counts_and_cost(labels, scores, threshold):
    selected = scores >= threshold
    tp = int(np.sum(selected & (labels == 1)))
    fp = int(np.sum(selected & (labels == 0)))
    fn = int(np.sum(~selected & (labels == 1)))
    tn = int(np.sum(~selected & (labels == 0)))
    return tp, fp, fn, tn, fp + 12 * fn

def choose_threshold(labels, scores):
    thresholds = np.r_[np.inf, np.sort(np.unique(scores))[::-1]]
    costs = [counts_and_cost(labels, scores, t)[-1] for t in thresholds]
    return thresholds[int(np.argmin(costs))]

def run_study():
    features, labels, fitting, tuning, inspection, reserve = read_study_data()
    scaler = StandardScaler().fit(features[fitting])
    x_fit = scaler.transform(features[fitting])
    x_tune = scaler.transform(features[tuning])
    x_inspect = scaler.transform(features[inspection])
    y_fit = labels[fitting]
    positive = np.flatnonzero(y_fit == 1)
    negative = np.flatnonzero(y_fit == 0)
    n_positive, n_negative = len(positive), len(negative)
    weights = np.where(y_fit == 1, len(y_fit) / (2 * n_positive),
                        len(y_fit) / (2 * n_negative))
    duplicate = np.random.default_rng(64).choice(
        positive, size=n_negative - n_positive, replace=True)
    retained = np.r_[np.random.default_rng(65).choice(
        negative, size=n_positive, replace=False), positive]
    synthetic = smote_points(x_fit[positive], n_negative - n_positive, k=3, seed=66)
    methods = [
        ("original", x_fit, y_fit, None),
        ("balanced_weight", x_fit, y_fit, weights),
        ("random_over", np.vstack([x_fit, x_fit[duplicate]]),
         np.r_[y_fit, np.ones(len(duplicate))], None),
        ("random_under", x_fit[retained], y_fit[retained], None),
        ("smote", np.vstack([x_fit, synthetic]),
         np.r_[y_fit, np.ones(len(synthetic))], None),
    ]
    for name, training_features, training_labels, weight in methods:
        parameters = fit_logistic(training_features, training_labels, weight, penalty=0.01)
        tuning_scores = predict_score(parameters, x_tune)
        threshold = choose_threshold(labels[tuning], tuning_scores)
        inspection_scores = predict_score(parameters, x_inspect)
        print(name, "training rows", len(training_labels), "threshold", threshold)
        print("at 0.5: TP FP FN TN cost",
              counts_and_cost(labels[inspection], inspection_scores, 0.5))
        print("at selected threshold: TP FP FN TN cost",
              counts_and_cost(labels[inspection], inspection_scores, threshold))
        print("AP", average_precision_score(labels[inspection], inspection_scores),
              "ROC-AUC", roc_auc_score(labels[inspection], inspection_scores),
              "Brier", brier_score_loss(labels[inspection], inspection_scores))

if __name__ == "__main__":
    run_study()
```

The reserved IDs are returned to make ownership explicit but never passed to prediction. Each procedure keeps its original fitted coefficients while its threshold is tuned; we do not refit on tuning data afterward and silently assume the score scale stays identical. A later production refit needs a compatible complete threshold/calibration protocol.

### Read the outcomes that actually occurred

The following numbers are from the retained author calculation:

| Procedure | Selected score threshold | Inspection TP / FP / FN / TN | Inspection cost FP+12 FN | AP | Top 10 positives |
| --- | ---: | --- | ---: | ---: | ---: |
| Original | 0.145006 | 2 / 12 / 5 / 181 | 72 | 0.173459 | 2 |
| Balanced weights | 0.740873 | 4 / 15 / 3 / 178 | 51 | 0.273317 | 1 |
| Random oversampling | 0.706116 | 4 / 17 / 3 / 176 | 53 | 0.278679 | 2 |
| Random undersampling | 0.772569 | 4 / 19 / 3 / 174 | 55 | 0.193550 | 2 |
| SMOTE | 0.759272 | 4 / 13 / 3 / 180 | 49 | 0.272615 | 1 |

The always-negative baseline has 193/200=96.5% accuracy, recall 0 and cost 84. Its precision is undefined. At threshold 0.5, the original fitted model makes one false alarm and detects none, for cost 85; the tuned threshold improves its realized cost to 72. The SMOTE procedure's tuned result has cost 49 rather than its default-threshold cost 65. Threshold selection and resampling have played different roles.

SMOTE has the lowest realized tuned cost among these five; random oversampling has the highest AP; the original model and two other procedures find two positives in their top ten, while balanced weights and SMOTE find one. Those are different questions with different winners. Neither the AP ranking nor the cost ranking certifies a universally best method. With only seven inspection positives, a single additional detection changes recall by 1/7≈0.143. The method/settings were not changed after seeing this table, and the 462 reserved proteins remain unscored.

The original model's inspection Brier score, mean(q−y)², is 0.033475; balanced weighting gives 0.131518 and SMOTE 0.121856. The smaller original value does not prove perfect calibration: Brier also reflects resolution and the event prevalence. It does show why a larger rare-class score or better chosen action rule is not automatically a better original-population probability estimate. The later [Calibration & Conformal Prediction](/learn/path/full-curriculum/calibration-conformal-prediction?module=classical-ml) develops probability assessment and calibration in full.

**Figure 5 — Each improvement has a named objective.** Show actual observed confusion counts at 0.5 and at the separately selected threshold, then a distinct ranking panel with AP and the top-ten record identities. The panel must not interpolate fake precision curves from the five method summaries. Use saved individual scores to reconstruct exact threshold steps. Keep the threshold-tuning curve labeled tuning, and the final locked inspection outcome labeled inspection.

**Investigation 5 — Inspect a queue without turning inspection into tuning.** Begin with the saved tuning scores of a chosen fitted method, allow a learner to record a threshold/cost prediction, and reveal the actual selected records and costs. Let them change hypothetical costs and recompute the tuning choice. For the declared cost pair, the record above is the exact oracle. A separate locked-result panel displays the original inspection outcome; if a learner experiments with inspection thresholds, label that mode exploratory and clear any claim of untouched assessment. Never expose the reserved proteins' predictions. This teaches the distinction by controlling actual records and choices, not by printing a warning after every click.

For an API-oriented implementation, a training-only sampler can be composed as below. Save as `yeast_smote_cv.py` beside the prior files. This additionally requires `imbalanced-learn`. The current documentation was inspected, but **this optional package program was not executed** in the content phase. It is a separate threefold development-CV illustration with sklearn's stated `C=1` convention, not a reproduction of the custom λ-normalized five-model table.

```python
import numpy as np
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from yeast_imbalance import read_study_data

features, labels, fitting, tuning, inspection, reserve = read_study_data()
# Use the original fitting partition only; other roles retain their declared ownership.
model = make_pipeline(
    StandardScaler(), SMOTE(k_neighbors=3, random_state=66),
    LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
)
folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=67)
scores = cross_val_score(model, features[fitting], labels[fitting],
                         cv=folds, scoring="average_precision", n_jobs=1)
print("three fitting-partition fold AP values", scores)
print("mean fold AP", np.mean(scores))
```

Every fold's scaler and sampler fit anew inside that fold. A shared random seed does not make its synthesized rows identical to our from-scratch routine: random draw order, neighbor ties and algorithm conventions can differ. Validate the learning procedure and synthetic geometry rather than inventing matching stdout.

## 8. Deeper choices: where the simple picture changes

The core workflow is already usable: define the decision, protect assessment data, compare a baseline with justified alternatives, and inspect counts at the intended operating point. The branches below explain what changes when ordinary interpolation or a single class weight is insufficient. They are further study, not prerequisites for the first-pass exercises.

### Where should synthetic points be placed?

Ordinary SMOTE chooses neighbors within the minority class. It does not inspect majority labels when drawing a point on the chosen segment. This is why moving the majority point in Investigation 4 changes the apparent conflict while leaving the synthetic coordinates unchanged. The algorithm's construction and our judgment of that construction are different operations.

**Borderline-SMOTE** first identifies minority observations in mixed-class neighborhoods and directs synthesis toward that boundary region. It can concentrate effort where a classifier is uncertain, but a mixed neighborhood can also contain mislabeled observations or genuine class overlap. A boundary is not automatically missing positive coverage.

**ADASYN** assigns more synthetic examples near minority observations whose neighborhoods contain a larger proportion of other-class observations. In a simplified binary description, let \(r_i\) be that fraction for minority observation \(i\). Normalize these fractions and allocate an intended synthetic budget \(G\) roughly as \(g_i=G r_i/\sum_j r_j\), then use minority neighbors for interpolation. Integer allocation and implementation rules mean the final count need not hit exact parity. If every \(r_i=0\), the normalization is undefined; this is a case to handle explicitly, not evidence that an algorithm has learned how to create useful examples there. Giving difficult regions more points can amplify label noise as well as useful boundary information.

**KMeans-SMOTE** uses a clustering step to restrict and allocate synthesis across suitable clusters. The cluster count and geometry become additional assumptions. **SVM-SMOTE** uses a fitted support-vector boundary to guide candidates. Its behavior depends on that boundary model. These methods replace one geometric assumption with a richer one; they do not eliminate the need to check whether the resulting features and labels represent possible observations. Compare a variant only when its mechanism addresses a visible failure of the simpler procedure.

### When removing data is useful—and what is actually removed

Random undersampling reduces the number of majority observations used in a fit. In our study it retained only 21 of 579 fitting negatives. That speeds some fits and changes the empirical class contribution, but may discard a rare *negative* subtype that is crucial for avoiding false alarms.

Distance-based undersampling makes that choice depend on geometry. The NearMiss variants are distinct:

| Rule | Majority observations preferentially retained |
| --- | --- |
| NearMiss-1 | Smallest average distance to a specified number of nearest minority observations |
| NearMiss-2 | Smallest average distance to a specified number of farthest minority observations |
| NearMiss-3 | First collect a specified number of nearest majority candidates around each minority observation; from these, favor the largest average distances to a specified number of nearest minority observations |

These are rules for retaining observations, not interchangeable descriptions of “remove points near the boundary.” Scaling and outliers affect the choices. An implementation's neighbor counts and sampling strategy are part of the algorithm specification.

Cleaning methods answer another question: which local configurations should be deleted? A **Tomek link** is an opposite-class pair whose members are each other's nearest neighbor, subject to a tie convention. A common policy removes the majority member; an explicit all-class policy may remove both. **Edited nearest neighbors** deletes selected observations whose labels disagree with a specified neighbor vote. The vote may require unanimity or a majority, and the classes eligible for removal must be stated. Disagreement is observable; an incorrect label is not established by disagreement alone.

SMOTE followed by Tomek or neighbor cleaning can first add coverage and then remove selected overlap. Cleaning changes the class counts again, so the result need not remain balanced. Inspect the retained/deleted identities and the decision cost. A tidier scatterplot is not a validation criterion.

### Ensembles can distribute the discarded information

The [decision-trees and random-forests lesson](/learn/path/full-curriculum/decision-trees-random-forests?module=classical-ml) showed how multiple fitted models can reduce dependence on one sample. In a balanced ensemble, each learner can receive a different resampled training subset. Across learners, more majority observations may participate than in a single undersampled fit. This can preserve useful variety while limiting the imbalance seen by each learner.

Balanced bagging and balanced random forests apply such sampling around individual learners or trees. EasyEnsemble combines ensembles fitted on different undersampled subsets; RUSBoost combines random undersampling with boosting. Their voting/averaging rules, sampling replacement, per-learner counts and loss remain substantive choices. Different learners seeing a point does not create a new independent protein, and the entire ensemble must stay within each training fold. Use the earlier ensemble mechanisms to reason about these procedures rather than memorizing a league table of sampler names.

### Focal loss changes which examples dominate an update

A large training set can contain many correctly classified, easy negatives. Even small individual losses may add up. **Focal loss** was introduced for dense object detection, where a detector evaluates many potential object locations and most are background. That setting gives a concrete reason to reduce the influence of already-easy cases.

For a binary example, let \(p_t=q\) when its true label is 1 and \(p_t=1-q\) when its true label is 0. With a class-specific positive factor \(\alpha_t\), focal loss is

\[
L_{\mathrm{focal}}=-\alpha_t(1-p_t)^\gamma\log p_t,\qquad \gamma\ge0.
\]

At \(\gamma=0\), this is weighted cross-entropy. For an easy example with \(p_t=.9\), \(\gamma=2\) multiplies its cross-entropy loss by .01; for a difficult example with \(p_t=.2\), the multiplier is .64. With \(\alpha_t=1\), ten thousand easy examples contribute about 1,053.61 total cross-entropy versus 16.09 from ten difficult examples. Under focal loss those totals become about 10.54 and 10.30. The balancing comes from the current prediction difficulty, not just the class label. These totals are a constructed loss calculation, not a training benchmark.

**Figure 6 — Loss mass from many easy cases.** Show the two populations with explicit counts, per-example loss, and total loss under cross-entropy and focal loss. A count-times-loss rectangle makes the mechanism visible. The chart must say loss, not gradient: differentiating the focal objective also differentiates the factor \((1-p_t)^\gamma\). For a positive example,

\[
\frac{dL}{dp_t}=\alpha_t\left[\gamma(1-p_t)^{\gamma-1}\log p_t-\frac{(1-p_t)^\gamma}{p_t}\right].
\]

The derivative with respect to a model's logit additionally multiplies by \(p_t(1-p_t)\). A claim that the training gradient is simply cross-entropy's gradient times .01 would miss the first term. Difficult examples can include annotation errors; emphasizing them is not always beneficial. Focal loss is a changed objective and does not generally retain ordinary log loss's original-posterior optimum. The later [loss-functions lesson](/learn/path/full-curriculum/loss-functions-ce-mse-focal-contrastive-triplet?module=deep-learning-fundamentals) develops network losses, gradients and training comparisons. It is not needed to use the cost-sensitive logistic workflow above.

### A changed prior has an exact correction under a specific assumption

Suppose sampling changes only the positive-class proportion from a deployment value \(\pi\) to a training value \(\rho\), while preserving both class-conditional feature distributions. Let \(q(x)\) be the true posterior in that sampled population. Bayes' rule gives

\[
\frac{p(x)}{1-p(x)}
=\frac{q(x)}{1-q(x)}
\frac{\pi/(1-\pi)}{\rho/(1-\rho)}.
\]

To see why, write each posterior odds as the same likelihood ratio \(f(x\mid Y=1)/f(x\mid Y=0)\) times its population's prior odds, then divide. This is a prior-shift calculation, not a universal repair of a model score. It requires nondegenerate priors and the unchanged-conditional-distribution assumption; the displayed finite-odds calculation also assumes scores strictly between 0 and 1, with endpoints handled by limits where meaningful.

If a balanced sample has \(\rho=.5\), deployment has \(\pi=.01\), and its posterior is \(q=.8\), the sampled odds are 4. Multiply by 1/99 to obtain deployment odds 4/99, hence \(p=4/103\approx.038835\). A sample-posterior value of .8 can correspond to less than 4% deployment probability. That is a change of population, not a contradiction.

Random case-control sampling can plausibly preserve class-conditionals when sampling is independent of features within each class. Ordinary SMOTE alters the minority feature distribution by interpolation. Its change is therefore not, in general, just a prior change. A finite regularized classifier can also be misspecified. Use representative held-out probability assessment and an appropriate calibration protocol when decisions require probabilities; do not apply an odds correction and declare the problem solved. Calibration and threshold selection have different roles even when both use development data.

This distinction is useful beyond fraud or diagnosis. In a materials screening experiment, scientists may deliberately measure many more promising candidates than their natural prevalence would provide. A score describing that enriched sample is not automatically the probability that a randomly chosen candidate will succeed. In industrial fault monitoring, the deployment class-conditionals themselves may change with a new sensor or operating regime, so a prior-only correction may be insufficient.

### More rows do not create more independent evidence

Seven inspection positives leave very limited information about sensitivity. Stratification can place positives in each fold, but the same positive observed across repeated splits is still one underlying observation. Repeated folds measure a procedure's sensitivity to particular resplits; their standard deviation is not automatically a confidence interval for future recall. Use uncertainty methods suited to the sampling design, keeping repeated subjects or related units together. Exact duplicate removal in this study addresses one concrete leakage route; it does not reveal unknown protein families or guarantee transfer to a new organism.

A multiclass problem introduces multiple class-specific errors and possibly a full action-cost matrix. A macro-average weights classes equally; a frequency-weighted average answers a different question. Resampling every class to the largest count is a candidate training design, not a universal target. In a multi-label problem the full label vector and missing-label status remain attached to each observation, as the conflicting-label example in §5 demonstrated. If rare positives are unlabelled rather than confirmed negative, the task also requires a label-observation model; class balancing cannot turn unknown truth into negative truth.

### Budget the actual operations

For \(m\) minority observations in \(d\) dimensions, our explicit all-pairs neighbor calculation needs \(O(m^2d)\) arithmetic and \(O(m^2)\) stored distances. Sorting every distance row costs \(O(m^2\log m)\); more specialized selection/search structures can change that work, with effectiveness depending on dimension and geometry. Producing \(G\) interpolated vectors then costs \(O(Gd)\), plus storage for the generated data. The neighbor search is over the minority points here, not automatically every point in the dataset.

If the majority count is \(M\ge m\), balancing by adding minority rows changes total rows from \(M+m\) to \(2M\). The expansion factor is \(2M/(M+m)\), below 2. A 99-to-1 dataset becomes 198 rows from 100, not a hundred times larger. Balancing by undersampling leaves \(2m\) rows. Class weights avoid materializing duplicate feature rows, although they can change conditioning, convergence and the training path. There is no zero-overhead or equal-runtime guarantee.

## 9. Practice: make the decision yourself

Try each question before opening its hint or solution. Questions 1–7 use the first-pass route. Questions 8–10 transfer the deeper ideas. A calculator or a short script is welcome; the target is a defensible explanation, not mental arithmetic speed.

### 1. What can 99% accuracy hide?

There are 1,200 observations,12 positive. Construct two confusion matrices with 99% accuracy: one with recall 0 and another with recall 1. Explain how the same accuracy can describe both.

<details><summary>Hint</summary>

Both matrices need 1,188 correct predictions and 12 errors. Assign those errors to different cells.

</details>
<details><summary>Solution</summary>

Always predicting negative gives TP 0, FP 0, FN 12, TN 1,188. Detecting all positives while making 12 false alarms gives TP 12, FP 12, FN 0, TN 1,176. Both have 1,188/1,200 accuracy. Their recalls are 0 and 1. Accuracy reports a particular equal-error-cost aggregate; the aggregate alone does not identify which errors occurred.

</details>

### 2. Ties are whole score groups

Four records have scores `[.95, .8, .8, .4]` and labels `[0, 1, 0, 1]`. With selection rule score≥threshold, compute precision and recall at threshold .8. If the review budget permits exactly two records, why is “take everything at least .8” not the same policy?

<details><summary>Hint</summary>

The threshold includes both records tied at .8. For an exact budget, state how the tie is broken before inspecting its labels.

</details>
<details><summary>Solution</summary>

At .8, TP 1, FP 2, FN 1, TN 0, so precision 1/3 and recall 1/2. The threshold selects three records. Taking exactly two must select the .95 record and one of the tied records. A fixed record-ID ordering or a predeclared random tie rule can do that; choosing the positive because its held-out truth is known leaks the outcome into the policy. The resulting top-two precision can differ depending on the legitimate tie rule.

</details>

### 3. Choose by expected cost

A false alarm costs 2 units and a missed positive costs 7. Correct decisions cost 0. A well-specified posterior gives \(p=.2\). Which action has lower expected cost? Derive the cutoff rather than applying .5 automatically.

<details><summary>Hint</summary>

Compare \(2(1-p)\) with \(7p\).

</details>
<details><summary>Solution</summary>

Selecting costs 1.6 in expectation and skipping costs 1.4, so skip. Selection is better when \(2(1-p)<7p\), or \(p>2/9\). At exactly 2/9 the actions tie under this cost model. The answer depends on a probability for the relevant population and on the stated costs; rarity alone did not decide it.

</details>

### 4. What does the weighted score mean?

At a feature value, the population positive probability is .2. Train unrestricted weighted binary log loss with positive weight 4 and negative weight 1. Find the population-optimal score, then invert it back to the original probability. What happens to the optimum if both weights are multiplied by 3?

<details><summary>Hint</summary>

Insert the values into the optimum derived in §4. The ratio of weights controls that optimum.

</details>
<details><summary>Solution</summary>

The optimum is \(q=(4\times.2)/(4\times.2+1\times.8)=.5\). Inverting gives \(p=w_-q/[w_+(1-q)+w_-q]=.5/(2+.5)=.2\). A weighted score .5 is not a claim that the original event probability is .5. Multiplying both weights by 3 leaves this population optimum unchanged; it scales the unnormalized expected loss. Our normalized finite-sample objective also cancels a common weight factor, while an objective normalized differently can change its effective regularization.

</details>

### 5. Interpolate a vector, then question its label

A minority anchor is \((1,2)\), its chosen minority neighbor is \((5,4)\), and \(u=.25\). Calculate the SMOTE point. If the training fold contains only three minority observations, can the usual “five other minority neighbors” setting be used? Finally, explain why a majority point near the generated location matters even though it did not enter the interpolation formula.

<details><summary>Hint</summary>

Use the same scalar fraction for both coordinates. Count neighbors after excluding the anchor itself.

</details>
<details><summary>Solution</summary>

The point is \((1,2)+.25(4,2)=(2,2.5)\). There are at most two other minority observations, so five-neighbor SMOTE is undefined in that fold; choose a justified smaller setting or a different procedure before evaluation. A nearby majority observation suggests overlap or an implausible minority-label assumption. Ordinary SMOTE does not resolve that conflict merely by creating the point.

</details>

### 6. Preserve the whole observation

Two training observations have feature values 0 and 4 and known label vectors `(A=1, B=0, C=1)` and `(A=1, B=1, C=0)`. You interpolate feature 2 while balancing label A. A colleague proposes assigning the midpoint label `(1, 1, 1)` because all three labels occur among the endpoints. Is that label established by the input? Give one coherent alternative that does not invent a label.

<details><summary>Hint</summary>

The feature interpolation establishes no rule for how B and C behave between endpoints.

</details>
<details><summary>Solution</summary>

No. The endpoints do not establish B or C at feature 2, and the proposed label combination was observed at neither endpoint. A domain-supported label-generation model could justify a new observation, but it must be stated and checked. Alternatively, oversample an entire existing row with its unchanged complete label vector, or use a fitting weight with an explicitly defined multi-label loss. Independent per-label synthetic matrices cannot be silently joined into one aligned dataset.

</details>

### 7. Choose the question before the winner

Using the observed Yeast table, identify the method with lowest declared inspection cost, the one with highest AP, and all methods with the most positives in their top ten. Would replacing the declared goal with AP after seeing these results preserve an untouched comparison? What does one additional detected positive change in recall?

<details><summary>Hint</summary>

Read three different columns. The inspection partition contains seven positives.

</details>
<details><summary>Solution</summary>

SMOTE has cost 49; random oversampling has AP .278679; original fitting, random oversampling and random undersampling each find two positives in their top ten. Changing the goal after inspecting outcomes is a new exploratory decision, not the original locked assessment. A new evaluation protocol is needed before presenting a newly selected procedure as independently assessed. One additional detection changes recall by 1/7≈.143. None of the three result columns removes that small-positive-count uncertainty.

</details>

### 8. Correct the sampling odds

Assume unchanged class-conditional feature distributions. A sampled population has positive prevalence .2, deployment prevalence is .02, and the sampled-population posterior is .5. Compute the deployment posterior. Explain why the same calculation is not automatically justified for SMOTE scores.

<details><summary>Hint</summary>

The sampled posterior odds are 1. Multiply by deployment prior odds divided by sampling prior odds.

</details>
<details><summary>Solution</summary>

The odds multiplier is \((.02/.98)/(.2/.8)=4/49\). Thus deployment odds are 4/49 and probability is 4/53≈.075472. SMOTE alters the distribution of minority features, and a fitted score may not equal the sampled population's true posterior. Those issues violate steps used in the derivation; knowing the two class proportions is insufficient.

</details>

### 9. Count storage and information separately

A fitting set has 960 majority and 40 minority observations. How many rows result from random oversampling to parity? From random undersampling to parity? Does the first procedure provide 960 independent minority examples?

<details><summary>Hint</summary>

Keep the original count of unique minority observations separate from the materialized row count.

</details>
<details><summary>Solution</summary>

Oversampling gives 1,920 rows, a 1.92-fold expansion from 1,000. Undersampling gives 80 rows. The duplicated minority rows still come from 40 underlying observations, so treating them as 960 independent events for an uncertainty calculation would exaggerate the evidence. The larger training matrix may change an optimization procedure while leaving the number of independently observed proteins unchanged.

</details>

### 10. Design a review queue under changing conditions

A factory can investigate ten sensor alerts per shift. One false alarm takes the same review time as one real fault. The classifier was trained on deliberately fault-enriched data, and next month a new sensor model will be installed. Describe a defensible development/evaluation plan. Explain where class weights, prior correction and a top-ten policy each fit, and name information that the prompt does not supply.

<details><summary>Hint</summary>

Separate fitting, probability interpretation, resource allocation and transfer to the new sensor. A prior adjustment assumes more than knowing the fault percentage.

</details>
<details><summary>Solution</summary>

Keep related machine histories and time boundaries intact; fit preprocessing and any sampler within the training role. Compare a baseline with justified weights or resampling using representative development data. Select a review policy there, with a label-independent tie rule and an explicit response when fewer than ten alerts have worthwhile expected value. A top-ten score policy enforces capacity; probability/cost thresholds answer an additional question about whether reviewing a candidate is worthwhile. Prior correction could apply if enrichment preserved class-conditionals and the fitted probabilities represent that enriched population. A new sensor can change feature distributions within each class, so collect or otherwise justify representative new-sensor assessment rather than assuming that correction suffices. Reserve a final future/group-separated assessment for the selected full procedure. The prompt leaves fault costs, enrichment mechanism, annotation completeness, sensor compatibility and population prevalence unspecified; identify these as design inputs, not arbitrary defaults. Weighted fitting alone does not supply them.

</details>

## 10. Readiness and the next lesson

You are ready to continue when you can construct a confusion table, explain why an accurate model can miss every rare event, derive an action cutoff from stated costs, distinguish a weighted score from an original-population probability, generate and question a SMOTE point, and place sampling and threshold choice on the correct side of the assessment boundary. Questions 1–7 check those skills. The advanced branches let you reason about richer samplers, changed priors and resource constraints when a task requires them.

The next module topic is [AutoML & Neural Architecture Search](/learn/path/full-curriculum/automl-neural-architecture-search-nas?module=classical-ml). We will let a procedure search over modeling choices. That only helps if the search is asked the right question: the objective, fold ownership, data geometry and decision cost you specified here must travel with it. Automating a leaky or irrelevant comparison makes it easier to repeat the same mistake at scale.

## References and other ways to learn

- **A visual sampler comparison:** the imbalanced-learn [comparison of oversampling methods](https://imbalanced-learn.org/stable/auto_examples/over-sampling/plot_comparison_over_sampling.html) shows actual input clouds, generated samples and fitted decision boundaries for duplication, SMOTE, ADASYN and variants, plus mixed/all-categorical examples. Its substantive example and code were inspected. Use the pictures to compare geometric assumptions; they are examples, not a universal performance ranking.
- **The original SMOTE mechanism:** Chawla et al., [SMOTE: Synthetic Minority Over-sampling Technique](https://arxiv.org/pdf/1106.1813), especially §4's construction and §6's categorical discussion. The paper motivates interpolated minority features and also reports limits, including an unfavorable Adult-dataset case. The current library guide clarifies present algorithm conventions; this lesson explicitly uses one interpolation fraction for a whole vector.
- **Current implementation definitions:** imbalanced-learn's [oversampling guide](https://imbalanced-learn.org/stable/over_sampling.html), [undersampling guide](https://imbalanced-learn.org/stable/under_sampling.html), [combined samplers](https://imbalanced-learn.org/stable/combine.html) and [balanced ensembles](https://imbalanced-learn.org/stable/ensemble.html) explain which observations are generated, retained or removed. Consult the exact sampler and version before treating two API names as equivalent. The [training/assessment pitfalls example](https://imbalanced-learn.org/stable/common_pitfalls.html) demonstrates why preprocessing and sampling belong within each training split.
- **Turn scores into decisions:** scikit-learn's [decision-threshold guide](https://scikit-learn.org/stable/modules/classification_threshold.html) separates score fitting from threshold tuning and illustrates CV-based threshold selection and fixed-threshold use. Its APIs are an alternative to our explicit threshold sweep; choose the score/cost objective appropriate to the task.
- **Why costs change the cutoff:** Elkan, [The Foundations of Cost-Sensitive Learning](https://cseweb.ucsd.edu/~elkan/rescale.pdf), §§1–3, derives cost-based decisions and the assumptions behind class-prior rescaling. Read after the two-action derivation here; finite, constrained learners need not behave like unrestricted population-optimal rules.
- **Understand ROC and PR geometry:** Davis and Goadrich, [The Relationship Between Precision-Recall and ROC Curves](https://mark.goadrich.com/articles/davisgoadrichpr.pdf), explains their fixed-population relationship and why PR interpolation needs care. For the exact noninterpolated score used in our table, consult [average_precision_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html). Average precision is not automatically the trapezoidal area under a plotted PR curve.
- **Where focal loss came from:** Lin et al., [Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002), §3 and its loss plots, gives the many-background-locations motivation and modulating factor. The inspected mechanism supports our loss calculation; its detection benchmarks were not reproduced by this lesson.
- **The actual observations:** Nakai, [Yeast](https://archive.ics.uci.edu/dataset/110/yeast), UCI Machine Learning Repository, [DOI record](https://doi.org/10.24432/C5KG68), CC BY 4.0. The offline copy and provenance identify all rows, score fields and the exact-duplicate repair. This is an observed protein-localization dataset, not an experiment establishing a production assay's costs or transfer performance.

Research and documentation inspection: 12 September 2026. The real-data author calculation used Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1 and scikit-learn 1.9.1. The optional imbalanced-learn program is deliberately marked unexecuted. These resources offer complementary mathematical, visual and code routes.
