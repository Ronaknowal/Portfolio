# Calibration & Conformal Prediction

A classifier assigns an unfamiliar recording a probability of .9 for “mechanical fault.” Another system returns a set: {loose bearing, rubbing belt}. A third predicts tomorrow's sound level as 120 dB with an interval from 113 to 127 dB. All three communicate uncertainty, but they make different promises. Understanding those promises is as important as producing the numbers.

**Calibration** asks whether forecasts with a stated probability have the corresponding outcome frequency in the population of interest. **Conformal prediction** turns a fixed prediction procedure and fresh labeled examples into sets with a particular coverage guarantee. Neither operation makes an inaccurate model accurate by definition. The useful result is a prediction whose meaning, construction, and limits the learner can explain.

The preceding PAC/VC lesson concerned population error and variation across training samples. Here we first examine the numerical meaning of a probability, then follow one prediction through probability calibration and conformalization. The core route is sections 1–8 and the practice. Section 9 develops optional extensions. You need conditional probability, averages, sorting, and the idea of a held-out sample. Logistic regression and evaluation metrics help; their required operations are refreshed locally.

By the end, you should be able to construct and interpret a reliability diagram, fit a simple probability map without leaking outcomes, calculate the exact split-conformal rank, distinguish marginal from subgroup coverage, and produce a report that compares usefulness as well as coverage.

## 1. A probability is a statement about a conditioning population

Suppose a weather service issues “70% chance of rain” on many days. Among days receiving that forecast, the rain frequency should approach .7 in the population described by the service. This is not a demand that any one day contain 70% rain, or that exactly seven of the next ten forecasts succeed.

Let Y be 1 when the event happens and 0 otherwise. Let P=p(X) be the model's probability for that event. Binary class-probability calibration means

\[
\mathbb E[Y\mid P]=P.
\]

The left side is the positive rate among inputs receiving a given forecast. The right side is the stated forecast. With continuously varying probabilities, conditioning is understood through a conditional expectation; we estimate it from nearby scores because an exact decimal may never repeat. A finite sample estimates this population relationship and can disagree with it through sampling noise.

### Class probability and confidence of being correct

For a binary prediction p=.2, the model assigns .2 to class 1 and .8 to class 0. If it predicts class 0, its top-class confidence is .8. A plot of p against the frequency of **class 1** is therefore different from a plot of maximum confidence against **whether the selected class was correct**.

This distinction can conceal real errors. Consider two equally common forecast groups. One receives p=.2 but has a true positive rate .3; the other receives p=.8 but has a true positive rate .9. Class-1 probabilities are too low in both groups. Yet top-class confidence is .8 for everyone, and average correctness is (.7+.9)/2=.8. A confidence-only diagram looks perfect because it merges two different mistakes.

For multiple classes, useful questions include calibration of each class's probability against that class's indicator, calibration of top confidence against correctness, and calibration of the entire predicted probability vector. These condition on different information. To examine the forecast for class k, use the pairs (p_k, 1{Y=k}) across the relevant evaluation population; restricting to examples with Y=k leaves only positive outcomes and is not a class-k reliability diagram. [Vaicenavicius and colleagues, calibration definitions and evaluation framework](https://proceedings.mlr.press/v89/vaicenavicius19a/vaicenavicius19a.pdf).

### Calibration can discard useful information

Imagine two equally common production lines whose true fault rates are .1 and .3. A model assigns .2 to both. It is calibrated when conditioning only on that score, because the combined rate is .2. A model that preserves the line information and reports .1 or .3 is also calibrated, with more **resolution**: it separates groups with different outcome frequencies.

Suppose releasing a faulty item costs 80 units, releasing a sound item costs nothing, and quarantining any item costs 20. Release has expected cost 80p; quarantine costs 20, so the threshold is p=.25. The coarse score releases everyone, costing 16 on average. Using the two rates releases the .1 group and quarantines the .3 group, costing (.5×8)+(.5×20)=14. Calibration supports sensible decisions using the information in a score; it does not guarantee that the score preserves all information useful for the decision.

Recall that Brier loss is the average squared probability error, (p−y)². In a group whose positive rate is r, forecasting p has expected loss r(1−p)²+(1−r)p²: weight the error for each possible outcome by how often it occurs. Averaging across our equally common groups gives .16 for the constant .2 score and .15 for the two exact group rates. Both forecasts are calibrated. The better proper score reflects additional useful information, not a calibration repair. This connects directly to the distinction between ranking, probability quality, and operational cost in Evaluation Metrics.

## 2. Build a reliability diagram from the actual observations

Use this small **constructed sample**. There are five observations with forecast .2, of which two are positive, and five with forecast .8, of which three are positive.

| Forecast | Count | Positive outcomes | Observed positive fraction | Observed minus forecast |
|---|---:|---:|---:|---:|
| .2 | 5 | 2 | .4 | +.2 |
| .8 | 5 | 3 | .6 | −.2 |

A reliability diagram places mean predicted class-1 probability on the horizontal axis and observed class-1 fraction on the vertical axis. The points are (.2,.4) and (.8,.6). The diagonal represents matching values. At x=.8, y=.6 is **below** the diagonal: the positive probability is overestimated. At x=.2, y=.4 is above: it is underestimated. For a top-confidence diagram, below the diagonal specifically means overconfidence about the selected class.

The accompanying count strip matters. Two positives out of five are weak evidence about a precise population rate. Two hundred out of five hundred provide substantially more precision. A dot without its denominator hides that difference. Empty bins have no estimated fraction and should not be mistaken for bins with zero positive outcomes.

### What binned Expected Calibration Error (ECE) measures

For bins B_b, define the sample summary

\[
\widehat{\mathrm{ECE}}=\sum_b\frac{n_b}{n}
\left|\bar y_b-\bar p_b\right|.
\]

With one bin for each forecast group, ECE is (.5×.2)+(.5×.2)=.2. Merge the entire sample into one bin and both means become .5, giving ECE=0. The model did not change. The binning erased opposing local discrepancies.

Binning creates two competing problems: wide bins hide variation, while small bins have noisy outcome counts. Even a perfectly calibrated population can produce nonzero empirical ECE. Conversely, zero empirical ECE for a particular binning does not prove calibration. Report the population, forecast definition, boundaries, bin counts, and sampling uncertainty along with the summary; choose the analysis before using it to select a model. Repeatedly searching bin schemes for the smallest ECE is another form of selection. [The primary evaluation study's section 4 discusses these estimation effects](https://proceedings.mlr.press/v89/vaicenavicius19a/vaicenavicius19a.pdf).

For our two-bin sample, replacing .2 and .8 with the observed fractions .4 and .6 gives zero **in-sample** ECE. Brier loss falls from .28 to .24; ROC AUC remains .6 because the ordering is unchanged. This is a useful calculation, but fitting and judging the repair on the same ten outcomes would not establish generalization.

**Investigation: make an aggregate error disappear.** Edit the individual forecast/outcome cards, choose boundaries, and record where the dots will move and whether ECE will rise or fall. Reveal only after committing the prediction. First merge the two bins without changing any observations. Then change one forecast or outcome instead. Finally reorder the cards: the result should stay the same. The evidence behind every plotted dot remains visible as a count and a list of its contributing observations.

The complete `reliability` function in [calibration_calculations.py](calibration_calculations.py) handles empty bins and includes p=1 in the final bin. Excluding forecasts exactly equal to 1 silently loses observations, often the most consequential ones.

## 3. Fit a probability map to independent predictions

A raw model score may rank examples well without being a probability. A support-vector classifier's decision score is one example. Applying a sigmoid merely puts that score between zero and one; it does not establish that .8 means an 80% event rate.

We keep the base predictor fixed and learn a map from its score to a probability using examples that did not fit that predictor. The map has its own parameters and can overfit. Its performance must be assessed on further observations.

### Sigmoid calibration: fit both slope and offset

Write the logistic function as σ(z)=1/(1+e⁻ᶻ). A sigmoid calibrator uses

\[
q(s)=\sigma(as+b).
\]

The slope controls how quickly probabilities change with score; the offset moves the probability scale. At q=.5, as+b=0. Changing b can therefore change predicted classes under a .5 threshold. A positive a preserves score order, a negative a reverses it, and a zero a produces a constant. Multiclass one-versus-rest mappings followed by normalization have additional interactions; their effects cannot be inferred from one binary curve.

For binary labels, fit a and b by minimizing average log loss. A numerically stable contribution is `logaddexp(0,z) - target*z`, with z=as+b. Its gradients are the averages of (q−target)s and q−target. This is the same residual mechanism as logistic regression, applied to one already-computed score.

Platt's method uses smoothed targets: positives receive (N₊+1)/(N₊+2), and negatives receive 1/(N₋+2). Smoothing discourages infinite logits on a separable calibration sample. It is part of the stated fit, not a promise of exact calibration. Our complete program implements the smoothed objective and checks the optimizer's result.

For scores [−3,−2,−1,0,1,2,3,4] and labels [0,1,0,0,1,0,1,1], it obtains a≈.2677017731 and b≈−.1338508865. Forecasts at the eight scores are approximately [.281516,.338664,.400939,.466587,.533413,.599061,.661336,.718484]. These are fitted values for a constructed example, not an independently measured improvement.

### Isotonic calibration: pool a local contradiction

Sometimes a sigmoid is too restrictive, but higher scores should still mean no lower probability. **Isotonic regression** fits nondecreasing probabilities by minimizing the sum of squared differences from the observed labels. Sort the observations by their scores while keeping labels attached; do not sort the labels separately.

Start with each distinct score as a block. A block's fitted probability is its fraction of positive outcomes. If a left block's mean exceeds its right neighbor's mean, the required monotonicity is violated. Merge them and replace both estimates by their combined, count-weighted mean. Continue backward if the merge creates a new violation. This is the pool-adjacent-violators algorithm.

In the eight-score example, the sequence of raw label means is [0,1,0,0,1,0,1,1]. The 1 followed by 0 merges to .5, then that block and the next 0 merge to 1/3. The later 1,0 pair merges to .5. The fitted values become

\[
[0,\tfrac13,\tfrac13,\tfrac13,\tfrac12,\tfrac12,1,1].
\]

Equal raw scores must receive equal fitted values. For scores [−2,−2,0,1] with labels [1,0,0,1], group the two −2 observations first: their mean is .5 with weight 2. Pooling with the zero at score 0 gives 1/3 with weight 3. The fitted values at distinct scores [−2,0,1] are [1/3,1/3,1]. Averaging block means without their counts would give the wrong result.

The stack implementation is linear after sorting; sorting ordinarily costs O(n log n). Fitted values at the observed knots are constant within pooled blocks. A library must also specify how to predict between knots and outside the observed range: our practical example uses scikit-learn's interpolation with endpoint clipping. The resulting interpolation need not be a staircase between every pair of distinct knots.

More flexibility requires enough informative calibration observations, especially in rare score regions. There is no universal sample count at which isotonic suddenly becomes preferable. It can introduce ties and change AUC, despite being monotone. Assess held-out proper scores, reliability counts, and relevant decisions together. [Scikit-learn's calibration guide, sigmoid and isotonic methods](https://scikit-learn.org/stable/modules/calibration.html).

**Investigation: repair the staircase.** Before revealing a merge, select the next violating adjacent blocks and predict their combined probability. Edit a label and trace the consequences backward. Reordering observations with the same score should have no effect. Changing a label can change several downstream fitted values. Overlay the sigmoid fit only after constructing the monotone solution, then inspect where the two assumptions produce different probabilities. A small calibration loss is a fit statistic, not a held-out score.

### Temperature scaling: change confidence while keeping the winning logit

A multiclass model often produces logits z₁,…,z_K. Softmax turns them into positive numbers summing to one. **Temperature scaling** uses one positive T:

\[
q_k=\frac{\exp(z_k/T)}{\sum_j\exp(z_j/T)}.
\]

For logits [3,1,0], T=1 produces approximately [.843795,.114195,.042010]. With T=2, it becomes [.628532,.231224,.140244]. The leading class remains first. Dividing all logits by the same positive number preserves their within-example order and ties, so the top-1 class and its accuracy are unchanged under the same tie rule. Probability thresholds and cost-sensitive decisions can still change. A claim about top-1 accuracy is not a guarantee about every downstream action or every ranking across different examples.

Fit T by minimizing held-out multiclass log loss. Using inverse temperature β=1/T, the objective is the average of `logsumexp(β*z) - β*z_true`; its derivative compares the probability-weighted mean logit with the true-class logit. Our instructional fit uses a declared positive search range and reports whether the answer is near its boundary. The base model's logits are cached, so fitting T need not repeatedly run the whole network.

The method can soften or sharpen probabilities, but one scalar cannot repair arbitrary class-specific biases. Vector or matrix scaling adds parameters and can change winning classes. Guo and colleagues found temperature scaling effective on the models and datasets they studied; that empirical finding is not a rule that every later network must be overconfident or improve under calibration. [Guo et al., method and results](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf).

## 4. Decide who is allowed to see each label

A clear data-flow diagram is more valuable than a method name when evaluating leakage. In our classification experiment, labels have four separate roles:

1. **Base fit:** learn preprocessing and the classifier.
2. **Probability calibration:** fit a map from the frozen model's predictions to probabilities.
3. **Conformal calibration:** choose a rank threshold using the now-frozen probability procedure.
4. **Final assessment:** measure probability quality, coverage, set sizes, and relevant slices.

The third role is separate because probability calibration is itself training. Reusing its outcomes to compute an ordinary split-conformal threshold treats those observations differently from a fresh test example. Holding out only the original classifier's training data is insufficient. Likewise, hyperparameter selection and early stopping consume label information. Either account for them within a suitable nested procedure or complete them before using genuinely separate calibration observations.

Scikit-learn 1.9.1 supports `CalibratedClassifierCV(FrozenEstimator(fitted_pipeline), method="sigmoid")`. The wrapper holds the fitted pipeline fixed and uses the provided observations to fit calibration. It cannot determine whether you supplied an honest independent split. The estimator's score interface matters: sigmoid calibration uses `decision_function` when available and otherwise uses `predict_proba`. GaussianNB has no decision function, so this is not automatically a sigmoid of its generative log odds. Saturated probabilities and unsaturated log odds contain different usable numerical information.

With cross-validation instead, keep the scaler, text vocabulary, and learned feature selection **inside** the base pipeline cloned for every fold. With `ensemble=False`, out-of-fold scores fit a calibrator and a base estimator is refitted on all supplied training data. With `ensemble=True`, fold-specific calibrated estimators are retained and their probabilities averaged. Both involve a different inference procedure from a frozen single base model. Calibration folds do not undo hyperparameter selection that already used their outcomes.

The current API also supports `method="temperature"`; for probability-only estimators it constructs log-probability logits with a numerical safeguard. SVC's `probability` parameter is deprecated in 1.9 and scheduled for removal in 1.11, so our example uses an explicit calibration wrapper. These are versioned software contracts, not timeless properties of the mathematics. [Current calibration API](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html), [SVC API](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html).

## 5. Conformal prediction starts with an ordering of mistakes

We now freeze the complete predictor and its score function. A **nonconformity score** is a number that is large when a proposed answer fits poorly. It need not be a calibrated probability.

For classification, one choice is s(x,y)=1−p_y(x): a class given a small model probability receives a large score. For regression, s(x,y)=|y−m(x)| measures the distance from a point prediction, in the target's units. The prediction set will contain candidate answers whose scores are not unusually large relative to fresh labeled calibration examples.

Suppose nine calibration scores, already sorted from **smallest to largest**, are

\[
[.05,.10,.15,.20,.25,.30,.40,.60,.90].
\]

For desired coverage 1−α=.8, calculate k=⌈(n+1)(1−α)⌉=⌈10×.8⌉=8. The threshold is the eighth smallest score, q=.6. For a new input, include every candidate y satisfying s(x,y)≤.6.

| New probability vector for classes A, B, C | Class scores 1−p | Returned set |
|---|---|---|
| [.80,.15,.05] | [.20,.85,.95] | {A} |
| [.45,.40,.15] | [.55,.60,.85] | {A,B} |
| [.34,.33,.33] | [.66,.67,.67] | Empty |

Equality is included. The last row is not a contradiction: this particular score can return an empty set when no class reaches the learned threshold. A larger set is not automatically produced for every difficult input. Empty, singleton, and large sets all deserve interpretation. A singleton is not automatically a label with 80% conditional probability of correctness; the theorem concerns a different probability.

### Why n+1 appears

Imagine including the next example's true-label score alongside the n calibration scores. If these n+1 scores are exchangeable and have no ties, the next score is equally likely to occupy any of their n+1 ranks. For k≤n, the event that it is no greater than the kth calibration score corresponds to its combined rank being at most k. Its probability is k/(n+1), which is at least 1−α by our choice of k.

The next example's true label is unknown when predicting. The algorithm therefore tests each possible label and retains those that would pass this same score comparison. Exactly when the true label passes, the set covers it.

Here **exchangeable** means that permuting the observations leaves their joint distribution unchanged. Independent draws from one distribution are a common sufficient condition, but exchangeability can also hold for dependent observations, such as a uniformly randomized ordering of a fixed finite collection. It is not a property created by shuffling an arbitrary time series before forecasting its future.

Conditional on any independent base-training and probability-calibration work, if the conformal observations and the new example are exchangeable and are scored by the same fixed rule,

\[
\Pr\{Y_{\rm new}\in C(X_{\rm new})\}\ge 1-\alpha.
\]

The probability averages over conformal calibration data and the new example. It is not a claim about each individual input or every realized calibration set. With ties, including equality preserves the lower guarantee and can produce extra coverage. The familiar upper limit 1−α+1/(n+1) additionally needs a no-ties condition or a suitable randomized construction. If every possible score equals .2, the nonrandomized set rule covers every true label: coverage is 1, even if the target is .8. [Split-conformal theorem and rank proof, Appendix D](https://arxiv.org/html/2107.07511v6#A4).

### The finite rank is not an interpolated percentile

If k≤n, use sorted_scores[k−1] in zero-based indexing. If k=n+1, use infinity. For our nine scores and α=.05, k=10 and the answer is infinity, so every possible answer is included. Clipping k to n would discard the protection requested by a target beyond the resolution of this small calibration set.

On our nine-score example, NumPy's default linear quantile at 8/9 returns about .633333, while `method="higher"` at that same level returns .9. Neither is the eighth order statistic .6: quantile conventions use different indexing rules. Rather than rely on a remembered recipe, our function computes k and selects the exact indexed value. The direct definition also handles the infinity case explicitly.

**Investigation: rotate the hidden next score.** Start with ten distinct score cards, including .95. Hide one as the future example, calculate the threshold from the other nine, and predict whether it is covered. Cycle through all ten possibilities: exactly eight pass at α=.2. Then make all ten scores equal and repeat. Edit the requested α, a score near the selected rank, or an irrelevant lower score. A threshold depends on an order statistic, so some numerical changes leave it unchanged.

The complete program records these exact rotations. They illustrate the symmetry behind the proof; they are not a simulation claiming to verify all exchangeable populations.

## 6. Adapt the set to the quantity you want to predict

### Constant-width regression intervals

With absolute residual scores, s(x,y)=|y−m(x)|≤q means

\[
C(x)=[m(x)-q,\ m(x)+q].
\]

Every interval has width 2q. The centers vary, but the widths do not. These are intervals for a **future response**, including its variability around the predictor. They are not confidence intervals for an unknown mean or credible intervals from a Bayesian posterior.

### Normalize by a local scale

Suppose an independent training procedure supplies a positive scale u(x), large in regions expected to have bigger errors. Use s(x,y)=|y−m(x)|/u(x). Calibration scores are now dimensionless and the final interval is

\[
[m(x)-q\,u(x),\ m(x)+q\,u(x)].
\]

The local scale need not be the true standard deviation for marginal conformal validity. Its quality affects usefulness. It must be fixed before ordinary conformal calibration, finite and positive. Training outcomes may be used to fit it; tuning it with the conformal calibration outcomes changes the procedure covered by the ordinary split-conformal proof. Zero or negative scales are invalid here.

Consider residual magnitudes [.5,1,1.5,2,3,4,5,6,8] and corresponding scales [1,1,1,1,2,2,2,3,4], in the same physical units. At α=.2, the absolute-residual threshold is 6 and the normalized threshold is 2. Predictions m=10 with scale 1 and m=20 with scale 3 receive intervals [8,12] and [14,26]. The absolute method gives [4,16] and [14,26]. This construction explains how useful scale information can avoid giving easy cases unnecessarily large intervals; it does not claim this scale model was learned accurately from data.

**Investigation: keep units and coverage separate.** Edit calibration residuals and scales, predict which order statistic changes, and construct the query intervals. Double every calibration and query scale: normalized q halves, leaving the physical intervals unchanged. Change the last two residuals to 12 and 16 while keeping their scales fixed: q rises from 2 to 4, doubling the half-widths. Calibration outcomes change the learned threshold; a choice of display units should not change membership.

### Conformalized quantile regression

A different approach learns lower and upper conditional quantile estimates L(x), U(x). One standard definition of the quantile at level τ is the smallest cutoff q for which P(Y≤q)≥τ; using the infimum handles distributions whose support needs it. In a continuous distribution with a strictly increasing cumulative probability near q, that probability equals τ. **Pinball loss** weights an underprediction by τ and an overprediction by 1−τ, so fitting it targets a quantile rather than a mean.

Approximate .05 and .95 quantiles supply an initial 90% interval, but fitted endpoints do not automatically have finite-sample coverage. Conformalized quantile regression, or **CQR**, calibrates the score

\[
s(x,y)=\max\{L(x)-y,\ y-U(x)\}.
\]

If y lies outside the initial interval, this is the distance beyond the nearer violated endpoint. Inside, it is nonpositive. The final set is [L(x)−q, U(x)+q]. A positive q expands the endpoints; a negative q can shrink an overly wide initial interval. The score definition admits an empty result if shrunken endpoints cross. Expanding such a set for a declared practical policy preserves inclusion, but its width and coverage should be evaluated as that enlarged procedure.

Our training code resolves crossed **base** quantile estimates by a fixed pointwise min/max rearrangement before computing any scores, and applies the same transformation at prediction. It never sorts each endpoint differently using the true answer. CQR adapts widths through its initial quantile fits, while keeping a marginal guarantee under the same exchangeability conditions. It does not guarantee correct coverage at every x. [Romano, Patterson and Candès, CQR score and construction](https://proceedings.neurips.cc/paper/2019/file/5103c3584b063c431bd1268e9b5e76fb-Paper.pdf).

## 7. Run two complete offline experiments

Download [calibration_calculations.py](calibration_calculations.py), [uncertainty_experiments.py](uncertainty_experiments.py), [banknote-subset.csv](banknote-subset.csv), and [airfoil-subset.csv](airfoil-subset.csv) into one folder. The first program contains complete numerical mechanisms; the second imports them and runs both data experiments. It does not download data or require access to the website. [Data provenance](data-provenance.md) gives creators, licenses, original row IDs, selection rules, units, and hashes.

Use Python 3.12 and a virtual environment. On Windows:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install numpy==2.3.5 scipy==1.18.1 scikit-learn==1.9.1
.venv\Scripts\python calibration_calculations.py
.venv\Scripts\python uncertainty_experiments.py
```

On macOS or Linux, create the environment with `python3 -m venv .venv`, then use `.venv/bin/python` in place of `.venv\Scripts\python`. The files write `checked-results.json` and `experiment-results.json`, retaining raw scores, predictions, intervals, data-role IDs, and measured counts. These are complete examples, including data loading, fitting, prediction and evaluation.

### Classification: separate two meanings of calibration

The Banknote Authentication data contain four wavelet-derived image features and binary class codes. Keep codes 0 and 1 as supplied; we do not infer a semantic mapping absent from the dataset metadata. Each row is one observation. Our retained 480-row subset uses the first 240 for the base pipeline, the next 80 for probability calibration, the next 80 for conformal calibration, and the last 80 for assessment. These allocations precede the comparisons.

The base pipeline is `StandardScaler` followed by an RBF SVC with C=1. We predeclare sigmoid calibration as the primary procedure, and also report isotonic, temperature, a naive sigmoid of the raw decision score, and the training-prior constant. All mappings use the same frozen base predictor where applicable. We do not select the best method using these final assessment results.

For each method, true-class conformal scores on its separate 80 examples determine k=⌈81×.9⌉=73. The last 80 observations assess both its probabilities and its sets:

| Fixed procedure | Correct classifications / 80 | Brier loss | Covered / 80 | Mean set size |
|---|---:|---:|---:|---:|
| Naive sigmoid of SVC score | 79 | .060271 | 69 | .8625 |
| Held-out sigmoid calibration | 79 | .007839 | 71 | .8875 |
| Held-out isotonic calibration | 79 | .004766 | 76 | .9500 |
| Held-out temperature scaling | 79 | .008521 | 69 | .8625 |
| Training-prior constant | 51 | .235538 | 80 | 2.0000 |

The sigmoid procedure has nine empty sets and 71 singletons. Its observed coverage is 88.75%, below 90%. That observation is not logically inconsistent with the marginal theorem: this is one realized conformal split and a finite assessment sample. Conversely, isotonic's 95% observation does not prove a population guarantee or establish it as the universally better calibrator. Its threshold is zero in this run because many conformal scores tie at zero. Ties are precisely why a no-ties upper bound is inapplicable.

All four SVC-derived scores have AUC=1 on these 80 observations, despite different Brier losses and set behavior. The prior baseline returns both labels for every observation: coverage is perfect and the sets convey no class distinction. That baseline makes the usefulness question explicit.

There is also a numerical lesson in the constant baseline. Its class-1 probability is 103/240. Computing `1 - probability <= q` includes equality at q=1−103/240. Rearranging this to `probability >= 1 - q` can fail by one floating-point rounding unit because the second subtraction is not exact. We use the same score computation for calibration and prediction, and the small program retains this boundary fixture. Algebraic equivalence does not excuse dropping a tied boundary in an implementation.

### Regression: sound pressure in an airfoil experiment

The Airfoil Self-Noise dataset records wind-tunnel measurements. Inputs are frequency in Hz, attack angle in degrees, chord length in metres, free-stream velocity in m/s, and displacement thickness in metres. The target is **scaled sound pressure level in dB**, the quantity supplied by the dataset; an interval width in dB is not a linear acoustic-pressure difference. UCI lists 1,503 observations. We retain 480 randomly ordered rows: 240 for fitting, 120 for conformal calibration, and 120 for final assessment. [UCI data description and attribution](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise).

Compare a constant predictor with absolute-residual intervals, a standardized ridge model with absolute-residual intervals, raw .05/.95 gradient-boosted quantile estimates, and CQR around those same quantile estimates. The quantile models have 80 trees of depth 2 and fixed seeds; no test outcomes select these settings. CQR uses rank 109 of 120 scores. Its q is .7663505895 dB, so it expands both raw endpoints by that amount.

| Fixed interval procedure | Covered / 120 | Mean width in dB |
|---|---:|---:|
| Training-mean constant + conformal residuals | 100 | 18.991917 |
| Ridge + conformal absolute residuals | 108 | 16.370437 |
| Raw fitted .05/.95 quantiles | 93 | 14.353792 |
| Conformalized quantile regression | 103 | 15.886494 |

CQR improves this sample's coverage over the unadjusted quantiles and remains slightly narrower on average than the ridge intervals. It also covers fewer observed responses than the ridge procedure. The table preserves that tradeoff; shorter intervals are not automatically better if they miss the outcomes the task needs to cover. Ridge's point MAE is about 3.939 dB compared with 5.765 for the training-mean baseline. Point-error metrics and interval metrics answer different questions.

The randomized allocation supports an exchangeability interpretation for predicting a randomly selected held-out row from this fixed corpus, conditional on the separately fitted model. Measurements share experimental conditions, so this does not establish an iid sample of future wind-tunnel campaigns or performance on a new airfoil design. An intended deployment on new experimental settings would require that grouping to determine the split and validation unit. This is a concrete distinction between a reproducible corpus experiment and its possible real-world generalization.

The plots use retained predictions and endpoints for every assessment row. A useful view shows horizontal interval segments with observed targets, and a second view shows widths versus input frequency. Frequency groups below 2,000 Hz and at least 2,000 Hz are declared diagnostic slices. For ridge, the two coverage counts are 54/66 and 54/54, despite overall 108/120. The same overall score hides quite different local behavior.

## 8. Interpret coverage without changing the probability statement

There are three levels worth keeping separate:

| Quantity | What is averaged or conditioned on? | What it establishes |
|---|---|---|
| Marginal conformal coverage | Calibration sample and next example | The stated lower bound under the score/sampling assumptions |
| Coverage conditional on one calibration sample | Fresh examples with that fitted threshold fixed | A random population property that can vary across calibration samples |
| Observed assessment coverage | A finite collection of recorded covered/not-covered outcomes | An estimate or finite-corpus measurement, with sampling limitations |

At a .9 target, a reported 92/100 is an observation, not a proof. A reported 87/100 is a reason to examine uncertainty, data roles, dependence, score versions and shifts, not by itself a mathematical refutation. Under iid continuous scores and a fixed independently trained model, the conditional coverage for the kth order statistic follows a Beta(k,n+1−k) distribution when k≤n. Its mean is k/(n+1); this explains variation between calibration samples. A finite iid assessment set adds binomial variation **conditional on the fixed threshold**. Marginally, its indicators share a random threshold and should not be treated as independent Bernoulli trials with parameter exactly 1−α. These continuous-score statements do not directly describe tied isotonic scores or our finite-corpus sampling scheme. [Calibration-size and assessment analysis](https://arxiv.org/html/2107.07511v6).

### Marginal coverage does not protect every group

If 80% of a population belongs to group A with coverage 1, and 20% belongs to group B with coverage .5, the marginal coverage is .8×1+.2×.5=.9. Half of group B's answers are missed. This is an exact constructed population comparison, not an estimate from the airfoil experiment.

A **group-conditional** or Mondrian construction can compute a separate rank threshold within each predeclared group. The group definition and score must be fixed independently of that group's conformal outcomes, and calibration/future examples need the appropriate within-group exchangeability. Small groups can require infinity to support a demanding coverage target. A guarantee for several specified groups is not a guarantee conditional on every possible x or on every overlapping slice chosen after inspecting failures.

Class-conditional conformal prediction is a related construction: compute q_k from calibration observations whose true class is k, then test candidate class k against q_k at prediction. The future label is unknown, but each candidate uses its own class threshold. This is different from the mistaken class-calibration plot that discards all negative outcomes for a class.

### A singleton is a selection event

Keeping only singleton sets changes the population under discussion. Ordinary marginal coverage does not guarantee the same rate among those retained predictions. A selective system should report how many observations it acts on, error among acted-on observations, and the handling of empty or multiple-label sets. If that selected population needs its own guarantee, use a construction that targets the corresponding conditional risk; the original marginal claim cannot simply be renamed.

### Changing the population or the procedure

A simple label-shift example makes the calibration issue tangible. Suppose a binary test has sensitivity .8 and false-positive rate .2, unchanged between populations. With event prevalence .5, the positive predictive value is .8. With prevalence .1, it becomes (.8×.1)/[(.8×.1)+(.2×.9)]≈.307692. The class-conditional test behavior stayed fixed; the posterior meaning of its positive result changed.

Conformal guarantees also depend on the relevant joint sampling structure. A new population, a changed preprocessing rule, a refitted model, an updated probability map, or a score selected using conformal labels can invalidate the old threshold's argument. Re-estimating a threshold on recent data helps only if the intended sampling assumptions and selection procedure are appropriate. For sequential data, a time-aware method must state its own guarantee; repeatedly shuffling observations does not turn next-month forecasting into the original exchangeable setting.

An acceptable report names the outcome/unit, data-role allocation, model and calibration versions, score definition, α, rank and threshold, equality policy, coverage counts, set-size/width distribution, important slices and sampling interpretation. It also records what will trigger a new evaluation. No universal ECE threshold, percentage drop or mandatory calibration method replaces that reasoning.

## 9. Optional deeper routes: change the score or the guarantee deliberately

### Adaptive classification sets and regularization

The score 1−p_y tests each class against a common probability cutoff. **Adaptive prediction sets (APS)** instead use the cumulative probability mass up to a candidate class after sorting classes by descending probability. With [.5,.3,.2], the cumulative scores are [.5,.8,1]. The score incorporates the competition among labels rather than only one probability.

At q=.8, the direct score-sublevel set contains the first two classes. At q=.7, it contains only the first. Some practical APS variants also include the next class whose addition crosses the cutoff, ensuring a nonempty boundary-expanded set. That is a different, more conservative set construction. Randomized variants handle boundary inclusion using an auxiliary random variable and must use matching calibration and prediction conventions. A nominal “APS” label without its tie and boundary rules is insufficient to reproduce a result.

RAPS adds a nonnegative penalty to cumulative scores for labels beyond a chosen rank, encouraging efficiency in large label spaces. Penalty and rank choices consume tuning information; freeze them before ordinary conformal calibration or use an analysis that accounts for their selection. Better adaptivity is a design aim, not a universal pointwise coverage guarantee. [APS construction and boundary discussion](https://arxiv.org/html/2107.07511v6).

### Reuse data with a different conformal procedure

Full conformal prediction treats a proposed test label symmetrically with training observations and recomputes the scores in that augmented sample. It can avoid a single held-out split, but naive implementations may refit for every test/candidate-label combination. Continuous regression labels require additional computational structure or a justified approximation, not a finite class loop.

Jackknife+ fits leave-one-out models and combines each omitted observation's residual with that model's prediction at the new x. This is not merely adding leave-one-out residuals around one final fit. Under exchangeability and a symmetric fitting algorithm, its basic worst-case guarantee is 1−2α in the original parameterization, although performance can be closer to 1−α under further conditions. CV+ substitutes fold-specific fits and has its own finite-sample correction. They should not inherit the split-conformal 1−α formula by analogy. Linear least-squares shortcuts need their own algebra; ordinary logistic regression does not have a general exact ridge-style hat-matrix leave-one-out formula. [Jackknife+ construction and theorems](https://stat.cmu.edu/~ryantibs/papers/jackknife.pdf).

### Shift, unusual applications, and what the guarantee targets

Under covariate shift, P(X) changes while P(Y|X) stays fixed. Weighted conformal methods can account for known density ratios and support conditions using a weighted score distribution, including weight associated with the new input. Estimated ratios add an estimation problem; arbitrary dataset shifts are not repaired by putting weights on the old quantile. The original weighted-conformal study itself uses airfoil measurements, making it a useful next experiment after understanding our unweighted protocol. [Tibshirani and colleagues, covariate-shift setting and Airfoil example](https://proceedings.neurips.cc/paper/2019/file/8fb21ee7a2207526da55a679f0332de2-Paper.pdf).

One useful extension is anomaly detection. If larger scores indicate more unusual behavior, a new score s can be compared with n exchangeable reference scores using p=(1+# reference scores ≥s)/(n+1). Flagging small p controls a marginal false-flag probability for a new in-distribution observation under that setup. It does not establish that a flagged item is faulty, and many simultaneous or repeated flags require a corresponding multiple-testing or sequential analysis. The extra one and tie direction echo the same rank logic as prediction sets.

For structured predictions, the cost of an omission may matter more than whether an entire set contains an exact label. **Conformal risk control** replaces the binary miscoverage loss with a specified bounded monotone loss over a nested family, under its stated conditions. A segmentation task might care about the fraction of missed relevant pixels. A selective classifier might instead care about errors among accepted cases; that loss need not satisfy the basic monotonicity condition and requires an appropriate procedure for the selected population, potentially a more general risk-control framework. Define the loss and the randomness being controlled before translating “90%” into a product claim. Venn predictors are another probabilistic framework; they are not a synonym for ordinary group-conditional conformal sets. [Risk-control and outlier-detection branches](https://arxiv.org/html/2107.07511v6#S4).

### Practical libraries and cost

MAPIE 1.5 provides `SplitConformalClassifier` and `SplitConformalRegressor`. A fitted estimator with `prefit=True` is followed by `conformalize` on the separate data and then `predict_set` or `predict_interval`. Explicitly choose the conformity score and inspect output shape: a classification set array has axes observation, class and confidence level. Its classes must be aligned with the estimator's class order, not assumed to be consecutive integers. The [current classifier API](https://mapie.readthedocs.io/en/stable/api/classification/) and [quick start](https://mapie.readthedocs.io/en/v1.5.0/content/getting-started/quick-start/) offer a library route after the exact from-scratch rank calculation. Our executed programs use NumPy/scikit-learn; this MAPIE route is documentation-based guidance, not a claimed executed benchmark.

A frozen sigmoid or temperature map is small, but fitting involves optimization iterations. Isotonic needs sorting and stored knots; a fold ensemble retains several base models. Split conformal requires calibration predictions and an order statistic, plus storing or summarizing scores according to the application. Class-set construction evaluates K candidates per observation for the simple score, while APS additionally sorts classes. Local scale or quantile models add their own training and inference costs. The overhead can be modest compared with the base model, but it is measurable and should not be described as always zero.

## 10. Practice: calculate, change, and interpret

### 1. A misleading confidence diagram

In two equally common groups, forecasts for class 1 are .1 and .9, but true positive rates are .2 and 1.0. What does a top-confidence diagram show? What does a class-1 diagram reveal?

<details><summary>Hint</summary>The first group predicts class 0; its correctness is one minus its positive rate.</details>
<details><summary>Solution</summary>Both groups have top confidence .9. Their correctness rates are .8 and 1, averaging .9, so the confidence-only diagram matches the diagonal. The class-1 points are (.1,.2) and (.9,1), each .1 above the diagonal. The different conditioning quantities conceal miscalibration when merged.</details>

### 2. Construct a different isotonic fit

Scores [1,2,3,4,5,6] have labels [0,1,0,1,0,1]. Find the monotone least-squares probabilities. Explain whether permuting the input row order changes the result.

<details><summary>Hint</summary>Pool adjacent violations after sorting by score.</details>
<details><summary>Solution</summary>The fit is [0,.5,.5,.5,.5,1]. Each 1,0 pair averages to .5; adjacent equal-mean blocks may remain separately represented without changing the predictions. Row order has no effect when scores and labels stay attached and ties are handled correctly. This fit is not an independent demonstration of population calibration.</details>

### 3. Compute a rank without a percentile shortcut

Fourteen calibration scores are the integers 1 through 14. For α=.2, compute k and q. For α=.02, what changes? Is the kth score counted from the largest or the smallest?

<details><summary>Hint</summary>Include the unseen next observation in the rank denominator.</details>
<details><summary>Solution</summary>k=ceil(15×.8)=12, so q=12, the twelfth smallest score. At α=.02, k=ceil(14.7)=15, larger than the fourteen available scores, so q=infinity. Clipping to 14 is a different procedure and loses the desired rank argument.</details>

### 4. Turn class scores into a set

A three-class predictor returns [.5,.35,.15]. A separately computed threshold is q=.65 for score 1−p_y. Which classes are included? What happens if q decreases to .6? Does a singleton then have 90% conditional correctness?

<details><summary>Hint</summary>Use the weak inequality on scores [.5,.65,.85].</details>
<details><summary>Solution</summary>At .65 include the first and second classes, including equality. At .6 only the first remains. Its singleton status does not establish 90% correctness conditional on that status or input. The nominal conformal coverage belongs to the sampling procedure and specified score, not to each displayed set.</details>

### 5. A negative CQR adjustment

An initial interval is [10,20]. Five calibration CQR scores are [−5,−4,−3,−2,−1] and α=.4. Find the conformalized interval. Why can the score be negative?

<details><summary>Hint</summary>k=ceil(6×.6), and the score measures how far a label lies beyond—or within—the endpoints.</details>
<details><summary>Solution</summary>k=4 and q=−2. The final interval is [10−(−2),20+(−2)]=[12,18]. Negative scores arise when outcomes fall inside their initial intervals; calibration can shrink excess width. This constructed sample does not prove that every future response lies in the shrunken interval.</details>

### 6. A data-flow bug

A team fits a neural network, chooses an epoch using validation labels, fits temperature on those same labels, and computes a split-conformal threshold from the same observations. A new independent test set is untouched. Which step needs redesign for the ordinary split-conformal argument?

<details><summary>Hint</summary>Consider the entire final score function, not only the original network weights.</details>
<details><summary>Solution</summary>The conformal observations helped select the model and fit temperature, so they are not scored symmetrically with new test examples by an independently fixed rule. Complete selection and probability fitting first, then use separate conformal observations, or use a specifically justified alternative procedure. An untouched test set measures the resulting system but does not retroactively validate the threshold construction.</details>

### 7. Evaluate the acoustic intervals

Ridge intervals cover 108/120 observations, with 54/66 below 2,000 Hz and 54/54 above. Write a two-sentence interpretation and a useful next development action without calling either empirical fraction a theorem.

<details><summary>Hint</summary>Separate overall measurement, subgroup measurement, and the sampling interpretation.</details>
<details><summary>Solution</summary>The observed overall coverage is 90%, while the lower-frequency slice is about 81.82% and the higher-frequency slice is 100% on this fixed assessment sample. These measurements identify uneven behavior; they neither prove exact population coverage nor supply a conditional guarantee for either slice. A useful next development experiment examines residual patterns and relevant experimental groups, designs a scale/quantile or group-specific procedure using development data, then obtains an appropriate new independent assessment rather than tuning on this final table.</details>

### 8. Can a narrower set be worse?

One system always returns all five labels. Another returns one label per observation but omits the true label on 30% of them. Which uncertainty report is more useful, and what is missing from that question?

<details><summary>Hint</summary>There is no single scalar objective without a task and error tolerance.</details>
<details><summary>Solution</summary>The first has perfect coverage and no discrimination; the second is selective but misses many answers. Compare coverage, size, consequence of omissions, downstream action and the requested guarantee under a declared task. Neither perfect coverage nor minimal set size alone determines usefulness. A method that targets an appropriate loss or acceptance policy may be needed.</details>

## 11. References and another way to learn

For a spoken explanation, the authors' [A Tutorial on Conformal Prediction](https://www.youtube.com/watch?v=nql000Lu_iE) develops the basic route, and [Part 2: Conditional Coverage and Diagnostics](https://www.youtube.com/watch?v=TRx4a2u-j7M) is the follow-up for interpreting the guarantee. Their [official video listing](https://stephenbates19.github.io/videos.html) identifies the recordings. Reconstruct our ten-card rank argument before moving to the second talk, and compare its conditioning language with the airfoil slices.

For interactive examples and library contracts, use [scikit-learn's calibration guide](https://scikit-learn.org/stable/modules/calibration.html). Read its definition and score discussion before copying a wrapper. [Guo et al.](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf) gives the temperature method and an actual empirical study; [Vaicenavicius et al.](https://proceedings.mlr.press/v89/vaicenavicius19a/vaicenavicius19a.pdf) helps explain why a single binned statistic misses part of calibration.

[Angelopoulos and Bates' tutorial](https://arxiv.org/html/2107.07511v6) offers diagrams, applications, evaluation and the finite-rank proof. Use the formal order-statistic definition and attend to its footnotes about ties; code conventions and a theorem's assumptions must agree. The CQR, weighted conformal and jackknife+ papers linked at their mechanisms are targeted deeper routes, each changing a specific part of the procedure.

The next topic in this module is **Rademacher Complexity & Generalization Bounds**. It returns to the generalization question from PAC/VC with a complexity measure sensitive to sampled inputs. The connection is the habit you have just practiced: define the random object, what was fitted using which information, and the exact event that a probability statement controls.
