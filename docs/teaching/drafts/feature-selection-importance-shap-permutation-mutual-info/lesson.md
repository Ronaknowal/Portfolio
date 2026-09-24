# Feature Selection & Importance: SHAP, Permutation & Mutual Information

A laboratory measures thirteen properties of each sample. A classifier makes useful predictions, but running every assay takes time. Which measurements could the laboratory stop collecting? Now imagine a second request: explain why the classifier gave one particular sample a high score. These sound similar, yet they need different experiments.

The preceding [Regularization](/learn/path/full-curriculum/regularization-l1-l2-elastic-net-dropout?module=classical-ml) lesson showed why different coefficient vectors can represent the same predictor. Here we ask more precisely what an input contributes: to the information in the data, to a fitted model's performance, or to one prediction relative to a reference. That precision makes an explanation useful rather than merely persuasive.

**First pass.** Read sections 1–6, work through the small count, permutation and coalition examples, then try practice 1–6. Section 6 connects them in a reproducible observed-data workflow. Section 7 explores deeper questions about dependence, search and stability; it is optional on a first visit. You need the earlier ideas of training versus validation, a prediction function and an average. Entropy and coalition notation are introduced here before use.

## 1. Ask which question you need to answer

**Feature selection** chooses which inputs a learning procedure will receive. Its goal might be comparable accuracy with fewer measurements, less memory, a simpler model, or a particular scientific investigation. It can help or hurt prediction; fewer columns is not automatically better. Nor does every model use every supplied column. A tree may never split on one of them.

**Feature importance** assigns a quantity to an input under a stated question. There is no single intrinsic importance number waiting inside a dataset.

| Question | What stays fixed? | Suitable starting point |
| --- | --- | --- |
| Does this measurement tell us anything about the target on its own? | Observed distribution and variable definitions | Mutual information or an appropriate univariate statistic |
| How much does this fitted model rely on the correctly paired measurement? | Model, assessment rows and performance metric | Permutation importance |
| Could we train a useful model without this measurement? | Training/assessment procedure; refit the model | Subset comparison or a removal-and-refit experiment |
| How is this prediction allocated relative to this reference? | Model, instance, output scale and missing-feature rule | Shapley/SHAP attribution |
| Would changing this real-world quantity change the outcome? | A stated causal problem | Causal assumptions and evidence beyond these diagnostics |

A large association may come from a measurement taken after the outcome becomes known. That input can score beautifully while being unavailable at prediction time. Establish availability and the unit of prediction before ranking anything. An identifier is not automatically forbidden: a known group identity can be useful in a suitable task. But memorizing unique row IDs will not teach a classifier how to handle unseen rows.

One helpful distinction is **selecting a useful subset** versus **discovering every relevant variable**. Two instruments may measure the same quantity. A low-cost predictor may need only one; a scientific inventory might care about both. State which goal is intended before interpreting an excluded column as irrelevant. The [Guyon–Elisseeff introduction](https://www.jmlr.org/papers/volume3/guyon03a/guyon03a.pdf) develops this distinction and the interaction examples behind it.

**Figure 1 — One dataset, different questions.** Show the same measurement cards entering four experimental routes: rank from training counts; shuffle a column into a fixed predictor; remove a column and refit; explain one prediction relative to background rows. Highlight where fitting occurs and where it does not. A fifth, separate causal-question branch names the additional assumptions and evidence it needs. This is an information-flow diagram, not interchangeable ranking bars.

## 2. Information before fitting: what does one variable reveal?

### Start with counts rather than a formula

Suppose eight observations have binary measurement X and binary target Y:

| | Y=0 | Y=1 | Row total |
| --- | ---: | ---: | ---: |
| X=0 | 3 | 1 | 4 |
| X=1 | 1 | 3 | 4 |
| Column total | 4 | 4 | 8 |

Before observing X, the target is evenly split. After seeing X=0, Y=0 occurs three quarters of the time; after X=1, Y=1 does. The measurement has reduced our uncertainty without perfectly predicting the answer.

For a discrete variable, **entropy** measures the average information required to identify its outcome:

\[
H(Y)=-\sum_y p(y)\log_2 p(y).
\]

A fair binary target has entropy one bit. A certain target has zero. An outcome that is rarer carries more information when it occurs, because −log₂ p is larger. We define a zero-probability contribution as zero by its limiting value; we do not take a literal logarithm of zero in code.

Conditional entropy averages the remaining uncertainty after X is known. In the table, each row has probabilities 3/4 and 1/4, so

\[
H(Y\mid X)= -\tfrac34\log_2\tfrac34-\tfrac14\log_2\tfrac14
\approx0.811278\text{ bits}.
\]

**Mutual information**, or MI, is the reduction:

\[
I(X;Y)=H(Y)-H(Y\mid X)\approx0.188722\text{ bits}.
\]

An equivalent form compares the joint distribution with what independence would predict:

\[
I(X;Y)=\sum_{x,y:p(x,y)>0}p(x,y)\log_2\frac{p(x,y)}{p(x)p(y)}.
\]

For independent variables the numerator equals the denominator in every occupied cell, making the log ratio zero. More generally the weighted sum is a divergence and is nonnegative, even though individual occupied cells can contribute negative terms. Zero population MI characterizes independence. MI is symmetric in X and Y, has no positive/negative direction like correlation, and is not an accuracy percentage.

**Investigation 1 — Change a count, change the information.** Edit the four cell counts, predict whether observing X reduces uncertainty more or less, then reveal row probabilities, each cell's contribution and the total. Compare the 3/1/1/3 table with 4/0/0/4, where one bit is revealed, and 2/2/2/2, where none is. Doubling every count leaves the empirical probabilities and MI unchanged; it does not leave the amount of statistical evidence unchanged. Keep both statements visible.

Here is a complete small calculation. Save as `information_from_counts.py`; it requires NumPy. The same computation was executed in the author's numeric record. Independent execution of this displayed program is part of implementation verification.

```python
import numpy as np

def information_bits(counts):
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 2 or not np.isfinite(counts).all():
        raise ValueError("Use a finite two-dimensional count table.")
    if (counts < 0).any() or counts.sum() <= 0:
        raise ValueError("Counts must be nonnegative with a positive total.")
    joint = counts / counts.sum()
    independent = joint.sum(axis=1, keepdims=True) * joint.sum(axis=0, keepdims=True)
    occupied = joint > 0
    contributions = np.zeros_like(joint)
    contributions[occupied] = joint[occupied] * np.log2(
        joint[occupied] / independent[occupied]
    )
    return contributions, float(contributions.sum())

for table in ([[3, 1], [1, 3]], [[4, 0], [0, 4]], [[2, 2], [2, 2]]):
    cells, total = information_bits(table)
    print(f"MI = {total:.6f} bits")
```

The corresponding totals are 0.188722, 1.000000 and 0.000000 bits.

### A pair can matter even when neither member matters alone

Consider four equally likely states:

| A | B | Y: are the bits different? |
| ---: | ---: | ---: |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

This is XOR. Knowing A alone leaves the two target values equally likely. So does B. Therefore I(A;Y)=I(B;Y)=0. Knowing the pair determines Y, so I((A,B);Y)=1 bit. A ranking that discards every zero-MI individual input would discard both essential parts of this particular mechanism.

This does not make all filters inherently univariate. A filter is independent of the final predictor; it can evaluate joint or conditional information. Univariate filters are a common, economical subclass. Their limitation is the question they ask, not a proof that all non-model criteria ignore interactions.

The opposite issue is redundancy: two exact copies can each reveal the same one bit, while together still reveal only one. Summing their individual MI values double-counts that information. Slightly correlated measurements are subtler: they can still supply complementary signal or independent measurement noise. A correlation threshold alone does not prove one is safe to discard.

**Figure 2 — The XOR square and its projections.** Label the four corners by target. Along either axis, combine the two overlapping target states and show the resulting half/half distribution. The joint square keeps the information that either one-dimensional projection loses. Put the duplicate-variable table beside it only after explaining complementarity, so the learner sees why “correlated” and “jointly useful” are different questions.

### Estimation is not population knowledge

The count-table value is an empirical estimate when counts come from a sample. With eight unique ID categories and a balanced binary target, each observed category has one known label. The empirical MI is one bit even if new IDs carry no target information. That is memorization in the contingency table, not established predictive signal.

Continuous variables need an estimator rather than a literal finite category per distinct value. Binning introduces a resolution choice: very fine bins can memorize, while coarse bins can hide relationships. Nearest-neighbor estimators use local distances instead; choosing neighborhood size trades resolution against estimator variability. Numeric storage does not decide the statistical type: category codes remain discrete, while a rounded measurement may be modeled as continuous if that matches the question and measurement process.

The current [`mutual_info_classif` contract](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html) distinguishes discrete and continuous inputs, uses a seed for tiny tie-breaking perturbations, returns **nats**, and clips negative estimates to zero. A nat uses the natural logarithm; divide by ln 2 to express the same quantity in bits. A returned zero is not a certificate of population independence. Its mixed continuous/discrete estimation is not simply “run the same continuous KSG formula on integer class labels.”

Other useful univariate statistics answer narrower questions. ANOVA's F statistic compares between-class mean variation with within-class variation, so equal class means can hide distributional differences. A contingency-table χ² statistic compares observed categorical counts with independent expected counts. Scikit-learn's `chi2` selector is intended for nonnegative count/frequency-like feature inputs; merely rescaling arbitrary continuous measurements to [0,1] does not establish the inferential assumptions of a count test. A score used for ranking and a calibrated significance test are distinct uses.

## 3. Select a subset with the learning procedure inside the boundary

Three practical families organize the search:

| Family | How it chooses | What to inspect |
| --- | --- | --- |
| Filter | Statistics of the training data, independently of the final fitted predictor | Joint versus univariate criterion, estimator resolution, redundancy |
| Wrapper | Fit and assess candidate input subsets with a chosen learner | Search strategy, complete validation boundary, computation |
| Embedded | Selection is part of fitting, such as an L1-penalized objective or tree split choices | Model assumptions, tuning and selection stability |

**Forward selection** starts from an empty subset and adds the candidate that produces the best chosen validation result. **Backward selection** starts with all available inputs and removes a candidate. These greedy procedures do not reconsider every past decision, so neither guarantees the globally best subset.

**Recursive feature elimination**, or RFE, fits a model, ranks its available inputs by a specified model quantity, removes the weakest and refits. Removing a column can change the ranking of the survivors. RFE's elimination criterion is not itself necessarily a validation score. RFECV adds a cross-validation procedure to select the retained size; its selected CV score still participated in selection. An outer assessment is needed to assess the whole recipe independently. These distinctions and the current selector APIs are documented in the [feature-selection guide](https://scikit-learn.org/stable/modules/feature_selection.html).

The previous regularization lesson already derived why L1 can produce zeros. It does not identify only truly useless inputs, and support can change nonmonotonically along a correlated-data path. A selected subset reflects the objective, feature scale, sample and penalty. A tree's internal importance can also feed a selector, but the importance itself is a score; a threshold or size rule is what turns it into a retained subset.

### See the search fail on a complete tiny world

Use the four equally likely XOR states. For each subset, let a lookup predictor output the most common target among states with those observed values, breaking ties toward zero. Empty, A-only and B-only subsets each achieve 1/2 accuracy; the pair achieves one. These are exact scores over a declared finite world, not held-out experimental estimates.

A forward rule that stops unless accuracy strictly improves never leaves the empty set. A rule forced to retain two inputs reaches the pair, while backward removal from the perfect pair would reject either one-column reduction under the same strict-improvement requirement. The final size and stopping rule are part of the algorithm, not administrative details.

**Investigation 2 — Traverse the subset lattice.** Edit the binary labels attached to the four displayed input states. Before running, predict whether the strict-improvement forward rule will discover a useful subset. Reveal the actual lookup predictions at each subset, then animate only the candidates the chosen rule evaluates. Change the labels to Y=A; now the A-only subset has accuracy one and the search finds it. Changing the order of the same states leaves all exact scores unchanged. A second mode forces two selections, making the original XOR interaction reachable. The interface must distinguish changing the learner's policy from discovering different data.

For a real dataset, the lookup table is replaced by a specified fitted learner and the score by suitable validation. A linear learner without interaction features cannot solve the XOR task just because a wrapper presents both columns. “Wrapper” is not a guarantee of interaction sensitivity.

### Protect the selection boundary

If a training fold selects features by their relation to Y, its selector must see only that fold's training rows and labels. Selecting on the full dataset first allows validation labels to influence the representation. A pipeline inside the outer CV loop enforces the usual fit/transform boundary. However, wrapping an **internally** cross-validated selector after a globally fitted transform can still expose its inner validation rows to that transform. Put every learned operation inside the actual boundary being claimed.

Selection size, thresholds, encodings, feature groups inferred from correlations, and any score-driven domain revision are all choices. If an inspection result causes another choice, those inspection rows become development information. The [previous CV lesson](/learn/path/full-curriculum/cross-validation-hyperparameter-tuning?module=classical-ml) explains nested assessment of the complete procedure. Leakage is a fact about information flow; it is not disproved by one run in which the corrected score happens to rise.

Computation depends on the actual search. Forward selection from d=5 to k=2 assesses 5+4=9 subsets. With three folds, that is 27 fits before a final refit. RFE removing one at a time from five to two fits at sizes five, four and three to choose removals, then fits the retained two-feature model: four fits in that simple protocol. Neither count is “one fit per original feature” universally. Batched removal changes the candidates and cost; a percentage-step implementation's exact convention must be checked.

## 4. Permutation: disturb an input, keep the model fixed

Suppose a fitted predictor has mean squared error 2 on the assessment rows. Copy those rows and shuffle one input column among them, leaving the target and other columns in their original rows. Its error is now 5. Under this shuffle, the error increase is three in squared-target units.

Repeated random donor permutations estimate

\[
\widehat{\operatorname{PI}}_j=
\frac1R\sum_{r=1}^R\left[
\frac1n\sum_i L(y_i,f(x_{i,-j},x_{\pi_r(i),j}))
-\frac1n\sum_i L(y_i,f(x_i))\right].
\]

Here π_r is the donor row ordering. The fitted function f is unchanged. For a higher-is-better score such as accuracy, use original score minus shuffled score instead. A positive value then consistently means performance deteriorated after disturbing the input.

Each shuffle preserves that column's empirical values but disturbs its pairing with targets **and other inputs**. It can leave some rows unchanged and does not make a finite sample perfectly independent. It may also create unusual combinations outside the observed joint distribution. This is an explicit perturbation experiment on a particular model, population sample and metric. It is not an automatically unbiased measure of a feature's causal effect or universal usefulness. The [permutation guide](https://scikit-learn.org/stable/modules/permutation_importance.html) describes the fixed-model procedure and metric-dependent interpretation.

### Redundancy does not make the fitted model refit itself

Let four rows have two identical sensor columns: both equal (−1,−1,1,1). The target is that same vector. Three predictors fit perfectly: f₁(x)=x₁, f₂(x)=x₂, and f₃(x)=(x₁+x₂)/2. Use the declared donor ordering (2,3,0,1), which exchanges the signs.

| Fixed predictor | Shuffle sensor 1: MSE increase | Shuffle sensor 2 | Shuffle both together |
| --- | ---: | ---: | ---: |
| First sensor only | 4 | 0 | 4 |
| Second sensor only | 0 | 4 | 4 |
| Average of sensors | 1 | 1 | 4 |

The first predictor cannot suddenly use sensor 2 after sensor 1 is corrupted; its equation never reads sensor 2. The average predictor already uses both, so corrupting one produces a different result. Removing sensor 1 **and refitting** would be a fourth experiment: a newly trained predictor could use sensor 2 alone. These are not conflicting answers to one question.

For a logical feature group, apply the **same donor ordering to all its columns**. This preserves their within-group pairing while disturbing their relation to the target and remaining groups. Independent permutations inside the group would answer another question and could turn a valid one-hot category into an impossible vector. For a preprocessing pipeline, permuting a raw categorical field before encoding often makes the intended group explicit.

**Investigation 3 — Follow the donor rows.** Edit the sensor rows, target values and coefficients of the fixed linear predictor. Choose one or both sensors to permute and record a predicted error change. Apply a visible donor mapping, then show original and hybrid rows, fixed-equation predictions and per-row error changes. A coefficient of zero supplies an exact null; switching from the first-only predictor to the average supplies a substantive contrast. No hidden refit is allowed.

This table also illustrates model uncertainty. Several models can perform equally well and rely on different inputs. [Fisher, Rudin and Dominici](https://www.jmlr.org/papers/volume20/18-760/18-760.pdf) formalize reliance across a specified set of well-performing models. Their model-class result is more than averaging one model's permutation repeats; the candidate class and allowed loss tolerance matter.

### What the uncertainty bars do and do not show

Repeat-to-repeat variation reflects random donor choices with the **same fitted model and assessment sample**. It does not include uncertainty from new training samples, a different model, different assessment cases or a changed deployment population. More repeats reduce Monte Carlo error in that fixed experiment; they cannot repair the wrong perturbation or a weak assessment design.

A negative measured importance means the shuffled version scored better on this assessment. Sampling variability, harmful fitted reliance or a metric-insensitive effect are possible explanations. Zero importance can arise because a feature is unused, because the metric did not change, or because the chosen perturbations did not alter relevant decisions. It does not alone prove independence from the target. Importance on training rows is permitted but describes training behavior; use suitable unseen assessment rows to investigate predictive behavior beyond fitting.

Tree impurity importance asks something else. For a split node t, weighted impurity reduction is its sample fraction times the parent impurity minus the child-weighted impurities. Sum the reductions at splits using a feature, then apply the estimator's normalization. It records how that particular fitted tree partitioned its training criterion. Many candidate splits can favor chance reductions, so a high-cardinality measurement can receive excessive training importance. Held-out perturbation tests a different property; neither score should be disguised as the other by normalizing all bars to a common-looking scale.

## 5. SHAP: allocate one prediction relative to a declared reference

Permutation importance begins with performance across labeled cases. **Shapley attribution** begins with one output and asks how to allocate its difference from a reference among input features. SHAP applies this idea to model explanations. The allocation can be useful, but it is incomplete until we define what a prediction means when only some inputs are supplied.

### First define the game

Suppose our model is f(a,b)=a+b+ab and the instance is (2,3), with prediction eleven. Use (0,0) as the declared reference. A coalition is simply a subset of features whose instance values are retained; replace the other values by the reference. Write v(S) for that coalition's output:

| Retained instance features S | Input actually evaluated | v(S) |
| --- | --- | ---: |
| None | (0,0) | 0 |
| A | (2,0) | 2 |
| B | (0,3) | 3 |
| A and B | (2,3) | 11 |

If A arrives first, it adds two; B then adds nine. If B arrives first, it adds three; A then adds eight. Average over the two possible arrival orders:

\[
\phi_A=(2+8)/2=5,\qquad \phi_B=(3+9)/2=6.
\]

They add to eleven, the difference from the reference. Each receives its standalone contribution plus half of the interaction six. The model itself remains nonlinear; the additive accounting is for this particular instance and reference.

**Figure 3 — Two arrival orders, one allocation.** Animate A→B and B→A with their actual intermediate inputs and output differences. Then join the two paths into the averages five and six. A waterfall starts at zero and ends at eleven, with no implication that five is the physical effect of manipulating A alone.

For d features, every ordering is equally weighted. If a subset S arrives before feature j, there are |S|! ways to order its members and (d−|S|−1)! ways to order those after j. Dividing by d! gives

\[
\phi_j=\sum_{S\subseteq F\setminus\{j\}}
\frac{|S|!(d-|S|-1)!}{d!}\,[v(S\cup\{j\})-v(S)].
\]

The notation v matters: it is the **specified coalition game**, not an ordinary model f mysteriously accepting missing columns. A telescoping sum along each ordering gives v(F)−v(∅); averaging preserves that total. Thus v(∅)+Σφ_j=v(F), commonly called efficiency or local accuracy when v(F)=f(x).

The Shapley rule also treats players symmetrically when all their marginal contributions match, gives zero to a player that changes no coalition value, and is linear when two games are added. These properties characterize the allocation **for a fixed game**. They do not uniquely choose a background population, certify causal truth, or prove that this is the best explanation for every user. The [original SHAP paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf) explicitly specifies the simplified-input mapping behind its uniqueness result.

### Replace missing coordinates using actual reference rows

One zero reference is often not meaningful. Instead, take a declared background collection B. For every background row, retain the instance values in S and fill the remaining coordinates from that row. Average the model outputs:

\[
v_{x,B}(S)=\frac1{|B|}\sum_{b\in B}f(x_S,b_{\bar S}).
\]

Missing coordinates from the **same donor row stay together**. This preserves dependence among those missing coordinates while breaking dependence between them and the retained values. It is often called a background-replacement or interventional game. It need not be a real-world intervention on the target-generating system.

For f(a,b)=a+b+ab, x=(2,3), and background rows (0,0),(1,1), the four coalition values become 1.5, 3.5, 5 and 11. The Shapley values are now four and 5.5, summing to 9.5 above the new baseline 1.5. **The model and instance prediction did not change.** The reference question changed. Nor is the average prediction necessarily the prediction at the average input: f(.5,.5)=1.25 differs from the average 1.5.

**Investigation 4 — Rebuild a coalition from its donor inputs.** Edit the instance, two background rows and the interaction coefficient in f(a,b)=a+b+γab. Record which feature should receive the larger attribution and what baseline should result. Reveal every hybrid input, its prediction, the coalition average and both ordering paths. Switching γ to zero removes order-dependent interaction increments; changing only a displayed true target leaves this prediction explanation unchanged. Zero-reference and two-row-reference presets reproduce the exact comparisons above.

Save the complete following program as `coalition_attribution.py`. It computes every coalition once, caches its value, then allocates the differences. It is intentionally bounded to small d; it is the mechanism, not a replacement for scalable explainers.

```python
import math
import numpy as np

def explain_background(predict, instance, background):
    instance = np.asarray(instance, dtype=float)
    background = np.asarray(background, dtype=float)
    if instance.ndim != 1 or background.ndim != 2:
        raise ValueError("Use one input vector and a matrix of background rows.")
    if background.shape[1] != len(instance) or len(background) == 0:
        raise ValueError("The background must have matching columns and some rows.")
    if not np.isfinite(instance).all() or not np.isfinite(background).all():
        raise ValueError("This teaching implementation requires finite inputs.")
    dimension = len(instance)
    if not 1 <= dimension <= 10:
        raise ValueError("Exhaustive teaching mode supports one to ten features.")
    values = np.empty(1 << dimension)
    for mask in range(len(values)):
        hybrid = background.copy()
        columns = [j for j in range(dimension) if mask & (1 << j)]
        hybrid[:, columns] = instance[columns]
        outputs = np.asarray(predict(hybrid), dtype=float)
        if outputs.shape != (len(background),) or not np.isfinite(outputs).all():
            raise ValueError("predict must return one finite scalar per input row.")
        values[mask] = outputs.mean()
    phi = np.zeros(dimension)
    for j in range(dimension):
        for mask in range(len(values)):
            if mask & (1 << j):
                continue
            size = mask.bit_count()
            weight = math.factorial(size) * math.factorial(dimension-size-1)
            weight /= math.factorial(dimension)
            phi[j] += weight * (values[mask | (1 << j)] - values[mask])
    return values[0], phi, values

if __name__ == "__main__":
    def model(rows):
        return rows[:, 0] + rows[:, 1] + rows[:, 0] * rows[:, 1]
    for reference in ([[0, 0]], [[0, 0], [1, 1]]):
        baseline, phi, coalitions = explain_background(model, [2, 3], reference)
        print("coalitions", coalitions)
        print("baseline", baseline, "contributions", phi)
        print("reconstructed prediction", baseline + phi.sum())
```

The equivalent author computation gives coalition arrays [0,2,3,11] and [1.5,3.5,5,11], attributions [5,6] and [4,5.5], and reconstructed prediction eleven in both cases. The empty subset is mask zero; A is bit zero and B is bit one. Neither a zero-valued feature nor a missing table entry automatically means an absent player.

### Dependence changes which question the game answers

Another game is observational conditioning:

\[
v_x^{\mathrm{cond}}(S)=E[f(X)\mid X_S=x_S].
\]

It asks what output to expect after learning those observed values under a specified joint distribution. Compare this with taking missing values from unconditional reference rows. Conditional expectations can respect dependence, but estimating them accurately is a separate statistical problem; an exact Shapley sum cannot repair inaccurate conditional estimates.

Let X₁=X₂ be a fair binary variable and let f(x)=x₁. Explain x=(1,1). The baseline is 1/2. Under conditioning, knowing either coordinate reveals both, so v({1})=v({2})=1 and each feature receives 1/4. Under background replacement, learning X₂ while replacing X₁ still averages to 1/2; the attributions are 1/2 for X₁ and zero for X₂.

There is no violation of the dummy-player rule: X₂ changes coalition values in the conditional game by revealing X₁. It does not change them in the replacement game. State whether the explanation concerns information revealed by observations or the model's response to replaced coordinates. [Aas, Jullum and Løland](https://martinjullum.com/publication/aas-2021-explaining/aas-2021-explaining.pdf) develop dependent-feature conditional estimation and distinguish these targets.

[Janzing, Minorics and Blöbaum](https://proceedings.mlr.press/v108/janzing20a/janzing20a.pdf), section 3, analyze this same duplicate-input example from the model-input intervention perspective. Their distinction between the algorithm's output and the real-world outcome is crucial: intervention on the inputs of a known program does not establish the effect of changing the corresponding physical quantities in the world.

Tree-path-dependent SHAP uses a tree's recorded path counts. It is **not generally the true conditional expectation under the data's joint distribution**. Interventional TreeSHAP uses an explicit background and a different game. Current `TreeExplainer` has an `auto` mode that chooses based on whether background data are supplied; specify the intended mode explicitly rather than relying on defaults. “Exact TreeSHAP” means exact for its defined algorithm/game assumptions, not exact causal discovery.

For classifier explanations, name the class and output units. Raw outputs can be margins or log odds for some models; probabilities require a probability-scale game. If baseline plus attributions reconstructs a logit, applying the sigmoid to the **total** yields the probability. Applying it separately to each contribution does not produce additive probability effects. A mean-absolute attribution summarizes magnitude across selected explained rows; it has no label-performance term and therefore cannot tell you what fraction of accuracy survives feature removal.

## 6. A complete observed-data workflow

### Question, data and boundaries

The [UCI Wine dataset](https://doi.org/10.24432/C5PC7J) contains 178 chemical analyses from three cultivars grown in the same Italian region. It provides thirteen numeric measurements and a cultivar label. This is cultivar classification, not prediction of wine quality. The preserved source file has no missing entries. Its supplied names/table do not establish measurement units for all columns, so the visual labels retain source-scale values without inventing physical units.

The thirteen fields are alcohol, malic acid, ash, alkalinity of ash, magnesium, total phenols, flavanoids, nonflavanoid phenols, proanthocyanins, color intensity, hue, OD280/OD315 and proline. The class column is the target and is excluded from the inputs. We retain the provider's source order and all rows.

We ask whether a shallow classifier can use fewer measurements and then inspect a separately declared small model. The boundaries are:

- 138 development rows and forty reserved rows, stratified split seed 51. The forty reserved rows receive no prediction or score here.
- Within development, 100 fitting rows and 38 inspection rows, stratified seed 52.
- Within the 100 fitting rows, three stratified folds, seed 53, compare MI-selected sizes three, six and thirteen. The selector and depth-three tree fit separately in each fold. The tree has a minimum of five fitting cases per leaf.

The inspection rows do not choose the subset size. Once inspected, they are part of what the author knows; use the still-reserved rows or appropriate new data if further changes are made. This small historical collection does not establish performance on a new region, vineyard, laboratory or measurement process. No corresponding grouping metadata supports such an assessment.

For a transparent local explanation, a second model uses four raw fields declared before fitting: alcohol, malic acid, flavanoids and proline. Four inputs allow all sixteen coalitions to be inspected. It is not the MI-selected model, and its smaller input set is not chosen by the inspection score. Both models use the same tree settings and fitting rows.

### Run the selection, then inspect the fixed model

Save this as `wine_feature_study.py` beside `coalition_attribution.py` and the provided `wine.data`. Setup: Python with NumPy and scikit-learn. The author's equivalent bounded calculation ran with Python 3.12.14, NumPy 2.3.5 and scikit-learn 1.9.1; these numbers are recorded observations, not newly executed browser results. The separately assembled displayed programs still require phase-two execution checks.

```python
from pathlib import Path
from functools import partial
import numpy as np
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from coalition_attribution import explain_background

NAMES = ["alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
         "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
         "color_intensity", "hue", "od280_od315", "proline"]

def run_study():
    raw = np.loadtxt(Path(__file__).with_name("wine.data"), delimiter=",")
    features, target = raw[:, 1:], raw[:, 0].astype(int)
    development, reserved = train_test_split(
        np.arange(len(target)), train_size=138, stratify=target, random_state=51
    )
    fitting, inspection = train_test_split(
        development, train_size=100, stratify=target[development], random_state=52
    )
    folds = list(StratifiedKFold(3, shuffle=True, random_state=53).split(
        features[fitting], target[fitting]
    ))
    score_mi = partial(mutual_info_classif, discrete_features=False,
                       n_neighbors=3, random_state=54)

    def build(k):
        return make_pipeline(
            SelectKBest(score_mi, k=k),
            DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=55)
        )

    candidate_scores = []
    for k in [3, 6, 13]:
        fold_scores = []
        for local_fit, local_validation in folds:
            fit_ids, validation_ids = fitting[local_fit], fitting[local_validation]
            candidate = build(k).fit(features[fit_ids], target[fit_ids])
            fold_scores.append(candidate.score(features[validation_ids], target[validation_ids]))
        candidate_scores.append((k, float(np.mean(fold_scores))))
        print("retained size", k, "mean CV accuracy", round(np.mean(fold_scores), 6))
    selected_k = max(candidate_scores, key=lambda pair: pair[1])[0]
    selected = build(selected_k).fit(features[fitting], target[fitting])
    selected_names = np.array(NAMES)[selected[0].get_support()].tolist()
    print("selected names", selected_names)
    print("selected-model inspection accuracy",
          selected.score(features[inspection], target[inspection]))

    small_columns = [0, 1, 6, 12]
    small_names = np.array(NAMES)[small_columns].tolist()
    small_features = features[:, small_columns]
    small_model = DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=5, random_state=55
    ).fit(small_features[fitting], target[fitting])
    original_prediction = small_model.predict(small_features[inspection])
    original_accuracy = np.mean(original_prediction == target[inspection])
    majority = np.argmax(np.bincount(target[fitting]))
    print("majority inspection accuracy", np.mean(target[inspection] == majority))
    print("four-field inspection accuracy", original_accuracy)
    donors = [np.random.default_rng(56 + r).permutation(len(inspection)) for r in range(20)]
    for column, name in enumerate(small_names):
        drops = []
        for donor in donors:
            changed = small_features[inspection].copy()
            changed[:, column] = changed[donor, column]
            shuffled_accuracy = np.mean(small_model.predict(changed) == target[inspection])
            drops.append(original_accuracy - shuffled_accuracy)
        print(name, "accuracy drop", round(np.mean(drops), 6),
              "permutation SD", round(np.std(drops), 6))

    reference_ids = np.sort(fitting)
    class_index = int(np.flatnonzero(small_model.classes_ == 1)[0])
    def probability_one(rows):
        return small_model.predict_proba(rows)[:, class_index]
    instance = small_features[inspection[0]]
    baseline, phi, coalition_values = explain_background(
        probability_one, instance, small_features[reference_ids]
    )
    print("explained source row", inspection[0], "class-1 probability", coalition_values[-1])
    print("background prediction", baseline, "contributions", phi)
    print("reconstruction error", baseline + phi.sum() - coalition_values[-1])
    return small_model, small_features, fitting, inspection, small_names

if __name__ == "__main__":
    run_study()
```

All Wine measurements are treated as continuous inputs to this MI estimator, following the source's measurement description; the target is discrete. Tree splits do not require standardization in this example. This is a declared teaching estimator choice, not a universal rule for integer-valued measurements. The size tie rule prefers the first, smaller listed size. The record contains nine selection-CV fits, one selected refit and one four-feature refit: eleven fits total.

### Read the recorded outcomes without inventing a winner

| MI-selected size | Correct counts in the three validation folds | Mean fold accuracy |
| ---: | --- | ---: |
| 3 | 30/34, 25/33, 26/33 | 0.809269 |
| 6 | 30/34, 29/33, 26/33 | 0.849673 |
| 13 | 30/34, 26/33, 28/33 | 0.839572 |

The rule selects six. Its final training-selected fields are alcohol, total phenols, flavanoids, color intensity, OD280/OD315 and proline. Across the three inner folds, the six retained identities are not identical. The column count is one hyperparameter; each fitted selector still learns a specific subset from its own rows.

The selected model classifies 36/38 inspection rows correctly, as does the independently declared four-field model. The training-majority baseline, always predicting class 2, is correct on 15/38. These small-sample results demonstrate the protocol and a possible measurement reduction. They do not prove that these are the best six chemical assays, that the two models are equivalent, or that the tiny CV difference is statistically decisive.

**Figure 4 — Selection changes a set, not just a score.** Place the actual fold-by-feature retained/not-retained matrix beside the size/accuracy table. Use three connected points only for the three tested sizes; do not interpolate an optimal count between them. Separate the final selected set from the fold-specific sets and the separately declared four-field model.

For the fixed four-field model, the recorded inspection permutation results are:

| Field | Mean accuracy decrease | SD across twenty donor permutations | Tree impurity importance |
| --- | ---: | ---: | ---: |
| Alcohol | 0.221053 | 0.048809 | 0.402897 |
| Malic acid | 0 | 0 | 0 |
| Flavanoids | 0.359211 | 0.082076 | 0.460171 |
| Proline | 0.189474 | 0.029539 | 0.136932 |

Accuracy decrease is a proportion: 0.221053 corresponds to about 22.1 percentage points in this averaging scheme. The impurity column is normalized training-criterion reduction. These are different units and mechanisms; no common heatmap should pretend the numbers measure the same thing. The tree never splits on malic acid, so changing that coordinate leaves its prediction function unchanged. That is a verified property of this fitted tree, not a statement that malic acid has no association with cultivar.

### Explain one actual prediction from actual coalition evaluations

Source row 104 (zero-based file index), the first inspection row, has (alcohol, malic acid, flavanoids, proline)=(12.51,1.73,1.92,672). The four-feature tree predicts class 2; its class-1 probability is zero. We explain the class-1 probability, using all 100 fitting rows as the background. Their average class-1 prediction is 0.33.

| Contribution | Probability units |
| --- | ---: |
| Baseline | +0.33 |
| Alcohol attribution | −0.45 |
| Malic acid attribution | 0 |
| Flavanoids attribution | +0.07 |
| Proline attribution | +0.05 |
| Reconstructed class-1 probability | 0 |

The negative attribution can have magnitude greater than the baseline because positive contributions offset part of it. An individual attribution is not a probability and need not lie between zero and one. The final probability does.

The alcohol-only coalition has value zero; the flavanoids-only coalition has value 0.42, proline-only 0.38, and flavanoids-plus-proline 0.62. Every coalition retaining this instance's alcohol has value zero in the saved tree. Averaging the marginal differences across all orderings yields the table, with floating-point reconstruction error about 1.1×10⁻¹⁶.

**Figure 5 — From a hybrid row to a probability waterfall.** Show the actual small tree, a selected coalition's one hundred hybrid rows, their leaf outcomes, and their averaged class-1 output. Let the learner inspect individual donors before looking at the waterfall. A companion scatter shows the twelve predeclared explained inspection rows, one actual point per row, rather than invented evenly spaced attribution dots.

**Investigation 5 — A changed sample takes a different tree path.** Edit the four source-scale measurements to update the class-1 probability and attribution immediately. From row 104, changing alcohol from 12.51 to 13.5 while keeping the other fields fixed makes the saved tree output one; its attributions become approximately (+0.216667,0,+0.206667,+0.246667) above the same 0.33 reference. Changing only malic acid from 1.73 to 4.1 leaves the prediction and all coalition values unchanged. Follow the actual split comparisons and hybrid donor outcomes to see why. These are model-input scenarios; the fitted tree and reference do not retrain after an edit.

For a deliberate reference contrast, use the twelve fitting rows with the largest source indices. The original file is arranged by class, so this is a class-3 cohort, **not a representative population sample**. Its baseline class-1 prediction is zero. For the same row 104, the contributions become approximately (−0.416667,0,+0.375,+0.041667), still reconstructing zero. The statement has changed from comparison with the whole fitting reference to comparison with that cohort. Selecting reference rows is a substantive explanation decision, not an invisible speed trick.

For a production tree explainer, the optional following program makes the same mode, background and output choice explicit. Save as `check_tree_explanation.py` beside the two prior programs. It additionally requires a compatible current `shap` installation; no SHAP package execution is claimed in this content phase, and no fabricated library stdout is attached. Phase two should record its version and check the comparison. The small exhaustive program remains the independent reference computation.

```python
import numpy as np
import shap
from coalition_attribution import explain_background
from wine_feature_study import run_study

model, features, fitting, inspection, names = run_study()
background = features[np.sort(fitting)]
rows = features[inspection[:12]]
explainer = shap.TreeExplainer(
    model, data=background, feature_perturbation="interventional",
    model_output="probability", feature_names=names
)
explanations = explainer(rows)
class_index = int(np.flatnonzero(model.classes_ == 1)[0])
class_one = explanations[:, :, class_index]
probabilities = model.predict_proba(rows)[:, class_index]
np.testing.assert_allclose(class_one.base_values + class_one.values.sum(axis=1),
                           probabilities, rtol=0, atol=1e-6)
baseline, exact_phi, _ = explain_background(
    lambda values: model.predict_proba(values)[:, class_index], rows[0], background
)
np.testing.assert_allclose(class_one.values[0], exact_phi, rtol=0, atol=1e-6)
np.testing.assert_allclose(class_one.base_values[0], baseline, rtol=0, atol=1e-6)
shap.plots.waterfall(class_one[0])
```

Current [`TreeExplainer` documentation](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html) describes its dependence modes, model-dependent raw outputs and multi-output shapes. Explicitly selecting the class avoids confusing a samples×features×classes result with a two-dimensional attribution matrix. A reconstruction check is necessary numerical evidence; by itself it cannot certify that the background or explanatory question is appropriate.

## 7. Deeper branches: choose the right information and explanation

### Conditional information and measurement budgets

Once a subset S is available, the additional information in candidate X_j is

\[
I(X_j;Y\mid X_S)=H(Y\mid X_S)-H(Y\mid X_S,X_j).
\]

In XOR, I(B;Y)=0 but I(B;Y|A)=1 bit. For an exact copy B=A, conditioning on A makes B contribute no additional information. This gives a principled target for complementarity, but estimating high-dimensional conditional information from few rows is difficult. A formula does not eliminate sample requirements or measurement error.

A Markov blanket B for target Y is a set that makes Y conditionally independent of the remaining inputs given B, under the stated distribution. That is stronger than selecting the largest marginal MI values or dropping highly correlated pairs. The later [Bayesian Networks & Causal Graphical Models](/learn/path/full-curriculum/bayesian-networks-causal-graphical-models?module=classical-ml) gives graph semantics and the assumptions connecting graph structure to such conditional relationships.

For a measurement budget, a subset can be assessed by prediction loss and acquisition cost together. A hypothetical pair of sensors might cost 2 and 8 units; an equally accurate substitute costing 3 could be useful even if a global importance bar is smaller. Cost may attach to a group: once an assay panel is run, several outputs may have almost no extra acquisition cost. Keeping one feature from every panel may therefore save fewer resources than keeping a larger number from one cheap panel. These are declared engineering costs, not costs inferred from attribution magnitudes.

Unsupervised selection has a different target again. Removing a constant training column is often useful; removing every low-variance column can discard a rare alarm that matters greatly for a high-cost event. Without target labels, use a declared objective such as reconstruction, reliable measurement or clustering quality. Earlier PCA and NMF construct new coordinates from inputs rather than simply selecting old columns. They may reduce dimension without reducing the number of raw sensors that must be collected.

### Stability, significance and repeated selection

A selector can be unstable while prediction is stable. The duplicate-sensor example makes that possible without any estimation noise. Across resampled training sets, record which inputs are selected and how their predictions compare on appropriate assessment cases. A selection frequency is a descriptive property of that resampling-and-fitting procedure, not automatically the probability that the feature is truly relevant. Formal stability-selection error bounds need additional assumptions and a specified algorithm.

Significance asks how unusual a statistic would be under a specified null model. An importance score alone is not a p-value. If one runs one hundred independent valid null tests at level 0.05, the probability of at least one false rejection is 1−0.95¹⁰⁰≈0.9941. Dependence changes that arithmetic; selection after seeing the results introduces further issues. False-discovery or family-wise procedures can be appropriate for valid input p-values, but they do not turn an arbitrary MI estimate into a significance test.

Random probe features can reveal suspicious overfitting behavior; consistently outranking a few such probes does not by itself prove relevance or guarantee an error rate. Do not remove inconvenient observations solely because they weaken the desired importance story. Investigate measurement and label quality using a documented rule. Selecting rows as well as columns is another part of the learning procedure and its assessment boundary.

A beeswarm spread over many explained cases measures variation of attribution **across those cases**. It is not a confidence interval over new training fits. A plot of importance across CV fits, a plot across permutation repeats and a plot across individual predictions visualize different distributions. Name which one is shown. The next Bias–Variance lesson develops why changing training data can change a fitted function even when its average performance is similar.

### Grouping players changes the attribution game

Grouping columns for a valid perturbation is often useful, but “add their individual Shapley values” and “treat the group as one player” can differ. Consider three players A,B,C with value one only when all three are present, and zero for every other coalition. Each individual Shapley value is 1/3, so A+B sum to 2/3. Now form two players: group G={A,B}, and C. Both are needed for the value one, so each grouped player receives 1/2. The possible arrival orders changed.

Use summed individual values when that is the declared aggregation, and grouped-game values when groups are the explanatory players. A valid one-hot feature can be treated as one original variable, avoiding impossible partial-category coalitions. Feature engineering can create similar choices: explaining alcohol and alcohol² as separate players is different from explaining the single raw alcohol measurement that generates both. The decision should follow what the learner or application regards as a meaningful change.

### Scale computation to the mechanism

For d raw inputs, exhaustive coalition enumeration requires 2ᵈ coalition values per explained instance; background replacement additionally evaluates B hybrid rows per coalition. Cached coalition values avoid recomputing the same subset for every feature. The local Wine calculation uses sixteen coalitions and one hundred background rows per instance, not an exponential native search over all thirteen source columns.

Sampling arrival orders estimates Shapley values using their average marginal increments; sampling coalitions with the Shapley kernel instead yields a weighted regression problem. These are related approximations, not identical algorithms. KernelSHAP uses special coalition-size weights and constraints at empty/full coalitions. Its cost depends on the evaluated coalitions, background size, model prediction cost and regression solve; there is no universal O(d²) end-to-end guarantee or fixed number of seconds.

Tree-specific algorithms exploit shared paths. The [original TreeSHAP algorithm](https://arxiv.org/pdf/1802.03888) gives an O(TLD²) bound per explained instance for T trees, at most L leaves and depth D in that algorithm; background-dependent variants and implementations add their own costs. That expression is not a benchmark for a million rows or a guarantee for every dependence model. Inspect the actual supported algorithm, approximation setting and output scale, then measure its cost on the intended workload. Never run large background explanations on every page render.

For differentiable models, gradients measure local sensitivity, while integrated gradients accumulates gradients along a specified path from a reference. Deep SHAP and gradient-based approximations make additional choices. They are useful extensions, but sharing an additive-looking plot does not make all methods exact Shapley estimators. Nor are attention weights automatically causal or faithful explanations. A later interpretability investigation should define its intervention, reference and evaluation just as carefully as this tabular lesson.

Finally, a global sum of mean absolute SHAP values is an attribution-magnitude total over a declared set of predictions. If you plot a cumulative **fraction** of that total, it must finish at one when the total is positive. If every attribution is zero, the fraction is undefined and should be labeled accordingly. Even a correct 95% cumulative fraction does not imply 95% retained accuracy, explained variance or information. For a measurement-removal decision, refit and assess the reduced-input recipe.

## 8. Practice: change the question and calculate its answer

Attempt the first six before opening solutions. The remaining questions use the deeper branches.

### 1. A different count table

The counts are [[2,0],[0,6]]. What are H(Y), H(Y|X) and I(X;Y) in bits? Why is perfect prediction now less than one bit of information?

<details><summary>Hint</summary>The target probabilities are 1/4 and 3/4. Conditional on either occupied row, the answer is certain.</details>
<details><summary>Solution</summary>H(Y)=−(1/4)log₂(1/4)−(3/4)log₂(3/4)≈0.811278 bits. Conditional entropy is zero, so MI is 0.811278 bits. Perfectly revealing a target cannot reveal more uncertainty than it originally had; this target is not balanced.</details>

### 2. A feature that becomes useful later

For fair independent A and B with Y=A XOR B, give I(A;Y) and I(B;Y). After A is known, how many additional bits about Y does B reveal? Would adding a perfect duplicate C=A create a second independent bit about Y? The optional deeper section names the additional-information quantity I(B;Y|A).

<details><summary>Solution</summary>The first two values are zero and the conditional value is one bit. C supplies no information beyond A because it is determined by A. A and B together determine Y; copying A does not create new target information.</details>

### 3. A partial permutation

Use the duplicate rows (−1,−1),(−1,−1),(1,1),(1,1), target (−1,−1,1,1), and predictor (x₁+x₂)/2. Shuffle only the first column using donors (0,2,1,3). What is the MSE increase? What happens if both columns use that same permutation?

<details><summary>Solution</summary>The first-column perturbation gives predictions (−1,0,0,1), with squared errors (0,1,1,0), so MSE rises by 0.5. Perturbing both gives (−1,1,−1,1), squared errors (0,4,4,0), and increase two. Some donor rows preserve values; a shuffle need not alter every case.</details>

### 4. Change the interaction strength

For f(a,b)=a+b+2ab, instance (1,2), and reference (0,0), calculate all four coalition values and the Shapley values. Then explain why the A-first increment differs from the A-second increment.

<details><summary>Solution</summary>Coalition values are 0,1,2,7. A receives (1+5)/2=3 and B receives (2+6)/2=4. The interaction term contributes four only after both variables are present; averaging arrival orders allocates two of that interaction to each.</details>

### 5. Information flow rather than a score test

A colleague selects the six highest-MI inputs using every development label, then cross-validates a model on those columns. After moving the selector inside the folds, the score unexpectedly increases slightly. Was the original procedure free of leakage?

<details><summary>Solution</summary>No. Validation labels influenced the original selected representation regardless of the observed score direction. The corrected experiment changes the information boundary; one realized difference is not a universal test for whether leakage occurred. Preserve an independent assessment of the full selection procedure.</details>

### 6. Read the actual Wine result

Malic acid has zero permutation importance and zero replacement SHAP in the four-field tree. What precisely has been established? A proposed reduced model is judged only by retaining 95% of mean absolute SHAP. What calculation is missing?

<details><summary>Solution</summary>The saved tree never uses malic acid, so replacing that coordinate cannot change its output. This does not establish population independence or that no other model can use it. To judge a reduced measurement set, fit the reduced-input learning procedure and assess it on appropriate protected data; attribution magnitude is not a retained-performance guarantee.</details>

### 7. Explain a different output — deeper

A binary model's raw margin explanation has baseline −0.2 and contributions +0.8 and −0.1. What is the reconstructed margin and its probability under the sigmoid? Can the two contributions be separately passed through the sigmoid and added?

<details><summary>Solution</summary>The margin is 0.5 and sigmoid(0.5)=1/(1+exp(−0.5))≈0.622459. The sigmoid is nonlinear, so transforming and adding the individual terms does not yield an additive probability explanation. A probability-space coalition game must explain that output directly.</details>

### 8. Count the search fits — deeper

Forward selection chooses three of six inputs using fourfold validation. Count candidate fits and one final refit. Compare simple RFE removing one at a time from six to three, including its final retained-model fit and no CV.

<details><summary>Solution</summary>Forward selection assesses 6+5+4=15 candidate subsets, with four fits each: sixty plus one final refit. RFE fits six-, five- and four-feature models to select removals, then the final three-feature model: four fits. These methods use different selection quantities and do not have identical statistical guarantees.</details>

### 9. Change the players — deeper

Four players A,B,C,D receive value one only when all four are present. Compare the sum of individual A+B+C Shapley values with the value assigned when G={A,B,C} is one player and D the other.

<details><summary>Solution</summary>The individual game is symmetric, so each receives 1/4 and the sum is 3/4. The two-player game is symmetric, so G receives 1/2. Grouping changes possible arrival orders and the allocation target; it is not generally the same as summing afterward.</details>

### 10. Design a useful deployment investigation — deeper

A factory wants to replace an expensive sensor with two cheaper correlated sensors. A fitted model gives the expensive sensor the largest permutation score. Propose a comparison that answers the factory's question, including the unit of assessment and actual costs.

<details><summary>Solution</summary>Define the future task, such as predicting faults on new machines or later operating periods, and split by that unit and information availability. Compare predeclared full and cheaper-sensor learning procedures, fitting every transform/selection stage within their development folds. Assess the locked procedures on protected machines or later periods, using relevant error costs and actual acquisition/maintenance costs. A fixed-model permutation tests present reliance; it does not evaluate the retrained substitute. Preserve calibration/threshold assessment if decisions depend on predicted probabilities.</details>

## 9. Readiness and what comes next

You are ready to continue when you can distinguish data information, fixed-model reliance, removal-and-refit performance and local attribution; calculate MI from a small table; trace a permutation donor; derive a two-feature Shapley allocation; and state the reference, output units and protected data boundary in an actual analysis. You should be able to explain why an excluded feature can still be informative and why an exact explanation can still answer the wrong question.

Next is [Bias–Variance Tradeoff & Learning Curves](/learn/path/full-curriculum/bias-variance-tradeoff-learning-curves?module=classical-ml). It develops what changes when training samples change, why prediction and selection stability can differ, and how to read learning curves. After that, Imbalanced Learning connects the metric and measurement choices here to rare events and unequal error costs.

## References and other ways to learn

- [Guyon and Elisseeff, An Introduction to Variable and Feature Selection](https://www.jmlr.org/papers/volume3/guyon03a/guyon03a.pdf): sections 2–4 give small geometrical examples of individual versus joint usefulness and subset search. Sections 5–7 connect construction, validation, stability and scientific discovery. Read the geometric examples alongside our exact XOR world; historical implementation recommendations need their assumptions checked.
- [Scikit-learn feature-selection guide](https://scikit-learn.org/stable/modules/feature_selection.html): the current map of variance filters, univariate selectors, RFE/RFECV, embedded selectors and sequential search. Useful when translating a declared selection question into an API.
- [Mutual information for classification](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html): estimator inputs, discrete/continuous flags, reproducibility and nat units. It is an estimator contract, not a promise to reveal every interaction.
- [Permutation feature importance](https://scikit-learn.org/stable/modules/permutation_importance.html): fixed-model algorithm and the role of the metric and assessment data.
- [Permutation with correlated features: visual/code example](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html): inspected tree/permutation plots, correlation dendrogram and a removal-and-refit comparison. Its correlation grouping is exploratory and uses the full X; for a protected assessment, fit grouping within the development boundary and choose a linkage appropriate to the distance. Its particular result is not a universal fallback rule for every fixed model.
- [Lundberg and Lee, A Unified Approach to Interpreting Model Predictions](https://proceedings.neurips.cc/paper_files/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf): sections 2–4 define the additive explanation, assumptions and coalition weighting. Our two-order calculation is preparation for its kernel-regression formulation.
- [SHAP waterfall notebook](https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/waterfall.html): a visual/code route for reading a single prediction decomposition, its background and log-odds units, then comparing individual cases. The inspected notebook also discusses why a striking pattern needs further investigation rather than instant causal interpretation.
- [TreeExplainer API](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html): explicitly choose dependence mode, background and output scale; inspect multi-output shapes before plotting.
- [Lundberg, Erion and Lee, Consistent Individualized Feature Attribution for Tree Ensembles](https://arxiv.org/pdf/1802.03888): section 3 explains shared-path computation and the stated TreeSHAP complexity; section 4 extends the game to interaction allocations. These algorithmic bounds are more informative than an unsupported timing comparison.
- [Aas, Jullum and Løland, Explaining Individual Predictions When Features Are Dependent](https://martinjullum.com/publication/aas-2021-explaining/aas-2021-explaining.pdf): the conditional-game construction and the separate problem of estimating missing-feature distributions.
- [Janzing, Minorics and Blöbaum, Feature Relevance Quantification in Explainable AI](https://proceedings.mlr.press/v108/janzing20a/janzing20a.pdf): section 3 distinguishes an intervention on a program's input from a causal claim about the world and explains the duplicate-input example.
- [Fisher, Rudin and Dominici, Model Class Reliance](https://www.jmlr.org/papers/volume20/18-760/18-760.pdf): sections 2–4 expand the question from one fitted model to a declared collection of well-performing models. Their formal reliance ratios and bounds have assumptions beyond our exact zero-loss demonstration.
- [UCI Wine dataset](https://doi.org/10.24432/C5PC7J): original description, attribution and CC BY 4.0 license. The offline packet preserves every numeric source row and the exact study split, model and attribution records.
