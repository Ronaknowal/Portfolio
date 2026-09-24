# Bias–Variance Tradeoff & Learning Curves

A model makes an error. Should you collect more examples, change its inputs, simplify it, or let it fit a richer relationship? Those actions solve different problems. This lesson gives you a way to reason about them and then check that reasoning against data.

Begin with a distinction: **one fitted model's mistake, a learning procedure's sensitivity to its training data, and the uncertainty remaining in the target are different quantities.** A learning curve helps investigate them; it does not directly display all three.

First pass: §§1–6 and practices 1–5. You will calculate a small decomposition, read three different kinds of curve, and propose a controlled next experiment. Sections 7–9 explain training optimism, classification and double descent; their additional mathematics is a deeper route. The earlier [regularization](/learn/path/full-curriculum/regularization-l1-l2-elastic-net-dropout?module=classical-ml), [feature selection](/learn/path/full-curriculum/feature-selection-importance-shap-permutation-mutual-info?module=classical-ml) and [cross-validation](/learn/path/full-curriculum/cross-validation-hyperparameter-tuning?module=classical-ml) lessons supply useful connections, refreshed where needed.

## 1. Hold the question still; change the training data

Imagine calibrating a sensor. At a fixed input setting, the average correct output is 10 units. You repeatedly collect a small training sample and fit the same procedure. Its predictions at that input might be 8, 10 and 12.

The average prediction is 10. There is no average offset here, but individual fitted models differ. Another procedure might always predict 9: it is more stable, yet systematically low.

Use these original finite teaching distributions, with each listed prediction equally likely and a fresh observed output of 9 or 11, also equally likely:

| Procedure | Predictions across training draws | Mean prediction | Squared offset from 10 | Prediction variance | Fresh-target noise variance | Expected squared error |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | 8, 10, 12 | 10 | 0 | \(8/3\) | 1 | \(11/3\) |
| B | 9, 9, 9 | 9 | 1 | 0 | 1 | 2 |

For A, average \((8-10)^2,(10-10)^2,(12-10)^2\): the result is \(8/3\). For B, every fitted prediction has the same offset −1. B has lower expected error in this example even though its average prediction is less accurate.

Picture a vertical ruler at the chosen input. Put the true mean at 10, predictions from different training samples as three marked dots, and their average as a separate tick. The distance from the average tick to 10 is the **bias**; the spread of prediction dots around their own average supplies the **variance**. A separate pair of target-outcome dots at 9 and 11 represents noise. Keeping these two kinds of dots separate prevents “the model is uncertain” from becoming an explanation for every source of error.

The learning procedure includes the model family, preprocessing, regularization, optimization and any randomization. Bias is a property of that procedure under a specified training-sampling process, not just a label attached to “linear” or “complex.” A richer model can have bias from shrinkage or incomplete optimization. A simpler model can be unbiased at a particular input even if it misses the relationship elsewhere.

Try editing the three prediction dots in the first investigation. Watch the total expected error change directly. Moving one dot toward the true mean can change both the average offset and the spread; the two terms must be recalculated together.

Nor does low variance mean low error. A broken program returning zero for every input can be perfectly stable.

## 2. Why the three terms add

At a fixed input \(x\), define

\[
f(x)=\mathbb E[Y\mid X=x],\qquad
\sigma^2(x)=\operatorname{Var}(Y\mid X=x).
\]

The first is the population's mean target at that input. The second describes how actual targets vary around that mean. Neither is normally known from a single real dataset.

Let \(D\) denote a random training dataset, including algorithmic randomness if relevant, and let \(\hat f_D(x)\) be the resulting prediction. Its average over repeated training draws is \(\bar f(x)=\mathbb E_D[\hat f_D(x)]\). For a fresh target independent of training given \(x\), with finite second moments,

\[
\mathbb E_{D,Y\mid x}\!\left[(Y-\hat f_D(x))^2\right]
=\underbrace{(\bar f(x)-f(x))^2}_{\text{squared bias}}
+\underbrace{\mathbb E_D[(\hat f_D(x)-\bar f(x))^2]}_{\text{prediction variance}}
+\underbrace{\sigma^2(x)}_{\text{target noise}}.
\]

Here is the mechanism, rather than a formula to memorize. Write the error as

\[
Y-\hat f_D(x)
=\underbrace{Y-f(x)}_{\text{fresh target deviation}}
+\underbrace{f(x)-\bar f(x)}_{\text{fixed offset}}
+\underbrace{\bar f(x)-\hat f_D(x)}_{\text{training-draw deviation}}.
\]

Square the sum. Each squared term has the meaning above. The mixed terms have expectation zero: the first and third deviations each have mean zero, and the fresh target deviation is independent of the trained prediction under the stated experiment. Thus the three contributions remain after averaging.

For all test inputs, average this identity over the intended test-input distribution. Noise may depend on \(x\); a constant noise line is appropriate only if that variance is constant. Equal weighting of a plotted grid measures error on that grid, not automatically error under the population's input frequencies.

The identity is about **expected squared error on fresh outcomes**. It does not say that training error equals bias squared plus noise, that every observed test error exceeds the noise floor, or that complexity must move bias and variance in opposite directions. The training–validation distinction returns in §7.

### “Irreducible” depends on the information available

Suppose a hidden setting \(Z\) is equally likely to be −1 or +1, and \(Y=X+Z+\varepsilon\), with independent noise of variance .25. Given only \(X\), the best mean predictor is \(X\), and remaining variance is \(1+.25=1.25\). If the setting \(Z\) is measured before prediction, the best mean becomes \(X+Z\), leaving .25.

The new feature changes the conditioning information. It does not refute the old noise floor; it defines a better-informed problem. Improving measurement may also change the target itself. Conversely, repeatedly fitting a larger model to exactly the same inputs cannot explain an independent future noise draw.

For a deeper statement, conditioning on \(X=x\),

\[
\operatorname{Var}(Y\mid X)
=\mathbb E[\operatorname{Var}(Y\mid X,Z)\mid X]
+\operatorname{Var}(\mathbb E[Y\mid X,Z]\mid X).
\]

The second term is the part that observing \(Z\) can explain in this example. It connects useful feature acquisition to the decomposition.

## 3. Retrain every possible tiny dataset

A simulation can reveal bias because we choose the truth. A real learning curve cannot usually do that. Let us first make the controlled case fully inspectable.

Use three fixed training inputs, \([-1,0,1]\), and a true response
\[
f(x)=1+x+c x^2.
\]
At each training input, independently add either \(-\sigma\) or \(+\sigma\). There are exactly \(2^3=8\) equally likely training datasets. Fit a constant, a line or a quadratic by least squares to every dataset. This experiment varies the training outcomes while holding the design points fixed; it is neither a bootstrap nor a simulation of random input locations.

For \(c=1,\sigma=.5\), the noiseless training targets are \([1,1,3]\). One possible noisy set is \([.5,1.5,2.5]\). All eight sets matter when defining the average fitted prediction.

At the test input \(x=.5\), the true mean is 1.75. The three procedures give:

| Fit | Mean prediction | Squared bias | Variance | Noise | Expected squared error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Constant | \(5/3\) | \(1/144\) | \(1/12\) | \(1/4\) | \(49/144=.340278\) |
| Line | \(13/6\) | \(25/144\) | \(11/96\) | \(1/4\) | \(155/288=.538194\) |
| Quadratic | \(7/4\) | 0 | \(23/128\) | \(1/4\) | \(55/128=.429688\) |

The constant is best **at this particular input**. Its small bias is partly a coincidence: its average level lies near the truth there. At \(x=0\), its squared bias is \(4/9\), and the quadratic has lower expected error. Never turn one probe point into a global model ranking.

The arithmetic behind the quadratic is particularly revealing. Its prediction at .5 is
\[
-.125\,y_{-1}+.75\,y_0+.375\,y_1.
\]
These interpolation weights sum to one. The mean prediction is 1.75; independence of the three noises gives variance
\[
.25\{(-.125)^2+.75^2+.375^2\}=23/128.
\]
One negative weight is normal for polynomial interpolation. It also shows why changing an observed target can move another prediction in the opposite direction.

Change the true curvature to \(c=0\), keeping the same inputs and noise. A line now has zero bias at .5 and expected error \(35/96=.364583\); the quadratic still has extra prediction variance and error \(55/128=.429688\). Extra flexibility did not buy a better approximation because a line already represents this truth.

### Complete calculation

Save this as a Python file and run it with NumPy. Arrays use rows for training-dataset realizations and columns for probe inputs. The variance divisor is 8 because we enumerate an entire equally weighted finite distribution; it is not an unbiased-sample-variance estimate.

A degree-\(d\) fit represents \(b_0+b_1x+\cdots+b_dx^d\). The design matrix has three rows and \(d+1\) columns; each column supplies one power of the input. Least squares chooses coefficients to minimize the sum of squared differences from the three targets. Solving for all eight target columns produces a coefficient array of shape \((d+1,8)\); evaluating at two probes gives eight predictions at each probe. This is why the program transposes its final product before averaging across training worlds.

~~~python
from itertools import product
import numpy as np

train_x = np.array([-1., 0., 1.])
probe = np.array([0., .5])
curvature, sigma = 1., .5
signs = np.array(list(product([-1., 1.], repeat=len(train_x))))
targets = 1 + train_x + curvature * train_x**2 + sigma * signs
truth = 1 + probe + curvature * probe**2

for degree in (0, 1, 2):
    design = np.vander(train_x, degree + 1, increasing=True)
    coefficients = np.linalg.lstsq(design, targets.T, rcond=None)[0]
    predicted = (
        np.vander(probe, degree + 1, increasing=True) @ coefficients
    ).T
    mean = predicted.mean(axis=0)
    bias_squared = (mean - truth)**2
    variance = predicted.var(axis=0)
    expected_error = ((predicted - truth)**2).mean(axis=0) + sigma**2
    print(degree, np.round(mean, 6), np.round(bias_squared, 6),
          np.round(variance, 6), np.round(expected_error, 6))
~~~

At the .5 probe, this produces the table above; at zero, the line and constant coincide, whereas the quadratic's expected error is .5. The larger retained [author calculation](author-calculations.py) checks this identity on 61 probe positions and also supplies a five-input version for investigation. It uses the same least-squares operation without clipping predictions or changing a hidden regularization parameter.

In the accompanying investigation, edit curvature, noise level and the probe location. Switch between fits to inspect the actual eight fitted curves, average curve and separate error contributions, including changes that preserve expected squared error. A second mode adds training inputs at \(-.5,.5\), enumerating 32 possible datasets. Setting noise to zero is a useful null: a correctly specified, identified polynomial reproduces the truth in every training draw.

### What can a bootstrap establish?

A bootstrap samples rows with replacement from the one dataset you possess. It can approximate aspects of a fitted procedure's sampling variability when its assumptions are suitable. It does not reveal the unknown \(f(x)\) merely by repeating the fit. Replacing \(f(x)\) with the original fitted model changes the target of a bias calculation.

Likewise, repeating random seeds on unchanged data measures conditional algorithmic variability, not all variability across newly collected training datasets. These are useful experiments when named correctly. They answer different questions from our exact eight-world calculation.

## 4. Three horizontal axes, three questions

| Curve | Horizontal axis | What stays fixed | Question |
| --- | --- | --- | --- |
| Learning curve | Number of training examples | Procedure and evaluation protocol | What happens as this procedure receives more data? |
| Validation curve | A hyperparameter, such as minimum leaf size | Data splits and other settings | Which inspected setting predicts better? |
| Training trajectory | Iteration, boosting round or epoch | Data and training run | What happens as this optimization run continues? |

All may show training and held-out error. They are not interchangeable. A larger minimum leaf size restricts a tree; a larger polynomial degree expands a function family; more trees in an averaging ensemble and more rounds in boosting have different effects.

To build a learning curve, choose the deployment-relevant split first. For each training portion, fit anew at each chosen sample size and score both that fitting subset and the associated held-out portion. Preprocessing belongs inside each fitted procedure. Use the same validation examples across sizes within a fold so a size comparison does not silently become a comparison of different test populations.

Sampling smaller training subsets introduces another choice. Randomly ordered prefixes are suitable for an exchangeable-data diagnostic. Chronological prefixes answer a different question because both time and amount of data change; grouped units need group-aware subsets. The real example below states exactly which experiment it runs.

One descending curve provides evidence over the inspected sizes. It does not guarantee its future slope. A plateau over a short, noisy interval does not prove you have exhausted all useful information.

## 5. A real question: predicting airfoil sound pressure

The supplied [Airfoil Self-Noise data](airfoil-self-noise.dat) contains 1,503 wind-tunnel measurements. The inputs are frequency (Hz), angle of attack (degrees), chord length (m), free-stream velocity (m/s), and suction-side displacement thickness (m); the target is scaled sound-pressure level (dB). This is a regression problem about aerodynamic measurements, not a classification problem about the presence of noise. [UCI dataset and license](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise).

We ask: **on row-level held-out measurements from this supplied collection, which procedures benefit from increasing training size?** The collection contains related experimental settings; its file does not supply a deployment-ready independent-run identifier. A random row split therefore does not establish performance on a new airfoil family or new experiment. For that goal, define the appropriate physical group and split it deliberately. Our scoped comparison uses a fixed development pool of 1,200 rows; 303 rows remain unused by this lesson.

Compare four prespecified procedures:

- A training-mean baseline.
- A standardized linear Ridge model with penalty 1.
- A regression tree allowed one training item per leaf.
- A tree requiring at least 20 items per leaf.

The tree contrast isolates a capacity restriction. The linear model provides another family; its preprocessing is fitted separately within each training subset. We use MSE in squared dB of the supplied target scale. This does not turn a squared error on a logarithmic sound-level scale into acoustic power error.

Download the data beside this complete program; install NumPy and scikit-learn if needed. Author calculations used NumPy 2.3.5 and scikit-learn 1.9.1.

~~~python
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split, learning_curve

data = np.loadtxt("airfoil-self-noise.dat")
x, y = data[:, :5], data[:, 5]
development, untouched = train_test_split(
    np.arange(len(y)), train_size=1200, random_state=41
)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
models = {
    "mean": DummyRegressor(strategy="mean"),
    "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.)),
    "tree_leaf1": DecisionTreeRegressor(
        min_samples_leaf=1, random_state=43),
    "tree_leaf20": DecisionTreeRegressor(
        min_samples_leaf=20, random_state=43),
}
for name, model in models.items():
    sizes, train_scores, valid_scores = learning_curve(
        model, x[development], y[development],
        train_sizes=[60, 120, 240, 480, 900], cv=cv,
        scoring="neg_mean_squared_error",
        shuffle=True, random_state=44, n_jobs=1,
    )
    print(name)
    for n, tr, va in zip(sizes, train_scores, valid_scores):
        print(n, round(-tr.mean(), 4), round(-va.mean(), 4))
~~~

Each of the five folds has 960 available training rows and 240 validation rows. Requested size 900 means 900 fitted rows per fold, not 900 total rows split five ways. The returned arrays have shape (5 sizes, 5 folds). The scoring interface maximizes scores, so it returns negative MSE; the program negates scores for ordinary positive error reporting. [Learning-curve API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html).

The recorded mean validation MSE is:

| Fitted rows per fold | Mean baseline | Ridge | Tree, leaf 1 | Tree, leaf 20 |
| ---: | ---: | ---: | ---: | ---: |
| 60 | 45.4545 | 24.9831 | 41.8405 | 39.9286 |
| 120 | 45.0810 | 23.8044 | 24.4041 | 32.3389 |
| 240 | 45.2469 | 23.6478 | 20.1754 | 28.1945 |
| 480 | 45.2065 | 23.4994 | 13.7464 | 21.6109 |
| 900 | 45.2220 | 23.3720 | 8.1412 | 16.6298 |

Several things are worth seeing together. Ridge is better with few examples. The unrestricted tree improves enough to pass it as training size grows. Requiring 20 items per leaf helps the tree at the smallest inspected size but hurts at the larger sizes. “Regularize a flexible model” is therefore a hypothesis to evaluate, not a guaranteed remedy.

The leaf-1 tree's training MSE is zero at every inspected size. Its validation MSE nevertheless decreases from 41.8405 to 8.1412. Exact training fit does not tell you whether the next larger training set will help, nor does zero training loss automatically imply useless predictions.

At size 900, Ridge has train/validation MSE 22.7528/23.3720; the leaf-20 tree has 12.5438/16.6298. A small Ridge gap does not certify proximity to the unknown noise floor: another inspected procedure already predicts much better.

The visual compares small multiples with shared axes and the mean baseline visible. Keep individual fold values available, with lines showing the mean. Fold-to-fold spread is a descriptive diagnostic; these folds overlap in their training data, so a standard deviation across five folds is not automatically a confidence interval for a difference. To compare two models, inspect their paired errors on matching observations and use an uncertainty method appropriate to the data dependence.

### Change a setting, then inspect a training trajectory

Append the following to the same program. The first comparison changes one tree setting at fixed fold-training size 960. The second traces one prespecified boosting fit on a separate split of the development pool.

~~~python
from sklearn.model_selection import validation_curve
from sklearn.ensemble import GradientBoostingRegressor

leaves = [1, 2, 5, 10, 20, 40]
tr, va = validation_curve(
    DecisionTreeRegressor(random_state=43), x[development], y[development],
    param_name="min_samples_leaf", param_range=leaves, cv=cv,
    scoring="neg_mean_squared_error", n_jobs=1,
)
for size, train_score, valid_score in zip(leaves, tr, va):
    print("leaf", size, round(-train_score.mean(), 4),
          round(-valid_score.mean(), 4))

fit_ids, monitor_ids = train_test_split(
    development, test_size=240, random_state=45
)
boosted = GradientBoostingRegressor(
    n_estimators=120, max_depth=2, learning_rate=.1, random_state=46
).fit(x[fit_ids], y[fit_ids])
train_mse = [np.mean((y[fit_ids] - p)**2)
             for p in boosted.staged_predict(x[fit_ids])]
monitor_mse = [np.mean((y[monitor_ids] - p)**2)
               for p in boosted.staged_predict(x[monitor_ids])]
for i in [0, 9, 29, 59, 119]:
    print("round", i + 1, round(train_mse[i], 4),
          round(monitor_mse[i], 4))
print("best inspected round", int(np.argmin(monitor_mse)) + 1)
~~~

Validation MSE for minimum leaf sizes 1, 2, 5, 10, 20, 40 is respectively 8.2079, 8.5475, 10.1516, 12.3799, 15.7762, 21.8071. This inspected restriction does not improve the score. The leaf-1 result differs from 8.1412 above because this run fits 960 rather than 900 rows per fold.

The boosting trace is:

| Round | Training MSE | Monitoring MSE |
| ---: | ---: | ---: |
| 1 | 41.5026 | 42.9204 |
| 10 | 26.9413 | 28.0979 |
| 30 | 17.5512 | 20.0168 |
| 60 | 13.0405 | 15.3584 |
| 120 | 9.1042 | 12.3338 |

The best of all 120 inspected rounds is 120. We have **not observed an optimal stopping point followed by deterioration**. If additional rounds are a worthwhile hypothesis, specify and test them on development data. Do not invent an overfitting turn merely because a standard illustration usually contains one. This staged comparison is also not a ranking against the five-fold experiment: the validation partitions differ.

These are model-development results. The 303 reserved rows were not used for any reported selection or score. A final performance report would first freeze a procedure and then evaluate it under an appropriate independent protocol.

### Keep the diagnostic affordable without changing its meaning

The learning-curve program performs \(4\times5\times5=100\) fits: four procedures, five sizes and five folds. More candidate settings multiply that work. The staged boosting display reuses one fitted run's intermediate predictions instead of refitting from scratch at every round. A small exact-world example exposes the mathematics cheaply; the real study answers a different empirical question.

For a larger problem, inspect a justified smaller grid first, retain split IDs and outcomes, and expand when the unresolved question warrants it. A curve on ten percent of the data cannot guarantee the shape on the other ninety percent. scikit-learn's incremental learning-curve mode requires the estimator's partial-fit interface; a warm-start flag alone does not make it valid. Changing to an incremental training procedure may also change the estimator being studied. Measure actual cost rather than assigning universal sample or runtime cutoffs.

## 6. Turn a curve into the next useful experiment

Before diagnosing a curve, verify that its axes, target, scoring direction and data units are what you think they are. Then separate observation from hypothesis.

| Observation | Plausible explanation | Useful discriminating experiment |
| --- | --- | --- |
| Both errors are poor relative to a meaningful baseline | Insufficient features/flexibility, too much shrinkage, poor optimization, or a bug | Fit a small known-solvable case; then change one of those constraints |
| Training error is low; held-out error is much larger | Sample sensitivity, leakage/mismatch, or an unrepresentative split may matter | Audit units/availability; compare paired results under a controlled regularization or data-size change |
| Held-out error improves over inspected sizes | More relevant data has helped this procedure over this range | Test a larger size with the same protocol and preserve the actual outcome |
| Both curves flatten near each other | The current family may be limited; target uncertainty or the sampled range may also dominate | Compare a justified alternate representation/model, inspect labels and slices; do not declare the noise floor known |
| Monitoring error worsens while fitting error falls | Further fitting is harming this measured validation objective | Assess stopping/regularization under a prespecified selection protocol |

A feature-selection result from the preceding lesson is one candidate input to this process. Importance does not prove that removing a feature will improve the learning curve. Retrain and evaluate the proposed reduced-input pipeline. If collecting a new measurement changes the prediction question or availability, document that change as well.

Data acquisition also has a composition question. Ten thousand near-duplicate measurements may add little independent information; a smaller set covering an underserved operating range may answer the actual failure. Plot errors by that range and define the desired deployment population. A global learning curve can hide a subgroup that still lacks coverage.

For practical planning, record: the observed result, your proposed mechanism, one changed setting/data collection, held-fixed quantities, evaluation unit, metric and what outcome would count against the hypothesis. This is more useful than assigning a categorical diagnosis from an arbitrary gap percentage.

## 7. Deeper: why training error is optimistic

The fitted model used the training outcomes, so they are not fresh observations independent of its predictions. The mixed-term argument in §2 cannot simply be applied to its residuals.

There are also two different risk targets. Conditional risk asks how the particular model trained on the observed \(D\) predicts fresh cases. Expected procedure risk averages that risk over new training datasets as well. An independent test set primarily evaluates the fitted model it is given; the bias–variance identity describes the repeated-training average. Cross-validation estimates depend on its refitting sizes and protocol. Keeping those targets distinct prevents a single test score from being mistaken for a direct population-variance measurement.

Consider correctly specified least squares with a fixed full-column-rank design matrix \(X\), \(n\) rows and \(p\) fitted coefficients, including an intercept if present. Write
\[
y=X\beta+\varepsilon,\quad
\mathbb E\varepsilon=0,\quad
\operatorname{Cov}(\varepsilon)=\sigma^2 I,
\]
and \(H=X(X^\top X)^{-1}X^\top\). This matrix projects outcomes onto the fitted column space, so \(\hat y=Hy\).

The training residual is \((I-H)\varepsilon\). Because \(I-H\) is a projection of rank \(n-p\),
\[
\mathbb E[\mathrm{MSE}_{\mathrm{train}}]
=\frac{\sigma^2}{n}\operatorname{tr}(I-H)
=\sigma^2(1-p/n).
\]

Now obtain independent new outcomes \(y'=X\beta+\varepsilon'\) **at those same input rows**. Their prediction errors are \(\varepsilon'-H\varepsilon\). Independence gives
\[
\mathbb E[\mathrm{MSE}_{\mathrm{new\ outcomes,\ same}\ X}]
=\sigma^2(1+p/n).
\]

The gap is \(2p\sigma^2/n\), while average fitted-prediction variance at those inputs is \(p\sigma^2/n\). Even in this favorable setting the gap is twice that variance, not a direct variance estimate.

Take \(n=6,p=2,\sigma^2=4\). Expected training error is \(8/3\), new-outcome error is \(16/3\), and prediction variance is \(4/3\). All arise from the same model.

For a genuinely new input vector \(x_*\), including any intercept coordinate, conditional on the fixed training design,
\[
\mathbb E[(Y_*-\hat f(x_*))^2\mid X,x_*]
=\sigma^2+\sigma^2 x_*^\top(X^\top X)^{-1}x_*,
\]
under the same correct-model and fresh-noise assumptions. Its leverage depends on location. The same-input average \(1+p/n\) is therefore not a universal out-of-distribution or random-design test-error formula.

This projection also exposes estimation versus approximation. If the true mean vector is \(f\) rather than \(X\beta\), training error gains \(\|(I-H)f\|^2/n\). High training error can contain approximation failure; small training error alone says little about a new input far from the fitted design.

### Effective degrees of freedom and regularization

For a fixed linear smoother \(\hat y=Sy\), training error under a mean vector \(f\) is
\[
\frac{\|(I-S)f\|^2}{n}
+\frac{\sigma^2}{n}\{n-2\operatorname{tr}(S)+\operatorname{tr}(S^\top S)\}.
\]
Fresh outcomes at the same inputs have expected error
\[
\frac{\|(I-S)f\|^2}{n}
+\sigma^2+\frac{\sigma^2}{n}\operatorname{tr}(S^\top S).
\]
Subtracting leaves \(2\sigma^2\operatorname{tr}(S)/n\). For ordinary least squares \(S=H\); for Ridge, the shrinkage matrix has smaller trace but may introduce bias. This explains the role of an optimism correction without assuming one parameter-count formula fits every learner.

Mallows-style \(C_p\) corrections estimate this expected optimism using a noise estimate. Likelihood criteria such as AIC and BIC answer related model-selection questions under their own assumptions; they are not measurements of the three decomposition terms. Section 9 of the earlier [Regularization lesson](/learn/path/full-curriculum/regularization-l1-l2-elastic-net-dropout?module=classical-ml) develops their different predictive and evidence-based aims, alongside minimum description length and an explicit coding example. Generalized cross-validation uses a linear smoother's effective degrees of freedom as an efficient leave-one-out approximation. Hyperparameter selection performed on the same observations adds another adaptive step, so fixed-\(S\) algebra is not automatically a complete correction for the selected procedure.

## 8. Deeper: classification and averaging

For a binary target with \(\eta(x)=P(Y=1\mid x)\), the squared probability error, or binary Brier loss, has the exact decomposition
\[
\mathbb E[(Y-\hat p_D(x))^2]
=(\eta(x)-\mathbb E\hat p_D(x))^2
+\operatorname{Var}(\hat p_D(x))
+\eta(x)(1-\eta(x)).
\]
This is an exact squared-loss identity, not an approximation to classification accuracy.

Thresholded class decisions behave differently. If \(\eta=.8\) and a training procedure predicts class 1 with probability \(q\), its expected zero-one error is
\[
.8(1-q)+.2q=.8-.6q.
\]
Changing \(q\) from 0 to .25 reduces error from .8 to .65 while introducing variation in the predicted class. Changing it from 1 to .75 increases error from .2 to .35. Thus increased variation may help or hurt, depending on which decisions it replaces. [Domingos's primary paper](https://homes.cs.washington.edu/~pedrod/papers/aaai00.pdf) develops loss-specific definitions; it should not be reduced to “the same three positive terms for every metric.”

A useful separate calculation explains averaging. Suppose \(B\) predictors at a fixed input have equal variance \(v\) and pairwise correlation \(\rho\). Their equal-weight average has variance
\[
v\left(\rho+\frac{1-\rho}{B}\right).
\]
Expand the variance of the sum: there are \(B\) individual variances and \(B(B-1)\) pair covariances, then divide by \(B^2\).

With \(v=4,\rho=.5,B=4\), the average variance is 2.5, not 1. Perfectly correlated predictions gain nothing from averaging; independent ones have variance \(v/B\). If the constituent mean predictions stay the same, averaging does not remove their common bias.

This calculation motivates bagging and why diversity matters, but bootstrap models from one dataset are not independent. Changing the training distribution through resampling can also alter their mean predictions. Boosting adds sequential corrective fits; it is not described completely by this fixed-constituent averaging calculation. The earlier ensemble lesson owns those training mechanisms.

## 9. Deeper: a tradeoff need not draw a U

The squared-error identity remains true when expectations exist. It does not prescribe how its terms move with width, depth, training size or optimization time. Double descent concerns those trajectories, not a failure of expanding a square.

To see a concrete alternative, consider a specified linear world with \(p\) independent standard-Gaussian input coordinates, true coefficient norm \(\|\beta\|=1\), and independent target noise of variance .04. Fit the **minimum Euclidean-norm unregularized least-squares solution**, using the pseudoinverse. Set \(\gamma=n/p\), the ratio of training rows to coordinates.

In the large-\(n,p\) approximation with fixed ratio, expected excess squared error is
\[
\begin{cases}
(1-\gamma)+.04\,\gamma/(1-\gamma),&\gamma<1,\\
.04/(\gamma-1),&\gamma>1.
\end{cases}
\]
Add .04 for fresh-target noise. These are asymptotic theoretical values, not a timing experiment or finite-sample measured result. [Nakkiran, Claims 1–2](https://arxiv.org/pdf/1912.07242).

| \(n/p\) | Approximate fresh-target MSE |
| ---: | ---: |
| .10 | .944444 |
| .50 | .580000 |
| .80 | .400000 |
| .90 | .500000 |
| .99 | 4.010000 |
| 1.01 | 4.040000 |
| 1.10 | .440000 |
| 1.50 | .120000 |
| 2.00 | .080000 |

Adding data initially helps, then hurts near the interpolation boundary, then helps again. The prediction-variance term below the boundary includes randomness of the observed input subspace as well as amplified target noise. At ratios close to one, small singular values make inversion especially sensitive to noise. More rows beyond that region constrain the fit differently.

The graph must mark the undefined/asymptotic singular boundary at \(\gamma=1\), not connect a smooth finite line through it. There is no simulated hardware or neural architecture in these numbers.

For this linear objective, convergent gradient descent from zero finds the minimum-norm solution. That statement does not establish that gradient descent finds an analogous minimum-norm predictor for every nonlinear network. Parameter count alone is also not a reliable interpolation threshold for arbitrary architectures and losses.

[Belkin and colleagues](https://arxiv.org/html/1812.11118v2) document model-wise examples, including tree ensembles; the phenomenon is not exclusive to neural networks. Deep networks add optimization-time and data-size behavior to investigate. The practical consequence is to measure the relevant trajectory and regularization choices, not to replace “smaller is always safer” with “bigger is always safer.”

## 10. Practice: from a calculation to a research decision

### 1. A new three-world sensor

The true mean at one input is 5. Equally likely fitted predictions are 3, 4 and 8. Fresh target noise has variance 2 and is independent of training. Compute bias, variance and expected squared error. Compare with a procedure always predicting 4.

<details><summary>Hint</summary>
Average predictions before measuring their spread. The target mean and average prediction need not be equal.
</details>
<details><summary>Solution</summary>
The mean is 5, so bias is zero. Prediction variance is \((4+1+9)/3=14/3\); expected error is \(20/3\). Always predicting 4 gives squared bias 1, variance 0, error 3. The stable biased procedure wins here. Noise has the same value in both comparisons.
</details>

### 2. Change the noise, keep the procedure

For the quadratic in §3 at \(x=.5\), keep curvature 1 and change \(\sigma\) to 1. Compute the new variance and total error. Then explain why the bias stays zero.

<details><summary>Hint</summary>
The noise variance multiplies the squared interpolation weights; the mean of each training noise remains zero.
</details>
<details><summary>Solution</summary>
Variance is \(23/32=.71875\); fresh-target noise is 1, so expected error is \(55/32=1.71875\). Correctly specified interpolation reproduces the quadratic mean in expectation at these full-rank input points. This conclusion relies on this estimator and experiment, not merely on containing the true function in a broad model family.
</details>

### 3. Diagnose the diagnosis

A colleague observes train/validation MSE 12/13 at several inspected sizes and says, “The model has low variance, the noise floor is 13, and collecting more data is pointless.” List what is observed and propose two distinct tests before accepting that conclusion.

<details><summary>Hint</summary>
You have no measured population mean or conditional noise variance. Consider another model and another part of the data-generation process.
</details>
<details><summary>Solution</summary>
The observed curves are close and flat over the inspected sizes. That does not identify the decomposition. Compare a justified richer/less-regularized procedure under the same splits, and inspect target measurement, feature availability, group composition or optimization with known cases. A meaningful new-size experiment can also test the local plateau. Report outcomes that would weaken each hypothesis, rather than calling a gap of 1 a variance estimate.
</details>

### 4. Investigate a new restriction on real data

Using only the supplied 1,200-row development pool, compare a tree with maximum depth 4 against the existing leaf-1 tree at the same five sizes and folds. Predict at which size, if any, the restriction will first stop helping. Keep the original feature set, row IDs, metric and shuffle settings fixed.

<details><summary>Hint</summary>
Add a separately named model, leaving the baseline procedure unchanged. A crossover may not occur in the inspected range.
</details>
<details><summary>Assessment and example conclusion</summary>
A complete answer contains the saved prediction, five paired score comparisons with fold values, and a conclusion tied to the actual result. “The restricted model was never better over these sizes” is acceptable if supported. Do not tune depth repeatedly and then present the selected score as independent evaluation. The 303 reserved rows remain unused while this development exercise continues.
</details>

### 5. Two sets of evidence

In an invented study, model A's validation error decreases 20→14→11 as fitted rows increase 100→300→900. Model B scores 13 on one different validation split with 900 fitted rows. Explain what may be concluded, and design the missing comparison.

<details><summary>Hint</summary>
The data-size trajectory and the model-family comparison are separate.
</details>
<details><summary>Solution</summary>
A improved over the inspected training sizes under its protocol. B's 13 cannot be cleanly ranked against A's 11 without accounting for the changed evaluation set. Fit the two prespecified procedures on matching training subsets and score the same validation units; inspect paired errors and relevant uncertainty. Neither trajectory certifies a future deployment improvement.
</details>

### 6. A new optimism calculation

For correctly specified fixed-design OLS with \(n=10,p=3,\sigma^2=2\), calculate expected training MSE, fresh-response MSE at the same inputs, their gap, and average fitted-prediction variance. State what changes for a new input location.

<details><summary>Solution</summary>
The values are 1.4, 2.6, 1.2 and .6. For a new \(x_*\), the variance contribution is \(2x_*^\top(X^\top X)^{-1}x_*\); the fixed-design average cannot be reused without its location/distribution assumptions. A rank-deficient design would also require the effective rank rather than blindly counting columns.
</details>

### 7. Is a probability loss the same as accuracy?

At one input, \(P(Y=1)=.7\). Procedure A always outputs probability .6; B outputs .4 or .8 with equal chance across independent training datasets. Compute expected Brier loss and expected classification error using threshold .5.

<details><summary>Hint</summary>
Both procedures average to .6, but only one crosses the class threshold across fits.
</details>
<details><summary>Solution</summary>
A has Brier loss \(.01+.21=.22\). B has the same squared bias .01 plus prediction variance .04 and noise .21, totaling .26. A always predicts class 1, giving error .3. B predicts 0 or 1 equally, giving error .5. This example compares two losses explicitly; it does not turn .04 into a universal classification-variance term.
</details>

### 8. Transfer the nonmonotonic example

Use §9's stated approximation at \(\gamma=.75\) and .9. Compute risk and explain why adding examples can worsen it without contradicting §2. Then name two assumptions you would check before applying that conclusion to a neural model.

<details><summary>Solution</summary>
Risk is \(.25+.12+.04=.41\) at .75, and \(.1+.36+.04=.5\) at .9. The risk identity allows a changing variance term; it does not assert monotonic learning curves. The example assumes isotropic Gaussian inputs, a correctly specified linear target, independent noise and minimum-norm ridgeless fitting in a large-dimensional limit. A neural model's data geometry, loss, regularization and optimization selection need their own evidence.
</details>

Core readiness: distinguish bias, prediction variability and target noise; calculate the changed finite example; tell the three plot axes apart; interpret real curves without inventing a diagnosis; design one useful controlled next experiment. Deeper readiness adds the fixed-design optimism derivation, loss-specific distinction and conditional double-descent explanation.

Next is [Imbalanced Learning](/learn/path/full-curriculum/imbalanced-learning-smote-cost-sensitive-learning?module=classical-ml). Carry forward a question that a single average curve can hide: whose mistakes does the metric count, and does the training/evaluation population give the rare cases enough attention?

## References & another way to learn it

- [Caltech — Learning From Data, Lecture 8](https://www.youtube.com/watch?v=zrEyxfl2-a8), with [official lecture slides](https://work.caltech.edu/slides/slides08.pdf). A visual mathematical alternative on repeated fits and learning curves, best after §§1–3. The full slide sequence was read; the video itself was not watched for this checkpoint. The slides use “bias” for the squared-bias contribution and sometimes compare fresh outcomes at fixed inputs; retain those conventions when following their derivations.
- [scikit-learn — Single estimator versus bagging](https://scikit-learn.org/stable/auto_examples/ensemble/plot_bias_variance.html). A complete alternative simulation that makes prediction spread visible. Useful after the exact finite calculation. Its constructed function and measured outputs are separate from our airfoil study.
- [scikit-learn — Learning and validation curves](https://scikit-learn.org/stable/modules/learning_curve.html), with [the learning_curve API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.learning_curve.html). Practical parameter, shape and score conventions for the programs. Documentation reviewed at version 1.9.1; a diagnostic curve is not a direct estimate of the unknown population decomposition.
- [Domingos — A Unified Bias-Variance Decomposition for Zero-One and Squared Loss](https://homes.cs.washington.edu/~pedrod/papers/aaai00.pdf). Primary loss-specific theory after §8; notation differs from this manuscript's signed bias.
- [Nakkiran — More Data Can Hurt for Linear Regression](https://arxiv.org/pdf/1912.07242). A short deeper exposition behind §9's explicitly asymptotic formulas. Read the setup and Claims 1–2 before interpreting its plot.
- [Belkin et al. — Reconciling modern machine learning practice and the bias-variance trade-off](https://arxiv.org/html/1812.11118v2). Broader model-wise evidence, including tree ensembles, with assumptions and experimental settings in the paper.
- [UCI — Airfoil Self-Noise](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise). Data description and CC BY 4.0 license; the offline file, extraction details and exact software/source hashes are in [data provenance](data-provenance.md).
