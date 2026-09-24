# Regularization: L1, L2, Elastic Net & Dropout

A model can explain the observations you collected in several different ways. Some explanations depend on large, finely balanced coefficients: increase one contribution and almost cancel it with another. A small change in the measurements may then change those coefficients dramatically. Other explanations use many weak contributions that might be real signal, or might just fit the particular sample.

**Regularization adds a preference to fitting.** It can favor smaller coefficients, fewer nonzero coefficients, smoother neighboring values, or predictions that remain useful when some intermediate inputs are randomly withheld. The preference changes the problem being solved. Whether it improves future predictions is something to assess using the data boundaries from the previous lesson.

**First pass.** Follow sections 1–6 to understand the objectives, work a tiny fit, compare real observations and explain dropout's train/evaluation distinction. Try practice 1–6. Sections 7–9 deepen the connection to linear algebra, Bayesian priors, parameterization and model-selection criteria; you can return to those branches after the core route. You need a weighted sum, squared error, means and the preceding fit/validation distinction. We introduce the extra notation as it becomes useful.

## 1. What preference are we adding?

Suppose a prediction is

\[
\hat y_i=b+x_{i1}w_1+\cdots+x_{id}w_d.
\]

There are n observed cases and d input features. The coefficient vector w tells us how each feature contributes; the intercept b provides a common offset. Ordinary least squares chooses these values to minimize the sum of squared residuals, where a residual is observed minus predicted value.

We will use **half the mean squared error**, plus a penalty:

\[
J(b,w)=\frac1{2n}\sum_{i=1}^{n}(y_i-b-x_i^\top w)^2
+\lambda\left[\rho\sum_j|w_j|+\frac{1-\rho}{2}\sum_jw_j^2\right].
\]

The factor one-half simplifies derivatives; averaging by n keeps the data term on a per-case scale. λ is the nonnegative penalty strength. The mixing fraction ρ lies between zero and one. The intercept is excluded from this penalty. This convention governs the main regression calculations; a deeper denoising example will explicitly declare its unaveraged objective:

| Choice | Penalty in this convention | What it encourages |
| --- | --- | --- |
| Ridge, or L2 | $\lambda\sum_j w_j^2/2$, using ρ=0 | Smaller coefficient norm; stable treatment of weakly determined directions |
| Lasso, or L1 | $\lambda\sum_j|w_j|$, using ρ=1 | Shrinkage with the possibility of exact zero coefficients |
| Elastic net | Both terms, using 0<ρ<1 | Sparse fits with an additional strictly convex preference |

The vertical bars mean absolute value: both +3 and −3 contribute 3 to an L1 penalty and 9 to a squared L2 penalty. “L1” and “L2” name norms, or ways to measure a vector's size. Neither name identifies which observations are allowed to influence fitting; preprocessing and penalty selection still belong inside the training/validation protocol.

A numerical comparison makes the tradeoff visible. In a one-coefficient problem, suppose the data term is $\frac12(w-3)^2$. With ridge strength λ=1, w=3 gives perfect data fit but total cost 4.5. At w=1.5, data cost is 1.125 and penalty is 1.125, totaling 2.25. A worse fit to these observed data is preferable under the new objective. That does **not** by itself establish that w=1.5 predicts future cases better.

**Figure 1 — Two quantities make one objective.** Plot the one-dimensional data cost, penalty and their sum, with labeled values at w=3 and w=1.5. Keep “training fit” and “penalty” visibly separate so a learner can explain why the minimum moves.

### Units change the meaning of a penalty

If a feature measured in meters is replaced by the same values in centimeters, its numeric values multiply by 100. Dividing its coefficient by 100 preserves every prediction, but its L1 cost divides by 100 and its squared L2 cost divides by 10,000. A raw coefficient penalty therefore favors that larger numerical feature scale for the same predictive contribution.

Standardizing a numeric column using its training mean and standard deviation is one useful way to make the preference refer to a one-standard-deviation change. It is not an instruction to erase meaningful physical units in every problem. A domain-specific penalty can deliberately assign different costs to different coefficients. Sparse indicator columns also need a considered convention: scaling a rare binary feature to unit variance changes the cost of its effect.

The preceding [Feature Scaling, Encoding & Imputation](/learn/path/full-curriculum/feature-scaling-encoding-imputation?module=classical-ml) supplies the fit/transform mechanics. Here the important question is: **what change in the original input does one unit of this coefficient represent?** Changing target units also changes the numerical balance between squared error, L1 and L2; a λ value is meaningful only with its objective and scaling convention.

The unpenalized intercept has a useful consequence. If every training target increases by seven, the fitted intercept can increase by seven while slopes stay the same. There is no reason to shrink that common offset toward zero merely because the measurement origin changed. Penalizing an intercept can be a deliberate modeling choice, but it should not happen accidentally because a column of ones was included in w.

## 2. Why L2 shrinks and L1 can select

### One coefficient, with the arithmetic exposed

Start with

\[
\frac12(w-z)^2+\frac\lambda2w^2.
\]

Here z is the coefficient preferred by the data alone in this simple normalized problem. The derivative is $(w-z)+\lambda w$. Setting it to zero gives

\[
w_{\text{ridge}}=\frac{z}{1+\lambda}.
\]

At z=3 and λ=1, the answer is 1.5. At z=0.4, it is 0.2. Both are pulled toward zero. In this scalar example a nonzero z does not become exactly zero at finite λ. In a multi-feature problem, an individual ridge coefficient can equal zero because of the data geometry; ridge simply has no threshold region that systematically creates sparsity.

Replace the squared penalty by $\lambda|w|$. For positive w, the derivative of the objective is $w-z+\lambda$, giving w=z−λ if that answer is positive. For negative w, the derivative is $w-z-\lambda$, giving w=z+λ if negative. If neither case is valid, the minimum is at zero. Combining the cases:

\[
S(z,\lambda)=\operatorname{sign}(z)\max(|z|-\lambda,0).
\]

This is **soft-thresholding**. It removes the central interval [−λ, λ] and shrinks surviving values toward zero. It differs from hard thresholding, which would keep a surviving z unchanged.

| Data preference z, with λ=1 | Ridge | Lasso | Elastic net, ρ=0.5 |
| ---: | ---: | ---: | ---: |
| 3 | 1.5 | 2 | 5/3 |
| 0.4 | 0.2 | 0 | 0 |
| −2 | −1 | −1 | −1 |

For elastic net the same calculation gives $S(z,\lambda\rho)/[1+\lambda(1-\rho)]$. Its L1 part sets the threshold and its L2 part changes the denominator. Elastic net is a family of preferences, not a guarantee that its answer or prediction error lies between those of separately tuned ridge and lasso.

**Investigation 1 — Where does a coefficient disappear?** Enter a data preference z, strength λ and mixing fraction ρ. Predict whether the answer is negative, zero or positive before solving. See the actual objective curve, derivative on each side and the soft-threshold interval. Changing z from 0.4 to 1.4 under lasso λ=1 changes the fitted coefficient from zero to 0.4. Moving z within [−1,1] leaves it zero, even though the data term changes. That flat output region is the mechanism behind sparsity.

### The two-dimensional geometry, without an exaggerated claim

A constrained version asks for the smallest data loss among coefficients inside a fixed penalty budget. In two dimensions, an L2-norm budget forms a disk and an L1-norm budget forms a diamond. The first data-loss contour that meets the allowed region identifies a constrained optimum. Diamond corners and faces make exact zero coordinates possible over a range of data preferences.

They do not force every optimum to a corner. With independent normalized coordinates, z=(3,0.4) and lasso λ=0.1, the answer is (2.9,0.3): both coordinates survive. At λ=1 the answer becomes (2,0). The elastic-net budget still has nonsmooth behavior where a coordinate crosses zero; its curved edges do not remove that threshold.

**Figure 2 — Actual contact, including a non-sparse case.** Draw disk, diamond and mixed-penalty boundaries with computed contours for these stated two-coordinate problems. Show both λ cases. In a penalized-to-constrained comparison, the matching budget is the penalty value attained by that solution; the same numerical λ is not automatically the same radius or budget across methods.

## 3. From one coefficient to a complete fit

### Separate the offset, then write the matrix equation

Let X contain the n rows of features, and let y contain the n targets. Subtract each training feature mean and the training target mean. Write the centered arrays as Z and t. After fitting slopes w, recover the intercept as $b=\bar y-\bar x^\top w$.

For ridge, differentiation gives

\[
(Z^\top Z+n\lambda I)w=Z^\top t.
\]

I is an identity matrix. The $n\lambda$ appears because our data term is averaged by n. For λ>0, any nonzero vector v satisfies

\[
v^\top(Z^\top Z+n\lambda I)v=\|Zv\|^2+n\lambda\|v\|^2>0.
\]

Thus the matrix is positive definite and the centered ridge slopes are unique, even if columns repeat or d>n. In code, solve the linear system instead of explicitly forming its inverse. For poorly conditioned problems, an SVD-based solver avoids forming the squared condition number of the normal equations; positive λ helps mathematically but does not excuse careless numerics.

### Coordinate descent: let each feature explain the remaining residual

Lasso and elastic net can update one coefficient at a time. Temporarily remove feature j's current contribution from the prediction. The partial residual is

\[
r_j=t-Zw+Z_{:,j}w_j.
\]

Define a data curvature and a residual association:

\[
a_j=\frac{Z_{:,j}^\top Z_{:,j}}n,\qquad
c_j=\frac{Z_{:,j}^\top r_j}n.
\]

The exact coordinate minimizer is

\[
w_j\leftarrow\frac{S(c_j,\lambda\rho)}{a_j+\lambda(1-\rho)}.
\]

Each update uses the current values of the other coefficients. A full pass through all coordinates is a **sweep**. With correlated columns, changing one coefficient changes the residual available to the next, so several sweeps can be needed. A coefficient that is zero during one sweep can become nonzero later; the partial residual can change.

For pure lasso, a zero coefficient at an optimum permits absolute residual association $|Z_{:,j}^\top(t-Zw)/n|$ at most λ. A nonzero coefficient requires equality to λ, with the association's sign matching the coefficient. Strict inequality therefore forces zero; equality alone can occur with a zero or nonzero coefficient. These are optimality conditions involving the **current full residual**, not a one-time test of the ordinary-least-squares coefficient or proof that a feature is irrelevant to the world.

### Four rows we can calculate by hand

Use the following constructed input:

| Row | First feature | Second feature | Target |
| ---: | ---: | ---: | ---: |
| 0 | 1 | 1 | 3.4 |
| 1 | 1 | −1 | 2.6 |
| 2 | −1 | 1 | −2.6 |
| 3 | −1 | −1 | −3.4 |

The means are zero, $Z^\top Z/n=I$ and $Z^\top t/n=(3,0.4)$. These columns are orthogonal: after accounting for one, the residual association of the other stays the same. At λ=1, one coordinate sweep therefore gives ridge (1.5,0.2), lasso (2,0), or elastic net with ρ=0.5 equal to (5/3,0). Their total objective values are 2.29, 2.58 and approximately 2.496667, respectively. These costs come from **different penalty functions**, so the smallest of those numbers is not a valid way to choose which family predicts best.

**Investigation 2 — Fit the residual, then inspect the prediction.** Edit the actual four rows, targets and two features. Before Apply, predict whether a chosen coefficient is zero, or predict one row's fitted value. Inspect partial residuals, c, a, threshold, coordinate update and the resulting prediction. Changing row 0's target from 3.4 to 7.4 gives lasso slopes (3,0.4) and intercept 1 at λ=1. Adding seven to every original target instead changes only the intercept to seven. The first edit adds a feature-specific association; the second changes the measurement origin.

### A complete small implementation

The following NumPy program implements the common objective, including an unpenalized intercept and residual updates. It checks the optimality conditions rather than stopping just because the last coefficient movement looks small. For nonzero w_j, the smooth gradient plus $\lambda\rho\operatorname{sign}(w_j)$ should be zero. For a zero coordinate, the smooth gradient may lie anywhere within [−λρ, λρ]. The maximum violation is reported as a residual, not as a test-set error.

Use Python with NumPy 2.3.5 (`python -m pip install numpy==2.3.5` in a new environment). Save as `coordinate_regularization.py` and run it. Equivalent calculations were executed in the supplied author script; the displayed standalone program awaits phase-two execution.

```python
import numpy as np

def soft_threshold(value, threshold):
    return np.sign(value) * np.maximum(np.abs(value) - threshold, 0.0)

def fit_coordinates(X, y, strength, ratio, tolerance=1e-10, max_sweeps=10000):
    X, y = np.asarray(X, float), np.asarray(y, float)
    if X.ndim != 2 or y.shape != (len(X),) or len(X) == 0:
        raise ValueError("X must be nonempty rows by features, y one target per row")
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise ValueError("Use finite data; handle missing observations before fitting")
    if not np.isfinite([strength, ratio]).all() or strength < 0 or not 0 <= ratio <= 1:
        raise ValueError("Use nonnegative strength and a mixing ratio in [0, 1]")
    n, p = X.shape
    mean_x, mean_y = X.mean(axis=0), y.mean()
    Z, target = X - mean_x, y - mean_y
    curvature = np.mean(Z * Z, axis=0)
    weight = np.zeros(p)
    residual = target.copy()
    for sweep in range(1, max_sweeps + 1):
        for j in range(p):
            partial = residual + Z[:, j] * weight[j]
            association = Z[:, j] @ partial / n
            denominator = curvature[j] + strength * (1 - ratio)
            weight[j] = (soft_threshold(association, strength * ratio)
                         / denominator) if denominator > 0 else 0.0
            residual = partial - Z[:, j] * weight[j]
        gradient = -(Z.T @ residual) / n + strength * (1 - ratio) * weight
        violation = np.where(
            weight != 0,
            np.abs(gradient + strength * ratio * np.sign(weight)),
            np.maximum(np.abs(gradient) - strength * ratio, 0),
        )
        optimality_residual = float(violation.max(initial=0))
        if optimality_residual <= tolerance:
            intercept = float(mean_y - mean_x @ weight)
            return weight, intercept, sweep, optimality_residual
    raise RuntimeError("Coordinate fit did not meet the requested optimality tolerance")

X = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], float)
y = np.array([3.4, 2.6, -2.6, -3.4])
for name, ratio in [("ridge", 0), ("lasso", 1), ("elastic_net", .5)]:
    weight, intercept, sweeps, residual = fit_coordinates(X, y, 1, ratio)
    print(name, np.round(weight, 6), round(intercept, 6), sweeps)
```

Expected rounded lines from the recorded equivalent calculation:

```text
ridge [1.5 0.2] 0.0 1
lasso [2. 0.] 0.0 1
elastic_net [1.666667 0.] 0.0 1
```

A constant centered feature has a=0 and no residual association. Setting its coefficient to zero is appropriate; when its objective is completely flat it is a declared representative solution. This instructional solver does not implement sparse storage, screening or optimized paths. Current library implementations use additional numerical machinery and diagnostics; the [scikit-learn linear-model guide](https://scikit-learn.org/stable/modules/linear_model.html#lasso) documents coordinate descent and its optimality-gap approach.

## 4. Correlated features: prediction and attribution are different questions

Imagine two sensors report exactly the same centered value x. The model's prediction depends only on the sum $s=w_1+w_2$, because $xw_1+xw_2=xs$. If the target is 2x, no amount of fitting those duplicate measurements can reveal which sensor “caused” the signal.

Use two rows x=−1 and x=1, with targets −2 and 2. Our data loss is $\frac12(s-2)^2$. With lasso λ=1, the optimal sum is s=1. Every nonnegative pair with that sum has the same data cost and the same L1 penalty: (1,0), (0.5,0.5), and (0,1) all minimize the objective. A coordinate solver starting from zero may return (1,0) because it visits the first column first. A different order can return the other endpoint. That algorithmic choice is not evidence about the sensors' scientific importance.

The squared L2 penalty prefers balanced coefficients because, for fixed s,

\[
w_1^2+w_2^2=\frac{s^2}{2}+\frac{(w_1-w_2)^2}{2}.
\]

The difference term is smallest when both weights equal s/2. With ridge λ=1, the optimum is (2/3,2/3). With elastic net λ=1 and ρ=0.5, it is (0.6,0.6). These methods also change the best sum; they do not merely redistribute the lasso answer.

**Figure 3 — One prediction, many coefficient allocations.** Put the duplicate-feature coefficients on a plane, show lines of constant sum, and highlight the lasso minimizer segment. Add the unique ridge and elastic-net solutions with their own objectives. A linked contribution strip shows what the two sensor readings add to the prediction.

For nearly identical columns the conclusion becomes a tendency, with conditions, rather than exact equality. The elastic-net L2 term supplies strict convexity and encourages similar coefficients for similarly scaled, strongly positively correlated columns. It does not promise that every correlated group will always be selected together or that feature selection will be perfectly stable. Full-column-rank lasso is unique even when its columns are correlated; correlation alone is not a proof of nonuniqueness. [Zou and Hastie's primary analysis](https://hastie.su.domains/Papers/B67.2%20%282005%29%20301-320%20Zou%20%26%20Hastie.pdf) develops grouping, and [Tibshirani's uniqueness paper](https://arxiv.org/pdf/1206.0313) states the more precise solution conditions.

This distinction matters in applications such as correlated chemical measurements or groups of gene-expression features. A sparse predictor may be cheaper to measure and easier to inspect. Its nonzero list is still a property of this fitted model, its feature representation and its penalty. Prediction, stable selection and causal explanation require different evidence. The next lesson on feature importance will make those questions explicit.

## 5. A real comparison: predicting airfoil sound measurements

The Airfoil Self-Noise collection contains 1,503 observations from aeroacoustic experiments. Each row gives frequency in hertz, angle of attack in degrees, chord length in meters, free-stream speed in meters per second and suction-side displacement thickness in meters. The target is scaled sound-pressure level in decibels. These are physical observations, not points drawn to guarantee that one regularizer wins. [UCI's dataset description](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise) supplies the measurement context and CC BY 4.0 license; the unchanged small data file and attribution accompany this lesson.

The task is numeric prediction for held-out rows under the declared row-level experiment. Related experimental settings occur in the collection, and independent run IDs are not available here. This assessment does not establish performance on an entirely new airfoil or experimental run.

We use a predeclared development set of 1,200 rows and reserve 303 rows, split with seed 41. The reserved rows receive no prediction or score in this lesson. Within development, three shuffled folds with seed 202 compare fitting choices. This is a **development comparison and selection record**. Its selected scores are not newly independent performance claims. The later learning-curves lesson reuses the input with a different protocol; its numbers should not be read as a direct contest against these fits.

### Let a linear model express curved relationships

For this comparison, transform the five raw inputs into their five original terms, five squared terms and ten pairwise products: twenty features total. A term such as frequency × chord can represent an interaction between measurements. The model remains linear in the twenty fitted coefficients, although its prediction is nonlinear in the original inputs.

Fit a scaler to those twenty columns inside each training fold, then fit the model. The polynomial recipe itself is fixed, but scaling learns from data and must stay within the fold. All three families use λ values `[0.001,0.01,0.1,1,10,100]`; elastic net fixes ρ=0.5 for this comparison. We also compute a training-mean baseline and unpenalized least squares with the same twenty-feature representation. Mean squared error has units of squared decibels.

### Library names do not define the objective

For scikit-learn's `Ridge`, the documented objective is summed squared error plus `alpha` times squared coefficient norm. For `Lasso` and `ElasticNet`, the data term is half the **mean** squared error. Therefore, to fit our common convention on n_fit rows:

| Estimator | Settings matching this manuscript |
| --- | --- |
| `Ridge` | `alpha = n_fit * strength` |
| `Lasso` | `alpha = strength` |
| `ElasticNet` | `alpha = strength`, `l1_ratio = ratio` |

At the same numeric `alpha`, ridge and lasso do not have the same normalized penalty strength. For logistic regression the inverse-strength parameter `C` also needs its estimator's documented loss normalization; “C=1/λ” without specifying that loss can miss a sample-count factor. The [current objectives](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression-and-classification) are the reference, rather than a shared parameter name.

### Complete offline program

Save as `airfoil_regularization.py` beside `airfoil-self-noise.dat`. In a new environment use `python -m pip install numpy==2.3.5 scikit-learn==1.9.1`. The equivalent full calculation ran serially in the author environment with no warnings. This program performs 54 candidate fits, three OLS fits and three selected refits; baseline means require no estimator fit. It does not train a neural network or download data.

```python
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

def make_model(family, strength, n_fit):
    if family == "ridge":
        fit = Ridge(alpha=n_fit * strength, solver="svd")
    elif family == "lasso":
        fit = Lasso(alpha=strength, max_iter=50000, tol=1e-8)
    elif family == "elastic_net":
        fit = ElasticNet(alpha=strength, l1_ratio=.5, max_iter=50000, tol=1e-8)
    elif family == "ols":
        fit = LinearRegression()
    else:
        raise ValueError(family)
    return make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False), StandardScaler(), fit,
    )

data = np.loadtxt(Path(__file__).with_name("airfoil-self-noise.dat"))
development, reserved = train_test_split(
    np.arange(len(data)), train_size=1200, random_state=41,
)
X, y = data[development, :5], data[development, 5]
splits = list(KFold(n_splits=3, shuffle=True, random_state=202).split(X))
strengths = [.001, .01, .1, 1, 10, 100]
mean_errors, ols_errors = [], []
for train, valid in splits:
    mean_errors.append(np.mean((y[valid] - y[train].mean()) ** 2))
    ols = make_model("ols", 0, len(train)).fit(X[train], y[train])
    ols_errors.append(np.mean((ols.predict(X[valid]) - y[valid]) ** 2))
print("baseline mean MSE", round(float(np.mean(mean_errors)), 6))
print("OLS mean MSE", round(float(np.mean(ols_errors)), 6))

for family in ["ridge", "lasso", "elastic_net"]:
    candidates = []
    for strength in strengths:
        errors, nonzero = [], []
        for train, valid in splits:
            model = make_model(family, strength, len(train)).fit(X[train], y[train])
            errors.append(np.mean((model.predict(X[valid]) - y[valid]) ** 2))
            nonzero.append(int(np.count_nonzero(model[-1].coef_)))
        score = float(np.mean(errors))
        candidates.append((score, strength))
        print(family, strength, round(score, 6), nonzero)
    score, selected_strength = min(candidates)  # lower strength resolves an exact tie
    fitted = make_model(family, selected_strength, len(X)).fit(X, y)
    print("selected", family, selected_strength,
          "development CV MSE", round(score, 6),
          "refit nonzero", int(np.count_nonzero(fitted[-1].coef_)))
```

Recorded development results, rounded to six decimals:

| λ | Ridge MSE | Lasso MSE | Elastic-net MSE |
| ---: | ---: | ---: | ---: |
| 0.001 | 17.315634 | 17.335409 | 17.325484 |
| 0.01 | 17.335789 | 17.347152 | 17.332697 |
| 0.1 | 17.694800 | 17.629206 | 17.650934 |
| 1 | 20.962663 | 22.179353 | 21.658337 |
| 10 | 34.763447 | 45.072768 | 45.072768 |
| 100 | 43.547835 | 45.072768 | 45.072768 |

The mean baseline is 45.072768 and unpenalized OLS is 17.350268. All three searches select the smallest λ in this predeclared grid, 0.001. Their selected differences are small; this experiment does not justify a strong ranking of families or claim a universal interior “sweet spot.” In a real development project, a boundary selection can motivate another declared search, with the assessment boundary still protected.

Lasso at λ=0.1 leaves nine nonzero coefficients in each of the three folds. At λ=10 it leaves none and predicts the training mean, matching the baseline exactly. The λ=0.001 final lasso refit on all development rows keeps all twenty terms, even though two individual folds kept nineteen. L1 can create sparsity; the selection objective and sample do not guarantee that the chosen fit will be sparse.

**Figure 4 — The observed path, including its unhelpful end.** Plot actual per-fold and mean validation MSE against λ on a logarithmic horizontal axis, with baseline lines. Beside it show signed coefficient paths from a specified fold and an explicit zero mark. These are recorded results, not a hand-drawn U-shaped curve. Selecting a point reveals its actual fold count, fitted coefficient values and score.

**Investigation 3 — Trace one airfoil prediction through its terms.** Use the saved final ridge model at λ=0.001. Edit the five physical measurements of one development observation, predict whether its output increases or decreases, then calculate the twenty polynomial values, their saved standardized values and signed contributions. The initial observation's recorded prediction is approximately 124.465305 dB. Changing a target label in this inference view cannot change the fitted prediction: no fitting takes place. Changing a raw feature changes its own, squared and interaction terms together. The display must not offer twenty unrelated editable scaled terms as if every combination represented a possible physical input.

The starting measurements are 1,250 Hz, 17.4 degrees, chord 0.0254 m, speed 31.7 m/s and displacement thickness 0.0176631 m. Keeping the other four values fixed and changing frequency to 1,750 Hz gives approximately 123.628646 dB, a decrease of 0.836659 dB. This is a deterministic scenario under the fitted model, not evidence that physically changing frequency causes that exact change in a new experiment.

### Selecting λ without leaking the scaler

An efficient regularization-path estimator such as `LassoCV` can reuse nearby solutions. But `Pipeline(StandardScaler(), LassoCV(...))` fits that outer scaler on all data supplied to the pipeline **before** the estimator runs its internal folds. Those inner validation rows then influenced scaling. The same issue arises if you compute `X_scaled` once and pass it to an internal CV estimator.

Our explicit loop fits the whole pipeline within each fold. A `GridSearchCV` wrapped around a complete pipeline is another clear option when its parameter convention matches the intended objective. Path efficiency is useful, but it does not move preprocessing inside folds automatically. After selection, refitting preprocessing and the chosen model on all development rows is appropriate; those final fitted transformations must travel with the model for inference.

## 6. Dropout: change what the learner sees during training

The first three methods add an explicit parameter cost. **Dropout randomly withholds selected input or intermediate values during fitting.** A later neural-network lesson will build layers in detail. For now, a hidden unit is simply a learned weighted sum followed by an activation function, and an activation is the value it passes onward. A dropout mask multiplies selected values by zero while leaving other paths available.

Use q for the **keep probability**, so the drop probability is 1−q. With inverted dropout, an activation a becomes

\[
\tilde a=\frac{m}{q}a,\qquad m\sim\operatorname{Bernoulli}(q),\quad0<q\le1.
\]

A Bernoulli variable equals one with probability q and zero otherwise. Dividing retained values by q gives $\mathbb E[\tilde a]=a$. For q=0.5, a value 2 becomes either zero or four, each with probability one-half. At ordinary deterministic evaluation, the dropout operation is the identity: it passes the original activation through.

### An exact connection to a penalty

Consider a linear prediction $\tilde y=\sum_j w_jx_jm_j/q$ with independent masks and a fixed target y. The mean prediction is $w^\top x$, but squared loss also responds to variation around that mean. Expanding the square gives

\[
\mathbb E_m\!\left[\frac12(y-\tilde y)^2\right]
=\frac12(y-w^\top x)^2
+\frac{1-q}{2q}\sum_j w_j^2x_j^2.
\]

The cross terms from independent centered mask noise vanish. Each retained/rescaled input has variance $x_j^2(1-q)/q$. Averaging over training rows therefore produces a data-dependent diagonal quadratic penalty. This is exact for this linear, squared-loss, independent-mask setup. Other losses and nonlinear networks require different analysis or approximations; dropout is not universally the same as adding a fixed L2 penalty. [Wager, Wang and Liang](https://nlp.stanford.edu/pubs/wager2013dropout.pdf) develop the more general feature-noising connection.

For x=(2,1), w=(1,−1), y=1 and q=0.5, the deterministic prediction is one and its half-squared loss is zero. Enumerate all four masks:

| Mask | Noisy prediction | Half-squared loss | Probability |
| --- | ---: | ---: | ---: |
| (0,0) | 0 | 0.5 | 1/4 |
| (0,1) | −2 | 4.5 | 1/4 |
| (1,0) | 4 | 4.5 | 1/4 |
| (1,1) | 2 | 0.5 | 1/4 |

The mean prediction is one but expected loss is 2.5. The penalty formula gives $(1-q)/(2q)\,(4+1)=2.5$, exactly matching the enumeration. This is why matching an average activation does not make a noisy training objective identical to a clean evaluation loss.

**Investigation 4 — Enumerate the masks.** Edit x, w, y and q for two contributions. Predict the mean output and whether expected noisy loss exceeds clean loss, then open the actual mask tree. Its branch probabilities change with q; there is no need for a misleading four-state uniform average when q is not one-half. Setting one contribution to zero makes its mask irrelevant. Setting q=1 removes the extra loss. Changing the target changes both losses, while their difference stays fixed for fixed x, w and q.

A complete exact enumeration uses only Python:

```python
from itertools import product

x, weight, target, keep = [2.0, 1.0], [1.0, -1.0], 1.0, 0.5
expected_prediction = expected_loss = 0.0
for mask in product([0, 1], repeat=2):
    probability = 1.0
    for kept in mask:
        probability *= keep if kept else 1 - keep
    prediction = sum(a * w * m / keep for a, w, m in zip(x, weight, mask))
    loss = (target - prediction) ** 2 / 2
    expected_prediction += probability * prediction
    expected_loss += probability * loss
    print(mask, probability, prediction, loss)
print(expected_prediction, expected_loss)
```

Its recorded baseline values are the table above and final values `1.0 2.5`. Keep must be positive; at q=1 the zero-probability branches simply contribute nothing. Complete dropout-network training is owned by the later [Dropout, DropPath & Stochastic Depth](/learn/path/full-curriculum/dropout-droppath-stochastic-depth?module=deep-learning-fundamentals) lesson rather than a second unexplained network here.

### What changes when the rest of the network is nonlinear?

An average input passed through a nonlinear operation need not equal the average of that operation's outputs. For a small example, let a noisy value be zero or two equally often, and pass it through $f(u)=\max(0,u-1)$. The mean of f is 0.5. Passing the mean input one through f gives zero. Consequently, ordinary dropout-off inference is not generally an exact average of every masked nonlinear network. The [original dropout paper](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) motivates and investigates the approximation; its benchmark outcomes are not universal guarantees.

In PyTorch, `nn.Dropout(p=...)` uses **drop probability**, and ordinary evaluation uses `model.eval()`. Disabling gradient recording with `no_grad()` or `inference_mode()` is a separate operation; it does not itself switch dropout into evaluation behavior. Intentional Monte Carlo dropout is another inference procedure, not an automatic uncertainty guarantee. The [API contract](https://docs.pytorch.org/docs/main/generated/torch.nn.Dropout.html) specifies element masking and inverted scaling. The later deep lesson owns mask placement, residual paths and normalization interactions.

Compare models on the same held-out cases using a clearly declared evaluation mode. A lower dropout-off loss on the **training rows** is still a training-data result; a gap between noisy training loss and clean loss does not prove generalization. Dropout rates, placement and combination with weight penalties require validation. There is no universal best rate, fixed extra-epoch multiplier or rule that adding more regularizers must improve a model.

## 7. Deeper branch: directions, paths and computation

### Ridge shrinks directions of information

Write a singular value decomposition of the centered design as $Z=U\Sigma V^\top$. The columns of V describe orthogonal directions in coefficient space; a singular value σ_j tells us how strongly changing that direction changes the fitted observations. Along a positive-singular-value direction, ridge uses

\[
w_\lambda=\sum_j\frac{\sigma_j}{\sigma_j^2+n\lambda}(u_j^\top t)v_j.
\]

The corresponding fitted-data component is multiplied by $\sigma_j^2/(\sigma_j^2+n\lambda)$. A large singular value retains more of its unpenalized fit. A small singular value is attenuated more, preventing division by a tiny number from producing a huge coefficient response. Components in the null space are set to zero by positive ridge regularization.

For example, with nλ=1, singular values 4 and 0.5 give fitted-component multipliers 16/17≈0.9412 and 0.25/1.25=0.2. With nλ=4 they become 0.8 and approximately 0.05882. The weakly identified direction receives much stronger relative shrinkage. This is a more accurate explanation of stabilization than saying every original feature coefficient is multiplied by the same number.

As λ decreases to zero, ridge approaches the minimum-Euclidean-norm least-squares solution. When the design is full column rank, that is the usual unique OLS solution. As λ grows, the slopes approach zero and the unpenalized intercept leaves the training mean prediction. Individual coefficients can move non-monotonically or cross zero in correlated designs; the scalar shrinkage formula does not describe each original coordinate independently.

**Figure 5 — Strong and weak directions.** Show the two analytic multipliers for singular values 4 and 0.5, along with coefficient-space directions and their corresponding data changes. Label the horizontal variable nλ so it matches the displayed equation. Do not mix it with a native lasso `alpha` axis.

### What an L1 path tells you

For centered data and pure lasso, the all-zero slope vector satisfies the optimality conditions when

\[
\lambda\ge\lambda_{\max}=\max_j|Z_{:,j}^\top t|/n.
\]

For the four-row example, λ_max=3. This gives a principled starting point for a path from a zero-slope fit toward less penalization. Nearby λ values can use the previous solution as a warm start. In general correlated designs, coordinates may enter, leave or change sign along the path; the number of selected features is not guaranteed to move monotonically at every path point.

When a lasso solution is nonunique, every minimizer has the same fitted values on the training design, although coefficient allocations can differ. There exists a sparse representative with a limited active set; under common general-position uniqueness conditions the number of nonzero coefficients is at most the design rank. This is more precise than asserting that **every** solution always has at most n nonzeros. In the duplicate-column example, one can distribute an optimal positive sum over many identical columns without changing fit or L1 cost. Prediction away from the observed design can differ if those columns no longer remain identical. The [uniqueness analysis](https://arxiv.org/pdf/1206.0313) is the reference for these distinctions.

### Choose a solver for the matrix you actually have

For dense n×d data with n≥d, forming the normal-equation matrix costs order nd² and a dense solve order d³; storing that matrix costs order d². If d is much larger than n, the identity

\[
w=Z^\top(ZZ^\top+n\lambda I)^{-1}t
\]

offers an n×n system instead. Use a solve here too. These dimensions suggest alternatives; they do not establish an exact practical crossover at n=d. Conditioning, sparsity, factorization reuse and available memory matter. Iterative least-squares or matrix-vector methods can avoid either dense Gram matrix.

The residual-maintaining coordinate implementation costs order nd per dense sweep. Recomputing the entire prediction $Zw$ separately for every coordinate would introduce unnecessary extra work. Sparse column storage can make an update depend on its nonzero entries; specialized solvers add screening, active sets and warm starts. The number of sweeps depends on tolerance and conditioning, so a universal “10–200 sweeps” promise is inappropriate. Check convergence warnings and optimality diagnostics before interpreting a fit.

For distributed ridge with manageable d, sums of local $Z_i^\top Z_i$ and $Z_i^\top t_i$ can recover the corresponding global sufficient statistics, provided centering, weights and normalization are handled consistently. The d² communication/storage requirement remains. Consensus optimization can distribute lasso-type objectives, but it needs its own convergence and communication design. A local solver does not become a distributed algorithm merely by putting its call in a task scheduler.

### Early stopping and weight decay are related, but have precise contracts

Under ordinary gradient descent on the unpenalized centered mean-square objective, initialized at zero, a direction whose Gram eigenvalue is a>0 has fitted-component factor $1-(1-\eta a)^t$ after t steps with step size η. Ridge's factor is $a/(a+\lambda)$. Both can suppress weakly learned directions, but they are different filters. For one step, η=0.1 and a values 1 and 4 give factors 0.1 and 0.4. Matching those with ridge would require λ values 9 and 6 respectively; one common λ does not reproduce both. Appropriate step-size conditions are also needed for the iteration to remain stable.

Likewise, a gradient step on a loss plus $\lambda\|w\|^2/2$ is

\[
w^+=(1-\eta\lambda)w-\eta\nabla L(w).
\]

For this ordinary update, a multiplicative decay with factor $1-\eta\lambda$ is equivalent. In an adaptive optimizer, adding λw to a gradient sends it through the optimizer's gradient transformation; decaying weights separately generally does not. That is the distinction behind AdamW. Inspect the optimizer's actual convention and parameter groups, including whether biases and normalization parameters are included. The [decoupled-weight-decay paper](https://arxiv.org/pdf/1711.05101) derives the difference. Detailed optimizer dynamics belong to the optimization lessons; a `weight_decay` argument is not a universal mathematical identity.

## 8. Deeper branch: a penalty encodes a representation

### Bayesian priors: track the noise scale

Assume $y\mid b,w,X$ has independent Gaussian errors with known variance σ². Ignoring constants, the negative log likelihood is $\|y-b\mathbf1-Xw\|^2/(2\sigma^2)$. A Gaussian prior $w_j\sim N(0,\tau^2)$ contributes $\|w\|^2/(2\tau^2)$. Multiplying the combined negative log posterior by σ²/n gives our ridge objective with

\[
\lambda=\frac{\sigma^2}{n\tau^2}.
\]

An independent Laplace prior with density proportional to $\exp(-|w_j|/s)$ instead gives lasso strength $\lambda=\sigma^2/(ns)$. The intercept can receive a separate prior or be treated as unpenalized. These are **maximum a posteriori**, or MAP, fits: the most favored parameter value under the stated likelihood/prior combination.

The factors matter. Writing “Gaussian variance 1/λ” without the likelihood and normalization can be wrong for the objective being used. If a prior and noise scale are held fixed while n changes, our normalized λ changes inversely with n. Holding λ fixed over different training sizes is a different convention, useful for a controlled regularization comparison but not the same fixed-prior experiment.

A Laplace prior is continuous; it assigns probability zero to any exact singleton w_j=0, as do other continuous densities. Its posterior mode can be exactly zero because of the density's kink. That does not give a posterior probability that a feature is absent. A full Bayesian analysis includes uncertainty and integrates predictions over parameter values; replacing it by one penalized fit discards that information. The [elastic-net paper's Bayesian section](https://hastie.su.domains/Papers/B67.2%20%282005%29%20301-320%20Zou%20%26%20Hastie.pdf) connects the priors, while this local derivation specifies our factors explicitly.

### Same predictor, different parameter penalty

Suppose a one-dimensional model is written with two factors, predicting abx, and the data cost is $\frac12(ab-1)^2$. Every pair with ab=1 has zero data cost. But adding $\lambda(a^2+b^2)$ gives different costs along that same-prediction curve: (1,1) costs 2λ, while (2,0.5) costs 4.25λ.

Balancing the factors minimizes the penalty **among zero-data-loss pairs**, but the full regularized optimum can prefer nonzero data loss. Let p=ab. Since $a^2+b^2\ge2|ab|=2|p|$, with equality attainable by equal-magnitude factors, the full problem reduces to

\[
\min_p\frac12(p-1)^2+2\lambda|p|.
\]

Soft-thresholding gives $p^*=\max(1-2\lambda,0)$. At λ=0.25, the optimal product is 0.5, achieved by a=b=√0.5 or both negative. The data cost is 0.125 and penalty 0.25, totaling 0.375; balanced zero-data-loss factors would total 0.5. At λ≥0.5, both optimal factors are zero. At λ=0, every ab=1 pair minimizes the unpenalized problem.

This is a concrete example of a parameterization changing what a familiar L2 penalty means for the represented function. It is not evidence that balancing arbitrary neural layers universally improves prediction. The learning point is to inspect the **whole objective**, not only a symmetry of its data-loss term.

**Figure 6 — The same-prediction curve and the actual optimum.** Show ab=1, the balanced point on it, and the balanced optimum with product 1−2λ. A linked scalar p plot exposes why staying on the zero-loss curve misses the full minimum.

### Sometimes smoothness is a better preference than small values

If w represents values at neighboring positions, penalizing differences can be more meaningful than pulling every value toward zero. Let L compute neighboring differences and minimize

\[
\frac12\|y-w\|^2+\frac\lambda2\|Lw\|^2.
\]

For three positions, take $Lw=(w_2-w_1,w_3-w_2)$. With y=(0,2,0) and λ=1, solve $(I+L^\top L)w=y$ to obtain (0.5,1,0.5). Ordinary identity-based ridge with the same unaveraged convention gives (0,1,0). Both reduce the middle spike, but the difference penalty spreads it across neighboring positions. Adding a constant to every input shifts the difference-penalty solution by the same constant because L annihilates a constant vector.

This is a small instance of **generalized Tikhonov regularization**, where $\|Lw\|^2$ expresses which patterns are expensive. A difference operator encourages smoothness; another L can encode a different scientifically justified relation. It is useful in inverse problems, where measurements are indirect and many latent signals could explain them. The condition for a unique generalized quadratic fit is that no nonzero direction lies in both the data operator's null space and L's null space. A difference penalty alone does not necessarily remove every ambiguity.

**Figure 7 — Penalize magnitude or neighboring change?** Display the observed three-position signal, identity-penalty solution and difference-penalty solution on common axes. Show the actual edge differences under each and the linear system used. Label this as an exact constructed denoising example, not an empirical claim about noise removal quality.

Other useful penalties encode other structures. An L1 penalty on differences, often called total-variation or fused regularization in appropriate settings, can favor piecewise-constant regions rather than smooth variation. A group-lasso penalty sums Euclidean norms of predeclared coefficient groups, allowing an entire group to become zero. A multi-task penalty can select the same input across several prediction outputs. These are different assumptions about where sparsity belongs: individual coefficients, neighboring changes, predefined groups or shared tasks. They are not interchangeable names for elastic net's tendency to balance correlated individual coefficients.

An engaging further example is reconstructing an image from a few line projections. The unknown pixel values form w, and a known projection operator maps them to measurements. If the image is sparse in the chosen representation, L1 regularization can express that prior structure. Most natural images are not sparse as raw pixels, so the representation is part of the scientific claim. The inspected [tomography reconstruction example](https://scikit-learn.org/stable/auto_examples/applications/plot_tomography_l1_reconstruction.html) shows the actual operator, synthetic image and comparison; its particularly favorable sparse image is not a guarantee of exact recovery for arbitrary scans.

## 9. Deeper branch: AIC, BIC and description length

Coefficient penalties are not the only way to control fitting flexibility. Suppose we compare candidate probability models, each fitted by maximum likelihood on the same observations. A more flexible model often has a larger training likelihood simply because it had more freedom. AIC, BIC and minimum description length account for complexity for different reasons. They do not turn a development-selected score into a fresh test result.

### AIC: correct optimism when estimating predictive fit

Let $\ell(\hat\theta)$ be the maximized natural-log likelihood and k the number of freely fitted parameters in a regular parametric model. The familiar formula is

\[
\operatorname{AIC}=-2\ell(\hat\theta)+2k.
\]

The negative likelihood term measures fit; smaller is better. The correction reflects that parameters were chosen on the data being scored. Under the usual regular, correctly specified parametric assumptions, it estimates expected predictive log-loss, up to a constant shared by candidates. The 2k correction is an asymptotic result, not a universal fee for every parameter in every algorithm; model misspecification can require a different optimism correction.

For a Gaussian regression with unknown noise variance estimated by maximum likelihood, substitution gives a data-dependent term $n\log(\mathrm{RSS}/n)$ plus constants, followed by 2k. Count an estimated variance parameter and intercept consistently. Known-variance formulas differ. Small-sample corrections such as AICc have model-specific assumptions; they are not a universal replacement for checking sample size, dependence or misspecification.

For shrinkage estimators, the effective flexibility can differ from the raw number of stored coefficients. The later [Bias–Variance & Learning Curves](/learn/path/full-curriculum/bias-variance-tradeoff-learning-curves?module=classical-ml) derives the fixed-linear-smoother optimism correction and its trace-based degrees of freedom. Counting twenty stored ridge coefficients as twenty freely fitted OLS coefficients would miss that shrinkage. Information-criterion implementations for lasso use additional model and variance-estimation assumptions; inspect those rather than attaching 2k to an arbitrary penalized training objective. The [scikit-learn criterion derivation](https://scikit-learn.org/stable/modules/linear_model.html#aic-and-bic-criteria) specifies its actual Gaussian convention.

### BIC: a large-sample evidence approximation

For the same kind of regular, fixed-dimensional model,

\[
\operatorname{BIC}=-2\ell(\hat\theta)+k\log n.
\]

The evidence for a model integrates likelihood over its parameter prior, rather than evaluating only the best point. A local quadratic approximation around the maximum makes each well-identified parameter direction contribute a width of order $n^{-1/2}$. Multiplying k such widths contributes $n^{-k/2}$; taking −2 times the log gives the k log n term. Prior densities and local curvature contribute terms that the basic large-n expression suppresses.

This argument needs an identifiable, regular interior solution and suitable priors, with dimension held fixed as n grows. In a correctly specified collection satisfying the needed conditions, BIC can consistently favor the correct model dimension. That is a different goal from minimizing predictive loss at a finite sample size. Neural networks, mixture singularities, growing dimensions and boundary parameters do not automatically satisfy this derivation. BIC values are not exact posterior probabilities. [Grünwald's technical discussion](https://homepages.cwi.nl/~pdg/ftp/mdlintro.pdf), sections 2.6.3 and 2.9.2, develops the evidence approximation and its limits.

Take two constructed fitted-model records on n=100 observations:

| Model | Maximized log likelihood | k | AIC | BIC |
| --- | ---: | ---: | ---: | ---: |
| Smaller | −150 | 3 | 306 | 313.8155 |
| Larger | −146 | 5 | 302 | 315.0259 |

The larger model improves −2 log likelihood by eight. AIC charges four for its two added parameters, while BIC charges about 9.2103. They choose differently because they use different justified approximations and goals. These are hand records for arithmetic, not empirical evidence that one criterion is better. Compare only compatible likelihoods on the same observations, with constants and target transformations treated consistently.

### MDL: pay to describe the explanation as well as its errors

Minimum description length asks how compactly a declared coding scheme can describe the observed data. In a simple two-part version, pay for a model/parameter description and then for the data given that description:

\[
L(\text{explanation})+L(\text{data}\mid\text{explanation}).
\]

Here L denotes code length, not the regression loss notation used earlier. A pattern that perfectly fits the observations is not free: the receiver must be told which pattern or parameter values were selected. For a discrete probability model, ideal data-code length is $-\log_2 P(\text{data}\mid\text{model})$. Continuous measurements additionally need a declared precision or corresponding density-based construction.

A tiny fully specified code makes this concrete. Both parties know the message contains sixteen bits. Two modes are allowed:

- Mode 0: send a zero flag followed by all sixteen literal bits, for seventeen bits total.
- Mode 1: send a one flag followed by a four-bit pattern; the receiver repeats that pattern four times, for five bits total.

For `0101010101010101`, mode 1 sends the flag and `0101`: five bits. The chosen pattern was learned from the data, and its four bits were paid for. For `0101010001010101`, no four-bit pattern repeated four times reproduces the message, so this scheme uses the seventeen-bit literal mode. These codes are unambiguous because the first flag identifies the remaining length. The scheme is intentionally limited; another declared code might exploit a different pattern. We are not claiming to compute the shortest possible program for every message.

**Figure 8 — Three complexity accounts.** Put AIC's 2k and BIC's k log n next to their shared fit term, using the computed table. Below, show the actual MDL flags, pattern/literal payload and decoded sixteen-bit result. Do not put bits and natural-log likelihood units on an unlabeled common axis.

Modern MDL includes refined universal codes, not only a hand-selected parameter code. For a finite discrete model class with a finite normalizer, normalized maximum likelihood assigns

\[
P_{\mathrm{NML}}(D)=\frac{P(D\mid\hat\theta_D)}{\sum_{D'}P(D'\mid\hat\theta_{D'})}.
\]

The denominator accounts for all datasets of the stated size that the model family can fit well. Its logarithm supplies a complexity cost and makes the expression a probability distribution. Some model classes have an infinite normalizer and require another construction. Under specific fixed-dimensional regular asymptotics, an MDL expression can share BIC's leading complexity term; **MDL and BIC are not identical in general**. The inspected [MDL tutorial](https://homepages.cwi.nl/~pdg/ftp/mdlintro.pdf) provides both the basic coding view and the refined distinction.

These criteria extend the same habit as regularization: state the preference, its units and assumptions, then distinguish the quantity optimized from the outcome ultimately needed. Cross-validation remains useful when it matches the intended future use and encompasses the whole selection recipe; analytic or coding criteria are useful when their assumptions and purpose fit the problem.

## 10. Practice: change the problem, then explain the answer

Try the first six without the deeper branches. Hints and solutions are optional so you can work independently before checking.

### 1. Shrinkage with a different sign

For $\frac12(w+2.4)^2$ and λ=0.6, find ridge, lasso and elastic-net ρ=0.5 coefficients. Explain why a zero answer is not appropriate here.

<details><summary>Hint</summary>The data preference z is −2.4. Apply the threshold before the elastic-net denominator.</details>
<details><summary>Solution</summary>Ridge is −2.4/1.6=−1.5. Lasso is −1.8. Elastic net is (−2.4+0.3)/1.3=−21/13≈−1.615385. The absolute data preference exceeds each relevant threshold.</details>

### 2. What changes under a different measurement origin?

In the four-row example, fit lasso with λ=0.5, then increase every target by three. Give both sets of slopes and intercepts. Would penalizing the intercept necessarily preserve this result?

<details><summary>Solution</summary>The slopes are (2.5,0) in both fits. The original intercept is zero and the shifted intercept is three. Excluding the intercept allows an exact translation without changing the slope objective. An intercept penalty introduces an extra cost for that translation and can change the result.</details>

### 3. A missing sample-count factor

You want our ridge objective with λ=0.2 on eighty training rows. Which `Ridge(alpha=...)` matches it? Which `Lasso(alpha=...)` matches pure L1 at the same λ convention? What happens to ridge's native alpha on a sixty-row fold?

<details><summary>Solution</summary>Ridge needs alpha=16 on eighty rows and alpha=12 on sixty rows. Lasso uses alpha=0.2 in either case. This follows from multiplying our ridge objective by 2n, not from treating the two parameter names as equivalent. It does not say that matching numerical λ gives the two penalty shapes identical effects.</details>

### 4. Duplicate sensors

The duplicate-feature example now has target 3x and lasso λ=0.5. Give the optimal coefficient sum and two different minimizers. What extra fact would you need before calling one sensor causally important?

<details><summary>Solution</summary>The optimal sum is 2.5. Pairs (2.5,0) and (1.25,1.25) both minimize the objective, as do other nonnegative allocations of that sum. The observational duplicate design does not identify which sensor is causally relevant; that requires an appropriate causal question, assumptions and evidence beyond this fit.</details>

### 5. Exact dropout without uniform mask probabilities

Let x=(1,2), w=(2,0), y=1 and keep probability q=0.75. Compute the clean prediction, expected noisy prediction, clean half-squared loss and expected noisy half-squared loss. Why does the second mask not affect the answer?

<details><summary>Hint</summary>Only the first coordinate contributes. Its noisy prediction is zero with probability 1/4 and 8/3 with probability 3/4.</details>
<details><summary>Solution</summary>Both clean and expected predictions are two. Clean half-squared loss is 1/2. Expected noisy loss is $(1/4)(1/2)+(3/4)(25/18)=7/6$. The difference is 2/3, matching $(1-q)/(2q)\,4$. The second coefficient is zero, so changing that mask cannot change the weighted sum.</details>

### 6. Read the actual experiment

A learner sees the λ=0.001 lasso result and says: “Lasso always selects fewer inputs than ridge, and the smallest score proves this family will win on a new airfoil.” Identify two separate errors. What does the λ=10 result legitimately demonstrate?

<details><summary>Solution</summary>The final selected lasso fit keeps all twenty terms, so L1 does not guarantee sparsity at the selected setting. These are development selection scores in a row-level design, not independent evidence about a new airfoil/run or a decisive family ranking. At λ=10 all lasso slopes are zero in the three folds, and the unpenalized intercept reproduces the corresponding mean baseline.</details>

### 7. A changed factor penalty — deeper

Minimize $\frac12(ab-1)^2+0.1(a^2+b^2)$. Give the optimal product and balanced factors. Compare its total cost with the balanced zero-data-loss pair (1,1).

<details><summary>Solution</summary>The product is 0.8 and equal-sign factors have magnitude √0.8≈0.894427. Data cost is 0.02, penalty is 0.16 and total is 0.18, below the 0.2 of (1,1). Minimizing the penalty while insisting on zero data loss misses the actual full optimum.</details>

### 8. A code has to pay for its chosen pattern — deeper

Using the declared sixteen-bit code, encode `1110111011101110`. Give the mode, payload and total length. If someone chooses a different four-bit pattern after seeing the data but charges only the flag, what is missing?

<details><summary>Solution</summary>Mode 1, payload 1110, total five bits. The receiver must learn which of sixteen possible patterns was selected, so the four-bit pattern description cannot be omitted. The receiver already knows the total message length and the repeat rule under this declared code.</details>

### 9. Criteria can disagree — deeper

On n=50 observations, a smaller model has log likelihood −80 and k=2. A larger model has log likelihood −77 and k=4. Compute AIC and BIC for both. Does disagreement imply a calculation error?

<details><summary>Solution</summary>AIC values are 164 and 162, favoring the larger model. BIC values are $160+2\log50≈167.8240$ and $154+4\log50≈169.6481$, favoring the smaller. The different penalties reflect different goals and assumptions; disagreement alone is not an error. These formulas still require compatible regular likelihood models and correctly counted parameters.</details>

### 10. A smoothness-preserving shift — deeper

Change the difference-penalty input from (0,2,0) to (3,5,3), keeping λ=1 and the unaveraged objective. Predict the solution without solving another matrix system. Would identity-based ridge make the same shift?

<details><summary>Solution</summary>The difference-penalty solution becomes (3.5,4,3.5), because adding a constant lies in L's null space. Identity-based ridge gives (1.5,2.5,1.5), so its output shift is only 1.5. The penalties express different preferences.</details>

## 11. Readiness and the next question

You are ready to move on when you can state the fitted objective and its normalization, calculate a shrinkage/threshold update, distinguish prediction from coefficient attribution, fit preprocessing within each validation fold, and explain why mean-preserving dropout still changes expected loss. You should also be able to read the real comparison without forcing a U shape or treating a development-selected score as independent evidence.

Next is [Feature Selection & Importance: SHAP, Permutation & Mutual Information](/learn/path/full-curriculum/feature-selection-importance-shap-permutation-mutual-info?module=classical-ml). A zero or large coefficient is only one kind of statement. We will ask which features are useful to a fitted predictor, how removing or perturbing a feature changes its performance, what information exists before fitting, and why none of those questions automatically identifies causes.

## References and other ways to learn

- [Scikit-learn linear models](https://scikit-learn.org/stable/modules/linear_model.html): inspect the actual ridge/lasso/elastic-net objectives, coordinate updates, path diagnostics, multi-task penalties and information-criterion assumptions. Compare each equation with its parameter names before copying a strength value between estimators.
- [Zou and Hastie, Regularization and Variable Selection via the Elastic Net](https://hastie.su.domains/Papers/B67.2%20%282005%29%20301-320%20Zou%20%26%20Hastie.pdf): sections 2–3 explain grouping and the original paper's distinction between a mixed-penalty estimate and its historical rescaled variant. Current `ElasticNet` follows its documented mixed objective; do not add the paper's extra rescaling to a library prediction automatically.
- [Tibshirani, The Lasso Problem and Uniqueness](https://arxiv.org/pdf/1206.0313): the KKT conditions, equal fitted-value property and uniqueness assumptions correct the oversimplified claim that correlation always makes lasso nonunique.
- [Compressive sensing: tomography reconstruction with an L1 prior](https://scikit-learn.org/stable/auto_examples/applications/plot_tomography_l1_reconstruction.html): a visual/code learning route from projections to an image. Read the synthetic image's sparsity assumption and operator construction before interpreting the favorable comparison.
- [Lasso model selection: AIC, BIC and cross-validation](https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html): an inspected criterion/path visualization and executable example. It illustrates estimator choices; when adapting its internally cross-validated estimator, keep learned preprocessing inside the folds as explained here.
- [Srivastava and colleagues, Dropout](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf): the mask/network diagrams, training procedure and empirical model-averaging discussion. Its particular architecture/rate findings are observations from its experiments, not universal placement rules.
- [Wager, Wang and Liang, Dropout Training as Adaptive Regularization](https://nlp.stanford.edu/pubs/wager2013dropout.pdf): deeper analysis of noising under generalized linear losses. Our exact two-input square-loss derivation is the preparation for its data-dependent penalties.
- [Loshchilov and Hutter, Decoupled Weight Decay Regularization](https://arxiv.org/pdf/1711.05101): the update equations explain why adaptive-optimizer weight decay needs a precise convention.
- [Grünwald, A Tutorial Introduction to the Minimum Description Length Principle](https://homepages.cwi.nl/~pdg/ftp/mdlintro.pdf): start with chapter 1's coding explanation, then use sections 2.5.3, 2.6.3 and 2.9.2 for normalized maximum likelihood, Bayesian evidence and the distinction from BIC. The extra branches are optional further study rather than prerequisites for the core regularization workflow.
- [Airfoil Self-Noise at UCI](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise): measurement definitions, attribution and data license. The accompanying packet keeps the unchanged numeric file and exact development/fold indices for offline reproduction.
