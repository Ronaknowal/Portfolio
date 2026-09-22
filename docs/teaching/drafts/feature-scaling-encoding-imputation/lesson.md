# Feature Scaling, Encoding & Imputation

> Current lab UX, 21 September 2026: controls show live calculations and topic-specific visuals without learner prediction entry, grading or guess-to-reveal screens. Genuine algorithm steps, separate practice and data-role boundaries remain. See [the current migration record](../../LIVE-EXPLORATION-CLASSICAL-EARLY.md).

A table is not yet a model input. A body mass of `4000` might mean grams, a category called `female` is not a smaller number than `male`, and a blank measurement is not a measured zero. Before fitting a model, we need a consistent way to represent what each entry means.

This lesson follows one practical question: **can measurements of a penguin help distinguish its species?** The answer will depend partly on the model, and partly on the representation that decides which measurements the model can compare. We will use real observations, inspect individual transformations, and keep a small set of rows aside to see what happens to previously unseen records.

**First pass:** read sections 1–6, work through the distance and mixed-table investigations, then try core practice questions 1–6. You need Python arrays, a table with rows and columns, averages, and the idea of predicting a label from examples. Each new formula is unpacked locally. Sections 7–9 are deeper branches on nonlinear representations, target encoding, and missing-data uncertainty; they extend the core rather than becoming prerequisites for finishing it.

## 1. Three different questions hiding inside “preprocessing”

Suppose one row contains a bill length, body mass, recorded sex, and species:

| Bill length | Body mass | Recorded sex | Species |
| --- | --- | --- | --- |
| 40 mm | 4,000 g | female | Adelie |

For our prediction task, species is the **target**: the answer available in the training examples. The other selected columns are **features**: information we intend to have when making a new prediction. A feature is not useful merely because it is present in the file; an identification number or a label recorded only after the answer is known can mislead the experiment.

Three preparation operations answer different questions:

| Operation | Question | Example |
| --- | --- | --- |
| Scaling | What numerical differences should have comparable influence? | Express a difference in body mass relative to its training spread |
| Encoding | How should a category become a model-readable representation? | Give each recorded category its own indicator coordinate |
| Imputation | What input should we supply where a measurement is absent? | Insert a training median and optionally retain a missingness indicator |

They can interact, but they are not interchangeable. Converting grams to kilograms changes units. Replacing a missing mass with 4,000 g makes an estimate. Encoding `not_recorded` as its own category records absence without inventing a biological category.

The preceding [NMF lesson](/learn/path/full-curriculum/non-negative-matrix-factorization-nmf?module=classical-ml) also depended on representation: its additive factors required nonnegative input. Subtracting a column mean can create negative numbers, so a preprocessing choice suitable for a distance model may violate an NMF input requirement. The useful question is always **what information and geometry does the next model need?**

**Figure 1 — One record, three meanings.** Follow an observed numeric cell, a category cell, and an absent cell through separate annotated transformations. Keep units beside original measurements and feature names beside the output coordinates. The target takes a separate path into fitting, never into the ordinary feature matrix.

## 2. Scaling changes the meaning of “nearby”

### A nearest-neighbor decision you can calculate

Consider a new measurement $Q=(40,4000)$, where the coordinates are bill length in millimeters and body mass in grams. Two possible neighbors are $A=(41,4100)$ and $B=(43,4001)$. These are constructed measurements for arithmetic, not rows claimed to come from the real dataset.

The usual squared Euclidean distance adds squared coordinate differences:

\[
d^2(Q,A)=(41-40)^2+(4100-4000)^2=10{,}001,
\]
\[
d^2(Q,B)=3^2+1^2=10.
\]

The raw-number rule chooses B. A difference of 100 g overwhelms a difference of a few millimeters. The calculation is well defined, but its implicit relative importance came from the units we happened to write.

Now decide, explicitly, that 1 mm and 100 g should each count as one unit of difference. Divide bill differences by 1 and mass differences by 100:

\[
d_s^2(Q,A)=1^2+1^2=2,\qquad d_s^2(Q,B)=3^2+0.01^2=9.0001.
\]

A is now nearer. Nothing moved in the physical world. We changed the ruler used by the model.

For positive divisors $s_j$, this rule is

\[
d_s^2(x,z)=\sum_j\frac{(x_j-z_j)^2}{s_j^2}.
\]

So scaling a feature by $1/s_j$ is equivalent to assigning its squared difference weight $1/s_j^2$. This is why scaling matters to nearest neighbors, k-means, and distance-based kernels. It also affects the meaning of coefficient penalties and can improve the numerical conditioning of gradient-based fitting. There is no theorem saying equal training variance is the best measure of relevance for every task.

**Investigation 1 — Choose the ruler, then inspect the neighbor.** Edit either candidate’s measurements or the mass divisor and follow which candidate is nearer. The display separates each feature’s contribution to squared distance. Valid input edits update the calculation immediately. Also try multiplying both divisors by the same positive number: all distances change by a common factor, but the neighbor ranking stays the same.

### Learn a ruler from training data

Often we do not have a justified domain divisor. A common baseline is to fit a separate mean and standard deviation for each feature:

\[
\mu_j=\frac1n\sum_i x_{ij},\qquad
s_j=\sqrt{\frac1n\sum_i(x_{ij}-\mu_j)^2},\qquad
z_{ij}=\frac{x_{ij}-\mu_j}{s_j}.
\]

Here $n$ counts training rows, $i$ selects a row, and $j$ selects a column. The divisor $n$, rather than $n-1$, matches `StandardScaler`'s population-style training variance. The purpose is a transformation, not an unbiased estimate of an unknown population variance.

Subtracting the same mean from two rows cancels in their difference. Centering therefore does not change their Euclidean separation; the division changes the relative feature weights. Centering still matters to other operations, including a model's intercept and ordinary PCA's variance interpretation.

Standardization makes a nonconstant training column have mean zero and variance one. It **does not turn a skewed distribution into a Gaussian distribution**. It also does not establish the assumptions needed for a regression confidence interval: the distribution of a feature and the distribution of a model's errors are different objects.

For a constant training column, the standard deviation is zero. A practical implementation uses a scale of one rather than dividing by zero. Its training values become zero after centering; a different future value need not become zero. Constant columns may be removed, but a value changing after training can also be a useful data-quality signal.

### An outlier makes the choice visible

Fit three scalers to the five training values `[1, 2, 3, 4, 100]`:

| Training value | Standard scaling | Min–max scaling | Median/IQR scaling |
| --- | ---: | ---: | ---: |
| 1 | −0.5383 | 0 | −1 |
| 2 | −0.5127 | 0.0101 | −0.5 |
| 3 | −0.4870 | 0.0202 | 0 |
| 4 | −0.4614 | 0.0303 | 0.5 |
| 100 | 1.9993 | 1 | 48.5 |
| New value 150 | 3.2810 | 1.5051 | 73.5 |

Standard scaling uses mean 22 and standard deviation about 39.0128. **Min–max scaling** subtracts the training minimum and divides by the training range: here $(x-1)/99$. **Robust scaling** here subtracts the training median 3 and divides by the interquartile range $Q_{75}-Q_{25}=4-2=2$, using the stated linear percentile convention.

The robust rule preserves visible separation among 1, 2, 3, and 4. It does not remove 100: that observation is still 48.5 transformed units away from the median. A new value 150 is outside the training min–max interval. Forcing it into `[0,1]` would be a separate clipping operation that discards how far outside the range it lies.

**Figure 2 — Five observations on three rulers.** Show the same identified observations on aligned number lines, with a full-range view and a clearly marked magnified view of the four small values. Do not stretch every set to the same unlabeled width: the point is to see how its actual coordinates changed.

### Column scaling is different from row normalization

For a row $x$, L2 normalization divides by its own length, $\|x\|_2=\sqrt{\sum_jx_j^2}$. Thus `[3,4]` and `[6,8]` both become `[0.6,0.8]`. Their direction survives; their overall size does not.

This can be useful when comparing the composition of documents rather than their lengths, or a spectrum's shape rather than its overall intensity. It would be a questionable default if total intensity or total body size carries the signal. A zero vector has no mathematical direction; implementations generally leave it zero.

Sparse matrices introduce another practical constraint. A table of mostly zero word counts can become dense if we subtract a nonzero column mean. `StandardScaler(with_mean=False)` or `MaxAbsScaler` can preserve zeros when appropriate. Sparse storage and row normalization are tools for a specific representation, not mandatory steps for every dataset.

For an ideal threshold decision tree, strictly increasing transformations preserve the order of observed values and therefore the possible training partitions. This explains why unit scaling is usually much less important there. Finite precision, histogram binning, clipping, and transformations that merge values qualify that statement; it is not a promise that every implementation gives identical predictions under every transformation.

## 3. Encoding categories means choosing relationships

### One-hot coordinates avoid an invented ordering

Suppose a feature records `red`, `green`, or `blue`. Assigning numbers 0, 1, and 2 allows a numerical model to treat blue as twice green or to place green between the other two. A color label does not imply those relationships.

One-hot encoding assigns one coordinate to each known category:

| Category | red | green | blue |
| --- | ---: | ---: | ---: |
| red | 1 | 0 | 0 |
| green | 0 | 1 | 0 |
| blue | 0 | 0 | 1 |

Any two different rows in this table are distance $\sqrt2$ apart. This is a chosen geometry: all different categories have equal separation in that feature block. With several categorical columns, each block contributes to the total distance, so mixing one-hot blocks and scaled measurements still requires judgment about their relative influence.

Dropping one column is sometimes useful for interpreting an unregularized linear model with an intercept. If all three columns are kept, their sum is the intercept column, so its coefficients are not uniquely identified. Predictions can still be fitted with a suitable numerical least-squares solver. Removing red also changes distances: red becomes `[0,0]`, one unit from green, while green and blue remain $\sqrt2$ apart. Regularization can likewise make the choice of reference coding affect fitted predictions. “Always drop the first category” is not a universal preparation rule.

**Figure 3 — A category simplex and a reference corner.** Connect the three full one-hot points with equal-length edges; alongside it, show the dropped-column representation and its unequal distances. The table remains visible so the geometry can be checked without spatial intuition.

### Missing, unknown, and rare are different states

A missing category means the value was not recorded. An unknown category means a value is present now but was absent from the fitted vocabulary. A rare category is known but has little training support.

For example, a device model called `sensor_C` may be new at prediction time. `OneHotEncoder(handle_unknown="ignore")` represents an unknown value with zeros across that categorical block. This avoids an exception, but it does not teach the model how `sensor_C` behaves. With a dropped reference column, all-zero encoding can also coincide with the reference category. Alternative policies include rejecting invalid input or deliberately grouping infrequent/new values into a fitted bucket; the choice belongs to the application.

In our penguin program, absent recorded sex becomes `not_recorded`, and that fitted category gets its own coordinate. We keep all known one-hot columns. At deployment, unexpected values should still be monitored even when prediction remains possible.

### When order is real

For `low`, `medium`, and `high`, an ordinal encoding may express useful order. The values `[0,1,2]` additionally give equal numerical gaps to a linear or distance model. An ordinal scale alone does not justify those gaps. A threshold tree can use the ordering without multiplying by a coefficient, but a single threshold still divides a contiguous portion of the order; it cannot select an arbitrary subset of categories in one split.

For very many categories, one-hot width can become expensive or weakly supported. The deeper branch explains target encoding and hashing. Some estimators also provide native categorical treatment. Check the estimator's actual interface and treatment of categories instead of assuming every model needs the same numeric encoding.

## 4. A missing value is a question, not a zero

### Begin with a transparent estimate

Consider measured lengths `[10, 20, missing, missing]`. Median imputation fills both missing cells with 15. The filled table has mean 15, but the two missing measurements have not been discovered. Both `[10,20,10,20]`, with mean 15, and `[10,20,50,60]`, with mean 35, are compatible with the observed cells.

This distinction matters even when prediction improves. An imputer supplies a usable model input; it does not certify that an estimated value was physically measured.

A numeric **missingness indicator** adds a second feature that is 1 when the original value was absent and 0 otherwise. Now an actual 15 and an imputed 15 need not look identical to the model. This can help when absence contains predictive information, such as an optional measurement that technicians order selectively. If collection policy changes, that relationship can change too.

Fit the replacement value on training rows, then reuse it for later rows. For entirely missing training columns, specify a stable output policy: dropping the column, keeping an explicit empty feature, or refusing an unusable input. Our program uses `keep_empty_features=True` so its column structure is retained, although its actual numeric training columns are not entirely missing.

### Why the reason for missingness matters

Let $R$ say whether a measurement was observed. **MCAR** means the missingness process is independent of the data values. **MAR** allows missingness to depend on observed information but, conditional on that information, not additionally on the missing values. **MNAR** allows a remaining dependence on those unseen values. A scale that fails above an unrecorded weight limit illustrates the last case.

These describe a data-generating process, not a property a median imputer can establish from a blank cell. Observed data alone generally cannot distinguish MAR from every MNAR alternative. Understanding collection and performing sensitivity analysis matter. Adding an indicator does not, by itself, solve MNAR or recover valid scientific uncertainty. [Van Buuren's missingness introduction](https://stefvanbuuren.name/fimd/sec-MCAR.html) develops these assumptions through concrete measurement examples.

### Borrowing information from other rows

Nearest-neighbor imputation estimates a missing feature from similar rows that actually contain that feature. With incomplete rows, distance must be calculated using their available overlap.

Use these three donor rows, whose columns are $a,b,c$, and query `[2,12,missing]`:

| Donor | a | b | c |
| --- | ---: | ---: | ---: |
| D1 | 1 | 10 | 100 |
| D2 | 3 | missing | 300 |
| D3 | missing | 14 | 500 |

The nan-aware distance used here takes the squared differences on shared observed coordinates and multiplies by $m/q$, where $m=3$ is the total number of features and $q$ the number jointly observed. The three squared distances are:

\[
D1:\tfrac32(1^2+2^2)=7.5,\quad
D2:3(1^2)=3,\quad
D3:3(2^2)=12.
\]

With two neighbors and uniform weights, D2 and D1 supply $c=(300+100)/2=200$. D2 is a valid donor even though another feature is absent. For a different missing target column, donor eligibility may differ. If no donor has a defined overlap distance, the implementation needs a fallback; scikit-learn uses the relevant training feature's average when available. Scaling of observed features still affects these distances. [The imputation guide](https://scikit-learn.org/stable/modules/impute.html#nearest-neighbors-imputation) describes this feature-by-feature donor behavior.

**Investigation 2 — Who is allowed to donate?** Edit a donor cell or mark it absent and follow the eligible donors and imputed value immediately. The display crosses out unavailable distance coordinates, shows $m/q$, ranks eligible donors, and highlights only the cells contributing to the estimate. Changing a donor's unused feature can leave the answer unchanged; deleting its target measurement can make it ineligible.

Iterative imputation takes a different approach: initialize missing cells, fit one incomplete column from the others using rows where that column is observed, update its missing entries, then cycle through columns. This models relationships that a separate median ignores. It still depends on the chosen conditional models and on how missingness arose. The deeper uncertainty section explains why one completed table is different from multiple imputation.

## 5. Learn the preparation rule without looking ahead

There are two distinct operations:

- **Fit:** learn training medians, means, scales, and categories.
- **Transform:** apply those already learned values to a table.

A new row should not redefine the ruler. If the training minimum and maximum are 1 and 100, the new value 150 maps to $149/99$; fitting min–max again on the new batch would create a different coordinate system.

The same principle applies to evaluation. Split rows before learning preprocessing statistics. Fit preparation and the model using training rows. Apply the fitted preparation to the held-out rows, then count correct predictions. An estimate of future performance is compromised when the fitting procedure gets information it would not have at prediction time. Looking at held-out feature distributions to choose a transformation can also make an experiment adaptive, even without reading labels.

**Figure 4 — A fitted object crosses the boundary; observations do not cross backward.** Training rows create a saved median/scaler/vocabulary bundle. Both training and later inputs pass through that same bundle. Held-out outcomes enter the final comparison only. A separate forbidden backward arrow depicts fitting the bundle on all rows.

A pipeline packages this sequence so the software can repeat it consistently. It does not repair a feature that already leaks the target, a split that puts repeated subjects on both sides, or a manually fitted transformer created before the split. Next, [Cross-Validation & Hyperparameter Tuning](/learn/path/full-curriculum/cross-validation-hyperparameter-tuning?module=classical-ml) will repeat this fit/transform boundary inside each training/validation partition.

## 6. A complete experiment on real penguin measurements

The supplied `penguins.csv` contains 344 observations in the openly available Palmer Penguins dataset. We use four numeric measurements and recorded sex to predict Adelie, Chinstrap, or Gentoo. Bill length and depth are in millimeters, flipper length in millimeters, and body mass in grams. Two rows lack each of the four measurements; eleven lack recorded sex. The dataset's original research context is ecological measurement, and this small classification exercise does not establish performance on every future population or collection protocol. Dataset attribution and the original variables are described by the [Palmer Penguins authors](https://allisonhorst.github.io/palmerpenguins/reference/penguins.html).

We reserve 86 rows and fit on 258, keeping roughly the same species proportions with a fixed stratified split. Here “accuracy” simply means correct species predictions divided by 86. The model takes the majority species among the five nearest training rows.

Save the supplied CSV beside this program as `penguins.csv`. A compatible environment is Python 3.12 with NumPy 2.3.5, pandas 3.0.1, and scikit-learn 1.9.1; for a fresh environment install those packages with `python -m pip install numpy==2.3.5 pandas==3.0.1 scikit-learn==1.9.1`. The following complete experiment was executed during authoring through an equivalent calculation script using those versions.

```python
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler,
)

data = pd.read_csv(Path(__file__).with_name("penguins.csv"))
numeric = [
    "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g",
]
categorical = ["sex"]
X = data[numeric + categorical]
y = data["species"]
train, test = train_test_split(
    np.arange(len(data)), test_size=0.25, random_state=20, stratify=y,
)

def make_model(scaler):
    numeric_steps = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", scaler),
    ])
    category_steps = Pipeline([
        ("impute", SimpleImputer(
            strategy="constant", fill_value="not_recorded",
            keep_empty_features=True,
        )),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    prepare = ColumnTransformer([
        ("numeric", numeric_steps, numeric),
        ("category", category_steps, categorical),
    ])
    return Pipeline([
        ("prepare", prepare),
        ("classify", KNeighborsClassifier(n_neighbors=5)),
    ])

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(np.zeros((len(train), 1)), y.iloc[train])
baseline_prediction = baseline.predict(np.zeros((len(test), 1)))
print("majority", int(np.sum(baseline_prediction == y.iloc[test])), "/", len(test))

models = {}
for name, scaler in [
    ("raw", "passthrough"),
    ("standard", StandardScaler()),
    ("minmax", MinMaxScaler()),
    ("robust", RobustScaler()),
]:
    model = make_model(scaler)
    model.fit(X.iloc[train], y.iloc[train])
    prediction = model.predict(X.iloc[test])
    correct = int(np.sum(prediction == y.iloc[test]))
    print(name, correct, "/", len(test), f"{correct / len(test):.4f}")
    models[name] = model

prepare = models["standard"].named_steps["prepare"]
print(prepare.get_feature_names_out())
print(np.round(prepare.transform(X.iloc[test[:1]]), 4))
```

The recorded results are:

| Preparation for numeric features | Correct / held-out rows | Accuracy |
| --- | ---: | ---: |
| Majority-species baseline | 38 / 86 | 0.4419 |
| Median imputation, no scaling | 67 / 86 | 0.7791 |
| Median imputation, standard scaling | 84 / 86 | 0.9767 |
| Median imputation, min–max scaling | 85 / 86 | 0.9884 |
| Median imputation, robust scaling | 85 / 86 | 0.9884 |

All four models use the same categorical preparation, split, and neighbor count. This controlled comparison makes the influence of a numerical ruler visible. One extra correct row does not establish that min–max or robust scaling is generally superior to standard scaling. We have now inspected this held-out set across several alternatives; selecting a procedure from these results would require a separate evaluation plan. The next lesson builds that plan.

The standard model's fitted medians are `[45.0,17.3,197.0,4000.0]`. After imputation, its training means are approximately `[43.9306,17.0992,200.6589,4190.9884]`, and its scales `[5.4000,1.9387,14.2307,806.6876]`.

The first held-out row, zero-based source row 309, is `[51.0,18.8,203.0,4100.0,"male"]`. It becomes:

\[
[1.3091,\;0.8773,\;0.1645,\;-0.1128,\;0,\;1,\;0].
\]

The final three columns mean `sex_female`, `sex_male`, and `sex_not_recorded`. The negative mass coordinate says this mass is below the fitted mean; it does not mean a negative mass. Species, island, year, and row number were not included as features in this experiment.

**Investigation 3 — Follow a record through a fitted table pipeline.** Inspect any supplied held-out row and edit an independent copy to follow each transformed cell immediately. The visual forks numeric and categorical columns, exposes the saved training statistics, and rejoins the seven named output coordinates. Every displayed value must refer to the current valid record and selected cell. The experiment does not silently refit when you edit a later input.

As a practical extension, compare errors rather than only the score: standard scaling misclassified two rows here, whereas the raw model misclassified nineteen. Inspect their actual measured values and nearest-neighbor contributions before inventing a story about why. Do not treat a species label as available input while exploring those errors.

## 7. Deeper branch: change the shape, not only the ruler

Affine scaling maps $x$ to $(x-a)/b$. It preserves relative gaps within a feature up to a common factor. Sometimes the modeling question calls for a nonlinear relationship instead.

### Logs and power transforms

If a quantity varies multiplicatively, `log` can make ratios into differences: $\log(100)-\log(10)=\log(10)-\log(1)$. This is useful when equal multiplicative changes should have equal influence, as in a model of concentrations or elapsed times spanning several orders of magnitude. It does not justify taking the logarithm of arbitrary signed measurements.

For strictly positive $x$, the Box–Cox family is

\[
g_\lambda(x)=
\begin{cases}(x^\lambda-1)/\lambda,&\lambda\ne0,\\\log x,&\lambda=0.\end{cases}
\]

Yeo–Johnson extends a related family to zero and negative values:

\[
g_\lambda(x)=
\begin{cases}
((x+1)^\lambda-1)/\lambda,&x\ge0,\lambda\ne0,\\
\log(x+1),&x\ge0,\lambda=0,\\
-((1-x)^{2-\lambda}-1)/(2-\lambda),&x<0,\lambda\ne2,\\
-\log(1-x),&x<0,\lambda=2.
\end{cases}
\]

Check the special cases rather than memorizing a name: $\lambda=1$ gives $g(x)=x$ on both sides. At $\lambda=0$, nonnegative inputs use `log1p`, but negative inputs use the negative quadratic branch. At $\lambda=2$, negative inputs use a logarithm, not a square root. `PowerTransformer` fits its parameter per training feature by a likelihood criterion and standardizes afterward by default. Better marginal symmetry is a possible useful result; it is not a guarantee of jointly Gaussian features or correctly modeled residuals. The [preprocessing guide's nonlinear section](https://scikit-learn.org/stable/modules/preprocessing.html#non-linear-transformation) gives the API and definitions.

### Quantiles answer a different question

A quantile transform replaces a value by where it lies in the fitted distribution. For training values `[1,2,3,4,100]`, a simple illustrative rank coordinate $(r-1)/(5-1)$ maps the sorted observations to `[0,.25,.5,.75,1]`. The large final gap becomes the same rank gap as the others. This rank calculation explains the idea; interpolation, ties, and endpoint handling in an actual transformer must be specified separately.

Mapping those probabilities through an inverse normal CDF gives normal-quantile coordinates, with finite endpoint handling in software. Rank ordering is generally retained where the map is strictly increasing, but original numerical gaps are not. Ties and saturation outside the fitted range can merge values, making a complete inverse impossible. A “Gaussian-looking” histogram can therefore hide an important loss of magnitude information.

**Figure 5 — Distance versus rank.** Draw labeled links from the five measured values to their exact rank coordinates. Keep 100 identified. Beside the rank map, show a log map with actual computed coordinates. Ask whether changing 100 to 1,000 changes its rank or its log distance; these representations preserve different information.

### Features can also express thresholds and interactions

Discretization assigns a value to a fitted interval: for thresholds 10 and 20, a quantity can be represented as `below 10`, `10 to below 20`, or `20 and above`. One-hot interval features let a linear model fit a stepwise response, at the cost of losing within-bin differences. Bin rules must be fitted on training data when they are data-dependent.

A polynomial map can instead add $x^2$ or $x_1x_2$. The model remains linear in its fitted coefficients while its response varies nonlinearly with the original variables. Spline bases provide smoother local building blocks. These are choices about which relationships a model can express, beyond simply fixing units. A custom deterministic transform, such as converting an angle to sine and cosine, can express that 359° and 1° are nearby. Keep the original unit and period explicit: `sin` and `cos` expect radians in NumPy.

This circular representation is particularly useful for direction or time of day. It avoids declaring midnight far from 23:59, while preserving the fact that morning and evening can differ. It would be inappropriate for elapsed time, where completing a 24-hour cycle does not erase duration.

## 8. Deeper branch: high-cardinality categories and target information

### Why a category average can accidentally contain the answer

Suppose many rows carry a product identifier, and the target is whether a product was returned. A smoothed target encoding represents category $c$ by

\[
t_c=\frac{\sum_{i:x_i=c}y_i+\alpha\mu}{n_c+\alpha},
\]

where $n_c$ counts training examples of the category, $\mu$ is the relevant training target mean, and $\alpha\ge0$ controls the pull toward that mean. An unseen category maps to $\mu$. A category with one positive example and $\alpha=2,\mu=.5$ maps to $2/3$, rather than an unqualified 1.

The danger is easiest to see without smoothing: if a category occurs once, its training encoded value equals that row's target. The model is being given part of the answer. Smoothing reduces this direct influence but does not replace a separation rule.

**Cross-fitting** generates each training row's encoded feature using other training rows. Divide the outer training set into internal folds. For one fold, learn category sums, counts, **and the prior mean** from the other folds, then encode the held-out internal rows. Repeat until each training row has an out-of-fold representation. Once the downstream model is trained, a new external row is encoded from statistics fitted on the whole outer training set. Outer evaluation targets are never used.

Here is an exact six-row example, with smoothing 2:

| Row | Category | Target | Internal fold | Cross-fitted value |
| --- | --- | ---: | ---: | ---: |
| 0 | A | 1 | 0 | 5/9 |
| 1 | A | 1 | 1 | 7/9 |
| 2 | B | 0 | 0 | 2/9 |
| 3 | B | 0 | 1 | 4/9 |
| 4 | C | 1 | 0 | 2/9 |
| 5 | C | 0 | 1 | 7/9 |

For fold 0, donor rows 1, 3, and 5 have targets `[1,0,0]`, so the prior is $1/3$. Row 0's A encoding is $(1+2/3)/3=5/9$. For fold 1, the donor targets are `[1,0,1]`, giving prior $2/3$; row 1 receives $(1+4/3)/3=7/9$.

Change only row 0's target from 1 to 0. Encodings for held-out fold 0 remain unchanged because their donor set did not change. Encodings for fold 1 become `[2/9,2/9,5/9]`. Notice that B changes even though no B target changed: the fold-specific prior changed. Computing one global prior before internal splitting would let a held-out target affect its own encoding through smoothing.

The following complete teaching calculation reproduces the table and this contrast using NumPy from the earlier setup:

```python
import numpy as np

def cross_fit(categories, target, folds, smoothing=2.0):
    encoded = np.empty(len(target), dtype=float)
    for held_fold in np.unique(folds):
        donors = folds != held_fold
        prior = target[donors].mean()
        for row in np.flatnonzero(~donors):
            matching = donors & (categories == categories[row])
            encoded[row] = (
                target[matching].sum() + smoothing * prior
            ) / (matching.sum() + smoothing)
    return encoded

categories = np.array(["A", "A", "B", "B", "C", "C"])
target = np.array([1., 1., 0., 0., 1., 0.])
folds = np.array([0, 1, 0, 1, 0, 1])
print(np.round(cross_fit(categories, target, folds), 6))
changed = target.copy()
changed[0] = 0.
print(np.round(cross_fit(categories, changed, folds), 6))
```

Expected arrays are `[.555556,.777778,.222222,.444444,.222222,.777778]` and `[.555556,.222222,.222222,.222222,.222222,.555556]`. This bounded calculation was executed during authoring.

**Investigation 4 — Trace which target can affect which encoded row.** Edit a category or target and follow the selected row’s encoding alongside the internal donor graph and separate category/prior contributions. Editing the inspected row's own target is a checked null; editing one of its donors can be a checked contrast. Fold membership stays visible throughout.

For production use, `TargetEncoder.fit_transform` supplies internal cross-fitting, whereas `fit(...).transform(...)` does not produce the same training representation. In scikit-learn 1.9, `cv` can accept a splitter or iterable of splits; older examples using encoder-level `shuffle` and `random_state` are being deprecated. Group or time relationships require appropriate internal and outer splits. The default shuffled split cannot decide that for you. See the [current TargetEncoder API](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.TargetEncoder.html) and its [worked cross-fitting example](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder_cross_val.html).

### Hashing trades a learned vocabulary for collisions

Feature hashing assigns each category or token to one of a fixed number of buckets. A signed version also assigns a deterministic sign and adds the signed feature value to that bucket. It is not merely a binary flag.

For a constructed map, let `apple` and `pear` both use bucket 0, with signs +1 and −1, and `banana` use bucket 1 with sign +1. Counts `apple:3, pear:1, banana:2` produce `[2,2]`. Counts `apple:2, banana:2` produce the same vector. The collision makes the original dictionary impossible to recover from this vector alone.

Hashing can bound memory and accept previously unseen names without growing a vocabulary. With suitable random-hash assumptions, signed hashing preserves inner products in expectation; that expectation is not a guarantee for every pair under a fixed small hash table. Increasing the number of buckets reduces typical collision pressure while increasing model width. The [original feature-hashing paper](https://arxiv.org/pdf/0902.2206) derives this tradeoff.

## 9. Deeper branch: completing data versus representing uncertainty

Iterative conditional prediction is useful for building model input, but one completed dataset treats its filled cells as if there were no uncertainty about them. A point estimate can look reasonable while standard errors are too small if uncertainty from missing data is ignored.

Multiple imputation creates several plausible completed datasets under an explicit imputation model, fits the intended analysis to each, then combines **analysis estimates**, rather than averaging the filled tables first. Properly representing uncertainty requires more than running a deterministic imputer with a different random seed; the imputations must reflect the relevant conditional uncertainty and assumptions.

For a scalar estimate, let $\hat\theta_k$ be the result from completed dataset $k$, $U_k$ its estimated variance, and $m$ the number of completed datasets. Define

\[
\bar\theta=\frac1m\sum_k\hat\theta_k,\quad
\bar U=\frac1m\sum_kU_k,\quad
B=\frac1{m-1}\sum_k(\hat\theta_k-\bar\theta)^2.
\]

Rubin's pooling rule uses total variance $T=\bar U+(1+1/m)B$. The first term captures uncertainty within each completed-data analysis; the second captures variation between plausible completions, with a finite-$m$ adjustment. For estimates `[9,10,11]` and within-analysis variances `[4,4,4]`, the pooled estimate is 10, $B=1$, and $T=4+4/3=16/3$. Its standard error is about 2.309, larger than 2 from treating the completion as certain. Constructing intervals also requires the appropriate degrees-of-freedom calculation and imputation assumptions; this small arithmetic example is not an automatic validity certificate. The [mice pooling documentation](https://amices.org/mice/reference/pool.html) explains the analysis-then-pool workflow and available small-sample treatment.

`IterativeImputer` is an experimental scikit-learn estimator and returns a single completion per transform. Its optional posterior sampling can support repeated completions under the chosen estimator, but a sound multiple-imputation analysis also requires a compatible scientific model, diagnostics, and pooling. A predictive pipeline and a scientific missing-data analysis share tools while answering different questions.

**Figure 6 — Several plausible tables, several estimates, one pooled analysis.** Branch an incomplete table into three symbolic completions, then show the exact estimates 9, 10, 11 and variances 4. Link these to within/between variance contributions. Do not depict repeated identical tables as uncertainty or imply the imputed cells were measured.

## 10. Practice: explain the representation before choosing the function

Try each question before opening its hint or solution. The first six use only the core route.

### 1. A changed ruler

Query `[0,0]` has candidates A `[2,60]` and B `[5,10]`. Which is nearest under raw squared distance? Which is nearest with divisors `[1,30]`?

<details><summary>Hint</summary>Calculate one contribution per feature; the divisor belongs inside the square.</details>
<details><summary>Solution</summary>Raw distances squared are 3,604 and 125, so B wins. Scaled distances squared are $4+4=8$ and $25+1/9=25.111\ldots$, so A wins. The input observations are unchanged; relative feature weighting changed.</details>

### 2. Fit once, transform later

A training column is `[2,4,6]`. Find its mean, population-style standard deviation, standardized value for a new 8, and min–max value for that 8. Should adding 8 to the later batch change the saved training statistics?

<details><summary>Hint</summary>The training squared deviations are 4, 0, and 4.</details>
<details><summary>Solution</summary>The mean is 4 and standard deviation $\sqrt{8/3}$. The standardized new value is $4/\sqrt{8/3}=\sqrt6\approx2.4495$. Min–max gives $(8-2)/(6-2)=1.5$. Transformation reuses the fitted statistics; refitting on later inputs would create a different map.</details>

### 3. What did normalization discard?

A spectrum `[2,1,2]` and another `[6,3,6]` are L2-normalized. What are the results? Would this be appropriate if total emitted energy is the prediction signal?

<details><summary>Solution</summary>The lengths are 3 and 9, so both map to `[2/3,1/3,2/3]`. Relative shape remains, but the threefold intensity difference disappears. If total energy matters, preserve a magnitude feature or choose another representation rather than discarding it blindly.</details>

### 4. A category that did not exist during fitting

A full one-hot vocabulary has `small`, `medium`, and `large`, and an unknown value uses an all-zero block. Compare unknown-to-small distance with small-to-medium distance. What application decision is hidden behind accepting the unknown value?

<details><summary>Solution</summary>The distances are 1 and $\sqrt2$. Ignoring unknown categories creates a representation with a specific geometry; it is not neutral. The application must decide whether a new value is valid, should trigger a review or fallback, or belongs in a deliberately learned other-category group.</details>

### 5. Changed donors

In the imputation table, change D2's `c` to missing. With two neighbors, what value replaces the query's missing `c`? What if D1's `c` changes from 100 to 140 while the other original cells remain unchanged?

<details><summary>Hint</summary>First decide who can donate the target feature, then use the distances on the query's observed coordinates.</details>
<details><summary>Solution</summary>With D2 ineligible, D1 and D3 supply $(100+500)/2=300$. In the separate second change, the original selected donors D2 and D1 remain nearest, so the estimate becomes $(300+140)/2=220$. Changing a value in the query's missing target column does not itself enter these overlap distances.</details>

### 6. Explain a transformed real record

For the fitted standard penguin pipeline, keep the first held-out record unchanged except set its body mass to missing. Predict that output coordinate. Does this mean the animal's true mass was 4,000 g?

<details><summary>Hint</summary>First apply the saved median, then the saved mean and scale.</details>
<details><summary>Solution</summary>The coordinate becomes $(4000-4190.9883721)/806.6875829\approx-0.2368$. This is a numerical estimate passed to the model. It is not a recovered observation. Other coordinate values and the fitted statistics remain unchanged.</details>

### 7. Which part of target encoding changed? — deeper

In the six-row example, change row 3's target from 0 to 1. Calculate row 0's new encoding. Does row 3's own cross-fitted encoding change?

<details><summary>Solution</summary>Row 0's donors are rows 1, 3, and 5, now with prior $2/3$. Its A donor is still positive, so its value becomes $7/9$. Row 3 belongs to held-out fold 1, whose donor targets are unchanged, so its own value remains $4/9$. The first change passes through the prior; the null demonstrates excluding one's own target.</details>

### 8. Identity is a useful transform check — deeper

Use the Yeo–Johnson formula at $\lambda=1$ on 3 and −3. Why is a generic claim that “negative inputs use a square root” wrong?

<details><summary>Solution</summary>For 3, $((3+1)^1-1)/1=3$. For −3, $-((1+3)^1-1)/1=-3$. Both branches depend on $\lambda$; the negative exponent is $2-\lambda$, with a logarithmic limit at $\lambda=2$, not a fixed square root.</details>

### 9. Pool estimates, not completed tables — deeper

Four completed-data analyses yield estimates `[8,10,10,12]`, each with variance 1. Find the pooled estimate and total variance using the stated rule.

<details><summary>Solution</summary>The average is 10. Squared deviations sum to 8, so $B=8/3$. Then $T=1+(1+1/4)(8/3)=13/3\approx4.3333$. Its standard error is $\sqrt{13/3}\approx2.0817$. Between-completion disagreement contributes substantial uncertainty; validity still depends on how the completions and analyses were constructed.</details>

## 11. Readiness and the next connection

You are ready to continue when you can trace one record through a fitted imputer, scaler, and encoder; explain how scaling changes a distance; distinguish a missing value from an unknown category; and keep the fit/transform boundary intact for a new row. Those are core skills, independent of whether you have finished the deeper branches.

The next topic is [Cross-Validation & Hyperparameter Tuning](/learn/path/full-curriculum/cross-validation-hyperparameter-tuning?module=classical-ml). Here, we held a split fixed to inspect a representation. Next, we will decide how to compare multiple procedures without repeatedly treating the same observations as fresh evidence. Later regularization makes the connection between feature units and coefficient penalties explicit; feature selection asks which input information should remain at all.

## References and other ways to learn

- [Scikit-learn preprocessing guide](https://scikit-learn.org/stable/modules/preprocessing.html): the technical reference for current scaler, nonlinear transform, categorical, binning, polynomial and custom-transform behavior. Read the subsection matching a representation you can already explain, then inspect its API rather than treating the whole page as a required first pass.
- [Compare scalers on data with outliers](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html): a visual alternative with full-range and magnified views of actual housing data. Compare robust scaling with a quantile transform and explain what happens to an identified extreme observation. The example's code and figure descriptions were inspected; its runtime is not our benchmark.
- [Imputation guide](https://scikit-learn.org/stable/modules/impute.html): current simple, iterative, neighbor and indicator behavior, including entirely missing columns. Use it to check a proposed imputer's exact contract.
- [Target Encoder's Internal Cross Fitting](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder_cross_val.html): a worked code-and-results alternative showing why near-unique categories can overfit without cross-fitting. Follow where `fit_transform` occurs inside its pipeline.
- [Palmer Penguins dataset and project](https://allisonhorst.github.io/palmerpenguins/): measurements, dataset context, and CC0 availability, by Allison Horst, Alison Hill, and Kristen Gorman, using Palmer Station penguin observations collected by Gorman and colleagues. Our attached CSV preserves the public file; the accompanying provenance records its exact hash and use.
- [Flexible Imputation of Missing Data: missingness concepts](https://stefvanbuuren.name/fimd/sec-MCAR.html) and [mice pooling reference](https://amices.org/mice/reference/pool.html): a conceptual measurement-based reading and a concrete analysis workflow for the deeper uncertainty branch. Do not substitute a prediction score for evidence that an inferential missingness assumption is valid.
- [Weinberger and colleagues, Feature Hashing for Large Scale Multitask Learning](https://arxiv.org/pdf/0902.2206): primary treatment of signed hashing and its inner-product analysis; useful when a fixed-memory representation matters more than recovering an explicit vocabulary.
