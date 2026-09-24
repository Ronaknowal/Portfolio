# PCA & Dimensionality Reduction

> Current lab UX, 21 September 2026: controls show live calculations and topic-specific visuals without learner prediction entry, grading or guess-to-reveal screens. Genuine algorithm steps, separate practice and data-role boundaries remain. See [the current migration record](../../LIVE-EXPLORATION-CLASSICAL-EARLY.md).

<!-- Content-first manuscript, 12 September 2026. Visual anchors refer to visual-specifications.md; they are authoring instructions, not learner-facing placeholders. Rendered visuals/labs and website integration are deferred. Preserve the explanations even when adapting their presentation. -->

## 1. Can we keep fewer numbers without losing the pattern?

Imagine receiving a spreadsheet with 178 wines and 13 chemical measurements for each wine. You want to compare the samples. Thirteen separate columns are easy to store but hard to see together: a scatterplot has only two axes. Which two numbers should represent each wine?

One option is to pick two existing measurements. Another is to make two new measurements by combining the original thirteen. If several chemical quantities rise and fall together, a shared combination could describe their variation more efficiently than separate columns.

**Principal component analysis, or PCA, finds such combinations.** It orders them by how much variation they capture in the data you give it. You can keep all the combinations as a new coordinate system, or keep only the first few to obtain a smaller representation. Keeping fewer coordinates is the dimensionality reduction step.

The aim here is concrete: make a compact view, recover approximate measurements from it, and decide whether the information lost is acceptable for the intended use.

Our real example is the [UCI Wine dataset](https://archive.ics.uci.edu/dataset/109/wine): chemical analyses from three cultivars grown in the same Italian region, supplied for studying wine origin. A row is one wine; the 13 numeric inputs include alcohol, malic acid, color intensity and proline. The cultivar label is separate. It will help us inspect the result, but PCA will receive only the measurements. The repository supplies [an offline CSV](wine.csv), with attribution in [the data notes](data-provenance.md).

First we will use four invented points whose arithmetic fits on a page. Then we will return to the wines and answer two different questions: what makes a useful two-dimensional picture, and what makes an acceptable compressed measurement record?

**First-pass route.** Read sections 1–6, try practice 1–4 in section 9, and use the readiness check in section 11. That gets you from a picture to a working analysis. Section 7 develops interpretation, section 8 connects PCA to clustering and prediction, and section 10 contains the deeper mathematics, computation and applications. The branches can be read when their questions become relevant. Allow roughly 40–55 minutes for the first pass and another 60–90 minutes for code and practice; derivations and extensions take additional time.

You need averages, squared distances and basic array operations. We will introduce the new mathematical notation as we use it. If `@` or array shapes are unfamiliar, review [NumPy arrays and broadcasting](/learn/path/full-curriculum/numpy-arrays-broadcasting-vectorization?module=programming-scientific-computing).

## 2. Rotate the ruler, then keep its readings

Suppose two sensors describe the position of a moving marker. Their readings have the same unit. We observe:

| Observation | First reading | Second reading |
| --- | ---: | ---: |
| A | 1 | 1 |
| B | 2 | 0 |
| C | 4 | 4 |
| D | 5 | 3 |

The points form a narrow diagonal cloud. If we could record only one number per observation, a ruler laid along that diagonal would distinguish the lower-left observations from the upper-right observations. A ruler laid across the cloud would mostly measure its small thickness.

<!-- Figure F1: four labeled observations, centroid, diagonal principal axis, perpendicular projection feet and score strip; immediate inline view. -->

### Put the origin in the middle of the cloud

The average reading is `(3, 2)`. Subtract it from every point:

| Observation | Centered first reading | Centered second reading |
| --- | ---: | ---: |
| A | −2 | −1 |
| B | −1 | −2 |
| C | 1 | 2 |
| D | 2 | 1 |

This operation is **centering**. It changes where zero is, while preserving all distances between points. We now describe differences from the average observation.

The fitted line passes through the mean in the original picture and through zero in the centered picture. Without centering, an SVD of the raw readings solves a different problem: fitting directions through the original zero. A large offset can then influence the direction. It need not dominate every dataset, but it is no longer the same centered PCA calculation.

### A direction tells us where the ruler points

A direction is a vector. Use

\[
v_1=\frac{1}{\sqrt 2}(1,1).
\]

Both entries are approximately `0.7071`. Dividing by `√2` makes its length one. This **unit-length** convention matters: otherwise doubling the numbers in the direction would double all ruler readings without improving the direction.

For a centered point `a = (a₁, a₂)`, its reading along the ruler is the **dot product**:

\[
z=a\cdot v_1=a_1v_{11}+a_2v_{12}.
\]

This number is its **score** on the first principal component. A direction belongs to the whole fitted model; a score belongs to one observation.

For A, the score is `−2/√2 − 1/√2 = −3/√2 ≈ −2.1213`. For D, it is `3/√2 ≈ 2.1213`. A negative score means “on the other side of the mean along the chosen direction,” not a negative concentration or an invalid measurement.

### Reconstruct an observation from its score

Multiply the score by the direction to return to a point on the line. Then add the mean to return to the original coordinates:

\[
\widehat{x}=\mu+zv_1.
\]

For A:

\[
(3,2)+\frac{-3}{\sqrt2}\frac{(1,1)}{\sqrt2}
=(3,2)+(-1.5,-1.5)=(1.5,0.5).
\]

The reconstructed A is close to `(1,1)`, but it is not identical. The residual—the original minus its reconstruction—is `(-0.5, 0.5)`.

| Observation | One score | Reconstruction | Squared distance lost |
| --- | ---: | --- | ---: |
| A | −2.1213 | (1.5, 0.5) | 0.5 |
| B | −2.1213 | (1.5, 0.5) | 0.5 |
| C | 2.1213 | (4.5, 3.5) | 0.5 |
| D | 2.1213 | (4.5, 3.5) | 0.5 |

A and B now have the same representation. Their difference lay across the ruler, in the direction we discarded. Compression has a visible meaning: two distinct observations can become indistinguishable.

**Pause and predict.** If you reverse the ruler, making the direction `−v₁`, what happens to A's score and reconstruction?

<details><summary>Check the reasoning</summary>

The score changes sign. The direction also changes sign, so their product does not: `(-z)(-v₁) = zv₁`. The reconstruction is still `(1.5, 0.5)`. This is why a sign-flipped component can describe exactly the same model.

</details>

### Investigation: find the most useful ruler

<!-- Lab L1: projection workbench. Compare live residuals on editable observations using a direction handle and exact-value keyboard inputs. -->

Start with the four observations. Record whether you expect turning the horizontal ruler toward the diagonal to decrease, increase or preserve the total squared reconstruction error. Rotate it, compare the result with your prediction, and inspect the perpendicular residual segments.

Then change one observation yourself. Does the best direction move toward it? Finally, translate all four observations by the same amount and refit. Explain why the mean moves but the centered geometry stays the same. Reset restores the four original observations so you can check the calculation above.

## 3. Why maximum spread and minimum loss give the same answer

For each centered point, the projection and its residual form a right triangle. The squared length of the original vector is the sum of the two squared lengths:

\[
\|a\|^2=\|\widehat a\|^2+\|a-\widehat a\|^2.
\]

The symbol `‖a‖²` means the sum of the squared coordinates of `a`. Add this identity over all observations:

\[
\text{total centered squared length}
=\text{retained squared length}+\text{residual squared length}.
\]

The total on the left does not change when we turn the ruler. Therefore the direction that retains the most squared length also loses the least. These are two views of the same optimization, not competing definitions of PCA.

For the four-point example, total centered squared length is `5 + 5 + 5 + 5 = 20`. The diagonal retains `18` and loses `2`. The horizontal ruler retains `10` and loses `10`. The perpendicular diagonal retains `2` and loses `18`.

<!-- Figure F2: three true-scale small multiples of the projection and retained/residual length totals, with the horizontal baseline. -->

### From spread to variance

The sample variance of a list of centered readings is their squared sum divided by `n − 1`, where `n` is the number of observations. For our four scores:

\[
\lambda_1=\frac{18}{4-1}=6.
\]

The perpendicular direction `v₂ = (1, −1)/√2` has score variance `2/3`. It is the second principal direction. Its scores complete the coordinate system:

| Observation | First score | Second score |
| --- | ---: | ---: |
| A | −3/√2 | −1/√2 |
| B | −3/√2 | 1/√2 |
| C | 3/√2 | −1/√2 |
| D | 3/√2 | 1/√2 |

Keeping both scores permits exact reconstruction. Keeping the first score retains

\[
\frac{6}{6+2/3}=0.9=90\%
\]

of the total sample variance. This is the **explained variance ratio**.

Its precise meaning is valuable: on the data used to fit this centered, unwhitened PCA, the retained coordinates contain 90% of the total squared deviation from the mean. The other 10% is reconstruction loss in the same geometry.

**What explained variance does—and does not—certify.** It measures this squared-error objective. It is not a percentage of facts, class information or scientific meaning preserved. A low-variance direction can contain the distinction a task needs; a large-variance direction can reflect an unwanted artifact. Section 8 constructs that case, and section 10.3 separates finite-sample variation from population structure. Use this distinction whenever interpreting a percentage below.

### Error units: three different summaries of the same residuals

Our example has total squared error, or **SSE**, equal to `2`. The average squared distance per observation is `2/4 = 0.5`. The average squared error per numeric entry is `2/(4×2) = 0.25`.

They are all correct, but answer different questions. A statement such as “MSE is 0.25” needs its denominator. Hereafter, a reconstruction MSE means an average over all entries; the projection workbench shows SSE so its pieces add directly.

## 4. Fit once, project, and reconstruct in Python

Let `X` have shape `(n, d)`: `n` observations and `d` input features. Fit a mean and `k` directions from permitted fitting data. Then transform any observation by subtracting that same mean and taking its `k` dot products.

<!-- Figure F3: X(n,d) -> centering -> A(n,d) -> multiply V_k(d,k) -> Z(n,k); inverse multiplication and mean restoration shown separately. Training and later-data branches use the same fitted quantities. -->

Use Python with NumPy and scikit-learn. If needed, install them in a project environment:

```sh
python -m pip install numpy scikit-learn
```

The examples were checked with Python 3.12.14, NumPy 2.3.5 and scikit-learn 1.9.1. They use the CPU and require no network once the packages and data are available. Printed floating-point values are rounded; tiny final-bit differences are normal.

### A compact implementation using SVD

SVD is a matrix decomposition that supplies the directions and their strengths. You can use the program now; section 10.1 explains why it computes PCA.

```python
import numpy as np

X = np.array([[1., 1.], [2., 0.], [4., 4.], [5., 3.]])
mean = X.mean(axis=0)
centered = X - mean
_, singular_values, directions = np.linalg.svd(centered, full_matrices=False)

k = 1
kept_directions = directions[:k]       # shape: (k, d)
scores = centered @ kept_directions.T # shape: (n, k)
reconstructed = scores @ kept_directions + mean
variances = singular_values**2 / (len(X) - 1)

print(np.round(variances, 4))
print(np.round(variances / variances.sum(), 4))
print(np.round(reconstructed, 4))
print(round(np.mean((X - reconstructed)**2), 4))
```

```text
[6.     0.6667]
[0.9 0.1]
[[1.5 0.5]
 [1.5 0.5]
 [4.5 3.5]
 [4.5 3.5]]
0.25
```

`directions` stores principal directions as **rows**. That is why the forward multiplication uses `.T` and the reconstruction does not. This convention agrees with scikit-learn's `components_`. [NumPy's SVD reference](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html) specifies the factor shapes.

This short program assumes a finite numeric matrix with at least two rows and nonzero total variation. Missing-value treatment and constant data are discussed in section 10.2. It is a teaching implementation, with no automatic choice of scaling or component count.

### The library version exposes the same operations

Run this after the preceding block:

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=1, svd_solver="full")
scores = pca.fit_transform(X)
reconstructed = pca.inverse_transform(scores)

new_observation = np.array([[6., 4.]])
new_scores = pca.transform(new_observation)
new_reconstruction = pca.inverse_transform(new_scores)

print(pca.components_.shape, scores.shape)
print(np.round(new_reconstruction, 4))
```

```text
(1, 2) (4, 1)
[[5.5 4.5]]
```

`fit` learns the mean and directions; `transform` uses them; `inverse_transform` reconstructs. For the new point, `(6,4) − (3,2) = (3,2)`. Its retained component is `(2.5,2.5)`, giving `(5.5,4.5)` after restoring the mean. We did not move the fitted ruler to accommodate the new observation.

PCA centers its inputs. It does **not** standardize their scales automatically. [PCA API reference](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).

## 5. Scaling changes which differences matter

A squared distance adds differences from all features. If one feature is in thousands and another is in tenths, the first can dominate that sum. A change of measurement unit can therefore change the PCA answer without changing the objects being measured.

### An exact unit-change experiment

Take four centered points: `(-2,-1)`, `(-2,1)`, `(2,-1)`, `(2,1)`. Variation along the horizontal coordinate is four times variation along the vertical coordinate. PC1 is horizontal and retains 80% of the variance.

Now multiply the second coordinate by 10—for example, express the same lengths in a unit ten times smaller. Horizontal variance stays `16/3`; vertical variance becomes `400/3`. PC1 becomes vertical and retains `400/416 ≈ 96.15%`.

PCA has answered the new numerical question correctly. We changed the relative penalty assigned to errors on the two axes.

**Standardization** divides each centered feature by its own standard deviation:

\[
y_{ij}=\frac{x_{ij}-\mu_j}{s_j}.
\]

A difference of one then means one fitted standard deviation for that feature. On the rectangle, standardization produces a square with equal variance along the two coordinates. No unique first direction is preferred. It does not uncover a secret diagonal; it makes the symmetry explicit.

<!-- Lab L2: unit-and-metric investigation, rectangle edits and unit multiplier, raw/standardized geometry; ties shown as ties. -->

In the scaling investigation, predict whether changing one unit will change the leading direction. Try a multiplier you choose, then repeat with standardization. Edit the rectangle's width or height and explain when there is a preferred direction and when there is a tie.

### Make a scaling decision for the wines

For this exploratory view, we want each chemical feature to contribute on a comparable relative scale. We will standardize. That is a modeling choice, not a universal preprocessing law. If a later task supplies measurement-error variances or physical costs, those may define a more appropriate weighting. Standardizing a nearly constant noisy feature can give its noise disproportionate influence.

The following analysis describes **all 178 supplied observations**. Its purpose is to compare representations of this fixed collection. The next section starts a separate train/validation analysis for choosing a representation for other observations.

```python
import numpy as np
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

wine = load_wine()  # included with scikit-learn; no dataset download
X, cultivar = wine.data, wine.target

raw_pca = PCA(svd_solver="full").fit(X)
scaler = StandardScaler().fit(X)
standardized = scaler.transform(X)
scaled_pca = PCA(svd_solver="full").fit(standardized)
scores = scaled_pca.transform(standardized)

print(X.shape, np.bincount(cultivar))
print(np.round(raw_pca.explained_variance_ratio_[:2], 4))
print(np.round(scaled_pca.explained_variance_ratio_[:3], 4))
print(round(scaled_pca.explained_variance_ratio_[:2].sum(), 4))
```

```text
(178, 13) [59 71 48]
[0.9981 0.0017]
[0.362  0.1921 0.1112]
0.5541
```

The raw first component is almost entirely the proline direction. Its roughly 99.8% variance share reflects the original column magnitudes. After standardization, PC1 retains about 36.2% and PC2 about 19.2%. Their combined 55.4% is a substantially less complete reconstruction of the standardized measurements than a casual two-dimensional picture might suggest.

That lower percentage does not make standardization a failure. We changed what counts as a large error. Comparing the raw and standardized percentages as though they measured the same objective would be like comparing a distance in centimeters with an unrelated distance in seconds.

<!-- Figure F4: Wine raw/standardized variance bars, standardized score scatter with axes marked 36.20%/19.21%, optional cultivar shapes, and coefficient display. Distinguish fixed-collection exploration from the next section's held-out workflow. -->

Plotting the cultivar labels over the scores lets us ask whether this unsupervised summary aligns with a known grouping. Do not include the label as a fourteenth numeric feature. Otherwise the plot would partly encode the answer we hoped to inspect. Class numbers `0,1,2` in scikit-learn are codes, not quantities with meaningful distances; the CSV uses the original `1,2,3` labels.

## 6. Choose the number of components for a stated purpose

Before choosing `k`, finish this sentence: “I need the representation to…”

| Purpose | A useful decision rule | What to examine |
| --- | --- | --- |
| Show the observations | Start with two components; inspect further pairs if helpful | Labeled score plots, retained variation and original features |
| Store approximate measurements | Smallest `k` meeting an explicit error budget | Reconstruction error, feature-level residuals and total storage |
| Help a prediction model | Compare the complete pipeline with a no-PCA baseline | Validation performance on the actual task |
| Reduce measurement noise | Evaluate against a justified signal/noise model or independent target | Error relative to the target, not merely the noisy input |

A **scree plot** shows the variance contributed by each component. A cumulative plot adds those contributions. An elbow can suggest where gains become smaller, but a clear elbow need not exist. For the standardized full Wine collection, the first two components retain 55.41%, eight retain 92.02%, and ten retain 96.17%. Those are useful accounting facts, not universal cutoff recommendations.

### Keep fitting information separate from evaluation information

To choose a transform for future observations, split the data first. Learn imputation, means, scales and PCA directions on the training portion. Apply those fitted operations unchanged to validation observations. If the observations are grouped by person, session, source or time, make the split reflect the intended use rather than mixing related rows arbitrarily.

This is the lesson's home for **data leakage**: even a transform that never reads labels can leak evaluation information through its fitted means or directions. A pipeline passed into cross-validation refits its preprocessing within each training fold. Putting previously transformed full data into cross-validation does not. [Scikit-learn's leakage guide](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage).

### A worked compression budget

For an exercise in measurement reconstruction, set this budget before examining the results: retain the fewest components whose validation squared error is at most **10% of the error from always returning the training mean**.

Our baseline stores no per-observation coordinates. Every reconstructed validation row is the training mean. We measure all errors in the same training-standardized coordinates. A ratio of `0.10` means 90% less squared error than this baseline on the validation rows; it is not a classification score.

```python
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

wine = load_wine()
train, validation = train_test_split(
    np.arange(len(wine.data)), test_size=0.25,
    random_state=42, stratify=wine.target
)
scaler = StandardScaler().fit(wine.data[train])
A = scaler.transform(wine.data[train])
B = scaler.transform(wine.data[validation])
pca = PCA(svd_solver="full").fit(A)

baseline_mse = np.mean((B - pca.mean_)**2)
ratios = []
for k in range(14):
    directions = pca.components_[:k]
    scores = (B - pca.mean_) @ directions.T
    reconstruction = scores @ directions + pca.mean_
    ratios.append(np.mean((B - reconstruction)**2) / baseline_mse)

print(len(train), len(validation))
print(round(baseline_mse, 4))
for k in [0, 2, 7, 8, 10, 13]:
    print(k, round(ratios[k], 4))
print(next(k for k, ratio in enumerate(ratios) if ratio <= 0.10))
```

```text
133 45
1.0963
0 1.0
2 0.4281
7 0.1265
8 0.096
10 0.0521
13 0.0
8
```

The label is used only to keep the three cultivar proportions represented in both portions, a **stratified split**. PCA still receives no labels. The baseline's validation MSE is `1.0963`; training standardization does not force validation variance or mean to equal the training values.

Seven components leave 12.65% of baseline error. Eight leave 9.60%, so **eight is the smallest count satisfying this particular budget**. Two components are useful for a picture but miss the budget substantially. If the goal had instead been “retain at least 95% of training variance,” this split would select ten components.

<!-- Lab L3: component-budget explorer. Exact training/validation split, empirical error curve including k=0 and all 13; user-set budget and feature-level residual inspection. -->

In the budget explorer, change the error budget and follow the first qualifying count and curve crossing immediately. Then choose a validation observation and compare its original and reconstructed feature values. Does acceptable average error hide a particularly poor measurement? This connects the whole-dataset curve to an individual record.

**Why not just minimize validation reconstruction error?** With one fixed orthonormal basis, each added component removes a nonnegative squared residual for every observation. The error therefore cannot increase as `k` grows—even on validation data. Minimizing that error alone selects all dimensions. Compression needs a budget, penalty or constraint in addition to error.

We have used validation results to choose a representation. They are development evidence. A final claim about performance on future data needs a further untouched evaluation set or an appropriate resampling protocol. Section 10.3 explains how this differs from denoising against a clean target.

### Can the measurements actually be recovered in their original units?

For a retained `k`, first reconstruct in the standardized coordinate system, then undo standardization:

```python
# Continue the training/validation program above.
k = 8
directions = pca.components_[:k]
scores = (B - pca.mean_) @ directions.T
reconstruction = scores @ directions + pca.mean_
original_units = scaler.inverse_transform(reconstruction)

print(original_units.shape)
```

```text
(45, 13)
```

The thirteen output columns are approximate recovered measurements, not thirteen independently retained numbers. Keep the training scales, means, component directions and feature order with the scores; they are the decoder.

## 7. Read components, scores and plots without mixing them up

### Coefficients describe a direction; scores describe observations

For the standardized Wine collection in section 5, PC1 gives approximately these weights to selected features:

| Feature | PC1 direction coefficient |
| --- | ---: |
| Total phenols | 0.3947 |
| Flavanoids | 0.4229 |
| Nonflavanoid phenols | −0.2985 |
| Proline | 0.2868 |

The complete score uses all thirteen coefficients. Holding the other standardized coordinates fixed, increasing flavanoids by one fitted standard deviation increases the score by about `0.4229`. This tells you how the score is calculated. It does not establish that changing a compound causes a change in cultivar or quality.

Call a component something descriptive only after inspecting its coefficients, the observations and the scientific context. “A contrast involving several phenolic measurements” is supported more directly than a claim that PC1 is an intrinsic quality axis. The sign convention can be reversed with no change in reconstructions, as section 2 showed.

The word **loading** has more than one convention. Here a loading means a unit direction coefficient, as in `components_`. Some sources use feature–score correlations or direction coefficients multiplied by the square root of an eigenvalue. Label the quantity being shown rather than assuming the numbers should match across software.

For a centered feature with sample standard deviation `sⱼ` and a nonzero-variance score, the feature–score correlation is

\[
\operatorname{corr}(X_j,Z_\ell)
=\frac{v_{j\ell}\sqrt{\lambda_\ell}}{s_j}.
\]

You can check the distinction in the four-point example: the first coefficient is `1/√2 ≈ 0.7071`, but its correlation with the first score is `√0.9 ≈ 0.9487`. A coefficient and a correlation answer different questions.

### A biplot overlays two kinds of objects

A **score plot** puts observations at their component coordinates. A **biplot** also draws arrows for features. To read one, ask what scales the arrows and observations use.

Our optional biplot uses observation scores `zᵢ` and feature arrows `aⱼ = (vⱼ₁, vⱼ₂)`. Their dot product gives the rank-two approximation to the centered feature value:

\[
\widehat{x}_{ij}-\mu_j=z_i\cdot a_j.
\]

For standardized PCA, that value is in standardized units. If arrows are enlarged for visibility, their display multiplier must be stated and removed before this calculation. With this particular scaling, arrow angles alone are not a general formula for original feature correlations. Other biplot scalings support different interpretations. [Jolliffe and Cadima, biplots and scaling conventions](https://pmc.ncbi.nlm.nih.gov/articles/PMC4792409/).

<!-- Figure F5: optional contribution/biplot reading panel, single selected observation and feature arrow, exact dot product and inverse scaling. Use the four-point calculation before the 13-feature view. -->

### Close in the picture can mean far in the omitted coordinates

If two observations overlap in PC1/PC2, inspect their remaining scores or original features before concluding they are duplicates. The first picture deliberately discards information. A point with an ordinary-looking score can also have a large residual away from the retained subspace. Section 10.5 uses that residual as a diagnostic.

## 8. Connect PCA to clustering and prediction

The previous module topic, [K-Means & Hierarchical Clustering](/learn/path/full-curriculum/k-means-hierarchical-clustering?module=classical-ml), made the representation part of the clustering question. PCA lets us make that connection exact.

### Rotating all coordinates preserves Euclidean distances

An orthonormal coordinate change preserves the length of every difference vector. For a complete matrix of orthonormal directions `V`:

\[
\|(x-y)V\|^2=\|x-y\|^2.
\]

Centering cancels from `x − y`. Keeping **all** coordinates therefore changes neither Euclidean pair distances nor the K-Means squared-distance objective for a corresponding partition. A numerical algorithm can still make different choices at ties; the mathematical objective has not changed.

Keeping only `k` directions gives

\[
\|x-y\|^2
=\|(x-y)V_k\|^2+\|(x-y)V_{\mathrm{discarded}}\|^2.
\]

This is the same right-triangle accounting used for reconstruction in section 3, now applied to a difference between two observations. Distances can only shrink under orthogonal projection; how much they shrink depends on the pair.

For A and B, the original squared distance is `2`. Their one-component scores coincide, so the retained squared distance is `0`; all `2` lies in the discarded direction. A 90% variance summary has erased 100% of this pair's distance. That is why a global percentage cannot guarantee each neighborhood is preserved.

### A 99% variance component can lose the entire label

Construct four observations `(−10,−1)`, `(−10,1)`, `(10,−1)`, `(10,1)`. Define a class by the sign of the second coordinate. PC1 is the horizontal axis and retains `100/101 ≈ 99.01%` of the variance. After keeping it, each horizontal location contains both classes at the same score. No deterministic classifier using that score alone can distinguish the two observations at either location.

Using the second coordinate alone separates the classes perfectly on this constructed dataset. We are comparing two objectives: explain feature variation and retain information about this label.

<!-- Lab L4: variance-versus-label investigation, editable spread and label rule, recorded prediction of distinguishability, full/PC1/PC2 views. -->

In the investigation, record whether the retained coordinate will still distinguish the labels. Change the horizontal spread and keep the label rule fixed. Then change which coordinate defines the labels. The latter is a useful null contrast: it changes the usefulness of a component without changing the PCA fit at all.

For a prediction task, compare a no-PCA pipeline with PCA pipelines using validation on the actual prediction metric. Section 10.4 gives a complete small example. If the goal is clustering, examine stability and relevant external evidence, not just the attractiveness of a PCA scatterplot.

Reducing `d` to `k` can reduce work in computing each pairwise distance. It does not reduce the number of pairs among `n` observations. An explicitly stored dense distance matrix is still `n × n`.

## 9. Practice: calculate, diagnose and transfer

Try each question before opening its hint or solution. The first four are the core checkpoint. Questions 5–8 connect to the later branches.

### 1. Change the data, keep the method

Use `(1,0)`, `(3,2)`, `(5,4)`. Calculate the mean, a first principal direction, all three scores, the sample variance of those scores and the one-component reconstruction SSE. Then add `(8,4)` as a **new observation**, using the original fitted transform. What is its reconstruction and squared error?

<details><summary>Hint</summary>

The original three points are exactly on a line of slope one. The new point is not on that line. Keep the original mean when transforming it.

</details>

<details><summary>Solution</summary>

The mean is `(3,2)`. Choose `v₁=(1,1)/√2`. Centered rows are `(-2,-2)`, `(0,0)`, `(2,2)` and scores are `−2√2, 0, 2√2`. Sample variance is `(8+0+8)/2=8`. Their one-component reconstruction SSE is zero.

The new centered point is `(5,2)`. Its score is `7/√2`; reconstructing gives `(3,2)+(3.5,3.5)=(6.5,5.5)`. Residual `(1.5,-1.5)` has squared length `4.5`. Exact training reconstruction did not imply exact reconstruction of every possible new point.

</details>

### 2. Change the direction instead of memorizing a percentage

In the four-point example from section 2, keep the **second** direction `(1,−1)/√2` instead of the first. Find A's score and reconstruction. What are the total SSE, per-entry MSE and retained variance fraction?

<details><summary>Hint</summary>

The two components divide total centered squared length `20` into `18` and `2`. Swapping which is retained swaps the loss.

</details>

<details><summary>Solution</summary>

A's score is `−1/√2`. Its reconstruction is `(3,2)+(-0.5,0.5)=(2.5,2.5)`. The total SSE is `18`; per-entry MSE is `18/8=2.25`; retained variance is `10%`. A's individual squared error is `4.5`. This checks whether you can trace the inverse mapping, not just recite “keep the largest component.”

</details>

### 3. Repair a misleading scaling conclusion

A report says: “The raw Wine first component retains 99.8%; the standardized first component retains only 36.2%. Standardization destroyed 63.6% of the information.” Explain the error and write a better one-sentence conclusion.

<details><summary>Hint</summary>

Ask what squared error each percentage uses as its denominator.

</details>

<details><summary>Solution</summary>

The representations use different feature weightings, so their percentages do not measure the same error objective. Neither percentage measures all useful information. One acceptable conclusion is: “Raw PCA is dominated by large-magnitude measurements, whereas standardized PCA spreads retained relative variation across more components; choose the weighting from the intended task.” A strong answer also identifies which original feature dominates rather than merely saying “scaling matters.”

</details>

### 4. Run a new compression budget

Run the complete program in section 6. Change the allowed validation error fraction from `0.10` to **`0.06`**. Predict the smallest acceptable count before reading the result. Report the count, the two errors bracketing the budget, the coordinate system and whether you have measured final test performance.

<details><summary>Hint</summary>

You need both the last count that fails and the first count that passes. Reuse the training transform; do not fit a fresh validation PCA.

</details>

<details><summary>Solution</summary>

Nine components leave about `0.072354` of the training-mean baseline error; ten leave about `0.052066`. The smallest acceptable count is **10**. Errors use training-standardized measurements on the 45 validation rows. These are model-selection results, not an untouched final test result. A different split can give a different count.

</details>

### 5. Diagnose a pipeline leak

A colleague computes `StandardScaler().fit_transform(X)` on the entire dataset, applies PCA once, and then uses cross-validation to select a classifier. Explain what must move inside the folds. Separately, why is the descriptive full-collection analysis in section 5 not claiming that kind of generalization evidence?

<details><summary>Solution</summary>

Both the fitted scaler and fitted PCA must be learned using each fold's training portion, along with the classifier. Pass the complete unfitted pipeline into cross-validation. Section 5 describes the observations supplied to the transform; it does not estimate predictive performance on unseen observations. The use of the data and the claim made about the output determine the evaluation obligation.

</details>

### 6. Audit a storage promise

You have `n=100` rows and `d=20` features. You keep `k=10` PCA scores per row. Count stored scalar values if you retain the score matrix, component directions and mean, with no feature scaling and the same scalar precision throughout. Does this halve storage? At what number of rows does this choice first save storage?

<details><summary>Hint</summary>

The decoder also takes space. Compare `nk + dk + d` with `nd`.

</details>

<details><summary>Solution</summary>

Original: `100×20=2000` scalars. Compressed representation plus decoder: `100×10 + 20×10 + 20=1220`, or 61% of the original—39% saved, not 50%. Savings require `10n+220 < 20n`, hence `n>22`; the first integer is 23. For 22 rows the counts tie. Metadata, storage format and quantization are outside this scalar-count calculation.

</details>

### 7. Separate uncorrelated from independent

Give `U` the values `−1,0,1` with equal probability and define `W=U²`. Show that `U` and `W` have zero covariance but are not independent. Why does this matter when describing PCA scores?

<details><summary>Solution</summary>

`E[U]=0`, `E[UW]=E[U³]=0`, so their covariance is zero. But `W=0` tells us `U=0`, while `W=1` tells us `|U|=1`; they are dependent. PCA diagonalizes the fitted sample covariance of its scores. It does not generally produce statistically independent variables. The later ICA topic studies a different objective.

</details>

### 8. Why does a noise-only scree plot have a leading component?

Run the short Gaussian example in section 10.3. Before executing it, decide whether the first two sample components must retain exactly 10% of variation. Explain the result without inventing a hidden two-dimensional cause. Change the seed and repeat; state what you would need before giving an empirical component a scientific name.

<details><summary>Solution and assessment</summary>

The population covariance is the 20-dimensional identity, so every fixed two-dimensional orthonormal projection captures 10% of population variance. PCA chooses its directions after inspecting a finite sample. The chosen directions capitalize on that sample's uneven spread. For seed 23 the first two retain about 24.59% of the sample variation. Another seed changes the number.

A good interpretation distinguishes the fitted sample result from population structure, and proposes evidence tied to the intended claim: repeated data, stability of the retained subspace, a suitable null/noise model, known measurement factors or held-out task performance. Naming the tallest component is not that evidence.

</details>

### Independent mini-project

Using the supplied Wine data, write a short analysis that answers **one** purpose: exploratory visualization, measurement compression, or prediction. State the unit of observation, which rows fit the transform, the scale choice, the baseline and the success criterion before reporting results. Include one changed setting and an observation-level diagnosis, not only a global score.

For compression, a complete submission contains the reproducible split, selected count, error against the mean baseline, a per-feature or per-row residual inspection, retained decoder information and a statement about what the validation result supports. The 6% variation above provides an exact numerical checkpoint; explaining why it leads to ten components is part of the task.

For prediction, use the pipeline in section 10.4 as a starting point, then change one justified modeling choice. Report the fold results for both PCA and the no-PCA baseline, not just whichever mean is higher. The point is to make and evaluate a decision, not to force PCA to win.

## 10. Deeper branches

### 10.1 Why eigenvectors and SVD compute the same PCA

Read this branch when you want to connect the geometric calculation to the matrix algorithm. [Matrix Decompositions](/learn/path/full-curriculum/matrix-decompositions-svd-qr-cholesky-lu?module=math-foundations) and [Eigenvalues & Eigenvectors](/learn/path/full-curriculum/eigenvalues-eigenvectors?module=math-foundations) provide extended background.

Write the centered data matrix as `A`. Its sample covariance matrix is

\[
C=\frac{A^\top A}{n-1}.
\]

The diagonal entries are feature variances. An off-diagonal entry measures how two features vary together. The matrix has shape `(d,d)`, is symmetric, and is positive semidefinite: for any direction `v`, `vᵀCv` is a squared length divided by `n−1`, so it cannot be negative.

For the four-point example:

\[
C=\frac13\begin{bmatrix}10&8\\8&10\end{bmatrix}.
\]

Multiplying by `(1,1)` gives `6(1,1)`; multiplying by `(1,−1)` gives `(2/3)(1,−1)`. These directions are **eigenvectors**: the covariance matrix stretches each without changing its direction. The stretch factors `6` and `2/3` are its eigenvalues.

Why should this solve the optimization? The score vector for a unit direction `v` is `Av`. Its sample variance is

\[
\frac{(Av)^\top(Av)}{n-1}=v^\top Cv.
\]

A symmetric covariance matrix admits an orthonormal eigenbasis. Express a unit direction as `v=Σⱼaⱼvⱼ`, where `Σⱼaⱼ²=1`. Its score variance is

\[
v^\top Cv=\sum_j a_j^2\lambda_j\leq\lambda_1.
\]

This is a weighted average of the eigenvalues, with nonnegative weights adding to one. It cannot exceed the largest eigenvalue. Choosing its eigenvector achieves the maximum. Restricting to directions perpendicular to that vector gives the next largest eigenvalue, and so on. Eigenvalues are ordered **nonincreasingly**, allowing ties.

You can also use a Lagrange multiplier for the constraint `vᵀv=1`. Differentiating `vᵀCv−λ(vᵀv−1)` gives `2Cv−2λv=0`. The stationary directions are eigenvectors; the weighted-average argument identifies the maximum rather than merely finding a stationary point.

Now take the reduced SVD:

\[
A=USV^\top,
\qquad
C=V\frac{S^2}{n-1}V^\top.
\]

Here `r=min(n,d)`, `U` has shape `(n,r)`, `S` is the diagonal `(r,r)` matrix of singular values, and `Vᵀ` has shape `(r,d)`. Zero singular values are allowed. The nonzero covariance eigenvalues are `sⱼ²/(n−1)`; any omitted covariance eigenvalues are zero. The rows of `Vᵀ` are the principal directions returned by the program in section 4.

The first `k` score columns can be calculated two ways:

\[
Z=AV_k=U_kS_k.
\]

This is the same score matrix: dot products with directions on the left, SVD factors on the right. Its sample covariance is diagonal, with the retained eigenvalues on the diagonal. The scores are uncorrelated **on the fitting sample**. Uncorrelated does not mean independent; practice 7 gives a counterexample.

For orthonormal retained directions, `VₖVₖᵀ` is the projection matrix in feature space. The reconstructed centered matrix is `A VₖVₖᵀ`. The singular values give its error directly:

\[
\|A-AV_kV_k^\top\|_F^2
=\sum_{j>k}s_j^2
=(n-1)\sum_{j>k}\lambda_j.
\]

The Frobenius norm squared, `‖·‖²_F`, adds the squares of all matrix entries. This identity explains the 90% variance/10% loss result without a separate error model. The truncated SVD is optimal among rank-at-most-`k` matrix approximations for this squared-error objective—the **Eckart–Young** result. It does not assert optimality for label preservation or nonlinear compression.

If you remember the covariance example from Eigenvalues & Eigenvectors, these are the same centered observations and the same eigenvalues. Here we have connected them to a fitted mean, reconstruction and a practical compression decision.

For an explicit check, rerun the first NumPy program in section 4 to restore its four-point variables, then run:

```python
# Continue the four-point NumPy example in section 4.
covariance = centered.T @ centered / (len(X) - 1)
eigenvalues, eigenvectors = np.linalg.eigh(covariance)
leading = eigenvectors[:, -1:]
eigen_reconstruction = (centered @ leading) @ leading.T + mean

print(np.round(eigenvalues[::-1], 4))
print(np.allclose(eigen_reconstruction, reconstructed))
print(round(float(np.sum(singular_values[1:]**2)), 4))
```

```text
[6.     0.6667]
True
2.0
```

The comparison uses reconstructions, so a harmless sign difference cannot make it fail. The geometry and covariance viewpoints have a long history: Pearson's closest-fit formulation dates to 1901 and Hotelling's statistical treatment to 1933. Their connection is useful because the two viewpoints lead to the same computation, not because the names must be memorized. [PCA review and historical references](https://pmc.ncbi.nlm.nih.gov/articles/PMC4792409/).

### 10.2 What happens at zero variance, ties and whitening?

**Rank.** Centering makes the rows sum to zero, so the centered matrix has rank at most `min(n−1,d)`. With four observations in twenty features, at most three principal components have nonzero sample variance. That is a finite-data constraint, not a discovery that the population has only three degrees of freedom. Retaining all nonzero training directions exactly reconstructs training data; a future point can still lie outside their span.

**Tied directions.** If two eigenvalues are equal, any orthonormal basis within their shared eigenspace is valid. If `k` cuts through that tie, the selected `k`-dimensional subspace can also be nonunique. If eigenvalues are merely close, small data changes can rotate individual directions substantially. Compare reconstructions or projection matrices, and compare whole tied subspaces where appropriate, rather than treating every changed vector as an error.

**Constant data.** If every observation is identical, total centered variance is zero. The mean reconstructs everything; a “fraction explained” divides zero by zero and has no informative value. If only one column is constant, it has no centered variation. `StandardScaler` leaves such a column with scale factor one rather than dividing by zero. Missing values require an explicit strategy before ordinary PCA; fitting an imputer follows the same information boundary as the other fitted operations in section 6.

**Whitening changes the metric again.** PCA alone rotates and optionally truncates. Whitening additionally divides each nonzero-variance score by the square root of its fitted variance:

\[
w_{ij}=z_{ij}/\sqrt{\lambda_j}.
\]

The retained score columns then have unit sample variance on the fitting data. Directions with small original variance get magnified relative to large-variance directions. Euclidean distances in whitened coordinates therefore differ from distances in ordinary PCA coordinates. This can suit a downstream model, but should be evaluated for that model's purpose.

For the four-point example, fit `PCA(n_components=2, whiten=True, svd_solver="full")`. In the checked scikit-learn version, the transformed columns have variance `1` using `ddof=1`, and `0.75` using `ddof=0`. These denominators differ: dividing by four rather than three changes the variance by `3/4`. Whitening is not a way to make a zero-variance component informative.

`StandardScaler` uses the `ddof=0` convention for its fitted scales, while PCA reports sample variances with `n−1`. For fully observed nonconstant columns standardized on `n` fitting observations, PCA's sample feature variances are consequently `n/(n−1)`. The common factor does not change directions or variance ratios, but it matters when checking absolute values. [StandardScaler conventions](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#notes).

### 10.3 Sampling variation is not hidden structure

Suppose twenty independent Gaussian measurements each have population variance one. The population covariance is the identity. It has no preferred direction. A finite sample will not have exactly that covariance; the entries fluctuate.

```python
import numpy as np
from sklearn.decomposition import PCA

rng = np.random.default_rng(23)
noise = rng.normal(size=(40, 20))
pca = PCA(svd_solver="full").fit(noise)
print(round(pca.explained_variance_ratio_[:2].sum(), 4))
```

```text
0.2459
```

These leading components explain 24.59% of this sample's variation, despite the population having no special two-dimensional subspace. PCA selected the strongest directions in the very sample being summarized. The [Random Matrix Theory](/learn/path/full-curriculum/random-matrix-theory?module=math-foundations) lesson develops how sample spectra behave when the number of features is large relative to the number of observations.

<!-- Figure F6: finite Gaussian sample spectrum and flat population 5%-per-direction reference; clearly simulated, no significance thresholds. -->

A useful stability check refits on repeated samples or resampled rows and compares retained subspaces. It should respect the data's grouping or dependence. Stable directions can still describe a stable artifact, so stability and scientific interpretation answer different questions.

**Denoising changes the target of error.** Suppose a true two-sensor signal is `(s,s)`, and a measurement adds error `(e,−e)`. Projection onto the diagonal removes that error exactly: the signal and error occupy perpendicular directions. If the signal variation dominates, PCA can identify that diagonal from suitable data.

If the error instead is `(e,e)`, it lies along the same direction as the signal. The diagonal projection preserves it. PCA cannot separate two contributions merely because we call one “noise.”

When evaluating denoising, compare a reconstruction to a justified clean target, independent measurement or explicit noise model. Reconstructing the noisy input in all dimensions gives zero input reconstruction error while preserving every noise realization. This is why the monotone input-error curve in section 6 is not, by itself, a denoising validation curve.

### 10.4 Evaluate PCA inside a prediction pipeline

Here is a complete development comparison using the same Wine data. It asks whether reducing the features helps a logistic-regression classifier predict the cultivar. Each candidate is evaluated on the same five stratified folds. The scaler and PCA are refitted inside each training fold.

```python
import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

wine = load_wine()
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for k in [None, 2, 8]:
    reduction = "passthrough" if k is None else PCA(
        n_components=k, svd_solver="full"
    )
    pipeline = make_pipeline(
        StandardScaler(), reduction, LogisticRegression(max_iter=2000)
    )
    accuracy = cross_val_score(
        pipeline, wine.data, wine.target,
        cv=folds, scoring="accuracy", n_jobs=1
    )
    print(k, np.round(accuracy, 4), round(accuracy.mean(), 4))
```

```text
None [0.9722 0.9722 0.9722 1.     1.    ] 0.9833
2 [0.9444 0.9722 0.9167 0.9714 0.9714] 0.9552
8 [0.9722 0.9722 0.9722 0.9714 1.    ] 0.9776
```

`None` labels the no-PCA baseline. On these folds, neither compressed candidate improves its mean accuracy. Eight components retain more predictive utility than two, but the original standardized features work well for this classifier and dataset size.

The fold values show variation hidden by a single mean. They are not independent replications or a confidence interval. Selecting a pipeline after inspecting them uses development evidence, as discussed in section 6. More components are not automatically better on another task, and the no-PCA comparison should remain part of the investigation. [Cross-Validation & Hyperparameter Tuning](/learn/path/full-curriculum/cross-validation-hyperparameter-tuning?module=classical-ml) develops the evaluation protocol further.

### 10.5 Two useful applications beyond a scatterplot

**Compression requires a decoder.** For `n` observations with `d` features, the original numeric matrix stores `nd` scalars. Keeping `k` scores per observation, `k` feature directions and a mean stores `nk + dk + d` scalars before any scaling metadata. For `n=1000`, `d=100`, `k=10`, that is `11,100` rather than `100,000` scalars at equal precision. Whether a particular file becomes that much smaller also depends on its format and encoding. Practice 6 shows why small datasets can have much less impressive savings.

For images, each column can be a pixel and each row an image. Reshape a principal direction into the original image layout: it becomes a **pattern of positive and negative pixel weights**. Reconstruction adds weighted patterns to the mean image. A direction can contain negative weights even when every observed pixel is nonnegative; the direction describes changes around the mean. Keeping two such patterns is a two-coordinate image representation, not an image with two pixels.

**A discarded direction can be a useful alarm.** Consider a constructed two-sensor system whose normal readings vary close to `(s,s)`. Fit its usual diagonal direction. A later centered observation `(3,3)` has a large score along the usual mode but zero residual. The observation `(3,−3)` has zero diagonal score but residual squared length `18`. A score-only display calls the second point ordinary-looking; its residual shows a strong disagreement between the sensors.

<!-- Figure F7: normal diagonal band, two labeled new observations, score position versus residual; calculated schematic, no empirical alarm threshold. -->

This suggests inspecting two diagnostics: how far an observation travels within the usual subspace, and how far it falls outside it. Turning those diagnostics into an operational alarm requires data about normal variation, relevant faults and false-alarm costs. The geometry alone does not supply an alarm threshold.

A related question appears with neural population recordings: a row might represent one time bin and columns different neurons. Scores summarize population variation; reconstructing reveals which activity patterns are discarded. Shuffling row order does not change the fitted covariance, so ordinary PCA by itself does not learn temporal dynamics. Connecting score points in time order adds a trajectory display, not a dynamical model. The planned [Dimensionality Reduction & Manifold Analysis for Neural Data](/learn/path/full-curriculum/dimensionality-reduction-manifold-analysis-for-neural-data?module=computational-neuroscience) develops the measurement, time-axis and validation issues needed for that use.

### 10.6 Choose a computation that fits the data

The mathematical objective does not prescribe forming a huge covariance matrix. For dense real data, an economy-size direct SVD costs on the order of `nd·min(n,d)` arithmetic operations; forming and diagonalizing `AᵀA` instead costs approximately `nd² + d³`. These are scaling models, not measured running times.

On the nonzero singular spectrum, forming `AᵀA` squares the condition number. That can obscure small components in floating-point arithmetic. Direct SVD avoids that particular loss of conditioning, while covariance eigendecomposition can still be an efficient choice for many rows and relatively few, well-conditioned features.

| Approach | What it computes or stores | When to consider it |
| --- | --- | --- |
| Direct full SVD | Complete reduced factorization, then truncation | Small/moderate dense problems; reliable reference for retained components |
| Covariance eigendecomposition | A `d×d` covariance matrix and its eigenvectors | Many more rows than features, with manageable `d²` storage |
| Randomized SVD | A sketch targeting roughly `k` directions | `k` much smaller than the smaller matrix dimension |
| Incremental PCA | Updates a retained approximation from batches | Rows do not all fit in memory |
| Truncated SVD without centering | Low-rank factors of the supplied matrix | Sparse count/TF–IDF matrices when uncentered approximation is the intended objective |

Randomized SVD uses a random test matrix `Ω` with about `k+p` columns, where `p` is oversampling. Compute `Y=AΩ`, find an orthonormal basis `Q` for those columns, and decompose the smaller matrix `QᵀA`. Power iterations can improve separation of leading directions at the cost of extra passes. Approximation quality depends on the spectrum and settings. For a fixed number of passes and small `k`, the leading dense multiplication work is roughly `nd(k+p)`; reduced factorizations add work. [Halko, Martinsson and Tropp, algorithm overview](https://arxiv.org/abs/0909.4061).

The sketch workspace is not the entire memory cost. A million-by-five-thousand float64 input alone contains `5×10⁹` numbers, or about **40 GB** in decimal units. Selecting 50 components does not make that input disappear. Batch processing, memory mapping or another data-access plan may still be necessary.

Incremental PCA keeps a compressed approximation between batches. Its answer can depend on batch size and order; it is not merely a different rounding of an exact full solution. If preprocessing scales are also learned from a stream, plan how they are fixed or updated consistently. Do not standardize each batch independently and assume the resulting coordinates have a common meaning. [IncrementalPCA reference](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html).

For the small dense examples in this lesson, `svd_solver="full"` makes the choice explicit. Current scikit-learn also supports `"covariance_eigh"`, `"randomized"` and `"arpack"`; the default `"auto"` selects according to shape and requested dimension. It is not a universal randomized-SVD default. For randomized comparisons, set `random_state`; record the solver and package version when reproducibility matters. Consult the API for its current sparse-input support and component-count constraints rather than inferring them from an older tutorial.

### 10.7 When the question calls for another method

PCA's best reconstruction is linear: the decoded points lie on a flat affine subspace. If observations follow a curved shape, a flat projection can overlap different parts of that shape. A two-dimensional surface rolled through three-dimensional space is a standard example: its intrinsic dimension is two, but flattening it is not the same operation as projecting onto a plane. A neighborhood method may help, provided its assumptions, sampling and parameters support the geometry. It is not guaranteed to unroll every such dataset correctly.

Use this comparison to choose what to investigate next:

| Need | Candidate and essential distinction |
| --- | --- |
| Keep a subset of actual measurements | Feature selection retains original columns; PCA usually mixes many columns in every score. |
| Use a nonlinear similarity with a PCA-like spectral approach | Kernel PCA centers a kernel matrix and finds components in its feature space. It supports a forward transform for new points. An approximate inverse is a separate fitted problem, not a prerequisite for the forward transform. Dense kernel storage grows as `n²`. [KernelPCA API](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html). |
| Preserve distances approximately without learning covariance directions | Random projections choose a suitable random map. Johnson–Lindenstrauss guarantees concern finite sets, distortion and probability, with target dimension scaling like `log(n)/ε²`; they do not order coordinates by explained variance. [Random projection guide](https://scikit-learn.org/stable/modules/random_projection.html). |
| Explore nonlinear neighborhoods | [t-SNE, UMAP & Manifold Learning](/learn/path/full-curriculum/t-sne-umap-manifold-learning?module=classical-ml) develops the objectives and interpretation limits. t-SNE compares neighborhood probability distributions, rather than minimizing PCA reconstruction loss. UMAP can transform new observations; it is not limited to static pictures. [UMAP's transform tutorial](https://umap-learn.readthedocs.io/en/latest/transform.html). |
| Seek components with stronger statistical separation | [ICA](/learn/path/full-curriculum/independent-component-analysis-ica?module=classical-ml) targets independence under additional assumptions. Decorrelation alone is weaker. |
| Describe nonnegative data through additive parts | [NMF](/learn/path/full-curriculum/non-negative-matrix-factorization-nmf?module=classical-ml) imposes nonnegativity rather than PCA's orthogonality. |
| Learn an encoder and decoder with nonlinear functions | An autoencoder trains a reconstruction model. Labels and a GPU are not inherently required; architecture, capacity and evaluation still matter. Its code coordinates need not be ordered principal components. |

There are also variants that change PCA itself: sparse directions can simplify interpretation; robust formulations can change how outliers influence the fit; functional PCA treats whole curves as observations; probabilistic PCA introduces a latent-variable and noise model. Each adds a modeling choice. “Robust PCA” in particular can refer to different formulations, including low-rank-plus-sparse decomposition. Use the reference's actual objective rather than assuming a variant is ordinary PCA with better guarantees.

**PCA versus regression.** Ordinary least squares predicts a designated response by minimizing vertical response errors. PCA treats the selected feature coordinates symmetrically under the chosen metric and minimizes perpendicular reconstruction errors. A best prediction line and a first principal axis therefore need not coincide. In the four-point example, regressing the second reading on the first gives slope `(8/3)/(10/3)=0.8`; PCA's first axis has slope `1`. Their objectives explain the difference.

## 11. What you should now be able to do

PCA learns a coordinate system from centered variation. Scores tell you where observations lie in that system. Keeping fewer scores loses the variation in discarded directions, and reconstruction makes that loss inspectable. Scaling sets the geometry; the purpose sets the acceptable loss.

Before moving on, check that you can:

- Explain a direction, a score and a reconstruction using the same point.
- Predict the effect of changing a unit, reversing a direction and discarding a coordinate.
- Reproduce the four-point result and the changed Wine error budget.
- State which data fitted the transform and what evidence your evaluation supplies.
- Give a case where high explained variance does not preserve a task-relevant distinction.

The next module topic is **[Clustering Evaluation & Validation (Silhouette, ARI, NMI)](/learn/path/full-curriculum/clustering-evaluation-validation-silhouette-ari-nmi?module=classical-ml)**. After constructing clusters and changing representations, the next question is how to evaluate the grouping. Follow that module sequence; the optional branches above are connections, not substitutions for the next lesson.

## References & another way to learn it

### Alternate explanations and practice

- **Jolliffe & Cadima, “Principal component analysis: a review and recent developments” (2016)** — [open survey](https://pmc.ncbi.nlm.nih.gov/articles/PMC4792409/). A substantial reference after the core lesson; section 2 connects definitions and graphical interpretation, while section 3 introduces extensions. Useful when different loading or biplot conventions seem inconsistent. It is a conceptual reference, not current Python API documentation.
- **James, Witten, Hastie, Tibshirani & Taylor, *An Introduction to Statistical Learning with Applications in Python*** — [official Chapter 12 resources](https://www.statlearning.com/resources-python) and [PCA notebook](https://islp.readthedocs.io/en/stable/labs/Ch12-unsup-lab.html#principal-components-analysis). Another hands-on route through scaled data, scores, biplots and variance plots. The notebook uses additional packages and datasets; consult its version instructions rather than replacing your environment blindly. The authors also provide the book through [their official site](https://www.statlearning.com/home).
- **Philippe Rigollet, MIT 18.650, Lecture 19: Principal Component Analysis** — [lecture video](https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/resources/lecture-19-video/) and [companion PCA slides](https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/d85e1a9d113142ade8ce5e4f5ef0b4e8_MIT18_650F16_PCA.pdf). A mathematical alternative after section 10.1, connecting projected variance and eigenvectors. The slides use a `1/n` empirical covariance convention; this lesson uses `1/(n−1)`. Directions and ratios agree, while absolute variances differ by the stated factor. The companion slides were reviewed for this recommendation; the full video was not watched.
- **Scikit-learn, “Importance of Feature Scaling”** — [worked Wine example](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html). An additional comparison of how scale affects PCA and downstream estimators. Read after sections 5–6 and distinguish its chosen protocol from the split used here.

### Data, computation and precise references

- **Aeberhard & Forina, Wine dataset** — [UCI record and license](https://archive.ics.uci.edu/dataset/109/wine), DOI [10.24432/C5PC7J](https://doi.org/10.24432/C5PC7J). Attribution and the offline mirror's transformation history are in the supplied [data notes](data-provenance.md). The [scikit-learn loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html) supplies the version used by the programs.
- **Numerical operations:** [NumPy SVD](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html), [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html), [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html), and [decomposition guide](https://scikit-learn.org/stable/modules/decomposition.html). Use these to check factor orientation, fit/transform semantics and solver behavior. The draft's tested versions are recorded beside the code; moving documentation can describe a newer release.
- **Evaluation:** [scikit-learn's common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html). A practical reference for fitted preprocessing and fold boundaries.
- **Halko, Martinsson & Tropp (2011), “Finding Structure with Randomness”** — [paper and free manuscript](https://arxiv.org/abs/0909.4061). Advanced reading for randomized low-rank factorization and the assumptions behind its approximation bounds. Read the algorithm overview before the proofs; it does not supply a machine-independent timing benchmark.

The small point clouds, projection calculations, sensor scenarios and practice problems in this lesson are constructed teaching examples. Wine results are calculations on the identified real dataset. The Gaussian spectrum is a seeded simulation. None is an invented hardware benchmark or a claim about every future dataset.
