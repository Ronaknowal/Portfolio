# Gaussian Mixture Models (GMM) & EM Algorithm

Stable ID: `gaussian-mixture-models-gmm-em-algorithm`. Classical ML, position 16. Content-first manuscript; the visual placements below refer to [the complete specifications](visual-specifications.md). Research and source conservation are recorded in [the design](../../GMM-LESSON-DESIGN.md).

## 1. Explain an overlapping population

A collection of flower measurements contains several shapes. A measurement in the middle could plausibly belong to more than one group. A measurement far away might be closer to one group than the others, yet still look unusual for the collection as a whole. We need a model that can answer both questions: **which group accounts for this observation, and how much density does the whole model assign here?**

A Gaussian mixture model builds a density by adding several bell-shaped components. Each component has a center, a spread and a share of the total population in the model. The component responsible for a particular observation is hidden. Expectation–maximization, or **EM**, fits the model by alternating between estimating those hidden memberships and updating the components from the estimates.

We will first use four invented measurements so every update can be checked by hand. Then we will fit real iris flower measurements. The practical question is whether several components describe unseen measurements better than a single Gaussian. We will return to that question with separate training, validation and test results.

**First-pass route.** Read §§1–8, doing the responsibility, EM and covariance investigations where they appear. Run the small EM program in §4 and the offline Iris program in §7. Attempt practice 1–5 in §11. Return to §9 for the lower-bound proof and to §10 for the k-means limit, conditional prediction and Bayesian extension; practice 6–8 develops those branches. Allow about 55–70 minutes for the core reading and 60–100 minutes for calculation and code practice. The deeper branches are another sitting.

You need weighted averages, variance, conditional probability, logarithms and the idea of a covariance matrix. We refresh them locally. Review [Probability Distributions & Bayes' Theorem](/learn/path/full-curriculum/probability-distributions-bayes-theorem?module=math-foundations), [Maximum Likelihood & MAP Estimation](/learn/path/full-curriculum/maximum-likelihood-map-estimation?module=math-foundations), and [PCA & Dimensionality Reduction](/learn/path/full-curriculum/pca-dimensionality-reduction?module=classical-ml) if those operations feel unfamiliar. NumPy array shapes are explained beside the program.

The previous [Anomaly & Outlier Detection](/learn/path/full-curriculum/anomaly-outlier-detection-isolation-forest-one-class-svm-lof?module=classical-ml) lesson separated a score from an alert decision. Here we learn a new source of scores: a fitted probability density.

## 2. A hidden selector followed by a measurement

Imagine making one observation in two steps. First select component A with probability 0.5 or component B with probability 0.5. If A is selected, draw a measurement from a normal distribution with mean −2 and variance 1. If B is selected, use mean 2 and variance 1. We observe the measurement but lose the selector's result.

The selector is a **latent variable**: part of the model that is not observed. Write it as $z_i$ for observation $i$. The measured value is $x_i$. A Gaussian's mean μ locates its center; its variance $v=σ^2$ measures squared spread. A standard deviation σ has the same units as $x$, while $v$ has squared units.

> **Inline visual F1 — Select, draw, hide.** A 50:50 selector branches to the two labeled Gaussian densities, then to a visible measurement. The latent selector is outlined; the observation is filled. Caption: “The model selects one component for each observation. Adding the weighted curves describes what we see after the selector is hidden.”

For one measurement, a Gaussian density is

\[
\mathcal N(x;\mu,v)=\frac{1}{\sqrt{2\pi v}}
\exp\left[-\frac{(x-\mu)^2}{2v}\right].
\]

The squared distance in the exponent lowers density away from the center. Dividing by the spread prevents a very wide bell from assigning a high density everywhere: its total area must remain 1. Here π in $2\pi$ is the circle constant. The separate symbol $π_k$ below is a component's mixing weight.

With $K$ components, mixing weights satisfy $π_k\geq0$ and $∑_kπ_k=1$. The mixture density is

\[
p(x\mid\theta)=\sum_{k=1}^K\pi_k\mathcal N(x;\mu_k,v_k).
\]

The symbol θ collects all weights, means and variances. A weighted component curve has area $π_k$; the sum has area 1. At the same location, add the heights of those curves. Do not average the means and then draw one bell: that would usually be a different density.

**Probability, density and interpretation.** This is the home for the distinction throughout the lesson. A continuous density is height per measurement unit; probability is area over an interval. An exact point has probability zero in this model. Density can exceed 1 when a bell is narrow. Component memberships below are probabilities *under the fitted model*, with its parameters treated as fixed. They are neither verified species probabilities nor an uncertainty distribution over the fitted parameters. The iris data are a deliberately balanced species collection, so fitted weights describe this sampling arrangement, not species prevalence in nature.

### Reverse the selector with Bayes' rule

At a measured $x_i$, calculate how much each component contributes to the mixture density. Divide each contribution by their sum:

\[
r_{ik}=P(z_i=k\mid x_i,\theta)
=\frac{\pi_k\mathcal N(x_i;\mu_k,v_k)}
{\sum_{j=1}^K\pi_j\mathcal N(x_i;\mu_j,v_j)}.
\]

This probability is a **responsibility**. The numerator is a component's weighted density at the observation. The denominator sums over components while keeping the *same observation $x_i$*. Each row of the responsibility matrix sums to 1. Responsibilities are soft allocations: a row such as $(0.7, 0.3)$ contributes 0.7 of an observation to A's fitted statistics and 0.3 to B's.

For our two bells, take $x=2$. A contributes $0.5\mathcal N(2;-2,1)\approx0.000066915$. B contributes $0.5\mathcal N(2;2,1)\approx0.199471140$. Their sum is 0.199538055, giving B responsibility 0.999664650.

Compare three observations under this *same fixed model*:

| Observation $x$ | Mixture density $p(x)$ | B responsibility | Negative log-density $-\log p(x)$ |
| --- | ---: | ---: | ---: |
| 0 | 0.053990967 | 0.500000000 | 2.918939 |
| 2 | 0.199538055 | 0.999664650 | 1.611750 |
| 8 | $3.0379414\times10^{-9}$ | $1-1.27\times10^{-14}$ | 19.612086 |

At 8, B explains almost all of an extremely small total. Responsibility compares the components *with one another*; density compares locations under the whole model. This is why a very decisive component assignment can coexist with a large anomaly score.

> **Investigation I1 — Which component, how much density?** Record whether A, B or a tie will have greater responsibility at a measurement you choose. Edit the measurement and component weight, then calculate. Inspect the weighted heights, their sum and the normalized allocation together. Next find two measurements with very similar B responsibility but very different total density. Use the identical-component case to explain why changing the measurement then leaves responsibilities equal to the mixing weights.

**Quick try.** Keep means −2 and 2 and variances 1, but set A's weight to 0.2. At $x=0$, predict the two responsibilities and whether total density changes.

<details><summary>Check the reasoning</summary>

The two Gaussian heights are equal at 0. Weighting and normalizing gives responsibilities $(0.2, 0.8)$. The weighted sum is the common height times $0.2+0.8=1$, so total density remains 0.053990967. A change in relative responsibility need not change the density at that location.

</details>

## 3. Fit components when membership is missing

We now keep the data fixed and learn the parameters. This is a different task from §2, which kept the parameters fixed and scored a new location.

For independent observations, the likelihood multiplies their densities. The log-likelihood adds their log-densities:

\[
\ell(\theta)=\sum_{i=1}^{n}\log\left[\sum_{k=1}^{K}
\pi_k\mathcal N(x_i;\mu_k,v_k)\right].
\]

Maximizing it rewards a model for assigning density near the observed values. The sum inside each logarithm couples the component parameters. Differentiating produces responsibilities that themselves depend on the unknown parameters, so there is no general one-shot closed-form solution. EM turns this coupling into repeated weighted fits.

Suppose we knew the selector for every row. We could fit each Gaussian to its assigned measurements. EM replaces those unavailable hard memberships with their current conditional expectations—the responsibilities. It alternates:

1. **E-step:** hold every parameter fixed; compute every responsibility row using Bayes' rule.
2. **M-step:** hold that entire responsibility matrix fixed; fit weights, means and spreads using it.

Let $N_k=∑_i r_{ik}$, the effective number of observations allocated to component $k$. Then

\[
\pi_k^{new}=\frac{N_k}{n},\qquad
\mu_k^{new}=\frac{\sum_i r_{ik}x_i}{N_k},\qquad
v_k^{new}=\frac{\sum_i r_{ik}(x_i-\mu_k^{new})^2}{N_k}.
\]

The weight is a share of the total effective count. The mean is a weighted average. The variance is weighted squared spread **around the new mean**. The denominator is $N_k$, not $N_k-1$: these are likelihood updates, rather than an unbiased sample-variance correction. These expressions assume $N_k>0$ and, for an ordinary nonsingular Gaussian, positive fitted variance.

The name “expectation” does not mean replacing each observation by an expected observation. The measured $x_i$ stays where it is. What changes is our expected hidden membership. “Maximization” means maximizing the expected *complete-data* log-likelihood with responsibilities held fixed, not maximizing the observed-data likelihood globally in one step.

### One complete EM cycle

Use four constructed observations, in arbitrary measurement units:

| ID | A | B | C | D |
| --- | ---: | ---: | ---: | ---: |
| Measurement | −2 | −1 | 1 | 2 |

Initialize two equally weighted components at means −1 and 1, each with variance 1. Component names here are **Left** and **Right**, to distinguish them from observation IDs A–D.

For observation A at −2, the Left-to-Right weighted-density ratio is

\[
\frac{e^{-(-2+1)^2/2}}{e^{-(-2-1)^2/2}}=e^4.
\]

So its Left responsibility is $e^4/(1+e^4)=0.982013790$. Doing the same for all four rows gives:

| Observation | $x_i$ | Left responsibility | Right responsibility |
| --- | ---: | ---: | ---: |
| A | −2 | 0.982013790 | 0.017986210 |
| B | −1 | 0.880797078 | 0.119202922 |
| C | 1 | 0.119202922 | 0.880797078 |
| D | 2 | 0.017986210 | 0.982013790 |

Left's effective count is exactly 2 by symmetry. Its weighted sum is

\[
(-2)(0.982013790)+(-1)(0.880797078)
+(1)(0.119202922)+(2)(0.017986210)
\approx-2.689649316.
\]

Dividing by 2 gives its new mean, −1.344824658. Its weighted second moment is

\[
\frac{4(0.982013790)+1(0.880797078)+1(0.119202922)+4(0.017986210)}{2}=2.5.
\]

Variance can be computed as weighted second moment minus squared weighted mean. Thus $v_{Left}^{new}=2.5-(-1.344824658)^2=0.691446639$. Right's new mean is 1.344824658 and its variance is the same. Both weights remain 0.5.

> **Inline visual F2 — Fractional observations become a weighted fit.** Show the four measurement locations, then split each observation's unit mass between two aligned component lanes. Mark each lane's effective count, weighted center and spread before and after the M-step. Caption: “The observations stay fixed. Their fractional allocations move the fitted centers and spreads.”

The total log-likelihood rises from −7.158186977 to −6.461856301. A larger log-likelihood is better even when both values are negative. On the next E-step, Left's responsibility for A rises to 0.999582068, because the updated Left component is closer and narrower. We recompute responsibilities after changing parameters; the earlier table describes the input to the first M-step, not the final fitted model.

**Complete a changed update.** Replace D's measurement 2 by 3, keeping the original starting parameters. Calculate the new Right mean after one E+M cycle. Hint: the Right responsibility at 3 is $1/(1+e^{-6})$, and the effective Right count is no longer 2.

<details><summary>Explained answer</summary>

The Right responsibilities for −2, −1, 1, 3 are approximately 0.017986210, 0.119202922, 0.880797078 and 0.997527377. Their sum is 2.015513587. Their measurement-weighted sum divided by that count gives 1.844792261. The new Right weight is 0.503878397 and its variance is 1.582910448. Moving the last observation affects the E-step allocation as well as the average; simply adding 1/2 to the old Right mean would hold the wrong weights fixed.

</details>

## 4. A compact EM program and a meaningful stopping rule

Before running an optimizer, make its permissible models explicit. An unconstrained Gaussian mixture can drive its likelihood upward without limit by shrinking a component around one observation. Our small program therefore maximizes the observed log-likelihood **subject to each variance being at least 0.05**. The constrained one-dimensional M-step replaces an unrestricted variance $s_k$ by $\max(s_k,0.05)$, written `np.maximum` in Python. This is a fixed model constraint, with variance units squared; it is not an additive adjustment to every variance.

Why this update? For fixed responsibilities and the new mean, the variance-dependent expected log-likelihood is $-N_k(\log  v+s_k/v)/2$ plus a constant. It rises up to $v=s_k$ and falls after that. Over $v\geq0.05$, its maximum is $s_k$ when feasible, otherwise the boundary 0.05. This gives an exact constrained M-step.

### Compute in log space

For a distant observation, directly evaluating both component densities can underflow to floating-point zero. Dividing those zeros by a clipped denominator still gives zero responsibilities, whose sum is zero.

Instead calculate each log weighted density $a_k=\log π_k+\log \mathcal N(x;μ_k,v_k)$. With $m=\max_k a_k$,

\[
\log p(x)=m+\log\sum_k e^{a_k-m},\qquad
r_k=e^{a_k-\log p(x)}.
\]

The largest shifted exponent is zero, so at least one exponential is 1. For $a=(-1000,-1001)$, the log-density is −999.686738312 and the responsibilities are $(0.731058579, 0.268941421)$. Both original exponentials can be zero in float64; their relative sizes remain available through the shifted calculation.

For a local reproduction environment, install the checked package versions once:

```text
python -m pip install "numpy==2.3.5" "scikit-learn==1.9.1"
```

Save this complete program as `em_1d.py`. Run `python em_1d.py` in an environment with NumPy. The fixture and selected values were checked during authoring with Python 3.12.14 and NumPy 2.3.5; the displayed program's full native-output verification in the lesson reader remains a phase-two task.

```python
import numpy as np

x = np.array([-2., -1., 1., 2.])
weights = np.array([0.5, 0.5])
means = np.array([-1., 1.])
variances = np.array([1., 1.])
variance_floor = 0.05

def expectation(x, weights, means, variances):
    log_joint = np.log(weights) - 0.5 * (
        np.log(2 * np.pi * variances)
        + (x[:, None] - means) ** 2 / variances
    )
    row_max = log_joint.max(axis=1, keepdims=True)
    log_density = row_max[:, 0] + np.log(
        np.exp(log_joint - row_max).sum(axis=1)
    )
    responsibility = np.exp(log_joint - log_density[:, None])
    return responsibility, log_density.sum()

responsibility, previous = expectation(x, weights, means, variances)
print(f"initial log-likelihood {previous:.6f}")
for iteration in range(1, 51):
    count = responsibility.sum(axis=0)
    weights = count / len(x)
    means = (responsibility.T @ x) / count
    scatter = (responsibility * (x[:, None] - means) ** 2).sum(axis=0)
    variances = np.maximum(scatter / count, variance_floor)
    responsibility, current = expectation(x, weights, means, variances)
    print(iteration, f"{current:.6f}")
    gain = (current - previous) / len(x)
    if abs(gain) < 1e-8:
        break
    previous = current

print("means", np.round(means, 6))
print("variances", np.round(variances, 6))
print("row sums", responsibility.sum(axis=1))
```

`x[:, None]` has shape $(n,1)$, while the parameter arrays have shape $(K,)$. Subtraction broadcasts to $(n,K)$: every observation against every component. The responsibility matrix has that same shape. Summing down rows gives $K$ effective counts; multiplying its transpose by `x` gives $K$ weighted sums.

The checked arithmetic predicts these printed values:

```text
initial log-likelihood -7.158187
1 -6.461856
2 -5.724278
3 -5.675743
4 -5.675742
5 -5.675742
means [-1.499994  1.499994]
variances [0.250018 0.250018]
row sums [1. 1. 1. 1.]
```

The final means are close to the hard group averages −1.5 and 1.5. The slight difference comes from the tiny soft allocations across the gap. The fitted variances stay above the floor, so it never becomes active in this run.

> **Investigation I2 — Can your initialization separate the data?** Record whether one complete EM cycle will increase log-likelihood or leave it unchanged. Edit a measurement or place the two initial means yourself, then compute the E-step and M-step separately. Watch fractional allocations, component curves and the objective change together. Compare an asymmetric initialization with two identical initial components. Finally use repeated observations to make the variance floor active, and explain which update hits the boundary.

### What improvement does—and does not—guarantee

Keep the guarantees here rather than adding a qualification to every trace. In exact arithmetic, an exact E-step followed by an M-step that increases its stated bound cannot decrease the observed log-likelihood. The proof is in §9. With our fixed positive variance floor, every component density is at most $1/\sqrt{2\pi(0.05)}$. Their weighted average has the same upper bound, so the total log-likelihood is bounded above by $n\log[1/\sqrt{2\pi(0.05)}]$ and its monotone values have a finite limit. This statement concerns **objective values**. Parameter convergence, stationarity and optimality require further conditions; mixtures can have equivalent labelings, stationary configurations and different local solutions.

For example, initialize both components at mean 0 and variance 2.5. Every responsibility is 0.5. Both M-steps return mean 0 and variance 2.5, so log-likelihood stays −7.508335597. Repeating an identical operation cannot spontaneously break the symmetry. The separated fit above has a higher objective. An unchanged objective therefore need not identify a useful solution.

For unconstrained mixtures with at least two components, one can stay broad and give positive density to every observation while another shrinks onto A at −2. A component centered exactly on A has density proportional to $1/σ$ there. As its standard deviation tends to zero, that row's log-density grows without bound. The remaining broad component keeps the other rows finite. For weights $(0.5, 0.5)$, means $(−2, 0)$, broad variance 4 and shrinking standard deviations $(1,0.1,0.01,0.001)$, the four-row log-likelihoods are −8.1221, −6.9453, −4.6696 and −2.3697. This is covariance collapse, not successful recovery of four measurements.

> **Inline visual F3 — A spike can game the likelihood.** Align the density curves for the same broad component and three shrinking standard deviations. Show a separate exact log-likelihood strip with a logarithmic σ axis. Caption: “A component can increase the likelihood by concentrating on one observed location. A fixed variance floor stops this particular route to infinity.”

The program stops when the absolute average log-likelihood change is small, or after 50 iterations. It is written for the supplied small, positive-variance fixture, not arbitrary adversarial inputs. A substantive decrease is a diagnostic: first compare the same data, units and objective at matching iteration boundaries; then check stale responsibilities, normalization, covariance calculations and numerical conditioning. Tiny floating-point differences need tolerance. A modified or approximate update may optimize a different objective. `GaussianMixture(reg_covar=...)` adds to the covariance diagonal; that operation is not the variance-floor maximizer derived above, and the exact proof must not be attached to it without examining the update.

## 5. Let covariance describe the shape

For a vector $x\in\mathbb R^d$, a component mean μ has $d$ entries and its covariance Σ is a $d\times d$ symmetric positive-definite matrix. Its diagonal entries are variances. Off-diagonal entries describe how coordinates vary together within that component.

The multivariate Gaussian density is

\[
\mathcal N(x;\mu,\Sigma)=
\frac{\exp[-\tfrac12(x-\mu)^T\Sigma^{-1}(x-\mu)]}
{(2\pi)^{d/2}\sqrt{\det\Sigma}}.
\]

The quadratic form $D^2=(x-μ)^TΣ^{-1}(x-μ)$ is squared **Mahalanobis distance**. It measures displacement relative to a component's spread and orientation. The determinant term accounts for the volume covered by that component. A wide component spreads its probability mass over more space and has a lower peak. Both terms matter in responsibilities.

Take a two-feature component with mean $(0, 0)$ and

\[
\Sigma=\begin{pmatrix}1&0.75\\0.75&1\end{pmatrix},\qquad
\Sigma^{-1}=\frac{1}{0.4375}
\begin{pmatrix}1&-0.75\\-0.75&1\end{pmatrix}.
\]

Both $(1, 1)$ and $(1, −1)$ are Euclidean distance √2 from the mean. Yet $D^2(1,1)=8/7\approx1.142857$, while $D^2(1,-1)=8$. Positive correlation makes moving up together familiar and moving in opposite directions unusual. Their component densities are 0.135882281 and 0.004407103, respectively.

The contour $D^2=1$ is an ellipse. Its long direction is $(1,1)/\sqrt2$ with semiaxis $\sqrt{1.75}$; its short direction is $(1,-1)/\sqrt2$ with semiaxis $\sqrt{0.25}=0.5$. These are the eigenvector directions and square roots of the covariance eigenvalues, connecting to PCA. This contour is a constant-density boundary; in two dimensions it encloses $1-e^{-1/2}\approx39.35\%$ of that Gaussian's mass, not 68%. The familiar 68% statement belongs to a one-dimensional interval within one standard deviation.

> **Investigation I3 — Same distance, different plausibility.** Record which of two editable points will have higher density. Change correlation while holding both marginal variances at 1. Compare the displacement arrows along the covariance's eigenvector directions, the ellipse and the quadratic forms. Find a correlation that makes the two points equally plausible; then choose a new point pair and explain the result without relying on the original answer.

For vector observations, the M-step uses the same effective counts and weighted means as before. Replace the scalar squared deviation with an outer product:

\[
\Sigma_k^{new}=\frac1{N_k}\sum_i r_{ik}
(x_i-\mu_k^{new})(x_i-\mu_k^{new})^T.
\]

Each outer product contributes all variances and cross-products at once. For example, a residual $(2, −1)$ produces $\begin{pmatrix}4&-2\\-2&1\end{pmatrix}$. Weight it by that row's responsibility, add over rows, then divide by the effective count. Compute covariance around the *new* mean. A Cholesky factorization and triangular solves evaluate the density efficiently; explicitly computing a matrix inverse for every row is unnecessary.

### Four covariance choices, four model families

| `covariance_type` | Restriction | Covariance parameters | Geometric consequence |
| --- | --- | ---: | --- |
| `full` | Separate general $Σ_k$ | $K d(d+1)/2$ | Components can have different orientations and spreads |
| `tied` | One shared general $Σ$ | $d(d+1)/2$ | Same ellipse shape and orientation, translated to different means |
| `diag` | Separate diagonal $Σ_k$ | $Kd$ | Axis-aligned ellipses in the chosen feature coordinates |
| `spherical` | $Σ_k=v_k I$ | $K$ | A separate radius scale per component |

Add $Kd$ mean parameters and $K-1$ free weights to each count. For $d=2,K=3$, totals are respectively 17, 11, 14 and 11. Tied and spherical happen to have equal totals here, while allowing different shapes.

> **Inline visual F4 — Which ellipse freedoms remain?** Four matched two-component panels show full, tied, diagonal and spherical covariance. All use the same means. Matrix cells are linked to semiaxis directions and lengths; the caption states which covariance entries are shared or zero. Equal visual area is not imposed when the specified determinants differ.

In a Gaussian component, a diagonal covariance makes the features conditionally independent **given that component**. Mixing those components can still create dependence overall. The coordinate system matters: rotating correlated data can turn an axis-aligned restriction into a different model family. `spherical` permits a different variance for each component; it is not a shared-variance k-means switch.

## 6. Decide what a fitted mixture is for

Fit a density when the question concerns where future measurements occur, how to sample representative values, or how to score unusual measurements. Use responsibilities when a soft component allocation helps a downstream task. These are related outputs with different uses.

For density estimation, compare mean log-density on observations the fitting procedure has not seen. For component allocations with external labels, a permutation-invariant score such as ARI can answer a different question: whether the fitted partition agrees with a reference partition. Review [Clustering Evaluation & Validation](/learn/path/full-curriculum/clustering-evaluation-validation-silhouette-ari-nmi?module=classical-ml) for that denominator and label-permutation reasoning.

A component is a building block of a density, not necessarily a real-world category. Several components may approximate one curved or heavy-tailed population. Conversely, overlapping categories may be described adequately by one broad component. For irregular dense regions separated by gaps, revisit [DBSCAN](/learn/path/full-curriculum/dbscan-density-based-clustering?module=classical-ml). K-means remains a useful baseline when Euclidean grouping is the intended task; GMM adds a density model and covariance assumptions.

### Choose complexity with a declared rule

More components create more possibilities for fitting the training data. AIC and BIC compare fit with model size:

\[
\mathrm{AIC}=-2\ell(\hat\theta)+2p,\qquad
\mathrm{BIC}=-2\ell(\hat\theta)+p\log n,
\]

where $p$ is the parameter count, $n$ is the number of fitting observations and $\ell(\hat\theta)$ is their total fitted log-likelihood. Smaller is preferred. The two penalties differ: for $n>e^2$, BIC charges more per parameter. When likelihoods are evaluated on the same observations and representation, this tradeoff is directly comparable. Do not compare raw-density scores across different units, dimensions or data subsets as though their scale were unchanged.

These criteria are tools with statistical assumptions, not a certificate of a true category count. Mixtures have boundary and identifiability complications; asymptotic model-selection statements require their own conditions. Our real-data workflow declares **highest validation mean log-density** as its selection rule, shows training BIC as a diagnostic and reserves a final test set. We will not choose a new rule after seeing test results.

As a small calculation, suppose two fitted candidates on the same $n=100$ observations have $(ℓ,p)=(-150,5)$ and $(−143,11)$. AIC values are 310 and 308, favoring the larger model. BIC values are $300+5log100=323.025851$ and $286+11log100=336.656872$, favoring the smaller one. The 14-unit reduction in $-2\ell$ pays AIC's 12-unit penalty increase but not BIC's 27.631021-unit increase.

### Give fitting several starting points

EM depends on initialization, so compare several starts under the same objective. Keep the best converged fit from each declared model family. `GaussianMixture` uses `n_init` for this. Its `init_params='kmeans'` initializes responsibilities using a k-means fit; the separate `'k-means++'` option uses that seeding method. They are distinct settings. Set seeds, initialization counts, tolerance and iteration caps explicitly so a comparison has a reproducible meaning.

Check `converged_`, component weights and covariance eigenvalues. An effective count near one can indicate a narrow feature, an unusual observation, a poor start or excessive flexibility. Inspect the rows it accounts for and compare alternatives; a small weight does not establish which explanation is correct. Swapping complete component parameter sets changes their names but leaves density and log-likelihood unchanged. Align component identities before comparing component-specific parameters across fits; ARI already ignores label permutations.

## 7. Fit real flower measurements offline

The [UCI Iris dataset](https://archive.ics.uci.edu/dataset/53/iris) contains four measurements in centimeters for 150 iris plants, 50 from each of three species. It is associated with Fisher's 1936 work on distinguishing species using measurements. We reuse those observed measurements for a different question: **can a mixture predict sepal length and width density for held-out rows better than one Gaussian?** Species is withheld from fitting and selection, then used for one final partition diagnostic.

The provided [iris.csv](iris.csv) is exported from the corrected Iris snapshot bundled with scikit-learn 1.9.1. It contains all four measurements and species names, plus a stable one-based observation ID; this program uses only the two sepal columns. The [provenance record](data-provenance.md) gives attribution, CC BY 4.0 licensing, snapshot details and the historical row corrections. No download happens when the program runs.

Split the 150 row IDs with one fixed, label-blind permutation: 90 training, 30 validation and 30 test. Fit the two feature means and standard deviations on training only. Apply that same transform everywhere. Standardization fixes the coordinate scale in which regularization is specified; it does not make a mixture Gaussian by itself. The train means are $(5.772222222, 3.080000000)$ cm and scales are $(0.829863145, 0.420898510)$ cm.

This is a small held-out demonstration for this curated dataset. It is not a field-sampling study of plants across regions or seasons. A deployed data-collection task needs a split matched to its actual source, grouping and time structure.

Save the program as `iris_density.py` beside `iris.csv`, then run `python iris_density.py`. It requires NumPy and scikit-learn. The author calculation run used Python 3.12.14, NumPy 2.3.5 and scikit-learn 1.9.1; numerical fitting can vary across versions. All 16 candidates below converged in that run.

```python
import csv
from pathlib import Path
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

with Path("iris.csv").open(newline="", encoding="utf-8") as stream:
    rows = list(csv.DictReader(stream))
X = np.array([[float(row["sepal_length_cm"]),
               float(row["sepal_width_cm"])] for row in rows])
species = np.array([row["species"] for row in rows])
order = np.random.default_rng(16).permutation(len(X))
train, validation, test = order[:90], order[90:120], order[120:]
scaler = StandardScaler().fit(X[train])
Z = scaler.transform(X)

fits = []
print("covariance K validation_log_density training_BIC")
for kind in ["full", "tied", "diag", "spherical"]:
    for k in range(1, 5):
        model = GaussianMixture(
            n_components=k, covariance_type=kind,
            n_init=5, random_state=16, reg_covar=1e-4,
            tol=1e-6, max_iter=500,
        ).fit(Z[train])
        if not model.converged_:
            raise RuntimeError(f"Unfinished fit: {kind}, K={k}")
        score = model.score(Z[validation])
        fits.append((score, kind, k, model))
        print(kind, k, f"{score:.6f}", f"{model.bic(Z[train]):.3f}")

_, kind, k, selected = max(fits, key=lambda fit: fit[0])
baseline = fits[0][3]  # full covariance, K=1
print("selected", kind, k)
print("test log-density", f"{selected.score(Z[test]):.6f}")
print("baseline test log-density", f"{baseline.score(Z[test]):.6f}")
print("test ARI", f"{adjusted_rand_score(species[test], selected.predict(Z[test])):.6f}")
first = test[:1]
print("first test ID", int(first[0] + 1))
print("responsibilities", np.round(selected.predict_proba(Z[first])[0], 6))
print("log-density", np.round(selected.score_samples(Z[first])[0], 6))
```

`fit` receives an $n\times2$ matrix and no species labels. `score` returns an average log-density; multiply by the number of evaluated rows to get their total log-likelihood. `score_samples` returns one log-density per row. `predict_proba` returns normalized responsibilities of shape $(n,K)$, despite wording about component density in some API descriptions. `predict` chooses the largest responsibility, an argmax, for each row. These [GaussianMixture methods](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html) have different output meanings.

The author calculation produced this candidate table. Values are in the fixed training-standardized coordinates; higher validation score and lower BIC are preferred:

| Covariance | $K$ | Validation mean log-density | Training BIC |
| --- | ---: | ---: | ---: |
| full | 1 | −2.934881 | 533.238 |
| full | 2 | **−2.668800** | 490.753 |
| full | 3 | −2.671122 | 496.832 |
| full | 4 | −3.256749 | **461.595** |
| tied | 1 | −2.934881 | 533.238 |
| tied | 2 | −2.729729 | 500.157 |
| tied | 3 | −2.741336 | 496.608 |
| tied | 4 | −2.787305 | 499.707 |
| diag | 1 | −2.939660 | 528.817 |
| diag | 2 | −2.963611 | 528.980 |
| diag | 3 | −3.253775 | 537.340 |
| diag | 4 | −2.820248 | 541.401 |
| spherical | 1 | −2.939660 | 524.317 |
| spherical | 2 | −2.922018 | 527.058 |
| spherical | 3 | −2.968764 | 527.253 |
| spherical | 4 | −2.925480 | 532.312 |

The declared validation rule selects full covariance with two components. Full $K=3$ is very close on validation, so this small split supplies little separation between those candidates. Training BIC selects full $K=4$. Inspecting that fit reveals a component with sepal-width variance exactly 0.0001 in standardized coordinates, the additive covariance regularization level. It concentrates on a narrow line of rounded measurements. Its strong training score travels poorly to validation rows.

> **Inline visual F5 — Training fit and held-out fit answer different questions.** Two aligned panels show all candidate validation scores and training BIC values by $K$, including $K=1$. A third small panel shows the full-$K=4$ component's narrow variance against the training observations. Caption: “The chosen validation rule selects full $K=2$. Training BIC favors a sharper $K=4$ fit; inspect the component geometry before interpreting that preference.”

Now reveal the reserved test result for the selected model and baseline:

```text
selected full 2
test log-density -3.011223
baseline test log-density -2.887249
test ARI 0.433446
first test ID 34
responsibilities [1. 0.]
log-density -4.861128
```

On these 30 test observations the selected mixture scores **0.123975 nats per row lower** than the baseline. This run supports using the mixture as an exploratory description, but supplies no held-out improvement over the simple baseline on that test subset. Keep the declared rule and report this outcome. A further comparison should use newly specified repeated validation or new data, while preserving an honest final evaluation boundary.

The ARI of 0.433446 compares the model's two-component argmax partition with the three species labels on exactly those 30 test rows. It is a separate assessment of agreement, not the density score used to select the model. Row 34 also reconnects to §2: one responsibility rounds to 1 while its log-density is −4.861128. Rounding hides the small second responsibility, not an uncertainty calculation about species identity.

## 8. Use the density, and diagnose its failures

### From density to an anomaly decision

For a new observation in the same representation, define $s(x)=-\log  p(x)$. A larger score means less fitted density. If a workflow needs alerts, choose a threshold using a separate appropriate calibration set and a stated false-alert or review-budget goal; report false positives and missed cases where labels exist. The fitted mixture supplies a score, and the decision protocol supplies its operational meaning.

As a teaching scenario, imagine a device with two legitimate operating modes whose standardized readings follow §2's two-bell model. Values near either mode receive density; a value of 8 gets score 19.612086 even though its Right responsibility is nearly 1. A provisional rule $s>10$ flags it. Choosing 10 is an explicit scenario choice, not a claimed general false-alarm guarantee. If the device's normal operating range changes, calibration and possibly the density model need reassessment. This gives the previous anomaly lesson's score/action separation a concrete probabilistic mechanism.

### Generate a new measurement

Sampling goes in the forward direction from §2: select $k\sim\mathrm{Categorical}(π)$, then sample from its Gaussian. For a vector model, write $Σ_k=L_kL_k^T$ using a Cholesky factor, draw $ε\sim\mathcal N(0,I)$, and return $x=μ_k+L_kε$. Since $E[ε]=0$ and $\operatorname{Cov}(ε)=I$, this gives component mean μ_k and covariance $L_kL_k^T=Σ_k$.

The mean of the two-bell model is $0.5(-2)+0.5(2)=0$. Its variance is

\[
\operatorname{Var}(X)=\sum_k\pi_k\left[v_k+(\mu_k-E[X])^2\right]
=0.5(1+4)+0.5(1+4)=5.
\]

Variance combines within-component spread with separation between component means. A single Gaussian with mean 0 and variance 5 matches these two moments but fills in the central gap differently. Sampling a mixture means selecting a component for each observation; averaging two independently drawn component samples would produce another distribution.

### A compact diagnostic map

| Observation | Inspect | Useful response |
| --- | --- | --- |
| A covariance is singular or nearly singular | Eigenvalues, effective counts, duplicated or collinear rows, units | Reduce flexibility, choose a scale-appropriate regularization or constrained model, and compare held-out behavior |
| Different starts give different fits | Final objective, component alignment and covariance geometry | Retain declared best converged starts and report material instability |
| Objective changes become tiny | Tolerance, per-row versus total change, parameter movement | Check stopping criteria and whether the resulting density serves the task |
| A narrow component improves only training fit | Its rows, variance floor/regularization, validation scores | Investigate rounded measurements and excessive flexibility |
| Gaussian ellipses poorly describe the shape | Residual geometry, transformed features, tail behavior | Reconsider the representation or a different density/clustering family |

A positive-definite covariance may have a **negative log-determinant**: for $Σ=0.25I$ in two dimensions, $\det\Sigma=0.0625$ and $\log\det\Sigma<0$. Invalidity concerns nonpositive eigenvalues, not the sign of the logarithm. Likewise, more dimensions do not mechanically force responsibilities to become uniform; irrelevant features, limited data and ill-conditioned estimation cause particular problems that must be checked.

For dense full covariance, a typical EM iteration costs $O(nKd^2+Kd^3)$: rowwise quadratic forms and scatter matrices, plus component matrix factorizations. Diagonal and spherical versions reduce the main work to $O(nKd)$. Storing responsibilities costs $O(nK)$, and full covariances cost $O(Kd^2)$. At $K=10,d=5000$, covariance entries alone occupy $10\cdot5000^2\cdot8=2{,}000{,}000{,}000$ bytes in float64—2 decimal GB, about 1.86 GiB—before data and working memory. These are operation/storage estimates, not measured runtimes.

There is no universal feature-count cutoff or fixed number of EM iterations that makes a model appropriate. Data per component, covariance structure, conditioning and the task determine the tradeoff. For much larger data, online sufficient-statistic updates are a separate algorithmic choice; `GaussianMixture` exposes batch `fit`, not `partial_fit`. Dimension reduction can make estimation tractable, while changing the density question to the retained representation.

## 9. Deeper branch: why the EM bound works

This branch derives the guarantee used in §4. It needs conditional probability and the concavity of logarithms. It does not require variational-inference machinery.

If the hidden selectors were observed, the complete-data log-likelihood would be

\[
\log p(X,Z\mid\theta)=\sum_i\sum_k\mathbf1[z_i=k]
\left(\log\pi_k+\log\mathcal N(x_i;\mu_k,\Sigma_k)\right).
\]

Here $\mathbf1[z_i=k]$ is 1 for the selected component and 0 otherwise. In the E-step, use the current parameters $θ^{old}$ to take its conditional expectation. Replace each indicator by $r_{ik}^{old}$. The resulting function is

\[
Q(\theta\mid\theta^{old})=
\sum_i\sum_k r_{ik}^{old}\log p(x_i,z_i=k\mid\theta).
\]

**Q is the expected complete-data log-likelihood.** It does not include the entropy term that will appear in the lower bound. This distinction keeps two related quantities from acquiring the same name.

For any positive normalized allocation $q_i(k)$, multiply and divide within the mixture sum:

\[
\log p(x_i\mid\theta)=
\log\sum_k q_i(k)\frac{p(x_i,k\mid\theta)}{q_i(k)}
\geq\sum_k q_i(k)\log\frac{p(x_i,k\mid\theta)}{q_i(k)}.
\]

The inequality follows because log is concave: log of a weighted average is at least the weighted average of the logs. Zero terms are handled by their limiting values where the support permits. Summing over observations defines the **evidence lower bound**, or ELBO,

\[
\mathcal F(q,\theta)=
\underbrace{\sum_{i,k}q_i(k)\log p(x_i,k\mid\theta)}_{\text{expected complete-data log-likelihood}}
+\underbrace{\left(-\sum_{i,k}q_i(k)\log q_i(k)\right)}_{H(q)}.
\]

In other words, $F=Q+H$ when $q=r^{old}$, with $H(q)=-\sum q\log q$. Entropy $H$ is fixed during the M-step, so maximizing Q also maximizes this bound.

The E-step sets $q_i(k)=p(k\mid x_i,θ^{old})$. Then $p(x_i,k\midθ^{old})/q_i(k)=p(x_i\midθ^{old})$, independent of $k$. Jensen's inequality is an equality there: the bound **touches** the old log-likelihood. The M-step raises the same bound while q stays fixed. Therefore

\[
\ell(\theta^{new})\ \geq\ 
\mathcal F(q^{old},\theta^{new})\ \geq\
\mathcal F(q^{old},\theta^{old})\ =\ \ell(\theta^{old}).
\]

This chain is the complete reason. Raising an arbitrary lower bound would not suffice; the equality at the old parameter is essential. An M-step may merely increase Q, instead of finding its exact maximum, and the chain still holds: this is **generalized EM**. A covariance floor belongs in the feasible set of that maximization; adding a penalty changes the objective whose bound must be tracked.

> **Inline visual F6 — Touch, lift, touch again.** Use the exact four-row example's objective and bound values at the old and new parameters, joined by a labeled inequality chain. Do not invent smooth objective curves in a fictitious single parameter. Caption: “The old E-step makes the bound equal to the old objective. The M-step raises that bound. The new objective lies at least as high.”

Another way to locate the gap is

\[
\ell(\theta)-\mathcal F(q,\theta)
=\sum_i\mathrm{KL}\big(q_i\,\|\,p(z_i\mid x_i,\theta)\big)\geq0.
\]

For discrete probabilities, $\mathrm{KL}(q\|p)=∑_kq_klog(q_k/p_k)$. The E-step makes q equal to the conditional distribution, closing this gap. The E-step itself leaves θ—and hence the observed log-likelihood—unchanged. It improves the auxiliary distribution so that the next parameter update is justified.

This perspective generalizes beyond mixtures: EM is a strategy for latent-variable likelihood problems whose conditional expectations and weighted parameter fits are manageable. Dempster, Laird and Rubin's 1977 paper organized this general method and its earlier special cases; it was not the first time anyone had fitted a mixture. Wu's 1983 convergence work distinguishes stationary limit points from convergence of a whole parameter sequence. The unconstrained singular mixture in §4 falls outside a casual “bounded likelihood” guarantee.

## 10. Deeper branch: connections you can derive

### Precisely how k-means appears

This optional connection uses squared distance and the responsibility formula. Suppose all components share a fixed isotropic covariance $σ^2 I$ and have equal fixed weights. Then

\[
r_{ik}=\frac{\exp[-\|x_i-\mu_k\|^2/(2\sigma^2)]}
{\sum_j\exp[-\|x_i-\mu_j\|^2/(2\sigma^2)]}.
\]

For fixed means, the largest responsibility belongs to a nearest Euclidean mean. With finite positive variance, however, ordinary EM still uses *all* soft responsibilities in its mean update. Taking an argmax only after fitting does not undo those soft training updates.

There are two precise connections:

* If we explicitly replace responsibilities by one-hot nearest-mean assignments and update each mean to its hard group's average, while preserving the common isotropic/equal-weight restrictions, we obtain Lloyd's k-means updates. Ties and empty groups need the same conventions on both sides.
* In the limit $σ^2\to0$, responsibilities concentrate on a unique nearest mean. If several means are equally near, ties remain; with equal weights they split equally before a hard tie convention is applied. This limit explains the correspondence, without making finite-variance soft EM identical to k-means.

With means 0 and 3, observation 1 has squared distances 1 and 4. Its responsibility for mean 0 is $1/[1+e^{-3/(2σ^2)}]$. It is 0.592667 at $σ^2=4$, 0.817574 at 1 and 0.997527 at 0.25; the midpoint 1.5 stays 0.5 at every variance. Decreasing variance sharpens unequal distances, not exact ties.

`covariance_type='spherical'` instead allows one learned variance **per component**, and `GaussianMixture` also learns its weights. That family does not enforce the restrictions above. Thresholding at 0.5 is not an argmax rule for three or more components: a valid row $(0.40, 0.35, 0.25)$ would select nothing, although its argmax is the first component.

### Condition on a measurement to predict another

This optional application needs the rule for a conditional Gaussian, which we state here. It extends the reverse-selector idea to regression.

Suppose a joint GMM models two quantities $u$ and $v$. We observe $u$ and want a distribution for $v$. First update the mixing weights using only the marginal density of $u$:

\[
\alpha_k(u)=\frac{\pi_k\mathcal N(u;\mu_{u,k},\Sigma_{uu,k})}
{\sum_j\pi_j\mathcal N(u;\mu_{u,j},\Sigma_{uu,j})}.
\]

Within each component, the conditional mean and variance are

\[
m_k(u)=\mu_{v,k}+\Sigma_{vu,k}\Sigma_{uu,k}^{-1}(u-\mu_{u,k}),
\quad
V_k=\Sigma_{vv,k}-\Sigma_{vu,k}\Sigma_{uu,k}^{-1}\Sigma_{uv,k}.
\]

Then $p(v\mid u)=∑_kα_k(u)\mathcal N(v;m_k(u),V_k)$. The covariance term shifts the predicted $v$ according to the observed deviation in $u$. These are conditional distributions from one fixed fitted joint model; fitting that model remains an earlier step.

In an invented two-mode measurement process, use equally weighted component means $(0,0)$ and $(2,4)$, each with covariance $\begin{pmatrix}1&0.5\\0.5&1\end{pmatrix}$. At $u=1$, the marginal $u$ densities tie, so the updated weights are $(0.5,0.5)$. Conditional means are 0.5 and 3.5, and both conditional variances are 0.75. Thus the prediction for $v$ is a two-component mixture, with mean 2 and variance $0.75+0.5(1.5^2)+0.5(1.5^2)=3$. Reporting only the mean 2 would conceal the two plausible outcomes. This is a useful reason to retain a predictive distribution rather than only a regression line.

### Bayesian mixtures change the fitting problem

Maximum-likelihood EM estimates one parameter set. A Bayesian mixture puts distributions on weights, means and covariances; variational inference approximates their posterior through an evidence bound. Scikit-learn's `BayesianGaussianMixture` supports a finite Dirichlet weight prior and a truncated Dirichlet-process construction. `n_components` is a finite cap in the implementation, even for the latter. Small weights can make some fitted components practically inactive.

An “effective component count” needs a declared weight threshold. With six available components and the same 90 standardized Iris training rows, our author calculation used a Dirichlet-process weight prior with concentrations 0.01, 1 and 10. Counting weights above 0.01 gives 3, 3 and 4. Counting weights above 0.05 gives 2 for all three. This is sensitivity to prior and reporting choices, not automatic proof of a biological category count.

To inspect the extension, append this complete block to the Iris program after its fit:

```python
from sklearn.mixture import BayesianGaussianMixture

for concentration in [0.01, 1., 10.]:
    bayesian = BayesianGaussianMixture(
        n_components=6, covariance_type="full",
        weight_concentration_prior_type="dirichlet_process",
        weight_concentration_prior=concentration,
        n_init=3, random_state=16, reg_covar=1e-4,
        tol=1e-6, max_iter=1000,
    ).fit(Z[train])
    print(concentration, np.round(bayesian.weights_, 4),
          int((bayesian.weights_ > 0.01).sum()),
          int((bayesian.weights_ > 0.05).sum()),
          bayesian.converged_)
```

Checked weights rounded to four decimals are:

| Concentration | Six weights | Count $>0.01$ | Count $>0.05$ |
| --- | --- | ---: | ---: |
| 0.01 | [0.0118, 0.3748, 0.6133, 0.0001, 0.0000, 0.0000] | 3 | 2 |
| 1 | [0.0117, 0.3713, 0.6075, 0.0055, 0.0027, 0.0014] | 3 | 2 |
| 10 | [0.0114, 0.3643, 0.5969, 0.0101, 0.0091, 0.0082] | 4 | 2 |

All three fits converged in the author calculation. Priors and the truncation cap remain modeling choices to assess. Do not compare this model's variational evidence bound numerically with ordinary GMM's observed-data log-likelihood as if they were the same objective. See the [BayesianGaussianMixture guide](https://scikit-learn.org/stable/modules/mixture.html#variational-bayesian-gaussian-mixture) after the core workflow.

## 11. Practise on changed problems

Attempt each task before opening its hint or solution. Explanations are part of the success criterion.

### 1. Reverse the allocation with unequal weights

Two one-dimensional components have weights $(0.25, 0.75)$, means $(0, 2)$ and variances $(1, 1)$. At $x=1$, compute the mixture density and both responsibilities. Then find the measurement at which the two responsibilities tie.

<details><summary>Hint</summary>

At 1 the Gaussian heights match. For the boundary, equate the two weighted densities and take logarithms; the squared terms simplify to a linear equation.

</details>
<details><summary>Explained solution</summary>

At 1, each Gaussian height is $e^{-1/2}/\sqrt{2\pi}=0.241970725$. The mixture density is that same height and responsibilities are $(0.25,0.75)$. The tie equation is $\log (0.25)-x^2/2=\log (0.75)-(x-2)^2/2$, giving $x=1-\log (3)/2=0.450693856$. The heavier component's region extends toward the lighter component's center. The midpoint alone ignores prior component weights.

</details>

### 2. Do an M-step from a new soft allocation

Measurements are 0, 2 and 5. Fixed responsibilities for component A are 0.8, 0.4 and 0.1; B gets the remainder. Find A's effective count, new weight, mean and variance. Then apply a minimum variance of 3.

<details><summary>Hint</summary>

Use the weighted first and second moments. The floor clips the result upward only when it is below 3.

</details>
<details><summary>Explained solution</summary>

$N_A=1.3$, weight $=1.3/3=13/30\approx0.433333$. Weighted sum is $0+0.8+0.5=1.3$, so mean $=1$. Weighted second moment is $(0+1.6+2.5)/1.3=41/13$. Variance is $41/13-1=28/13\approx2.153846$, or equivalently $(0.8\cdot1+0.4\cdot1+0.1\cdot16)/1.3$. The constrained variance becomes 3. Adding 3 would implement a different update.

</details>

### 3. Repair plausible numerical reasoning

Someone reports responsibilities $(0,0,0)$ for a distant point, uses the sum of its clipped component densities as “probability of its cluster,” and rejects $\begin{pmatrix}0.25&0\\0&0.25\end{pmatrix}$ because its log-determinant is negative. Diagnose all three statements and give a concrete repair.

<details><summary>Hint</summary>

Check the required sum of a responsibility row, the units of a density and the eigenvalues of the covariance.

</details>
<details><summary>Explained solution</summary>

A responsibility row must sum to 1. Evaluate log weighted densities and normalize using log-sum-exp; denominator clipping cannot recover ratios already lost to underflow. The sum of weighted component densities is the marginal measurement density. A component's probability conditional on that measurement is its weighted density divided by the sum. The matrix has eigenvalues 0.25 and 0.25, both positive; its determinant is 0.0625 and a negative log-determinant is valid. A successful Cholesky factor is $0.5I$.

</details>

### 4. Change the geometry without changing distance

With mean $(0,0)$, compare density at $(2,1)$ and $(2,−1)$ under covariances [[1,0.5],[0.5,1]], I, and [[1,−0.5],[−0.5,1]]. Predict all three orderings before calculating. Give exact squared Mahalanobis distances and explain why the Euclidean distances tie.

<details><summary>Hint</summary>

Both squared Euclidean distances are 5. For correlation rho and marginal variances 1, the quadratic form is $(x^2+y^2-2\rho xy)/(1-\rho^2)$.

</details>
<details><summary>Explained solution</summary>

For positive correlation the squared distances are $4$ and $28/3$, so $(2,1)$ has greater density. At zero correlation they are both $5$, so the densities tie. For negative correlation the squared distances reverse to $28/3$ and $4$. The determinants agree between the positive and negative cases at $3/4$, so their normalization factors agree. Equal Euclidean distance is insufficient because the cross-product contributes with a different sign.

</details>

### 5. Make and assess a real-data decision

Using the supplied CSV and §7's fixed split, restrict candidates to $K=1,2$, allowing all four covariance types. Select by validation score before reading test results. Reproduce the selected model and baseline test scores, then explain whether you would claim a predictive improvement. Finally, hide or permute the species column and explain which calculations should stay unchanged.

<details><summary>Hint</summary>

The reduced candidate set still includes the validation winner. Species is used in exactly one reported diagnostic.

</details>
<details><summary>Explained solution</summary>

The reduced search selects full $K=2$, validation mean log-density −2.668800. The test scores remain −3.011223 for the selected mixture and −2.887249 for full $K=1$; the mixture is lower by 0.123975 nats per row. Report that the validation preference did not yield a test improvement in this demonstration. No need to force a positive conclusion. Feature scaling, fits, responsibilities, densities, BIC and the selected candidate stay unchanged when species labels are changed. ARI may change after a permutation of labels *across observations*. Merely renaming the three species consistently leaves ARI unchanged.

</details>

### 6. Find the missing line in a proof

An argument says: “F is below the likelihood. An update increases F. Therefore the likelihood increases.” Explain why this reasoning is incomplete, give two pairs of numbers that refute it, then supply the EM property that repairs it.

<details><summary>Hint</summary>

A lower bound may rise while remaining below a falling quantity. EM does more than construct any lower bound.

</details>
<details><summary>Explained solution</summary>

Old values $F=0,ℓ=10$ and new values $F=1,ℓ=2$ satisfy both lower-bound inequalities and increase F, while likelihood decreases. EM's E-step ensures $F(q^{old},θ^{old})=ℓ(θ^{old})$. Holding that q fixed in an improving M-step then gives the three-term chain in §9. A higher arbitrary bound is insufficient.

</details>

### 7. Separate a hard limit from a library setting

With equal weights, means 0 and 3, and shared variance 1, calculate the first component's responsibility at $x=0.5$. Then set shared variance to 0.25. Explain why neither calculation proves `GaussianMixture(covariance_type='spherical')` implements k-means.

<details><summary>Hint</summary>

The squared distances are 0.25 and 6.25. Identify every model restriction and which weights the mean update uses.

</details>
<details><summary>Explained solution</summary>

Responsibilities are $1/(1+e^{-3})=0.952574127$ and $1/(1+e^{-12})=0.999993856$. They approach a one-hot choice as shared variance tends to zero for a unique nearest mean. Finite-variance EM still updates means using soft weights. The spherical API learns separate component variances and weights, so it also lacks the shared-variance/equal-weight restrictions. Explicit hard nearest-center assignments plus hard mean updates yield the Lloyd correspondence.

</details>

### 8. Retain predictive spread in a two-mode process

Use §10's joint model, but observe $u=0$. Find the updated weight of the component centered at $(2,4)$, both conditional means, and the conditional mean of $v$. Explain why the covariance matters even before mixing the conditional predictions.

<details><summary>Hint</summary>

The marginal $u$ density ratio for components two versus one is $e^{-2}$. Each conditional mean shifts by $0.5(u-μ_u)$.

</details>
<details><summary>Explained solution</summary>

The second weight is $e^{-2}/(1+e^{-2})=0.119202922$. Conditional means are 0 and $4+0.5(0-2)=3$; both variances remain 0.75. The conditional mean is $0.880797078(0)+0.119202922(3)=0.357608766$. Correlation changes the within-component conditional means from 0 and 4 to 0 and 3. Ignoring covariance would change the prediction even if the updated component weights were correct.

</details>

## 12. Readiness and the next question

A GMM selects a hidden component and then generates a measurement. Bayes' rule reverses that process into responsibilities. EM uses those probabilities as fractional memberships, then updates weighted statistics. Its objective is a density fit; covariance structure, initialization and the evaluation protocol determine what that fit can support.

You are ready to continue when you can explain the 8-versus-2 density contrast, calculate one changed EM cycle, identify the constrained objective in the small program, read a covariance ellipse and reproduce the held-out Iris comparison. For the deeper route, repair the missing equality in the EM proof, derive the k-means limit and keep a conditional mixture distribution instead of reporting only its mean.

Next is [t-SNE, UMAP & Manifold Learning](/learn/path/full-curriculum/t-sne-umap-manifold-learning?module=classical-ml). We have fitted a density in declared measurement coordinates. The next lesson asks how to display neighborhood relationships in fewer dimensions, and what a picture preserves. Later, ICA's hidden variables represent simultaneous source contributions; a GMM's selector chooses one component for an observation. Keeping these meanings of “hidden component” distinct will make those connections easier.

## References & another way to learn it

**Alternate explanations.** These supplement the complete lesson above.

* Christopher Bishop, [*Pattern Recognition and Machine Learning*, chapter 9](https://www.microsoft.com/en-us/research/wp-content/uploads/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf), freely available textbook PDF. §§9.2–9.4 develop mixtures, singular likelihoods, EM, the k-means relationship and the lower bound. Best after the hand example; matrix notation and probability are assumed. Chapter list and relevant pages were read during authoring; this is a theory reference, not current library documentation.
* Andrew Ng, [Mixtures of Gaussians and the EM Algorithm](https://cs229.stanford.edu/notes_archive/cs229-notes7b.pdf), short Stanford course notes. Useful after §3 for seeing the updates in compact mathematical form. Their φ and $w_j^{(i)}$ correspond to our mixing weights and responsibilities. All four pages were reviewed.
* Tengyu Ma and Andrew Ng, [The EM Algorithm](https://cs229.stanford.edu/notes-spring2019/cs229-notes8.pdf), Stanford notes dated May 13, 2019. Read the Jensen and EM sections alongside §9. They supply a slower route from concavity to the bound; review the exact convergence distinctions in §4 of this lesson when reading informal convergence wording.
* Andrew Ng, [Stanford CS229 Lecture 12](https://see.stanford.edu/Course/CS229/43), video with [substantive transcript](https://see.stanford.edu/materials/aimlcs229/transcripts/MachineLearning-Lecture12.html). Its transition from density estimation to hidden labels and soft weighted updates provides another verbal explanation after §§2–3. The relevant transcript sections and companion notes were reviewed; the video was not watched. The transcript has recognition errors and missing equations, so use the notes for exact notation. Earlier k-means discussion and later Jensen discussion make it a longer resource than the core GMM explanation.

**Exact technical and data references.**

* Dempster, Laird and Rubin (1977), [Maximum Likelihood from Incomplete Data via the EM Algorithm](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x). The original general framework; the publisher's abstract and bibliographic record were reviewed, with mathematical derivations cross-checked through Bishop and Stanford. Full publisher access may require a subscription.
* C. F. Jeff Wu (1983), [On the Convergence Properties of the EM Algorithm](https://doi.org/10.1214/aos/1176346060). Distinguishes stationary limit points and parameter-sequence convergence under explicit conditions. The paper's abstract was reviewed; full-text retrieval failed during authoring, so no detailed theorem-number guidance is claimed.
* Scikit-learn, [GaussianMixture API](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html) and [mixture guide](https://scikit-learn.org/stable/modules/mixture.html). Parameter, scoring and covariance references checked against documentation identifying itself as 1.9.1. Use these for library semantics; the mathematically qualified convergence discussion in this lesson is more precise than the guide's broad overview wording.
* Scikit-learn, [BayesianGaussianMixture API](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.BayesianGaussianMixture.html). Optional prior/truncation branch after §10; a variational evidence objective rather than ordinary maximum-likelihood EM.
* Fisher, R. (1936), [Iris, UCI Machine Learning Repository](https://doi.org/10.24432/C56C76), CC BY 4.0. The dataset page, measurement units and licensing were reviewed; the supplied file is the corrected [scikit-learn Iris snapshot](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html), with exact provenance in the accompanying file.
