# Gaussian Processes (GP)

A temperature probe gives you two readings along a pipe. You want the temperature between them, but you also need to decide where another measurement would be useful. A curve alone leaves out the second question. A Gaussian process lets you express which curves are plausible, update those beliefs with observations, and inspect uncertainty at places you have not measured.

The key move is to describe how **function values vary together**. If nearby temperatures usually move together, one reading tells us something about its neighbors. How far that information travels is a modeling decision, encoded in a covariance function.

**First pass:** follow sections 1–7, including the two-observation calculation and the CO₂ experiment. You should finish able to explain a GP prediction, distinguish its two uncertainty bands, run a small regression, and recognize a misleading forecast. Section 8 explores measurement selection; section 9 develops classification, scalable inference, and the connection to kernel ridge regression. Those branches need the core equations but are optional on a first reading.

You need vectors, matrix multiplication, an average, and the idea of a normal distribution. A normal variable has a center called its mean and spread described by its variance; standard deviation is the square root of variance. We introduce the required Gaussian conditioning operation here. The previous CRF lesson modeled dependent discrete labels. Here the dependent quantities are numerical function values, and Gaussian algebra makes the basic regression calculation exact.

## 1. A distribution over function values

Imagine recording a possible temperature at each of three positions. One possible state is the vector `[1.0, 0.6, −0.2]`. Another is `[−0.5, 0.1, 0.4]`. Drawing many such vectors and joining values at their positions gives many possible curves. The lines joining the points are a drawing convention; a finite drawing is not the entire continuous function.

**Inline figure F1 — from vector to curve:** three labeled positions, a three-coordinate sample, and the corresponding connected points. Show several samples from the same joint distribution. Ask: why should the middle point not be sampled independently of its neighbors?

A **Gaussian process** is a collection of random variables, one for each input, such that every finite collection has a joint multivariate normal distribution. We write

\[
f\sim\operatorname{GP}(m,k),\qquad
(f(x_1),\ldots,f(x_n))^T\sim\mathcal N(m_X,K_{XX}).
\]

Here $m(x)$ is the prior mean at input $x$, and $K_{ij}=k(x_i,x_j)$ is the covariance between two function values. The word “process” does not require time: inputs could be positions, material compositions, or settings of an expensive simulator. “Gaussian” describes distributions of function values, not a requirement that the inputs form a bell curve.

Covariance records whether deviations from the mean tend to move together. Its diagonal entries are variances. If outputs are degrees Celsius, covariance has units °C². Correlation divides covariance by the two standard deviations and is dimensionless. A covariance need not lie between zero and one, and negative covariance can be valid.

There is a constraint: any finite kernel matrix must be symmetric and positive semidefinite. In plain terms, every weighted combination of its random variables must have nonnegative variance:

\[
\operatorname{Var}(a^Tf_X)=a^TK_{XX}a\geq0.
\]

An arbitrary “similarity” score does not necessarily satisfy this requirement. For example, the symmetric matrix 
$\begin{bmatrix}1&2\\2&1\end{bmatrix}$
would give the difference of its variables variance $1+1-2(2)=-2$. It cannot be a covariance matrix.

There is a familiar finite-dimensional example. Let $f(x)=a+bx$, with independent $a,b\sim\mathcal N(0,1)$. Every vector of function values is a linear transformation of Gaussian weights, so this is a GP with $m(x)=0$ and $k(x,z)=1+xz$. Bayesian linear regression already supplies uncertainty over functions. More flexible kernels let us work without explicitly constructing a large feature vector. This connection is developed from both weight and function viewpoints in [GPML, chapter 2](https://gaussianprocess.org/gpml/chapters/RW2.pdf).

**Quick prediction.** For this random-line prior, can a sampled curve have a sudden bend? No: its possible curves are straight lines. Gaussian marginals alone do not mean “anything can happen”; the covariance restricts how values relate.

## 2. One measurement: the entire mechanism in two numbers

Separate the underlying quantity $f(x)$ from a measurement:

\[
y=f(x)+\varepsilon,\qquad \varepsilon\sim\mathcal N(0,\sigma_n^2).
\]

The noise has mean zero and is independent of the function. A measurement can lie above or below the underlying curve. With noise, a sensible fitted curve need not pass through every observation.

Consider two locations, an observed location $x_0$ and a target $x_*$. Give both latent values prior mean zero and variance one. Their covariance is ρ. Suppose the measurement noise variance is 0.25 and we observe $y_0=2$. The measurement has variance $1+0.25=1.25$. Conditioning gives

\[
\underbrace{\mathbb E[f_*\mid y_0]}_{\text{updated center}}
=\frac{\rho}{1.25}\,2,
\qquad
\underbrace{\operatorname{Var}(f_*\mid y_0)}_{\text{remaining uncertainty}}
=1-\frac{\rho^2}{1.25}.
\]

For ρ = 0.5, the mean becomes **0.8** and the variance becomes **0.8**. A positively related location moves upward when the observed location is high. The reduction in variance is $0.5^2/1.25=0.2$: some uncertainty was shared with the measurement and has now been resolved.

At ρ = 0, the reading changes neither the target mean nor its variance. With no covariance, these jointly Gaussian quantities are independent. Returning to ρ = 0.5, if the reading changes from 2 to −2 while the covariance and noise stay fixed, the target mean changes sign, but its variance stays 0.8. The observed value tells us **where** to move; the covariance and observation precision tell us **how much information** the measurement contains.

**Inline figure F2 — a conditional slice:** a joint Gaussian ellipse for $y_0$ and $f_*$, a vertical line at $y_0=2$, and the resulting one-dimensional conditional distribution centered at 0.8. Place the variance subtraction beside its width. A separate zero-covariance panel shows why the slice has the same width and center as the marginal.

There are two different future questions:

| Question | Distribution in this example | What remains uncertain? |
| --- | --- | --- |
| What is the underlying value $f_*$? | $\mathcal N(0.8,0.8)$ | The latent function |
| What would a new measurement $y_*$ report? | $\mathcal N(0.8,1.05)$ | The function plus new independent noise |

The second variance is $0.8+0.25$. A pointwise 95% Bayesian credible interval for the latent value is $0.8\pm1.96\sqrt{0.8}$, approximately [−0.953, 2.553]. The new-observation predictive interval is approximately [−1.208, 2.808]. These statements are conditional on the specified model and hyperparameters. They are not a promise that 95% of an entire curve lies inside a pointwise band, nor automatic frequentist coverage on a different data-generating process.

**Investigation I1 — move a measurement, watch information travel.** Before revealing an update, record which will change: the mean, latent width, both, or neither. Edit the actual observation locations and values; then compare short and long correlation ranges. The plot must distinguish latent and observation bands. Keep the noise and kernel fixed for the value-only comparison. Explain the result using the covariance connection, not merely “the curve moved.”

## 3. Many measurements: condition one larger Gaussian

Let $X$ contain $n$ observed inputs and $X_*$ contain $q$ target inputs. Define $r=y-m_X$, the observed residual from the prior mean. Let $R$ be the $n\times n$ observation-noise covariance; independent equal-variance noise gives $R=\sigma_n^2I$.

The joint model and conditional result are

\[
\begin{bmatrix}y\\f_*\end{bmatrix}
\sim\mathcal N\left(
\begin{bmatrix}m_X\\m_*\end{bmatrix},
\begin{bmatrix}K_{XX}+R&K_{X*}\\K_{*X}&K_{**}\end{bmatrix}\right),
\]
\[
C=K_{XX}+R,\qquad
\mu_*=m_*+K_{*X}C^{-1}r,\qquad
\Sigma_*=K_{**}-K_{*X}C^{-1}K_{X*}.
\]

Track the shapes: $C$ is $n\times n$; $K_{X*}$ is $n\times q$; μ is a $q$-vector; Σ is $q\times q$. Its diagonal gives marginal variances. Off-diagonal entries tell us how prediction errors at different targets remain related. Drawing an independent error bar at each target does not display that relationship.

The mean formula starts from the prior and adds an observation-based correction. The covariance formula starts from prior uncertainty and subtracts the part explained by observations. With fixed kernel and noise, adding an independent noisy observation cannot increase the conditional variance. Refitting hyperparameters changes the model itself, so comparisons across refits do not inherit that guarantee.

Known unequal Gaussian noise is still exact: use $R=\operatorname{diag}(\sigma_1^2,\ldots,\sigma_n^2)$. Known correlated Gaussian noise can also be handled with a full $R$, with appropriate cross-covariance terms if future noise is correlated with past noise. Unknown input-dependent noise requires an additional estimation model; it is not the same problem as simply supplying known variances.

### A two-observation calculation

Use inputs $X=[0,2]$, values $y=[1,-1]$, zero mean, noise variance 0.25, and the radial basis function (RBF) kernel

\[
k(x,z)=\exp\left[-\frac{(x-z)^2}{2\ell^2}\right],\qquad \ell=1.
\]

The off-diagonal covariance is $e^{-2}\approx0.135335$, so

\[
C=\begin{bmatrix}1.25&0.135335\\0.135335&1.25\end{bmatrix}.
\]

At the midpoint $x_*=1$, both cross-covariances are $e^{-1/2}\approx0.606531$. The opposite observed values cancel in the mean, giving zero. But their information does not cancel: latent variance falls from one to **0.468895**. A zero prediction can be an informed estimate rather than an absence of evidence.

| Target $x_*$ | Posterior mean | Latent variance | New-observation variance |
| --- | ---: | ---: | ---: |
| 0 | 0.775717 | 0.199407 | 0.449407 |
| 1 | 0 | 0.468895 | 0.718895 |
| 2 | −0.775717 | 0.199407 | 0.449407 |
| 4 | −0.121112 | 0.985182 | 1.235182 |

At 4 the data have little influence under this kernel. Farther away, the RBF cross-covariances approach zero: the mean returns to the prior mean and latent variance returns to the prior variance, one. The uncertainty does not disappear because the mean returns to zero.

**Inline figure F3 — the calculation as a map:** align observation locations, the $2\times2$ matrix, the target's two cross-covariances, and the final mean/variance. Highlight equal midpoint connections while keeping the observed signs visible.

### Compute by solving, not by explicitly inverting

For a positive-definite $C$, Cholesky factorization gives $C=LL^T$. Solve $Lz=r$ and $L^T\alpha=z$. For all targets, solve $LV=K_{X*}$. Then

\[
\mu_*=m_*+K_{X*}^T\alpha,\qquad
\Sigma_*=K_{**}-V^TV.
\]

This avoids explicitly forming a matrix inverse and reuses the same factor for many predictions. Here is the complete small example. Save it as `gp_conditioning.py`. In an isolated Python environment install `numpy==2.3.5 scipy==1.18.1`, then run `python gp_conditioning.py`.

```python
import numpy as np
from scipy.linalg import solve_triangular

def rbf(x, z, length=1.0):
    distances = np.asarray(x)[:, None] - np.asarray(z)[None, :]
    return np.exp(-0.5 * (distances / length) ** 2)

def predict(x, y, targets, noise_variance=0.25):
    covariance = rbf(x, x) + noise_variance * np.eye(len(x))
    lower = np.linalg.cholesky(covariance)
    intermediate = solve_triangular(lower, y, lower=True)
    weights = solve_triangular(lower.T, intermediate, lower=False)
    cross = rbf(x, targets)
    solved_cross = solve_triangular(lower, cross, lower=True)
    mean = cross.T @ weights
    latent_covariance = rbf(targets, targets) - solved_cross.T @ solved_cross
    log_marginal = (
        -0.5 * y @ weights
        - np.log(np.diag(lower)).sum()
        - 0.5 * len(x) * np.log(2 * np.pi)
    )
    return mean, latent_covariance, log_marginal

x = np.array([0.0, 2.0])
y = np.array([1.0, -1.0])
targets = np.array([0.0, 1.0, 2.0, 4.0])
mean, covariance, log_marginal = predict(x, y, targets)
print(np.round(mean, 6))
print(np.round(np.diag(covariance), 6))
print(round(float(log_marginal), 6))
changed = predict(x, np.array([3.0, -2.0]), targets)
print(np.allclose(covariance, changed[1]))
```

Output:

```text
[ 0.775717  0.       -0.775717 -0.121112]
[0.199407 0.468895 0.199407 0.985182]
-2.952256
True
```

The last line checks a conceptual claim: changing only measured values preserves covariance. A materially negative computed variance signals a problem. Tiny negative roundoff may be clipped only within a documented numerical tolerance. If Cholesky fails, investigate an invalid kernel, duplicate noise-free observations, numerical scale, or nearly dependent rows. Small diagonal jitter can stabilize a valid near-singular system, but it changes that system; choose and report it relative to the covariance scale. Do not disguise substantial extra modeled noise as a numerical detail.

## 4. Kernels express the kinds of change you expect

For the RBF kernel $k=\sigma_f^2\exp[-r^2/(2\ell^2)]$, $r=|x-z|$, $\sigma_f^2$ is latent variance and ℓ is a length scale in input units. It is a correlation range, not a period. Smaller ℓ means observations have more local influence. It does not give a periodic prior or force the curve to interpolate noisy measurements.

In the same two-observation problem, midpoint mean is zero at every listed scale, but uncertainty differs:

| RBF length ℓ | Midpoint latent variance | Mean at target 4 | Why it changes |
| --- | ---: | ---: | --- |
| 0.3 | 0.999976 | approximately 0 | Neither observation strongly informs the gap |
| 1 | 0.468895 | −0.121112 | Information connects nearby locations |
| 3 | 0.127300 | −0.867255 | Long-range dependence strongly constrains the gap |

**Inline figure F4 — same random draws, different assumptions:** show three prior samples for each length scale on the same axes, alongside the corresponding covariance matrix. Reuse the same underlying normal draws so that changing the assumption, rather than unrelated randomness, drives the comparison. These are computed draws, not hand-drawn curves labeled as samples.

The following choices answer different modeling questions. The normalized stationary forms below are multiplied by an output variance amplitude when needed.

| Kernel | Form or construction | Modeling question |
| --- | --- | --- |
| RBF | $e^{-r^2/(2\ell^2)}$ | Is an extremely smooth latent function plausible? |
| Matérn 3/2 | $(1+\sqrt3r/\ell)e^{-\sqrt3r/\ell}$ | Should changes be less smooth than an RBF permits? |
| Matérn 5/2 | $(1+\sqrt5r/\ell+5r^2/(3\ell^2))e^{-\sqrt5r/\ell}$ | Is a smoother, but still finite-smoothness, model appropriate? |
| Periodic | $e^{-2\sin^2(\pi r/p)/\ell^2}$ | Should positions one period $p$ apart share the same latent value? |
| Linear | $\sigma_b^2+\sigma_w^2xz$ | Could a random intercept and slope explain the function? |
| Rational quadratic | $(1+r^2/(2a\ell^2))^{-a},\ a>0$ | Would a mixture of RBF length scales help? |

Matérn parameter ν controls mean-square differentiability: an integer-order mean-square derivative of order $j$ exists when ν > $j$. Thus 3/2 and 5/2 permit one and two such derivatives. RBF permits every order. This is a property of the stochastic model, not something proved by a smooth-looking finite plot. [GPML, chapter 4, §§4.1–4.2](https://gaussianprocess.org/gpml/chapters/RW4.pdf) develops these distinctions.

Adding valid kernels gives another valid kernel. If $f=f_1+f_2$ and the two zero-mean GP components are independent, their covariances add. A trend plus a seasonal effect therefore has a direct probabilistic interpretation.

Multiplying valid kernels also gives a valid kernel. For example, periodic × RBF preserves seasonal resemblance while making it fade across distant years. This is often called locally periodic covariance. The resulting model is a GP with that product covariance; multiplying two GP sample functions does **not** generally produce Gaussian function values.

For multiple input features, an RBF can use

\[
k(x,z)=\sigma_f^2\exp\left[-\frac12\sum_j\frac{(x_j-z_j)^2}{\ell_j^2}\right].
\]

A large fitted $\ell_j$ means the model changes little across that feature's observed range, holding others fixed. This automatic relevance determination (ARD) is sensitive to units, correlated inputs, and fitting assumptions; it does not establish causal importance. Fit any scaling on training inputs and preserve it for later inputs.

**Try a counterexample.** Under a perfectly periodic kernel, a point ten complete periods away can be strongly related to an observation. “Farther away always means more uncertainty” is therefore an RBF-style intuition, not a universal GP rule.

## 5. Learn hyperparameters while checking the task you actually care about

Kernel parameters and noise assumptions affect both the fit and the uncertainty. One way to choose them is the **log marginal likelihood**. It is the log density of all training observations after integrating out the latent function values:

\[
\log p(y\mid X,\theta)
=-\tfrac12r^TC^{-1}r-\tfrac12\log|C|-\tfrac n2\log(2\pi).
\]

The first term penalizes residuals in directions the covariance considers unlikely. The determinant accounts for the volume over which probability density is distributed. The last term normalizes the Gaussian. A model cannot freely broaden every direction to fit anything without changing how much density it gives the actual observations. Using $C=LL^T$, compute half the log determinant as $\sum_i\log L_{ii}$, as the program did.

For our deliberately conflicting observations `[1, −1]`, log marginal likelihood is −2.861021, −2.952256, and −4.022773 for lengths 0.3, 1, and 3. Among these three fixed candidates, the short scale gives the observations higher density. This is a small comparison, not proof that the short scale is globally optimal or that shorter scales always win.

Optimizing θ integrates out $f$ but **does not integrate out θ**. A single optimized setting is an empirical-Bayes, plug-in choice. Full hyperparameter inference averages predictions over a posterior on θ and can reflect additional uncertainty. Optimization can have local optima; multiple initialized fits and sensible bounds help diagnose this, but do not guarantee a global optimum. [GPML, chapter 5, §§5.3–5.4](https://gaussianprocess.org/gpml/chapters/RW5.pdf) treats marginal likelihood alongside cross-validation.

Training marginal likelihood and held-out forecasting answer different questions. A high training density does not show that the next two years will be forecast well. Preserve a development protocol that matches the future use, compare a simple baseline, and keep final test outcomes out of model selection.

## 6. A real experiment: predict future monthly CO₂

The NOAA Global Monitoring Laboratory measures atmospheric CO₂ at Mauna Loa. Our offline file contains monthly means for **1990–1999**, 120 rows. Outputs are parts per million (ppm). These are historical observations, not simulated points. The source marks interpolated months with negative spread/uncertainty fields; none of the selected months has that flag. The retained provenance records the download and public-domain attribution. [NOAA data and explanation](https://gml.noaa.gov/ccgg/trends/data.html).

Question: after observing through December 1997, how well can a small GP forecast the next 24 monthly means? To choose its kernel family, first simulate an earlier decision:

1. Fit on 1990–1995, 72 months.
2. Predict 1996–1997, 24 development months; choose the lower mean absolute error (MAE).
3. Refit the chosen family on 1990–1997, then evaluate 1998–1999 once.

MAE is Σ|actual − predicted| divided by the number of predictions. It has the same units as the observations. We also show root mean squared error (RMSE), which emphasizes larger errors, and count actual observations inside nominal pointwise 95% predictive intervals. The later evaluation lesson develops these metrics more broadly.

We compare an RBF-only model with a sum of linear, periodic, and RBF components. The latter can express continuing trend, an annual pattern, and smooth departures. Annual period is fixed at one year. Input is `(year − 1990) + (month − 0.5)/12`, an evenly spaced monthly coordinate. We subtract the training mean and add it back at prediction; future observations do not determine that mean.

The observation noise variance **0.09 ppm²** is a fixed instructional modeling assumption, corresponding to standard deviation 0.3 ppm. It is not NOAA's reported measurement uncertainty or a fitted scientific conclusion. The final test will help expose the limitations of this simple noise model and covariance family.

Save `mauna-loa-monthly.csv` beside the following `co2_gp.py` program. The lesson download contains these historical rows and provenance, so running it requires no data fetch. In the environment used for the small example, install `scikit-learn==1.9.1` and run `python co2_gp.py`.

```python
from pathlib import Path
import numpy as np
from sklearn.base import clone
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, ExpSineSquared

data = np.genfromtxt(
    Path(__file__).with_name("mauna-loa-monthly.csv"),
    delimiter=",", names=True,
)
x = ((data["year"] - 1990) + (data["month"] - 0.5) / 12)[:, None]
y = data["co2_ppm"]
training = np.arange(72)
development = np.arange(72, 96)
test = np.arange(96, 120)
noise_variance = 0.09
families = {
    "rbf": 4.0 * RBF(1.0, (0.2, 20.0)),
    "trend_periodic": (
        DotProduct(1.0, sigma_0_bounds="fixed")
        + 4.0 * ExpSineSquared(
            1.0, 1.0, length_scale_bounds=(0.2, 5.0),
            periodicity_bounds="fixed",
        )
        + RBF(3.0, (0.5, 20.0))
    ),
}

def fit(kernel, rows):
    center = y[rows].mean()
    model = GaussianProcessRegressor(
        kernel=clone(kernel), alpha=noise_variance,
        normalize_y=False, n_restarts_optimizer=1, random_state=23,
    )
    model.fit(x[rows], y[rows] - center)
    return model, center

def evaluate(model, center, rows):
    mean, latent_sd = model.predict(x[rows], return_std=True)
    mean = mean + center
    observation_sd = np.sqrt(latent_sd**2 + noise_variance)
    errors = y[rows] - mean
    mae = np.abs(errors).mean()
    rmse = np.sqrt(np.mean(errors**2))
    covered = np.count_nonzero(np.abs(errors) <= 1.959963984540054 * observation_sd)
    return float(mae), float(rmse), int(covered)

development_scores = {}
for name, kernel in families.items():
    model, center = fit(kernel, training)
    scores = evaluate(model, center, development)
    development_scores[name] = scores[0]
    print(name, "development", tuple(round(v, 6) for v in scores))

selected = min(development_scores, key=development_scores.get)
model, center = fit(families[selected], np.arange(96))
print("selected", selected)
print("test", tuple(round(v, 6) for v in evaluate(model, center, test)))
seasonal_naive = y[84 + np.arange(24) % 12]
print("seasonal_naive MAE", round(float(np.abs(y[test] - seasonal_naive).mean()), 6))
```

The kernel's amplitude and permitted length scales are optimized during fitting; `alpha` adds the fixed training-noise variance. Because these kernels contain no WhiteKernel, `return_std` describes the latent process in this setup. The explicit addition of 0.09 constructs the new-observation variance. A model with WhiteKernel has different prediction semantics, so adding its noise again would double count it. [GaussianProcessRegressor API](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html).

The retained author run used Python 3.12.14 and the package versions above:

| Evaluation | MAE (ppm) | RMSE (ppm) | Observations inside 95% predictive intervals |
| --- | ---: | ---: | ---: |
| RBF, development | 5.432943 | 5.884270 | 14/24 |
| Trend + periodic + RBF, development | 0.320601 | 0.416399 | 24/24 |
| Selected family refitted, final test | 1.233775 | 1.312967 | 13/24 |

The seasonal-naive baseline repeats each month of 1997 for both forecast years. It uses no 1998 observations to predict 1999 and gets test MAE **3.813333 ppm**. The selected GP predicts levels better in this experiment, but its test uncertainty is much less reliable than the development result suggested.

**Inline figure F5 — forecast and failure:** plot observed monthly values and the chosen forecast on a calendar axis. Mark the training/development/test boundaries and distinguish points outside the test predictive band. A paired residual panel shows the sign and size of errors. A narrow band can be visually attractive and still miss most of an important change.

The count 13/24 is an observed coverage diagnostic for this one correlated time period; it is not an independent-binomial estimate from 24 unrelated cases. Optimized hyperparameters, covariance misspecification, changing growth, and the simplified noise assumptions can all affect coverage. This experiment does not identify a unique cause. A useful next development study would test additional earlier forecast origins and inspect residual structure before choosing richer trend or noise assumptions. Do not tune on these test outcomes and continue calling the same period untouched test data.

**Investigation I2 — read the forecast before scoring it.** Compare the two development predictions on the actual month axis, record which will continue trend and seasonality, then reveal errors and bands. With a frozen choice, inspect the test forecast and predict whether “lower MAE” implies “well-calibrated intervals.” For an editable modeling task, choose the training cutoff and forecast horizon within the supplied historical period; make the role of each slice explicit. In this exploration, kernel parameters stay at the values learned on 1990–1995; conditioning and centering use the selected available prefix. Changing a cutoff creates a new experiment, not a correction to the reported fixed test or a fresh hyperparameter refit.

This example connects to [scikit-learn's longer CO₂ kernel-design walkthrough](https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_co2.html). Its locally periodic construction provides a useful extension; the experiment here uses a smaller, independently specified period and a held-out comparison.

## 7. Practice: calculate, diagnose, and transfer

### A. A weaker connection

Both latent variances are one, observation noise variance is one, the cross-covariance is 0.4, and the observed value is −3. Calculate the posterior mean, latent variance, and a future observation variance with the same noise. What changes if the cross-covariance is zero?

<details><summary>Hint</summary>
The observed variable's variance includes noise. Use that total in both denominators; add future noise only after calculating latent variance.
</details>
<details><summary>Solution</summary>
The observed variance is 2. Mean $0.4(-3)/2=-0.6$; latent variance $1-0.16/2=0.92$; observation variance $0.92+1=1.92$. Zero cross-covariance gives mean 0, latent variance 1, and observation variance 2. An observation can be extreme while teaching nothing about an independent target.
</details>

### B. A constant prior is a strong claim

Let $k(x,z)=1$ everywhere, with zero observation noise. Can this model accommodate two distinct values, 1 and −1, at different inputs? Would a small jitter fix the modeling issue?

<details><summary>Hint</summary>
Calculate the prior variance of $f(x)-f(z)$.
</details>
<details><summary>Solution</summary>
It is $1+1-2=0$, so the values must be equal almost surely. The conflicting observations have no support under the model, and the covariance matrix is singular. Added diagonal variance permits observation disagreement only by changing the noise assumptions. Numerical stabilization does not make a constant latent function capable of varying.
</details>

### C. Diagnose a suspicious improvement

A colleague changes observation values, leaves locations and all hyperparameters fixed, and reports much narrower latent bands. Name a precise check. Then explain when a changed width could be legitimate.

<details><summary>Hint</summary>
Find where $y$ appears in the conditional covariance formula.
</details>
<details><summary>Solution</summary>
There is no $y$ in that formula. Compare the two covariance arrays while holding kernel, noise, input preprocessing, and target grid fixed. In the supplied implementation they must match to numerical tolerance. If the fit also re-estimated kernel/noise parameters or target normalization, the model changed, and widths may legitimately differ. Log those settings before diagnosing the solver.
</details>

### D. A forecast review

An engineer says, “Our test MAE beats seasonal-naive, so the GP's 95% band is validated.” Write a short correction and one next study using the supplied experiment.

<details><summary>Hint</summary>
Point accuracy and interval performance measure different things. Preserve the used test's status.
</details>
<details><summary>Solution</summary>
“The GP improves test MAE from 3.813 to 1.234 ppm, but only 13 of 24 test observations are inside its nominal 95% intervals. That period does not support the interval claim.” An appropriate next study uses multiple earlier training cutoffs, forecasts a fixed horizon, and compares residual patterns and interval behavior for predeclared kernels/noise assumptions. Reserve a later, genuinely unused period for a subsequent final evaluation. Do not merely enlarge bands until this test count looks satisfactory.
</details>

### E. Your own kernel proposal

A sensor signal has a drifting baseline and a daily cycle whose shape slowly changes. Propose a covariance composition; explain how two distant readings at the same hour should relate.

<details><summary>Hint</summary>
The seasonal effect needs both recurrence and decay across days.
</details>
<details><summary>Solution</summary>
One defensible model is a long-scale RBF or explicit trend component plus periodic($p=1$ day) × RBF with a longer day-to-day decay scale, plus separately modeled observation noise. Same-hour readings remain strongly related nearby in time, but the periodic component's covariance fades over many days. Validate the decay scale using held-out future periods. Different compositions can be justified if their assumptions and evaluation protocol are explicit.
</details>

## 8. Deeper application: where should we measure next?

Read this branch after you can interpret the conditional covariance. Suppose the pipe experiment's main goal is reducing uncertainty at target $x_t=1$, and a new noisy reading costs the same at either candidate location. Use the **current posterior covariance** $c_D$, after accounting for existing data. A measurement at candidate $z$, with independent noise variance $\sigma_n^2$, reduces target variance by

\[
\Delta(z)=\frac{c_D(x_t,z)^2}{c_D(z,z)+\sigma_n^2}.
\]

This is the one-observation update again, now starting from the current posterior rather than the original prior. The candidate's own uncertainty is only part of the decision: it must also inform the target.

For the two-observation RBF example, candidates 1 and 4 reduce target variance by **0.305834** and **0.001888**, respectively. Location 4 is quite uncertain but weakly related to the target after existing observations. Measuring near the gap is much more useful for this particular goal. With zero target-candidate covariance, the reduction is exactly zero. With very noisy new observations, the reduction tends toward zero even if the locations are related.

**Inline figure F6 / investigation I3 — place the next probe:** target location stays visually distinct from editable candidate locations. Show covariance connections, not merely the widest uncertainty region. Record a candidate choice before showing variance reduction. Move the target and choose again: the best measurement is a decision relative to a goal.

An optimization goal is different. If you seek a small objective value, a common acquisition is expected improvement. For a noiseless incumbent $b$, predictive mean μ and standard deviation $s>0$, let $z=(b-\mu)/s$. Then

\[
\operatorname{EI}=(b-\mu)\Phi(z)+s\phi(z),
\]

where Φ and φ are the standard normal cumulative distribution and density. For $b=1$, candidate A with μ=0.8, $s=0.1$ has EI 0.200849; B with μ=1, $s=0.5$ has EI 0.199471. A slightly wins despite B's greater uncertainty. At $s=0$, use the continuous limit $\max(b-\mu,0)$. With noisy observations, the best observed value need not be a known latent incumbent; noisy acquisitions must account for that distinction. Expected improvement chooses experiments that might improve an objective, while target-variance reduction chooses experiments that clarify a target. Neither is a universal “pick the most uncertain” rule.

**Transfer check.** Suppose A instead has μ=1.2, $s=0.1$, while B has μ=1, $s=0.2$, with incumbent 1. Which has greater expected improvement?

<details><summary>Hint</summary>
A can improve only through the low tail of a distribution mostly above the incumbent. B is centered at the incumbent, so its first EI term is zero.
</details>
<details><summary>Solution</summary>
A has $z=-2$, EI approximately 0.000849. B has $z=0$, EI $0.2/\sqrt{2\pi}\approx0.079788$, so B wins. Uncertainty is useful when it creates a meaningful chance of improvement; its effect depends on the mean and objective too.
</details>

## 9. Deeper connections and extensions

### Classification changes the likelihood

A Gaussian latent value can drive a class probability through a sigmoid: $p(y_i=1\mid f_i)=1/(1+e^{-f_i})$. The likelihood is Bernoulli, so multiplying it by the Gaussian prior no longer produces an exact Gaussian posterior. A Laplace approximation finds the posterior mode $\hat f$ and uses local curvature to approximate its shape:

\[
q(f)=\mathcal N(\hat f,(K^{-1}+W)^{-1}),
\quad W_{ii}=\pi_i(1-\pi_i),\quad\pi_i=\operatorname{sigmoid}(\hat f_i).
\]

The prediction integrates the sigmoid over uncertain latent values. Generally $\mathbb E[\operatorname{sigmoid}(f_*)]\ne\operatorname{sigmoid}(\mathbb E[f_*])$. The latter discards uncertainty before converting to probability. A probability close to 0.5 can reflect ambiguous outcomes or uncertain latent values; one probability alone does not separate those causes. Expectation propagation and variational inference provide other approximations. The [scikit-learn GP guide](https://scikit-learn.org/stable/modules/gaussian_process.html) explains its Laplace classifier and contrasts its multiclass strategies with a direct joint multiclass likelihood.

### What makes exact regression expensive?

For dense $n\times n$ covariance, storage is $O(n^2)$, and one Cholesky factorization is $O(n^3)$. Hyperparameter fitting repeats expensive evaluations. After fitting, a single mean prediction takes $O(n)$ algebra beyond kernel evaluation; its variance requires a triangular solve taking $O(n^2)$. For many targets, batch solves reuse the factor. A full covariance among $q$ targets additionally needs $O(q^2)$ output storage and cross-target work. Prediction is not uniformly linear just because the mean is.

At $n=50{,}000$, one dense float64 matrix alone occupies $8n^2=20\times10^9$ bytes, about 20 GB decimal. Factorization workspaces and copies require more. There is no universal row count at which a GP becomes unusable: kernel structure, precision, hardware, repeated fits, and latency requirements matter.

Inducing-variable methods summarize the function through $m\ll n$ latent values $u=f(Z)$. The locations $Z$ need not be a subset of observed inputs. Under a Gaussian model, define $Q=K_{XZ}K_{ZZ}^{-1}K_{ZX}$. Titsias's variational regression bound is

\[
\log\mathcal N(y;0,Q+\sigma_n^2I)
-\frac{\operatorname{tr}(K_{XX}-Q)}{2\sigma_n^2}.
\]

The trace term penalizes latent variance the inducing representation leaves unexplained. It is an approximation objective with a reason for its correction, not a claim that selected points exactly replace all data. Dense inducing calculations commonly involve $O(nm^2+m^3)$ work. [Titsias, 2009, equation 9](https://proceedings.mlr.press/v5/titsias09a/titsias09a.pdf).

Stochastic variational methods keep a distribution $q(u)$ and optimize an evidence lower bound whose likelihood contribution is a sum over observations. Minibatches estimate that sum, while a KL term compares $q(u)$ with its prior. This gives a route to large datasets and non-Gaussian likelihoods; approximation quality still depends on the representation and optimization. [Hensman, Fusi and Lawrence, 2013](https://arxiv.org/abs/1309.6835).

Structured kernel interpolation instead approximates covariance using interpolation onto inducing locations with useful grid structure. Fast matrix-vector products can support iterative linear solves, with costs depending on that structure and convergence. [Wilson and Nickisch, 2015](https://proceedings.mlr.press/v37/wilson15.html). Neither technique makes every arbitrary kernel calculation exact and linear-time.

Random features approximate a kernel by a fixed feature inner product. Use the **same sampled feature map** for training and prediction. If a feature matrix has $n\times D$ entries, its storage is $O(nD)$; that does not make a dense ridge solve $O(nD)$. Forming its normal matrix costs $O(nD^2)$, with $O(D^3)$ factorization, before choices such as iterative optimization. A Gaussian prior on finite feature weights defines an approximate GP and can retain Bayesian uncertainty; using only point-estimated linear weights does not automatically do so.

### The GP–kernel ridge connection, with its boundary

Kernel ridge regression minimizes

\[
\frac1n\sum_i(y_i-f(x_i))^2+\lambda\|f\|_{\mathcal H_k}^2.
\]

Its fitted function is $k(x,X)(K+n\lambda I)^{-1}y$. For a zero-mean GP with the same fixed kernel and independent Gaussian noise, the posterior mean is identical when **$\sigma_n^2$ = nλ**. If the loss were a sum instead of an average, the matching factor would change. The deterministic regularization objective alone does not supply the GP's posterior intervals. [Kanagawa et al., 2018, Proposition 3.6](https://arxiv.org/pdf/1807.02582).

There is a subtle difference between a posterior mean and a sampled path. Brownian covariance $k(s,t)=\min(s,t)$ on [0,1] has an RKHS of absolutely continuous functions anchored at zero with square-integrable derivative. Brownian sample paths almost surely do not belong to that RKHS, even though its geometry defines the kernel. A fitted mean can be much more regular than a typical random path. This infinite-dimensional example should not be generalized to every finite-rank GP: the random-line GP has paths in its finite-dimensional span. The same [survey's §4.1](https://arxiv.org/pdf/1807.02582) explains why sample-path membership needs care.

### Functions can connect different kinds of observations

For a sufficiently differentiable kernel, differentiating it gives covariance between a function and a derivative:

\[
\operatorname{Cov}(f(x),f'(z))=\partial_zk(x,z),\quad
\operatorname{Cov}(f'(x),f'(z))=\partial_x\partial_zk(x,z).
\]

For RBF amplitude one, $\operatorname{Var}(f'(x))=1/\ell^2$, with units of output²/input². Slope measurements from a simulator can therefore enter the same joint conditioning system as value measurements. The kernel must support the derivatives being observed; a rough prior cannot be differentiated merely because the program can differentiate its formula away from the diagonal.

A related extension gives a kernel two output indices, $k((x,a),(z,b))$, to express covariance between different sensors or tasks. Separately fitting one GP per output assumes away those cross-output connections. These extensions are valuable when there is a defensible relationship between measurements, and require checking the resulting joint covariance rather than treating every input column as interchangeable.

**Connection check.** An average-loss KRR fit has 40 examples and λ=0.025. What noise variance matches its mean under the fixed-kernel, zero-mean GP assumptions? What additional claim would be unjustified from the KRR objective alone?

<details><summary>Hint</summary>
Keep the factor of $n$ from the average loss.
</details>
<details><summary>Solution</summary>
The matching variance is $40(0.025)=1$, not 0.025. Claiming posterior credible intervals from the deterministic KRR objective alone is unjustified: those intervals need probabilistic assumptions, including the GP prior and observation model.
</details>

## 10. References & another way to learn it

- [Rasmussen and Williams, *Gaussian Processes for Machine Learning*](https://gaussianprocess.org/gpml/chapters/RW.pdf): the free canonical textbook. Chapter 2 develops regression from weights and functions; chapter 4 explains covariance choices; chapter 5 treats model selection. Chapters 3 and 8 extend the core to classification and approximations. Use these after the numerical conditioning example.
- [Görtler, Kehlbeck and Deussen, *A Visual Exploration of Gaussian Processes*](https://distill.pub/2019/visual-exploration-gaussian-processes/): an interactive article for seeing joint Gaussians, conditioning, and function samples. Its geometric view is especially useful if matrix notation feels disconnected from the picture. Keep variance and standard deviation distinct when translating a covariance diagonal into a plotted width.
- [scikit-learn, Gaussian processes guide](https://scikit-learn.org/stable/modules/gaussian_process.html): practical reference for regression, classification, kernels, and API assumptions. Consult it when choosing how to represent known noise or reading prediction output.
- [scikit-learn, CO₂ forecasting example](https://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gpr_co2.html): a longer worked kernel-composition example. Compare its locally periodic structure with the simpler fixed experiment here; its reported result is a different experiment.
- [Rasmussen and Ghahramani, Cambridge lectures 3–4](https://mlg.eng.cam.ac.uk/teaching/4f13/1213/lect0304.pdf): compact lecture notes on the move from Bayesian linear models to GP regression. Suitable after section 3 as another mathematical route.
- [Kanagawa et al., *Gaussian Processes and Kernel Methods: A Review on Connections and Equivalences*](https://arxiv.org/pdf/1807.02582): advanced reading for the exact KRR correspondence and distinctions between RKHS functions and sample paths.
- [Titsias, variational inducing variables](https://proceedings.mlr.press/v5/titsias09a.html), [Hensman et al., stochastic variational GPs](https://arxiv.org/abs/1309.6835), and [Wilson and Nickisch, structured kernel interpolation](https://proceedings.mlr.press/v37/wilson15.html): three different mechanisms for scaling inference. Read the mechanism you need rather than treating their complexity statements as interchangeable.

The next topic in this module is **Semi-Supervised Learning**. Here unlabeled locations acquired predictions through a covariance model and observed numerical values. Next, unlabeled examples help classification through assumptions about input geometry, class structure, or agreeing views. A large unlabeled collection is useful only when those assumptions connect its structure to the labels we need.
