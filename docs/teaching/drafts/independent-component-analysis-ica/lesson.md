# Independent Component Analysis (ICA)

Two sensors can both contain the same two signals in different proportions. A loud pulse in one recording might come from the event you care about, an interfering source, or both. Instead of keeping the direction with the largest variation, can we find combinations that separate the contributions?

**Independent component analysis** estimates a linear representation whose component signals are as statistically independent as its model and estimation method can make them. We will first separate an exact four-state mixture. Then we will ask a narrower, measurable question of a real electrical recording: does an ICA component track a simultaneously recorded reference more closely than an original channel or a principal component?

**First-pass route.** Read sections 1–5, using the mixing figure and the rotation investigation as you go. Run the short NumPy example in section 5, then read and run the real-data comparison in section 6. Try practice 1, 2 and 4 in section 9 before the readiness check. Sections 7 and 8 are deeper branches for component removal, objectives, computation and extensions. Expect about 45–55 minutes of reading on the first route, plus 45–75 minutes for code and practice; the deeper branches add about 25 minutes.

You need dot products, matrix multiplication, an average and variance. [PCA & Dimensionality Reduction](/learn/path/full-curriculum/pca-dimensionality-reduction?module=classical-ml) supplies the geometry of projections, eigenvectors and reconstruction. [Probability Distributions & Bayes’ Theorem](/learn/path/full-curriculum/probability-distributions-bayes-theorem?module=math-foundations) reviews independence and moments. We will introduce the specific probability and optimization ideas locally. Python examples need NumPy and scikit-learn; the supplied CSV lets the real example run offline.

The previous topic, [t-SNE, UMAP & Manifold Learning](/learn/path/full-curriculum/t-sne-umap-manifold-learning?module=classical-ml), asks how to display relationships among observations in a small number of coordinates. ICA asks a different question about the observations’ **generating mixture**. Neither a separated-looking embedding nor uncorrelated PCA coordinates supplies that generating model.

## 1. Follow one sample through a mixture

Suppose source values at one instant are $s_1=1$ and $s_2=-1$. Sensor 1 receives twice the first source plus the second; sensor 2 receives the first plus twice the second:

$$
x_1=2s_1+s_2=1,\qquad x_2=s_1+2s_2=-1.
$$

The compact notation is

$$
x=As,\qquad
A=\begin{bmatrix}2&1\\1&2\end{bmatrix}.
$$

Here $s$ is the column of two source amplitudes, $x$ is the column of two observed amplitudes, and $A$ is the **mixing matrix**. Its column $a_j$ tells how source $j$ contributes across sensors. Its row $i$ gives sensor $i$’s recipe. The coefficients have units of observed amplitude per source amplitude; our hand example uses arbitrary units.

**Inline figure M1 — two sources, weighted contributions, two sums.** Show the actual $2,-1,1,-2$ contributions, with each arrow labeled by its coefficient, next to the corresponding entries of $A$. A source column is a pattern across sensors, not an unmixing direction.

If we knew $A$, ordinary algebra would solve the problem:

$$
A^{-1}=\frac13\begin{bmatrix}2&-1\\-1&2\end{bmatrix},
\qquad s_1=(2x_1-x_2)/3,\quad s_2=(-x_1+2x_2)/3.
$$

Substituting $x=(1,-1)^T$ recovers $(1,-1)^T$. The difficult part of **blind source separation** is estimating the recipes when only many observed $x$’s are available. “Blind” describes that missing mixing information; assumptions still do substantial work.

For a dataset, each row is one simultaneous observation and each column is a sensor. Thus $X$ has shape $n\times d$, $S$ has shape $n\times k$, and $A$ has shape $d\times k$. With row storage, the same relation is $X=SA^T$. An unmixing operator $B$ with shape $k\times d$ produces $\hat S=(X-\mathbf1\mu^T)B^T$, where $\mu$ is the fitted vector of sensor means. We reserve $W$ below for the rotation **after whitening**, so $B=WK$ when $K$ is the whitener.

### The model’s conditions belong here

Our exact model is a **noiseless, instantaneous, constant linear mixture**: the sensor reading now depends on source values now through one fixed matrix. The basic identifiable case has mutually independent, nondegenerate sources, at most one Gaussian source, and a square invertible mixing matrix. With more sensors than sources, a full-column-rank mixing model can first be represented in its source-dimensional signal subspace. More sources than sensors requires additional structure and a different separation method. The moment calculations in this lesson assume finite variances, and kurtosis additionally needs finite fourth moments.

Independence here is between the source variables at the same observation. A signal may still resemble its own recent past. Ordinary FastICA uses the distribution of simultaneous samples rather than an explicit temporal model. Autocorrelation changes how much independent information a recording supplies, so 20,000 time samples need not provide the information of 20,000 independent draws.

A real room introduces propagation delays and echoes, so the cocktail-party story is a useful motivation for this simplified model. Biological recordings also contain measurement noise and sources that may share activity. We will use the model as a tool and evaluate its result, with the model conditions available here whenever needed. The foundational tutorial introduces this same distinction between the ideal mixture and its applications. [Hyvärinen & Oja, 2000, sections 1–2](https://www.cs.helsinki.fi/u/ahyvarin/papers/NN00new.pdf).

## 2. Zero correlation can hide complete dependence

The expectation $E[u]$ is a probability-weighted average. Covariance measures whether two centered variables tend to have products of the same sign:

$$
\operatorname{Cov}(u,v)=E[(u-Eu)(v-Ev)].
$$

Zero covariance is one equality about an average. **Independence** is stronger: for every pair of events about the variables, the probability of both equals the product of the individual probabilities. For densities, this becomes $p(u,v)=p(u)p(v)$. Independence implies zero covariance when the required moments exist.

For an exact counterexample, let $u$ take $-1,0,1$, each with probability $1/3$, and let $v=u^2$. Then $Eu=0$, $Ev=2/3$, and $E[uv]=E[u^3]=0$, so covariance is zero. Nevertheless, observing $v=0$ tells us $u=0$ exactly. The joint probability of $u=0,v=0$ is $1/3$, whereas the product of the marginals is $1/9$.

**Inline figure D1 — conditional support.** Three labeled points lie on $v=u^2$. Beside them, reveal how selecting $v=0$ reduces the possible $u$’s to one. The covariance cancellation and the conditional restriction are different visible facts.

PCA finds orthogonal directions that diagonalize covariance. Whitening also rescales them to unit variance. Neither operation tests all these joint probabilities. For a **jointly Gaussian** vector, zero cross-covariances do imply independence. Having separately Gaussian-looking histograms is a weaker observation than establishing a joint Gaussian model.

### An exact source distribution we can carry through every step

Let $s_1$ and $s_2$ be independent fair choices from $\{-1,1\}$. There are four equally likely source states:

| Source state $(s_1,s_2)$ | Mixed observation $(x_1,x_2)$ |
| --- | --- |
| $(-1,-1)$ | $(-3,-3)$ |
| $(-1,1)$ | $(-1,1)$ |
| $(1,-1)$ | $(1,-1)$ |
| $(1,1)$ | $(3,3)$ |

Every marginal sign has probability $1/2$; every pair has probability $1/4$. These four rows enumerate a designed probability distribution. They are not a claim that four arbitrary measurements suffice to fit useful real-world ICA.

The source means are zero and $E[ss^T]=I$, the identity matrix. The mixed covariance is

$$
\Sigma_x=AIA^T=\begin{bmatrix}5&4\\4&5\end{bmatrix}.
$$

For example, sensor 1 has average squared value $(9+1+1+9)/4=5$, and the average product of sensor readings is $(9-1-1+9)/4=4$.

## 3. Whitening removes a stretch, leaving a separation question

The covariance has eigenvectors $v_+=(1,1)^T/\sqrt2$ and $v_-=(1,-1)^T/\sqrt2$, with eigenvalues 9 and 1. Multiplying $\Sigma_xv_+=9v_+$ verifies the first pair. PCA projects onto these directions. It obtains scores

$$
p_+=(x_1+x_2)/\sqrt2,\qquad p_-=(x_1-x_2)/\sqrt2.
$$

Their variances are 9 and 1. **Whitening** divides each score by its standard deviation:

$$
z_1=\frac{x_1+x_2}{3\sqrt2}=\frac{s_1+s_2}{\sqrt2},\qquad
z_2=\frac{x_1-x_2}{\sqrt2}=\frac{s_1-s_2}{\sqrt2}.
$$

Now $E[zz^T]=I$. Yet $z_1=0$ forces $z_2$ to be $\pm\sqrt2$, never zero. Both individual zero events have probability $1/2$, but their intersection has probability zero. The whitened coordinates are still dependent mixtures.

**Inline figure W1 — four corresponding clouds.** Source square → observed elongated cloud → PCA-whitened diamond → recovered source square. Preserve the identity of each of the four states with labels, not just colors. Print covariance beneath the observed and whitened panels, and show the missing joint zero event beneath the diamond. The source and recovered square are the same distribution, reached once from the constructed inputs and once from the unmixing operation.

A final linear combination recovers the sources:

$$
\begin{bmatrix}s_1\\s_2\end{bmatrix}
=\frac1{\sqrt2}\begin{bmatrix}1&1\\1&-1\end{bmatrix}
\begin{bmatrix}z_1\\z_2\end{bmatrix}.
$$

This matrix is orthogonal: its rows have length one and dot product zero. It includes a reflection; when we speak informally of the remaining “rotation,” the allowed orthogonal transformations include reflections and sign changes.

In general, if $\Sigma_x=VDV^T$ with positive eigenvalues, the whitener is $K=D^{-1/2}V^T$. Then $z=K(x-\mu)$ has identity covariance. With unit-variance independent sources in the square noiseless model, $Q=KA$ obeys $QQ^T=I$. That calculation explains why searching over orthogonal $W$’s after whitening is sufficient. It reduces the unknown scaling and shearing before the independence search.

If an eigenvalue is zero, dividing by its square root is impossible; a constant or redundant channel provides no new direction. Very small eigenvalues can amplify noise. Estimate effective rank and choose a defensible subspace. Reducing to $k<d$ principal coordinates before ICA chooses a **variance-based subspace**; it does not select the $k$ most independent or most non-Gaussian physical sources.

## 4. Why non-Gaussianity gives a direction

For a centered variable with nonzero variance, its **excess kurtosis** is

$$
\kappa(y)=\frac{E[y^4]}{E[y^2]^2}-3.
$$

The subtraction gives a Gaussian value of zero. Our unit-variance binary source has $E[s^4]=1$, hence $\kappa=-2$. A unit-variance Laplace source has excess kurtosis 3. Positive and negative departures can both supply useful information.

Take independent, centered unit-variance sources and a unit-length combination $y=a s_1+b s_2$, with $a^2+b^2=1$. Expanding the fourth power gives

$$
E[y^4]=a^4E[s_1^4]+6a^2b^2+b^4E[s_2^4].
$$

The odd cross-terms vanish because each contains a zero source mean. Subtract $3(a^2+b^2)^2$, and the result is

$$
\kappa(y)=a^4\kappa(s_1)+b^4\kappa(s_2).
$$

For two binary sources with $a=b=1/\sqrt2$, this is $-1$, compared with $-2$ for either unmixed source. For two Laplace sources it is $1.5$, compared with 3. **Equal source kurtoses do not destroy separation:** the fourth powers change with the direction. At angle $\theta$, two equal Laplace sources give $3(\cos^4\theta+\sin^4\theta)$, with maxima at source axes rather than a ring of equal maxima.

The central limit theorem offers the intuition that many independent contributions can make a standardized sum more Gaussian. The fourth-moment identity is the exact reason in our example. The general theorem is a limiting statement under conditions; it does not say every mixture of two arbitrary distributions improves every measure of Gaussianity.

### The Gaussian ambiguity is a population fact

If $s_1,s_2$ are independent standard Gaussians, their joint density is proportional to $\exp[-(s_1^2+s_2^2)/2]$. An orthogonal transformation preserves that sum of squares. The transformed pair has the same joint Gaussian distribution and independent coordinates. Observations alone cannot tell which of these bases was the original source basis.

With two or more Gaussian sources, their Gaussian subspace has this unresolved rotation. At most one Gaussian source is allowed in the usual identifiable ICA model; other non-Gaussian sources can then determine the remaining direction. This is about what the probability model identifies, not whether a numerical solver happens to return an array. Finite Gaussian samples can have accidental fourth-moment structure, and a solver can follow it.

Kurtosis is only one diagnostic. A non-Gaussian variable taking $0$ with probability $2/3$, and $\pm\sqrt3$ with probability $1/6$ each, has variance 1 and fourth moment 3: its excess kurtosis is also zero. Its sixth moment is 9, whereas a standard Gaussian’s is 15. Therefore a threshold such as “all kurtoses close to zero” cannot establish Gaussianity or decide ICA suitability by itself.

### Investigation: rotate a distribution, not just its covariance

In the rotation investigation, record whether your proposed new projection will have greater, equal or smaller absolute excess kurtosis than the active one. Enter an angle of your own before revealing the result. The covariance remains $I$ under every orthogonal rotation, while the joint support and fourth moment can change.

Start with the binary source distribution above. Try a projection halfway between its source directions, then a direction near one source. Return with the Gaussian population selected. Explain what the Gaussian null case removes from the search. This activity’s exact statistic, source-family controls and prediction rules are specified in [the visual packet](visual-specifications.md#r1-rotation-investigation).

### A broader objective

For a continuous variable, differential entropy is $H(y)=-\int p(y)\log p(y)\,dy$: an average log-density measure, not a histogram bar’s height. Among distributions with a fixed variance, the Gaussian has maximum entropy. **Negentropy** measures the gap $J(y)=H(y_{\mathrm G})-H(y)$, where the Gaussian has the same variance.

Estimating a full density can be difficult. FastICA commonly uses a nonquadratic contrast related to an approximation $J(y)\propto[E G(y)-E G(\nu)]^2$, with $\nu\sim N(0,1)$. This is a surrogate for searching, not an exact measured mutual information. Choices include $G(u)=\log\cosh(u)$, $G(u)=-e^{-u^2/2}$, and $G(u)=u^4/4$. The cube choice makes a particularly transparent calculation; fourth powers are also sensitive to unusually large observations. The log-cosh derivative grows more gently. No one nonlinearity is best for every source distribution.

The entropy argument in this paragraph concerns continuous densities. The binary hand example uses exact probabilities and kurtosis; it is not being assigned a differential entropy. Section 8 connects the continuous objective to independence and likelihood.

## 5. Follow a FastICA update

For whitened observations $z_i\in\mathbb R^k$, a unit vector $w$ produces a component $y_i=w^Tz_i$. The unit-length constraint keeps its variance at one, so a larger objective must come from changing the distribution rather than simply magnifying every amplitude.

Let $g=G'$. A one-component FastICA iteration computes

$$
r=\frac1n\sum_i z_i g(w^Tz_i)
-\left[\frac1n\sum_i g'(w^Tz_i)\right]w,
\qquad w_{\mathrm{new}}=r/\|r\|.
$$

The first average weights each observation by a nonlinear function of its current projection. The second term corrects the current direction; normalization restores the constraint. In the log-cosh version, $g(u)=\tanh u$ and $g'(u)=1-\tanh^2u$.

For a hand trace, use the four whitened diamond points from section 3 and $w=(0.8,0.6)$. With $g(u)=u^3$, the projections are the signed values $0.8\sqrt2$ and $0.6\sqrt2$. The first average is $(2(0.8)^3,2(0.6)^3)=(1.024,0.432)$. The average derivative is $E[3y^2]=3$. Thus

$$
r=(1.024,0.432)-3(0.8,0.6)=(-1.376,-1.368).
$$

After normalization and an optional sign flip for easier comparison, the next vector is approximately $(0.7091653,0.7050423)$. It moved toward $(1,1)/\sqrt2$, which extracts $s_1$. This is the same final combination we found by algebra in section 3, now approached by a distribution-based update.

**Inline figure F1 — projection, nonlinear weighting, correction, new direction.** Pair the four projection values with the two averaged-vector terms, then show old and new directions on a unit circle. The equation is a sequence of operations on the same four observations.

### Why this update has that form

At a stationary point of $E[G(w^Tz)]$ constrained by $\|w\|^2=1$, the objective gradient must align with $w$: $E[z g(w^Tz)]-\beta w=0$. The multiplier $\beta$ accounts for the constraint; multiplying by $w^T$ gives $\beta=E[y g(y)]$ at a stationary point.

Newton’s method solves an equation using its derivative. The derivative matrix here contains $E[zz^Tg'(w^Tz)]-\beta I$. FastICA makes the approximation $E[zz^Tg']\approx E[g']I$ in whitened coordinates. Simplifying the resulting Newton-like step and discarding a scalar that normalization will remove gives the update above. Whitening makes $E[zz^T]=I$; it does **not** by itself make that factorization exact. The method searches for a fixed direction, with local convergence depending on the distribution, contrast and initialization. [Hyvärinen & Oja, section 6, equations 41–44](https://www.cs.helsinki.fi/u/ahyvarin/papers/NN00new.pdf).

For several components, unconstrained repetitions could rediscover the same direction. **Deflation** estimates them one at a time. After **each update**, subtract its projection on every previously found unit direction, then normalize. If a previous direction is $q$, subtract $(q^Tr)q$. Orthogonality enforces distinct uncorrelated coordinates in whitened space; the nonlinear objective still supplies the separation criterion.

**Parallel or symmetric FastICA** updates all rows of $W$ and orthogonalizes them together, using $W\leftarrow(WW^T)^{-1/2}W$ when that inverse square root exists. A sign-aware stopping test is $1-|w_{\mathrm{new}}^Tw|<\text{tolerance}$. Parallel and antiparallel vectors describe the same source direction. A small directional change records numerical convergence, which is distinct from a successful application diagnostic.

### Complete NumPy example

Use a Python environment with NumPy and scikit-learn. The author’s calculation snapshot used Python 3.12.14 and these package versions; install them once if they are not already available:

```sh
python -m pip install numpy==2.3.5 scipy==1.18.1 scikit-learn==1.9.1
```

Save the following program as `ica_by_hand.py` and run `python ica_by_hand.py`. This program uses the full four-state distribution, not sampled sine waves claimed to be independent. It implements deflation with the orthogonalization inside the iteration. The finite, full-rank fixture is supplied; a zero update or a rank-deficient input needs diagnosis before normalization in a general-purpose implementation.

The covariance average divides by four because the rows enumerate four equiprobable states. NumPy’s `eigh` returns eigenvalues in ascending order; section 3 deliberately listed the larger one first. The code’s whitened axes can therefore differ in order and sign from the figure while representing the same information.

```python
import numpy as np

S = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]])
A = np.array([[2., 1.], [1., 2.]])
X = S @ A.T
mean = X.mean(axis=0)
centered = X - mean
values, vectors = np.linalg.eigh(centered.T @ centered / len(X))
K = (vectors / np.sqrt(values)).T
Z = centered @ K.T

def separate(Z, seed=12, max_iter=200, tol=1e-10):
    rng = np.random.default_rng(seed)
    W = np.zeros((Z.shape[1], Z.shape[1]))
    for j in range(len(W)):
        w = rng.normal(size=Z.shape[1])
        w -= W[:j].T @ (W[:j] @ w)
        w /= np.linalg.norm(w)
        for _ in range(max_iter):
            y = Z @ w
            r = Z.T @ (y**3) / len(Z) - (3*y*y).mean() * w
            r -= W[:j].T @ (W[:j] @ r)
            new = r / np.linalg.norm(r)
            change = 1 - abs(new @ w)
            w = new
            if change < tol:
                break
        else:
            raise RuntimeError("Iteration limit reached")
        W[j] = w
    return W

W = separate(Z)
estimated = Z @ W.T
B = W @ K
C = np.corrcoef(S.T, estimated.T)[:2, 2:]
reconstructed = estimated @ np.linalg.inv(B).T + mean
print(np.round(Z.T @ Z / len(Z), 6))
print(np.round(np.max(np.abs(C), axis=1), 6))
print(np.max(np.abs(reconstructed - X)) < 1e-10)
```

Expected rounded result for this fixture:

```text
[[1. 0.]
 [0. 1.]]
[1. 1.]
True
```

The output checks whitening, source agreement and reconstruction separately. Here the two high correlations must match different recovered columns; inspect $C$ to confirm the one-to-one correspondence. In a general $k$-source benchmark, match components by a one-to-one assignment maximizing total absolute correlation, then resolve signs and scales. Taking each row’s best match independently can reuse one recovered component and exaggerate recovery.

The exact hand calculation and a small author probe support these expected results. They are not a completed production/runtime verification campaign; the implementation packet identifies the later execution checks.

## 6. A real recording: fit blindly, choose with development data, evaluate later

An abdominal electrode records overlapping electrical activity. In the **Abdominal and Direct Fetal ECG Database**, researchers recorded four abdominal channels and a simultaneous direct fetal ECG reference to study fetal-heartbeat measurement from abdominal signals. The supplied extract is the first 20 seconds of record `r01`, version 1.0.0, sampled at 1,000 Hz. A row is one instant, not one person. The four abdominal values are inputs; the direct channel is an external comparison signal. [Dataset description and acquisition details](https://physionet.org/content/adfecgdb/1.0.0/).

The immediate question is deliberately measurable without specialist physiology: **which representation contains a single coordinate with stronger absolute linear correlation to that reference over a later four-second interval?** Correlation measures aligned waveform variation. It is not a fetal-beat detector, a clinical accuracy measure, or a count of recovered physiological sources. A reference measured at a different location can differ in waveform and polarity. This is a short within-recording investigation; evaluating a clinical or cross-person claim would require a different task, endpoints and independent participants.

Download [the provided CSV](r01-first20s.csv) and keep it beside `ica_recording.py`. Save the program below under that name, then run `python ica_recording.py` from their directory. After the one-time package installation, the program needs no network access. [Data provenance](data-provenance.md) gives the original authors, ODC-By 1.0 license, source and extract hashes, exact columns and conversion. The stored integers are ADC counts. The program converts them to microvolts using the EDF header’s calibration. The provider already performed acquisition/filtering steps; the lesson adds no filter or resampling.

**Inline figure P1 — a chronological evidence boundary.** The first 12 seconds feed the PCA/ICA fits. Seconds 12–16 select one coordinate within each representation using reference correlation. Seconds 16–20 evaluate those already selected coordinates. The direct reference has no arrow into either fitted decomposition. It does have an arrow into development selection: the overall selection procedure therefore uses reference information even though ICA fitting is unsupervised.

Before running, predict which of raw channels, PCA coordinates or ICA coordinates will give the largest held-out absolute correlation. The comparison uses all four coordinates for both decompositions. It chooses the coordinate within each method only on development data, then freezes the choice. The record, interval, split and ICA settings were fixed before observing the comparison; we keep an inconvenient outcome rather than search for a better seed or time window.

```python
import numpy as np
from sklearn.decomposition import FastICA, PCA

digital = np.loadtxt("r01-first20s.csv", delimiter=",", skiprows=1)
microvolts = (digital + 32768) * (6553.6 / 65535) - 3276.8
reference = microvolts[:, 0]
X = microvolts[:, 1:]
train = slice(0, 12000)
development = slice(12000, 16000)
test = slice(16000, 20000)

def correlation_with_reference(values, target):
    joined = np.column_stack([values, target])
    return np.corrcoef(joined, rowvar=False)[-1, :-1]

pca = PCA(n_components=4, svd_solver="full").fit(X[train])
ica = FastICA(n_components=4, whiten="unit-variance",
              whiten_solver="svd", algorithm="parallel", fun="logcosh",
              random_state=7, max_iter=1000, tol=1e-5).fit(X[train])
representations = [("channel", X), ("PCA", pca.transform(X)),
                   ("ICA", ica.transform(X))]
for name, values in representations:
    dev = correlation_with_reference(values[development], reference[development])
    chosen = int(np.argmax(np.abs(dev)))
    result = correlation_with_reference(values[test], reference[test])[chosen]
    print(name, chosen + 1, f"{abs(dev[chosen]):.6f}", f"{abs(result):.6f}")
print("ICA iterations", ica.n_iter_)
```

Author probe result with Python 3.12.14, NumPy 2.3.5 and scikit-learn 1.9.1; columns are representation, chosen one-based coordinate, development $|r|$, test $|r|$:

```text
channel 3 0.201203 0.119806
PCA 4 0.184580 0.450178
ICA 2 0.169804 0.343966
ICA iterations 14
```

PCA coordinate 4 has the largest test value in this fixed comparison. ICA coordinate 2 improves on the selected raw channel, but its independence-oriented objective does not optimize this reference-correlation diagnostic. The change from development to test also makes the short interval’s variability visible. The practical conclusion is to retain the comparison and investigate the signal/task relationship before preferring an ICA pipeline. The lower-variance PCA coordinate mattered here; automatically discarding it would have removed that candidate.

**Inline figure E1 — inspect the outcome.** Display the three test correlations on a common 0–1 axis including zero, and aligned 16–18 second traces of the reference and selected coordinates. Use explicit display normalization and the same time axis. Keep the numeric table above available. The goal is to connect correlation with waveform agreement, not to label every peak.

The library handles fitted centering and whitening. `ica.components_` is $B=WK$, shape $4\times4$; `mixing_` is its pseudoinverse, shape $4\times4$; `mean_` has four sensor means. `transform(X_new)` uses those fitted quantities. Calling `fit_transform` on a new interval would estimate a new decomposition and invalidate a comparison that assumes fixed component identities. `whiten='unit-variance'` normalizes fitted source variances; `whiten=False` expects an already whitened input. Explicit settings avoid relying on an older default. [FastICA API](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html).

A full-rank `inverse_transform(transform(X))` reconstructs sensor values to numerical precision even when the components are unhelpful. Any invertible change of coordinates can do that. Reconstruction checks algebra and information retention; the reference comparison asks about this application. If you reduce the component count, reconstruction instead returns the retained subspace contribution plus the mean.

To study stability, compare several predeclared seeds and non-overlapping recording blocks, align component sign/permutation before comparing them, and report all runs. Use development data for any method or component-count choice. A final test interval stays untouched until those choices are frozen. Randomly mixing neighboring samples across folds would make the rows seem more independent than the recording permits. The provider’s offline filtering also means this extract cannot establish a real-time pipeline’s performance.

## 7. Deeper branch: what is a component, and what happens if we remove it?

**Return here after the first-pass example.** An ICA representation has useful ambiguities even in its ideal identifiable setting. Since

$$
x=\sum_j a_js_j,
$$

multiplying $s_j$ by any nonzero $c$, and dividing $a_j$ by $c$, leaves every observation unchanged. Negative $c$ includes a sign flip. Reordering sources and the corresponding columns also preserves the sum. ICA therefore identifies sources up to scale, sign and permutation. A unit-variance convention fixes scale in a useful way, but component index and polarity are still conventions. The model supplies no default ranking by explained variance.

For sensor reconstruction, the **mixing column** $a_j$ tells where a component contributes. The **unmixing row** $b_j^T$ tells which sensor combination estimates it. These are dual operations, not equal vectors: for our $A$, mixing column 1 is $(2,1)^T$, while inverse row 1 is $(2,-1)/3$. Plotting an inverse row as if it were a spatial contribution pattern reverses the meaning.

In the four-state example, each unit-variance source contributes $\lVert a_j\rVert^2=5$ to the sum of sensor variances. More generally the contribution is $\operatorname{Var}(s_j)\lVert a_j\rVert^2$, provided the sources are uncorrelated and amplitudes use compatible sensor units. Doubling a source and halving its column leaves that product unchanged. The column norm alone is not a universal energy measure across normalization conventions.

Suppose a domain investigation identifies component 2 as an unwanted contribution. Removing it means setting its scores to zero and reconstructing:

$$
x_{\rm kept}=x-a_2s_2.
$$

At the worked instant $s=(1,-1)^T$, observed $x=(1,-1)^T$. Component 2 contributes $(-1,-2)^T$; subtracting that contribution leaves $(2,1)^T$, the first source’s contribution. The altered result is not expected to equal the original sensors.

**Investigation C1 — edit a contribution.** Enter a new two-source amplitude pair and choose which component to keep. Record a predicted sensor amplitude before reconstructing. Observe the contribution vectors, their sum, and the removed difference. Then rescale a source and inversely rescale its mixing column: the reconstructed observations should stay fixed. See [the full contract](visual-specifications.md#c1-component-contribution-investigation).

In EEG/MEG practice, a component’s time course, spatial pattern and relationship to an auxiliary eye or cardiac channel can help identify an artifact candidate. A statistical component label such as “blink-like” is an interpretation based on this evidence. It is not an anatomical source location or guaranteed neurophysiological cause. Removing a component also removes any wanted activity it contains, which is why before/after task-signal checks and sensitivity to exclusion choices matter. [MNE’s artifact tutorial](https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html) shows this inspect–exclude–reconstruct workflow. Detailed filtering, referencing, rank changes and experimental leakage belong to the planned [Neural Preprocessing, Artifact Rejection and Leakage-Safe Pipelines](/learn/path/full-curriculum/neural-preprocessing-artifact-rejection-and-leakage-safe-pipelines?module=computational-neuroscience) lesson.

## 8. Deeper branch: objectives, extensions and computation

**This branch connects the mechanism to the wider ICA literature.** Read it when you want to distinguish a model, an estimation objective and an algorithm for that objective.

### Independence, entropy and likelihood

For continuous components with suitable finite entropies, total dependence can be expressed as

$$
I(y_1,\ldots,y_k)=\sum_jH(y_j)-H(y).
$$

This is the KL divergence between the joint density and the product of its marginals, so it is nonnegative and zero exactly at mutual independence. It is commonly called total correlation or multi-information when there are more than two components. ICA seeks to **minimize** it.

For a whitened vector $z$ and square orthogonal $W$, $y=Wz$ has unit marginal variances, and $H(y)=H(z)+\log|\det W|=H(z)$. The Gaussian entropy reference is fixed too. Therefore

$$
\sum_j J(y_j)=\text{constant}-\sum_j H(y_j),
$$

so maximizing the sum of marginal negentropies is equivalent to minimizing total dependence under these conditions. This is the formal connection behind section 4. Maximizing the sum of marginal entropies would point in the opposite direction when joint entropy is fixed.

A likelihood formulation starts by choosing source densities $p_j$. For square invertible $B$, a change of variables gives

$$
p_x(x)=|\det B|\prod_j p_j(b_j^T(x-\mu)).
$$

For independent observation vectors, the dataset log likelihood is $n\log|\det B|+\sum_{i,j}\log p_j(b_j^T(x_i-\mu))$. For temporally dependent recordings, that sum is a marginal fitting contrast rather than the full time-series joint likelihood. The determinant accounts for how a linear transformation changes volume. Without it, changing scale could appear beneficial for the wrong reason. Incorrect source-density choices can also change the estimator.

Infomax connects an appropriately chosen nonlinear output transformation to entropy maximization and this likelihood perspective. It is an alternative estimation route, not a claim that deterministic input–output mutual information equals dependence among recovered coordinates. FastICA is a fixed-point algorithm tied to specified contrasts. Picard is another optimizer using preconditioning and an approximate Hessian; its published comparisons concern stated objectives and datasets, not a universal speed or stability ranking. The lineage from early adaptive separation, Comon’s ICA formulation, Infomax and fixed-point methods helps explain why several algorithms share the ICA name. [Canonical book, chapters 7–14](https://www.cs.helsinki.fi/u/ahyvarin/papers/bookfinal_ICA.pdf), [Picard paper](https://arxiv.org/abs/1706.08171).

### When another model is needed

| Situation | What changes in the reasoning? |
| --- | --- |
| Additive sensor noise $x=As+\epsilon$ | Whitened covariance includes noise; sample directions can be biased or noise-amplified. A noisy latent-variable model can make that assumption explicit. Discarding low-variance coordinates may also discard a weak wanted source. |
| Several Gaussian sources with different temporal structure | Marginal non-Gaussianity cannot identify their Gaussian subspace. Methods based on several lagged covariance matrices, such as SOBI, use information ordinary FastICA ignores. Distinct lag profiles and their assumptions must be established. |
| Delays or reverberation | The model becomes $x_t=\sum_\ell A_\ell s_{t-\ell}$. Multiplying by one instantaneous inverse generally leaves delayed terms. A convolutive source-separation method addresses this different model. |
| More sources than sensors | A rectangular underdetermined mixture cannot be inverted to recover arbitrary source values. Sparsity or other additional structure can support specialized methods. Disabling an orthogonality constraint does not solve that counting problem. |
| Nonlinear mixing | An expressive encoder can reconstruct data without identifying independent generating causes. Nonlinear ICA needs additional identifiable structure; an ordinary autoencoder or VAE is not automatically a source-separation solution. |
| Nonnegative additive data | NMF constrains factors to be nonnegative. That is a different structural assumption from ICA independence, with its own nonuniqueness and interpretation questions. |

For the delay row, a two-tap example makes the issue concrete: $x_t=As_t+Cs_{t-1}$. Even if $A$ is known, $A^{-1}x_t=s_t+A^{-1}Cs_{t-1}$. The extra term survives. The planned [Source Separation & Audio Denoising](/learn/path/full-curriculum/source-separation-audio-denoising-demucs-band-split-rnn?module=nlp-cv-multimodal) topic will provide the application route beyond the instantaneous model.

### Cost and reproducibility

For $n$ observations and $d$ sensors, forming a dense covariance and diagonalizing it costs approximately $O(nd^2+d^3)$, with the exact method and shape affecting the practical choice. SVD can avoid explicitly forming that covariance. After retaining $k$ dimensions, each parallel FastICA iteration costs $O(nk^2+k^3)$: projected/nonlinear averages plus symmetric orthogonalization. Deflation has repeated data passes and projections on earlier directions. More iterations, components and samples all add work.

Storing $X$ requires $O(nd)$ values; a dense $d\times d$ float64 covariance requires $8d^2$ bytes—800,000,000 bytes at $d=10,000$. This is a storage calculation, not a timing benchmark. There is no universal “five times components squared” sample threshold that ensures reliable recovery. Distribution shape, dependence, noise, conditioning and the intended error criterion affect the data requirement.

The real example here uses only four sensor channels and 12,000 training instants. It is a small CPU exercise. Set an iteration cap, retain convergence warnings, inspect rank, and record versions and seeds. If convergence is poor, diagnose scaling, rank, outliers, contrast and model mismatch before merely raising the cap. A fixed seed makes the computational starting point reproducible; it does not make the estimate insensitive to changed data.

## 9. Practice with changed inputs

Attempt each task before opening its hint and solution. The calculations here change the demonstrated numbers or the decision being made.

### 1. A new sensor recipe

You observe $x=(5,-1)^T$ under $A=\begin{bmatrix}3&1\\1&1\end{bmatrix}$. Recover the two source values. Then give one different source/mixing pair that generates exactly the same observations, using scale ambiguity.

<details><summary>Hint</summary>
Subtract the second sensor equation from the first. To preserve observations after rescaling, alter the corresponding column, not the corresponding row.
</details>

<details><summary>Solution</summary>

The equations are $3s_1+s_2=5$ and $s_1+s_2=-1$. Subtracting gives $2s_1=6$, hence $s_1=3,s_2=-4$. Replace $s_1$ by 6 and column 1 by $(1.5,0.5)^T$, leaving column 2 at $(1,1)^T$. The reconstructed readings are $9-4=5$ and $3-4=-1$. Changing a row instead would alter a sensor recipe and would not implement this ambiguity.
</details>

### 2. Equal kurtosis, changed mixing weights

Two independent standardized Laplace sources have excess kurtosis 3. A unit projection uses weights $a=\sqrt3/2,b=1/2$. Find its variance and excess kurtosis. Compare it with equal weighting and a pure source. Explain whether the equal source kurtoses make separation impossible.

<details><summary>Hint</summary>
Use $a^2+b^2=1$ for variance and fourth powers for the excess kurtosis.
</details>

<details><summary>Solution</summary>

Variance is 1. Kurtosis is $3(9/16+1/16)=30/16=1.875$. Equal weighting gives 1.5 and a pure source gives 3. The equal marginal kurtoses are compatible with a directional contrast; there is no flat ring. The value 1.875 is the exact independent variation to reproduce in the rotation investigation at 30° from a source axis.
</details>

### 3. Repair two plausible implementations

Program A uses $g(u)=u$ on whitened data. Program B estimates every row independently and subtracts previously found directions only after each row has converged. Explain the failure mechanism in each and state a repair.

<details><summary>Hint</summary>
For A, replace $E[z(w^Tz)]$ by $E[zz^T]w$. For B, consider two initializations entering the same attraction region.
</details>

<details><summary>Solution</summary>

A gives $r=Iw-w=0$, so normalization is undefined. Variance has no preferred direction after whitening; use a suitable nonquadratic contrast. B can converge repeatedly to the same direction. Subtracting the already found direction only at the end can leave a near-zero residual, and the intermediate search never respected the constraint. Orthogonalize inside every iteration before normalizing, or use a symmetric multi-component algorithm.
</details>

### 4. Choose without looking at the answer interval

A new recording gives these **signed** correlations with an external reference:

| Component | Development | Test |
| --- | --- | --- |
| 1 | −0.60 | 0.15 |
| 2 | 0.45 | −0.80 |
| 3 | 0.20 | 0.30 |

The protocol is “choose largest development absolute correlation, then report the test absolute correlation.” Which coordinate and result belong in the report? A colleague wants to change the selection after seeing the test column. What should happen next?

<details><summary>Hint</summary>
The test column is for the already fixed choice, even if a different coordinate looks more attractive there.
</details>

<details><summary>Solution</summary>

Select component 1 using $|-0.60|=0.60$ and report test $|0.15|=0.15$. Reporting 0.80 would evaluate a selection made with test information. A revised selection rule becomes a new method to develop and evaluate on fresh held-out data. An actual polarity reversal between intervals is also a useful stability finding; taking absolute values was a declared diagnostic choice, not a way to erase it from investigation.
</details>

### 5. A nuisance component contains wanted activity

In a simulation, the true task signal is $q_t$, but an ICA candidate is $u_t=b_t+0.2q_t$, where $b_t$ is nuisance activity. Its mixing column is $a=(2,-1)^T$. What task contribution is removed when the entire candidate is excluded? Propose a check before deciding whether that removal is acceptable.

<details><summary>Hint</summary>
Multiply the whole component by its mixing column before separating wanted and nuisance terms.
</details>

<details><summary>Solution</summary>

Exclusion removes $a b_t+0.2a q_t$, including wanted signal $(0.4q_t,-0.2q_t)^T$. In this simulation, compare reconstructed task amplitudes or task-event recovery with the known $q_t$ before and after exclusion. On measured data, use a justified task endpoint, auxiliary information and sensitivity to plausible exclusion sets. Reduced visible artifact amplitude alone does not answer the task-preservation question.
</details>

### 6. Design a modest follow-up

Keep the real-data program’s fitting and selection boundary. Propose a follow-up that asks whether its finding persists, without using the existing test result to choose a favorable replacement. State the unit of evaluation and one outcome that would make you revise the conclusion.

<details><summary>Hint</summary>
Changing only the seed measures one kind of variability. A later block and another participant ask different questions.
</details>

<details><summary>Example solution and success criteria</summary>

Predeclare several later non-overlapping blocks and the same fitting/development/test durations; carry all four-channel inputs and the same fixed settings into each. Report each method’s selected-coordinate diagnostic for every block, including failures to converge. Keep participant identity separate, and reserve different participants for a future cross-person claim. If PCA’s advantage reverses across blocks or all correlations collapse, revise the original finding to describe its dependence on that short interval. A good answer states the question, respects information availability, keeps the baseline, records unsuccessful runs, and distinguishes within-recording robustness from population generalization. A new complete clinical study is outside this small exercise.
</details>

## 10. Readiness and the next question

You are ready to move on from the first-pass route when you can explain why the whitened diamond is dependent, calculate a fourth-moment contrast on changed weights, trace a normalized FastICA update, and keep fitting, coordinate selection and evaluation distinct in the recording example. After the deeper component-removal branch, also distinguish a mixing column from an unmixing row and predict what component exclusion subtracts from the sensors.

Try these from memory: What assumption makes an orthogonal search sufficient after whitening? Why can two Gaussian sources rotate without changing the observed model? Why does exact reconstruction say little about source usefulness? Why do component numbers need matching across fits?

Next, [Non-Negative Matrix Factorization (NMF)](/learn/path/full-curriculum/non-negative-matrix-factorization-nmf?module=classical-ml) asks what changes when both factors must be nonnegative and combine additively. That constraint can suit counts or magnitudes. It supplies a different factorization goal; physical meaning and uniqueness still need evidence.

## 11. References & another way to learn it

- **Hyvärinen & Oja — [Independent Component Analysis: Algorithms and Applications](https://www.cs.helsinki.fi/u/ahyvarin/papers/NN00new.pdf).** Free author-hosted tutorial, useful after sections 3–5. Sections 2–6 connect identifiability, non-Gaussianity, whitening and fixed points. The relevant model and algorithm passages were read; examples use older notation/software context.
- **Hyvärinen, Karhunen & Oja — [Independent Component Analysis](https://www.cs.helsinki.fi/u/ahyvarin/papers/bookfinal_ICA.pdf).** Author-hosted book manuscript. Chapters 6–10 deepen whitening and estimation objectives; chapters 13, 15–19 and 22 organize practical issues, noise, temporal structure, convolutive mixing and brain-imaging applications. The contents and selected relevant passages were inspected, not the entire book. Matrix calculus and probability are useful for its proofs.
- **Andrew Ng / Stanford — [CS229 Lecture 15](https://see.stanford.edu/Course/CS229/45), [YouTube recording](https://www.youtube.com/watch?v=QGd06MTRMHs), and [substantive transcript](https://see.stanford.edu/materials/aimlcs229/transcripts/MachineLearning-Lecture15.html).** Another route through the mixture model, Gaussian symmetry and likelihood/CDF reasoning. The official bookmarks locate ICA at 39:49 and the algorithm at 47:41. The ICA transcript and companion notes were reviewed; the video/audio was not watched or evaluated. Use the current lesson/API reference for software details, and expect transcription errors in formulas.
- **Andrew Ng — [CS229 ICA notes](https://cs229.stanford.edu/notes2021fall/cs229-notes11.pdf).** A short mathematical alternative for section 8’s likelihood route. The model, ambiguities, Gaussian example and change-of-variables/likelihood derivation were inspected. This is not a FastICA implementation tutorial.
- **scikit-learn — [FastICA reference](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html).** Consult after running the real example for `components_`, `mixing_`, whitening and iteration semantics. Parameter and attribute sections were checked against the installed 1.9.1 snapshot. Moving stable documentation can change.
- **MNE — [Repairing artifacts with ICA](https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html).** An application tutorial showing fitted decompositions, component inspection, auxiliary-channel evidence and exclusion/reconstruction. Filtering, fitting and component-identification passages were reviewed, not executed here. It assumes knowledge of EEG/MEG recordings and uses a separate MNE API.
- **Jezewski and colleagues / PhysioNet — [Abdominal and Direct Fetal ECG Database, v1.0.0](https://physionet.org/content/adfecgdb/1.0.0/).** The source of the actual simultaneous measurements. Read the acquisition description before interpreting the example. The local extract is distributed with [its attribution and calibration](data-provenance.md) under [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/).
- **Ablin, Cardoso & Gramfort — [Faster Independent Component Analysis by Preconditioning with Hessian Approximations](https://arxiv.org/abs/1706.08171).** Advanced alternative-optimizer reading after section 8. The abstract and author description of its objective/preconditioner were checked; this lesson did not reproduce its benchmarks. Treat speed comparisons as specific experimental results.

Authoring scope, claim locators, original-source conservation, calculation evidence and phase-two continuation live in [the ICA design record](../../ICA-LESSON-DESIGN.md).
