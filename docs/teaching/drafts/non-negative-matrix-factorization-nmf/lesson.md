# Non-Negative Matrix Factorization (NMF)

Suppose you have hundreds of small pictures of handwritten digits. You could store every picture separately. Could you instead learn a small collection of reusable ink patterns and describe each picture by how much of each pattern to add?

That is the central question of **non-negative matrix factorization**. It learns nonnegative patterns and nonnegative amounts whose sums approximate your observations. The same arithmetic can describe a document as a combination of word patterns or a measured spectrum as a combination of spectral patterns. The useful result is a compact, inspectable representation of the measurements.

**First pass:** read §§1–6, work through the additive reconstruction and update investigations as they appear, then attempt exercises 1–5 in §9. This route takes you from matrix entries to a complete offline fit and an interpretation of its errors. Return to §7 for optimization, nonnegative rank and separability, and §8 for topic models, spectra and streaming. Exercises 6–8 assess those deeper branches. Allow roughly 50 minutes for the core reading and another hour for its code and practice; the deeper branches are a separate sitting.

## 1. A component is a pattern, and an activation is an amount

The preceding [ICA lesson](/learn/path/full-curriculum/independent-component-analysis-ica?module=classical-ml) looked for independent hidden sources in mixtures of signals. NMF asks for a different property: all the numbers used to build an observation must be nonnegative. Independence is not part of its basic objective.

You need to multiply a row of numbers by a matrix and add squared errors. We will refresh both. A gradient means the local direction in which an error changes; the first fit can be followed before reading its derivation.

Our convention is **observations in rows, features in columns**:

| Matrix | Shape | Meaning |
| --- | --- | --- |
| \(X\) | \(n\times d\) | The observed nonnegative measurements |
| \(W\) | \(n\times k\) | How much each observation uses each component |
| \(H\) | \(k\times d\) | The component patterns in the original features |
| \(\widehat X=WH\) | \(n\times d\) | Reconstructed measurements |

The letter \(k\) is the **number of components**. People sometimes call it the factorization rank, although the product can have matrix rank smaller than \(k\). A column of \(W\) follows one component across observations; a row of \(H\) follows that component across features. Some references transpose the whole convention. Check shapes before translating a formula.

Consider three constructed observations with three features:

\[
X=\begin{bmatrix}2&1&3\\1&2&3\\3&3&6\end{bmatrix},\qquad
W=\begin{bmatrix}2&1\\1&2\\3&3\end{bmatrix},\qquad
H=\begin{bmatrix}1&0&1\\0&1&1\end{bmatrix}.
\]

The first component contributes equally to features 1 and 3. The second contributes equally to features 2 and 3. Observation 1 is

\[
2[1,0,1]+1[0,1,1]=[2,1,3].
\]

To find one reconstructed cell, multiply corresponding entries and add:

\[
\widehat X_{ij}=\sum_{r=1}^{k}W_{ir}H_{rj}.
\]

For observation 1, feature 3, this is \(2\cdot1+1\cdot1=3\). Every contribution is zero or positive. There is no cancellation between components.

**Inline figure — build one observation.** Show the two three-cell component strips, scale them by 2 and 1, and align their contributions beneath the observed strip. Selecting feature 3 highlights the two entries that add to 3. Shapes and cells should do the explaining before the compact matrix notation returns.

**Investigation — build a new mixture.** Edit the two activation amounts and one component cell, record whether a chosen reconstructed feature will increase, decrease or stay fixed, then apply the change. Find two different mixtures with the same third feature but different first features. The goal is to reason about contributions, not identify a preset by its name.

### The interpretation contract

Nonnegativity guarantees additive reconstruction. Recognizable physical parts, sparse factors, independence and a unique explanation require additional assumptions or evidence. A component can be broad, overlap another component or combine unrelated physical processes. Calling it “eye,” “topic” or “material” is an interpretation to evaluate against its features and domain evidence. NMF weights are not probabilities unless a specified normalization gives them that meaning.

This distinction also separates the recent lessons: a GMM responsibility is a normalized conditional probability of a mixture assignment; a t-SNE coordinate locates a point in a neighborhood map; an ICA coordinate estimates a source under independence assumptions; an NMF activation contributes to an additive reconstruction. None is a generic unit of “hidden meaning.”

## 2. What gets optimized?

Usually the observations do not fit a small number of patterns exactly. We therefore choose \(W,H\geq0\) to minimize a reconstruction loss. For the **squared Frobenius loss**,

\[
F(W,H)=\frac12\|X-WH\|_F^2
=\frac12\sum_{i=1}^{n}\sum_{j=1}^{d}(X_{ij}-\widehat X_{ij})^2.
\]

The subscript \(F\) means: square every matrix entry, sum, then take the square root for the norm. We square that norm in the objective. The one-half cancels a factor of two when differentiating; it does not change which factors minimize the loss.

If an observation is \([2,1,3]\) and its reconstruction is \([1.5,1,2.5]\), the residual is \([.5,0,.5]\). Its contribution to \(F\) is \(\tfrac12(.25+0+.25)=.25\). Keep the residual signed when displaying it: positive means missing reconstructed mass; negative means excess.

This objective values an absolute error of 2 equally at a feature value of 2 and a feature value of 20. Rescaling one feature by ten can make its squared error a hundred times more influential. Thus preprocessing is a modeling decision. Standard centering would introduce negative entries and change the additive interpretation. Dividing all image entries by the known maximum 16, as we do later, preserves zero and relative weights. Arbitrarily shifting a signed dataset until it is nonnegative introduces a baseline pattern the model must explain.

**Inline figure — residual image.** For the exact three-feature calculation, align observed, reconstructed and signed residual strips. Later the same correspondence becomes three 8×8 images. Use a diverging scale for residuals and a sequential nonnegative scale for intensity; those encodings answer different questions.

### A loss encodes which discrepancies matter

For a nonnegative observation \(x\) and positive reconstruction \(y\), two other common entrywise losses are

\[
d_{\mathrm{KL}}(x,y)=x\log(x/y)-x+y,
\qquad
d_{\mathrm{IS}}(x,y)=x/y-\log(x/y)-1.
\]

The first is **generalized Kullback–Leibler divergence**, summed over entries. With the convention \(0\log(0/y)=0\), a zero observation contributes \(y\). If \(x>0,y=0\), the divergence is infinite. It becomes the usual KL divergence when the arrays are normalized probability distributions. The second is **Itakura–Saito divergence**, used for strictly positive entries here.

Compare the same absolute overestimate:

| Observed \(x\), reconstructed \(y\) | \(\tfrac12(x-y)^2\) | Generalized KL | Itakura–Saito |
| --- | ---: | ---: | ---: |
| 2, 4 | 2 | .613706 | .193147 |
| 20, 22 | 2 | .093796 | .004401 |

KL and IS distinguish these two contexts. They still distinguish themselves: scaling both \(x\) and \(y\) by \(c>0\) scales squared loss by \(c^2\), KL by \(c\), and leaves IS unchanged. These follow by substituting into the formulas; “relative error” is too vague to describe all three.

Independent Gaussian errors with common variance yield squared-error fitting of the means. Independent Poisson counts with means \(\widehat X_{ij}\) yield generalized-KL fitting after terms independent of the factors are removed. Choosing the latter is an assumption about count variability, not a rule that every count dataset follows a Poisson model. TF-IDF values, for example, are weighted text features rather than integer counts.

These losses belong to the beta-divergence family, with \(\beta=2,1,0\) respectively. The library supports them, but its coordinate-descent solver uses Frobenius loss; the multiplicative solver supports the other beta losses. Strictly positive input is required for its \(\beta\leq0\) cases. [Scikit-learn's NMF guide](https://scikit-learn.org/stable/modules/decomposition.html#nmf-with-a-beta-divergence) documents those conventions.

## 3. Learn one factor while holding the other still

The difficult part is that both the component patterns and their amounts are unknown. Changing both at once creates a jointly nonconvex problem. A natural strategy is to alternate:

1. Hold the activations \(W\) fixed and improve the patterns \(H\).
2. Hold the newly updated patterns fixed and improve \(W\).
3. Repeat while monitoring the objective and a stopping criterion.

One especially transparent method uses **multiplicative updates**. For the squared loss above,

\[
H\leftarrow H\odot\frac{W^\top X}{(W^\top W)H},\qquad
W\leftarrow W\odot\frac{XH^\top}{W(HH^\top)}.
\]

The symbol \(\odot\) and the fraction mean entrywise multiplication and division. Ordinary adjacent matrix products still mean matrix multiplication. The second update uses the new \(H\).

Here is the reason for the ratio. The gradient with respect to \(H\) is

\[
\nabla_HF=(W^\top W)H-W^\top X.
\]

The first term reflects the reconstruction currently produced; the second reflects the observed data. If the second is larger at a cell, the gradient is negative there, so increasing that cell can reduce loss. Multiplying by their ratio increases it. If the first term is larger, the ratio decreases it. A positive value multiplied by a nonnegative ratio stays nonnegative.

### A full numerical step

Use the \(X\) from §1, but initialize the unknown factors as

\[
W^{(0)}=\begin{bmatrix}1&.5\\.5&1\\1&1\end{bmatrix},\qquad
H^{(0)}=\begin{bmatrix}1&.2&.8\\.2&1&.8\end{bmatrix}.
\]

The initial loss is 17.06. To update \(H_{11}\), the observed-data numerator is
\(1\cdot2+.5\cdot1+1\cdot3=5.5\).
The first row of \(W^\top W\) is \([2.25,2]\), so the denominator is \(2.25\cdot1+2\cdot.2=2.65\).
Therefore \(H_{11}\) becomes \(1\cdot5.5/2.65=2.075472\).

Updating all entries gives

\[
H^{(1)}\approx\begin{bmatrix}2.075472&.408163&2.470588\\.408163&2.075472&2.470588\end{bmatrix}.
\]

Using this new pattern matrix in the \(W\) update gives

\[
W^{(1)}\approx\begin{bmatrix}.826888&.393655\\.393655&.826888\\1.212145&1.212145\end{bmatrix}.
\]

The loss after both updates is .0394474. Notice that some activations decreased even though the corresponding pattern entries increased. What matters is their product.

**Investigation — one alternating update.** Edit a measurement in \(X\), inspect the selected pattern cell's numerator and denominator, and record whether that cell will grow, shrink or remain unchanged. Advance the \(H\) and \(W\) phases separately. The same selected cell must remain highlighted in the products and reconstruction. An exact-fit reset provides the null case: every relevant ratio is 1 and the factors stay fixed.

### Complete NumPy program

Create a Python environment with `python -m pip install numpy`. Save this as `nmf_step.py` and run `python nmf_step.py`. The example has strictly positive initial factors and no entirely zero data row or column, so its displayed update needs no added denominator constant. The supported numerical example is small and bounded.

```python
import numpy as np

X = np.array([[2., 1., 3.], [1., 2., 3.], [3., 3., 6.]])
W = np.array([[1., .5], [.5, 1.], [1., 1.]])
H = np.array([[1., .2, .8], [.2, 1., .8]])

def objective(X, W, H):
    return np.sum((X - W @ H) ** 2) / 2

print(0, round(objective(X, W, H), 8))
for iteration in range(1, 41):
    H *= (W.T @ X) / ((W.T @ W) @ H)
    W *= (X @ H.T) / (W @ (H @ H.T))
    if iteration in (1, 2, 10, 40):
        print(iteration, round(objective(X, W, H), 8))
```

The bounded author calculation executed these updates with NumPy 2.3.5:

```text
0 17.06
1 0.03944744
2 0.02823377
10 0.00180608
40 5e-08
```

The fit approaches the exact product from §1, while its factor values need not match the chosen factors there. The next section explains why.

### Why a flat error trace is not enough

A zero multiplied by a ratio remains zero. An entry initialized at exactly zero can become **zero locked** even when increasing it would improve the fit. For example, hold \(W=[1]\), set \(X=[2,1]\) and \(H=[0,1]\). The gradient for the first \(H\) entry is \(-2\), so a positive move helps. A guarded multiplicative implementation that leaves zero entries at zero cannot make that move. By contrast, the nonnegative least-squares solution with this fixed \(W\) is exactly \([2,1]\).

The original update's upper-bound argument establishes non-increasing objective values under its mathematical conditions. Objective decrease, stationarity, a local minimum and a global minimum are different statements. At a nonnegative boundary, stationarity permits a positive gradient at a zero variable: movement toward negative values is forbidden. A zero gradient everywhere is neither the correct boundary test nor a proof of a local minimum. [Lin's analysis](https://www.csie.ntu.edu.tw/~cjlin/papers/multconv.pdf), §§II–IV, separates these issues and supplies modified updates with a convergence argument. We derive the relevant conditions in §7.

Adding an epsilon to every denominator or clipping factors after every step changes the algorithm. It can be useful numerical engineering, but the unmodified proof cannot simply be copied onto that changed procedure. For ordinary work, use a maintained solver with documented behavior and inspect its convergence information.

## 4. The same observations can have different explanations

Even before considering local optimization, the data may admit several exact nonnegative factorizations.

The factors from §1 yield \(X\) exactly. So do

\[
W_2=\begin{bmatrix}1.5&.5\\.5&1.5\\2&2\end{bmatrix},\qquad
H_2=\begin{bmatrix}1.25&.25&1.5\\.25&1.25&1.5\end{bmatrix}.
\]

Check the first row: \(1.5[1.25,.25,1.5]+.5[.25,1.25,1.5]=[2,1,3]\). The second explanation has components that each contribute to every feature. They are not simply the first components with their order or scale changed. The observations alone have not identified which set of components is physically real.

**Inline figure — two cones, same observations.** Plot the first two features. The first dictionary's rays follow the coordinate axes; the second dictionary's rays point through \((1.25,.25)\) and \((.25,1.25)\). The observed points \((2,1),(1,2),(3,3)\) lie in both cones. Feature 3 equals the sum of the first two throughout this exact example, so this 2D view preserves the relevant containment relationship.

Two simpler ambiguities always deserve attention:

- **Permutation:** exchange two rows of \(H\) and the corresponding columns of \(W\). The product stays the same.
- **Scale:** multiply row \(r\) of \(H\) by any \(c>0\) and divide column \(r\) of \(W\) by \(c\). Every contribution stays the same.

Consequently, a larger raw activation in one fit does not establish a stronger physical component than in another fit. First align component identities and choose a stated scale convention.

### Normalize while preserving the product

Let \(s_r=\sum_jH_{rj}>0\). Define \(\widetilde H_{rj}=H_{rj}/s_r\) and \(\widetilde W_{ir}=W_{ir}s_r\). Then \(\widetilde W\widetilde H=WH\), and every pattern sums to one. A zero pattern contributes nothing and is handled separately rather than divided by zero.

For the first observation of §1, both pattern sums are 2. The normalized patterns are \([.5,0,.5]\) and \([0,.5,.5]\); the activations become \([4,2]\). Their sum, 6, is the reconstructed total mass. Dividing these new activations by 6 gives mixture proportions \([2/3,1/3]\) **for the normalized reconstruction**. The original activation vector \([2,1]\) was not itself a probability distribution.

This normalization preserves reconstruction, but generally changes a penalty on factor magnitudes. Apply it for inspection after an unpenalized fit, or explicitly account for it when the optimization includes regularization.

## 5. Fit real digits and inspect the result

Our [offline CSV](digits-300.csv) contains 300 real digit images: 30 examples of each label from the scikit-learn optical-digits collection. E. Alpaydin and C. Kaynak collected the underlying [UCI Optical Recognition of Handwritten Digits dataset](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits) for recognition research. Each image has 64 block counts arranged as 8×8; each count is an integer from 0 to 16. These are reduced handwritten bitmaps, not MNIST images. Attribution, selection and the CC BY 4.0 license are in [data provenance](data-provenance.md).

The scientific question here is narrower than recognition: **can a small additive dictionary reconstruct previously withheld images, and what do its patterns actually look like?** Labels construct a balanced teaching collection and stratify the split; they are never factorization features. Writer identities are unavailable in this extract, so this split concerns withheld images within the collection, not a claim about new writers.

We divide every block count by 16. Use 180 images to learn patterns, 60 for validation comparisons, and reserve 60 for a final diagnostic. For a new row, `transform` finds its nonnegative activation amounts while keeping the fitted dictionary fixed. It is a small optimization problem, not multiplication by the dictionary transpose as in an orthogonal PCA projection.

**Inline figure — fit versus transform.** In the training lane, both \(W_{\mathrm{train}}\) and \(H\) receive update arrows. In the validation/test lane, \(H\) is shown as the same locked dictionary and only the new \(W\) receives an update arrow. This describes evaluation of a dictionary on new images. For a purely descriptive analysis of one fixed collection, fitting the whole collection answers a different question and can be stated as such.

Install `numpy` and `scikit-learn` in your own environment, save the CSV beside this program as `digits-300.csv`, save the code as `nmf_digits.py`, and run `python nmf_digits.py`.

```python
import numpy as np
from sklearn.decomposition import NMF, PCA
from sklearn.model_selection import train_test_split

data = np.loadtxt('digits-300.csv', delimiter=',', skiprows=1)
source_rows, labels = data[:, 0].astype(int), data[:, 1].astype(int)
X = data[:, 2:] / 16
train, rest = train_test_split(np.arange(len(X)), test_size=.4,
                              random_state=19, stratify=labels)
validation, test = train_test_split(rest, test_size=.5,
                                  random_state=19, stratify=labels[rest])

for k in (1, 4, 8, 16):
    for seed in (7, 19):
        model = NMF(n_components=k, init='random', solver='cd',
                    random_state=seed, max_iter=2000, tol=1e-5)
        W_train = model.fit_transform(X[train])
        W_validation = model.transform(X[validation])
        H = model.components_
        train_mse = np.mean((X[train] - W_train @ H) ** 2)
        validation_mse = np.mean((X[validation] - W_validation @ H) ** 2)
        print(k, seed, round(train_mse, 6), round(validation_mse, 6))

# A fixed eight-component comparison chosen for inspectable image panels.
nmf = NMF(n_components=8, init='random', solver='cd', random_state=19,
          max_iter=2000, tol=1e-5).fit(X[train])
W_test = nmf.transform(X[test])
nmf_reconstruction = W_test @ nmf.components_
pca = PCA(n_components=8, svd_solver='full').fit(X[train])
pca_reconstruction = pca.inverse_transform(pca.transform(X[test]))
mean_reconstruction = np.repeat(X[train].mean(axis=0)[None, :], len(test), axis=0)
for name, reconstruction in [('mean', mean_reconstruction),
                              ('PCA8', pca_reconstruction),
                              ('NMF8', nmf_reconstruction)]:
    print(name, round(np.mean((X[test] - reconstruction) ** 2), 6))
print('first held-out source row', source_rows[test[0]])
print('first reconstructed image')
print(nmf_reconstruction[0].reshape(8, 8).round(2))
```

The author calculation executed the same data, split, fits and quantities under scikit-learn 1.9.1. The validation table is:

| Components | Seed | Training MSE | Validation MSE |
| ---: | ---: | ---: | ---: |
| 1 | 7 | .072144 | .072803 |
| 1 | 19 | .072144 | .072803 |
| 4 | 7 | .040409 | .041899 |
| 4 | 19 | .040409 | .041900 |
| 8 | 7 | .024750 | .025548 |
| 8 | 19 | .025147 | .026315 |
| 16 | 7 | .012295 | .013548 |
| 16 | 19 | .012729 | .015289 |

MSE means the average squared residual across all selected images and all 64 scaled features. Its units are squared fractions of the maximum block count. Here additional components improve validation reconstruction over the inspected range. The higher-rank runs also show an initialization effect. At one component, both seeds nearly agree: a useful null rather than a reason to invent variability.

The separately specified eight-component comparison gives test MSE .069995 for the training-mean image, .019678 for PCA and .025381 for NMF. PCA wins this reconstruction comparison. It is allowed signed, centered patterns and uses an additional mean image; NMF supplies the additive constraint we wanted to inspect. This is a comparison of those representation choices, not equal storage bits or a digit-classification benchmark. There is no reason to tune the example until NMF wins.

**Inline figure — actual factors and reconstructed images.** Reshape the eight actual \(H\) rows into 8×8 patterns. Show a selected held-out image, each weighted contribution \(W_{ir}H_{r,:}\), their sum and the signed residual. First show all panels with a common intensity scale; a clearly labeled per-pattern-normalized inspection toggle may reveal shape, with its scale factor kept visible. A normalized display must not silently replace the numbers used in reconstruction.

For the selected image, rank components by the sum of their actual contributions, not by raw \(W_{ir}\). Removing a component changes the reconstruction by exactly its contribution image. You can now say whether a pattern is concentrated around a stroke, spreads over several regions, or overlaps another pattern. That is stronger evidence than naming every component a digit part in advance.

### Choosing a useful component count

If your actual objective is validation MSE among these candidates, 16 components with seed 7 is the best inspected candidate. The fixed eight-component panel was chosen for a readable demonstration, not mislabeled as the selected optimum. A genuine subsequent test evaluation would fit the chosen procedure according to its declared train/validation policy and evaluate once on the untouched test set. Repeatedly trying choices after viewing test results turns the test set into further validation.

The best attainable unregularized training error cannot increase when another component is allowed: an old fit can be embedded by adding a zero component. A particular local solver run can break that visual trend by finding a worse solution. Neither an elbow nor a stable cluster assignment identifies a universal “true number of topics.” Choose a count using the task: reconstruction on withheld data, stable interpretable patterns, downstream usefulness and the cost of a larger dictionary. When comparing patterns across seeds, normalize and match them one-to-one before measuring similarity.

## 6. Practical choices that change the model

### Solvers and initialization

With \(H\) fixed, fitting each row of \(W\) is a nonnegative least-squares problem with \(k\) unknowns. There are \(n\) such row problems. With \(W\) fixed, fitting each column of \(H\) gives \(d\) problems. These are convex subproblems; alternating between them does not make the joint problem convex.

**Coordinate descent** updates one coefficient or block using the other current values. It can activate a zero entry when the feasible descent direction points into the positive region. A full alternating NNLS method solves each subproblem to an appropriate accuracy; one sweep of coordinate updates should not be described as an exact solve of every subproblem. Multiplicative updates offer an especially readable mechanism and support different losses. There is no solver ranking independent of matrix shape, sparsity, stopping criteria and requested accuracy.

NNDSVD initializes nonnegative factors from singular-vector information. `nndsvd` retains zeros, `nndsvda` fills those zeros with the data mean, and `nndsvdar` uses small random fills. In the inspected scikit-learn version, `init=None` chooses `nndsvda` when the component count fits within the matrix dimensions, otherwise random initialization. Set parameters explicitly in a reproducible lesson. For multiplicative fitting, initial zeros deserve special attention because of zero locking. [NMF API](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html).

### Sparsity is an additional preference

To prefer fewer active contributions or narrower patterns, add a penalty such as

\[
F(W,H)+\lambda_W\sum_{ir}W_{ir}+\lambda_H\sum_{rj}H_{rj}.
\]

For nonnegative factors these sums are their entrywise L1 norms. Penalizing \(W\) discourages an observation from spreading mass across many components; penalizing \(H\) discourages a component from spreading mass across many features. The effect depends on scale and the complete objective. A sparsity penalty is not a constraint that all component word lists become distinct.

Scikit-learn exposes `alpha_W`, `alpha_H` and `l1_ratio`, including mixed L1/L2 penalties. Its objective scales the \(W\) penalty by the number of features and the \(H\) penalty by the number of samples, so a hand-written \(\lambda\) is not automatically the same numerical parameter. The [regularization lesson](/learn/path/full-curriculum/regularization-l1-l2-elastic-net-dropout?module=classical-ml) develops this statistical tradeoff after preprocessing and validation.

### A diagnosis table

| Observation | Inspect next | A justified response |
| --- | --- | --- |
| Input contains negative values | Units, centering and the intended meaning of zero | Choose a meaningful nonnegative representation or a method supporting signed values |
| Objective barely changes | Initialization, stopping tolerance, gradients at active/boundary variables, iteration limit | Distinguish a good fit from boundary stalling; compare a documented alternate solver |
| Similar top words or patterns | Full normalized factors and activations across observations | Check redundancy, a shared background and genuine overlap; do not infer an exact topic count from top words alone |
| Different component numbers between runs | Permutation and scale matching | Compare contributions and aligned shapes |
| Low reconstruction error but unhelpful factors | The task's semantic or scientific diagnostics | Revisit representation, loss, constraints and count rather than rewarding error alone |
| Sparse matrix becomes huge | Intermediate products and storage format | Keep data sparse and use small Gram products; avoid constructing \(WH\) merely to update factors |

## 7. Deeper: geometry, optimization and the limits of factorization

This branch explains why alternating updates work, why general NMF remains hard, and when an additional geometric assumption helps. The core fit and interpretation do not depend on completing its proofs.

### Separate convexity and joint nonconvexity

For fixed \(W\), the Hessian of a column's least-squares objective is \(W^\top W\), which is positive semidefinite because \(v^\top W^\top Wv=\|Wv\|^2\geq0\). The nonnegative feasible region is convex. The corresponding statement holds for rows of \(W\) with \(H\) fixed.

For joint nonconvexity, take the scalar problem \(f(w,h)=\tfrac12(1-wh)^2\). Both \((w,h)=(1,1)\) and \((2,.5)\) have zero loss. Their midpoint \((1.5,.75)\) has product 1.125 and loss .0078125. Convexity would require the midpoint loss to be at most zero. This directly tests the objective, rather than incorrectly concluding that any loss of a bilinear product must be nonconvex.

### The majorization argument

For a column \(h\) of \(H\), let \(A=W^\top W\) and \(b=W^\top x\). Its objective has gradient \(Ah-b\) and Hessian \(A\). Assume the fixed \(W\) has no all-zero component column and the current point \(h'\) is positive; then every \((Ah')_r>0\). An all-zero component column is unused and must be removed or handled separately before this inverse-based derivation. Choose the diagonal matrix
\(D_{rr}=(Ah')_r/h'_r\).
The quadratic function

\[
G(h,h')=F(h')+(h-h')^\top\nabla F(h')
+\tfrac12(h-h')^\top D(h-h')
\]

touches the objective at \(h'\) and lies above it. To see the key inequality, write \(z_r=h'_ru_r\). Since \(A\) is symmetric with nonnegative entries,

\[
z^\top(D-A)z=\frac12\sum_{rs}A_{rs}h'_rh'_s(u_r-u_s)^2\geq0.
\]

Minimizing this separable quadratic gives
\(h=h'-D^{-1}(Ah'-b)=h'\odot b/(Ah')\), the multiplicative rule. Therefore
\(F(h_{\mathrm{new}})\leq G(h_{\mathrm{new}},h')\leq G(h',h')=F(h')\).
This is an upper-bound minimization argument. EM in the earlier GMM lesson used a lower bound while maximizing a log likelihood; the inequality direction changes with the optimization problem. [Lee and Seung's original algorithm paper](https://papers.nips.cc/paper_files/paper/2000/file/f9d1152547c0bde01830b7e8bd60024c-Paper.pdf).

For the nonnegative constraints, the first-order conditions are

\[
W,H\geq0,\quad \nabla_WF,\nabla_HF\geq0,\quad
W\odot\nabla_WF=0,\quad H\odot\nabla_HF=0.
\]

A positive variable must have zero gradient; a zero variable must not have a negative gradient pointing into a feasible decrease. The zero-locked example violates the latter condition. Factor rescaling can also change the size of a gradient-based diagnostic without changing reconstruction, so a stopping metric needs a stated scale convention.

### Nonnegative rank and an informative surprise

The ordinary rank of a matrix measures how many signed linear basis directions suffice for exact reconstruction. **Nonnegative rank** is the smallest \(k\) permitting an exact nonnegative factorization. It is at least ordinary rank and can be larger.

Consider

\[
S=\begin{bmatrix}0&0&1&1\\1&0&0&1\\1&1&0&0\\0&1&1&0\end{bmatrix}.
\]

Its ordinary rank is 3: the sum of rows 1 and 3 equals the sum of rows 2 and 4, and the first three rows are independent. Its nonnegative rank is 4. Each nonnegative rank-one contribution has rectangular positive support and cannot place positive mass in one of the zero cells, since another contribution cannot cancel it. The positive positions \((1,3),(2,4),(3,1),(4,2)\) cannot share a single such rectangle pairwise: for any pair, at least one crossed cell is zero. At least four rank-one contributions are necessary; taking \(W=I_4,H=S\) shows four suffice.

**Inline figure — rectangles cannot cross a zero.** Draw the binary matrix and these four marked cells. Selecting a pair reveals the crossed zero that prevents one positive rectangle from covering both. This makes the difference between rank and nonnegative rank concrete, rather than asking the learner to accept a complexity slogan.

General exact NMF includes NP-hard instances, as the canonical survey discusses. Nonconvexity alone would not prove NP-hardness. Structured cases can be much easier. Under a **separability** assumption, every needed component direction appears among the observed rows (after our row-oriented convention). With nonzero rows normalized to sum to one, all other rows lie in the convex hull of these anchor rows. Finding its extreme points can identify candidate patterns instead of searching for arbitrary hidden directions. Noise, redundant anchors and rank/conditioning assumptions matter to algorithmic guarantees.

For example, observations \([1,0],[0,1],[.25,.75],[.6,.4]\) visibly include the two endpoints. Removing both endpoints leaves many wider enclosing segments consistent with the remaining mixtures, just as in §4. A general unconstrained NMF fit is not automatically a separable model. The survey's geometric algorithms and their explicit assumptions are a useful next theoretical reading. [Gillis, *The Why and How of NMF*](https://arxiv.org/pdf/1401.5226), §§3.2 and 4.

## 8. Deeper applications: words, spectra and streams

### Documents as nonnegative word patterns

Suppose the vocabulary is `[orbit, rocket, goal, team]` and two component rows are \([3,2,0,0]\) and \([0,0,1,4]\). A document activation \([2,1]\) reconstructs \([6,4,1,4]\). The first component supplies most of the space-related words; the second supplies the sports-related words. A mixed document can use both without forcing a hard class assignment.

Real text needs a vocabulary and weighting policy. `CountVectorizer` produces counts; `TfidfVectorizer` downweights words common across the fitted corpus. Vocabulary filtering and IDF estimation belong inside the training split when evaluating new-document behavior. Check actual text examples and full weight patterns, not just attractive top-word lists. Removing a domain word as a “stop word” can remove the signal you wanted to discover.

Here is a complete constructed transfer example. Save it as `nmf_words.py`; it needs the same NumPy/scikit-learn installation as §5.

```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF

documents = ['orbit rocket orbit', 'rocket orbit rocket',
             'goal team goal', 'team goal team',
             'orbit rocket goal team']
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(documents)
model = NMF(n_components=2, init='nndsvda', solver='mu',
            beta_loss='kullback-leibler', max_iter=1000,
            tol=1e-5, random_state=19)
W = model.fit_transform(X)
H = model.components_
mass = H.sum(axis=1)
normalized_patterns = H / mass[:, None]
amounts = W * mass
print(vectorizer.get_feature_names_out())
print(normalized_patterns.round(3))
print(amounts.round(3))
new = vectorizer.transform(['rocket team'])
print((model.transform(new) @ H).round(3))
```

The exact vocabulary order is `['goal', 'orbit', 'rocket', 'team']`; each normalized nonzero pattern sums to 1. The fitted factor values are computed when you run the program; this additional program has not been executed in the content phase, so no fitted numbers are presented as observed outputs. Inspect whether the two patterns divide the vocabulary as expected and how the mixed document uses them. This tiny constructed corpus exposes the operation; the real-data evidence for this lesson remains the digit experiment.

Probabilistic latent semantic analysis can express a normalized count table using a latent-topic mixture. The KL objective has a corresponding likelihood interpretation when totals and factors are normalized appropriately. Latent Dirichlet Allocation adds a hierarchical generative model with Dirichlet priors; normalizing arbitrary NMF factors after fitting does not supply that prior model or its posterior uncertainty. Nor does choosing LDA establish calibration of a scientific claim. The comparison is about modeling assumptions and the desired output, not a universal speed ranking.

### A spectrum can combine materials, with explicit assumptions

In a simple linear mixing model, a measurement across three wavelength bands might be \(.3[.2,.6,.4]+.7[.8,.3,.1]=[.62,.39,.19]\). The two vectors describe material spectra, and the coefficients are nonnegative amounts. If physics and calibration justify abundance fractions, impose a sum-to-one constraint as well; unconstrained NMF does not supply it.

This connection explains both the appeal of NMF and the need for separability or other information: if a pure material is observed, its spectrum can anchor the mixture geometry. If every observation is mixed, several dictionaries may explain it. Nonlinear light interactions, an unknown background and wavelength-dependent measurement uncertainty may require a richer model. Weighted least squares uses each measurement's uncertainty to set its influence; equal Frobenius weights silently assume equal precision. The hyperspectral treatment and environmental-factorization references in the [canonical survey](https://arxiv.org/pdf/1401.5226) provide the documented application context; the three-band numbers here are a constructed illustration.

For audio, the signed waveform first becomes a nonnegative magnitude or power spectrogram. Factor rows can represent frequency patterns and activations can vary across time, depending on orientation. Reconstructing a usable waveform additionally requires a treatment of phase and source assignment; magnitude addition is a modeling approximation, not the same exact linear model used for instantaneous ICA. These domain details belong in the source-separation lesson rather than being hidden inside the word “isolate.”

### What larger matrices cost

With dense \(X\), efficient Frobenius updates cost on the order of \(ndk+(n+d)k^2\) per alternating sweep. Compute denominators as \((W^\top W)H\) and \(W(HH^\top)\), rather than first forming the full \(n\times d\) reconstruction. If \(X\) has \(s\) stored nonzeros, the data products can use roughly \(sk\) work, while the factor and Gram terms remain. This advantage is available to properly organized multiplicative updates as well as coordinate methods.

Sparse input does not make the dense factors free. Storing \(W,H\) costs \(k(n+d)\) numbers; computing every residual still touches many reconstructed entries unless a suitable algebraic objective computation is used. Do not infer elapsed seconds from these operation counts.

In an online dictionary method, encode a batch using the current dictionary, then update the dictionary using information retained from past batches. For squared loss, sums of activation outer products and data–activation products form sufficient quadratic statistics for the fixed past encodings. In our orientation those statistics have shapes \(k\times k\) and \(d\times k\). They compress old data contributions for this update, while changing encodings or the data distribution introduces further choices. Forgetting factors reduce the weight of older batches.

`MiniBatchNMF` exposes a maintained incremental route; its batch size, loss and convergence behavior should be selected from its current documentation and measured for the real task. General online dictionary-learning results have assumptions and do not automatically apply to every beta divergence or streaming implementation. [MiniBatchNMF documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.MiniBatchNMF.html).

Tensor factorization extends this idea to data with three or more axes, such as time × frequency × sensor. Keeping those axes can preserve relationships lost by flattening them. CP and Tucker impose different factor structures; they require their own tensor-shaped derivations. They are optional specialist extensions, not prerequisites for the matrix workflow here.

## 9. Practice: explain, change, diagnose

Attempt the task before opening a hint or solution. The numeric cases differ from the running fit.

### 1. Reconstruct and preserve scale

With \(H=[[2,0,1],[0,1,2]]\) and \(w=[1,3]\), reconstruct the observation. Normalize each pattern to sum to one while preserving the product. What are the normalized mixture proportions and reconstructed total?

<details><summary>Hint</summary>
Multiply each activation by its pattern's old row sum when dividing that pattern by the same sum.
</details>

<details><summary>Solution</summary>
The reconstruction is \([2,3,7]\). Both patterns sum to 3, so normalized patterns are \([2/3,0,1/3]\) and \([0,1/3,2/3]\), with new activations \([3,9]\). Their total is 12 and their normalized proportions are \([1/4,3/4]\). The third feature is large because both patterns contribute to it, not because the observation belongs to a third component.
</details>

### 2. One coordinate's update

Hold \(W=[[1],[2]]\), use \(X=[[2,1],[4,3]]\), and initialize \(H=[[1,1]]\). Compute one Frobenius multiplicative update of \(H\). Does it solve the fixed-\(W\) least-squares problem in this special case?

<details><summary>Hint</summary>
Here \(W^\top W\) is the scalar 5; calculate both entries of \(W^\top X\).
</details>

<details><summary>Solution</summary>
The numerator is \([10,7]\) and the denominator is \([5,5]\), giving \(H=[2,1.4]\). Reconstruction becomes \([[2,1.4],[4,2.8]]\), with half-squared Frobenius loss \(.1\). Each feature has one positive coefficient with unconstrained optimum \(b/5\), so this update reaches the NNLS optimum for fixed \(W\). That one-dimensional coincidence does not make a general simultaneous update an exact solution of a multivariable NNLS problem.
</details>

### 3. A misleading component claim

A colleague shows two nonnegative factors with low residual error and says, “We have proved there are exactly six biological sources; the largest activation identifies the strongest source.” Identify three missing pieces of reasoning and propose evidence to collect.

<details><summary>Hint</summary>
Separate component count, scientific interpretation, and the scale ambiguity.
</details>

<details><summary>Solution</summary>
Six was a chosen factor count and needs a task-based comparison with other counts. Nonnegative components need biological validation, such as agreement with independent markers or controlled mixtures, before being called physical sources. Raw activations depend on dictionary scale; inspect normalized contributions using meaningful measurement units. A defensible study would record those choices, stability across fits and samples, withheld reconstruction or downstream outcomes, and external biological evidence. Merely reducing residual error addresses only the reconstruction question.
</details>

### 4. Investigate an actual held-out image

In the digit fit, choose a test image other than the first. Compute each component's total contribution, remove the largest contributor, and report the before/after MSE for that image. Predict which pixels will lose reconstructed intensity before calculating the removal.

<details><summary>Hint</summary>
For selected row \(i\), the contribution totals are `W_test[i] * H.sum(axis=1)`. Remove the chosen rank-one contribution from the reconstructed row; leave the remaining amounts fixed.
</details>

<details><summary>Solution and success criteria</summary>
The removed image is exactly `W_test[i, r] * H[r]`. Every removed pixel contribution is nonnegative. Some overpredicted pixels can move closer to their observations, but at an exact rowwise NNLS optimum the total row MSE cannot improve by removing a component: setting its coefficient to zero was already a feasible choice. Numerical fits meet that expectation only to their optimization accuracy. Report the source-row ID, selected component, actual contribution vector and both MSE values. Check that the difference between before and after equals the contribution elementwise. Explain which individual pixel errors improve and which worsen. A component with zero activation supplies an exact unchanged case. This task holds the other contributions fixed; reoptimizing them would be a separate comparison.
</details>

### 5. Repair an evaluation pipeline

A document study fits TF-IDF and NMF on all documents, divides the activations into train/test, and reports a classifier score on the latter as evidence for new-document performance. Repair the order. What should happen to a completely unseen word at inference?

<details><summary>Hint</summary>
Identify every object whose learned state used information from the test documents.
</details>

<details><summary>Solution</summary>
Split documents first under the intended deployment unit. Fit vocabulary, IDF and NMF dictionary on training documents only, then train the classifier using those activations. Apply the frozen vectorizer and dictionary to validation/test documents. A word absent from the fitted vocabulary contributes no feature in this fixed representation; inspect how often that occurs and whether it makes a document's representation uninformative. If model selection is repeated, fit the entire pipeline separately within each training fold. The next two lessons develop preprocessing and cross-validation in detail.
</details>

### 6. Deeper: test joint convexity with different points

For \(f(w,h)=\tfrac12(2-wh)^2\), compare \((1,2)\), \((4,.5)\) and their midpoint. Use the result to test convexity.

<details><summary>Solution</summary>
Both endpoints have zero loss. The midpoint is \((2.5,1.25)\), whose product is 3.125 and loss is \(\tfrac12(1.125)^2=.6328125\). This exceeds the average endpoint loss, contradicting convexity. The test concerns the actual squared-loss function.
</details>

### 7. Deeper: choose a loss by its scaling behavior

A positive spectral observation and its prediction are both multiplied by 3 because of a common gain change. Derive how Frobenius, KL and IS losses change. Which comparison is gain-invariant?

<details><summary>Solution</summary>
The residual triples, so squared loss increases by 9. In KL, the log ratio is unchanged and both outside linear terms triple, so loss increases by 3. In IS only the ratio appears, so the loss is unchanged. IS is gain-invariant for this joint rescaling. Whether that is desirable depends on whether absolute signal strength carries information for the task.
</details>

### 8. Deeper: boundary stationarity

For fixed \(W=[1]\), \(X=[3,1]\) and \(H=[0,2]\), compute \(\nabla_HF\). Which coordinates violate the nonnegative first-order conditions, and how would a feasible improving move behave?

<details><summary>Solution</summary>
The gradient is \([-3,1]\). The zero first coordinate has negative gradient: increasing it is a feasible descent move, so it violates stationarity. The positive second coordinate has nonzero gradient: decreasing it slightly is also feasible and improves loss. The fixed-\(W\) optimum is \([3,1]\). A multiplicative zero-locked first coordinate could remain wrong even while the second coordinate improves.
</details>

## 10. Readiness and the next step

For the core route, you should be able to trace a single reconstructed cell, explain one ratio update, preserve a product under factor normalization, fit a frozen dictionary to new observations, and interpret an actual residual without assuming that a component is a physical source. The deeper route adds the nonnegative first-order conditions, the difference between ordinary and nonnegative rank, and the extra assumption behind anchor-based recovery.

Next is [Feature Scaling, Encoding & Imputation](/learn/path/full-curriculum/feature-scaling-encoding-imputation?module=classical-ml). NMF made preprocessing consequences visible: centering changes signs, feature scaling changes squared-error influence, and a learned transformation must be fitted within the intended information boundary. The next lesson builds those choices into a complete representation pipeline.

## References & another way to learn it

- [Nicolas Gillis, *The Why and How of Nonnegative Matrix Factorization*](https://arxiv.org/pdf/1401.5226) — freely available survey/chapter. Read the image/text examples for another visual explanation, then §3 for algorithms and §3.2 for separable geometry. The survey uses observations in columns, so transpose the matrix convention when comparing it with this lesson. Its numerical benchmark is historical, not a prediction of current library timings.
- [Lee and Seung, *Algorithms for Non-negative Matrix Factorization*](https://papers.nips.cc/paper_files/paper/2000/file/f9d1152547c0bde01830b7e8bd60024c-Paper.pdf) — original seven-page algorithm paper. Best after the ratio walkthrough; it develops both Euclidean and generalized-KL updates and the auxiliary-function argument. Read it together with the next reference for precise convergence distinctions.
- [Chih-Jen Lin, *On the Convergence of Multiplicative Update Algorithms for NMF*](https://www.csie.ntu.edu.tw/~cjlin/papers/multconv.pdf) — deeper mathematical reading on boundary behavior, first-order conditions and modified updates. Requires comfort with gradients and limit points; §II explains the central issue before the proof.
- [Scikit-learn decomposition guide](https://scikit-learn.org/stable/modules/decomposition.html#nmf) and [NMF API](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html) — current parameter semantics, losses, initialization and transformation behavior, inspected as version 1.9.1. Use these when reproducing the programs rather than copying defaults from an older tutorial.
- [Scikit-learn's example comparing decomposition patterns](https://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html) — an alternate visual activity: compare what different constraints make visible in the same image collection. Its example requires its own dataset retrieval; this lesson supplies a separate offline digit dataset.
- [Optical Recognition of Handwritten Digits, UCI](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits) — data collection, feature construction and attribution. See the local provenance before reusing or exporting the provided subset.

The linked papers and substantive documentation were read for this draft. No video is required to complete its explanations or exercises; the inspected visual example and canonical chapter provide alternate learning routes.
