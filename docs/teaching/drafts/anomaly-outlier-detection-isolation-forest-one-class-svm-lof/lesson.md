# Anomaly & Outlier Detection (Isolation Forest, One-Class SVM, LOF)

## Find observations worth investigating—and explain the comparison

A machine usually runs near a familiar temperature. Today it is much hotter. That reading might indicate a fault, a planned operating change, a different load, or a faulty sensor. An anomaly detector can help identify an unusual observation. Deciding what happened requires additional evidence.

The useful starting question is: **unusual compared with what, and what will we do if we find it?**

You will learn three comparisons. Isolation Forest asks how easily random cuts separate an observation. Local Outlier Factor (LOF) compares its neighborhood with nearby neighborhoods. One-Class SVM learns a boundary around a reference population. You will then turn their scores into decisions and compare them with a simple baseline on a real machine-temperature series.

**First pass:** read sections 1–8, work through the temperature example in section 10, and attempt practices A–F. Sections 9 and 11 develop optimization, computational and monitoring details. You do not need to finish those deeper branches before trying the basic investigations.

You need distances, averages and the idea of a training set. We will explain the logarithm, exponential and kernel notation where they enter. A score is a number used to order observations, not automatically a probability.

## 1. Name the observation before choosing an algorithm

Suppose our reference measurements are 0, 1, 2, 3 and 12. The gap between 3 and 12 is large relative to the other gaps. That makes 12 a reasonable candidate for investigation. It does not tell us whether 12 is a mistake. If the values represent operating modes, 12 could be legitimate. If units were entered incorrectly, the same geometry could reveal a data-quality problem.

An observation can also be a transaction, session, patient measurement, image or time window. Its representation determines which differences are visible. A detector given only temperature cannot recognize “ordinary under heavy load but excessive while idle” unless load, operating mode or an appropriate residual enters the representation.

| Case | What is unusual? | Example and needed representation |
|---|---|---|
| Point anomaly | One observation differs from the chosen reference | A sensor reports 900 while comparable readings are around 90 |
| Contextual anomaly | The observation is unusual under its context | 90 is ordinary under load but unusual while idle; include or condition on load |
| Collective anomaly | A group or sequence is unusual even when individual values look ordinary | An unusually long flat trace; represent duration, variability or the sequence |

These cases can overlap. Putting a one-hour change beside the current temperature introduces a little temporal context. It does not turn an ordinary tabular detector into a complete sequence model.

The previous [DBSCAN lesson](/learn/path/full-curriculum/dbscan-density-based-clustering?module=classical-ml) supplies a useful boundary: a point labeled noise was not attached to a density-connected component under a particular metric and parameter choice. That is a geometric result. Here we additionally specify a reference population and decision rule. A dense collection of repeated bad measurements can form a cluster; a rare legitimate operating mode can lie outside every cluster.

**Try it.** A detector flags a rare but scheduled shutdown. Has the algorithm necessarily failed?

**Explanation.** It may have correctly found an unusual operating state. It fails the intended alerting task if the system was supposed to suppress scheduled shutdowns and had the information needed to do so. The task and available context matter.

## 2. Separate fitting, scoring and taking action

**Outlier detection:** you have one collection that may already contain unusual observations. Fit a comparison to that collection and inspect its members. Reviewing a batch of sensor records for data-quality issues is one example.

**Novelty detection:** fit on a reference collection intended to represent acceptable behavior, then score later observations. Learning from a reviewed operating period and monitoring the next period is one example. “Reference” is a declared assumption, not a guarantee that the historical data are clean.

In scikit-learn, this distinction determines which LOF methods may be called. Other literature sometimes uses novelty more broadly for newly encountered concepts; here we use the library's fitting-and-query convention.

Separate three objects:

\[
\text{reference data}\longrightarrow\text{fitted comparison},
\qquad x\longrightarrow A(x),
\qquad \text{alert if }A(x)>\tau.
\]

Here \(A\) is oriented so that **larger means more unusual**, and \(\tau\) is a chosen threshold. Fitting determines the score. Thresholding determines which observations receive attention. Two thresholds can give different alert counts from exactly the same model and ranking.

[Inline figure: a reference period builds the scaler and detector; a later calibration period sets the threshold; a final period supplies untouched scores and evaluation. Labels and event annotations enter the evaluation lane, not the fitting lane.]

Do not fit a scaler on the complete series before splitting. That lets later information influence earlier distances. For repeated customers, devices or patients, keep groups separate when the intended test is performance on new groups. For future monitoring, preserve chronological order. Nearby rows can share one physical event even when their timestamps differ.

### Use one score orientation at the decision boundary

The following contracts refer to the APIs inspected for scikit-learn 1.9.1. Keep the model's own offset distinct from a threshold calibrated by your application.

| Output | Direction and meaning | Our anomaly-oriented quantity |
|---|---|---|
| Isolation Forest: score_samples(X) | Smaller is more unusual; negative isolation-based anomaly score | −score_samples(X) |
| One-Class SVM: score_samples(X) | Unshifted kernel decision score; smaller is less compatible with the reference region | −score_samples(X) |
| LOF: negative_outlier_factor_ | Negative training-row LOF values | −negative_outlier_factor_ |
| LOF, novelty=True: score_samples(X_new) | Negative LOF-style scores for new queries against the frozen reference | −score_samples(X_new) |

The decision function subtracts an offset from the normality-oriented score. A negative decision value is outside the estimator's selected boundary. It is not a calibrated anomaly probability, and the kernel decision value is not a Euclidean distance in the original space.

The rest of the lesson uses \(A>\tau\), with **strict** inequality. Scores tied at the threshold remain unflagged. A percentile setting therefore need not flag exactly the requested fraction.

## 3. Isolation Forest: an empty gap can make a point easy to separate

Choose a cut uniformly along the interval from 0 to 12. Any cut between 3 and 12 separates the last point from the other four. That interval occupies 9 of the 12 units, so its first-cut isolation probability is

\[
\frac{12-3}{12-0}=\frac34.
\]

To isolate 0 on that first cut, the cut must fall between 0 and 1. Its probability is only \(1/12\). No classifier has learned the label “bad.” The empty gap gives 12 many opportunities for early separation.

In several dimensions, an isolation tree chooses a feature and a cut between its current minimum and maximum, then repeats inside each child. A path records how many cuts a query follows before a terminal node. Many random trees reduce dependence on any one lucky cut.

[Investigation: edit the five positions, commit which point will have the shortest average path, and reveal the cut intervals and paths. Move 12 to 4: the endpoint remains a little easier to isolate, but the exceptional gap disappears. Put every point at the same coordinate: no separating cut exists.]

### Why a terminal node can contain several observations

A practical tree stops at a depth limit, commonly \(\lceil\log_2\psi\rceil\) for subsample size \(\psi\), or when it cannot separate remaining values. If the terminal node contains \(m\) reference rows, stopping does not mean all \(m\) were individually isolated. Add an average remaining-path correction \(c(m)\).

\[
c(m)=2H_{m-1}-\frac{2(m-1)}m,
\qquad H_r=1+\frac12+\cdots+\frac1r,
\]

with \(c(0)=c(1)=0\), \(c(2)=1\). This normalizer comes from an average search-path calculation, not a model of anomaly probability. Implementations often approximate the harmonic term for larger \(m\); our tiny calculation uses the exact finite sum.

If a query reaches depth \(d\) at a leaf of size \(m\), its corrected path is \(h=d+c(m)\). Average across trees, then normalize:

\[
s(x)=2^{-\mathbb E[h(x)]/c(\psi)}.
\]

If the average equals the normalizer, the exponent is −1 and the score is \(1/2\). Shorter paths give larger scores. The exponential rescales a path statistic; 0.8 does **not** mean an 80% chance of failure.

For the five positions, \(c(5)=77/30\approx2.5667\). Integrating all possible one-dimensional cut intervals exactly, with depth cap 3 and the stated leaf correction, gives:

| Position | Expected corrected path | Rounded score |
|---:|---:|---:|
| 0 | \(31/12\approx2.5833\) | 0.497755 |
| 1 | \(73/22\approx3.3182\) | 0.408159 |
| 2 | \(17/5=3.4\) | 0.399239 |
| 3 | \(17/6\approx2.8333\) | 0.465258 |
| 12 | \(841/660\approx1.2742\) | 0.708845 |

These are expectations over this one-dimensional construction, **not** outputs promised from a finite random forest or the library. With 0, 1, 2, 3, 4, the endpoints tie at about 0.569715. With five identical values, the corrected path is \(c(5)\) everywhere and the scores are all 0.5. Equal scores provide no ranking among those observations.

### A compact isolation tree you can inspect

**Run the examples.** Use Python 3.12 in a virtual environment. From a terminal in your example folder, install the versions used for the calculations:

~~~sh
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux alternative:
# source .venv/bin/activate
python -m pip install numpy==2.3.5 pandas==3.0.1 scikit-learn==1.9.1
~~~

Save each Python block in its own file: isolation_example.py, lof_modes.py, lof_arithmetic.py, kernel_boundary.py and temperature_monitor.py, in that order. Run a file with, for example, python isolation_example.py. The temperature program also needs the linked CSV and JSON beside it. These small teaching programs expose the calculation; they are not hardened production monitoring services.

This complete program uses one-dimensional trees, exact harmonic correction, a local random generator and subsampling. It accepts ordinary finite numeric inputs with at least two rows. The library's construction and harmonic approximation need not match its scores.

~~~python
import math
import numpy as np


def correction(n):
    if n <= 1:
        return 0.0
    return 2 * sum(1 / j for j in range(1, n)) - 2 * (n - 1) / n


def grow(values, depth, limit, rng):
    lo, hi = float(values.min()), float(values.max())
    if len(values) <= 1 or depth == limit or lo == hi:
        return {"size": len(values)}
    cut = rng.uniform(lo, hi)
    left = values < cut
    if not left.any() or left.all():
        return {"size": len(values)}
    return {
        "cut": cut,
        "left": grow(values[left], depth + 1, limit, rng),
        "right": grow(values[~left], depth + 1, limit, rng),
    }


def path(tree, query, depth=0):
    if "size" in tree:
        return depth + correction(tree["size"])
    child = "left" if query < tree["cut"] else "right"
    return path(tree[child], query, depth + 1)


def fit_forest(values, n_trees=200, max_samples=256, seed=17):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Use at least two finite one-dimensional observations.")
    if type(max_samples) is not int or max_samples < 2:
        raise ValueError("Use an integer max_samples >= 2.")
    if type(n_trees) is not int or n_trees < 1:
        raise ValueError("Use a positive integer number of trees.")
    size = min(max_samples, len(values))
    rng = np.random.default_rng(seed)
    trees = [grow(rng.choice(values, size, replace=False), 0,
                  math.ceil(math.log2(size)), rng) for _ in range(n_trees)]
    return trees, size


def anomaly_scores(fitted, queries):
    trees, size = fitted
    mean_paths = [np.mean([path(tree, x) for tree in trees]) for x in queries]
    return 2 ** (-np.asarray(mean_paths) / correction(size))


x = np.array([0., 1., 2., 3., 12.])
fitted = fit_forest(x)
scores = anomaly_scores(fitted, x)
print("Most easily isolated position:", x[np.argmax(scores)])
print("Duplicate-only scores:", anomaly_scores(fit_forest([2., 2., 2.]), [2., 2.]))
~~~

The seeded example should identify 12; the duplicate-only construction gives scores 0.5 and 0.5. Inspect the fitted size: requesting max_samples 256 does not put 256 observations in a five-row tree. Normalizing as though it did changes the score's meaning.

### Subsampling and representation

If unusual observations form a group, members can shield one another from rapid isolation: one form of **masking**. A smaller subsample may leave fewer members together and expose their separation. It can also omit a legitimate rare mode. There is no universally best subsample size.

The original method uses axis-aligned cuts. Translating or positively rescaling a coordinate preserves the corresponding uniform-cut construction in exact arithmetic. Arbitrary rotations generally do not. Irrelevant features can waste cuts. Random-projection variants change the mechanism and should not be silently substituted for the method derived here.

## 4. LOF: compare local spacing with nearby local spacing

Consider two legitimate groups on a line: 0, 1, 2 and 20, 24, 28. The second is more spread out. A global nearest-distance rule can penalize it even when its spacing is internally consistent. LOF asks: **is this point less locally supported than its neighbors are?**

We use exactly \(k=2\) other reference rows, with stable row-order tie breaking. Distance is the absolute difference. Self means the same observation ID, not every row at the same coordinate.

### Step 1: measure each neighbor's usual radius

For reference point \(o\), let \(r_k(o)\) be its kth selected neighbor's distance.

| Reference point | Two neighbors | \(r_2\) |
|---:|---|---:|
| 0 | 1, 2 | 2 |
| 1 | 0, 2 | 1 |
| 2 | 1, 0 | 2 |
| 20 | 24, 28 | 8 |
| 24 | 20, 28 | 4 |
| 28 | 24, 20 | 8 |

### Step 2: put a floor under each distance

\[
\operatorname{reach}_k(p,o)=\max\{d(p,o),r_k(o)\}.
\]

The radius belongs to **neighbor \(o\)**. The floor prevents an exceptionally short distance to \(o\) from creating an arbitrarily large local density simply because \(p\) nearly coincides with it. Reversing \(p,o\) changes the radius used, so reachability need not be symmetric.

If you read the optional density-hierarchy branch previously, keep the formulas separate: OPTICS reachability uses the source point's core radius, while HDBSCAN mutual reachability includes both endpoint core radii. LOF uses the neighbor's radius in this directed comparison. Our k selects other rows, excluding the training row itself; DBSCAN's min_samples count includes the point itself. None of those earlier optional formulas is needed to perform the maxima below.

### Step 3: take the reciprocal of average reach

For positive mean reach,

\[
\operatorname{lrd}_k(p)=
\left(\frac1k\sum_{o\in N_k(p)}\operatorname{reach}_k(p,o)\right)^{-1}.
\]

The reciprocal is large when average reach is small. Its units are inverse distance. It is **not a probability density normalized to integrate to one**.

At reference point 1, both distances are 1, but the radii of neighbors 0 and 2 are both 2. Both reaches are therefore 2:

\[
\operatorname{lrd}(1)=\frac1{(2+2)/2}=\frac12.
\]

At point 0, reaches to 1 and 2 are 1 and 2, giving lrd \(2/3\). All six density proxies are

\[
2/3,\quad1/2,\quad2/3,\quad1/6,\quad1/8,\quad1/6.
\]

### Step 4: take a dimensionless comparison

\[
\operatorname{LOF}_k(p)=\frac1k\sum_{o\in N_k(p)}
\frac{\operatorname{lrd}_k(o)}{\operatorname{lrd}_k(p)}.
\]

Near 1 means comparable local support. Larger means the selected neighbors have greater local density proxies than the query. Values below 1 are possible; there is no theorem that every acceptable observation scores exactly 1.

For point 1, both neighbor densities are \(2/3\), so LOF is \((2/3)/(1/2)=4/3\). For point 0, mean neighbor density is \((1/2+2/3)/2=7/12\), so LOF is \((7/12)/(2/3)=7/8\). The scaled second group has the same pattern:

\[
7/8,\quad4/3,\quad7/8,\quad7/8,\quad4/3,\quad7/8.
\]

Even this regular example has non-unit values. “Anything above 1 is bad” confuses a finite-neighborhood effect with an application decision.

### Farther from a reference point, yet lower LOF

Freeze the six rows as the reference. Query 4 has neighbors 2 and 1, distances 2 and 3, and radius floors 2 and 1. Reaches are 2 and 3, so its density is \(2/5\). Mean reference density is \(7/12\):

\[
\operatorname{LOF}(4)=\frac{7/12}{2/5}=\frac{35}{24}\approx1.4583.
\]

Query 17 has neighbors 20 and 24. Reaches are \(\max(3,8)=8\) and \(\max(7,4)=7\), so its density is \(2/15\). Mean neighbor density is \(7/48\):

\[
\operatorname{LOF}(17)=\frac{7/48}{2/15}=\frac{35}{32}\approx1.09375.
\]

Query 17 is farther from its nearest reference point—3 units rather than 2—but has the lower LOF. Its nearby group normally has larger spacing.

[Investigation: commit which query is more locally isolated, then drag the query. Neighbor IDs, radii, each maximum, reciprocal and ratio update together. Change k from 2 to 3: neighborhoods cross the gap and the original contrast can disappear.]

### Ties and duplicates change the contract

The original paper includes every point at distance at most the kth-neighbor distance, so ties can give more than \(k\) neighbors. Our small calculation selects exactly \(k\), as the library's fixed-neighbor calculation does; tied selections may depend on implementation ordering. State the convention before comparing numbers.

Repeated coordinates require care. Exclude a training row by identity, not by removing every zero distance or blindly dropping the first sorted entry. If all relevant reaches vanish, unstabilized reciprocals are infinite and ratios can be undefined. Library stabilization makes the arithmetic computable; it does not create separation between identical records. Our hand calculation uses distinct coordinates and positive reaches.

## 5. LOF fitting mode is part of the mathematics

A training row excludes its own ID from its neighborhood. A **new query** has no training ID. If it equals a training coordinate, that reference row is a valid zero-distance neighbor.

Consequently, novelty query scoring on the training array does not recover training-row scores. In our example, training points 1 and 24 have LOF \(4/3\). Treating those coordinates as new queries gives \(7/8\): their neighbor sets changed.

~~~python
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

x = np.array([0., 1., 2., 20., 24., 28.])[:, None]
model = LocalOutlierFactor(n_neighbors=2, novelty=True).fit(x)
print("Training LOF:", np.round(-model.negative_outlier_factor_, 6))
print("New-query LOF:", np.round(-model.score_samples([[6.], [17.]]), 6))
print("Wrong training comparison:", np.round(-model.score_samples(x), 6))
~~~

For the inspected version, rounded values are:

~~~text
Training LOF: [0.875    1.333333 0.875    0.875    1.333333 0.875   ]
New-query LOF: [2.625   1.09375]
Wrong training comparison: [0.875 0.875 0.875 0.875 0.875 0.875]
~~~

Use the default novelty=False with fit_predict when reviewing the fitted collection. Use novelty=True when later queries are the task, and score those queries with score_samples, decision_function or predict. New queries do not become one another's neighbors, so scoring a batch does not adapt the frozen reference.

### The core LOF arithmetic in a small program

This complete example exposes the floors and ratios. It uses exactly k other rows, stable ties and distinct one-dimensional reference values. It deliberately omits a production search index and duplicate stabilization.

~~~python
import numpy as np

x = np.array([0., 1., 2., 20., 24., 28.])
k = 2
if len(np.unique(x)) != len(x) or not 1 <= k < len(x):
    raise ValueError("Use distinct coordinates and 1 <= k < n.")

distance = abs(x[:, None] - x[None, :])
np.fill_diagonal(distance, np.inf)  # Exclude this row's own identity.
neighbors = np.argsort(distance, axis=1, kind="stable")[:, :k]
radius = distance[np.arange(len(x)), neighbors[:, -1]]
reach = np.maximum(distance[np.arange(len(x))[:, None], neighbors],
                   radius[neighbors])
lrd = 1 / reach.mean(axis=1)
lof = (lrd[neighbors] / lrd[:, None]).mean(axis=1)

queries = np.array([4., 6., 17.])
query_distance = abs(queries[:, None] - x[None, :])
query_neighbors = np.argsort(query_distance, axis=1, kind="stable")[:, :k]
query_reach = np.maximum(
    query_distance[np.arange(len(queries))[:, None], query_neighbors],
    radius[query_neighbors],
)
query_lrd = 1 / query_reach.mean(axis=1)
query_lof = (lrd[query_neighbors] / query_lrd[:, None]).mean(axis=1)
print("Reference LOF:", np.round(lof, 6))
print("Query LOF:", np.round(query_lof, 6))
~~~

Query results are approximately 1.458333, 2.625 and 1.09375. The full pairwise matrix makes the calculation inspectable but uses quadratic storage. Use a suitable neighbor search rather than this matrix for a large dataset.

## 6. One-Class SVM: build a compatibility boundary from similarities

Suppose we have reviewed reference observations but few failures. We can ask for a region that represents the reference reasonably well while allowing some training observations to fall outside.

One-Class SVM constructs a separating hyperplane in a feature space. A **kernel** calculates similarities in that space without explicitly constructing every feature. The RBF kernel is

\[
K(x,z)=\exp(-\gamma\|x-z\|^2),\qquad\gamma>0.
\]

When \(x=z\), squared distance is zero and similarity is 1. Greater distance reduces it. Increasing gamma makes similarity decay over a shorter input distance. Features and units must therefore be meaningful before choosing gamma.

A fitted decision has the form

\[
g(x)=\sum_i\alpha_iK(x_i,x)-\rho.
\]

The nonnegative weights identify reference observations that contribute to the boundary; observations with nonzero weights are support vectors. The offset rho determines the zero contour. Positive g is inside the selected region; negative g is outside.

This does not imply a convex or connected region in the input space. A hyperplane in a nonlinear feature space can describe separated regions in the original coordinates.

### Two reference observations expose the mechanism

Take \(x_1=-1,x_2=1\) and \(\nu=1/2\), using the normalized optimization derived in section 9. Symmetry gives \(\alpha_1=\alpha_2=1/2\). Both reference observations lie on the boundary:

\[
\rho=\frac{1+e^{-4\gamma}}2,
\qquad
g(x)=\frac{e^{-\gamma(x+1)^2}+e^{-\gamma(x-1)^2}}2-\rho.
\]

At midpoint 0 the similarities agree:

\[
g(0)=e^{-\gamma}-\frac{1+e^{-4\gamma}}2.
\]

For gamma 0.1, this is approximately +0.069677: the midpoint belongs to the learned region. For gamma 1, it is approximately −0.141278: the midpoint is outside while both reference points remain on the boundary.

At gamma 1, moving a little inward from either reference produces positive scores, whereas the midpoint and far-away points have negative scores. The nonnegative region has separated pieces. Describing every One-Class SVM region as “one connected blob of normal data” would miss this behavior.

[Investigation: predict the midpoint sign before changing gamma. Reveal the two similarity curves, their weighted sum, the rho line and the signed difference. Keep a reference point selected while varying gamma: its score remains zero in this exact symmetric example.]

~~~python
import numpy as np

def decision(x, gamma):
    rho = (1 + np.exp(-4 * gamma)) / 2
    return (np.exp(-gamma * (x + 1)**2)
            + np.exp(-gamma * (x - 1)**2)) / 2 - rho

for gamma in [0.1, 1.0]:
    print(f"gamma={gamma:.1f}: midpoint={decision(0, gamma):+.6f}, "
          f"reference={decision(1, gamma):+.6f}")
~~~

~~~text
gamma=0.1: midpoint=+0.069677, reference=+0.000000
gamma=1.0: midpoint=-0.141278, reference=+0.000000
~~~

### Nu is not tomorrow's fault prevalence

In exact optimization, under the conditions in section 9, nu bounds a fraction of training margin violations and a fraction of support vectors. It is a constraint-and-regularization parameter, not known anomaly prevalence, a promised future false-positive rate, or a guarantee that exactly nu times n training predictions are negative.

Solver tolerances and the treatment of boundary points matter when comparing software counts with the theorem. Use a separate calibration set when the application needs an explicit alert budget.

The API's gamma='scale' sets gamma using fitted-data variance. It does not standardize each feature. Nor is gamma the RBF standard deviation: in the alternative form \(\exp(-\|x-z\|^2/(2\sigma^2))\), gamma equals \(1/(2\sigma^2)\).

## 7. Choose a comparison that matches the task

| Question about the representation | Mechanism to investigate | What to inspect |
|---|---|---|
| Do unusual observations separate after few random cuts? | Isolation Forest | Feature relevance, axis orientation, subsample variability, rare legitimate groups |
| Do acceptable regions have different local spacings? | LOF | Neighbor count, metric, duplicates, whether the reference represents those regions |
| Can reviewed observations define a useful nonlinear support region? | One-Class SVM | Scaling, gamma, nu, kernel cost, later-data stability |
| Is a simple baseline sufficient? | Domain rule, residual or robust deviation | Its assumptions and calibration under the same evaluation protocol |

There is no universal ranking here. Hold the available training information and evaluation protocol fixed. Giving one method reviewed normal data while another receives a contaminated batch changes the task as well as the algorithm.

Some useful applications make the representation issue concrete:

* **Calibration drift in an instrument.** Monitor residuals against a stable reference standard, rather than raw readings that legitimately change with the specimen. Within-instrument monitoring and transfer to unseen instruments need different splits.
* **A stuck sensor.** Each repeated value may be common. Near-zero trailing variability and unusual duration expose the collective event that a level-only detector can miss.
* **Scientific sample or manufacturing-batch review.** A rare specimen may be the most interesting legitimate observation. Rank for inspection and preserve IDs and raw measurements rather than automatically deleting it.
* **Unexpected access patterns.** Session-level features can expose a new combination of ordinary actions. Repeated activity by one account calls for account-aware evaluation when testing transfer to new accounts.

Each example specifies a unit, context, reference and intended response. “Remove every outlier before training” is not an adequate policy: unusual measurements may be the investigation's target.

## 8. A good ranking still needs a decision policy

A hypothetical detector catches 80% of faults and flags 1% of nonfault observations. Faults occur in 0.1% of 100,000 observations. Expected counts are:

* 100 faults, of which 80 are flagged;
* 99,900 nonfault observations, of which 999 are flagged;
* 1,079 total alerts, with only \(80/1079\approx7.4\%\) corresponding to faults.

These are specified hypothetical rates, not a claim about any real application's prevalence. A low false-positive rate can still overwhelm reviewers because the nonfault population is much larger.

For prevalence p, sensitivity t and false-positive rate f, provided alert probability is positive,

\[
P(\text{fault}\mid\text{alert})=\frac{pt}{pt+(1-p)f}.
\]

The numerator counts the fraction that are faults and flagged; the denominator counts all flagged observations. If nobody is flagged, precision is undefined rather than automatically 0 or 1.

[Investigation: enter prevalence, sensitivity, false-positive rate and review budget. Commit a predicted fraction of useful alerts, then reveal a labeled population/count diagram. Change prevalence while holding both conditional operating rates fixed.]

### Calibrate the action separately

With representative reviewed calibration data, choose a threshold using workload, costs or a labeled operating point. An unlabeled empirical score quantile controls an **observed calibration alert fraction**, not a known false-positive fraction.

Sorted calibration scores 1, 1, 2, 4, 4 illustrate ties. Threshold 4 and strict greater-than produce zero alerts. The score threshold cannot select exactly 20% of these rows. A top-B policy could select B rows, but needs a tie rule and has a different contract.

Numeric contamination in Isolation Forest and LOF selects their fitted score offset. It does not discover the true fraction of faults. For the same other fitting settings, changing that offset changes labels without changing score ranking. One-Class SVM has nu rather than contamination, and its effect is not merely a post-fit percentile.

If reliable labels exist, inspect precision–recall behavior, recall at an affordable budget and actual counts. ROC summaries do not display alert workload by themselves. Changing only the threshold cannot improve ranking AUC: score order is unchanged. Labels obtained only for reviewed top-ranked items do not reveal recall among everything never reviewed.

For incidents, specify whether evaluation counts **rows or events**. Consecutive positives from one physical incident are not independent successful detections. State alert grouping, event matching and the latency definition.

## 9. Deeper: why One-Class SVM has these constraints

Let phi be the feature map and n the number of reference observations. The normalized primal is

\[
\min_{w,\rho,\xi}\quad
\frac12\|w\|^2+\frac1{\nu n}\sum_{i=1}^n\xi_i-\rho
\]

subject to

\[
\langle w,\phi(x_i)\rangle\ge\rho-\xi_i,\qquad
\xi_i\ge0,\qquad0<\nu\le1.
\]

The term −rho rewards moving the separating level from the origin. The norm restrains w; slack permits a reference observation below the level at a cost. This is the origin-separating One-Class SVM formulation. A support-vector enclosing-sphere formulation is related, but should not be conflated with it without stating equivalence conditions.

Introduce nonnegative multipliers alpha for the first constraints and beta for nonnegative slack. Stationarity of the Lagrangian gives

\[
w=\sum_i\alpha_i\phi(x_i),\qquad
\sum_i\alpha_i=1,\qquad
\alpha_i+\beta_i=\frac1{\nu n}.
\]

Substitution yields

\[
\min_\alpha\frac12\sum_{i,j}\alpha_i\alpha_jK(x_i,x_j),
\qquad0\le\alpha_i\le\frac1{\nu n},\qquad\sum_i\alpha_i=1.
\]

A positive-semidefinite kernel makes this a convex quadratic problem. In the two-reference example, symmetry and minimizing this quadratic give equal weights; the cap \(1/(\nu n)=1\) allows them.

The standard nu property is stated for an exact solution with nonzero rho. A **strict margin violator** has positive slack, forcing its multiplier to the cap. If there are m such observations,

\[
\frac{m}{\nu n}\le\sum_i\alpha_i=1
\quad\Rightarrow\quad m\le\nu n.
\]

If s support vectors each contribute at most \(1/(\nu n)\) to a sum of 1,

\[
1\le\frac{s}{\nu n}\quad\Rightarrow\quad s\ge\nu n.
\]

These concern strict training violations and nonzero multipliers, not unseen labels. A point exactly on the boundary is not a strict violator. Approximate software results with tiny signed decisions near zero are not exact theorem evidence.

A free support vector, with \(0<\alpha_i<1/(\nu n)\), gives by complementary slackness

\[
\rho=\sum_j\alpha_jK(x_j,x_i).
\]

If there is no free support vector, use appropriate offset bounds instead of this convenient equality. Library coefficient normalizations can differ from the paper formulation; account for them before comparing multiplier values.

## 10. Real monitoring: temperature, change and alert workload

We use the supplied machine_temperature_system_failure.csv from the Numenta Anomaly Benchmark (NAB). It records temperature of an industrial machine's internal component. The source describes a planned shutdown and later failure-related behavior. The supplied annotations contain **four time windows**, not verified fault labels for every row or a documented one-to-one mapping between windows and the prose description.

The [CSV](machine_temperature_system_failure.csv) and [annotation JSON](nab-event-windows.json) are pinned to an upstream commit and supplied offline with the repository's [MIT license](NAB-LICENSE.txt); see [dataset provenance](dataset-provenance.md). The source does not specify temperature units or timestamp timezone. We use **recorded temperature units** and timestamps as supplied.

### Inspect the stream before fitting

There are 22,695 raw rows and 22,683 unique timestamps. Twelve extra rows share timestamps. We explicitly average measurements at the same timestamp, a chosen measurement policy rather than silent deletion.

For level \(v_t\), form

\[
x_t=[v_t,\quad v_t-v_{t-1\mathrm{h}}].
\]

The second feature looks up the exact timestamp one hour earlier; it does not assume that 12 rows earlier always means one hour. Exclude rows without that earlier measurement—12 here. No interpolation or future value fills them in.

This is causal once current-timestamp measurements are available. If duplicate records arrive late, deployment needs a closing delay or revision policy. Retrospective aggregation does not establish zero-latency operation.

| Role | Time interval | Feature rows |
|---|---|---:|
| Fit reference and scaler | Before 6 December 2013 | 885 |
| Calibrate threshold | 6 December through before 10 December | 1,152 |
| Inspect later performance | From 10 December onward | 20,634 |

The early reference is an operating assumption, not certified fault-free because annotation windows start later. Fit the scaler only there; freeze its means and scales for later periods.

### Give a simple baseline the same opportunity

Use absolute level deviation from the reference median divided by reference median absolute deviation (MAD):

\[
A_{\mathrm{base}}(v)=
\frac{|v-\operatorname{median}(v_{\mathrm{fit}})|}
{\operatorname{median}(|v_{\mathrm{fit}}-\operatorname{median}(v_{\mathrm{fit}})|)}.
\]

Here median is about 81.8620 and MAD is 4.42426. We use raw MAD, without a normal-consistency multiplier. A zero MAD requires another declared scale or a constant-reference policy; this dataset's MAD is positive.

The baseline sees level only; learned detectors see level and one-hour change. Keep that representation difference visible. It is not an algorithm comparison on identical features.

All learned detectors share reference rows and scaler. Their settings are fixed for this example: 100 isolation trees, subsample 256, seed 17; RBF One-Class SVM gamma 0.5, nu 0.05; novelty LOF with 20 neighbors. Compare two predeclared calibration quantiles, 0.95 and 0.99, without selecting a winner from test annotations.

### Complete offline analysis

Save this program beside the supplied CSV and JSON. It needs NumPy, pandas and scikit-learn and downloads nothing. Author calculations used Python 3.12.14, NumPy 2.3.5, pandas 3.0.1 and scikit-learn 1.9.1. Later releases may require API and output checks.

~~~python
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

base = Path(__file__).resolve().parent
raw = pd.read_csv(base / "machine_temperature_system_failure.csv",
                  parse_dates=["timestamp"])
level = raw.groupby("timestamp", sort=True)["value"].mean()
previous = level.reindex(level.index - pd.Timedelta(hours=1))
features = pd.DataFrame({
    "level": level.to_numpy(),
    "one_hour_change": level.to_numpy() - previous.to_numpy(),
}, index=level.index).dropna()

fit = features.index < "2013-12-06"
cal = (features.index >= "2013-12-06") & (features.index < "2013-12-10")
test = features.index >= "2013-12-10"
scaler = StandardScaler().fit(features.loc[fit])
x_fit, x_cal, x_test = [scaler.transform(features.loc[m]) for m in [fit, cal, test]]

median = features.loc[fit, "level"].median()
mad = (features.loc[fit, "level"] - median).abs().median()
if mad <= 0:
    raise ValueError("This baseline requires a positive reference MAD.")
scores = {"Baseline": (
    abs(features.loc[cal, "level"].to_numpy() - median) / mad,
    abs(features.loc[test, "level"].to_numpy() - median) / mad,
)}
models = {
    "Isolation Forest": IsolationForest(n_estimators=100, max_samples=256,
        contamination="auto", random_state=17, n_jobs=1),
    "One-Class SVM": OneClassSVM(kernel="rbf", gamma=0.5, nu=0.05),
    "LOF novelty": LocalOutlierFactor(n_neighbors=20, novelty=True,
                                     contamination="auto"),
}
for name, model in models.items():
    model.fit(x_fit)
    scores[name] = (-model.score_samples(x_cal), -model.score_samples(x_test))

windows = json.loads((base / "nab-event-windows.json").read_text())[
    "realKnownCause/machine_temperature_system_failure.csv"]
times = features.index[test]
window_masks = [(times >= pd.Timestamp(a)) & (times <= pd.Timestamp(b))
                for a, b in windows]
inside = np.logical_or.reduce(window_masks)
print("fit/cal/test:", int(fit.sum()), int(cal.sum()), int(test.sum()))
for name, (cal_score, test_score) in scores.items():
    for q in [0.95, 0.99]:
        threshold = np.quantile(cal_score, q, method="higher")
        alert = test_score > threshold
        hits = sum(bool((alert & mask).any()) for mask in window_masks)
        print(f"{name:16s} q={q:.2f} alerts={alert.sum():5d} "
              f"inside={np.sum(alert & inside):4d} "
              f"outside={np.sum(alert & ~inside):4d} windows={hits}/4")
~~~

The fixed author calculation produced:

| Method | Calibration quantile | Test alerts | Inside windows | Outside windows | Windows with an alert |
|---|---:|---:|---:|---:|---:|
| Baseline | 0.95 | 4,416 | 1,379 | 3,037 | 4/4 |
| Baseline | 0.99 | 1,461 | 1,016 | 445 | 4/4 |
| Isolation Forest | 0.95 | 10,530 | 1,722 | 8,808 | 4/4 |
| Isolation Forest | 0.99 | 1,548 | 971 | 577 | 4/4 |
| One-Class SVM | 0.95 | 9,719 | 1,347 | 8,372 | 4/4 |
| One-Class SVM | 0.99 | 9,232 | 1,319 | 7,913 | 4/4 |
| LOF novelty | 0.95 | 8,771 | 1,275 | 7,496 | 4/4 |
| LOF novelty | 0.99 | 7,837 | 1,138 | 6,699 | 4/4 |

The test has 2,268 rows inside windows and 18,366 outside. All methods hit all four windows, yet workload differs greatly. The 0.99 threshold gives 11 calibration alerts out of 1,152 but can give thousands of later alerts. An empirical calibration percentile did not guarantee similar behavior under a changing distribution.

**Outside-window alerts are unmatched workload, not verified false positives.** Inside-window rows are not individually confirmed faults either. We have not run NAB's official scoring, identified precise fault onsets or performed a prospective early-warning study. An alert at a window's beginning is relative to an annotation boundary, not proof of prediction before failure.

[Investigation: view the actual trace with shaded windows. Predict whether moving from the 95th to the 99th calibration percentile loses a window hit. Reveal row alerts, unmatched workload and window hits together. In this run, workload falls while the window-hit count stays unchanged.]

Investigate representation and reference stability instead of tuning until the complex detector beats the baseline. Documented load or operating mode could help if available; we cannot invent those missing variables. Another study could compare level-only and level-plus-change representations on a separate validation period, then lock the choice before evaluating a final future period.

## 11. Deeper: local bounds, resource costs and changing references

### When does local regularity keep LOF near 1?

Suppose all reachability distances needed for a point **and its neighbors' density calculations** lie between positive a and b. Each average reach lies in [a,b], each local density in [1/b,1/a], and each ratio in [a/b,b/a]. Averaging preserves the bounds:

\[
\frac ab\le\operatorname{LOF}\le\frac ba.
\]

When a and b are close, this interval stays near 1. The assumption covers neighbors' own neighborhoods too; checking only the query's distances is insufficient. It explains local regularity, not a universal alert threshold.

### What grows with reference size?

For T isolation trees with capped subsample size psi, the original average-case account is roughly \(T\psi\log\psi\) construction and \(nT\log\psi\) scoring for n rows. Feature processing and tree shape contribute too. Reading a massive dataset still costs work; fixed subsamples do not make the complete application independent of its input size.

LOF needs neighborhoods and local density statistics. A brute-force distance matrix uses quadratic storage. Search indexes can help in favorable dimensions; high-dimensional distances can make both search and interpretation difficult. Approximate neighbors change scores and require a declared tradeoff.

Kernel One-Class SVM can require substantial pairwise-kernel work and storage. Its practical limit depends on data, solver and kernel, not a universal row cutoff. Kernel approximations with a suitable linear one-class learner use a different computation and should be evaluated rather than advertised as identical.

### Updating a reference changes the question

Frozen novelty LOF does not learn from each new batch. Neither does a fitted One-Class SVM or ordinary Isolation Forest. Changed equipment behavior can make a former reference unsuitable.

State when a new reference is allowed, which reviewed observations may enter it, and how old incidents are protected from being normalized away. Compare distributions and workload over time, but stable scores alone do not prove stable fault detection. Keep model, scaler and threshold versions together. Rolling updates must use only information available by that time; hindsight selection of a clean-looking reference leaks future knowledge.

## 12. Practice: explain the mechanism before choosing a label

Each task changes the worked example. Try the question before opening its hint or solution. Round at the final step.

### A. A different isolation gap

Reference values are 0, 2, 3, 4, 10. A first cut is uniform between the minimum and maximum. What is the probability of immediately isolating 10? What is it for 0? Does the larger probability establish that 10 is faulty?

**Hint.** Identify the intervals giving a singleton and divide their lengths by the full span.

**Solution.** Cuts in (4,10) isolate 10, giving 6/10=0.6. Cuts in (0,2) isolate 0, giving 2/10=0.2. Endpoint choices have probability zero in this continuous calculation. This is a geometric separation advantage, not a fault label.

### B. Finish a truncated path

A depth-capped tree was fitted on four rows. A query reaches depth 2 in a leaf with two reference rows. Compute corrected path and normalized score. What goes wrong if the denominator uses a requested max_samples of 256?

**Hint.** c(2)=1 and c(4)=13/6. The denominator describes the actual fitted sample. This state is reachable: successive splits can leave 3 and then 2 rows on the query path.

**Solution.** The path is \(2+1=3\). Its single-tree normalized score is

\[
2^{-3/(13/6)}=2^{-18/13}\approx0.3830.
\]

Using c(256) compares this four-row tree with a much larger reference construction and inflates the score. This is a supplied terminal-state calculation, not a claim that one tree gives a reliable forest estimate.

### C. Change the local reach calculation

A query's three neighbors are at distances 0.4, 0.6, 1.0. Their kth-neighbor radii are 0.3, 0.8, 0.5, and their reference local densities are 1, 2, 1.5. Find the query's density and LOF. Then multiply every distance and radius by 10 while preserving geometry. What happens to all local densities and LOF?

**Hint.** Apply each maximum. Take the reciprocal of average reach, not an average of reciprocals.

**Solution.** Reaches are 0.4, 0.8, 1.0; their mean is 11/15. Query density is 15/11. Mean neighbor density is 1.5, so LOF is \(1.5/(15/11)=1.1\). A common positive scale factor of 10 divides every density, including reference densities, by 10. Ratios remain 1.1. Rescaling only one coordinate of multidimensional data need not preserve the geometry.

### D. Diagnose the invalid comparison

A novelty LOF model is fitted on reviewed references. You compare negative score_samples(reference) with negative negative_outlier_factor_, expecting equality. They differ. Is this necessarily a bug? How should you compare training and later observations?

**Hint.** Draw the neighbors of a new query that equals a training coordinate.

**Solution.** Query scoring may include the coordinate-matching reference row; training LOF excludes its own ID. These are different neighborhood contracts. Use training factors for an in-sample diagnostic and query scores on a separate later calibration/test set. Do not pool them as identical held-out measurements.

### E. Widen the reference anchors

Move the One-Class SVM anchors to −2 and +2, keep nu=1/2, and set gamma=0.25. Derive the midpoint decision. Compare with anchors −1,+1 at gamma=1. Why do they agree?

**Hint.** Squared separation is now 16; squared midpoint distance is 4.

**Solution.** Equal weights still apply. Rho is \((1+e^{-16\gamma})/2=(1+e^{-4})/2\). The midpoint sum is \(e^{-4\gamma}=e^{-1}\), so \(g(0)\approx-0.141278\). Doubling distances and dividing gamma by four preserves every RBF exponent. Units and gamma must be considered together.

### F. A team with 200 review slots

There are 50,000 observations, fault prevalence 0.2%, sensitivity 90%, and false-positive rate 0.5%. Compute expected alerts and precision. Is a 200-review budget sufficient? Does reviewing only the top 200 preserve 90% sensitivity?

**Hint.** Count fault and nonfault populations separately.

**Solution.** There are 100 faults, giving 90 true alerts. Of 49,900 nonfault observations, 249.5 are expected to be flagged. Total expected alerts are 339.5, with precision \(90/339.5\approx26.51\%\). Fractional expected counts describe an average, not a fractional record. The budget is insufficient. Raising the threshold or taking the top 200 changes the operating point, so the old sensitivity cannot simply be carried over.

### G. Training bound or test promise?

An exact One-Class SVM solution has n=80, nu=0.15 and nonzero rho. Give the strict training violation bound and support-vector bound. Does it guarantee at most 12 false alerts in the next 80 observations?

**Hint.** Each multiplier is at most 1/12; their sum is 1.

**Solution.** At most 12 strict training violators and at least 12 support vectors. No future false-alert bound follows: the future population can differ, boundary labels are separate, and the theorem contains no future fault labels.

### H. Write an honest temperature recommendation

Use the 0.99 rows in the real-data table. Which method has the smallest unmatched row workload? How much larger is the One-Class SVM unmatched workload? Why is that insufficient to establish the best fault detector? Write five sentences including a next experiment.

**Hint.** Use the same 18,366 outside-window rows, preserve annotation limits and remember the different feature sets.

**Solution.** The baseline has 445 unmatched alerts; One-Class SVM has 7,913, approximately 17.78 times as many. A suitable report is:

“Under the fixed reference period and 0.99 calibration quantile, all methods alerted in all four published windows. The level-only baseline had 445 outside-window row alerts, compared with 577 for Isolation Forest, 7,913 for One-Class SVM and 6,699 for novelty LOF. These are not verified false positives, and four-window coverage does not measure precise onset or alert usefulness. The baseline uses level, whereas the learned detectors use level plus one-hour change. I would compare matched representations on an additional validation period, investigate operating-regime change, and reserve a later period for a locked final comparison.”

### I. An event metric can hide repeated work

Two non-overlapping event windows cover rows 3–5 and 9–11. A alerts at 3,4,5,9,10,11; B at 3,9,12. Find window hits, row alerts and outside-window alerts. Which has higher event recall under this definition? Can you infer pointwise fault precision?

**Hint.** Extra alerts within one window do not create new events.

**Solution.** Both hit 2/2 windows. A has six row alerts and none outside; B has three row alerts and one outside. Window-hit recall ties. Window annotations do not identify which individual rows were faulty, so pointwise fault precision is undetermined.

### J. Make a stuck sensor visible

A healthy sensor often reports near 40; a stuck sensor reports exactly 40 for three hours. A level-only detector gives ordinary scores. Propose a causal feature and comparison. Name a legitimate state that could resemble the episode.

**Hint.** The unusual property belongs to the sequence.

**Solution.** Use trailing variability or elapsed duration since the last change, computed from current and earlier readings only. Compare with reviewed durations under the same operating mode, considering measurement resolution. A controlled constant process, quantization or scheduled idle state can also be flat. The feature makes the pattern visible without establishing its cause.

## 13. From unusual scores to a probability model

You can now trace scores to paths through random cuts, ratios of local reachability densities or a kernel compatibility boundary. You can separate those scores from thresholds, review workload and annotated events.

The next topic, [Gaussian Mixture Models (GMM) & EM Algorithm](/learn/path/full-curriculum/gaussian-mixture-models-gmm-em-algorithm?module=classical-ml), introduces a learned probability density and soft component responsibilities. They answer different questions. A point far from every component can strongly prefer one component relative to the others while receiving little total density. Turning even a valid density into an alert still needs a reference population, representation and decision policy.

## References and another way to learn

* [Chandola, Banerjee and Kumar, Anomaly Detection: A Survey](https://arindam.cs.illinois.edu/papers/09/anomaly.pdf). The canonical map: begin with section 2 for data, anomaly types, labels and outputs. Its classification, nearest-neighbor and clustering sections position these mechanisms among other families. The authoring review inspected the taxonomy and relevant passages, not every derivation.
* [Liu, Ting and Zhou, Isolation Forest](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf). Read tree construction and path normalization in sections 2–3, then subsampling. The experiments motivate choices rather than guarantee a universally optimal setting.
* [Breunig, Kriegel, Ng and Sander, LOF: Identifying Density-Based Local Outliers](https://sigmodrecord.org/publications/sigmodRecord/0006/pdfs/LOF_%20Identifying%20Density-Based%20Local%20Outliers.pdf). Sections 3–5 define reachability, tied neighborhoods and local bounds. Compare their possibly larger tied neighborhoods with our explicitly fixed-k calculation.
* [Schölkopf and colleagues, Estimating the Support of a High-Dimensional Distribution](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-99-87.pdf). The primal/dual development and nu-property proof explain the training-bound conditions. Best after section 9 rather than as a first introduction to kernels.
* [scikit-learn outlier and novelty guide](https://scikit-learn.org/stable/modules/outlier_detection.html), [IsolationForest API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html), [OneClassSVM API](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html), and [LocalOutlierFactor API](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html). Use these for method availability, orientation and parameter behavior; reviewed release 1.9.1. The [LOF novelty example](https://scikit-learn.org/stable/auto_examples/neighbors/plot_lof_novelty_detection.html) gives another worked frozen-reference/later-query view.
* [Nicolas Goix, Anomaly detection algorithms in Scikit-Learn—recorded talk](https://webcast.in2p3.fr/video/anomaly_detection_algorithms_in_scikitlearn), with [companion slides](https://ngoix.github.io/nicolas_goix_osi_presentation.pdf). A short visual alternative for fitting modes and random isolation. The 15-slide companion was read and recording page checked; the recording was not watched in full. This 2015 resource is for intuition; use current APIs rather than copying historical code.
* [NAB data descriptions](https://github.com/numenta/NAB/blob/ea702d75cc2258d9d7dd35ca8e5e2539d71f3140/data/README.md), [pinned annotation windows](https://github.com/numenta/NAB/blob/ea702d75cc2258d9d7dd35ca8e5e2539d71f3140/labels/combined_windows.json), and [duplicate-timestamp report](https://github.com/numenta/NAB/issues/376). These support the real example's provenance and preprocessing. Supplied offline files and MIT notice are documented beside this lesson.
