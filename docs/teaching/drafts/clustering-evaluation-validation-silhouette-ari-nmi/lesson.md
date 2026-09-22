# Clustering Evaluation & Validation (Silhouette, ARI, NMI)

> Current lab UX, 21 September 2026: controls show live calculations and topic-specific visuals without learner prediction entry, grading or guess-to-reveal screens. Genuine algorithm steps, separate practice and data-role boundaries remain. See [the current migration record](../../LIVE-EXPLORATION-CLASSICAL-EARLY.md).

*Content-first manuscript, 12 September 2026. Visual markers refer to the accompanying specifications. Author status: expected numerical results are derived or calculated from the specified fixtures; complete displayed-program verification and runtime/visual implementation are deferred. Approximately 65–80 minutes of reading, plus 60–100 minutes of practice.*

## 1. Two groups look cleaner. Three groups match the species better. Which result should we keep?

Imagine receiving 150 iris specimens. For each one, you have sepal length, sepal width, petal length and petal width. You group specimens using those four measurements, without supplying their species names to the algorithm. A two-group result separates the measurements more cleanly by one geometric score. A three-group result agrees more closely with the recorded species.

That is a real result you will reproduce here. Neither number needs to be wrong. They answer different questions.

**Clustering evaluation means collecting evidence about a proposed grouping. Validation means deciding whether that evidence supports the use you intend.** A grouping useful for choosing representative specimens may differ from one useful for identifying species. Start by naming the decision, then choose the evidence.

You already know how k-means and hierarchies build groups. PCA added another decision: which representation to preserve. This lesson asks how to judge those choices. You will calculate individual scores, inspect whole partitions, challenge a result with controlled changes and finish with a reproducible report.

**First pass:** read sections 1–7, try the silhouette and partition investigations, and run Programs 1 and 2. Continue through the Iris comparison, rejection policy and stability in sections 8–10; choose one of Programs 4–5 to run. Finish with the report task in section 12. The explicitly marked deeper branches on secondary indices, information distances and null-reference model selection can wait. You can follow the core with averages and fractions; the needed probability and logarithm ideas are introduced here.

### Four questions, four kinds of evidence

| Question | What is compared? | Useful evidence |
| --- | --- | --- |
| Are these groups compact and separated in the chosen geometry? | Data distances and one partition | Silhouette, scatter/neighbor inspection, within-group distortion |
| Do these groups agree with a specified reference? | Two partitions of the **same observations** | Contingency table, ARI, NMI/AMI, split/merge diagnosis |
| Would a reasonable change produce similar groups? | Repeated fits or perturbed data on a comparable population | Membership stability, cluster survival, sensitivity by group |
| Does the grouping help the intended task? | A frozen procedure versus a task baseline | Held-out distortion, retrieval utility, reproducible external relationships or a prospective task outcome |

These are complementary questions, not four votes in a contest. A reference can be another algorithm's partition rather than annotated classes; an external comparison does not require a claim that either partition is the truth.

**One caution to carry through the lesson:** every score is conditional on its observation population, representation and comparison rule. State those once in an evaluation record. Then interpret each result within that record. There is no universal silhouette threshold, correct cluster count, or agreement score that substitutes for a task definition.

## 2. Keep the objects fixed before comparing their groups

A **hard partition** assigns each observation to exactly one nonempty group. Its group names are arbitrary. For eight observations A–H, these assignments describe the same partition:

```text
ID          A B C D E F G H
Partition U 0 0 0 0 1 1 1 1
Renamed U   7 7 7 7 3 3 3 3
```

All four first observations remain together, and all four last observations remain together. Raw label accuracy is zero, because none of the numbers match. The grouping is unchanged.

**[Figure V1: shared observation IDs, two aligned groupings and selected pair links.]**

Now change the second row to `0 0 1 1 0 0 1 1`. A and E become neighbors in the second grouping while A and C stop being neighbors. That is a change in membership, not a rename. Good partition-comparison measures detect the second change and ignore the first.

Use stable IDs to align the rows. If one program sorts observations by label while another retains input order, comparing their label arrays by position answers a different, accidental question. Join on IDs and verify that the retained ID sets match before calculating an agreement score.

Our core metrics compare hard partitions. Soft probabilities, overlapping memberships, and unassigned observations require an explicit conversion or a measure designed for those objects. Section 9 handles unassigned observations. A conversion from probabilities to the largest-probability label deliberately discards uncertainty.

## 3. Silhouette: inspect one point's neighborhood before averaging

Take six locations on a number line:

```text
ID        A B C       D E F
Location  0 1 2       7 8 9
Group     L L L       R R R
```

For C, at location 2, ask two questions.

1. How far is C from the **other** members of L? The distances are 2 and 1, whose average is 1.5. Call this within-group average **a(C)**.
2. What is C's average distance to the other group R? Its distances to D, E and F are 5, 6 and 7, averaging 6. Call this **b(C)**.

C's own group is close while the competing group is much farther away. Normalize the difference by the larger average:

\[
s(C)=\frac{6-1.5}{6}=0.75.
\]

**[Figure V2: distance fan for C, with two own-group distances and three other-group distances; average brackets 1.5 and 6. The same IDs appear in the sorted silhouette bars.]**

With more than two groups, first calculate an average distance from C to **each** other group. Then take the smallest of those averages. The competing group is not chosen by its center or by whichever individual point happens to be closest.

For observation i in group C(i), the definitions are:

\[
a(i)=\frac{1}{|C(i)|-1}\sum_{j\in C(i),j\ne i}d(i,j),
\]
\[
b(i)=\min_{G\ne C(i)}\frac{1}{|G|}\sum_{j\in G}d(i,j),
\]
\[
s(i)=\frac{b(i)-a(i)}{\max\{a(i),b(i)\}}.
\]

Here d(i,j) is the declared nonnegative dissimilarity between observations. In this example it is absolute separation on the line, measured in the line's units. Both a and b have those units; their ratio makes s dimensionless.

For positive a or b, the score lies between −1 and 1. If b≥a, then s=1−a/b: the score rises as the own-group average becomes small relative to the alternative. If a>b, then s=b/a−1: the score is negative because another group has the smaller average distance. Negative silhouette tells you exactly this geometric fact. Inspect that point; its sign is not a class-label error detector.

### Read the silhouette plot as a distribution

The six point scores are:

| ID | a | b | s |
| --- | ---: | ---: | ---: |
| A | 1.5 | 8 | .812500 |
| B | 1 | 7 | .857143 |
| C | 1.5 | 6 | .750000 |
| D | 1.5 | 6 | .750000 |
| E | 1 | 7 | .857143 |
| F | 1.5 | 8 | .812500 |

The mean is approximately .806548. The actual silhouette plot gives every observation a horizontal bar, sorted within its group, with a vertical zero line and a line for the overall mean. Group height reveals group size. Bars extending left of zero locate poor within-versus-between distance contrasts. A thick positive band plus a small negative band can have a favorable mean while leaving a small group poorly represented.

The usual overall mean weights observations equally: a group of 100 contributes 100 times as much as a singleton. An equally weighted mean of group means is a different summary. Name it if you use it.

### Conditions and edge cases

Use at least two groups and fewer groups than observations. The standard scikit-learn API rejects a one-group or all-singleton partition. For a singleton inside an otherwise valid partition, use s=0: there is no other own-group member from which to estimate cohesion. If both a and b are zero, use s=0 by convention. A precomputed dissimilarity matrix should have finite nonnegative entries, a zero diagonal and, for our undirected interpretation, symmetry. Similarities where larger means closer must first be turned into a justified dissimilarity.

Multiplying every distance by the same positive constant leaves s unchanged: that factor cancels from numerator and denominator. Changing relative feature weights can change it. Tied competing-group averages give the same b whichever tied group is named; the score remains defined.

### Investigation: change a membership, then explain every moving bar

**[Lab L1: editable points and memberships, distance fan and live silhouette plot.]** Move a point or assign it to another group and follow its silhouette and the overall mean immediately. Follow the links from that point to the observations contributing to a and b. Reset and apply a common scale change as a control experiment.

Try an input of your own rather than only the presets. Which bars change when one point changes groups? The edited observation is only part of the answer: other points now average over different neighbors too.

### Program 1 — calculate the bars

The page's eventual labs are visual models; these Python programs run locally. Use Python 3.12, create a folder and run `python -m venv .venv`. Activate it with `.venv\Scripts\activate.bat` in Windows Command Prompt, or `source .venv/bin/activate` on macOS/Linux. If your launcher is `py -3.12` or `python3.12`, use that to create the environment. Install the dependencies with `python -m pip install numpy==2.3.5 scipy==1.18.1 scikit-learn==1.9.1`. Save each complete program in its own `.py` file and run it with the environment's Python. With no activation, invoke `.venv\Scripts\python.exe` or `.venv/bin/python` directly.

Save this block as `silhouette_walkthrough.py`, then run `python silhouette_walkthrough.py`. It exposes the averaging mechanism, then compares it with the library. The small finite-coordinate examples here are teaching inputs; use maintained library implementations for production data handling.

```python
import numpy as np
from sklearn.metrics import silhouette_samples


def silhouettes(points, labels):
    points = np.asarray(points, dtype=float)
    labels = np.asarray(labels)
    groups = np.unique(labels)
    if not 2 <= len(groups) < len(points):
        raise ValueError("Use 2 to n-1 nonempty groups.")
    distance = np.linalg.norm(points[:, None] - points[None, :], axis=2)
    result = np.zeros(len(points))
    for i, group in enumerate(labels):
        own = labels == group
        own[i] = False
        if not own.any():
            continue
        a = distance[i, own].mean()
        b = min(distance[i, labels == other].mean()
                for other in groups if other != group)
        result[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0
    return result


x = np.array([0, 1, 2, 7, 8, 9.0])[:, None]
groups = np.array([0, 0, 0, 1, 1, 1])
scores = silhouettes(x, groups)
print("scores:", np.round(scores, 6).tolist())
print("mean:", round(float(scores.mean()), 6))
print("matches library:", np.allclose(scores, silhouette_samples(x, groups)))
print("common scale unchanged:", np.allclose(scores, silhouettes(10*x, groups)))
changed = np.array([0, 0, 1, 1, 1, 1])
print("changed C:", round(float(silhouettes(x, changed)[2]), 6))
print("changed mean:", round(float(silhouettes(x, changed).mean()), 6))
```

Expected results:

```text
scores: [0.8125, 0.857143, 0.75, 0.75, 0.857143, 0.8125]
mean: 0.806548
matches library: True
common scale unchanged: True
changed C: -0.75
changed mean: 0.459394
```

After C moves to R, its new own-group mean is 6 while its average to A and B is 1.5. These are exactly the two quantities exchanged in its score: (1.5−6)/6=−.75. The other bars must be recomputed, rather than flipping only C's sign.

**Try:** move C's location from 2 to 3 while keeping the original memberships. Find its new score before running code.

<details><summary>Hint</summary>Its own distances become 3 and 2; its competing distances become 4, 5 and 6.</details>
<details><summary>Explanation</summary>a=2.5, b=5, so s=.5. This changes the geometry, while the previous experiment changed membership. A and B's within averages change too.</details>

## 4. A score evaluates a geometry as well as a partition

PCA taught you that dropping directions changes distances and whitening changes their relative weights. This matters even if you keep the memberships fixed. Suppose one coordinate measures length in centimetres and another records a category code. Treating both as Euclidean axes is already an evaluation choice; a score cannot repair an inappropriate distance.

For numeric coordinates, a weighted Euclidean distance is

\[
d_w(x,y)^2=\sum_r w_r(x_r-y_r)^2,\qquad w_r\ge0.
\]

Scaling coordinate r by √wᵣ implements that weighting. Scaling one coordinate by ten multiplies its squared-distance contribution by 100. Standardization uses variability to set relative scales; it does not establish which measurements matter for a scientific task.

Separate two experiments:

- **Rescore:** hold one partition fixed and change the evaluation geometry. This isolates the score's representation sensitivity.
- **Refit:** change the representation, fit the clustering again, then evaluate. Now both geometry and membership can change.

A full-dimensional, unwhitened orthogonal PCA rotation preserves Euclidean pair distances. Fixed-partition silhouettes must remain the same, up to rounding. A two-dimensional projection or whitening has no such general invariance. This is the direct bridge from PCA's distance geometry to evaluation.

### Shape can conflict with average pairwise separation

Imagine two concentric rings. Distant points around one ring may be farther apart than a point and members of the other ring. A partition into rings can therefore have modest Euclidean silhouette, while slicing the picture into compact left and right groups scores better. Both calculations can be correct. One criterion measures average pairwise compactness; another grouping question might concern connected dense structures.

**[Figure V3: the same ring observations, ring membership versus left/right membership, with computed silhouette distributions; schematic group names are not algorithm predictions.]** For the specified 32 points on radii 1 and 2, the ring partition has mean silhouette about .074594; the left/right partition has about .320613. Both use exactly the same Euclidean distances. These are constructed memberships chosen to expose the criterion, not fitted outputs.

The next lesson will show how density-based algorithms express the second idea. Here, the important move is to inspect the assumed geometry and the question behind a score disagreement. A positive silhouette is not enough to declare the ring slices biologically or semantically correct.

### Deeper branch: what CH, Davies–Bouldin and Dunn measure

Read this branch when you want to select complementary diagnostics rather than accumulate unrelated score columns.

For Euclidean data, let μ be the full-data mean, μⱼ a group's mean and nⱼ its size. Write within- and between-group sums of squares as

\[
W=\sum_j\sum_{x\in C_j}\|x-\mu_j\|^2,
\qquad B=\sum_j n_j\|\mu_j-\mu\|^2.
\]

Expanding x−μ=(x−μⱼ)+(μⱼ−μ), the cross term sums to zero within a group. Therefore total squared variation is W+B. This is the same mean-centering identity behind k-means, now reorganized into an evaluation statistic.

The **Calinski–Harabasz index** is

\[
\mathrm{CH}=\frac{B/(k-1)}{W/(n-k)}.
\]

It rewards large separation of means compared with remaining within-group variation, adjusted by these degrees-of-freedom factors. Its ordinary formula needs 2≤k<n and W>0. For our six points, W=4, B=73.5, n=6 and k=2, giving CH=73.5. Higher CH means stronger separation under this centroid-scatter criterion. Do not read it as an F-test p-value for clusters selected from those same observations.

For **Davies–Bouldin**, define each group's scatter as the average Euclidean distance Sⱼ to its center, and center separation Mⱼₗ=‖μⱼ−μₗ‖. Each group takes its worst competing ratio (Sⱼ+Sₗ)/Mⱼₗ; DB averages those ratios over groups. Lower is favorable. The six-point scatters are both 2/3 and the centers are 7 apart, giving DB=4/21≈.190476. Coincident centers make the ordinary ratio undefined or infinite; inspect that degeneracy instead of treating a library fallback as strong separation.

The basic **Dunn index** divides the smallest cross-group point distance by the largest within-group diameter. Our six-point example gives 5/2=2.5. One bridging point or one extreme within-group distance can dominate it. Different publications use different definitions of inter-group separation or diameter, so report the exact variant.

Silhouette averages distances from each point; CH and DB use group centers/scatter; Dunn uses extremes. Their agreement is informative when their distinct summaries suit the task. They need not agree, and no rule here guarantees a preferred k.

## 5. ARI: count the pairs before correcting for chance

Return to the eight IDs. Let the reference partition be U=`00001111`, and the candidate be V=`00010111`: D and E have exchanged groups. Instead of matching label numbers, inspect the 28 unordered pairs of distinct IDs.

Every pair falls into one of four cases:

| Pair decision | Together in V | Apart in V |
| --- | ---: | ---: |
| Together in U | TP: together in both | FN: split by V |
| Apart in U | FP: merged by V | TN: apart in both |

“Positive” means the pair is together; it does not mean a positive class. The **Rand index**, RI=(TP+TN)/28, is the fraction of pair decisions on which the partitions agree.

Rather than visiting all pairs, count intersections of groups:

| U group \ V group | V0 | V1 | U size |
| --- | ---: | ---: | ---: |
| U0 | 3 | 1 | 4 |
| U1 | 1 | 3 | 4 |
| V size | 4 | 4 | 8 |

A cell containing m observations contributes m(m−1)/2 pairs that are together in both partitions. Let **S** be the sum of these cell pair counts, **A** the sum of reference-group pair counts, **B** the sum of candidate-group pair counts, and **M**=n(n−1)/2 the total pairs.

Here S=3+3=6, A=B=6+6=12 and M=28. Thus TP=S=6, FN=A−S=6, FP=B−S=6, and TN=M−A−B+S=10. RI=16/28=4/7≈.571429.

**[Lab L2: same-ID membership editor connected to contingency cells and a triangular 28-pair board. Rename, swap or split groups and follow the score and pair-board changes immediately.]**

### Why RI's natural chance baseline is not zero

Suppose the reference groups remain size four and four. Randomly assign V's four zero labels and four one labels to the eight fixed IDs. This preserves group sizes but removes their association with U: the **fixed-margin permutation null**.

For any pair already together in U, the probability of also being together in V is B/M. There are A such pairs, so the expected number together in both is E[S]=AB/M. Consequently

\[
E[\mathrm{RI}]=1-\frac{A+B}{M}+\frac{2AB}{M^2}.
\]

For A=B=12, M=28, this expectation is 25/49≈.510204. Observed RI=.571429 is somewhat above that baseline.

RI itself remains interpretable: it describes pairwise agreement. **Adjusted Rand index** asks how much agreement remains after subtracting this chosen chance expectation. Its conventional normalization is

\[
\mathrm{ARI}=\frac{S-AB/M}{(A+B)/2-AB/M}.
\]

For the exchanged D/E example,

\[
\mathrm{ARI}=\frac{6-36/7}{12-36/7}=\frac18=.125.
\]

A value of 1 means the same partition up to names. Zero means S equals its null expectation. Negative means less together-pair agreement than that expectation. The attainable lower bound depends on the partition sizes; a negative result is not a calculation failure.

The normalization uses (A+B)/2 as its standard upper reference for S. When the two size profiles differ, that reference need not be attainable by a contingency table with those exact margins. In particular, “ARI=1” still requires identical nonempty groups, not merely the best possible matching between incompatible group sizes.

Two null models must not be mixed. If each observation independently chooses one of k equally probable labels in each partition, a pair agrees with probability (1/k)²+(1−1/k)². At k=5 that is .68. This independent-assignment model does **not** fix the resulting group sizes. ARI's conditional calculation instead fixes the observed margins. Neither model says that every random realization receives its expectation.

### Program 2 — pair counting, then the compact formula

Save as `partition_pairs.py`. The first computation directly checks pair decisions; the second groups them through a contingency table. Their equality links the visual pair board to the efficient calculation.

```python
from collections import Counter
from itertools import combinations
from math import comb
from sklearn.metrics import adjusted_rand_score, rand_score

u = [0, 0, 0, 0, 1, 1, 1, 1]
v = [0, 0, 0, 1, 0, 1, 1, 1]
cells = Counter(zip(u, v))
rows, columns = Counter(u), Counter(v)
pairs = comb(len(u), 2)
together_both = sum(comb(n, 2) for n in cells.values())
together_u = sum(comb(n, 2) for n in rows.values())
together_v = sum(comb(n, 2) for n in columns.values())
expected = together_u * together_v / pairs
upper = (together_u + together_v) / 2
ari = (together_both - expected) / (upper - expected)
agreements = sum((u[i] == u[j]) == (v[i] == v[j])
                 for i, j in combinations(range(len(u)), 2))
print("S A B M:", together_both, together_u, together_v, pairs)
print("RI:", round(agreements / pairs, 6))
print("ARI:", round(ari, 6))
print("library:", round(rand_score(u, v), 6),
      round(adjusted_rand_score(u, v), 6))
renamed = [9 if label == 0 else 4 for label in v]
print("renamed ARI:", round(adjusted_rand_score(u, renamed), 6))
```

Expected output from the worked arithmetic and author metric probe:

```text
S A B M: 6 12 12 28
RI: 0.571429
ARI: 0.125
library: 0.571429 0.125
renamed ARI: 0.125
```

This short program illustrates a nondegenerate case. When both partitions are the same single group, or both are all singletons, the displayed ARI quotient has a zero denominator; the library uses 1 for perfect agreement. Empty or one-row comparisons have no empirical pair evidence even where an API supplies a convenient identity result. Report the sample count with the metric.

## 6. NMI: how much does one label tell us about the other?

Choose one of A–H uniformly at random. U and V are now two labels attached to that randomly chosen observation. A cell count nᵤᵥ divided by n is the joint probability of seeing those two labels together. Row and column proportions are their marginal probabilities.

**Entropy** H(U)=−Σᵤp(u)log₂p(u) measures average uncertainty in bits. A balanced two-group partition has H(U)=1 bit. A single group has entropy zero: its label conveys no distinction between observations.

**Mutual information**, I(U;V), measures the reduction in uncertainty about U after learning V:

\[
I(U;V)=H(U)-H(U\mid V)
=\sum_{u,v:p(u,v)>0}p(u,v)\log_2\frac{p(u,v)}{p(u)p(v)}.
\]

Here conditional entropy H(U|V) is the average entropy remaining within the V groups. The same quantity also equals H(V)−H(V|U), so MI is symmetric. Its upper bound is min(H(U),H(V)); there is no fixed upper bound independent of the available labels and sample size.

### A refinement makes the distinction visible

Take U=`00001111` and V=`00112233`. V splits each U group into two pure subgroups. Knowing V tells you U exactly, so H(U|V)=0 and I(U;V)=1 bit. Knowing U leaves a choice between two V groups, so H(V|U)=1 bit. H(V)=2 bits.

**[Figure V4: one-bit U split followed by an additional split within each group; aligned contingency tiles show which information is preserved and which distinction V adds.]**

The arithmetic version of **normalized mutual information** is

\[
\mathrm{NMI}_{\rm arithmetic}=\frac{2I(U;V)}{H(U)+H(V)}.
\]

It gives 2/3 for this refinement. The geometric version gives I/√(H(U)H(V))=1/√2≈.707107. Both describe the same partitions with different normalizations. Always name the normalizer; this lesson uses arithmetic NMI, the current scikit-learn default. A `min` normalizer would give 1 to this strict refinement, so a score of 1 under that convention need not mean identical partitions.

NMI uses the empirical table: I=0 means that table factors exactly into its margins. Independent random finite assignments usually produce a table that does not factor exactly. Normalization restricts the range but does not remove that finite-sample association.

### AMI: subtract the association expected under the same fixed margins

Keep both partitions' group sizes and randomly permute one partition over the IDs. Compute MI for those rearrangements. **Adjusted mutual information** subtracts the mean MI of that null experiment:

\[
\mathrm{AMI}=\frac{I-E[I]}{(H(U)+H(V))/2-E[I]}.
\]

This uses the arithmetic convention too. AMI can be negative, equals 1 for identical partitions, and has zero expectation under the nondegenerate fixed-margin null. It is neither a probability nor a significance test. Choose adjustment when comparison against that null is part of the question; choose a declared unadjusted normalization when that is the required descriptive quantity. There is no k=10 boundary at which the meaning suddenly changes.

**[Lab L3: construct two labelings and directly compare observed MI/NMI/AMI with the complete finite permutation distribution for eight IDs.]**

For balanced two-by-two margins on eight IDs there are C(8,4)=70 assignments of the four V0 labels. The overlap of U0 and V0 can be 0,1,2,3,4, occurring 1,16,36,16,1 times. Equal overlap 2 gives an exactly independent table and NMI=0. Complete agreement at overlap 0 or 4 gives NMI=1. Averaging all 70 NMI values gives about .114844, while the mean ARI and AMI are zero. The positive unadjusted baseline emerges from the entire distribution, not a fixed penalty for “many clusters.”

### Program 3 — enumerate the chance experiment

```python
from itertools import combinations
import numpy as np
from sklearn.metrics import (adjusted_rand_score,
                             normalized_mutual_info_score,
                             adjusted_mutual_info_score)

reference = np.array([0, 0, 0, 0, 1, 1, 1, 1])
results = []
overlaps = np.zeros(5, dtype=int)
for chosen in combinations(range(8), 4):
    candidate = np.ones(8, dtype=int)
    candidate[list(chosen)] = 0
    overlaps[np.sum(candidate[:4] == 0)] += 1
    results.append([
        adjusted_rand_score(reference, candidate),
        normalized_mutual_info_score(reference, candidate, average_method="arithmetic"),
        adjusted_mutual_info_score(reference, candidate, average_method="arithmetic")
    ])
means = np.mean(results, axis=0)
print("assignments:", len(results))
print("overlap counts:", overlaps.tolist())
print("mean NMI:", round(float(means[1]), 6))
print("adjusted means close to zero:", np.allclose(means[[0, 2]], 0, atol=1e-12))
```

Expected result, from the author enumeration of these exact 70 assignments:

```text
assignments: 70
overlap counts: [1, 16, 36, 16, 1]
mean NMI: 0.114844
adjusted means close to zero: True
```

The tolerance in the final line handles rounding of a mathematically zero mean. It does not turn every individual adjusted score into zero.

### Deeper branch: calculate E[MI] without enumerating all permutations

Let aᵤ be the size of U group u and bᵥ the size of V group v. Under a random fixed-margin permutation, the overlap R in that cell is hypergeometric:

\[
P(R=r)=\frac{\binom{a_u}{r}\binom{n-a_u}{b_v-r}}{\binom{n}{b_v}},
\quad \max(0,a_u+b_v-n)\le r\le\min(a_u,b_v).
\]

This counts ways to choose bᵥ observations: r from inside Uᵤ and the remainder from outside. For each cell, average its contribution (r/n)log(nr/(aᵤbᵥ)) with these probabilities, treating r=0 as contribution zero. Sum over cells. Linearity of expectation makes this valid even though different cells are dependent. The entropies in the AMI denominator stay fixed because the margins stay fixed.

The logarithm base cancels from NMI and AMI if used consistently. Raw MI from scikit-learn uses natural logarithms, so it is reported in nats; divide by log(2) for bits. This is a unit conversion, not a different association.

For the constant-partition cases, use the library's declared convention: both single-group partitions give NMI=AMI=1; one constant and one nonconstant partition give 0. An implementation that returns 1 whenever either entropy is zero is wrong. A raw formula's zero denominator needs its own case analysis.

## 7. Explain which groups split or merge

An overall agreement score should lead you back to the contingency table. A row spread over several columns shows a reference group split by the candidate. A column receiving observations from several rows shows groups merged by the candidate. Inspect small groups separately: a group of m contributes C(m,2) together-pairs, so large groups can dominate pair summaries. Information summaries also use observation proportions, rather than weighting every group equally.

For U=`00001111` and the refinement V=`00112233`, each predicted group contains only one reference category. Its **purity** is 1, despite splitting each reference group in half. Purity chooses the largest reference count in each candidate group and sums those counts, divided by n. Every-singleton clustering has purity 1 for any reference labeling. This is why purity must be accompanied by evidence about fragmentation.

Two directional information summaries reveal this:

- **Homogeneity**, I/H(U), asks how well the candidate groups determine reference category. It is 1 for the pure refinement.
- **Completeness**, I/H(V), asks how well reference category determines candidate group. It is 1/2 for the refinement.

With the standard zero-entropy conventions, their harmonic mean is **V-measure**. When both entropies are positive, algebra gives V=2I/(H(U)+H(V)), exactly arithmetic NMI. The same 2/3 appears again because these are two descriptions of the same normalization. Homogeneity and completeness name the asymmetry that the symmetric result hides.

**Fowlkes–Mallows** uses pair precision TP/(TP+FP) and pair recall TP/(TP+FN), taking their geometric mean. For the D/E exchange, both are 6/12, so FM=.5. Unlike ARI, FM does not subtract a chance baseline. It can be helpful when the consequences of merging unrelated records and splitting related records need to be discussed directly—for example, grouping duplicate catalogue entries. That is an original teaching application: pairs are records, “together” means a proposed duplicate relationship, and false merges can join different products.

### Deeper branch: matching labels and measuring information lost

Sometimes a task really requires mapping discovered groups to named classes. Maximizing a one-to-one matching through the contingency table then produces a **matched classification accuracy**. It answers a mapping question with a stated constraint. Purity instead permits several discovered groups to map to the same class. If the numbers of groups differ, unmatched groups need a declared handling rule. Do not silently turn either into a general clustering score. Once a mapping is learned for prediction, evaluate that mapping on held-out records rather than rematching to each test set's known answers.

Another comparison is **variation of information**:

\[
\mathrm{VI}(U,V)=H(U\mid V)+H(V\mid U)=H(U)+H(V)-2I(U;V).
\]

It adds the information lost in each direction, so lower is closer. The pure refinement costs one bit: there is no uncertainty about U given V, but one bit about V given U. VI is a metric on finite partitions of a common set; ordinary “1−arithmetic NMI” and chance-adjusted scores should not be assumed to inherit metric properties. Use this branch when a method actually needs a distance between whole partitions, rather than only a reported similarity.

## 8. Real data: separate geometric selection from reference agreement

The supplied [iris.csv](iris.csv) contains 150 specimens, four morphological measurements in centimetres and species labels. Fisher's historical classification data asked how measurements distinguish species. Our task is different: build geometric groups without using species, then compare the resulting partitions with that recorded category. The offline file uses the corrected Iris copy bundled with scikit-learn 1.9.1, with IDs 0–149. [Data attribution and license](data-provenance.md) identify UCI's CC BY 4.0 source and the exact copy.

This first analysis is **descriptive**: all 150 specimens define both the fit and its reported geometry. It does not estimate performance on future specimens. Section 12 gives a separate frozen-report protocol for that purpose.

Use Euclidean k-means with 20 initializations and `random_state=17`. Compare raw measurements, standardized measurements, two unwhitened principal components of standardized data, and two whitened components. Keep species out of fitting and selection. Inspect k=2 and k=3 as candidate resolutions; neither is named “correct” in advance.

### Program 4 — reproduce the disagreements

Save `iris.csv` beside this file, named `iris_evaluation.py`. The column selection deliberately excludes the row identifier and species.

```python
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_score, silhouette_samples,
                             adjusted_rand_score, adjusted_mutual_info_score)

data = np.genfromtxt(Path(__file__).with_name("iris.csv"),
                     delimiter=",", skip_header=1)
x, species = data[:, 1:5], data[:, 5].astype(int)
scaled = StandardScaler().fit_transform(x)
representations = {
    "raw4": x,
    "scaled4": scaled,
    "pca2": PCA(2, svd_solver="full").fit_transform(scaled),
    "white2": PCA(2, whiten=True, svd_solver="full").fit_transform(scaled),
}
for name, z in representations.items():
    for k in [2, 3]:
        labels = KMeans(n_clusters=k, n_init=20, random_state=17).fit_predict(z)
        values = silhouette_samples(z, labels)
        print(name, k, f"sil={values.mean():.6f}",
              f"ARI={adjusted_rand_score(species, labels):.6f}",
              f"AMI={adjusted_mutual_info_score(species, labels):.6f}",
              f"negative={np.sum(values < 0)}")
```

For the declared data and settings, the expected values are:

```text
raw4 2 sil=0.681046 ARI=0.539922 AMI=0.653838 negative=0
raw4 3 sil=0.552819 ARI=0.730238 AMI=0.755119 negative=0
scaled4 2 sil=0.581750 ARI=0.568116 AMI=0.731585 negative=0
scaled4 3 sil=0.459948 ARI=0.620135 AMI=0.655223 negative=2
pca2 2 sil=0.614520 ARI=0.568116 AMI=0.731585 negative=0
pca2 3 sil=0.509168 ARI=0.620135 AMI=0.655223 negative=0
white2 2 sil=0.487163 ARI=0.568116 AMI=0.731585 negative=0
white2 3 sil=0.479638 ARI=0.580252 AMI=0.611417 negative=1
```

**[Figure V5 and Lab L4: real-data evaluation workspace. Separate panels show silhouette and reference agreement; group-size bars and a contingency table remain visible. A fixed-partition rescore mode allows feature-weight changes without refitting.]**

Return to the opening question. In raw measurement space, k=2 has silhouette .681046 versus .552819 for k=3. But species agreement rises from ARI .539922 to .730238. If the task is agreement with species, the three-group result is stronger on these data. If the task is a compact two-prototype summary, the two-group result answers that different constraint. A report should retain the disagreement and explain which decision it serves.

The standardized and two-PC k=3 fits give the same partition up to names in this fixture, even though their silhouettes differ. Comparing their aligned label arrays directly gives ARI=1; every pair has the same together/apart status. Equal agreement with species alone would not establish that identity. The labels used for the external comparison have not changed; the distances used for silhouette have. This makes the rescore/refit distinction tangible. In the lab, freeze the standardized partition and choose your own positive weight for petal width. Watch the silhouette contributions update as the frozen partition is rescored. Then use a common multiplier on every weight as the null control: silhouette should stay fixed.

After examining the aggregate scores, sort silhouette bars within each group and open the species contingency table. Which species gets split? Which candidate group merges reference categories? Read individual specimen IDs and measurements for the low bars. Raw size and representative measurements are often more interpretable than another decimal place in a global score.

When using a two-coordinate scatter to show four-dimensional distances, label it as a **view**. The silhouette calculation still uses its declared feature space. A pair that looks close in the view can differ along an omitted direction.

## 9. Unassigned observations: a cleaner score can describe fewer cases

Some algorithms leave observations unassigned, often encoding that state as −1. The next lesson explains how DBSCAN produces such results. For now, consider the evaluation contract when a program returns them.

Ordinary partition metrics do not attach a special meaning to the number −1. Passing it to ARI treats all −1 observations as one group. Passing it to silhouette treats −1 as a group too. If your intended meaning is “these observations are unrelated rejections,” that is not the same assertion.

You have several legitimate reporting policies, provided you name the question:

1. **All-row partition comparison:** score every row, interpreting each returned label—including −1—as a group. This measures agreement under that encoding.
2. **Conditional assigned-row evaluation:** exclude unassigned rows, then score the remaining population. Report **coverage = assigned rows / all original rows** and the retained IDs.
3. **Separate rejection evaluation:** if a trusted rejection target exists, evaluate that binary decision separately from how the accepted observations are grouped.

**[Figure V6: six observation IDs pass through an assigned-row filter; a population counter and silhouette bars before/after make the denominator change visible.]**

Use the six line points again. If C is rejected, labels become `[0,0,-1,1,1,1]`. Treating −1 as a singleton group gives a mean silhouette about .469841. Dropping C gives about .838314 on five observations, with coverage 5/6. This is a valid improvement in the *conditional* geometric summary. It is not a like-for-like improvement across all six observations.

When comparing two methods that reject different rows, report each method's coverage, then optionally compare both on the **intersection** of their assigned IDs. That common subset fixes the comparison population; it also narrows the question to observations both methods accept. Retain the all-row counts so difficult cases do not disappear from the report. After filtering, check that silhouette still has 2≤k<n; otherwise report “undefined,” not zero.

## 10. Stability: would this grouping survive the change we care about?

An optimizer can return the same answer repeatedly because a dominant feature separates the data. That may be useful, or the dominant feature may be the ID of the recording machine. Stability is evidence about sensitivity, not about which scientific distinction deserves a cluster.

Name the perturbation:

| What changes? | What this experiment asks |
| --- | --- |
| Initialization only, fixed observations | Did the optimizer find different solutions? |
| Observation frequencies or a resample | Does the grouping depend strongly on this sample? |
| Measurement noise within a justified scale | Does plausible measurement error change membership? |
| Features or representation | Is the chosen geometry doing most of the work? |
| New cohort/time window | Does the fitted grouping or its task relationship replicate? |

Do not combine all these changes into one unlabeled spread. Start with one controlled change and add others when the task calls for them. Keep observation IDs aligned throughout.

### Make the resampling effect visible without optimizer randomness

Consider the fixed probe locations `[0,1,4,5,8,9]`. Fit two one-dimensional groups to weighted copies of those locations. To isolate sampling, choose the **globally lowest** weighted squared-error split among the five possible contiguous cuts. This tiny exact solver differs from iterative k-means: it removes initialization as a cause of disagreement.

- With weights `[3,3,1,1,1,1]`, the means are .5 and 6.5. Predicting the six original probes gives `[0,0,1,1,1,1]`.
- With weights `[1,1,1,1,3,3]`, the means are 2.5 and 8.5. The same probes get `[0,0,0,0,1,1]`.

The locations are unchanged. Changing which ones occur more often shifts the best representation and moves the middle pair's group assignment. Comparing the resample's row positions would obscure this mechanism. Comparing predictions on the six fixed IDs shows it.

**[Lab L5: editable multiplicities feed two weighted fits, then two aligned prediction strips on a shared probe ruler. Edit a weight and follow the selected pair’s membership and aligned comparison immediately.]**

The lab also permits k=1. Both fits then assign every probe to the same group and have perfect partition agreement. This null case explains why maximizing stability alone can prefer a trivial answer. The useful question is whether a nontrivial grouping with the needed resolution survives justified perturbations.

### Program 5 — compare the same probes

```python
from itertools import combinations
import numpy as np
from sklearn.metrics import adjusted_rand_score


def exact_line_centers(x, weight, k):
    # x is sorted; this small example has strictly positive weights.
    best = None
    for cuts in combinations(range(1, len(x)), k-1):
        groups = np.split(np.arange(len(x)), cuts)
        centers = tuple(np.average(x[g], weights=weight[g]) for g in groups)
        cost = sum(np.sum(weight[g] * (x[g]-center)**2)
                   for g, center in zip(groups, centers))
        candidate = (float(cost), centers)
        if best is None or candidate < best:
            best = candidate
    return np.array(best[1])


x = np.array([0, 1, 4, 5, 8, 9.0])
weights = [np.array([3, 3, 1, 1, 1, 1]), np.array([1, 1, 1, 1, 3, 3])]
predictions = []
for weight in weights:
    centers = exact_line_centers(x, weight, 2)
    labels = np.abs(x[:, None] - centers).argmin(axis=1)
    predictions.append(labels)
    print("centers:", centers.tolist(), "probe labels:", labels.tolist())
print("probe ARI:", round(adjusted_rand_score(*predictions), 6))
print("one-group ARI:", adjusted_rand_score(np.zeros(6), np.zeros(6)))
```

Expected centers and labels are given above; the two-by-two contingency table has cells 2,0,2,2, with row and column sizes 2/4 and 4/2. Thus S=3, A=B=7, M=15, yielding ARI=−1/14≈−.071429. The final line is 1.0.

For real-data bootstrap stability, sample **training observation IDs with replacement**, refit all learned preprocessing and clustering on that resample, and predict a fixed development probe set. Compare probe labels with the reference fit or with other resamples, stating which. Draws that lose the required number of distinct observations must be counted and handled explicitly, not silently replaced until the result looks stable.

K-means has a natural nearest-center assignment for new probes. A hierarchy does not generally provide a native out-of-sample prediction rule. One alternative is to cluster subsamples and compare memberships on their shared original IDs; another is to define and justify a separate assignment rule. These are different stability experiments. A co-assignment matrix can record how often a pair is together **among runs where that pair is jointly observed**; dividing by all runs would incorrectly count absent observations as separated.

## 11. Deeper branch: tendency, relative selection and computation

The first pass can skip this branch. Return when you must choose an evaluation procedure for a larger analysis or defend a null-reference comparison.

### Does the data support grouping beyond a chosen reference?

An algorithm asked for k groups will usually return k groups even from a smooth cloud. **Cluster tendency** asks whether relevant grouping structure is present before treating those divisions as meaningful. You must specify what “no relevant grouping” would look like for your measurements. Uniformly distributed points in a bounding box, a unimodal correlated cloud and shuffled feature columns are different reference models.

The **gap statistic** compares observed within-group dispersion with dispersion from a chosen null generator. For k-means with squared Euclidean distances, use Wₖ equal to within-cluster SSE. Simulate B reference datasets, apply the same fitting procedure to each, and compute

\[
\operatorname{Gap}(k)=\frac1B\sum_{b=1}^B\log W_{kb}^{*}-\log W_k.
\]

A larger gap means more compactness than that reference produces at the same k. Let sₖ be the simulated log-dispersion standard deviation multiplied by √(1+1/B). A conventional Tibshirani rule picks the first k satisfying Gap(k)≥Gap(k+1)−sₖ₊₁. Include k=1. Declare what happens if no tested k qualifies, and avoid k with zero dispersion because its logarithm is undefined.

For an original teaching table with Gap values `.10,.42,.46,.44` and next-point uncertainties `.03,.08,.06,.07`, k=1 fails because .10<.42−.08. k=2 qualifies because .42≥.46−.06. The rule chooses 2 although the maximum displayed gap is at 3. These values illustrate the decision rule; they are not measured Iris results.

A uniform box can be generated in original coordinates or after an orthogonal PCA rotation. The latter is a change of the reference model's bounding geometry, not PCA whitening. If you use R's `clusGap`, name `spaceH0`, distance power and selection rule: its defaults need not match the squared-SSE and `Tibs2001SEmax` convention above. The maintained [clusGap manual](https://stat.ethz.ch/R-manual/R-devel/library/cluster/html/clusGap.html) explains those choices.

Model-based likelihood criteria such as BIC for Gaussian mixtures answer another question under a probability model. Choose them when that model and its likelihood meaning match the task; the later Gaussian-mixture lesson owns their full derivation. Graph modularity and density-validity measures similarly require their own graph or density definitions.

### A hierarchy has a fidelity question before it has a cut

For a dendrogram, a pair's **cophenetic distance** is the height at which its two observations first join the same branch. Compare that vector of pair heights with the original vector of pairwise distances. Their Pearson correlation, when both vectors vary, measures how well this hierarchy preserves the distance pattern. [SciPy's `cophenet` documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.cophenet.html) states this representation and API.

For line points A=0,B=1,C=3, the original AB/AC/BC distances are `[1,3,2]`. Single linkage merges A/B at 1 and joins C at 2, producing cophenetic values `[1,2,2]`; their correlation is √3/2≈.866025. The hierarchy compresses two different distances into the same merge height. This evaluates the whole dendrogram's distance fidelity, not its agreement with categories or the usefulness of a particular cut. A constant vector makes correlation undefined; choosing k still needs the partition-level questions already developed here.

### What scales, and what changes the quantity?

For n rows and d coordinates, straightforward exact Euclidean silhouette evaluates O(n²d) distance work. A full dense float64 n-by-n matrix occupies 8n² bytes: at n=100,000 this is 80 billion bytes, about 74.5 GiB. Exact computation need not store that entire matrix. The current scikit-learn implementation computes distances in chunks and accumulates group sums; chunking reduces working memory, not the number of relevant distances.

There are two different sampling approaches:

1. **Subset silhouette:** take m rows and compute silhouette entirely inside that subset. This is what `silhouette_score(sample_size=m)` does. It changes both the focal observations and their own/competing groups. Rare groups may vanish or become singletons, and the result is not generally an unbiased estimate of full-data silhouette.
2. **Sample focal rows, retain all comparison rows:** select m focal IDs and compute each one's exact a and b against the full fixed partition. A simple random sample mean of those fixed per-point scores is unbiased for their full-data mean. This takes O(mnd) distance work; report the sampling variability and inspect rare groups separately.

A stratified focal sample can protect small groups. Combine its group means using original group proportions if the target is the observation-weighted full-data mean. Using equal group weights instead targets the equally weighted cluster summary introduced in section 3.

Replacing each group's observations with its center is cheaper, but changes silhouette's definition. In general, distance to a mean is not the mean distance. Call such a result a centroid proxy and evaluate whether that approximation serves your task. Do not label it exact silhouette.

ARI and unadjusted MI avoid listing every pair. After label encoding, count nonzero contingency cells and margins. With hashing, count construction has expected O(n) time; operations over a dense r-by-c table cost O(rc), while sparse forms can exploit occupied cells. A library's sorting-based encoding may add O(n log n) work. AMI additionally sums expected information over possible cell overlaps; it does not share a universal linear-time cost claim with ARI. Benchmark actual data sizes only when timing is the question.

## 12. A report you can defend, then try on your own

For the opening Iris question, a useful report might say:

> I treated the four cm measurements as Euclidean coordinates and fitted k-means with 20 initializations, seed 17. On these 150 specimens, two groups had higher silhouette, while three groups agreed more closely with species. For describing species-related morphology, I would inspect the three-group contingency table and its remaining merges. For a two-representative compression constraint, I would retain two groups and report distortion. This is a descriptive analysis of this dataset.

That statement connects evidence to a decision. It also makes clear what a later analysis must establish if the groups are to be used on new observations.

### Separate development from the final report

If the intended use concerns new specimens, reserve a final report set before selecting the representation, k or interpretation. Use a development fit set to learn preprocessing and centers. A separate selection set can compare candidates. Once the selection rule and chosen pipeline are frozen, apply them to the report set once.

**[Figure V7: specimen-ID flow through fit, selection and frozen report; learned scales/centers travel forward, report labels have no backward arrow.]**

The terms “validation” and “test” differ across papers; the role of the data matters more than the name. Unsupervised fitting can still leak information: estimating scaling, selecting PCA dimensions or choosing k after viewing the final report makes that report part of development. Exploratory analysis of all available data is legitimate when described as exploration, as in section 8.

For a compactness task, one defensible rule is: fix standardized four-feature geometry, try k=2,3,4, require each training group to contain at least 10 specimens, and choose the highest selection-set silhouette. The minimum size is an illustrative operational choice. Report squared distance to the selected training centers on untouched observations against a one-center training-mean baseline. Inspect species agreement separately without using it to choose the model. This ties the task to a defined representation and a natural baseline.

### Program 6 — freeze the decision before opening the report

Save `iris_frozen_report.py` beside the supplied CSV. Follow the 90/30/30 disjoint IDs: preprocessing is fitted on the 90 fit rows, selection uses only the 30 selection rows, and final evaluation uses the remaining 30 rows. A raised “no eligible candidate” is an explicit outcome, not a request to use report results to revise the rule.

```python
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

data = np.genfromtxt(Path(__file__).with_name("iris.csv"), delimiter=",", skip_header=1)
x, species = data[:, 1:5], data[:, 5].astype(int)
order = np.random.default_rng(23).permutation(len(x))
fit_ids, select_ids, report_ids = order[:90], order[90:120], order[120:]
candidates = []
for k in [2, 3, 4]:
    model = make_pipeline(StandardScaler(),
                          KMeans(n_clusters=k, n_init=20, random_state=17))
    model.fit(x[fit_ids])
    selected_labels = model.predict(x[select_ids])
    group_count = len(np.unique(selected_labels))
    sizes = np.bincount(model[-1].labels_, minlength=k)
    if sizes.min() < 10 or not 2 <= group_count < len(select_ids):
        continue
    score = silhouette_score(model[0].transform(x[select_ids]), selected_labels)
    candidates.append((score, -k, model))
if not candidates:
    raise ValueError("No eligible candidate under the development rule.")
selection_score, negative_k, model = max(candidates, key=lambda row: row[:2])

# All choices above are now frozen.
report_labels = model.predict(x[report_ids])
baseline = make_pipeline(StandardScaler(),
                         KMeans(n_clusters=1, n_init=1, random_state=17))
baseline.fit(x[fit_ids])
error = np.mean(model.transform(x[report_ids]).min(axis=1)**2)
base_error = np.mean(baseline.transform(x[report_ids]).min(axis=1)**2)
print("split sizes:", len(fit_ids), len(select_ids), len(report_ids))
print("selected k:", -negative_k, "selection silhouette:", round(selection_score, 6))
print("report distortion / baseline:", round(error, 6), round(base_error, 6))
print("report species ARI:", round(adjusted_rand_score(species[report_ids], report_labels), 6))
```

Expected result for this prespecified fixture:

```text
split sizes: 90 30 30
selected k: 2 selection silhouette: 0.556802
report distortion / baseline: 1.375135 4.673251
report species ARI: 0.628523
```

All three candidates pass the training-size requirement; k=2 wins the selection silhouette comparison. The report distortion falls from about 4.67 for one representative to 1.38 for two. That is evidence for representation quality in the declared standardized geometry; it does not establish species recovery. The separately reported ARI describes that other question.

Because both pipelines fit their scaler on the same fit rows, the two distortion numbers use the same standardized coordinate system. A report-set silhouette, if added, measures distances among report observations grouped by the frozen assignment; a held-out center distortion measures distance to training representatives. These are different held-out summaries. Small group counts and uncertain estimates deserve inspection rather than forced comparison to a threshold.

### Independent practice

Try before opening the explanations. Each problem changes the input, the comparison, or the intended use.

**1. A foreign group's nearest point is misleading.** A selected point has a=2. Distances to foreign group R are `[1,9,11]`; distances to group S are `[4,4]`. Which group determines b and what is s?

<details><summary>Hint</summary>Average within each candidate group before taking a minimum.</details>
<details><summary>Explanation</summary>R's mean is 7; S's is 4. Therefore b=4 and s=.5. Choosing R because it contains the point at distance 1 would reverse the conclusion.</details>

**2. Cross the labels.** For U=`00001111` and V=`00110011`, compute S,A,B,M, RI and ARI. Then explain why NMI is zero while ARI is negative.

<details><summary>Hint</summary>The contingency table is a two-by-two table of twos. Compare observed S with AB/M, not with zero.</details>
<details><summary>Explanation</summary>S=4,A=B=12,M=28. TN=8, RI=12/28=3/7. Expected S=36/7, so ARI=−1/6. Every joint cell has probability 1/4 and each marginal 1/2, so the empirical labels are independent and MI=NMI=0. Under the fixed-margin random experiment, some arrangements have positive MI; AMI is therefore negative too (about −.129745). Independence of this empirical table and chance adjustment are distinct reference points.</details>

**3. Pure fragments.** Give every one of the eight observations its own candidate group. U is still `00001111`. Calculate purity, H(V), MI and arithmetic NMI; predict ARI.

<details><summary>Hint</summary>A singleton ID determines its reference group. But many distinctions in V have no counterpart in U.</details>
<details><summary>Explanation</summary>Purity=1; H(V)=log₂8=3 bits; I=H(U)=1 bit. NMI=2/(1+3)=.5. No pair is together in V, so S=B=0 and ARI=0. Under these fixed margins, every singleton relabeling conveys the same one bit about U, so AMI is 0 apart from rounding. High purity and perfect group granularity are not the same question.</details>

**4. A small group vanishes from the headline.** A group of 90 has mean silhouette .8; a group of 10 has mean −.4. Compute the observation-weighted overall mean and the equally weighted group mean. Which report would you choose for a task that must represent the small group?

<details><summary>Hint</summary>Use 90/100 and 10/100 for the first average, then 1/2 and 1/2.</details>
<details><summary>Explanation</summary>The observation mean is .68, while the group mean is .2. Report both with sizes and the small group's distribution when it has an explicit task role. An equal-group score alone can overemphasize arbitrary fragmentation, so preserve the partition and weighting definition.</details>

**5. Rejecting difficult observations.** Method A assigns all 100 records with silhouette .45. Method B assigns 60 with silhouette .70. A colleague declares B better. Design a fair comparison.

<details><summary>Hint</summary>Name the population behind each mean and the task's cost for rejection.</details>
<details><summary>Explanation</summary>Report 100% versus 60% coverage and each conditional score. Compare both partitions on the same 60 IDs as a separate diagnostic, then examine the rejected 40 and the operational cost of leaving them without a group. If rejection is allowed, predeclare its acceptable coverage/cost. Neither number alone resolves the tradeoff.</details>

**6. Repair the stability code.** Two bootstrap fits each return a length-100 label array, and a script computes ARI between them in sampled-row order. What is wrong? Offer two valid repairs.

<details><summary>Hint</summary>Equal array length does not mean equal observation identity.</details>
<details><summary>Explanation</summary>Bootstrap position 7 can refer to different original IDs in the two fits, and IDs can repeat. Predict a fixed probe set with an agreed assignment rule and compare its aligned labels, or compare unique original IDs in the intersection of subsamples using a defined rule for duplicates. Name the common population and preserve how many observations were actually compared.</details>

**7. Exact changed-input acceptance.** Replace the Program 5 probe locations with `[0,1,2,8,9,10]` and use uniform weights in both fits. Derive the best two-group centers and cost. Then multiply every coordinate by 3.

<details><summary>Hint</summary>Check the five contiguous cuts. Within each three-point run the mean is its middle value.</details>
<details><summary>Explanation</summary>The best cut separates the two runs, with centers 1 and 9 and total squared cost 4. After scaling, centers are 3 and 27, cost 36; memberships and their ARI remain unchanged. Silhouette also stays unchanged because all distances scale by 3. This connects weighted fitting, pair agreement and the earlier distance-ratio invariant.</details>

**8. New-context capstone.** You group museum objects to select 20 items for a compact handling study. Curatorial categories are available; one is rare. Propose your representation, baseline, selection evidence and final report. How would the plan change if the goal became recovering curatorial categories?

<details><summary>Hint</summary>Representative coverage and category recovery need different success criteria even on the same objects.</details>
<details><summary>Example acceptable response</summary>For handling coverage, use measurements relevant to size/material/fragility with justified scales, compare held-out representation error and rare-object coverage against a simple sampling or one-representative baseline, and keep the 20-item budget fixed. Use curatorial categories to inspect coverage without claiming they are the optimization target. For category recovery, reserve annotated examples for evaluation and report contingency rows, ARI/AMI and rare-category split/merge behavior. Learn any mapping or representation choice on development data. A final report identifies observation IDs, rejection rules and selection choices so another person can reproduce the comparison.</details>

### Ready to continue?

You should be able to explain a negative silhouette using actual distances; calculate a pair contingency table and its chance adjustment; distinguish NMI's normalization from AMI's null; show why a rename has no effect; identify the population behind a conditional score; and design a stability or held-out comparison that keeps IDs aligned.

Next is [**DBSCAN & Density-Based Clustering**](/learn/path/full-curriculum/dbscan-density-based-clustering?module=classical-ml). It changes the grouping mechanism from centroid compactness to density connectivity and can leave points unassigned. You now have the tools to explain why its results may disagree with k-means on silhouette, and how to report that disagreement without hiding the rejected observations.

## References & another way to learn it

### Explanations and practice

- **Zaki & Meira, Data Mining and Machine Learning, chapter 17:** [chapter contents](https://dataminingbook.info/toc/) and [author lecture slides](https://www.cs.rpi.edu/~zaki/DMML/slides/pdf/ychap17.pdf). A canonical map of external, internal and relative validation, useful after the core walkthrough. The slides cover more indices than you need for a first report; use the named definitions rather than assuming all score directions/conventions agree.
- **Berkeley Data100, lecture 24.5, “Picking the number of clusters”:** [official lecture page](https://ds100.org/sp21/lecture/lec24/), [video playlist](https://www.youtube.com/playlist?list=PLQCcNQgUcDfqgm0VJbNx-Gqp4bpQ8tzwo), and [companion notebook](https://ds100.org/sp21/resources/assets/lectures/lec23/lec23.html). A visual alternative for elbow reasoning and sorted silhouette bars. The relevant companion cells 74–79 and official item listing were inspected, not video playback. This is archived 2021 material; use this lesson's current API conventions. An earlier dendrogram example in that notebook substitutes synthetic heights, so do not reuse it as a quantitative linkage implementation.
- **von Luxburg, Clustering Stability: An Overview:** [open survey](https://arxiv.org/pdf/1007.1075), especially section 2. A deeper explanation of perturbations and why stability-based model selection is more subtle than choosing a large agreement score. Later asymptotic arguments need more probability than this lesson's finite experiments.
- **Ullmann and colleagues, Validation of cluster analysis results on validation data:** [open paper](https://arxiv.org/html/2103.01281v2), especially its discovery-versus-validation framework. Useful when writing an analysis plan that goes beyond a benchmark score.

### Definitions, implementations and data

- [Scikit-learn evaluation guide](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation), [silhouette API](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html), [Rand/ARI](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html), [NMI](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.normalized_mutual_info_score.html), and [AMI](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_mutual_info_score.html). Consult the exact metric's conventions; the author calculation used version 1.9.1.
- **Vinh, Epps & Bailey, 2010:** [open original paper](https://jmlr.org/papers/volume11/vinh10a/vinh10a.pdf). Sections 3–4 explain normalization, metric properties and chance correction; helpful for the deeper information branch.
- **R cluster package, gap statistic:** [maintainer documentation](https://stat.ethz.ch/R-manual/R-devel/library/cluster/html/clusGap.html). Precise reference-model and one-standard-error rule choices; not a claim that all packages share its defaults.
- **Iris source and offline copy:** [UCI dataset](https://archive.ics.uci.edu/dataset/53/iris), [local CSV](iris.csv), [provenance and license](data-provenance.md). Keep `row_id` and `species` out of the feature matrix.

The design record contains exact research locators and author-check scope. The supplied specifications describe unimplemented visual work; phase two still needs real program executions, numerical/interaction review, readable desktop/mobile rendering and integration.
