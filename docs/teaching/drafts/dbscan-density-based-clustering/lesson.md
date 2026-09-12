# DBSCAN & Density-Based Clustering

Suppose you mark where a survey found flowers along a winding trail. You want groups of nearby observations, including groups that bend with the trail, and you want to leave isolated observations ungrouped. Choosing two centers answers a different question: which center is closest to each observation? DBSCAN instead asks whether observations can be linked through sufficiently crowded neighborhoods.

Later we will return to real flower measurements: the 150-row Iris dataset used in the preceding clustering-evaluation lesson. Can a local-density rule recover useful groups from the four measurements, and how much of the dataset does it leave out? The trail below is an invented hand-calculation example; Iris is measured data. They have different jobs.

**First pass:** read §§1–8, work the neighbor table and border prediction, and run Programs 1–3. Then attempt practice A–E and the Iris report in §13. This route teaches you to explain and apply DBSCAN. Return to §§9–12 for density hierarchies, larger datasets and new observations; those are deeper branches. Allow roughly 60–75 minutes reading for the core and another 60–90 minutes for code and practice. Coordinates, distance comparisons and a Python loop are enough to start. The [K-Means lesson](/learn/path/full-curriculum/k-means-hierarchical-clustering?module=classical-ml) and [Clustering Evaluation](/learn/path/full-curriculum/clustering-evaluation-validation-silhouette-ari-nmi?module=classical-ml) provide useful comparisons; the necessary local ideas are refreshed here.

This is a written lesson awaiting website implementation. Visual callouts identify the planned representations, specified in the accompanying file. Numerical tables explicitly distinguish author calculations from complete-program outputs awaiting phase-two execution.

## 1. Start with a radius and a count

Take ten survey observations on a straight trail. Every row represents one observation, even if two rows eventually have identical coordinates. Their positions, in invented meters relative to a landmark, are:

| ID | A | B | C | D | E | F | G | H | I | J |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| x | −1.75 | −1.50 | −1.25 | −1.00 | 1.00 | 1.25 | 1.50 | 1.75 | 0 | 4 |

Along this line, distance is `abs(x−y)`. In a two-coordinate map it becomes Euclidean distance: square the horizontal and vertical differences, add them, then take the square root. A displacement of 3 meters horizontally and 4 vertically has distance √(9+16)=5 meters.

Choose a radius **ε=1 meter** and a count **m=4 observations**. The library names these `eps` and `min_samples`; papers often call the count `MinPts`. The neighborhood of row p is

\[
N_\varepsilon(p)=\{q:d(p,q)\leq\varepsilon\}.
\]

In words, collect every observation no farther than the radius. **Count p itself. Include a point exactly on the boundary.** We use this convention throughout. A neighborhood is a set of row identities, not a set of unique coordinate values.

At D=−1, the interval is [−2,0]. It contains A,B,C,D and I: five observations. At I=0, the interval is [−1,1]. It contains D,I,E: three. D is a crowded starting point even though I, one of its neighbors, is not.

> **Inline figure F1 — one radius, two counts.** Two aligned trail rows show D's closed interval and roster of five, then I's closed interval and roster of three. Endpoints are filled, and the selected row is counted in each roster. Exact x positions determine spacing; label placement may use leader lines.

We now name three types:

| Type | Test | Role |
| --- | --- | --- |
| **Core** | At least m rows in its own neighborhood | Can extend a cluster through its neighbors. |
| **Border** | Not core, but inside a core row's neighborhood | Joins that cluster; cannot extend it. |
| **Noise** | Neither core nor adjacent to any core | Receives no cluster assignment. |

For the trail, counts are `[4,4,4,5,5,4,4,4,3,1]`. A–H are core. I is border because D and E are core neighbors. J is noise: it has only itself and no core neighbor. In sklearn, noise has label −1. These labels describe the chosen density rule. An observation labeled noise can be a valid rare flower, a sampling artifact or a measurement error; deciding which is the next lesson's anomaly-detection task.

**Pause:** If a row has only three neighbors when m=4, have you established that it is noise? No: you have established only that it is not core. You still need to inspect its core neighbors.

## 2. Build the core graph before assigning the border

A graph is a collection of vertices and links. Make one vertex for each **core** row. Join two core vertices when their distance is at most ε. A connected component is a group in which you can travel between every pair along links.

The trail has two core components: `{A,B,C,D}` and `{E,F,G,H}`. Within each group, the largest separation is .75 meter, so all pairs are linked at ε=1. Between the groups, the closest core pair is D,E, distance 2 meters, so there is no core link between them.

Now attach each noncore row to any component containing one of its core neighbors. I lies exactly 1 meter from both D and E. It can attach to either component, but it cannot join the two components together: I is not a core vertex. J attaches to neither.

> **Inline figure F2 — the bridge that cannot transmit.** Draw the two core components as filled markers and I as a hollow marker between them. Solid lines join core vertices; dotted attachment lines join D–I and I–E. A second row shows the tempting wrong graph, with I used as a transmitting bridge, crossed out. J remains visible at its actual coordinate.

This construction explains the shape freedom. A long chain of core rows can turn a corner or curve around an empty region; it does not need one representative center. Consecutive links must be short, but the two ends of a long component can be much farther apart than ε. Sparse gaps stop expansion only when there is no chain of qualifying core rows across them.

The graph also gives a correctness argument. Starting from a core vertex and visiting its core neighbors repeatedly reaches every core vertex in its connected component: follow any path one edge at a time. It cannot reach a different core component, since that would supply a path connecting the two. Adding adjacent noncore rows afterward preserves this core partition. This is why the graph description and the expansion algorithm below agree.

### Which results depend on input order?

Core, border and noise **types** depend on distances and counts. They stay the same if rows are reordered. The partition of core rows also stays the same. Numeric cluster IDs can be renamed, so compare sets of row IDs rather than asking whether an integer label changed.

A border row shared by several core components is different. Ordinary DBSCAN assigns it to the first component that reaches it. With A visited first, I joins the left group; with H visited first, I joins the right group. The row remains border in both runs. Neither assignment is a mathematical claim that one side is more probable.

> **Investigation L1 — can one row join two groups together?** Record a prediction, initially unanswered, about I's type and number of core components. Move I to a position you choose, or edit a trail coordinate, then apply the change. Inspect the neighbor roster, core graph and final attachments. Reverse the starting order separately. Feedback names the decisive neighbors and distinguishes a genuine component change from a renamed label. The answer for every editable position is not prewritten beside the control.

## 3. Reachability: why the arrows matter

The vocabulary in papers expresses the same mechanism.

- q is **directly density-reachable from p** when p is core and q is in p's neighborhood. The arrow points from the transmitting core to its neighbor.
- q is **density-reachable from p** when there is a chain of these arrows. Every transmitting row along the chain is core; the final row may be border. A zero-step chain lets a row reach itself.
- p and q are **density-connected** when some row o can reach both of them.

D→I is allowed at ε=1,m=4. I→D is not: I has only three neighbors. Thus direct reachability, and reachability in general, need not be symmetric. Density connectivity is symmetric because exchanging p and q does not change “o reaches both.”

There is a subtle reason for keeping the core graph explicit. Density connectivity need not be transitive across all rows. In our trail, D and I are density-connected; I and E are density-connected; D and E are not. The shared border row makes the first two statements true without supplying a core path for the third. On core rows, connected components give the unambiguous equivalence classes. Ordinary DBSCAN then makes a disjoint border-assignment choice.

Some descriptions call each core component plus all its reachable border points a density cluster. Such border-inclusive sets can overlap. Distinguish those mathematical sets from the single integer label per row returned by a usual implementation. The variant **DBSCAN*** keeps only the core components and leaves all noncore rows unassigned. We will meet it again when constructing a density hierarchy.

## 4. A complete small implementation

**Program 1 question:** Can we recover the two trail groups while preventing I from transmitting expansion?

Save this as `trail_dbscan.py` and run `python trail_dbscan.py` with Python 3. It needs only the standard library. The implementation is for small finite coordinate lists, positive ε and positive integer m. It stores the neighborhood lists so the mechanism is inspectable.

Later programs also use NumPy and scikit-learn. In your chosen Python environment, install them with `python -m pip install numpy scikit-learn`, save each complete block as its own `.py` file, then run `python filename.py`. The real-data block specifies its accompanying CSV. The author calculation snapshot used NumPy 2.3.5 and scikit-learn 1.9.1; package-specific output is not promised across every future version.

```python
from math import dist


def dbscan(points, eps, minimum):
    neighbors = [
        [j for j, other in enumerate(points) if dist(point, other) <= eps]
        for point in points
    ]
    core = [len(row) >= minimum for row in neighbors]
    labels = [-1] * len(points)
    cluster = 0

    for seed in range(len(points)):
        if not core[seed] or labels[seed] != -1:
            continue
        labels[seed] = cluster
        pending = [seed]
        while pending:
            current = pending.pop()
            for other in neighbors[current]:
                if labels[other] != -1:
                    continue
                labels[other] = cluster
                if core[other]:
                    pending.append(other)
        cluster += 1

    types = [
        "core" if core[i] else "border" if labels[i] != -1 else "noise"
        for i in range(len(points))
    ]
    return labels, types


positions = [-1.75, -1.5, -1.25, -1, 1, 1.25, 1.5, 1.75, 0, 4]
points = [(x, 0) for x in positions]
labels, types = dbscan(points, eps=1, minimum=4)
for name, label, kind in zip("ABCDEFGHIJ", labels, types):
    print(name, label, kind)
```

Derived result, also matched by the author’s sklearn fixture probe; the exact complete program will be executed during implementation:

```text
A 0 core
B 0 core
C 0 core
D 0 core
E 1 core
F1 core
G 1 core
H 1 core
I 0 border
J -1 noise
```

Only a core row enters `pending`. Every newly assigned row receives its label once, so border rows do not transmit and the finite loop terminates. This code precomputes the types, avoiding a separate visited/noise-candidate state. An on-demand batch expansion can visit a noncore row early and tentatively leave it unassigned; a later core expansion may still assign it as border. “Visited” must not mean “permanently noise.”

Reverse `points`, run again and restore original row order before comparing. The components stay intact while I's attachment changes. Sorting rows is therefore a reproducibility choice, not a repair for every border ambiguity.

## 5. Choose a scale, then inspect what it does

ε has distance units; m is a count. The pair specifies a local crowding rule, not a requested number of clusters. Increasing ε at fixed m expands every neighborhood. A core row stays core, and an assigned row cannot become noise. However, the **number** of clusters can first increase as new core components appear and later decrease as components join.

The trail calculation demonstrates both creation and merging:

| ε, meters | Core count | Border count | Noise count | Core components / returned clusters |
| ---: | ---: | ---: | ---: | ---: |
| .125 | 0 | 0 | 10 | 0 |
| .5 | 4 | 4 | 2 | 2 |
| .75 | 8 | 0 | 2 | 2 |
| 1 | 8 | 1 | 1 | 2 |
| 1.25 | 9 | 0 | 1 | 1 |

These are exact fixture calculations, checked with sklearn 1.9.1. At 1.25, I gains C and F as neighbors. Its count reaches 5; it becomes core and supplies a path between D and E. The repeated one-component result now has a different explanation from “everything is close to a center.”

At fixed ε, raising m can remove core vertices, break components or turn observations into border/noise. It need not reduce the number of clusters. If m=1, every row is core because it counts itself: DBSCAN reduces to connected components of the radius-neighbor graph, with no noise. These limiting cases are valuable checks on your reasoning.

### The neighbor-distance plot has an exact interpretation

For each row p, sort its distances to **all rows including itself**. Let c_m(p) be entry m, counting from 1. Then

\[
p\text{ is core at radius }\varepsilon
\quad\Longleftrightarrow\quad c_m(p)\leq\varepsilon.
\]

For m=4, the trail's c-values in A–J order are `[.75,.5,.5,.75,.75,.5,.5,.75,1.25,2.75]`. This is the same core test as counting a radius neighborhood, viewed in the opposite direction: instead of fixing radius and asking how many, fix count and ask how far.

If m exceeds the total row count, there is no mth entry. Define c_m=∞ for this mathematical comparison: no finite radius can make a core. The displayed neighbor-query program uses m=4 on ten rows, where the requested entry exists.

If a library omits the query row, this is the distance to its (m −1)th other neighbor. With an explicit sklearn query `kneighbors(X)`, the row itself is present, so request `n_neighbors=m` and use column `m-1`. An omitted query `kneighbors()` uses a different self-exclusion convention; do not mix the two calls.

**Program 2 question:** At which exact radii can the trail's core set change?

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

X = np.array([-1.75, -1.5, -1.25, -1, 1, 1.25, 1.5, 1.75, 0, 4])[:, None]
m = 4
search = NearestNeighbors(n_neighbors=m).fit(X)
distances, _ = search.kneighbors(X)
core_radius = distances[:, m - 1]
print(core_radius.tolist())
print(np.sort(core_radius).tolist())
print(int(np.sum(core_radius <= 1)))
```

Derived output: the vector above; sorted vector `[.5,.5,.5,.5,.75,.75,.75,.75,1.25,2.75]`; core count 8. For a real dataset, plot every sorted value against its rank. A bend can suggest a density scale worth trying. The plot does not know the scientific meaning of a group, and some datasets have no distinctive bend. Examine a range of radii and report the corresponding core components, coverage and stability.

> **Inline figure F3 — counts and radii are inverse views.** Align D's five-row roster with its ordered distances 0,.25,.5,.75,1. Highlight the fourth value. Beside it, plot the complete ten-point sorted c_m curve, an ε line and the count beneath it. Keep J's 2.75 value visible while retaining enough vertical resolution to distinguish .5,.75 and 1.25.

## 6. Units and representation are part of the model

If the trail is converted from meters to centimeters, multiply every coordinate **and ε** by 100. Every comparison is preserved, so the core graph is identical. Multiplying coordinates alone keeps the number 1 but changes its meaning from 1 meter to 1 centimeter.

For features with different meanings, use an explicit weighted distance:

\[
d_w(x,z)=\sqrt{\sum_{j=1}^{d} w_j(x_j-z_j)^2},\qquad w_j\geq0.
\]

Multiplying feature j by√w_j turns ordinary Euclidean distance in the transformed coordinates into this distance. Standardization is one choice: subtract each feature's mean and divide by its standard deviation. It gives equal numerical weight to a one-standard-deviation change, which may or may not match your question. In a map measured in meters on both axes, independently standardizing the axes can distort physical proximity. Choose deliberately.

A concrete geometry check uses four corners `(0,0),(1,0),(0,2),(1,2)`, ε=1,m=2. Initially there are two horizontal pairs. Multiply only y by .5, keeping ε=1: every point can reach its horizontal and vertical neighbors, so the radius graph becomes one component. Transform both axes by the same positive factor and ε by that factor: the original result returns. This null distinguishes a change of units from a change of metric.

> **Investigation L2 — change units or change the question?** Record predicted component count, then choose an unsolved feature multiplier/weight and radius. Show actual transformed points with equal-axis scales, the selected row's neighborhood shape and exact distances. Offer a paired unit-conversion action that also converts ε; compare its unchanged graph with one-axis weighting. Reset preserves no committed prediction.

For geographic positions, raw latitude/longitude degrees are not uniform meter coordinates. With a spherical Earth approximation, sklearn's haversine metric takes `[latitude,longitude]` in radians and returns an angle. A 2 km radius corresponds to `2/6371 ≈ .000313922` radians if you explicitly choose Earth radius 6371 km. A suitable local projected coordinate system can instead supply meter coordinates. Geography determines that choice; standardizing latitude and longitude does not fix the distance model.

**Geographic program question:** Can the first three locations connect through short links even though their endpoints exceed the chosen radius?

```python
import numpy as np
from sklearn.cluster import DBSCAN

latitude_longitude_degrees = np.array([[0, 0], [0, .01], [0, .02], [0, 1]])
X = np.deg2rad(latitude_longitude_degrees)
labels = DBSCAN(eps=2 / 6371, min_samples=2,
                metric="haversine", algorithm="ball_tree").fit_predict(X)
print(labels.tolist())
```

Derived result: `[0,0,0,-1]`. Adjacent first-three separations are about 1.112 km; their endpoints are about 2.224 km apart. The chain still connects them. This original equatorial example is a unit calculation, not a geographic benchmark.

## 7. A real question: which Iris measurements form dense groups?

Iris contains 150 flowers, with sepal length/width and petal length/width measured in centimeters,50 from each of three named species. A sepal is the outer flower part below the petals. The data are associated with Fisher's 1936 study of using multiple measurements to distinguish groups. Here we ask a different, unsupervised question: which measurements are crowded together without providing species to the clustering rule?

The supplied [iris.csv](iris.csv) contains row IDs 0–149, those four features in that order and species codes 0=setosa,1=versicolor,2=virginica. It is the corrected Iris copy bundled with sklearn 1.9.1, exported without row reordering; see [provenance](data-provenance.md). UCI distributes Iris under CC BY 4.0. This repeats the preceding lesson's data on purpose: changing the grouping rule while keeping the records recognizable makes the comparison useful.

Our **data unit** is one measured flower. The following fit is a descriptive analysis of this finite collection, not a test of performance on future flowers. All 150 feature rows are available to fit the scaler. Species are held aside until after the density settings have been declared. A predictive task would need a separate frozen reference set and an explicit new-row rule; §12 explains that difference.

Use all four standardized features, m=5, and inspect ε=.3,.5,.8,1 as a declared sensitivity grid. These are teaching settings to investigate, not a claim that the best radius is known in advance. Keep the feature order and metric fixed while comparing radii. Always show the number of returned clusters, noise, core/border counts and **coverage = assigned rows /150**. A one-cluster/all-assigned baseline has 100% coverage; an all-noise result has 0%. Neither alone answers the flower question.

### Program 3: an offline report, including the rejected rows

Put `iris.csv` beside `iris_dbscan.py`. Install NumPy and scikit-learn if needed, then run `python iris_dbscan.py`. The author calculation used NumPy 2.3.5 and sklearn 1.9.1; these are recorded snapshots, not an assertion that all future versions print identically.

```python
from pathlib import Path
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

data = np.genfromtxt(Path(__file__).with_name("iris.csv"), delimiter=",", skip_header=1)
row_ids = data[:, 0].astype(int)
features = data[:, 1:5]
species = data[:, 5].astype(int)
X = StandardScaler().fit_transform(features)

print("eps clusters core border noise coverage silhouette ARI_all")
for eps in [.3, .5, .8, 1.0]:
    model = DBSCAN(eps=eps, min_samples=5).fit(X)
    labels = model.labels_
    assigned = labels >= 0
    groups = np.unique(labels[assigned])
    core = len(model.core_sample_indices_)
    border = int(assigned.sum()) - core
    noise = int((~assigned).sum())
    silhouette = (silhouette_score(X[assigned], labels[assigned])
                  if 2 <= len(groups) < assigned.sum() else float("nan"))
    ari_all = adjusted_rand_score(species, labels)
    print(f"{eps:.1f} {len(groups)} {core} {border} {noise} "
          f"{assigned.mean():.3f} {silhouette:.3f} {ari_all:.3f}")
    print("noise_ids", row_ids[~assigned].tolist())
```

The elementary author probe produced the following summary; this table is the reproducible target for the completed program's later execution. The full row-ID lists are in `author-calculations.json` and should also be inspected in your run.

| ε | Clusters | Assigned group sizes | Noise | Coverage | Silhouette, assigned rows only | ARI, all 150 rows with −1 treated as one label |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| .3 | 3 | 13,12,5 | 120 | .200 | .630 | .088 |
| .5 | 2 | 45,71 | 34 | .773 | .656 | .442 |
| .8 | 2 | 49,97 | 4 | .973 | .598 | .552 |
| 1 | 2 | 49,98 | 3 | .980 | .595 | .554 |

The three-cluster result at .3 retains only 30 flowers. Its cluster count matching the three species is therefore a weak reason to prefer it. At .8, two large dense groups cover 146 flowers. Revealing species after the fit shows a group related to setosa and another joining many versicolor/virginica measurements. The result is a density grouping in the four-feature representation, not a renamed species classifier.

The silhouette compares average distances to the same cluster and the nearest alternative cluster; it is not a centroid calculation. Its conditional rows change between these runs. ARI compares two partitions through pair agreement and adjusts for chance. Passing all labels into ARI explicitly makes the noise rows one predicted group. If you exclude noise, publish the retained IDs and coverage; if comparing two excluded-noise scores, also examine the intersection of retained IDs. These are the reporting conventions from the preceding lesson, applied to a method that can abstain.

> **Investigation L3 — a better score for fewer flowers?** Before revealing species, record whether a learner-chosen radius/count will raise coverage and how many groups they expect. Apply it, inspect the four-feature report and a clearly labeled two-feature projection with the same row IDs. Reveal species only after recording a parameter decision. Then compare conditional scores with coverage, including common-row comparison. The plotting projection is not the space used by the four-feature fit.

## 8. Know when a single radius is the wrong tool

A small radius can miss a diffuse group; a larger radius can join dense groups that you wanted separate. This is a real incompatibility on some datasets, not always a tuning failure.

Consider these three intended trail groups, using m=3:

- Left: `[0,.125,.25,.375]`.
- Middle: `[.75,.875,1,1.125]`.
- Right: `[5,5.75,6.5,7.25]`.

At ε=.25, the first two groups are separate core components and all four right-hand rows are noise. At ε=.375, the first two join through the .375 gap. The right group requires ε≥.75 before either inner row has three neighbors. Therefore no ε can recover all three intended groups with this m: the radius needed by the right group has already joined the left and middle groups.

Now change only the right group to `[5,5.125,5.25,5.375]`. At ε=.25 all three groups appear. This equal-density null shows exactly what caused the earlier conflict.

> **Inline figure F4 and investigation L4 — incompatible intervals.** Three aligned groups share a true-distance axis; below them show “keep left/middle separate: ε<.375” and “make right group viable: ε≥.75.” Their acceptable intervals do not overlap. Learners commit a prediction, edit the right-group spacing or middle-group offset, then seek an overlapping interval and test a chosen radius. A parameter table must accompany the geometry; no axis break may disguise an actual gap.

Use this decision table once you have counted neighborhoods:

| Observation | Inspect next |
| --- | --- |
| Everything is noise | Units, m versus sample size, and the distribution of c_m. With m>n no core row is possible. |
| One large connected component | Core links crossing supposed boundaries; a bridge can connect distant endpoints. |
| Some groups vanish before others separate | Different local densities, feature representation, or a density hierarchy. |
| Repeated measurements create unexpected cores | Whether duplicate rows are distinct events or accidental data duplication; counts embody the data unit. |
| Results change greatly under mild feature changes | Whether the metric has a stable meaning; inspect relevant features and perturbations. |
| A two-dimensional embedding looks convincing but the original neighbors disagree | Which space defines the task. PCA discards directions; other embeddings can distort density. A picture is not a density-preservation guarantee. |

In high dimensions, irrelevant coordinates can dominate distance. For independent noise coordinates with variance σ², two independent rows contribute an expected 2 dσ² to squared distance from d such coordinates. This explains one route by which useful small differences can be overwhelmed. It does not establish that every high-dimensional dataset is equidistant. Feature selection, domain distances and dimensionality reduction are choices to assess; none guarantees meaningful density clusters.

<details><summary>Optional shape comparison: rings rather than centers</summary>

Two concentric rings make the difference between connected groups and center-based groups visible. Construct 12 equally spaced rows on a circle of radius 1 and 36 on radius 3. This is a synthetic geometry experiment, not measured flower data. Adjacent inner/outer gaps are about .518/.523; the gap between rings is at least 2. At ε=.6 and m=3, each row counts itself and its two ring neighbors. Every row is core, each ring is connected, and the rings cannot connect to each other.

Two distinct Euclidean K-Means centers instead divide space by their perpendicular bisector. That straight boundary cannot recover these two concentric rings: a line placing the whole inner ring on one side cannot also put every outer-ring row on the other. This comparison is about the specified ring grouping, not a proof that DBSCAN is preferable for every task.

> **Inline figure F7 — two grouping questions on the same rings.** Equal-scale panels use the identical 48 computed coordinates. One shows DBSCAN's ring components and short neighbor links; the other shows the actual fixed-seed K-Means centers, bisector and memberships. Label the radii and specify that ring identity is a constructed reference. A small retained-row table replaces dense point labels.

**Shape program question:** Do these two rules recover the constructed ring identities?

```python
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score

inner_angle = 2 * np.pi * np.arange(12) / 12
outer_angle = 2 * np.pi * np.arange(36) / 36
inner = np.column_stack([np.cos(inner_angle), np.sin(inner_angle)])
outer = 3 * np.column_stack([np.cos(outer_angle), np.sin(outer_angle)])
X = np.vstack([inner, outer])
ring = np.r_[np.zeros(12, dtype=int), np.ones(36, dtype=int)]
density = DBSCAN(eps=.6, min_samples=3).fit(X)
centers = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
print("core rows", len(density.core_sample_indices_))
print("DBSCAN ring ARI", round(adjusted_rand_score(ring, density.labels_), 3))
print("KMeans ring ARI", round(adjusted_rand_score(ring, centers.labels_), 3))
```

The author fixture calculation gives 48 core rows, DBSCAN ring ARI 1.000 and this K-Means fit's ring ARI −.016. The code's rounded display may print `1.0` rather than `1.000`. These are a constructed-reference comparison, not a general scientific-quality score. The coordinates, both fitted partitions and actual centers are retained in `supplementary-author-calculations.json`; complete-program execution remains a later implementation check.

</details>

## 9. Deeper branch: OPTICS keeps an ordering of density structure

Read this branch after you can compute a core distance. OPTICS separates exploring density structure from extracting one flat clustering.

The core distance c_m(o) is the smallest radius making o core, using our self-inclusive count. OPTICS may limit neighborhood searches by `max_eps`; if o cannot gather m rows within that limit, its core distance is undefined for that run. From an already processed core row o, define a candidate reachability for p:

\[
r(p\mid o)=\max\{c_m(o),d(o,p)\}.
\]

The maximum enforces both requirements: the source must be core at that scale, and the source must reach the destination. Notice which row supplies the core distance: **o**, the source. This is directed information, unlike the symmetric mutual distance in §10.

OPTICS maintains the best candidate reachability discovered for each unprocessed row. It processes a row with the smallest available candidate next, updates nearby candidates using the new row, and restarts when no finite candidate remains. The output is an ordering plus core/reachability values. `labels_` additionally depends on an extraction method; the ordering itself is not a requested flat partition.

For our trail with m=4 and `max_eps=2`, the author calculation gives:

| Ordered ID | A | B | C | D | I | E | F | G | H | J |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reachability | undefined | .75 | .5 | .5 | 1 | 1.25 | .75 | .5 | .5 | undefined |
| Core distance | .75 | .5 | .5 | .75 | 1.25 | .75 | .5 | .5 | .75 | undefined |

The first undefined value marks a restart, not proof of noise. A cut at ε=1 can start a cluster at A because its core distance is .75. It can start another at E: E's reachability is 1.25, but its own core distance is .75. Rows with reachability ≤1 follow the current cluster. A high-reachability row whose core distance is also >1 remains noise. This is why painting every bar above the cut as noise gives the wrong interpretation.

> **Inline figure F5 — high bars can start clusters.** Use the exact ordered table. Show undefined reachability as a separate open restart marker above a finite axis, never as a fake finite height. A/E get explicit “core start” markers under an ε=1 line; J gets “no core start.” The low stretches B–D and F–H form two visible valleys. An accompanying row links every bar to its original ID.

**Program 4 question:** How does the density ordering become labels at a chosen radius?

```python
import numpy as np
from sklearn.cluster import OPTICS, cluster_optics_dbscan

X = np.array([-1.75, -1.5, -1.25, -1, 1, 1.25, 1.5, 1.75, 0, 4])[:, None]
model = OPTICS(min_samples=4, max_eps=2).fit(X)
labels = cluster_optics_dbscan(
    reachability=model.reachability_,
    core_distances=model.core_distances_,
    ordering=model.ordering_, eps=1,
)
print(model.ordering_.tolist())
print(model.reachability_[model.ordering_].tolist())
print(labels.tolist())
```

Expected values are the ordering/reachability table above (`inf` denotes undefined in sklearn) and labels `[0,0,0,0,1,1,1,1,0,-1]`. They were probed with sklearn 1.9.1. Here the result matches ordinary DBSCAN's selected assignment; extraction conventions can differ for border rows in general.

A single horizontal cut still chooses one global scale. The alternative **xi extraction** looks for sufficiently steep relative changes in the reachability profile, governed by `xi`, minimum cluster size and predecessor correction. It can identify nested structures at different scales. For the maintained implementation, use `OPTICS(min_samples=..., cluster_method="xi", xi=..., min_cluster_size=...)`; inspect its ordering as well as labels. xi is a structural extraction choice, not an automatic replacement for explaining what groupings you want. The sklearn 1.9.1 implementation documents quadratic time for its ordering search, so an index alone does not establish an O(n log n) run.

## 10. Deeper branch: HDBSCAN builds and selects a density hierarchy

HDBSCAN offers another way to explore multiple density levels. First increase the cost of connections involving sparse rows. For distinct p,q, define

\[
d_{\mathrm{mr}}(p,q)=\max\{c_m(p),c_m(q),d(p,q)\}.
\]

This **mutual reachability** requires both endpoints to be core at a given scale. In the trail, A–B has ordinary distance .25 but mutual distance .75 because A needs radius .75 to become core. D–I has ordinary distance 1 but mutual distance 1.25 because I needs 1.25. Thus the tempting border bridge is delayed until I can really transmit.

For a threshold ε, retain vertices with c_m≤ε and mutual edges≤ε. The connected components are the core-only DBSCAN* structure at that scale. Vertices with c_m>ε are the sparse ones excluded at that scale. This condition is worth reading directly from the maximum: an endpoint needing a **larger**, not smaller, radius cannot qualify yet.

Instead of retaining all pairwise edges, a **minimum spanning tree (MST)** connects all rows with minimum total edge weight. It preserves the connected components obtained by cutting edges above any threshold. Why? If a low-weight path connected two groups but the tree only connected them through a larger edge, one edge on that path could replace the larger tree edge and reduce total weight, contradicting minimality. Equal-weight alternatives can yield different trees while preserving this threshold connectivity.

Here is the complete small dense-graph construction. It is a teaching step toward the hierarchy, not a full HDBSCAN implementation.

**MST program question:** Which local radius delays the border bridge, and how many edges retain the threshold connectivity?

```python
import numpy as np

X = np.array([-1.75, -1.5, -1.25, -1, 1, 1.25, 1.5, 1.75, 0, 4])[:, None]
D = np.abs(X - X.T)
core = np.sort(D, axis=1)[:, 3]  # fourth entry, including self
W = np.maximum(np.maximum(core[:, None], core[None, :]), D)
np.fill_diagonal(W, 0)

inside = {0}
edges = []
while len(inside) < len(X):
    weight, source, target = min(
        (W[i, j], i, j)
        for i in inside for j in range(len(X)) if j not in inside
    )
    edges.append((source, target, float(weight)))
    inside.add(target)

print(core.tolist())
print(len(edges))
print(float(W[0, 1]), float(W[3, 8]))
```

Derived results: the same ten core distances from §5;9 tree edges; selected weights `.75 1.25`. The identical core distances are deliberate: the neighbor-count view, OPTICS core distances when within its search limit, and mutual-reachability construction use the same local radius definition. The tree code scans candidate edges repeatedly for clarity; use a maintained implementation for large graphs.

### Condense the tree, then choose branches

Let λ=1/ε. Moving toward larger λ means demanding higher density. A component can split, and rows can fall away. `min_cluster_size` governs which branches are large enough to retain in a **condensed tree**. It is different from m, which defines local core distances.

For a branch C born at λ_b, let λ_p be the level at which member p leaves that branch, either falling out or entering a retained child. Its stability is

\[
\operatorname{stability}(C)=\sum_{p\in C}(\lambda_p-\lambda_b).
\]

It counts persistence across density levels, weighted by the rows that remain. Take an illustrative eligible six-row parent born at λ=1 whose rows enter two three-row children at λ=3. Parent stability is 6(3−1)=12. If each child lasts until λ=6, each stability is 3(6−3)=9; selecting both children yields 18, better than 12. If they last only until λ=4, the children total 6 and the parent wins. These numbers are exact arithmetic for a declared abstract condensed tree, not inferred lifetimes for Iris. Treat this parent as a nonroot candidate; a library's root/single-cluster eligibility policy is an additional choice.

The excess-of-mass (`eom`) selection compares a candidate parent's stability with the sum obtainable from its eligible selected descendants, subject to taking a nonoverlapping set of branches. It does not independently choose every stable node: a selected parent and child would double-count their observations. `leaf` selection instead favors terminal retained branches and often gives a finer partition.

> **Inline figure F6 — a lifetime tree with two possible selections.** Put λ on a shared vertical axis with marked levels 1,3,4,6. Width represents row count 6→3+3. Shade either the parent's area 12 or the children's combined area 18/6. Explicit “row count × density-level duration” captions connect the diagram to the sum. On phones stack the two scenarios rather than shrink the labels.

For a library fit using our convention:

**Hierarchy program question:** Which trail rows receive a selected cluster, and how do their membership strengths differ?

```python
import numpy as np
from sklearn.cluster import HDBSCAN

X = np.array([-1.75, -1.5, -1.25, -1, 1, 1.25, 1.5, 1.75, 0, 4])[:, None]
model = HDBSCAN(min_cluster_size=4, min_samples=4,
                cluster_selection_method="eom", copy=True).fit(X)
print(model.labels_.tolist())
print(model.probabilities_.round(3).tolist())
```

The author probe gives labels `[0,0,0,0,1,1,1,1,0,-1]` and membership strengths rounded to three decimals `[1,1,1,1,1,1,1,1,.6,0]`. Full formatted-program execution is deferred to implementation. `probabilities_` describes strength of membership in the selected structure. It is not a calibrated probability that a flower belongs to a biological species or that an observation is safe. The Gaussian-mixture lesson later introduces responsibilities from an explicit probability model.

**Library convention:** sklearn 1.9.1 HDBSCAN includes the query row in `min_samples`; contrib `hdbscan.HDBSCAN` excludes it for this parameter. To align neighbor counts where both settings are valid, sklearn m corresponds to contrib m−1: for example, 4 versus 3. This does not make their entire APIs or every tied result identical. Also choose `min_cluster_size`, `min_samples`, metric and extraction method deliberately. Hierarchical density methods can help with differing densities, but they still encode modeling choices.

## 11. Deeper branch: what costs time and memory?

DBSCAN repeatedly needs radius neighborhoods. With n rows and d features, brute-force distance evaluation across all queries costs O(n² d). A version retaining every neighborhood additionally stores O(n·average_neighbor_count) entries, up to O(n²). A version that queries neighborhoods as needed can use linear auxiliary storage apart from the data/index; storing an n×n distance matrix is not inherent in the definition of DBSCAN.

Spatial indexes can prune distance work on suitable low-dimensional data. A useful way to read a radius query cost is **search overhead plus the neighbors returned**. Returning n neighbors cannot take O(log n) just because a tree was used. Large radii, dense neighborhoods and unhelpful high-dimensional geometry can remove much of the advantage. sklearn DBSCAN bulk-computes neighborhoods, so selecting `ball_tree` does not eliminate the memory for the returned neighborhoods.

At 50,000 rows, one dense float 64 distance matrix alone contains 2.5 billion entries and occupies 20 billion bytes, about 18.6 GiB. That arithmetic is not a measured runtime. Profile the intended data, radius, dimension and metric before picking an implementation; avoid universal row-count thresholds or claimed speedups from a synthetic curve.

Two practical reductions preserve the question when applied correctly:

1. **Compress exact duplicates and retain multiplicity.** If repeated rows truly represent multiple observations, a unique row gets a positive weight equal to its count. The weighted neighborhood sum reproduces the original count. Removing repeats without weights changes density.
2. **Represent a genuinely sparse radius-neighbor graph.** Store distances for pairs within the chosen radius and pass an appropriate sparse precomputed distance graph. Omitting a genuine neighbor can destroy a core or bridge. Missing entries mean absent edges, not zero distance; handle true zero-distance duplicate rows deliberately, commonly through the weighted compression first.

**Duplicate program question:** Can a smaller input preserve three repeated observations without changing their neighborhood count?

```python
import numpy as np
from sklearn.cluster import DBSCAN

X = np.array([[0.], [0.], [0.], [2.]])
unique, inverse, counts = np.unique(X, axis=0, return_inverse=True, return_counts=True)
model = DBSCAN(eps=.25, min_samples=3).fit(unique, sample_weight=counts)
print(model.labels_[inverse].tolist())
print(DBSCAN(eps=.25, min_samples=3).fit_predict(unique).tolist())
```

Derived output: `[0,0,0,-1]` with multiplicities; `[-1,-1]` on the unweighted unique rows. sklearn also permits special negative weights that can inhibit core status; those are not observation multiplicities and fall outside the positive-count interpretation used here.

Partitioned processing adds a second problem: a row near a partition boundary needs neighbors from the other side. A halo of radius ε can supply those neighbors when the partition geometry and distance support such a halo, but correct global results also require global neighbor counts, core classification and component reconciliation across partitions. Running local DBSCAN independently and concatenating labels is insufficient. Approximate neighbors similarly change the model if they miss relevant links. These are implementation tradeoffs to measure, not guaranteed speed gains.

## 12. Deeper branch: what about a new observation?

sklearn DBSCAN has `fit`/`fit_predict`, not a general `predict` method. Its clustering describes the rows present during the fit. Adding a new row and refitting can create a core, attach noise or join two components. That is why assigning a new row is a separate contract.

For example, a frozen-reference extension could examine nearby **training core** rows: return −1 if there are none, return their component if all agree, and declare ambiguity if several components qualify. This assigns against a fixed structure without allowing the new row to change it. It differs from refitting on the combined dataset. The R `dbscan` package provides a particular core-neighbor prediction extension; its existence does not add the same method to sklearn or remove the need to state your policy.

A monitoring workflow should fit its scaler and reference clusters on the allowed reference period, choose a decision/novelty policy on separate calibration data when available, then evaluate the later population. It should not fit transformations on future observations merely because the clustering itself is unsupervised. The next lesson, [Anomaly Detection](/learn/path/full-curriculum/anomaly-outlier-detection-isolation-forest-one-class-svm-lof?module=classical-ml), develops reference sets, scores and decisions. A density noise label alone cannot choose an alert threshold or its cost.

## 13. Practice: change the data, then explain the result

Try each task before opening its hint or solution. Ordinary label numbers may permute; compare row identities and core components.

### A. A new five-row trail

Rows A–E are at `[0,.25,.5,.75,2]`. With ε=.25,m=3, classify every row and give the clusters. Then raise m to 4.

<details><summary>Hint</summary>
List each closed interval's neighbors, including its center. The first and fourth rows need not be core to join.
</details>
<details><summary>Solution</summary>
Counts are 2,3,3,2,1. B/C are core, A/D are border, E noise; one assigned group A–D. At m=4 none is core, so all five are noise. The loss of the core vertices removes the support for the former borders as well.
</details>

### B. Move a shared border off the line

Use the ten-row trail, but move I to `(0,.125)` while the other rows stay at y=0. Keep ε=1,m=4. Does reversing row order still change I's assignment?

<details><summary>Hint</summary>
Compute its distance to D and E before considering order.
</details>
<details><summary>Solution</summary>
Both distances are √(1+.125²)>1, so I is adjacent to neither core component and is noise. A–H remain core and form two components; J is still noise. Reversal can rename components but cannot attach I. This is a null case for the border-order effect: order matters only when a noncore row actually has eligible core neighbors in different components.
</details>

### C. A tempting graph proof

A colleague says: “Density connectivity is symmetric, so connectedness through any intermediate row is transitive and every point gets a unique cluster.” Identify the missing condition with the original trail.

<details><summary>Hint</summary>
Compare D–I, I–E and D–E at ε=1,m=4.
</details>
<details><summary>Solution</summary>
D reaches itself and I; E reaches itself and I. But no core path connects D and E because I cannot transmit. Thus density connectivity through arbitrary rows is not transitive. Restricting to core vertices gives graph components, then handle shared borders separately. Symmetry alone never implies transitivity.
</details>

### D. Duplicates and the count convention

There are three distinct observation rows at x=2 and one at x=5. Use ε=.125,m=3. What happens before and after unweighted deduplication? What happens at m=1 on the original data?

<details><summary>Hint</summary>
Count row identities, then compare that count with the number of coordinate values.
</details>
<details><summary>Solution</summary>
The three repeated rows are core and form one component; x=5 is noise. Unweighted deduplication leaves two rows with only one neighbor each, so both are noise at m=3. Positive weights 3 and 1 restore the original density rule. At m=1 all four rows are core: the three duplicates form one component, x=5 a second, and no row is noise.
</details>

### E. A unit conversion with a false fix

Add a fifth row `(3,0)` to the four corners `(0,0),(1,0),(0,2),(1,2)`. Start with ε=1 and m=2. A colleague converts only the vertical coordinate from meters to centimeters and multiplies ε by 100. Will all neighborhood decisions remain the same? Give a sound transformation. What if the fifth row is absent?

<details><summary>Hint</summary>
The horizontal difference has not grown by 100. Compare the new row's distance to `(1,0)` with the old and new radius.
</details>
<details><summary>Solution</summary>
No for these five rows. Originally `(3,0)` has only itself within radius 1 and is noise; the other rows form two horizontal pairs. After the partial conversion and radius change, its horizontal distance 2 to `(1,0)` is below 100, so it becomes core and joins the lower pair. Uniformly scaling both coordinates and ε preserves every comparison. Alternatively, keep ε in meters and weight the squared centimeter coordinate by 1/10,000, restoring its meter contribution. With only the original four corners, the flawed conversion happens to leave the two horizontal pairs unchanged: horizontal distance 1 remains admitted and vertical distance 200 remains excluded. One unchanged finite result does not prove a transformation preserves distances generally.
</details>

### F. Repair the incompatible density fixture

Keep the dense groups from §8 but move the right group to `[5,5.25,5.5,5.75]`. With m=3, find a radius recovering all three groups and give its usable interval before the dense groups join.

<details><summary>Hint</summary>
The two interior right-group rows become core before its endpoints do. The dense-group gap remains .375.
</details>
<details><summary>Solution</summary>
At ε=.25, the right interior rows have three neighbors and its endpoints attach as border. The first two groups remain separate until ε reaches .375. Thus every ε in[.25,.375) recovers the three groups. At the upper endpoint the closed-boundary core link merges the dense groups. This changed-data repair differs from simply reducing m until every stray pair qualifies.
</details>

### G. A perfect score on the survivors

For standardized Iris at ε=.5, compare m=5 with m=10. Your task is to inspect density groups across the whole collection. Is an assigned-row ARI of 1 sufficient reason to prefer m=10?

<details><summary>Hint</summary>
Keep the original 150 row denominator. Inspect how many flowers each score evaluated.
</details>
<details><summary>Solution</summary>
The author probe gives m=5:116 assigned,34 noise,2 clusters, assigned-row ARI≈.631 and all-row ARI≈.442. For m=10:61 assigned,89 noise,3 clusters, assigned-row ARI 1.000 but all-row ARI≈.279. The perfect result describes only 40.7% of the collection. It can be useful if the stated task deliberately seeks a small unambiguous subset, but does not establish coverage of the whole collection. Publish both populations and evaluate both methods on their common retained IDs if comparing conditional agreement.
</details>

### H. Read an OPTICS cluster start

At cut ε=.6, an ordered row has reachability .9 and core distance .4. The next row has reachability .5. Explain their extraction roles. What if the first core distance were .8?

<details><summary>Hint</summary>
A high reachability value can indicate a start, not just noise.
</details>
<details><summary>Solution</summary>
The first row starts a new cluster: it is not reached at .6 from the preceding expansion, but it is core at .6 itself. The next row can join the current cluster through its .5 reachability. If the first core distance is .8, it cannot start a cluster at .6; it is noise at this step. Assigning the next row then depends on whether a valid current cluster has been established earlier—do not infer the whole extraction from two isolated numbers without ordering state.
</details>

### I. Choose a branch without double-counting

An eight-row parent is born at λ=2; all rows leave for two four-row children at λ=5. Both children persist until λ=7. Compare selecting the parent with selecting both children. Then change only their exit level to 9.

<details><summary>Hint</summary>
Each stability uses its own birth level. Compare sums of disjoint selections.
</details>
<details><summary>Solution</summary>
Parent stability 8(5−2)=24. At exit 7 the children contribute 4(7−5)+4(7−5)=16, so the parent wins. At exit 9 they contribute 32, so the children win. Selecting the parent and both children is inadmissible because their rows overlap. This is a declared condensed-tree exercise; no data fit or claim of a globally correct scientific partition is hidden in its arithmetic.
</details>

### J. Why an index cannot erase a dense output

For 100,000 rows, ε is large enough that every row is a neighbor of every other. Estimate the number of stored directed neighborhood entries and explain why “tree queries are logarithmic” does not solve the memory problem.

<details><summary>Hint</summary>
Multiply the number of queries by the number of results per query.
</details>
<details><summary>Solution</summary>
There are 10¹⁰ entries, including self. Just eight-byte indices require 80 billion bytes, before arrays, distances and interpreter overhead. A search index can reduce the work of finding sparse neighborhoods; it cannot return 10¹⁰ explicit entries in logarithmic total time. Consider an implementation that avoids retaining all neighborhoods, a different task/scale, or valid data reduction. Changing ε only to make a benchmark fast changes the clustering question.
</details>

### K. Independent Iris report

Reproduce Program 3, then change the representation to the four original centimeter features and use ε=.5,m=5. Freeze this choice before revealing species. Write a five-sentence report: data unit/representation, parameter rule, assigned/noise counts, conditional versus all-row comparison, and the next observation you would inspect. Add one radius of your own; justify it from its neighbor-distance plot rather than species.

<details><summary>Hint</summary>
Remove the scaler, keep the same row IDs and feature order, and calculate coverage before interpreting a score. The same numerical radius now has different units.
</details>
<details><summary>One checkable result and an acceptable interpretation</summary>
For the fixed changed setting, the author probe found 2 clusters of 49 and 84 rows,17 noise, coverage 133/150≈.887, assigned-row silhouette≈.735, all-row ARI≈.521 and assigned-row ARI≈.607. An acceptable report says: “I clustered 150 measured flowers in four-dimensional centimeter space. At ε=.5 cm and m=5, two groups retain 133 flowers and 17 are unassigned. The assigned-only geometry is fairly separated in this representation. Species agreement is a separate retrospective comparison, with 17 rows omitted from the conditional score. I would inspect those 17 rows and their core neighborhoods before choosing whether the grouping serves the question.” Your additional radius may produce several defensible conclusions; success requires correct denominators, explicit settings and evidence, not matching a preferred cluster count.
</details>

### L. A frozen reference versus refitting

A new row is within ε of training core rows from both trail components. Specify the result under the “return ambiguous when components disagree” policy from §12. Is that necessarily the result of refitting DBSCAN with the new row?

<details><summary>Hint</summary>
Does the new row contribute to old neighbor counts under both procedures?
</details>
<details><summary>Solution</summary>
The frozen-reference policy returns ambiguity. Refit includes the new observation in neighborhood counts; it can make a formerly noncore row core and merge components, or leave a shared-border assignment depending on its position/count. The two procedures answer different questions. Record which one your future-observation evaluation uses.
</details>

## 14. What to remember, and another way to learn it

DBSCAN groups rows through a graph of locally crowded observations. Radius and count define core rows; core paths define the stable components; neighboring noncore rows attach at the border. Units, sampling and the treatment of unassigned observations are part of the analysis. A density hierarchy explores several scales, and an anomaly workflow adds a separate decision about unusual observations.

You are ready to continue when you can draw a neighborhood with the self/boundary convention, explain why a border cannot transmit, predict a change without confusing label numbers with groups, and produce a real-data report that includes the rejected rows. Use the next Anomaly Detection lesson to move from “not assigned under this density rule” to a stated reference population, score and decision.

**Another explanation or activity**

- [UBC CPSC 330 DBSCAN video, Varada Kolhatkar](https://youtu.be/T4NLsrUaRtg), with [its public lecture companion](https://ubc-cs.github.io/cpsc330-2023W1/lectures/15_DBSCAN-hierarchical.html): a visual beginner revisit after §§1–5, including neighborhood growth and parameter changes. The substantive companion was read; the video was not watched. Use this lesson's precise border/noise rule and reporting conventions rather than the companion's shorthand or its plotting-only replacement for `predict`.
- [Hahsler, Piekenbrock and Doran, “dbscan: Fast Density-Based Clustering with R”](https://www.jstatsoft.org/article/view/v091i01): a free 30-page canonical exposition with algorithm definitions, neighbor-search discussion and worked parameter sensitivity. Good after the core route even if you use Python. The paper's strict-radius notation and R-specific prediction/neighbor conventions differ from the explicit conventions here; its historical performance tables are not current benchmarks.
- [“How HDBSCAN works,” implementation authors](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html): a visual intermediate account of mutual reachability, MST, condensation and stability. Read after §10; it makes the tree operations visible and uses the separate contrib package's conventions.

**Precise references and data**

- [Ester, Kriegel, Sander and Xu 1996, original DBSCAN paper](https://file.biolab.si/papers/1996-DBSCAN-KDD.pdf): §§3–4 introduce the neighborhood/reachability mechanism and expansion algorithm. Its spatial-database experiments are historical evidence for those settings, not the source of any invented modern timing here.
- [sklearn DBSCAN 1.9.1 API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html), [OPTICS](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html), [HDBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html): current parameter meanings, implementation notes and the cross-package HDBSCAN self-count distinction. Recheck the version when reproducing an API-dependent detail.
- [UCI Iris](https://archive.ics.uci.edu/dataset/53/iris), [sklearn loader](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html), and the supplied [data provenance](data-provenance.md): measured feature definitions, license and exact offline copy.
