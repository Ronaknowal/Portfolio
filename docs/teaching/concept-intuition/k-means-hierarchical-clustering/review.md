# K-means and hierarchy: conceptual-transition review

26 September 2026, author full-source reading. The implemented JSX is the canonical manuscript. The lesson was already unusually strong through the core route, with actual movable centers, geometry, seeding, linkage/cut and palette investigations. Improvements therefore target specific later transitions rather than expanding an already sufficient opening. All programs, ten exercises, capstone, data and numerical results remain unchanged.

| Exact location and hurdles | Disposition |
| --- | --- |
| §1 grouping question, unlabeled data, representative vs tree, k vs KNN, six-point transfer | Retained: real Faithful scatter and six-row example explicitly distinguish descriptive and predictive tasks. |
| §2 assign/mean/iterate, inertia, boundary, representative, ties | Retained: complete P2 distances and mean, live Lloyd steps, mean-optimum figure and tie checkpoint. |
| §3 units/weights, square-root transformation, scaling/centering/constants/categories, held-out geometry | Retained: exact dollars example, controllable geometry and actual geyser disagreement, complete scaling program. |
| §4 separable subproblems, mean proof/weights, hardness scope, D² sampling, zero-total and expected theorem | Retained: local arithmetic and seeding distribution, complete proof and precise expectation boundary. |
| §5 monotonic vs finite stopping, finite configurations, tolerance/empty/ties, bad fixed point, native comparison | Retained: explicit 1/2 sequence and rectangle errors 9 versus 1, complete status-returning code. |
| §6 linkage meanings, alternate fixture, dendrogram axes/cuts, tied-height count, real tree | Retained: four mechanisms drawn, lab's alternate chain changes tree, exact four-group impossibility explained. |
| §7 Ward increment/size effect/height, telescoping, hybrid, MST, inversions | Gap fixed for MST threshold interpretation: diagram on 0,1,2,6 exposes local edges vs distant endpoints. Ward's original movement derivation, exact merge table, connected cost and centroid inversion explanation retained. |
| §8 estimator state, new rows, starts, stopping differences, input-shape contracts | Retained: code and prose map actual fitted quantities, condensed vs square distance trap and lack of hierarchy predict. |
| §9 k complexity, elbow/log scale, silhouette, label-invariant agreement, stability/leakage/gap | Retained: plotted repeated runs, exact one-point silhouette, 4-row label reversal and future-wait information boundary. |
| §10 palette, repeated-color weights, byte rounding, storage, perceptual objective | Retained: complete live palette and stored reconstruction calculation; distinct-color transfer exercise. |
| §11 computational cost, incremental mean, pair storage, BIRCH summaries, IVF routing | Gaps fixed: running-count table 0→3→5 and weighted-arrival explanation; BIRCH exact cost diagram shows internal scatter retained and original split excluded. Existing resource bounds and retrieval recall tradeoff retained. |
| §12 Voronoi limits, alternative objective owners, spherical, kernel, GMM small-variance, high-dimensional caveats | Gaps fixed: unit-vector mean vs renormalized direction and zero-sum case; explicit φ(x)=(x,x²) kernel-distance calculation yielding 8.5 by both routes. Existing hard/soft/tie limit and method ownership remain. |
| §13 practice and §14 capstone | All ten changed tasks, hints/solutions, complete held-out report and independent variations preserved. |

## Research and representations

Consulted the current [scikit-learn clustering guide](https://scikit-learn.org/stable/modules/clustering.html), including the original method definitions and retained mini-batch/BIRCH reference, and [Google's k-means limitations explainer](https://developers.google.com/machine-learning/clustering/kmeans/advantages-disadvantages), 26 September 2026. Used these to locate where a compact overview still needs a visible mechanism. Existing original proofs, ISLR/ESL and annotated MIT transcript references retained. No source's general algorithm-ranking claim adopted.

`KMeansStructureFigures.jsx` adds two static SVG figures, scoped by `k-means-structure.css`:

- Single-link path: positions are quantitatively 0,1,2,6 on one axis; edge labels are 1,1,4. Solid amber denotes threshold-retained edges and dashed neutral the removed edge. Caption explains component membership and complete-link contrast. Located inside the graph connection deeper branch.
- Compressed scatter: the original points 0,2 and their mean 1/new representative 4 use one common spatial scale. Caption exposes N, LS, SS and both equal cost calculations 20. Explicitly distinguishes the restricted assignment objective from re-splitting the original points.
- Running mean table: actual arrivals and counts, not generic step cards. Spherical/kernel bridges use concrete vectors/products; another picture would add little beyond their existing adjacent geometry.

Author parse/arithmetic and retained-source checks are in `author-checks.json`. Independent full reading and actual desktop/phone figures remain pending integration owner. No browser review, fresh fit or universal optimality claim is implied.

Final copy-only follow-through: Renamed the observed program-wrapper label from Before running to Investigate; its code-reading question and all programs remain unchanged.
