# KNN: intuition at every learning transition

26 September 2026. Full production JSX reading and targeted revision, including deeper theory and all twelve practice solutions. Existing topic body is directly authored, with canonical code held in its unchanged example data. Root owns independent/browser closure.

| Location | Assessment and action |
| --- | --- |
| §1 cases → representation/distance → retrieve → combine | Existing lifecycle figure and actual eight-row live map already expose the sequence. Retained. |
| §2 metric family and contours | Coordinate difference example and live equal-distance contour changes sufficient. Corrected missing spaces that interrupted the prose. |
| §2 units → standardization → learned geometry | Existing unit-conversion lab exposes changed identity of the nearest row; train-only formula and zero-variance behavior retained. |
| §2 direction versus magnitude → cosine/normalized Euclidean | Existing vector diagram, norm expansion and complete program sufficient; retained. |
| §3 retrieval versus aggregation → normalized shares | Added explicit masses 1 and 1/3 becoming shares 3/4 and 1/4 before the symbolic definition. Preserved exact-match and class-tie cases. |
| §3 local regression → no extrapolation | Existing adjustable curve and exact changed-membership result sufficient. Retained. |
| §3 Voronoi ownership → half-spaces | Added one-dimensional cancellation example A=0,B=4 giving q≤2 before the general proof. No unnecessary full boundary simulator. |
| §4 complete scratch estimator → tie/scaling/cost contracts | Complete program retained, with existing full-sort versus partial-selection explanation. |
| §5 k and validation → calibration | Retained exact uniform/inverse k=n contrast, complete pipeline, self-neighbor warning and data boundary diagram. |
| §6 KD tree → lower bound → equality handling | Existing spatial search lab, four-row visit trace, independent exact program and tie example sufficient. |
| §6 BallTree alternative | Added query/incumbent/containing-ball diagram and closest-possible-point reasoning before triangle-inequality formula. |
| §7 exact versus approximate search → candidate recall versus task quality | Existing removable-candidate lab directly demonstrates distinction; cost table and held-out workload guidance retained. |
| §8 high-dimensional mass → radius → nuisance versus intrinsic dimension | Existing volume lab, exact s^d calculation and PCA counterexample retained. No universal high-dimensional failure claim added. |
| §9 local Bayes risk → noisy neighbor → asymptotic bound | Added area-accurate independent-label probability diagram at the local argument, prior to distributional theorem. Kept all convergence qualifications and Jensen/multiclass derivations. |
| §9 consistency → rate balance | Added two-job explanation for growing k while k/n shrinks, then noise versus neighborhood-mixing explanation before smoothness/dimension rate calculation. |
| §10 applications: longitude seam, vector outputs, forest representation | Existing complete geographic calculation and target-vector example, leaf encoding and training-boundary explanation sufficient. Retained. |
| §11 failures, radius fallback, imbalance and drift | Existing diagnosis table and causal reasoning retained. |
| §12–13 transfer and onward connection | All twelve exercises, worked capstone and next-topic order retained. New diagrams support existing branch-bound and noisy-label practice. |

## New diagrams

- `NeighborBallBoundFigure`: constructed Euclidean geometry q=(0,0), c=(6,0), radius=2, incumbent=(3,0). Isotropic scale 25 SVG units per distance unit, circle radius 50. Nearest possible group distance 4 exceeds incumbent distance 3; alternate incumbent distance 5 would require search. Amber segment encodes the lower bound, not an actual hidden member. HTML supplies complete interpretation; narrow SVG retains labels.
- `NeighborNoiseFigure`: local η=0.8. Independent neighbor/query labels give .64, .16, .16, .04 joint masses. Rectangle widths and heights encode the .8/.2 marginals, so cells have correct areas. Amber disagreement cells also have printed percentages. Their sum .32 differs from majority rule's .2 risk. This is a specified local probability model, not a finite-sample benchmark or unconditional limit theorem.

## Research actually read

Read the [Cornell CS4780 creator lecture notes](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote02_kNN.html) through its metric, local Bayes-error, convergence and dimensionality explanations. Adopted the useful idea of exposing both label draws; retained this lesson's more careful assumptions rather than copying informal universal assertions or its examples. Read [scikit-learn neighbors](https://scikit-learn.org/stable/modules/neighbors.html), neighbor weighting and KD/BallTree mechanism passages. Its broad complexity summaries are not substituted for this lesson's explicit worst-case accounting. Existing annotated lecture/video links retained; no new claim to have watched the video.

Checked both new calculations, bound equality/alternate state, and changed JSX parsing; hashes in `author-checks.json`. No old native fits rerun, no programs/results removed. Browser and independent review are pending root, not author-certified.
