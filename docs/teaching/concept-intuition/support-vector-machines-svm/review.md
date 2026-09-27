# SVM: full reading-path review

26 September 2026. Production JSX is the directly authored manuscript; all canonical programs and original executable results preserved.

| Location / hurdle | Before/after disposition |
| --- | --- |
| §1 score, normal, projection, normalized corridor | Existing geometric live lab and projection calculation directly distinguish scale from distance. Retained. |
| §2 active observations, movement, certificates, perturbation bound | Existing one-dimensional analytic fixture and support lab suffice; retained degeneracy conditions. |
| §3 slack → hinge → C → nonunique bias | Table alone made shortfall less visible. Added three common-axis rulers: wrong-side, correct-but-short and beyond-margin. Existing live C/bias lab and sample-count normalization remain. |
| §4 dual lower bound → stationarity → feasible certificate | Added explicit two-sided bound intuition before Lagrangian. Added three computed feasible brackets for the existing C=.25 two-point problem, converging to .375. |
| §4 complementary slackness → support cases → uniqueness | Added pressure/slack interpretation immediately before exact implications; retained correct one-way logic and duplicate-coefficient distinctions. |
| §5 feature map → kernel → PSD | Existing XOR feature map/live contributions sufficient. Added weighted-feature-length reason before PSD condition, rather than presenting a legal-kernel test without motivation. |
| §5 kernel composition, RBF scale/proof, far-field behavior | Preserved complete derivations and interpretation; they extend the now grounded dot-product condition. |
| §6 balanced pair move → box interval → curvature → degenerate pair | Original live SMO lab and opposite-label exact-zero-curvature example sufficient. Retained full solver, bias minimization, gap stopping and cost explanation. |
| §7 model selection and pipelines | Actual fitted C/γ map, train/validation distinction and complete library comparison retained. |
| §8 calibration, weights, multiclass | Existing data-role and score-shape explanations sufficient; retained tested-version contract. |
| §8 precomputed query/training matrix | Added per-query contribution diagram with explicitly owned A/B columns. Same linear predictor can be reconstructed from both rows, exposing why column identity matters. |
| §9 ε-insensitive loss → two-slack dual → target units | Tube lab and exact scalar optimum retained. Added above/below corrective-direction explanation before two dual prices and their difference; retained target-rescaling proof and runnable example. |
| §10 sequence features and lost information | Spectrum diagram, identical-count counterexample and program retained. |
| §10 finite approximations → RFF → Nyström | Added comparison of work location and coordinate construction before the two methods. Added landmark-reference/rescaling explanation before eigenvalue expression. Exact costs, unbiased-kernel expectation versus finite draw and train-only landmark ownership retained. |
| §11 diagnosis, changed transfer/report | Every existing task, hint, explanation, reference and onward topic retained. |

## Diagram specifications and checked numbers

`HingeShortfallFigure`: label-signed scores −.2,.6,1.7 map to losses 1.2,.4,0. Common coordinate scale; amber length ends at requested score 1 and is absent beyond it. Plain text distinguishes score from Euclidean input distance. No draggable handle is implied.

`SvmObjectiveBoundsFigure`: x=[−1,1], y=[−1,1], C=.25, α=[a,a], w=2a,b=0. At a=0: [D,P]=[0,.5]; a=.1: [.18,.42]; a=.25: [.375,.375]. Shared axis 0–.5 with deliberately coincident markers at exact optimum. All points satisfy dual balance/box bounds; full mathematical certificate, not a generalization score.

`PrecomputedKernelFigure`: x_train=[0,2], signed coefficients [−.5,.5], b=−1. Queries [1,3] give K=[[0,2],[0,6]] and scores [0,2]. Distinct references explicitly named in each column; wrapping HTML keeps equations readable. Matrix happens to be square because this illustrative example has two queries and two training points; text and the preserved 5×40 practice explain that query-by-training is the general shape.

## Research and evidence

Read [Cornell's SVM lecture notes](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote09.html) for geometry-to-constrained-objective progression. Existing MIT transcript/Platt/LIBSVM citations remain; this pass does not claim a fresh full video or full paper read. New diagrams use locally derived exact examples. Author parsed changed JSX, checked all hinge values, dual feasibility/brackets/monotone narrowing and kernel-row scores. Existing native solver fits and library results are unchanged and not rerun. Source-bound results: `author-checks.json`; independent and browser closure belongs to root.
