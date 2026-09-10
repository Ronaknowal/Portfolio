# Optimal Transport — independent review

Bounded independent content/model review completed after the author corrected one numerical defect. This is separate from the author's browser verification, integrated review and user acceptance. The reviewer did not edit the owned lesson, examples or models.

Stable ID: `optimal-transport-wasserstein-distance-sinkhorn`, mathematics position 24. Read the full body, eight complete programs and expected outputs, pure models, design and native verification logic. The author's known missing Lagrangian plus signs were already being corrected; the reviewed displayed derivation includes the required additions.

## Finding and resolution

`orderedTransport` previously advanced a source or destination when remaining mass was at most `1e-14`. This discarded a genuinely positive mass. Reproduction: source locations `[0,100]`, weights `[1e-15,1−1e-15]`; target locations `[−100,100]`, weights `[5e-16,1−5e-16]`; order 4. The old code moved only `5e-16` of the source at 0, reporting cost `5e-8`. That source must split into two moves of `5e-16`, both distance 100, giving total `1e-7`. An absolute marginal residual could look small while the relative transport error was large.

The author changed advancement to exhausted mass `<=0` and added the regression. The minimum-and-subtract step already makes at least one active remainder zero, so this does not need a positive truncation threshold for progress. The independent exporter and check were rerun against the corrected model: cost `1.0000000000000001e-7`; both small entries are retained. No unresolved actionable finding remains in this bounded review.

## Mathematical review

- Coupling row/column constraints, two-by-two feasible interval, endpoint optimum and finite price certificate are consistent, including zero weights. Weak duality follows the weighted-slack identity; the Lagrangian signs and route inequalities agree. Attainment/strong duality is confined to the finite feasible bounded problem.
- The rooted Wasserstein definition distinguishes a metric power from the actual cost matrix, retaining order at least 1 and finite-moment qualifications. The finite glued coupling and Minkowski argument explain the triangle inequality without claiming it for arbitrary route costs.
- Weighted monotone matching, uncrossing, CDF area for W1 and the quantile formula for general order are consistent. Unequal weights/counts are not replaced by an equal-count rank shortcut.
- The convention is `Fε = cost + εΣπ(logπ−1) = cost−εH(π)−ε`. Differentiation gives the stated product scaling, the equal-weight logistic reference is correct, and the extra constant is retained in displayed values. Positive active marginals/finite costs justify uniqueness and positive entries; forbidden edges are explicitly outside that simple argument.
- The finite linear-cost bias bound follows from comparing exact objectives and bounding joint entropy between the larger marginal entropy and their sum. It is not applied to stopped infeasible iterates. The maximum-entropy optimal-plan limit and independent-coupling large-ε limit are correctly qualified to the finite problem.
- KL-reference regularization differs by `ε[H(a)+H(b)+1]` from this convention. The three full objectives cancel those marginal-only constants. Subtracting the linear terms alone is not identified with Sinkhorn divergence.
- The nonnegativity/separation paragraph uses the supported bounded-support Euclidean costs and does not promise a general metric or statistical unbiasedness. Independently checked [Feydy et al., equations 1–3 and Theorem 1](https://proceedings.mlr.press/v89/feydy19a/feydy19a.pdf), including its compact-space positive-universal-kernel hypotheses and stated Euclidean extension.
- Barycentric projection is explicitly a conditional mean that generally loses the target law. The finite variance example and changed practice agree with the law of total variance. Displacement interpolation has the correct endpoints, and the constant-speed W2 claim is limited to an optimal Euclidean quadratic coupling.
- The stated Brenier conditions require finite second moments and an absolutely continuous source. The one-dimensional Normal calculation follows from maximal covariance and is correct. These scoped conditions were compared with [Computational Optimal Transport, theorem 2.1 and its surrounding discussion](https://arxiv.org/html/1803.00567v4).
- The particular unbalanced penalty genuinely permits both marginal changes; the single-route positive optimum, derivative and curvature are correct. Finite sliced directions are not equated with the spherical integral; the opposite-diagonal example exposes dependence missed by coordinate marginals. The Gromov objective compares internal distances, is generally nonconvex, and the translated-segment example correctly separates intrinsic from absolute geometry.
- All visible changed numerical practice solutions were checked, including the `1.3` price certificate, rooted units, changed CDF areas, entropy self-value, projection variance and duplication/rescaling acceptance criteria.

## Complementary independent checks

Commands actually executed from repository root:

```powershell
node scratch/export-optimal-transport-independent.mjs
scratch/lesson-tools/Scripts/python.exe scratch/optimal-transport-independent-review/check.py
```

The reviewer used model outputs plus separate SciPy linear programs and directly evaluated entropy/KL identities. The final run passed:

- 24 weighted 3-to-4-location LP comparisons at orders 1.5 and 4, with non-integer/irrational locations and unequal counts/weights. The author suite covers other configurations; these deliberately extend its integer orders.
- 21 entropy-versus-KL objective checks across original, shifted, self and split-atom problems at three ε values. Constants are evaluated independently with SciPy `xlogy`.
- Three complete invariance configurations: adding row/column costs leaves the plan unchanged and shifts the objective by their marginal-weighted sum; splitting every atom in two preserves the aggregated plan and linear cost, changes raw full objective by `−ε log4`, and leaves the consistently debiased divergence unchanged.
- The corrected tiny-mass regression described above.

Largest numerical discrepancy among the complementary checks: `7.46813721974604e-12`, against a `1e-9` tolerance; the tiny-mass cost has its separate `1e-20` tolerance. Artifacts are in `scratch/optimal-transport-independent-review/`: fixtures, check script, results and source hash snapshot. The reviewer read the author's native verification and expected outputs but did not relabel the author's eight-program execution or browser evidence as independently rerun work. Finite numerical checks do not prove the general theorems.

## Reviewed identity

SHA-256 after the model correction:

| Source | SHA-256 |
| --- | --- |
| `src/learn/data/topics/optimal-transport-wasserstein-distance-sinkhorn.jsx` | `a9730a9bd8cc39b800723dd8a8b7722dd106344ab40e858ae1bfd0afa14b7060` |
| `src/learn/data/optimal-transport-models.js` | `9d82310a960a4c7b82ab3009d0839f290e45b76434c18b0d916d52648039bd01` |
| `src/learn/data/optimal-transport-examples.js` | `47fab6c1d8e3e40c68fcf0c7c7fcb990ae514f04e06ed3c7e545c2f8bba130b2` |

The author was still completing narrow equation layout and browser QA during this review. Later formatting must preserve formulas and assumptions; any substantive change needs a targeted follow-up. This record certifies its explicit bounded scope, not future source or user approval.
