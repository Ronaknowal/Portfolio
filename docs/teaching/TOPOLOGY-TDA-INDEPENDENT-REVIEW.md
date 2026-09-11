# Topology & Topological Data Analysis: independent review

This is the completed independent review of mathematics position 40, `topology-topological-data-analysis-tda`, closed against the author's `2026-09-10T22:39:38.023Z` freeze. It is separate from the author's numerical/browser checks and root's integrated production review. The complete lesson, individual brief, design, nine actual Python programs, model functions, lab components and scoped CSS were read. All six actual production hashes match the author freeze and the final complementary run. No unresolved material finding remains in this bounded review.

## Mathematical and teaching scope

The review checked the progression from modeled spaces and relative openness to simplicial closure, chains over F₂, the boundary-of-boundary identity and quotient homology. A continuous collapse is not confused with a homeomorphism; homotopy equivalence, the coefficient field, and the limits of additive homology are explicit. Rips uses pairwise distance as its scale, while Čech uses balls of half that radius. The equilateral-triangle comparison and restricted-Voronoi qualification avoid confusing clique completion with a common intersection.

The persistence explanation distinguishes induced homology maps from unrelated Betti counts. The actual reducer uses face-compatible order and earlier-column additions. The displayed representative is a cycle belonging to the indicated class, not a unique or shortest geometric loop. Tied computational prefixes are distinguished from sublevel scales, zero-length bookkeeping is identified, and positive intervals retain multiplicity. The complete 2-skeleton suffices for H₀/H₁; its artificial H₂ is not reported as the homology of the full Rips complex. Right-censored bars and the complete filtration's essential classes are separate.

The matching contract allows unmatched copies on both sides, with half-persistence cost to the diagonal. Its finite diagrams do not silently absorb essential points. Fixed-complex filtration stability and paired-point displacement assumptions are stated; the lesson does not turn a small fraction of outliers into a Hausdorff-distance guarantee. Closed pixel cells, corner connectivity, the planar Euler calculation and changed boundary conventions are explicit. The delay-map determinant calculation teaches the actual finite ellipse/segment example without claiming a general reconstruction theorem or temporal direction from an unordered point cloud.

Persistence images integrate weighted Gaussian mass in birth-persistence coordinates, account for the finite window and do not renormalize it into a posterior. Landscapes retain ranked tents rather than adding all bars into one curve. The classification program separates training, validation, final test and later stress diagnostics, includes a simpler baseline, and reports a case where that baseline wins. Mapper edges depend on common observation IDs after within-cover clustering. The visible graph is a 1-skeleton; the lesson explicitly warns when a common intersection supplies omitted higher-dimensional fillings.

All twelve substantial final tasks and their hint/solution progression were read, including changed chains, rectangle scales, diagonal matching, perturbations, pixels, delays, descriptors and Mapper. The programs have visible questions and complete expected output. The exact original connectivity program and output are conserved. No second material mathematical or teaching defect was found in this bounded review.

## One finding and its repair

The first independent run found cancellation in both the exported JavaScript Gaussian interval helper and the actual displayed Python helper. The accepted interval `[0, 1e-20]` returned zero even though its probability mass is representable, approximately `3.9894228040143266e-21`. Consequently, a supported custom image pixel returned zero instead of approximately `3.8079030136375245e-21`. Several other tiny intervals lost relative accuracy. The fixed unit-width browser presets did not expose this edge, but the helpers' admitted domains did.

The author repaired both actual implementations with a narrow-interval midpoint expansion through the fourth derivative, using a log-density/width product and an explicit small-width criterion. The nine displayed programs were regenerated. This reviewer reran the actual code against 85-digit integration, preserving the original failing result separately. All nine changed JavaScript intervals, all nine actual Python helper intervals and the custom pixel pass. The custom pixel's relative error is `1.41e-17`; the original Python tiny interval is positive and agrees to approximately `1.99e-15` relative error. This is an arithmetic repair, not a claim that every arbitrary floating-point input is exact.

The initial failure is retained in `scratch/topology-tda-independent-review/initial-arithmetic-finding.json`; the [durable final evidence](evidence/topology-tda-independent-review.json) embeds both the initial and final results so the history survives scratch cleanup.

## Complementary executable evidence

Run from the application repository:

```powershell
scratch/lesson-tools/Scripts/python.exe -X utf8 -I scripts/verify-topology-tda-independent.py
```

The repair run at `2026-09-10T22:31:00.546784+00:00` passed. The closing run at `2026-09-10T22:40:00.387341+00:00` reran the same checks against all six final frozen sources and passed:

| Independent check | Actual evidence |
| --- | --- |
| Displayed programs and preservation | All nine standalone programs executed and matched exact stdout. The original code/output were compared directly with the archived source blocks and the archived source hash. |
| Changed finite filtrations | Eighteen 4–6-vertex filtrations with tied integer scales; the oracle uses a lowest-pivot bit-vector span calculation, separately from the model's largest-pivot column reducer. |
| Reduction operations | 1,386 actual trace stages preserve boundary-of-recorded-combination, homogeneous dimension, earlier-column use and strict pivot descent. |
| Representative classes | 330 returned cycles are independent of earlier boundaries at creation, survive to the specified death, and become boundaries when paired. Active representatives are independent modulo boundaries at actual scales. Relabeling vertices preserves positive barcode multiplicities. |
| Dimension truncation | The four-vertex 2-skeleton has Betti vector `[1,0,1,0]`; filling the tetrahedron gives `[1,0,0,0]`. |
| Matching/stability transfer | Thirty-six H₀/H₁ comparisons of perturbed point clouds use an independent augmented bipartite matching feasibility oracle, including diagonal dummy matches, and satisfy the stated twice-displacement bound. |
| Mapper and the omitted nerve | Twenty-seven changed cover/clustering states use exact set intersections for edges and memberships. Six include higher common intersections whose triangle boundaries are nonzero in the graph chain space. |
| Gaussian and pixel arithmetic | Nine changed intervals in each actual language implementation and one tiny custom pixel compared with 85-digit quadrature. Actual binary64 endpoints are preserved in the reference rather than replaced by idealized decimal inputs. |

The independent script is semantically named and self-contained: it imports the actual topic-owned models/examples and does not depend on a generated author fixture. Python and its embedded JavaScript were formatted with normalized-AST conservation. It is complementary to the author's broader graph/pixel/rank/matching checks, not a second claim to have executed those checks independently.

## Selected primary-source inspection

The reviewer opened the [ETH persistence notes](https://ti.inf.ethz.ch/ew/courses/TDA25/Chapter4.pdf), particularly sections 4.2 and 4.3.2, to compare the induced-map, basis-choice, tied ordering and matrix-pairing contracts. The lesson properly uses its stated half-open sublevel convention and does not copy an informal essentially-linear runtime assertion from the notes.

The reviewer also inspected [ETH Mapper section 7.3](https://ti.inf.ethz.ch/ew/courses/TDA25/Chapter7.pdf): pullback components, common-point nerves, the point-cloud clustering adaptation and optional 1-skeleton output support the distinctions checked here. The [JMLR persistence-images publication page](https://jmlr.org/papers/v18/16-337.html) was opened; a fresh PDF fetch failed, so a new full-paper read is not claimed. The Gaussian integral was instead checked directly by independent numerical integration. The author's more extensive exact research scope, including the later Hatcher checks, belongs to the [lesson design](TOPOLOGY-TDA-LESSON-DESIGN.md).

## Final identity, visual review and limits

The [author's exact freeze](evidence/topology-tda-author-review.json) matches all six production sources in the independent run. The final body hash is `b5b08bf419a5892ea94421538cb222d7fe4294b5a6c2d12c279bff339bfa0eb1`; the model hash is `f7e6e6363dea8f1a041986b4bb8a807119adf4f3333e91c7d47c781ad059c9fb`. Full examples, lab, CSS and brief hashes are in the durable independent JSON, together with the author packet's hash.

The author's actual-font browser run at `2026-09-10T22:35:40.358Z` passed 458 states at each of 1440, 390 and 320 pixels. That run checks the actual programs, equations, controls, keyboard behavior, reset/back, disclosure and anchor states; it is author evidence, not a duplicate independent browser execution. The reviewer opened nine exact final captures and checked them against the mathematics and ordinary reading flow:

| Capture under `scratch/topology-tda-browser/` | Actual independent visual inspection |
| --- | --- |
| `square-persistence-390.png` | Selected perimeter cycle, H₀ multiplicity, H₁ endpoints, essential-bar separation and dimension disclaimer. |
| `reduction-matrix-320.png` | Current XOR chain and boundary agree with the matrix; narrow matrix panning is explicitly explained. |
| `matching-optimum-390.png` | Distinct copies, selected partners, L∞ squares, diagonal contract and cost 2. |
| `pixel-ring-320.png` | Eight closed pixels, shared edges/vertices, Euler zero and one hole; labels and controls fit. |
| `landscape-image-1440.png` | Ranked tents, integrated pixel contributions, selected rectangle, finite-window mass and color-scale explanation. |
| `mapper-chain-1440.png` | The same point ID appears in two cover clusters and connects the corresponding graph nodes; source and layout coordinates are distinguished. |
| `reading-3-390.png` | The cycle-versus-boundary explanation and F₂ arithmetic arrive before the equation and fit the normal reader. |
| `reading-9-320.png` | The descriptor introduction, ranked-tent mechanism and wrapped equation remain readable on a narrow screen. |
| `inline-1-390.png` | Pairwise-touching balls lack a common triple point; the second scale meets at the shared center with the stated radius convention. |

These files were reopened after the final full capture run; their hashes were recorded immediately and reconfirmed at closure. The independent JSON attributes image creation to the author and actual image inspection to this reviewer. The earlier provisional screenshots and first arithmetic run are not mislabeled as the final version.

Finite tests do not prove the general persistence decomposition, matching-stability or nerve theorems. The review does not establish statistical usefulness on an unseen application, full graduate topology coverage, a learner study or user acceptance. No production source was edited by this reviewer; the author made the reported repair.
