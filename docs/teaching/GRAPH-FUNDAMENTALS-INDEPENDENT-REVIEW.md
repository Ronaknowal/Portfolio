# Graph Fundamentals — independent finite review

10 September 2026 UTC. Topic `graph-fundamentals-adjacency-laplacian-connectivity`. This is a bounded complementary review of the author-owned lesson; it does not replace the author's native/browser review, parent integration or user acceptance.

## Inspected scope and conclusions

Read the complete current body, all eleven complete displayed Python programs, the pure graph models, all figure/lab render logic, and the individual design and verification records. Checked the changed practice calculations and the reasoning behind source-row adjacency versus incoming aggregation, weighted walk products versus simple paths, reflexive reachability, directed versus undirected components, signed local cancellation versus energy, incidence orientation, component and circulation nullspaces, isolate/loop normalization, conserved means, periodic/lazy updates, anchored uniqueness and the maximum principle. Read the GCN shape/normalization and prediction-time information contracts.

The mathematical proofs and changed exercises are coherent within their stated finite, undirected, nonnegative-weight assumptions. The directed counterexample is valid; the loop convention is declared consistently; anchored uniqueness follows from the zero-boundary energy argument. A stationary degree distribution is not incorrectly used at a zero-volume isolate, and the ordinary and weighted means are kept distinct. The GCN section correctly qualifies the surviving square-root-degree mode and does not infer a universal learned-network limit. No additional mathematical or essential prerequisite blocker was found in this finite read.

## Numerical finding and repair

The original accepted-input model `graphNormalizations(2, [[0,1,1e-310]])` formed infinite reciprocal degrees and consequently nonfinite row-normalized/transition entries, although the two-node normalized matrices have finite mathematical entries. The actual displayed Python helper `normalized_operators([[0,1e-310],[1e-310,0]])` likewise returned `[[nan, inf], [inf, nan]]` for its transition. These weights are outside the current slider fixtures, but were accepted by the exported numerical interfaces. This was an arithmetic-range contract issue, not a failure of the isolate theorem.

The author repaired both implementations with explicit rejection when a positive degree's reciprocal is not representable. The independent regression verifies rejection at the smallest subnormal, `1e-320` and `1e-310`, while `1e-308` remains accepted and produces finite, correct two-node operators. Exact input zero still follows the declared isolate rule. This does not claim robust arithmetic for every possible ill-conditioned graph. The body's equations and normal UI outputs were unchanged; the author separately rechecked actual display of the changed program.

## Complementary executable evidence

Run `scratch/lesson-tools/Scripts/python.exe -X utf8 scripts/verify-graph-fundamentals-independent.py`. The actual result is `scratch/graph-fundamentals-independent-review/results.json`, initially completed at 18:53:42 UTC. It loads the real example functions and current JS models rather than separate illustrative implementations.

- **81 changed anchored cycle networks:** independently derive the two unknown values from the resistance ratios along the two series paths between opposite anchors. Compare exact `Fraction` results from the displayed harmonic helper and actual JS solutions. The maximum JS discrepancy in these cases was zero.
- **81 loop/partial/coordinate states:** adding self-loops leaves the harmonic solution unchanged; an appended unanchored isolate stays explicitly undetermined. Reverse every incidence orientation and preserve energy/net outflow. Independently check row-stochasticity, degree stationarity, the square-root-degree null direction, normalized coordinate intertwining and the isolate diagonal contrast.
- **13 exact changed-input iterations:** use the three-node path's closed-form nonconstant eigenmodes and rational factors `(3/4)^k`, `(1/4)^k` as the oracle for the displayed conservative-iteration helper.
- **64 simple four-vertex graphs plus an isolate:** enumerate vertex triples directly, independently of matrix cubes, and compare incident/total triangle counts and the low-degree policy from the actual displayed helper.
- **Four JS and four native normalization boundaries:** test the repaired reciprocal contract and the retained representable near-boundary case.

All eleven programs were executed while loading their native helper functions; their exact displayed stdout assertions belong to the author's suite and are not relabeled as new independent assertions here. These finite complementary checks support the implementation and written arguments; they are not a proof for arbitrary floating-point inputs.

## Source and visual cross-checks

Independently inspected [Spielman's Laplacian lecture, section 2.2](https://www.cs.yale.edu/homes/spielman/561/lect02-15.pdf) for the single-edge outer product and local operator, and the current [NetworkX 3.6.1 normalized-Laplacian reference](https://networkx.org/documentation/stable/reference/generated/networkx.linalg.laplacianmatrix.normalized_laplacian_matrix.html) for row order, summed parallel weights and self-loop row-degree conventions. The finite derivations were checked directly; the notes were not treated as an unchecked authority.

Opened the author's actual 390-pixel `ordinary-section-8-390.png`, `loop-and-isolate-390.png` and `connected-interpolation-390.png` under `scratch/graph-fundamentals-browser/`. Their visible values and explanations agree with the models: the loop gives B degree 4 and row `[0.5,0.25,0.25,0]`, the isolate holds, and changed anchors 6/1 give B=13/3 with the connected D/E values equal to 1. These are explicitly author-captured screenshots independently opened here, not a newly claimed comprehensive browser run. The author owns the full desktop/mobile/keyboard/code-rendering evidence.

## Final reviewed identity

Exact current production fingerprints, author-freeze matching and the arithmetic repair disposition are recorded in `evidence/graph-fundamentals-independent-review.json`. The source/body and lab/CSS/brief did not change during this review; only the author-owned model and one normalization program received the reported narrow repair. No production source was edited by this reviewer.

Closed against the final author freeze at **2026-09-10T18:55:55.437494Z**. All six final source hashes exactly match the independently executed snapshot. Final model SHA-256: `ba77012855dd218f7be27f58cae58ed789475359cb1febef41cdf9ec23d82546`; examples: `940d6038e91a4249d29849c706b869ee4f04afc9bf4c5bd2f34a2c369605a454`. No unresolved material finding remains within this review scope.
