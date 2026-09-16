# Manifold learning labs/models — complementary independent review

Reviewed 14 September 2026 by the inline-figure author, who did not author `ManifoldLabs.jsx`, `ManifoldShared.jsx`, `manifold-models.js`, or `manifold-data.js`. This is independent scrutiny of those labs and models; it does not certify the reviewer's own figures. No material mathematical or prediction-state findings remain open in this review's scope.

## Scope and prior evidence

Read all four labs, the shared components, and the complete pure models against the full saved manuscript, visual specifications, and approved design. Also reviewed the author-owned [349-assertion model checks](manifold-models.json), [90 native metric checks](manifold-data.json), and [four executed Python programs and installed-source checks](manifold-native.json). Those checks were not represented as my independent executions or needlessly repeated. Native evidence includes real 300×2 UMAP coordinates, preserved original source IDs and nine coordinate arrays, the source 30 local/global reversal, the fixed-P tiny trace, and held-out transform evaluation.

The source review found:

| Investigation | Independent conclusion |
|---|---|
|A · graph route|Proposed endpoint IDs are compared on the old graph before the candidate graph is applied. Coordinate/radius validation preserves the active graph, and no-path results remain distinct from numeric distance. Stable IDs survive edits.|
|B · conditional probability|Old/new probabilities concern the same proposed candidate and distance row; entropy uses a consistent logarithm convention. The calculation does not assume every middle-ranked probability is monotone in bandwidth.|
|C · fuzzy graph and optional pair|Directed support, supplied local scales, fuzzy union, ideal single-pair cost, and real UMAP fitting are distinguished. Absent support and weights0/1 receive correct boundary treatment. Editing a candidate graph retires the pending pair prediction; applying a graph remounts the pair state.|
|D · local neighbor audit|The answer lists remain withheld until committed reveal. Results bind to source ID, k, map identity, and metric. Changes retire stale results. Rankings use unrounded coordinates/features, with label colors excluded from the fit and original source IDs used throughout.|

The model implementations preserve self-exclusion, exact ordered-pair normalization and total-gradient signs. The four-vertex Rips formula agrees with its stated bounded domain: triangles kill cycles and the tetrahedron is counted at the filled threshold. MDS and LLE fixtures encode the exact manuscript calculations.

## Additional independent executions

[Independent executable evidence](manifold-independent-labs-checks.json) contains source hashes and complete expected neighbor lists. The bounded check used Microsoft Edge against the real local production preview with web fonts loaded, plus an independent raw-distance ranking calculation.

1. **Storage-order metamorphic check.** Independently ranked source 306, k=20, actual UMAP coordinates using direct squared Euclidean sums. Reversing both row storage and paired coordinate storage leaves all original/map source-ID lists and the retained set unchanged.
2. **Changed endpoint baseline.** In A, choose C/E, then draft D=(1,3) at radius 1. The old graph baseline is 2; applying the candidate yields no path and correctly evaluates the committed prediction.
3. **A middle Gaussian probability can decrease.** In B, candidate C with distances [1,2,3] increases when sigma changes 1→2.75, then decreases at 2.75→3. Independently computed probabilities are 0.3403799325222413 and 0.340271984831862. Both recorded predictions and displayed values agree.
4. **An additional real-data query.** In D, choose source 306, k=20, `umap-n15-d01-s7`. Before reveal there are no neighbor-list nodes. After a committed prediction of 17, the full original and mapped source-ID lists equal the independent rankings and the retained count is 17/20.

All four checks passed. The evidence records the exact source state tested; the subsequent shared component changes adjust plot/graph geometry. They do not change neighbor arithmetic or lab prediction state. Their visual effects receive separate final shared-layout inspection.

## Teaching and interface assessment

Each lab has a concrete prediction, a meaningful user-controlled mathematical input, an explicit commit/apply or reveal boundary, and an observable result tied to that input. The source and changed-case walkthrough do not expose Investigation D's local answer before commitment. Errors preserve useful state. Optional label overlays and the optional pair cost remain clearly separated from model fitting and the central investigation.

This is an expert teaching/interface review, not a real learner study. Browser and mathematical checks establish the bounded claims above, not universal usability or unrestricted-input correctness. The parent incorporated the independently identified enlarged-text checkbox-wrap repair into the scoped lesson stylesheet; final [parent browser evidence](manifold-browser.json) records that visual closure.
