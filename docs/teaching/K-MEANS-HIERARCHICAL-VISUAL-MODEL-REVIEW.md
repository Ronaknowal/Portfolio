# K-Means & Hierarchical Clustering: independent visual-model review

Source/numerical review closed 11 September 2026 at 17:09:49 UTC. The reviewer authored the separate runnable Python examples, but did **not** author the pure models, interactive labs, inline figures or their CSS reviewed here. The root owns the complete lesson, actual-font browser checks and integration. This is a complementary review of those independently authored representations, not an independent review of the reviewer's own Python examples.

The initial five-file hashes, inputs/check scope, numerical results and corrections are recorded in [the evidence packet](evidence/k-means-hierarchical-visual-model-review.json). The numerical model hash is `7ed25d9b1a26d33364c80f9102a03e5e4cc7b6c9e108ea5ad06994e2f38f050d`; the labs hash at that source review was `42d1c2fd556b19f1be3c444206549c7a0db66698bc2eef55e7d8e904ef4d9800`. Its post-check change was the stopped-seeding caption; the numerical model and both figures/styles retained their checked identity at that stage. The subsequent label-layout amendment and its final source identity are recorded separately below.

## Inspected scope

Read the complete `k-means-hierarchical-models.js`, `KMeansHierarchicalLabs.jsx`, lab CSS, `KMeansHierarchicalFigures.jsx` and figure CSS. Read the author's model-checking scope to avoid duplicating its broad suite. Checked assignment/movement state meaning; tie, duplicate and empty-center policy; D² interval selection and all-zero state; metric/unit transformation; four linkage definitions, Ward scaling and count/height cuts; palette initialization, weights and integer reconstruction; and the four inline figures' calculations, representations and accessible explanations.

The representations express their quantities consistently:

- The scatter uses the same numeric scale for both coordinates in a square viewport. Transformed feature distances therefore remain visible as transformed geometry. Residual links connect the points to the centers selected by the **displayed phase's** labels. A mean-update phase deliberately retains the preceding labels until assignment runs again.
- D² interval widths, the draw marker, selected-row text, selected center and exact/rounded table derive from one state. The new visible color key identifies points without requiring a hover. All-zero distances produce a stop, not an invented uniform D² distribution; the separate native implementation explains its explicit fixed-slot fallback.
- Dendrogram branches use actual child and merge heights, with a linear vertical scale. Ward uses `sqrt(2Δ)`, not plain centroid distance. A count cut selects a merge prefix; a height cut includes all equal-height merges. The caption, tree, original-coordinate scatter and member-set table agree on this distinction.
- Palette images use the actual rounded centers and assignments whose integer-channel error is reported. Floating-center SSE and displayed integer SSE remain separate. Initialization is explicitly deterministic farthest-first after the most frequent color, not a D² sample. Numeric sRGB error and payload/codec claims are kept distinct.
- Diagnostic lines/whiskers use the actual retained 150-row program outputs. The axes are linear; ranges are observed restart variation, not confidence intervals. The mean figure explicitly says its sloping ownership links are layout while the error uses horizontal scalar differences. The linkage ruler's distances and Ward units are consistent. The ring figure is explicitly schematic, with an enclosing-ring limitation that follows from a straight two-center boundary.

Controls have visible labels; state explanations use text alongside color. Points have row/cluster labels and shape variation; centers use crosses. Data tables have captions, column headers and focusable scrolling regions. This source inspection does not by itself prove screen-reader behavior, mobile label clearance or successful rendered interaction.

## Complementary calculations

Fourteen focused cases passed without rerunning the author's full model suite:

- Four changed five-point hierarchies compared to SciPy's independently implemented single, complete, average and Ward heights.
- Two changed nonuniform-weight examples checked with exact rational weighted means and a weighted pairwise SSE identity, including retention of an empty center.
- All six UI palette sizes fitted separately with scikit-learn, using the same declared starting centers and frequency weights. Centers were matched by an assignment algorithm rather than compared as arbitrary integer cluster labels. Floating objectives agreed; actual reconstructed integer RGB errors were checked separately.
- The running six-point assignment/mean bridge is exactly **19.25 → 7.6875**. The scalar figure's errors are **50 → 38** for centers 2 → 4 on points `[1,2,9]`; the linkage figure's Ward increase is **36**.

All four diagnostic rows were compared to the already executed `production` program's stdout in [the native packet](evidence/k-means-hierarchical-native.json). Those program executions were reused, not rerun or attributed to this model review. The standalone native programs retain different explicitly declared palette and scaling fixtures; their numeric outputs are not presented as the UI's data.

## Findings and closure

1. **Arithmetic-range contract.** The other independent reviewer found that the native Ward helper accepted a distinct separation whose square underflowed. This reviewer confirmed the paired pure-model issue and informed its owner: finite coordinate bounds alone did not prevent a nonzero geometric separation from becoming zero. The model author added narrow arithmetic guards. This reviewer then independently rechecked four direct rejection cases: nonzero squared separation, positive Ward scaling, positive weighted error and a D² probability that underflows. A representable `1e-150` separation remains positive; genuine duplicates still give zero. No broader arbitrary-range audit is claimed. The native helper's separately authorized amendment is recorded in the native packet and is not counted as independent verification of the reviewer's own code.
2. **Stopped-seeding caption.** With identical rows, only C0 was drawn but the scatter caption still described a selected C1. Reported to the lab author; the final source now says **“C0 already represents every row; no second center is drawn.”** The main stopping explanation was already correct. The conditional final caption was inspected after the repair.

The root separately requested the clearer initial-assignment caption and visible D² legend. Their final source was read here; they are not represented as discoveries by this reviewer. No remaining material source/math finding was identified within this bounded scope.

## Limits and next stage

At the initial source-review stage, the root's actual-font desktop/390/320 browser review was separate and still needed to close the rendered reading, controls, keyboard and selected-image evidence. That stage established source geometry and numerical correspondence without claiming opened screenshots, a browser suite, integration or user acceptance. The subsequent finite image inspections are recorded below. No frozen native program was rerun for this review, and no model/lab/figure production file was edited by this reviewer.

## Subsequent finite opened-image review

After the source/numerical review, the root supplied three final production captures. This reviewer actually opened `evidence/k-means-lloyd-desktop.png`, `evidence/k-means-tree-mobile.png` and `evidence/k-means-diagnostics-mobile.png`. Their exact hashes and review times are appended to the evidence packet. This is an additional visual inspection, not a rerun of the root's browser suite.

The narrow dendrogram has readable heights, row/cluster labels and the horizontal cut. The narrow diagnostic plots, range explanation and numeric table are clear; no issue was identified in those two captures.

The Lloyd desktop capture revealed a concrete graph-reading problem: the alternating label offsets face **into** the close diagonal P0/P1 pair, causing their labels to overlap. The C0 label touches the nearby P1 marker, and the upper P4 label lies over the center marker. The root was asked for a focused label-placement correction, keeping coordinates fixed and using outward point-label offsets and clear center-label positions (or leaders). The numeric table remains useful but does not make these overlapping graph labels clear. The original crowded image is preserved at [its archived path](archive/k-means-hierarchical-before-label-layout/k-means-lloyd-desktop.png), retaining its original hash. The finding was pending at this stage and is closed by the amendment below.

## Final label-layout amendment: closed

The focused archived/current source diff was read in full. Only scatter annotations, their leader lines and explanatory copy changed. Point and center coordinates, the equal-axis projection, assignments and residual endpoints remain unchanged. The new layout chooses text positions against marker, tick and previously placed label boxes; short gray leaders connect displaced text back to its actual marker. Its fixed-size labels remain readable as the plot narrows. The bounded candidate search operates on the small displayed fixtures. The plot's `ResizeObserver` is attached to a stable container and disconnected on unmount; there is no added timer or persistent event listener. This is a source-level lifecycle assessment, not a performance benchmark.

The final labs hash is `6c5ea036d7751e02a0b3925cbd372a0a449f6ba5624aa6525f3ac1ddd1a47273`, and the final lab CSS hash is `421f42263440d73139f2437a402de95c7ae7a28105cdb9c8b41ae4a2b4e857c4`. They match the root's [successful final production packet](evidence/k-means-hierarchical-browser.json), checked at **17:34:20 UTC**. That packet owns the 54 intended-font label states across 1366/390/320 widths and explicitly reuses the archived 12 full production cases. Those browser checks were read and attributed to the root, not rerun by this reviewer. The separate failed temporary author harness is not used as successful evidence here.

This reviewer actually opened the final [desktop Lloyd image](evidence/k-means-lloyd-desktop.png), [phone Lloyd image](evidence/k-means-lloyd-mobile.png) and [desktop seeding image](evidence/k-means-seeding-desktop.png). In both Lloyd captures, P0/P1 and P4/P5 labels are separated from each other and from the center markers; C0/C1 labels have clear positions. The short leaders remain distinguishable from the dashed residuals, and the nearby explanation states their different roles. In the seeding capture, coincident point/selected-center markers retain their true geometry while their labels are separately identifiable. The probability strip, selected P4 key and accompanying table remain readable. No remaining material label-placement issue was found in these operated images.

The evidence packet preserves the earlier source/math findings, the original crowded-image hash and the exact hashes of these three newly opened images. No numerical/native suite or complete browser campaign was repeated for this display-only closure. It does not claim exhaustive clearance for arbitrary point clouds, measured performance, independent integration or user acceptance.
