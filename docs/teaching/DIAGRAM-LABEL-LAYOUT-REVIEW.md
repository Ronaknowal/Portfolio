# Diagram label and annotation layout review

14 September 2026. This is the bounded rendering review assigned within the user's request to repair the distorted GMM illustration and prevent similar failures elsewhere. It is not a renewed mathematical/content audit or approval of all interactive states.

## Scope and confirmed repairs

| Topic / component | Confirmed issue and repair |
| --- | --- |
| Eigenvalues & Eigenvectors — `EigenvalueLabs.jsx` | A negative-x observation ID crowded a vertical-axis tick. Its label now faces away from the axis; the observation and projection do not move. |
| Matrix Calculus & Jacobians — `MatrixCalculusLabs.jsx` | The logarithmic error explanation touched the horizontal-axis caption. They now have separate annotation rows. |
| Convex Optimization — `ConvexOptimizationLabs.jsx` and its stylesheet | The shared mobile SVG rule overrode the topic's font size, crowding endpoint ticks and clipping titles. A scoped font rule and dedicated annotation margins keep the titles and captions separate. Endpoint tick alignment faces inward. |
| Non-Convex Optimization Landscape — `NonconvexLandscapeLabs.jsx` | The top coordinate label overlapped the figure title. Coordinate labels now sit beside the sampled field. |
| Bayesian Inference & Conjugate Priors — `BayesianInferenceLabs.jsx` | The interval plot's measurement-axis caption extended past the SVG. Extra bottom margin contains it. |
| Maximum Likelihood & MAP — `MaximumLikelihoodLabs.jsx` | Lower-left horizontal and vertical ticks crowded each other. Endpoint alignment and a slightly lower tick row separate them. |
| Spectral Graph Theory — `SpectralGraphLabs.jsx` and its stylesheet | Grouped node IDs and centroid IDs occupied the same space. Squares retain their actual coordinates; a wrapping key identifies each centroid and its plotted coordinates. A background stroke keeps group text distinct from an axis behind it. |
| Stochastic Processes — `StochasticProcessesLabs.jsx` | The return-arrow glyph crowded the hidden-state title. A path arrowhead belongs to the connector, with separate heading space. |
| Queueing Theory — `QueueingLabs.jsx` | The unit heading crowded the top occupancy tick. It now begins over the plot area. |
| K-Nearest Neighbors — `KnnLabs.jsx` and its stylesheet | Nearby B and C IDs collided. Fixture-specific label offsets and short leaders distinguish the IDs without moving their rows. Rendered review also caught B1 touching B2's marker; that annotation was moved above-right. |
| Ensemble Methods & Stacking — `EnsembleMethodsLabs.jsx` | A zero-error line crossed its numeric label; probability-map corner ticks crowded; the stump's row labels touched its caption. Numeric annotations are offset from the reference line, endpoint ticks face inward, and the stump caption has a separate row. |
| Recommender Systems — `RecommenderLabs.jsx` | Vector IDs crossed an axis/caption and local-objective corner ticks crowded. Labels offset from the vectors, the horizontal axis caption uses the bottom margin, and contour captions/ticks have separate space. |
| PDEs, Conservation & Boundary Conditions — `PdeLabs.jsx` | The time-axis title touched its upper endpoint value. It now has its own upper plot margin. |
| Naive Bayes — `NaiveBayesLabs.jsx` and its stylesheet | Shared mobile SVG typography crowded tick rows, captions and lower-left corners. Topic-scoped typography, separate annotation margins and inward endpoint alignment preserve the existing plots. |

Only these topic-owned rendering files and four topic-owned stylesheets changed under this assignment. Numerical model files, datasets, observation positions, computed paths, data-space coordinate maps, units and axis domains were preserved. Additional annotation space does not rescale one quantitative axis independently.

## Triage dispositions

The all-published geometry scans identify candidates, not confirmed defects. Spectral graph edges intentionally terminate beneath opaque node circles: text/edge bounding-box intersections there are not visible text collisions. Grid and zero/reference lines may also share a text bounding box while a background stroke keeps the text readable. KNN intentionally preserves a labelled, keyboard-focusable horizontal drawing region on narrow screens; a viewport-width crop of the SVG is not the entire scrollable figure.

The changes address actual annotation collisions and clipping. They do not shift data points, widen mathematical distributions, jitter coincident observations, truncate numeric evidence, or remove substantive teaching material to obtain cleaner pictures.

## Rendered verification and limits

The first combined build's selected figures were captured in headless Microsoft Edge with document fonts ready at desktop width 1366 and phone width 320. All 26 selected screenshots were viewed: one informative figure per initial family at each width. The reviewed desktop and phone Convex plots contain their titles, ticks and captions; the Spectral centroid key wraps on phones and leaves the grouped node labels separate. The other selected figures retain their calculations while separating the affected annotation rows.

That actual image review identified the additional KNN label-to-marker collision described above. Root's subsequent geometry scan identified the additional ensemble-stump, recommender-axis and Naive Bayes cases. Their final rendered closure and exact source/build hashes are recorded in the companion evidence record and final closure below; they were not inferred from the earlier screenshots.

This is a scoped default-state rendering review with relevant subsequent cases, not a claim that every slider combination, hidden branch, browser zoom, font fallback or assistive-technology mode was exercised. Root owns final production integration, all-route geometry triage/disposition, the ledger and the updated authoring policy.
## Additional all-route candidate dispositions

Root subsequently assigned nine additional topic families for candidate review. Ten representative desktop images were captured and actually viewed (both complex-plane figures); the source layering was inspected. No broad content audit or all-state approval is implied.

| Candidate | Source and rendered disposition |
| --- | --- |
| Expression tree, SVG 7 | `ExpressionTreeFigure.jsx` draws edges, then opaque node circles (`#29251b` / `#152720`), then symbols. The rendered operators and leaf values are clear. Intentional node layering; no repair. |
| Combinatorial primal-dual graph, SVG 1 | `CombinatorialOptimizationLabs.jsx` draws each dual load over a filled `#151a20` rectangle and vertex names over filled circles. The displayed zeros and names are clear. Intentional backplates; no repair. |
| Mapper nerve, SVG 16 | `TopologyTdaLabs.jsx` draws all edges before opaque filled circles and node IDs. The six default IDs are clear. Intentional node layering; no repair. |
| Reductions cover graph, SVG 1 | `IntractabilityLabs.jsx` draws matching/ordinary edges before node circles; `intractability-labs.css` supplies opaque `#101820` / `#17382b` fills. All names are clear. Intentional node layering; no repair. |
| Network-flow original/residual networks, SVGs 0–1 | Both use the same `NetworkFlowLabs.jsx` renderer. All original/path edges precede the edge-label rectangles; `network-flow-labs.css` gives them `#0c1011` fill. Representative SVG 0 shows clear 0/1 labels. Intentional backplates, including the shared residual rendering order; no repair. |
| Sets/logic Hasse graph, SVG 3 | `SetsLogicLabs.jsx` renders cover edges before filled node circles and values. The values 2, 3 and 6 are clear. Intentional node layering; no repair. |
| Matrix decomposition output, SVG 3 | Actual defect: `CoordinatePlane` puts x ticks inside the drawing and then paints output vectors over them. The output/approximation line visibly touches 1 and 2 in the captured example. Reported for annotation-only repair. |
| Probability mixed-distribution CDF, SVG 3 | Actual defect: ordinate zero starts at x=25, baseline159 while a horizontal zero segment runs from x=25 at y=155. The image visibly strikes through the zero. Reported for moving only the annotation left of that segment. |
| Complex/Fourier plane, SVGs 0 and 11 | SVG 0 is a tight candidate; SVG 11 visibly confirms the issue: the dashed sum vector cuts the origin zero. `Plane` paints that label before vectors without a backplate. Reported for drawing the label last with an opaque background stroke, preserving the vector path. |

The representative captures are `evidence/screenshots/layout-triage-*.png`. The matrix, probability and complex findings must not be marked resolved merely because other graph candidates were intentional layering; their final repair/check disposition belongs in the final integration record.
## Annotation repairs from the additional review

The three confirmed issues above were repaired in their topic-owned components. Matrix decomposition retains its coordinate grid, curves, vectors and scales and moves horizontal tick labels into a dedicated bottom gutter. The mixed-distribution CDF moves only its ordinate-zero label left of the horizontal zero segment. The complex-plane origin label is now painted after the vectors with a dark background stroke; its actual point and all vectors remain unchanged. The covered segment behind that label is intentional annotation layering, not an unresolved crossing or a changed vector.

The final affected-figure capture stage in `evidence/diagram-label-layout-review.json` binds and closes these three repairs together with the Naive Bayes count-label halo. Earlier rendered results for unchanged figures remain applicable; they are not represented as a new mathematical or whole-state test run.

## Final closure

All 15 final affected captures were actually viewed: Naive reliability, Matrix decomposition output, mixed-distribution CDF and both flagged complex planes, each at 1366, 390 and 320 pixels. Their confirmed defects are resolved in those states. The matrix ticks occupy the separate bottom gutter, the CDF ordinate zero sits left of the zero segment, and the Complex/Naive background strokes protect glyphs while the quantitative paths remain unchanged. A line bounding-box intersection behind those intentional annotation strokes is not a visible unresolved collision.

[Source-bound evidence](evidence/diagram-label-layout-review.json) preserves the earlier 13 follow-up captures, final 15 captures, final build-manifest hash, all 21 changed source-file hashes and the six checked unchanged layering dispositions. Source/build hashes were checked before and after the final capture pass; no source drift occurred during capture. The first 26-view pass and ten later candidate images are separately described above rather than counted as a new all-state validation. There are no unresolved confirmed layout defects in this assigned, rendered scope.

Root retains responsibility for final ledger/inventory reconciliation, the production build and other reviewers' scopes. No numerical model, dataset or mathematical path was changed by this reviewer, and no pending content packet was implemented.
