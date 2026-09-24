# Shared plots and foreground labels — rendered review

14 September 2026. This bounded review supports the user's request to repair distorted illustrations and find the same failure elsewhere. It does not certify every figure, state, device, or scientific claim on the website.

## Shared Plot

The old shared renderer put long rotated axis captions, full floating-point endpoint strings and series legends inside a fixed-size SVG. Captions could leave its bounds, endpoint labels could collide, and legends could cover plotted data. The repair keeps the original point values, series order and numeric domains. It moves captions and legends to wrapping HTML, keeps tick text at a readable pixel size by measuring the mounted container, and formats only axis tick labels. If compact limits would become indistinguishable or overlap, the chart shows min/max with the complete numeric limits in flowing text. Exact data mounts only when explicitly opened, with at most 50 rows per page. The resize observer skips unchanged widths and disconnects on unmount; no dependency was added.

[Source-bound shared Plot browser evidence](evidence/shared-plot-layout-browser.json) records 17 passing checks on four existing published routes at 1366, 760, 390 and 320 px: Hyena's long decimal endpoints and 12 exact values, DAPO's multi-line legend, byte-level tokenization's long horizontal caption, and GPU autoscaling's 144 values across 50/50/44 rows. Keyboard disclosure works; closing removes the table rows. All five retained final screenshots were actually inspected, with explicit sticky-header clearance. The later combined build changes other figures; this evidence is reused for the unchanged two Plot source files rather than presented as a rerun against that later build.

## Six foreground-line candidate families

Nine initial candidate images and their source were inspected; the final [source-bound browser report](evidence/foreground-label-layout-browser.json) records 38 passing cases and 15 inspected final captures.

| Family | Finding and disposition |
| --- | --- |
| Multivariate calculus | The name of the base point lay across an outgoing arrow. Labels now use the free direction opposite outgoing arrows; opposed arrows use a perpendicular offset. Point/vector positions are unchanged. Six changed directions and the opposed-arrow case were checked. |
| Convex duality | The upper price label lay on the optimal-price reference. It now occupies the clear header band; the value is unchanged. The zero-budget, initial-price 20, state 20 case also passed. |
| Hypothesis testing | Reference lines crossed interval-method names and the alternative-distribution title. Reference markers now occupy each plotted row, preserving their horizontal positions and leaving the label bands clear. Default layouts, a changed effect preset, two-sided power at effects 0 and 4, and success-probability references 0, .5, 1 were checked. |
| Exponential families | The continuous mean guide crossed an axis tick and the spread summary. Two aligned guide segments now belong to their data rows. Dataset coordinates, mean and spread summaries are unchanged. |
| Differential geometry | The B name was on the chord. It now sits radially outside the circle; a small background halo keeps coordinate axes from obscuring it at cardinal positions. Six endpoint angles were checked; the 0, 90 and 180 degree captures were inspected. The arc/chord endpoints and lengths are unchanged. |
| f-divergence/IPM | No change. The detector's stem/zero overlap is occluded by the later-painted Q point: centre 175, radius 8, stem endpoint 183. The visible zero is unobstructed in the inspected desktop and mobile images. This is a specific drawing-order finding, not a generic exemption for lines through text. |

Defaults were checked at 1366, 390 and 320 px. Both numerical mark coordinates and lesson models were preserved. The scripts' geometry candidates were interpreted against actual rendered images; deliberate coordinate-axis and point occlusion remain distinguished from foreground-line collisions. These checks do not replace the parent task's integration, curriculum, ledger and broader route review.

Only the five named lesson renderer files and the shared Plot JSX/CSS were changed in this review. Temporary before-triage and duplicate captures were removed after selecting the retained final evidence. Existing manuscripts, lesson prose, example data, pure models and unrelated work were not edited.

## Independent check of the allocation and backtracking repairs

The root-authored GMM allocation reflow was checked against the source and its desktop and 320 px captures. Independent logistic-responsibility arithmetic agrees with the existing model to 1e−12: the Left count is 2, mean −1.344824658054, variance 0.691446639091 and standard deviation 0.831532704763. The direct log-density calculation moves from −7.158186977137 to −6.461856301295. Each responsibility column still has unit mass. All four before/after strips use the same affine coordinate map over [−3.4,3.4]; their bars show mean ± standard deviation. The components retain equal panel widths and stack on phones. The two captures show clear, separated descriptions and marks. These computations and source hashes are saved in the browser report's separate `independentRootFigureReview` record.

The backtracking CSS change was also checked against source and the initial/explored desktop captures. It preserves all node numbers, edge positions and visited/pruned/solution classes. Moving opacity from the whole unseen group to its node circle/number keeps the annotation backplate opaque; the unseen annotation uses a deliberately muted fill. The take/skip names stay readable across the underlying edges in both inspected states.

This review found one additional real annotation defect in the EM lab: the **Identical components** preset drew Left and Right on top of one another at their shared mean. The follow-up repair uses a wrapping HTML key with the existing colors/dash patterns, and removes the two positional SVG names. It preserves curve/guide/measurement coordinates and statistics. The existing GMM browser verifier now checks all four presets before and after a full cycle at desktop and 320 px. The final root production build and 14-case GMM browser suite passed, including those preset/state checks. Both final identical-key screenshots (desktop and 320 px) were actually inspected: the key stays complete and separated, and the coincident curves keep their original numeric geometry. The follow-up is closed; no broader content review is claimed here.
