# Independent reader and learning-experience review

Date: 14 September 2026. Topic: `t-sne-umap-manifold-learning`, Classical ML position 17. **Reader/figure review complete after focused corrections; no open content or learning-experience finding in this review scope.** Final website integration and user acceptance remain separate.

## Scope and independence

This reviewer authored `ManifoldLabs.jsx`, `ManifoldShared.jsx` and `manifold-labs.css`. The independent scope here is the **other authors' full reader body, nine inline figures and their explanations**, checked against the complete retained manuscript, specifications and design record. It includes all four displayed programs and seven changed practice tasks. This is not independent certification of the reviewer's own labs or shared rendering helpers. Their author geometry checks are explicitly separated below; the [independent technical review](manifold-independent-technical-review.md), [independent lab checks](manifold-independent-labs-checks.json) and root-owned [browser evidence](manifold-browser.json) supply complementary review of that work.

The review read the full production body and all nine figure implementations, inspected the native code/output bindings, and looked at the actual retained desktop and narrow screenshots. No beginner participant or screen-reader listening session was conducted: pedagogical judgments are an independent heuristic assessment, not measured learning outcomes.

## Findings and closure

1. **P2 — Bernoulli entropy sign in section 6.** Both the manuscript and reader initially said that adding the fixed Bernoulli entropy to cross-entropy gives KL. The identity is `KL(Bern(w) || Bern(ν)) = CE(w,ν) − H(w)`. A separate simple check at `w = ν = 0.5` gives CE = H = 0.6931471805599453 and KL = 0. The owner corrected both sources to **subtracting** and explained why the fixed input-graph term leaves the minimizing layout unchanged. The final source was reread at reader line 215 and manuscript line 198. Closed.
2. **P2 — F4 narrow table split numeric tokens.** In the initial [320 px capture](screenshots/manifold-figures/f4-narrow-before.png), values such as `−0.077785` broke between the last two digits. The table fit its bounds but was hard to use for exact arithmetic. The figure owner changed narrow numerical rows to stacked labeled readouts, retaining the same six-decimal values. The refreshed [F4 narrow capture](screenshots/manifold-figures/f4-narrow.png) was inspected: each gradient, step and new position is now an intact token with its label. The same treatment protects F6's coordinate disclosure. Closed.

The figure owner's larger compact-SVG glyphs also improve the narrow number lines without shifting model coordinates. The root-owned correction distinguishing a common rigid rotation from individual point relocation was read in its final form; its mathematical finding and closure are recorded by the technical reviewer rather than attributed to this review.

## Complete nine-item learning-experience checklist

1. **Route.** The introduction gives the practical digits question, named prerequisites and a first-pass route through sections 1–8 and practices 1–4. Sections 9 and 10 explicitly begin the deeper classical-method and derivation branches. The source keeps all thirteen sections and the next-topic ICA connection; implementation did not truncate the prepared packet.
2. **Cautions.** The map-reading contract has a clear home in section 3. The explanations of probability versus density, fixed-collection exploration versus held-out prediction, and ideal full-pair cost versus sampled UMAP serve different questions at their point of use. Displayed code prints measurements rather than cautionary prose. No repeated-caution rewrite is needed. The root's compact-status correction removes a duplicated lab verdict paragraph; that is an author/interface correction, not an independent lab finding here.
3. **Real question.** The opening 300 handwritten images return as actual pixel grids, source IDs, saved coordinates and a PCA baseline. Section 7 answers the original ten-neighbor question using measured R and T. Section 8 then changes the task to held-out prediction and reports the unfavorable UMAP result as well as the baselines. Selection by label is disclosed; the embedding features are pixels divided by 16.
4. **Labs as investigations.** The reader's placement and prompts lead from constructing graph edges, through Gaussian preference and UMAP memberships, to an original-space digit audit. Each asks for a changed choice and explanation rather than a completed trace alone. Arbitrary coordinate/distance/query edits supply choices not already solved by the prose; the equal-distance, missing-support and repeated-seed cases are deliberate nulls. Because this reviewer authored the labs, their actual commitment/invalidation/undo/validation behavior is judged by the separate independent and root browser records linked above. The numerical records cover all saved maps at k = 5, 10, 20, not just the opening query.
5. **Figures.** The nine figures have distinct jobs, detailed below. F4's broken numeric tokens were repaired. F6 includes PCA on an honest 0–1 retention scale and does not expose D's initial local answer. Quantitative circles/coordinates retain equal scales within maps. Captions identify exact calculations, native fits or algorithm diagrams instead of passing illustrative layouts off as measurements.
6. **Connections.** Section 9 explicitly turns the earlier U path lengths into Isomap/MDS coordinates. The MDS/PCA Gram-matrix connection is stated, and practice 5 connects the same line through MDS and LLE. The full t-SNE gradient derivation explains ordered pairs and the derivative of normalization. UMAP's self-column convention, fuzzy union, sampled optimization, transform, parameter roles and computational boundaries remain present. The canonical manifold alternatives are either developed or deliberately routed to spectral geometry/TDA; the site implementation did not replace them with a method-name menu.
7. **Code.** All four programs keep fitting, neighbor ranking, scoring, transforming or gradient updates at the center. They have complete imports and data/setup instructions; no browser optimizer executes on page mount. An independent binding check recomputed each displayed `example.code` SHA-256 and compared `example.expected` byte-for-byte with [native evidence](manifold-native.json): all four matched. Existing native executions were reused rather than rerun by this reviewer. The recorded 12-thread digits fit and one-thread UMAP distinction is visible in the surrounding explanation.
8. **Practice.** Seven tasks change quantities and/or decisions: a new sensor corridor; a non-Gaussian entropy row; repair of a misleading score; changed directed memberships and analytic optima; MDS/LLE reconstruction on a different line; a broken deployment pipeline; and two independent digit queries. Numerical solutions supply reproducible values with reasoning. Hints and solutions are separate closed disclosures. The final audit gives success criteria rather than a fabricated universal answer.
9. **Screenshots.** Informative states were inspected, not only default wrappers: F4's signed contributions and corrected exact readouts; F6's actual full desktop maps and coincident seed markers; F9's filled versus unfilled complexes; the root's saturated fuzzy union and source30 pixel-neighbor list. The close-point graph and negative-tick mobile repairs have separate author evidence below. These images complement numerical and DOM assertions; a passing assertion was not treated as proof of a readable drawing.

## Inline-visual reading pass

| Figure | Reading question and inspected outcome |
| --- | --- |
| F1 | The same A–G identities make the two-unit opening and six-unit route directly comparable. Desktop side-by-side and narrow stacked captures keep unit edges and route indices readable. |
| F2 | Arrow direction and solid/dashed patterns separate input-only and map-only choices. The four-row audit makes every false input rank 2; R = 0 and T = 0.5 have distinct explanations. The narrow capture retains identities even where A and C are near each other. |
| F3 | The distance strip, raw weights and normalized bars preserve B/C/D identity. Both bar panels use 0–1 scales; D's genuinely short bar has its numeric value alongside. Desktop and narrow captures show the probability and entropy/perplexity stages. |
| F4 | A–B and A–C attraction, A–D repulsion and their total have one signed displacement scale. The step-size multiplication is separate from the negative gradient. Desktop geometry is clear; corrected narrow readouts retain complete values. |
| F5 | The crossed-out self edge is visibly excluded, three strengths sum to the log₂4 target, and two reciprocal directions use one shared distance. The fixed union stays visible while candidate separations change. The narrow progression preserves graph versus coordinates. |
| F6 | Actual PCA and t-SNE maps retain the same selected observed image; the chart retains the PCA baseline and coincident seed marks without jitter. The companion native T/continuity table does not call either score a percentage of retained neighbors. Desktop and narrow captures were read; source306 was subsequently selected for a focused mobile/shared-axis check. |
| F7 | D → B → a centered line stays explicit, with exact fractions, gap 2, gap 3 and total distance 5. The narrow matrix labels and recovered-coordinate picture preserve the Gram-product example. |
| F8 | Solid/dashed reconstruction arrows retain weights 2/3 and 1/3 while the measured gaps double. The narrow lines use the same coordinate scale, making the changed spacing visible rather than normalizing both drawings into one picture. |
| F9 | Metric duplicate locations are separate from the abstract complex drawing. The square has an unfilled cycle at ε = 1; the complete threshold state is filled, with explicit triangle/tetrahedral language. The projected duplicate pairs and [1,√2) interval are not represented as a fitted UMAP result. |

## Author-only shared geometry closure

During the shared rendering pass, a supported graph edit `D = (0.25,1), G = (4,4), ε = 1` was captured with the actual Edge/browser fonts. Its initial B/D node disks overlapped. `GraphMarks` now retains the numerical centers, uses smaller marks for close points and places separate opaque-backed leader labels around them; the full applied ID/coordinate list is visible. A 390/320 px follow-up found all seven IDs once, B/D callouts, no label/text overlap and no clipped ticks. The [before image](screenshots/manifold-graph-close-before.png) and [320 px after image](screenshots/manifold-graph-close-after-320.png) were visually inspected.

The shared map plot's left gutter was widened while retaining a 230×230 equal-scale data area. F6's negative `−22.2` tick now remains complete at 320 px, with a measured 13 px glyph height. [Source306's final mobile map](screenshots/manifold-shared-f6-source306-320.png) was inspected after keyboard-addressable source selection, switching the mobile map and enabling digit colors. [Focused geometry evidence](manifold-shared-geometry.json) records both widths. These are **author checks of this reviewer's shared code**, independently supplemented by the technical review's geometry invariants; they are not mislabeled as independent findings.

The geometry capture preceded only the final outcome-live-region amendment in `ManifoldShared.jsx`; that amendment changed result announcement markup, not `EqualPlot`, `DigitScatter`, graph geometry or CSS. The root owns the final browser check of the announcement region and final source reconciliation. No assistive-technology listening result is claimed.

## Reviewed source identity

SHA-256 of the final reader/figure/program/packet source inspected for this review:

| Source | SHA-256 |
| --- | --- |
| `src/learn/data/topics/t-sne-umap-manifold-learning.jsx` | `5a149b1cea182600d624492d9b252fa89aa9b35046197980505d781caacfff85` |
| `src/learn/components/lesson-labs/ManifoldFigures.jsx` | `d95fdffdbd8b3c2021e7faa6f70ee74ce94360a32244a55b7d808c84c63218f3` |
| `src/learn/components/lesson-labs/manifold-figures.css` | `c3ae1be13d6c7cd29b0786766e6398318b74d33b19e53602158e60dc3f1a3402` |
| `src/learn/data/manifold-examples.js` | `5896f48a0c7c2d2741b6fac101cb61d3f9d7b1a1b5cd6920c20bf47addcdf9c2` |
| `docs/teaching/drafts/t-sne-umap-manifold-learning/lesson.md` | `52214b4bee8bc04ed84054cb5ec7ed725a11261840273b35dec4d90a930160d3` |
| `docs/teaching/drafts/t-sne-umap-manifold-learning/visual-specifications.md` | `d30a770dde8b7aeb388201ec78596b89517d952387f17949ca8c5ae51753ec01` |

The review did not rewrite historical completed lessons, approve deployment or infer user acceptance. Later source changes need a relevant delta review; unchanged native and figure evidence can be reused.
