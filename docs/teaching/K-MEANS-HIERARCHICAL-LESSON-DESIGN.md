# K-Means & Hierarchical Clustering — lesson design and implementation

Scope authorized 11 September 2026: the next single topic, `k-means-hierarchical-clustering`, Classical Machine Learning position 11 of 39. Retain the title: metric choice, initialization, quantization and linkage are necessary parts of these two methods, not a reason to enumerate applications in the heading. Preserve its stable ID, memberships and position. The previous first-ten increment remains closed.

## Learning contract

A learner can calculate an assignment and mean update, distinguish a fitted partition from discovered truth, justify a distance representation, explain and implement careful seeding, read and cut a dendrogram with the correct height convention, choose an appropriate flat or hierarchical workflow, and submit a reproducible clustering report with limitations.

Prerequisite: the implemented K-Nearest Neighbors lesson supplies vectors, distance, units and training-owned transformations. This lesson refreshes squared distance and arithmetic means locally; it does not require a previous clustering course. NumPy code is explained with shapes. The beginner route reaches both complete algorithms and interpretable output; mathematical proofs and specialist connections are progressive branches, not substitutes for core reasoning.

Anchor: represent observations as points and choose a small set of representatives or a nested grouping. Preserve the original six labeled points P0=(1,0), P1=(1.5,.5), P2=(3,2), P3=(3.5,2), P4=(7,5), P5=(7.5,5.5). Preserve the original 150-observation `make_blobs` generator as a clearly synthetic library comparison. Use independently changed rectangles, axis scales and a constructed color raster to test transfer. No synthetic fixture establishes natural categories, clinical validity or a general implementation-speed ranking.

## Retain, repair and scope ownership

The entire old lesson was reviewed by the author and research reviewer. [Immutable original bytes](evidence/k-means-hierarchical-baseline.json) permit a coverage comparison; removing an incorrect statement is not removal of a curriculum topic.

| Existing idea / discovery | Decision and depth | Destination |
| --- | --- | --- |
| Objectives, Voronoi assignment, mean minimizer, Lloyd convergence | Retain and derive; repair strict-decrease and boundedness-only termination argument; declare ties, empty centers and early stopping | Core, sections 2–4 |
| k-means++ theory and scratch code | Retain complete runnable code; distinguish vanilla D² expectation guarantee from greedy sklearn seeding; handle duplicate/all-zero cases | Core seeding and deeper guarantee |
| All four linkage criteria, Ward scratch code and SciPy tree | Retain; distinguish ΔSSE, sqrt(2Δ) height, ordinary distance, exact merge-count cuts and tied-height cuts | Core hierarchical route |
| 150-point demo, trajectories, elbow, silhouette, distance matrix | Replace hand-authored numbers and a mislabeled 45-point plot with computed examples/figures. Diagnostics support a decision, not a certificate of true k | Core examples and validation bridge |
| Scaling, outliers, initialization, high-dimensional geometry, unequal groups, arbitrary labels | Retain with explicit mechanisms/counterexamples and reasonable boundaries; no always-scale rule or automatic anomaly label | Core diagnosis |
| Quantization and image palettes | Deepen: weighted repeated colors, centroid versus stored byte palette, actual distortion and honest bit accounting | This topic: direct objective application |
| ANN coarse codebooks, mini-batches and clustering features | Explain enough to map assignment to routing/stream summaries; show approximation limits and memory arithmetic | This topic with links for deeper systems work |
| BIRCH, medoids, spherical/kernel k-means, GMM and graph/density alternatives | Retain scoped mechanisms and selection boundaries; correct count-weight and covariance-limit claims | Deeper branches and actual existing follow-ons |
| PCA/UMAP as preprocessing and automatic biological interpretation | Replace blanket advice with representation/validation reasoning. PCA lesson is the actual next page; its own derivation remains there | Linked next topic; scoped destination note if a new useful instruction is needed |
| Silhouette, label permutations, stability and reference labels | Teach minimum needed to report results, with changed exercises. The existing evaluation topic explicitly owns full metric derivations, ARI/NMI and selection uncertainty | Destination note for clustering evaluation; do not rewrite it here |
| Long unsupported historical priority/popularity assertions | Keep sourced useful quantization/history connection; remove unsupported priority, universal performance and popularity claims | References / application explanation |

The scoped owner check inspected KNN's actual current blueprint and lesson, the PCA and GMM introductions and the clustering evaluation source. It is not an audit of every curriculum entry.

## Hurdles, representations and learning evidence

| Hurdle | Explanation / example | Representation and learner action | Independent evidence |
| --- | --- | --- | --- |
| No labels does not mean no assumptions | Representation, metric and k determine the question | Observation → feature vector → distance → partition, with center/group terminology | Diagnose a misleading customer segmentation |
| Two coupled unknowns | Freeze centers to assign, freeze assignments to take means | Lloyd phase map with point-center segments, old/new centers and exact current objective | Changed hand calculation and nonoptimal rectangle fixed point |
| Units define similarity | Squared axis differences receive squared scale factors | Feature-geometry comparison in the actual transformed coordinates | Convert a unit with and without the compensating weight |
| Random seeding is not farthest-first | D² defines probability mass, not a deterministic ranking | Cumulative sampling strip tied to actual points; zero-mass state explicit | Calculate probabilities and all-duplicate behavior |
| A hierarchy is a sequence of irreversible decisions | Different linkages ask different pair questions | Dendrogram and labeled scatter connected to selected merge prefix or height | Changed pair sets, tied-height cut and row-order interpretation |
| Ward has two commonly conflated quantities | Parallel-axis identity gives Δ; SciPy height is sqrt(2Δ) | Merge table/caption exposes both units next to tree | Direct union SSE versus Δ, and native SciPy comparison |
| Small error does not select the task | More centers can represent more detail | Actual native seed/k diagnostics, named metrics and failure examples | Explain local-fit nonmonotonicity, silhouette limits and stability |
| A representative has a concrete use | Replace each RGB vector by its palette entry | Original/reconstructed pixel mosaic with counts and calculated distortion | Weighted unique colors versus repeated pixels; storage caveats |
| A fitted model must be usable and evaluated | Fit transformations/centers on development data, freeze and report | Complete native pipeline and cluster profile output | Changed held-out capstone and explained diagnosis |

Each visual has a distinct purpose and runs bounded calculations on small fixtures. Geometry represents the displayed metric; dendrogram leaf spacing is layout, not distance. Numerical tables and member labels supplement color. Controls must be keyboard usable with reset/back behavior and legible narrow layouts. There is no lab quota. Static explanatory figures remain visible beside their concepts; optional details cannot conceal the first explanation.

## Implementation and evidence ownership

Six stages follow the [canonical workflow](../../LESSON-TEACHING-STANDARD.md#six-stage-authoring-workflow); quality-first guidance permits further relevant improvement at any stage.

- Author/integration owner: root. Lesson body, blueprint, scoped notes, metadata, browser review and final handoff.
- Research/coverage reviewer: scientific_visual_improvements. Complete original read; primary claims and annotated alternate resources in [source review](K-MEANS-HIERARCHICAL-SOURCE-REVIEW.md). Later independent complete-lesson review must be recorded separately from research.
- Native-example author: testing_documentation_completion. Complete Python programs and their executed outputs; independent native oracles/changed fixtures, with explicit runtime versions.
- Visual/model author: workflow_visual_improvements. Topic-native components, pure models and meaningful invariants; their own tests are author evidence, not independent review.

Current stage: complete, 11 September 2026. All six authoring stages are closed: assessment/research, complete implementation, author verification, independent review, focused corrections/browser closure, and website integration/handoff. The full production browser pass and subsequent focused label-layout amendment succeeded. No material finding remains open. User acceptance is separate; the next module topic is PCA and awaits the user's next scope.

Planned checks are source-bound rather than a campaign to rerun unchanged historical work: mean and Ward identities, exact small-fixture partition comparison, duplicate/tie/early-stop contracts, common unit-scale invariants, current native library output, browser controls and diagram interpretation at desktop/390px/keyboard, strict module sequence and ID/publication conservation, selected-topic lazy closure, production build. No benchmark claims will be inferred from schematic curves. Reading/practice estimates will be set after the complete material exists.

## Implemented learning experience

The fourteen-section route moves from choosing meaningful features through assignment/mean updates, metric choice, initialization, linkage and tree cuts to diagnosis, useful applications, changed practice and a complete held-out capstone. Mathematical proofs and specialist connections extend the visible beginner explanation. Retained depth includes the mean identity, careful seeding guarantee, empty/tied cases, Ward's two scales, all four linkages, current library workflows, mini-batches, weighted palettes, clustering features and approximate-search routing. The original six-point and 150-row examples remain identifiable.

Five investigations use different representations: assignment/residual geometry; transformed feature geometry; a D² sampling strip linked to points; an operable dendrogram with exact merge/count cuts; and a reconstructed color mosaic. Four inline figures explain the mean, linkage definitions, measured diagnostics and the ring counterexample. Nine changed exercises include hints and reasoned solutions. Six standalone Python programs expose their actual executed output, and the capstone includes an independent variation rather than asking the learner to copy the demonstrated result.

References contain annotated primary documentation, original research, educational articles and a direct MIT lecture/YouTube alternative. The [source review](K-MEANS-HIERARCHICAL-SOURCE-REVIEW.md) records the actual reviewed material, including the official lecture transcript; this is not a claim that the video was watched. Root additionally read the linked scikit-learn clustering-assumptions tutorial and dendrogram example. The lesson remains self-contained.

Newly useful follow-on instructions were saved for [PCA](topic-notes/pca-dimensionality-reduction.md) and [clustering evaluation](topic-notes/clustering-evaluation-validation-silhouette-ari-nmi.md). Future authors must assess their reasoning; neither destination was rewritten or automatically declared reviewed. PCA is the actual next module entry, regardless of publication state.

## Evidence and resolved findings

- [Native execution packet](evidence/k-means-hierarchical-native.json): six full programs, 49 changed oracle cases and five rejection cases, with NumPy 2.3.5, SciPy 1.18.1 and scikit-learn 1.9.1. A later focused Ward repair retains the original execution identity and reruns only its affected program, five boundaries and three library fixtures.
- [Pure-model author packet](evidence/k-means-hierarchical-models.json): exact small-fixture objectives, linkage, seeding, weights, cuts and reconstruction checks, with focused arithmetic amendments and archived original evidence. The visible UI is a bounded teaching model, not a general arbitrary-range numerical library.
- [Complete independent lesson review](K-MEANS-HIERARCHICAL-INDEPENDENT-REVIEW.md): original/replacement coverage, all complete programs, prose, mathematics and visual source, plus complementary calculations. Findings about the first Python execution bridge and Ward underflow are closed. The first-time learner now receives environment/install/save/run guidance; unrepresentable positive squared costs produce a rescaling error instead of false coincidence.
- [Independent model and visual review](K-MEANS-HIERARCHICAL-VISUAL-MODEL-REVIEW.md): fourteen changed library/rational/weighted cases and comparison of every diagnostic row to actual native output. The stopped-seeding caption and arithmetic-range findings are closed. Its later opened-image finding is tracked separately from the already completed numerical review.
- [Full production browser pass before the label-only amendment](archive/k-means-hierarchical-before-label-layout/k-means-hierarchical-browser.json): twelve passing cases against Edge 152. All six displayed programs/outputs, fourteen anchors, controls and reset/back, keyboard ranges, narrow 390/320 layouts, exact module order, stable completion, selected-topic loading and import/render recovery were operated. A real label/input association defect was repaired before this pass: the range now has an explicit unique input ID and associated label/output.
- [Latest production evidence](evidence/k-means-hierarchical-browser.json) binds the current checked source/build and retained earlier evidence. The final build passed in 17.52 seconds, followed by 54 focused annotation states across 1366/390/320 widths. Root and the independent visual reviewer opened the corrected [desktop Lloyd](evidence/k-means-lloyd-desktop.png), [phone Lloyd](evidence/k-means-lloyd-mobile.png) and [seeding](evidence/k-means-seeding-desktop.png) captures. Point and center labels are readable, leaders identify unchanged markers, and the original collision finding is closed. Selected tree, mean, diagnostic and palette captures retain their earlier review. DOM overlap checks were supplemented by actual image interpretation.

Revisions were driven by those concrete findings. Source hashes identify what was checked; they do not replace reasoning or numerical/rendered evidence. Unchanged native, mathematical and reader checks are reused rather than relabeled as fresh executions after a CSS or annotation change.

## Integration boundaries and handoff

Curriculum and generated-artifact checks passed: 28 modules, 1,218 stable topic IDs, 352 individual briefs, seven guided paths, 228 publication mappings and 245 separately loaded outlines. Classical ML remains 39 topics and this lesson stays at position 11, after Survival Analysis and before PCA. Only this topic's selected body and actual shared dependencies load on its fresh reader route; no unrelated lesson body or syllabus outline was requested.

Production builds use the installed Vite entry directly because this machine's global npm shim is unavailable. Existing warnings in the untouched Bayesian Networks lesson and the shared navigation chunk remain outside this change. Browser payload figures are built file sizes and gzip estimates, not measured network compression, latency or a device-performance benchmark. Browser checks loaded the site's intended Google Fonts through the verification environment; no application network or security configuration was changed.

The final selected-route JavaScript total is 1,448,219 built bytes, with a 325,636-byte gzip estimate, including the actual shared runtime/navigation/math dependencies. The final label layout retains bounded fixed-fixture work and disconnects its resize observer on unmount. It does not alter the numerical model or displace plotted data to make labels fit.

The eight reviewed production sources are bound in [the completed scoped ledger](classical-ml-unsupervised-progress.json); the inventory reflects this exact-source implementation review. Retained captures, original source, compact evidence and small finding archives support future work without reopening completed suites. Three disposable working files were removed: the final build log, the failed temporary label-harness log and that superseded harness. Its honest failed outcome remains in the [layout amendment](evidence/k-means-label-layout-amendment.json); it is not counted as a successful check. The temporary production preview was stopped; the user's development preview and unrelated work were preserved.

There is no remaining implementation action in this increment. User acceptance remains separate, and no deployment was performed. The next request may authorize PCA, but this one-topic increment does not.

## Proposed revision, 12 September 2026 — awaiting the user's decision

The user asked for an independent quality opinion on the completed lesson and then for the resulting plan to be implemented so the result could be judged. The revision below is implemented in the working tree and is **not** committed, reviewed or accepted. The eight-source hashes in the earlier records and the ledger no longer describe the working tree; every displayed program was re-executed and the model, native and browser verifiers were rerun and pass on the new sources. Independent review of the changed prose, figures and labs has not happened.

Content changes: one caution callout in section 1 replaces the per-paragraph qualifications (disclaimer phrases roughly halved, from 27 to 19 across a longer body); an explicit first-pass route; the real Old Faithful dataset (272 rows, Härdle 1991 version via the R datasets mirror) introduced in section 1, fitted in section 8 and used for the k = 1..8 diagnostics on a logarithmic inertia axis; a forty-leaf SciPy Ward dendrogram of a deterministic subsample; the Lloyd/Ward 7.6875 coincidence made explicit with the Ward-initialized k-means trick; verified facts added with sources (NP-hardness of the objective, Vattani's exponential lower bound, smoothed polynomial iterations, Murtagh and Legendre on the two Ward conventions, centroid-linkage inversions, Lloyd's 1957 note); a changed-unit geyser exercise; a changed-seed capstone self-check drawn from the native oracle; ESL and ISLR added to the alternate resources. The displayed Python programs drop the arithmetic-underflow guards and printed moral sentences so the mechanisms stay visible; validation is a finiteness check, stated as such in prose.

Lab changes: every investigation records a prediction and compares it with the model's answer. Lloyd accepts any two seed rows, draws the nearest-center bisector and has a run-to-fixed-point control. Feature geometry enumerates all seven two-group splits instead of comparing two initializations. Seeding accepts successive D² draws and shows exact probability against 200 seeded draws and the uniform baseline. Hierarchy adds a chain-plus-triple fixture with exact unit spacing on which single and complete linkage disagree, plus merge stepping. Palette adds a 37-color sky raster and a 160-color gradient. New evidence captures are in `evidence/`; the old Lloyd, seeding, tree, mean, diagnostics and palette captures were regenerated by the updated browser verifier.

If the user accepts, the next actions are: independent review against the new source identity, a refreshed source-bound ledger, and a commit. If the user declines, `git checkout -- .` and removal of the four untracked files restore the reviewed state.
