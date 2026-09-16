# t-SNE, UMAP & Manifold Learning — design and content handoff

12 September 2026. Classical ML position 17 of 39, stable ID `t-sne-umap-manifold-learning`. Requested mode: **research/write only, content first**. This record owns scoped design, research, author evidence and continuation; root owns the shared delivery ledger and current handoff. No commit or publication was made by this author.

| State | Current result |
| --- | --- |
| Content | Complete manuscript, worked examples, changed practice/hints/solutions, source annotations and actionable visual/investigation contracts |
| Implementation | Not started for this revision; older published body remains unchanged |
| Computational verification | Bounded content probes executed for exact examples, probability/gradient reasoning and real-data PCA/t-SNE inputs; UMAP library execution and formal native/model campaigns deferred |
| Browser/visual review | Not started; all figures and investigations are specifications |
| Independent review | Not started; root's batch reconciliation is distinct from formal phase-two independent review |
| User review | Not requested or inferred |
| Next action | On explicit finish authorization, run topic inventory with `--work finish`, read this complete packet, then implement, execute, review and integrate it |

## Authoritative packet and source conservation

- [Manuscript](drafts/t-sne-umap-manifold-learning/lesson.md)
- [Visual and investigation specifications](drafts/t-sne-umap-manifold-learning/visual-specifications.md)
- [Offline real data](drafts/t-sne-umap-manifold-learning/digits-300.csv) and [provenance](drafts/t-sne-umap-manifold-learning/data-provenance.md)
- [Reproducible content calculations](drafts/t-sne-umap-manifold-learning/content-calculations.py) and [actual calculated inputs](drafts/t-sne-umap-manifold-learning/calculated-inputs.json)
- [Destination-note disposition](topic-notes/t-sne-umap-manifold-learning.md)

The entire original `src/learn/data/topics/t-sne-umap-manifold-learning.jsx` was read, including all eleven sections, source annotations, purported outputs and exercises. Its baseline is the batch's commit `8c5da59f18516be77c29d5aeeafca3decca4f738`; original working-file SHA-256 on entry was `3d8d75e6a52660038c97286e979ec514352acfb9fc1d8cfdd08cbd24410029a3`. No redundant original copy is created. Root binds the final packet hashes in the two-phase ledger. Those hashes establish identity, not correctness.

Read the nested `AGENTS.md`, complete current teaching standard, handoff, topic-design brief, relevant ML/domain playbook, learning code standard, note policy and returned note. Ran `node scripts/build-curriculum-inventory.mjs --topic t-sne-umap-manifold-learning --work content`; it identified the old publication with both new-revision phases not started and the TDA-origin note. Existing unrelated work was preserved.

## Scope and learning contract

Retain the current title, stable identity and module order. The title genuinely promises more than two visualization APIs. Preserve Isomap, MDS and LLE with actual mechanisms and worked calculations; place their longer treatment on a clearly marked return route. No syllabus rename or new topic is justified.

Beginner finish line: explain why a neighbor graph can add a shortcut; follow distance→affinity→map; audit a selected observation with actual neighbor identities; distinguish map area from member count. Intermediate finish line: compare real PCA/t-SNE representations for a chosen metric/k and use correct training/transform boundaries. Deeper finish line: calculate classical MDS/LLE examples, derive the t-SNE gradient, distinguish a stated UMAP full-pair cost from its sampled implementation, and explain a controlled topology distortion.

First-pass route: sections 1–8, investigations A–D, digits program, practice 1–4. Deeper route: sections 9–10 plus changed MDS/LLE/deployment/independent practice. Final readiness questions only require the core mechanism and audit; optional eigenvector derivations do not silently become a core prerequisite.

Prerequisites checked against current content/briefs:

| Required skill | Exact topic / actual bridge | Decision |
| --- | --- | --- |
| Euclidean distance, weighted sums and array rows | `Vectors, Matrices & Tensor Operations`; local coordinates and units are introduced in §§1–2 | Local examples are self-contained |
| Scaling, linear reconstruction, fitted transform | `PCA & Dimensionality Reduction`, inspected current scaling and fit/validation sections | Link/review, then explicitly state why common pixel scaling differs from feature standardization |
| Density versus assignment weights | Preceding `Gaussian Mixture Models (GMM) & EM Algorithm`, coordinated with author | Bridge without requiring its pending implementation or reusing responsibilities as neighbor probabilities |
| Graph vertices, weighted edges and shortest paths | `Graph Fundamentals (Adjacency, Laplacian, Connectivity)` brief inspected | Teach finite graph mechanics locally; no unexplained graph-theory prerequisite |
| Symmetric eigenproblems and graph Laplacian | `Eigenvalues & Eigenvectors`, `Spectral Graph Theory` current brief | Deeper-only with Gram/eigenvector refresher and explicit ownership link |
| Future separation of measured mixtures | Next `Independent Component Analysis (ICA)`, coordinated with author | End with source-separation question on measured matrix; never suggest feeding a t-SNE map to ICA |

## Preserve, repair and extend

| Original useful coverage / weakness | Current treatment and learning reason |
| --- | --- |
| Curved structure, Swiss roll, geodesic motivation | Preserve with exact U route and radius-shortcut calculation before broad manifold language |
| t-SNE conditional probabilities, perplexity, symmetrization, Student-t, KL and gradient | Preserve and derive; add achievable-perplexity/tie example and exact finite-difference-checked tiny optimizer |
| Early exaggeration and initialization | Preserve, repair loss-normalization interpretation, remove universal PCA-initialization and deterministic claims |
| UMAP local scales/fuzzy union/cross-entropy | Preserve, correct exponent clamp, self-count calibration convention and membership interpretation; separate graph construction, ideal full-pair loss and actual sampled schedule |
| UMAP new-point support and supervised use | Repair contradictory “no transform”/“parametric graph” claims, separate ordinary transform from encoder, teach frozen preprocessing and fold-local coordinate systems |
| Isomap/LLE/MDS comparison | Deepen to actual matrix, local reconstruction and globally constrained coordinate calculation; distinguish strain/stress and sparse/dense solver choices |
| Scaling | Preserve memory arithmetic and algorithmic mechanisms; remove unmeasured speed rankings, invented seconds, hard sample ceilings and n^1.14 law; correct 500k float64 matrix from 2 PB to 2 TB |
| Digits application | Preserve useful domain, replace unsupported “verified” outputs/representative-centroid points with embedded licensed measurements and native coordinates |
| Interpretation cautions | Consolidate at §3; repair claim that every plotted neighbor is an input neighbor and distinguish plotted area from count |
| Practice | Replace answer-implied diagnoses with changed geometry, entropy, edge weights, MDS/LLE values, pipeline repair and a reproducible independent image audit |
| References | Keep canonical theory and accessible alternatives; no unsupported historical dominance, universal ranking or “watched video” claims |

Applications have distinct instructional purposes: digits test whether real relationships survive; the new-image pipeline distinguishes fixed-collection mapping from inference; the four-point Rips example isolates topology changes without specialized biology assumptions. No decorative industry list or unverified clinical/scientific claim is added.

## Canonical-reference coverage check

Canonical orientation: the full [scikit-learn manifold chapter](https://scikit-learn.org/stable/modules/manifold.html), with original t-SNE and UMAP papers as mechanism references. Read chapter section list and mechanisms 2.2.1–2.2.9. This is an accessible canonical implementation chapter, not a claim that no other standard reference exists.

| Canonical headline | Disposition |
| --- | --- |
| Isomap graph→shortest paths→eigenproblem | Core bridge §2, complete return branch §9.2 |
| Standard LLE, local weights/regularization | Complete §9.3 with exact changed practice |
| Modified LLE and rank/regularization issue | Explained scoped extension in §9.3; specialist derivation deliberately outside this broad lesson |
| Hessian LLE and tangent-space alignment | Mechanistic comparison §9.4 with local-flat/second-order prerequisite bridge; full derivations signposted as specialist reference |
| Spectral/Laplacian embedding | Objective and graph/operator distinction §9.4; full spectral theory already has a current lesson/brief |
| Classical, metric, nonmetric MDS | Explicit objectives and exact line recovery §9.1; PCA equivalence connected rather than merely repeating a number |
| t-SNE crowding, asymmetric objective, optimization, approximation | §§4–5,8,10, including normalization derivative and implementation-specific conventions |
| Original t-SNE paper's SNE lineage, random-walk/landmark large-data branch | Symmetry/heavy-tail improvements covered; later tree/interpolation methods teach practical scaling. Historical random-walk derivation deferred as optional source reading; it is not necessary for the selected CPU workflow |
| UMAP theory, graph construction, optimization, transform | §6 plus §8; categorical proofs and nerve-theorem assumptions remain with mathematical follow-up, not silently asserted as practical guarantees |
| UMAP actual effective loss versus nominal formula | Included in core §6 because omitting it would make the pseudocode/formula explanation inaccurate |
| Density-aware, supervised, parametric variants | Ordinary transform and parametric distinction taught; supervised fit boundary made explicit. Full densMAP and supervised-UMAP workflows deferred pending a task with a density/label-aware objective, not claimed equivalent to ordinary UMAP |

The existing TDA note requested a controlled before/after topology comparison and neighborhood audit. It is included as F2/D and §9.5/F9, adapted to exact constructed geometry. No actual persistent-homology runtime or topology certificate is claimed. The note stays visibly open for phase-two implementation closure. No new cross-topic omission was established requiring a separate destination note; known spectral/TDA/geometry/ICA owners are explicitly linked and root owns the batch's NMF note.

## Hurdles, representations and assessment

| Hurdle | Mechanism / representation | Complete example / evidence |
| --- | --- | --- |
| Ambient shortcut versus surface route | F1 + editable graph investigation A | Six-unit U, radius changes, changed sensor corridor |
| A map can invent or omit neighbors | F2 identity arrows + D query tile audit | Four-point R=0,T=.5; actual source30 local reversal |
| Perplexity is entropy, not a hard k | F3 normalized bars + B editable distance row | Numeric Gaussian row, equal-distance null, changed entropy task |
| P versus Q and total motion | F4 signed contributions, gradient derivation | Four-point exact optimizer and numerical derivative |
| Fuzzy strength versus calibrated probability; graph versus layout | F5 calibration/union + C editable local formula and ideal pair separation | log₂4 calibration, .625 union, finite ideal optimum, null edges |
| Useful picture versus valid inference | Real digit maps, original neighbor tiles, separate train/validation program | Same 300 rows; label-use disclosure; changed pipeline repair |
| MDS/Isomap/LLE are different preservation choices | F7 matrix-to-line, F8 local recipe, graph reuse | Exact centered coordinates, reconstruction weights, changed practice |
| Topological input versus projected picture | F9 thresholded square and duplicate projection | Exact [1,√2) loop interval lost after projection |

See the specification for every individual figure's entities, formulas, exact data, placement, mobile/text alternatives and pending rendering checks. Every investigation specifies unset committed predictions keyed to current inputs, an entity edit not resolved in prose, meaningful checked contrast and null. No figure/lab count is used as quality evidence.

## Claim/source ledger and review boundaries

All sources below retrieved 12 September 2026. Versions are document or local-runtime snapshots, not assertions about newest release. No long passages copied. Original author explanations, calculations and code teach the lesson; links supply verification and alternative reading.

| Claim / source locator | What was reviewed and used | Caveat / outcome |
| --- | --- | --- |
| [t-SNE original paper](https://jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf), §§2–4, Appendix A | Probability definition, crowding, symmetrization, gradient, early optimization; Appendix A inspected directly | No borrowed empirical outputs; derivation independently checked numerically |
| [TSNE API](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html), parameters | 1.9.1 auto rate max(N/α/4,50), iteration rename, init, perplexity, approximation conventions | Native1.9.1 fits recorded; no generic claim that every t-SNE implementation lacks transform |
| [Trustworthiness API](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.trustworthiness.html), definition/neighbor constraint | Rank penalty, k<n/2 | Local stable-sort retention distinguished from native library score sorting with ties |
| [Manifold chapter](https://scikit-learn.org/stable/modules/manifold.html), full section list and method sections | MDS distinctions, Isomap stages, LLE regularization/variants, Laplacian/tangent formulations | Original Isomap/LLE author-hosted pages were attempted but failed through the browser tool; do not claim those original PDFs were read |
| [Swiss Roll And Swiss-Hole Reduction](https://scikit-learn.org/stable/auto_examples/manifold/plot_swissroll.html) | Official same-family geometric comparison and its code | Reference not rerun; the lesson's worked graph is its own exact fixture |
| [UMAP explanation](https://umap-learn.readthedocs.io/en/latest/how_umap_works.html), complete page | Topology motivation, local scales and graph→layout explanation | Its broad topology phrasing is interpreted through explicit finite graph/layout boundaries here |
| [UMAP source](https://umap-learn.readthedocs.io/en/latest/_modules/umap/umap_.html), `smooth_knn_dist`, fuzzy union, parameter validation, transform path | Self-column exclusion, log₂k target, positive-distance rho, clamp, scale floors, graph composition | Documentation header0.5.8; latest source pages can move. Recheck installed-version source in phase two |
| [UMAP parameters](https://umap-learn.readthedocs.io/en/latest/parameters.html) | n_neighbors/min_dist/spread/output-dimension meanings and limits | Parameter roles are not empirical guarantees of global/density preservation |
| [UMAP paper](https://arxiv.org/html/1802.03426v3), algorithm/optimization sections | Theory versus discrete graph and sampled optimizer | Categorical proof sections not claimed fully derived or independently verified |
| [Damrich & Hamprecht2021](https://proceedings.neurips.cc/paper/2021/file/2de5d16682c3c35007e4e92982f1a2ba-Paper.pdf), §§3–6 | Nominal/actual loss distinction, role of negative sampling, controlled ring counterexample, effective-loss argument | No large experiment reproduction; avoid importing one convention's exact effective-loss coefficient into another implementation |
| [UMAP transform](https://umap-learn.readthedocs.io/en/latest/transform.html) and [reproducibility](https://umap-learn.readthedocs.io/en/latest/reproducibility.html) | Fitted reference embedding, train/test boundary, seed/thread tradeoff | UMAP dependency absent; complete program outputs remain unexecuted |
| [Parametric UMAP](https://arxiv.org/abs/2009.12981) | Encoder distinction from ordinary graph-based extension | No neural performance guarantee or executable encoder workflow claimed |
| [openTSNE simple usage](https://opentsne.readthedocs.io/en/stable/examples/01_simple_usage/01_simple_usage.html) | Reference-map transform example |1.0.0 documentation header; not installed or executed |
| [Barnes–Hut paper](https://jmlr.org/papers/v15/vandermaaten14a.html) and [FIt-SNE](https://www.nature.com/articles/s41592-018-0308-4) | Tree and interpolation acceleration mechanisms | No timings or universal hardware speedup borrowed |
| [UCI optical digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits) and [sklearn dataset description](https://scikit-learn.org/stable/datasets/toy_dataset.html#optical-recognition-of-handwritten-digits-dataset) | Dataset purpose/collectors, block-count features, original test portion, license | Balanced subset/retrieval/source rows documented; not MNIST or a representative prevalence sample |
| [Distill t-SNE essay](https://distill.pub/2016/misread-tsne/) | Substantive density, spacing, parameter and convergence explanation | Page read; controls not operated in this phase;2016 defaults not treated as current |
| [McInnes slide transcript](https://speakerdeck.com/lmcinnes/a-guide-to-dimension-reduction) and [talk listing](https://pydata.org/nyc2018/schedule/presentation/1/) | Substantive matrix-factorization/neighbor-graph slides, especially slides48–74 | Video not watched; honest alternate medium through creator's slides. Simplifications not used as precise algorithm definitions |

## Checks actually performed

The scoped authoring command ran successfully and the returned notes were read. Native runtime inspected only at the exact shared path `scratch/lesson-tools/Scripts/python.exe`: Python3.12.14, NumPy2.3.5, scikit-learn1.9.1; `umap` module absent. No environment mutation or recursive scratch scan occurred.

Executed `scratch/lesson-tools/Scripts/python.exe docs/teaching/drafts/t-sne-umap-manifold-learning/content-calculations.py`. The first content probe checked six PCA-init t-SNE fits and revealed the equal-seed null. A focused extension added the two random-init fits and replaced a tie-affected native identity-score null with a tie-free constructed one; the regenerated input record faithfully reflects the extended calculation. This is a bounded content-data exercise, not an implementation campaign.

Recorded outcomes in the linked JSON:

- CSV shape300×66,30 rows per label; all original feature integers retained. File hash matches provenance. Native coordinates for PCA plus eight t-SNE parameter/initialization settings, with R/T/continuity at k5,10,20.
- R10: PCA.376333; t-SNE PCA-init p5.710333,p30.770333,p80.762333. Seeds7 and19 are exactly equal at each PCA-init perplexity in this runtime. Random-init p30 R10=.772000 and.769667 supplies the contrast.
- U radius.75/1/1.5/2: no route/6/4.828427/2. Unequal-distance Gaussian rows change probability and entropy as stated; equal-distance rows remain uniform. Fuzzy union and ideal pair costs match the manuscript.
- Four-point t-SNE P is symmetric, off-diagonal normalized; objective.0535224029→.0182274938 under the specified200 updates. Analytic gradient versus central finite differences maximum absolute error1.2095e−10.
- Exact MDS B agrees with centered line outer product; eigenvalues numerically0,0,38/3. False-neighbor four-point fixture R1=0,T1=.5; identical tie-free fixture R1=T1=1. Rigid-transform retention is1.

Supplemental bounded query audit read the already computed coordinates without refitting. At source_row30,k10, original neighbor IDs `[0,166,229,160,36,276,140,266,178,79]`; PCA preserves6, t-SNE p30/s7 preserves5. The specification stores both complete map-neighbor lists and additional k5/k20 local-reversal examples. This establishes the promised local-versus-global contrast. Matrix memory, changed entropy/fuzzy/MDS/LLE exercises and square-filtration states were independently calculated from their equations; no runtime persistence algorithm was claimed.

Final scoped content checks: Python fenced programs parse; written practice uses closed hint/solution disclosures; relative packet links resolve; internal `/learn/path/` IDs match the publication manifest; CSV and coordinates remain finite with consistent row identity; manuscript/spec source values agree. The root will validate and bind the final phase checkpoint after any reconciliation fixes.

**Deferred:** full displayed-program extraction/execution, UMAP installs/fits/transform scores, runtime React/SVG/models/labs, native/browser verifier suites, figures at actual desktop/phone sizes, accessibility/keyboard interaction, performance/loading, formal independent review and production integration. No app build was run for this documentation-only packet.

## Author learning-experience assessment

This is the author's heuristic assessment of the written packet, not a novice study, independent review or rendered-page pass.

1. **Route:** core route explicitly ends after a useful real-data audit and inference boundary; matrix/eigenvector and derivation branches are marked where they begin. Final retrieval stays within that core.
2. **Cautions:** map geometry/count/density claims live together in§3. Achievable perplexity lives in§4, optimizer conventions§5, nominal-versus-sampled UMAP§6, fitting boundary§8. Removed the original repeated warning cycle; teaching code prints values only. Later references to these distinctions have a new local calculation or operation rather than an appended generic disclaimer.
3. **Real question:** opening digit similarity question returns with measured R10 and exact image IDs. The selection rule and original collection purpose are visible. Local row30 reverses the aggregate ranking, preventing a simplistic winner story.
4. **Investigations:** all four contracts have unset recorded predictions, editable geometric/data entities, active-input keys, meaningful contrast and a null. UMAP C explores actual local formulas and a clearly labeled ideal pair; it does not impersonate the sampled library optimizer. Fixture evidence is distinguished from pending control/render checks.
5. **Figures:** inline reading pass found the main hidden steps: route versus shortcut, probability normalization, pair-versus-total force, directed-to-union edge, actual pixel neighbors, MDS distance-to-coordinate recovery, LLE weights and filled versus unfilled topology. Each has a dedicated representation at its explanatory location. Shared scales, PCA baseline and phone composition are specified; perceptibility at rendered size is pending.
6. **Connections:** counted route distance becomes Isomap input; Gram construction explains PCA/MDS agreement; the same union strength feeds the ideal pair example; ICA begins with measured mixtures. Canonical LLE variants/spectral methods are retained with explained ownership and depth.
7. **Code:** compact full-input programs emphasize affinity/gradient/audit/transform operations. Numerical validation is outside the displayed mechanism. Unexecuted UMAP outputs are clearly labeled and never replaced with fabricated numbers.
8. **Practice:** changed corridor, probability distribution, fuzzy weights, local-recipe geometry and pipeline task require independent reasoning; exact outcomes are supplied. The real-data task gives ID-based success checks and permits a local result differing from the average.
9. **Screenshots:** deliberately not taken in content-first mode. The specification names informative states to capture later, including source30 after prediction check, seed null/random-init contrast and full-width map/axis readability. This item remains pending, not passed.

No material written-content gap remains. Implementation can improve a representation or correct content with an explicit reason and refreshed source checkpoint; it should not repeat the research phase by default or treat this record as approval of unbuilt visuals.

## Prepared-content implementation — 14 September 2026

The user authorized finishing the next complete research/write packet. The `--work finish` preflight selected **t-SNE, UMAP & Manifold Learning**, Classical ML position 17 of 39, revision 1. This continues the saved manuscript rather than starting another research campaign. [The entry baseline](evidence/manifold-implementation-baseline.json) preserves its previous phase state, all other 176 ledger-entry hashes and the publication manifest. The earlier deferred statements above describe phase one; this section and the central ledger own the current implementation state.

The reader now contains all 13 sections, nine inline figures, four distinct investigations, four complete Python programs, seven independent practice problems with separately closed hints/solutions, annotated alternate resources and the existing ICA continuation. Core and deeper routes remain separate. No topic title, stable ID, catalogue order or publication mapping was changed. A topic-owned blueprint now describes the actual learning path.

### Content conservation and necessary corrections

- Preserved the route/shortcut, false/missing-neighbor, Gaussian normalization, t-SNE gradient, UMAP calibration/union, real-digit audit, MDS, LLE and Rips-complex figures. Preserved specialist spectral methods and the exact optimizer derivation, rather than abbreviating the prepared scope to a pair of library introductions.
- Completed the deferred UMAP examples and replaced pending execution prose with actual measured outputs, versions, interpretation and limits. The four displayed programs match the executed exported code. The original nine saved PCA/t-SNE layouts and 300 source identities remain intact; one actual UMAP layout is added.
- Independent review corrected the sign in `KL = cross-entropy − Bernoulli entropy`. It also corrected a new implementation sentence that had grouped rigid rotations with point relocation: common rigid rotations preserve distances and neighbor selections.
- Kept resource purpose, level and version caveats in the reader; detailed author/reviewer activity stays in this record. No claim is made to have watched the linked talk video. The phase-one source ledger still records which article, paper sections and slide transcript were examined.

### Implementation and reproducible evidence

| Concern | Implemented treatment and evidence |
| --- | --- |
| Geometry investigation | Editable actual coordinates/endpoints and radius; old/new comparisons use the same proposed endpoints; disconnected routes, deterministic shortest-path ties, step inspection, reset/undo and invalid-input retention. Close points retain their true centers and use separately placed ID labels with leaders. |
| Probability investigation | Candidate distances and bandwidth, normalized bars and exact entropy terms; candidate or perplexity prediction; equal-distance null and middle-candidate nonmonotonicity. |
| Fuzzy graph investigation | Editable directed local membership inputs and graph support; exact fuzzy union; a separately scoped, explicitly ideal single-pair loss. An unapplied graph edit clears its pair prediction without resetting the applied separation. |
| Real-image investigation | All 300 actual images, ten saved maps, k = 5/10/20, source-ID keyboard selection, prediction before revealing lists, and at most 40 neighbor tiles. Global scores remain distinct from local counts. Display-only map tabs preserve the answer; consequential edits invalidate it. |
| Model/data verification | [Models](evidence/manifold-models.json): 349 assertions across six groups, including finite-difference gradient checks on original and changed inputs. [Data](evidence/manifold-data.json): 300 × 64 original pixel values, ten layouts, 90 measured diagnostics, stable source IDs, and the source 30 PCA 6/t-SNE 5 local reversal. |
| Native execution | [Native evidence](evidence/manifold-native.json): all four complete programs executed and their exported code replayed. UMAP 0.5.12 produces R10 = 0.7143, T10 = 0.9890. Held-out pixels/PCA-10/UMAP-10 accuracies are 0.9733/0.9733/0.9467, or 73/73/71 correct out of 75. The reader interprets this small split without a universal algorithm ranking. |
| Runtime convention | Python 3.12.14, NumPy 2.3.5, SciPy 1.18.1, scikit-learn 1.9.1 and UMAP 0.5.12. Original t-SNE arrays reproduce with 12 BLAS/OpenMP threads; TSNE `n_jobs=1` alone does not constrain every numerical pool. UMAP/other native programs use one numerical thread. Installed graph calibration, union, sampled optimization and transform source were inspected; hashes are in native evidence. |
| Downloads | Topic-owned public CSV, saved-coordinate JSON, provenance/license, runtime versions, actual UMAP coordinates and all four Python programs. Nothing is fetched from a third-party model service when a lab mounts; no browser optimizer is run. |
| Browser integration | [Production browser evidence](evidence/manifold-browser.json): nine behavior groups covering complete displayed content, four investigations, meaningful nulls/edits, actual IDs, mobile tabs, keyboard alternatives, completion persistence, sequence, failed-import recovery and active-lesson loading. Desktop/intermediate/phone checks use actual loaded fonts; zoom-equivalent checks use 320 CSS px and twice the pixel density. |

The build keeps the lesson and its numerical data in its on-demand module. The measured current lesson chunk is approximately 102 kB gzip; its exact decoded/compressed sizes and build identity are recorded in browser evidence. This is a current payload measurement, not a before/after speed claim. No other lesson body was requested on the initial route, and returning to the lesson reused its loaded module. The shared reader needed `min-width: 0` to allow its flex item to shrink at intermediate widths; unchanged PCA and GMM pages were also checked at 1366/780/390 px after that repair.

### Independent review and visual closure

Authorship and review are deliberately distinguished. The root integrated the reader/blueprint; separate authors implemented the figures, investigations and models/data. The model author independently reviewed the other authors' JSX and state transitions in [the technical review](evidence/manifold-independent-technical-review.md), including 92 state-helper assertions and 60 focused geometry/state delta assertions. The figure author independently checked four complementary lab/model risks in [the additional checks](evidence/manifold-independent-labs-checks.json). The lab author independently reviewed the reader and figures using the separate [learning-experience assessment](evidence/manifold-learning-review.md). These are expert heuristic reviews, not a novice usability study or user acceptance.

The [figure visual review](evidence/manifold-figures-review.md) records the actual all-nine-figure pass at desktop, intermediate and narrow widths, plus enlarged text and changed F6 selection/color/disclosure. Small-screen repairs keep F4 numeric values intact, stack dense F6 metrics and give shared axes enough room for negative ticks. [Shared geometry evidence](evidence/manifold-shared-geometry.json) checks D = (0.25, 1), G = (4, 4), radius 1 at 390/320 px: every ID remains readable and positions remain exact. A concise visible status avoids repeating the result paragraph; a persistent polite result region announces the applied outcome, with historical feedback outside it.

Root inspected the retained desktop graph and narrow probability, fuzzy-union, selected-digit-map and image-neighbor captures, including the close-point correction, plus F1, F4 and F9. Foreground-line/label triage candidates behind opaque graph node disks are intentional; they were visually inspected, not treated as confirmed text collisions. Figure-specific visual review is separate from a no-overflow assertion. A final enlarged-text check identified an unbroken resource-list token and a checkbox caption; topic-scoped wrapping fixes both, and the final browser run checks 780 px with 200% root text without page overflow.

### Author learning-experience closure and continuation

1. **Route:** the reader starts with the 300-image inspection question; core sections lead to a measured audit and a fitting boundary before the explicitly deeper branches.
2. **Cautions:** the central map-reading contract is applied through fresh calculations, without replacing each section with another generic warning.
3. **Real question:** source 30 reverses the global PCA/t-SNE ranking, and the held-out classifier comparison supplies an honest negative result for UMAP.
4. **Learner action:** all four investigations record a prediction, act on the topic's entities and include tested contrast/null cases. Editing invalidates stale answers and preserves the applied state.
5. **Perceptible figures:** each hidden mechanism has its own representation; labels, filled topology, actual pixels, equal metric axes and exact numerical alternatives were reviewed at rendered sizes.
6. **Connections:** graph routes become Isomap distances; probabilities become forces; graph weights remain distinct from a sampled layout; matrix/recipe methods and ICA ownership remain explicit.
7. **Code:** four compact programs teach the mechanism; native verification stays outside learner code. Published expected outputs are actual executions.
8. **Practice:** changed geometry, entropy, fuzzy weights, MDS/LLE positions, a deployment failure and an independent image audit require transfer rather than copying.
9. **Visual inspection:** actual informative states and images were examined. Expert inspection and bounded fixtures cannot guarantee every possible user-created configuration, but the discovered close-point and responsive failures were repaired and checked.

The content packet is retained with its reconciled prose and original calculation inputs. The one-use author conversion script was removed after producing the reviewed source. Native runtimes and referenced evidence remain useful for reproducing this work. Automatic approval review rejected the model author's optional deletion of the scratch execution directory because it contains untracked scripts/data; that directory was retained and the rejected deletion was not retried indirectly. It does not define unfinished lesson work.

Revision 1 content and implementation are complete after final source reconciliation; [the integration snapshot](evidence/manifold-integration.json) binds the final source and conservation checks. **Next eligible prepared topic: Independent Component Analysis (ICA), Classical ML position 18.** It remains content-complete and implementation-not-started; this request does not authorize its implementation.
