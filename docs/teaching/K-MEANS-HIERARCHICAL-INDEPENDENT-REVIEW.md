# K-Means & Hierarchical Clustering: independent complete-lesson review

Reviewed 11 September 2026 by `scientific_visual_improvements`. This reviewer researched the original lesson but did not author the replacement production sources. **Source, mathematics and pedagogy review; no independent browser run or screenshot inspection is claimed.** Root owns the actual-route browser review and final integration.

Status: **closed at 2026-09-11 17:15:53 UTC** against the final eight-source identity below. Both reported findings are repaired and received focused closure. No unresolved material source, mathematical or pedagogical finding remains within this bounded review. Actual rendered browser behavior and production integration remain separate gates.

## Material actually inspected

Read the entire replacement body, including all optional branches, nine changed practice tasks with separate hints/explanations, capstone instructions and annotated resources. Read all six complete exported Python programs and their expected output, the pure JS model, all five labs, four inline figures, both CSS files, the stable-ID blueprint and full design. Compared this coverage with the entire original read documented in [the source review](K-MEANS-HIERARCHICAL-SOURCE-REVIEW.md).

The replacement retains the original six-point Ward data and 150-row blob generator; objective/mean derivations; initialization; four linkage definitions; complete native/library fitting; diagnostic reasoning; scale/memory; shape, units, high-dimensional data and outlier cautions; labels; and mixture/medoid/kernel/BIRCH/FAISS connections. It repairs the inaccurate original claims rather than retaining them as authoritative prose. Quantization and a frozen held-out report now provide explicit end-to-end outcomes. The old generic text traces are replaced by mechanism-specific forms with described conventions.

All important mathematical read-through checks passed: the two conditional minimizations; consistent ties/empty slots and finite stabilization; fixed point versus optimum; weighted means and feature weights; vanilla versus greedy seeding; Ward's factor of two and telescoping increments; monotone height versus merge-count cuts; single-linkage connectivity; k-selection and silhouette qualifications; count-weighted palette/CF identities; spherical versus ordinary centers; kernel-mean expansion; and the unique-nearest small-variance mixture limit. The nine changed practice solutions were recalculated during the read, including the multi-step 0,2,3,9 Lloyd example and changed 3-versus-2 Ward merge.

The beginner route defines the data unit and grouping question, gives a manual assignment and mean update before formal notation, and revisits the same six points for the hierarchy. Optional branches add proofs and specialist connections while core explanations remain visible. The initially missing first-program execution bridge is now repaired, as recorded below. The actual next topic remains PCA in the Classical ML module; deeper evaluation/density/mixture links are not substituted for that sequence.

## Reused author evidence, with attribution

- [Native author evidence](evidence/k-means-hierarchical-native.json): six complete program outputs, **49** independent native oracle states and **5** explicit rejected cases, timestamp `2026-09-11T16:45:46.356564+00:00`. Its examples hash matched the initial read. Reviewed the recorded outputs and relevant contracts; did **not** rerun those complete programs or claim ownership of their checks.
- [Model author evidence](evidence/k-means-hierarchical-models.json): **126 grouped checks and 13 rejected inputs**, timestamp `2026-09-11T16:51:55.264Z`. Its model/lab/CSS hashes matched the initial read. Read the verifier's relevant invariants and reused this unchanged campaign rather than repeating it.
- Earlier [original-source review](K-MEANS-HIERARCHICAL-SOURCE-REVIEW.md): current-library definitions and source inspection, vanilla theorem scope, direct Ward arithmetic, and the entire official MIT hierarchical-clustering transcript. The replacement correctly annotates that lecture's centroid-linkage convention and heuristic gap interpretation. No video playback is claimed.

## Complementary checks actually executed

At `2026-09-11T17:03:01.544Z`, an inline Node module imported the actual JS model and passed:

1. **20 isometry/partition states:** points `(0,0),(.2,.1),(2,0),(2.1,.2),(.4,3)` versus the rotation/translation `(x,y) → (100−y,−70+x)`. For each of single/complete/average/Ward, compared all merge heights and SSE increments, then co-membership for every k from 1 to 5. Height differences were below `1e−12`, cost differences below `1e−11`; partitions agreed.
2. **Five independent graph-closure comparisons:** at thresholds `0,.23,1,1.81,3.01`, directly formed pairwise Euclidean threshold edges using `Math.hypot`, took Boolean transitive closure, and compared every pair's connectivity with the actual single-linkage height cut. All agreed. This checks the explanatory MST/connectivity bridge by a different mechanism, not another agglomeration loop.
3. **One changed weighted-mean case:** points `(1,−2),(2,0),(7,3)`, weights `2,3,1`, memberships `0,0,1`, prior centers `(0,0),(0,1),(5,5)`. Compared actual weighted mean/error with explicitly expanded repeated rows and confirmed the third empty center remained `(5,5)`. Centers agreed exactly; errors agreed within `1e−12`.

Separately extracted only imports and function definitions from the actual displayed Ward program using Python's AST, without running its unchanged demonstration. Singleton separations `1` and `1e−100` returned the expected heights and costs. Separation `1e−200` exposed the arithmetic finding below. No scratch files, production mutation, build or browser session was created for these checks.

## Findings and disposition

### 1. First optional program lacks an execution bridge

The first `Program` appears in section 3, before any install/save/run instructions. Its `Program` wrapper renders the question and `RunnableExample`; the shared component shows code/output only. The page assumes imports succeed but supplies no direct action for a first-time reader to run the complete example. Requested a concise local setup note before that block: required Python/packages, save the full block as a file, execute it with the same Python environment, then compare output. This need not change the programs or add another lab.

Disposition: **resolved by the author, independently reread in the final body**. Before the first scaling program, section 3 now distinguishes browser labs from locally executed Python; creates a Python 3.12 virtual environment; gives operating-system-specific activation and a direct-interpreter fallback; installs the three exact recorded package versions; and tells the reader to save the complete block as `clustering_scaling.py`, run it in that same environment, and compare output. Later blocks are explicitly standalone files. This closes the source/pedagogy finding. The reviewer did not install another environment or claim a browser rendering check.

### 2. Native Ward silently loses a nonzero singleton height

Actual initial helper `ward_merges([[0.0],[1e-200]])` passes its finite-coordinate and magnitude checks, but squares the difference to zero. It returns both `delta=0` and `height=0`, although the correct singleton height is the representable number `1e-200`. A coherent teaching arithmetic contract should reject/rescale an unrepresentable squared cost, as the Lloyd helper already does, rather than claim coincidence. There is no request for arbitrary precision or an unbounded numeric API.

Observed initial results:

| Separation | Returned height | Returned Δ | Correct singleton height |
| --- | ---: | ---: | ---: |
| 1 | 1 | .5 | 1 |
| 1e−100 | 1e−100 | 5e−201 | 1e−100 |
| 1e−200 | 0 | 0 | 1e−200 |

Disposition: **resolved and independently reproduced on the final displayed helper**. The author now checks nonzero coordinate differences whose squares become zero and positive squared separation whose Ward scaling becomes zero, with an explicit rescaling error. The reviewer reread the actual guard and again extracted only the displayed Ward imports/functions with Python's AST. The three original probes now return `(height, Δ) = (1,.5)` and `(1e−100,5e−201)`, while `1e−200` raises `ValueError: Rescale coordinates: squared separation is outside this arithmetic range.` No unchanged demonstration or unrelated program was rerun.

The native packet's `amendments` entry at `2026-09-11T17:05:13.505532+00:00` preserves the original module bytes, binds the before/after hashes, records unchanged Ward stdout and the five other unchanged programs, and attributes the author's focused five boundary cases plus three changed Ward/library fixtures separately. Those author checks were inspected and reused, not relabeled as reviewer execution. The fixed visual fixtures do not expose this case, so this is a changed-native-input contract finding, not a claimed browser defect. The broader JS functions describe bounded teaching fixtures rather than a general numerical library; this review does not certify every representable floating-point input outside actual UI states.

## Initial exact source identity

These hashes bind the complete initial read and its 26 complementary model checks, distinguishing them from subsequent arithmetic guards, copy repairs and browser styling changes. The final amended identity is listed separately; earlier evidence is not retroactively claimed to have run on amended bytes.

| Source | SHA256 |
| --- | --- |
| `src/learn/data/topics/k-means-hierarchical-clustering.jsx` | `0fa55c15ecfac4c00eb5f17b2ecd1b577721951859367c829b0f82b6b1310878` |
| `src/learn/data/k-means-hierarchical-examples.js` | `a74cd60a330ce9f2e8703887edeb008c48ad5fd7d1538838d85d502852b3675b` |
| `src/learn/data/k-means-hierarchical-models.js` | `50c1deafd66c785b3cf10419e9bc10bf5059deeee364fa79fff62a92d6543c20` |
| `src/learn/components/lesson-labs/KMeansHierarchicalLabs.jsx` | `73aac20e9777f0cd1934787e28401df38582f7f8c553b2174e85347277fe6fbb` |
| `src/learn/components/lesson-labs/k-means-hierarchical-labs.css` | `e789c5d4dd5d57ce149119e56f4d2924a0504faff8e11c446e14868d712d9ad3` |
| `src/learn/components/lesson-labs/KMeansHierarchicalFigures.jsx` | `a92ed0e2afb46582521543d2cdbb68daad7e2039f2da648e28aaae9336192d5e` |
| `src/learn/components/lesson-labs/k-means-hierarchical-figures.css` | `d591dae1795fec22b6d2440de21e3968e9c8be17ae1ce3c73ec6b43b0d9a028e` |
| `src/learn/data/curriculum/blueprints/k-means-hierarchical-clustering.js` | `6f74f6a098dc78efbd85b99cdf6672dd8594925d5267594a73cf09a66441d86a` |

## Focused final-source review and evidence reuse

The final body was reread at the setup bridge, wrapped equations, silhouette notation, dendrogram leaf legend and capstone introduction. The multiline metric/objective/update/seeding/SSE/Ward/silhouette formulas preserve their mathematical expressions. Silhouette distances `a(i), b(i)` now differ visibly from assignment index `a_i`; the leaf legend explains `P4·2` as observation and cluster; and the capstone explains its illustrative thresholds, bootstrap resampling, fixed comparison rows and distance-returning `transform` before the code. The four-row diagnostic figure's production bytes remain identical to the complete initial read. The final figure CSS adds only narrower optional-branch horizontal padding below 360px; this source review does not establish rendered fit.

Read the separate [independent visual-model review](K-MEANS-HIERARCHICAL-VISUAL-MODEL-REVIEW.md) and its [exact evidence](evidence/k-means-hierarchical-visual-model-review.json). That reviewer independently checked the amended numerical model, including positive-square/Ward/weight/probability range guards, representable tiny separation and true duplicates, along with 14 complementary hierarchy/weighted/palette/figure cases. Its final numerical-model hash matches the final identity below. Its source-level stopped-seeding caption correction and the original 126-check author campaign remain separately attributed. This review did not repeat either campaign or claim its initial 26 checks ran after those arithmetic amendments.

After that record closed, root's actual-browser work found an implicit range-label association problem. This reviewer read the final `Range` component and `useId` import: one unique `controlId` connects the outer label's `htmlFor`, the range input's `id` and the output's `htmlFor`, avoiding implicit association with the preceding labelable output. This is a focused source check of the parent-reported repair; root owns the pending actual keyboard/label regression check. The final lab hash below therefore intentionally differs from the earlier visual-model packet. No numerical model change accompanied this final association repair.

## Final exact source identity

Read from disk at the focused closure. These eight hashes include the last range-label repair, unlike the earlier author/model-review snapshot.

| Source | SHA256 |
| --- | --- |
| `src/learn/data/topics/k-means-hierarchical-clustering.jsx` | `bf89d7ba6add457467b803aaf5458936de661ee3227e857dd697483661f7596f` |
| `src/learn/data/k-means-hierarchical-examples.js` | `49989054e94ec00232f88462e39f3777786e363a0673407489e5e18d77b060b2` |
| `src/learn/data/k-means-hierarchical-models.js` | `7ed25d9b1a26d33364c80f9102a03e5e4cc7b6c9e108ea5ad06994e2f38f050d` |
| `src/learn/components/lesson-labs/KMeansHierarchicalLabs.jsx` | `715a18e243d9f1115d287e1d314c3d7ca0fc42d306718261e35f49cd3ea29226` |
| `src/learn/components/lesson-labs/k-means-hierarchical-labs.css` | `5d45d3ea076cdf00275c3c74ffeb352cb7151ef515ca2638122a5cd6d8a45477` |
| `src/learn/components/lesson-labs/KMeansHierarchicalFigures.jsx` | `a92ed0e2afb46582521543d2cdbb68daad7e2039f2da648e28aaae9336192d5e` |
| `src/learn/components/lesson-labs/k-means-hierarchical-figures.css` | `25f9e3311801550d271c8b45b589fc80eb520ef00dd20d23419ec8b8ec83b07b` |
| `src/learn/data/curriculum/blueprints/k-means-hierarchical-clustering.js` | `6f74f6a098dc78efbd85b99cdf6672dd8594925d5267594a73cf09a66441d86a` |

## Scope limits

No independent browser interaction, screenshot inspection, fresh performance benchmark or full-environment dependency install occurred in this stage. Visual review here concerns source-derived quantities, labels, geometry and explanatory contracts; root's actual-font responsive/keyboard/reading evidence must establish rendering. Native outputs and the finite campaigns above do not certify global clustering optima beyond the specifically enumerated examples. Full integration and user acceptance remain separate from this review.
