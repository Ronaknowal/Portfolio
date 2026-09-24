# Graph Fundamentals: author verification

Stable ID: `graph-fundamentals-adjacency-laplacian-connectivity`. Mathematics position 30. Author verification completed across 10–11 September 2026 local time. The final native check ran at 18:20:08 UTC on 10 September; the final complete browser review ran at 18:29:50 UTC. This is author evidence, separate from independent review, root production integration and user acceptance.

Read the [lesson design](GRAPH-FUNDAMENTALS-LESSON-DESIGN.md) for the scope decision, prerequisite and destination review, research inspected and representation contracts. The [durable evidence](evidence/graph-fundamentals-author-review.json) contains the exact final production and verification-source fingerprints, native results, browser results and hashes of the final screenshots actually opened.

## What the rewrite teaches and preserves

The lesson starts with an actual five-station picture beside its weighted adjacency matrix. It then develops row/column conventions, weighted walks, weak/strong connectivity, local disagreement and total energy, incidence currents, component kernels, isolate and loop conventions, three averaging rules, anchored interpolation, graph features, one fully explained GCN-shaped calculation and prediction-time data availability. Proof hypotheses and counterexamples are part of the ordinary reading route. Spectral eigenmodes/cuts/clustering remain the next topic.

The [original source archive](evidence/graph-fundamentals-original-content.json) preserves the full previous body. The exact original five-vertex connectivity/Laplacian Python program and output remain in the new examples. Native verification compares their trimmed bytes. Useful earlier graph modeling, sparse storage, connectivity, normalization, features and evaluation outcomes are retained; the rewrite clarifies them rather than discarding them. Title, stable ID and module order are unchanged.

All eleven complete Python programs have a visible prediction question, actual verified output and an explanation. Python setup is supplied before the first program. Ten independent changed-context tasks have separate optional hints and explained answers. The exercises include constructing counterexamples and modifying actual learner functions, not only copying shown calculations.

## Visual and interaction evidence

| Representation and placement | Actual contract checked |
| --- | --- |
| Initial graph beside adjacency, at the first matrix definition | Declared label order, mirrored undirected cells, disconnected blocks and the missing bridge are visible before a learner reaches code. |
| Graph/matrix/walk investigation, after the walk mechanism | Edge/direction changes update the same adjacency and enumerated weighted walks. Zero-step identity/absence, weighted return, directed reverse edge and reset passed. |
| Directed reachability inline figure | A→B↔C has one weak component and two strong components; the table includes zero-step reflexivity. |
| Laplacian/energy investigation | Signed local contributions can cancel while whole-edge energies remain positive. Signal/bridge keyboard ranges, componentwise constants and reset passed. Bars state their changing numerical scale and count each undirected edge once. |
| Incidence node→edge→node figure | The actual drops, currents and signed outflows agree with the declared orientation and weights. Units are conditional on a physical conductance interpretation. |
| Normalization workbench | H L H gives zero isolate rows; the naive identity formula differs. Sticky-isolate P has row sum one. A self-loop changes row degree and normalized operators while canceling from L. Both constant and square-root-degree inputs were inspected. |
| Averaging graph and trajectories | Conservative ordinary mean, degree-weighted mean, periodic neighbor averaging, lazy damping and the growing discrete counterexample are calculated from actual iterates. Previous/step/reset and step 24 passed. No empirical time or benchmark claim. |
| Anchored interpolation investigation | Unanchored components remain explicitly undetermined. Independent boundary, bridge and anchor changes repair the correct components; solved values/residuals agree with independent systems. |
| Prediction-time inline comparison | Known structure at times 2/3 is distinguished from the future target edge at time 6 for a time-5 prediction. Test-node membership alone is not called leakage. |

The final browser script used Microsoft Edge, actual Space Grotesk and JetBrains Mono fonts, and viewports 1440, 390 and 320 pixels wide. At each width it checked all ten anchors by keyboard and actual arrival coordinates, five investigations with changed and boundary states, all eleven exact question/program/output matches, ten hint/solution pairs, sixteen rendered display equations and seven direct annotated references. It found no page errors, unexpected console errors, page/container overflow, SVG text overflow or math-renderer errors. The development HMR socket is deliberately closed by the harness; its separately recorded Vite warning is an environment consequence, not a hidden lesson error.

Ninety-four final captures were saved under `scratch/graph-fundamentals-browser/`. Actual opened final images include every inline figure, all five investigations, loop/isolate and growing-mode states, connected interpolation, ordinary sections 4/6/8/9, the changed harmonic solution and references. The durable record names and hashes the opened subset; saving a screenshot alone is not claimed as visual inspection. Native code/output can scroll locally, preserving conventional program formatting.

### Defects and refinements resolved during review

- Added the initial graph/matrix inline figure after finding that a beginner otherwise had to imagine the graph before reaching the later workbench.
- Corrected the examples' result field to the shared renderer's `expected` contract, regenerated every program and verified visible output. Program semantics were preserved.
- Wrapped the isolate transition/generator equation across logical lines at 320 pixels. Scoped SVG typography, loop bounds and trajectory margins avoid clipped labels without shrinking all lesson text.
- Used a compact viewport for the three-node directed figure. Removed authoring-process commentary from learner explanations and corrected singular walk-result grammar.
- The first rapid-scroll screenshots contained transient unpainted browser tiles. Fresh DOM/geometry and a separate real-browser capture showed the full graph. The final harness waits two animation frames and 150 ms after each long-page jump; opened final energy captures contain the complete D—E edge, node labels and controls. No model or product CSS workaround was introduced for this capture issue.
- Harness-only corrections used valid attribute selectors for numbered anchors and actual keyboard modality for focus assertions.

## Native and independent-oracle checks

Executed:

```text
scratch/lesson-tools/Scripts/python.exe -X utf8 -I scripts/prepare-graph-fundamentals-examples.py
node scripts/format-graph-fundamentals.cjs
scratch/lesson-tools/Scripts/python.exe -X utf8 -I scripts/verify-graph-fundamentals.py
node scripts/review-graph-fundamentals.cjs
node scripts/review-graph-fundamentals-reading.cjs
```

The browser command used the approved read-only loopback/public-font network path. The final formatting comparison conserved normalized JavaScript AST/string values and CSS meaning; the subsequent singular/plural result-copy correction intentionally changed only that visible sentence.

The separate final reading check confirms that Python setup precedes the first program and captures its actual preceding question at all three widths. The 390/320 images were opened. Final production source freeze: 2026-09-10T18:34:10.109910Z, with six exact hashes in the durable record.

Actual environment: Python 3.12.14, NumPy 2.3.5, NetworkX 3.6.1. All eleven standalone programs ran in isolated Python processes and matched stdout. The original program/output preservation assertion passed.

The complementary oracle suite checked 140 weighted undirected graphs against NetworkX/NumPy and exact arithmetic, 64 directed fixtures for strong/weak/reflexive reachability, 45 weighted walk cases against explicit sequence products, 420 anchored/partly unanchored systems against independently solved NumPy systems and 45 averaging configurations against matrix powers. It checked empty/isolated vertices, duplicate edges, self-loops, absent zero-weight edges, disconnected components and fifteen rejected input contracts. Tiny nonzero readouts remain scientific notation rather than being erased as zero.

Changed practice executed actual learner functions for reordered adjacency and row/column action, changed walk weights, signed incidence/energy, ordinary-versus-weighted averaging and rational interpolation. The four-node 1–2–1 weighted path with boundary 0 and 12 gives 24/5 and 36/5; an extra unanchored isolate triggers the native program's explicitly stated global-uniqueness rejection. The browser model deliberately supports a partial determined/undetermined result. A directed counterexample independently demonstrates that undirected symmetry/kernel claims cannot be transferred blindly.

Maximum checked energy discrepancy: 0. Maximum anchored residual: 8.881784197001252e−15. These finite checks complement the lesson's proofs; they are not exhaustive mathematical verification.

## Research scope and remaining boundaries

The design records exact inspected portions of Spielman's Laplacian and random-walk notes, MIT's incidence lecture summary, NetworkX's normalized-Laplacian source and clustering definitions, and Kipf/Welling's normalization and learning setup. The official MIT video title, creator and course description were verified; full-video playback is not claimed. The lesson supplies both the alternate recorded-lecture route and the inspected written summary.

All graphs and plotted values are original finite mathematical examples computed in code. No force-layout geometry, clinical/sensor accuracy, convergence speed benchmark, physical experiment or GCN benchmark was reproduced. Smoothing is a modeling assumption. General directed/signed operators, full mixing theory, spectral partitioning and trained neural architectures require their own stated assumptions and later owners.

Root owns production loading/build, shared registry and curriculum validation. The completed [independent review](GRAPH-FUNDAMENTALS-INDEPENDENT-REVIEW.md) and [its durable evidence](evidence/graph-fundamentals-independent-review.json) check the final paired amendment below. User acceptance remains pending; publication and author verification do not substitute for it.

## Independent-review arithmetic amendment

The independent reviewer found that an accepted edge weight of `1e-310` produces a degree whose reciprocal overflows. `graphNormalizations` now explicitly rejects non-finite inverse degrees with `RangeError` and asks for rescaled weights. This preserves its exposed finite inverse-degree contract and the existing zero-degree convention.

The full native/oracle suite passed again at 2026-09-10T18:49:34.198594Z. Three additional regression inputs (`1e-310`, `5e-309`, `Number.MIN_VALUE`) are rejected; a representable reciprocal at weight `1e-308` still gives the correct finite normalized Laplacian, walk generator and stochastic transition. The amended model hash is `ba77012855dd218f7be27f58cae58ed789475359cb1febef41cdf9ec23d82546`; the durable record preserves the earlier model hash/freeze and the amended freeze identity.

The same guard was then applied to the actual displayed Python `normalized_operators` helper, using `math.isfinite` and `ValueError`. Both language implementations passed the full native suite and paired boundary checks at 2026-09-10T18:52:41.684641Z. All eleven stdout values remain unchanged. The amended examples hash is `940d6038e91a4249d29849c706b869ee4f04afc9bf4c5bd2f34a2c369605a454`.

`node scripts/review-graph-normalization-range.cjs` passed a targeted actual browser check at 1440/390/320 with public fonts loaded: the corrected import/guard and unchanged output are rendered, the question still precedes the program, and there are no page errors or page overflow. The final 320-pixel capture was opened. The durable record retains this additional browser evidence and both amendments, while the earlier full investigation checks remain separately attributed. Body, lab rendering, CSS and brief fingerprints are unchanged; the bounded lab controls cannot select the rejected subnormal weights.
