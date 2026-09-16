# Independent technical review: manifold lesson implementation

Date: 14 September 2026. Status: **PASS after three targeted corrections**. No open mathematical or prediction-state defect was identified in the reviewed source snapshot.

## Scope and independence

The reviewer authored the separate manifold models/data/native-example layer, but did **not** author the four JSX sources reviewed here. This review covers other authors' lesson body, nine figures, shared investigation components and four investigations. It uses the complete saved manuscript and visual specifications, exact fixtures and native outputs as references. It does not claim independent review of the reviewer's own model/data/example implementation.

The review is bounded to mathematical fidelity, semantic data use and prediction transitions. Root and the figure author own actual desktop/phone captures, geometry, browser actions and final integration. The state probe below is not a substitute for those checks.

## Findings, changes and closure

| Finding | Consequence and requested correction | Final inspected state |
| --- | --- | --- |
| P2: IdealPairLab's React key included the outer **unapplied** graph-draft key. | Editing d or a local scale remounted the pair and reset an already applied r=2 to r=1 even though the active graph weight had not changed. This violated the graph/layout separation and last-valid-state contract. | `ManifoldLabs.jsx:205–213,282` now keys the pair by reset count and applied graph inputs. A separate `graphDraftKey` effect clears the prediction/history while retaining the active separation and weight. Source closure passed. Root owns the browser regression. |
| P2: The body said adding Bernoulli entropy to cross-entropy gives KL. | For a fixed graph pair, KL(Bern(w) ∥ Bern(ν)) = CE(w,ν) **minus** H(w). The prepared manuscript contained the same wording, so this was a necessary mathematical correction during implementation. | `t-sne-umap-manifold-learning.jsx:215` now says subtracting, and explains that the constant depends only on the input graph. Independent numeric check at w=.625, ν=.2: CE=1.0895775270141415; H=.6615632381579821; KL=.42801428885615933; CE−H equals KL. |
| P2: New map prose grouped rigid rotation with moving/jittering observations. | A common rigid rotation preserves pair distances and all neighbor selections. Grouping it with pointwise relocation contradicted the explicitly taught invariance/null. | `t-sne-umap-manifold-learning.jsx:253` now distinguishes individual point changes from a common rigid rotation. Source closure passed. |

Only the relevant authors changed production sources. The reviewer recorded findings and inspected the resulting corrections.

## Mathematical and data consistency

- **Radius route and Isomap:** the U has six unit edges at radius 1; corner shortcuts at 1.5; a direct crossing at 2; a disconnected state at .75. The graph investigation recomputes the baseline on the old graph using the **newly proposed endpoints**, preserving a like-for-like comparison. Coordinate edits, no-path reporting, positive radius bounds, duplicate rejection and lexicographic route semantics agree with the specification. The finite route list gives edge and cumulative lengths.
- **Local Gaussian row:** both the figure and investigation use d²/(2σ²), normalize across the same identities and label entropy in bits and perplexity as 2ᴴ. The investigation compares the proposed distance row at old versus proposed bandwidth, rather than accidentally changing both row and bandwidth between its baselines. Its displayed raw Gaussian weights are deliberately distinguished from the numerically stabilized normalization. Equal-distance and common-scale explanations are correct; it does not promise monotonic probability for every middle candidate.
- **t-SNE:** P is the retained exact fixture; its symmetric ordered-pair normalization agrees with the factor 4 gradient. F4 passes the retained P and initial Y into the verified objective, draws **negative-gradient** contributions and applies the .5 step size separately. The pair/total distinction, signs of A's three contributions, Q normalization Z=4, full initial gradient table and centered first step are consistent. The body preserves normalized KL versus exaggeration boundaries and differentiates Z to obtain repulsion. Its cited ≤1.3×10⁻¹⁰ error explicitly belongs to the retained authoring probe, not a claim about every changed-map numerical derivative.
- **UMAP:** the calibration counts three nonself strengths for a neighbor array of length 4, targeting log₂4=2. Reciprocal memberships use the **same physical distance** with separate local scales; the .5/.25 strengths combine to .625. Retention flags change directed support independently, offsets/scales are validated, the clamp is retained, and the graph/local-scales label is explicit. The pair panel holds the active graph weight, distinguishes the analytic single-pair BCE from sampled optimization, handles w=0/1 asymptotic minima and shows an out-of-range finite minimum honestly. Reflection changes labels/orientation while retaining separation and cost.
- **Identity and metrics:** both native maps and every 8×8 tile use original `sourceRow` identity. Rows and coordinate positions are joined by the same saved array order. The query model receives full-precision coordinates; rounded text never becomes a fit or ranking input. Local neighbors exclude self and use the stated source-ID tie rule. Native T/continuity are labeled with their separate scikit-learn sorting convention. R remains the retained directed selection fraction, not a rank-weighted score. Source30's local reversal remains available without being revealed in D's initial question.
- **Actual results and interpretation:** the body uses the executed example module through `RunnableExample`'s actual `expected` contract. It reports the preserved reference-map scores, actual UMAP 0.5.12 output, 12-thread digits-versus-one-thread UMAP execution distinction, and unfavorable 73/75 versus 71/75 held-out result without extrapolating a universal ranking. All four programs remain complete. The ten stored maps and downloads support independent audits without a browser optimizer.
- **Classical methods and topology:** F7 retains the exact distance matrix, one-ninth Gram matrix, centered coordinates and eigenvalue 38/3. F8's input/output locations preserve the 2/3,1/3 recipe while doubling distances; it does not generalize nonnegative weights to all LLE. F9 distinguishes metric positions from abstract complex drawing positions, preserves duplicate identities and zero-distance edges, and includes filled triangles/tetrahedral simplex at the correct closed thresholds. The loop interval is [1,√2), with no corresponding projected interval. These are constructed examples rather than fabricated UMAP outputs.

## Prediction-transition review

All four investigations use the same inspected state helper. Predictions start unset; an explicit commit stores a complete draft snapshot/key; an apply without a matching commit cannot evaluate; edits clear the commitment; results retain their historical status; invalid drafts preserve active inputs; undo restores the prior active snapshot and clears prediction; reset restores useful initial inputs. Numeric count/union controls validate their ranges and supported increments before enabling commit.

| Investigation | Question inputs and reveal boundary inspected |
| --- | --- |
| A — graph | Coordinates, radius and endpoint IDs are in the draft key. Switching only the point-edit selector does not mutate coordinates. The old/new route comparison uses the same proposed endpoints. Disconnected baselines switch to the existence question. |
| B — probability | Distances, bandwidth, candidate and question type are in the key. Changing the distance row deliberately affects both bandwidth-comparison rows and is explained beside the prediction. |
| C — fuzzy graph and ideal pair | Shared distance, both offsets/scales and both retention flags are keyed. Pair predictions use the active graph's w. After the targeted repair, an outer draft change invalidates the pair answer without changing applied separation. Applying a new graph starts the corresponding pair context. |
| D — digit audit | CSV hash, source ID, k, metric and saved-map key are stored together. Query/k/map edits invalidate revealed counts and neighbor lists. The ordinary display tab and optional color overlay are presentation-only. Candidate lists, local score and relationship edges stay hidden until the matching committed reveal. |

A bounded Node probe extracted the actual `useManifoldInvestigation` function from `ManifoldShared.jsx` and supplied a deterministic `useState` slot harness. **92 assertions passed** over the four lab input shapes: unset state; refused uncommitted apply; committed key; cleared prediction after edit; unchanged active values; applied result; cleared commitment; invalid draft retention; hidden current result/history; blocked invalid commit; undo; and reset. The probe evaluates the actual state-helper function but does not mount React, inspect asynchronous browser effects or substitute for the root's browser regression. Per-lab evaluation formulas and reveal conditions were read directly against the models and specification.

## Reviewed source hashes

SHA-256 binds this technical review to the final inspected snapshot. Later source changes require a relevant delta review; unchanged numerical data/model evidence is reusable.

| Source | SHA-256 |
| --- | --- |
| `src/learn/components/lesson-labs/ManifoldLabs.jsx` | `d438e2b2922eaafc0e4aeac0268a895fa6d178b6abcaee0de5a6640aa5552679` |
| `src/learn/components/lesson-labs/ManifoldShared.jsx` | `3b23be0b7836d0197213de9907e616172a26414dd29cc5e0a507630c9645d7ea` |
| `src/learn/components/lesson-labs/ManifoldFigures.jsx` | `d95fdffdbd8b3c2021e7faa6f70ee74ce94360a32244a55b7d808c84c63218f3` |
| `src/learn/data/topics/t-sne-umap-manifold-learning.jsx` | `5a149b1cea182600d624492d9b252fa89aa9b35046197980505d781caacfff85` |

References used: the complete retained `docs/teaching/drafts/t-sne-umap-manifold-learning/lesson.md` and `visual-specifications.md`; `manifold-models.json`, `manifold-data.json`, and `manifold-native.json` for established numerical/runtime facts. Review limitations and source ownership remain distinct from model authorship, visual review and user acceptance.

## Final geometry/state delta review — 14 September

The integration owner requested a bounded follow-up for `GraphMarks`, the wider `EqualPlot` gutter and shortened apply-status wording. **Geometry and state fidelity pass.** The unchanged figures/body need no repeat mathematical campaign; their earlier source hashes still match.

- `GraphMarks` derives each marker's `px=sx(x)` and `py=sy(y)` from the entered coordinates. Radius reduction and separately positioned label/backplate/leader objects do not enter the graph model, edge endpoint calculation or route length. The label search reserves actual marker bounds and axis-label space and prefers locations clear of foreground edges. The always-visible applied-ID/coordinate sentence remains an independent text equivalent. This source review does not certify every possible label intersection; root owns rendered-layout evidence.
- `EqualPlot` uses `230/span` for x and `−230/span` for y after the gutter change. The 330×298 viewBox changes margins; the actual plot remains a 230×230 square, so distances and circles preserve equal axis scale. Digit rows and graph points still pass their exact coordinates through these functions.
- The apply handler still evaluates the committed snapshot, stores the full `evaluation.message` and observed result, updates active inputs, keeps the previous state for undo, and clears prediction/commit. Only its live status text changed. The semantic select-label association added in this revision also preserves control values and handlers.

A focused source-extraction probe passed **60 geometry/state assertions**: the U, a close A/B fixture, and a concentrated quarter-grid fixture with a distant point retained exact marker centers, equal x/y factors, available close-point labels and unchanged source point objects. A focused apply probe retained exact observed values and full feedback after the shorter status change. These are pure source/hook probes, not React DOM or postbuild browser evidence.

The accessibility follow-up is **closed in source**. The initial generic live status no longer announced the actual outcome. The final `Prediction` component now retains one persistent, initially empty `data-manifold-result-announcement` region with `aria-live="polite"` and `aria-atomic="true"`; a current applied result fills it with the actual feedback. The generic status is non-live for that applied-result state, avoiding duplicate live announcements. Historical feedback remains outside the result live region, while draft/commit status messages retain their ordinary live behavior. This preserves both the stored result and the specification's outcome-announcement intent. Root owns postbuild browser verification; this source closure does not claim an assistive-technology listening test.

The following hashes supersede the earlier table for this bounded delta snapshot:

| Source | SHA-256 |
| --- | --- |
| `src/learn/components/lesson-labs/ManifoldLabs.jsx` | `7513207cf1cbcc1a7ccaf8c540e2bb72163b5e741643a31787c80355f052a6ef` |
| `src/learn/components/lesson-labs/ManifoldShared.jsx` | `d8cdd1540ca5c9e51e1e778346c20a7aec0e593b36d59597784c12f5a664050a` |
| `src/learn/components/lesson-labs/ManifoldFigures.jsx` | `d95fdffdbd8b3c2021e7faa6f70ee74ce94360a32244a55b7d808c84c63218f3` |
| `src/learn/data/topics/t-sne-umap-manifold-learning.jsx` | `5a149b1cea182600d624492d9b252fa89aa9b35046197980505d781caacfff85` |
