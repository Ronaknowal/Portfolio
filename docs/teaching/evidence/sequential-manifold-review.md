# Sequential manifold implementation review — 16 September 2026

Scope: Classical ML position 17 only, `t-sne-umap-manifold-learning`. Independent reviewer read the complete prepared manuscript, visual/investigation specifications, current reader, all nine figure implementations, four investigations including the optional ideal-pair stage, shared state helper, models, styles and executed example module. The integration owner owns the new production-browser run and repairs. This record does not advance the queue.

## Finding M17-1 — comparison feedback hides a graded change

**P2, open at this review snapshot.** Investigation B grades the unrounded change at tolerance 1e−9 but displays both comparison values at five decimal places. Supported inputs can therefore produce feedback that rejects “Stay the same” while printing `1 → 1`. The individual probability display at six decimal places also prints `1` in the example below. The numerical model is correct; the explanatory feedback does not expose the distinction it grades.

Reproduction:

1. In B set distances B/C/D to `0.25, 3, 6`, set bandwidth to `0.25`, commit a prediction and apply to establish that active bandwidth.
2. Change only proposed bandwidth to `0.5`, leave candidate B selected, predict “Stay the same”, commit and apply.
3. The grader observes “lower” but its message prints `Candidate B probability: 1 → 1`. With the same bandwidth comparison and the perplexity question, it observes “higher” while printing `Perplexity: 1 → 1`.

Direct evaluation of the current model gives:

| Quantity | σ = 0.25 | σ = 0.5 | Difference |
| --- | --- | --- | --- |
| B probability | 1 | 0.9999999827421723 | −1.72578277e−8 |
| Perplexity | 1 | 1.0000003257415508 | +3.25741551e−7 |

The relevant source is `ManifoldLabs.jsx`: `changeName` at line 14, comparison/evaluation at lines 190–195; `ManifoldShared.jsx` owns the five-place `format` default. Repair should preserve the mathematical distinction while making the compared values or signed difference legible and stating the equality tolerance. Merely widening the grading tolerance to conceal the result changes the question. Include both probability and perplexity in the affected browser regression, and retain exact-equality/null cases.

## Coverage and complementary review

No missing prepared section, numerical derivation, example, figure, practice problem or substantive investigation contract was found. The reader retains all 13 sections, nine mechanism-specific figures, four complete programs and seven independently closed hint/solution pairs. The numerical and visual representations track the same entities: U point IDs, directed ordered t-SNE pairs, actual source-row IDs and original digit pixels, MDS A/B/C, local reconstruction weights and coincident projection identities.

The complete manuscript-to-reader comparison preserves the core/deeper split; Isomap/MDS/LLE and specialist spectral methods; normalization, entropy and t-SNE gradient; fuzzy calibration versus union versus sampled layout; original-space metrics; held-out transform boundaries; computational limits; changed-constraint practice; annotated resources; and the ICA continuation. The addition of actually executed UMAP results is reconciled in the manuscript and implementation record. No new external factual claim was introduced by this review.

Investigations A/B show their last applied state rather than evaluating a changed draft early. C also retains the last applied connection and its separately scoped ideal-pair curve. Its initial connection is the already-worked example; that is not evidence of a changed-draft answer leak. D keeps local lists, count and relationship edges behind the matching prediction/reveal, while global metrics remain visible. F6 provides actual pixels/coordinates and aggregate scores without printing D's local answer. Source IDs, not subset indices, join the image tiles and maps. The local tie rule is stated separately from native library rank scoring. Null/support/boundary semantics are consistent in the inspected implementation, apart from the visibility issue above.

Source/CSS inspection found no competing class rule flattening the membership stroke-width attributes, and the SVG simplex fill has an explicit CSS fill. The reviewer also inspected retained narrow screenshots `manifold-fuzzy-390.png` and `manifold-figures/f9-phone.png`: the saturated membership strokes and filled topology states actually paint. These old images are corroborating visual evidence, not a claim to have run the current production browser. The integration owner's current run remains necessary for its selected changed states and paint/geometry checks.

## Reused evidence and source identity

All 12 source digests in `manifold-browser.json` matched the current tree at review entry, including the body, model/data/example modules, figures/labs/shared helper, three topic styles, blueprint and shared index stylesheet. All four displayed `manifoldExamples` code SHA-256 values and expected outputs exactly matched `manifold-native.json`. Existing native execution and the prior model/data/independent reviews therefore remain applicable; the browser record did not contain the near-saturation comparison above and cannot close it.

Reviewed entry hashes (SHA-256):

| File | Digest |
| --- | --- |
| `src/learn/data/topics/t-sne-umap-manifold-learning.jsx` | `5a149b1cea182600d624492d9b252fa89aa9b35046197980505d781caacfff85` |
| `src/learn/data/manifold-models.js` | `472158d05199facd42bd860dfe08b1780b7c3b5e523b39b0519a16ab4142ff10` |
| `src/learn/data/manifold-data.js` | `d76e917248b1d01180cef670f9489b369af430c84377593dbeac591d8cd966a0` |
| `src/learn/data/manifold-examples.js` | `5896f48a0c7c2d2741b6fac101cb61d3f9d7b1a1b5cd6920c20bf47addcdf9c2` |
| `src/learn/components/lesson-labs/ManifoldLabs.jsx` | `7513207cf1cbcc1a7ccaf8c540e2bb72163b5e741643a31787c80355f052a6ef` |
| `src/learn/components/lesson-labs/ManifoldShared.jsx` | `d8cdd1540ca5c9e51e1e778346c20a7aec0e593b36d59597784c12f5a664050a` |
| `src/learn/components/lesson-labs/ManifoldFigures.jsx` | `d95fdffdbd8b3c2021e7faa6f70ee74ce94360a32244a55b7d808c84c63218f3` |

Disposition at initial review: full content/source review complete with one actionable new finding. Closure requires the integration owner's repair and affected source/browser delta, not another native fit or broad content rewrite. Only this review record was edited by the reviewer.

## Repair delta and boundary review

**M17-1 fixed in source; production-browser evidence pending from integration owner.** `comparisonValues` retains compact feedback unless it would hide a change larger than the grading tolerance, then prints twelve significant digits. Both probability/perplexity and the optional ideal-pair cost feedback use it; B explains rounding and the 1e−9 equality convention. The numerical model and grading direction remain unchanged.

The reviewer executed the actual helper extracted from `ManifoldShared.jsx`, rather than a rewritten approximation. An exhaustive sweep of sorted three-distance rows on the supported quarter-unit grid, all twelve supported bandwidths and all bandwidth pairs, plus ideal-pair small-change/equality probes, examined **811,368 comparisons**. All **673,132** comparisons graded as changes retained distinct printed values with the correct numerical direction. Sorting loses no candidate possibilities because all three probabilities are tested. This is model/helper evidence, not React or paint evidence.

Delta hashes: `ManifoldShared.jsx` = `15a9068b37671c3417dab6beb5f09b2f87cd4c8ce0161e378279a34745de1a80`; `ManifoldLabs.jsx` = `382fe2a70b35a7971620ecf23596157ac1e6073e7180c6a1bd9b84582aa8d3b6`.

**M17-2 — P2, open at delta review:** C rejects the inclusive boundary of its stated estimate tolerance. Starting from its exact default union `0.625`, enter either legal prediction `0.620` or `0.630`, commit and apply. Both are mathematically 0.005 away, but JavaScript subtraction gives `0.0050000000000000044`, causing `Math.abs(computed.weight - prediction) <= 0.005` to reject them. This was reproduced directly against `umapConnection` and the actual grading expression. Use a small floating-point allowance within this bounded 0–1 domain; regression should accept both boundaries and reject `0.619` and `0.631`. The displayed prompt says “checked within 0.005,” so an inclusive threshold is the established contract.

**M17-2 fixed in source.** The final predicate adds `1e−12` to the `0.005` inclusive threshold. The reviewer extracted and executed the actual `predictionMatches` expression against the actual default connection: `0.620`, `0.625` and `0.630` pass; `0.619`, `0.631`, `0` and `1` fail. This allowance is far below the prediction field's `0.001` increment and preserves rejection just outside the contract. The new browser regression explicitly covers both boundary and both adjacent outside values. Final reviewed `ManifoldLabs.jsx` SHA-256 is `c9d573d804a2e4239c0a051fd385d62f4b7bd28c4cee02d2616203e4576b4724`; the helper hash remains `15a9068b37671c3417dab6beb5f09b2f87cd4c8ce0161e378279a34745de1a80`.

Independent source review is now closed with both findings repaired and no remaining source blocker. The integration owner reported M17-1's affected production-browser groups passing and is rebuilding/rerunning for M17-2; its final source-bound browser record owns that runtime closure. No additional topic or unrelated source was reviewed or edited.
