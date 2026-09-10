# Vectors, Matrices & Tensor Operations — verification

10 September 2026. Stable ID `vectors-matrices-tensor-operations`. [Teaching design and source-review ledger](VECTORS-MATRICES-TENSORS-DESIGN.md). This record covers this lesson only, not the wider curriculum's completion.

## Implementation and preservation

The title and route are unchanged. The original affine fixture, image-batch transformations, token projection, indexing/masking, reductions, broadcasting, transpose, reshape, moveaxis, stacking/concatenation/splitting coverage is retained and developed with complete programs and interpreted outputs. Added geometric foundations, units, basis images, projection derivation, linear versus affine maps, composition order, rank/null-space and solvability readiness. Corrected the original conflation of complex transpose and conjugate transpose; qualified cosine at zero, matrix rank versus array-axis count, and legal-but-semantically-wrong broadcasting.

Four distinct investigations address vector projection, matrix geometry, product row/column accumulation and tensor reduction provenance. Three inline figures address head-to-tail addition, composition order and transpose versus reshape. Six independent exercises have optional hints and explained solutions; distributed diagnosis and exploratory transfer supplement them. Descriptive exercise titles avoid apparently skipped numbering when returning to the final practice section. The next route is Matrix Decompositions, preserving the module sequence. No LeetCode quota was imposed on mathematical practice.

## Computational evidence

Run `node scripts/verify-vector-tensor.mjs`. It calls `scripts/verify-vector-tensor-native.py` with the configured lesson Python runtime and executes all ten displayed standalone NumPy programs. Actual result: **pass**, Python 3.12.14 and NumPy 2.3.5. Machine evidence: `scratch/vector-tensor-verification/results.json` (07:17:04 UTC).

| Independent check | Actual coverage |
| --- | --- |
| Projection versus `numpy.linalg.lstsq` | 3,969 signed/zero direction-vector pairs; reconstruction, residual orthogonality, cosine conditions and squared-distance decomposition |
| Matrix maps versus NumPy | 343 map/input cases across all seven maps |
| Matrix products versus NumPy | 540 seeded signed products over all dimensions 1–3 |
| Selected cell / partial sum | 6,480 states checked against independent NumPy dot-prefix calculations |
| Tensor reduction versus NumPy | 189 reductions from fixed and random tensors |
| Highlight/source provenance | 2,268 independently enumerated coordinate/value correspondences |
| Worked and transfer contracts | 11 groups, including null-space systems, shape rules, reindexing, batching and exercise results |
| Invalid input contracts | 19 grouped checks covering parser shape/finite/bounds, dot dimensions, product compatibility, partial terms and tensor axes/fixtures |
| Complete displayed programs | 10 exact stdout comparisons: projection, maps, resources, affine, reduction, broadcasting, reindex, images, tokens and nullspace |

The oracle does not compare an implementation only with itself: projection uses a least-squares solver; products/reductions use native NumPy; highlighted provenance uses independent coordinate enumeration. Invented fixtures are mathematical teaching examples, not measured datasets or benchmarks.

## Browser and visual evidence

Run `node scripts/review-vector-tensor.cjs`. Actual full interaction pass: **1440×1000 and 390×1000**, Edge headless, reduced motion. Machine evidence: `scratch/vector-tensor-browser/results.json` (07:25:11 UTC). At each width:

- 18 projection states; signed, zero, perpendicular and extreme inputs; all direction presets, keyboard ranges, reset, exact readings and SVG endpoint consistency.
- 28 matrix-map states; all maps with zero/signed/extreme inputs, exact row/column readings, shared plotting domain and coordinate-to-SVG mapping.
- 53 product selection/partial states and seven invalid input scenarios; active calculation survives invalid submission; every output cell, stepping, backtracking and reset.
- 48 tensor dataset/axis/output states; exact highlighted sources, values, means, output identities and reset.
- Ten complete rendered programs, ten unique working section anchors, six initially closed independent-practice hint/solution pairs, keyboard disclosure, next-module route, safe external-link attributes, minimum control height, no page errors and no horizontal page overflow.

The first full run had twelve source links. The final annotated direct-video and basis/dimension additions bring this to fourteen; `node scripts/review-vector-tensor-reading.cjs` separately checks the final source and ordinary-reading state at both widths. This supplemental pass verifies final aligned equations and descriptive practice headings after the last source polish. Evidence: `scratch/vector-tensor-browser/reading-results.json` and its dot-reading/map-reading/practice-reading/sources screenshots. No KaTeX errors or horizontal page overflow were found.

Screenshots were actually opened, including projection, shear, product, tensor provenance, reindexing, head-to-tail addition, collapsed map, composition, ordinary dot-product reading, practice and resources. The initial map plot made the unit square too small; the final shared adaptive domain exposes both original and transformed geometry. The initial projection was nearly parallel; v=(1,4) now makes the perpendicular remainder visible while preserving the earlier v=(3,2) worked example as an explicit control task. Long formulas now use aligned lines instead of unnecessary mobile sideways reading. These are visual review findings, not inferred from a successful build.

Formatting evidence: `scratch/vector-tensor-verification/formatting-results.json`; Babel-normalized AST equality across the owned lesson, lab and model files, including JSX text and raw template values. This check is after the deliberate formula/title edits, so it establishes that the subsequent formatting itself changed no behavior or content.

## Sources and limits

Primary textbook and NumPy API content was substantively reviewed; the design record identifies exact pages and limitations. The 3Blue1Brown direct video is annotated using the reviewed creator's substantive text companion. MIT session notes, problems and solved examples were read as extracted PDF text. No entire video viewing, transcript inspection, live learner study, or accessibility certification is claimed.

The later einsum lesson has a saved [destination note](topic-notes/tensor-algebra-einsum-notation.md), including explicit-output/diagonal semantics and a proposed mathematical tensor bridge. The receiving author must investigate and decide rather than blindly paste it. A scoped NumPy 2.3.5 check confirms `ii->i` on [[1,2],[3,4]] is [1,4], implicit `ii` is 5, and `i->` on [1,2,3] is 6.

Parent reported an integrated production build and loading/recovery checks passing before the final equation/title/formatting polish. Parent owns the final build after the source-freeze signal; this record does not label that earlier build as proof of the later edits. The lesson is ready for that integration and user review.
