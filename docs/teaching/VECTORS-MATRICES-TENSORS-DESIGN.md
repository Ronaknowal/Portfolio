# Vectors, Matrices & Tensor Operations: teaching design

10 September 2026. Stable ID `vectors-matrices-tensor-operations`; Mathematical & Statistical Foundations, Linear Algebra. Read current handoff, teaching standard, mathematical domain playbook, topic-design and engineering standards, approved Linux and current Trees/Graphs teaching examples. Exact topic-plan command returned a published legacy lesson requiring individual design; no destination note existed and the open bit-manipulation inbox item was unrelated.

## Scope, preservation and finish line

Retain the title and identity: the existing title already covers the coherent scope. The old lesson had useful array-shape, affine-layer, broadcasting, reshape/moveaxis/reduction, concatenation and token-projection coverage. It lacked actual geometric teaching, substitutions, visual investigations, complete independent practice, annotated sources and verified outputs. Preserve those useful operations while teaching what they mean, including vector norm/dot/projection, matrix column images, composition order, span/rank/null-space and solvability readiness for the next Matrix Decompositions lesson. This is not a full course in abstract tensor analysis, decomposition algorithms or least squares.

Prerequisites: signed arithmetic, coordinates, simple algebra/function substitution and the Pythagorean theorem; explicitly refresh these at use. Recorded prerequisite titles: Algebra, Functions, Exponentials & Logarithms; Geometry, Trigonometry & Coordinate Reasoning. NumPy/Python is an optional computational route, with full imports and fixtures, not a prerequisite for following mathematical reasoning. Explain rows-as-observations versus mathematical column-vector conventions directly.

Observable outcomes: name axes/units and distinguish object dimension from array-axis count; add/scale/project vectors with conditions; connect matrix columns and row dot products to one output; explain composition and failure of commutativity; predict tensor reduction/reindex/broadcast shape and meaning; diagnose legal-but-wrong operations; combine these in an independent measurement/prediction task; recognize zero-direction, lost-information and floating-point limitations.

## Coverage decisions

| Idea | Existing evidence | Decision and owner |
| --- | --- | --- |
| Shapes and typed data | Existing scalar/vector/matrix/tensor overview | Retain, deepen units, named axes and coordinate choices; current lesson |
| Vector addition, length, dot, projection | Dot formula only; no geometric derivation | Teach here with numerical substitution and projection residual; geometry prerequisites refreshed |
| Matrix as a map, columns and rank loss | Existing weighted-sum sentence | Teach column images and linearity; introduce span/independence/rank/null space and one small solve because next decomposition lesson needs them |
| Matrix multiplication / affine batches | Existing 4×3/3×2 program | Retain full original fixture as runnable example; new pack/resource fixture teaches units and one cell before neural notation |
| Transpose, complex transpose | Existing conflation of transpose with Hermitian operation | State separate definitions and uses; ordinary transpose still exists over complex entries |
| Tensor reshape/moveaxis/reduce/stack/concat/split/index/mask | Existing API overview and partial snippets | Retain all with visible provenance and complete examples; Tensor Algebra & Einsum remains the existing owner for general contractions |
| Token projection and averaging | Existing unanswered task | Full numerical verification and explained independent variation; pad masking qualifies the mean |
| Coordinate-invariant tensor law | Not taught by old numerical-array definition | Explain computational terminology boundary; abstract multilinear/tensor transformation laws belong to Tensor Algebra & Einsum, not an unmotivated detour |

## Teaching sequence and representations

1. A displacement and a batch of orders: coordinates/named axes/units; immediately visible labelled matrix, not abstract notation first.
2. Addition/scaling, unit directions and length; show a head-to-tail coordinate figure. Dot product as weighted sum then geometry; derive projection coefficient from perpendicular residual.
3. Vector projection investigation: initial direction u=(2,1), initial v=(1,4), editable v within ±4; same-scale coordinate axes, v/p/residual arrows, signed dot and exact formula. This initial vector makes the perpendicular remainder visible; setting v=(3,2) revisits the preceding hand calculation. Prediction, negative/zero/perpendicular cases and reset. Direction presets include zero, which disables projection with explanation. Numeric coordinates remain authoritative when arrows coincide.
4. Matrices as linear maps: fixed preset maps, editable input coordinates, input square and transformed parallelogram, basis images, column combination and row calculation. Identity, shear, quarter rotation, reflection, collapse and zero show distinct consequences. No animation needed. Rank-loss and zero-origin relationship visible.
5. Products as many weighted sums: row×column explorer with order/resource fixtures, output-cell selection, term stepping and full product; held-fixed labels and exact partial sum. Inputs bounded and validated; malformed or incompatible matrices preserve active calculation. Units distinguish inner-match from semantic compatibility.
6. Composition, transpose and lost information: compact order contrast with explicit coordinates; two small systems explain impossible/nonunique/unique without teaching decomposition methods.
7. Tensors: two session slices, axes named session/time/channel. Reduction investigation highlights precisely the source cells for selected output, names surviving axes and shows sum/count. Static transpose-versus-reshape comparison preserves value identity and exposes changed grouping.
8. Broadcasting, reductions, index/mask/stack/concat/moveaxis and affine batch examples; shape alignment table, silent square-matrix broadcasting counterexample. More complete numeric programs than the original.
9. Independent practice, hints, explained solutions, diagnostic checks, readiness and actual next Matrix Decompositions link; annotated alternate resources.

Each visual is calculated from exact invented fixtures or elementary formulas, never a benchmark or measured phenomenon. Input/output plots use equal x/y scale and labelled coordinates; geometry only represents commensurate spatial components. The map explorer fits one shared domain to the current input, output, basis, unit square and column contributions; both plots always have identical ranges. The note explicitly explains range refitting. This makes the unit square visible without falsifying relative geometry. Projection keeps a fixed ±6 domain. Data feature axes need not be physical directions. At narrow widths diagrams fit with native text or stack panels; exact tables may scroll locally with keyboard. Long mathematical equalities are split into readable aligned lines. All controls use native HTML, visible focus, text status and deterministic reset; no reliance on color alone or hover. The browser is a bounded teaching model, not arbitrary Python execution.

## Applications and learning purpose

- Invented assembly orders: order×product counts multiplied by product×resource requirements. Shows why the inner names and units, not just equal integers, must match. Add one spare panel per order to distinguish linear from affine.
- Geometric direction extraction: split a displacement into along-direction and perpendicular parts, including a negative projection. This is an exact mathematical scenario, not a claim about a deployed sensor.
- Two-channel difference as common-offset rejection: [-1,1]·(signal+[c,c]) is unchanged, since [-1,1]·[c,c]=0. A simple, slightly less familiar null-space consequence; does not claim removal of arbitrary noise or a clinical measurement benefit.
- Batched affine scores and token projections retain the old practical ML connection. Scores are numbers, not automatically probabilities; averaging mixes exactly the selected axis and padding requires a mask.

## Research ledger

Sources reviewed 10 September 2026. No entire recording watched. Content and API examples are authored independently around local fixtures.

| Primary/creator resource | What was actually inspected / use / limitation |
| --- | --- |
| https://textbooks.math.gatech.edu/ila/dot-product.html | Interactive Linear Algebra textbook page: norm, dot, perpendicularity; supports real Euclidean assumptions |
| https://textbooks.math.gatech.edu/ila/matrix-multiplication.html | Textbook composition/row-column rule and caveats; supports order and algebra, not empirical performance |
| https://textbooks.math.gatech.edu/ila/dimension.html and https://textbooks.math.gatech.edu/ila/rank-thm.html | Substantive text reviewed: basis, dimension, column/null spaces and rank plus nullity equals the number of columns; exact small systems independently checked |
| https://www.3blue1brown.com/lessons/linear-transformations/ | Substantive creator's text adaptation with basis-image equations and examples; useful animated/video alternative for section 3. Full video not watched. Precise algebraic linearity definition in our lesson avoids relying on a slogan about lines. |
| https://www.youtube.com/watch?v=kYB8IZa5AuE | Direct Chapter 3 recording identity resolved and metadata inspected; fit verified from the substantive creator's text adaptation above. No whole-video viewing or transcript review claimed. |
| https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/pages/ax-b-and-the-four-subspaces/the-geometry-of-linear-equations/ | Session overview plus linked four-page summary, two-page problems and three-page solutions PDFs read as extracted text. Checked row/column geometry, dependent-column and shape practice. Lecture/transcript not watched/read. Intro undergraduate extension after matrix maps; older mathematical content remains useful. |
| https://numpy.org/doc/stable/reference/generated/numpy.matmul.html | Current stable docs: last-two-axis products, batch broadcast and 1-D promotion, no scalar @, no automatic conjugation |
| https://numpy.org/doc/stable/user/basics.broadcasting.html | Right-aligned equal-or-one rules, conceptual repetition and large-result memory caveat |
| https://numpy.org/doc/stable/reference/generated/numpy.reshape.html | Element-count/C-index-order interpretation and view/copy limits |
| https://numpy.org/doc/stable/reference/generated/numpy.moveaxis.html | Axis movement preserves remaining order; returned view |
| https://numpy.org/doc/stable/reference/generated/numpy.mean.html | Reduction axes/keepdims and floating-point accumulator precision |
| https://numpy.org/doc/stable/reference/generated/numpy.transpose.html, https://numpy.org/doc/stable/reference/generated/numpy.stack.html and https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html | Axis permutation, new-axis versus existing-axis joining and shape constraints |

Tested runtime: Python 3.12.14 / NumPy 2.3.5; current stable docs identify 2.5. Established APIs available in the tested version are used; the installed version is not claimed to be the latest. Further actual checks belong to [the verification record](VECTORS-MATRICES-TENSORS-VERIFICATION.md).

One scoped adjacent discovery was routed to [Tensor Algebra & Einsum Notation](topic-notes/tensor-algebra-einsum-notation.md): explicit output may retain a repeated diagonal label, and computational-array terminology deserves a later mathematical transformation-law bridge. Both entries remain open for the destination author; the destination lesson was not rewritten or broadly audited.

## Verification plan and status

Independent Python/NumPy oracles for every pure model, all small signed/zero geometric inputs, every product output/partial term, shape validation and source-cell reduction provenance. Execute every complete printed program and compare exact displayed outputs; independently verify numeric claims, hand results, shape mappings, composition counterexample and rank-loss systems. Browser at1440/390: all controls/presets/cells/steps/reset/errors, keyboard, formula rendering, all route anchors, source links, no blocking overflow/errors, opened screenshots and ordinary-reading visual pass. Parent handles full build/curriculum registration.

Implementation complete. Computational verification passed: ten displayed programs and independent NumPy/least-squares oracles. Browser interaction and opened visual review passed at 1440 and 390 pixels; a supplemental ordinary-reading pass verifies final equation layout, descriptive practice titles and fourteen source links. Formatting of owned lesson/component/model source preserves normalized AST including template raw values and JSX text. Parent owns the final integrated production build after this final source polish. User review has not been performed; no learning study or global curriculum completeness claim.
