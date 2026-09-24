# Authoring notes: Tensor Algebra & Einsum Notation

Canonical topic ID: `tensor-algebra-einsum-notation`

## 2026-09-10 — Separate repeated labels, diagonal selection and explicit output

- Status: implemented
- Origin: `vectors-matrices-tensor-operations`; [design record](../VECTORS-MATRICES-TENSORS-DESIGN.md).
- Destination and ownership rationale: this existing lesson owns contraction notation; the introductory operations lesson teaches axis identity and reductions first. General einsum rules belong here rather than adding an early notation detour.
- Idea and learning benefit: a learner should predict both values and surviving axes without the incorrect shortcut that every repeated input label is always reduced.
- Existing coverage: checked `src/learn/data/topics/tensor-algebra-einsum-notation.jsx`, section 1 states “repeated labels are summed over”; section 3 correctly shows `ii->` but has no retained-diagonal counterexample. The output-label qualification needs to be made explicit rather than leaving two rules apparently in conflict.
- Proposed treatment: teach explicit output first, with `ii->i` (diagonal retained), `ii->` (trace), `ij->j` (reduction even though i occurs once), then compare implicit mode. A label repeated within one operand selects its diagonal; output labels decide what survives in explicit mode. Explain ellipsis/broadcasting separately. No title change required by this discovery.
- Explanation/example: for A=[[1,2],[3,4]], `ii->i` yields [1,4], while implicit `ii` and explicit `ii->` yield 5. Mark the selected diagonal cells, then show the optional reduction arrow. Predict `i->` on [1,2,3] before calculating 6.
- Prerequisites and boundaries: named axes, indexing, diagonal, sum and shape; arbitrary index notation and contraction optimization come later. Recheck NumPy semantics and distinguish mathematical Einstein conventions from NumPy's extended API.
- Evidence: [NumPy einsum reference](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html), substantive Notes reviewed 10 September 2026; implicit mode, repeated labels in one operand, explicit output and ellipsis sections support this distinction. Version-specific examples should be executed again by the receiving author.
- Resolution: incorporated into core section3 and the explicit-expression inspector. The same matrix figure separates diagonal selection from optional reduction; examples compare implicit `ji` with explicit `ji->ji` and preserve a repeated k in `ik,kj->ikj`. No rename required.
- Implementation/verification links: `src/learn/data/topics/tensor-algebra-einsum-notation.jsx`, [design](../TENSOR-ALGEBRA-EINSUM-DESIGN.md), [verification](../TENSOR-ALGEBRA-EINSUM-VERIFICATION.md). NumPy2.3.5 tests independently compare 544 model contractions and depicted source addresses, including singleton/diagonal distinctions.

## 2026-09-10 — Computational arrays and mathematical tensors

- Status: adapted
- Origin: `vectors-matrices-tensor-operations`; its first section distinguishes a numerical tensor's array-axis count from vector-space dimension and matrix rank.
- Destination and ownership rationale: this topic can connect index notation to multilinear objects and change of basis after learners can track axes. The introductory operations lesson would become less approachable if it immediately developed covariant/contravariant transformation laws.
- Idea and learning benefit: recognize that a library's multidimensional array does not automatically specify how a mathematical or physical quantity transforms when coordinates change. An array's shape alone does not establish its tensor meaning.
- Existing coverage: destination currently teaches computational contractions; no change-of-basis transformation law is taught. The origin now states the terminology boundary rather than claiming all arrays are coordinate-independent tensors.
- Proposed treatment: investigate a progressively disclosed deeper section contrasting one geometric vector represented in two bases, a linear map's matrix in changed coordinates, and an arbitrary image array with named axes. Only introduce upper/lower indices if needed and with explicit conventions; bridge any fuller differential-geometry owner if one fits better after checking the current catalogue. Do not infer from this note that advanced tensor calculus must be forced into the core.
- Explanation/example: the same displacement should remain geometrically unchanged after its coordinate pair and basis change together. A diagram could separate the object from its numerical representation and contrast this with simply transposing an array.
- Prerequisites and boundaries: basis, coordinate transformation and linear maps first. Existing matrix-decomposition coverage may supply bases, but transformation-law conventions require dedicated authoritative research; that research has not been completed in this scoped introductory rewrite.
- Evidence: NumPy documentation uses tensor/array language for indexed computation; this entry is a curriculum proposal, not evidence of a physical application. Research a primary mathematical treatment and independently verify any proposed transformation example before implementing.
- Resolution: adapted into section8: vector and functional pairing, map similarity, metric congruence, a linked basis investigation and independently checked complete solve example. A deeper branch distinguishes tensor types, multilinearity and upper/lower transformation factors. Broader tensor-field differentiation is deliberately left to the existing geometry owner, with a [new destination note](differential-geometry-riemannian-manifolds.md) about differential versus gradient conventions. No full tensor-calculus scope is implied.
- Implementation/verification links: [design](../TENSOR-ALGEBRA-EINSUM-DESIGN.md), [verification](../TENSOR-ALGEBRA-EINSUM-VERIFICATION.md); 324 fixed-basis model cases checked against NumPy solves and invariants, plus 50 unseen outer-product basis transformations. MIT transcript portions and exact review bounds are recorded in the source ledger.
