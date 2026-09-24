# NumPy foundations: design and verification record

Current visual follow-up, 10 September 2026: [scientific representation review](SCIENTIFIC-VISUAL-REVIEW.md) records the inline reshape/transpose comparison and fresh native/browser evidence. The existing labs and executable examples were retained; the dated reimplementation record below describes the prior increment.

Date: 9 September 2026. Scope: topic 2 in the user's explicitly authorised first-five reimplementation. Stable topic ID: `numpy-arrays-broadcasting-vectorization`. This record implements the [teaching standard](../../LESSON-TEACHING-STANDARD.md); it does not replace the [current handoff](../../LESSON-AUTHORING-HANDOFF.md).

## Learning contract and original weaknesses

The learner has just studied Python names, values, lists, indexing, comparisons, imports and functions. Matrix algebra, machine learning and a prior scientific-computing course are not prerequisites for the core route. The local bridges explain NumPy's comma-separated axis indexing, shape tuples, None as an inserted axis, boolean masks, dtype, views and numeric assertions before the complete report relies on them.

The earlier lesson had substantial technical coverage and independently runnable examples, but its first pass was closer to an API survey. A single generic broadcasting control did not expose source-to-output cells, shape versus scientific meaning, shared storage, or reduction contributors. Its mixed temperature/humidity fixture needed a caveat when averaging rows. The revision preserves advanced API depth in optional branches and uses one core dataset in which both columns are temperatures from sensors measuring the same room.

The practical anchor is calibration of three times × two sensors: `[[18, 20], [24, 26], [30, 32]]` in °C. Rows are time, columns are sensor. Known offsets `[2, 4]` give equal corrected pairs `[[16, 16], [22, 22], [28, 28]]`. These values and assumptions are invented for transparent calculation; they are not empirical sensor evidence. A later complete workflow introduces one missing reading and a fourth row, preserving original time indices through filtering.

The beginner finish line is a checked report that preserves raw measurements, selects observations deliberately, applies per-sensor calibration, reduces the correct axis and keeps original labels. Optional branches retain sorting, advanced indexing, linear algebra, train/test preprocessing, generators, storage and performance diagnosis. The mathematics behind decompositions, a full estimator API, Fourier analysis, masked-array internals, arbitrary stride tricks and hardware benchmarking remain follow-on subjects, not claims of mastery here.

## Outcome and assessment map

| Outcome / hurdle | Mechanism and representation | Worked evidence | Practice / success | Depth |
| --- | --- | --- | --- | --- |
| Read axes, dtype and coordinates | A labelled time × sensor grid; one coordinate per axis; scalar versus one/two-axis shape | `create` reports shape, size, itemsize and Python-list versus array multiplication | Predict fourth-row shape (4, 2), explain unchanged ndim | Core |
| Select observations without losing meaning | Selected output cells link to their original coordinates; integer index drops an axis, slice retains one | `select`: column, one-column table, row mask, cell mask, combined conditions | Predict mask [False, True, False] and shape (1, 2); compare equal values with different structure | Core |
| Predict mutation through views | Array objects point to shared or independent buffers; relative byte offsets expose strides | `memory`: view/copy/advanced read, writes, shares_memory and direct indexed assignment | Choose copy before mutation; diagnose chained advanced indexing and in-place view update | Core, non-contiguous/object nuance deeper |
| Predict broadcasting and map its operands | Align trailing dimensions; length-one input coordinates stay at zero; output cell selects exact inputs | `broadcast`: per-sensor, per-time singleton, invalid trailing axis, valid outer difference | Trace 32−4=28; repair (3,) offsets; transfer to batch×height×width×channel | Core |
| Choose a reduction from the scientific question | Highlight all contributing cells for one surviving mean; keepdims explicitly retains length-one axis | `reduce`: per-sensor/per-time/global means, centering and argmax | Explain (3, 1) per-time means and rows [-1, 1]; distinguish same units from mixed units | Core, variance/empty/multi-axis deeper |
| Separate reshape from transpose | Selection explorer maps same-shaped outputs back to different source cells | `shapes`: transpose, reshape, vector T, newaxis, concatenate, stack, linspace | Explain why regrouping is not a sensor/time axis swap | Core |
| Separate elementwise contributions from contraction | Two-sensor calculation table connects multiplication with the final weighted sum | `weighted`: `X*w` versus `X@w`, shape and numerical checks | Trace 4.5+15=19.5 and explain which dimension disappears | Core; general linear algebra deeper |
| Handle dtype and numerical limits intentionally | Concrete overflow and casting failures; explicit finite masks and tolerances | `numerical`, retained `numpyMissing` | Explain convert-before-overflow; choose missing policy; diagnose eager division within where | Core, ufunc mask details deeper |
| Combine transformations into a reliable report | Five-stage workflow, labelled intermediates, accepted original time indices | `workflow` includes data, validation, calculations, output, assertions | Changed row [26,28] yields means22 and warm times[2,4] | Core synthesis |
| Transfer to a new task without copying a recipe | Energy meters, multiplication gains, sum rather than mean, square-shape axis trap | `transfer` is hidden behind optional hint and complete explained solution | Totals[80,65,70], days[0,3], raw unchanged, meter deviations average to0 | Independent core assessment |

## Visual contracts

All figures are original code-native representations. Appearance marks selection with a border and explicit coordinates/source text, not color alone. Static labelled grids explain entity organisation. Interactive figures use ordinary buttons/selects, visible focus, aria-pressed and nearby live causal feedback. No randomness or motion is required.

### Coordinate and selection explorer

- Question: which original readings survive, and what shape and meaning do they retain?
- Starting state: original (3,2) temperature grid; `X[:,0]` selected as the first investigation, output concealed.
- Prediction: name values, surviving axes and their lengths before revealing.
- Controls: eight finite presets; reveal/hide; select an output cell; reset.
- Consequence: output geometry reflects its shape and selection highlights its original input cell. Feedback explains axis removal/retention and source coordinates.
- Transfer: row mask and cell mask can contain equal numbers but encode different structure; transpose and reshape can share shape but differ in meaning.
- Boundary: only the displayed fixed numeric selections. Memory sharing is taught in its own activity.
- Independent check: every shape, selected value and source index compared with real NumPy. Values are unique, so oracle source provenance can be recovered independently from NumPy's result.

### Shared-storage explorer

- Question: does writing through picked change X, and which physical slot explains the effect?
- Starting state: a fresh contiguous float64 (3,2) array and selected first column. Relative addresses show 8-byte elements and a 16-byte view stride.
- Prediction: determine whether selected slot 1 will also change X[1,0].
- Controls: basic view, explicit numeric copy, advanced-index copy; choose slot0/1/2; write99; undo; reset.
- Consequence: shared buffer A changes in both arrays for a view; independent buffer B changes alone for a copy. Address labels and causal text identify the same or different storage explicitly.
- Transfer: place copy before the first mutation when raw measurements must remain intact.
- Boundary: buffers/offsets are schematic, not actual addresses. Object dtypes, negative strides and general reshape cases are excluded and explained separately.
- Independent check: all three strategies with no mutation or each of three writes; values, shares_memory and real strides compared with NumPy.

### Operand-to-output broadcasting explorer

- Question: which offset reaches a chosen result cell, and when does a shape reject or silently answer another question?
- Starting state: (3,2) readings minus (2,) sensor offsets; output concealed; aligned dimensions visible.
- Prediction: compatibility, output shape and one result's inputs.
- Controls: sensor/time/scalar/invalid/outer presets; reveal/hide; choose any output; reset.
- Consequence: selected output highlights both source operands and shows the exact subtraction equation; invalid dimensions produce a named mismatch explanation instead of fake output.
- Transfer: contrast (3,) versus (3,1); inspect the square outer-difference matrix, diagonal and lower triangle.
- Boundary: fixed-data subtraction, displayed scalars and at most two axes. No Python execution or allocation/performance simulation. The generic pure model is also tested on non-displayed multidimensional and empty shapes.
- Independent check: all five presets plus144 generated shape pairs, including scalars, zeros, singleton and incompatible axes; NumPy subtraction and broadcast_to independently supply outputs and source-index maps.

### Reduction-contributor explorer

- Question: which inputs contribute to each mean, and which coordinate survives?
- Starting state: X, axis0, keepdimsFalse; result concealed.
- Prediction: number and labels of output means.
- Controls: axis0/1, keepdimsFalse/True, reveal/hide, output selection, reset.
- Consequence: contributing cells highlight together and their sum/divisor is shown beside the surviving mean; geometry changes with keepdims.
- Transfer: subtract each time's mean using (3,1), explaining why (3,) fails against (3,2).
- Boundary: fixed finite unweighted means. Missing values, empty groups and numerical precision limits are covered explicitly outside the model.
- Independent check: both axes with both keepdims settings against real NumPy; expected contributors also visible in exact equations.

## Claim and source ledger

Retrieved 9 September 2026 with the web tool. Moving stable documentation identified itself as NumPy2.5; executable evidence below is NumPy2.3.5. The lesson does not claim those are the same release or that the runtime is latest. Core APIs used are common to those checked contracts. Source text was used for technical checking; examples and representations were designed for this curriculum rather than copied from documentation.

| Claim / convention checked | Primary source and locator | Evidence / qualification |
| --- | --- | --- |
| ndarray shape, ndim, size, dtype, coordinate indexing | [Beginner guide](https://numpy.org/doc/stable/user/absolute_beginners.html), array attributes/indexing | Shape is coordinate organisation, not axis meaning or units; custom values verified in create/select |
| Integer indexing drops axes, slices retain; advanced selection copies; direct assignment differs | [Indexing guide](https://numpy.org/doc/stable/user/basics.indexing.html), basic/advanced indexing and assignment | Examples restrict to ordinary numeric arrays; scalar selection called scalar |
| Views share buffers; reshape may copy; strides define byte steps | [Copies and views](https://numpy.org/doc/stable/user/basics.copies.html), view/copy/reshape sections | Real shares_memory and stride evidence for all pictured strategies |
| Object-array copy is shallow | [numpy.copy](https://numpy.org/doc/stable/reference/generated/numpy.copy.html), Notes | Independent numeric storage is not a recursive copy of Python objects |
| Right alignment, singleton reuse, invalid dimensions and input-memory economy | [Broadcasting guide](https://numpy.org/doc/stable/user/basics.broadcasting.html), general rules/examples | Computed outputs/intermediates can still be large; no speed claim |
| Axis and keepdims, accumulation precision | [numpy.mean](https://numpy.org/doc/stable/reference/generated/numpy.mean.html), parameters/Notes | Fixed float64 examples; same units required for scientific interpretation |
| N−ddof and variance/std distinction | [numpy.std](https://numpy.org/doc/stable/reference/generated/numpy.std.html), ddof/Notes | Usual sampling assumptions qualified; no claim sample standard deviation is unbiased |
| Reshape's logical ordering versus transpose | [numpy.reshape](https://numpy.org/doc/stable/reference/generated/numpy.reshape.html), order/Notes | Default C order specified; no universal view guarantee |
| @ contraction versus elementwise multiplication | [numpy.matmul](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html), Notes | Core restricts to (3,2)@(2,); deeper batch contract explained separately |
| Overflow and dtype limits | [Data types](https://numpy.org/doc/stable/user/basics.types.html), overflow | Int8 wrap and pre-promotion verified in real runtime |
| allclose may broadcast and tolerance is scale dependent | [numpy.allclose](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html), Notes | Core separates shape and numerical checks |
| where on divide and uninitialised excluded output | [numpy.divide](https://numpy.org/doc/stable/reference/generated/numpy.divide.html), where/out | Retained complete example independently executed; Python eager argument evaluation explained |
| np.vectorize is convenience wrapping Python calls | [numpy.vectorize](https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html), Notes | No unmeasured acceleration claim |
| Repeated advanced indices require deliberate accumulation | [ufunc.at](https://numpy.org/doc/stable/reference/generated/numpy.ufunc.at.html), main contract | Initial add.at URL failed; corrected official ufunc.at page read successfully |
| argsort returns indices; tie stability is a choice | [numpy.argsort](https://numpy.org/doc/stable/reference/generated/numpy.argsort.html), return/parameters | Earlier complete sort output rerun; preserves aligned-record reasoning |
| Solving and SVD reconstruction contracts | [solve](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html), Notes; [SVD](https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html), return and decomposition | Retained example rerun; nonunique singular vectors and conditioning cautions retained |
| Random stream compatibility requires more than seed | [Random compatibility](https://numpy.org/doc/stable/reference/random/compatibility.html) | Repeatability asserted within tested setup, not across all releases/call patterns |
| NPY shape/dtype storage, object pickle issue | [numpy.save](https://numpy.org/doc/stable/reference/generated/numpy.save.html), Notes/allow_pickle | Retained numeric in-memory round trip rerun; full I/O follows in next actual topic |

## Implementation and evidence

- Lesson: [topic source](../../src/learn/data/topics/numpy-arrays-broadcasting-vectorization.jsx). Stable ID and hasIntegratedGuide preserved.
- Interaction: [labs](../../src/learn/components/lesson-labs/numpy-foundations-labs.jsx), [scoped styles](../../src/learn/components/lesson-labs/numpy-foundations-labs.css), [pure models](../../src/learn/data/numpy-foundations-model.js).
- [New complete examples](../../src/learn/data/numpy-foundations-examples.js); [retained optional depth](../../src/learn/components/lesson-labs/numpy-foundations-deeper.jsx) uses existing shared examples without modifying them.
- Runtime verifier: [Node comparison](../../scripts/numpy-foundations-verify.mjs), [independent Python/NumPy oracle](../../scripts/numpy-foundations-oracle.py).
- Fresh runtime result: 8 coordinate selections,12 memory states,149 broadcast cases,4 reductions; all shape/value/source-index checks passed. All10 new and5 retained displayed code/output pairs passed with no stderr. Python3.12.14, NumPy2.3.5. Machine-readable result: `scratch/numpy-foundations/runtime-results.json`.
- Browser verifier: [desktop/mobile checks](../../scripts/numpy-foundations-browser.cjs). At1440 and390 CSS pixels: all4 labs render, selection output links to its source, view/copy writes and undo agree, invalid broadcasts recover/reset, reduction axis/keepdims results agree, keyboard Enter activates a chosen output and visible focus is solid, all8 lesson anchors resolve, page overflow0, no page errors. Result: `scratch/numpy-foundations/browser-results.json`.
- Browser checks caught and corrected a mobile-only bug: rotating a full-width arrow expanded its hit area over output cells. The arrow is now bounded to36px and does not receive pointer events. This was a real interaction fix, not a forced test click.
- Inspected mobile broadcasting/storage and desktop selection/reduction screenshots; cell labels, source coordinates and equations remain readable. Tall NumPy element captures can include fixed global navigation at the capture's current scroll position; normal viewport/interaction checks passed. The final batch's scoped data-lab and OOP captures exclude fixed navigation during capture only; this does not change runtime behavior or hide navigation during normal interactions.
- The continuation link was verified against the live catalogue and points to `scientific-file-formats-schemas-reliable-data-i-o`. No publication-based skipping was introduced.
- `scripts/review-programming-batch-two.cjs` now retains decorator/testing checks and delegates current NumPy coverage to `numpy-foundations-browser.cjs`; the combined public command passed at 1440 and 390 pixels. The old native batch-two script retains earlier NumPy fixtures as regression evidence, not complete coverage of the rewritten lesson.
- [The first-five integration record](../../FIRST-FIVE-REIMPLEMENTATION.md) owns final application build, cross-page browser evidence, curriculum conservation and shared handoff status.

## Review status and limits

| Review | State |
| --- | --- |
| Implementation | Complete within the authorised topic scope |
| Computational verification | Passed, including independent real NumPy model comparisons and all displayed complete examples |
| Browser and visual review | Passed the stated desktop/mobile/keyboard checks; final integrated build and cross-page evidence are recorded in the linked first-five integration record |
| Coverage/pedagogy | Author heuristic review against Linux's mechanism-first contract; four distinct conceptual hurdles mapped to representations, predictions and transfer |
| Actual beginner study | Not conducted; completion clicks and automated tests do not establish mastery |
| User review | Pending review of this first-five increment |
| Next action | Review the delivered first-five increment; user and learner feedback can guide further refinement without treating publication as acceptance |

No arbitrary-code runtime, physical sensor validation, general NumPy emulator, performance benchmark or exhaustive numerical-analysis course is claimed. Several optional APIs have further edge cases; future dedicated lessons should retain source/version checks and deepen them when they become scoped outcomes.
