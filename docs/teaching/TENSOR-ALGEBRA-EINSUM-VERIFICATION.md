# Tensor Algebra & Einsum Notation — verification

10 September 2026. Stable ID `tensor-algebra-einsum-notation`. [Design and research](TENSOR-ALGEBRA-EINSUM-DESIGN.md). Implementation and author verification are complete; root integration and user acceptance remain separate. Lesson/component source is frozen after the independent-review correction and formatting at **08:57:06 UTC**; styles last changed at08:48:19.

## What changed and why

Retained the original matrix-product, trace, column-sum and outer-product examples, attention contractions, shape/ellipsis/implicit-order cautions, optimization concern and batch-covariance task. Rebuilt the explanation around one selected output and its actual source entries. Explicit output now correctly distinguishes diagonal selection, reduction of a once-occurring index and a retained shared index. Attention has complete inputs, scaled dot products, stable masked softmax and separate value-feature axes; covariance includes centering, denominator and estimator assumptions.

Four investigations serve distinct learning needs: an editable bounded contraction interpreter, per-key attention products/weights/contributions, two contraction orders with first-temporary shapes and exact multiplication counts, and a basis/coordinate/functional/metric investigation. Two inline figures explain diagonal selection before reduction and same-batch versus all-pairs comparisons. The deeper mathematical branch derives fixed-space transformation laws without claiming to teach all tensor calculus. Kinetic energy provides a separate physical-units application with stated classical/Cartesian assumptions.

Ten complete NumPy programs, six independent tasks with separate hints/explained solutions, checkpoints and optional deeper branches retain useful detail. Topic title, stable identity, module position and next Randomized Linear Algebra bridge are preserved. Both Vectors destination proposals are implemented or adapted; the reasoned [Differential Geometry note](topic-notes/differential-geometry-riemannian-manifolds.md) persists the differential/covector versus metric-gradient extension.

## Actual numerical evidence

`node scripts/verify-einsum.mjs` calls `scripts/verify-einsum-native.py`. Pass **08:39:54 UTC** in `scratch/einsum-verification/results.json`. Python **3.12.14**, NumPy **2.3.5**. No package installation needed. No numerical model or displayed program changed after this pass; later model formatting had normalized AST equality.

| Check | Result and independence |
| --- | --- |
| Complete learner programs | 10 native programs, every displayed stdout block checked exactly |
| Contraction interpreter | 544 cases compared with native NumPy einsum for output shapes and every cell; each depicted source address/value/product independently checked against native indexing |
| Attention model | 56 batch/query/mask/scaling cases against independent matrix products and direct exponentials on these safe bounded scores |
| Coordinate model | 324 basis/vector states against NumPy solve/map/metric operations and invariant quantities |
| Contraction costs | 1,296 dimension cases checked by enumerating conventional dense multiplication operations and independently allocating first-temporary shapes |
| Learner attention function | 32 unseen B/T/S/D/F combinations with masks, compared with per-query/key/component loops; six invalid-input groups |
| Additional numerical transfer | Six covariance fixtures against np.cov and positivity/symmetry; 50 unseen type(2,0) outer-product transformations against separately transformed vectors |
| Independent practice | Five changed numerical groups covering weighting, covariance/Gram, masked retrieval and functional pairing |
| Runtime boundaries | Seven groups including empty reductions, zero vectors, singleton broadcasting and rejected diagonal/output labels |
| Model invalid contracts | 20 grouped expression/shape/mask/basis/dimension guards |

The teaching inspector deliberately accepts only positive axis sizes1–4, at most four labels and one/two supplied operands in explicit mode. The native runtime checks cover empty arrays separately; they do not falsely imply this browser inspector implements the entire NumPy API. Counts compare conventional dense scalar multiplications, not NumPy's FLOP convention or hardware performance. The coordinate-gradient example in the saved destination note was separately executed with NumPy solve, returning new components [3,−1] and reconstructed old vector [2,−1].

## Actual browser and visual evidence

`node scripts/review-einsum.cjs` passed **08:49:40 UTC** in Edge headless at **1440×1000 and 390×1000**. Evidence: `scratch/einsum-browser/results.json`.

At each width:

- 22 selected output entries across all eight contraction presets, plus a custom retained-k expression; invalid expressions preserve the last valid result.
- 56 attention mask/query/batch/scaling states, including single-key weights; an all-masked row reports an error and removes the obsolete context.
- Five valid cost comparisons, including reversed preference, equal scalar case and maximum bounded dimensions; five invalid input groups preserve valid results.
- 16 basis/vector states, correct measurement/metric readouts, finite SVG geometry and deterministic resets.
- Ten displayed complete programs and expected outputs, ten unique working section anchors, six independent practice sections, nine annotated reference links and the adjacent module link.
- Keyboard select navigation, Tab focus, Enter/Space hint disclosure, native controls at least44px high; no page errors, KaTeX errors, page overflow or displayed-equation overflow.

The initial harness used an exact label-text query for a nested select whose text included its option labels. Scoped prefix label selectors corrected that test-only lookup; the controls themselves already worked. A real narrow-screen issue was found: a long explanatory phrase inside the optional tensor transformation formula caused horizontal math scrolling. That phrase was removed from the formula while its type is explained in adjacent prose. Basis arrows gained visible b1/b2 labels, and mobile inputs align by their lower edges.

After the final intentional word/number spacing polish and masked-equation correction, `node scripts/review-einsum-reading.cjs` passed **08:58:24 UTC** at **1440,390 and320px** (`scratch/einsum-browser/reading-results.json`): six ordinary-reading sections, four labelled basis presets at each width, all displayed formulas with disclosures open, nine references and no overflow/errors. It also verifies the piecewise equation/readout and two masked interaction states per width. The script's original filename used the first digit of a section number, causing section10 to overwrite the section1 screenshot; the filename now uses the complete number and the reading capture was rerun. This was an evidence-file issue, not a lesson navigation defect.

Independent cross-review caught a real mismatch between the displayed masked-attention formula and its correctly masked runtime: the denominator restricted allowed keys but a literal numerator still gave a positive weight for a blocked key. The formula now defines Z over allowed keys and W piecewise, with blocked weight explicitly zero; the lab sentence similarly qualifies its exponential expression. Native models/programs were already correct and were not altered. The independent counterexample and two further nonorthogonal/nonidentity-metric checks are recorded in `scratch/einsum-cross-review/results.json` and `scratch/einsum-independent-review.py`. Functional pairing, map action, metric congruence and type(2,0) outer-product invariants passed. The corrected formula and masked lab were rechecked at all three widths; final formula screenshots were opened at390 and320, with no clipping or ambiguous blocked-key case. Array-index wording was also narrowed so it does not imply a zero-based convention for the later geometric v₁/v₂ notation.

Actual opened screenshots include desktop contraction and cost investigation; mobile contraction, attention, basis and inline diagonal; ordinary-reading axes, notation, covariance, basis and independent practice; final mobile sources; final shear390 and rotation320 basis figures; corrected masked-equation and attention captures. Fixed Learn header visibility is suppressed only during isolated component captures and restored afterward. Ordinary page screenshots retain normal navigation. Screenshot paths follow `scratch/einsum-browser/`; final reading files use `final-reading-<section>-<width>.png`, and final basis/source/masked-equation captures are similarly named.

## Limits and reproducibility

`scratch/einsum-verification/formatting-results.json` records conventional formatting with normalized Babel AST equality, preserving raw template and JSX text values. Numerical examples and figures are declared invented fixtures, not empirical datasets, attention-quality evaluations or performance measurements. The source ledger distinguishes substantive NumPy/paper/textbook reading from MIT transcript portions and recording identity; no complete video viewing is claimed. No observed beginner study, external deployment or user approval is implied by these author checks.

Root registered the individual blueprint and owns generated navigation, curriculum conservation, production build, loading/recovery checks and the full-goal ledger. No shared publication mapping or global curriculum order was edited by this topic implementation.
