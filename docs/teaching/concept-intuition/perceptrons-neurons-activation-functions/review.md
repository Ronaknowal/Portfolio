# Perceptrons, neurons and activations: concept-level review

26 September 2026. Full implementation revision. The entire production lesson, its existing figures and implementation-depth record were read. This is an author assessment; independent and rendered review remain pending. Keep the title and stable ID: the changes explain the existing scope.

## Concept map and actual disposition

| Concept / location | Assessment and change | Representation / transfer |
| --- | --- | --- |
| Opening; neuron as a function | Existing handwriting question is useful. Removed the preceding catalogue of control instructions that used terms before introducing them. Added a short causal reading guide after the first-pass route. | Existing observed digit fixture later closes the opening question. |
| §1 weighted contributions, bias, activation | Existing numeric 4-score trace and weighted evidence figure already distinguish products, sum and activation; retained. | Existing geometry lab varies input/weight/bias. |
| §1 score versus distance; positive scaling and zero vector | Complete numbers and boundary lab already expose this distinction; retained. | Changed-boundary practice and null case retained. |
| §2 signed margin and perceptron update | Formula stated direction but did not walk a complete mistaken row before the epoch trace. Added the original (1,2), w=(−1,0), b=0, step=.5 calculation. | MistakeCorrectionFigure aligns old parameters, added increments and new parameters; score −1→2, increase 3. |
| §2 augmented bias input | Added why constant input 1 gives an always-available offset and one unified update, rather than only telling readers to append it in code. | Same three aligned parameter columns; existing complete program retained. |
| §2 AND versus XOR and tie policy | Full truth table, executed update counts and inequality contradiction already explicit; retained. | Existing XOR construction and changed-corner practice. |
| §2 convergence argument | Added the proof's goal and two growth-rate meanings before/after the retained bound. | Projection grows at least linearly; norm at most as square root; no artificial geometric plot needed. |
| §3 hidden feature / ramp cancellation | Existing table, worked figure and repair lab follow every row; retained. | All-corner repair exercise retained. |
| §3 affine collapse versus nonlinear representation | Existing exact composition identity and XOR contrast suffice; retained. | Explicit architecture boundary retained. |
| §3 batch rows, neurons, transpose and bias broadcasting | Added a complete two-example/two-neuron matrix example, not another shape-only box. | BatchNeuronCorrespondence follows both columns, output/ReLU values and one bias per neuron. Practice 4 changes the shape. |
| §4 activation slopes and chain sensitivities | Existing values, derivatives, counterexample weight factors and live curve investigation already connect local and whole-path effects; retained. | Existing negative operating-point and ten-layer sensitivity practice. |
| §4 stable scratch/library code | Complete mechanism program, shared conventions and negative-branch change exercise retained byte-for-byte. | Existing API mapping and exact-value output. |
| §4 softmax versus independent outputs | Added same-input competition contrast and explanation of common-factor cancellation. | SoftmaxCompetitionFigure changes only score 3; shared probability strips show other shares falling. |
| §5 loss versus correct counts | Added why probabilities .51 and .99 count identically but give different CE; prior experiment unchanged. | Measured paired-error and loss displays remain the evidence for actual fitted models. |
| §5 training loop and data roles | Existing five-operation loop, real fixture, shapes and exact programs already linked; retained. | No new fit, changed benchmark, fabricated accuracy or hidden dataset. |
| §6 diagnosis, symmetry and checkpoint compatibility | Existing diagnosis table and identical-unit explanation suffice; retained. | Useful distinctions remain before the optional branch. |
| §7 negative branches, GELU/SiLU and deterministic weighting | Added hard-cutoff→continuous-fraction motivation and multiplier/product representation. | SmoothGateDecomposition uses a common 0–1 multiplier scale at z=−1,+1; labels distinguish multiplier from signed output. |
| §7 negative slope; Mish | Added the competing effects behind SiLU's negative slope and a plain description of Mish's composed fraction. | Reuse the same multiplier interpretation and existing activation lab rather than duplicating all curves. |
| §7 SwiGLU versus scalar activation | Existing separate learned projections and coordinatewise table already make the distinction. New caption explicitly distinguishes a positive scalar multiplier from a signed learned SiLU gate. | Original three-projection program and equal-budget exercise retained. |
| §7 approximation and memory accounting | Existing ramp decomposition, triangle plots, units, raw-tensor count and parameter counts already expose their mechanisms; retained. | Original pulse-construction and budget exercises retained. |
| §8–9 practice, sources and next lesson | Read complete hints/solutions and references; preserved. | Original changed-input tasks, eight end exercises and executable variation remain. |

## Research consulted

- Stanford CS231n, Neural Networks Part 1, layer-wise organization and feedforward computation: https://cs231n.github.io/neural-networks-1/ . Read the full page, concentrating on matrix/neuron correspondence. The diagram-to-matrix teaching decision is useful; historical blanket recommendations about sigmoid, capacity and depth were not adopted.
- 3Blue1Brown, creator-hosted neural-network text companion: https://www.3blue1brown.com/lessons/neural-networks/ . Reviewed the weighted sum, bias, layer matrix and function passages. No full-video watch claim; retained the lesson's precise ReLU and probability boundaries.
- Cornell CS4780 perceptron notes: https://www.cs.cornell.edu/courses/cs4780/2022sp/notes/LectureNotes06.html . Read update geometry, bias augmentation and proof. Added an original small calculation and the two-growth-rate explanation; did not claim the learner algorithm knows the separator.
- Original GELU paper §2, https://arxiv.org/html/1606.08415v5 . Reviewed the stochastic motivation versus deterministic expected output and CDF-based multiplier. The new small values are locally checked, not a copied benchmark or a claim that observed scores must be Gaussian.

## Visual and implementation contract

Four static native HTML/CSS representations sit beside the specific new bridges. Aligned parameter columns, matrix entries, common-total strips and multiplier bars each encode a different mechanism. Text equivalents include exact arithmetic. The two-column layouts stack at reading-container width 470px; labels remain HTML, with no shrunken SVG text. No control-shaped decoration is used. Existing live labs keep their behavior and useful context.

Production JSX is the full-mode written-content source; this record supplies the visual specifications. No manuscript duplicate is needed. The data builder only owns the old measured-data module and downloads and does not overwrite the page. New small calculations live in perceptron-intuition.js; new figures and CSS are topic owned.

## Author checks and remaining work

Run scripts/verify-perceptron-intuition.mjs. Its source-bound receipt checks independently calculated update direction/size, matrix products, the probability competition/cancellation, parsed JSX and exact identity of prior examples/models/downloads against the pre-revision baseline. A Python SciPy calculation separately checked the normal-CDF constants. These checks supplement the full reading assessment, not replace it.

Independent review and actual desktop/narrow browser inspection remain pending. Earlier native fit results are reused only for unchanged source/data; no new training run or learner trial is claimed.
