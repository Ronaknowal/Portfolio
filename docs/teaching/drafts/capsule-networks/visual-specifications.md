# Capsule Networks: visual and investigation specifications

## Live exploration contract — 21 September 2026

Open each investigation with its current inputs, intermediate mechanism and complete current output visible. Apply valid edits to meaningful entities immediately and update diagrams, tables, units and causal explanation together. No prediction entry, predicted-answer choices, commitment, prediction grading or answer-unlock feature is part of this packet, even optionally. Model predictions and mathematical masks/gates remain subject matter.

Use the topic-specific controls and checked fixtures below. Pair sliders or direct manipulation with labeled keyboard/numeric controls; keep presets as starting points, not the only editable values. A pinned baseline preserves its inputs, seed, units and outputs while the current case changes. Explain both a meaningful contrast and an unchanged/null result, then connect the observed effect to a practical design decision. Reset restores the stated fixture and current result. Invalid text has a local explanation and a clearly identified last valid result; never silently clamp or pair new inputs with old output.

Step/Back and bounded Run controls advance a real computation or reveal its chronological stages, not permission to view an answer. Show the current state and its result throughout. Keep exact small calculations live. For costly frozen inference, debounce or run bounded work with pending/current state labels and stale-result cancellation; inspect saved measurements without implying fresh training. Respect reduced motion, keep focus stable and avoid announcing every animation frame. Independent written practice and its hints/solutions stay separate.

Phase two must test default results without any action, meaningful edits, quick consecutive edits, valid extremes, null/invalid cases, reset, linked-view agreement, keyboard operation and readable phone layouts. The mathematical/reference checks already specified below remain; these live browser checks have not been performed in this content-only revision.

### Topic-specific live route
**Manipulate votes and vector geometry.** Edit capsule votes, routing iterations, vector magnitude/direction and supported retained image/latent coordinates.
**See the consequence.** Show coupling rows, vote contributions, squash length/direction, current parent vectors and saved/frozen-model outputs. Step routing to inspect its computation, with all current outputs visible.
**Decision connection.** Distinguish agreement from activation magnitude, pose changes from class evidence and a model intervention from a new empirical result.


Stable ID capsule-networks. Content-first revision 1; implementation not started. This file specifies the learner experience and numerical contracts, not implemented SVG, React, workers or browser verification. Read lesson.md, data-provenance.md, complete programs and saved results together. No fixed number of labs is implied for other topics.

## Representation plan and shared contracts

Use arrows, local coordinate frames, assignment edges, tensor correspondence and actual image reconstructions. The purpose is to expose the computation. Avoid representing every section as the same sliders above a paragraph. The core routing explorer needs a coupled arrow field and routing table; the geometry branch needs commuting diagrams; the fitted example needs pixels, scores and masked reconstructions.

All investigations open with current computed vectors, lengths and outputs visible. Valid edits update the same bounded model used by every arrow, table and readout. Reset restores the fixture; Step advances a routing iteration and exposes the current coupling and sum. No predicted answer is collected.

Show categorical changes such as winner/tie beside exact vector lengths and output differences. Explain the responsible vote or routing term using actual current values. There are no answer choices, correctness verdicts or background calculations that leave an old output attached to new inputs.

Arrow coordinates are abstract dimensionless capsule coordinates. Image pixels are intensity fractions 0–1; CSV integers are 0–16 and scaled once. Class lengths are bounded scores, not probabilities. Routing couplings are normalized rows, not class confidence. Keep decimals and rounding only in display; compute at full chosen precision. Class ties use the lowest parent index and must be captioned in tasks involving a tie.

No animation is necessary to understand a state. Use explicit next/previous controls with a stable step counter, visible current values and textual tables. Respect reduced motion. All inputs have labels, ranges, keyboard equivalents and error states. Color is redundant with child/parent names, arrowhead shape or line pattern. Use meaningful contrast, focus order and at least 44px touch controls. On narrow screens stack paired panels; preserve table headers and an accessible row-oriented text view. No hover-only formula, tooltip-only axis unit, color-only Calculate and explain or endlessly auto-running simulation.

## 1. From tensor channels to one capsule

Placement: §1 and §4, with the same numerical index convention. This is an explanatory correspondence, not a prediction lab.

Draw a 4×4 grid, four capsule types at each cell, four coordinate slots per type. Selecting a cell highlights its 16 original channels, regrouping into four arrows/vectors. The exact source layout is B×16×4×4, channel = type×4 + coordinate. The flattened child index is ((row×4 + column)×4 + type). Show source coordinate (channel,row,column) and target (child,coordinate). Batch B remains separate.

Default cell row 1, column 2, type 3 maps to child (1×4+2)×4+3 = 27, channels 12–15. Input a changed cell/type, e.g. row 2, column 1, type 0 → child 36. Render no invented activation values: use either the index-only construction or calculated real primary vectors from the retained model. Clearly label which.

Caption: regrouping preserves values and location/type identity through the index; it does not infer pose semantics. Accessibility: list source/target coordinate pairs. Phase two checks all 256 coordinate mappings and round trip, boundary child 0/63, B>1 separation and row/column labels.

The original large CapsNet topology is a separate static shape strip in §8. Display actual 28→20→6 sizes, 32 types×8 coordinates, 1,152 children, 10×16 class vectors and masked 160-value decoder input. Derive its count table from the exact formulas in mechanics-results.json rather than hand-copying inconsistent labels.

## 2. Vote workshop: agreement, cancellation and row normalization

Placement: §2–3, immediately beside the concrete three-child/two-parent example. One shared investigator exposes three linked representations:

- Two parent coordinate planes, each displaying the three child votes and their weighted contributions;
- Three child nodes with two outgoing edges labeled c(i,A), c(i,B);
- A step table of logits, couplings, sums, squashed outputs, lengths and dot-product agreements.

Input is the actual votes[3][2][2] in mechanics-results.json. Initially votes child1 A[2,0]/B[0,1], child2 A[2,0]/B[0,-1], child3 A[0,1]/B[0,2]. Do not drag a parent output; it must be recomputed. Learners can edit any vote coordinate via keyboard fields or drag an arrow endpoint; range [-3,3], step .1, finite numbers only. Coordinate edits genuinely alter entities. Keep parent planes at a stable shared scale; never rescale each parent to appear equally large. All three child contributions remain distinguishable.

Mechanism: b=0; c = stable softmax over the two parents for each child; s[j]=sum_i c[i,j]*vote[i,j]; v=s*norm(s)/(1+norm(s)^2); agreement[i,j]=dot(vote[i,j],v[j]); b += agreement if another step follows. Maximum 8 steps. The displayed state is the output of step t and its input logits/couplings; “Next” uses that agreement to compute step t+1. Restart b after every edited input, never reuse logits from the previous votes. Smooth interpolation between states must not be labeled an actual routing iteration.

Initial task: observe whether A, B or a tie has the longer vector after 3 steps. The current computed result is visible. Calculate and explain lengths with tie tolerance 1e−8, numeric length estimates tolerance .002. First demonstration is explained only in the current live view. Next investigation Show current outputs with each valid input change.

Checked reference: baseline step1 lengths [.8095238095,.5], step2 [.9149600672,.6992512291], step3 [.9346149254,.7785845120]. Step3 A couplings [.8996394794,.9899663614,.1075606613]. Contrast changes child2 A to [-2,0], giving step3 [.6490798364,.8584637810]. Saved full traces include all numbers and intermediates, not just an outcome.

Nulls: zero all votes → zero outputs/uniform coupling; identical-parent votes [1,0] from all three children → identical outputs/uniform rows at all steps; permuting child records together → unchanged parent output. Renaming a child only changes labels. A common rotation of every 2D vote rotates outputs by the same orthogonal transformation and preserves lengths/couplings; check this before offering a rotation control. Multiplying all votes by a scalar generally changes lengths and subsequent couplings.

Useful follow-up: add two versus four identical children in a separate bounded preset using the same generic arithmetic (up to 4 children). At one step, lengths .5 versus .8; parent outputs remain equal. This reveals weighted sum versus incoming weighted average. Any change in child count recompute current outputs and resets b. If implementation supports only the fixed three-child editor, teach this second comparison as a calculated static figure with the values and leave the separate exercise intact.

Feedback should identify the dominant changed contribution and whether cancellation, direction, magnitude or relative agreement caused the result. It must not assert that the longer parent is a true object or that low entropy means correctness. A mini entropy trace uses only saved/computed actual steps; concentration and class winner are different outputs.

Text alternative: six editable (x,y) pairs; the same output table and equations. Mobile: parent A/B tabs or vertically stacked planes with a persistent shared step control, never require simultaneous tiny hover targets. Phase two verifies scalar NumPy fixtures, row sums, bounds, clipping-free arrows, zero norms, identical rows, permutation, edit invalidation, stale-task cancellation and keyboard operation.

## 3. Squash: move along an arrow or turn it

Placement: §2 introductory curve and §7 deeper sensitivity. These may share one calculation component but expose depth progressively.

Core graph has x = input length r, y = output length r²/(1+r²), r in [0,6]. Mark (0,0), (1,.5) and radius sqrt(4.25) with the core example. Show the input vector and squashed vector on one coordinate plane. The vertical axis is score length, not probability. Analytic function, not measured data; caption as such. Sample the function as needed for plotting, retaining explicit formula and exact critical points.

Deeper mode shows a small radial and perpendicular perturbation at the same starting vector, with output difference and tangent arrows. Display eigenvalues 2r/(1+r²)² and r/(1+r²). Controls edit actual s=(x,y) within [-5,5], perturbation magnitude [0,.05] and direction radial/perpendicular; r=0 disables an undefined direction and uses explicitly chosen Cartesian axes instead. No unsupported “gradient explodes at zero” tooltip.

Live observation: display which perturbation yields the larger local output change for the chosen nonzero vector; compute finite differences for the selected magnitude, not merely compare against eigenvalues if the perturbation is finite. Zero perturbation is a null task with identical output; at zero report no direction. Baseline s[.3,.4] gives radial .64 and tangent .4; [3,4] gives radial .0147928994 and tangent .1923076923. The infinitesimal comparison reverses. The analytic curve's inflection is 1/sqrt3, not its half-height r=1.

The original manuscript derivative fixture is in mechanics-results.json. Phase two independently checks analytic derivatives, finite perturbation result checks, zero behavior, display units, error tolerance and reduced-motion mode. Avoid finite-difference step extremes that create an inaccurate demonstration without explaining cancellation.

## 4. Fit comparison: distinguish decisions, margins and information availability

Placement: §5. This evidence figure uses actual saved data, not an interactive retraining promise. Show paired seeds 1–3 and both trained routing settings. Separate classification correct count, margin loss and predicted-mask reconstruction MSE; each has its own labeled axis. Include all six results, with table access and no cropped axis designed to amplify a one-image difference. Hover only supplements visible labels.

A second matrix displays inference counts 1,2,3,5 for each fixed trained model. Each row is one immutable model. Clicking a cell reveals predictions changed versus that model's training-time count, including the difference between “same count” and “same predictions.” Source: author-check-results.json comparison_summary plus calculated-inputs.json inference_iterations. Seed3 trained1: inference2 correct117 but one changed prediction. Cells do not update the model weights or training metrics.

Display training accuracy 280/280 separately. No smoothing of recorded trajectories; steps 0,1,100,300,600 can be connected as recorded checkpoints with straight segments, but do not claim values between them were measured. No timing axis or hardware comparison is available.

Training-mean-image reconstruction baseline MSE .07295990735 is a horizontal reference only on the MSE panel. Show predicted-class and true-label-conditioned MSE under distinct names, because the latter uses additional information. All labels are loaded from verified retained rows, never passed to the encoder in a reconstruction widget.

This explanatory evidence selection has no invented practice show the computed values and their cause. A follow-up asks the learner to choose the claim supported by the displayed contrast, e.g. “lower margin loss in these paired fits” versus “uniformly better classification.” compare against the exact table and explain scope. It must not use a canned always-positive capsule verdict.

Phase two checks every plotted point against JSON, paired-seed identity, row/cell controls, viewport text, exact denominator120 and source model identity.

## 5. Actual pixel and latent-coordinate investigations

Placement: §6. Use a **single fixed seed-1 three-step model** initially. Offer source251/actual4 and source40/actual9, the first two development rows, not hand-picked favorable outputs. A later optional switch to the seed-1 uniform model is allowed if its distinct weights are loaded on demand; never silently substitute a model. Changing actual specimen changes model inputs; changing only a displayed ID/title does not.

Panel A: 8×8 editable pixel grid, original ghost image, class-length bar chart and predicted-class reconstruction. Each pixel is a numeric 0–1 value; keyboard editing maps to same array. Allow one-pixel nonwrapping right/down/left/up shifts with zero fill and a zero-shift null. A “Restore input” control differs from full lab reset. The retained author contrast flips pixel row3,col3 to1−old; use its actual before/after numbers and lengths from author-check-results.json. Learners must also make their own unsolved edits within the same bounds.

**Live input effect:** edit the input and display original versus changed argmax, including ties, alongside class-vector and score changes. Recompute the frozen model after valid edits. The actual label remains a reference annotation; it does not enter the encoder or predicted-mask decoder.

Panel B: fixed encoder result, selected class mask and one selected vector coordinate (0–7) with delta [-.2,.2]. The coordinate edit acts only on the decoder input, visibly separated from the original class-score panel. Show original and edited masked reconstructions plus a signed difference image on a symmetric scale. Preserve the original encoder scores; if showing edited vector norm, label it as a separately recomputed norm, not a new encoder prediction. Do not name axes “width/angle/thickness” without evidence.

**Live latent effect:** edit the selected coordinate/class and display the reconstruction array difference immediately. Use max-absolute tolerance 1e−7 in the float64 reference and a browser tolerance established by parity checks. Delta ±.1 contrasts and delta 0 nulls are saved for both models/two examples. Editing only a masked-out class by .1 leaves reconstruction exactly unchanged.

Input contract for the model: reshape16 channels into type4,coord4,H4,W4, reorder to row,column,type,coord; squash; W64×10×8×4; softmax parents; three full routing steps; lengths; masked decoder80→64→128→64 sigmoid. Native float32 and independent NumPy computations match selected outputs within6.39e−8 for capsule coordinates and2.98e−7 for reconstruction pixels. Saved complete model state and architecture are present; no pretrained dependency.

Performance: only the topic's needed selected model, two inputs and compact evidence are included in the lesson's lazy asset route. The whole calculated-inputs.json contains six-run author evidence and must not become an eager client import. Convert required weights to a precise compact representation with provenance and parity checks in phase two. Run bounded inference in a worker if needed; latest request revision wins, old work cannot overwrite edited state, inputs cannot enqueue unbounded simultaneous computations. Debounce expensive drag inference or require an explicit Run . Do not retrain in-browser or increase image/class dimensions.

Text/mobile: editable cell table with row/column, ten-score table, reconstruction-intensity table and concise max-change/changed-class description. Canvas is not the only representation. Handle missing/corrupt model load with an honest retry state; never replace results with a generic toy model. Phase two checks independent fixtures, all shifts including zero, pixel bounds, mask index, default labels, saved-state identity, null handling, worker cancellation, pending-state reset and accessible controls.

## 6. Coordinate frames and EM: optional mechanisms

Placement: §7. A static or stepped commuting square shows explicit homogeneous matrices M, W and G from mechanics-results.json. Top path: transform part then predict whole; bottom: predict whole then transform. Both end at (-3,1). A second vector fixture uses W=diag(2,1), u[1,2], R90 and highlights unequal endpoints [-4,1] versus[-2,2]. The first is exact supplied geometry, the second an unconstrained-map counterexample. Do not depict either as fitted capsule geometry.

Compare both path endpoints immediately and show the residual vector. The transformation controls act on the part and whole in the stated coordinate frames; the diagram explains which route preserves the relationship.

EM panel: two vote-coordinate plots, three children, separate child activation bars, parent mean/diagonal spread ellipses and responsibility rows. Use mechanics-results.json em fixture. Each parent sees a different vote from the same child. First means .48/1.84, masses1.25 each; step3 means .10880966/2.81926704. Second-axis spread is floor .01, explicitly labeled “variance floor,” not observed nonzero spread. R rows sum1; parent activations need not.

Display the current component means and their motion under active/inactive-child vote edits. Show actual weighted sufficient statistics and explain when an inactive child contributes zero. This is a model-state comparison, not a scored response.

This is an explicitly constructed EM illustration, not a fitted matrix-capsule model or a benchmark. Browser phase checks log-domain normalization, finite variance, negligible mass, no accumulated activation multiplication, result checks against current inputs, source contracts and labels. Full STAR/VB architectures remain annotated optional references, not nonfunctional browser controls.

## Completion boundary

Implementing these representations, rendering diagrams, worker/model packaging, measuring browser performance, native/browser parity, keyboard/accessibility checks and independent phase-two review are all deferred. Content-stage checks covered the stated numerical fixtures, real fitted outputs and manuscript logic only. Keep this packet's offline inputs/programs/results until its later authorized finish phase has consumed them.

## Precise fixture and scope details retained for implementation

**Diagonal EM.** Edit child activation in [0,1] or a vote coordinate in [−4,4]. Recompute q=aR each step using `diagonal_em` in `capsule-mechanics.py`, mass guard 1e−12, variance floor .01, beta constants 0 and λ schedule .5/.75/1. Changing an inactive third child's vote leaves output means unchanged, although its responsibility row may change. At all activations 0, show “no active evidence; mean not identified” and stop; finite guarded defaults are not a discovered object.


## Code-to-mechanism implementation contract — 22 September 2026

Place the manuscript's new implementation route next to its stated concept section. Preserve the named axes, state and algorithm steps when implementing figures; the complete teaching programs are content inputs, not a hidden replacement for learner-visible code. Render long source only on demand with keyboard-scrollable code and wrapping download labels. Keep constructed comparison fixtures separate from recorded training experiments. No browser execution of Python, pretrained-model download or GPU experiment is required to operate a lab.

The ownership map in design.md identifies which operations are implemented here and which actual sources are reused. Both paths must be findable: the transparent mechanism and the ordinary package/tool route, followed by the changed-input practice. There is no guess-entry, prediction submission or answer-unlock state. Current outputs remain visible while the learner edits meaningful inputs; separate written practice can retain hints and solutions.
