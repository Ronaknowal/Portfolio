# ConvNeXt: figure and investigation specifications

## Live exploration contract — 21 September 2026

Open each investigation with its current inputs, intermediate mechanism and complete current output visible. Apply valid edits to meaningful entities immediately and update diagrams, tables, units and causal explanation together. No prediction entry, predicted-answer choices, commitment, prediction grading or answer-unlock feature is part of this packet, even optionally. Model predictions and mathematical masks/gates remain subject matter.

Use the topic-specific controls and checked fixtures below. Pair sliders or direct manipulation with labeled keyboard/numeric controls; keep presets as starting points, not the only editable values. A pinned baseline preserves its inputs, seed, units and outputs while the current case changes. Explain both a meaningful contrast and an unchanged/null result, then connect the observed effect to a practical design decision. Reset restores the stated fixture and current result. Invalid text has a local explanation and a clearly identified last valid result; never silently clamp or pair new inputs with old output.

Step/Back and bounded Run controls advance a real computation or reveal its chronological stages, not permission to view an answer. Show the current state and its result throughout. Keep exact small calculations live. For costly frozen inference, debounce or run bounded work with pending/current state labels and stale-result cancellation; inspect saved measurements without implying fresh training. Respect reduced motion, keep focus stable and avoid announcing every animation frame. Independent written practice and its hints/solutions stay separate.

Phase two must test default results without any action, meaningful edits, quick consecutive edits, valid extremes, null/invalid cases, reset, linked-view agreement, keyboard operation and readable phone layouts. The mathematical/reference checks already specified below remain; these live browser checks have not been performed in this content-only revision.

### Topic-specific live route
**Inspect modern convolution blocks.** Change stage dimensions, normalization groups, GRN feature cells, valid visible-patch selections and branch-folding coefficients.
**See the consequence.** Show parameter counts, shared GRN denominator, changed feature maps, reconstruction consequences and folded-kernel equality immediately. Preserve image masking as the learning objective, not UI answer hiding.
**Decision connection.** Separate architecture from recipe, global channel context from local normalization, and valid reparameterization from a changed function.


Content-first specification,13 September2026. No SVG, React, browser model or visual implementation exists in this packet. Implement only under a later finish authorization, retaining the numerical contracts below.

## Shared interaction and evidence contract

Each investigation opens with current outputs and uses one complete state for inputs, mask, model identity, normalization axes and selected output. Valid edits update that state and all dependent results together. A retained baseline is immutable; no prediction, submission or completion score belongs to the lab.

All activities require editable mathematical entities, such as feature cells, reduction groups, image pixels, patch visibility, kernel coefficients and affine parameters. Presets help comparison but cannot replace those edits. Provide explicit Apply/Calculate rather than computing on every pointer movement. No fitting, pretrained downloads, model training or random full-size network construction in the browser.

Use separate controls for Show the current computed result and its contributing terms immediately. Exact values: default tolerance max(1e−6,1e−4|target|) for small hand calculations; saved small-network float32 versus double reconstruction tolerance1e−5. For zero-change prediction, maximum absolute change≤1e−5 for browser model; analytically exact zero remains explained as exact. Direction choices use a neutral zone ±1e−5; do not Calculate and explain numerical noise as a meaningful change. When bounds or equalities make more than one proposed category valid, Calculate and explain the explicit actual category rather than force a preferred story.

Feedback states the changed input, the intermediate statistic or path that carried the change, and the computed outcome. Show original and edited values side by side in the current live view. No perpetual green pass banner, auto-filled answer, celebratory score for selecting the provided worked fixture or animated outcome . Reset restores the stated inputs and immediately displays their computed result.

All matrix cells have row/column/channel names, exact numeric text and keyboard controls. Use semantic labels and focus-visible buttons, not color alone. Arrow keys may traverse an accessible matrix with an equivalent ordinary table available. Errors identify the invalid cell/value and leave prior valid state intact. If pointer drawing is offered, the equivalent numeric editor remains available. Touch targets at least44px; on narrow screens stack source→statistic→output rather than shrink numbers. Keep legends next to the relevant diagram, per-activity control count small, reduced-motion support, no obligatory animation. Dynamic announcements occur after a meaningful settled change in a polite status region.

Offline links/downloads preserve attribution. Assets load only with this lesson or upon opening the specific optional activity. The two seed1 masked models are small; keep the shared dataset/results out of compact navigation and unrelated routes. Do not import all chapter models or include large ImageNet parameter tensors. Release activity state on unmount; heavy plot redraw or inference is initiated deliberately and, if necessary after measurement, moved to a worker. Bounds below are mathematical safeguards, not fabricated performance targets. Phase two measures actual behavior.

## 1. Architecture-versus-recipe experiment genealogy

Placement lesson§1 after the comparison record. Purpose: distinguish an architecture intervention, a recipe intervention and an uncontrolled combined comparison. A branching diagram is more suitable than a generic slider because the causal question depends on which earlier configuration a change starts from.

Draw two initial nodes for old/enhanced ResNet recipe, then a short annotated chain: depthwise replacement→widening→move depthwise before expansion→increase spatial kernel. An optional expanded view can show remaining roadmap steps. Nodes visibly distinguish published measurements from calculated operation counts. Do not redraw a monotone improvement curve or fabricate intermediate benchmarks.

Published source: ConvNeXt V1 §2, Figure2, Appendix C **Table10**, whose label and values were checked. Small-regime values: baseline76.13; enhanced recipe78.82±.07; stage ratio79.36±.07; patchify79.51±.18; depthwise78.28±.08; width80.50±.02; inverting80.64±.03; moving depthwise79.92±.08; k5 80.35±.08; k7 80.57±.14; rejected k9 80.57±.06 and k11 80.47±.11; GELU80.62±.14; fewer activations81.27±.06; fewer normalizers81.41±.09; LN81.47±.09; separate downsampling81.97±.06. The enhanced/modernized results are three-seed averages; baseline76.13 is the cited prior implementation result. If using an expanded graph, retain the paper's GFLOPs convention and distinguish it from our narrower MAC counts.

The basic figure only needs the mechanism-changing subset. Hover/focus notes identify what was held fixed and what changed. Text fallback narrates the same sequence and names failed/rejected branches. A baseline/final comparison cannot be described as an isolated effect of depthwise convolution or GELU. Optional learner task is to propose which comparison answers a given intervention question; it complements, rather than replaces, the genuine numerical edit activities below.

## 2. Block anatomy and normalization reduction lattice

Placement§2: first a static shape ribbon connecting DW7→channelLN→expand→GELU→project→LayerScale/DropPath→residual add. Color denotes axis roles, with labels/chip shapes duplicating the meaning. The input skip must join after the branch mask; the spatial filter precedes expansion; the second linear transformation has no extra activation. V2 view replaces LayerScale with expanded-width GRN after GELU and before projection. Do not depict GRN at widthC when it acts at4C.

Show NCHW↔NHWC as a change of axis order, not a spatial/channel mixer. A separate small memory diagram may show fixed logical NCHW dimensions with different strides; do not claim permutation alone copies memory. Shape table remains the authoritative text alternative.

Interactive reduction lattice: start one specimen with H1,W2,C2, feature values locations[1,3] and[101,103]. Channels editable within[-200,200], mode channelLN or singleGroup, epsilon fixed1e−6, affine gamma1/beta0 initially. Learner can edit a value and select the grouping by marking the actual reduction axes. Valid input changes recompute every dependent result and explanation; retained baselines keep their original inputs.

Ask before computation: select exactly which output cells will change when the chosen input cell changes, and optionally inspect the first location's first normalized value. Calculate and explain affected-set equality from the actual output comparison, not from a hardcoded “LN changes two” rule; degenerate edits may produce no change. Show mean/variance calculated from highlighted input set, then arrows to the normalized cells.

Default fixture outputs are in author-check-results.json normalization_axes. ChannelLN both locations≈[-.9999995,.9999995]; singleGroup produces different location offsets. For a contrasting input edit, changing only101→105 in the two-channel example changes magnitudes only slightly through epsilon because two unequal channels remain symmetric standardized pairs. Therefore switch to the checked three-channel fixture[1,3,7],[101,103,107] for a clearly nontrivial affected-set activity: change the second location's101→105. All three outputs of that location change while the first location is untouched; exact values are in normalization_axes.three_channel. The checked null adds10 to all three channels of only the second location, producing maximum output change0. The two-channel fixture primarily teaches the axis distinction, and its nearly constant pair is a useful degeneracy rather than a large-change promise.

Readiness transfer: change one entire location by a constant, then change one channel among three; explain why locationwise shift invariance differs from arbitrary cell invariance. Bounds H≤2,W≤3,C≤4; no requirement that norm mean/variance be reused across specimens.

## 3. Stage grid and parameter budget

Placement§3. Static proportional grids56,28,14,7 with channel-width rails96,192,384,768 and repeated blocks3,3,9,3. A selected input patch remains visually traceable through downsampling; avoid suggesting one feature cell is literally an input-pixel average. Show spatial support separately from learned channel meaning.

Optional counter: select width C∈[4,256], expansion t∈[1,6], odd k∈{3,5,7,9}, H,W∈[1,56], and V1/V2. The learner edits the operator definition; calculate components rather than return a stored family total. V1 parameters=2tC²+(k²+t+5)C. V2 replaces C LayerScale parameters with 2tC GRN parameters, yielding2tC²+(k²+3t+4)C. Conv/linear MACs=HW(k²C+2tC²), excluding other operations. If offering depthwise filtering after expansion, recompute its width as tC and display that different block.

Checked fixtures: C96,t4,k7 gives79,296 V1 and79,968 V2 parameters; practice C64,t3,k5 gives26,688 V1 parameters and5,130,496 MACs at14². Doubling both spatial dimensions quadruples MACs and leaves block parameters unchanged. Switching V1→V2 leaves conv/linear MACs unchanged while adding GRN work excluded from this counter. Show the current computed result and its contributing terms immediately. Complete family counts/shapes are in block-check-results.json. Distinguish actual named V1 configurations from shape-only V2 Small/XLarge combinations that the helper constructs without claiming official V2 variants.

## 4. GRN channel maps with a shared denominator

Placement §4. Show each channel as a2×1 spatial map, length bar G, common mean denominator, relative bar R, then output map. Distinguish the LN diagram's grouping per location from this spatial reduction per channel. Inputs and G use activation units; R and gamma are dimensionless; beta uses activation units.

Defaults: maps A[3,4], B[0,12], gamma[.5,−.5], beta[0,0], epsilon1e−6. Editable cell values[-30,30], gamma[-1,1], beta[-5,5], H≤3,W≤3,C≤4. Plain text lists every formula and intermediate. Changing one channel updates the others' denominator; gamma's sign does not establish whether a feature is useful.

Live observation: edit B's second cell12→0 and watch A(first)'s output and its change from the baseline update immediately. Original R[.588235225,1.411764540], output A[3.882352837,5.176470450], B[0,3.529412761]. Edited outputs A[5.9999988,7.9999984], B zeros. Calculate and explain any allowed edit against the actual formula. Alternate B12→24 makes A(first) approximately3.517241; show its denominator14.5 in the current live view.

Checked nulls in author-check-results: gamma=beta=0 returns the input exactly; zero input and beta0 return zero even with nonzero gamma. Another specimen does not affect this specimen's dense GRN; compute per image if offering a batch. A spatial permutation of input and output commutes with this GRN because it preserves norms. This is not an invariance claim for the entire CNN.

Optional gradient branch starts gamma/beta0 and loss.5sumY². Inspect which parameters can receive a gradient before any update; reveal gamma gradients[14.70588062,203.29409373], beta gradients[7,12], dL/dX=X. Recorded central-difference agreement6.3e−10. Optional one-step SGD uses learner-chosen eta≤1e−3 and fresh numerical evaluation; do not assume larger steps improve loss. Identity initialization does not mean GRN starts participating only after training.

The feature-diversity panel in §6 uses the six actual runs. Show near-zero channel-norm fraction separately from mean(1−cos)/2 over valid nonself pairs. Do not mark higher distance as inherently better. These statistics do not use labels; the supervised probe uses training labels.

## 5. Masked-image information boundary and specimen investigation

Placement §5 before the experiment: split the image into two visibly separate paths. Input: patch mask→stem→visible feature grid→masked encoder→encoded features plus mask tokens→decoder→reconstruction. Target: original image→hidden-pixel loss. Mark where no label is supplied and where the later frozen probe uses training labels. Draw each hidden2×2 patch as a unit; zero-valued dark background must look different from unobserved input.

In §6 use source251/label4 and source40/label9 from calculated-inputs.json with the two seed1 models. Retain original data and provenance. User-edited specimens are counterfactual edits, not new real recordings; do not substitute invented drawings for the recorded observations.

Inputs: specimen, model variant, actual8×8 normalized pixel values[0,1], exactly6 visible cells in the4×4 patch grid. Initial selection uses its first fixed development mask. Numeric pixel edits default to increments1/16, allowing arbitrary bounded real values. The patch editor swaps one visible and one hidden patch, preserving6; disable Apply until both endpoints are selected. Reset restores the stated inputs and immediately displays their computed result.

Display current reconstruction MSE and its change from the original beside the reconstruction and difference map. The patch mask determines which pixels enter the model and which reconstruction errors are assessed; it never hides the lab's answer. Pixel and mask edits update the actual bounded inference result.

in the current live view, display four aligned8×8 grids: original target, current visible input, reconstruction, hidden-pixel error. Offer an additional output-difference grid on demand. Use the same value palette across comparisons and an exact numeric view. Reconstruction values are unbounded linear outputs; annotate display clipping and never alter MSE with it. Visible errors may be displayed separately but are not scored. Coordinates are zero-based row/column.

The forward contract is masked-reconstruction.py and independent_prediction in author-checks.py:

1. Repeat the4×4 visibility mask into2×2 patches; mask the image before Conv1→12, k2s2.
2. Apply channel LayerNorm12, epsilon1e−6, then mask inactive features.
3. Each of two blocks: masked input; DW3p1 plus bias; mask; channel LN; linear12→48 and exact GELU; mask; optional per-image GRN over spatial norms and the48-channel mean; linear48→12; residual add; mask.
4. Insert the learned mask token only at inactive positions. Use one unmasked decoder block without GRN, a1×1 head12→4, and pixel shuffle2 with channel order(row offset,column offset)=(0,0),(0,1),(1,0),(1,1).
5. Score original hidden targets by averaging40 pixel errors, using the complete saved seed1 state.

No native sparse backend or fitting is needed in the browser. Independent scalar NumPy loops agree with all four stored reconstructions within3.21e−7. Checked interventions:

|Variant/specimen|Hidden(0,0)flip: predictionchange|MSEbefore→after|Visible(0,2)flip:maxpredictionchange|
|---|---|---|---|
|absent/251|0|.053903108→.080158571|.455133147|
|absent/40|0|.068924853→.089879007|.082818690|
|present/251|0|.054046563→.080387244|.506609512|
|present/40|0|.068796620→.090517749|.090682697|

These visibility and pixel null/contrast fixtures are actually checked. Swapping the first row-major visible patch with the first hidden patch is also checked: maximum prediction changes for absent/251, absent/40, present/251, present/40 are .431955433,1.171942695,.413967947,1.219148770. Exact changed masks are retained in each intervention row. No “every visible edit always changes prediction” assertion; the selected checks demonstrate that it can.

The reconstruction trace uses actual steps0,1,100,300,600, with markers for observations and connecting segments rather than an invented fitted curve or uncertainty band. Show one paired seed plus the mean-image baseline, with an optional full-seed table. Keep probe counts on a separate0–120 axis: raw118, absent115, present116. Expose denominators and metric names. Probe training counts280/280 do not imply zero reconstruction error.

## 6. Linear-branch folding stencil

Optional placement §7. Draw three branches—Conv3+BN, Conv1+BN and identity—beside one folded3×3 kernel. Offer a shared activation after each equivalent sum, plus the separate-branch-activation failure case.

Editable coefficient[-3,3], convolution bias[-2,2], fixed BN mean[-3,3], variance[.01,9], gamma[-3,3], beta[-2,2], and input pixel[-30,30]. Require aligned odd kernel centers, unit stride, equal output shape and matching identity channels. Default single-channel5×5 image has row-major values1…25. The3×3 kernel contains0…8 divided by10;1×1 kernel=2. Convolution biases=.4,−.3. First BN mean1,variance4,gamma3,beta−.2; second mean−2,variance1,gamma.5,beta.7; epsilon1e−5.

Compute each branch's W'=γW/sqrt(v+eps), b'=γ(b−μ)/sqrt(v+eps)+β. Add centered padded kernels and identity delta. Show the current computed result and its contributing terms immediately. Default center output111.0498261257; maximum error over all outputs2.84e−14. Full folded kernel and bias are retained in author-check-results.json.

The checked negative fixture uses branch1 Conv(image) and branch2 Conv(−image), comparing separate ReLUs with ReLU of the sum. Maximum difference23.44987925. Show the negated input in the graph; it differs from the default positive-input fixture. The scalar practice x=−2 gives2 versus0. Learner edits may make equality hold at a particular input; Calculate and explain the actual result rather than claiming nonlinear branches differ on every input.

Text fallback walks through each folded coefficient and one output dot product. Keep BN in evaluation mode: training-batch statistics cannot be represented by the same static kernel using stored running statistics. No latency or speedup is claimed; a later authorized deployment investigation could measure that separately.

## Deferred implementation checks

Phase two implements and validates the actual browser models and diagrams, keyboard and small-viewport layout, graphical legibility, numeric formatting/rounding, loading/error recovery, live recomputation, reduced motion and route-lazy imports. Verify every worked/null/changed fixture against the retained source, including the checked three-channel-axis and patch-swap fixtures, before exposing them. Run complete optional architecture code as needed when its content changes; do not repeat the unchanged six fits merely to reconfirm recorded outputs. Formal independent review and application build/browser checks are all deferred.


## Code-to-mechanism implementation contract — 22 September 2026

Place the manuscript's new implementation route next to its stated concept section. Preserve the named axes, state and algorithm steps when implementing figures; the complete teaching programs are content inputs, not a hidden replacement for learner-visible code. Render long source only on demand with keyboard-scrollable code and wrapping download labels. Keep constructed comparison fixtures separate from recorded training experiments. No browser execution of Python, pretrained-model download or GPU experiment is required to operate a lab.

The ownership map in design.md identifies which operations are implemented here and which actual sources are reused. Both paths must be findable: the transparent mechanism and the ordinary package/tool route, followed by the changed-input practice. There is no guess-entry, prediction submission or answer-unlock state. Current outputs remain visible while the learner edits meaningful inputs; separate written practice can retain hints and solutions.
